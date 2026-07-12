import asyncio
import json
import re
import time
import datetime
import uuid
from collections import deque
from typing import Dict
from dataclasses import dataclass, field

import astrbot.api.star as star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api import logger
from astrbot.api.message_components import Plain
from astrbot.core.agent.message import TextPart
from astrbot_model_usage import schedule_model_usage

MODEL_USAGE_SOURCE = "heartflow"


@dataclass
class JudgeResult:
    """判断结果数据类"""
    relevance: float = 0.0
    willingness: float = 0.0
    social: float = 0.0
    timing: float = 0.0
    continuity: float = 0.0  # 新增：与上次回复的连贯性
    reasoning: str = ""
    should_reply: bool = False
    confidence: float = 0.0
    overall_score: float = 0.0
    related_messages: list = None
    trigger_event: object = None  # 批量判断时，被选为走 LLM 的主 event 引用
    batch_size: int = 1

    def __post_init__(self):
        if self.related_messages is None:
            self.related_messages = []


@dataclass
class RawMessage:
    """原始群聊消息条目"""
    sender_name: str
    sender_id: str
    content: str
    timestamp: float
    is_bot: bool = False
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class QueuedMessage:
    """一條等待防抖判斷結果的消息。"""
    raw_message: RawMessage
    event: object
    waiter: asyncio.Future


@dataclass
class ChatState:
    """群聊状态数据类"""
    energy: float = 1.0
    last_reply_time: float = 0.0
    last_energy_update_time: float = 0.0
    last_reset_date: str = ""
    total_messages: int = 0
    total_replies: int = 0


@dataclass
class DebounceState:
    """防抖合并状态"""
    timer: asyncio.TimerHandle | None = None
    timer_generation: int = 0
    pending: list[QueuedMessage] = field(default_factory=list)
    is_judging: bool = False
    judge_start_time: float = 0.0
    judge_task: asyncio.Task | None = None
    # 旧批次仍在判断时，新消息进入下一批
    next_pending: list[QueuedMessage] = field(default_factory=list)


def _extract_json(text: str) -> dict:
    """从模型返回的文本中稳健地提取 JSON 对象。

    依次尝试：
    1. 直接解析
    2. 去除 markdown 代码块后解析
    3. 正则提取第一个 {...} 子串后解析
    """
    text = text.strip()

    # 1. 直接尝试
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 去除 markdown 代码块
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. 正则提取最外层 {...}
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError(f"无法从文本中提取有效 JSON: {text[:200]}")


def _clamp_score(v) -> float:
    """将模型返回的分数值钉位到 [0, 10]。"""
    try:
        return max(0.0, min(10.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


# Gemini/Google GenAI 结构化输出 Schema
JUDGE_JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "relevance": {"type": "INTEGER", "description": "内容相关度(0-10)"},
        "willingness": {"type": "INTEGER", "description": "回复意愿(0-10)"},
        "social": {"type": "INTEGER", "description": "社交适宜性(0-10)"},
        "timing": {"type": "INTEGER", "description": "时机恰当性(0-10)"},
        "continuity": {"type": "INTEGER", "description": "对话连贯性(0-10)"},
        "reasoning": {"type": "STRING", "description": "详细分析原因"},
    },
    "required": [
        "relevance", "willingness", "social", "timing", "continuity", "reasoning"
    ],
}


class HeartflowPlugin(star.Star):

    def __init__(self, context: star.Context, config):
        super().__init__(context)
        self.config = config

        # 判断模型配置（带 fallback 链）
        self.judge_provider_name = self.config.get("judge_provider_name", "")
        fallback_names = [
            self.config.get(f"judge_provider_fallback_{i}", "")
            for i in range(1, 6)
        ]
        # 有序 provider 名称列表，过滤空值和重复
        seen = set()
        self.judge_provider_chain = []
        for name in [self.judge_provider_name] + fallback_names:
            if name and name not in seen:
                seen.add(name)
                self.judge_provider_chain.append(name)
        if not self.judge_provider_chain:
            logger.warning("心流插件未配置任何判断模型提供商")
        else:
            logger.info(f"心流判断模型 fallback 链: {' -> '.join(self.judge_provider_chain)}")

        # 心流参数配置
        self.reply_threshold = self.config.get("reply_threshold", 0.6)
        self.energy_decay_rate = self.config.get("energy_decay_rate", 0.1)
        self.energy_recovery_rate = self.config.get("energy_recovery_rate", 0.02)
        self.context_messages_count = self.config.get("context_messages_count", 5)
        self.judge_context_count = self.config.get("judge_context_count", self.context_messages_count)
        self.min_reply_interval = self.config.get("min_reply_interval_seconds", 0)
        self.whitelist_enabled = self.config.get("whitelist_enabled", False)
        self.chat_whitelist = self.config.get("chat_whitelist", [])

        # 群聊状态管理
        self.chat_states: Dict[str, ChatState] = {}

        # 原始群聊消息缓冲区：{unified_msg_origin: deque[RawMessage]}
        # 记录所有群聊原始消息（无论是否触发 LLM），用于判断上下文
        self._raw_msg_buffer: Dict[str, deque] = {}
        self._raw_msg_buffer_size = max(self.context_messages_count, self.judge_context_count) * 4  # 缓冲区保留更多条以备用

        # 防抖合并状态
        self._debounce_states: Dict[str, DebounceState] = {}
        self.debounce_seconds = self.config.get("debounce_seconds", 5.0)
        self.judge_timeout_seconds = self.config.get("judge_timeout_seconds", 30.0)
        self.energy_system_enabled = self.config.get("energy_system_enabled", True)
        self.max_cached_messages = max(1, self.config.get("max_cached_messages", 10))

        # 判断配置
        self.judge_max_retries = max(0, self.config.get("judge_max_retries", 3))  # 确保最小为0
        
        # 判断权重配置
        self.weights = {
            "relevance": self.config.get("judge_relevance", 0.25),
            "willingness": self.config.get("judge_willingness", 0.2),
            "social": self.config.get("judge_social", 0.2),
            "timing": self.config.get("judge_timing", 0.15),
            "continuity": self.config.get("judge_continuity", 0.2)
        }
        # 检查权重和
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(f"判断权重和不为1，当前和为{weight_sum}")
            # 进行归一化处理
            self.weights = {k: v / weight_sum for k, v in self.weights.items()}
            logger.info(f"判断权重和已归一化，当前配置为: {self.weights}")

        logger.info("心流插件已初始化")

    async def judge_with_tiny_model(self, event: AstrMessageEvent) -> JudgeResult:
        """使用小模型进行智能判断（支持多 provider fallback）"""

        if not self.judge_provider_chain:
            logger.warning("小参数判断模型提供商名称未配置，跳过心流判断")
            return JudgeResult(should_reply=False, reasoning="提供商未配置")

        # 获取群聊状态
        chat_state = self._get_chat_state(event.unified_msg_origin)

        # 获取当前对话的人格系统提示词，让模型了解大参数LLM的角色设定
        persona_system_prompt = await self._get_persona_system_prompt(event)
        logger.debug(f"小参数模型使用人格提示词: {'有' if persona_system_prompt else '无'} | 长度: {len(persona_system_prompt) if persona_system_prompt else 0}")

        # 构建判断上下文
        chat_context = self._build_chat_context(event)
        recent_messages = self._get_recent_messages(event)
        last_bot_reply = self._get_last_bot_reply(event)

        judge_prompt = f"""
你是群聊机器人的决策系统，需要判断是否应该主动回复以下消息。

## 机器人角色设定
{persona_system_prompt if persona_system_prompt else "默认角色：智能助手"}

## 当前群聊情况
- 群聊ID: {event.unified_msg_origin}
- 我的精力水平: {chat_state.energy:.1f}/1.0
- 上次发言: {self._get_minutes_since_last_reply(event.unified_msg_origin)}分钟前

## 群聊基本信息
{chat_context}

## 最近{self.context_messages_count}条对话历史
{recent_messages}

## 上次机器人回复
{last_bot_reply if last_bot_reply else "暂无上次回复记录"}

## 待判断消息
发送者: {event.get_sender_name()}
内容: {event.message_str}
时间: {datetime.datetime.now().strftime('%H:%M:%S')}

## 评估要求
请从以下5个维度评估（0-10分），**重要提醒：基于上述机器人角色设定来判断是否适合回复**：

1. **内容相关度**(0-10)：消息是否有趣、有价值、适合我回复
   - 考虑消息的质量、话题性、是否需要回应
   - 识别并过滤垃圾消息、无意义内容
   - **结合机器人角色特点，判断是否符合角色定位**

2. **回复意愿**(0-10)：基于当前状态，我回复此消息的意愿
   - 考虑当前精力水平和心情状态
   - 考虑今日回复频率控制
   - **基于机器人角色设定，判断是否应该主动参与此话题**

3. **社交适宜性**(0-10)：在当前群聊氛围下回复是否合适
   - 考虑群聊活跃度和讨论氛围
   - **考虑机器人角色在群中的定位和表现方式**

4. **时机恰当性**(0-10)：回复时机是否恰当
   - 考虑距离上次回复的时间间隔
   - 考虑消息的紧急性和时效性

5. **对话连贯性**(0-10)：当前消息与上次机器人回复的关联程度
   - 如果当前消息是对上次回复的回应或延续，应给高分
   - 如果当前消息与上次回复完全无关，给中等分数
   - 如果没有上次回复记录，给默认分数5分

**回复阈值**: {self.reply_threshold} (综合评分达到此分数才回复)
"""

        # 遍历 fallback 链，依次尝试每个 provider
        judge_data = await self._call_judge_providers(event.unified_msg_origin, judge_prompt, persona_system_prompt)

        # 所有 provider 都失败，返回保守默认分数
        if judge_data is None:
            logger.warning(f"所有判断方式均失败，返回保守默认分数")
            return JudgeResult(
                relevance=5.0, willingness=5.0, social=5.0,
                timing=5.0, continuity=5.0,
                reasoning=f"所有判断方式均失败，使用默认分数",
                should_reply=False, confidence=0.5, overall_score=0.5,
            )

        # 从 judge_data 提取分数并钉位到 [0, 10]
        relevance = _clamp_score(judge_data.get("relevance", 0))
        willingness = _clamp_score(judge_data.get("willingness", 0))
        social = _clamp_score(judge_data.get("social", 0))
        timing = _clamp_score(judge_data.get("timing", 0))
        continuity = _clamp_score(judge_data.get("continuity", 0))

        # 计算综合评分
        overall_score = (
            relevance * self.weights["relevance"] +
            willingness * self.weights["willingness"] +
            social * self.weights["social"] +
            timing * self.weights["timing"] +
            continuity * self.weights["continuity"]
        ) / 10.0

        should_reply = overall_score >= self.reply_threshold

        logger.debug(f"小参数模型判断成功，综合评分: {overall_score:.3f}, 是否回复: {should_reply}")

        return JudgeResult(
            relevance=relevance, willingness=willingness,
            social=social, timing=timing, continuity=continuity,
            reasoning=judge_data.get("reasoning", ""),
            should_reply=should_reply,
            confidence=overall_score,
            overall_score=overall_score,
            related_messages=[],
        )

    async def _call_judge_providers(self, umo: str, judge_prompt: str, persona_system_prompt: str = "") -> dict | None:
        """遍历 fallback 链尝试 provider，返回解析后的 dict 或 None"""
        if not self.judge_provider_chain:
            logger.warning("小参数判断模型提供商名称未配置，跳过心流判断")
            return None

        judge_data = None
        used_provider_name = None

        provider_count = len(self.judge_provider_chain)
        for provider_index, provider_name in enumerate(self.judge_provider_chain, start=1):
            started_at = time.monotonic()
            try:
                provider = self.context.get_provider_by_id(provider_name)
                if not provider:
                    logger.warning(f"未找到提供商: {provider_name}，尝试下一个")
                    continue

                # 优先尝试 structured output（仅 Google provider 支持）
                if self._is_google_provider(provider):
                    call_path = "structured_output"
                    provider_call = self._judge_with_structured_output(
                        provider, judge_prompt, umo
                    )
                else:
                    call_path = "text_chat"
                    provider_call = self._judge_with_text_chat(
                        provider, judge_prompt, persona_system_prompt, umo
                    )

                logger.debug(
                    f"开始判断 provider [{provider_index}/{provider_count}] "
                    f"{provider_name} | 路径: {call_path} | "
                    f"超时: {self.judge_timeout_seconds}s"
                )
                judge_data = await asyncio.wait_for(
                    provider_call,
                    timeout=max(0.0, self.judge_timeout_seconds),
                )
                elapsed = time.monotonic() - started_at

                if judge_data is not None:
                    used_provider_name = provider_name
                    logger.info(
                        f"判断 provider 成功: {provider_name} | 耗时: {elapsed:.2f}s"
                    )
                    break

                logger.warning(
                    f"判断 provider 未返回有效结果: {provider_name} | "
                    f"耗时: {elapsed:.2f}s，尝试下一个"
                )

            except asyncio.TimeoutError:
                elapsed = time.monotonic() - started_at
                logger.warning(
                    f"判断 provider 超时: {provider_name} | "
                    f"耗时: {elapsed:.2f}s，尝试下一个"
                )
                continue
            except Exception as e:
                elapsed = time.monotonic() - started_at
                logger.warning(
                    f"provider {provider_name} 判断失败: {e} | "
                    f"耗时: {elapsed:.2f}s，尝试下一个"
                )
                continue

        if judge_data is not None:
            if used_provider_name != self.judge_provider_name and len(self.judge_provider_chain) > 1:
                logger.info(f"主 provider 失败，由备用 provider 兜底成功: {used_provider_name}")

        return judge_data

    def _is_google_provider(self, provider) -> bool:
        """检测 provider 是否为 Google GenAI（支持 structured output）"""
        try:
            if not hasattr(provider, 'client'):
                return False
            client = provider.client
            module = type(client).__module__ or ""
            return "google" in module and "genai" in module
        except Exception:
            return False

    async def _judge_with_structured_output(
        self, judge_provider, judge_prompt: str, umo: str
    ) -> dict | None:
        """使用 Google GenAI 原生结构化输出进行判断。

        返回解析后的 dict，或在失败时返回 None。
        直接调用底层 genai client，绕过 AstrBot 重试层，故此处自行重试。
        """
        from google.genai import types

        client = judge_provider.client
        model = judge_provider.model_name

        # AsyncClient 直接调用；sync Client 需要通过 .aio 访问
        if hasattr(client, 'aio'):
            gen = client.aio.models.generate_content
        else:
            gen = client.models.generate_content

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JUDGE_JSON_SCHEMA,
        )

        max_retries = max(1, self.judge_max_retries)

        for attempt in range(max_retries):
            started_at = time.time()
            try:
                response = await gen(
                    model=model,
                    contents=judge_prompt,
                    config=config,
                )
            except asyncio.CancelledError:
                schedule_model_usage(
                    context=self.context,
                    umo=umo,
                    provider=judge_provider,
                    provider_model=model,
                    source=MODEL_USAGE_SOURCE,
                    status="aborted",
                    started_at=started_at,
                    ended_at=time.time(),
                )
                raise
            except Exception as e:
                schedule_model_usage(
                    context=self.context,
                    umo=umo,
                    provider=judge_provider,
                    provider_model=model,
                    source=MODEL_USAGE_SOURCE,
                    status="error",
                    started_at=started_at,
                    ended_at=time.time(),
                )
                if attempt < max_retries - 1:
                    logger.warning(f"结构化输出失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                else:
                    logger.warning(f"结构化输出重试 {max_retries} 次后仍失败: {e}")
                continue

            schedule_model_usage(
                context=self.context,
                umo=umo,
                provider=judge_provider,
                provider_model=model,
                source=MODEL_USAGE_SOURCE,
                response=response,
                status="completed",
                started_at=started_at,
                ended_at=time.time(),
            )
            try:
                judge_data = json.loads(response.text)
                if attempt > 0:
                    logger.info(f"结构化输出在第 {attempt + 1} 次尝试成功: {judge_data}")
                else:
                    logger.debug(f"结构化输出成功: {judge_data}")
                return judge_data
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"结构化输出 JSON 解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                else:
                    logger.warning(f"结构化输出 JSON 解析重试 {max_retries} 次后仍失败: {e}")
        return None

    async def _judge_with_text_chat(
        self,
        judge_provider,
        judge_prompt: str,
        persona_system_prompt: str,
        umo: str,
    ) -> dict | None:
        """使用 text_chat 进行判断（适用于非 Google 提供商）。

        返回解析后的 dict，或在失败时返回 None。
        底层 AstrBot 会自动重试 HTTP 错误（最多5次），此处失败即放弃该 provider。
        """
        # 构建完整的判断提示词
        complete_judge_prompt = "你是一个专业的群聊回复决策系统，能够准确判断消息价值和回复时机。"
        if persona_system_prompt:
            complete_judge_prompt += f"\n\n你正在为以下角色的机器人做决策：\n{persona_system_prompt}"
        complete_judge_prompt += "\n\n**重要提醒：你必须严格按照JSON格式返回结果，不要包含任何其他内容！请不要进行对话，只返回JSON！**\n\n"
        complete_judge_prompt += judge_prompt
        complete_judge_prompt += """\n\n请严格按以下JSON格式回复，不要添加任何其他内容：
{"relevance": 分数, "willingness": 分数, "social": 分数, "timing": 分数, "continuity": 分数, "reasoning": "详细分析原因"}
"""

        max_retries = max(1, self.judge_max_retries)

        for attempt in range(max_retries):
            started_at = time.time()
            try:
                llm_response = await judge_provider.text_chat(
                    prompt=complete_judge_prompt,
                    contexts=[],  # 不传对话历史，防止角色扮演污染
                    image_urls=[],
                )
            except asyncio.CancelledError:
                schedule_model_usage(
                    context=self.context,
                    umo=umo,
                    provider=judge_provider,
                    source=MODEL_USAGE_SOURCE,
                    status="aborted",
                    started_at=started_at,
                    ended_at=time.time(),
                )
                raise
            except Exception as e:
                schedule_model_usage(
                    context=self.context,
                    umo=umo,
                    provider=judge_provider,
                    source=MODEL_USAGE_SOURCE,
                    status="error",
                    started_at=started_at,
                    ended_at=time.time(),
                )
                # HTTP 错误底层已重试过，直接放弃
                logger.warning(f"text_chat 调用失败: {e}")
                return None

            schedule_model_usage(
                context=self.context,
                umo=umo,
                provider=judge_provider,
                source=MODEL_USAGE_SOURCE,
                response=llm_response,
                status="completed",
                started_at=started_at,
                ended_at=time.time(),
            )
            try:
                content = llm_response.completion_text.strip()
                logger.debug(f"小参数模型原始返回内容: {content[:200]}...")

                return _extract_json(content)

            except Exception as e:
                # HTTP 成功但返回非 JSON，底层不会重试，由上层重试
                if attempt < max_retries - 1:
                    logger.warning(f"text_chat JSON 解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                else:
                    logger.warning(f"text_chat JSON 解析重试 {max_retries} 次后仍失败: {e}")
        return None

    def _record_raw_message(self, event: AstrMessageEvent, is_bot: bool = False) -> RawMessage:
        """将消息写入原始消息缓冲区"""
        umo = event.unified_msg_origin
        if umo not in self._raw_msg_buffer:
            self._raw_msg_buffer[umo] = deque(maxlen=self._raw_msg_buffer_size)
        raw_message = RawMessage(
            sender_name=event.get_sender_name(),
            sender_id=str(event.get_sender_id()),
            content=event.message_str,
            timestamp=time.time(),
            is_bot=is_bot,
        )
        self._raw_msg_buffer[umo].append(raw_message)
        return raw_message

    def _get_raw_buffer(self, umo: str) -> list[RawMessage]:
        """获取缓冲区中的消息列表（时间顺序）"""
        return list(self._raw_msg_buffer.get(umo, []))

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=1000)
    async def on_group_message(self, event: AstrMessageEvent):
        """群聊消息处理入口"""

        # 检查基本条件
        if not self._should_process_message(event):
            return

        # 第一时间记录原始消息，无论是否最终触发 LLM
        raw_message = self._record_raw_message(event, is_bot=False)

        try:
            # 防抖模式：消息进入防抖窗口，await 判断结果 Future
            if self.debounce_seconds > 0:
                await self._handle_message_with_debounce(event, raw_message)
                return

            # 非防抖模式：直接判断单条消息
            judge_result = await self.judge_with_tiny_model(event)

            if judge_result.should_reply:
                logger.info(f"🔥 心流触发主动回复 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f} | {judge_result.reasoning[:50]}...")

                event.is_at_or_wake_command = True
                event.set_extra("heartflow_triggered", True)

                self._update_active_state(event, judge_result)
                logger.info(f"💖 心流设置唤醒标志 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f} | {judge_result.reasoning[:50]}...")
                return
            else:
                logger.debug(f"心流判断不通过 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f} | 原因: {judge_result.reasoning[:30]}...")
                self._update_passive_state(event, judge_result)

        except Exception as e:
            logger.error(f"心流插件处理消息异常: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def _handle_message_with_debounce(
        self,
        event: AstrMessageEvent,
        raw_message: RawMessage | None = None,
    ):
        """防抖模式消息处理：加入 pending，启动/重置倒计时，并 await 判断结果 Future。

        每条消息的 on_group_message 都会阻塞在 await Future 上，
        直到倒计时结束、_judge_batch 完成并 set_result。
        判断通过则只让最后一条消息的 event 走 LLM，其余 stop_event。
        """
        umo = event.unified_msg_origin
        state = self._get_debounce_state(umo)

        raw_msg = raw_message or RawMessage(
            sender_name=event.get_sender_name(),
            sender_id=str(event.get_sender_id()),
            content=event.message_str,
            timestamp=time.time(),
            is_bot=False,
        )
        waiter = asyncio.get_running_loop().create_future()
        queued = QueuedMessage(raw_message=raw_msg, event=event, waiter=waiter)

        # 判断中收到的新消息：进入"下一批"，等旧批次结束后由 _promote_next_pending 接管
        if state.is_judging:
            state.next_pending.append(queued)
            if len(state.next_pending) > self.max_cached_messages:
                evicted = state.next_pending.pop(0)
                evicted.event.stop_event()
                if not evicted.waiter.done():
                    evicted.waiter.set_result(JudgeResult(
                        should_reply=False,
                        reasoning="下一批缓存已满，释放较旧消息",
                    ))
            logger.debug(f"判断中，消息进入下一批缓存 [{umo[:20]}...] 下一批 {len(state.next_pending)} 条")
        else:
            # 非判断中：进入 pending，启动倒计时
            state.pending.append(queued)
            logger.debug(f"防抖入队 [{umo[:20]}...] 当前积压 {len(state.pending)} 条")
            self._start_debounce_timer(umo)

        # shield 避免一條 event 被取消時連帶取消同批的結果。
        try:
            judge_result = await asyncio.shield(waiter)
        except asyncio.CancelledError:
            raise
        except Exception:
            event.stop_event()
            return

        self._apply_judge_result_to_event(event, judge_result)

    def _apply_judge_result_to_event(self, event: AstrMessageEvent, judge_result: JudgeResult):
        """根据判断结果决定当前 event 的去留。"""
        umo = event.unified_msg_origin
        if judge_result.should_reply:
            # 只让主 event（最后一条消息）走 LLM，其他 stop
            if judge_result.trigger_event is event:
                event.is_at_or_wake_command = True
                event.set_extra("heartflow_triggered", True)
                event.set_extra("heartflow_batch_messages", judge_result.related_messages)
                self._update_active_state(event, judge_result)
                logger.info(f"💖 心流设置唤醒标志 | {umo[:20]}... | 评分:{judge_result.overall_score:.2f} | {judge_result.reasoning[:50]}...")
                return  # pipeline 继续走 LLM
            else:
                event.stop_event()
                return
        else:
            event.stop_event()
            return

    def _get_debounce_state(self, umo: str) -> DebounceState:
        """获取或创建防抖状态"""
        if umo not in self._debounce_states:
            self._debounce_states[umo] = DebounceState()
        return self._debounce_states[umo]

    def _start_debounce_timer(self, umo: str, delay: float | None = None):
        """启动/重置防抖倒计时（当前批次）"""
        state = self._get_debounce_state(umo)
        if state.timer is not None:
            state.timer.cancel()
            logger.debug(f"防抖倒计时重置 [{umo[:20]}...] 剩余 {len(state.pending)} 条待判断")
        state.timer_generation += 1
        generation = state.timer_generation
        loop = asyncio.get_running_loop()
        state.timer = loop.call_later(
            self.debounce_seconds if delay is None else max(0.0, delay),
            lambda: asyncio.create_task(self._on_debounce_timer(umo, generation))
        )

    async def _on_debounce_timer(self, umo: str, generation: int | None = None):
        """倒计时回调：开始批量判断（当前批次）"""
        state = self._get_debounce_state(umo)
        if generation is not None and generation != state.timer_generation:
            return
        state.timer = None
        if not state.pending:
            return

        # 冷却检查：冷却中则不判断，消息重新进入倒计时
        if self.min_reply_interval > 0:
            elapsed_seconds = self._get_seconds_since_last_reply(umo)
            if elapsed_seconds < self.min_reply_interval:
                remaining = self.min_reply_interval - elapsed_seconds
                logger.debug(f"冷却中，距上次回复还有 {remaining:.0f}s，延迟判断 [{umo[:20]}...]")
                self._start_debounce_timer(umo, delay=remaining)
                return

        queued_batch = state.pending
        state.pending = []
        state.is_judging = True
        state.judge_start_time = time.time()
        batch_items = [(item.raw_message, item.event) for item in queued_batch]

        logger.debug(f"防抖窗口结束 [{umo[:20]}...] 共 {len(batch_items)} 条消息，开始判断")

        judge_result = None
        try:
            state.judge_task = asyncio.create_task(self._judge_batch(umo, batch_items))
            judge_result = await state.judge_task
        except Exception as e:
            logger.error(f"批量判断异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            judge_result = JudgeResult(
                should_reply=False,
                reasoning=f"批量判断异常: {e}",
                batch_size=len(batch_items),
            )
        finally:
            state.is_judging = False
            state.judge_task = None
            if judge_result is None:
                judge_result = JudgeResult(
                    should_reply=False,
                    reasoning="判断被取消",
                    batch_size=len(batch_items),
                )
            if not judge_result.should_reply:
                self._update_passive_state_batch(umo, judge_result, len(batch_items))
            for queued_message in queued_batch:
                if not queued_message.waiter.done():
                    queued_message.waiter.set_result(judge_result)
            # 旧批次结束，若有 next_pending 则转为新批次
            self._promote_next_pending(umo)

    def _promote_next_pending(self, umo: str):
        """旧批次结束后，把 next_pending 提升为新批次并启动倒计时。"""
        state = self._get_debounce_state(umo)
        if not state.next_pending:
            return
        state.pending = state.next_pending
        state.next_pending = []
        logger.debug(f"下一批提升为新批次 [{umo[:20]}...] {len(state.pending)} 条，启动新倒计时")
        self._start_debounce_timer(umo)

    async def _judge_batch(self, umo: str, batch_items: list) -> JudgeResult:
        """批量判断一批消息，只返回 JudgeResult（不触发回复，不更新状态）"""
        if not batch_items:
            return JudgeResult(should_reply=False, reasoning="空批次")

        trigger_event = batch_items[-1][1]  # 最后一条消息的 event
        batch_raw_msgs = [item[0] for item in batch_items]

        # 获取人格提示词
        persona_system_prompt = await self._get_persona_system_prompt(trigger_event)
        logger.debug(f"小参数模型使用人格提示词: {'有' if persona_system_prompt else '无'} | 长度: {len(persona_system_prompt) if persona_system_prompt else 0}")

        # 构建判断上下文
        chat_state = self._get_chat_state(umo)
        chat_context = self._build_chat_context(trigger_event)
        recent_messages = self._get_recent_messages_for_batch(umo, batch_raw_msgs)
        last_bot_reply = self._get_last_bot_reply_by_umo(umo)

        # 构造 batch 待判断消息文本
        batch_lines = []
        for m in batch_raw_msgs:
            prefix = "[机器人]" if m.is_bot else f"[{m.sender_name}]"
            time_str = datetime.datetime.fromtimestamp(m.timestamp).strftime('%H:%M:%S')
            batch_lines.append(f"{prefix} {time_str}: {m.content}")
        batch_text = "\n".join(batch_lines)

        judge_prompt = f"""
你是群聊机器人的决策系统，需要判断是否应该主动回复以下消息。

## 机器人角色设定
{persona_system_prompt if persona_system_prompt else "默认角色：智能助手"}

## 当前群聊情况
- 群聊ID: {umo}
- 我的精力水平: {chat_state.energy:.1f}/1.0
- 上次发言: {self._get_minutes_since_last_reply(umo)}分钟前

## 群聊基本信息
{chat_context}

## 最近{self.context_messages_count}条对话历史
{recent_messages}

## 上次机器人回复
{last_bot_reply if last_bot_reply else "暂无上次回复记录"}

## 待判断消息（本次共 {len(batch_raw_msgs)} 条，是短时间内群里的连续对话）
{batch_text}

## 评估要求
请从以下5个维度评估（0-10分），**重要提醒：基于上述机器人角色设定来判断是否适合回复**。
以上 {len(batch_raw_msgs)} 条消息是短时间内群里的连续对话，请判断是否值得作为一整段参与，而非逐条评估。

1. **内容相关度**(0-10)：消息是否有趣、有价值、适合我回复
   - 考虑消息的质量、话题性、是否需要回应
   - 识别并过滤垃圾消息、无意义内容
   - **结合机器人角色特点，判断是否符合角色定位**

2. **回复意愿**(0-10)：基于当前状态，我回复此消息的意愿
   - 考虑当前精力水平和心情状态
   - 考虑今日回复频率控制
   - **基于机器人角色设定，判断是否应该主动参与此话题**

3. **社交适宜性**(0-10)：在当前群聊氛围下回复是否合适
   - 考虑群聊活跃度和讨论氛围
   - **考虑机器人角色在群中的定位和表现方式**

4. **时机恰当性**(0-10)：回复时机是否恰当
   - 考虑距离上次回复的时间间隔
   - 考虑消息的紧急性和时效性

5. **对话连贯性**(0-10)：当前消息与上次机器人回复的关联程度
   - 如果当前消息是对上次回复的回应或延续，应给高分
   - 如果当前消息与上次回复完全无关，给中等分数
   - 如果没有上次回复记录，给默认分数5分

**回复阈值**: {self.reply_threshold} (综合评分达到此分数才回复)
"""

        # 调用 provider fallback 链
        judge_data = await self._call_judge_providers(umo, judge_prompt, persona_system_prompt)

        if judge_data is None:
            logger.warning(f"所有判断方式均失败，返回保守默认分数 [{umo[:20]}...]")
            return JudgeResult(
                relevance=5.0, willingness=5.0, social=5.0,
                timing=5.0, continuity=5.0,
                reasoning="所有判断方式均失败，使用默认分数",
                should_reply=False, confidence=0.5, overall_score=0.5,
                batch_size=len(batch_items),
            )

        relevance = _clamp_score(judge_data.get("relevance", 0))
        willingness = _clamp_score(judge_data.get("willingness", 0))
        social = _clamp_score(judge_data.get("social", 0))
        timing = _clamp_score(judge_data.get("timing", 0))
        continuity = _clamp_score(judge_data.get("continuity", 0))

        overall_score = (
            relevance * self.weights["relevance"] +
            willingness * self.weights["willingness"] +
            social * self.weights["social"] +
            timing * self.weights["timing"] +
            continuity * self.weights["continuity"]
        ) / 10.0

        should_reply = overall_score >= self.reply_threshold

        judge_result = JudgeResult(
            relevance=relevance, willingness=willingness,
            social=social, timing=timing, continuity=continuity,
            reasoning=judge_data.get("reasoning", ""),
            should_reply=should_reply,
            confidence=overall_score,
            overall_score=overall_score,
            related_messages=batch_raw_msgs,
            trigger_event=trigger_event if should_reply else None,
            batch_size=len(batch_items),
        )

        if judge_result.should_reply:
            logger.info(f"🔥 心流触发主动回复 | {umo[:20]}... | 评分:{judge_result.overall_score:.2f} | {judge_result.reasoning[:50]}...")
        else:
            logger.debug(f"判断不通过 [{umo[:20]}...] 评分 {judge_result.overall_score:.2f}")

        return judge_result

    def _get_recent_messages_for_batch(self, umo: str, batch_msgs: list) -> str:
        """获取 batch 之前的历史消息（不含 batch 本身）。

        规则：从 _raw_msg_buffer 取最近 N 条（N=context_messages_count）。
        batch 消息已通过 _record_raw_message 写入 buffer 末尾，需排除。
        如果 batch 长度 M >= N，仍取 N 条历史（batch 不截断，全量在 prompt 另一段展示）。
        """
        msgs = self._get_raw_buffer(umo)
        batch_ids = {message.message_id for message in batch_msgs}
        batch_positions = [
            index for index, message in enumerate(msgs)
            if message.message_id in batch_ids
        ]
        if batch_positions:
            history = msgs[:min(batch_positions)]
        elif batch_msgs:
            first_batch_time = min(message.timestamp for message in batch_msgs)
            history = [message for message in msgs if message.timestamp < first_batch_time]
        else:
            history = msgs

        recent = history[-self.context_messages_count:] if len(history) > self.context_messages_count else history

        if not recent:
            return "暂无对话历史"

        lines = []
        for m in recent:
            prefix = "[机器人]" if m.is_bot else f"[{m.sender_name}]"
            lines.append(f"{prefix}: {m.content}")
        return "\n".join(lines)

    def _get_last_bot_reply_by_umo(self, umo: str) -> str | None:
        """从原始消息缓冲区获取上次机器人的回复内容。"""
        msgs = self._get_raw_buffer(umo)
        for m in reversed(msgs):
            if m.is_bot and m.content.strip():
                return m.content
        return None

    def _update_passive_state_batch(self, umo: str, judge_result: JudgeResult, batch_count: int):
        """更新被动状态（批量，未回复）"""
        chat_state = self._get_chat_state(umo)
        chat_state.total_messages += batch_count

        if self.energy_system_enabled:
            chat_state.energy = min(1.0, chat_state.energy + self.energy_recovery_rate)

        logger.debug(f"更新被动状态(批量) [{umo[:20]}...] | 消息+{batch_count} | 精力: {chat_state.energy:.2f} | 原因: {judge_result.reasoning[:30]}...")

    @filter.after_message_sent()
    async def on_after_message_sent(self, event: AstrMessageEvent):
        """在消息发送后将机器人的回复写入原始消息缓冲区，以便后续判断参考"""
        if not self.config.get("enable_heartflow", False):
            return

        result = event.get_result()
        if result is None or not result.chain:
            return

        # 提取回复的纯文本内容
        reply_text = "".join(
            comp.text for comp in result.chain if isinstance(comp, Plain)
        ).strip()
        if not reply_text:
            return

        umo = event.unified_msg_origin
        if umo not in self._raw_msg_buffer:
            self._raw_msg_buffer[umo] = deque(maxlen=self._raw_msg_buffer_size)
        self._raw_msg_buffer[umo].append(RawMessage(
            sender_name="bot",
            sender_id="bot",
            content=reply_text,
            timestamp=time.time(),
            is_bot=True,
        ))
        logger.debug(f"机器人回复已写入缓冲区: {umo[:20]}... | {reply_text[:40]}...")

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        """心流触发时，在 LLM 请求前注入一条提示，让大模型知道自己是主动参与群聊的"""
        if not event.get_extra("heartflow_triggered"):
            return
        if not req or not hasattr(req, "system_prompt"):
            return
        note = "（注意：本次是你主动参与群聊的，不是用户叫你。回复应自然随意，像普通群成员一样加入话题。）"
        req.system_prompt = (req.system_prompt or "") + "\n" + note

        batch_messages = event.get_extra("heartflow_batch_messages", [])
        prior_messages = batch_messages[:-1]
        if prior_messages and hasattr(req, "extra_user_content_parts"):
            lines = []
            for message in prior_messages:
                time_str = datetime.datetime.fromtimestamp(message.timestamp).strftime("%H:%M:%S")
                lines.append(f"[{message.sender_name}/{time_str}]: {message.content}")
            req.extra_user_content_parts.append(TextPart(
                text=(
                    "<system_reminder>以下是本次防抖窗口中、当前消息之前的群聊消息：\n"
                    + "\n".join(lines)
                    + "\n</system_reminder>"
                )
            ))

    def _should_process_message(self, event: AstrMessageEvent) -> bool:
        """检查是否应该处理这条消息"""

        # 检查插件是否启用
        if not self.config.get("enable_heartflow", False):
            return False

        # 跳过已经被其他插件或系统标记为唤醒的消息
        if event.is_at_or_wake_command:
            logger.debug(f"跳过已被标记为唤醒的消息: {event.message_str}")
            return False

        # 检查白名单
        if self.whitelist_enabled:
            if not self.chat_whitelist:
                logger.debug(f"白名单为空，跳过处理: {event.unified_msg_origin}")
                return False

            if event.unified_msg_origin not in self.chat_whitelist:
                logger.debug(f"群聊不在白名单中，跳过处理: {event.unified_msg_origin}")
                return False

        # 跳过机器人自己的消息
        if event.get_sender_id() == event.get_self_id():
            return False

        # 跳过空消息
        if not event.message_str or not event.message_str.strip():
            return False

        # 冷却时间校验：防止短时间内连续触发
        if self.min_reply_interval > 0 and self.debounce_seconds <= 0:
            minutes = self._get_minutes_since_last_reply(event.unified_msg_origin)
            elapsed_seconds = minutes * 60
            if elapsed_seconds < self.min_reply_interval:
                logger.debug(f"冷却中，距上次回复还有 {self.min_reply_interval - elapsed_seconds:.0f}s")
                return False

        return True

    def _get_chat_state(self, chat_id: str) -> ChatState:
        """获取群聊状态"""
        if chat_id not in self.chat_states:
            self.chat_states[chat_id] = ChatState()

        # 检查日期重置
        today = datetime.date.today().isoformat()
        state = self.chat_states[chat_id]

        if state.last_reset_date != today:
            state.last_reset_date = today
            if self.energy_system_enabled:
                # 每日重置时恢复一些精力
                state.energy = min(1.0, state.energy + 0.2)

        # 精力系统关闭时恒为 1.0
        if not self.energy_system_enabled:
            state.energy = 1.0
            return state

        # 基于时间流逝自然恢复精力，不改写真实的上次回复时间。
        now = time.time()
        if state.last_energy_update_time <= 0:
            state.last_energy_update_time = state.last_reply_time or now
        if state.last_energy_update_time > 0:
            elapsed_minutes = (now - state.last_energy_update_time) / 60.0
            time_recovery = elapsed_minutes * (self.energy_recovery_rate * 5)
            state.energy = min(1.0, state.energy + time_recovery)
            state.last_energy_update_time = now

        return state

    def _get_minutes_since_last_reply(self, chat_id: str) -> int:
        """获取距离上次回复的分钟数"""
        chat_state = self._get_chat_state(chat_id)

        if chat_state.last_reply_time == 0:
            return 999  # 从未回复过

        return int((time.time() - chat_state.last_reply_time) / 60)

    def _get_seconds_since_last_reply(self, chat_id: str) -> float:
        """获取距离上次回复的秒数，保留小数用于精确安排冷却。"""
        chat_state = self._get_chat_state(chat_id)
        if chat_state.last_reply_time == 0:
            return float("inf")
        return max(0.0, time.time() - chat_state.last_reply_time)

    def _get_recent_messages(self, event: AstrMessageEvent) -> str:
        """从原始消息缓冲区获取最近的消息历史（用于小参数模型判断）。

        包含所有群聊成员的消息，而非仅 LLM 处理过的消息。
        """
        msgs = self._get_raw_buffer(event.unified_msg_origin)
        # 排除当前这条消息（已被 _record_raw_message 写入），取之前的若干条
        if msgs and msgs[-1].content == event.message_str:
            msgs = msgs[:-1]
        recent = msgs[-self.context_messages_count:] if len(msgs) > self.context_messages_count else msgs

        if not recent:
            return "暂无对话历史"

        lines = []
        for m in recent:
            prefix = "[机器人]" if m.is_bot else f"[{m.sender_name}]"
            lines.append(f"{prefix}: {m.content}")
        return "\n".join(lines)

    def _get_last_bot_reply(self, event: AstrMessageEvent) -> str | None:
        """从原始消息缓冲区获取上次机器人的回复内容。"""
        msgs = self._get_raw_buffer(event.unified_msg_origin)
        for m in reversed(msgs):
            if m.is_bot and m.content.strip():
                return m.content
        return None

    def _build_chat_context(self, event: AstrMessageEvent) -> str:
        """构建群聊上下文摘要信息。"""
        chat_state = self._get_chat_state(event.unified_msg_origin)

        # 检查上次机器人回复后群里有没有人接话（评估回复质量）
        msgs = self._get_raw_buffer(event.unified_msg_origin)
        post_reply_engagement = ""
        found_bot = False
        user_msgs_after_bot = 0
        for m in reversed(msgs):
            if m.is_bot:
                found_bot = True
                break
            user_msgs_after_bot += 1
        if found_bot:
            if user_msgs_after_bot >= 3:
                post_reply_engagement = "（上次回复后群里进行了热烈讨论）"
            elif user_msgs_after_bot == 0:
                post_reply_engagement = "（上次回复后无人接话）"

        if chat_state.total_messages > 100:
            activity_level = "高"
        elif chat_state.total_messages > 20:
            activity_level = "中"
        else:
            activity_level = "低"

        context_info = f"最近活跃度: {activity_level}\n"
        context_info += f"历史回复率: {(chat_state.total_replies / max(1, chat_state.total_messages) * 100):.1f}%\n"
        context_info += f"当前时间: {datetime.datetime.now().strftime('%H:%M')}"

        if post_reply_engagement:
            context_info += f"\n回复效果: {post_reply_engagement}"
            
        return context_info

    def _update_active_state(self, event: AstrMessageEvent, judge_result: JudgeResult):
        """更新主动回复状态"""
        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)

        # 更新回复相关状态
        now = time.time()
        chat_state.last_reply_time = now
        chat_state.last_energy_update_time = now
        chat_state.total_replies += 1
        chat_state.total_messages += max(1, judge_result.batch_size)

        # 精力消耗（回复后精力下降）
        if self.energy_system_enabled:
            chat_state.energy = max(0.1, chat_state.energy - self.energy_decay_rate)

        logger.debug(f"更新主动状态: {chat_id[:20]}... | 精力: {chat_state.energy:.2f}")

    def _update_passive_state(self, event: AstrMessageEvent, judge_result: JudgeResult):
        """更新被动状态（未回复）"""
        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)

        # 更新消息计数
        chat_state.total_messages += 1

        # 精力恢复（不回复时精力缓慢恢复）
        if self.energy_system_enabled:
            chat_state.energy = min(1.0, chat_state.energy + self.energy_recovery_rate)

        logger.debug(f"更新被动状态: {chat_id[:20]}... | 精力: {chat_state.energy:.2f} | 原因: {judge_result.reasoning[:30]}...")

    # 管理员命令：查看心流状态
    @filter.command("heartflow")
    async def heartflow_status(self, event: AstrMessageEvent):
        """查看心流状态"""

        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)

        status_info = f"""
🔮 心流状态报告

📊 **当前状态**
- 群聊ID: {event.unified_msg_origin}
- 精力水平: {chat_state.energy:.2f}/1.0 {'🟢' if chat_state.energy > 0.7 else '🟡' if chat_state.energy > 0.3 else '🔴'}
- 上次回复: {self._get_minutes_since_last_reply(chat_id)}分钟前

📈 **历史统计**
- 总消息数: {chat_state.total_messages}
- 总回复数: {chat_state.total_replies}
- 回复率: {(chat_state.total_replies / max(1, chat_state.total_messages) * 100):.1f}%

⚙️ **配置参数**
- 回复阈值: {self.reply_threshold}
- 判断提供商链: {' -> '.join(self.judge_provider_chain) if self.judge_provider_chain else '(未配置)'}
- 最大重试次数: {self.judge_max_retries}
- 白名单模式: {'✅ 开启' if self.whitelist_enabled else '❌ 关闭'}
- 白名单群聊数: {len(self.chat_whitelist) if self.whitelist_enabled else 0}
- 防抖时间: {self.debounce_seconds}s
- 判断超时: {self.judge_timeout_seconds}s
- 精力系统: {'✅ 开启' if self.energy_system_enabled else '❌ 关闭'}
- 缓存消息上限: {self.max_cached_messages}

🎯 **评分权重**
- 内容相关度: {self.weights['relevance']:.0%}
- 回复意愿: {self.weights['willingness']:.0%}
- 社交适宜性: {self.weights['social']:.0%}
- 时机恰当性: {self.weights['timing']:.0%}
- 对话连贯性: {self.weights['continuity']:.0%}

🎯 **插件状态**: {'✅ 已启用' if self.config.get('enable_heartflow', False) else '❌ 已禁用'}
"""

        event.set_result(event.plain_result(status_info))

    # 管理员命令：重置心流状态
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("heartflow_reset")
    async def heartflow_reset(self, event: AstrMessageEvent):
        """重置心流状态"""

        chat_id = event.unified_msg_origin
        if chat_id in self.chat_states:
            del self.chat_states[chat_id]

        event.set_result(event.plain_result("✅ 心流状态已重置"))
        logger.info(f"心流状态已重置: {chat_id}")

    async def _get_persona_system_prompt(self, event: AstrMessageEvent) -> str:
        """获取当前对话的人格系统提示词"""
        try:
            persona_mgr = self.context.persona_manager

            # 获取当前对话，尝试拿到会话绑定的 persona_id
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            persona_id: str | None = None
            if curr_cid:
                conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
                if conversation:
                    persona_id = conversation.persona_id

            # 用户显式取消人格
            if persona_id == "[%None]":
                return ""

            if persona_id:
                # 直接通过 PersonaManager 查询数据库
                try:
                    persona = await persona_mgr.get_persona(persona_id)
                    return persona.system_prompt or ""
                except ValueError:
                    logger.debug(f"未找到人格 {persona_id}，回退到默认人格")

            # 无 persona_id 或查询失败，使用默认人格
            default_persona = await persona_mgr.get_default_persona_v3(event.unified_msg_origin)
            return default_persona.get("prompt", "")

        except Exception as e:
            logger.debug(f"获取人格系统提示词失败: {e}")
            return ""
