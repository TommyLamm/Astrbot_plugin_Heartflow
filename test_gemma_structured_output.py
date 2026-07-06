"""测试 Gemma 4 模型的 structured output 支持。

用法:
  1. 设置环境变量 GEMINI_API_KEY，或在下方 API_KEY 变量填入密钥
  2. 运行: python test_gemma_structured_output.py
  3. 可选参数: python test_gemma_structured_output.py <model_name>

测试内容:
  - 直接调用 google-genai 的 AsyncClient (与 AstrBot provider.client 类型一致)
  - 使用 response_schema 强制 JSON 输出
  - 对比有无 reasoning 字段的表现
  - 测试同步 Client 和 AsyncClient 两种调用路径
"""

import asyncio
import json
import os
import sys
import traceback
from typing import Any

# ========== 配置区 ==========
# 在这里填入你的 API key，或设置环境变量 GEMINI_API_KEY
API_KEY = "GEMINI_API_KEY"

# 默认测试模型，可通过命令行参数覆盖
DEFAULT_MODEL = "gemma-4-26b-a4b-it"

# 测试用 prompt（模拟 Heartflow 插件的判断场景）
TEST_PROMPT = """你是群聊机器人的决策系统，需要判断是否应该主动回复以下消息。

## 机器人角色设定
张三，42岁，郑州农村人，县城面馆老板。性格热情耿直，吃苦耐劳，遇事乐观。

## 待判断消息
发送者: Tommy
内容: 123
时间: 18:24:05

## 评估要求
请从以下5个维度评估（0-10分）：
1. 内容相关度(0-10)：消息是否有趣、有价值、适合我回复
2. 回复意愿(0-10)：基于当前状态，我回复此消息的意愿
3. 社交适宜性(0-10)：在当前群聊氛围下回复是否合适
4. 时机恰当性(0-10)：回复时机是否恰当
5. 对话连贯性(0-10)：当前消息与上次机器人回复的关联程度
"""

# Schema 1: 不含 reasoning（Heartflow 插件当前使用的版本）
SCHEMA_NO_REASONING = {
    "type": "OBJECT",
    "properties": {
        "relevance": {"type": "INTEGER", "description": "内容相关度(0-10)"},
        "willingness": {"type": "INTEGER", "description": "回复意愿(0-10)"},
        "social": {"type": "INTEGER", "description": "社交适宜性(0-10)"},
        "timing": {"type": "INTEGER", "description": "时机恰当性(0-10)"},
        "continuity": {"type": "INTEGER", "description": "对话连贯性(0-10)"},
    },
    "required": ["relevance", "willingness", "social", "timing", "continuity"],
}

# Schema 2: 含 reasoning（用于对比测试）
SCHEMA_WITH_REASONING = {
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


def get_api_key() -> str:
    global API_KEY
    if API_KEY:
        return API_KEY
    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if env_key:
        return env_key
    print("ERROR: 未找到 API key。")
    print("请设置环境变量 GEMINI_API_KEY，或在脚本顶部 API_KEY 变量中填入密钥。")
    sys.exit(1)


def ensure_deps() -> None:
    try:
        import google.genai  # noqa
    except ImportError:
        print("google-genai 未安装，正在安装...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-genai"])
        print("安装完成。")


def print_separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def parse_response(response: Any) -> dict | None:
    """从 genai 响应中提取并解析 JSON。"""
    if not response:
        return None

    # response.text 会拼接所有非 thought part 的文本
    try:
        text = response.text
    except Exception as e:
        print(f"  [WARN] response.text 访问失败: {e}")
        text = None

    if text:
        text = text.strip()
        print(f"  [response.text] 长度: {len(text)}")
        print(f"  [response.text] 内容: {text[:500]}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试提取 {...}
            import re
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
            print(f"  [WARN] response.text 不是有效 JSON")

    # 回退：手动遍历 parts
    print("  [INFO] 尝试手动遍历 parts...")
    try:
        candidates = response.candidates
        if not candidates:
            return None
        parts = candidates[0].content.parts
        json_part = None
        thought_parts = []
        for p in parts:
            if getattr(p, "thought", False):
                thought_parts.append(p.text or "")
            elif p.text:
                if json_part is None:
                    json_part = p.text
        if thought_parts:
            print(f"  [thought] 共 {len(thought_parts)} 段思考，总长度: {sum(len(t) for t in thought_parts)}")
            print(f"  [thought] 首段预览: {thought_parts[0][:200] if thought_parts else '(空)'}")
        if json_part:
            print(f"  [json part] 长度: {len(json_part)}")
            print(f"  [json part] 内容: {json_part[:500]}")
            try:
                return json.loads(json_part)
            except json.JSONDecodeError:
                import re
                m = re.search(r"\{.*\}", json_part, re.DOTALL)
                if m:
                    return json.loads(m.group())
    except Exception as e:
        print(f"  [ERROR] 遍历 parts 失败: {e}")

    return None


async def run_async_client(api_key: str, model: str) -> None:
    """测试 AsyncClient 路径（AstrBot provider.client 实际使用的类型）。"""
    print_separator(f"AsyncClient 测试 | 模型: {model}")

    from google import genai
    from google.genai import types

    # 关键：AsyncClient 的 API 调用直接是 async，不需要 .aio
    client = genai.Client(api_key=api_key)

    print(f"  Client 类型: {type(client).__module__}.{type(client).__name__}")
    print(f"  hasattr(client, 'aio'): {hasattr(client, 'aio')}")
    print(f"  hasattr(client, 'models'): {hasattr(client, 'models')}")
    print(f"  hasattr(client, 'async_models'): {hasattr(client, 'async_models')}")

    # 探测可用的异步访问路径
    async_models = None
    if hasattr(client, 'aio'):
        async_models = client.aio.models
        print(f"  异步路径: client.aio.models (类型: {type(async_models).__name__})")
    elif hasattr(client, 'models') and hasattr(client.models, 'generate_content'):
        # 某些版本 Client.models.generate_content 本身就是 async
        async_models = client.models
        print(f"  异步路径: client.models (类型: {type(async_models).__name__})")
    else:
        print("  [ERROR] 找不到异步 generate_content 路径")
        return

    # 测试 1: 不含 reasoning 的 schema
    print("\n  --- 测试 1: schema 不含 reasoning ---")
    try:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SCHEMA_NO_REASONING,
        )
        response = await async_models.generate_content(
            model=model,
            contents=TEST_PROMPT,
            config=config,
        )
        result = parse_response(response)
        if result:
            print(f"  [结果] 解析成功: {result}")
            assert "relevance" in result, "缺少 relevance 字段"
            print("  [PASS] structured output (无 reasoning) 工作正常")
        else:
            print("  [FAIL] 未能解析出 JSON")
    except Exception as e:
        print(f"  [ERROR] 调用失败: {e}")
        traceback.print_exc()

    # 测试 2: 含 reasoning 的 schema
    print("\n  --- 测试 2: schema 含 reasoning ---")
    try:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SCHEMA_WITH_REASONING,
        )
        response = await async_models.generate_content(
            model=model,
            contents=TEST_PROMPT,
            config=config,
        )
        result = parse_response(response)
        if result:
            print(f"  [结果] 解析成功: {result}")
            if "reasoning" in result:
                reasoning_text = result.get("reasoning", "")
                print(f"  [reasoning] 长度: {len(reasoning_text)}")
                print(f"  [reasoning] 预览: {reasoning_text[:150]}")
            print("  [PASS] structured output (含 reasoning) 工作正常")
        else:
            print("  [FAIL] 未能解析出 JSON")
    except Exception as e:
        print(f"  [ERROR] 调用失败: {e}")
        traceback.print_exc()


async def run_plain_text_chat(api_key: str, model: str) -> None:
    """对比测试: 不用 structured output，纯 prompt 要求 JSON。"""
    print_separator(f"纯 text_chat 对比测试 | 模型: {model}")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    async_models = client.aio.models if hasattr(client, 'aio') else client.models

    prompt = TEST_PROMPT + """
**重要：请严格按以下JSON格式回复，不要添加任何其他内容：**
{"relevance": 分数, "willingness": 分数, "social": 分数, "timing": 分数, "continuity": 分数}
"""

    try:
        response = await async_models.generate_content(
            model=model,
            contents=prompt,
        )
        text = ""
        try:
            text = response.text
        except Exception:
            pass
        print(f"  [原始返回] 长度: {len(text) if text else 0}")
        print(f"  [原始返回] 内容: {text[:500] if text else '(空)'}")

        # 统计 thought parts
        try:
            parts = response.candidates[0].content.parts
            thought_count = sum(1 for p in parts if getattr(p, "thought", False))
            non_thought_count = len(parts) - thought_count
            print(f"  [parts] thought: {thought_count}, non-thought: {non_thought_count}")
        except Exception:
            pass

        if text:
            try:
                data = json.loads(text.strip().strip("`").removeprefix("json").strip())
                print(f"  [解析成功] {data}")
            except json.JSONDecodeError:
                import re
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    try:
                        data = json.loads(m.group())
                        print(f"  [正则提取成功] {data}")
                    except json.JSONDecodeError:
                        print("  [FAIL] 无法解析为 JSON")
                else:
                    print("  [FAIL] 未找到 JSON 结构")
    except Exception as e:
        print(f"  [ERROR] 调用失败: {e}")
        traceback.print_exc()


def main() -> None:
    ensure_deps()

    api_key = get_api_key()
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    print(f"API Key: {api_key[:8]}...{api_key[-4:]} (长度 {len(api_key)})")
    print(f"测试模型: {model}")

    asyncio.run(run_async_client(api_key, model))
    asyncio.run(run_plain_text_chat(api_key, model))

    print_separator("测试完成")


if __name__ == "__main__":
    main()
