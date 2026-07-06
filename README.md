# 心流插件（Heartflow）

Heartflow 是一個 AstrBot 群聊主動回覆插件。它使用獨立的小模型判斷是否值得參與對話，再交由 AstrBot 目前配置的主 LLM 生成實際回覆。

## 功能

- 雙 LLM 架構：判斷模型負責決策，主 LLM 負責生成回覆。
- 消息防抖：短時間內的連續消息會合併為一個批次判斷。
- 連續循環：判斷或回覆期間收到的新消息會進入下一批，不會阻塞後續循環。
- 完整批次上下文：判斷通過後，主 LLM 會收到同一防抖窗口內的完整文字消息。
- Provider fallback：主判斷模型失敗時，可依序嘗試最多五個備用 provider。
- 群聊隔離：每個 `unified_msg_origin` 擁有獨立的防抖、統計及精力狀態。
- 白名單、最短回覆間隔、精力系統及自訂評分權重。
- Provider 級逾時保護：單一 provider 逾時後會取消該次呼叫並繼續 fallback 鏈。

## 工作流程

```text
群聊消息
  ↓
防抖窗口（新消息會重置倒計時）
  ↓
批量判斷模型（主 provider → fallback 1 → ... → fallback 5）
  ↓
未通過：停止事件並更新狀態
通過：最後一個 event 繼續 AstrBot pipeline
  ↓
把批次內較早的消息注入主 LLM，保留最後一條原始 prompt
  ↓
AstrBot 主 LLM 生成並發送回覆
```

當一批消息正在判斷時，新消息會存入下一批快取。舊批結束後，下一批會重新開始防抖倒計時。主 LLM 生成或發送回覆期間到達的消息也會進入新的循環。

## 判斷維度

判斷模型會按以下五個維度評分，每項範圍為 0–10：

1. 內容相關度（`relevance`）
2. 回覆意願（`willingness`）
3. 社交適宜性（`social`）
4. 時機恰當性（`timing`）
5. 對話連貫性（`continuity`）

加權結果會正規化為 0–1，達到 `reply_threshold` 才會觸發主 LLM。權重總和不等於 1 時，插件會自動正規化。

預設權重：

```python
{
    "relevance": 0.25,
    "willingness": 0.20,
    "social": 0.20,
    "timing": 0.15,
    "continuity": 0.20,
}
```

## 安裝與基本設定

1. 透過 AstrBot 插件管理介面安裝本插件，或將插件目錄放入 AstrBot 的插件目錄。
2. 重新載入插件或重啟 AstrBot。
3. 在 AstrBot 中準備至少一個可用的判斷模型 provider。
4. 設定 `judge_provider_name`。
5. 將 `enable_heartflow` 設為 `true`。

建議判斷模型使用延遲較低、JSON 輸出穩定的小模型。Google GenAI provider 會優先使用 structured output；其他 provider 會使用文字 JSON 解析流程。

## 配置參數

### 核心與 provider

| 參數 | 預設值 | 說明 |
| --- | ---: | --- |
| `enable_heartflow` | `false` | 啟用心流主動回覆。 |
| `judge_provider_name` | `""` | 主要判斷模型 provider ID。未設定時不會觸發。 |
| `judge_provider_fallback_1`、`judge_provider_fallback_2`、`judge_provider_fallback_3`、`judge_provider_fallback_4`、`judge_provider_fallback_5` | `""` | 依序嘗試的備用 provider ID；留空即略過。 |
| `judge_max_retries` | `3` | 單一 provider 的最大嘗試次數；`0` 或 `1` 都表示只嘗試一次。 |
| `reply_threshold` | `0.6` | 觸發回覆的綜合分數門檻，範圍 0–1。 |

### 防抖與頻率控制

| 參數 | 預設值 | 說明 |
| --- | ---: | --- |
| `debounce_seconds` | `5.0` | 收到消息後等待的秒數；期間有新消息便重置。設為 `0` 會停用防抖並逐條判斷。 |
| `judge_timeout_seconds` | `30.0` | 每個判斷 provider 的超時秒數。逾時後自動嘗試下一個 fallback；最壞等待時間約為此值乘以已配置 provider 數量。 |
| `max_cached_messages` | `10` | 判斷期間下一批可保留的消息數；超出時釋放最舊消息，只保留最近 N 條。 |
| `min_reply_interval_seconds` | `0` | 兩次主動觸發之間的最短秒數；`0` 表示不限制。防抖模式下，冷卻期間的消息仍會保留。 |

### 上下文、白名單與精力

| 參數 | 預設值 | 說明 |
| --- | ---: | --- |
| `context_messages_count` | `5` | 建立單條及批量判斷上下文時使用的最近消息數。 |
| `judge_context_count` | `10` | 原始消息緩衝容量的計算基準之一；目前實際判斷上下文條數仍由 `context_messages_count` 控制。 |
| `whitelist_enabled` | `false` | 僅處理白名單內的群聊。 |
| `chat_whitelist` | `[]` | 允許處理的完整群聊 SID 列表，可用 `/sid` 查詢。 |
| `energy_system_enabled` | `true` | 是否啟用回覆消耗及時間恢復機制。停用時精力固定為 1.0。 |
| `energy_decay_rate` | `0.1` | 每次主動回覆消耗的精力。 |
| `energy_recovery_rate` | `0.02` | 未回覆及時間經過時的精力恢復係數。 |

### 評分權重

| 參數 | 預設值 |
| --- | ---: |
| `judge_relevance` | `0.25` |
| `judge_willingness` | `0.20` |
| `judge_social` | `0.20` |
| `judge_timing` | `0.15` |
| `judge_continuity` | `0.20` |

## 白名單

啟用 `whitelist_enabled` 後，只有 `chat_whitelist` 中的群聊會進入判斷流程。

取得 SID：

1. 在目標群聊發送 `/sid`。
2. 複製 AstrBot 回傳的完整 SID。
3. 將 SID 加入 `chat_whitelist`。

白名單啟用但列表為空時，插件不會處理任何群聊。

## 管理命令

- `/heartflow`：顯示目前群聊的精力、統計、provider 鏈、防抖設定及評分權重。
- `/heartflow_reset`：重置目前群聊的心流狀態；需要管理員權限。

## 防抖行為說明

- 同一群聊的多條消息共用一個防抖窗口。
- 每次新消息都會使窗口重新倒數。
- 判斷只允許批次最後一個 event 繼續主 LLM pipeline，其餘 event 會停止。
- 主 LLM 的原始 prompt 仍是最後一條消息；較早的批次消息會作為額外使用者內容注入，因此不會重寫 event。
- 一個 event 被取消不會取消同批其他 event 的等待結果。
- 已失效的 timer callback 會被忽略，避免舊倒計時提前消耗新批次。
- 單一判斷 provider 超時或異常時會繼續下一個 fallback；全部 provider 均失敗才採保守策略不回覆。

## 故障排查

### 完全沒有回覆

- 確認 `enable_heartflow = true`。
- 確認 `judge_provider_name` 是 AstrBot 中有效的 provider ID。
- 檢查 fallback provider 是否配置正確。
- 若啟用白名單，確認完整 SID 已加入 `chat_whitelist`。
- 檢查 `reply_threshold` 是否過高。
- 查看日誌中是否出現 provider、JSON 解析或判斷逾時錯誤。

### 回覆過於頻繁

- 提高 `reply_threshold`。
- 增加 `min_reply_interval_seconds`。
- 增加 `energy_decay_rate`。
- 延長 `debounce_seconds`，讓連續消息合併判斷。

### 回覆只像在回答最後一條消息

- 確認正在使用包含本次防抖修復的版本。
- 查看日誌是否顯示批量判斷。
- 確認主 LLM provider 支援 AstrBot 的 `extra_user_content_parts`。

### 下一批長時間沒有處理

- 檢查 `judge_timeout_seconds` 及已配置 provider 數量；整條鏈最壞等待時間約為兩者乘積。
- 檢查判斷 provider 的網路狀態及回應延遲。
- 查看是否仍處於 `min_reply_interval_seconds` 冷卻期。

## 開發與測試

執行本地回歸測試：

```bash
python -m pytest -q
python -m py_compile main.py test_gemma_structured_output.py tests/conftest.py tests/test_debounce.py
```

`test_gemma_structured_output.py` 是需要 API key 的手動整合測試，不會被 pytest 自動收集。

## 授權

本插件依 [LICENSE](LICENSE) 所列條款授權。
