# Heartflow 防抖循環修復設計

## 目標

讓防抖模式可靠地完成以下循環：收集消息、批量判斷、把完整批次交給主 LLM、判斷或回覆期間繼續收集下一批，並在逾時後仍能推進。

## 狀態與資料流

- 每個 `unified_msg_origin` 維持獨立的 `DebounceState`。
- 倒計時內的消息進入 `pending`，判斷開始後的新消息進入 `next_pending`。
- 判斷開始時建立並保存明確的判斷 task。watchdog 逾時時取消該 task、以保守結果釋放當前批次，並立即提升 `next_pending`。
- 判斷通過時，只有批次最後一個 event 繼續 AstrBot pipeline；其他 event 停止。
- 完整批次文字保存到觸發 event 的 extra。`on_llm_request` 使用 AstrBot 的 `ProviderRequest.extra_user_content_parts` 注入批次內容，不修改 `event.message_str`，也不依賴內建群聊 context 是否啟用。

## 上下文正確性

- 每條 `RawMessage` 帶有插件內唯一 ID。
- 取得批次之前的歷史時，按 ID 排除當前批次，而不是假設批次位於 buffer 尾端。這允許下一批消息在 async 人格查詢期間寫入 buffer，而不造成錯誤切片或重複。
- 注入主 LLM 的批次內容保留發送者、時間和文字；最後一條仍是 AstrBot 原始 prompt，不重複注入。

## 冷卻與統計

- 防抖啟用時，入口不因最短回覆間隔而丟棄消息；冷卻檢查移到批次倒計時回調。
- 冷卻期間保留 `pending`，倒計時延後到剩餘冷卻結束，而不是反覆固定延遲。
- 成功與未成功批次都按實際批次消息數更新 `total_messages`；成功批次只增加一次 `total_replies`。

## 邊界處理

- 使用 timer generation/token 忽略已失效但已排入 event loop 的舊倒計時回調，確保最後一條消息確實重置防抖窗口。
- `max_cached_messages` 截斷下一批時，被淘汰 event 立即停止並釋放，不繼續佔用共享 Future 的等待者。
- 插件卸載或單一 event 取消不應取消其他 event 共用的 Future；等待共用 Future 時使用 cancellation shielding。

## 測試

新增不需要真實 AstrBot provider 的單元測試，覆蓋：

1. 多條消息合併後只有最後 event 通過，且主 LLM 收到完整批次。
2. 判斷逾時會取消 task 並推進下一批。
3. 判斷期間新消息不污染當前批次之前的歷史。
4. 啟用冷卻時消息仍會入隊。
5. 成功批次統計使用完整批次大小。
6. 已失效 timer callback 不提前結束新窗口。

現有 Gemma 手動整合腳本不作為 pytest 單元測試收集；將避免其帶參數函數被 pytest 誤認為 fixture 測試。
