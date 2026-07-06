# Provider 級 Timeout 與 Fallback 修復設計

## 問題

目前 `judge_timeout_seconds` 包住整個 `_judge_batch`。第一個 provider 若掛起到期限，整條判斷 task 會被取消，後續 fallback provider 永遠沒有執行機會。實際配置含三個 provider、timeout 30 秒時，第一個 NVIDIA provider 佔滿 30 秒後便終止整批。

## 核准方案

將 `judge_timeout_seconds` 的語義改為「每個 provider 的最大判斷時間」，不再作為整條 provider 鏈的共同期限。

每個 provider 依序執行：

1. 記錄 provider ID、序號、呼叫路徑及開始時間。
2. 使用 `asyncio.wait_for` 限制該 provider 的完整判斷流程，包括插件層重試。
3. 成功時記錄耗時並立即返回結果。
4. timeout、例外或無有效 JSON 時記錄耗時並繼續下一個 provider。
5. 全部 provider 均失敗後返回保守的不回覆結果。

三個 provider 且 timeout 為 30 秒時，最壞 provider 等待時間為 90 秒；人格與 prompt 建構時間不計入 provider timeout。

## 狀態機整合

- `_on_debounce_timer` 仍建立並持有 `judge_task`，以便插件 task 被外部取消時能清理。
- 移除 `_on_debounce_timer` 對整個 `_judge_batch` 的 30 秒 `wait_for`。
- provider 級 timeout 只取消當前 provider coroutine，不取消整批與下一批狀態。
- `_judge_batch` 完成或全部失敗後，沿用現有 waiter 廣播及 `next_pending` 提升流程。

## 日誌

每次 provider 呼叫至少輸出：

- 開始：provider ID、鏈中位置、structured output 或 text chat 路徑、timeout 秒數。
- 成功：provider ID、耗時。
- timeout：provider ID、耗時，並明確說明將嘗試下一個 provider。
- 無有效結果或例外：provider ID、耗時，並明確說明將嘗試下一個 provider。

日誌不得包含 API key、完整 prompt 或其他憑證。

## 配置與文件

- 保留配置名稱 `judge_timeout_seconds`，預設值維持 30 秒。
- 更新 `_conf_schema.json` 與 README，說明 timeout 是每個 provider 的期限，整條鏈的最壞時間約為 `timeout × provider 數量`。
- `judge_max_retries` 行為不在本次修改中重構；provider timeout 包住該 provider 的完整重試流程。

## 測試

1. 第一個 provider 掛起並 timeout 時，第二個 provider 仍會執行並可成功返回。
2. 第一個 provider 丟出例外或返回 `None` 時，下一個 provider 仍會執行。
3. 所有 provider timeout 時返回 `None`，不留下未完成 task。
4. `_on_debounce_timer` 不會在第一個 provider timeout 後取消整批。
5. 既有防抖、下一批提升、取消隔離及批次上下文測試保持通過。
