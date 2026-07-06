# Heartflow Debounce Cycle Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make debounced message batches reach the reply LLM intact and keep later batches progressing across provider timeouts, cooldowns, and timer races.

**Architecture:** Keep one state machine per unified message origin. Queue entries receive independent waiter futures, while a cancellable judge task owns each active batch; the triggering event carries batch metadata into `on_llm_request`, which appends prior batch messages as a `TextPart` without rewriting the original prompt.

**Tech Stack:** Python 3.10+, asyncio, dataclasses, pytest/pytest-asyncio, AstrBot plugin APIs.

---

## File structure

- Modify `main.py`: debounce state machine, batch metadata injection, history identity, cooldown and statistics.
- Create `tests/conftest.py`: minimal AstrBot API stubs for isolated plugin unit tests.
- Create `tests/test_debounce.py`: async regression tests for each confirmed state-machine failure.
- Modify `test_gemma_structured_output.py`: keep the network script manual instead of accidental pytest collection.

### Task 1: Establish an isolated regression-test harness

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_debounce.py`
- Modify: `test_gemma_structured_output.py:170,248,314-315`

- [ ] **Step 1: Add AstrBot stubs and fake events**

Create decorator, logger, `Plain`, `TextPart`, fake event, fake request, and a `plugin_factory` fixture. Import the real `main.py` after installing stubs in `sys.modules`.

- [ ] **Step 2: Stop collecting manual provider calls**

Rename `test_async_client` and `test_plain_text_chat` to `run_async_client` and `run_plain_text_chat`, including their `__main__` calls.

- [ ] **Step 3: Verify the harness**

Run: `python -m pytest --collect-only -q`

Expected: collection succeeds and no fixture errors reference `api_key` or `model`.

### Task 2: Deliver the complete batch to the reply LLM

**Files:**
- Modify: `main.py:16-69,453-557,559-577,661-786,878-886`
- Test: `tests/test_debounce.py`

- [ ] **Step 1: Write failing tests**

Add tests asserting that a successful three-message batch lets only the last event continue, saves all batch messages on it, and appends the first two formatted messages to `req.extra_user_content_parts` while leaving `req.prompt` unchanged.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_debounce.py -k "batch_reaches_reply_llm" -q`

Expected: FAIL because the trigger event currently stores no batch metadata and `on_llm_request` only modifies the system prompt.

- [ ] **Step 3: Implement minimal batch metadata flow**

Return the buffered `RawMessage` from `_record_raw_message`, reuse it in the queue, set `JudgeResult.related_messages`, attach it to the trigger event, and append a `TextPart` containing all prior batch messages during `on_llm_request`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_debounce.py -k "batch_reaches_reply_llm" -q`

Expected: PASS.

### Task 3: Make timeout and queue ownership deterministic

**Files:**
- Modify: `main.py:56-69,508-659,788-802`
- Test: `tests/test_debounce.py`

- [ ] **Step 1: Write failing timeout, overflow, cancellation, and stale-timer tests**

Assert that a timed-out judge task receives cancellation and the next batch finishes on its own timeout cycle; an overflowed oldest event is stopped and released; cancellation of one event does not cancel another waiter; and a stale timer generation cannot consume a newly reset batch.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_debounce.py -k "timeout or overflow or cancellation or stale_timer" -q`

Expected: failures showing the active judge remains alive, shared futures are poisoned or retained, and stale callbacks lack generation checks.

- [ ] **Step 3: Implement per-entry waiters and cancellable judge ownership**

Introduce a queued-entry dataclass containing raw message, event, and waiter. Await waiters through `asyncio.shield`, broadcast each batch result, cancel and await the active judge task on timeout, then clear judging state and promote the next batch in `finally`. Add `timer_generation` and reject callbacks whose generation is no longer current.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_debounce.py -k "timeout or overflow or cancellation or stale_timer" -q`

Expected: PASS with no pending-task warnings.

### Task 4: Correct history, cooldown, and statistics

**Files:**
- Modify: `main.py:36-54,453-464,595-608,804-829,918-964,1030-1058`
- Test: `tests/test_debounce.py`

- [ ] **Step 1: Write failing correctness tests**

Assert that messages arriving during persona lookup do not appear in the prior-history block, cooldown-enabled debounce still queues incoming messages, cooldown scheduling uses the actual remaining seconds, and a successful N-message batch increments `total_messages` by N and `total_replies` by one.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_debounce.py -k "history or cooldown or statistics" -q`

Expected: failures from tail slicing, entry-time cooldown rejection, and single-message active accounting.

- [ ] **Step 3: Implement identity-based history and batch accounting**

Give each recorded raw message an ID and find the earliest batch ID in the buffer to establish the history cutoff. Bypass entry-time cooldown only in debounce mode, schedule the timer for the precise remaining cooldown, separate energy-recovery timestamps from `last_reply_time`, and carry `batch_size` in `JudgeResult` for active accounting.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_debounce.py -k "history or cooldown or statistics" -q`

Expected: PASS.

### Task 5: Full verification

**Files:**
- Verify: `main.py`, `tests/test_debounce.py`, `test_gemma_structured_output.py`

- [ ] **Step 1: Run plugin regression tests**

Run: `python -m pytest -q`

Expected: all local tests pass without fixture or pending-task errors.

- [ ] **Step 2: Run syntax compilation**

Run: `python -m py_compile main.py test_gemma_structured_output.py tests/conftest.py tests/test_debounce.py`

Expected: exit code 0.

- [ ] **Step 3: Inspect the final diff**

Run: `git diff --check` and `git diff --stat`

Expected: no whitespace errors; changes are limited to the planned plugin, tests, and documentation files.
