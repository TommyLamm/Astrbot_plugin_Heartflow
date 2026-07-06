# Provider Timeout Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure a hanging judge provider times out independently so later providers in the configured fallback chain still run.

**Architecture:** Move `asyncio.wait_for` from the whole `_judge_batch` operation into `_call_judge_providers`, wrapping one provider attempt at a time. Preserve the debounce state machine and its owned judge task, while adding provider-path and elapsed-time logs and synchronizing configuration documentation with the new per-provider timeout semantics.

**Tech Stack:** Python 3.10+, asyncio, pytest/AnyIO, AstrBot provider APIs, JSON configuration schema.

---

## File structure

- Modify `main.py`: provider-level timeout, fallback continuation, elapsed-time logs, removal of whole-batch timeout handling.
- Modify `tests/test_debounce.py`: fallback and debounce integration regression tests.
- Modify `_conf_schema.json`: per-provider timeout wording.
- Modify `README.md`: timeout behavior and worst-case chain duration.

### Task 1: Reproduce fallback starvation

**Files:**
- Test: `tests/test_debounce.py`

- [ ] **Step 1: Write a failing provider-chain test**

Add an async test with two fake providers. The first provider coroutine sleeps beyond `judge_timeout_seconds`; the second returns a valid judge dictionary. Assert that `_call_judge_providers` returns the second result, records both calls in order, and cancels the first coroutine.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_debounce.py -k provider_timeout_allows_fallback -q`

Expected: FAIL because `_call_judge_providers` currently awaits the first provider without an individual timeout.

- [ ] **Step 3: Write all-provider-timeout coverage**

Add a test with two hanging providers and assert the method returns `None` within approximately `2 × judge_timeout_seconds`, with both provider coroutines cancelled and no pending task left behind.

- [ ] **Step 4: Verify RED**

Run: `python -m pytest tests/test_debounce.py -k all_provider_timeouts -q`

Expected: FAIL or hit the test safety timeout because no provider-level deadline exists.

### Task 2: Implement per-provider timeout

**Files:**
- Modify: `main.py:325-363,612-678`
- Test: `tests/test_debounce.py`

- [ ] **Step 1: Wrap each provider invocation**

In `_call_judge_providers`, choose the structured-output or text-chat coroutine, log its provider ID/path/position, and await it through `asyncio.wait_for(..., timeout=self.judge_timeout_seconds)`. Catch `asyncio.TimeoutError` separately, log elapsed time, and continue the loop.

- [ ] **Step 2: Remove the whole-chain timeout**

In `_on_debounce_timer`, continue creating `state.judge_task` but await it directly. Retain exception-to-conservative-result handling, waiter resolution, passive statistics, and next-batch promotion.

- [ ] **Step 3: Verify GREEN**

Run: `python -m pytest tests/test_debounce.py -k "provider_timeout or all_provider_timeouts" -q`

Expected: all selected tests PASS without pending-task warnings.

- [ ] **Step 4: Verify debounce regressions**

Run: `python -m pytest tests/test_debounce.py -q`

Expected: all debounce tests PASS, including cancellation and next-batch promotion.

### Task 3: Synchronize user-facing documentation

**Files:**
- Modify: `_conf_schema.json:140-145`
- Modify: `README.md`

- [ ] **Step 1: Update schema wording**

Describe `judge_timeout_seconds` as the maximum time for one provider's complete judgment flow. State that timeout continues to the next fallback provider.

- [ ] **Step 2: Update README wording**

Document that worst-case provider wait is approximately `judge_timeout_seconds × configured provider count`, and that one timeout no longer cancels the full chain.

- [ ] **Step 3: Validate documentation**

Run a JSON parse of `_conf_schema.json`, then `git diff --check`.

Expected: JSON parses successfully and no whitespace errors are reported.

### Task 4: Full verification

**Files:**
- Verify: `main.py`, `_conf_schema.json`, `README.md`, `tests/test_debounce.py`

- [ ] **Step 1: Run complete tests**

Run: `python -m pytest -q`

Expected: all tests PASS with no warnings about destroyed or pending tasks.

- [ ] **Step 2: Compile Python sources**

Run: `python -m py_compile main.py test_gemma_structured_output.py tests/conftest.py tests/test_debounce.py`

Expected: exit code 0.

- [ ] **Step 3: Audit the final diff**

Run: `git diff --check` and inspect `git diff --stat` plus the relevant timeout code.

Expected: changes are limited to timeout behavior, regression tests, schema, and README documentation.
