import asyncio
import time
from collections import deque

import pytest

from main import ChatState, JudgeResult, RawMessage
from conftest import FakeEvent, FakeRequest


async def wait_until(predicate, timeout=0.2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not reached before timeout")
        await asyncio.sleep(0.001)


def test_harness_loads_plugin(plugin_factory):
    plugin = plugin_factory()
    assert plugin.debounce_seconds == 0.01


@pytest.mark.anyio
async def test_batch_reaches_reply_llm_without_rewriting_prompt(plugin_factory):
    plugin = plugin_factory()
    events = [
        FakeEvent("first", sender_name="alice", sender_id="1"),
        FakeEvent("second", sender_name="bob", sender_id="2"),
        FakeEvent("latest", sender_name="carol", sender_id="3"),
    ]
    messages = [
        RawMessage("alice", "1", "first", 1.0),
        RawMessage("bob", "2", "second", 2.0),
        RawMessage("carol", "3", "latest", 3.0),
    ]
    result = JudgeResult(
        should_reply=True,
        overall_score=0.9,
        reasoning="reply",
        related_messages=messages,
        trigger_event=events[-1],
    )

    for event in events:
        plugin._apply_judge_result_to_event(event, result)

    assert events[0].stopped is True
    assert events[1].stopped is True
    assert events[2].stopped is False
    assert events[2].get_extra("heartflow_batch_messages") == messages

    request = FakeRequest(prompt="latest")
    await plugin.on_llm_request(events[2], request)

    assert request.prompt == "latest"
    assert len(request.extra_user_content_parts) == 1
    injected = request.extra_user_content_parts[0].text
    assert "alice" in injected and "first" in injected
    assert "bob" in injected and "second" in injected
    assert "carol" not in injected and "latest" not in injected


@pytest.mark.anyio
async def test_provider_timeout_does_not_cancel_whole_batch(plugin_factory):
    plugin = plugin_factory(debounce_seconds=0.005, judge_timeout_seconds=0.01)
    calls = 0
    first_cancelled = asyncio.Event()

    async def judge(_umo, _items):
        nonlocal calls
        calls += 1
        if calls == 1:
            try:
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                first_cancelled.set()
                raise
        return JudgeResult(should_reply=False, reasoning="skip")

    plugin._judge_batch = judge
    first_event = FakeEvent("first")
    first = asyncio.create_task(plugin._handle_message_with_debounce(first_event))
    state = plugin._get_debounce_state(first_event.unified_msg_origin)
    await wait_until(lambda: state.is_judging)
    second = asyncio.create_task(plugin._handle_message_with_debounce(FakeEvent("second")))

    await asyncio.wait_for(asyncio.gather(first, second), timeout=0.3)

    assert first_cancelled.is_set() is False
    assert calls == 2


@pytest.mark.anyio
async def test_overflow_releases_evicted_event_immediately(plugin_factory):
    plugin = plugin_factory(
        debounce_seconds=0.005,
        judge_timeout_seconds=1,
        max_cached_messages=2,
    )
    release_first = asyncio.Event()
    calls = 0

    async def judge(_umo, _items):
        nonlocal calls
        calls += 1
        if calls == 1:
            await release_first.wait()
        return JudgeResult(should_reply=False, reasoning="skip")

    plugin._judge_batch = judge
    active_event = FakeEvent("active")
    active = asyncio.create_task(plugin._handle_message_with_debounce(active_event))
    state = plugin._get_debounce_state(active_event.unified_msg_origin)
    await wait_until(lambda: state.is_judging)
    overflow_events = [FakeEvent(str(index)) for index in range(3)]
    waiting = [
        asyncio.create_task(plugin._handle_message_with_debounce(event))
        for event in overflow_events
    ]
    await wait_until(lambda: len(state.next_pending) == 2)
    await asyncio.sleep(0)

    try:
        assert overflow_events[0].stopped is True
        assert waiting[0].done() is True
    finally:
        release_first.set()
        await asyncio.wait_for(asyncio.gather(active, *waiting), timeout=0.3)


@pytest.mark.anyio
async def test_cancelling_one_waiter_does_not_cancel_batch(plugin_factory):
    plugin = plugin_factory(debounce_seconds=0.02)

    async def judge(_umo, _items):
        return JudgeResult(should_reply=False, reasoning="skip")

    plugin._judge_batch = judge
    first_event = FakeEvent("first")
    second_event = FakeEvent("second")
    first = asyncio.create_task(plugin._handle_message_with_debounce(first_event))
    second = asyncio.create_task(plugin._handle_message_with_debounce(second_event))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    await asyncio.wait_for(second, timeout=0.2)
    assert second_event.stopped is True


@pytest.mark.anyio
async def test_stale_timer_generation_cannot_consume_reset_batch(plugin_factory):
    plugin = plugin_factory(debounce_seconds=1)
    calls = 0

    async def judge(_umo, _items):
        nonlocal calls
        calls += 1
        return JudgeResult(should_reply=False, reasoning="skip")

    plugin._judge_batch = judge
    event = FakeEvent("first")
    waiter = asyncio.create_task(plugin._handle_message_with_debounce(event))
    await asyncio.sleep(0)
    state = plugin._get_debounce_state(event.unified_msg_origin)
    stale_generation = state.timer_generation
    plugin._start_debounce_timer(event.unified_msg_origin)

    await plugin._on_debounce_timer(event.unified_msg_origin, stale_generation)
    assert calls == 0

    state.timer.cancel()
    await plugin._on_debounce_timer(
        event.unified_msg_origin,
        state.timer_generation,
    )
    await asyncio.wait_for(waiter, timeout=0.1)
    assert calls == 1


def test_batch_history_excludes_current_and_later_messages(plugin_factory):
    plugin = plugin_factory(context_messages_count=10)
    umo = "test:GroupMessage:group"
    old = RawMessage("old-user", "0", "old history", 1.0)
    first = RawMessage("alice", "1", "batch first", 2.0)
    second = RawMessage("bob", "2", "batch second", 3.0)
    later = RawMessage("carol", "3", "next batch", 4.0)
    plugin._raw_msg_buffer[umo] = deque([old, first, second, later], maxlen=40)

    history = plugin._get_recent_messages_for_batch(umo, [first, second])

    assert "old history" in history
    assert "batch first" not in history
    assert "batch second" not in history
    assert "next batch" not in history


def test_debounce_mode_queues_messages_during_reply_cooldown(plugin_factory):
    plugin = plugin_factory(debounce_seconds=1, min_reply_interval=60)
    event = FakeEvent("during reply")
    plugin.chat_states[event.unified_msg_origin] = ChatState(
        last_reply_time=time.time(),
    )

    assert plugin._should_process_message(event) is True


@pytest.mark.anyio
async def test_cooldown_reschedules_for_actual_remaining_seconds(plugin_factory):
    plugin = plugin_factory(debounce_seconds=100, min_reply_interval=10)
    event = FakeEvent("queued")
    plugin.chat_states[event.unified_msg_origin] = ChatState(
        last_reply_time=time.time() - 1,
    )
    waiter = asyncio.create_task(plugin._handle_message_with_debounce(event))
    await asyncio.sleep(0)
    state = plugin._get_debounce_state(event.unified_msg_origin)
    state.timer.cancel()
    captured_delays = []
    plugin._start_debounce_timer = (
        lambda _umo, delay=None: captured_delays.append(delay)
    )

    await plugin._on_debounce_timer(
        event.unified_msg_origin,
        state.timer_generation,
    )

    assert len(captured_delays) == 1
    assert 8 <= captured_delays[0] <= 10
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter


def test_energy_recovery_does_not_change_last_reply_time(plugin_factory):
    plugin = plugin_factory()
    umo = "test:GroupMessage:group"
    reply_time = time.time() - 120
    plugin.chat_states[umo] = ChatState(
        energy=0.5,
        last_reply_time=reply_time,
    )

    plugin._get_chat_state(umo)

    assert plugin.chat_states[umo].last_reply_time == reply_time


def test_successful_batch_counts_every_message_once(plugin_factory):
    plugin = plugin_factory()
    event = FakeEvent("latest")
    result = JudgeResult(
        should_reply=True,
        reasoning="reply",
        trigger_event=event,
        batch_size=3,
    )

    plugin._apply_judge_result_to_event(event, result)

    state = plugin.chat_states[event.unified_msg_origin]
    assert state.total_messages == 3
    assert state.total_replies == 1


class FakeJudgeProvider:
    def __init__(self, result=None, delay=0):
        self.result = result
        self.delay = delay
        self.calls = 0
        self.cancelled = False

    async def run(self):
        self.calls += 1
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return self.result


class FakeProviderContext:
    def __init__(self, providers):
        self.providers = providers

    def get_provider_by_id(self, provider_id):
        return self.providers.get(provider_id)


@pytest.mark.anyio
async def test_provider_timeout_allows_fallback_to_succeed(plugin_factory):
    first = FakeJudgeProvider(delay=0.2)
    expected = {
        "relevance": 8,
        "willingness": 8,
        "social": 8,
        "timing": 8,
        "continuity": 8,
        "reasoning": "fallback succeeded",
    }
    second = FakeJudgeProvider(result=expected)
    plugin = plugin_factory(judge_timeout_seconds=0.02)
    plugin.context = FakeProviderContext({"first": first, "second": second})
    plugin.judge_provider_name = "first"
    plugin.judge_provider_chain = ["first", "second"]
    plugin._is_google_provider = lambda _provider: False

    async def call_text(provider, _prompt, _persona):
        return await provider.run()

    plugin._judge_with_text_chat = call_text

    result = await asyncio.wait_for(
        plugin._call_judge_providers("umo", "prompt"),
        timeout=0.15,
    )

    assert result == expected
    assert first.calls == 1
    assert first.cancelled is True
    assert second.calls == 1


@pytest.mark.anyio
async def test_all_provider_timeouts_return_none_without_pending_tasks(plugin_factory):
    first = FakeJudgeProvider(delay=0.2)
    second = FakeJudgeProvider(delay=0.2)
    plugin = plugin_factory(judge_timeout_seconds=0.01)
    plugin.context = FakeProviderContext({"first": first, "second": second})
    plugin.judge_provider_name = "first"
    plugin.judge_provider_chain = ["first", "second"]
    plugin._is_google_provider = lambda _provider: False

    async def call_text(provider, _prompt, _persona):
        return await provider.run()

    plugin._judge_with_text_chat = call_text
    started = time.monotonic()

    result = await asyncio.wait_for(
        plugin._call_judge_providers("umo", "prompt"),
        timeout=0.15,
    )

    assert result is None
    assert first.cancelled is True
    assert second.cancelled is True
    assert first.calls == 1 and second.calls == 1
    assert time.monotonic() - started < 0.1


@pytest.mark.anyio
@pytest.mark.parametrize("failure_mode", ["none", "exception"])
async def test_invalid_provider_result_continues_fallback(
    plugin_factory,
    failure_mode,
):
    first = FakeJudgeProvider()
    expected = {"relevance": 7, "reasoning": "second provider"}
    second = FakeJudgeProvider(result=expected)
    plugin = plugin_factory(judge_timeout_seconds=0.02)
    plugin.context = FakeProviderContext({"first": first, "second": second})
    plugin.judge_provider_name = "first"
    plugin.judge_provider_chain = ["first", "second"]
    plugin._is_google_provider = lambda _provider: False

    async def call_text(provider, _prompt, _persona):
        if provider is first:
            first.calls += 1
            if failure_mode == "exception":
                raise RuntimeError("provider failed")
            return None
        return await provider.run()

    plugin._judge_with_text_chat = call_text

    result = await plugin._call_judge_providers("umo", "prompt")

    assert result == expected
    assert first.calls == 1
    assert second.calls == 1
