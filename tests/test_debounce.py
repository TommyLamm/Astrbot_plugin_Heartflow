import pytest

from main import JudgeResult, RawMessage
from conftest import FakeEvent, FakeRequest


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
