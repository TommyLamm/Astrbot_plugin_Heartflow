import asyncio
import json
import sys
import types
from types import SimpleNamespace

import pytest

import main


class FakeProvider:
    provider_config = {"id": "gemini_chat"}
    model_name = "gemma-4-31b-it"

    def __init__(self, results):
        self.results = iter(results)

    async def text_chat(self, **_kwargs):
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.mark.anyio
async def test_text_chat_records_each_json_attempt(plugin_factory, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "schedule_model_usage", lambda **kwargs: calls.append(kwargs))
    provider = FakeProvider(
        [
            SimpleNamespace(completion_text="not json", usage=object()),
            SimpleNamespace(completion_text=json.dumps({"relevance": 8}), usage=object()),
        ]
    )
    plugin = plugin_factory(context=object(), judge_max_retries=2)

    result = await plugin._judge_with_text_chat(provider, "prompt", "", "umo:test")

    assert result == {"relevance": 8}
    assert [call["status"] for call in calls] == ["completed", "completed"]
    assert all(call["response"] for call in calls)
    assert all(call["source"] == "heartflow" for call in calls)


@pytest.mark.anyio
async def test_text_chat_records_api_error(plugin_factory, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "schedule_model_usage", lambda **kwargs: calls.append(kwargs))
    plugin = plugin_factory(context=object(), judge_max_retries=1)

    result = await plugin._judge_with_text_chat(
        FakeProvider([RuntimeError("failed")]), "prompt", "", "umo:test"
    )

    assert result is None
    assert len(calls) == 1
    assert calls[0]["status"] == "error"
    assert "response" not in calls[0]


@pytest.mark.anyio
async def test_text_chat_records_cancellation_as_aborted(plugin_factory, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "schedule_model_usage", lambda **kwargs: calls.append(kwargs))

    class SlowProvider(FakeProvider):
        async def text_chat(self, **_kwargs):
            await asyncio.sleep(10)

    plugin = plugin_factory(context=object(), judge_max_retries=1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            plugin._judge_with_text_chat(SlowProvider([]), "prompt", "", "umo:test"),
            timeout=0.001,
        )

    assert len(calls) == 1
    assert calls[0]["status"] == "aborted"


@pytest.mark.anyio
async def test_google_structured_output_records_response(plugin_factory, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "schedule_model_usage", lambda **kwargs: calls.append(kwargs))

    genai = types.ModuleType("google.genai")
    genai.types = SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)

    response = SimpleNamespace(text='{"relevance": 9}', usage_metadata=object())

    async def generate_content(**_kwargs):
        return response

    provider = FakeProvider([])
    provider.client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )
    plugin = plugin_factory(context=object(), judge_max_retries=1)

    result = await plugin._judge_with_structured_output(provider, "prompt", "umo:test")

    assert result == {"relevance": 9}
    assert len(calls) == 1
    assert calls[0]["status"] == "completed"
    assert calls[0]["response"] is response
    assert calls[0]["provider_model"] == "gemma-4-31b-it"
