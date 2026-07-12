import asyncio
import json
import sys
import types
from types import SimpleNamespace

import pytest


class FakeDatabase:
    def __init__(self, fail=False):
        self.fail = fail
        self.records = []

    async def insert_provider_stat(self, **kwargs):
        self.records.append(kwargs)
        if self.fail:
            raise RuntimeError("database unavailable")


class FakeContext:
    def __init__(self, db, providers=None):
        self.db = db
        self.providers = providers or {}

    def get_db(self):
        return self.db

    def get_provider_by_id(self, provider_id):
        return self.providers.get(provider_id)


class FakeTextProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.provider_config = {"id": "judge-provider"}
        self.model_name = "judge-model"

    def get_model(self):
        return self.model_name

    async def text_chat(self, **_kwargs):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


async def drain_stats(plugin):
    while plugin._provider_stat_tasks:
        await asyncio.gather(*list(plugin._provider_stat_tasks))


def make_text_response(text, *, input_other=0, input_cached=0, output=0, usage=True):
    token_usage = None
    if usage:
        token_usage = SimpleNamespace(
            input_other=input_other,
            input_cached=input_cached,
            output=output,
        )
    return SimpleNamespace(completion_text=text, usage=token_usage)


@pytest.mark.anyio
async def test_text_chat_usage_is_written_to_astrbot_stats(plugin_factory):
    db = FakeDatabase()
    provider = FakeTextProvider(
        [
            make_text_response(
                '{"relevance": 8, "reasoning": "ok"}',
                input_other=120,
                input_cached=30,
                output=18,
            )
        ]
    )
    plugin = plugin_factory(
        context=FakeContext(db),
        judge_max_retries=1,
        _provider_stat_tasks=set(),
    )

    result = await plugin._judge_with_text_chat(provider, "prompt", "", "umo-1")
    await drain_stats(plugin)

    assert result["relevance"] == 8
    assert len(db.records) == 1
    record = db.records[0]
    assert record["umo"] == "umo-1"
    assert record["provider_id"] == "judge-provider"
    assert record["provider_model"] == "judge-model"
    assert record["status"] == "completed"
    assert record["agent_type"] == "internal"
    assert record["stats"]["token_usage"] == {
        "input_other": 120,
        "input_cached": 30,
        "output": 18,
    }
    assert record["stats"]["end_time"] >= record["stats"]["start_time"]


@pytest.mark.anyio
async def test_every_json_retry_records_its_token_usage(plugin_factory):
    db = FakeDatabase()
    provider = FakeTextProvider(
        [
            make_text_response("not json", input_other=50, output=5),
            make_text_response(
                '{"relevance": 7, "reasoning": "retry ok"}',
                input_other=60,
                output=6,
            ),
        ]
    )
    plugin = plugin_factory(
        context=FakeContext(db),
        judge_max_retries=2,
        _provider_stat_tasks=set(),
    )

    result = await plugin._judge_with_text_chat(provider, "prompt", "", "umo-2")
    await drain_stats(plugin)

    assert result["reasoning"] == "retry ok"
    assert len(db.records) == 2
    assert [record["stats"]["token_usage"]["output"] for record in db.records] == [
        5,
        6,
    ]


@pytest.mark.anyio
async def test_missing_usage_still_records_completed_call(plugin_factory):
    db = FakeDatabase()
    provider = FakeTextProvider(
        [make_text_response('{"relevance": 6, "reasoning": "ok"}', usage=False)]
    )
    plugin = plugin_factory(
        context=FakeContext(db),
        judge_max_retries=1,
        _provider_stat_tasks=set(),
    )

    await plugin._judge_with_text_chat(provider, "prompt", "", "umo-3")
    await drain_stats(plugin)

    assert db.records[0]["stats"]["token_usage"] == {
        "input_other": 0,
        "input_cached": 0,
        "output": 0,
    }


@pytest.mark.anyio
async def test_provider_error_is_recorded_without_changing_fallback(plugin_factory):
    db = FakeDatabase()
    provider = FakeTextProvider([RuntimeError("provider unavailable")])
    plugin = plugin_factory(
        context=FakeContext(db),
        judge_max_retries=1,
        _provider_stat_tasks=set(),
    )

    result = await plugin._judge_with_text_chat(provider, "prompt", "", "umo-error")
    await drain_stats(plugin)

    assert result is None
    assert db.records[0]["status"] == "error"
    assert db.records[0]["stats"]["token_usage"] == {
        "input_other": 0,
        "input_cached": 0,
        "output": 0,
    }


@pytest.mark.anyio
async def test_provider_timeout_is_recorded_as_aborted(plugin_factory):
    class SlowProvider(FakeTextProvider):
        async def text_chat(self, **_kwargs):
            await asyncio.sleep(1)

    db = FakeDatabase()
    provider = SlowProvider([])
    context = FakeContext(db, {"slow": provider})
    plugin = plugin_factory(
        context=context,
        judge_provider_name="slow",
        judge_provider_chain=["slow"],
        judge_max_retries=1,
        judge_timeout_seconds=0.01,
        _provider_stat_tasks=set(),
    )
    plugin._is_google_provider = lambda _provider: False

    result = await plugin._call_judge_providers("umo-timeout", "prompt")
    await drain_stats(plugin)

    assert result is None
    assert db.records[0]["status"] == "aborted"
    assert db.records[0]["umo"] == "umo-timeout"


@pytest.mark.anyio
async def test_stats_write_failure_does_not_change_judge_result(plugin_factory):
    db = FakeDatabase(fail=True)
    provider = FakeTextProvider(
        [make_text_response('{"relevance": 9, "reasoning": "ok"}', output=4)]
    )
    plugin = plugin_factory(
        context=FakeContext(db),
        judge_max_retries=1,
        _provider_stat_tasks=set(),
    )

    result = await plugin._judge_with_text_chat(provider, "prompt", "", "umo-4")
    await drain_stats(plugin)

    assert result["relevance"] == 9
    assert len(db.records) == 1


@pytest.mark.anyio
async def test_google_structured_usage_is_written_to_astrbot_stats(
    plugin_factory,
    monkeypatch,
):
    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    genai_module.types = SimpleNamespace(GenerateContentConfig=GenerateContentConfig)
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)

    response = SimpleNamespace(
        text=json.dumps({"relevance": 8, "reasoning": "structured"}),
        usage_metadata=SimpleNamespace(
            prompt_token_count=90,
            cached_content_token_count=20,
            candidates_token_count=12,
        ),
    )

    class Models:
        async def generate_content(self, **_kwargs):
            return response

    provider = SimpleNamespace(
        client=SimpleNamespace(models=Models()),
        model_name="gemini-judge",
        provider_config={"id": "google-judge"},
        get_model=lambda: "gemini-judge",
    )
    db = FakeDatabase()
    plugin = plugin_factory(
        context=FakeContext(db),
        judge_max_retries=1,
        _provider_stat_tasks=set(),
    )

    result = await plugin._judge_with_structured_output(provider, "prompt", "umo-5")
    await drain_stats(plugin)

    assert result["reasoning"] == "structured"
    assert db.records[0]["stats"]["token_usage"] == {
        "input_other": 90,
        "input_cached": 20,
        "output": 12,
    }
