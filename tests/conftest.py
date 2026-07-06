import sys
import types
from dataclasses import dataclass, field

import pytest


class _Logger:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


class _Filter:
    class EventMessageType:
        GROUP_MESSAGE = "group"

    class PermissionType:
        ADMIN = "admin"

    def __getattr__(self, _name):
        return lambda *args, **kwargs: lambda func: func


class Plain:
    def __init__(self, text=""):
        self.text = text


class TextPart:
    def __init__(self, text=""):
        self.text = text


def _install_astrbot_stubs():
    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    star = types.ModuleType("astrbot.api.star")
    event = types.ModuleType("astrbot.api.event")
    components = types.ModuleType("astrbot.api.message_components")
    core = types.ModuleType("astrbot.core")
    agent = types.ModuleType("astrbot.core.agent")
    message = types.ModuleType("astrbot.core.agent.message")

    star.Star = type("Star", (), {})
    star.Context = object
    event.AstrMessageEvent = object
    event.filter = _Filter()
    components.Plain = Plain
    message.TextPart = TextPart
    api.logger = _Logger()
    astrbot.api = api

    sys.modules.update(
        {
            "astrbot": astrbot,
            "astrbot.api": api,
            "astrbot.api.star": star,
            "astrbot.api.event": event,
            "astrbot.api.message_components": components,
            "astrbot.core": core,
            "astrbot.core.agent": agent,
            "astrbot.core.agent.message": message,
        }
    )


_install_astrbot_stubs()

from main import HeartflowPlugin  # noqa: E402


@dataclass
class FakeEvent:
    message_str: str
    sender_name: str = "user"
    sender_id: str = "1"
    unified_msg_origin: str = "test:GroupMessage:group"
    is_at_or_wake_command: bool = False
    stopped: bool = False
    extras: dict = field(default_factory=dict)

    def get_sender_name(self):
        return self.sender_name

    def get_sender_id(self):
        return self.sender_id

    def get_self_id(self):
        return "bot"

    def stop_event(self):
        self.stopped = True

    def set_extra(self, key, value):
        self.extras[key] = value

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)


@dataclass
class FakeRequest:
    prompt: str = "latest"
    system_prompt: str = ""
    extra_user_content_parts: list = field(default_factory=list)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def plugin_factory():
    def make(**overrides):
        plugin = object.__new__(HeartflowPlugin)
        plugin.config = {"enable_heartflow": True}
        plugin._debounce_states = {}
        plugin._raw_msg_buffer = {}
        plugin._raw_msg_buffer_size = 40
        plugin.chat_states = {}
        plugin.context_messages_count = 5
        plugin.judge_context_count = 5
        plugin.debounce_seconds = 0.01
        plugin.judge_timeout_seconds = 0.05
        plugin.min_reply_interval = 0
        plugin.whitelist_enabled = False
        plugin.chat_whitelist = []
        plugin.max_cached_messages = 10
        plugin.energy_system_enabled = True
        plugin.energy_decay_rate = 0.1
        plugin.energy_recovery_rate = 0.02
        for key, value in overrides.items():
            setattr(plugin, key, value)
        return plugin

    return make
