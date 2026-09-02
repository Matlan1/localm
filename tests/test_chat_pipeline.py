# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chat-pipeline hooks: the ChatPipeline registry/runners (unit) and the
inlet/stream/outlet chain wired through the real /v1/chat/completions handler
with a stub engine and a synthetic plugin (integration).

Open mode (no API key) so the chat route and the mounted plugin are reachable;
the engine is a MagicMock whose chat_stream echoes the last message content, so
an inlet transform is observable in the reply.
"""

import asyncio
import json
import textwrap

import pytest
from fastapi.testclient import TestClient

from localm.inference.chat_pipeline import ChatHookContext, ChatPipeline


def _ctx():
    return ChatHookContext(model_id="m", stream=False, request_id="req-1")


# --------------------------------------------------------------------------- #
#  Unit: ChatPipeline registry + runners                                      #
# --------------------------------------------------------------------------- #

def test_unknown_phase_rejected():
    p = ChatPipeline()
    with pytest.raises(ValueError):
        p.add_hook("middle", lambda m, c: m)


def test_non_callable_rejected():
    p = ChatPipeline()
    with pytest.raises(TypeError):
        p.add_hook("inlet", 123)


def test_has_and_empty_pass_through():
    p = ChatPipeline()
    assert not p.has("inlet")
    msgs = [{"role": "user", "content": "x"}]
    assert asyncio.run(p.run_inlet(msgs, _ctx())) is msgs       # unchanged object
    assert p.run_stream("tok", _ctx()) == "tok"
    assert asyncio.run(p.run_outlet("txt", msgs, _ctx())) == "txt"


def test_priority_then_registration_order():
    order = []
    p = ChatPipeline()
    p.add_hook("inlet", lambda m, c: (order.append("b"), m)[1], priority=10, plugin="b")
    p.add_hook("inlet", lambda m, c: (order.append("a"), m)[1], priority=0, plugin="a")
    p.add_hook("inlet", lambda m, c: (order.append("c"), m)[1], priority=10, plugin="c")
    asyncio.run(p.run_inlet([], _ctx()))
    # priority 0 first, then the two priority-10 hooks in registration order
    assert order == ["a", "b", "c"]


# --------------------------------------------------------------------------- #
#  thinking-model <think> nudge in regular chat                               #
# --------------------------------------------------------------------------- #

def _think_ctx(model_id):
    return ChatHookContext(model_id=model_id, stream=False, request_id="r")


def test_is_thinking_model_detection():
    from localm.inference.model_family import is_thinking_model
    for name in ("deepseek-r1-7b", "qwq-32b", "qwen3-8b", "magistral-small",
                 "Llama-3.3-8B-Instruct-Thinking-High-Reasoning"):
        assert is_thinking_model(name), name
    for name in ("llama3.1-8b", "mistral-7b", "gemma3", "phi-4-mini", ""):
        assert not is_thinking_model(name), name


def test_thinking_inlet_inserts_system_when_absent():
    from localm.plugins.builtin.chat.plug import _thinking_inlet, _THINK_INSTRUCTION
    msgs = [{"role": "user", "content": "hi"}]
    out = _thinking_inlet(msgs, _think_ctx("deepseek-r1-7b"))
    assert out is msgs and msgs[0]["role"] == "system"
    assert _THINK_INSTRUCTION in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_thinking_inlet_appends_to_existing_system():
    from localm.plugins.builtin.chat.plug import _thinking_inlet, _THINK_INSTRUCTION
    msgs = [{"role": "system", "content": "You are a poet."},
            {"role": "user", "content": "hi"}]
    out = _thinking_inlet(msgs, _think_ctx("qwen3-8b"))
    assert out is msgs and len(msgs) == 2          # no new message inserted
    assert msgs[0]["content"].startswith("You are a poet.")
    assert _THINK_INSTRUCTION in msgs[0]["content"]


def test_thinking_inlet_skips_when_already_instructed():
    # The coder case: a system prompt that already steers <think> must not be
    # doubled up.
    from localm.plugins.builtin.chat.plug import _thinking_inlet
    msgs = [{"role": "system", "content": "Reason inside <think>...</think> first."},
            {"role": "user", "content": "hi"}]
    before = [dict(m) for m in msgs]
    assert _thinking_inlet(msgs, _think_ctx("qwq-32b")) is None
    assert msgs == before                          # untouched


def test_thinking_inlet_noop_for_non_thinking_model():
    from localm.plugins.builtin.chat.plug import _thinking_inlet
    msgs = [{"role": "user", "content": "hi"}]
    before = [dict(m) for m in msgs]
    assert _thinking_inlet(msgs, _think_ctx("llama3.1-8b")) is None
    assert msgs == before


# --------------------------------------------------------------------------- #
#  role-word self-reference nudge (REC-D1-ROLEWORD) in regular chat           #
# --------------------------------------------------------------------------- #

def test_role_word_inlet_inserts_system_when_absent():
    from localm.plugins.builtin.chat.plug import _role_word_inlet, _ROLE_WORD_INSTRUCTION
    msgs = [{"role": "user", "content": "hi"}]
    out = _role_word_inlet(msgs, _think_ctx("llama3.1-8b"))
    assert out is msgs and msgs[0]["role"] == "system"
    assert _ROLE_WORD_INSTRUCTION in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_role_word_inlet_appends_to_existing_system():
    from localm.plugins.builtin.chat.plug import _role_word_inlet, _ROLE_WORD_INSTRUCTION
    msgs = [{"role": "system", "content": "You are a poet."},
            {"role": "user", "content": "hi"}]
    out = _role_word_inlet(msgs, _think_ctx("gemma3"))
    assert out is msgs and len(msgs) == 2          # no new message inserted
    assert msgs[0]["content"].startswith("You are a poet.")
    assert _ROLE_WORD_INSTRUCTION in msgs[0]["content"]


def test_role_word_inlet_skips_when_already_instructed():
    from localm.plugins.builtin.chat.plug import _role_word_inlet
    msgs = [{"role": "system", "content": "Never refer to yourself in the third person."},
            {"role": "user", "content": "hi"}]
    before = [dict(m) for m in msgs]
    assert _role_word_inlet(msgs, _think_ctx("llama3.1-8b")) is None
    assert msgs == before                          # untouched


def test_role_word_inlet_applies_regardless_of_model_family():
    from localm.plugins.builtin.chat.plug import _role_word_inlet, _ROLE_WORD_INSTRUCTION
    for model_id in ("deepseek-r1-7b", "llama3.1-8b", "mistral-7b", ""):
        msgs = [{"role": "user", "content": "hi"}]
        out = _role_word_inlet(msgs, _think_ctx(model_id))
        assert out is msgs
        assert _ROLE_WORD_INSTRUCTION in msgs[0]["content"], model_id


def test_thinking_and_role_word_inlets_compose_on_one_system_message():
    from localm.plugins.builtin.chat.plug import (
        _thinking_inlet, _role_word_inlet, _THINK_INSTRUCTION, _ROLE_WORD_INSTRUCTION,
    )
    msgs = [{"role": "user", "content": "hi"}]
    msgs = _thinking_inlet(msgs, _think_ctx("deepseek-r1-7b")) or msgs
    msgs = _role_word_inlet(msgs, _think_ctx("deepseek-r1-7b")) or msgs
    assert len([m for m in msgs if m["role"] == "system"]) == 1
    assert _THINK_INSTRUCTION in msgs[0]["content"]
    assert _ROLE_WORD_INSTRUCTION in msgs[0]["content"]


def test_failure_isolation_keeps_going_and_preserves_value():
    p = ChatPipeline()
    p.add_hook("inlet", lambda m, c: m + ["one"], priority=0, plugin="ok1")

    def boom(m, c):
        raise RuntimeError("boom")

    p.add_hook("inlet", boom, priority=1, plugin="bad")
    p.add_hook("inlet", lambda m, c: m + ["two"], priority=2, plugin="ok2")
    out = asyncio.run(p.run_inlet([], _ctx()))
    # the raising hook is skipped; the value from before it flows on
    assert out == ["one", "two"]


def test_sync_and_async_inlet_and_outlet():
    p = ChatPipeline()
    p.add_hook("inlet", lambda m, c: m + ["sync"], plugin="s")

    async def amod(m, c):
        return m + ["async"]

    p.add_hook("inlet", amod, plugin="a")
    assert asyncio.run(p.run_inlet([], _ctx())) == ["sync", "async"]

    async def aout(text, m, c):
        return text + "!"

    p.add_hook("outlet", aout, plugin="a")
    assert asyncio.run(p.run_outlet("hi", [], _ctx())) == "hi!"


def test_inplace_mutation_returning_none_is_respected():
    p = ChatPipeline()

    def mutate(m, c):
        m.append("added")            # mutate in place, return None

    p.add_hook("inlet", mutate, plugin="m")
    msgs = []
    out = asyncio.run(p.run_inlet(msgs, _ctx()))
    assert out == ["added"]


def test_stream_sync_transform():
    p = ChatPipeline()
    p.add_hook("stream", lambda t, c: t.upper(), plugin="up")
    assert p.run_stream("abc", _ctx()) == "ABC"


def test_async_stream_hook_is_skipped():
    p = ChatPipeline()

    async def a(t, c):
        return t.upper()

    p.add_hook("stream", a, plugin="bad")
    # async stream hooks are unsupported (hot path); the piece passes unchanged
    assert p.run_stream("abc", _ctx()) == "abc"


def test_remove_plugin_drops_only_that_plugins_hooks():
    p = ChatPipeline()
    p.add_hook("inlet", lambda m, c: m, plugin="keep")
    p.add_hook("inlet", lambda m, c: m, plugin="drop")
    p.add_hook("outlet", lambda t, m, c: t, plugin="drop")
    p.add_hook("stream", lambda t, c: t, plugin="keep")
    p.remove_plugin("drop")
    assert p.has("inlet") and p.has("stream")
    assert not p.has("outlet")                      # only drop had an outlet
    assert len(p._hooks["inlet"]) == 1
    assert p._hooks["inlet"][0].plugin == "keep"


# --------------------------------------------------------------------------- #
#  Integration: hooks through the real /v1/chat/completions handler           #
# --------------------------------------------------------------------------- #

@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _make_plugin(root, name, body):
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{name}"\nregister = "plug"\n',
        encoding="utf-8")
    (pdir / "plug.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return pdir


def _echo_engine():
    """Engine stub whose chat_stream yields the last message's text content, so
    an inlet that rewrites that content is visible in the reply."""
    from unittest.mock import MagicMock
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.count_tokens.return_value = 5

    def _stream(messages, **kw):
        last = messages[-1].get("content", "") if messages else ""
        if not isinstance(last, str):
            last = " ".join(p.get("text", "") for p in (last or [])
                            if p.get("type") == "text")
        return iter([last])

    engine.chat_stream.side_effect = _stream
    type(engine).loaded = property(lambda self: True)
    engine.supports_images = True
    return engine


HOOK_PLUGIN = '''
    def _inlet(messages, ctx):
        if messages and isinstance(messages[-1].get("content"), str):
            messages[-1]["content"] = "INLET:" + messages[-1]["content"]
        return messages

    def _stream(token, ctx):
        return token.upper()

    def _outlet(text, messages, ctx):
        return text + " [OUT]"

    def register(host):
        host.register_chat_hook("inlet", _inlet)
        host.register_chat_hook("stream", _stream)
        host.register_chat_hook("outlet", _outlet)

    def unregister():
        pass
'''

BAD_INLET_PLUGIN = '''
    def _boom(messages, ctx):
        raise RuntimeError("inlet exploded")

    def register(host):
        host.register_chat_hook("inlet", _boom)

    def unregister():
        pass
'''


def _install(app, plugins_root, name):
    from localm.plugins.engine import PluginManager
    mgr = PluginManager(app, external_root=plugins_root, builtin_root=None)
    mgr.discover()
    mgr.install(name)
    return mgr


def _post(client, payload):
    return client.post("/v1/chat/completions", json=payload)


def test_pipeline_present_on_created_app(env):
    from localm.inference.http_server import create_app
    app = create_app(_echo_engine())
    assert isinstance(app.state.chat_pipeline, ChatPipeline)


def test_no_hooks_is_unchanged(env):
    from localm.inference.http_server import create_app
    app = create_app(_echo_engine())
    with TestClient(app) as c:
        r = _post(c, {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hi"


def test_inlet_and_outlet_transform_non_streaming(env):
    from localm.inference.http_server import create_app
    app = create_app(_echo_engine())
    _make_plugin(env / "plugins", "hookplug", HOOK_PLUGIN)
    _install(app, env / "plugins", "hookplug")
    with TestClient(app) as c:
        r = _post(c, {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    # echo returns the inlet-rewritten content ("INLET:hi"); outlet appends.
    assert r.json()["choices"][0]["message"]["content"] == "INLET:hi [OUT]"


def test_stream_hook_transforms_streamed_tokens(env):
    from localm.inference.http_server import create_app
    app = create_app(_echo_engine())
    _make_plugin(env / "plugins", "hookplug", HOOK_PLUGIN)
    _install(app, env / "plugins", "hookplug")
    with TestClient(app) as c:
        r = _post(c, {"model": "m", "stream": True,
                      "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    content = ""
    for line in r.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        delta = json.loads(line[len("data: "):])["choices"][0]["delta"]
        if delta.get("content"):
            content += delta["content"]
    # inlet rewrites to "INLET:hi"; the stream hook upper-cases each piece
    assert content == "INLET:HI"


def test_disable_stops_hooks_without_restart(env):
    from localm.inference.http_server import create_app
    app = create_app(_echo_engine())
    _make_plugin(env / "plugins", "hookplug", HOOK_PLUGIN)
    mgr = _install(app, env / "plugins", "hookplug")
    with TestClient(app) as c:
        first = _post(c, {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert first.json()["choices"][0]["message"]["content"] == "INLET:hi [OUT]"

    mgr.disable("hookplug")
    with TestClient(app) as c:
        after = _post(c, {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert after.status_code == 200
    assert after.json()["choices"][0]["message"]["content"] == "hi"


def test_failing_inlet_is_isolated(env):
    from localm.inference.http_server import create_app
    app = create_app(_echo_engine())
    _make_plugin(env / "plugins", "badplug", BAD_INLET_PLUGIN)
    _install(app, env / "plugins", "badplug")
    with TestClient(app) as c:
        r = _post(c, {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    # a hook that raises is skipped; the turn still succeeds with original content
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hi"
