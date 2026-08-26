# SPDX-License-Identifier: AGPL-3.0-or-later
"""/v1/completions runs the same chat-pipeline hooks and the same audit log /
transcript that /v1/chat/completions runs on every turn, so raw-completion
traffic is not a parallel, unguarded path to the model.

These tests drive the real /v1/completions handler (streaming + non-streaming)
with a stub engine and a synthetic hook plugin, asserting the inlet/stream/outlet
hooks run, that the exchange is handed to _audit_exchange, and that the
text_completion response shape is unchanged.
"""

import json
import textwrap

import pytest
from fastapi.testclient import TestClient


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


def _echo_engine():
    """Engine stub whose chat_stream yields the last message's text content."""
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


def _make_plugin(root, name, body):
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{name}"\nregister = "plug"\n',
        encoding="utf-8")
    (pdir / "plug.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return pdir


def _install(app, plugins_root, name):
    from localm.plugins.engine import PluginManager
    mgr = PluginManager(app, external_root=plugins_root, builtin_root=None)
    mgr.discover()
    mgr.install(name)
    return mgr


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


def _post(client, payload):
    return client.post("/v1/completions", json={"model": "m", **payload})


def _sse_text(resp):
    out = ""
    for line in resp.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[len("data: "):])
        piece = chunk["choices"][0].get("text") or ""
        out += piece
    return out


# --------------------------------------------------------------------------- #
#  Response shape unchanged when no hooks are present                          #
# --------------------------------------------------------------------------- #

def test_no_hooks_completion_unchanged(env):
    from localm.inference.http_server import create_app
    app = create_app(_echo_engine())
    with TestClient(app) as c:
        r = _post(c, {"prompt": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "hi"


def test_no_hooks_completion_stream_unchanged(env):
    from localm.inference.http_server import create_app
    app = create_app(_echo_engine())
    with TestClient(app) as c:
        r = _post(c, {"prompt": "hi", "stream": True})
    assert r.status_code == 200
    assert _sse_text(r) == "hi"


# --------------------------------------------------------------------------- #
#  Hooks run on /v1/completions                                                #
# --------------------------------------------------------------------------- #

def test_completion_inlet_and_outlet_transform(env):
    from localm.inference.http_server import create_app
    app = create_app(_echo_engine())
    _make_plugin(env / "plugins", "hookplug", HOOK_PLUGIN)
    _install(app, env / "plugins", "hookplug")
    with TestClient(app) as c:
        r = _post(c, {"prompt": "hi"})
    assert r.status_code == 200
    # echo returns the inlet-rewritten prompt ("INLET:hi"); outlet appends.
    assert r.json()["choices"][0]["text"] == "INLET:hi [OUT]"


def test_completion_stream_hook_transforms(env):
    from localm.inference.http_server import create_app
    app = create_app(_echo_engine())
    _make_plugin(env / "plugins", "hookplug", HOOK_PLUGIN)
    _install(app, env / "plugins", "hookplug")
    with TestClient(app) as c:
        r = _post(c, {"prompt": "hi", "stream": True})
    assert r.status_code == 200
    # inlet rewrites to "INLET:hi"; the stream hook upper-cases each piece.
    assert _sse_text(r) == "INLET:HI"


def test_failing_inlet_isolated_completion(env):
    from localm.inference.http_server import create_app
    app = create_app(_echo_engine())
    _make_plugin(env / "plugins", "badplug", BAD_INLET_PLUGIN)
    _install(app, env / "plugins", "badplug")
    with TestClient(app) as c:
        r = _post(c, {"prompt": "hi"})
    assert r.status_code == 200
    assert r.json()["choices"][0]["text"] == "hi"   # raising hook skipped


# --------------------------------------------------------------------------- #
#  /v1/completions is audited                                                  #
# --------------------------------------------------------------------------- #

def _recorder(monkeypatch):
    import localm.inference.http_server as hs
    calls = []

    def _rec(audit, transcript, messages, reply):
        calls.append({"messages": messages, "reply": reply})
    monkeypatch.setattr(hs, "_audit_exchange", _rec)
    return calls


def test_completion_audited_non_streaming(env, monkeypatch):
    from localm.inference.http_server import create_app
    calls = _recorder(monkeypatch)
    app = create_app(_echo_engine())
    with TestClient(app) as c:
        r = _post(c, {"prompt": "audit-me"})
    assert r.status_code == 200
    assert len(calls) == 1, "completion exchange was not audited"
    last_user = calls[0]["messages"][-1]["content"]
    assert "audit-me" in last_user
    assert calls[0]["reply"] == "audit-me"


def test_completion_audited_streaming(env, monkeypatch):
    from localm.inference.http_server import create_app
    calls = _recorder(monkeypatch)
    app = create_app(_echo_engine())
    with TestClient(app) as c:
        r = _post(c, {"prompt": "audit-stream", "stream": True})
    assert r.status_code == 200
    assert _sse_text(r) == "audit-stream"
    assert len(calls) == 1, "streamed completion exchange was not audited"
    assert calls[0]["reply"] == "audit-stream"
