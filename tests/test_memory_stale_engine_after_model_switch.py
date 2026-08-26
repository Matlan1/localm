# SPDX-License-Identifier: AGPL-3.0-or-later
"""Memory consolidation must follow the CURRENT engine, never one snapshotted at
plugin-load time.

PluginManager.inference_engine (plugins/engine.py) is a LIVE property resolving
http_server._engines[http_server._active_model_name] on every access; switch_engine
(http_server.py) rebinds that to a brand-new Engine object per model and unloads
the old one. register() therefore stashes the HOST and every use site resolves
host.engine() fresh (plug._live_engine()), never caching its return value.

Negative case: a fake host whose .engine() mimics the live property (its return
value can change after register() without register() being called again) proves
the route follows the CURRENT engine.
"""

from __future__ import annotations

import json
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.builtin.memory import plug


class _SwappableHost:
    """Mimics PluginHost.engine() -> PluginManager.inference_engine: a LIVE
    lookup whose return value can change out from under a caller who resolved it
    earlier."""

    def __init__(self):
        self.current = None

    def mount_router(self, router):
        pass

    def register_chat_hook(self, phase, fn, priority=0):
        pass

    def engine(self):
        return self.current


class _StubEngine:
    loaded = True

    def __init__(self, reply):
        self._reply = reply
        self.calls = 0
        self.caller_thread = None

    def chat_stream(self, messages, **kw):
        self.calls += 1
        self.caller_thread = threading.get_ident()
        yield self._reply


def _facts_reply():
    return json.dumps({"facts": [{"fact": "User's name is Ada", "confidence": 0.9}]})


def _seed_session(home):
    sdir = home / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"type": "user", "data": {"content": "my name is Ada"}},
        {"type": "llm", "data": {"content": "Noted, Ada!"}},
    ]
    (sdir / "s.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_embed_fn", lambda: None)
    monkeypatch.setenv("LOCALM_MODE", "log")               # writes allowed
    return tmp_path


def test_live_engine_reflects_host_engine_on_every_call(monkeypatch):
    """Unit-level proof: _live_engine() is a live pass-through to the stashed
    host, not a value pinned at register() time."""
    # register() does `global _HOST; _HOST = host` - a raw module-global mutation
    # a plain function call can't undo. Snapshot it via monkeypatch FIRST so
    # teardown restores whatever _HOST held at entry, regardless of what
    # register() sets it to in between.
    monkeypatch.setattr(plug, "_HOST", None)
    host = _SwappableHost()
    plug.register(host)                     # runs "at startup": nothing loaded yet
    assert plug._live_engine() is None

    eng_a = _StubEngine(_facts_reply())
    host.current = eng_a                    # the first model finishes loading
    assert plug._live_engine() is eng_a

    eng_b = _StubEngine(_facts_reply())      # switch_engine: a NEW Engine object
    host.current = eng_b                     # replaces eng_a (which is unloaded)
    assert plug._live_engine() is eng_b, (
        "stale: _live_engine() still returned the engine captured at an earlier "
        "moment instead of following the host's current live value")
    assert plug._live_engine() is not eng_a


def test_consolidate_route_follows_a_model_switch(home, monkeypatch):
    """Route-level proof: register() happens before any model is loaded (None
    engine), so consolidate 503s. A model loads, then the user SWITCHES to a
    different model (a brand-new engine object, per switch_engine).
    Consolidation must use the NEW engine."""
    monkeypatch.setattr(plug, "_HOST", None)      # undo the module-global leak
    _seed_session(home)
    host = _SwappableHost()
    plug.register(host)

    app = FastAPI()
    app.include_router(plug._router)
    client = TestClient(app)

    # Startup: register() ran, but no model is loaded yet.
    assert client.post("/api/memory/consolidate").status_code == 503

    # First model loads.
    eng_a = _StubEngine(_facts_reply())
    host.current = eng_a
    r1 = client.post("/api/memory/consolidate")
    assert r1.status_code == 200, r1.text
    assert eng_a.calls > 0, "consolidation never called the first engine"

    # User switches models: a brand-new Engine object replaces the old one.
    eng_b = _StubEngine(_facts_reply())
    host.current = eng_b
    r2 = client.post("/api/memory/consolidate")
    assert r2.status_code == 200, r2.text
    assert eng_b.calls > 0, (
        "consolidation after a model switch never reached the NEW engine - it is "
        "still pinned to whatever was resolved earlier (#959)")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
