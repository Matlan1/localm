# SPDX-License-Identifier: AGPL-3.0-or-later
"""The chat model grows its memory UNATTENDED. After a turn, a debounced
background pass distils durable facts into the store, so a default (opted-in)
install accumulates memory with no manual step.

These tests exercise the real trigger logic in localm.plugins.builtin.chat.plug
(the gating, debounce, privacy block, and single-flight guard), driving the
background pass synchronously via a stubbed engine so no real model is needed.
"""

from __future__ import annotations

import json
import time

import pytest


@pytest.fixture
def chat_home(tmp_path, monkeypatch):
    import localm.config as cfg
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    import localm.plugins.builtin.memory.plug as plug
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_memory_root", lambda: tmp_path / "memory")
    monkeypatch.setattr(plug, "_embed_fn", lambda: None)
    # Reset module state between tests.
    plug._auto_running = False
    return tmp_path, plug


class _SyncThread:
    """Drop-in for threading.Thread that runs the target synchronously on
    .start(), so the background consolidation is deterministic in tests."""

    def __init__(self, target=None, daemon=None):
        self._target = target

    def start(self):
        if self._target:
            self._target()


def _record_thread(sink):
    """A Thread stub that records the spawned target instead of running it, so a
    test can assert whether a background pass would have been spawned."""
    class _Rec:
        def __init__(self, target=None, daemon=None):
            sink.append(target)

        def start(self):
            pass
    return _Rec


class _StubEngine:
    loaded = True

    def __init__(self, reply):
        self._reply = reply
        self.calls = 0

    def chat_stream(self, messages, **kw):
        self.calls += 1
        yield self._reply


def _seed_session(home, text="User: my name is Ada and I prefer metric units\n"
                             "Assistant: Noted, Ada!"):
    sdir = home / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"type": "user", "data": {"content": "my name is Ada and I prefer metric units"}},
        {"type": "llm", "data": {"content": "Noted, Ada!"}},
    ]
    (sdir / "s.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _facts_reply():
    return json.dumps({"facts": [
        {"fact": "User's name is Ada", "confidence": 0.9},
        {"fact": "User prefers metric units", "confidence": 0.9}]})


def test_auto_consolidation_grows_memory_after_a_turn(chat_home, monkeypatch):
    home, plug = chat_home
    monkeypatch.setenv("LOCALM_MODE", "log")            # writes allowed
    _seed_session(home)
    eng = _StubEngine(_facts_reply())
    monkeypatch.setattr(plug, "_live_engine", lambda: eng)
    # Run the background pass synchronously so the test is deterministic.
    monkeypatch.setattr(plug._threading, "Thread", _SyncThread)

    plug._memory_outlet("reply text", [], {})

    store = plug._chat_store()
    texts = [r.text for r in store.all()]
    assert any("Ada" in t for t in texts), f"memory did not grow: {texts}"
    assert plug._auto_running is False                   # flag cleared after run


def test_privacy_mode_never_auto_consolidates(chat_home, monkeypatch):
    home, plug = chat_home
    monkeypatch.setenv("LOCALM_MODE", "privacy")
    _seed_session(home)
    eng = _StubEngine(_facts_reply())
    monkeypatch.setattr(plug, "_live_engine", lambda: eng)
    spawned = []
    monkeypatch.setattr(plug._threading, "Thread", _record_thread(spawned))

    plug._memory_outlet("reply", [], {})

    assert spawned == [], "a consolidation thread was spawned in privacy mode"
    assert eng.calls == 0
    assert not (home / "memory").exists() or not list((home / "memory").rglob("*.jsonl"))


def test_debounce_blocks_a_second_run_within_the_interval(chat_home, monkeypatch):
    home, plug = chat_home
    monkeypatch.setenv("LOCALM_MODE", "log")
    _seed_session(home)
    monkeypatch.setattr(plug, "_live_engine", lambda: _StubEngine(_facts_reply()))
    spawned = []
    monkeypatch.setattr(plug._threading, "Thread", _record_thread(spawned))
    # First outlet stamps the marker (recent), so the second is debounced.
    plug._auto_stamp(time.time())
    plug._memory_outlet("reply", [], {})
    assert spawned == [], "ran again within the debounce interval"


def test_disabled_config_skips_auto(chat_home, monkeypatch):
    home, plug = chat_home
    monkeypatch.setenv("LOCALM_MODE", "log")
    (home / "config.json").write_text(
        json.dumps({"memory_auto_consolidate": False}), encoding="utf-8")
    _seed_session(home)
    monkeypatch.setattr(plug, "_live_engine", lambda: _StubEngine(_facts_reply()))
    spawned = []
    monkeypatch.setattr(plug._threading, "Thread", _record_thread(spawned))
    plug._memory_outlet("reply", [], {})
    assert spawned == [], "auto ran despite memory_auto_consolidate=False"


def test_no_model_loaded_skips_auto(chat_home, monkeypatch):
    home, plug = chat_home
    monkeypatch.setenv("LOCALM_MODE", "log")
    _seed_session(home)
    monkeypatch.setattr(plug, "_live_engine", lambda: None)          # no model
    spawned = []
    monkeypatch.setattr(plug._threading, "Thread", _record_thread(spawned))
    plug._memory_outlet("reply", [], {})
    assert spawned == []


def test_outlet_returns_text_unchanged_and_never_raises(chat_home, monkeypatch):
    home, plug = chat_home
    # Force an internal error; the outlet must still return the text.
    monkeypatch.setattr(plug, "_maybe_auto_consolidate",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert plug._memory_outlet("the reply", [], {}) == "the reply"


def test_marker_persists_last_run_across_reads(chat_home):
    home, plug = chat_home
    assert plug._auto_last_run() == 0.0                 # never run
    plug._auto_stamp(1234567.0)
    assert plug._auto_last_run() == 1234567.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
