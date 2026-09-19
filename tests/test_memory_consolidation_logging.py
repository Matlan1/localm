# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for memory consolidation observability and progress logging.

Ensures that background auto-consolidation and manual consolidation passes emit
clear progress logs (start pass, extraction, candidate evaluation, episodic
summarization, and pass completion) rather than running silently during heavy
compute. Verifies that prompt/response logging is gated on debug_content_enabled.
"""

from __future__ import annotations

import json
import logging
import time

import pytest

from localm.plugins.builtin.memory import plug


@pytest.fixture
def mem_env(tmp_path, monkeypatch):
    import localm.config as cfg
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_memory_root", lambda: tmp_path / "memory")
    monkeypatch.setattr(plug, "_embed_fn", lambda: None)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    monkeypatch.setenv("LOCALM_MODE", "log")
    plug._auto_running = False

    sdir = tmp_path / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"type": "user", "data": {"content": "I prefer dark mode in all applications."}},
        {"type": "llm", "data": {"content": "Understood, dark mode is saved."}},
    ]
    sfile = sdir / "s1.jsonl"
    sfile.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return tmp_path


class _StubEngine:
    loaded = True

    def __init__(self, reply_map: dict | None = None, default_reply: str = "{}"):
        self.reply_map = reply_map or {}
        self.default_reply = default_reply
        self.calls = 0

    def chat_stream(self, messages, **kw):
        self.calls += 1
        prompt = messages[0]["content"] if messages else ""
        for key, val in self.reply_map.items():
            if key in prompt:
                yield val
                return
        yield self.default_reply


def test_auto_consolidate_logs_progress(mem_env, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="localm")
    extract_reply = json.dumps({"facts": [{"fact": "User prefers dark mode", "confidence": 0.9}]})
    engine = _StubEngine({"Extract ONLY durable": extract_reply})
    monkeypatch.setattr(plug, "_live_engine", lambda: engine)

    plug._auto_consolidate_bg()

    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "memory auto-consolidate: starting background memory synthesis pass" in blob
    assert "memory synthesis: analyzing session history" in blob
    assert "memory consolidation: extracting candidate facts" in blob
    assert "memory consolidation: extracted 1 candidate fact(s)" in blob
    assert "memory auto-consolidate: pass complete" in blob


def test_auto_consolidate_content_gating_disabled(mem_env, monkeypatch, caplog):
    monkeypatch.setattr("localm.debuglog.debug_content_enabled", lambda: False)
    caplog.set_level(logging.DEBUG, logger="localm")
    secret_fact = "secret_keyword_super_confidential"
    extract_reply = json.dumps({"facts": [{"fact": secret_fact, "confidence": 0.9}]})
    engine = _StubEngine({"Extract ONLY durable": extract_reply})
    monkeypatch.setattr(plug, "_live_engine", lambda: engine)

    plug._auto_consolidate_bg()

    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "memory auto-consolidate prompt:" not in blob
    assert "memory auto-consolidate response:" not in blob
    assert secret_fact not in blob


def test_auto_consolidate_content_gating_enabled(mem_env, monkeypatch, caplog):
    monkeypatch.setattr("localm.debuglog.debug_content_enabled", lambda: True)
    caplog.set_level(logging.DEBUG, logger="localm")
    sample_fact = "User prefers dark mode"
    extract_reply = json.dumps({"facts": [{"fact": sample_fact, "confidence": 0.9}]})
    engine = _StubEngine({"Extract ONLY durable": extract_reply})
    monkeypatch.setattr(plug, "_live_engine", lambda: engine)

    plug._auto_consolidate_bg()

    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "memory auto-consolidate prompt:" in blob
    assert "memory auto-consolidate response:" in blob
    assert sample_fact in blob


def test_store_episodes_logging(mem_env, caplog):
    caplog.set_level(logging.INFO, logger="localm")
    store = plug._chat_store()

    # Settle the session file by backdating mtime past EPISODIC_SETTLE_SECONDS
    sfile = mem_env / "sessions" / "s1.jsonl"
    old_time = time.time() - (plug.EPISODIC_SETTLE_SECONDS + 60)
    import os
    os.utime(sfile, (old_time, old_time))

    def complete(prompt):
        return "Summary: User prefers dark mode."

    stored = plug._store_episodes(store, complete)
    assert stored == 1

    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "memory consolidation: checking 1 session file(s) for episodic capture" in blob
    assert "memory consolidation: summarizing session s1" in blob
    assert "memory consolidation: stored 1 episodic summary record(s)" in blob
