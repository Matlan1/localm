# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chat-side episodic capture.

synthesize_memory must not summarise the CONCATENATION of every recent session
into at most ONE episodic record: that collapses N sessions into <=1 episode and
re-summarises the same sessions on every run. Each NEW session file (past a
persisted watermark) becomes its own episode, tagged with its source session, and
a processed session is never re-summarised.
"""

from __future__ import annotations

import json
import os

import pytest

from localm.plugins.builtin.memory import plug


@pytest.fixture
def memhome(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    monkeypatch.setenv("LOCALM_MODE", "log")
    (tmp_path / "sessions").mkdir()
    return tmp_path


def _write_session(home, name, content, mtime):
    p = home / "sessions" / f"{name}.jsonl"
    rows = [{"type": "user", "data": {"content": content}},
            {"type": "llm", "data": {"content": "understood"}}]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    os.utime(p, (mtime, mtime))            # deterministic ordering for the watermark
    return p


# Distinct summaries per topic so they do NOT collapse under the 0.85 dedup.
_SUMMARIES = {
    "rust": "Worked through Rust ownership rules and lifetime errors",
    "hiking": "Planned a hiking trip to the Alps for next month",
    "budget": "Reviewed the quarterly budget and cut two line items",
}


def _episode_stub():
    """Model stub: no durable facts, and a distinct clean summary per session topic."""
    def complete(prompt):
        if "Extract ONLY durable" in prompt:
            return json.dumps({"facts": []})
        if "Decide the single best action" in prompt:
            return json.dumps({"decision": "NO_OP", "confidence": 0.9})
        lo = prompt.lower()
        for k, v in _SUMMARIES.items():
            if k in lo:
                return v
        return "{}"
    return complete


def _episodics(store):
    return [r for r in store.all() if r.kind == "episodic"]


def test_one_episode_per_session(memhome):
    _write_session(memhome, "s1", "Let us discuss Rust today", 1000)
    _write_session(memhome, "s2", "I want to plan a hiking trip", 2000)
    _write_session(memhome, "s3", "Let us review the budget", 3000)
    plug.synthesize_memory(_episode_stub())
    eps = _episodics(plug._chat_store())
    assert len(eps) == 3                   # one per session
    joined = " ".join(e.text for e in eps)
    assert "Rust" in joined and "hiking" in joined and "budget" in joined


def test_episode_carries_source_session_meta(memhome):
    _write_session(memhome, "sessA", "Let us discuss Rust today", 1500)
    plug.synthesize_memory(_episode_stub())
    eps = _episodics(plug._chat_store())
    assert eps and eps[0].meta.get("session") == "sessA"
    assert eps[0].meta.get("session_mtime") == 1500


def test_watermark_prevents_reprocessing(memhome):
    _write_session(memhome, "s1", "Let us discuss Rust today", 1000)
    plug.synthesize_memory(_episode_stub())
    assert len(_episodics(plug._chat_store())) == 1
    # A second run with NO new session must add NO new episode (watermark skips it).
    plug.synthesize_memory(_episode_stub())
    assert len(_episodics(plug._chat_store())) == 1


def test_only_new_sessions_processed_after_watermark(memhome):
    _write_session(memhome, "s1", "Let us discuss Rust today", 1000)
    _write_session(memhome, "s2", "I want to plan a hiking trip", 2000)
    plug.synthesize_memory(_episode_stub())
    assert len(_episodics(plug._chat_store())) == 2
    _write_session(memhome, "s3", "Let us review the budget", 3000)
    plug.synthesize_memory(_episode_stub())
    eps = _episodics(plug._chat_store())
    assert len(eps) == 3                   # only s3 was newly summarised
    assert sum("budget" in e.text for e in eps) == 1


def test_watermark_advances_even_with_no_usable_summary(memhome):
    # A session whose summary is unusable ("{}") still advances the watermark, so
    # it is not retried forever.
    _write_session(memhome, "s1", "an unrelated topic with no stub match", 1000)
    plug.synthesize_memory(_episode_stub())
    assert len(_episodics(plug._chat_store())) == 0
    assert plug._read_episodic_watermark(plug._chat_store()) >= 1000


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
