# SPDX-License-Identifier: AGPL-3.0-or-later
"""REG-591: the episodic pass must be BOUNDED per run and must not re-summarise a
still-growing session.

HIGH: on the FIRST pass (no watermark sidecar -> watermark 0.0) `new_files` is
EVERY session file, and the loop ran one real model generation per file, serially,
with no per-run cap. A user with a large accumulated history (e.g. 200 sessions)
upgrades and the first auto-consolidation monopolises the single inference engine
for one generation per file, minutes-to-hours, starving chat. Pre-#591 the episodic
pass made exactly ONE model call total. Fix: cap the number of generations per run
and drain the backlog over several runs, advancing the watermark only past the
files actually processed.

MEDIUM: an active/growing session file's mtime keeps advancing past the watermark,
so it was re-summarised (from partial content) on every later run, accumulating
overlapping partial episodes. Fix: only summarise a SETTLED session (untouched for
a quiet window) and do not advance the watermark past an unsettled one, so a
growing session is summarised exactly ONCE, after it goes quiet.

These tests drive the SAME public signature the pre-fix code has (so stashing the
fix yields a clean behavioural failure, not an import error): they assert on the
GENERATION COUNT (pre-fix == one per file; post-fix bounded) and never reference
the fix's tuning constants, and they set mtimes relative to the real clock (settled
= far past; active = seconds ago), so the settle window's exact value is irrelevant.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from localm.plugins.builtin.memory import plug


@pytest.fixture
def memhome(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    monkeypatch.setenv("LOCALM_MODE", "log")
    (tmp_path / "sessions").mkdir()
    return tmp_path


def _write_session(home, name, mtime, content=None):
    p = home / "sessions" / f"{name}.jsonl"
    rows = [{"type": "user", "data": {"content": content or f"Let us discuss {name}"}},
            {"type": "llm", "data": {"content": "understood"}}]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    os.utime(p, (mtime, mtime))
    return p


def _counting_complete():
    """A summariser stub that COUNTS real generations and returns a distinct usable
    summary each call (so the count reflects model calls regardless of dedup)."""
    calls = {"n": 0}

    def complete(prompt):
        calls["n"] += 1
        return f"Reviewed distinct workstream {calls['n']} and its open questions"

    return complete, calls


_SETTLED = 1_000_000.0        # mtime far in the past: settled under any quiet window
_LONG_AGO = 100_000.0         # seconds; older than any reasonable settle window


# --------------------------------------------------------------------------- #
#  HIGH: bounded first pass                                                    #
# --------------------------------------------------------------------------- #

def test_first_pass_is_bounded(memhome):
    """A large pre-existing history (no watermark) must NOT generate one summary
    per file in a single pass."""
    n_files = 12
    for i in range(n_files):
        _write_session(memhome, f"s{i:03d}", _SETTLED + i)
    complete, calls = _counting_complete()

    plug._store_episodes(plug._chat_store(), complete, embed_fn=None)

    assert 1 <= calls["n"] < n_files, (
        f"episodic pass ran {calls['n']} generations for {n_files} files in one run; "
        f"pre-fix it ran one per file (unbounded)")


def test_backlog_drains_over_runs_without_skipping(memhome):
    """Bounding must not DROP files: the backlog drains over successive runs and
    every settled session is eventually summarised exactly once."""
    n_files = 12
    for i in range(n_files):
        _write_session(memhome, f"s{i:03d}", _SETTLED + i)
    total = 0
    store = plug._chat_store()
    for _ in range(30):                                 # generous upper bound on runs
        complete, calls = _counting_complete()
        plug._store_episodes(store, complete, embed_fn=None)
        assert calls["n"] < n_files                     # each run stays bounded
        total += calls["n"]
        if calls["n"] == 0:
            break
    assert total == n_files, (
        f"expected every one of {n_files} sessions summarised exactly once across "
        f"runs, got {total} total generations")


def test_tied_mtimes_at_cap_boundary_not_skipped(memhome):
    """Tie-safety: a bulk LOCALM_HOME restore, or a coarse-granularity volume
    (FAT/exFAT/SMB, 1-2s mtime resolution), gives MANY session files one IDENTICAL
    mtime - exactly the large first-pass history REG-591 targets. The per-run cap
    must not permanently skip the tied files left unprocessed when it breaks
    mid-group (a strict `>` watermark filter would exclude them forever). Every
    tied session must still drain, exactly once, over successive runs."""
    n_files = plug.EPISODIC_MAX_PER_RUN * 3
    for i in range(n_files):
        _write_session(memhome, f"s{i:03d}", _SETTLED)   # all identical mtime
    total = 0
    store = plug._chat_store()
    for _ in range(30):
        complete, calls = _counting_complete()
        plug._store_episodes(store, complete, embed_fn=None)
        assert calls["n"] <= plug.EPISODIC_MAX_PER_RUN   # bound holds even all-tied
        total += calls["n"]
        if calls["n"] == 0:
            break
    assert total == n_files, (
        f"tied-mtime backlog skipped files: {total}/{n_files} summarised "
        f"(watermark tie boundary dropped the rest)")


# --------------------------------------------------------------------------- #
#  MEDIUM: a growing session is not summarised while active, nor re-summarised #
# --------------------------------------------------------------------------- #

def test_active_session_not_summarised_until_settled(memhome):
    active_mtime = time.time() - 5.0                    # written seconds ago: still active
    _write_session(memhome, "live", active_mtime)
    store = plug._chat_store()

    complete, calls = _counting_complete()
    plug._store_episodes(store, complete, embed_fn=None)
    assert calls["n"] == 0, "an active/growing session was summarised while incomplete"
    # The watermark must NOT have advanced past the unsettled file, or it would
    # never be summarised once it settles.
    assert plug._read_episodic_watermark(store) < active_mtime


def test_settled_session_summarised_exactly_once(memhome):
    p = _write_session(memhome, "chat1", time.time() - 5.0)   # active first
    store = plug._chat_store()

    # Run 1: active -> skipped.
    complete1, calls1 = _counting_complete()
    plug._store_episodes(store, complete1, embed_fn=None)
    assert calls1["n"] == 0

    # The session ends; its file settles (mtime far enough in the past).
    settled_mtime = time.time() - _LONG_AGO
    os.utime(p, (settled_mtime, settled_mtime))

    # Run 2: settled -> summarised exactly once.
    complete2, calls2 = _counting_complete()
    plug._store_episodes(store, complete2, embed_fn=None)
    assert calls2["n"] == 1, "a settled session was not summarised"

    # Run 3: already past the watermark, so it is NOT re-summarised even though a
    # real session's mtime keeps advancing (the pre-fix duplicate-episode bug).
    complete3, calls3 = _counting_complete()
    plug._store_episodes(store, complete3, embed_fn=None)
    assert calls3["n"] == 0, "a processed session was re-summarised (duplicate episode)"


def test_existing_short_history_unaffected(memhome):
    """A handful of settled sessions (under the cap) still all summarise in one run,
    exactly as before - the bound must not change small-history behaviour."""
    for i in range(3):
        _write_session(memhome, f"old{i}", 1000.0 + i)  # long-settled
    complete, calls = _counting_complete()
    plug._store_episodes(plug._chat_store(), complete, embed_fn=None)
    assert calls["n"] == 3
