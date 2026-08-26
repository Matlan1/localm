# SPDX-License-Identifier: AGPL-3.0-or-later
"""The episodic pass must be BOUNDED per run and must not re-summarise a
still-growing session.

On the FIRST pass (no watermark sidecar -> watermark 0.0) `new_files` is EVERY
session file. Without a per-run cap that is one real model generation per file,
serially, so a user with a large accumulated history (e.g. 200 sessions)
monopolises the single inference engine for minutes to hours and starves chat. The
number of generations per run is therefore capped, the backlog drains over several
runs, and the watermark advances only past the files actually processed.

An active/growing session file's mtime keeps advancing past the watermark, so
without a settle check it is re-summarised from partial content on every later run,
accumulating overlapping partial episodes. Only a SETTLED session (untouched for a
quiet window) is summarised, and the watermark does not advance past an unsettled
one, so a growing session is summarised exactly ONCE, after it goes quiet.

These tests assert on the GENERATION COUNT and never reference the tuning
constants, and they set mtimes relative to the real clock (settled = far past;
active = seconds ago), so the settle window's exact value is irrelevant.
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
    mtime. The per-run cap must not permanently skip the tied files left
    unprocessed when it breaks mid-group (a strict `>` watermark filter would
    exclude them forever). Every tied session must still drain, exactly once, over
    successive runs."""
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
    # real session's mtime keeps advancing.
    complete3, calls3 = _counting_complete()
    plug._store_episodes(store, complete3, embed_fn=None)
    assert calls3["n"] == 0, "a processed session was re-summarised (duplicate episode)"


def test_existing_short_history_unaffected(memhome):
    """A handful of settled sessions (under the cap) all summarise in one run: the
    bound must not change small-history behaviour."""
    for i in range(3):
        _write_session(memhome, f"old{i}", 1000.0 + i)  # long-settled
    complete, calls = _counting_complete()
    plug._store_episodes(plug._chat_store(), complete, embed_fn=None)
    assert calls["n"] == 3


# --------------------------------------------------------------------------- #
#  A RESUMED session SUPERSEDES its own episode                               #
# --------------------------------------------------------------------------- #
# A session summarised at mtime M, then RESUMED (a user continues the
# conversation the next day), has its mtime advance to M2 > M, so it re-crosses
# the watermark and is summarised again. Each episode is tagged with
# meta={"session": stem}, and that tag is read back so the second summary
# replaces the first instead of becoming a second record for the same
# conversation.

def _summaries(*texts):
    """A stub returning a scripted, genuinely different summary per call."""
    it = iter(texts)
    calls = {"n": 0}

    def complete(prompt):
        calls["n"] += 1
        return next(it)

    return complete, calls


def _episodes(store):
    return [r for r in store.all() if r.kind == "episodic"]


def test_resumed_session_supersedes_its_episode_instead_of_duplicating(memhome):
    """One episode PER SESSION. A resumed session's later summary covers the WHOLE
    conversation, so it must REPLACE that session's earlier partial episode, not add
    a second record for the same stem."""
    store = plug._chat_store()
    _write_session(memhome, "mysession", _SETTLED, content="rust ownership please")
    c1, k1 = _summaries("Worked through Rust ownership rules and lifetime errors")
    plug._store_episodes(store, c1, embed_fn=None)
    assert k1["n"] == 1 and len(_episodes(plug._chat_store())) == 1

    # The user RESUMES the same session the next day: the file grows and its mtime
    # advances past the watermark, then settles again.
    _write_session(memhome, "mysession", _SETTLED + 200_000.0,
                   content="rust ownership, then the whole budget and hiring plan")
    c2, k2 = _summaries("Planned the quarterly budget cuts and the hiring freeze")
    plug._store_episodes(store, c2, embed_fn=None)

    eps = _episodes(plug._chat_store())
    stems = [e.meta.get("session") for e in eps]
    assert len(eps) == 1, (
        f"a resumed session accumulated {len(eps)} episodes for one stem "
        f"({stems}); it must supersede its own earlier episode")
    assert "budget" in eps[0].text, "the superseding episode must be the NEWER summary"
    assert eps[0].meta.get("session_mtime") == _SETTLED + 200_000.0, (
        "the superseded episode must re-stamp the session mtime it now reflects")


def test_resumed_session_with_same_story_does_not_duplicate(memhome):
    """The near-duplicate path: a resumed session whose summary is substantively the
    same must also stay at ONE episode (and must not add a second)."""
    store = plug._chat_store()
    _write_session(memhome, "steady", _SETTLED, content="rust ownership please")
    c1, _ = _summaries("Worked through Rust ownership rules and lifetime errors")
    plug._store_episodes(store, c1, embed_fn=None)

    _write_session(memhome, "steady", _SETTLED + 200_000.0, content="rust ownership again")
    c2, _ = _summaries("Worked through Rust ownership rules and lifetime errors again")
    plug._store_episodes(store, c2, embed_fn=None)
    assert len(_episodes(plug._chat_store())) == 1


def test_preexisting_duplicates_for_one_stem_are_collapsed(memhome):
    """A store can already hold several overlapping partials for one session. When
    that session is next processed, they collapse to the single fullest
    record."""
    from localm.memory import MemoryRecord
    store = plug._chat_store()
    for i, txt in enumerate(["Partial one about rust ownership",
                             "Partial two about rust and cargo",
                             "Partial three about rust, cargo and clippy"]):
        store.add(MemoryRecord(text=txt, kind="episodic", source="synth",
                               importance=0.4,
                               meta={"session": "dupe", "session_mtime": 1000.0 + i}))
    assert len(_episodes(plug._chat_store())) == 3       # three overlapping partials

    _write_session(memhome, "dupe", _SETTLED, content="rust, cargo, clippy, the lot")
    c, _ = _summaries("Covered the whole Rust toolchain: ownership, cargo and clippy")
    plug._store_episodes(store, c, embed_fn=None)

    eps = _episodes(plug._chat_store())
    assert len(eps) == 1, f"expected the duplicates to collapse to one, got {len(eps)}"
    assert "toolchain" in eps[0].text, "the surviving record must be the fullest summary"


def test_collapse_only_touches_the_processed_stem(memhome):
    """The collapse must be bounded: another session's episodes are untouched."""
    from localm.memory import MemoryRecord
    store = plug._chat_store()
    store.add(MemoryRecord(text="An unrelated episode about hiking the Alps",
                           kind="episodic", source="synth", importance=0.4,
                           meta={"session": "other", "session_mtime": 500.0}))
    for i in range(2):
        store.add(MemoryRecord(text=f"Partial {i} about rust ownership rules",
                               kind="episodic", source="synth", importance=0.4,
                               meta={"session": "dupe", "session_mtime": 1000.0 + i}))
    _write_session(memhome, "dupe", _SETTLED, content="rust the lot")
    c, _ = _summaries("Covered the whole Rust toolchain end to end")
    plug._store_episodes(store, c, embed_fn=None)

    eps = _episodes(plug._chat_store())
    stems = sorted((e.meta or {}).get("session") for e in eps)
    assert stems == ["dupe", "other"], f"expected one per stem, got {stems}"


def test_distinct_sessions_still_get_their_own_episodes(memhome):
    """The per-stem supersede must NOT collapse genuinely different sessions into
    one blob episode."""
    store = plug._chat_store()
    _write_session(memhome, "s_rust", _SETTLED, content="rust ownership please")
    _write_session(memhome, "s_hike", _SETTLED + 1, content="plan a hiking trip")
    c, _ = _summaries("Worked through Rust ownership rules and lifetime errors",
                      "Planned a hiking trip to the Alps for next month")
    plug._store_episodes(store, c, embed_fn=None)
    eps = _episodes(plug._chat_store())
    assert len(eps) == 2, "two distinct sessions must yield two episodes"
    assert {e.meta.get("session") for e in eps} == {"s_rust", "s_hike"}
