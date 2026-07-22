# SPDX-License-Identifier: AGPL-3.0-or-later
"""Episodic memory for the coder: store, BM25 retrieval, reflection, and the
Agent integration (recall injection + close-time reflection, privacy/restricted
gated).

Every test isolates LOCALM_HOME so the per-project episode store writes under a
tmp dir, never the user's real data.
"""

from __future__ import annotations

import time

import pytest

from localm.audit import SessionMode
from localm.plugins.coder.episodes import (
    Episode,
    EpisodeStore,
    consolidate,
    reflect_and_store,
    render_for_prompt,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


class _StubBackend:
    model_id = "stub-model"
    native_tools = False

    def set_tools(self, defs):
        pass


class _ChatBackend(_StubBackend):
    """A backend whose .chat returns a canned reflection reply and records calls."""

    def __init__(self, reply: str):
        self._reply = reply
        self.calls: list = []

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        return self._reply


_REPLY = ('{"summary": "added retry with backoff", '
          '"what_worked": "exponential backoff", "what_failed": "", '
          '"lesson": "cap backoff at 30s"}')


# --------------------------------------------------------------------------- #
#  Store: persistence, cap, per-project keying                                #
# --------------------------------------------------------------------------- #

def test_store_add_all_round_trip(home, tmp_path):
    store = EpisodeStore(tmp_path)
    store.add(Episode(task="t1", lesson="L1", files=["a.py"]))
    eps = store.all()
    assert len(eps) == 1
    assert eps[0].lesson == "L1" and eps[0].files == ["a.py"]
    # Atomic write leaves no temp file.
    assert list(store.path.parent.glob("*.tmp")) == []


def test_from_dict_drops_unknown_fields_and_defaults(home):
    ep = Episode.from_dict({"task": "t", "lesson": "L", "bogus": 1})
    assert ep.task == "t" and ep.lesson == "L"
    assert ep.outcome == "ok" and ep.files == []


def test_store_is_per_project(home, tmp_path):
    a = tmp_path / "proj_a"; a.mkdir()
    b = tmp_path / "proj_b"; b.mkdir()
    EpisodeStore(a).add(Episode(task="ta", lesson="La"))
    assert EpisodeStore(a).all() and not EpisodeStore(b).all()
    assert EpisodeStore(a).path != EpisodeStore(b).path


def test_store_evicts_by_value_not_arrival_order(home, tmp_path, monkeypatch):
    """At the cap the LEAST VALUABLE episode goes, not simply the oldest.

    Replaces the old FIFO pin (`test_store_caps_to_newest`). Under that behavior
    the rich failure lesson below - added FIRST - was always the first thing
    discarded, however much it had to teach; that is finding 39 in one line.
    Ages are staggered so plain recency cannot be what saves it."""
    import localm.plugins.coder.episodes as ep_mod
    monkeypatch.setattr(ep_mod, "_MAX_EPISODES", 3)
    store = ep_mod.EpisodeStore(tmp_path)
    now = time.time()

    # Added first and the OLDEST, but it carries a full failure lesson.
    store.add(ep_mod.Episode(
        task="migrate the users table", outcome="incomplete",
        summary="tried to run the migration", what_worked="alembic downgrade -1",
        what_failed="alembic upgrade head deadlocked against the open session",
        lesson="close the psql session before running alembic upgrade",
        ts=now - 2 * 86400))
    # Three thin, newer records: a bare one-line lesson each. Subjects are kept
    # unrelated so the dedup pass has nothing to collapse and eviction is what is
    # actually under test here.
    thin = [("rename the config loader", "config lives in one place"),
            ("bump the pillow dependency", "pin the minor version"),
            ("delete the unused sprite sheet", "check references first")]
    for i, (t, lsn) in enumerate(thin):
        store.add(ep_mod.Episode(task=t, lesson=lsn, ts=now - (2 - i)))

    kept = [e.task for e in store.all()]
    assert len(kept) == 3
    assert "migrate the users table" in kept          # value beat arrival order
    assert "rename the config loader" not in kept     # the weakest went instead
    assert [e.task for e in store.last_evicted] == ["rename the config loader"]


def test_dedup_collapses_near_identical_without_losing_a_distinct_one(home, tmp_path):
    """A restatement of a lesson MERGES into it; a genuinely different lesson is
    untouched. The negative half is the point: an over-eager dedup that also ate
    the distinct episode would be memory loss dressed up as tidying."""
    store = EpisodeStore(tmp_path)
    a = store.add(Episode(task="fix the flaky upload integration test",
                          lesson="raise the upload timeout to 30 seconds",
                          files=["tests/test_upload.py"]))
    b = store.add(Episode(task="add database migrations with alembic",
                          lesson="use alembic revision autogenerate",
                          files=["migrations/env.py"]))
    a2 = store.add(Episode(task="fix the flaky upload integration tests",
                           lesson="raise the upload timeout to 30 sec",
                           files=["tests/conftest.py"]))

    eps = store.all()
    assert len(eps) == 2, [e.task for e in eps]
    lessons = {e.lesson for e in eps}
    # The distinct lesson survives untouched...
    assert "use alembic revision autogenerate" in lessons
    assert any(e.id == b.id for e in eps)
    # ...and the two near-identical ones are now ONE record, keeping the newer
    # wording, both file lists, and a merge count.
    merged = next(e for e in eps if e.id == a2.id)
    assert merged.lesson == "raise the upload timeout to 30 sec"
    assert merged.merged == 1
    assert set(merged.files) == {"tests/conftest.py", "tests/test_upload.py"}
    # The collapsed original is archived, not destroyed.
    arch = store.forgotten()
    assert [r["id"] for r in arch] == [a.id]
    assert arch[0]["reason"] == "merged"


def test_merge_keeps_a_failed_predecessors_warning(home, tmp_path):
    """When a task that once FAILED is done again successfully, the restatement
    absorbs the older record - and must carry its what_failed forward. Dropping it
    would delete the most valuable half of the lesson (audit cluster 11) at exactly
    the moment the merge looks harmless. The outcome, though, is the newer run's:
    it really did finish this time."""
    store = EpisodeStore(tmp_path)
    store.add(Episode(task="migrate the users table to uuid keys",
                      outcome="incomplete",
                      what_failed="alembic upgrade deadlocked on the open session",
                      lesson="close psql before running the migration"))
    newer = store.add(Episode(task="migrate the users table to uuid key",
                              outcome="ok",
                              lesson="close psql before running the migration"))
    eps = store.all()
    assert len(eps) == 1
    got = eps[0]
    assert got.id == newer.id and got.outcome == "ok"
    assert "deadlocked" in got.what_failed        # the warning survived the merge


def test_dedup_does_not_collapse_a_shared_generic_lesson(home, tmp_path):
    """Two DIFFERENT tasks that happen to yield the same generic advice stay two
    episodes: the dedup signature is task+lesson, not the lesson alone."""
    store = EpisodeStore(tmp_path)
    store.add(Episode(task="wire up the billing webhook receiver",
                      lesson="write the test first"))
    store.add(Episode(task="port the image thumbnailer to pillow",
                      lesson="write the test first"))
    assert len(store.all()) == 2


def test_evicted_episode_is_recoverable_from_the_archive(home, tmp_path, monkeypatch):
    """Archive-before-drop: what the cap evicts is still on disk and restorable.
    Negative half first - it really is gone from live recall."""
    import localm.plugins.coder.episodes as ep_mod
    monkeypatch.setattr(ep_mod, "_MAX_EPISODES", 2)
    store = ep_mod.EpisodeStore(tmp_path)
    now = time.time()
    doomed = store.add(ep_mod.Episode(task="the oldest thin task",
                                      lesson="thin lesson", ts=now - 3))
    store.add(ep_mod.Episode(task="second task", lesson="second lesson", ts=now - 2))
    store.add(ep_mod.Episode(task="third task", lesson="third lesson", ts=now - 1))

    # Gone from recall...
    assert doomed.id not in {e.id for e in store.all()}
    # ...but archived with the reason, and restorable.
    arch = store.forgotten()
    assert [r["id"] for r in arch] == [doomed.id]
    assert arch[0]["reason"] == "cap" and arch[0]["lesson"] == "thin lesson"

    back = store.restore(doomed.id)
    assert back is not None and back.id == doomed.id
    assert doomed.id in {e.id for e in store.all()}
    assert back.lesson == "thin lesson"
    # Live again means no longer listed as forgotten.
    assert [r["id"] for r in store.forgotten()] == []
    # And a restore of something that was never archived says so, rather than
    # inventing an episode.
    assert store.restore("nope-not-here") is None


def test_restore_survives_the_decay_that_evicted_it(home, tmp_path, monkeypatch):
    """A restored episode is re-stamped to now, so the very age that got it
    evicted cannot immediately evict it again - which would make restore a silent
    no-op. Its id is preserved so an old citation still resolves."""
    import localm.plugins.coder.episodes as ep_mod
    monkeypatch.setattr(ep_mod, "_MAX_EPISODES", 2)
    store = ep_mod.EpisodeStore(tmp_path)
    now = time.time()
    old = store.add(ep_mod.Episode(task="ancient task", lesson="ancient lesson",
                                   ts=now - 400 * 86400))
    store.add(ep_mod.Episode(task="second task", lesson="second lesson", ts=now))
    store.add(ep_mod.Episode(task="third task", lesson="third lesson", ts=now))
    assert old.id not in {e.id for e in store.all()}

    back = store.restore(old.id)
    assert back is not None
    live = {e.id for e in store.all()}
    assert old.id in live                       # it stuck
    assert back.ts > now - 60                   # re-stamped, not left ancient


def test_restore_does_not_discard_recovery_copies_it_displaces(home, tmp_path,
                                                               monkeypatch):
    """Restoring into a FULL store evicts something else, and that eviction's own
    recovery copy must survive the archive rewrite. Reading the archive before
    add() and writing it back afterwards silently dropped it."""
    import localm.plugins.coder.episodes as ep_mod
    monkeypatch.setattr(ep_mod, "_MAX_EPISODES", 2)
    store = ep_mod.EpisodeStore(tmp_path)
    now = time.time()
    first = store.add(ep_mod.Episode(task="rename the config loader",
                                     lesson="config lives in one place", ts=now - 3))
    store.add(ep_mod.Episode(task="bump the pillow dependency",
                             lesson="pin the minor version", ts=now - 2))
    store.add(ep_mod.Episode(task="delete the unused sprite sheet",
                             lesson="check references first", ts=now - 1))
    assert [r["id"] for r in store.forgotten()] == [first.id]

    # Restoring `first` pushes the store back over the cap, evicting another.
    assert store.restore(first.id) is not None
    displaced = store.last_evicted
    assert displaced, "the restore should have displaced something"
    archived = {r["id"] for r in store.forgotten()}
    assert {e.id for e in displaced} <= archived, (
        "the displaced episode's recovery copy was thrown away by the rewrite")
    assert first.id not in archived          # it is live again, not still forgotten


def test_forget_one_episode_by_id(home, tmp_path):
    """Targeted forget: the whole point of stable ids. Archived, so an accidental
    forget is recoverable; unknown ids report failure instead of silently doing
    nothing."""
    store = EpisodeStore(tmp_path)
    keep = store.add(Episode(task="keep this one", lesson="keep"))
    drop = store.add(Episode(task="a totally different subject", lesson="drop"))

    assert store.forget(drop.id) is True
    assert [e.id for e in store.all()] == [keep.id]
    assert [r["reason"] for r in store.forgotten()] == ["forget"]
    assert store.forget("no-such-id") is False
    assert store.restore(drop.id) is not None    # recoverable


def test_clear_removes_the_archive_too(home, tmp_path, monkeypatch):
    """"Cleared episodic memory" has to be true. Leaving the lesson text sitting
    in the sidecar would be a privacy claim that did not hold."""
    import localm.plugins.coder.episodes as ep_mod
    monkeypatch.setattr(ep_mod, "_MAX_EPISODES", 1)
    store = ep_mod.EpisodeStore(tmp_path)
    store.add(ep_mod.Episode(task="first secret task", lesson="secret one"))
    store.add(ep_mod.Episode(task="a wholly different second task", lesson="two"))
    assert store.archive_path.is_file()          # the evicted one landed there

    store.clear()
    assert not store.path.is_file()
    assert not store.archive_path.is_file()
    assert store.all() == [] and store.forgotten() == []


def test_archive_is_capped(home, tmp_path, monkeypatch):
    import localm.plugins.coder.episodes as ep_mod
    monkeypatch.setattr(ep_mod, "_MAX_EPISODES", 1)
    monkeypatch.setattr(ep_mod, "_ARCHIVE_MAX", 3)
    store = ep_mod.EpisodeStore(tmp_path)
    # Unrelated subjects, so each add EVICTS its predecessor at the cap rather
    # than merging into it - the archive cap is what is under test.
    for t, lsn in [("rename the config loader", "config lives in one place"),
                   ("bump the pillow dependency", "pin the minor version"),
                   ("delete the unused sprite sheet", "check references first"),
                   ("wire up the billing webhook", "verify the signature"),
                   ("port the thumbnailer to pillow", "write the test first"),
                   ("shard the analytics table", "backfill in batches")]:
        store.add(ep_mod.Episode(task=t, lesson=lsn))
    assert [r["reason"] for r in store.forgotten()] == ["cap"] * 3   # bounded


def test_failed_archive_is_reported_not_swallowed(home, tmp_path, monkeypatch, caplog):
    """Rule 5: dropping an episode without a recovery copy is a real failure. It
    stays best-effort (the write still lands - refusing it would break a working
    setup over a sidecar), but it must leave a WARNING, not vanish."""
    import logging

    import localm.plugins.coder.episodes as ep_mod
    monkeypatch.setattr(ep_mod, "_MAX_EPISODES", 1)
    store = ep_mod.EpisodeStore(tmp_path)
    store.add(ep_mod.Episode(task="first task about widgets", lesson="one"))

    def boom(self, *a, **kw):
        raise OSError(28, "No space left on device")

    from pathlib import Path
    monkeypatch.setattr(Path, "write_text", boom)
    with caplog.at_level(logging.WARNING, logger="localm"):
        # write_text is broken for the main log too, so the add itself raises;
        # what matters is that the archive failure was reported first.
        try:
            store.add(ep_mod.Episode(task="a second unrelated task", lesson="two"))
        except OSError:
            pass
    assert any("could not archive" in r.getMessage() for r in caplog.records)
    assert store.last_archive_ok is False


def test_clear_removes_the_log(home, tmp_path):
    store = EpisodeStore(tmp_path)
    store.add(Episode(task="t", lesson="L"))
    assert store.path.is_file()
    store.clear()
    assert not store.path.is_file() and store.all() == []


def test_all_skips_malformed_lines(home, tmp_path):
    store = EpisodeStore(tmp_path)
    store.add(Episode(task="t", lesson="L"))
    store.path.write_text(store.path.read_text(encoding="utf-8") + "not json\n",
                          encoding="utf-8")
    assert len(store.all()) == 1        # the bad line is skipped, not fatal


def test_concurrent_add_and_all_survive_a_racing_replace(home, tmp_path):
    """A writer looping add() and readers looping all() hit the same file
    concurrently (this is the real shape of a GUI poll racing a session-close
    reflection write). On Windows, add()'s atomic temp-file replace can
    momentarily deny a concurrent open of the destination and vice versa; both
    sides must retry through the transient PermissionError, not raise it."""
    import threading
    import time

    store_w = EpisodeStore(tmp_path)
    errors: list = []
    stop = threading.Event()

    def writer():
        n = 0
        while not stop.is_set():
            try:
                store_w.add(Episode(task=f"t{n}", lesson="L"))
            except Exception as e:                       # pragma: no cover - failure path
                errors.append(("write", e))
            n += 1
            time.sleep(0.005)

    def reader():
        store_r = EpisodeStore(tmp_path)
        while not stop.is_set():
            try:
                store_r.all()
            except Exception as e:                        # pragma: no cover - failure path
                errors.append(("read", e))
            time.sleep(0.005)

    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    time.sleep(1.0)
    stop.set()
    for t in threads:
        t.join(timeout=5)

    assert errors == []


def test_add_retries_through_transient_permission_errors_and_succeeds(home, tmp_path, monkeypatch):
    """Fault-injects PermissionError on tmp.replace() a few times before letting
    it through, directly exercising add()'s retry-with-backoff path (the fix for
    the racing-replace flake above) without depending on real OS scheduling
    luck to hit the window."""
    from pathlib import Path

    store = EpisodeStore(tmp_path)
    real_replace = Path.replace
    calls = {"n": 0}

    def flaky_replace(self, target):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError(13, "Access is denied")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    ep = store.add(Episode(task="t", lesson="L"))
    assert calls["n"] == 4          # 3 denials, then the real replace succeeds
    assert ep.lesson == "L"
    assert [e.lesson for e in store.all()] == ["L"]


def test_add_raises_once_the_retry_budget_is_exhausted(home, tmp_path, monkeypatch):
    """A PermissionError that never clears must still surface as a real
    failure, not be silently swallowed - the retry is bounded, not infinite."""
    import localm.plugins.coder.episodes as ep_mod
    from pathlib import Path

    monkeypatch.setattr(ep_mod, "_PERMISSION_RETRY_DELAYS", (0.001, 0.001))  # keep the test fast

    def always_denied(self, target):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(Path, "replace", always_denied)
    store = EpisodeStore(tmp_path)
    with pytest.raises(PermissionError):
        store.add(Episode(task="t", lesson="L"))


def test_all_retries_through_transient_permission_errors_and_succeeds(home, tmp_path, monkeypatch):
    """Same as above for all()'s read side: a concurrent add()'s replace can
    momentarily deny the open, and all() must retry through it, not raise."""
    from pathlib import Path

    store = EpisodeStore(tmp_path)
    store.add(Episode(task="t", lesson="L"))       # seed the file for real first
    real_read_text = Path.read_text
    calls = {"n": 0}

    def flaky_read_text(self, *a, **kw):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError(13, "Access is denied")
        return real_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)
    eps = store.all()
    assert calls["n"] == 4
    assert [e.lesson for e in eps] == ["L"]


def test_all_raises_once_the_retry_budget_is_exhausted(home, tmp_path, monkeypatch):
    import localm.plugins.coder.episodes as ep_mod
    from pathlib import Path

    store = EpisodeStore(tmp_path)
    store.add(Episode(task="t", lesson="L"))       # seed the file for real first
    monkeypatch.setattr(ep_mod, "_PERMISSION_RETRY_DELAYS", (0.001, 0.001))

    def always_denied(self, *a, **kw):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(Path, "read_text", always_denied)
    with pytest.raises(PermissionError):
        store.all()


# --------------------------------------------------------------------------- #
#  Retrieval (BM25)                                                           #
# --------------------------------------------------------------------------- #

def test_search_ranks_relevant_first(home, tmp_path):
    store = EpisodeStore(tmp_path)
    store.add(Episode(task="set up postgres database migrations", lesson="use alembic"))
    store.add(Episode(task="fix css flexbox layout on mobile", lesson="use min-width"))
    hits = store.search("how do I run database migrations", k=2)
    assert hits and hits[0].lesson == "use alembic"


def test_search_returns_nothing_when_irrelevant(home, tmp_path):
    store = EpisodeStore(tmp_path)
    store.add(Episode(task="fix css flexbox layout", lesson="use min-width"))
    # No token overlap -> below the relevance floor -> nothing injected.
    assert store.search("postgres database migration alembic") == []
    assert store.search("quantum chromodynamics lagrangian gluon") == []


def test_search_silent_on_stopword_only_overlap(home, tmp_path):
    # A shared STOPWORD ("the"/"to"/"for") must not clear the relevance floor. BM25
    # has no stopword removal, so before the content-word filter an unrelated task
    # that merely contained "the" recalled irrelevant lessons and injected them into
    # the coder prompt. The lexical signal is now content-words-only, so this stays
    # silent.
    store = EpisodeStore(tmp_path)
    store.add(Episode(task="fix the flaky file-upload integration test",
                      lesson="raise the upload test timeout"))
    assert store.search("configure the kubernetes ingress controller for the cluster") == []
    assert store.search("update the billing invoice to the new tax rate") == []


def test_search_empty_store(home, tmp_path):
    assert EpisodeStore(tmp_path).search("anything") == []


def _kw_embed(texts):
    """Deterministic keyword-bucket 'embedding' so cosine reflects topical
    relatedness in-process (no model): [is-networking, is-ui, bias]."""
    net = {"retry", "backoff", "http", "client", "api", "network", "connection",
           "connections", "unreliable", "resilient", "server", "servers",
           "request", "requests", "flaky", "gracefully", "upstream", "handle"}
    ui = {"css", "flexbox", "layout", "navbar", "dropdown", "menu", "style",
          "mobile", "screens", "grid", "gap", "media", "min-width"}
    out = []
    for t in texts:
        w = set(t.lower().split())
        out.append([1.0 if w & net else 0.0, 1.0 if w & ui else 0.0, 0.2])
    return out


def test_search_semantic_recall_beats_lexical(home, tmp_path, monkeypatch):
    """With an embedder, a task phrased differently from a lesson is still
    recalled (cosine), where BM25 alone finds nothing; unrelated stays silent."""
    import localm.plugins.coder.episodes as em
    monkeypatch.setattr(em, "_embed_fn", lambda: _kw_embed)
    store = EpisodeStore(tmp_path)
    store.add(Episode(task="add retry logic with exponential backoff", lesson="cap backoff"))
    store.add(Episode(task="style the navbar dropdown", lesson="use flexbox gap"))
    q = "handle unreliable upstream connections gracefully"   # no shared tokens w/ retry
    sem = [e.lesson for e in store.search(q, k=2)]
    assert any("backoff" in l for l in sem)                   # semantic finds it
    # BM25-only (embedder off) misses it entirely
    monkeypatch.setattr(em, "_embed_fn", lambda: None)
    assert store.search(q, k=2) == []
    # and an unrelated task stays silent even with the embedder
    monkeypatch.setattr(em, "_embed_fn", lambda: _kw_embed)
    assert store.search("bake a chocolate cake", k=2) == []


def test_clear_removes_vector_sidecar(home, tmp_path, monkeypatch):
    import localm.plugins.coder.episodes as em
    monkeypatch.setattr(em, "_embed_fn", lambda: _kw_embed)
    store = EpisodeStore(tmp_path)
    store.add(Episode(task="add retry logic", lesson="backoff"))
    store.search("network retry")                              # populates the sidecar
    vec = store.path.with_suffix(".vec.json")
    assert vec.is_file()
    store.clear()
    assert not vec.is_file() and not store.path.is_file()


# --------------------------------------------------------------------------- #
#  Reflection                                                                 #
# --------------------------------------------------------------------------- #

def test_reflect_and_store_parses_and_stores(home, tmp_path):
    store = EpisodeStore(tmp_path)
    ep = reflect_and_store(store, task="do X", diff="--- a\n+++ b", outcome="ok",
                           files=["a.py"], turns=3, complete=lambda p: _REPLY, ts=1.0)
    assert ep.lesson == "cap backoff at 30s"
    stored = store.all()
    assert len(stored) == 1
    assert stored[0].summary == "added retry with backoff"
    assert stored[0].files == ["a.py"] and stored[0].outcome == "ok"


def test_reflect_handles_fenced_json(home, tmp_path):
    store = EpisodeStore(tmp_path)
    reflect_and_store(store, task="t", diff="", outcome="ok", files=[], turns=0,
                      complete=lambda p: '```json\n{"lesson": "fenced"}\n```', ts=1.0)
    assert store.all()[0].lesson == "fenced"


def test_reflect_extracts_json_wrapped_in_prose(home, tmp_path):
    store = EpisodeStore(tmp_path)
    reply = 'Sure! Here it is:\n{"lesson": "embedded"} \nHope that helps.'
    reflect_and_store(store, task="t", diff="", outcome="ok", files=[], turns=0,
                      complete=lambda p: reply, ts=1.0)
    assert store.all()[0].lesson == "embedded"


def test_reflect_skips_empty_when_model_gives_nothing(home, tmp_path):
    store = EpisodeStore(tmp_path)
    reflect_and_store(store, task="t", diff="", outcome="ok", files=[], turns=0,
                      complete=lambda p: "sorry, no idea", ts=1.0)
    assert store.all() == []        # nothing parseable -> not stored


def test_reflect_survives_a_model_error(home, tmp_path):
    store = EpisodeStore(tmp_path)

    def boom(prompt):
        raise RuntimeError("model down")

    reflect_and_store(store, task="t", diff="", outcome="ok", files=[], turns=0,
                      complete=boom, ts=1.0)
    assert store.all() == []        # best-effort: no crash, nothing stored


def test_reflect_feeds_error_trace_to_the_model(home, tmp_path):
    # Cluster 13: the reflection must SEE the tool/command failures, not just the
    # diff, so it can actually fill what_failed. Capture the prompt the model gets.
    store = EpisodeStore(tmp_path)
    seen = {}

    def capture(prompt):
        seen["prompt"] = prompt
        return ('{"summary": "s", "what_failed": "the pytest run failed", '
                '"lesson": "check imports first"}')

    reflect_and_store(
        store, task="fix the failing test", diff="--- a\n+++ b", outcome="incomplete",
        files=["t.py"], turns=4,
        errors="run_tests: ModuleNotFoundError: no module named foo\n"
               "run_shell: git apply failed: patch does not apply",
        complete=capture, ts=1.0)
    p = seen["prompt"]
    assert "TOOL FAILURES AND ERRORS" in p
    assert "ModuleNotFoundError" in p and "git apply failed" in p
    stored = store.all()
    assert stored and stored[0].what_failed == "the pytest run failed"


def test_reflect_stores_thin_failure_episode_when_model_unusable(home, tmp_path):
    # Cluster 11: a failed session whose model produces nothing usable must still
    # record the failure lesson from the raw error evidence - deterministically.
    store = EpisodeStore(tmp_path)
    reflect_and_store(
        store, task="add the migration", diff="", outcome="incomplete",
        files=[], turns=6,
        errors="run_shell: alembic: command not found\n"
               "run_shell: alembic: command not found",   # deduped in the summary
        complete=lambda p: "hmm, I am not sure", ts=1.0)
    stored = store.all()
    assert len(stored) == 1
    ep = stored[0]
    assert ep.summary == "session did not complete"
    assert "alembic: command not found" in ep.what_failed
    # deduped: the repeated identical line collapses to one
    assert ep.what_failed.count("alembic: command not found") == 1


def test_reflect_thin_failure_label_when_completed_with_errors(home, tmp_path):
    store = EpisodeStore(tmp_path)
    reflect_and_store(
        store, task="t", diff="", outcome="ok", files=[], turns=2,
        errors="read_file: no such file: missing.py",
        complete=lambda p: "no idea", ts=1.0)
    stored = store.all()
    assert len(stored) == 1
    assert stored[0].summary == "session completed with errors"
    assert "missing.py" in stored[0].what_failed


def test_reflect_no_thin_episode_without_error_evidence(home, tmp_path):
    # Unusable model reply AND no error trace -> still nothing stored (unchanged):
    # a blank record would only dilute retrieval.
    store = EpisodeStore(tmp_path)
    reflect_and_store(store, task="t", diff="", outcome="incomplete", files=[],
                      turns=0, errors="", complete=lambda p: "no idea", ts=1.0)
    assert store.all() == []


# --------------------------------------------------------------------------- #
#  Ids                                                                        #
# --------------------------------------------------------------------------- #

def test_add_assigns_a_stable_id_that_survives_a_round_trip(home, tmp_path):
    store = EpisodeStore(tmp_path)
    ep = store.add(Episode(task="t", lesson="L"))
    assert ep.id
    assert [e.id for e in store.all()] == [ep.id]
    assert [e.id for e in EpisodeStore(tmp_path).all()] == [ep.id]   # reopened


def test_legacy_record_without_an_id_gets_a_stable_derived_one(home, tmp_path):
    """A record written before ids existed still has to be citable and
    forgettable, without a migration write: the id derives from its own content,
    so it is the same on every read."""
    store = EpisodeStore(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        '{"task": "old task", "lesson": "old lesson", "ts": 1700000000.0}\n',
        encoding="utf-8")
    first = store.all()[0]
    assert first.id
    assert EpisodeStore(tmp_path).all()[0].id == first.id           # stable
    assert store.forget(first.id) is True                           # and usable


def test_identical_episodes_still_get_distinct_ids(home, tmp_path):
    """Two records with the same content AND timestamp derive the same id; the
    store disambiguates rather than letting one shadow the other."""
    store = EpisodeStore(tmp_path)
    a = store.add(Episode(task="alpha subject", lesson="one", ts=1.0))
    b = store.add(Episode(task="beta subject entirely", lesson="two", ts=1.0))
    # Force the collision path: a third record whose derived id equals a's.
    c = store.add(Episode(task="alpha subject", lesson="one", ts=1.0), dedup=False)
    ids = [e.id for e in store.all()]
    assert len(ids) == len(set(ids)) == 3
    assert a.id != c.id and b.id not in (a.id, c.id)


# --------------------------------------------------------------------------- #
#  Rendering                                                                  #
# --------------------------------------------------------------------------- #

def test_render_for_prompt():
    assert render_for_prompt([]) == ""
    block = render_for_prompt([Episode(task="t", lesson="do X", what_failed="thing Y")])
    assert "Past lessons" in block
    assert "lesson: do X" in block
    assert "avoid: thing Y" in block


def test_what_worked_is_load_bearing(home, tmp_path):
    """It used to be written and then never read by anything - not searched, not
    rendered, not weighed (audit finding 39, "what_worked is a dead field"). All
    three now consume it."""
    import localm.plugins.coder.episodes as em

    # 1. rendered on recall
    block = render_for_prompt([Episode(task="t", lesson="do X",
                                       what_worked="ran pytest -x first")])
    assert "worked: ran pytest -x first" in block

    # 2. searchable: a task matching ONLY the what_worked text still recalls it
    store = EpisodeStore(tmp_path)
    store.add(Episode(task="unrelated wording here", lesson="unrelated wording too",
                      what_worked="regenerate the protobuf stubs with grpcio-tools"))
    assert store.search("regenerate the protobuf stubs")

    # 3. weighed at eviction: same age, the one carrying it is worth more
    now = time.time()
    with_it = Episode(task="a", lesson="L", what_worked="W", ts=now)
    without = Episode(task="b", lesson="L", ts=now)
    assert em._episode_value(with_it, now) > em._episode_value(without, now)


# --------------------------------------------------------------------------- #
#  Opt-in LLM consolidation                                                   #
# --------------------------------------------------------------------------- #

_MERGE_REPLY = ('{"summary": "two takes on the same retry work", '
                '"what_worked": "exponential backoff", '
                '"what_failed": "a fixed 1s sleep", '
                '"lesson": "back off exponentially and cap at 30s"}')


def test_consolidate_merges_related_lessons_and_archives_the_originals(home, tmp_path):
    store = EpisodeStore(tmp_path)
    a = store.add(Episode(task="add retry logic to the http client",
                          lesson="use exponential backoff", ts=1000.0))
    b = store.add(Episode(task="add retry logic to the http uploader",
                          lesson="cap the backoff at 30s", ts=2000.0))
    far = store.add(Episode(task="restyle the settings page navbar",
                            lesson="use flexbox gap", ts=3000.0))

    rep = consolidate(store, complete=lambda p: _MERGE_REPLY)
    assert rep["groups"] == 1 and rep["merged"] == 1 and rep["replaced"] == 2
    assert rep["archived"] == 2 and rep["skipped"] == 0

    eps = store.all()
    assert len(eps) == 2
    assert far.id in {e.id for e in eps}                     # unrelated untouched
    merged = next(e for e in eps if e.id != far.id)
    assert merged.lesson == "back off exponentially and cap at 30s"
    # Reversible: both inputs are in the archive and can come back.
    arch_ids = {r["id"] for r in store.forgotten()}
    assert {a.id, b.id} == arch_ids
    assert all(r["reason"] == "consolidated" for r in store.forgotten())
    assert store.restore(a.id) is not None


def test_consolidate_leaves_a_group_alone_when_the_merge_is_unusable(home, tmp_path):
    """A weak model that returns nothing usable must not cost the user their
    lessons - destroying real memory is worse than not consolidating."""
    store = EpisodeStore(tmp_path)
    a = store.add(Episode(task="add retry logic to the http client",
                          lesson="use exponential backoff", ts=1000.0))
    b = store.add(Episode(task="add retry logic to the http uploader",
                          lesson="cap the backoff at 30s", ts=2000.0))
    rep = consolidate(store, complete=lambda p: "hmm, I am not sure")
    assert rep["groups"] == 1 and rep["merged"] == 0 and rep["skipped"] == 1
    assert {e.id for e in store.all()} == {a.id, b.id}       # untouched
    assert store.forgotten() == []


def test_consolidate_survives_a_model_error(home, tmp_path):
    store = EpisodeStore(tmp_path)
    store.add(Episode(task="add retry logic to the http client", lesson="backoff"))
    store.add(Episode(task="add retry logic to the http uploader", lesson="cap it"))

    def boom(prompt):
        raise RuntimeError("model down")

    rep = consolidate(store, complete=boom)
    assert rep["skipped"] == 1 and rep["merged"] == 0
    assert len(store.all()) == 2


def test_consolidate_is_never_automatic(home, tmp_path):
    """It must not fire on close(): a local model rewriting stored memory is
    where one bad merge poisons every later run, so it stays user-triggered."""
    from localm.audit import SessionMode
    store = EpisodeStore(tmp_path)
    store.add(Episode(task="add retry logic to the http client", lesson="backoff"))
    store.add(Episode(task="add retry logic to the http uploader", lesson="cap it"))
    agent = _agent(tmp_path, backend=_ChatBackend(_REPLY), mode=SessionMode.LOG)
    agent._changed_files = {"foo.py": {"original": None, "writes": 1,
                                       "last_tool": "write_file"}}
    agent._episode_task = "add retry"
    agent.close()
    # close() reflects (a third episode) but consolidates nothing.
    assert EpisodeStore(tmp_path).forgotten() == []


# --------------------------------------------------------------------------- #
#  Agent integration                                                          #
# --------------------------------------------------------------------------- #

def _agent(tmp_path, backend=None, **kw):
    from localm.plugins.coder.agent import Agent
    return Agent(backend or _StubBackend(), cwd=tmp_path, **kw)


def test_agent_enables_episodic_by_default(home, tmp_path):
    # In a normal (non-privacy) session, episodic memory is on by default.
    agent = _agent(tmp_path, mode=SessionMode.LOG)
    assert agent._episodic is True
    assert agent._episode_store is not None


def test_privacy_mode_disables_episodic(home, tmp_path):
    # Privacy mode disables the coder's episodic memory ENTIRELY (no recall AND no
    # write): the store is not even opened, so past-session lessons never reach the
    # model. Mirrors the chat memory's "fully off in privacy" contract.
    agent = _agent(tmp_path, mode=SessionMode.PRIVACY)
    assert agent._episodic is False
    assert agent._episode_store is None
    # And a pre-existing lesson is NOT recalled in privacy mode.
    EpisodeStore(tmp_path).add(Episode(
        task="add retry logic to the http client",
        lesson="exponential backoff capped at 30s"))
    assert agent._with_episodes("add retry logic") == "add retry logic"


def test_privacy_recall_opt_in_for_coder(home, tmp_path, monkeypatch):
    """With the coder privacy-recall opt-in on, a privacy-mode session RECALLS past
    lessons (read-only) but still writes NO new episode at close - reading existing
    lessons never creates a new trace."""
    import localm.config as cfg
    monkeypatch.setattr(cfg, "load_config", lambda: {
        "coder_episodic_memory": True,
        "memory_recall_in_privacy": True,
        "memory_recall_in_privacy_coder": True})
    EpisodeStore(tmp_path).add(Episode(
        task="add retry logic to the http client",
        lesson="exponential backoff capped at 30s"))
    agent = _agent(tmp_path, backend=_ChatBackend(_REPLY), mode=SessionMode.PRIVACY)
    assert agent._episodic is True                            # recall enabled
    out = agent._with_episodes("add retry logic to the uploader")
    assert "exponential backoff" in out                       # recalled read-only
    # A privacy session still writes NO new episode at close.
    agent._changed_files = {"foo.py": {"original": None, "writes": 1,
                                       "last_tool": "write_file"}}
    agent._episode_task = "add retry"
    agent.close()
    assert len(EpisodeStore(tmp_path).all()) == 1             # only the seeded lesson


def test_restricted_session_disables_episodic(home, tmp_path):
    agent = _agent(tmp_path, restricted=True)
    assert agent._episodic is False
    assert agent._episode_store is None


def test_config_off_disables_episodic(home, tmp_path, monkeypatch):
    import localm.config as cfg
    monkeypatch.setattr(cfg, "load_config", lambda: {"coder_episodic_memory": False})
    agent = _agent(tmp_path, mode=SessionMode.LOG)
    assert agent._episodic is False


def test_with_episodes_injects_relevant_lesson(home, tmp_path):
    EpisodeStore(tmp_path).add(Episode(
        task="add retry logic to the http client",
        lesson="exponential backoff capped at 30s"))
    agent = _agent(tmp_path, mode=SessionMode.LOG)
    out = agent._with_episodes("add retry logic to the http client uploader")
    assert "Past lessons" in out
    assert "exponential backoff" in out
    assert "## Task" in out


def test_with_episodes_noop_when_no_relevant_history(home, tmp_path):
    EpisodeStore(tmp_path).add(Episode(task="totally unrelated css work",
                                       lesson="use grid"))
    agent = _agent(tmp_path, mode=SessionMode.LOG)
    assert agent._with_episodes("quantum chromodynamics solver") == \
        "quantum chromodynamics solver"


def test_recalled_lesson_ids_are_recorded_for_the_run(home, tmp_path):
    """Retrieval used to render the lessons and throw the Episode objects away, so
    a lesson that steered a run badly was invisible afterwards and there was no
    handle to forget it by. The run now records id + text, emits it, and audits
    it - and the id it reports is the one targeted forget takes."""
    seeded = EpisodeStore(tmp_path).add(Episode(
        task="add retry logic to the http client",
        lesson="exponential backoff capped at 30s"))
    events: list = []
    agent = _agent(tmp_path, mode=SessionMode.LOG, on_event=events.append)
    audited: list = []
    agent._audit.episodes_recalled = audited.append

    out = agent._with_episodes("add retry logic to the http client uploader")
    assert "exponential backoff" in out                    # it really was injected

    assert [u["id"] for u in agent._episodes_used] == [seeded.id]
    assert agent._episodes_used[0]["lesson"] == "exponential backoff capped at 30s"
    assert agent._episodes_used[0]["outcome"] == "ok"
    assert agent._episodes_degrade_reason == ""
    # surfaced on the event stream (GUI/SSE)...
    recalled = [e for e in events if e["type"] == "episodes_recalled"]
    assert len(recalled) == 1
    assert [e["id"] for e in recalled[0]["episodes"]] == [seeded.id]
    # ...and in the audit trail.
    assert audited == [agent._episodes_used]
    # The reported id is the one that forgets it.
    assert EpisodeStore(tmp_path).forget(seeded.id) is True


def test_silent_recall_records_the_reason_and_emits_nothing(home, tmp_path):
    """Nothing relevant is a legitimate outcome, but it must be distinguishable
    from a BROKEN recall - so it is recorded, while the noisy surfaces stay quiet
    (silence-when-irrelevant is the contract)."""
    EpisodeStore(tmp_path).add(Episode(task="totally unrelated css work",
                                       lesson="use grid"))
    events: list = []
    agent = _agent(tmp_path, mode=SessionMode.LOG, on_event=events.append)
    agent._with_episodes("quantum chromodynamics solver")
    assert agent._episodes_used == []
    assert agent._episodes_degrade_reason == "no relevant past lesson"
    assert [e for e in events if e["type"] == "episodes_recalled"] == []


def test_failed_recall_is_recorded_not_swallowed(home, tmp_path):
    """Rule 5: a recall that BLEW UP is not the same as one that found nothing.
    The run still proceeds (best-effort), but the failure leaves a trace."""
    agent = _agent(tmp_path, mode=SessionMode.LOG)

    def boom(task, k=3):
        raise RuntimeError("store unreadable")

    agent._episode_store.search = boom
    assert agent._with_episodes("do the thing") == "do the thing"   # run continues
    assert agent._episodes_used == []
    assert "recall failed" in agent._episodes_degrade_reason
    assert "store unreadable" in agent._episodes_degrade_reason


def test_privacy_recall_opt_in_writes_no_audit_entry(home, tmp_path, monkeypatch):
    """The new citation surface must not create a trace privacy mode forbids: in
    a privacy session the audit log is a NullAuditLog, so recording what was
    recalled writes nothing to disk."""
    import localm.config as cfg
    monkeypatch.setattr(cfg, "load_config", lambda: {
        "coder_episodic_memory": True,
        "memory_recall_in_privacy": True,
        "memory_recall_in_privacy_coder": True})
    EpisodeStore(tmp_path).add(Episode(
        task="add retry logic to the http client",
        lesson="exponential backoff capped at 30s"))
    agent = _agent(tmp_path, mode=SessionMode.PRIVACY)
    from localm.audit import NullAuditLog
    assert isinstance(agent._audit, NullAuditLog)
    out = agent._with_episodes("add retry logic to the uploader")
    assert "exponential backoff" in out              # recalled read-only
    assert agent._episodes_used                      # tracked in memory for the run
    assert agent._audit.path is None                 # but nothing on disk


def test_close_writes_episode_on_changes(home, tmp_path):
    from localm.audit import SessionMode
    agent = _agent(tmp_path, backend=_ChatBackend(_REPLY), mode=SessionMode.LOG)
    # Simulate a substantive session (a file was written).
    agent._changed_files = {"foo.py": {"original": None, "writes": 1,
                                       "last_tool": "write_file"}}
    agent._episode_task = "add retry logic"
    agent.close()                                   # on_event is None -> synchronous
    eps = agent._episode_store.all()
    assert len(eps) == 1
    assert eps[0].lesson == "cap backoff at 30s"
    assert "foo.py" in eps[0].files


def test_close_skips_episode_in_privacy_mode(home, tmp_path):
    agent = _agent(tmp_path, backend=_ChatBackend(_REPLY), mode=SessionMode.PRIVACY)
    agent._changed_files = {"foo.py": {"original": None, "writes": 1,
                                       "last_tool": "write_file"}}
    agent.close()
    # Privacy: episodic memory is fully off - the store is never even opened...
    assert agent._episode_store is None
    # ...and nothing was written to disk for this project.
    assert EpisodeStore(tmp_path).all() == []


def test_close_writes_nothing_without_changes(home, tmp_path):
    from localm.audit import SessionMode
    agent = _agent(tmp_path, backend=_ChatBackend(_REPLY), mode=SessionMode.LOG)
    agent.close()                                   # no changed files
    assert agent._episode_store.all() == []


def test_restricted_session_neither_recalls_nor_writes(home, tmp_path):
    from localm.audit import SessionMode
    # A pre-existing owner lesson must NOT be recalled by a restricted session,
    # and a restricted session must NOT write a new one.
    EpisodeStore(tmp_path).add(Episode(task="owner lesson about the database",
                                       lesson="owner only"))
    agent = _agent(tmp_path, backend=_ChatBackend(_REPLY),
                   mode=SessionMode.LOG, restricted=True)
    assert agent._with_episodes("work on the database") == "work on the database"
    agent._changed_files = {"foo.py": {"original": None, "writes": 1,
                                       "last_tool": "write_file"}}
    agent.close()
    assert len(EpisodeStore(tmp_path).all()) == 1   # only the pre-existing one


def test_gui_session_reflects_off_the_event_loop(home, tmp_path):
    # GUI/web sessions (on_event set) must run the close-time model call on a
    # background thread so close() never blocks the asyncio loop it runs on.
    import threading
    import time

    from localm.audit import SessionMode

    started = threading.Event()
    release = threading.Event()

    class _BlockingBackend(_StubBackend):
        def chat(self, messages, **kw):
            started.set()
            release.wait(5)
            return _REPLY

    agent = _agent(tmp_path, backend=_BlockingBackend(), mode=SessionMode.LOG,
                   on_event=lambda e: None)
    agent._changed_files = {"foo.py": {"original": None, "writes": 1,
                                       "last_tool": "write_file"}}
    agent._episode_task = "do a thing"

    t0 = time.monotonic()
    agent.close()
    assert time.monotonic() - t0 < 1.0              # did not block on the model call
    assert started.wait(5)                          # reflection thread did start it
    assert agent._episode_store.all() == []         # not written while blocked
    release.set()
    for _ in range(100):                            # episode lands after chat returns
        if agent._episode_store.all():
            break
        time.sleep(0.02)
    assert len(agent._episode_store.all()) == 1


# --------------------------------------------------------------------------- #
#  CLI transparency: list / clear stored lessons                              #
# --------------------------------------------------------------------------- #

def test_cli_episodes_list_and_forget(home, tmp_path, monkeypatch):
    from click.testing import CliRunner
    from localm.plugins.engine import PluginManager
    monkeypatch.setattr(PluginManager, "is_active", lambda self, name: True)
    EpisodeStore(tmp_path).add(Episode(task="t", outcome="ok", lesson="remember this"))

    from localm.plugins.coder.cli import main
    runner = CliRunner()
    r = runner.invoke(main, ["--cwd", str(tmp_path), "--episodes"])
    assert r.exit_code == 0, r.output
    assert "remember this" in r.output

    r = runner.invoke(main, ["--cwd", str(tmp_path), "--forget-episodes"])
    assert r.exit_code == 0, r.output
    assert EpisodeStore(tmp_path).all() == []
