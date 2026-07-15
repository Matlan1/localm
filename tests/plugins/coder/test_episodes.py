# SPDX-License-Identifier: AGPL-3.0-or-later
"""Episodic memory for the coder: store, BM25 retrieval, reflection, and the
Agent integration (recall injection + close-time reflection, privacy/restricted
gated).

Every test isolates LOCALM_HOME so the per-project episode store writes under a
tmp dir, never the user's real data.
"""

from __future__ import annotations

import pytest

from localm.audit import SessionMode
from localm.plugins.coder.episodes import (
    Episode,
    EpisodeStore,
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


def test_store_caps_to_newest(home, tmp_path, monkeypatch):
    import localm.plugins.coder.episodes as ep_mod
    monkeypatch.setattr(ep_mod, "_MAX_EPISODES", 3)
    store = ep_mod.EpisodeStore(tmp_path)
    for i in range(5):
        store.add(ep_mod.Episode(task="task %d" % i, lesson="lesson %d" % i))
    tasks = [e.task for e in store.all()]
    assert tasks == ["task 2", "task 3", "task 4"]      # oldest two dropped


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
#  Rendering                                                                  #
# --------------------------------------------------------------------------- #

def test_render_for_prompt():
    assert render_for_prompt([]) == ""
    block = render_for_prompt([Episode(task="t", lesson="do X", what_failed="thing Y")])
    assert "Past lessons" in block
    assert "lesson: do X" in block
    assert "avoid: thing Y" in block


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
