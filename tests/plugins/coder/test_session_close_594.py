# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two properties of the coder's session-close path.

(a) THE REFLECTION TRIGGER. The close-time episodic reflection fires for a run
    that GENUINELY failed (max_turns, or a circuit breaker - both set
    ``_last_run_ok=False``), and not for a routine read-only run that collected
    incidental tool errors, nor for a USER-initiated stop (Ctrl-C, declining
    "keep going?"). In the CLI ``on_event`` is None, so the reflection runs
    SYNCHRONOUSLY: a 1024-token model inference between the user typing exit and
    getting their shell back.

(b) CLOSE MUST NOT RUN ON THE EVENT LOOP. The async route
    ``DELETE /api/coder/sessions/{id}`` reaches ``CoderSession.close()`` ->
    ``agent.close()`` -> ``_maybe_store_episode()`` -> ``_detect_shell_changes()``,
    which runs blocking ``subprocess.run(["git","status","--porcelain"],
    timeout=10)`` plus up to two ``git diff`` calls (timeout 15 each).

The on-loop tests use a structural oracle, not a timing one:
``asyncio.get_running_loop()`` succeeds only on the event-loop thread and raises
RuntimeError in a threadpool worker.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.audit import SessionMode


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


class _CountingBackend(_StubBackend):
    """Records every reflection call. ``calls`` is the observable: in the CLI
    each one is a synchronous 1024-token inference on quit."""

    def __init__(self):
        self.calls: list = []

    def chat(self, messages, **kw):
        self.calls.append(kw)
        return ('{"summary": "s", "what_worked": "", "what_failed": "f", '
                '"lesson": "l"}')


def _agent(tmp_path, backend=None, **kw):
    from localm.plugins.coder.agent import Agent
    return Agent(backend or _StubBackend(), cwd=tmp_path, mode=SessionMode.LOG, **kw)


# --------------------------------------------------------------------------- #
#  (a) Ordinary read-only sessions must not reflect on quit                    #
# --------------------------------------------------------------------------- #

class TestReflectionTrigger:
    def test_ok_readonly_session_with_incidental_errors_does_not_reflect(
            self, home, tmp_path):
        """`localm coder "why does X crash"` - reads around, two tool calls
        incidentally fail (a missing file, a failed grep), the run completes
        fine, nothing is written. Quitting makes no model call at all."""
        backend = _CountingBackend()
        agent = _agent(tmp_path, backend=backend)
        agent._episode_task = "why does the importer crash"
        assert agent._last_run_ok is True
        agent._error_trace = ["read_file: no such file: importer.py",
                              "run_shell: grep: importer: no such file"]

        agent.close()

        assert backend.calls == [], (
            "an ordinary read-only session that changed nothing ran a model "
            "reflection at close - a surprise hang on quit")
        assert agent._episode_store.all() == []

    def test_many_incidental_errors_alone_still_do_not_reflect(self, home, tmp_path):
        """A bare error COUNT is not a failure signal: a run that completed OK
        has no lesson, however many incidental errors it collected."""
        backend = _CountingBackend()
        agent = _agent(tmp_path, backend=backend)
        agent._episode_task = "look around"
        agent._error_trace = [f"read_file: missing {i}" for i in range(8)]
        assert agent._last_run_ok is True

        agent.close()
        assert backend.calls == []

    def test_user_stopped_session_does_not_reflect(self, home, tmp_path):
        """A user-initiated stop (Ctrl-C, or declining "keep going?") is not a
        failure with a lesson, so it does not reflect."""
        backend = _CountingBackend()
        agent = _agent(tmp_path, backend=backend)
        agent._episode_task = "have a look"
        agent._last_run_ok = False
        agent._user_stopped = True

        agent.close()
        assert backend.calls == []
        assert agent._episode_store.all() == []

    def test_genuine_failure_with_no_change_still_reflects(self, home, tmp_path):
        """A run that actually failed (max_turns or a circuit breaker, never a
        user stop) still records its lesson even though it wrote no files."""
        backend = _CountingBackend()
        agent = _agent(tmp_path, backend=backend)
        agent._episode_task = "fix the failing import"
        agent._last_run_ok = False           # max_turns / circuit breaker
        agent._error_trace = ["run_shell: pytest failed"]

        agent.close()
        assert len(backend.calls) == 1, "a genuine failure lesson must still reflect"
        eps = agent._episode_store.all()
        assert len(eps) == 1 and eps[0].outcome == "incomplete"

    def test_clean_session_still_reflects_nothing(self, home, tmp_path):
        backend = _CountingBackend()
        agent = _agent(tmp_path, backend=backend)
        agent._episode_task = "what does this do"
        agent.close()
        assert backend.calls == []

    def test_user_stopped_resets_between_runs(self, home, tmp_path):
        """The flag is per-run state: _loop resets it, so a stopped run followed
        by a genuinely failed one still reflects."""
        from unittest.mock import patch
        from localm.plugins.coder.agent import Agent
        backend = _CountingBackend()
        agent = _agent(tmp_path, backend=backend)
        agent._user_stopped = True           # a previous run was stopped

        def _fake_call_llm(self, messages, interactive):
            raise RuntimeError("stop here")

        with patch.object(Agent, "_call_llm", _fake_call_llm):
            with pytest.raises(RuntimeError):
                agent.run_task("next task")
        assert agent._user_stopped is False, "_loop must reset the per-run flag"


class _StopMidRunBackend(_CountingBackend):
    """Stands in for the user hitting Ctrl-C DURING generation: request_stop()
    lands while the model call is in flight. _loop clears a stale
    _stop_requested at entry, so a stop set BEFORE the run is ignored."""

    def __init__(self, agent_box):
        super().__init__()
        self._box = agent_box

    def chat(self, messages, **kw):
        if kw.get("max_tokens") == 1024:      # the close-time reflection call
            return super().chat(messages, **kw)
        self._box["agent"].request_stop()
        return "partial answer"


def test_loop_marks_a_user_stop_and_not_a_max_turns_failure(home, tmp_path):
    """Driven through the REAL loop, not hand-set state: a stop that arrives
    during a run marks _user_stopped, while exhausting max_turns does not, and
    only the latter reflects at close."""
    box: dict = {}
    backend = _StopMidRunBackend(box)
    agent = _agent(tmp_path, backend=backend)
    box["agent"] = agent
    agent.run_task("do something")
    assert agent._last_run_ok is False
    assert agent._user_stopped is True
    agent.close()
    assert backend.calls == [], "a user stop must not reflect at close"

    # max_turns is a genuine failure, NOT a user stop -> it still reflects.
    backend2 = _CountingBackend()
    agent2 = _agent(tmp_path, backend=backend2, max_turns=0)
    agent2._episode_task = "t"
    agent2.run_task("do something")
    assert agent2._last_run_ok is False
    assert agent2._user_stopped is False
    agent2.close()
    assert len(backend2.calls) == 1, "a max_turns failure must still reflect"


class _StopFirstRunThenAnswerBackend(_StopMidRunBackend):
    """Run 1 is stopped mid-generation; every later run answers cleanly with no
    tool call. Inherits the parent's contract that ``calls`` records ONLY the
    close-time reflection (max_tokens=1024), so an ordinary turn stays invisible
    to it and the assertion below is about the reflection alone.

    The clean answers differ from each other: an identical repeat would trip the
    repeat-response circuit breaker, which sets _last_run_ok=False and would arm
    the trigger for a reason unrelated to the stop.
    """

    def __init__(self, agent_box):
        super().__init__(agent_box)
        self.stops_left = 1
        self.answers = 0

    def chat(self, messages, **kw):
        if kw.get("max_tokens") == 1024:        # the close-time reflection call
            return super().chat(messages, **kw)
        if self.stops_left > 0:
            self.stops_left -= 1
            return super().chat(messages, **kw)  # requests stop, keeps the partial
        self.answers += 1
        return f"had a look, nothing to change (note {self.answers})"


def test_one_stopped_run_does_not_arm_reflection_for_the_rest_of_the_session(
        home, tmp_path):
    """One stopped run must not arm the close-time reflection permanently.

    _had_any_failure is SESSION-level and cleared only by reset(), while the
    trigger's user-stop guard reads the PER-RUN _user_stopped, which the next
    run re-arms to False.

    Drives the REAL loop for both runs; the behaviour lives in _loop's finally,
    so hand-set flags cannot see it.
    """
    box: dict = {}
    backend = _StopFirstRunThenAnswerBackend(box)
    agent = _agent(tmp_path, backend=backend)
    box["agent"] = agent

    agent.run_task("start something")             # run 1: stopped mid-generation
    assert agent._user_stopped is True
    assert agent._last_run_ok is False

    agent.continue_task("now a small question")   # run 2: clean, writes nothing
    assert agent._user_stopped is False, "the per-run flag must re-arm"
    assert agent._last_run_ok is True, (
        "run 2 must COMPLETE - if a breaker tripped it, this test would go green "
        "or red for a reason unrelated to the stop")
    assert agent.changed_files() == [], "this session must write no files"

    agent.close()

    assert backend.calls == [], (
        "a stopped EARLIER run left the close-time reflection armed: after a "
        "clean session the user still waits on a 1024-token inference at quit")
    assert agent._episode_store.all() == []


def test_a_stop_after_a_genuine_failure_still_keeps_that_lesson(home, tmp_path):
    """The mirror: a run that GENUINELY failed keeps its lesson even when the
    user stops the NEXT one. The user-stop guard applies to the per-run term
    only, never to the session-level failure record.
    """
    box: dict = {}
    backend = _StopMidRunBackend(box)
    agent = _agent(tmp_path, backend=backend, max_turns=0)
    box["agent"] = agent
    agent._episode_task = "fix the failing import"

    agent.run_task("do something")        # max_turns=0: a genuine failure
    assert agent._last_run_ok is False
    assert agent._user_stopped is False, "max_turns is not a user stop"
    assert agent._had_any_failure is True, "the session-level record must hold it"

    agent.max_turns = 3                   # let the next run actually reach the model
    agent.continue_task("carry on then")  # run 2: the user stops it
    assert agent._user_stopped is True
    assert agent._last_run_ok is False

    agent.close()

    assert len(backend.calls) == 1, (
        "the earlier genuine failure lost its lesson because the LAST run "
        "happened to be a user stop")
    assert len(agent._episode_store.all()) == 1


# --------------------------------------------------------------------------- #
#  (b) Closing a session must not block the event loop                         #
# --------------------------------------------------------------------------- #

def _coder_app(tmp_path, monkeypatch, *, api_key="ownersecret"):
    home_dir = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home_dir))
    monkeypatch.setenv("LOCALM_API_KEY", api_key)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home_dir)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home_dir / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home_dir / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home_dir / "registry.json")
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("coder")

    async def switch_model(name):
        pass

    from localm.plugins.gui.web import attach_gui
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "m")
    return app


def test_delete_session_route_does_not_close_on_the_event_loop(tmp_path, monkeypatch):
    """The real async DELETE route must not run the blocking close (git status +
    up to two git diffs, each with a multi-second timeout) on the loop thread.

    Oracle: asyncio.get_running_loop() succeeds ONLY on the event-loop thread and
    raises RuntimeError in a threadpool worker. Structural, so no sleep and no
    load-sensitive timing.
    """
    proj = tmp_path / "proj"
    proj.mkdir()
    app = _coder_app(tmp_path, monkeypatch)
    app.state.root_dir = str(proj)
    owner = {"Authorization": "Bearer ownersecret"}

    from localm.plugins.coder.sessions import CoderSession
    seen: dict = {}
    real_close = CoderSession.close

    def _probing_close(self):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True      # running ON the event-loop thread
        except RuntimeError:
            seen["on_loop"] = False     # off-loop (threadpool worker): correct
        return real_close(self)

    monkeypatch.setattr(CoderSession, "close", _probing_close)

    with TestClient(app) as client:
        r = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "mode": "log"})
        assert r.status_code == 200, r.text
        sid = r.json()["id"]
        d = client.delete(f"/api/coder/sessions/{sid}", headers=owner)
        assert d.status_code == 200, d.text
        assert d.json()["status"] == "closed"

    assert seen.get("on_loop") is False, (
        "session close ran ON the event loop: its blocking git subprocess would "
        "freeze every concurrent request")


def test_delete_session_route_still_404s_for_an_unknown_id(tmp_path, monkeypatch):
    """Offloading must not change the route's contract."""
    proj = tmp_path / "proj"
    proj.mkdir()
    app = _coder_app(tmp_path, monkeypatch)
    app.state.root_dir = str(proj)
    owner = {"Authorization": "Bearer ownersecret"}
    with TestClient(app) as client:
        assert client.delete("/api/coder/sessions/nope",
                             headers=owner).status_code == 404


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_delete_session_route_offloads_a_real_git_detecting_close(tmp_path,
                                                                 monkeypatch):
    """A GUI session that ran a run_shell (so the git baseline is captured) but
    recorded no write-tool change is exactly when close() reaches
    _detect_shell_changes() and shells out to git. The real git work must happen
    off the loop."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (("init",), ("config", "user.email", "t@example.com"),
                 ("config", "user.name", "t")):
        subprocess.run(["git", *args], cwd=str(repo), check=True,
                       capture_output=True, text=True)
    (repo / "base.py").write_text("print('base')\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True,
                   capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=str(repo), check=True,
                   capture_output=True, text=True)

    app = _coder_app(tmp_path, monkeypatch)
    app.state.root_dir = str(repo)
    owner = {"Authorization": "Bearer ownersecret"}

    seen: dict = {}
    from localm.plugins.coder.agent import persistence as _persist
    real_detect = _persist._PersistenceMixin._detect_shell_changes

    def _probing_detect(self):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        return real_detect(self)

    monkeypatch.setattr(_persist._PersistenceMixin, "_detect_shell_changes",
                        _probing_detect)

    with TestClient(app) as client:
        r = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(repo), "mode": "log"})
        assert r.status_code == 200, r.text
        sid = r.json()["id"]
        sess = app.state.coder_sessions.get(sid)
        # Stand in for a run_shell having run: the baseline is what makes close()
        # reach git at all.
        sess.agent._shell_baseline_captured = True
        sess.agent._git_baseline = frozenset()
        sess.agent._last_run_ok = False       # a genuine failure -> close reflects
        (repo / "generated.py").write_text("print('gen')\n", encoding="utf-8")

        assert client.delete(f"/api/coder/sessions/{sid}",
                             headers=owner).status_code == 200

    assert seen.get("on_loop") is False, (
        "the git change-detection subprocess ran ON the event loop")
