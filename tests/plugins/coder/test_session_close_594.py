# SPDX-License-Identifier: AGPL-3.0-or-later
"""REG-594 (regression audit 2026-07-14), two MEDIUM findings on the coder's
session-close path.

#594 broadened the close-time episodic reflection so a FAILED session with no
file change still records its lesson (failure lessons were systematically absent,
audit cluster 11). The feature is right; the trigger and the threading were not.

(a) session.py:152 - TRIGGER TOO BROAD. ``had_failure = (not self._last_run_ok)
    or len(self._error_trace) >= 2``. ``_error_trace`` accumulates EVERY failed
    tool call (a missing read_file, a failed grep), so a routine read-only
    investigation that wrote nothing took the reflection path. In the CLI
    ``on_event`` is None, so the reflection runs SYNCHRONOUSLY: a 1024-token model
    inference between the user typing exit and getting their shell back. It also
    fired for a USER-initiated stop (Ctrl-C, declining "keep going?"), which is the
    exact moment the user is asking to leave.
    The ``>= 2`` clause is also redundant for genuine failures: a truly broken run
    trips a circuit breaker (_GLOBAL_ERROR_ABORT=6, _REPEAT_RESPONSE_ABORT=5, or
    the per-tool streak) or max_turns, and every one of those already sets
    ``_last_run_ok=False``. So it only ever added false positives.

(b) session.py:147 - BLOCKING GIT ON THE EVENT LOOP. The async route
    ``DELETE /api/coder/sessions/{id}`` called ``_sessions(request).remove(id)``
    inline -> ``CoderSession.close()`` -> ``agent.close()`` ->
    ``_maybe_store_episode()`` -> ``_detect_shell_changes()``, which runs blocking
    ``subprocess.run(["git","status","--porcelain"], timeout=10)`` plus up to two
    ``git diff`` calls (timeout 15 each). On the loop thread that freezes the whole
    server - every concurrent request, SSE stream, and portmux-muxed connection -
    for the git duration.

The on-loop tests here use a structural oracle, not a timing one:
``asyncio.get_running_loop()`` succeeds only on the event-loop thread and raises
RuntimeError in a threadpool worker. So it detects the defect deterministically,
with no sleeps and no flakiness under load.
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
    """Records every reflection call. ``calls`` is the observable: in the CLI each
    one is a synchronous 1024-token inference the user waits for on quit."""

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
#  (a) The trigger: ordinary read-only sessions must not reflect on quit       #
# --------------------------------------------------------------------------- #

class TestReflectionTrigger:
    def test_ok_readonly_session_with_incidental_errors_does_not_reflect(
            self, home, tmp_path):
        """THE REGRESSION. `localm coder "why does X crash"` - reads around, two
        tool calls incidentally fail (a missing file, a failed grep), the run
        completes fine, nothing is written. Quitting must be instant: no model
        call at all."""
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
        """The threshold was 2; the point is that a bare error COUNT is not a
        failure signal at all. A run that still completed OK has no lesson,
        however many incidental errors it collected along the way."""
        backend = _CountingBackend()
        agent = _agent(tmp_path, backend=backend)
        agent._episode_task = "look around"
        agent._error_trace = [f"read_file: missing {i}" for i in range(8)]
        assert agent._last_run_ok is True

        agent.close()
        assert backend.calls == []

    def test_user_stopped_session_does_not_reflect(self, home, tmp_path):
        """A user-initiated stop (Ctrl-C, or declining "keep going?") is not a
        failure with a lesson - it is the user asking to leave NOW. Reflecting
        there is the worst possible moment for a 1024-token inference."""
        backend = _CountingBackend()
        agent = _agent(tmp_path, backend=backend)
        agent._episode_task = "have a look"
        agent._last_run_ok = False
        agent._user_stopped = True

        agent.close()
        assert backend.calls == []
        assert agent._episode_store.all() == []

    def test_genuine_failure_with_no_change_still_reflects(self, home, tmp_path):
        """The #594 FEATURE must survive: a run that actually failed (max_turns or
        a circuit breaker - never a user stop) still records its lesson even
        though it wrote no files."""
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
        """The flag is per-run state: a stopped run followed by a genuinely failed
        one must still reflect, or one Ctrl-C would mute lessons for the rest of
        the session."""
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
    lands while the model call is in flight, which is the real path (_loop clears
    a stale _stop_requested at entry, so a stop set BEFORE the run is ignored by
    design - "a stale stop must not kill a new task")."""

    def __init__(self, agent_box):
        super().__init__()
        self._box = agent_box

    def chat(self, messages, **kw):
        if kw.get("max_tokens") == 1024:      # the close-time reflection call
            return super().chat(messages, **kw)
        self._box["agent"].request_stop()
        return "partial answer"


def test_loop_marks_a_user_stop_and_not_a_max_turns_failure(home, tmp_path):
    """Pin the flag to the REAL loop, not just to hand-set state: a stop that
    arrives during a run marks _user_stopped, while exhausting max_turns does
    not - and only the latter reflects at close."""
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


# --------------------------------------------------------------------------- #
#  (b) The event loop: closing a session must not block it                     #
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
    """The concrete REG-594b scenario: a GUI session that ran a run_shell (so the
    git baseline is captured) but recorded no write-tool change. That is exactly
    when close() reaches _detect_shell_changes() and shells out to git. Prove the
    real git work happens off the loop."""
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
