# SPDX-License-Identifier: AGPL-3.0-or-later
"""Coder episode TRIGGERS + reflection evidence.

These drive the REAL Agent dispatch / close path (not mocks of the unit under
test): the tool-failure trace is captured through the real _execute_tool, git
change detection runs against a real throwaway repo, and spawn_agent folding runs
through the real tool. Every test isolates LOCALM_HOME under a tmp dir.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from localm.audit import SessionMode
from localm.plugins.coder.parser import ToolCall


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


_GOOD_REPLY = ('{"summary": "ran the generator", "what_worked": "the codegen script", '
               '"what_failed": "", "lesson": "regenerate after schema edits"}')


class _ChatBackend(_StubBackend):
    """Backend whose .chat returns a canned reflection reply."""

    def __init__(self, reply: str = _GOOD_REPLY):
        self._reply = reply
        self.calls: list = []

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        return self._reply


class _ScriptBackend(_StubBackend):
    """Backend that replays a fixed sequence of assistant turns (the last repeats)."""

    model_id = "script-model"

    def __init__(self, replies):
        self._replies = list(replies)
        self.i = 0

    def chat(self, messages, **kw):
        r = self._replies[min(self.i, len(self._replies) - 1)]
        self.i += 1
        return r


def _agent(tmp_path, backend=None, **kw):
    from localm.plugins.coder.agent import Agent
    return Agent(backend or _StubBackend(), cwd=tmp_path, **kw)


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=str(cwd), check=True,
                   capture_output=True, text=True)


# --------------------------------------------------------------------------- #
#  The tool-failure trace is captured and reaches the reflection              #
# --------------------------------------------------------------------------- #

def test_tool_failure_is_recorded_in_error_trace(home, tmp_path):
    # A read_file on a missing path fails through the real dispatch path and
    # lands in the bounded error trace that feeds the reflection.
    agent = _agent(tmp_path, mode=SessionMode.LOG)
    call = ToolCall(name="read_file", args={"path": "does_not_exist.py"},
                    raw="", start=0, end=0)
    result = agent._execute_tool(call, interactive=False)
    assert not result.ok
    assert agent._error_trace and agent._error_trace[0].startswith("read_file:")


def test_error_trace_is_bounded(home, tmp_path):
    from localm.plugins.coder.agent.constants import _MAX_ERROR_TRACE
    agent = _agent(tmp_path, mode=SessionMode.LOG)
    for i in range(_MAX_ERROR_TRACE + 10):
        agent._record_error("t", f"error number {i}")
    assert len(agent._error_trace) == _MAX_ERROR_TRACE
    # Newest kept (the ones that ended the run), oldest dropped.
    assert agent._error_trace[-1] == "t: error number %d" % (_MAX_ERROR_TRACE + 9)


def test_failed_no_change_session_stores_thin_failure_episode(home, tmp_path):
    # An investigation-only session that failed (no file change) still records a
    # failure lesson when the model reflects nothing usable.
    agent = _agent(tmp_path, backend=_ChatBackend("no idea, sorry"),
                   mode=SessionMode.LOG)
    agent._episode_task = "find why the importer crashes"
    agent._last_run_ok = False
    agent._error_trace = ["read_file: no such file: importer.py",
                          "run_shell: grep: importer: no such file"]
    agent.close()                                   # on_event None -> synchronous
    eps = agent._episode_store.all()
    assert len(eps) == 1
    assert eps[0].outcome == "incomplete"
    assert "importer" in eps[0].what_failed


# --------------------------------------------------------------------------- #
#  The CLI's synchronous close-time reflection is bounded                     #
# --------------------------------------------------------------------------- #

def test_cli_close_reflection_is_bounded_not_unbounded(home, tmp_path, monkeypatch):
    """A no-file-change FAILED session (max_turns / a circuit breaker / a failed
    verify oracle) must not block CLI exit for the full duration of a slow or
    wedged model call. Patches the deadline down so the test itself stays fast
    while still proving the bound is real."""
    import threading
    import time

    import localm.plugins.coder.agent.session as _session_mod
    monkeypatch.setattr(_session_mod, "_CLI_REFLECTION_DEADLINE_S", 0.2)

    started = threading.Event()
    release = threading.Event()

    class _BlockingBackend(_StubBackend):
        def chat(self, messages, **kw):
            started.set()
            release.wait(5)          # would hang the test outright if unbounded
            return _GOOD_REPLY

    agent = _agent(tmp_path, backend=_BlockingBackend(), mode=SessionMode.LOG)
    agent._episode_task = "investigate a crash"
    agent._last_run_ok = False        # no file change, but a real failure
    agent._error_trace = ["read_file: no such file: importer.py"]

    t0 = time.monotonic()
    agent.close()                     # on_event None -> synchronous, now bounded
    elapsed = time.monotonic() - t0

    assert started.wait(5), "premise broken: the model call never started"
    release.set()                     # free the leaked daemon thread
    assert elapsed < 2.0, (
        f"close() took {elapsed:.2f}s - the reflection deadline did not bound it")
    # On timeout, episodes.py's thin-failure fallback still fires from the real
    # error trace, so the session's lesson is recorded.
    eps = agent._episode_store.all()
    assert len(eps) == 1
    assert eps[0].outcome == "incomplete"
    assert "importer" in eps[0].what_failed


def test_cli_close_reflection_stores_the_full_episode_within_deadline(
        home, tmp_path, monkeypatch):
    """Negative for the bound: a normal-speed reflection must not be truncated
    or downgraded to the thin fallback just because it is wrapped in a deadline.
    The bound must change only the worst case, not what gets stored in the
    common one."""
    import localm.plugins.coder.agent.session as _session_mod
    monkeypatch.setattr(_session_mod, "_CLI_REFLECTION_DEADLINE_S", 5.0)

    agent = _agent(tmp_path, backend=_ChatBackend(), mode=SessionMode.LOG)
    agent._episode_task = "find why the importer crashes"
    agent._last_run_ok = False
    agent._error_trace = ["read_file: no such file: importer.py"]
    agent.close()

    eps = agent._episode_store.all()
    assert len(eps) == 1
    # The FULL reflection landed (from _GOOD_REPLY), not the thin fallback.
    assert eps[0].summary == "ran the generator"
    assert eps[0].lesson == "regenerate after schema edits"


def test_cli_close_prints_a_reflecting_notice_before_the_synchronous_call(
        home, tmp_path, monkeypatch):
    """The synchronous wait must be visible, not a silent hang."""
    import localm.plugins.coder.agent.session as _session_mod
    printed: list = []
    monkeypatch.setattr(_session_mod, "print_info", printed.append)

    agent = _agent(tmp_path, backend=_ChatBackend(), mode=SessionMode.LOG)
    agent._episode_task = "investigate a crash"
    agent._last_run_ok = False
    agent._error_trace = ["read_file: no such file: importer.py"]
    agent.close()

    assert any("reflect" in m.lower() for m in printed), (
        "no visible notice was printed before the synchronous close-time "
        "reflection")


def test_gui_session_reflection_stays_unbounded_and_gets_no_notice(
        home, tmp_path, monkeypatch):
    """Negative for the bound's SCOPE: the GUI/web path (on_event set) already
    backgrounds the call off the event loop with nobody waiting on it, so it
    must keep passing deadline=None and print no CLI-only notice."""
    import localm.plugins.coder.agent.session as _session_mod
    printed: list = []
    monkeypatch.setattr(_session_mod, "print_info", printed.append)
    monkeypatch.setattr(_session_mod, "_CLI_REFLECTION_DEADLINE_S", 0.01)

    agent = _agent(tmp_path, backend=_ChatBackend(), mode=SessionMode.LOG,
                   on_event=lambda e: None)
    agent._episode_task = "investigate a crash"
    agent._last_run_ok = False
    agent._error_trace = ["read_file: no such file: importer.py"]
    agent.close()

    for _ in range(100):                # the background thread stores async
        if agent._episode_store.all():
            break
        import time
        time.sleep(0.02)
    eps = agent._episode_store.all()
    assert len(eps) == 1
    # The full reflection landed, not the thin fallback: the CLI deadline does
    # not apply on this path.
    assert eps[0].summary == "ran the generator"
    assert printed == [], "the GUI path must not print the CLI-only notice"


def test_clean_no_change_session_stores_nothing(home, tmp_path):
    # The benign case stays silent: no changes, no failures.
    agent = _agent(tmp_path, backend=_ChatBackend(), mode=SessionMode.LOG)
    agent._episode_task = "what does this function do"
    assert agent._last_run_ok is True
    agent.close()
    assert agent._episode_store.all() == []


# --------------------------------------------------------------------------- #
#  run_shell writes recovered via git; delegated work folded                   #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_shell_driven_session_stores_episode_via_git(home, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    (repo / "base.py").write_text("print('base')\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "base")

    agent = _agent(repo, backend=_ChatBackend(), mode=SessionMode.LOG)
    assert agent._episodic is True

    # Real dispatch of a run_shell command captures the pre-mutation git baseline
    # (a clean tree here). No write-tool ever runs, so the write-tracker stays empty.
    probe = ToolCall(name="run_shell", args={"command": "git --version"},
                     raw="", start=0, end=0)
    agent._execute_tool(probe, interactive=False)
    assert agent._shell_baseline_captured is True
    assert agent._git_baseline == frozenset()       # clean tree at baseline

    # A codegen shell command then writes a new file (simulated portably).
    (repo / "generated.py").write_text("print('gen')\n", encoding="utf-8")
    agent._episode_task = "regenerate the client from the schema"

    agent.close()
    eps = agent._episode_store.all()
    assert len(eps) == 1, "shell-driven change was not detected/reflected"
    assert "generated.py" in eps[0].files
    assert eps[0].lesson == "regenerate after schema edits"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_git_fallback_ignores_preexisting_dirty_tree(home, tmp_path):
    # A file already dirty BEFORE the session (in the baseline) must NOT be
    # attributed to this session: no session change -> no episode.
    repo = tmp_path / "repo2"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    (repo / "base.py").write_text("v1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "base")
    # Pre-existing uncommitted edit (the user's, not this session's).
    (repo / "base.py").write_text("v2 dirty\n", encoding="utf-8")

    agent = _agent(repo, backend=_ChatBackend(), mode=SessionMode.LOG)
    probe = ToolCall(name="run_shell", args={"command": "git --version"},
                     raw="", start=0, end=0)
    agent._execute_tool(probe, interactive=False)
    assert "base.py" in agent._git_baseline           # already dirty at baseline
    agent._episode_task = "look around"
    agent.close()
    # The session changed nothing of its own -> nothing to learn, nothing stored.
    assert agent._episode_store.all() == []


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_git_fallback_diff_is_scoped_to_session_delta(home, tmp_path):
    # The reflected WORK LOG must contain only THIS session's change, not a
    # pre-existing dirty file that happened to be uncommitted before the session.
    repo = tmp_path / "repo3"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    (repo / "base.py").write_text("committed_v1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "base")
    # Pre-existing uncommitted edit (NOT this session's work).
    (repo / "base.py").write_text("PREEXISTING_DIRTY_EDIT\n", encoding="utf-8")

    captured = {}

    class _Capture(_StubBackend):
        def chat(self, messages, **kw):
            captured["prompt"] = messages[0]["content"]
            return _GOOD_REPLY

    agent = _agent(repo, backend=_Capture(), mode=SessionMode.LOG)
    probe = ToolCall(name="run_shell", args={"command": "git --version"},
                     raw="", start=0, end=0)
    agent._execute_tool(probe, interactive=False)          # baseline: base.py dirty
    assert "base.py" in agent._git_baseline
    # The session then creates a NEW file (its own work).
    (repo / "session_out.py").write_text("SESSION_MADE_THIS\n", encoding="utf-8")
    agent._episode_task = "generate session_out.py"
    agent.close()

    eps = agent._episode_store.all()
    assert len(eps) == 1
    assert eps[0].files == ["session_out.py"]              # list scoped to the delta
    work_log = captured["prompt"]
    assert "SESSION_MADE_THIS" in work_log                 # this session's change is in
    assert "PREEXISTING_DIRTY_EDIT" not in work_log        # the pre-existing one is NOT


def test_absorb_child_state_folds_changes_and_errors(home, tmp_path):
    parent = _agent(tmp_path, mode=SessionMode.LOG)
    child = _agent(tmp_path, mode=SessionMode.LOG)
    parent._changed_files = {"x.py": {"original": b"old", "writes": 1,
                                      "last_tool": "edit_file"}}
    child._changed_files = {
        "x.py": {"original": None, "writes": 2, "last_tool": "write_file"},
        "z.py": {"original": None, "writes": 1, "last_tool": "write_file"},
    }
    child._error_trace = ["run_shell: boom"]

    parent._absorb_child_state(child)

    # Overlapping file: writes summed, parent's first-seen original kept.
    assert parent._changed_files["x.py"]["writes"] == 3
    assert parent._changed_files["x.py"]["original"] == b"old"
    assert parent._changed_files["x.py"]["last_tool"] == "write_file"
    # Child-only file absorbed.
    assert "z.py" in parent._changed_files
    # Failure trace folded so delegated failures reach the parent's reflection.
    assert "run_shell: boom" in parent._error_trace


def test_spawn_agent_folds_child_changes_into_parent(home, tmp_path):
    # Real spawn_agent: the child writes a file, and the parent's changed-files
    # tracker must include it after the tool returns (the child is never close()d).
    from localm.plugins.coder.tools.agents import tool_spawn_agent

    write_call = ('<tool_call>\n'
                  '{"name": "write_file", "args": {"path": "child.py", '
                  '"content": "print(1)\\n"}}\n'
                  '</tool_call>')
    parent = _agent(tmp_path, backend=_ScriptBackend([write_call, "all done"]),
                    mode=SessionMode.LOG, auto_approve=True)

    result = tool_spawn_agent(tmp_path, task="create child.py",
                              _parent_agent=parent)
    assert result.ok, result.output
    assert (tmp_path / "child.py").exists()
    parent_paths = [c["path"] for c in parent.changed_files()]
    assert "child.py" in parent_paths
