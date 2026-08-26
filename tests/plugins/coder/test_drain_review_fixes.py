# SPDX-License-Identifier: AGPL-3.0-or-later
"""The drain-review fixes.

THE LOAD-BEARING TEST IS THE FIRST ONE. `dispatch_parallel` could not run at
all: the dispatcher injects the hidden ``_parent_agent`` argument for a
hardcoded list of tool names, the tool was not in it, and its first statement is
``if _parent_agent is None: return error``. So the model was told the tool
existed, the user was shown a confirmation card, and every call failed. A test
file that invokes ``tool_dispatch_parallel(...)`` DIRECTLY with
``_parent_agent=parent`` supplied by hand cannot see that, because supplying it
by hand is the one thing the product never does.

``test_every_registry_tool_receives_its_hidden_args_through_the_real_dispatcher``
is the test that catches it: it walks the WHOLE registry, dispatches each tool
through the real ``Agent._execute_tool``, and asserts the tool function actually
received every hidden parameter its signature declares. It is derived from
signatures rather than a hand-kept list, so a tool added tomorrow with a new
hidden argument is covered without anyone remembering to update this file.
"""

from __future__ import annotations

import inspect
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from localm.plugins.coder import child_limit
from localm.plugins.coder.audit import SessionMode
from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.tools import ToolResult
from localm.plugins.coder.tools.git import (
    WORKTREE_PREFIX, git_list_child_worktrees, git_prune_child_worktrees,
)
from localm.plugins.coder.tools.registry import TOOL_REGISTRY


# --------------------------------------------------------------------------- #
#  Harness
# --------------------------------------------------------------------------- #

def _run(cwd: Path, *args: str) -> str:
    out = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True)
    return (out.stdout + out.stderr).strip()


class _Stub:
    model_id = "test-model"
    native_tools = False
    supports_grammar = False
    last_usage = {"total_tokens": 0}
    _base_url = "http://127.0.0.1:8642/v1"

    def chat(self, messages, **kw):
        return "Done."

    def chat_stream(self, messages, **kw):
        yield "Done."


# The real Agent class, captured at import time; the child-faking fixtures below
# patch localm.plugins.coder.agent.Agent, the name _run_one_child resolves
# lazily. Only the children are substituted.
from localm.plugins.coder.agent import Agent as _RealAgent


def _agent(cwd, **kw):
    """A real Agent, so tests go through the REAL dispatcher."""
    Agent = _RealAgent
    with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        PM.build.return_value.file_count.return_value = 0
        PM.build.return_value.truncated = False
        kw.setdefault("mode", SessionMode.LOG)
        kw.setdefault("auto_approve", True)
        return Agent(_Stub(), cwd=cwd, self_verify=False, **kw)


def _call(agent, name, **args):
    """Dispatch a tool the way the product does: through Agent._execute_tool."""
    return agent._execute_tool(
        ToolCall(name=name, args=args, raw="", start=0, end=0), interactive=False)


def _hidden_params(fn) -> list[str]:
    """Underscore-prefixed parameters: the ones a model's tool call can never
    supply, because they are absent from the tool's declared JSON schema."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):       # pragma: no cover - builtins only
        return []
    return [
        name for name, p in sig.parameters.items()
        if name.startswith("_")
        # **_ / *_ are catch-alls, not real hidden arguments.
        and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    ]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "proj"
    r.mkdir()
    _run(r, "git", "init", "-b", "trunk")
    _run(r, "git", "config", "user.email", "test@example.invalid")
    _run(r, "git", "config", "user.name", "Test")
    _run(r, "git", "config", "commit.gpgsign", "false")
    (r / "shared.txt").write_text("original\n", encoding="utf-8")
    _run(r, "git", "add", "-A")
    _run(r, "git", "commit", "-m", "initial")
    return r


@pytest.fixture(autouse=True)
def clean_gate():
    child_limit._reset_for_tests()
    yield
    child_limit._reset_for_tests()


# --------------------------------------------------------------------------- #
#  The registry-wide dispatch contract
# --------------------------------------------------------------------------- #

def test_every_registry_tool_receives_its_hidden_args_through_the_real_dispatcher(
        tmp_path, monkeypatch):
    """EVERY registry tool, dispatched for real, gets every hidden arg it declares.

    Run in PRIVACY mode because that is the maximal-injection configuration (it
    adds ``_privacy`` on top of ``_parent_agent`` and ``_session``), so one pass
    covers every injection site.

    The tool FUNCTION is swapped for a recorder: the subject here is the
    dispatcher's injection, not what the tools do, and actually running
    ``git_push`` or ``run_shell`` for all 32 tools is not something a unit test
    should do. Everything up to and including the call is the real code path.
    """
    agent = _agent(tmp_path, mode=SessionMode.PRIVACY)
    # Pin the network policy to allow; every fn is a recorder.
    monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "allow")

    missing: list[str] = []
    never_called: list[str] = []
    checked: list[str] = []

    for name, td in sorted(TOOL_REGISTRY.items()):
        hidden = _hidden_params(td.fn)
        if not hidden:
            continue
        checked.append(name)
        seen: dict = {}

        def recorder(cwd, _seen=seen, **kwargs):
            _seen["called"] = True
            _seen["kwargs"] = kwargs
            return ToolResult.success("recorded")

        monkeypatch.setattr(td, "fn", recorder)
        _call(agent, name)

        if not seen.get("called"):
            never_called.append(name)
            continue
        got = seen.get("kwargs", {})
        for param in hidden:
            if param not in got:
                missing.append(f"{name} never received {param}")

    # Fails if nothing was dispatched at all.
    assert checked, "no registry tool declares a hidden parameter - harness broken"
    assert not never_called, (
        "these tools never reached their function, so nothing was verified: "
        f"{never_called}")
    assert not missing, (
        "the dispatcher advertises these tools but does not give them the hidden "
        "argument they guard on, so every real call fails: " + "; ".join(missing))
    # dispatch_parallel is in the sweep.
    assert "dispatch_parallel" in checked


def test_the_parent_agent_injection_set_matches_the_registry(tmp_path):
    """The named set and the registry signatures cannot drift apart silently."""
    from localm.plugins.coder.agent.constants import _PARENT_AGENT_TOOLS

    by_signature = {
        name for name, td in TOOL_REGISTRY.items()
        if "_parent_agent" in _hidden_params(td.fn)
    }
    assert by_signature == set(_PARENT_AGENT_TOOLS), (
        "every tool that takes _parent_agent must be injected: "
        f"signatures={sorted(by_signature)} injected={sorted(_PARENT_AGENT_TOOLS)}")


def test_dispatch_parallel_gets_past_its_parent_agent_guard(tmp_path):
    """The live reproduction, as a regression test.

    An EMPTY repo is the cheap probe: reaching "no commits yet" proves the call
    got past the ``_parent_agent`` guard, past task normalisation, past the
    git-repo check and into the dispatch body - without spawning a child.
    """
    proj = tmp_path / "empty"
    proj.mkdir()
    _run(proj, "git", "init", "-b", "trunk")
    agent = _agent(proj)

    res = _call(agent, "dispatch_parallel", tasks=["a"])

    assert "requires a running parent agent" not in res.output, (
        "dispatch_parallel is still dead on arrival: " + res.output)
    assert "no commits yet" in res.output, res.output


def test_dispatch_parallel_follows_the_delegation_disable_family(tmp_path):
    """Disabling spawn_agent must not leave an equivalent tool dispatchable."""
    from localm.plugins.coder.agent.constants import expand_shell_disable

    expanded = expand_shell_disable(frozenset({"spawn_agent"}))
    assert "dispatch_parallel" in expanded, (
        "a caller that switched delegation off can still dispatch children")


# --------------------------------------------------------------------------- #
#  The process-wide child-budget leak
# --------------------------------------------------------------------------- #

def test_a_malformed_model_value_is_rejected_cleanly(repo):
    """A dict/list `model` is a shape a local model plausibly emits."""
    agent = _agent(repo)
    before = child_limit.available()

    for bad in ({"name": "q"}, ["q"]):
        res = _call(agent, "dispatch_parallel",
                    tasks=[{"task": "x", "model": bad}])
        assert not res.ok
        assert "must be a model name" in res.output, res.output
        assert "unhashable" not in res.output, (
            "the malformed value still escapes as a raw TypeError: " + res.output)

    assert child_limit.available() == before, (
        "two malformed calls leaked the process-wide child budget")


def test_the_child_budget_is_returned_when_the_acquire_loop_itself_raises(
        repo, monkeypatch):
    """The STRUCTURAL half, with an oracle that can actually tell.

    The window is between the FIRST successful acquire and the `try`. Raising
    from anything further in (say the first `_git` call) does NOT discriminate:
    that call is already inside the try, and the `finally` already releases. So
    this raises from the acquire LOOP itself, on the second slot, which is only
    covered once the loop moved inside the try/finally. Hoist the acquire back
    out and this goes red; the sibling malformed-model test would not.
    """
    before = child_limit.available()
    assert before >= 2, "this test needs at least two free slots to be meaningful"

    real_acquire = child_limit.try_acquire
    calls = {"n": 0}

    def acquire_once_then_explode(kind, label):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_acquire(kind, label)
        raise RuntimeError("synthetic failure with one slot already held")

    monkeypatch.setattr(child_limit, "try_acquire", acquire_once_then_explode)
    agent = _agent(repo)
    res = _call(agent, "dispatch_parallel", tasks=["a", "b"])

    assert not res.ok
    assert calls["n"] == 2, "the acquire loop did not reach the second slot"
    assert child_limit.available() == before, (
        "the slot taken before the raise was never returned - the acquire is "
        "outside the try/finally again, and child_limit has no reaper")


# --------------------------------------------------------------------------- #
#  A failed child must not be reported ok
# --------------------------------------------------------------------------- #

class _FailingChild:
    """A child that FAILS the way the real Agent does: run_task RETURNS the
    failure message instead of raising, and records the verdict on itself."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.cwd = Path(kwargs["cwd"])
        self.name = kwargs.get("name", "child")
        self.turns = 10
        self.last_run_ok = False
        self._error_trace = [{"tool": "run_shell", "error": "exit 1"}]

    def run_task(self, task: str) -> str:
        (self.cwd / "half.txt").write_text("partial\n", encoding="utf-8")
        return "[max_turns=10 reached]"


def test_a_failed_child_is_reported_as_error_not_ok(repo, monkeypatch):
    import localm.plugins.coder.agent as agent_mod
    monkeypatch.setattr(agent_mod, "Agent", _FailingChild)

    agent = _agent(repo)
    res = _call(agent, "dispatch_parallel", tasks=["a"])

    assert "[error]" in res.output, (
        "a child that hit max_turns was reported to the model as ok:\n" + res.output)
    assert "[ok]" not in res.output
    assert not res.ok, "ToolResult.ok is True for a run in which every child failed"


def test_a_failed_childs_error_trace_reaches_the_parent(repo, monkeypatch):
    import localm.plugins.coder.agent as agent_mod
    monkeypatch.setattr(agent_mod, "Agent", _FailingChild)

    agent = _agent(repo)
    assert agent._error_trace == []
    _call(agent, "dispatch_parallel", tasks=["a"])

    assert any(e.get("error") == "exit 1" for e in agent._error_trace), (
        "the child's errors were discarded: " + repr(agent._error_trace))


def test_a_failed_background_child_is_not_absorbed_as_ok(tmp_path):
    """The background path (persistence.py)."""
    from localm.plugins.coder import delegated as _delegated

    agent = _agent(tmp_path)
    finished = [{
        "id": "job1", "label": "worker", "state": "done",
        "result": {"summary": "[max_turns=10 reached]", "turns": 10, "ok": False,
                   "branch": "coder/worker-abc", "base": "deadbeef",
                   "file_count": 1, "diff": "diff --git a/x b/x"},
    }]

    class _Registry:
        def drain_finished(self, kind=None):
            return finished

        def get(self, job_id):
            return None

    with patch("localm.plugins.coder.background.get_registry",
               return_value=_Registry()):
        notes = agent._drain_background_agents()

    sets = list(getattr(agent, "_delegated", []))
    assert sets and sets[0].status == "error", (
        "a failed background sub-agent was recorded as an ok change set: "
        f"{[s.status for s in sets]}")
    assert notes and "DID NOT COMPLETE" in notes[0], notes
    assert _delegated.footer_for(agent)


def test_polling_a_failed_background_child_does_not_report_it_finished(tmp_path):
    """The model's own polling surface (check_agent_job) is a third X3 site."""
    from localm.plugins.coder.tools.agents import tool_check_agent_job

    class _Job:
        kind = "agent"

        def status(self):
            return {"state": "done", "label": "worker", "warnings": [],
                    "result": {"summary": "[max_turns=10 reached]", "turns": 10,
                               "ok": False}}

        def elapsed(self):
            return 1.0

    class _Registry:
        def get(self, job_id):
            return _Job()

        def ids(self, kind=None):
            return ["job1"]

    with patch("localm.plugins.coder.background.get_registry",
               return_value=_Registry()):
        res = tool_check_agent_job(tmp_path, "job1")

    assert "DID NOT COMPLETE" in res.output, res.output
    assert "finished in" not in res.output, res.output
    # A successful poll does not count against the failure breaker.
    assert res.ok


# --------------------------------------------------------------------------- #
#  A queued child that never started, and the budget it holds
# --------------------------------------------------------------------------- #

class _SlowChild:
    """Child 1 runs long enough to burn the batch deadline; child 2 would be
    quick but never gets a slot, because only one token was available."""

    started = threading.Event()
    release = threading.Event()
    ran: list = []

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.cwd = Path(kwargs["cwd"])
        self.name = kwargs.get("name", "child")
        self.turns = 1
        self.last_run_ok = True

    def run_task(self, task: str) -> str:
        _SlowChild.ran.append(self.name)
        _SlowChild.started.set()
        # Held open past the batch deadline, then released by the test.
        _SlowChild.release.wait(timeout=30)
        return f"{self.name} done"


@pytest.fixture
def slow_child(monkeypatch):
    _SlowChild.started = threading.Event()
    _SlowChild.release = threading.Event()
    _SlowChild.ran = []
    import localm.plugins.coder.agent as agent_mod
    monkeypatch.setattr(agent_mod, "Agent", _SlowChild)
    yield _SlowChild
    _SlowChild.release.set()


def test_synchronous_spawn_agent_does_not_report_a_failed_child_as_finished(
        repo, monkeypatch):
    """The FOURTH X3 site, and the most-used delegation surface."""
    from localm.plugins.coder.tools import agents as ag

    class _Child:
        turns = 7
        last_run_ok = False
        _error_trace: list = []
        _changed_files: dict = {}

        def run_task(self, task):
            return "[max_turns=7 reached]"

    monkeypatch.setattr(ag, "_prepare_child",
                        lambda *a, **kw: (_Child(), "the task"))
    agent = _agent(repo)
    res = ag.tool_spawn_agent(repo, task="do a thing", _parent_agent=agent)

    assert "DID NOT COMPLETE" in res.summary, res.summary
    assert "finished in" not in res.summary, res.summary


def test_the_scoped_prune_probe_forces_english_git_messages(repo, monkeypatch):
    """The line we parse is gettext-translated.

    On a git build shipping message catalogs, a localized line fails the regex,
    the fail-closed branch reports OUR OWN records as foreign, and cleanup is
    disabled permanently on that machine. The probe must pin the locale.
    """
    from localm.plugins.coder.tools import git as gitmod

    seen: dict = {}
    real_git = gitmod._git

    def spy(cwd, *args, **kw):
        if "prune" in args and "-n" in args:
            seen["env"] = kw.get("env")
        return real_git(cwd, *args, **kw)

    monkeypatch.setattr(gitmod, "_git", spy)
    gitmod.git_prune_child_worktrees(repo)

    env = seen.get("env")
    assert env is not None, "the prune probe ran with the ambient locale"
    assert env.get("LC_ALL") == "C", env.get("LC_ALL")
    # env is merged onto the real environment, never passed bare.
    assert "PATH" in env or "Path" in env, sorted(env)[:10]


def test_a_queued_child_that_never_started_says_so_and_leaves_no_worktree(
        repo, slow_child):
    """One slot, two tasks, deadline burned by the first child.

    The second child never runs. Calling that a 600s timeout whose 'worktree is
    still held by the running thread' is false in every clause, and the teardown
    skips its worktree on that basis, leaking it with no reaper.
    """
    # Take one of the two slots so the dispatch can only get one.
    outsider = child_limit.try_acquire("test", "outsider")
    assert outsider is not None

    agent = _agent(repo)
    res = _call(agent, "dispatch_parallel", tasks=["a", "b"], timeout_s=1)

    assert slow_child.ran == ["child1"], (
        f"expected only child1 to run, got {slow_child.ran}")
    assert "never started" in res.output, res.output
    assert "[not_started]" in res.output, res.output

    # Nothing of the never-started child is left on disk: no worktree, and no
    # branch from `git worktree add -b`.
    slow_child.release.set()
    leftover = [p for p in git_list_child_worktrees(repo) if "child2" in p.name]
    assert leftover == [], f"the never-started child leaked a worktree: {leftover}"
    branches = _run(repo, "git", "branch", "--list", "coder/*")
    assert "child2" not in branches, (
        f"the never-started child left an empty branch behind: {branches}")


def test_the_budget_is_held_until_an_abandoned_child_actually_ends(repo, slow_child):
    """X10: releasing on return admitted new children beside a live abandoned one."""
    agent = _agent(repo)
    before = child_limit.available()

    _call(agent, "dispatch_parallel", tasks=["a"], timeout_s=1)

    # The child was abandoned but its thread is still running, so its slot is
    # still taken.
    assert child_limit.available() == before - 1, (
        "the budget was returned while an abandoned child was still running")

    # Let it finish; the done-callback returns the slot.
    slow_child.release.set()
    deadline = time.monotonic() + 10
    while child_limit.available() != before and time.monotonic() < deadline:
        time.sleep(0.05)
    assert child_limit.available() == before, (
        "the slot was never returned after the abandoned child ended")


# --------------------------------------------------------------------------- #
#  The repo-wide prune must not discard a worktree we do not own
# --------------------------------------------------------------------------- #

def _register_worktree(repo: Path, path: Path, branch: str) -> None:
    _run(repo, "git", "worktree", "add", str(path), "-b", branch)
    assert path.exists()


def test_a_foreign_worktree_record_survives_the_coder_prune(repo, tmp_path):
    """A user worktree on a drive that is not mounted looks 'missing' to git.

    `git worktree prune` takes no pathspec, so an unscoped prune in the coder's
    teardown drops that record too. Recovery needs `git worktree repair`, and
    the coder never locks the worktrees it does not own.
    """
    foreign = tmp_path / "user-work"
    _register_worktree(repo, foreign, "user/feature")
    ours = tmp_path / f"{WORKTREE_PREFIX}child1-abc123"
    _register_worktree(repo, ours, "coder/child1-abc123")

    # Both directories vanish.
    import shutil
    shutil.rmtree(foreign)
    shutil.rmtree(ours)

    msg, ok = git_prune_child_worktrees(repo)

    assert not ok, "the prune ran even though it would drop a foreign record"
    assert "user-work" in msg, msg
    listed = _run(repo, "git", "worktree", "list")
    assert "user-work" in listed, (
        "the user's worktree record was destroyed by the coder's cleanup:\n" + listed)


def test_the_scoped_prune_still_reaps_our_own_records(repo, tmp_path):
    """The fires-control for the test above: with only OUR record stale, it prunes.

    Without this, 'never prune anything' would pass the foreign-record test while
    silently disabling the cleanup.
    """
    ours = tmp_path / f"{WORKTREE_PREFIX}child1-abc123"
    _register_worktree(repo, ours, "coder/child1-abc123")
    import shutil
    shutil.rmtree(ours)
    assert f"{WORKTREE_PREFIX}child1-abc123" in _run(repo, "git", "worktree", "list")

    msg, ok = git_prune_child_worktrees(repo)

    assert ok, msg
    assert f"{WORKTREE_PREFIX}child1-abc123" not in _run(
        repo, "git", "worktree", "list")


def test_the_background_finalizer_uses_the_scoped_prune(repo, tmp_path):
    """The background teardown must use the scoped prune, not a bare one."""
    from localm.plugins.coder.tools.agents import _finalize_isolated_child

    foreign = tmp_path / "user-work"
    _register_worktree(repo, foreign, "user/feature")
    import shutil
    shutil.rmtree(foreign)

    ours = tmp_path / f"{WORKTREE_PREFIX}bg-abc123"
    _register_worktree(repo, ours, "coder/bg-abc123")
    info = {"worktree": ours, "repo": repo, "base": "HEAD",
            "branch": "coder/bg-abc123", "name": "bg", "run_id": "abc123"}

    out = _finalize_isolated_child(info)(object())

    listed = _run(repo, "git", "worktree", "list")
    assert "user-work" in listed, (
        "the background teardown destroyed a foreign worktree record:\n" + listed)
    # The skip is reported, not swallowed.
    assert "user-work" in (out.get("cleanup_warning") or ""), out


# --------------------------------------------------------------------------- #
#  A destructive tool must not run beside a slow peer
# --------------------------------------------------------------------------- #

def test_a_destructive_tool_does_not_run_beside_an_abandoned_peer(tmp_path,
                                                                  monkeypatch):
    """run_tests is non-destructive and a real suite outlives the batch deadline.

    The timeout path cancels (a no-op on a running future) and shuts the pool down
    without joining, so without this gate the destructive segment starts while the
    peer is still executing - the exact stacked concurrency destructive=True
    prevents.
    """
    agent = _agent(tmp_path)
    monkeypatch.setattr(type(agent), "_PARALLEL_BATCH_TIMEOUT_S", 0.2)
    monkeypatch.setattr(type(agent), "_ABANDONED_PEER_GRACE_S", 0.2)

    release = threading.Event()
    executed: list = []

    def fake_execute(call, interactive=False):
        executed.append(call.name)
        if call.name in ("read_file", "grep"):
            release.wait(timeout=30)
        return ToolResult.success("done")

    monkeypatch.setattr(agent, "_execute_tool", fake_execute)

    calls = [
        ToolCall(name="read_file", args={}, raw="", start=0, end=0),
        ToolCall(name="grep", args={}, raw="", start=0, end=0),
        ToolCall(name="dispatch_parallel", args={}, raw="", start=0, end=0),
    ]
    try:
        blocks = agent._execute_tools(calls, interactive=False)
    finally:
        release.set()

    assert "dispatch_parallel" not in executed, (
        "the destructive tool ran while a non-destructive peer was still going")
    joined = "\n".join(blocks)
    assert "runs alone" in joined, joined
    assert "still running" in joined, joined


def test_the_destructive_gate_survives_into_the_next_turn(tmp_path, monkeypatch):
    """The refusal tells the model to wait, so the NEXT turn must gate too.

    _execute_tools runs once per turn. An abandoned list living in that call
    frame is empty again immediately, so a model that did what the refusal said
    walks into an ungated dispatch while the peer is still running - the fix's
    own advice routing it into the hole the fix exists to close.
    """
    agent = _agent(tmp_path)
    monkeypatch.setattr(type(agent), "_PARALLEL_BATCH_TIMEOUT_S", 0.2)
    monkeypatch.setattr(type(agent), "_ABANDONED_PEER_GRACE_S", 0.2)

    release = threading.Event()
    executed: list = []

    def fake_execute(call, interactive=False):
        executed.append(call.name)
        if call.name in ("read_file", "grep"):
            release.wait(timeout=30)
        return ToolResult.success("done")

    monkeypatch.setattr(agent, "_execute_tool", fake_execute)

    turn1 = [ToolCall(name=n, args={}, raw="", start=0, end=0)
             for n in ("read_file", "grep")]
    # A later, separate turn.
    turn2 = [ToolCall(name="dispatch_parallel", args={}, raw="", start=0, end=0)]
    try:
        agent._execute_tools(turn1, interactive=False)
        blocks = agent._execute_tools(turn2, interactive=False)
    finally:
        release.set()

    assert "dispatch_parallel" not in executed, (
        "the destructive tool ran on the next turn while the abandoned peer from "
        "the previous turn was still running")
    assert "still running" in "\n".join(blocks)


def test_a_destructive_tool_still_runs_when_its_peers_finished(tmp_path,
                                                               monkeypatch):
    """Fires-control for the test above: the gate must not block the normal case."""
    agent = _agent(tmp_path)
    executed: list = []

    def fake_execute(call, interactive=False):
        executed.append(call.name)
        return ToolResult.success("done")

    monkeypatch.setattr(agent, "_execute_tool", fake_execute)
    calls = [
        ToolCall(name="read_file", args={}, raw="", start=0, end=0),
        ToolCall(name="grep", args={}, raw="", start=0, end=0),
        ToolCall(name="dispatch_parallel", args={}, raw="", start=0, end=0),
    ]
    agent._execute_tools(calls, interactive=False)

    assert "dispatch_parallel" in executed, (
        "the new gate blocks a destructive tool even with no live peer")
