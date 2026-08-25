# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for worktree-isolated parallel dispatch (tools/parallel.py).

These drive the REAL code path against a REAL temporary git repository - real
`git worktree add`, real commits, real diffs. Only the child Agent is substituted,
because running two actual local models in a unit test would make the suite depend
on a GPU. The isolation being tested (two worktrees, untouched parent tree, real
cleanup) is exercised for real, not mocked.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from localm.plugins.coder import child_limit
from localm.plugins.coder.tools import parallel as par
from localm.plugins.coder.tools.git import git_list_child_worktrees


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _run(cwd: Path, *args: str) -> str:
    out = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True)
    return (out.stdout + out.stderr).strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real git repo with one commit, so `git worktree add` has a base."""
    r = tmp_path / "proj"
    r.mkdir()
    _run(r, "git", "init", "-b", "trunk")
    _run(r, "git", "config", "user.email", "test@example.invalid")
    _run(r, "git", "config", "user.name", "Test")
    _run(r, "git", "config", "commit.gpgsign", "false")
    (r / "shared.txt").write_text("original\n", encoding="utf-8")
    (r / "README.md").write_text("# proj\n", encoding="utf-8")
    _run(r, "git", "add", "-A")
    _run(r, "git", "commit", "-m", "initial")
    return r


@pytest.fixture(autouse=True)
def clean_gate():
    """Every test starts with the full child budget free."""
    child_limit._reset_for_tests()
    yield
    child_limit._reset_for_tests()


class FakeAgent:
    """Stands in for the real Agent. Writes a file inside whatever cwd it is given.

    Registers every cwd it ever saw so a test can prove two children were handed
    two DIFFERENT directories.
    """

    seen_cwds: list[Path] = []
    behaviour = {}          # name -> callable(agent) run inside run_task

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.cwd = Path(kwargs["cwd"])
        self.name = kwargs.get("name", "child")
        self.turns = 1
        FakeAgent.seen_cwds.append(self.cwd)

    def run_task(self, task: str) -> str:
        hook = FakeAgent.behaviour.get(self.name)
        if hook is not None:
            return hook(self)
        (self.cwd / "shared.txt").write_text(f"written by {self.name}\n",
                                             encoding="utf-8")
        return f"{self.name} done"


@pytest.fixture(autouse=True)
def fake_agent(monkeypatch):
    """Patch the Agent the tool lazily imports (same seam spawn_agent's tests use)."""
    FakeAgent.seen_cwds = []
    FakeAgent.behaviour = {}
    import localm.plugins.coder.agent as agent_mod
    monkeypatch.setattr(agent_mod, "Agent", FakeAgent)
    yield FakeAgent
    FakeAgent.seen_cwds = []
    FakeAgent.behaviour = {}


class DummyParent:
    """Minimal stand-in for the parent Agent."""

    def __init__(self, cwd: Path, **kw):
        self.cwd = cwd
        self.backend = type("B", (), {"model_id": "test-model",
                                      "_base_url": "http://127.0.0.1:8642/v1"})()
        self.auto_approve = kw.get("auto_approve", True)
        self.dry_run = False
        self.always_confirm = None
        self.confirm_handler = kw.get("confirm_handler")
        self.mode = None
        self.restricted = False
        self.disabled_tools = frozenset()
        self.scope = kw.get("scope")
        self._interactive = kw.get("interactive", False)
        self.verify_cmd = kw.get("verify_cmd")


# --------------------------------------------------------------------------
# REQUIRED: two tasks editing the SAME file do not interfere
# --------------------------------------------------------------------------

def test_two_children_editing_the_same_file_do_not_interfere(repo):
    parent = DummyParent(repo)
    res = par.tool_dispatch_parallel(
        repo, tasks=["edit shared", "also edit shared"], _parent_agent=parent,
    )
    assert res.ok, res.output

    # Each child ran in a DIFFERENT directory, and neither was the parent repo.
    assert len(FakeAgent.seen_cwds) == 2
    a, b = FakeAgent.seen_cwds
    assert a != b, "both children shared one directory - no isolation"
    assert a.resolve() != repo.resolve() and b.resolve() != repo.resolve()

    # Both children's work survived, each on its own branch, neither overwriting
    # the other. This is the property the shared-cwd design could not provide.
    branches = _run(repo, "git", "branch", "--list", "coder/*")
    assert branches.count("coder/") == 2, branches
    contents = set()
    for line in branches.splitlines():
        br = line.strip().lstrip("* ").strip()
        contents.add(_run(repo, "git", "show", f"{br}:shared.txt"))
    assert contents == {"written by child1", "written by child2"}, contents


def test_parent_working_tree_is_untouched_until_an_explicit_merge(repo):
    before_head = _run(repo, "git", "rev-parse", "HEAD")
    parent = DummyParent(repo)
    res = par.tool_dispatch_parallel(
        repo, tasks=["one", "two"], _parent_agent=parent,
    )
    assert res.ok, res.output

    # The file the children both rewrote is untouched in the parent.
    assert (repo / "shared.txt").read_text(encoding="utf-8") == "original\n"
    # HEAD did not move and the tree is clean: nothing was merged or committed here.
    assert _run(repo, "git", "rev-parse", "HEAD") == before_head
    status = _run(repo, "git", "status", "--porcelain")
    assert status == "", f"parent tree was modified: {status}"

    # And the report says so plainly.
    assert "NOTHING HAS BEEN MERGED" in res.output


def test_diffs_are_returned_for_review_and_name_their_branch(repo):
    parent = DummyParent(repo)
    res = par.tool_dispatch_parallel(repo, tasks=["a", "b"], _parent_agent=parent)
    assert "written by child1" in res.output
    assert "written by child2" in res.output
    assert "coder/child1-" in res.output and "coder/child2-" in res.output
    # A concrete, correct next step for the human, not an auto-merge.
    assert "git" in res.output and "merge" in res.output


# --------------------------------------------------------------------------
# REQUIRED: worktrees cleaned up on success AND failure
# --------------------------------------------------------------------------

def test_worktrees_are_cleaned_up_on_the_success_path(repo):
    parent = DummyParent(repo)
    res = par.tool_dispatch_parallel(repo, tasks=["a", "b"], _parent_agent=parent)
    assert res.ok
    assert git_list_child_worktrees(repo) == [], "worktrees leaked on success"
    wt_root = repo / ".claude" / "worktrees"
    leftovers = [p for p in (wt_root.iterdir() if wt_root.exists() else [])
                 if p.name.startswith("coder-child-")]
    assert leftovers == [], f"worktree directories left on disk: {leftovers}"


def test_worktrees_are_cleaned_up_when_a_child_raises(repo):
    def boom(agent):
        raise RuntimeError("child exploded")

    FakeAgent.behaviour = {"child1": boom}
    parent = DummyParent(repo)
    res = par.tool_dispatch_parallel(repo, tasks=["a", "b"], _parent_agent=parent)

    # The failure is REPORTED, not swallowed.
    assert "child exploded" in res.output
    # ...and the failing child's worktree is still cleaned up.
    assert git_list_child_worktrees(repo) == [], "worktrees leaked on the failure path"
    # The healthy sibling still produced reviewable work.
    assert "written by child2" in res.output


def test_child_budget_is_released_even_when_a_child_raises(repo):
    def boom(agent):
        raise RuntimeError("nope")

    FakeAgent.behaviour = {"child1": boom, "child2": boom}
    parent = DummyParent(repo)
    par.tool_dispatch_parallel(repo, tasks=["a", "b"], _parent_agent=parent)
    assert child_limit.available() == child_limit.MAX_CONCURRENT_CHILDREN, \
        "concurrency budget leaked after a failure"


# --------------------------------------------------------------------------
# REQUIRED: a hung child is bounded and reported, never hangs the parent
# --------------------------------------------------------------------------

def test_a_hung_child_is_bounded_and_reported(repo):
    release = threading.Event()

    def hang(agent):
        # Bounded so the test cannot wedge the suite if the guard regresses.
        release.wait(timeout=30)
        return "eventually"

    FakeAgent.behaviour = {"child1": hang}
    parent = DummyParent(repo)

    started = time.monotonic()
    try:
        res = par.tool_dispatch_parallel(
            repo, tasks=["hang", "fine"], timeout_s=2, _parent_agent=parent,
        )
        elapsed = time.monotonic() - started

        # The parent returned promptly instead of waiting on the hung child.
        assert elapsed < 25, f"parent blocked {elapsed:.1f}s on a hung child"
        assert "timeout" in res.output.lower() or "budget" in res.output.lower()
        # The healthy sibling's result still came back.
        assert "written by child2" in res.output
    finally:
        release.set()
        time.sleep(0.2)


def test_two_hung_children_do_not_double_the_parents_wait(repo):
    """Both children hanging must still cost ONE budget, not one budget each.

    Waiting `timeout_s` on each future in turn would block the parent for twice
    what the caller asked for. A test that hangs only ONE child cannot see this,
    because the healthy sibling's future returns immediately.
    """
    release = threading.Event()

    def hang(agent):
        release.wait(timeout=30)
        return "late"

    FakeAgent.behaviour = {"child1": hang, "child2": hang}
    parent = DummyParent(repo)
    started = time.monotonic()
    try:
        res = par.tool_dispatch_parallel(
            repo, tasks=["a", "b"], timeout_s=3, _parent_agent=parent,
        )
        elapsed = time.monotonic() - started
        assert elapsed < 6, (
            f"two hung children blocked the parent for {elapsed:.1f}s on a 3s "
            "budget - the per-future timeout is restarting instead of sharing "
            "one deadline"
        )
        assert res.output.lower().count("budget") >= 2 or \
            res.output.lower().count("timeout") >= 2
    finally:
        release.set()
        time.sleep(0.2)


def test_a_hung_childs_worktree_is_reported_not_silently_leaked(repo):
    release = threading.Event()

    def hang(agent):
        release.wait(timeout=30)
        return "late"

    FakeAgent.behaviour = {"child1": hang}
    parent = DummyParent(repo)
    try:
        res = par.tool_dispatch_parallel(
            repo, tasks=["hang", "ok"], timeout_s=2, _parent_agent=parent,
        )
        # We must SAY the worktree was left behind rather than pretend it is gone.
        assert "WARNING" in res.output
        assert "left in place" in res.output or "still holds it" in res.output
    finally:
        release.set()
        time.sleep(0.2)


# --------------------------------------------------------------------------
# The cap, and the shared budget
# --------------------------------------------------------------------------

def test_three_tasks_are_rejected_with_the_design_reason(repo):
    parent = DummyParent(repo)
    res = par.tool_dispatch_parallel(repo, tasks=["a", "b", "c"], _parent_agent=parent)
    assert not res.ok
    assert "at most 2" in res.output
    assert "deliberate design constraint" in res.output
    # Nothing was created for a rejected dispatch.
    assert git_list_child_worktrees(repo) == []


def test_dispatch_is_rejected_when_the_shared_budget_is_already_spent(repo):
    held = [child_limit.try_acquire("background", "job-a"),
            child_limit.try_acquire("background", "job-b")]
    try:
        parent = DummyParent(repo)
        res = par.tool_dispatch_parallel(repo, tasks=["a"], _parent_agent=parent)
        assert not res.ok
        # The rejection NAMES who holds the budget, so it is actionable.
        assert "job-a" in res.output and "job-b" in res.output
    finally:
        for t in held:
            child_limit.release(t)


def test_partial_budget_degrades_and_says_so(repo):
    held = child_limit.try_acquire("background", "job-a")
    try:
        parent = DummyParent(repo)
        res = par.tool_dispatch_parallel(repo, tasks=["a", "b"], _parent_agent=parent)
        assert res.ok, res.output
        # Degradation is announced, not silent.
        assert "reduced parallelism" in res.output
        # Both tasks still ran and produced reviewable work.
        assert "written by child1" in res.output
        assert "written by child2" in res.output
    finally:
        child_limit.release(held)


# --------------------------------------------------------------------------
# Confirmation routing
# --------------------------------------------------------------------------

def test_unattended_parent_still_fails_closed():
    """The 2026-07-09 bypass fix must survive the serialising wrapper.

    A parent with no handler and no interactive terminal has nobody to ask, so the
    child must get None (which execution.py turns into a denial) - never a
    permissive default that self-approves.
    """
    parent = DummyParent(Path("."), confirm_handler=None, interactive=False)
    assert par._serialised_confirm_handler(parent, "child1") is None


def test_parent_handler_is_inherited_and_serialised():
    parent = DummyParent(Path("."), confirm_handler=lambda call: True)
    handler = par._serialised_confirm_handler(parent, "child1")
    assert handler is not None
    assert handler(type("C", (), {"name": "write_file"})()) is True


def test_interactive_parent_without_handler_reaches_its_terminal():
    """An interactive REPL parent confirms via _confirm_tool, not a handler."""
    parent = DummyParent(Path("."), confirm_handler=None, interactive=True)
    parent._confirm_tool = lambda call: True
    handler = par._serialised_confirm_handler(parent, "child1")
    assert handler is not None, "interactive parent lost its confirmation channel"
    assert handler(type("C", (), {"name": "run_shell"})()) is True


def test_concurrent_children_never_prompt_at_the_same_time():
    """THE SERIALISATION TEST. Two children prompting at once on one terminal
    would interleave their questions and leave the human unable to tell which
    answer applied to which child."""
    overlap = []
    inside = []
    lock = threading.Lock()

    def slow_handler(call):
        with lock:
            inside.append(1)
            overlap.append(len(inside))
        time.sleep(0.05)
        with lock:
            inside.pop()
        return True

    parent = DummyParent(Path("."), confirm_handler=slow_handler)
    handlers = [par._serialised_confirm_handler(parent, f"child{i}") for i in range(2)]

    threads = []
    for h in handlers:
        for _ in range(5):
            threads.append(threading.Thread(
                target=h, args=(type("C", (), {"name": "write_file"})(),)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max(overlap) == 1, f"{max(overlap)} children prompted concurrently"


# --------------------------------------------------------------------------
# Input handling and preconditions
# --------------------------------------------------------------------------

def test_non_git_directory_is_rejected_clearly(tmp_path):
    plain = tmp_path / "notarepo"
    plain.mkdir()
    parent = DummyParent(plain)
    res = par.tool_dispatch_parallel(plain, tasks=["a"], _parent_agent=parent)
    assert not res.ok
    assert "git repository" in res.output


def test_requires_a_parent_agent(repo):
    res = par.tool_dispatch_parallel(repo, tasks=["a"], _parent_agent=None)
    assert not res.ok


@pytest.mark.parametrize("bad", [None, 123, [123]])
def test_malformed_tasks_are_rejected(repo, bad):
    parent = DummyParent(repo)
    res = par.tool_dispatch_parallel(repo, tasks=bad, _parent_agent=parent)
    assert not res.ok


def test_task_objects_and_bare_strings_are_both_accepted(repo):
    parent = DummyParent(repo)
    res = par.tool_dispatch_parallel(
        repo,
        tasks=[{"task": "do a thing", "name": "Refactor Auth!"}, "plain string"],
        _parent_agent=parent,
    )
    assert res.ok, res.output
    # The unsafe label was slugified into a usable branch name.
    assert "coder/refactor-auth-" in res.output


def test_two_distinct_models_surface_the_residency_condition(repo, monkeypatch):
    """Two models only run side by side if both fit in VRAM; say so rather than
    letting the user assume parallelism they may not be getting."""
    def fake_backend(model, port=8642):
        return type("B", (), {"model_id": model, "_base_url": ""})()

    import localm.plugins.coder.backends.http as http_mod
    monkeypatch.setattr(http_mod, "make_localm_backend", fake_backend)

    parent = DummyParent(repo)
    res = par.tool_dispatch_parallel(
        repo,
        tasks=[{"task": "a", "name": "c1", "model": "model-a"},
               {"task": "b", "name": "c2", "model": "model-b"}],
        _parent_agent=parent,
    )
    assert "two different models were requested" in res.output
    # And each child reports the model it ACTUALLY ran on, so an eviction-driven
    # fallback is visible instead of silent.
    assert "model:    model-a" in res.output
    assert "model:    model-b" in res.output


def test_single_model_dispatch_does_not_nag_about_residency(repo):
    """The safe default (both children on the resident model) says nothing."""
    parent = DummyParent(repo)
    res = par.tool_dispatch_parallel(repo, tasks=["a", "b"], _parent_agent=parent)
    assert "two different models were requested" not in res.output


def test_child_does_not_get_wider_reach_than_its_parent(repo):
    """A parent working in repo/sub must not spawn a child at the worktree ROOT.

    cwd is what confines the file tools, so handing the child the root would let it
    read and write files the parent itself could not touch - a quiet widening of
    reach, in the very feature whose job is confinement.
    """
    sub = repo / "sub"
    sub.mkdir()
    (sub / "inner.txt").write_text("inner\n", encoding="utf-8")
    _run(repo, "git", "add", "-A")
    _run(repo, "git", "commit", "-m", "add sub")

    captured = {}

    def record(agent):
        captured["cwd"] = agent.cwd
        return "ok"

    FakeAgent.behaviour = {"child1": record}
    parent = DummyParent(sub)
    par.tool_dispatch_parallel(sub, tasks=["a"], _parent_agent=parent)

    child_cwd = captured["cwd"]
    assert child_cwd.name == "sub", (
        f"child ran at {child_cwd}, widening its reach beyond the parent's 'sub'"
    )
    # Still inside an isolated worktree, not the real repo.
    assert "coder-child-" in str(child_cwd)
    assert child_cwd.resolve() != sub.resolve()


def test_child_is_confined_to_its_worktree_and_knows_where_it_is(repo):
    parent = DummyParent(repo, scope="src/**")
    captured = {}

    def record(agent):
        captured["cwd"] = agent.cwd
        captured["scope"] = getattr(agent, "scope", "MISSING")
        captured["worktree_path"] = getattr(agent, "worktree_path", None)
        captured["branch"] = getattr(agent, "worktree_branch", None)
        return "ok"

    FakeAgent.behaviour = {"child1": record}
    par.tool_dispatch_parallel(repo, tasks=["a"], _parent_agent=parent)

    # cwd is the isolation mechanism (every file tool resolves through _confine).
    assert captured["cwd"].resolve() != repo.resolve()
    assert "coder-child-" in str(captured["cwd"])
    # A parent scope is inherited as an ADDITIONAL narrowing.
    assert captured["scope"] == "src/**"
    # The child can report which worktree/branch its diff belongs to.
    assert captured["worktree_path"] and captured["branch"]


# --------------------------------------------------------------------------
# An isolated child's diff lands in a tree the parent's own verify_cmd (if
# any) never sees, so it needs an oracle of its own - see
# tools/agents.py:_isolated_verify_cmd.
# --------------------------------------------------------------------------

def test_isolated_child_inherits_an_explicit_parent_verify_cmd(repo):
    """An explicit choice at the parent must not be silently replaced by a
    different auto-detected command for the child."""
    parent = DummyParent(repo, verify_cmd="pytest tests/only_this.py -x")
    captured = {}

    def record(agent):
        captured["verify_cmd"] = agent.verify_cmd
        return "ok"

    FakeAgent.behaviour = {"child1": record}
    par.tool_dispatch_parallel(repo, tasks=["a"], _parent_agent=parent)

    assert captured["verify_cmd"] == "pytest tests/only_this.py -x"


def test_isolated_child_without_parent_verify_cmd_detects_its_own(repo):
    """The common case: a one-shot task-mode parent never sets verify_cmd at
    all (see core.py's constructor comment), so "inherit only" would leave
    every isolated child exactly as unverified as today. The child must still
    get a real oracle, detected against ITS OWN worktree."""
    (repo / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    _run(repo, "git", "add", "-A")
    _run(repo, "git", "commit", "-m", "add a pytest marker")

    parent = DummyParent(repo)          # verify_cmd defaults to None
    captured = {}

    def record(agent):
        captured["verify_cmd"] = agent.verify_cmd
        return "ok"

    FakeAgent.behaviour = {"child1": record}
    par.tool_dispatch_parallel(repo, tasks=["a"], _parent_agent=parent)

    cmd = captured["verify_cmd"]
    assert cmd is not None, "isolated child got no verify_cmd at all"
    assert "pytest" in " ".join(str(part) for part in cmd)


def test_isolated_child_verify_failure_is_reported_as_error_not_ok(repo):
    """The severity of the defect: a child whose diff fails verification must
    be reported as an error, never as ok (which the GUI shows as SUCCESS).

    FakeAgent's own run_task is a full stand-in, so this hook plays the part
    of the real Agent's own pre-done gate (loop.py's _run_verify_gate) -
    running the SAME verify_cmd this fix threads to the child and setting
    last_run_ok from its real exit code, exactly as loop.py does on exhausted
    retries. Before this fix verify_cmd was always None here, so a child could
    never even attempt this and always fell back to reporting ok.

    The precondition is asserted OUTSIDE the hook, on the main thread, after
    dispatch returns - never by raising inside the hook itself. A raise there
    runs on the pool worker, where dispatch_parallel's own exception handler
    (its `except Exception as exc: outcome.seal("error", ...)`) catches it and
    reports the identical "[error]" symptom this test is trying to prove for a
    real verification failure - so an assertion inside the hook cannot tell
    "verify_cmd reached the child and failed" from "the hook itself crashed
    for an unrelated reason", which defeats the whole point of the test.
    """
    captured = {}

    def fails_its_own_verification(agent):
        captured["verify_cmd"] = agent.verify_cmd
        if agent.verify_cmd is not None:
            from localm.plugins.coder import verify as _verify
            code, _out = _verify.run_verify(agent.verify_cmd, agent.cwd)
            agent.last_run_ok = (code == 0)
        return "child claims done"

    FakeAgent.behaviour = {"child1": fails_its_own_verification}
    parent = DummyParent(
        repo, verify_cmd=[sys.executable, "-c", "import sys; sys.exit(1)"])
    res = par.tool_dispatch_parallel(repo, tasks=["a"], _parent_agent=parent)

    assert captured.get("verify_cmd") is not None, "no verify_cmd reached the child"
    assert "[error]" in res.output, res.output
    assert "[ok]" not in res.output, res.output
    assert res.ok is False


# --------------------------------------------------------------------------
# THE RACE: an abandoned worker must never overwrite the parent's verdict
#
# A child that outlives the batch deadline is ABANDONED, not stopped, so its
# thread is still alive and still holding the shared _ChildOutcome while the
# parent reports on it. Before the seal, that worker's `outcome.status = "ok"`
# landed on top of the parent's "timeout" and the model was handed a delegated
# job that read [ok].
#
# These tests DRIVE that interleaving instead of racing for it. The window is
# genuinely microseconds wide in production, so a test that merely hangs a child
# and hopes would pass on the broken code most runs - which is the same thing as
# not testing it. Nothing about the mechanism is faked: the real parent, the real
# pool, the real _run_one_child and the real _ChildOutcome all run. Only the
# SCHEDULING is pinned, by releasing the hung child at the exact point between
# the parent stamping its verdict (the `_FuturesTimeout` branch) and reading it
# back (the commit/teardown/report pass), then waiting for the write to land.
# --------------------------------------------------------------------------

def _drive_late_write(monkeypatch, release: threading.Event,
                      wrote: threading.Event, late_child: str = "child1"):
    """Pin the interleaving that produces the late write.

    Two seams, both observation-only:

    * ``_run_one_child`` is wrapped so the test can tell when the REAL function
      has returned, i.e. when its write to the shared outcome has landed. The
      real function is what runs; the wrapper only signals afterwards.
    * ``ThreadPoolExecutor.shutdown`` is the parent's first unconditional step
      after the wait loop has stamped every verdict and before it reads any of
      them back. Releasing the hung child there, and blocking until its write is
      done, puts the worker's write squarely inside the window under test.
    """
    real_run_one_child = par._run_one_child

    def observed(parent, spec, child_cwd, branch, max_turns, outcome):
        try:
            real_run_one_child(parent, spec, child_cwd, branch, max_turns, outcome)
        finally:
            if spec["name"] == late_child:
                wrote.set()

    monkeypatch.setattr(par, "_run_one_child", observed)

    real_pool_cls = par.ThreadPoolExecutor

    class _PinnedPool(real_pool_cls):
        def shutdown(self, *a, **kw):
            # The parent has stamped its verdicts and has not yet read one back.
            release.set()
            # Do not proceed until the abandoned worker's write has actually
            # happened, or "the race did not occur this run" could pass for green.
            assert wrote.wait(timeout=30), "the abandoned child never finished"
            return super().shutdown(*a, **kw)

    monkeypatch.setattr(par, "ThreadPoolExecutor", _PinnedPool)


def test_an_abandoned_child_cannot_report_ok_after_the_parent_gave_up(
        repo, monkeypatch):
    """The invariant: once the parent has reported a child as timed out, no later
    write from that child can flip its observed status to success."""
    release, wrote = threading.Event(), threading.Event()

    def hang(agent):
        # Bounded, so a regression cannot wedge the suite.
        assert release.wait(timeout=30), "test driver never released the child"
        (agent.cwd / "shared.txt").write_text("late work\n", encoding="utf-8")
        return "finished, but far too late"

    FakeAgent.behaviour = {"child1": hang}
    _drive_late_write(monkeypatch, release, wrote)
    parent = DummyParent(repo)

    res = par.tool_dispatch_parallel(
        repo, tasks=[{"name": "child1", "task": "hang"},
                     {"name": "child2", "task": "fine"}],
        timeout_s=1, _parent_agent=parent,
    )

    # The whole point. On the pre-seal code this reads "=== child1 [ok] ===".
    assert "=== child1 [timeout] ===" in res.output, (
        "an abandoned child's late write overwrote the parent's verdict:\n"
        + res.output)
    assert "=== child1 [ok] ===" not in res.output
    # ...and every other consumer of that status agrees.
    assert "1/2 child(ren) finished" in res.summary, res.summary
    # The late result is REFUSED, not swallowed: we say the work exists.
    assert "too late to be used" in res.output, res.output


def test_the_late_result_is_reported_rather_than_silently_dropped(
        repo, monkeypatch):
    """Refusing the write must not hide that the child finished, or where its
    files went (AGENTS.md rule 5)."""
    release, wrote = threading.Event(), threading.Event()

    def hang(agent):
        assert release.wait(timeout=30)
        return "done at last"

    FakeAgent.behaviour = {"child1": hang}
    _drive_late_write(monkeypatch, release, wrote)

    res = par.tool_dispatch_parallel(
        repo, tasks=[{"name": "child1", "task": "hang"},
                     {"name": "child2", "task": "fine"}],
        timeout_s=1, _parent_agent=DummyParent(repo),
    )
    assert "NOTE:" in res.output
    assert "AFTER the batch was given up on" in res.output
    # It names the verdict the child tried to report, so nothing is lost.
    assert "reporting 'ok'" in res.output, res.output
    # The worktree it left behind is still pointed at.
    assert "left in place" in res.output


def test_a_late_write_cannot_flip_the_tools_ok_flag_or_the_change_set(
        repo, monkeypatch):
    """Not just the rendered text: the machine-readable consumers too.

    ``ToolResult.ok`` gates the parent's own last_run_ok and the --ci exit code,
    and the DelegatedChangeSet status is what /diff shows. A single abandoned
    child flipping to ok would make all three claim a success.
    """
    from localm.plugins.coder import delegated as _delegated

    release, wrote = threading.Event(), threading.Event()

    def hang(agent):
        assert release.wait(timeout=30)
        return "too late"

    FakeAgent.behaviour = {"child1": hang}
    _drive_late_write(monkeypatch, release, wrote)
    recorded = []
    monkeypatch.setattr(_delegated, "record",
                        lambda _p, cs: recorded.append(cs))

    res = par.tool_dispatch_parallel(
        repo, tasks=[{"name": "child1", "task": "hang"}],
        timeout_s=1, _parent_agent=DummyParent(repo),
    )
    # The ONLY child was abandoned, so the dispatch did not succeed.
    assert res.ok is False, "a late write flipped the tool's ok flag"
    assert [cs.status for cs in recorded] == ["timeout"], (
        f"the /diff change-set records {[cs.status for cs in recorded]}")


def test_a_child_that_finishes_in_time_still_reports_ok(repo, monkeypatch):
    """The fires-control's other half: the seal must not suppress ALL writes.

    Runs through the SAME pinned-pool driver, so this cannot pass merely because
    the seam was absent - only because the child published before any seal.
    """
    release, wrote = threading.Event(), threading.Event()
    # Released up front: the child never blocks, so it publishes long before the
    # deadline and the parent's wait returns normally with nothing to seal.
    release.set()

    FakeAgent.behaviour = {}
    _drive_late_write(monkeypatch, release, wrote)

    res = par.tool_dispatch_parallel(
        repo, tasks=[{"name": "child1", "task": "quick"}],
        timeout_s=30, _parent_agent=DummyParent(repo),
    )
    assert "=== child1 [ok] ===" in res.output, res.output
    assert res.ok is True
    assert "1/1 child(ren) finished" in res.summary
    # No refusal happened, so there is nothing to warn about.
    assert "too late to be used" not in res.output


def test_a_failed_child_that_finishes_in_time_still_reports_error(repo):
    """The seal must not launder a genuine failure into a success either."""
    def fail(agent):
        agent.last_run_ok = False
        return "hit max_turns"

    FakeAgent.behaviour = {"child1": fail}
    res = par.tool_dispatch_parallel(
        repo, tasks=[{"name": "child1", "task": "doomed"}],
        timeout_s=30, _parent_agent=DummyParent(repo),
    )
    assert "=== child1 [error] ===" in res.output, res.output
    assert res.ok is False
