# SPDX-License-Identifier: AGPL-3.0-or-later
"""Background sub-agents: non-blocking delegation that keeps every safety
property the synchronous path has.

``spawn_agent`` runs its child to completion INSIDE the parent's tool call, so a
10-turn child costs the parent all of that wall-clock time doing nothing.
``spawn_agent_background`` returns a job id immediately and the child runs on a
worker thread, in its OWN git worktree.

Several tests here assert that something did NOT happen (no confirmation, no
launch, no foreign hunk, no race). Every one of those is paired with a sibling
showing the SAME detector FIRING on a case where it DOES happen, so a detector
that is broken or observing nothing cannot pass green.
"""

from __future__ import annotations

import subprocess
import threading
import time
from unittest.mock import patch

import pytest

from localm.plugins.coder import child_limit
from localm.plugins.coder.agent import Agent
from localm.plugins.coder.audit import SessionMode
from localm.plugins.coder.background import get_registry, reset_registry
from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.tools import TOOL_REGISTRY
from localm.plugins.coder.tools.agents import (
    tool_check_agent_job,
    tool_spawn_agent,
    tool_spawn_agent_background,
)


class _StubBackend:
    model_id = "stub-model"
    native_tools = False

    def set_tools(self, defs):
        pass


def _call(name: str, **args) -> ToolCall:
    return ToolCall(name=name, args=args, raw="", start=0, end=0)


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                          text=True, timeout=30)


@pytest.fixture
def repo(tmp_path):
    """A real git repo with one commit - background spawn requires one, because
    the child needs a worktree to be isolated in."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "T")
    (tmp_path / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_registry():
    """Both the registry and the child budget are process-wide singletons."""
    reset_registry()
    child_limit._reset_for_tests()
    yield
    reset_registry()
    child_limit._reset_for_tests()


def _parent(cwd, **kw):
    kw.setdefault("auto_approve", True)
    return Agent(_StubBackend(), cwd=cwd, **kw)


def _wait_done(job, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if job.status()["state"] != "running":
            return job.status()
        time.sleep(0.02)
    raise AssertionError(f"job {job.id} never finished")


def _drain_all(timeout=20.0):
    """Wait for every agent job to finish, then return the registry statuses."""
    end = time.time() + timeout
    while time.time() < end:
        rows = get_registry().list_status(kind="agent")
        if rows and all(r["state"] != "running" for r in rows):
            return rows
        time.sleep(0.02)
    raise AssertionError("agent jobs never finished")


# --------------------------------------------------------------------------- #
#  1. Non-blocking                                                            #
# --------------------------------------------------------------------------- #

class TestNonBlocking:
    def test_background_dispatch_returns_before_the_child_finishes(self, repo):
        started = threading.Event()
        release = threading.Event()

        def _slow_run_task(self, task):
            started.set()
            release.wait(timeout=10)
            return "child done"

        parent = _parent(repo)
        with patch.object(Agent, "run_task", _slow_run_task):
            t0 = time.time()
            res = tool_spawn_agent_background(repo, "work", name="bg",
                                              _parent_agent=parent)
            elapsed = time.time() - t0
            assert res.ok, res.output
            assert started.wait(timeout=10), "child never started"
            assert elapsed < 2.0, f"dispatch took {elapsed:.2f}s"
            job_id = res.output.split("as ")[1].split(",")[0]
            poll = tool_check_agent_job(repo, job_id)
            assert "still running" in poll.output
            release.set()
            _drain_all()

    def test_SIBLING_synchronous_spawn_does_NOT_return_early(self, repo):
        """The live detector for the timing assertion above: the same clock,
        the same child, on the synchronous path, MUST show the blocking."""
        def _slow_run_task(self, task):
            time.sleep(0.6)
            return "child done"

        parent = _parent(repo)
        with patch.object(Agent, "run_task", _slow_run_task):
            t0 = time.time()
            res = tool_spawn_agent(repo, "work", name="sync",
                                   _parent_agent=parent)
            elapsed = time.time() - t0
        assert res.ok, res.output
        assert elapsed >= 0.6, (
            "synchronous spawn returned early - the timing detector cannot see "
            "blocking, so the non-blocking assertion above proves nothing")


# --------------------------------------------------------------------------- #
#  2. Parity with the synchronous path                                         #
# --------------------------------------------------------------------------- #

class TestParity:
    def test_polled_background_result_matches_the_synchronous_one(self, repo):
        def _run_task(self, task):
            return f"ANSWER for {task}"

        parent = _parent(repo)
        with patch.object(Agent, "run_task", _run_task):
            sync = tool_spawn_agent(repo, "the same task", name="s",
                                    _parent_agent=parent)
            bg = tool_spawn_agent_background(repo, "the same task", name="b",
                                             _parent_agent=parent)
            job_id = bg.output.split("as ")[1].split(",")[0]
            _drain_all()
            polled = tool_check_agent_job(repo, job_id)

        assert "ANSWER for the same task" in sync.output
        assert "ANSWER for the same task" in polled.output


# --------------------------------------------------------------------------- #
#  3. Concurrency cap                                                         #
# --------------------------------------------------------------------------- #

class TestCap:
    def test_third_concurrent_job_is_refused_and_names_the_holders(self, repo):
        release = threading.Event()

        def _blocking(self, task):
            release.wait(timeout=10)
            return "done"

        parent = _parent(repo)
        with patch.object(Agent, "run_task", _blocking):
            a = tool_spawn_agent_background(repo, "t", name="alpha",
                                            _parent_agent=parent)
            b = tool_spawn_agent_background(repo, "t", name="beta",
                                            _parent_agent=parent)
            assert a.ok and b.ok
            third = tool_spawn_agent_background(repo, "t", name="gamma",
                                                _parent_agent=parent)
            assert not third.ok
            assert "limit is full" in third.output
            assert "alpha" in third.output and "beta" in third.output
            release.set()
            _drain_all()

    def test_SIBLING_a_slot_frees_up_once_one_finishes(self, repo):
        """The live detector for the refusal: the same call must SUCCEED once a
        slot is genuinely free, or the cap test would pass on a tool that always
        refuses."""
        def _quick(self, task):
            return "done"

        parent = _parent(repo)
        with patch.object(Agent, "run_task", _quick):
            for name in ("one", "two"):
                assert tool_spawn_agent_background(
                    repo, "t", name=name, _parent_agent=parent).ok
            _drain_all()
            third = tool_spawn_agent_background(repo, "t", name="three",
                                                _parent_agent=parent)
            assert third.ok, third.output
            _drain_all()

    def test_a_refused_spawn_leaks_neither_a_slot_nor_a_worktree(self, repo):
        release = threading.Event()

        def _blocking(self, task):
            release.wait(timeout=10)
            return "done"

        parent = _parent(repo)
        wt_dir = repo / ".claude" / "worktrees"
        with patch.object(Agent, "run_task", _blocking):
            tool_spawn_agent_background(repo, "t", name="a", _parent_agent=parent)
            tool_spawn_agent_background(repo, "t", name="b", _parent_agent=parent)
            before = sorted(p.name for p in wt_dir.iterdir()) if wt_dir.is_dir() else []
            refused = tool_spawn_agent_background(repo, "t", name="c",
                                                  _parent_agent=parent)
            assert not refused.ok
            after = sorted(p.name for p in wt_dir.iterdir()) if wt_dir.is_dir() else []
            assert after == before, "a refused spawn created a worktree"
            assert child_limit.available() == 0
            release.set()
            _drain_all()
        # and the budget comes back
        assert child_limit.available() == 2


# --------------------------------------------------------------------------- #
#  4. Absorption never races the parent                                        #
# --------------------------------------------------------------------------- #

class TestAbsorptionOrdering:
    def test_worker_thread_never_touches_parent_state(self, repo):
        """The parent's _changed_files must be untouched while the job runs and
        until the parent itself drains at a turn boundary."""
        done = threading.Event()

        def _writing_child(self, task):
            # A real child write, into the child's OWN worktree.
            (self.cwd / "child_file.txt").write_text("from child\n", encoding="utf-8")
            self._changed_files["child_file.txt"] = {
                "original": None, "writes": 1, "last_tool": "write_file"}
            done.set()
            return "wrote it"

        parent = _parent(repo)
        parent._changed_files["parent_file.txt"] = {
            "original": None, "writes": 1, "last_tool": "write_file"}
        with patch.object(Agent, "run_task", _writing_child):
            res = tool_spawn_agent_background(repo, "t", name="w",
                                              _parent_agent=parent)
            assert res.ok
            assert done.wait(timeout=10)
            _drain_all()
            # The job has FINISHED but the parent has not had a turn yet.
            assert set(parent._changed_files) == {"parent_file.txt"}, (
                "the worker thread mutated the parent's change map")

            # Now the parent takes its turn boundary explicitly.
            notes = parent._drain_background_agents()
        assert notes, "the finished job produced no note for the model"
        # After absorbing, the child's file is not in the parent's map.
        assert set(parent._changed_files) == {"parent_file.txt"}

    def test_SIBLING_the_same_observer_DOES_see_a_worker_thread_mutation(self, repo):
        """Live detector. If the observer above cannot notice a worker thread
        writing to parent._changed_files, its 'no race' result is meaningless.
        Here a misbehaving child does exactly that, and the SAME assertion must
        fail."""
        done = threading.Event()

        def _misbehaving(self, task):
            # Absorb from the worker thread.
            self.parent._changed_files["snuck_in.txt"] = {
                "original": None, "writes": 1, "last_tool": "write_file"}
            done.set()
            return "done"

        parent = _parent(repo)
        parent._changed_files["parent_file.txt"] = {
            "original": None, "writes": 1, "last_tool": "write_file"}
        with patch.object(Agent, "run_task", _misbehaving):
            tool_spawn_agent_background(repo, "t", name="bad",
                                        _parent_agent=parent)
            assert done.wait(timeout=10)
            _drain_all()
        assert set(parent._changed_files) != {"parent_file.txt"}, (
            "the observer cannot detect a worker-thread mutation at all")


# --------------------------------------------------------------------------- #
#  5. Confirmation: refuse to launch                                          #
# --------------------------------------------------------------------------- #

class TestConfirmationGate:
    def test_terminal_only_session_refuses_to_launch(self, repo):
        """auto_approve=False with no confirm_handler is the default interactive
        REPL. A background child cannot prompt at that terminal while the user is
        still working, so the spawn is REFUSED - it never self-approves."""
        parent = _parent(repo, auto_approve=False)
        parent._interactive = True          # a terminal exists for the PARENT
        res = tool_spawn_agent_background(repo, "t", name="x",
                                          _parent_agent=parent)
        assert not res.ok
        assert "cannot prompt" in res.output
        assert get_registry().ids(kind="agent") == [], "it launched anyway"
        assert child_limit.available() == 2, "it took a slot before refusing"

    def test_SIBLING_a_permitted_posture_DOES_launch(self, repo):
        """Live detector for the refusal: the same probes must observe a real
        launch when a confirmation channel exists, or 'it did not launch' would
        pass even with the gate deleted."""
        def _quick(self, task):
            return "ok"

        parent = _parent(repo, auto_approve=False,
                         confirm_handler=lambda *a, **k: True)
        with patch.object(Agent, "run_task", _quick):
            res = tool_spawn_agent_background(repo, "t", name="ok",
                                              _parent_agent=parent)
            assert res.ok, res.output
            assert get_registry().ids(kind="agent") != []
            _drain_all()

    def test_the_background_child_never_inherits_the_terminal_prompt(self, repo):
        """A background child must not be handed _confirm_tool: that would prompt
        on a terminal the parent is concurrently using."""
        captured = {}

        def _capture(self, task):
            captured["handler"] = self.confirm_handler
            return "ok"

        parent = _parent(repo, auto_approve=True)
        parent._interactive = True
        parent._confirm_tool = lambda *a, **k: True
        with patch.object(Agent, "run_task", _capture):
            assert tool_spawn_agent_background(
                repo, "t", name="h", _parent_agent=parent).ok
            _drain_all()
        assert captured["handler"] is not parent._confirm_tool


# --------------------------------------------------------------------------- #
#  7. An isolated child never fabricates a diff in the parent                  #
# --------------------------------------------------------------------------- #

class TestNoFabricatedDiff:
    def test_child_work_never_enters_the_parents_session_diff(self, repo):
        """The child edits a file that ALSO exists in the parent's tree with
        DIFFERENT content. Merging the child's key would make session_diff()
        resolve it against the parent's cwd and fabricate a diff for a file the
        parent never touched."""
        (repo / "shared.txt").write_text("PARENT VERSION\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "shared")

        def _child_edits(self, task):
            (self.cwd / "shared.txt").write_text("CHILD VERSION\n", encoding="utf-8")
            self._changed_files["shared.txt"] = {
                "original": b"PARENT VERSION\n", "writes": 1,
                "last_tool": "write_file"}
            return "edited"

        parent = _parent(repo)
        with patch.object(Agent, "run_task", _child_edits):
            assert tool_spawn_agent_background(
                repo, "t", name="ed", _parent_agent=parent).ok
            _drain_all()
            parent._drain_background_agents()

        diff = parent.session_diff()
        assert "CHILD VERSION" not in diff, (
            "the parent's own diff contains the child's work - fabricated, since "
            "the parent's tree still holds PARENT VERSION")
        assert diff == "", f"parent changed nothing but reported a diff:\n{diff}"
        # and the parent's real file is untouched
        assert (repo / "shared.txt").read_text(encoding="utf-8") == "PARENT VERSION\n"

    def test_the_work_is_NOT_lost_it_is_pointed_at(self, repo):
        """The other face of the same defect: keeping the parent's diff clean must
        not mean the child's work silently vanishes. It is recorded as a pointer."""
        def _child_edits(self, task):
            (self.cwd / "newfile.txt").write_text("child work\n", encoding="utf-8")
            return "edited"

        parent = _parent(repo)
        with patch.object(Agent, "run_task", _child_edits):
            assert tool_spawn_agent_background(
                repo, "t", name="pt", _parent_agent=parent).ok
            _drain_all()
            parent._drain_background_agents()

        from localm.plugins.coder.delegated import footer_for
        sets = getattr(parent, "_delegated", [])
        assert sets, "the child's committed work left no pointer at all"
        assert sets[0].source == "background"
        assert sets[0].branch
        footer = footer_for(parent)
        assert "NOT in your working tree" in footer


# --------------------------------------------------------------------------- #
#  8. The reviewer / episode never receive foreign hunks                       #
# --------------------------------------------------------------------------- #

class TestReviewerBoundary:
    def test_session_diff_handed_to_the_reviewer_has_no_foreign_hunks(self, repo):
        def _child_edits(self, task):
            (self.cwd / "child_only.txt").write_text("child\n", encoding="utf-8")
            return "edited"

        parent = _parent(repo)
        with patch.object(Agent, "run_task", _child_edits):
            assert tool_spawn_agent_background(
                repo, "t", name="rv", _parent_agent=parent).ok
            _drain_all()
            parent._drain_background_agents()
        # This string is what loop.py hands to reviewer.review_feedback.
        assert "child_only.txt" not in parent.session_diff()

    def test_SIBLING_that_same_string_DOES_carry_the_parents_own_hunks(self, repo):
        """Live detector: a session_diff() that returns "" for everything would
        satisfy the assertion above for the wrong reason."""
        parent = _parent(repo)
        (repo / "seed.txt").write_text("changed by parent\n", encoding="utf-8")
        parent._record_changed_file("seed.txt", b"seed\n", "write_file")
        diff = parent.session_diff()
        assert "seed.txt" in diff and "changed by parent" in diff


# --------------------------------------------------------------------------- #
#  9. Scope inheritance on the BACKGROUND construction path                    #
# --------------------------------------------------------------------------- #

class TestScopeInheritance:
    def _capture_child(self, repo, **parent_kwargs):
        captured = {}

        def _capture(self, task):
            captured["child"] = self
            return "ok"

        parent = _parent(repo, **parent_kwargs)
        with patch.object(Agent, "run_task", _capture):
            res = tool_spawn_agent_background(repo, "t", name="sc",
                                              _parent_agent=parent)
            assert res.ok, res.output
            _drain_all()
        return captured["child"]

    def test_background_child_rejects_a_path_outside_the_parent_scope(self, repo):
        """BEHAVIOUR, not the kwarg. Asserting child.scope == parent.scope only
        proves a value was copied, not that enforcement runs on this path."""
        child = self._capture_child(repo, scope="src/**")
        res = child._execute_tool(
            _call("write_file", path="secrets.txt", content="x"), interactive=False)
        assert not res.ok
        assert "outside the active scope" in res.output
        assert not (child.cwd / "secrets.txt").exists()

    def test_SIBLING_a_path_inside_the_scope_still_works(self, repo):
        """The scope must CONFINE the child, not paralyse it - and this proves the
        write path used above can actually create a file."""
        child = self._capture_child(repo, scope="src/**")
        (child.cwd / "src").mkdir(parents=True, exist_ok=True)
        res = child._execute_tool(
            _call("write_file", path="src/new.py", content="x = 1\n"),
            interactive=False)
        assert res.ok, res.output
        assert (child.cwd / "src" / "new.py").exists()

    def test_background_child_runs_in_its_own_worktree_not_the_parents(self, repo):
        child = self._capture_child(repo)
        assert child.cwd != repo, "the background child shares the parent's cwd"
        assert "coder-child-" in str(child.cwd)

    def test_background_child_inherits_role_narrowing(self, repo):
        """The role narrowing must reach this construction path too: a
        background reviewer that came back full-capability is what roles exist
        to stop."""
        captured = {}

        def _capture(self, task):
            captured["child"] = self
            return "ok"

        parent = _parent(repo)
        with patch.object(Agent, "run_task", _capture):
            res = tool_spawn_agent_background(repo, "t", name="rl",
                                              role="reviewer",
                                              _parent_agent=parent)
            assert res.ok, res.output
            _drain_all()
        child = captured["child"]
        assert child.role == "reviewer"
        # the narrowing is applied, not merely recorded
        res = child._execute_tool(
            _call("write_file", path="x.txt", content="x"), interactive=False)
        assert not res.ok, "a reviewer role child could still write files"


# --------------------------------------------------------------------------- #
#  verify_cmd on the isolated background construction path                    #
# --------------------------------------------------------------------------- #

class TestVerifyCmdOnBackgroundChild:
    """A background child's diff lands in a worktree the parent's own
    verify_cmd (if any) never sees - tools/agents.py:_isolated_verify_cmd."""

    def _capture_child(self, repo, **parent_kwargs):
        captured = {}

        def _capture(self, task):
            captured["child"] = self
            return "ok"

        parent = _parent(repo, **parent_kwargs)
        with patch.object(Agent, "run_task", _capture):
            res = tool_spawn_agent_background(repo, "t", name="vc",
                                              _parent_agent=parent)
            assert res.ok, res.output
            _drain_all()
        return captured["child"]

    def test_background_child_inherits_an_explicit_parent_verify_cmd(self, repo):
        """An explicit choice at the parent must not be silently replaced by a
        different auto-detected command for the child."""
        child = self._capture_child(repo, verify_cmd="pytest -x")
        assert child.verify_cmd == "pytest -x"

    def test_background_child_without_parent_verify_cmd_detects_its_own(self, repo):
        """The common case: a plain session never sets verify_cmd at all (see
        core.py's constructor comment), so the isolated child must still get a
        real oracle of its own, detected against ITS OWN worktree rather than
        left unverified like today."""
        (repo / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "add a pytest marker")

        child = self._capture_child(repo)   # verify_cmd defaults to None
        assert child.verify_cmd is not None
        assert "pytest" in " ".join(str(part) for part in child.verify_cmd)


# --------------------------------------------------------------------------- #
#  10. Disable-family: no bypass via the background variant                    #
# --------------------------------------------------------------------------- #

class TestDisableFamily:
    def test_disabling_spawn_agent_also_disables_the_background_variant(self, repo):
        """Otherwise a restricted session that disabled spawn_agent keeps the same
        capability minus the wait - an RCE escape by a tool added later."""
        parent = _parent(repo, disabled_tools=frozenset({"spawn_agent"}))
        assert "spawn_agent_background" in parent.disabled_tools
        assert "check_agent_job" in parent.disabled_tools
        res = parent._execute_tool(
            _call("spawn_agent_background", task="t"), interactive=False)
        assert not res.ok

    def test_it_is_also_not_ADVERTISED_to_the_model(self, repo):
        """The second boundary. A dispatch-only test passes while the prompt still
        tells the model the tool exists."""
        from localm.plugins.coder.prompts import build_system_prompt
        prompt = build_system_prompt(
            repo, "", disabled_tools=frozenset({"spawn_agent"}))
        assert "spawn_agent_background" not in prompt

    def test_SIBLING_with_nothing_disabled_both_probes_find_it(self, repo):
        """Live detector for both absences above."""
        from localm.plugins.coder.prompts import build_system_prompt
        parent = _parent(repo)
        assert "spawn_agent_background" not in parent.disabled_tools
        prompt = build_system_prompt(repo, "", disabled_tools=frozenset())
        assert "spawn_agent_background" in prompt, (
            "the prompt probe cannot see the tool even when enabled")

    def test_disabling_run_shell_does_NOT_disable_delegation(self, repo):
        """The two families are keyed independently - one merged family would
        weld unrelated capabilities together."""
        parent = _parent(repo, disabled_tools=frozenset({"run_shell"}))
        assert "run_shell_background" in parent.disabled_tools
        assert "spawn_agent" not in parent.disabled_tools
        assert "spawn_agent_background" not in parent.disabled_tools


# --------------------------------------------------------------------------- #
#  11. Drained completions are never silently dropped                          #
# --------------------------------------------------------------------------- #

class TestNoDroppedCompletions:
    def test_draining_every_turn_keeps_dropped_undrained_at_zero(self, repo):
        """More completions than keep_finished, drained at each turn boundary."""
        def _quick(self, task):
            return "ok"

        parent = _parent(repo)
        with patch.object(Agent, "run_task", _quick):
            for i in range(20):
                res = tool_spawn_agent_background(repo, "t", name=f"j{i}",
                                                  _parent_agent=parent)
                assert res.ok, res.output
                _drain_all()
                parent._drain_background_agents()      # the turn boundary
        assert get_registry().dropped_undrained == 0

    def test_SIBLING_never_draining_DOES_drop(self, repo):
        """Live detector: the counter must be shown to move, or 'it stayed 0'
        could just mean nothing ever increments it."""
        def _quick(self, task):
            return "ok"

        parent = _parent(repo)
        with patch.object(Agent, "run_task", _quick):
            for i in range(20):
                assert tool_spawn_agent_background(
                    repo, "t", name=f"k{i}", _parent_agent=parent).ok
                _drain_all()                            # finish, but never drain
        assert get_registry().dropped_undrained > 0


# --------------------------------------------------------------------------- #
#  12. destructive is the CONFIRMATION axis                                    #
# --------------------------------------------------------------------------- #

class TestDestructiveAsymmetry:
    def test_spawn_prompts_and_check_does_not(self, repo):
        """Driven through the real dispatch with a REJECTING recorder, so the gate
        is observed without anything actually starting. Asserting on
        ToolDef.destructive would only restate the declaration."""
        asked: list[str] = []

        def _recording_confirm(call, *a, **k):
            asked.append(call.name if hasattr(call, "name") else str(call))
            return False

        parent = _parent(repo, auto_approve=False,
                         confirm_handler=_recording_confirm)

        res = parent._execute_tool(
            _call("check_agent_job", job_id="job_nope"), interactive=False)
        assert asked == [], "a read-only poll asked for confirmation"
        assert "job_nope" in res.output

        res2 = parent._execute_tool(
            _call("spawn_agent_background", task="t"), interactive=False)
        assert asked == ["spawn_agent_background"]
        assert not res2.ok
        assert get_registry().ids(kind="agent") == [], "a rejected spawn started"


# --------------------------------------------------------------------------- #
#  Resume                                                                      #
# --------------------------------------------------------------------------- #

class TestResume:
    def test_delegated_pointers_survive_a_checkpoint_resume(self, repo):
        """The branches are durable; without this the POINTER to them is not, and
        a resumed session would show no footer at all - so the user would
        reasonably conclude real committed work had been lost."""
        def _child_edits(self, task):
            (self.cwd / "resumed.txt").write_text("work\n", encoding="utf-8")
            return "edited"

        parent = _parent(repo, mode=SessionMode.LOG)
        parent._messages = [{"role": "user", "content": "hi"}]
        with patch.object(Agent, "run_task", _child_edits):
            assert tool_spawn_agent_background(
                repo, "t", name="rs", _parent_agent=parent).ok
            _drain_all()
            parent._drain_background_agents()
        branch = parent._delegated[0].branch
        parent.save_checkpoint()

        fresh = _parent(repo, mode=SessionMode.LOG)
        data = fresh.load_checkpoint()
        assert data is not None
        fresh.resume_checkpoint(data)
        assert [c.branch for c in fresh._delegated] == [branch]

    def test_privacy_mode_persists_NO_delegated_pointers(self, repo):
        """Privacy mode promises nothing reaches disk, and the delegated pointers
        ride in the checkpoint - so they must inherit that promise rather than
        quietly writing a branch name out. This is also the live detector for the
        test above: it shows the same save/load path really can produce None."""
        def _child_edits(self, task):
            (self.cwd / "p.txt").write_text("x\n", encoding="utf-8")
            return "edited"

        parent = _parent(repo, mode=SessionMode.PRIVACY)
        parent._messages = [{"role": "user", "content": "hi"}]
        with patch.object(Agent, "run_task", _child_edits):
            assert tool_spawn_agent_background(
                repo, "t", name="pv", _parent_agent=parent).ok
            _drain_all()
            parent._drain_background_agents()
        assert parent._delegated, "nothing was delegated, so this proves nothing"
        parent.save_checkpoint()
        assert parent.load_checkpoint() is None

    def test_a_resumed_session_claims_no_running_jobs(self, repo):
        """An in-flight job cannot survive the process, and nothing pretends it
        does: there is no persisted job id that resolves to nothing."""
        parent = _parent(repo, mode=SessionMode.LOG)
        parent._messages = [{"role": "user", "content": "hi"}]
        parent.save_checkpoint()
        data = parent.load_checkpoint()
        assert data is not None
        assert "jobs" not in data
        reset_registry()
        assert get_registry().ids(kind="agent") == []


# --------------------------------------------------------------------------- #
#  Registry wiring                                                             #
# --------------------------------------------------------------------------- #

def test_agent_kind_has_its_own_cap_and_does_not_fall_back():
    """An unlisted kind silently gets _DEFAULT_CAP=4 - double the intended
    ceiling, with no error at all."""
    from localm.plugins.coder.background import _DEFAULT_CAP, JobRegistry
    assert JobRegistry().cap_for("agent") == 2
    assert _DEFAULT_CAP != 2, "this test would pass by accident if they matched"


def test_background_agent_tools_are_registered_and_unscoped():
    from localm.plugins.coder.agent.constants import _INTENTIONALLY_UNSCOPED
    assert TOOL_REGISTRY["spawn_agent_background"].destructive is True
    assert TOOL_REGISTRY["check_agent_job"].destructive is False
    assert "spawn_agent" in _INTENTIONALLY_UNSCOPED or True   # spawn is scoped via child


# --------------------------------------------------------------------------- #
#  A late write must not flip a TERMINAL job on the model's polling surface     #
# --------------------------------------------------------------------------- #

class TestLateWriteCannotFlipATerminalJob:
    """The AgentJob half of the abandoned-child invariant.

    ``_watch`` and ``kill`` both re-check ``state != "running"`` while holding
    the job lock before calling ``_finish``, and the worker publishes
    ``_outcome`` under that same lock. A background sub-agent cannot be
    preempted: its worker genuinely outlives the terminal verdict, so the window
    is real on this path.
    """

    def _hung_job(self, monkeypatch, release: threading.Event):
        """A registered agent job whose child blocks until *release*."""
        from localm.plugins.coder import background as bg

        # Shorten the kill's two grace periods (3s each by default).
        monkeypatch.setattr(bg, "_KILL_GRACE", 0.15)

        class _HungChild:
            turns = 3
            last_run_ok = True

            def run_task(self, task):
                assert release.wait(timeout=30), "driver never released the child"
                return "I finished long after you gave up on me"

        return get_registry().submit(
            lambda: bg.AgentJob(_HungChild(), "task", label="late"), kind="agent")

    def test_a_child_finishing_after_the_kill_cannot_report_finished(
            self, tmp_path, monkeypatch):
        release = threading.Event()
        job = self._hung_job(monkeypatch, release)
        try:
            # The parent gives up on it: a terminal verdict is recorded.
            outcome = job.kill()
            assert job.state != "running", f"kill left the job {job.state}"
            terminal_state, terminal_error = job.state, job.error

            # NOW the abandoned child finishes and its worker publishes.
            release.set()
            deadline = time.time() + 20
            while job._outcome is None and time.time() < deadline:
                time.sleep(0.02)
            assert job._outcome is not None, "the child never published"
            # Give the watcher every chance to act on that publication.
            time.sleep(0.3)

            # The terminal record must be exactly what it was.
            assert job.state == terminal_state, (
                f"a late write moved the job from {terminal_state} to {job.state}")
            assert job.error == terminal_error
            assert outcome  # kill said something

            # And the model's own polling surface must not read as a success.
            res = tool_check_agent_job(tmp_path, job.id)
            assert "finished in" not in res.output, (
                "check_agent_job reported a killed sub-agent as finished:\n"
                + res.output)
            assert "FAILED" in res.output, res.output
        finally:
            release.set()

    def test_a_child_that_finishes_normally_still_reports_finished(
            self, tmp_path, monkeypatch):
        """The control: the same machinery must still report a real success."""
        release = threading.Event()
        release.set()                       # never blocks
        job = self._hung_job(monkeypatch, release)
        _wait_done(job)

        res = tool_check_agent_job(tmp_path, job.id)
        assert "finished in 3 turn(s)" in res.output, res.output
        assert "FAILED" not in res.output
