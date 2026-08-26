# SPDX-License-Identifier: AGPL-3.0-or-later
"""spawn_agent: serialised like the destructive tool it is, and confined like its
parent.

Two independent properties:

1. The spawn_agent ToolDef must carry ``destructive=True``. Without it
   ``_execute_tools`` batches consecutive spawn_agent calls into a
   ThreadPoolExecutor, so two spawns in ONE model turn run children CONCURRENTLY
   in the SAME cwd, breaking the invariant ``_execute_tools``' own docstring
   states ("destructive calls are always run alone, in order, to avoid unintended
   interactions"). The batch also carries a 120s deadline whose pool is abandoned
   with ``shutdown(wait=False)``, so a timed-out child keeps writing to the tree
   while ``_absorb_child_state`` mutates the parent's state.

2. The child Agent must inherit the parent's ``scope`` as well as its auto_approve
   / dry_run / always_confirm / confirm_handler / mode / restricted /
   disabled_tools. ``scope`` and ``restricted`` are independent request fields and
   spawn_agent is only disabled for a RESTRICTED session, so an owner working
   under ``--scope`` would otherwise spawn a child with no path confinement.

The concurrency tests drive the REAL ``_execute_tools`` and detect overlap with a
``threading.Barrier``: if the two calls run in parallel both reach the barrier and
it releases; if they run serially the first waits out its timeout and breaks it.
``test_two_read_file_calls_DO_run_concurrently`` is the control that proves the
detector can actually see concurrency, so a passing serialisation test means
something.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

from localm.plugins.coder.agent import Agent
from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.tools import TOOL_REGISTRY, ToolResult, tool_spawn_agent


class _StubBackend:
    model_id = "stub-model"
    native_tools = False

    def set_tools(self, defs):
        pass


def _call(name: str, **args) -> ToolCall:
    return ToolCall(name=name, args=args, raw="", start=0, end=0)


def _overlap_count(agent: Agent, calls: list[ToolCall], timeout: float = 0.75) -> int:
    """Run *calls* through the real _execute_tools and return how many of them were
    inside _execute_tool at the same moment. 0 == they were serialised."""
    barrier = threading.Barrier(len(calls))
    overlapped: list[str] = []

    def _fake_execute_tool(call, interactive=False):
        try:
            barrier.wait(timeout=timeout)
            overlapped.append(call.name)
        except threading.BrokenBarrierError:
            pass          # serialised: nobody else ever showed up
        return ToolResult.success("ok")

    with patch.object(agent, "_execute_tool", _fake_execute_tool):
        blocks = agent._execute_tools(calls, interactive=False)
    assert len(blocks) == len(calls)      # every call still reports a result
    return len(overlapped)


class TestSpawnAgentIsSerialised:
    def test_spawn_agent_is_marked_destructive(self):
        assert TOOL_REGISTRY["spawn_agent"].destructive is True

    def test_two_spawn_agent_calls_land_in_separate_segments(self, tmp_path):
        """The segmentation itself: two spawns must not share one parallel batch."""
        agent = Agent(_StubBackend(), cwd=tmp_path)
        calls = [_call("spawn_agent", task="a"), _call("spawn_agent", task="b")]
        assert _overlap_count(agent, calls) == 0

    def test_spawn_agent_between_reads_still_runs_alone(self, tmp_path):
        """Ordering is preserved and the spawn is not swept into either read batch."""
        agent = Agent(_StubBackend(), cwd=tmp_path)
        calls = [_call("read_file", path="a"), _call("spawn_agent", task="x"),
                 _call("read_file", path="b")]
        # The spawn sits between two singleton read segments, so nothing can pair up.
        assert _overlap_count(agent, calls) == 0

    def test_two_read_file_calls_DO_run_concurrently(self, tmp_path):
        """Control: the detector CAN see concurrency, so the assertions above are
        evidence and not an artefact of a barrier that never releases."""
        agent = Agent(_StubBackend(), cwd=tmp_path)
        calls = [_call("read_file", path="a"), _call("read_file", path="b")]
        assert _overlap_count(agent, calls) == 2


def _spawn_and_capture_child(tmp_path, **parent_kwargs) -> Agent:
    """Spawn a real child via tool_spawn_agent, short-circuiting run_task so no LLM
    call is needed, and return the constructed child for direct inspection."""
    parent = Agent(_StubBackend(), cwd=tmp_path, **parent_kwargs)
    captured = {}

    def _fake_run_task(self, task):
        captured["child"] = self
        return "child done"

    with patch.object(Agent, "run_task", _fake_run_task):
        result = tool_spawn_agent(tmp_path, "do work", _parent_agent=parent)
    assert result.ok, result.output
    return captured["child"]


class TestSpawnAgentInheritsScope:
    def test_child_inherits_the_parent_scope(self, tmp_path):
        child = _spawn_and_capture_child(tmp_path, scope="src/**")
        assert child.scope == "src/**"

    def test_child_rejects_a_path_outside_the_parent_scope(self, tmp_path):
        """End to end through the child's own dispatch path, not just the kwarg:
        a scoped parent must not be able to delegate its way out of the scope."""
        child = _spawn_and_capture_child(tmp_path, scope="src/**")
        res = child._execute_tool(
            _call("write_file", path="secrets.txt", content="x"), interactive=False)
        assert not res.ok
        assert "outside the active scope" in res.output
        assert not (tmp_path / "secrets.txt").exists()

    def test_child_still_allows_a_path_inside_the_scope(self, tmp_path):
        """Sanity: inheriting the scope must confine the child, not paralyse it."""
        (tmp_path / "src").mkdir()
        child = _spawn_and_capture_child(tmp_path, scope="src/**")
        res = child._execute_tool(
            _call("write_file", path="src/new.py", content="x = 1\n"),
            interactive=False)
        assert res.ok, res.output
        assert (tmp_path / "src" / "new.py").exists()

    def test_unscoped_parent_still_spawns_an_unscoped_child(self, tmp_path):
        child = _spawn_and_capture_child(tmp_path)
        assert child.scope is None
