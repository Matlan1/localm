"""
Tests for the coder QoL round: changed-files tracker + session diff,
mid-task message queue, circuit breaker, parallel execution (ordering +
batch timeout), undo stack, checkpoint/resume, stop midstream, and the
patch-mode write-tool fallthrough guard.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.coder.audit import SessionMode
from localm.plugins.coder.tools import ToolDef, ToolResult


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

class _Scripted:
    """Minimal backend: one canned response per chat() call, repeats the last."""
    model_id = "test-model"
    native_tools = False
    supports_grammar = False
    last_usage = {"total_tokens": 5}

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, **kw):
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return r

    def chat_stream(self, messages, **kw):
        yield self.chat(messages, **kw)


def _make_agent(tmp_path: Path, responses=("Done.",), **kwargs):
    from localm.plugins.coder.agent import Agent
    backend = _Scripted(responses)
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        kwargs.setdefault("auto_approve", True)
        kwargs.setdefault("self_verify", False)
        agent = Agent(backend=backend, cwd=tmp_path, **kwargs)
    return agent


def _tc(name: str, **args) -> str:
    return "<tool_call>" + json.dumps({"name": name, "args": args}) + "</tool_call>"


def _make_call(name: str, **args):
    c = MagicMock()
    c.name = name
    c.args = args
    return c


# ---------------------------------------------------------------------------
#  Changed-files tracker + session diff
# ---------------------------------------------------------------------------

class TestChangedFiles:
    def test_write_tracked_as_created(self, tmp_path):
        agent = _make_agent(tmp_path, [
            _tc("write_file", path="a.py", content="x = 1\n"), "Done."])
        agent.run_task("make a.py")
        files = agent.changed_files()
        assert len(files) == 1
        f = files[0]
        assert f["path"] == "a.py" and f["created"] and f["exists"]
        assert f["writes"] == 1 and f["last_tool"] == "write_file"

    def test_cumulative_diff_uses_first_original(self, tmp_path):
        """Two edits to a pre-existing file diff against the ORIGINAL content,
        not the intermediate state."""
        (tmp_path / "a.txt").write_text("v1\n", encoding="utf-8")
        agent = _make_agent(tmp_path, [
            _tc("write_file", path="a.txt", content="v2\n"),
            _tc("edit_file", path="a.txt", old="v2", new="v3"),
            "Done."])
        agent.run_task("rewrite a.txt twice")
        f = agent.changed_files()[0]
        assert f["writes"] == 2 and not f["created"]
        diff = agent.session_diff("a.txt")
        assert "-v1" in diff and "+v3" in diff
        assert "v2" not in diff          # the intermediate state never appears

    def test_session_diff_all_files_and_unknown_path(self, tmp_path):
        agent = _make_agent(tmp_path, [
            _tc("write_file", path="a.txt", content="A\n")
            + _tc("write_file", path="b.txt", content="B\n"),
            "Done."])
        agent.run_task("two files")
        full = agent.session_diff()
        assert "b/a.txt" in full and "b/b.txt" in full
        assert agent.session_diff("never-touched.txt") == ""

    def test_failed_write_not_tracked(self, tmp_path):
        agent = _make_agent(tmp_path, [
            _tc("edit_file", path="missing.txt", old="x", new="y"), "Done."])
        agent.run_task("edit a file that does not exist")
        assert agent.changed_files() == []

    def test_dry_run_not_tracked(self, tmp_path):
        agent = _make_agent(tmp_path, [
            _tc("write_file", path="a.py", content="x"), "Done."],
            dry_run=True)
        agent.run_task("write")
        assert agent.changed_files() == []
        assert not (tmp_path / "a.py").exists()


# ---------------------------------------------------------------------------
#  Mid-task message queue
# ---------------------------------------------------------------------------

class TestMessageQueue:
    def test_queued_message_delivered_as_steering_note(self, tmp_path):
        agent = _make_agent(tmp_path, ["Done."])
        agent.queue_message("also add logging")
        agent.run_task("do the thing")
        steering = [m for m in agent._messages
                    if m["role"] == "user"
                    and "[user steering note" in str(m["content"])]
        assert len(steering) == 1
        assert "also add logging" in steering[0]["content"]
        assert agent.queued_count() == 0

    def test_queue_drained_between_turns(self, tmp_path):
        """A message queued during turn 1 reaches the conversation before the
        turn-2 LLM call."""
        agent = _make_agent(tmp_path, [
            _tc("list_dir", path="."), "Done."])

        original_chat = agent.backend.chat
        def chat_and_queue(messages, **kw):
            if agent.backend.calls == 0:
                agent.queue_message("steer!")
            return original_chat(messages, **kw)
        agent.backend.chat = chat_and_queue

        agent.run_task("look around")
        idx = [i for i, m in enumerate(agent._messages)
               if "[user steering note" in str(m.get("content", ""))]
        assert idx, "steering note never delivered"
        # The note precedes the final assistant message
        final_idx = max(i for i, m in enumerate(agent._messages)
                        if m["role"] == "assistant")
        assert idx[0] < final_idx

    def test_queue_is_thread_safe_smoke(self, tmp_path):
        import threading
        agent = _make_agent(tmp_path)
        threads = [threading.Thread(target=agent.queue_message, args=(f"m{i}",))
                   for i in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert agent.queued_count() == 20


# ---------------------------------------------------------------------------
#  Circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_four_identical_failures_abort_the_task(self, tmp_path):
        agent = _make_agent(tmp_path, [
            _tc("read_file", path="missing.txt")], max_turns=10)
        final = agent.run_task("read the missing file")
        assert "circuit breaker" in final
        assert agent.last_run_ok is False
        # 4 failing turns, not the full budget
        assert agent.turns == 4

    def test_success_resets_the_streak(self, tmp_path):
        (tmp_path / "real.txt").write_text("hi", encoding="utf-8")
        agent = _make_agent(tmp_path, [
            _tc("read_file", path="missing.txt"),
            _tc("read_file", path="missing.txt"),
            _tc("read_file", path="real.txt"),     # success resets streak
            _tc("read_file", path="missing.txt"),
            "Done."], max_turns=10)
        final = agent.run_task("flaky reads")
        assert "circuit breaker" not in final
        assert agent.last_run_ok is True

    def test_streak_hints_escalate(self, tmp_path):
        agent = _make_agent(tmp_path)
        call = _make_call("read_file", path="nope.txt")
        r1 = agent._execute_tool(call, interactive=False)
        r2 = agent._execute_tool(call, interactive=False)
        r3 = agent._execute_tool(call, interactive=False)
        assert "Hint" not in r1.output
        assert "failed twice in a row" in r2.output
        assert "3 times consecutively" in r3.output


# ---------------------------------------------------------------------------
#  Parallel execution
# ---------------------------------------------------------------------------

class TestParallelExecution:
    def test_results_keep_call_order(self, tmp_path):
        for name in ("a", "b", "c"):
            (tmp_path / f"{name}.txt").write_text(name.upper(), encoding="utf-8")
        agent = _make_agent(tmp_path, [
            _tc("read_file", path="a.txt")
            + _tc("read_file", path="b.txt")
            + _tc("read_file", path="c.txt"),
            "Done."])
        agent.run_task("read all three")
        results_msg = next(m["content"] for m in agent._messages
                           if m["role"] == "user"
                           and "<tool_result" in str(m["content"]))
        pos = [results_msg.find(f"<path>{n}.txt</path>") for n in ("a", "b", "c")]
        assert all(p >= 0 for p in pos)
        assert pos == sorted(pos)

    def test_batch_timeout_reports_hung_tool(self, tmp_path):
        from localm.plugins.coder.agent import Agent

        def hang(cwd, **kw):
            time.sleep(1.5)
            return ToolResult.success("finally")

        def quick(cwd, **kw):
            return ToolResult.success("fast")

        fake = {
            "slow_probe": ToolDef("slow_probe", hang, "slow", {}, destructive=False),
            "fast_probe": ToolDef("fast_probe", quick, "fast", {}, destructive=False),
        }
        agent = _make_agent(tmp_path, [
            _tc("slow_probe") + _tc("fast_probe"), "Done."])
        with patch.object(Agent, "_PARALLEL_BATCH_TIMEOUT_S", 0.2), \
             patch.dict("localm.plugins.coder.agent.TOOL_REGISTRY", fake):
            agent.run_task("race")
        results_msg = next(m["content"] for m in agent._messages
                           if "<tool_result" in str(m.get("content", "")))
        assert "did not finish within" in results_msg
        assert "fast" in results_msg            # the quick one still completed


# ---------------------------------------------------------------------------
#  Undo stack
# ---------------------------------------------------------------------------

class TestUndoStack:
    def test_undo_restores_previous_content(self, tmp_path):
        (tmp_path / "a.txt").write_text("before", encoding="utf-8")
        agent = _make_agent(tmp_path, [
            _tc("write_file", path="a.txt", content="after"), "Done."])
        agent.run_task("overwrite")
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "after"
        msg = agent.undo()
        assert "restored" in msg
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "before"

    def test_undo_deletes_newly_created_file(self, tmp_path):
        agent = _make_agent(tmp_path, [
            _tc("write_file", path="new.txt", content="x"), "Done."])
        agent.run_task("create")
        msg = agent.undo()
        assert "file was new" in msg
        assert not (tmp_path / "new.txt").exists()

    def test_undo_empty_stack_returns_none(self, tmp_path):
        agent = _make_agent(tmp_path)
        assert agent.undo() is None

    def test_undo_list_most_recent_first(self, tmp_path):
        agent = _make_agent(tmp_path, [
            _tc("write_file", path="one.txt", content="1"),
            _tc("write_file", path="two.txt", content="2"),
            "Done."])
        agent.run_task("two writes")
        stack = agent.undo_list()
        assert [Path(e["path"]).name for e in stack] == ["two.txt", "one.txt"]
        assert all(e["tool"] == "write_file" for e in stack)


# ---------------------------------------------------------------------------
#  Checkpoint / resume
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_save_load_resume_roundtrip(self, tmp_path):
        agent = _make_agent(tmp_path, mode=SessionMode.LOG)
        agent._messages = [{"role": "user", "content": "hello"}]
        agent._turns = 3
        agent._total_tokens = 99
        agent.save_checkpoint()
        assert agent._checkpoint_path.is_file()

        fresh = _make_agent(tmp_path, mode=SessionMode.LOG)
        data = fresh.load_checkpoint()
        assert data is not None and data["turns"] == 3
        fresh.resume_checkpoint(data)
        assert fresh._messages == [{"role": "user", "content": "hello"}]
        assert fresh.turns == 3 and fresh.total_tokens == 99

    def test_privacy_mode_never_writes_checkpoint(self, tmp_path):
        agent = _make_agent(tmp_path, mode=SessionMode.PRIVACY)
        agent._messages = [{"role": "user", "content": "secret"}]
        agent.save_checkpoint()
        assert not agent._checkpoint_path.exists()

    def test_corrupted_checkpoint_returns_none(self, tmp_path):
        agent = _make_agent(tmp_path, mode=SessionMode.LOG)
        agent._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        agent._checkpoint_path.write_text("{not json", encoding="utf-8")
        assert agent.load_checkpoint() is None

    def test_clear_checkpoint(self, tmp_path):
        agent = _make_agent(tmp_path, mode=SessionMode.LOG)
        agent._messages = [{"role": "user", "content": "x"}]
        agent.save_checkpoint()
        agent.clear_checkpoint()
        assert not agent._checkpoint_path.exists()

    def test_clean_finish_discards_checkpoint(self, tmp_path):
        agent = _make_agent(tmp_path, ["Done."], mode=SessionMode.LOG)
        agent._messages = [{"role": "user", "content": "stale"}]
        agent.save_checkpoint()
        agent.run_task("finish cleanly")
        assert not agent._checkpoint_path.exists()


# ---------------------------------------------------------------------------
#  Stop midstream
# ---------------------------------------------------------------------------

class TestStopMidstream:
    def test_stop_during_generation_keeps_partial_text(self, tmp_path):
        agent = _make_agent(tmp_path, ["partial answer"])

        original_chat = agent.backend.chat
        def chat_then_stop(messages, **kw):
            out = original_chat(messages, **kw)
            agent.request_stop()
            return out
        agent.backend.chat = chat_then_stop

        final = agent.run_task("long thing")
        assert final == "partial answer"
        assert agent.last_run_ok is False

    def test_stale_stop_does_not_kill_next_task(self, tmp_path):
        agent = _make_agent(tmp_path, ["Done."])
        agent.request_stop()              # stale - set before the task starts
        final = agent.run_task("new task")
        assert final == "Done."
        assert agent.last_run_ok is True


# ---------------------------------------------------------------------------
#  Patch-mode fallthrough guard
# ---------------------------------------------------------------------------

class TestPatchModeGuard:
    def test_uncapturable_write_tool_is_blocked(self, tmp_path):
        nb = {"cells": [{"cell_type": "code", "source": ["x = 1\n"],
                         "outputs": []}], "metadata": {}, "nbformat": 4,
              "nbformat_minor": 5}
        nb_path = tmp_path / "n.ipynb"
        nb_path.write_text(json.dumps(nb), encoding="utf-8")

        agent = _make_agent(tmp_path)
        agent.patch_mode = True
        call = _make_call("edit_notebook_cell", path="n.ipynb",
                          cell_index=0, source="x = 2")
        result = agent._execute_tool(call, interactive=False)
        assert not result.ok and "[patch-mode]" in result.output
        # disk untouched
        assert json.loads(nb_path.read_text(encoding="utf-8")) == nb
