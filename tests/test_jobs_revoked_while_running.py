# SPDX-License-Identifier: AGPL-3.0-or-later
"""A scheduled job whose creating key is revoked while the job runs is stopped
and its output discarded: at once for a revoke in this process (the auth revoke
listener), within the key-watch interval for a revoke elsewhere, an expiry or a
narrowed scope. A job created by the owner key is never tracked. A run that did
nothing on purpose is recorded as skipped, not ok."""

from __future__ import annotations

import threading
import time

import pytest

from localm import auth
from localm.plugins.builtin.jobs import runner
from localm.plugins.builtin.jobs.store import Job


def _job(owner="k" * 64):
    return Job(name="j", task_kind="rag", collection="kb", schedule_kind="interval",
               schedule=60, owner=owner)


@pytest.fixture
def revocable(monkeypatch):
    monkeypatch.setattr(runner, "_created_by_owner", lambda job: False)
    monkeypatch.setattr(runner, "_creator_refusal", lambda job: None)


def test_the_listener_is_registered_with_auth():
    assert runner._revoke_live_runs in auth._revoke_listeners


def test_a_revoke_in_this_process_discards_the_running_jobs_output(revocable, monkeypatch):
    job = _job()

    def rag(j):
        auth._credentials_changed(j.owner)
        return "indexed 3 files"

    monkeypatch.setattr(runner, "_run_rag", rag)
    out = runner.run_job(job)
    assert out["status"] == "error"
    assert "revoked" in out["error"] and "discarded" in out["error"]
    assert out["output"] == ""
    assert runner._LIVE_RUNS == {}


def test_another_keys_revoke_leaves_the_run_alone(revocable, monkeypatch):
    job = _job()

    def rag(j):
        auth._credentials_changed("a" * 64)
        return "indexed"

    monkeypatch.setattr(runner, "_run_rag", rag)
    assert runner.run_job(job)["status"] == "ok"


def test_the_key_watch_stops_a_run_whose_key_went_away_elsewhere(monkeypatch):
    monkeypatch.setattr(runner, "_created_by_owner", lambda job: False)
    monkeypatch.setattr(runner, "_KEY_WATCH_INTERVAL_S", 0.05)
    monkeypatch.setattr(runner, "_key_watch_running", False)
    calls = {"n": 0}

    def refusal(job):
        calls["n"] += 1
        return None if calls["n"] == 1 else "the key could not be re-confirmed"

    monkeypatch.setattr(runner, "_creator_refusal", refusal)

    def rag(j):
        deadline = time.monotonic() + 5
        while runner._current_live_run().revoked is None and time.monotonic() < deadline:
            time.sleep(0.01)
        return "indexed"

    monkeypatch.setattr(runner, "_run_rag", rag)
    out = runner.run_job(_job())
    assert out["status"] == "error" and "revoked" in out["error"]


def test_a_coder_agent_is_asked_to_stop_when_its_key_is_revoked():
    stopped = threading.Event()

    class _Agent:
        def cancel(self, reason):
            assert reason == runner._REVOKED_DURING_RUN
            stopped.set()

    live = runner._LiveRun(_job())
    live.agent = _Agent()
    live.revoke(runner._REVOKED_DURING_RUN)
    assert stopped.is_set() and live.revoked == runner._REVOKED_DURING_RUN


def test_a_stop_before_the_coder_loop_starts_is_not_lost(tmp_path):
    from unittest.mock import MagicMock, patch

    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = "test-model"
    backend.native_tools = False
    with patch("localm.plugins.coder.agent.ProjectMap") as pm, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        pm.build.return_value.file_count.return_value = 0
        agent = Agent(backend=backend, cwd=tmp_path)
    live = runner._LiveRun(_job())
    live.agent = agent
    live.revoke(runner._REVOKED_DURING_RUN)
    agent.run_task("edit every file")
    assert backend.chat.call_count == 0 and backend.chat_stream.call_count == 0, (
        "a cancelled run asks the model nothing")
    assert agent.stop_pending


def test_a_revoked_memory_run_stops_at_its_next_model_call(monkeypatch):
    live = runner._LiveRun(_job())
    with runner._LIVE_RUNS_LOCK:
        runner._LIVE_RUNS[threading.get_ident()] = live
    try:
        live.revoke(runner._REVOKED_DURING_RUN)
        with pytest.raises(PermissionError, match="stopped"):
            runner._raise_if_revoked()
    finally:
        with runner._LIVE_RUNS_LOCK:
            runner._LIVE_RUNS.pop(threading.get_ident(), None)
    runner._raise_if_revoked()


def test_the_key_watch_keeps_a_run_when_the_keystore_cannot_be_read(monkeypatch):
    monkeypatch.setattr(runner, "_created_by_owner", lambda job: False)
    monkeypatch.setattr(runner, "_KEY_WATCH_INTERVAL_S", 0.02)
    monkeypatch.setattr(runner, "_key_watch_running", False)
    monkeypatch.setattr(auth, "key_hash_live_checked", lambda h: None)
    monkeypatch.setattr(runner, "_creator_refusal", lambda job: "key not found")
    live = runner._register_live_run(_job())
    try:
        time.sleep(0.2)
        assert live.revoked is None
    finally:
        with runner._LIVE_RUNS_LOCK:
            runner._LIVE_RUNS.pop(threading.get_ident(), None)


def test_an_owner_keys_job_is_not_tracked(monkeypatch):
    monkeypatch.setattr(runner, "_created_by_owner", lambda job: True)
    assert runner._register_live_run(_job()) is None


def test_a_deliberately_skipped_run_is_recorded_as_skipped(monkeypatch):
    monkeypatch.setattr(runner, "_creator_refusal", lambda job: None)
    monkeypatch.setattr(runner, "_created_by_owner", lambda job: True)

    def rag(j):
        raise runner._Skipped("memory synthesis skipped (privacy mode)")

    monkeypatch.setattr(runner, "_run_rag", rag)
    out = runner.run_job(_job(owner=None))
    assert out["status"] == "skipped"
    assert out["output"] == "memory synthesis skipped (privacy mode)"
    assert out["error"] is None
