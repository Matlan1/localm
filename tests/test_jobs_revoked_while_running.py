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
        def request_stop(self):
            stopped.set()

    live = runner._LiveRun(_job())
    live.agent = _Agent()
    live.revoke(runner._REVOKED_DURING_RUN)
    assert stopped.is_set() and live.revoked == runner._REVOKED_DURING_RUN


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
