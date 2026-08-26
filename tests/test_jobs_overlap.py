# SPDX-License-Identifier: AGPL-3.0-or-later
"""Job-run overlap guard: while a previous run is in flight, a second run is
skipped rather than stacking another model load. Also covers freeing a
self-loaded engine after the run, while leaving a passed-in engine alone.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _make_job(**kw):
    from localm.plugins.builtin.jobs.store import Job
    base = dict(name="t", task_kind="chat", prompt="hi",
                schedule_kind="interval", schedule=60)
    base.update(kw)
    return Job(**base)


class _OkRun:
    """A run_job stand-in that records each call and returns an ok result."""

    def __init__(self):
        self.calls = []

    def __call__(self, job, *, engine=None):
        self.calls.append(job.id)
        return {"status": "ok", "output": "ran", "error": None,
                "started": 0.0, "finished": 0.0}


# --------------------------------------------------------------------------- #
#  runguard primitive                                                          #
# --------------------------------------------------------------------------- #

def test_run_slot_is_exclusive():
    from localm.plugins.builtin.jobs.runguard import is_running, run_slot
    assert is_running() is False
    with run_slot() as got1:
        assert got1 is True
        assert is_running() is True
        with run_slot() as got2:
            assert got2 is False          # already held -> the second caller skips
    assert is_running() is False          # released on exit


# --------------------------------------------------------------------------- #
#  Scheduler tick skips while a run is in flight                               #
# --------------------------------------------------------------------------- #

def test_tick_skips_when_a_run_is_in_flight(home):
    from localm.plugins.builtin.jobs import runguard
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore

    store = JobStore()
    job = store.add(_make_job(schedule=60, last_run=None))
    runner = _OkRun()
    sched = JobScheduler(store, run_job=runner)

    # Simulate a previous run (scheduled or GUI run-now) still holding the slot.
    assert runguard._RUN_LOCK.acquire(blocking=False)
    try:
        ran = sched.tick(now=1000.0)
    finally:
        runguard._RUN_LOCK.release()

    assert ran == []                          # tick skipped its due job
    assert runner.calls == []                 # the runner was never invoked
    assert store.list_results(job.id) == []   # nothing recorded
    # A cron job's minute is not consumed on a skip (interval job here, but assert the
    # job is still due once the slot frees):
    assert sched.due(store.get(job.id), now=1000.0) is True

    # With the slot free, the same tick runs it.
    ran2 = sched.tick(now=1000.0)
    assert ran2 == [job.id]
    assert runner.calls == [job.id]


def test_tick_runs_normally_when_slot_free(home):
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore

    store = JobStore()
    job = store.add(_make_job(schedule=60, last_run=None))
    runner = _OkRun()
    ran = JobScheduler(store, run_job=runner).tick(now=1000.0)
    assert ran == [job.id] and runner.calls == [job.id]


# --------------------------------------------------------------------------- #
#  Self-loaded engine is unloaded; a passed-in engine is not                  #
# --------------------------------------------------------------------------- #

class _Eng:
    def __init__(self):
        self.unloaded = 0

    def chat_stream(self, messages, **kw):
        yield "ok"

    def unload(self):
        self.unloaded += 1


def test_self_loaded_engine_is_unloaded(home, monkeypatch):
    from localm.plugins.builtin.jobs import runner
    monkeypatch.setenv("LOCALM_NET_MODE", "off")     # keep the chat path simple
    eng = _Eng()
    # _load_engine returns (engine, reused); a fresh runner-loaded engine is reused=False.
    monkeypatch.setattr(runner, "_load_engine", lambda model: (eng, False))

    result = runner.run_job(_make_job(task_kind="chat", prompt="hello"), engine=None)
    assert result["status"] == "ok"
    assert eng.unloaded == 1          # the run freed the model it loaded itself


def test_passed_in_engine_is_not_unloaded(home, monkeypatch):
    from localm.plugins.builtin.jobs import runner
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    eng = _Eng()
    result = runner.run_job(_make_job(task_kind="chat", prompt="x"), engine=eng)
    assert result["status"] == "ok"
    assert eng.unloaded == 0          # the host's shared engine must never be unloaded


# --------------------------------------------------------------------------- #
#  run-now route returns a clear busy response (no stacked load)              #
# --------------------------------------------------------------------------- #

def test_run_now_busy_returns_409(home, monkeypatch):
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)

    import localm.plugins.builtin.jobs.runner as runner
    monkeypatch.setattr(
        runner, "run_job",
        lambda job, engine=None: {"status": "ok", "output": "ROUTE-OK",
                                  "error": None, "started": 0.0, "finished": 0.0})

    store_root = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "builtin"
    from localm.plugins.builtin.jobs import runguard
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    mgr = PluginManager(app, store_root=store_root, installed_root=home / "plugins")
    mgr.install("jobs")

    with TestClient(app) as c:
        jid = c.post("/api/jobs", json={"name": "j", "prompt": "hi",
                                        "schedule_kind": "interval",
                                        "schedule": 120}).json()["id"]
        # Hold the run slot to simulate a run already in flight.
        assert runguard._RUN_LOCK.acquire(blocking=False)
        try:
            r = c.post(f"/api/jobs/{jid}/run")
        finally:
            runguard._RUN_LOCK.release()
        assert r.status_code == 409
        assert "in progress" in r.json()["detail"]

        # Once the slot is free, run-now works.
        r = c.post(f"/api/jobs/{jid}/run")
        assert r.status_code == 200 and r.json()["output"] == "ROUTE-OK"
