# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scheduled-jobs routes and store at their edges: a plain jobs key cannot delete
a shell-enabled job, PUT can clear model, cwd and scope, a run that finishes after
its job was deleted leaves nothing behind, and cancelling an operation that already
ended says so."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.test_jobs_owner_binding import (  # noqa: F401
    OPT_IN, _h, _mk_keys, _shell_job, jobs_app,
)


def test_a_plain_jobs_key_cannot_delete_a_shell_enabled_job(jobs_app, monkeypatch):  # noqa: F811
    (b,) = _mk_keys(["jobs", "chat"])
    _shell_job(owner=None)
    with TestClient(jobs_app) as c:
        jid = c.get("/api/jobs", headers=_h(b)).json()["jobs"][0]["id"]
        r = c.delete(f"/api/jobs/{jid}", headers=_h(b))
        assert r.status_code == 403, r.text
        assert c.get(f"/api/jobs/{jid}", headers=_h(b)).status_code == 200
        monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
        assert c.delete(f"/api/jobs/{jid}", headers=_h("ownersecret")).status_code == 200


def test_put_clears_scope_and_cwd_and_keeps_fields_left_out(jobs_app):  # noqa: F811
    with TestClient(jobs_app) as c:
        r = c.post("/api/jobs", json={"name": "j", "prompt": "hi", "scope": "src/**"})
        assert r.status_code == 200, r.text
        jid = r.json()["id"]
        r = c.put(f"/api/jobs/{jid}", json={"scope": None})
        assert r.status_code == 200, r.text
        assert r.json()["scope"] is None
        assert r.json()["prompt"] == "hi", "a field left out keeps its value"
        r = c.post("/api/jobs", json={"name": "c", "task_kind": "coder", "prompt": "p",
                                      "cwd": "."})
        cid = r.json()["id"]
        r = c.put(f"/api/jobs/{cid}", json={"cwd": None})
        assert r.status_code == 400, "a coder job still needs a cwd"
        assert "cwd" in r.text


def test_a_run_finishing_after_its_job_was_removed_records_nothing(tmp_path):
    from localm.plugins.builtin.jobs.store import Job, JobStore
    store = JobStore(root=tmp_path / "jobs")
    job = Job(name="j", task_kind="chat", prompt="hi", schedule_kind="interval",
              schedule=60)
    store.add(job)
    claimed = store.claim(job.id)
    assert claimed is not None
    assert store.remove(job.id)
    rid = store.record_result(job.id, {"status": "ok", "output": "x",
                                       "started": 1.0, "finished": 2.0})
    assert rid is None
    assert not (tmp_path / "jobs" / "results" / job.id).exists()
    job2 = Job(name="k", task_kind="chat", prompt="hi", schedule_kind="interval",
               schedule=60)
    store.add(job2)
    assert store.record_result(job2.id, {"status": "ok", "output": "y",
                                         "started": 1.0, "finished": 2.0})


def test_cancelling_an_operation_that_ended_says_so(tmp_path):
    import time

    from fastapi import FastAPI

    from localm.jobmanager import JobManager, _ActivityStore
    from localm.plugins.gui.routes import jobs as routes
    app = FastAPI()
    mgr = JobManager(store=_ActivityStore(tmp_path / "activity"), reconcile=False)
    routes.register(app, mgr)
    done = mgr.start_fn("t", lambda job: True)
    deadline = time.monotonic() + 10
    while done.status == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert done.status != "running"
    with TestClient(app) as c:
        r = c.post(f"/api/jobs/{done.id}/cancel")
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "not_running"
        assert r.json()["state"] == done.status


def test_a_result_is_saved_when_the_job_check_cannot_be_made(tmp_path, monkeypatch):
    from localm.plugins.builtin.jobs import store as store_mod
    from localm.plugins.builtin.jobs.store import Job, JobStore
    store = JobStore(root=tmp_path / "jobs")
    job = Job(name="j", task_kind="chat", prompt="hi", schedule_kind="interval",
              schedule=60)
    store.add(job)
    monkeypatch.setattr(store_mod, "_RESULT_CHECK_WAIT_S", 0.2)
    lock = store.root / "jobs.json.lock"
    lock.write_bytes(b"999999:" + b"f" * 32)
    try:
        rid = store.record_result(job.id, {"status": "ok", "output": "kept",
                                           "started": 1.0, "finished": 2.0})
    finally:
        lock.unlink()
    assert rid
    assert [r["output"] for r in store.list_results(job.id)] == ["kept"]


def test_run_now_releases_its_claim_when_the_result_cannot_be_saved(jobs_app,  # noqa: F811
                                                                     monkeypatch):
    from localm.plugins.builtin.jobs.store import JobStore

    def broken(self, job_id, result):
        raise TimeoutError("lock held elsewhere")

    with TestClient(jobs_app, raise_server_exceptions=False) as c:
        jid = c.post("/api/jobs", json={"name": "j", "prompt": "hi"}).json()["id"]
        with monkeypatch.context() as m:
            m.setattr(JobStore, "record_result", broken)
            assert c.post(f"/api/jobs/{jid}/run").status_code == 500
        assert JobStore().get(jid).claim is None
        assert c.post(f"/api/jobs/{jid}/run").status_code == 200, "no stale claim"


def test_cancelling_a_marked_but_unfinished_operation_asks_again(tmp_path):
    import threading
    import time

    from fastapi import FastAPI

    from localm.jobmanager import JobManager, _ActivityStore
    from localm.plugins.gui.routes import jobs as routes
    app = FastAPI()
    mgr = JobManager(store=_ActivityStore(tmp_path / "activity"), reconcile=False)
    routes.register(app, mgr)
    gate = threading.Event()
    job = mgr.start_fn("t", lambda j: gate.wait(10) or True)
    try:
        with TestClient(app) as c:
            assert c.post(f"/api/jobs/{job.id}/cancel").json()["status"] == "cancelling"
            assert job.status == "cancelled" and job.finished_at is None
            assert c.post(f"/api/jobs/{job.id}/cancel").json()["status"] == "cancelling"
    finally:
        gate.set()
        deadline = time.monotonic() + 10
        while job.finished_at is None and time.monotonic() < deadline:
            time.sleep(0.01)
