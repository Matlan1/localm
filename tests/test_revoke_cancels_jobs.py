# SPDX-License-Identifier: AGPL-3.0-or-later
"""Revoking a key, its expiry and clearing every key cancel the ordinary jobs the
key started; jobs of another key and of the owner run on. The reply to a revoke
lists the jobs."""

from __future__ import annotations

import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

from localm import auth, jobmanager, scopes
from localm.inference.http_server import create_app

OWNER = "owner-key-owner-key-1234"


def _wait(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


@pytest.fixture
def mgr(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_API_KEY", OWNER)
    m = jobmanager.JobManager(store=jobmanager._ActivityStore(tmp_path / "activity"),
                              reconcile=False)
    yield m
    for j in list(m._jobs.values()):
        j.cancel()
    assert _wait(lambda: all(j.finished_at is not None for j in m._jobs.values()))


def _blocking(job):
    return job.cancel_event.wait(30)


def _start(m, key=None, kind="test"):
    return m.start_fn(kind, _blocking, owner=auth._hash_key(key) if key else None,
                      label="the label")


def test_a_revoke_cancels_only_that_keys_jobs(mgr):
    a = auth.create_key("a", [scopes.CHAT])
    b = auth.create_key("b", [scopes.CHAT])
    mine, other, owned = _start(mgr, a["key"]), _start(mgr, b["key"]), _start(mgr, OWNER)
    open_mode = _start(mgr)
    assert auth.revoke_key(a["id"])
    assert mine.status == "cancelled" and mine.key_revoked
    assert _wait(lambda: mine.finished_at is not None)
    assert [j.status for j in (other, owned, open_mode)] == ["running"] * 3
    assert any("revoked" in e.get("text", "") for e in mine._history)


def test_clearing_every_key_cancels_key_jobs_and_not_the_owners(mgr, monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    auth.set_api_key(OWNER)
    a = auth.create_key("a", [scopes.CHAT])
    mine, owned = _start(mgr, a["key"]), _start(mgr, OWNER)
    auth.clear_api_key()
    assert mine.status == "cancelled"
    assert owned.status == "running"


def test_a_revoke_by_another_process_and_an_expiry_are_noticed(mgr, monkeypatch):
    monkeypatch.setattr(jobmanager, "_KEY_WATCH_SECONDS", 0.05)
    a = auth.create_key("a", [scopes.CHAT])
    b = auth.create_key("b", [scopes.CHAT], expires=time.time() + 0.5)
    c = auth.create_key("c", [scopes.CHAT])
    removed, expiring, live = (_start(mgr, k["key"]) for k in (a, b, c))
    auth._save_keystore([r for r in auth._load_keystore() if r["id"] != a["id"]])
    assert _wait(lambda: removed.status == "cancelled")
    assert _wait(lambda: expiring.status == "cancelled")
    assert live.status == "running"
    live.cancel()
    assert _wait(lambda: not mgr._watching_keys)


def test_an_unreadable_keystore_cancels_nothing(mgr, monkeypatch):
    a = auth.create_key("a", [scopes.CHAT])
    job = _start(mgr, a["key"])
    monkeypatch.setattr(auth, "key_hash_live_checked", lambda h: None)
    mgr._cancel_jobs_of_dead_keys()
    assert job.status == "running"


def test_a_job_registered_after_its_key_was_revoked_is_cancelled(mgr):
    a = auth.create_key("a", [scopes.CHAT])
    assert auth.revoke_key(a["id"])
    job = _start(mgr, a["key"])
    assert job.status == "cancelled"


def test_a_revoke_ends_the_child_process_of_a_command_job(mgr, monkeypatch):
    a = auth.create_key("a", [scopes.CHAT])
    real = jobmanager.subprocess.Popen
    monkeypatch.setattr(
        jobmanager.subprocess, "Popen",
        lambda argv, **kw: real([sys.executable, "-c", "import time; time.sleep(60)"],
                                creationflags=getattr(jobmanager.subprocess,
                                                      "CREATE_NO_WINDOW", 0), **kw))
    job = mgr.start_cli("pull", ["pull", "x"], owner=auth._hash_key(a["key"]))
    assert _wait(lambda: job._proc is not None)
    assert auth.revoke_key(a["id"])
    assert _wait(lambda: job._proc.poll() is not None)
    assert _wait(lambda: job.finished_at is not None) and job.status == "cancelled"


def test_a_cancelled_callback_that_raises_job_cancelled_ends_cancelled(mgr):
    go = threading.Event()

    def work(job):
        go.wait(10)
        job.raise_if_cancelled()
        return True

    job = mgr.start_fn("rag-index", work)
    job.cancel()
    go.set()
    assert _wait(lambda: job.finished_at is not None)
    assert job.status == "cancelled"
    assert any(e.get("text") == "Stopped before it was finished." for e in job._history)
    done = mgr.start_fn("rag-index", lambda j: j.raise_if_cancelled() is None)
    assert _wait(lambda: done.finished_at is not None) and done.status == "done"


def test_an_indexing_jobs_progress_stops_a_cancelled_run(mgr, tmp_path):
    from localm.plugins.builtin.rag.plug import _job_progress
    from localm.rag import Collection
    base = tmp_path / "collections"
    base.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(3):
        (docs / f"d{i}.txt").write_text(f"document number {i}", encoding="utf-8")
    Collection("c", base=base).create()
    reading = []

    def work(job):
        tell = _job_progress(job)

        def on_progress(text, **kw):
            if "reading" in text:
                reading.append(text)
                if len(reading) == 2:
                    job.cancel()
            tell(text, **kw)

        Collection("c", base=base, load=False, cache=False).add_paths(
            [docs], on_progress=on_progress)
        return True

    job = mgr.start_fn("rag-index", work)
    assert _wait(lambda: job.finished_at is not None)
    assert job.status == "cancelled" and len(reading) == 2
    saved = Collection("c", base=base, cache=False)
    assert saved.documents() == [] and not saved.corrupt
    Collection("c", base=base, load=False, cache=False).add_paths([docs])
    assert len(Collection("c", base=base, cache=False).documents()) == 3


def test_the_reply_lists_the_cancelled_jobs(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_API_KEY", OWNER)
    monkeypatch.setattr("localm.inference.routes.keys._JOB_END_WAIT_S", 5.0)
    app = create_app(None)
    m = jobmanager.JobManager(store=jobmanager._ActivityStore(tmp_path / "activity"),
                              reconcile=False)
    app.state.jobs = m
    release = threading.Event()
    with TestClient(app) as c:
        own = {"Authorization": f"Bearer {OWNER}"}
        a = auth.create_key("a", [scopes.CHAT])
        quick = _start(m, a["key"], kind="pull")
        body = c.delete(f"/v1/keys/{a['id']}", headers=own).json()
        assert body == {"status": "revoked", "id": a["id"], "cancelled_jobs": [
            {"id": quick.id, "kind": "pull", "label": "the label", "ended": True}]}

        monkeypatch.setattr("localm.inference.routes.keys._JOB_END_WAIT_S", 0.1)
        admin = auth.create_key("admin", [scopes.KEYS_ADMIN], allow_privileged=True)
        b = auth.create_key("b", [scopes.CHAT])
        stuck = m.start_fn("voice-model-download", lambda j: release.wait(20),
                           owner=auth._hash_key(b["key"]), label="secret label")
        body = c.delete(f"/v1/keys/{b['id']}",
                        headers={"Authorization": f"Bearer {admin['key']}"}).json()
        assert body["cancelled_jobs"] == [
            {"id": stuck.id, "kind": "voice-model-download", "label": None,
             "ended": False}]
        release.set()
        assert _wait(lambda: stuck.finished_at is not None)
        d = auth.create_key("d", [scopes.CHAT])
        assert c.delete(f"/v1/keys/{d['id']}", headers=own).json()["cancelled_jobs"] == []
        assert c.delete(f"/v1/keys/{d['id']}", headers=own).status_code == 404


def test_a_cancelled_browser_download_kills_its_installer():
    from localm.browser import provision
    started = threading.Event()
    stop = threading.Event()
    out = {}

    def run():
        out["result"] = provision._stream_install(
            [sys.executable, "-c",
             "import time; print('go', flush=True); time.sleep(60)"],
            on_progress=lambda line: started.set(), timeout=60,
            should_cancel=stop.is_set, kill=jobmanager._terminate_process_tree)

    t = threading.Thread(target=run)
    t.start()
    assert started.wait(20)
    begun = time.monotonic()
    stop.set()
    t.join(20)
    assert not t.is_alive() and time.monotonic() - begun < 20
    code, lines = out["result"]
    assert code is None and lines[-1] == "(the download was cancelled)"
