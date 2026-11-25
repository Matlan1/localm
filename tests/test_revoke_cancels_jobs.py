# SPDX-License-Identifier: AGPL-3.0-or-later
"""Revoking a key, its expiry and clearing every key cancel the ordinary jobs the
key started; jobs of another key and of the owner run on. The reply to a revoke
lists the jobs."""

from __future__ import annotations

import asyncio
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


def test_a_subscriber_whose_loop_is_closed_stops_nothing(mgr, monkeypatch):
    monkeypatch.setattr(jobmanager, "_KEY_WATCH_SECONDS", 0.05)
    a, b = (auth.create_key(n, [scopes.CHAT]) for n in "ab")
    jobs = [_start(mgr, a["key"]), _start(mgr, a["key"]), _start(mgr, b["key"])]
    for job in jobs:
        async def subscribe(job=job):
            job.subscribe()
        asyncio.run(subscribe())
        assert job.subscriber_count == 1
    assert auth.revoke_key(a["id"])
    assert [j.status for j in jobs] == ["cancelled", "cancelled", "running"]
    auth._save_keystore([])
    assert _wait(lambda: jobs[2].status == "cancelled")
    assert _wait(lambda: all(j.finished_at is not None for j in jobs))
    assert _wait(lambda: all(j.subscriber_count == 0 for j in jobs))
    assert _wait(lambda: all(any("revoked" in e.get("text", "") for e in list(j._history))
                             for j in jobs))


def test_a_job_is_cancelled_before_its_subscribers_are_told(mgr, monkeypatch):
    a = auth.create_key("a", [scopes.CHAT])
    job = _start(mgr, a["key"])
    seen = []
    real = job.push

    def push(event):
        seen.append(job.cancel_requested)
        real(event)

    monkeypatch.setattr(job, "push", push)
    assert auth.revoke_key(a["id"])
    assert seen and all(seen)


def test_an_unreadable_keystore_cancels_nothing(mgr, monkeypatch):
    a = auth.create_key("a", [scopes.CHAT])
    job = _start(mgr, a["key"])
    with monkeypatch.context() as m:
        m.setattr(auth, "key_hash_live_checked", lambda h: None)
        mgr._check_keys_of_jobs()
        assert job.status == "running"
    assert auth.revoke_key(a["id"])
    assert job.status == "cancelled"


def test_a_job_registered_after_its_key_was_revoked_is_cancelled(mgr):
    a = auth.create_key("a", [scopes.CHAT])
    assert auth.revoke_key(a["id"])
    job = _start(mgr, a["key"])
    assert job.status == "cancelled"


@pytest.fixture
def child(monkeypatch):
    """Every command job runs a child that sleeps. Yields the Popen objects and an
    event a test can clear to hold the start of the child back."""
    real = jobmanager.subprocess.Popen
    made, go = [], threading.Event()
    go.set()

    def popen(argv, **kw):
        assert go.wait(20)
        made.append(real([sys.executable, "-c", "import time; time.sleep(60)"],
                         creationflags=getattr(jobmanager.subprocess,
                                               "CREATE_NO_WINDOW", 0), **kw))
        return made[-1]

    class _Subprocess:
        Popen = staticmethod(popen)

        def __getattr__(self, name):
            import subprocess
            return getattr(subprocess, name)

    monkeypatch.setattr(jobmanager, "subprocess", _Subprocess())
    yield made, go
    go.set()
    for p in made:
        p.kill()
        p.wait(10)


def test_a_command_job_registered_after_the_revoke_starts_no_process(mgr, child):
    made, _go = child
    a = auth.create_key("a", [scopes.CHAT])
    assert auth.revoke_key(a["id"])
    for kind in ("pull", "runtime-update"):
        job = mgr.start_cli(kind, ["x"], owner=auth._hash_key(a["key"]))
        assert _wait(lambda: job.finished_at is not None)  # noqa: B023
        assert job.status == "cancelled"
    assert made == []


def test_a_revoke_between_registration_and_the_process_start_ends_it(mgr, child):
    made, go = child
    a = auth.create_key("a", [scopes.CHAT])
    go.clear()
    job = mgr.start_cli("pull", ["x"], owner=auth._hash_key(a["key"]))
    time.sleep(0.2)
    assert auth.revoke_key(a["id"]) and job.status == "cancelled"
    go.set()
    assert _wait(lambda: made and made[0].poll() is not None)
    assert _wait(lambda: job.finished_at is not None) and job.status == "cancelled"


def test_the_watch_stops_a_cancelled_job_whose_process_still_lives(mgr, child, monkeypatch):
    monkeypatch.setattr(jobmanager, "_KEY_WATCH_SECONDS", 0.05)
    made, _go = child
    a = auth.create_key("a", [scopes.CHAT])
    mgr.start_cli("pull", ["x"], owner=auth._hash_key(a["key"]))
    assert _wait(lambda: made and mgr._watched_jobs()[0].process_alive)
    stop, calls = made[0].terminate, []

    def terminate():
        calls.append(1)
        if len(calls) > 2:
            stop()

    made[0].terminate = terminate
    assert auth.revoke_key(a["id"])
    assert _wait(lambda: made[0].poll() is not None) and len(calls) >= 3
    assert _wait(lambda: not mgr._watching_keys)


@pytest.mark.parametrize("kind", sorted(jobmanager._FINISHES_AFTER_REVOKE))
def test_a_job_that_would_break_the_install_runs_to_its_end(mgr, child, kind):
    made, _go = child
    a = auth.create_key("a", [scopes.CHAT])
    job = mgr.start_cli(kind, ["x"], owner=auth._hash_key(a["key"]))
    assert _wait(lambda: made and job._proc is not None)
    assert auth.revoke_key(a["id"])
    assert job.status == "running" and made[0].poll() is None
    [row] = mgr.jobs_of_revoked_key(auth._hash_key(a["key"]), wait_s=5)
    assert row["cancelled"] is False and row["ended"] is False
    mgr._check_keys_of_jobs()
    assert job.status == "running"
    assert any("runs to its end" in e.get("text", "") for e in job._history)


def test_a_job_whose_key_is_already_gone_at_registration_is_cancelled(mgr, monkeypatch):
    monkeypatch.setattr(jobmanager, "_KEY_WATCH_SECONDS", 0.05)
    a = auth.create_key("a", [scopes.CHAT])
    auth._save_keystore([])
    job = _start(mgr, a["key"])
    assert _wait(lambda: job.status == "cancelled")


def test_a_job_registered_while_the_keystore_cannot_be_read_stays_revocable(
        mgr, monkeypatch):
    a = auth.create_key("a", [scopes.CHAT])
    with monkeypatch.context() as m:
        m.setattr(auth, "key_hash_live_checked", lambda h: None)
        job = _start(mgr, a["key"])
        mgr._check_keys_of_jobs()
        assert job.status == "running"
    mgr._check_keys_of_jobs()
    assert job.status == "running"
    assert auth.revoke_key(a["id"]) and job.status == "cancelled"


def test_an_owner_session_from_before_a_key_roll_is_the_owner(mgr, monkeypatch):
    from localm import sessions
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    auth.set_api_key(OWNER)
    before = auth._hash_key(OWNER)
    sessions.create(scopes={"admin"}, key_hash=before, owner_key_minted=True)
    auth.set_api_key("another-owner-key-9876")
    job = mgr.start_fn("test", _blocking, owner=before)
    assert job.key_hash is None
    mgr._check_keys_of_jobs()
    assert job.status == "running"
    stranger = mgr.start_fn("test", _blocking, owner="f" * 64)
    mgr._check_keys_of_jobs()
    assert stranger.status == "cancelled"


def test_a_revoke_ends_the_child_process_of_a_command_job(mgr, child):
    a = auth.create_key("a", [scopes.CHAT])
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


def test_the_reply_lists_the_keys_jobs(monkeypatch, tmp_path):
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
        assert body == {"status": "revoked", "id": a["id"], "jobs": [
            {"id": quick.id, "kind": "pull", "label": "the label",
             "cancelled": True, "ended": True}]}

        monkeypatch.setattr("localm.inference.routes.keys._JOB_END_WAIT_S", 0.1)
        admin = auth.create_key("admin", [scopes.KEYS_ADMIN], allow_privileged=True)
        b = auth.create_key("b", [scopes.CHAT])
        stuck = m.start_fn("voice-model-download", lambda j: release.wait(20),
                           owner=auth._hash_key(b["key"]), label="secret label")
        body = c.delete(f"/v1/keys/{b['id']}",
                        headers={"Authorization": f"Bearer {admin['key']}"}).json()
        assert body["jobs"] == [
            {"id": stuck.id, "kind": "voice-model-download", "label": None,
             "cancelled": True, "ended": False}]
        release.set()
        assert _wait(lambda: stuck.finished_at is not None)
        d = auth.create_key("d", [scopes.CHAT])
        assert c.delete(f"/v1/keys/{d['id']}", headers=own).json()["jobs"] == []
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
