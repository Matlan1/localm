# SPDX-License-Identifier: AGPL-3.0-or-later
"""sessions.json and jobs.json read-modify-writes serialise ACROSS OS processes.

Each store's in-process lock cannot order a write made by another process (the
``localm key recover`` / ``key clear`` CLI against a running server's session
stamps, or ``localm jobs ...`` against the scheduler's result stamps). Both
stores also hold ``config._cross_process_lock`` on their file for the whole
read-modify-write, so a write that started first cannot land on top of one that
finished after it and undo it.

The race tests spawn a REAL second interpreter whose write is slowed down
between its read and its write, and only proceed once it has read."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import localm.config as cfg
from localm import sessions
from localm.plugins.builtin.jobs.store import Job, JobStore


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    h.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(sessions, "_CACHE", {"mtime": None, "records": None})
    monkeypatch.setattr(sessions, "_last_used_writes", {})
    return h


def _child_env(home: Path) -> dict:
    env = dict(os.environ)
    env["LOCALM_HOME"] = str(home)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return env


def _start_slow_writer(home: Path, script: str) -> subprocess.Popen:
    """Run *script* in a second interpreter and return once it printed READ
    (it has read the store and is sleeping before its write)."""
    proc = subprocess.Popen([sys.executable, "-c", script], env=_child_env(home),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    line = proc.stdout.readline()
    if line.strip() != "READ":
        proc.kill()
        _out, err = proc.communicate(timeout=30)
        pytest.fail(f"the writer process never reached its write: {line!r} {err[-2000:]}")
    return proc


def _finish(proc: subprocess.Popen) -> None:
    _out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, err[-2000:]


_SLOW_SESSION_CREATE = (
    "import time\n"
    "from localm import sessions as s\n"
    "real = s._save\n"
    "def slow(records):\n"
    "    print('READ', flush=True)\n"
    "    time.sleep(1.5)\n"
    "    real(records)\n"
    "s._save = slow\n"
    "s.create(scopes={'chat'}, key_hash='CHILD')\n"
)


def test_revoke_all_is_not_undone_by_a_write_from_another_process(home):
    before = [sessions.create(scopes={"admin"}, key_hash="KH") for _ in range(3)]
    proc = _start_slow_writer(home, _SLOW_SESSION_CREATE)
    try:
        revoked = sessions.revoke_all()
    finally:
        _finish(proc)

    assert revoked is not None and revoked >= len(before)
    sessions._CACHE.update(mtime=None, records=None)
    alive = [sid for sid in before if sessions.lookup(sid) is not None]
    assert alive == [], (
        f"{len(alive)} revoked session(s) resolve again: a write from another "
        "process landed on top of revoke_all")


def test_a_session_write_waits_for_a_lock_held_by_another_process(home, monkeypatch):
    sid = sessions.create(scopes={"admin"}, key_hash="KH")
    lock = sessions.sessions_file().with_name("sessions.json.lock")
    lock.write_bytes(b"999999:" + b"f" * 32)
    monkeypatch.setattr(cfg, "_CROSS_LOCK_TIMEOUT", 0.3)

    assert sessions.revoke_all() is None, (
        "revoke_all reported a result while another process held the store")
    assert sessions.lookup(sid) is not None
    with pytest.raises(TimeoutError):
        sessions.create(scopes={"admin"}, key_hash="KH")

    lock.unlink()
    assert sessions.revoke_all() == 1
    assert sessions.lookup(sid) is None


def test_a_held_lock_costs_the_request_path_at_most_its_short_budget(home):
    sid = sessions.create(scopes={"admin"}, key_hash="KH")
    lock = sessions.sessions_file().with_name("sessions.json.lock")
    lock.write_bytes(b"999999:" + b"f" * 32)
    sessions._last_used_writes.clear()

    started = time.monotonic()
    rec = sessions.lookup(sid)
    assert sessions.remember_owner_key_minted(sid) is False
    elapsed = time.monotonic() - started

    assert rec is not None
    assert elapsed < 2 * sessions._BEST_EFFORT_LOCK_TIMEOUT_S + 1.0, elapsed
    assert sessions._BEST_EFFORT_LOCK_TIMEOUT_S < cfg._CROSS_LOCK_TIMEOUT


def test_a_failed_save_leaves_the_cache_holding_what_is_on_disk(home, monkeypatch):
    sid = sessions.create(scopes={"admin"}, key_hash="OLD")
    assert sessions.lookup(sid)["key_hash"] == "OLD"

    def _fail(_records):
        raise OSError("disk full")
    monkeypatch.setattr(sessions, "_save", _fail)

    assert sessions.relink_key_hash("OLD", "NEW") == 0
    assert sessions.lookup(sid)["key_hash"] == "OLD", (
        "a relink that was never written is visible through the cache")


def test_threads_in_one_process_still_serialise(home):
    errors = []

    def _mint():
        try:
            for _ in range(5):
                sessions.create(scopes={"chat"}, key_hash="T")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_mint) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert errors == []
    records = json.loads(sessions.sessions_file().read_text(encoding="utf-8"))
    assert len(records) == 20
    assert not sessions.sessions_file().with_name("sessions.json.lock").exists()


def test_a_missing_data_dir_is_not_created_by_a_revoke(tmp_path, monkeypatch):
    missing = tmp_path / "absent"
    monkeypatch.setenv("LOCALM_HOME", str(missing))
    monkeypatch.setattr(sessions, "_CACHE", {"mtime": None, "records": None})
    assert sessions.revoke_all() == 0
    assert sessions.revoke("x") is False
    assert not missing.exists()


_SLOW_JOB_STAMP = (
    "import sys, time\n"
    "from localm.plugins.builtin.jobs.store import JobStore\n"
    "store = JobStore()\n"
    "real = store._write_all\n"
    "def slow(jobs):\n"
    "    print('READ', flush=True)\n"
    "    time.sleep(1.5)\n"
    "    real(jobs)\n"
    "store._write_all = slow\n"
    "store.record_result(sys.argv[1], {'status': 'ok', 'finished': 1.0, 'output': 'x'})\n"
)


def _job(store: JobStore) -> Job:
    return store.add(Job(id="j1", name="j", task_kind="chat", prompt="hi",
                         schedule_kind="interval", schedule=3600, enabled=True))


def test_jobs_disable_is_not_undone_by_a_result_stamp_from_another_process(home):
    store = JobStore()
    _job(store)
    proc = subprocess.Popen(
        [sys.executable, "-c", _SLOW_JOB_STAMP.replace("sys.argv[1]", "'j1'")],
        env=_child_env(home), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)
    line = proc.stdout.readline()
    if line.strip() != "READ":
        proc.kill()
        pytest.fail(f"the stamping process never reached its write: {line!r} "
                    f"{proc.communicate(timeout=30)[1][-2000:]}")
    try:
        store.update("j1", enabled=False)
    finally:
        _finish(proc)

    job = JobStore().get("j1")
    assert job.enabled is False, (
        "a result stamp written by another process re-enabled a disabled job")
    assert job.last_status == "ok"


def test_a_job_write_waits_for_a_lock_held_by_another_process(home, monkeypatch):
    store = JobStore()
    _job(store)
    lock = store.root / "jobs.json.lock"
    lock.write_bytes(b"999999:" + b"f" * 32)
    monkeypatch.setattr(cfg, "_CROSS_LOCK_TIMEOUT", 0.3)

    with pytest.raises(TimeoutError):
        store.update("j1", enabled=False)
    assert JobStore().get("j1").enabled is True

    rid = store.record_result("j1", {"status": "ok", "finished": 2.0})
    assert rid
    assert JobStore().get("j1").last_result_id is None

    lock.unlink()
    store.update("j1", enabled=False)
    assert JobStore().get("j1").enabled is False
