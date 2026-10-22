# SPDX-License-Identifier: AGPL-3.0-or-later
"""update_config() and update_registry() must serialize ACROSS OS processes.

config._io_lock (threading.RLock) is in-process by construction: it cannot
serialize two SEPARATE localm OS processes. The CLI `localm config` racing a
running server's PATCH /v1/config, two CLI invocations, or a CLI `pull` racing
the GUI for the registry are genuinely different OS processes, each with its own
_io_lock, so one writer's read-modify-write can complete entirely inside the
other's read-to-write window, silently losing whichever change gets
read-before-written-back last.

update_config()/update_registry() therefore also hold a cross-process lock FILE
(config._cross_process_lock, os.O_CREAT|O_EXCL) across the whole
read-modify-write, so a second process attempting the same operation blocks
until the first fully commits.

These tests spawn two REAL, independent Python processes - not threads - so
they exercise actual cross-process contention. See
test_app_lifecycle_1_atomic_config.py for the same-process (threading) coverage
of update_config()'s own locking and the CLI/HTTP call sites."""

import json
import os
import subprocess
import sys
import threading
import time
import types

import pytest

import localm.config as cfg


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    h.mkdir()
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    return h


# Each worker is a genuinely separate python process. LOCALM_HOME is set in the
# process environment before the interpreter starts, so config.py picks it up at
# import-time HOME_DIR resolution. The mutator sleeps BEFORE mutating, widening
# the read-modify-write window so two processes started back-to-back overlap.
_WORKER = (
    "import sys, time\n"
    "import localm.config as c\n"
    "fn = getattr(c, sys.argv[1])\n"
    "key, value, delay = sys.argv[2], sys.argv[3], float(sys.argv[4])\n"
    "def _mut(d):\n"
    "    time.sleep(delay)\n"
    "    d[key] = value\n"
    "fn(_mut)\n"
)

# Same, generous delay for BOTH processes: whichever wins the lock file holds it
# for the full delay, so the second process is still blocked on acquisition when
# the first commits.
_RACE_DELAY = 0.3


def _spawn_two_writers(home_dir, fn_name, key_a, key_b):
    env = dict(os.environ)
    env["LOCALM_HOME"] = str(home_dir)
    # Make the child import the same localm this test runs (the worktree), not
    # the venv's editable install.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, fn_name, key_a, "writer-a", str(_RACE_DELAY)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE),
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, fn_name, key_b, "writer-b", str(_RACE_DELAY)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE),
    ]
    for i, p in enumerate(procs):
        out, err = p.communicate(timeout=60)
        assert p.returncode == 0, (
            f"writer {i} ({fn_name}) crashed (rc={p.returncode}): "
            f"{err.decode('utf-8', 'replace')[-2000:]}")


def test_update_config_survives_a_real_concurrent_process(home):
    """Two SEPARATE OS processes both calling update_config() concurrently
    (the CLI `localm config` racing a running server's PATCH /v1/config, which
    both route through update_config()) must not lose either process's
    change."""
    cfg.save_config({"n_ctx": 4096})
    _spawn_two_writers(cfg.HOME_DIR, "update_config", "temperature", "main_gpu_index")
    final = json.loads(cfg.CONFIG_FILE.read_text(encoding="utf-8"))
    assert final.get("temperature") == "writer-a"
    assert final.get("main_gpu_index") == "writer-b", (
        "a concurrent update_config() write from a SEPARATE OS process was "
        "silently lost - the cross-process lock did not close the race")


def test_update_registry_survives_a_real_concurrent_process(home):
    """Two SEPARATE OS processes both calling update_registry() concurrently
    (e.g. two `localm pull` invocations, or a pull racing the GUI) must not
    lose either process's registered model entry."""
    cfg.save_registry({"seed": {"path": "Z:/seed.gguf"}})
    _spawn_two_writers(cfg.HOME_DIR, "update_registry", "model-a", "model-b")
    final = json.loads(cfg.REGISTRY_FILE.read_text(encoding="utf-8"))
    assert final.get("model-a") == "writer-a"
    assert final.get("model-b") == "writer-b", (
        "a concurrent update_registry() write from a SEPARATE OS process was "
        "silently lost - the cross-process lock did not close the race")


# --------------------------------------------------------------------------- #
#  Deterministic (non-timing-dependent) coverage of the lock mechanism itself #
# --------------------------------------------------------------------------- #

def test_cross_process_lock_reclaims_a_stale_lock_file(home, monkeypatch, capsys):
    """A lock file left behind by a crashed holder must not wedge every future
    write forever - it is reclaimed once older than the staleness timeout, and
    the reclaim is surfaced, not silent."""
    monkeypatch.setattr(cfg, "_CROSS_LOCK_STALE_AGE", 0.05)
    monkeypatch.setattr(cfg, "_CROSS_LOCK_TIMEOUT", 5.0)
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    lockpath.write_text("99999999", encoding="utf-8")
    old = cfg.time.time() - 10  # far older than the 0.05s staleness threshold
    os.utime(lockpath, (old, old))

    cfg.save_config({"n_ctx": 4096})
    result = cfg.update_config(lambda c: c.__setitem__("temperature", 0.5))

    assert result["temperature"] == 0.5
    assert not lockpath.exists(), "the lock must be released after the write"
    assert "reclaiming stale lock" in capsys.readouterr().err


def test_cross_process_lock_times_out_on_a_live_holder(home, monkeypatch):
    """A lock file that is FRESH (not stale) must be respected - a second
    acquirer waits, and gives up with a clear TimeoutError rather than
    silently proceeding unprotected past a lock it never actually got."""
    monkeypatch.setattr(cfg, "_CROSS_LOCK_STALE_AGE", 3600.0)
    monkeypatch.setattr(cfg, "_CROSS_LOCK_TIMEOUT", 0.2)
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    cfg.ensure_dirs()
    lockpath.write_text("12345", encoding="utf-8")
    try:
        with pytest.raises(TimeoutError):
            cfg.update_config(lambda c: c.__setitem__("temperature", 0.5))
    finally:
        lockpath.unlink()


def test_a_lock_timeout_is_a_config_lock_timeout(home, monkeypatch):
    monkeypatch.setattr(cfg, "_CROSS_LOCK_STALE_AGE", 3600.0)
    monkeypatch.setattr(cfg, "_CROSS_LOCK_TIMEOUT", 0.2)
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    cfg.ensure_dirs()
    lockpath.write_text("12345", encoding="utf-8")
    try:
        with pytest.raises(cfg.ConfigLockTimeout):
            cfg.update_config(lambda c: c.__setitem__("temperature", 0.5))
    finally:
        lockpath.unlink()


def test_settings_write_blocked_by_another_process_answers_busy_not_500(
        home, monkeypatch, caplog):
    """PATCH /v1/config while another localm process holds config.json.lock
    answers 503 with Retry-After, says nothing was saved, and logs no traceback."""
    import logging

    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app

    monkeypatch.setenv("LOCALM_API_KEY", "owner-key-for-the-busy-settings-test")
    monkeypatch.setattr(cfg, "_CROSS_LOCK_STALE_AGE", 3600.0)
    monkeypatch.setattr(cfg, "_CROSS_LOCK_TIMEOUT", 0.3)
    cfg.save_config({"temperature": 0.7})
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    caplog.set_level(logging.WARNING, logger="localm")
    with TestClient(create_app(None), raise_server_exceptions=False) as client:
        lockpath.write_text("12345", encoding="utf-8")
        try:
            response = client.patch(
                "/v1/config", json={"temperature": 0.2},
                headers={"Authorization": "Bearer owner-key-for-the-busy-settings-test"})
        finally:
            lockpath.unlink()
    assert response.status_code == 503
    assert response.headers.get("retry-after")
    assert "busy" in response.json()["detail"]
    assert "Internal server error" not in response.text
    assert cfg.load_config()["temperature"] == 0.7
    assert not [r for r in caplog.records if r.exc_info]


def test_cross_process_lock_is_released_on_mutator_exception(home):
    """A mutator that raises must not leave the lock file behind - a failed
    write must never wedge every future writer."""
    cfg.save_config({"n_ctx": 4096})

    def _boom(c):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        cfg.update_config(_boom)

    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    assert not lockpath.exists()
    # A second, well-behaved writer must still succeed immediately afterward.
    result = cfg.update_config(lambda c: c.__setitem__("temperature", 0.5))
    assert result["temperature"] == 0.5


# --------------------------------------------------------------------------- #
#  Fencing-token hardening                                                    #
# --------------------------------------------------------------------------- #

def test_cross_process_lock_release_does_not_steal_a_reclaimed_lock(home, monkeypatch, capsys):
    """If THIS process's hold legitimately outlives _CROSS_LOCK_STALE_AGE (a slow
    write, not a crash) and a waiter reclaims it as stale, this process's own
    release must NOT delete the new holder's still-active lock file - that would
    let a THIRD writer in while the second still believes it holds exclusive
    access. A bare create/delete marker with no per-acquisition token fails
    this: its unconditional `finally: lockpath.unlink()` deletes whatever lock
    is on disk, not specifically the one it created."""
    monkeypatch.setattr(cfg, "_CROSS_LOCK_STALE_AGE", 0.05)
    monkeypatch.setattr(cfg, "_CROSS_LOCK_TIMEOUT", 5.0)
    cfg.ensure_dirs()
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    real_getpid = cfg.os.getpid

    # Two genuinely separate OS processes have different real pids; fake that
    # here so this exercises the stale-reclaim/fencing path rather than the
    # same-process reentrancy guard.
    monkeypatch.setattr(cfg.os, "getpid", lambda: 111111)
    cm_a = cfg._cross_process_lock(cfg.CONFIG_FILE)
    cm_a.__enter__()  # simulated "process A" acquires
    monkeypatch.setattr(cfg.os, "getpid", real_getpid)
    old = cfg.time.time() - 10  # A's hold is now far older than the stale threshold
    os.utime(lockpath, (old, old))

    # A separate OS process has its own pid AND its own in-memory record of the
    # locks it holds (config._held_lock_tokens), so give B an empty record as a
    # process boundary would. Ownership is keyed on the fencing token rather than
    # the pid number, because pids are reused across process lifetimes.
    monkeypatch.setattr(cfg, "_held_lock_tokens", {})
    monkeypatch.setattr(cfg.os, "getpid", lambda: 222222)
    cm_b = cfg._cross_process_lock(cfg.CONFIG_FILE)
    cm_b.__enter__()  # simulated "process B" sees A's lock as stale and reclaims it
    assert "reclaiming stale lock" in capsys.readouterr().err
    b_token = lockpath.read_bytes()
    monkeypatch.setattr(cfg.os, "getpid", real_getpid)

    cm_a.__exit__(None, None, None)  # A finally finishes and releases
    assert lockpath.exists(), (
        "A's release deleted B's live lock - the exact cascading-theft race "
        "the fencing token exists to prevent")
    assert lockpath.read_bytes() == b_token, "B's lock must be untouched by A's release"

    cm_b.__exit__(None, None, None)  # B releases normally afterward
    assert not lockpath.exists()


def test_cross_process_lock_cleans_up_orphan_on_acquire_write_failure(home, monkeypatch):
    """A transient failure WRITING the just-created lock marker (ENOSPC, a
    momentary AV lock on the file we just made) must not leak an orphaned,
    unowned lock file - that would block every other config/registry writer
    install-wide until the next staleness reclaim (up to _CROSS_LOCK_STALE_AGE)."""
    cfg.ensure_dirs()
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")

    def flaky_write(fd, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cfg.os, "write", flaky_write)
    with pytest.raises(OSError):
        with cfg._cross_process_lock(cfg.CONFIG_FILE):
            pass
    assert not lockpath.exists(), (
        "a failed acquire must not leave its just-created lock file behind")


def test_cross_process_lock_nested_same_process_call_fails_fast(home, monkeypatch):
    """A mutator that calls update_config() again on the SAME file from the same
    thread is not supported (this lock, unlike _io_lock, is not reentrant) - it
    must fail immediately with a clear error identifying itself as the holder,
    not silently stall for the whole timeout window and then raise a misleading
    'held by another localm process' TimeoutError."""
    monkeypatch.setattr(cfg, "_CROSS_LOCK_TIMEOUT", 5.0)
    cfg.save_config({"n_ctx": 4096})

    def _outer(c):
        cfg.update_config(lambda c2: c2.__setitem__("main_gpu_index", 1))

    start = cfg.time.time()
    with pytest.raises(RuntimeError, match="already held by this same process"):
        cfg.update_config(_outer)
    elapsed = cfg.time.time() - start
    assert elapsed < 2.0, (
        f"nested call took {elapsed:.1f}s - it must fail immediately, not wait "
        "out the cross-process timeout")


# --------------------------------------------------------------------------- #
#  A writer waiting out the cross-process lock must not hold _io_lock: every  #
#  reader (auth.require_auth -> load_config on every request) takes it.       #
# --------------------------------------------------------------------------- #

def test_reader_is_not_blocked_by_a_writer_waiting_on_the_cross_process_lock(home, monkeypatch):
    """While update_config() sits in the cross-process lock wait (a fresh lock
    file left by another localm process, e.g. a hard-killed `localm pull`),
    load_config() from another thread must return at once. _io_lock is taken
    only around the read-mutate-write itself; the wait holds _rmw_lock, which
    readers never take."""
    cfg.save_config({"n_ctx": 4096})
    monkeypatch.setattr(cfg, "_CROSS_LOCK_STALE_AGE", 3600.0)
    monkeypatch.setattr(cfg, "_CROSS_LOCK_TIMEOUT", 1.5)
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    lockpath.write_text("12345:foreign", encoding="utf-8")

    # Proves the writer is INSIDE the wait loop (not merely started) before the
    # reader is timed: the first backoff call can only happen from that loop.
    in_wait = threading.Event()
    real_backoff = cfg._cross_lock_backoff

    def backoff(attempt):
        in_wait.set()
        real_backoff(attempt)

    monkeypatch.setattr(cfg, "_cross_lock_backoff", backoff)
    outcome = {}

    def writer():
        try:
            cfg.update_config(lambda c: c.__setitem__("temperature", 0.5))
        except BaseException as e:
            outcome["exc"] = e

    t = threading.Thread(target=writer)
    t.start()
    try:
        assert in_wait.wait(5.0), "the writer never entered the cross-process wait"
        start = time.monotonic()
        result = cfg.load_config()
        elapsed = time.monotonic() - start
    finally:
        t.join(10.0)
        try:
            lockpath.unlink()
        except FileNotFoundError:
            pass
    assert result.get("n_ctx") == 4096
    assert elapsed < 0.5, (
        f"load_config() blocked {elapsed:.2f}s behind a writer that was only "
        "WAITING for the cross-process lock - _io_lock was held across the wait")
    assert isinstance(outcome.get("exc"), TimeoutError), (
        f"the writer must time out on the foreign lock, got {outcome.get('exc')!r}")


# --------------------------------------------------------------------------- #
#  Release must retry the unlink: a waiter reading the file pins it (Windows) #
# --------------------------------------------------------------------------- #

def _probe_read_handle_blocks_unlink(tmp_dir):
    """Assert the fault injection can fire on this box: an open read handle on
    a file makes os.unlink raise PermissionError."""
    probe = tmp_dir / "probe.txt"
    probe.write_text("x", encoding="utf-8")
    fh = open(probe, "rb")
    try:
        with pytest.raises(PermissionError):
            probe.unlink()
    finally:
        fh.close()
    probe.unlink()


@pytest.mark.skipif(os.name != "nt", reason="an open handle blocks unlink only on Windows")
def test_release_retries_unlink_while_a_reader_pins_the_lock_file(home):
    """A waiter in another process reads the lock file (read_bytes) at the
    instant the holder releases it. On Windows that handle makes the holder's
    unlink raise a sharing violation; the release must retry until the handle
    goes away, never leave an orphaned lock behind."""
    cfg.ensure_dirs()
    _probe_read_handle_blocks_unlink(cfg.HOME_DIR)
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    fh = None
    try:
        with cfg._cross_process_lock(cfg.CONFIG_FILE):
            fh = open(lockpath, "rb")
            threading.Timer(0.3, fh.close).start()
        assert not lockpath.exists(), (
            "the released lock file survived a 0.3s pinned read handle - the "
            "release gave up on the first sharing violation and orphaned it")
    finally:
        if fh is not None and not fh.closed:
            fh.close()
        try:
            lockpath.unlink()
        except FileNotFoundError:
            pass


@pytest.mark.skipif(os.name != "nt", reason="an open handle blocks unlink only on Windows")
def test_release_reports_a_lock_file_it_could_not_remove(home, caplog):
    """A handle held past the whole retry budget leaves the lock file behind;
    that is reported at WARNING naming the file, never silently."""
    cfg.ensure_dirs()
    _probe_read_handle_blocks_unlink(cfg.HOME_DIR)
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    fh = None
    try:
        with caplog.at_level("WARNING", logger="localm"):
            with cfg._cross_process_lock(cfg.CONFIG_FILE):
                fh = open(lockpath, "rb")
            # The handle is still open here: every retry has failed.
        assert lockpath.exists(), "the injection did not take: the file was removed"
        warnings = [r for r in caplog.records
                    if r.levelname == "WARNING" and lockpath.name in r.getMessage()]
        assert warnings, (
            f"no WARNING named {lockpath.name}; records: "
            f"{[r.getMessage() for r in caplog.records]}")
    finally:
        if fh is not None and not fh.closed:
            fh.close()
        try:
            lockpath.unlink()
        except FileNotFoundError:
            pass


def _fail_lock_read(monkeypatch, lockpath, exc_factory, fails):
    """Make Path.read_bytes on *lockpath* raise ``exc_factory()`` for the first
    *fails* calls (None: always), then behave normally. Returns the call counter."""
    real = type(lockpath).read_bytes
    calls = {"n": 0}

    def read_bytes(self):
        if self == lockpath:
            calls["n"] += 1
            if fails is None or calls["n"] <= fails:
                raise exc_factory()
        return real(self)

    monkeypatch.setattr(type(lockpath), "read_bytes", read_bytes)
    return calls


def test_release_reports_a_lock_file_it_could_not_read(home, monkeypatch, caplog):
    """A release whose read of its own lock file fails leaves the file for
    stale-age reclaim; that is reported at WARNING naming the file, never
    silently."""
    cfg.ensure_dirs()
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    try:
        with caplog.at_level("WARNING", logger="localm"):
            with cfg._cross_process_lock(cfg.CONFIG_FILE):
                calls = _fail_lock_read(
                    monkeypatch, lockpath, lambda: OSError(5, "injected read failure"), None)
            assert calls["n"] >= 1, "the injection did not take: the release never read the lock"
        assert lockpath.exists(), "the injection did not take: the file was removed"
        warnings = [r for r in caplog.records
                    if r.levelname == "WARNING" and lockpath.name in r.getMessage()
                    and "read" in r.getMessage()]
        assert len(warnings) == 1, (
            f"expected exactly one read WARNING naming {lockpath.name}; records: "
            f"{[r.getMessage() for r in caplog.records]}")
    finally:
        monkeypatch.undo()
        try:
            lockpath.unlink()
        except FileNotFoundError:
            pass


def test_release_of_an_already_removed_lock_file_is_silent(home, caplog):
    """A lock file another process already reclaimed is the ordinary case of a
    release finding nothing to read: no warning."""
    cfg.ensure_dirs()
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    with caplog.at_level("WARNING", logger="localm"):
        with cfg._cross_process_lock(cfg.CONFIG_FILE):
            lockpath.unlink()
    assert not [r for r in caplog.records if lockpath.name in r.getMessage()]


@pytest.mark.skipif(os.name != "nt", reason="sharing-violation retry is Windows only")
def test_release_retries_a_transient_lock_read_failure(home, monkeypatch, caplog):
    """A sharing violation on the release's read is ridden out: the lock file is
    removed and nothing is reported."""
    cfg.ensure_dirs()
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")

    def sharing_violation():
        e = PermissionError(13, "injected sharing violation")
        e.winerror = 32
        return e

    with caplog.at_level("WARNING", logger="localm"):
        with cfg._cross_process_lock(cfg.CONFIG_FILE):
            calls = _fail_lock_read(monkeypatch, lockpath, sharing_violation, 1)
    monkeypatch.undo()
    assert calls["n"] >= 2, "the read was not retried"
    assert not lockpath.exists()
    assert not [r for r in caplog.records if lockpath.name in r.getMessage()]


# --------------------------------------------------------------------------- #
#  The stale-lock reclaim must not delete a lock a NEW holder already took     #
# --------------------------------------------------------------------------- #

def _stale(lockpath, token=b"4242:deadbeef", age=10.0):
    lockpath.write_bytes(token)
    t = cfg.time.time() - age
    os.utime(lockpath, (t, t))
    return token, lockpath.stat().st_mtime


def test_a_stale_lock_is_reclaimed_when_nothing_changed(home, monkeypatch):
    monkeypatch.setattr(cfg, "_CROSS_LOCK_STALE_AGE", 0.05)
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    token, mtime = _stale(lockpath)
    assert cfg._reclaim_stale_lock(lockpath, token, mtime) is True
    assert not lockpath.exists()


def test_a_stale_lock_is_not_reclaimed_after_a_new_holder_took_it(home, monkeypatch):
    """Two waiters can both reach the stale branch. The first reclaims and
    creates its own lock; the second must NOT then delete that fresh lock and
    create a second one, which would put both inside the lock at once."""
    monkeypatch.setattr(cfg, "_CROSS_LOCK_STALE_AGE", 0.05)
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    token, mtime = _stale(lockpath)
    # The first waiter reclaimed it and took the lock itself: new bytes, new mtime.
    lockpath.write_bytes(b"777:feedface")
    now = cfg.time.time()
    os.utime(lockpath, (now, now))

    assert cfg._reclaim_stale_lock(lockpath, token, mtime) is False
    assert lockpath.read_bytes() == b"777:feedface", (
        "the stale-lock reclaim deleted a lock a new holder had already taken")


def test_a_stale_lock_whose_bytes_changed_in_place_is_not_reclaimed(home, monkeypatch):
    monkeypatch.setattr(cfg, "_CROSS_LOCK_STALE_AGE", 0.05)
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    token, mtime = _stale(lockpath)
    lockpath.write_bytes(b"777:feedface")
    os.utime(lockpath, (mtime, mtime))
    assert cfg._reclaim_stale_lock(lockpath, token, mtime) is False
    assert lockpath.exists()


def test_an_already_gone_lock_lets_the_caller_retry(home, monkeypatch):
    monkeypatch.setattr(cfg, "_CROSS_LOCK_STALE_AGE", 0.05)
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    token, mtime = _stale(lockpath)
    lockpath.unlink()
    assert cfg._reclaim_stale_lock(lockpath, token, mtime) is True


def test_an_unreadable_stale_lock_is_still_reclaimed(home, monkeypatch):
    """A stale lock whose bytes cannot be read must not wedge every writer: the
    mtime test alone decides."""
    monkeypatch.setattr(cfg, "_CROSS_LOCK_STALE_AGE", 0.05)
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    token, mtime = _stale(lockpath)
    real_read = type(lockpath).read_bytes

    def refuse(self):
        if self == lockpath:
            raise PermissionError("held open by another handle")
        return real_read(self)

    monkeypatch.setattr(type(lockpath), "read_bytes", refuse)
    assert cfg._reclaim_stale_lock(lockpath, token, mtime) is True
    assert not lockpath.exists()


def test_a_stale_lock_touched_since_it_was_judged_is_not_reclaimed(home, monkeypatch):
    """The mtime check alone must decide this one. The bytes are unchanged and
    the file is still older than the staleness age, so neither of the other two
    checks can refuse it; only the timestamp differs from the one it was judged
    by, which means the holder wrote to it after the judgement."""
    monkeypatch.setattr(cfg, "_CROSS_LOCK_STALE_AGE", 0.05)
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    token, mtime = _stale(lockpath, age=10.0)
    moved = cfg.time.time() - 5.0
    os.utime(lockpath, (moved, moved))
    assert cfg.time.time() - moved > cfg._CROSS_LOCK_STALE_AGE
    assert cfg._reclaim_stale_lock(lockpath, token, mtime) is False
    assert lockpath.read_bytes() == token


def test_the_acquirer_waits_when_the_stale_reclaim_refuses(home, monkeypatch):
    """The call site must go THROUGH the reclaim and respect its answer. When the
    reclaim refuses (the lock changed under it), the acquirer waits and times out
    rather than deleting the file and taking the lock anyway."""
    monkeypatch.setattr(cfg, "_CROSS_LOCK_STALE_AGE", 0.05)
    monkeypatch.setattr(cfg, "_CROSS_LOCK_TIMEOUT", 0.6)
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    token, _ = _stale(lockpath)
    calls = []

    def refuse(path, judged, judged_mtime):
        calls.append(path)
        return False

    monkeypatch.setattr(cfg, "_reclaim_stale_lock", refuse)
    with pytest.raises(TimeoutError):
        with cfg._cross_process_lock(cfg.CONFIG_FILE):
            pass
    assert calls, "the acquirer never consulted the stale-lock reclaim"
    assert lockpath.read_bytes() == token, (
        "the acquirer deleted a lock the reclaim had refused to remove")


_HF_PIN = "LOCALM_PINNED_HF_HOME"


def _token_path(env):
    return cfg._default_hf_token_path(env)


class TestUserCommandEnvHuggingFace:
    def test_a_pinned_home_and_the_default_login_location_are_dropped_with_the_marker(self):
        env = {_HF_PIN: "/pin", "HF_HOME": "/pin", "KEEP": "1"}
        env["HF_TOKEN_PATH"] = _token_path(env)
        assert cfg.user_command_env(env) == {"KEEP": "1"}

    def test_a_home_the_user_chose_is_kept_and_the_marker_is_dropped(self):
        env = {_HF_PIN: "/pin", "HF_HOME": "/mine", "HF_TOKEN_PATH": "/t"}
        assert cfg.user_command_env(env) == {"HF_HOME": "/mine", "HF_TOKEN_PATH": "/t"}

    def test_a_login_location_the_user_chose_is_kept_when_the_home_is_dropped(self):
        env = {_HF_PIN: "/pin", "HF_HOME": "/pin", "HF_TOKEN_PATH": "/my/token"}
        assert cfg.user_command_env(env) == {"HF_TOKEN_PATH": "/my/token"}

    def test_a_home_without_a_marker_is_the_users_own_and_is_kept(self):
        env = {"HF_HOME": "/mine", "HF_TOKEN_PATH": _token_path({})}
        assert cfg.user_command_env(env) == env

    def test_a_marker_without_a_home_leaves_nothing_behind(self):
        assert cfg.user_command_env({_HF_PIN: "/pin", "KEEP": "1"}) == {"KEEP": "1"}

    def test_an_environment_with_none_of_the_variables_is_returned_unchanged(self):
        env = {"KEEP": "1"}
        assert cfg.user_command_env(env) == {"KEEP": "1"}

    def test_the_login_location_follows_the_xdg_cache_of_the_environment(self):
        env = {_HF_PIN: "/pin", "HF_HOME": "/pin", "XDG_CACHE_HOME": "/xdg"}
        env["HF_TOKEN_PATH"] = _token_path(env)
        assert env["HF_TOKEN_PATH"] == os.path.join("/xdg", "huggingface", "token")
        assert cfg.user_command_env(env) == {"XDG_CACHE_HOME": "/xdg"}

    def test_the_callers_dict_is_not_modified(self):
        env = {_HF_PIN: "/pin", "HF_HOME": "/pin"}
        cfg.user_command_env(env)
        assert env == {_HF_PIN: "/pin", "HF_HOME": "/pin"}


@pytest.mark.parametrize("var", [v for v, _ in cfg._GPU_CACHE_PINS])
class TestUserCommandEnvGpuCaches:
    def test_a_pinned_cache_variable_is_dropped_with_its_marker(self, var):
        marker = cfg._gpu_pin_marker(var)
        env = {var: "/pin", marker: "/pin", "KEEP": "1"}
        assert cfg.user_command_env(env) == {"KEEP": "1"}

    def test_a_value_the_user_chose_is_kept_and_the_marker_is_dropped(self, var):
        marker = cfg._gpu_pin_marker(var)
        env = {var: "/mine", marker: "/pin"}
        assert cfg.user_command_env(env) == {var: "/mine"}

    def test_a_variable_without_a_marker_is_the_users_own_and_is_kept(self, var):
        assert cfg.user_command_env({var: "/mine"}) == {var: "/mine"}

    def test_a_marker_without_the_variable_is_dropped(self, var):
        assert cfg.user_command_env({cfg._gpu_pin_marker(var): "/pin"}) == {}

    def test_an_absent_variable_without_a_marker_is_left_absent(self, var):
        assert cfg.user_command_env({"KEEP": "1"}) == {"KEEP": "1"}


class TestReclaimStaleLockAge:
    def _lock(self, tmp_path, age):
        lockpath = tmp_path / "x.lock"
        lockpath.write_bytes(b"1:tok")
        stamp = lockpath.stat().st_mtime - age
        os.utime(lockpath, (stamp, stamp))
        return lockpath, b"1:tok", lockpath.stat().st_mtime

    def test_a_lock_younger_than_the_stale_age_is_left_in_place(self, tmp_path):
        lockpath, token, mtime = self._lock(tmp_path, age=0)
        assert cfg._reclaim_stale_lock(lockpath, token, mtime) is False
        assert lockpath.exists()

    def test_a_lock_older_than_the_stale_age_is_removed(self, tmp_path):
        lockpath, token, mtime = self._lock(tmp_path, age=cfg._CROSS_LOCK_STALE_AGE + 60)
        assert cfg._reclaim_stale_lock(lockpath, token, mtime) is True
        assert not lockpath.exists()

    def test_a_lock_exactly_as_old_as_the_stale_age_is_not_yet_stale(
            self, tmp_path, monkeypatch):
        lockpath, token, mtime = self._lock(tmp_path, age=0)
        monkeypatch.setattr(cfg, "time", types.SimpleNamespace(
            time=lambda: mtime + cfg._CROSS_LOCK_STALE_AGE))
        assert cfg._reclaim_stale_lock(lockpath, token, mtime) is False
        assert lockpath.exists()

    def test_a_lock_just_past_the_stale_age_is_removed(self, tmp_path, monkeypatch):
        lockpath, token, mtime = self._lock(tmp_path, age=0)
        monkeypatch.setattr(cfg, "time", types.SimpleNamespace(
            time=lambda: mtime + cfg._CROSS_LOCK_STALE_AGE + 0.001))
        assert cfg._reclaim_stale_lock(lockpath, token, mtime) is True
        assert not lockpath.exists()


# --------------------------------------------------------------------------- #
#  Retry policy of the lock-file read, judged as each platform would          #
# --------------------------------------------------------------------------- #

class _OsAs:
    """``config.os`` as seen on another platform: ``name`` replaced, the rest
    from the real module."""

    def __init__(self, name, **overrides):
        self.name = name
        self.__dict__.update(overrides)

    def __getattr__(self, attr):
        return getattr(os, attr)


def _denial(winerror=None):
    e = PermissionError(13, "injected denial")
    if winerror is not None:
        e.winerror = winerror
    return e


def _no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(cfg, "time", types.SimpleNamespace(
        sleep=sleeps.append, time=time.time, monotonic=time.monotonic))
    return sleeps


_RELEASE_READ_WARNING = (
    "could not read lock file %s on release (%s); leaving it in "
    "place, so config/registry writes in every localm process "
    "fail on it until it is %.0fs old and reclaimed")


def test_lock_read_on_posix_gives_up_at_once_and_reports_it(tmp_path, monkeypatch, caplog):
    lockpath = tmp_path / "config.json.lock"
    lockpath.write_bytes(b"1:aa")
    monkeypatch.setattr(cfg, "os", _OsAs("posix"))
    sleeps = _no_sleep(monkeypatch)
    calls = _fail_lock_read(monkeypatch, lockpath, _denial, None)
    with caplog.at_level("WARNING", logger="localm"):
        assert cfg._read_lock_for_release(lockpath) is None
    assert calls["n"] == 1 and sleeps == []
    [r] = [r for r in caplog.records if "lock file" in str(r.msg)]
    assert r.msg == _RELEASE_READ_WARNING
    assert r.args[0] == "config.json.lock"
    assert isinstance(r.args[1], PermissionError)
    assert r.args[2] == cfg._CROSS_LOCK_STALE_AGE


def test_lock_read_on_windows_rides_out_a_sharing_violation(tmp_path, monkeypatch):
    lockpath = tmp_path / "config.json.lock"
    lockpath.write_bytes(b"1:aa")
    monkeypatch.setattr(cfg, "os", _OsAs("nt"))
    sleeps = _no_sleep(monkeypatch)
    _fail_lock_read(monkeypatch, lockpath, lambda: _denial(32), cfg._REPLACE_RETRIES - 1)
    assert cfg._read_lock_for_release(lockpath) == b"1:aa"
    assert len(sleeps) == cfg._REPLACE_RETRIES - 1


def test_lock_read_on_windows_stops_after_its_retry_budget(tmp_path, monkeypatch):
    lockpath = tmp_path / "config.json.lock"
    lockpath.write_bytes(b"1:aa")
    monkeypatch.setattr(cfg, "os", _OsAs("nt"))
    sleeps = _no_sleep(monkeypatch)
    calls = _fail_lock_read(monkeypatch, lockpath, lambda: _denial(32), None)
    assert cfg._read_lock_for_release(lockpath) is None
    assert calls["n"] == cfg._REPLACE_RETRIES
    assert len(sleeps) == cfg._REPLACE_RETRIES - 1


def test_lock_read_on_windows_does_not_retry_a_stable_denial(tmp_path, monkeypatch):
    lockpath = tmp_path / "config.json.lock"
    lockpath.write_bytes(b"1:aa")
    monkeypatch.setattr(cfg, "os", _OsAs("nt"))
    sleeps = _no_sleep(monkeypatch)
    calls = _fail_lock_read(monkeypatch, lockpath, lambda: _denial(1314), None)
    assert cfg._read_lock_for_release(lockpath) is None
    assert calls["n"] == 1 and sleeps == []


def test_replace_on_windows_does_not_retry_a_stable_denial(tmp_path, monkeypatch):
    calls = []

    def replace(src, dst):
        calls.append((src, dst))
        raise _denial(1314)
    monkeypatch.setattr(cfg, "os", _OsAs("nt", replace=replace))
    sleeps = _no_sleep(monkeypatch)
    with pytest.raises(PermissionError):
        cfg._replace_atomic(tmp_path / "a", tmp_path / "b")
    assert len(calls) == 1 and sleeps == []
