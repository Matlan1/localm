# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two properties of config.py's locking/retry paths.

_cross_process_lock must check STALENESS before "is this lock's recorded pid MY
pid?". PID reuse across process lifetimes is routine, so a lock LEAKED by a
crashed process whose pid the OS later hands to a new localm process must still
be reclaimed rather than read as "already held by this same process (nested
call)".

_read_json / _replace_atomic retry a PermissionError _REPLACE_RETRIES times
(~1s of time.sleep) while holding _io_lock. That retry is for the TRANSIENT
Windows sharing violation (a concurrent atomic replace, antivirus, the
indexer). On POSIX, EACCES from open() is a stable state, so it must not be
retried - load_config() sits on the per-request auth path (auth.require_auth).

Neither is covered by tests/test_config_cross_process_lock.py: it only ever
leaks a lock whose pid differs from the reclaiming process's, and it only uses
readable temp files (plus a Windows-only real file lock the retry is SUPPOSED to
ride out).
"""

import asyncio
import builtins
import os

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


def _leak_stale_lock(lockpath, pid, age_s):
    """Write a lock file as a CRASHED holder would leave it: a valid
    ``<pid>:<nonce>`` token on disk with nobody holding it, aged *age_s*."""
    lockpath.write_bytes(f"{pid}:deadbeefdeadbeefdeadbeefdeadbeef".encode("ascii"))
    old = cfg.time.time() - age_s
    os.utime(lockpath, (old, old))


# --------------------------------------------------------------------------- #
#  Stale lock whose recorded pid collides with the CURRENT pid                 #
# --------------------------------------------------------------------------- #

def test_stale_lock_with_colliding_self_pid_is_reclaimed(home, capsys):
    """A crashed process leaked CONFIG.lock holding pid N; the OS later gave pid
    N to THIS process. The lock is long dead (older than
    _CROSS_LOCK_STALE_AGE) and must be reclaimed - a pid NUMBER matching ours is
    not evidence that WE hold it."""
    cfg.ensure_dirs()
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    _leak_stale_lock(lockpath, os.getpid(), cfg._CROSS_LOCK_STALE_AGE + 10)

    cfg.update_config(lambda c: c.__setitem__("n_ctx", 8192))

    assert cfg.load_config()["n_ctx"] == 8192, "the config write must go through"
    assert "reclaiming stale lock" in capsys.readouterr().err
    assert not lockpath.exists(), "the reclaimed lock must be released again"


def test_stale_lock_with_colliding_self_pid_does_not_wedge_the_registry(home, capsys):
    """The same leak on REGISTRY.lock must not wedge model pull / rm / alias."""
    cfg.ensure_dirs()
    lockpath = cfg.REGISTRY_FILE.with_name(cfg.REGISTRY_FILE.name + ".lock")
    _leak_stale_lock(lockpath, os.getpid(), cfg._CROSS_LOCK_STALE_AGE + 10)

    cfg.update_registry(lambda r: r.__setitem__("m", {"path": "x.gguf"}))

    assert "m" in cfg.load_registry()
    capsys.readouterr()


def test_fresh_foreign_lock_with_colliding_self_pid_waits_and_times_out(home, monkeypatch):
    """A pid collision on a lock we do NOT hold must not be mistaken for a
    nested call: it is a foreign lock, so the outcome is the normal
    wait-then-TimeoutError ('held by another localm process'), never an
    immediate RuntimeError. This lock is FRESH, so reordering staleness ahead of
    the pid check does not satisfy it either."""
    monkeypatch.setattr(cfg, "_CROSS_LOCK_TIMEOUT", 0.3)
    cfg.ensure_dirs()
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    _leak_stale_lock(lockpath, os.getpid(), 0)   # fresh: someone else holds it NOW

    with pytest.raises(TimeoutError, match="held by another localm process"):
        cfg.update_config(lambda c: c.__setitem__("n_ctx", 8192))


def test_genuine_nested_call_still_fails_fast(home, monkeypatch):
    """The nested-call guard still fires. A mutator that calls back into
    update_config() on the same file is unsupported and fails immediately with
    the clear error, rather than stalling for the timeout or reclaiming its own
    live lock."""
    monkeypatch.setattr(cfg, "_CROSS_LOCK_TIMEOUT", 5.0)
    cfg.save_config({"n_ctx": 4096})

    def _outer(c):
        cfg.update_config(lambda c2: c2.__setitem__("main_gpu_index", 1))

    start = cfg.time.time()
    with pytest.raises(RuntimeError, match="already held by this same process"):
        cfg.update_config(_outer)
    assert cfg.time.time() - start < 2.0, "must fail fast, not wait out the timeout"


def test_nested_guard_survives_an_outer_hold_longer_than_the_stale_age(home, monkeypatch):
    """A nested call is identified as OURS by the fencing token we actually
    hold, not by the lock's age, so the guard still fires when the outer
    critical section has outlived _CROSS_LOCK_STALE_AGE (a slow write under
    antivirus). A staleness-only check would let the inner call reclaim its own
    outer lock and proceed."""
    monkeypatch.setattr(cfg, "_CROSS_LOCK_STALE_AGE", 0.05)
    monkeypatch.setattr(cfg, "_CROSS_LOCK_TIMEOUT", 5.0)
    cfg.ensure_dirs()
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")

    with cfg._cross_process_lock(cfg.CONFIG_FILE):
        old = cfg.time.time() - 10          # our OWN live hold is now "stale"-aged
        os.utime(lockpath, (old, old))
        with pytest.raises(RuntimeError, match="already held by this same process"):
            with cfg._cross_process_lock(cfg.CONFIG_FILE):
                pass


def test_released_lock_is_forgotten_so_a_later_leak_is_not_read_as_ours(home, capsys):
    """The in-process record of what we hold must be dropped on release, or a
    later lock file carrying OUR pid would still be read as a live nested hold."""
    cfg.ensure_dirs()
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")

    with cfg._cross_process_lock(cfg.CONFIG_FILE):
        pass
    _leak_stale_lock(lockpath, os.getpid(), cfg._CROSS_LOCK_STALE_AGE + 10)

    cfg.update_config(lambda c: c.__setitem__("n_ctx", 2048))
    assert cfg.load_config()["n_ctx"] == 2048
    capsys.readouterr()


# --------------------------------------------------------------------------- #
#  PermissionError retry is for the TRANSIENT case only                        #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def no_sleep(monkeypatch):
    """Record every backoff instead of actually stalling the test ~1s."""
    slept = []
    monkeypatch.setattr(cfg.time, "sleep", lambda s: slept.append(s))
    return slept


def _deny_reads(monkeypatch, target, err):
    """Make open(*target*) raise *err*. Injected rather than produced for real
    because a POSIX EACCES cannot be created on the Windows box this suite runs
    on (and a real mode-000 file is not readable back for cleanup); the code
    under test branches on the exception class, which is what this reproduces."""
    attempts = []
    real_open = builtins.open

    def fake_open(path, *a, **kw):
        if str(path) == str(target):
            attempts.append(1)
            raise err
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", fake_open)
    return attempts


def test_posix_permission_denied_falls_back_immediately(home, monkeypatch, no_sleep, capsys):
    """THE REGRESSION. On POSIX, EACCES from open() is a stable state - the file
    is mode 000 or root-owned. Retrying it 16 times burns ~1s of time.sleep while
    holding _io_lock (serializing config access process-wide, including the
    per-request auth path) and cannot ever succeed. Fall back at once instead."""
    monkeypatch.setattr(cfg.os, "name", "posix")
    target = home / "config.json"
    target.write_text("{}", encoding="utf-8")
    attempts = _deny_reads(monkeypatch, target, PermissionError(13, "Permission denied"))

    assert cfg._read_json(target, {"fell_back": True}) == {"fell_back": True}
    assert len(attempts) == 1, f"retried a stable EACCES {len(attempts)} times"
    assert no_sleep == [], f"stalled {sum(no_sleep):.2f}s under _io_lock for nothing"
    assert "unreadable" in capsys.readouterr().err, "the failure must still be surfaced"


def test_windows_transient_sharing_violation_is_still_ridden_out(home, monkeypatch, no_sleep):
    """The retry still fires. On Windows a concurrent atomic replace /
    antivirus / the indexer holds the file for microseconds and raises
    PermissionError; that IS transient and must be retried, or a passing scanner
    makes live settings be discarded."""
    monkeypatch.setattr(cfg.os, "name", "nt")
    target = home / "config.json"
    target.write_text("{}", encoding="utf-8")
    err = PermissionError(13, "Permission denied")
    err.winerror = 32                       # ERROR_SHARING_VIOLATION
    attempts = _deny_reads(monkeypatch, target, err)

    cfg._read_json(target, {})
    assert len(attempts) == cfg._REPLACE_RETRIES, "the transient retry was lost"
    assert no_sleep, "the transient retry must back off between attempts"


def test_windows_transient_retry_recovers_when_the_lock_clears(home, monkeypatch, no_sleep):
    """NEGATIVE CASE, positive direction: the retry must actually RETURN the file
    once the transient lock clears, not merely spin and fall back."""
    monkeypatch.setattr(cfg.os, "name", "nt")
    target = home / "config.json"
    target.write_text('{"n_ctx": 4096}', encoding="utf-8")
    real_open = builtins.open
    calls = []

    def flaky_open(path, *a, **kw):
        if str(path) == str(target) and len(calls) < 3:
            calls.append(1)
            err = PermissionError(13, "Permission denied")
            err.winerror = 32
            raise err
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", flaky_open)
    assert cfg._read_json(target, {}) == {"n_ctx": 4096}


def test_posix_replace_permission_error_is_not_retried(home, monkeypatch, no_sleep):
    """_replace_atomic has the same defect on the WRITE path, where the stall is
    held under _io_lock AND the cross-process lock. A POSIX permission failure on
    os.replace is stable; surface it at once."""
    monkeypatch.setattr(cfg.os, "name", "posix")
    attempts = []

    def denied(src, dst):
        attempts.append(1)
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cfg.os, "replace", denied)
    with pytest.raises(PermissionError):
        cfg._replace_atomic(home / "a", home / "b")
    assert len(attempts) == 1, f"retried a stable EACCES {len(attempts)} times"
    assert no_sleep == []


def test_windows_replace_sharing_violation_is_still_retried(home, monkeypatch, no_sleep):
    """NEGATIVE CASE: the Windows write-path retry (WinError 5 under an AV lock,
    WinError 32 under a concurrent reader) must survive - it is the documented
    reason _replace_atomic has a retry at all."""
    monkeypatch.setattr(cfg.os, "name", "nt")
    attempts = []

    def denied(src, dst):
        attempts.append(1)
        err = PermissionError(13, "Permission denied")
        err.winerror = 5
        raise err

    monkeypatch.setattr(cfg.os, "replace", denied)
    with pytest.raises(PermissionError):
        cfg._replace_atomic(home / "a", home / "b")
    assert len(attempts) == cfg._REPLACE_RETRIES, "the transient retry was lost"
    assert no_sleep


# --------------------------------------------------------------------------- #
#  The async config routes must not block the event loop                       #
# --------------------------------------------------------------------------- #

def _config_endpoint(path, method):
    from fastapi import FastAPI

    import localm.inference.routes.config as config_routes

    class _Ctx:
        class mode:
            value = "standard"

    app = FastAPI()
    config_routes.register(app, _Ctx())
    return next(r for r in app.routes
                if getattr(r, "path", None) == path
                and method in getattr(r, "methods", set())).endpoint


class _FakeRequest:
    pass


async def _ticks_while_running(coro):
    """Drive *coro* as a task and count 10ms heartbeat ticks that land while it
    is still pending. The ticks can only advance if the EVENT LOOP is free, so
    this measures loop starvation directly rather than asserting on a mock."""
    task = asyncio.ensure_future(coro)
    ticks = 0
    while not task.done() and ticks < 300:
        await asyncio.sleep(0.01)
        ticks += 1
    with pytest.raises(TimeoutError):
        await task            # the foreign lock is never released -> it times out
    return ticks


@pytest.mark.anyio
async def test_patch_config_does_not_freeze_the_event_loop(home, monkeypatch):
    """patch_config is `async def`. A lock held by ANOTHER localm process (the
    user running `localm config ...` while the GUI is up) must not park the
    server event loop inside _cross_lock_backoff's time.sleep for up to
    _CROSS_LOCK_TIMEOUT. A blocked loop ticks 0 times for the whole wait."""
    monkeypatch.setattr(cfg, "_CROSS_LOCK_TIMEOUT", 0.4)
    cfg.save_config({"n_ctx": 4096})
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    _leak_stale_lock(lockpath, 999999, 0)      # another process holds it, right now

    patch_config = _config_endpoint("/v1/config", "PATCH")
    ticks = await _ticks_while_running(
        patch_config({"temperature": 0.9}, _FakeRequest()))

    assert ticks >= 5, (
        f"the event loop only advanced {ticks} times while a contended "
        "update_config() ran - it was blocked in time.sleep on the loop thread")


@pytest.mark.anyio
async def test_set_media_config_does_not_freeze_the_event_loop(home, monkeypatch):
    """The same defect 68 lines below patch_config, on the same blocking call."""
    monkeypatch.setattr(cfg, "_CROSS_LOCK_TIMEOUT", 0.4)
    cfg.save_config({"n_ctx": 4096})
    lockpath = cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".lock")
    _leak_stale_lock(lockpath, 999999, 0)

    set_media_config = _config_endpoint("/v1/media/config/{name}", "POST")
    ticks = await _ticks_while_running(
        set_media_config("image", {"delete_outputs": True}, _FakeRequest()))

    assert ticks >= 5, (
        f"the event loop only advanced {ticks} times while a contended "
        "update_config() ran - it was blocked in time.sleep on the loop thread")
