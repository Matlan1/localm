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
