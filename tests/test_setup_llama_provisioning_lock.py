# SPDX-License-Identifier: AGPL-3.0-or-later
"""_provisioning_lock: cross-process single-flight around setup-llama's own
provisioning steps. The GUI's standalone runtime-update button is a SECOND
trigger onto the same directory that a `localm update` re-provision or a user's
own `setup-llama` invocation can already be mutating.

Cross-process atomicity is the actual claim, so the load-bearing test spawns a
REAL second interpreter rather than mocking pid_alive - a unit test that only
monkeypatches the liveness check cannot demonstrate that mkdir is atomic across
two processes, only that the Python-level logic branches correctly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from localm import setup_llama as sl

# The holder subprocess is a script run by path, which does NOT get cwd inserted
# onto sys.path (only -m and -c do), so PYTHONPATH is forced explicitly to make
# it import the SAME localm tree this test process resolved `sl` from.
_REPO_ROOT = str(Path(sl.__file__).resolve().parents[1])

# A fixed, argv-driven holder script (no string interpolation into code: every
# value crosses the process boundary as a plain argv element, never templated
# into source text). Takes <target-dir> <marker-file> <hold-seconds>, acquires
# the lock, writes the marker to signal "I am holding it now", sleeps, exits.
_HOLDER_SCRIPT = (Path(__file__).parent / "_setup_llama_lock_holder.py")


def test_lock_round_trips_cleanly(tmp_path):
    """The ordinary case: acquire, do work, release - no lock dir left behind."""
    target = tmp_path / "rt"
    target.mkdir()
    lock = sl._provision_lock_path(target)
    assert not lock.exists()
    with sl._provisioning_lock(target):
        assert lock.is_dir()
        assert json.loads((lock / sl._PROVISION_LOCK_OWNER).read_text())["pid"] == os.getpid()
    assert not lock.exists(), "the lock must be released once the with-block exits"


def test_lock_released_even_when_the_body_raises(tmp_path):
    """finally releases the lock on ANY exit, not only a clean one - a leaked
    lock from an ordinary exception would wedge every later provision until
    someone deletes a folder by hand."""
    target = tmp_path / "rt"
    target.mkdir()
    with pytest.raises(ValueError):
        with sl._provisioning_lock(target):
            raise ValueError("boom")
    assert not sl._provision_lock_path(target).exists()


def test_lock_released_even_on_sys_exit(tmp_path):
    """main()'s body calls sys.exit() liberally (RuntimeInUseError, a bad
    archive, ...). SystemExit must not leak the lock either, or a legitimate
    refusal inside the locked section would wedge the NEXT run."""
    target = tmp_path / "rt"
    target.mkdir()
    with pytest.raises(SystemExit):
        with sl._provisioning_lock(target):
            sys.exit(1)
    assert not sl._provision_lock_path(target).exists()


def test_busy_lock_with_a_live_holder_refuses_fast_and_names_the_pid(tmp_path):
    """A lock recording THIS test process's own pid (verifiably alive) must
    refuse rather than block or steal - and the refusal names the pid so a
    stuck user knows what to look at."""
    target = tmp_path / "rt"
    target.mkdir()
    lock = sl._provision_lock_path(target)
    lock.mkdir()
    (lock / sl._PROVISION_LOCK_OWNER).write_text(json.dumps({"pid": os.getpid()}))

    with pytest.raises(sl.ProvisioningBusyError) as exc:
        with sl._provisioning_lock(target):
            pytest.fail("must never enter the body while busy")
    assert str(os.getpid()) in exc.value.reason
    # Refused, not stolen: the live holder's lock (and its content) survive.
    assert lock.is_dir()


def test_busy_lock_with_unreadable_owner_refuses_without_stealing(tmp_path):
    """No pid recorded at all (an older-format lock, or a crash in the mkdir-
    then-write-owner window) must NOT be treated as free - stealing here is
    exactly how two provisions end up interleaved."""
    target = tmp_path / "rt"
    target.mkdir()
    lock = sl._provision_lock_path(target)
    lock.mkdir()   # no owner.json written

    with pytest.raises(sl.ProvisioningBusyError):
        with sl._provisioning_lock(target):
            pytest.fail("must never enter the body when the owner cannot be read")
    assert lock.is_dir(), "an unreadable lock must be left in place, not deleted"


def test_stale_lock_from_a_dead_pid_is_reclaimed(tmp_path, monkeypatch):
    """A lock naming a pid that is PROVABLY gone is self-healing: the next
    caller reclaims it and proceeds, rather than being wedged forever by a
    process that crashed without releasing."""
    target = tmp_path / "rt"
    target.mkdir()
    lock = sl._provision_lock_path(target)
    lock.mkdir()
    dead_pid = 99999999   # not a real pid on any sane box
    (lock / sl._PROVISION_LOCK_OWNER).write_text(json.dumps({"pid": dead_pid}))
    monkeypatch.setattr("localm.instances.pid_alive", lambda pid: pid != dead_pid)

    with sl._provisioning_lock(target):
        # Reclaimed and re-acquired under OUR pid, not left as the dead one.
        assert json.loads((lock / sl._PROVISION_LOCK_OWNER).read_text())["pid"] == os.getpid()
    assert not lock.exists()


def test_provisioning_busy_error_exits_non_zero_and_says_why(capsys):
    with pytest.raises(SystemExit) as exc:
        sl._exit_provisioning_busy(sl.ProvisioningBusyError("Another setup-llama run is busy."))
    assert exc.value.code != 0
    assert "Another setup-llama run is busy." in capsys.readouterr().out


# --------------------------------------------------------------------------- #
#  Cross-process proof: two REAL interpreters, not a mocked liveness check.    #
# --------------------------------------------------------------------------- #

def test_lock_actually_serializes_across_two_real_processes(tmp_path):
    """The load-bearing test. A first, real subprocess takes the lock and
    holds it briefly; a second, concurrent attempt in THIS process must be
    refused immediately (not after waiting out the hold) - proving mkdir's
    atomicity is doing the work, not merely the Python-level branch logic a
    same-process mock would exercise."""
    target = tmp_path / "rt"
    target.mkdir()
    marker = tmp_path / "holding.marker"
    hold_s = "3.0"

    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT
    holder = subprocess.Popen(
        [sys.executable, str(_HOLDER_SCRIPT), str(target), str(marker), hold_s],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        deadline = time.time() + 15.0
        while not marker.exists():
            if holder.poll() is not None:
                pytest.fail(f"holder process exited early: {holder.stdout.read()}")
            if time.time() > deadline:
                pytest.fail("holder never reported taking the lock")
            time.sleep(0.05)

        started = time.time()
        with pytest.raises(sl.ProvisioningBusyError):
            with sl._provisioning_lock(target):
                pytest.fail("a second, concurrent acquire must never enter the body")
        elapsed = time.time() - started
        assert elapsed < 1.0, (
            f"the refusal took {elapsed:.2f}s - it must fail FAST, not block "
            "for anywhere near the holder's hold time")
    finally:
        holder.wait(timeout=15)
    assert holder.returncode == 0, holder.stdout.read()
    assert not sl._provision_lock_path(target).exists(), (
        "the holder must have released the lock on its own clean exit")
