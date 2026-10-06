# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two pulls of the same URL must not interleave into one .part file.

The destination and its ``.part`` are derived from the URL, so two pulls of the
same URL target one file. Each reads the ``.part``'s current size to decide
append-or-truncate, and with no lock the second one reads a size the first is
still changing. They then write into the same handle: the download "succeeds"
and fails its hash, or - with no ``--sha256`` to check it against - registers
as a working model that is not one.

THE CONTENDERS ARE PROCESSES, NOT THREADS. The GUI starts a pull by spawning
``localm pull`` as a child, and a user can run the same command in a terminal
at the same time, so a ``threading.Lock`` would serialise nothing. The central
test here therefore drives TWO REAL INTERPRETERS: a monkeypatched liveness
check cannot demonstrate atomicity across processes, which is the whole claim.

STALENESS IS DECIDED BY PID LIVENESS AND THE HOLDER'S START IDENTITY, NEVER BY
ELAPSED TIME OR THE WALL CLOCK. Any fixed timeout eventually reclaims a live
holder's lock, and a large model on a slow link is exactly the download that
outlives a generous one. A clock step moves the time, the boot time and, on
Linux, psutil's create_time of a process already running, so none of them can
tell a live holder from a replaced one. Every uncertainty KEEPS the lock, and
the tests below pin both directions: a live holder is never evicted, a
proven-dead or replaced one always is.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

import localm.config as config
import localm.model_manager as model_manager
from localm.model_manager.pull import (
    PullInFlight,
    _part_lock,
    _part_lock_dir,
    _part_lock_holder_is_gone,
)
from tests._process_identity import (
    a_forward_step_past_boot,
    spawn_on_this_tree,
    start_identity_of,
    started_an_hour_earlier,
    step_the_clock,
    this_pid_space,
)

ANOTHER_PID_SPACE = "0123456789abcdef"


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    (h / "models").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(model_manager, "MODELS_DIR", h / "models")
    monkeypatch.setattr(config, "HOME_DIR", h)
    monkeypatch.setattr(config, "MODELS_DIR", h / "models")
    monkeypatch.setattr(config, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(config, "REGISTRY_FILE", h / "registry.json")
    return h


# --------------------------------------------------------------------------
#  The claim: atomicity across real processes
# --------------------------------------------------------------------------

CONTEND = '''
    import sys, time
    from localm.model_manager.pull import _part_lock, PullInFlight
    try:
        with _part_lock(sys.argv[1]):
            # Hold it long enough that the sibling is certainly contending for
            # a HELD lock rather than arriving after it was released.
            print("WON", flush=True)
            time.sleep(float(sys.argv[2]))
    except PullInFlight as e:
        print("LOST", flush=True)
'''

HOLD = '''
    import os, sys
    from localm.model_manager.pull import _part_lock
    with _part_lock(sys.argv[1]):
        print("HELD", os.getpid(), flush=True)
        sys.stdin.read()
'''

REPORT = '''
    import json, os, sys
    from localm.instances import process_start_identity
    from localm.model_manager.pull import _pid_space_id
    print(os.getpid(), _pid_space_id(),
          json.dumps(process_start_identity(os.getpid())), flush=True)
    sys.stdin.read()
'''


def _spawn(script: str, home_dir, *args):
    return spawn_on_this_tree(script, home_dir, *args)


def _hold(home_dir, filename: str = "m.gguf", prefix=()):
    """A real process holding the lock on *filename* until its stdin closes,
    started through the command *prefix* when one is given.

    Returns the process and the holder's own pid, which on Windows differs from
    ``Popen.pid`` when ``sys.executable`` is a venv launcher.
    """
    p = spawn_on_this_tree(HOLD, home_dir, filename, stdin=subprocess.PIPE,
                           prefix=prefix)
    first = p.stdout.readline().split()
    if first[:1] != ["HELD"]:
        _release(p)
        pytest.fail(f"the holder did not take the lock: {first} "
                    f"{p.stderr.read()}")
    return p, int(first[1])


def _idle_child():
    """A live process that exits when its stdin closes."""
    return subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE)


def _release(p) -> None:
    p.stdin.close()
    p.wait(timeout=60)


def _write_owner(d, pid, **fields) -> None:
    """Write a lock record for *pid*, in this process's pid space unless
    *fields* names a ``space``."""
    if "space" not in fields:
        fields["space"] = this_pid_space()
    d.mkdir(parents=True)
    (d / "owner.json").write_text(
        json.dumps({"pid": pid, "filename": "m.gguf", **fields}),
        encoding="utf-8")


def _record(d):
    """The lock's owner record as text, or None when it does not exist."""
    f = d / "owner.json"
    return f.read_text(encoding="utf-8") if f.exists() else None


def test_two_real_interpreters_cannot_both_hold_the_lock(home):
    """Exactly one of two real processes may write the .part.

    No mocks and no patched liveness: two OS processes race for the same lock
    and the OS decides.
    """
    a = _spawn(CONTEND, home, "m.gguf", 2.0)
    b = _spawn(CONTEND, home, "m.gguf", 2.0)
    out_a, err_a = a.communicate(timeout=60)
    out_b, err_b = b.communicate(timeout=60)

    results = sorted([out_a.strip(), out_b.strip()])
    assert a.returncode == 0, err_a
    assert b.returncode == 0, err_b
    assert results == ["LOST", "WON"], (
        f"both processes reported {results} - the lock did not serialise two "
        f"real interpreters.\nA stderr: {err_a}\nB stderr: {err_b}")


def test_two_real_interpreters_on_different_files_both_proceed(home):
    """The lock is per DESTINATION, not global: concurrent downloads of
    unrelated models both proceed.
    """
    a = _spawn(CONTEND, home, "one.gguf", 0.2)
    b = _spawn(CONTEND, home, "two.gguf", 0.2)
    out_a, err_a = a.communicate(timeout=60)
    out_b, err_b = b.communicate(timeout=60)

    assert out_a.strip() == "WON", err_a
    assert out_b.strip() == "WON", err_b


def test_the_lock_is_released_when_the_holder_exits(home):
    """A lock taken and dropped by a real process leaves nothing behind."""
    p = _spawn(CONTEND, home, "m.gguf", 0.05)
    out, err = p.communicate(timeout=60)
    assert out.strip() == "WON", err
    assert not _part_lock_dir("m.gguf").exists(), (
        "the lock directory outlived the process that held it")
    with _part_lock("m.gguf"):
        pass


# --------------------------------------------------------------------------
#  Staleness: liveness, never elapsed time
# --------------------------------------------------------------------------

def test_a_live_holders_lock_is_never_reclaimed(home):
    """A slow-but-healthy download keeps its lock however long it runs, where a
    timeout-based rule would eventually evict it."""
    holder = subprocess.Popen([sys.executable, "-c",
                               "import time; time.sleep(30)"])
    try:
        d = _part_lock_dir("m.gguf")
        d.mkdir(parents=True)
        (d / "owner.json").write_text(
            json.dumps({"pid": holder.pid, "filename": "m.gguf",
                        "started": 0.0}), encoding="utf-8")
        # The injection took: a REAL live process owns this lock, and its
        # recorded start time is the epoch, so any elapsed-time rule would call
        # it stale immediately.
        assert holder.poll() is None, "the holder process died before the test"

        with pytest.raises(PullInFlight) as e:
            with _part_lock("m.gguf"):
                pass
        assert str(holder.pid) in str(e.value)
        assert (d / "owner.json").exists(), (
            "the live holder's own lock record was destroyed by the refusal")
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_a_dead_holders_lock_is_reclaimed(home):
    """A crashed download's lock is reclaimed rather than wedging the
    destination forever."""
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait(timeout=30)

    d = _part_lock_dir("m.gguf")
    d.mkdir(parents=True)
    (d / "owner.json").write_text(
        json.dumps({"pid": dead.pid, "filename": "m.gguf",
                    "started": 0.0}), encoding="utf-8")
    assert dead.poll() is not None, "the supposedly dead holder is still alive"

    with _part_lock("m.gguf"):
        rec = json.loads((d / "owner.json").read_text(encoding="utf-8"))
    assert rec["pid"] == os.getpid(), (
        "the lock was not actually re-taken by this process")


def test_a_hard_killed_holders_lock_is_reclaimed(home):
    """A real holder killed while it holds the lock leaves the record it wrote
    behind; the next pull reclaims it."""
    import signal
    from localm import instances
    holder, pid = _hold(home)
    try:
        d = _part_lock_dir("m.gguf")
        before = json.loads(_record(d))
        # The injection took: the holder wrote its own record, pid space included.
        assert before["pid"] == pid and before.get("space")
        os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        deadline = time.monotonic() + 30
        while instances.pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not instances.pid_alive(pid), "the holder survived the kill"

        with _part_lock("m.gguf"):
            rec = json.loads(_record(d))
        assert rec["pid"] == os.getpid(), (
            "a killed holder's lock was not reclaimed")
    finally:
        _release(holder)


@pytest.mark.parametrize("direction", ["forward", "back"])
def test_a_live_holder_keeps_its_lock_across_a_clock_step(home, monkeypatch,
                                                          direction):
    """A clock step while a download runs (NTP after sleep, a VM or WSL guest
    resyncing) leaves the live holder's lock in place and refuses a second
    pull. The forward step is an hour larger than the machine's uptime.
    """
    import psutil
    from localm import instances
    holder, pid = _hold(home)
    try:
        d = _part_lock_dir("m.gguf")
        before = _record(d)
        # The injection took: a real, live process holds this lock.
        assert json.loads(before)["pid"] == pid
        assert instances.pid_alive(pid)

        step = a_forward_step_past_boot() if direction == "forward" else -3600.0
        boot = psutil.boot_time()
        refused = None
        with monkeypatch.context() as m:
            step_the_clock(m, step)
            assert abs(psutil.boot_time() - (boot + step)) < 5.0
            gone = _part_lock_holder_is_gone(d)
            try:
                with _part_lock("m.gguf"):
                    pass
            except PullInFlight as e:
                refused = e

        assert _record(d) == before, (
            f"the live holder's lock record changed after a {step:+.0f} s "
            f"clock step")
        assert holder.poll() is None, "the holder died during the test"
        assert gone is False, (
            f"a live holder was judged gone after a {step:+.0f} s clock step")
        assert refused is not None and str(pid) in str(refused)
    finally:
        _release(holder)


def test_a_live_pid_now_naming_a_different_process_is_reclaimed(home):
    """The recorded pid is alive but is not the process that took the lock:
    its start identity differs from the recorded one, so the lock is
    reclaimed."""
    other = _idle_child()
    try:
        ident = start_identity_of(other.pid)
        d = _part_lock_dir("m.gguf")
        _write_owner(d, other.pid, start=started_an_hour_earlier(ident),
                     started=time.time())
        assert other.poll() is None, "the pid's current process died early"

        with _part_lock("m.gguf"):
            rec = json.loads(_record(d))
        assert rec["pid"] == os.getpid(), (
            "the lock was not actually re-taken by this process")
    finally:
        _release(other)


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="the boot id is read from Linux's /proc")
@pytest.mark.parametrize("ticks", ["same", "earlier"])
def test_a_record_under_another_boot_id_keeps_the_lock_while_its_pid_is_alive(
        home, ticks):
    """A record under another boot id comes from before a reboot or from
    another machine with this host name, which cannot be told apart here, so a
    live pid keeps the lock whatever its start ticks."""
    other = _idle_child()
    try:
        ident = start_identity_of(other.pid)
        assert ident["boot"], "no boot id was read on Linux"
        if ticks == "earlier":
            ident = started_an_hour_earlier(ident)
        d = _part_lock_dir("m.gguf")
        _write_owner(d, other.pid, started=time.time(), start={
            **ident, "boot": "00000000-0000-0000-0000-000000000000"})
        before = _record(d)

        refused = None
        try:
            with _part_lock("m.gguf"):
                pass
        except PullInFlight as e:
            refused = e
        assert _record(d) == before, (
            "a lock recorded under another boot id was reclaimed from a live "
            "pid")
        assert refused is not None
    finally:
        _release(other)


def _foreign(ident):
    """A start identity in the other supported platform's shape."""
    if "ticks" in ident:
        return {"created": 1.0}
    return {"boot": "00000000-0000-0000-0000-000000000000", "ticks": 1}


@pytest.mark.parametrize("start", [
    "absent", None, {}, "foreign",
    {"boot": None, "ticks": True},
    {"boot": None, "ticks": "12"},
    {"created": "12.5"},
    {"created": float("inf")},
    {"created": float("nan")},
], ids=["absent", "null", "empty", "foreign", "bool-ticks", "text-ticks",
        "text-created", "inf-created", "nan-created"])
def test_a_live_holder_without_a_comparable_start_identity_keeps_the_lock(
        home, start):
    """Uncertainty KEEPS the lock: a record with no start identity, or one
    that cannot be compared with the live pid's, is not evidence the holder
    died."""
    other = _idle_child()
    try:
        fields = {"started": 0.0}
        if start == "foreign":
            fields["start"] = _foreign(start_identity_of(other.pid))
        elif start != "absent":
            fields["start"] = start
        d = _part_lock_dir("m.gguf")
        _write_owner(d, other.pid, **fields)
        before = _record(d)

        refused = None
        try:
            with _part_lock("m.gguf"):
                pass
        except PullInFlight as e:
            refused = e
        assert _record(d) == before
        assert refused is not None and str(other.pid) in str(refused)
    finally:
        _release(other)


def test_a_live_holder_keeps_the_lock_when_its_identity_cannot_be_read_now(
        home, monkeypatch):
    """A record that would prove a replaced process keeps the lock when the
    live pid's own start identity cannot be read."""
    from localm.model_manager import pull
    other = _idle_child()
    try:
        ident = start_identity_of(other.pid)
        d = _part_lock_dir("m.gguf")
        _write_owner(d, other.pid, start=started_an_hour_earlier(ident),
                     started=time.time())
        before = _record(d)
        monkeypatch.setattr(pull.instances, "process_start_identity", lambda pid: None)

        refused = None
        try:
            with _part_lock("m.gguf"):
                pass
        except PullInFlight as e:
            refused = e
        assert _record(d) == before
        assert refused is not None
    finally:
        _release(other)


def _exited_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait(timeout=60)
    return p.pid


def _exited_holder():
    """The pid and start identity of a process that has exited, read while it
    was alive."""
    p = subprocess.Popen(
        [sys.executable, "-c",
         "import os, sys; print(os.getpid(), flush=True); sys.stdin.read()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    pid = int(p.stdout.readline())
    start = start_identity_of(pid)
    _release(p)
    return pid, start


@pytest.mark.parametrize("pid_state", ["alive", "dead"])
def test_a_record_from_another_pid_space_is_never_reclaimed(home, pid_state):
    """A record written in another pid space (another pid namespace, the
    other side of a data folder shared between Windows and WSL, another
    Windows machine) names a pid this process cannot look up, so it keeps the
    lock whether that number is alive here or not and whatever start identity
    it carries."""
    other = _idle_child()
    try:
        if pid_state == "alive":
            pid = other.pid
            start = started_an_hour_earlier(start_identity_of(pid))
        else:
            pid, start = _exited_pid(), None
        d = _part_lock_dir("m.gguf")
        _write_owner(d, pid, space=ANOTHER_PID_SPACE, start=start,
                     started=time.time())
        before = _record(d)

        refused = None
        try:
            with _part_lock("m.gguf"):
                pass
        except PullInFlight as e:
            refused = e
        assert _record(d) == before, (
            "a lock recorded in another pid space was reclaimed")
        assert refused is not None and str(pid) in str(refused)
    finally:
        _release(other)


def test_a_record_naming_no_pid_space_is_judged_by_liveness_alone(home):
    """A record with no pid space keeps the lock while its pid is alive,
    whatever start identity it carries; with a dead pid it is reclaimed (see
    test_a_dead_holders_lock_is_reclaimed)."""
    other = _idle_child()
    try:
        ident = start_identity_of(other.pid)
        d = _part_lock_dir("m.gguf")
        _write_owner(d, other.pid, space=None,
                     start=started_an_hour_earlier(ident), started=time.time())
        before = _record(d)

        refused = None
        try:
            with _part_lock("m.gguf"):
                pass
        except PullInFlight as e:
            refused = e
        assert _record(d) == before
        assert refused is not None and str(other.pid) in str(refused)
    finally:
        _release(other)


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="pid namespaces are a Linux kernel feature")
def test_a_holder_in_another_pid_namespace_keeps_its_lock(home):
    """A real holder in its own pid namespace records a pid that names a
    different, live process here; its lock stays held."""
    import shutil as _shutil
    from localm import instances
    unshare = _shutil.which("unshare")
    if unshare is None:
        pytest.skip("unshare is not installed")
    ns = (unshare, "--user", "--map-root-user", "--pid", "--fork",
          "--mount-proc")
    probe = subprocess.run([*ns, "true"], capture_output=True, text=True,
                           timeout=60)
    if probe.returncode != 0:
        pytest.skip("cannot create a user and pid namespace here: "
                    + probe.stderr.strip())
    holder, _ = _hold(home, prefix=ns)
    try:
        d = _part_lock_dir("m.gguf")
        before = _record(d)
        rec = json.loads(before)
        # The injection took: the recorded pid names a live process here that
        # is not the holder.
        assert instances.pid_alive(rec["pid"])
        assert start_identity_of(rec["pid"]) != rec["start"]

        refused = None
        try:
            with _part_lock("m.gguf"):
                pass
        except PullInFlight as e:
            refused = e
        assert _record(d) == before, (
            "a live holder in another pid namespace lost its lock")
        assert holder.poll() is None, "the holder died during the test"
        assert refused is not None
    finally:
        _release(holder)


def test_the_pid_space_id_differs_between_platforms_on_one_host(monkeypatch):
    from localm.model_manager import pull
    monkeypatch.setattr(pull, "_PID_SPACE", None)
    monkeypatch.setattr(sys, "platform", "win32")
    windows = pull._pid_space_id()
    monkeypatch.setattr(pull, "_PID_SPACE", None)
    monkeypatch.setattr(sys, "platform", "linux")
    linux = pull._pid_space_id()
    assert windows != linux


def test_a_process_and_an_observer_read_the_same_start_identity(home):
    """The identity a process records for itself equals the one another
    process reads for its pid, and two processes of one pid table compute the
    same pid space."""
    child = spawn_on_this_tree(REPORT, home, stdin=subprocess.PIPE)
    try:
        line = child.stdout.readline()
        fields = line.strip().split(" ", 2)
        if len(fields) != 3 or not fields[0].isdigit():
            _release(child)
            pytest.fail(f"the child did not report: {line!r} "
                        f"{child.stderr.read()}")
        pid_text, space_text, own_text = fields
        assert space_text == this_pid_space()
        assert start_identity_of(int(pid_text)) == json.loads(own_text)
    finally:
        _release(child)


def test_a_record_from_another_machine_with_this_host_name_is_never_reclaimed(
        home, monkeypatch):
    """A holder on another machine that shares the data folder and the host
    name records a pid that may be alive here as an unrelated process. On
    Windows its MachineGuid gives it another pid space; on Linux the pid space
    matches but its start identity carries another boot id. Either way the
    lock stays."""
    from localm.model_manager import pull
    other = _idle_child()
    try:
        ident = start_identity_of(other.pid)
        if "ticks" in ident:
            remote_space = this_pid_space()
            remote_start = {**started_an_hour_earlier(ident),
                            "boot": "00000000-0000-0000-0000-000000000000"}
        else:
            with monkeypatch.context() as m:
                m.setattr(pull, "_PID_SPACE", None)
                m.setattr(pull, "_machine_guid", lambda: "another-machine-guid")
                remote_space = pull._pid_space_id()
            # The injection took: the same host name, another MachineGuid.
            assert remote_space != this_pid_space()
            remote_start = started_an_hour_earlier(ident)
        d = _part_lock_dir("m.gguf")
        _write_owner(d, other.pid, space=remote_space, start=remote_start,
                     started=time.time())
        before = _record(d)

        refused = None
        try:
            with _part_lock("m.gguf"):
                pass
        except PullInFlight as e:
            refused = e
        assert _record(d) == before, (
            "a lock recorded on another machine with this host name was "
            "reclaimed")
        assert refused is not None
    finally:
        _release(other)


@pytest.mark.skipif(sys.platform != "win32",
                    reason="MachineGuid is a Windows registry value")
def test_the_windows_machine_guid_is_read():
    from localm.model_manager.pull import _machine_guid
    assert _machine_guid(), "no MachineGuid was read"


def test_the_lock_records_its_holders_pid_space_and_start_identity(home):
    from localm.instances import process_start_identity
    from localm.model_manager.pull import _pid_space_id
    with _part_lock("m.gguf"):
        rec = json.loads(_record(_part_lock_dir("m.gguf")))
    assert rec["pid"] == os.getpid()
    assert rec["space"] == _pid_space_id()
    assert rec["start"] == process_start_identity(os.getpid())


@pytest.mark.parametrize("body", [
    "{not json",              # unreadable
    '{"pid": "banana"}',      # a pid that is not a pid
])
def test_an_unidentifiable_holder_keeps_the_lock(home, body):
    """Uncertainty KEEPS the lock.

    An owner record that cannot be read is not evidence its owner died. The
    refusal names the directory, so a user who is certain can clear it by hand.
    """
    d = _part_lock_dir("m.gguf")
    d.mkdir(parents=True)
    (d / "owner.json").write_text(body, encoding="utf-8")

    with pytest.raises(PullInFlight) as e:
        with _part_lock("m.gguf"):
            pass
    assert str(d) in str(e.value), str(e.value)


def test_the_lock_is_dropped_even_when_the_download_raises(home):
    """A failing download must not wedge the destination for its own pid."""
    with pytest.raises(RuntimeError):
        with _part_lock("m.gguf"):
            raise RuntimeError("download blew up")
    assert not _part_lock_dir("m.gguf").exists()
    with _part_lock("m.gguf"):
        pass


# --------------------------------------------------------------------------
#  Taking over a stale lock
# --------------------------------------------------------------------------

RACE = '''
    import os, sys, time
    from pathlib import Path
    import localm.model_manager.pull as pull
    role, signals = sys.argv[1], Path(sys.argv[2])
    real = pull._part_lock_holder_is_gone
    verdicts = []

    def await_signal(name):
        deadline = time.monotonic() + 30
        while not (signals / name).exists():
            if time.monotonic() > deadline:
                print("TIMEOUT", name, flush=True)
                os._exit(3)
            time.sleep(0.01)

    def gated(d):
        verdict = real(d)
        verdicts.append(verdict)
        if len(verdicts) == 1:
            if role == "A":
                await_signal("b-checked")
            else:
                (signals / "b-checked").touch()
                await_signal("a-entered")
        return verdict

    pull._part_lock_holder_is_gone = gated
    try:
        with pull._part_lock("m.gguf"):
            print("WON", os.getpid(), flush=True)
            if role == "A":
                (signals / "a-entered").touch()
            sys.stdin.read()
    except pull.PullInFlight:
        print("LOST", os.getpid(), flush=True)
'''

RECHECK_RACE = '''
    import os, sys, time
    from pathlib import Path
    import localm.model_manager.pull as pull
    role, signals = sys.argv[1], Path(sys.argv[2])
    real = pull._part_lock_holder_is_gone
    verdicts = []

    def await_signal(name):
        deadline = time.monotonic() + 30
        while not (signals / name).exists():
            if time.monotonic() > deadline:
                print("TIMEOUT", name, flush=True)
                os._exit(3)
            time.sleep(0.01)

    def gated(d):
        verdict = real(d)
        verdicts.append(verdict)
        if role == "A" and len(verdicts) == 2:
            (signals / "a-rechecked").touch()
            await_signal("b-done")
        if role == "B" and len(verdicts) == 1:
            await_signal("a-rechecked")
        return verdict

    pull._part_lock_holder_is_gone = gated
    try:
        with pull._part_lock("m.gguf"):
            print("WON", os.getpid(), flush=True)
            if role == "B":
                (signals / "b-done").touch()
            sys.stdin.read()
    except pull.PullInFlight:
        print("LOST", os.getpid(), flush=True)
        if role == "B":
            (signals / "b-done").touch()
'''

TOMBSTONE_RACE = '''
    import os, sys, time
    from pathlib import Path
    import localm.model_manager.pull as pull
    role, signals = sys.argv[1], Path(sys.argv[2])
    real_gone, real_rename = pull._part_lock_holder_is_gone, pull._rename_dir
    calls = {"gone": 0, "tombstone": 0}

    def await_signal(name):
        deadline = time.monotonic() + 30
        while not (signals / name).exists():
            if time.monotonic() > deadline:
                print("TIMEOUT", name, flush=True)
                os._exit(3)
            time.sleep(0.01)

    def gated_gone(d):
        verdict = real_gone(d)
        calls["gone"] += 1
        if role == "B" and calls["gone"] == 1:
            (signals / "b-checked").touch()
            await_signal("a-at-tombstone")
        return verdict

    def gated_rename(src, dst):
        to_tombstone = Path(dst).name.startswith(pull._LOCK_TOMBSTONE_PREFIX)
        if to_tombstone:
            calls["tombstone"] += 1
        first = role == "A" and to_tombstone and calls["tombstone"] == 1
        if first:
            await_signal("b-checked")
        real_rename(src, dst)
        if first:
            (signals / "a-at-tombstone").touch()
            await_signal("b-done")

    pull._part_lock_holder_is_gone = gated_gone
    pull._rename_dir = gated_rename
    try:
        with pull._part_lock("m.gguf"):
            print("WON", os.getpid(), flush=True)
            if role == "B":
                (signals / "b-done").touch()
            sys.stdin.read()
    except pull.PullInFlight:
        print("LOST", os.getpid(), flush=True)
        if role == "B":
            (signals / "b-done").touch()
'''

GUARD = '''
    import os, sys
    from pathlib import Path
    import localm.model_manager.pull as pull
    with pull._reclaim_guard(Path(sys.argv[1]), sys.argv[2]):
        print("GUARDING", os.getpid(), flush=True)
        sys.stdin.read()
'''


def _write_stale_lock():
    """A lock left by a holder that has exited, in this pid space."""
    d = _part_lock_dir("m.gguf")
    pid, start = _exited_holder()
    _write_owner(d, pid, start=start, started=0.0)
    return d


def _first_line(p):
    """The first stdout line of *p*, split; on an empty or unexpected line the
    process is released and the test fails with its stderr."""
    words = p.stdout.readline().split()
    if words[:1] not in (["WON"], ["LOST"], ["GUARDING"]):
        _release(p)
        pytest.fail(f"child reported {words}: {p.stderr.read()}")
    return words


def test_two_pulls_that_both_found_a_stale_lock_do_not_both_take_it(home, tmp_path):
    """Two real processes reach a crashed download's lock together and both
    judge its holder gone before either takes it over. Exactly one takes the
    lock; the other refuses and leaves the winner's record alone."""
    d = _write_stale_lock()
    signals = tmp_path / "signals"
    signals.mkdir()
    a = spawn_on_this_tree(RACE, home, "A", signals, stdin=subprocess.PIPE)
    b = spawn_on_this_tree(RACE, home, "B", signals, stdin=subprocess.PIPE)
    try:
        won_a = _first_line(a)
        won_b = _first_line(b)
        rec = json.loads(_record(d) or "null")
        # The injection took: both processes checked the stale record before
        # either took the lock over.
        assert (signals / "b-checked").exists() and (signals / "a-entered").exists()
        assert won_a[0] == "WON", f"the first process did not take the lock: {won_a}"
        assert rec is not None and rec["pid"] == int(won_a[1]), (
            "the first holder's lock record was replaced by a second process "
            "that had judged the same stale lock gone")
        assert won_b[0] == "LOST", (
            f"both processes hold the lock: A={won_a} B={won_b}")
    finally:
        _release(a)
        _release(b)


def test_two_takeovers_of_one_stale_lock_are_serialised(home, tmp_path):
    """One process has re-checked the stale lock under its reclaim guard and
    is about to take it over when a second process, which also found it
    stale, arrives. The second refuses instead of taking it over too."""
    d = _write_stale_lock()
    signals = tmp_path / "signals"
    signals.mkdir()
    a = spawn_on_this_tree(RECHECK_RACE, home, "A", signals,
                           stdin=subprocess.PIPE)
    b = spawn_on_this_tree(RECHECK_RACE, home, "B", signals,
                           stdin=subprocess.PIPE)
    try:
        won_b = _first_line(b)
        won_a = _first_line(a)
        rec = json.loads(_record(d) or "null")
        # The injection took: B found the lock stale after A had re-checked it.
        assert (signals / "a-rechecked").exists()
        assert won_a[0] == "WON", f"the first process did not take the lock: {won_a}"
        assert rec is not None and rec["pid"] == int(won_a[1]), (
            "the lock record is not the first process's")
        assert won_b[0] == "LOST", (
            f"both processes took the stale lock over: A={won_a} B={won_b}")
    finally:
        _release(a)
        _release(b)


def test_a_takeover_keeps_its_guard_until_the_lock_is_re_created(home, tmp_path):
    """One process has moved the stale lock it took over to a tombstone when a
    second process, which also found it stale, arrives. The second refuses
    instead of taking the lock over before the first one renames its own
    record in."""
    d = _write_stale_lock()
    signals = tmp_path / "signals"
    signals.mkdir()
    a = spawn_on_this_tree(TOMBSTONE_RACE, home, "A", signals,
                           stdin=subprocess.PIPE)
    b = spawn_on_this_tree(TOMBSTONE_RACE, home, "B", signals,
                           stdin=subprocess.PIPE)
    try:
        won_b = _first_line(b)
        won_a = _first_line(a)
        rec = json.loads(_record(d) or "null")
        # The injection took: B judged the lock before A moved it to a
        # tombstone, and waited until it had.
        assert (signals / "b-checked").exists()
        assert (signals / "a-at-tombstone").exists()
        assert won_a[0] == "WON", f"the first process did not take the lock: {won_a}"
        assert rec is not None and rec["pid"] == int(won_a[1]), (
            "the lock record is not the first process's")
        assert won_b[0] == "LOST", (
            f"both processes took the stale lock over: A={won_a} B={won_b}")
    finally:
        _release(a)
        _release(b)


def test_a_pull_arriving_while_a_takeover_has_moved_the_lock_aside_takes_it(
        home, tmp_path):
    """A pull that arrives after a takeover has moved the stale lock to a
    tombstone, and before it renames its own record in, takes the lock; the
    takeover then refuses. Exactly one of them holds it."""
    d = _write_stale_lock()
    signals = tmp_path / "signals"
    signals.mkdir()
    (signals / "b-checked").touch()
    a = spawn_on_this_tree(TOMBSTONE_RACE, home, "A", signals,
                           stdin=subprocess.PIPE)
    try:
        # The injection took: the takeover has moved the stale lock aside.
        _await_file(signals / "a-at-tombstone")
        assert not d.exists()
        try:
            with _part_lock("m.gguf"):
                rec = json.loads(_record(d))
                (signals / "b-done").touch()
                won_a = _first_line(a)
        finally:
            (signals / "b-done").touch()
        assert rec["pid"] == os.getpid()
        assert won_a[0] == "LOST", f"both processes held the lock: A={won_a}"
    finally:
        _release(a)


def test_the_reclaim_guard_name_is_shorter_than_the_lock_name_and_never_one():
    from localm.model_manager.pull import _reclaim_guard_path
    names = ("m.gguf", "m" * 240 + ".gguf", "x.lock")
    guards = set()
    for name in names:
        d = _part_lock_dir(name)
        guard = _reclaim_guard_path(d, name)
        assert guard.parent == d.parent
        assert len(guard.name) < len(d.name)
        assert not guard.name.lower().endswith(".lock")
        guards.add(guard)
    assert len(guards) == len(names), "two file names share one reclaim guard"
    assert len(_part_lock_dir("m" * 240 + ".gguf").name) == 255


def test_a_guard_held_for_one_file_does_not_block_a_takeover_of_another(home):
    d = _write_stale_lock()
    guard = spawn_on_this_tree(GUARD, home, _part_lock_dir("other.gguf"),
                               "other.gguf", stdin=subprocess.PIPE)
    try:
        _first_line(guard)
        with _part_lock("m.gguf"):
            rec = json.loads(_record(d))
        assert rec["pid"] == os.getpid(), (
            "a stale lock was not taken over while another file's guard was "
            "held")
    finally:
        _release(guard)


def _case_insensitive(directory) -> bool:
    probe = directory / "CaseProbe"
    probe.write_text("", encoding="utf-8")
    try:
        return (directory / "caseprobe").exists()
    finally:
        probe.unlink()


def test_two_spellings_of_one_file_name_share_one_reclaim_guard(home):
    """On a case-insensitive filesystem two spellings of one file name name one
    lock, so a takeover under one spelling waits for another spelling's guard."""
    d = _write_stale_lock()
    if not _case_insensitive(d.parent):
        pytest.skip("the data folder is on a case-sensitive filesystem")
    before = _record(d)
    guard = spawn_on_this_tree(GUARD, home, _part_lock_dir("M.GGUF"), "M.GGUF",
                               stdin=subprocess.PIPE)
    try:
        _first_line(guard)
        refused = None
        try:
            with _part_lock("m.gguf"):
                pass
        except PullInFlight as e:
            refused = e
        assert _record(d) == before, (
            "a stale lock was taken over under one spelling while its guard "
            "was held under another")
        assert refused is not None
    finally:
        _release(guard)


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="a 255-byte lock name needs a filesystem without "
                           "a path length limit below it")
def test_a_stale_lock_with_the_longest_lockable_name_is_taken_over(home):
    longest = "m" * 240 + ".gguf"
    d = _part_lock_dir(longest)
    pid, start = _exited_holder()
    _write_owner(d, pid, start=start, started=0.0)
    with _part_lock(longest):
        assert json.loads(_record(d))["pid"] == os.getpid()


def test_a_takeover_in_progress_elsewhere_refuses_and_leaves_the_stale_lock(home):
    """While another process holds the reclaim guard for this file, a pull
    refuses rather than taking the stale lock over, and takes it once the
    guard is free."""
    d = _write_stale_lock()
    before = _record(d)
    guard = spawn_on_this_tree(GUARD, home, d, "m.gguf", stdin=subprocess.PIPE)
    try:
        _first_line(guard)
        refused = None
        try:
            with _part_lock("m.gguf"):
                pass
        except PullInFlight as e:
            refused = e
        assert _record(d) == before, (
            "a stale lock was taken over while another process held its "
            "reclaim guard")
        assert refused is not None and "taken over" in str(refused)
    finally:
        _release(guard)

    with _part_lock("m.gguf"):
        assert json.loads(_record(d))["pid"] == os.getpid()


def test_a_reclaimer_killed_while_guarding_does_not_wedge_the_next_takeover(home):
    """The OS releases the reclaim guard of a process that dies holding it, so
    the next pull still takes the stale lock over."""
    import signal
    from localm import instances
    d = _write_stale_lock()
    guard = spawn_on_this_tree(GUARD, home, d, "m.gguf", stdin=subprocess.PIPE)
    try:
        pid = int(_first_line(guard)[1])
        os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        deadline = time.monotonic() + 30
        while instances.pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not instances.pid_alive(pid), "the guarding process survived the kill"
    finally:
        _release(guard)

    deadline = time.monotonic() + 30
    while True:
        try:
            with _part_lock("m.gguf"):
                rec = json.loads(_record(d))
            break
        except PullInFlight:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)
    assert rec["pid"] == os.getpid()


def test_a_takeover_that_cannot_lock_the_guard_refuses_with_the_reason(
        home, monkeypatch):
    """A reclaim guard the filesystem cannot lock is not a reason to take the
    stale lock over unguarded: the pull refuses, naming the error and the lock
    to remove by hand."""
    import errno
    from localm.model_manager import pull
    d = _write_stale_lock()
    before = _record(d)

    def cannot_lock(f):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(pull, "_lock_guard_file", cannot_lock)
    with pytest.raises(PullInFlight) as e:
        with _part_lock("m.gguf"):
            pass
    assert _record(d) == before
    assert "No locks available" in str(e.value) and str(d) in str(e.value)


def test_a_release_leaves_a_lock_whose_record_is_no_longer_this_one(home,
                                                                   caplog):
    """Release removes the lock only when its record still carries this
    acquisition's token; anything else is left in place with a warning."""
    import logging
    d = _part_lock_dir("m.gguf")
    with caplog.at_level(logging.WARNING):
        with _part_lock("m.gguf"):
            rec = json.loads(_record(d))
            (d / "owner.json").write_text(
                json.dumps({**rec, "token": "another-acquisition"}),
                encoding="utf-8")
    assert d.exists(), "a release removed a lock another acquisition recorded"
    assert json.loads(_record(d))["token"] == "another-acquisition"
    assert str(d) in caplog.text


@pytest.mark.parametrize("record", ["missing", "unreadable"])
def test_a_release_leaves_a_lock_whose_record_cannot_be_read(home, caplog,
                                                            record):
    """A record that cannot be read at release time is not proof the lock is
    this acquisition's: it is left in place with a warning."""
    import logging
    d = _part_lock_dir("m.gguf")
    with caplog.at_level(logging.WARNING):
        with _part_lock("m.gguf"):
            if record == "missing":
                (d / "owner.json").unlink()
            else:
                (d / "owner.json").write_text("{not json", encoding="utf-8")
    assert d.exists(), "a release removed a lock whose record it could not read"
    assert str(d) in caplog.text


def test_each_acquisition_records_its_own_token(home):
    d = _part_lock_dir("m.gguf")
    tokens = []
    for _ in range(3):
        with _part_lock("m.gguf"):
            tokens.append(json.loads(_record(d))["token"])
    assert all(isinstance(t, str) and len(t) >= 32 for t in tokens), tokens
    assert len(set(tokens)) == 3, f"acquisitions reused a token: {tokens}"


# --------------------------------------------------------------------------
#  A crash never leaves a lock without its record
# --------------------------------------------------------------------------

CRASH_BEFORE_RECORD = '''
    import os, sys
    import localm.model_manager.pull as pull
    real = pull.instances.process_start_identity

    def die_on_own_identity(pid):
        if pid == os.getpid():
            os._exit(0)
        return real(pid)

    pull.instances.process_start_identity = die_on_own_identity
    with pull._part_lock(sys.argv[1]):
        print("HELD", flush=True)
'''

CRASH_IN_RECORD_WRITE = '''
    import os, sys
    import localm.model_manager.pull as pull

    def die_after_creating_the_record(staging, payload):
        open(staging / "owner.json", "x", encoding="utf-8").close()
        print("RECORD-CREATED", staging.name, flush=True)
        os._exit(0)

    pull._write_lock_record = die_after_creating_the_record
    with pull._part_lock(sys.argv[1]):
        print("HELD", flush=True)
'''

CRASH_IN_RELEASE = '''
    import os, shutil, sys
    from pathlib import Path
    import localm.model_manager.pull as pull

    def die_after_removing_the_record(path, *args, **kwargs):
        os.unlink(Path(path) / "owner.json")
        print("RECORD-REMOVED", flush=True)
        os._exit(0)

    with pull._part_lock(sys.argv[1]):
        print("HELD", flush=True)
        shutil.rmtree = die_after_removing_the_record
'''

RELEASE_GATED = '''
    import os, shutil, sys, time
    from pathlib import Path
    import localm.model_manager.pull as pull
    signals = Path(sys.argv[2])
    real = shutil.rmtree

    def await_signal(name):
        deadline = time.monotonic() + 30
        while not (signals / name).exists():
            if time.monotonic() > deadline:
                print("TIMEOUT", name, flush=True)
                os._exit(3)
            time.sleep(0.01)

    def gated(path, *args, **kwargs):
        owner = Path(path) / "owner.json"
        if owner.exists():
            os.unlink(owner)
        (signals / "a-releasing").touch()
        await_signal("b-done")
        return real(path, *args, **kwargs)

    with pull._part_lock(sys.argv[1]):
        print("HELD", os.getpid(), flush=True)
        sys.stdin.readline()
        shutil.rmtree = gated
    print("RELEASED", flush=True)
'''

RELEASE_ON_SIGNAL = '''
    import os, sys, time
    from pathlib import Path
    import localm.model_manager.pull as pull
    signals = Path(sys.argv[2])
    with pull._part_lock(sys.argv[1]):
        print("HELD", os.getpid(), flush=True)
        deadline = time.monotonic() + 30
        while not (signals / "b-looked").exists():
            if time.monotonic() > deadline:
                print("TIMEOUT", flush=True)
                os._exit(3)
            time.sleep(0.01)
    print("RELEASED", flush=True)
'''


def _litter(locks):
    """Entries in *locks* other than reclaim guard files."""
    return sorted(p.name for p in locks.iterdir() if not p.name.endswith(".grd"))


def _await_file(path, timeout=30.0):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            pytest.fail(f"{path.name} never appeared")
        time.sleep(0.01)


def test_a_holder_killed_before_writing_its_record_does_not_wedge_the_lock(home):
    """A real process dies while taking the lock, when it reads its own start
    identity for the owner record. The next pull takes the lock and nothing is
    left behind."""
    p = spawn_on_this_tree(CRASH_BEFORE_RECORD, home, "m.gguf")
    out, err = p.communicate(timeout=60)
    # The injection took: the process exited inside the acquisition, at the
    # point that reads its own start identity for the record.
    assert p.returncode == 0 and "HELD" not in out, (out, err)
    assert "Traceback" not in err, err
    d = _part_lock_dir("m.gguf")

    with _part_lock("m.gguf"):
        rec = json.loads(_record(d))
    assert rec["pid"] == os.getpid(), "the lock was not taken by this process"
    assert _litter(d.parent) == [], _litter(d.parent)


def test_a_holder_killed_while_writing_its_record_does_not_wedge_the_lock(home):
    """A real process dies after creating its owner record file and before
    writing it. The next pull takes the lock; the dead process's unfinished
    acquisition stays behind outside the lock path."""
    p = spawn_on_this_tree(CRASH_IN_RECORD_WRITE, home, "m.gguf")
    out, err = p.communicate(timeout=60)
    words = out.split()
    # The injection took: the process created an empty record file in its
    # staging directory and exited before taking the lock.
    assert words[:1] == ["RECORD-CREATED"] and "HELD" not in words, (out, err)
    assert p.returncode == 0, err
    d = _part_lock_dir("m.gguf")
    staging = d.parent / words[1]
    assert (staging / "owner.json").stat().st_size == 0
    assert not d.exists()

    with _part_lock("m.gguf"):
        rec = json.loads(_record(d))
    assert rec["pid"] == os.getpid(), "the lock was not taken by this process"
    assert _litter(d.parent) == [staging.name], _litter(d.parent)


def test_a_holder_killed_while_releasing_does_not_wedge_the_lock(home):
    """A real process dies during its release, after its owner record is
    removed and before the rest of the lock is. The next pull takes the lock
    and nothing is left behind."""
    p = spawn_on_this_tree(CRASH_IN_RELEASE, home, "m.gguf")
    out, err = p.communicate(timeout=60)
    # The injection took: the holder took the lock, then died in its release
    # right after removing the record.
    assert out.split() == ["HELD", "RECORD-REMOVED"], (out, err)
    assert p.returncode == 0, err
    d = _part_lock_dir("m.gguf")

    with _part_lock("m.gguf"):
        rec = json.loads(_record(d))
    assert rec["pid"] == os.getpid(), "the lock was not taken by this process"
    assert _litter(d.parent) == [], _litter(d.parent)


def test_a_pull_arriving_during_a_release_takes_the_lock(home, tmp_path):
    """A pull that arrives while a real holder is in the middle of releasing
    the lock takes it instead of refusing."""
    signals = tmp_path / "signals"
    signals.mkdir()
    a = spawn_on_this_tree(RELEASE_GATED, home, "m.gguf", signals,
                           stdin=subprocess.PIPE)
    try:
        first = a.stdout.readline().split()
        assert first[:1] == ["HELD"], (first, a.stderr.read())
        a.stdin.write("\n")
        a.stdin.flush()
        # The injection took: the holder is inside its release, with its
        # record already removed.
        _await_file(signals / "a-releasing")
        d = _part_lock_dir("m.gguf")
        try:
            with _part_lock("m.gguf"):
                rec = json.loads(_record(d))
        finally:
            (signals / "b-done").touch()
        assert rec["pid"] == os.getpid()
        assert a.stdout.readline().split() == ["RELEASED"]
    finally:
        _release(a)


def test_a_pull_that_finds_the_lock_released_before_reading_it_takes_it(
        home, tmp_path, monkeypatch):
    """The lock exists when a pull tries to take it and is released before
    the pull reads its record. The pull takes it instead of refusing."""
    from localm.model_manager import pull
    signals = tmp_path / "signals"
    signals.mkdir()
    a = spawn_on_this_tree(RELEASE_ON_SIGNAL, home, "m.gguf", signals,
                           stdin=subprocess.PIPE)
    try:
        first = a.stdout.readline().split()
        assert first[:1] == ["HELD"], (first, a.stderr.read())
        real = pull._part_lock_owner
        calls = []

        def after_the_release(d):
            calls.append(d)
            if len(calls) == 1:
                (signals / "b-looked").touch()
                assert a.stdout.readline().split() == ["RELEASED"]
            return real(d)

        monkeypatch.setattr(pull, "_part_lock_owner", after_the_release)
        d = _part_lock_dir("m.gguf")
        with _part_lock("m.gguf"):
            monkeypatch.undo()
            rec = json.loads(_record(d))
        # The injection took: the pull read the lock while it was held.
        assert calls, "the pull never read the lock record"
        assert rec["pid"] == os.getpid()
    finally:
        _release(a)


def test_an_empty_lock_directory_is_taken_over(home):
    """An empty lock directory, as a crash in an earlier version of the lock
    leaves, is taken over and nothing is left behind."""
    d = _part_lock_dir("m.gguf")
    d.mkdir(parents=True)
    with _part_lock("m.gguf"):
        rec = json.loads(_record(d))
    assert rec["pid"] == os.getpid()
    assert _litter(d.parent) == [], _litter(d.parent)


def test_an_empty_lock_directory_is_not_taken_over_while_its_guard_is_held(home):
    """Only a pull holding the reclaim guard replaces an empty lock
    directory."""
    d = _part_lock_dir("m.gguf")
    d.mkdir(parents=True)
    guard = spawn_on_this_tree(GUARD, home, d, "m.gguf", stdin=subprocess.PIPE)
    try:
        _first_line(guard)
        with pytest.raises(PullInFlight) as e:
            with _part_lock("m.gguf"):
                pass
        assert d.is_dir() and not any(d.iterdir()), (
            "an empty lock directory was replaced while another process held "
            "its reclaim guard")
        assert "taken over" in str(e.value)
    finally:
        _release(guard)


def test_leftovers_of_earlier_crashes_are_removed(home):
    """Released locks and unfinished acquisitions left by dead processes are
    removed by the next pull; an acquisition in progress in a live process is
    left alone."""
    locks = _part_lock_dir("m.gguf").parent
    locks.mkdir(parents=True)
    tomb = locks / ".pull-tomb-0123"
    tomb.mkdir()
    (tomb / "junk").write_text("x", encoding="utf-8")
    (locks / ".pull-acq-empty").mkdir()
    pid, start = _exited_holder()
    _write_owner(locks / ".pull-acq-dead", pid, start=start)
    live = _idle_child()
    try:
        _write_owner(locks / ".pull-acq-live", live.pid)
        with _part_lock("m.gguf"):
            pass
        assert _litter(locks) == [".pull-acq-live"], _litter(locks)
    finally:
        _release(live)


def test_a_staging_directory_removed_before_its_record_is_written_is_restaged(
        home, monkeypatch):
    """A pull whose unfinished acquisition is removed by another pull's
    leftover sweep before its record is written starts that acquisition
    again."""
    from localm.model_manager import pull
    real = pull._write_lock_record
    calls = []

    def swept_first(staging, payload):
        calls.append(staging)
        if len(calls) == 1:
            os.rmdir(staging)
        return real(staging, payload)

    monkeypatch.setattr(pull, "_write_lock_record", swept_first)
    with _part_lock("m.gguf"):
        rec = json.loads(_record(_part_lock_dir("m.gguf")))
    assert len(calls) == 2, calls
    assert rec["pid"] == os.getpid()
    assert _litter(_part_lock_dir("m.gguf").parent) == []


# --------------------------------------------------------------------------
#  Placement and wiring
# --------------------------------------------------------------------------

def test_the_lock_lives_beside_the_models_dir_not_inside_it(home):
    """The lock directory sits beside models/, not inside it: sync_models_dir
    walks that tree, and a tidy-up of stray files there would be free to delete
    a live lock."""
    d = _part_lock_dir("m.gguf")
    models = home / "models"
    assert models not in d.parents, f"{d} is inside the models dir"
    assert d.parent.parent == models.parent


def test_pull_url_refuses_and_leaves_the_part_file_untouched(home):
    """With the lock held, a second _pull_url does not open the .part at all.
    The bytes are asserted first, before the return value.
    """
    from unittest.mock import MagicMock

    from localm.model_manager import pull as _pull

    part = home / "models" / "m.gguf.part"
    part.write_bytes(b"FIRST-DOWNLOAD-BYTES")
    before = part.read_bytes()

    # Asserted from OUTSIDE the call, never by raising from a stub: _pull_url
    # catches NetworkPolicyError around exactly this region, so an assertion
    # raised inside it would be an input to that code rather than a failure the
    # runner sees.
    reached_network = MagicMock(side_effect=RuntimeError("unreachable"))
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(_pull, "_ssrf_resolve_final_url", reached_network)
    try:
        with _part_lock("m.gguf"):
            # The injection took: the lock is held, so the call below is
            # genuinely contending rather than arriving at a free destination.
            assert _part_lock_dir("m.gguf").exists()
            ok = _pull._pull_url("https://example.invalid/m.gguf", "m")
    finally:
        monkeypatch.undo()

    assert part.read_bytes() == before, (
        "a second pull wrote into the .part file while another download held "
        "it - this is the interleaving that corrupts the download")
    reached_network.assert_not_called()
    assert ok is False
