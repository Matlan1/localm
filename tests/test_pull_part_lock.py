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
    from localm.model_manager.pull import _pid_space_id, _process_start_identity
    print(os.getpid(), _pid_space_id(),
          json.dumps(_process_start_identity(os.getpid())), flush=True)
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
        monkeypatch.setattr(pull, "_process_start_identity", lambda pid: None)

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
    from localm.model_manager.pull import _pid_space_id, _process_start_identity
    with _part_lock("m.gguf"):
        rec = json.loads(_record(_part_lock_dir("m.gguf")))
    assert rec["pid"] == os.getpid()
    assert rec["space"] == _pid_space_id()
    assert rec["start"] == _process_start_identity(os.getpid())


@pytest.mark.parametrize("body", [
    None,                     # no owner record at all
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
    if body is not None:
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
