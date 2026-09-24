# SPDX-License-Identifier: AGPL-3.0-or-later
"""portmux's stop-signal and crash-watchdog helpers on their failure paths.

Each helper degrades instead of raising: a hook that is already gone, a signal
whose handler cannot be read or set, a wakeup socket that fails, a watchdog
script that is missing or cannot be spawned. These run on every platform; the
end-to-end signal tests in test_portmux_lifecycle.py only reach some of these
paths where the platform defines SIGHUP.
"""

import json
import signal
import socket
import subprocess
import threading
import types

import pytest

from localm import portmux


@pytest.fixture
def stop_state(monkeypatch):
    """Fresh portmux stop state; a re-delivered signal is recorded, not raised.
    Yields that record."""
    monkeypatch.setattr(portmux, "_active_runs", 0)
    monkeypatch.setattr(portmux, "_stop_requested", False)
    monkeypatch.setattr(portmux, "_stopping", False)
    monkeypatch.setattr(portmux, "_stop_hooks", [])
    raised = []
    monkeypatch.setattr(portmux.signal, "raise_signal", raised.append)
    yield raised


@pytest.fixture
def warnings(monkeypatch):
    """Every portmux._log.warning call, as its argument tuple."""
    seen = []
    monkeypatch.setattr(portmux, "_log",
                        types.SimpleNamespace(warning=lambda *a: seen.append(a)))
    return seen


class _Sock:
    """recv() returns, or raises, the queued items in order."""

    def __init__(self, items):
        self.items = list(items)

    def recv(self, _n):
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _refuse(*_args):
    raise ValueError("refused")


# --- stop hooks and the stop-signal handler ---------------------------------

def test_discarding_a_hook_that_is_not_registered_leaves_the_others(stop_state):
    def kept():
        return None

    def never_added():
        return None

    portmux._stop_hooks.append(kept)
    portmux._discard_stop_hook(never_added)
    assert portmux._stop_hooks == [kept]


def test_a_stop_signal_whose_default_cannot_be_restored_is_not_redelivered(
        stop_state, monkeypatch):
    monkeypatch.setattr(portmux.signal, "signal", _refuse)
    portmux._on_stop_signal(signal.SIGTERM, None)
    assert stop_state == []
    assert portmux._stop_requested is False


# --- the wakeup-socket helper -----------------------------------------------

def test_the_waker_stops_reading_when_its_socket_errors(stop_state, monkeypatch):
    monkeypatch.setattr(portmux, "_active_runs", 1)
    hits = []
    portmux._stop_hooks.append(lambda: hits.append(1))
    portmux._deliver_woken_stops(_Sock([OSError("closed")]), {int(signal.SIGTERM)})
    assert hits == []
    assert portmux._stop_requested is False


def test_the_waker_requests_a_stop_only_for_a_routed_signal(stop_state, monkeypatch):
    monkeypatch.setattr(portmux, "_active_runs", 1)
    hits = []
    portmux._stop_hooks.append(lambda: hits.append(1))
    routed = int(signal.SIGTERM)
    other = routed + 1
    portmux._deliver_woken_stops(_Sock([bytes([other]), b""]), {routed})
    assert hits == []
    assert portmux._stop_requested is False
    portmux._deliver_woken_stops(_Sock([bytes([other]), bytes([routed]), b""]), {routed})
    assert hits == [1]
    assert portmux._stop_requested is True


def test_no_waker_starts_when_no_socket_pair_can_be_made(monkeypatch, warnings):
    def no_pair():
        raise OSError("no socket pair")
    monkeypatch.setattr(socket, "socketpair", no_pair)
    assert portmux._start_stop_waker([signal.SIGTERM]) is None
    assert len(warnings) == 1
    assert "stop signals will wait for the main thread" in warnings[0][0]


def test_no_waker_starts_and_its_sockets_close_when_the_wakeup_fd_is_refused(
        monkeypatch, warnings):
    made = []
    real_pair = socket.socketpair

    def pair():
        ends = real_pair()
        made.extend(ends)
        return ends
    monkeypatch.setattr(socket, "socketpair", pair)
    monkeypatch.setattr(portmux.signal, "set_wakeup_fd", _refuse)
    assert portmux._start_stop_waker([signal.SIGTERM]) is None
    assert len(made) == 2 and all(s.fileno() == -1 for s in made)
    assert len(warnings) == 1


def test_the_waker_ends_even_when_the_previous_wakeup_fd_cannot_be_restored(
        stop_state, monkeypatch):
    calls = []

    def set_wakeup_fd(fd):
        calls.append(fd)
        if len(calls) == 1:
            return -1
        raise ValueError("cannot restore")
    monkeypatch.setattr(portmux.signal, "set_wakeup_fd", set_wakeup_fd)
    before = {t.ident for t in threading.enumerate()}
    stop = portmux._start_stop_waker([signal.SIGTERM])
    helpers = [t for t in threading.enumerate()
               if t.name == "localm-stop-signals" and t.ident not in before]
    assert stop is not None and len(helpers) == 1
    stop()
    assert calls[1] == -1, "stop() did not try to put the previous wakeup fd back"
    assert not helpers[0].is_alive()


# --- route_stop_signals ------------------------------------------------------

def test_route_stop_signals_skips_a_signal_it_cannot_inspect(monkeypatch):
    monkeypatch.setattr(portmux.signal, "getsignal", _refuse)
    with portmux.route_stop_signals(("SIGTERM",)) as routed:
        assert routed == []


def test_route_stop_signals_exits_cleanly_when_a_signal_cannot_be_inspected_on_exit(
        stop_state, monkeypatch):
    original = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        with portmux.route_stop_signals(("SIGTERM",)) as routed:
            assert routed == [signal.SIGTERM]
            monkeypatch.setattr(portmux.signal, "getsignal", _refuse)
    finally:
        monkeypatch.undo()
        signal.signal(signal.SIGTERM, original if original is not None else signal.SIG_DFL)


# --- the crash-recovery watchdog spawn --------------------------------------

@pytest.fixture
def spawn(monkeypatch, tmp_path):
    """A repo root holding the watchdog script, a crash dir, and a recording
    Popen. Yields (repo root, recorded (argv, kwargs) calls)."""
    import localm.bugreport
    import localm.updater
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "crash_recovery_watchdog.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(localm.updater, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(localm.bugreport, "_crash_dir", lambda home=None: tmp_path / "crashes")
    monkeypatch.delenv("LOCALM_CRASH_WATCHDOG", raising=False)
    monkeypatch.delenv("LOCALM_CRASH_WATCHDOG_HISTORY", raising=False)
    monkeypatch.setattr(portmux.sys, "argv", ["localm", "serve"])
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kw: calls.append((argv, kw)))
    yield tmp_path, calls


def _spawn(port=8443):
    portmux._spawn_crash_recovery_watchdog(host="127.0.0.1", port=port, tls=False,
                                           instance_id="inst-1")


def _arg(argv, flag):
    return argv[argv.index(flag) + 1]


def test_no_watchdog_is_spawned_when_its_script_is_missing(spawn):
    root, calls = spawn
    (root / "scripts" / "crash_recovery_watchdog.py").unlink()
    _spawn()
    assert calls == []


def test_the_relaunch_names_the_port_only_when_there_is_one(spawn):
    _root, calls = spawn
    _spawn(port=8443)
    _spawn(port=0)
    with_port = json.loads(_arg(calls[0][0], "--relaunch-argv"))
    without_port = json.loads(_arg(calls[1][0], "--relaunch-argv"))
    assert with_port[-2:] == ["-p", "8443"]
    assert "-p" not in without_port


def test_the_restart_history_is_handed_to_the_next_watchdog(spawn, monkeypatch):
    _root, calls = spawn
    monkeypatch.setenv("LOCALM_CRASH_WATCHDOG_HISTORY", "1700000000,1700000100")
    _spawn()
    assert _arg(calls[0][0], "--restart-history") == "1700000000,1700000100"


def test_without_restart_history_the_watchdog_gets_none(spawn):
    _root, calls = spawn
    _spawn()
    assert "--restart-history" not in calls[0][0]


def test_a_watchdog_that_fails_to_spawn_is_logged_and_never_raises(
        spawn, monkeypatch, warnings):
    def fail(argv, **kw):
        raise OSError("spawn failed")
    monkeypatch.setattr(subprocess, "Popen", fail)
    _spawn()
    assert len(warnings) == 1
    assert "could not spawn the crash-recovery watchdog" in warnings[0][0]


def test_a_spawn_failure_whose_warning_also_fails_still_never_raises(spawn, monkeypatch):
    def fail(argv, **kw):
        raise OSError("spawn failed")

    def broken_warning(*_a):
        raise RuntimeError("logging is torn down")
    monkeypatch.setattr(subprocess, "Popen", fail)
    monkeypatch.setattr(portmux, "_log", types.SimpleNamespace(warning=broken_warning))
    _spawn()
