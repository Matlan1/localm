# SPDX-License-Identifier: AGPL-3.0-or-later
"""The tray Restart/Stop control surface must forward the running server's real
instance_id (and, for restart, its real port) into
hs._do_restart/hs._do_shutdown.

appface calls on_restart/on_stop with NO arguments
(threading.Thread(target=self.on_restart)), and both
http_server._do_restart/_do_shutdown are keyword-only with None defaults, so
wiring the bare functions directly makes every tray Restart/Stop call
disarm_crash_guard(instance_id=None), which clears the LEGACY unscoped marker
and leaves the real per-instance marker armed. _do_restart also loses its port,
letting the re-exec'd server land on a different one.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from localm.plugins.gui.cli import _tray_callbacks


def _app(instance_id="abc123", instance_port=8642):
    return SimpleNamespace(state=SimpleNamespace(
        instance_id=instance_id, instance_port=instance_port))


def test_on_restart_forwards_the_real_instance_id_and_port():
    app = _app(instance_id="inst-42", instance_port=9001)
    hs = MagicMock()
    on_restart, _ = _tray_callbacks(app, hs)

    on_restart()

    hs._do_restart.assert_called_once_with(instance_id="inst-42", port=9001)


def test_on_stop_forwards_the_real_instance_id():
    app = _app(instance_id="inst-42", instance_port=9001)
    hs = MagicMock()
    _, on_stop = _tray_callbacks(app, hs)

    on_stop()

    hs._do_shutdown.assert_called_once_with(instance_id="inst-42")


def test_callbacks_take_no_arguments():
    """appface invokes these via threading.Thread(target=callback) with no
    args - the call shape the wiring has to survive. Confirm both
    callables tolerate a genuinely empty call, not just that they exist."""
    app = _app()
    hs = MagicMock()
    on_restart, on_stop = _tray_callbacks(app, hs)

    on_restart()   # must not raise TypeError: missing required argument
    on_stop()


def test_reads_instance_state_lazily_not_at_wire_time():
    """app.state.instance_id/instance_port are not set until
    instances.advertise() runs (inside hs.run_advertised(), called AFTER the
    tray is wired), so the callbacks read app.state at CALL time rather than
    capturing a value (e.g. via functools.partial) at wire time."""
    app = SimpleNamespace(state=SimpleNamespace())   # instance_id/port NOT YET set
    hs = MagicMock()
    on_restart, on_stop = _tray_callbacks(app, hs)

    # advertise() runs here, between wiring and the user ever clicking
    # anything - simulates hs.run_advertised()'s later, real assignment.
    app.state.instance_id = "inst-99"
    app.state.instance_port = 7777

    on_restart()
    on_stop()

    hs._do_restart.assert_called_once_with(instance_id="inst-99", port=7777)
    hs._do_shutdown.assert_called_once_with(instance_id="inst-99")


def test_missing_instance_state_falls_back_to_none_not_a_crash():
    """A defensive fallback, not the expected path: if a callback somehow
    fires before advertise() ever ran, it must degrade to the same
    None-scoped (legacy-marker) behaviour rather than raise AttributeError
    into the tray's background thread."""
    app = SimpleNamespace(state=SimpleNamespace())   # no instance_id/port at all
    hs = MagicMock()
    on_restart, on_stop = _tray_callbacks(app, hs)

    on_restart()
    on_stop()

    hs._do_restart.assert_called_once_with(instance_id=None, port=None)
    hs._do_shutdown.assert_called_once_with(instance_id=None)
