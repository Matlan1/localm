# SPDX-License-Identifier: AGPL-3.0-or-later
"""NEW-CRASH-NOTICE-USELESS (B): the tray Restart/Stop control surface must forward the running server's real instance_id (and, for restart, its real port) into hs._do_restart/hs._do_shutdown - not silently drop them."""

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
    """appface invokes these via threading.Thread(target=callback) with no args - the exact call shape that broke before this fix."""
    app = _app()
    hs = MagicMock()
    on_restart, on_stop = _tray_callbacks(app, hs)

    on_restart()   # must not raise TypeError: missing required argument
    on_stop()


def test_reads_instance_state_lazily_not_at_wire_time():
    """The whole point of the fix: app.state.instance_id/instance_port are not set until instances.advertise() runs (inside hs.run_advertised(), called AFTER the tray is wired) - so the callbacks must read app.state at CALL time, not capture a value (e.g. via functools.partial) at wire time, when it would..."""
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
    """A defensive fallback, not the expected path: if a callback somehow fires before advertise() ever ran, it must degrade to the same None-scoped (legacy-marker) behaviour rather than raise AttributeError into the tray's background thread."""
    app = SimpleNamespace(state=SimpleNamespace())   # no instance_id/port at all
    hs = MagicMock()
    on_restart, on_stop = _tray_callbacks(app, hs)

    on_restart()
    on_stop()

    hs._do_restart.assert_called_once_with(instance_id=None, port=None)
    hs._do_shutdown.assert_called_once_with(instance_id=None)
