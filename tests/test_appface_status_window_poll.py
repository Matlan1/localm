# SPDX-License-Identifier: AGPL-3.0-or-later
"""_StatusWindow._poll turns queued messages into label and window changes.

The Tk root and labels are recorders, so no window exists; the real
_StatusWindow runs its real _poll."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from localm import appface


@pytest.fixture()
def win():
    w = appface._StatusWindow(name="localm", url="http://127.0.0.1:8642")
    w._root = MagicMock()
    w._status_lbl = MagicMock()
    w._addr_lbl = MagicMock()
    return w


def _calls(mock):
    return [c[0] for c in mock.method_calls]


def test_error_shows_the_message_in_red_and_raises_the_window(win):
    win.set_error("model failed to load")
    win._poll()
    win._status_lbl.configure.assert_called_once_with(
        text="model failed to load", fg=appface._RED)
    assert _calls(win._root) == ["deiconify", "lift", "after"]


def test_show_deiconifies_lifts_and_focuses(win):
    win.show()
    win._poll()
    assert _calls(win._root) == ["deiconify", "lift", "focus_force", "after"]
    win._status_lbl.configure.assert_not_called()


def test_ready_withdraws_when_hiding_on_ready(win):
    win.hide_on_ready = True
    win.set_ready()
    win._poll()
    win._status_lbl.configure.assert_called_once_with(text="Running", fg=appface._GREEN)
    win._addr_lbl.configure.assert_called_once_with(text=win.url)
    assert _calls(win._root) == ["withdraw", "after"]


def test_ready_leaves_the_window_up_otherwise(win):
    win.hide_on_ready = False
    win.set_ready()
    win._poll()
    win._status_lbl.configure.assert_called_once_with(text="Running", fg=appface._GREEN)
    assert _calls(win._root) == ["after"]


def test_a_window_call_that_raises_does_not_stop_the_poll(win):
    win._root.deiconify.side_effect = RuntimeError("gone")
    win.set_error("bad")
    win.show()
    win._poll()
    win._root.after.assert_called_once()


def test_the_poll_reschedules_itself(win):
    win._poll()
    win._root.after.assert_called_once_with(300, win._poll)
