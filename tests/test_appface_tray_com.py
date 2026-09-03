# SPDX-License-Identifier: AGPL-3.0-or-later
"""_WinTray's tray thread must establish an STA COM apartment before its
message loop runs and release it when the thread ends, on every exit path.

TrackPopupMenu's internal message pump can trigger outgoing COM calls (UI
Automation / accessibility notifications for the open popup), and a thread
with no COM apartment established raises RPC_E_CANTCALLOUT_ININPUTSYNCCALL
when that happens - the exact HRESULT captured in a real crash trace for
issue #1578. _com_init/_com_uninit and their wiring into _run are the fix.
"""

import sys
import threading
from unittest.mock import MagicMock

import pytest

from localm import appface


def _tray():
    return appface._WinTray(name="LocaLM", url="https://127.0.0.1:1", logfile=None,
                            on_restart=None, on_stop=None)


def test_com_uninit_with_none_is_a_safe_noop():
    _tray()._com_uninit(None)   # must not raise


def test_com_init_returns_none_and_does_not_raise_off_windows_or_on_failure(
        monkeypatch):
    """On any platform without ctypes.windll (or if ole32 rejects the call),
    _com_init must degrade to None rather than raise into the tray thread."""
    tray = _tray()
    fake_ole32 = MagicMock()
    fake_ole32.CoInitializeEx.return_value = -2147417850  # RPC_E_CHANGED_MODE
    fake_windll = MagicMock(ole32=fake_ole32)
    monkeypatch.setattr("ctypes.windll", fake_windll, raising=False)

    result = tray._com_init()

    assert result is None
    fake_ole32.CoUninitialize.assert_not_called()


def test_com_init_returns_the_ole32_handle_on_success(monkeypatch):
    tray = _tray()
    fake_ole32 = MagicMock()
    fake_ole32.CoInitializeEx.return_value = 0   # S_OK
    fake_windll = MagicMock(ole32=fake_ole32)
    monkeypatch.setattr("ctypes.windll", fake_windll, raising=False)

    result = tray._com_init()

    assert result is fake_ole32
    fake_ole32.CoInitializeEx.assert_called_once_with(
        None, tray._COINIT_APARTMENTTHREADED)


def test_com_uninit_calls_couninitialize_on_the_returned_handle():
    tray = _tray()
    fake_ole32 = MagicMock()

    tray._com_uninit(fake_ole32)

    fake_ole32.CoUninitialize.assert_called_once_with()


def test_com_uninit_never_raises_even_if_couninitialize_does():
    tray = _tray()
    fake_ole32 = MagicMock()
    fake_ole32.CoUninitialize.side_effect = OSError("boom")

    tray._com_uninit(fake_ole32)   # must not raise


@pytest.mark.skipif(sys.platform != "win32",
                    reason="_run reads ctypes.windll, which is Windows-only")
def test_run_releases_the_com_apartment_even_when_the_message_loop_raises(
        monkeypatch):
    """The regression this fix exists to prevent: a resource (here, the COM
    apartment) acquired at the top of _run must be released via the finally,
    not only on the clean-exit path. Simulate the message loop blowing up and
    confirm _com_uninit still fires, with the exception still propagating
    (this is cleanup, not new exception-swallowing)."""
    tray = _tray()
    sentinel_ole32 = object()
    monkeypatch.setattr(tray, "_com_init", lambda: sentinel_ole32)
    uninit_calls = []
    monkeypatch.setattr(tray, "_com_uninit", lambda h: uninit_calls.append(h))
    monkeypatch.setattr(
        tray, "_run_message_loop",
        MagicMock(side_effect=RuntimeError("message loop exploded")))

    with pytest.raises(RuntimeError, match="message loop exploded"):
        tray._run()

    assert uninit_calls == [sentinel_ole32]


@pytest.mark.skipif(sys.platform != "win32",
                    reason="_run reads ctypes.windll, which is Windows-only")
def test_run_releases_the_com_apartment_on_the_clean_exit_path(monkeypatch):
    tray = _tray()
    sentinel_ole32 = object()
    monkeypatch.setattr(tray, "_com_init", lambda: sentinel_ole32)
    uninit_calls = []
    monkeypatch.setattr(tray, "_com_uninit", lambda h: uninit_calls.append(h))
    monkeypatch.setattr(tray, "_run_message_loop", MagicMock(return_value=None))

    tray._run()

    assert uninit_calls == [sentinel_ole32]


@pytest.mark.skipif(sys.platform != "win32", reason="real ole32 is Windows-only")
def test_real_coinitializeex_on_a_fresh_thread_succeeds():
    """Pins the actual Win32 contract this fix depends on: a brand-new thread
    that has never touched COM can always establish an STA apartment (S_OK or
    S_FALSE, never a negative HRESULT), and CoUninitialize on it never raises.
    Real ctypes calls, no mocking - this is an OS contract, not our code."""
    result = {}

    def _worker():
        tray = _tray()
        ole32 = tray._com_init()
        result["ole32"] = ole32
        tray._com_uninit(ole32)

    t = threading.Thread(target=_worker, name="test-com-init-fresh-thread")
    t.start()
    t.join(timeout=5)

    assert not t.is_alive(), "worker thread did not finish within 5s"
    assert result["ole32"] is not None, (
        "CoInitializeEx failed on a brand-new thread with no prior COM "
        "state - see the debug log for the returned HRESULT")
