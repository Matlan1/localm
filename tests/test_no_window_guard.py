# SPDX-License-Identifier: AGPL-3.0-or-later
"""The refusals in tests/_no_window.py: on Windows a test cannot start a
process with a visible console, open a file or a browser with the shell, or
show a Tk window."""

import asyncio
import multiprocessing
import os
import subprocess
import sys
import threading
import webbrowser

import pytest

from tests import _no_window as nw
from tests._tk_root import build_tk_root, collect_tk_garbage  # noqa: F401

_needs_windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows process creation")


class _Stop(Exception):
    pass


@pytest.fixture
def refusals():
    """Yield a function returning the reasons refused since the test started,
    then drop those refusals so the autouse fixture in tests/conftest.py does
    not fail the test for them."""
    before = dict(nw.HITS)

    def new():
        return [reason for (reason, origin), n in nw.HITS.items()
                if n > before.get((reason, origin), 0)]

    yield new
    nw.HITS.clear()
    nw.HITS.update(before)


@pytest.fixture
def spy(monkeypatch):
    """Record every CreateProcess call that gets past the guard, then start the
    process for real."""
    calls = []
    real = nw._real_create

    def record(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(nw, "_real_create", record)
    return calls


@pytest.fixture
def stop_spy(monkeypatch):
    """Record every CreateProcess call that gets past the guard and start
    nothing."""
    calls = []

    def record(*args):
        calls.append(args)
        raise _Stop

    monkeypatch.setattr(nw, "_real_create", record)
    return calls


def _flags(call):
    return call[5]


def _hidden_startupinfo(call):
    si = call[8]
    return bool(si.dwFlags & nw.STARTF_USESHOWWINDOW) and si.wShowWindow == nw.SW_HIDE


def _ours(calls, needle):
    """The one recorded call whose command line contains *needle*."""
    mine = [c for c in calls if needle in str(c[1])]
    assert len(mine) == 1, [str(c[1]) for c in calls]
    return mine[0]


# --------------------------------------------------------------------------- #
#  hidden(): the decision, on every platform                                   #
# --------------------------------------------------------------------------- #

def test_a_plain_spawn_keeps_its_flags_and_gains_sw_hide():
    flags, si, refusal = nw.hidden(0x00000200, None)
    assert refusal is None
    assert flags == 0x00000200
    assert si.dwFlags & nw.STARTF_USESHOWWINDOW and si.wShowWindow == nw.SW_HIDE


def test_detached_process_is_refused_unless_allowed():
    _, _, refusal = nw.hidden(nw.DETACHED_PROCESS, None)
    assert refusal and "DETACHED_PROCESS" in refusal
    flags, _, refusal = nw.hidden(nw.DETACHED_PROCESS, None, detached_allowed=True)
    assert refusal is None and flags == nw.DETACHED_PROCESS


def test_a_visible_new_console_is_refused():
    shown = nw._Info(dwFlags=nw.STARTF_USESHOWWINDOW, wShowWindow=1)
    for si in (None, nw._Info(), shown):
        _, _, refusal = nw.hidden(nw.CREATE_NEW_CONSOLE, si)
        assert refusal and "CREATE_NEW_CONSOLE" in refusal


def test_a_hidden_new_console_becomes_create_no_window():
    si = nw._Info(dwFlags=nw.STARTF_USESHOWWINDOW, wShowWindow=nw.SW_HIDE)
    flags, _, refusal = nw.hidden(nw.CREATE_NEW_CONSOLE | 0x00000200, si)
    assert refusal is None
    assert flags == nw.CREATE_NO_WINDOW | 0x00000200


def test_the_callers_startupinfo_is_never_modified():
    si = nw._Info(dwFlags=0x100, wShowWindow=5)
    _, out, _ = nw.hidden(0, si)
    assert (si.dwFlags, si.wShowWindow) == (0x100, 5)
    assert out is not si
    assert out.dwFlags == 0x100 | nw.STARTF_USESHOWWINDOW and out.wShowWindow == nw.SW_HIDE


# --------------------------------------------------------------------------- #
#  The armed guard on Windows                                                  #
# --------------------------------------------------------------------------- #

@_needs_windows
def test_the_guard_is_armed_for_the_whole_suite():
    import _winapi
    import tkinter
    assert nw.armed()
    assert _winapi.CreateProcess is nw.create_process
    assert os.startfile is nw.startfile
    assert webbrowser.open is nw.browser_open
    assert tkinter.Tk.__init__ is nw.tk_init
    assert tkinter.Toplevel.__init__ is nw.toplevel_init
    assert tkinter.Wm.deiconify is nw.deiconify and tkinter.Wm.wm_deiconify is nw.deiconify


@_needs_windows
def test_cmd_exit_starts_with_a_hidden_startupinfo(spy):
    assert subprocess.run(["cmd", "/c", "exit 0"]).returncode == 0
    assert _hidden_startupinfo(_ours(spy, "exit 0"))


@_needs_windows
def test_asyncio_subprocesses_start_with_a_hidden_startupinfo(spy):
    async def run():
        proc = await asyncio.create_subprocess_exec("cmd", "/c", "exit 7")
        return await proc.wait()

    assert asyncio.run(run()) == 7
    assert _hidden_startupinfo(_ours(spy, "exit 7"))


@_needs_windows
def test_multiprocessing_spawn_goes_through_the_guard(stop_spy):
    proc = multiprocessing.get_context("spawn").Process(target=os.getpid)
    with pytest.raises(_Stop):
        proc.start()
    assert _hidden_startupinfo(_ours(stop_spy, "multiprocessing"))


@_needs_windows
@pytest.mark.parametrize("flags", [nw.CREATE_NEW_CONSOLE, nw.DETACHED_PROCESS],
                         ids=["new-console", "detached"])
def test_a_visible_console_request_is_refused_before_anything_starts(
        stop_spy, refusals, flags):
    caught = None
    try:
        subprocess.run(["cmd", "/c", "exit 0"], creationflags=flags)
    except RuntimeError as exc:
        caught = exc
    assert [c for c in stop_spy if "exit 0" in str(c[1])] == []
    assert isinstance(caught, RuntimeError) and "BLOCKED by tests/_no_window.py" in str(caught)
    assert len(refusals()) == 1


@_needs_windows
def test_a_hidden_new_console_starts_with_create_no_window(stop_spy, refusals):
    new_console = nw.CREATE_NEW_CONSOLE
    si = subprocess.STARTUPINFO(dwFlags=subprocess.STARTF_USESHOWWINDOW,
                                wShowWindow=nw.SW_HIDE)
    with pytest.raises(_Stop):
        subprocess.run(["cmd", "/c", "exit 0"], creationflags=new_console, startupinfo=si)
    flags = _flags(_ours(stop_spy, "exit 0"))
    assert refusals() == []
    assert flags & nw.CREATE_NO_WINDOW and not flags & nw.CREATE_NEW_CONSOLE


@_needs_windows
def test_allow_detached_lets_detached_through_on_its_own_thread_only(stop_spy, refusals):
    detached = nw.DETACHED_PROCESS
    with nw.allow_detached():
        with pytest.raises(_Stop):
            subprocess.run(["cmd", "/c", "exit 0"], creationflags=detached)
        other = []

        def elsewhere():
            try:
                subprocess.run(["cmd", "/c", "exit 1"], creationflags=detached)
            except BaseException as exc:
                other.append(exc)

        t = threading.Thread(target=elsewhere)
        t.start()
        t.join(30)
    assert _flags(_ours(stop_spy, "exit 0")) & nw.DETACHED_PROCESS
    assert [c for c in stop_spy if "exit 1" in str(c[1])] == []
    assert len(other) == 1 and isinstance(other[0], RuntimeError)
    assert len(refusals()) == 1
    with pytest.raises(RuntimeError):
        subprocess.run(["cmd", "/c", "exit 0"], creationflags=detached)


@_needs_windows
def test_a_swallowed_refusal_is_still_recorded(stop_spy, refusals):
    new_console = nw.CREATE_NEW_CONSOLE
    try:
        subprocess.Popen(["cmd", "/c", "exit 0"], creationflags=new_console)
    except Exception:
        pass
    assert [c for c in stop_spy if "exit 0" in str(c[1])] == []
    assert len(refusals()) == 1 and "CREATE_NEW_CONSOLE" in refusals()[0]


@_needs_windows
def test_startfile_and_webbrowser_are_refused(tmp_path, refusals):
    assert os.startfile is nw.startfile and webbrowser.open is nw.browser_open
    with pytest.raises(RuntimeError, match="os.startfile"):
        os.startfile(str(tmp_path / "missing.txt"))
    with pytest.raises(RuntimeError, match="webbrowser.open"):
        webbrowser.open("http://127.0.0.1:9/")
    with pytest.raises(RuntimeError, match="webbrowser.open"):
        webbrowser.open_new_tab("http://127.0.0.1:9/")
    assert len(refusals()) == 3


@_needs_windows
def test_tk_windows_start_withdrawn_and_stay_withdrawn():
    tkinter = pytest.importorskip("tkinter")
    assert tkinter.Tk.__init__ is nw.tk_init
    assert tkinter.Toplevel.__init__ is nw.toplevel_init
    assert tkinter.Wm.deiconify is nw.deiconify and tkinter.Wm.wm_deiconify is nw.deiconify
    root = build_tk_root(tkinter.Tk)
    try:
        assert root.state() == "withdrawn"
        top = tkinter.Toplevel(root)
        assert top.state() == "withdrawn"
        root.deiconify()
        top.deiconify()
        assert root.state() == "withdrawn" and top.state() == "withdrawn"
        root.update()
        assert root.state() == "withdrawn" and top.state() == "withdrawn"
        assert not root.winfo_ismapped() and not top.winfo_ismapped()
    finally:
        root.destroy()
