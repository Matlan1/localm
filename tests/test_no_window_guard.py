# SPDX-License-Identifier: AGPL-3.0-or-later
"""The suite-wide guard in tests/_no_window.py: every process a test starts on
Windows runs with CREATE_NO_WINDOW and a hidden STARTUPINFO, and a request for
a visible window is refused before anything starts."""

import asyncio
import multiprocessing
import os
import subprocess
import sys
import webbrowser

import pytest

from tests import _no_window as nw

_needs_windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows process creation")


class _Stop(Exception):
    pass


@pytest.fixture
def refusals():
    """Yield a function returning the reasons refused since the test started,
    then drop those refusals so the suite-wide fixture does not fail the test
    for the refusals it asked for on purpose."""
    before = dict(nw.HITS)

    def new():
        return [reason for (reason, _), n in nw.HITS.items()
                if n > before.get((reason, _), 0)]

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


def _ours(calls, needle):
    """The one recorded call whose command line contains *needle*."""
    mine = [c for c in calls if needle in str(c[1])]
    assert len(mine) == 1, [str(c[1]) for c in calls]
    return mine[0]


def _hidden_startupinfo(call):
    si = call[8]
    return bool(si.dwFlags & nw.STARTF_USESHOWWINDOW) and si.wShowWindow == nw.SW_HIDE


# --------------------------------------------------------------------------- #
#  hidden(): the decision, on every platform                                   #
# --------------------------------------------------------------------------- #

def test_a_plain_spawn_gains_create_no_window_and_sw_hide():
    flags, si, refusal = nw.hidden(0x00000200, None)
    assert refusal is None
    assert flags == 0x00000200 | nw.CREATE_NO_WINDOW
    assert si.dwFlags & nw.STARTF_USESHOWWINDOW and si.wShowWindow == nw.SW_HIDE


def test_detached_process_is_refused():
    _, _, refusal = nw.hidden(nw.DETACHED_PROCESS, None)
    assert refusal and "DETACHED_PROCESS" in refusal


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
    assert nw.armed()
    assert _winapi.CreateProcess is nw.create_process
    assert os.startfile is nw.startfile
    assert webbrowser.open is nw.browser_open
    assert os.system is nw.system


@_needs_windows
def test_cmd_exit_reaches_createprocess_with_create_no_window(spy):
    assert subprocess.run(["cmd", "/c", "exit 0"]).returncode == 0
    call = _ours(spy, "exit 0")
    assert _flags(call) & nw.CREATE_NO_WINDOW
    assert _hidden_startupinfo(call)


@_needs_windows
def test_the_child_runs_on_a_console_of_its_own(spy):
    probe = ("import ctypes; a = (ctypes.c_uint * 16)(); "
             "print(ctypes.windll.kernel32.GetConsoleProcessList(a, 16))")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "1"


@_needs_windows
def test_an_unredirected_child_still_writes_to_the_tests_stdout(capfd, spy):
    subprocess.run([sys.executable, "-c", "print('reached the parent')"])
    assert "reached the parent" in capfd.readouterr().out
    assert _flags(_ours(spy, "reached the parent")) & nw.CREATE_NO_WINDOW


@_needs_windows
def test_asyncio_subprocesses_get_create_no_window(spy):
    async def run():
        proc = await asyncio.create_subprocess_exec("cmd", "/c", "exit 7")
        return await proc.wait()

    assert asyncio.run(run()) == 7
    call = _ours(spy, "exit 7")
    assert _flags(call) & nw.CREATE_NO_WINDOW
    assert _hidden_startupinfo(call)


@_needs_windows
def test_multiprocessing_spawn_gets_create_no_window(stop_spy):
    proc = multiprocessing.get_context("spawn").Process(target=os.getpid)
    with pytest.raises(_Stop):
        proc.start()
    call = _ours(stop_spy, "multiprocessing")
    assert _flags(call) & nw.CREATE_NO_WINDOW
    assert _hidden_startupinfo(call)


@_needs_windows
def test_os_system_runs_hidden_and_keeps_its_exit_code(spy):
    assert nw.system("exit 3") == 3
    assert _flags(_ours(spy, "exit 3")) & nw.CREATE_NO_WINDOW


@_needs_windows
@pytest.mark.parametrize("flags", [subprocess.CREATE_NEW_CONSOLE, subprocess.DETACHED_PROCESS],
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
    si = subprocess.STARTUPINFO(dwFlags=subprocess.STARTF_USESHOWWINDOW,
                                wShowWindow=nw.SW_HIDE)
    with pytest.raises(_Stop):
        subprocess.run(["cmd", "/c", "exit 0"], creationflags=subprocess.CREATE_NEW_CONSOLE,
                       startupinfo=si)
    flags = _flags(_ours(stop_spy, "exit 0"))
    assert refusals() == []
    assert flags & nw.CREATE_NO_WINDOW and not flags & subprocess.CREATE_NEW_CONSOLE


@_needs_windows
def test_a_swallowed_refusal_still_fails_the_test(stop_spy, refusals):
    try:
        subprocess.Popen(["cmd", "/c", "exit 0"], creationflags=subprocess.CREATE_NEW_CONSOLE)
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
    assert len(refusals()) == 2
