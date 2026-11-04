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
from pathlib import Path

import pytest

from tests import _no_window as nw
from tests._exe_stub import write_exe_stub
from tests._tk_root import build_tk_root, collect_tk_garbage  # noqa: F401

_needs_windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows process creation")


class _Stop(Exception):
    pass


_Refused = pytest.fail.Exception


@pytest.fixture(autouse=True)
def _guard_is_armed_before_the_test_body():
    """On Windows, fail before the test body runs when the guard is not in place,
    so a test here cannot reach a real window through an unarmed guard."""
    if sys.platform == "win32":
        import _winapi
        assert nw.armed()
        assert _winapi.CreateProcess is nw.create_process
        assert os.startfile is nw.startfile and webbrowser.open is nw.browser_open
    yield


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
def test_a_visible_console_request_is_refused_before_anything_starts(stop_spy, flags):
    with pytest.raises(_Refused, match="BLOCKED by tests/_no_window.py"):
        subprocess.run(["cmd", "/c", "exit 0"], creationflags=flags)
    assert [c for c in stop_spy if "exit 0" in str(c[1])] == []


@_needs_windows
def test_a_hidden_new_console_starts_with_create_no_window(stop_spy):
    new_console = nw.CREATE_NEW_CONSOLE
    si = subprocess.STARTUPINFO(dwFlags=subprocess.STARTF_USESHOWWINDOW,
                                wShowWindow=nw.SW_HIDE)
    with pytest.raises(_Stop):
        subprocess.run(["cmd", "/c", "exit 0"], creationflags=new_console, startupinfo=si)
    flags = _flags(_ours(stop_spy, "exit 0"))
    assert flags & nw.CREATE_NO_WINDOW and not flags & nw.CREATE_NEW_CONSOLE


def test_allow_window_nests_and_ends_with_its_block():
    assert not nw.window_allowed()
    with nw.allow_window():
        assert nw.window_allowed()
        with nw.allow_window():
            assert nw.window_allowed()
        assert nw.window_allowed()
    assert not nw.window_allowed()


def test_allow_window_ends_when_its_block_raises():
    with pytest.raises(_Stop):
        with nw.allow_window():
            raise _Stop
    assert not nw.window_allowed()


def test_importing_the_tests_package_without_pytest_arms_nothing_and_imports_nothing():
    code = ("import sys; from tests import _no_console_window, _no_window; "
            "print(_no_window.armed(), 'pytest' in sys.modules, 'tkinter' in sys.modules)")
    root = str(Path(__file__).resolve().parents[1])
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": root}, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "False", "False"]


@_needs_windows
def test_allow_window_stops_the_console_guard_adding_a_flag(monkeypatch):
    from tests import _no_console_window as guard
    calls = []

    def record(*args):
        calls.append(args)
        raise _Stop

    monkeypatch.setattr(guard, "has_console", lambda: False)
    monkeypatch.setattr(guard, "_real_create", record)
    with pytest.raises(_Stop):
        subprocess.run(["cmd", "/c", "exit 0"])
    assert _flags(_ours(calls, "exit 0")) & nw.CREATE_NO_WINDOW
    calls.clear()
    with nw.allow_window():
        with pytest.raises(_Stop):
            subprocess.run(["cmd", "/c", "exit 0"])
    assert not _flags(_ours(calls, "exit 0")) & nw.CREATE_NO_WINDOW


@_needs_windows
@pytest.mark.parametrize("flags", [nw.CREATE_NEW_CONSOLE, nw.DETACHED_PROCESS],
                         ids=["new-console", "detached"])
def test_allow_window_passes_a_console_request_through_unchanged_on_every_thread(
        stop_spy, flags):
    other = []

    def elsewhere():
        try:
            subprocess.run(["cmd", "/c", "exit 1"], creationflags=flags)
        except BaseException as exc:
            other.append(exc)

    with nw.allow_window():
        with pytest.raises(_Stop):
            subprocess.run(["cmd", "/c", "exit 0"], creationflags=flags)
        t = threading.Thread(target=elsewhere)
        t.start()
        t.join(30)
    mine = _ours(stop_spy, "exit 0")
    assert _flags(mine) & flags and not _flags(mine) & nw.CREATE_NO_WINDOW
    assert not _hidden_startupinfo(mine)
    assert _flags(_ours(stop_spy, "exit 1")) & flags
    assert len(other) == 1 and isinstance(other[0], _Stop)
    with pytest.raises(_Refused):
        subprocess.run(["cmd", "/c", "exit 2"], creationflags=flags)
    assert [c for c in stop_spy if "exit 2" in str(c[1])] == []


@_needs_windows
def test_allow_window_calls_the_real_startfile_and_browser(monkeypatch):
    opened = []
    monkeypatch.setattr(nw, "_real_startfile", lambda path, *a, **k: opened.append(("file", path)))
    monkeypatch.setattr(nw, "_real_browser_open", lambda url, *a, **k: opened.append(("url", url)))
    with nw.allow_window():
        nw.startfile("x.txt")
        webbrowser.open("http://127.0.0.1:9/")
    assert opened == [("file", "x.txt"), ("url", "http://127.0.0.1:9/")]
    with pytest.raises(_Refused):
        nw.startfile("x.txt")
    assert len(opened) == 2


@_needs_windows
def test_allow_window_lets_tk_windows_show(monkeypatch):
    calls = []

    class _Fake:
        def withdraw(self):
            calls.append("withdraw")

    monkeypatch.setattr(nw, "_real_tk_init", lambda self, *a, **k: calls.append("tk"))
    monkeypatch.setattr(nw, "_real_toplevel_init", lambda self, *a, **k: calls.append("top"))
    monkeypatch.setattr(nw, "_real_deiconify", lambda self: calls.append("deiconify"))
    monkeypatch.setattr(nw, "_real_iconify", lambda self: calls.append("iconify"))
    monkeypatch.setattr(nw, "_real_state", lambda self, new=None: calls.append(("state", new)))
    monkeypatch.setattr(nw, "_real_dialog_show", lambda self, **k: calls.append("dialog"))
    monkeypatch.setattr(nw, "_real_wait_visibility", lambda self, w=None: calls.append("wait"))
    with nw.allow_window():
        nw.tk_init(_Fake())
        nw.toplevel_init(_Fake())
        nw.deiconify(_Fake())
        nw.iconify(_Fake())
        nw.state(_Fake(), "normal")
        nw.dialog_show(_Fake())
        nw.wait_visibility(_Fake())
    assert calls == ["tk", "top", "deiconify", "iconify", ("state", "normal"), "dialog", "wait"]
    calls.clear()
    nw.tk_init(_Fake())
    nw.toplevel_init(_Fake())
    nw.deiconify(_Fake())
    nw.iconify(_Fake())
    nw.state(_Fake(), "normal")
    assert calls == ["tk", "withdraw", "top", "withdraw"]
    calls.clear()
    nw.state(_Fake())
    nw.state(_Fake(), "withdrawn")
    assert calls == [("state", None), ("state", "withdrawn")]
    calls.clear()
    with pytest.raises(_Refused, match="native Tk dialog"):
        nw.dialog_show(_Fake())
    with pytest.raises(_Refused, match="wait_visibility"):
        nw.wait_visibility(_Fake())
    assert calls == []


@_needs_windows
def test_except_exception_does_not_swallow_a_refusal(stop_spy):
    new_console = nw.CREATE_NEW_CONSOLE
    swallowed = False
    with pytest.raises(_Refused, match="CREATE_NEW_CONSOLE"):
        try:
            subprocess.Popen(["cmd", "/c", "exit 0"], creationflags=new_console)
        except Exception:
            swallowed = True
    assert not swallowed
    assert [c for c in stop_spy if "exit 0" in str(c[1])] == []


@_needs_windows
def test_a_refusal_imports_nothing_when_pytest_is_gone_from_sys_modules(monkeypatch):
    monkeypatch.delitem(sys.modules, "pytest")
    with pytest.raises(_Refused, match="os.startfile"):
        nw.startfile("x.txt")
    assert "pytest" not in sys.modules


@_needs_windows
def test_startfile_and_webbrowser_are_refused(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(nw, "_real_startfile", lambda *a, **k: opened.append(a))
    monkeypatch.setattr(nw, "_real_browser_open", lambda *a, **k: opened.append(a))
    with pytest.raises(_Refused, match="os.startfile"):
        nw.startfile(str(tmp_path / "missing.txt"))
    with pytest.raises(_Refused, match="webbrowser.open"):
        webbrowser.open("http://127.0.0.1:9/")
    with pytest.raises(_Refused, match="webbrowser.open"):
        webbrowser.open_new_tab("http://127.0.0.1:9/")
    assert opened == []


@_needs_windows
def test_tcl_without_tk_still_works():
    tkinter = pytest.importorskip("tkinter")
    interp = tkinter.Tcl()
    assert interp.tk.eval("expr {2 + 3}") == "5"


@_needs_windows
def test_tk_windows_start_withdrawn_and_stay_withdrawn(monkeypatch):
    tkinter = pytest.importorskip("tkinter")
    assert tkinter.Tk.__init__ is nw.tk_init
    assert tkinter.Toplevel.__init__ is nw.toplevel_init
    assert tkinter.Wm.deiconify is nw.deiconify and tkinter.Wm.wm_deiconify is nw.deiconify
    assert tkinter.Wm.iconify is nw.iconify and tkinter.Wm.wm_iconify is nw.iconify
    assert tkinter.Wm.state is nw.state and tkinter.Wm.wm_state is nw.state
    reached = []
    monkeypatch.setattr(nw, "_real_deiconify", lambda self: reached.append("deiconify"))
    monkeypatch.setattr(nw, "_real_iconify", lambda self: reached.append("iconify"))
    root = build_tk_root(tkinter.Tk)
    try:
        assert root.state() == "withdrawn"
        top = tkinter.Toplevel(root)
        assert top.state() == "withdrawn"
        root.deiconify()
        top.deiconify()
        root.iconify()
        root.state("normal")
        assert reached == []
        assert root.state() == "withdrawn" and top.state() == "withdrawn"
        root.update()
        assert root.state() == "withdrawn" and top.state() == "withdrawn"
        assert not root.winfo_ismapped() and not top.winfo_ismapped()
    finally:
        root.destroy()


# --------------------------------------------------------------------------- #
#  A text file named *.exe that a child could launch                           #
# --------------------------------------------------------------------------- #

def _fake(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not a program", encoding="utf-8")
    return path


def _real_stub(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MZ" + b"\0" * 62)
    return path


def _scan(tmp_path, command, *, env=None, cwd=None, application=None):
    return nw.fake_exe_in_reach(application, command, env, cwd,
                                roots=[nw._norm(tmp_path.parent)])


def test_an_exe_without_the_mz_header_in_the_cwd_is_found(tmp_path):
    fake = _fake(tmp_path / "uv.exe")
    assert _scan(tmp_path, "cmd /c setup.bat", cwd=str(tmp_path)) == str(fake)


@pytest.mark.parametrize("where", [".uv/uv.exe", ".venv/Scripts/localm.exe",
                                   ".python/cpython/python.exe"])
def test_the_folders_a_setup_script_uses_are_scanned(tmp_path, where):
    fake = _fake(tmp_path / where)
    assert _scan(tmp_path, "cmd /c setup.bat", cwd=str(tmp_path)) == str(fake)


def test_an_empty_exe_is_not_a_program(tmp_path):
    (tmp_path / "uv.exe").write_bytes(b"")
    assert _scan(tmp_path, "cmd /c x.bat", cwd=str(tmp_path)) == str(tmp_path / "uv.exe")


def test_a_real_stub_passes(tmp_path):
    _real_stub(tmp_path / "uv.exe")
    _real_stub(tmp_path / ".venv" / "Scripts" / "localm.exe")
    assert _scan(tmp_path, "cmd /c setup.bat", cwd=str(tmp_path)) is None


def test_the_childs_own_path_is_scanned_not_only_ours(tmp_path):
    fake = _fake(tmp_path / "tools" / "uv.exe")
    other = tmp_path / "other"
    other.mkdir()
    env = {"Path": os.pathsep.join([str(tmp_path / "tools"), r"C:\Windows\System32"])}
    assert _scan(tmp_path, "cmd /c x.bat", env=env, cwd=str(other)) == str(fake)
    assert _scan(tmp_path, "cmd /c x.bat", env={"Path": r"C:\Windows\System32"},
                 cwd=str(other)) is None


def test_a_path_folder_is_scanned_top_level_only(tmp_path):
    _fake(tmp_path / "tools" / "sub" / "uv.exe")
    other = tmp_path / "other"
    other.mkdir()
    assert _scan(tmp_path, "cmd /c x.bat", env={"PATH": str(tmp_path / "tools")},
                 cwd=str(other)) is None


def test_the_program_given_as_a_path_is_checked(tmp_path):
    fake = _fake(tmp_path / "prog" / "tool.exe")
    other = tmp_path / "other"
    other.mkdir()
    assert _scan(tmp_path, f'"{fake}" --version', cwd=str(other)) == str(fake)
    assert _scan(tmp_path, None, application=str(fake), cwd=str(other)) == str(fake)


def test_the_folder_of_a_script_on_the_command_line_is_scanned(tmp_path):
    fake = _fake(tmp_path / "scripts" / ".uv" / "uv.exe")
    other = tmp_path / "other"
    other.mkdir()
    script = tmp_path / "scripts" / "setup.bat"
    assert _scan(tmp_path, f'cmd.exe /d /c "{script}"', cwd=str(other)) == str(fake)


def test_folders_outside_the_temp_roots_are_never_scanned(tmp_path):
    _fake(tmp_path / "uv.exe")
    assert nw.fake_exe_in_reach(None, "cmd /c x.bat", {"PATH": str(tmp_path)}, str(tmp_path),
                                roots=[nw._norm(tmp_path / "elsewhere")]) is None


def test_a_temp_root_itself_is_not_scanned(tmp_path):
    _fake(tmp_path / "uv.exe")
    assert nw.fake_exe_in_reach(None, "cmd /c x.bat", None, str(tmp_path),
                                roots=[nw._norm(tmp_path)]) is None


def test_the_scan_stops_at_its_depth(tmp_path):
    too_deep = tmp_path.joinpath(*(["d"] * (nw.SCAN_DEPTH + 1)))
    _fake(too_deep / "uv.exe")
    assert _scan(tmp_path, "cmd /c x.bat", cwd=str(tmp_path)) is None
    last = tmp_path.joinpath(*(["d"] * nw.SCAN_DEPTH))
    _fake(last / "uv.exe")
    assert _scan(tmp_path, "cmd /c x.bat", cwd=str(tmp_path)) == str(last / "uv.exe")


def test_the_scan_stops_at_its_entry_limit(tmp_path, monkeypatch):
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("x", encoding="utf-8")
    _fake(tmp_path / "zz" / "uv.exe")
    monkeypatch.setattr(nw, "SCAN_ENTRY_LIMIT", 3)
    assert _scan(tmp_path, "cmd /c x.bat", cwd=str(tmp_path)) is None


def test_a_rewritten_exe_is_looked_at_again(tmp_path):
    exe = _real_stub(tmp_path / "uv.exe")
    assert _scan(tmp_path, "cmd /c x.bat", cwd=str(tmp_path)) is None
    exe.write_text("now text", encoding="utf-8")
    assert _scan(tmp_path, "cmd /c x.bat", cwd=str(tmp_path)) == str(exe)


@_needs_windows
def test_the_pytest_temp_folder_counts_as_a_temp_root(tmp_path):
    assert nw.under_temp(tmp_path)
    assert not nw.under_temp(Path(os.environ["SystemRoot"]) / "System32")
    assert not nw.under_temp(Path(__file__).resolve().parent)


@_needs_windows
@pytest.mark.parametrize("command", [
    ["cmd", "/c", r".\setup.bat"],
    ["cmd.exe", "/d", "/c", "exit 0"],
    ["powershell", "-NoProfile", "-Command", "exit 0"],
])
def test_a_text_exe_in_the_cwd_is_refused_before_any_child_starts(stop_spy, tmp_path, command):
    fake = _fake(tmp_path / ".uv" / "uv.exe")
    with pytest.raises(_Refused, match="not a program") as info:
        subprocess.run(command, cwd=tmp_path)
    assert str(fake) in str(info.value) and "Add-Type" in str(info.value)
    assert stop_spy == []


@_needs_windows
def test_a_text_exe_on_the_childs_path_is_refused_before_any_child_starts(stop_spy, tmp_path):
    tools = tmp_path / "tools"
    _fake(tools / "uv.exe")
    other = tmp_path / "other"
    other.mkdir()
    env = {**os.environ, "PATH": f"{tools};{os.environ['PATH']}"}
    with pytest.raises(_Refused, match="not a program"):
        subprocess.run(["cmd.exe", "/c", "exit 0"], cwd=other, env=env)
    assert stop_spy == []


@_needs_windows
def test_a_text_exe_started_directly_is_refused_before_any_child_starts(stop_spy, tmp_path):
    fake = _fake(tmp_path / "prog" / "tool.exe")
    with pytest.raises(_Refused, match="not a program"):
        subprocess.Popen([str(fake), "--version"])
    assert stop_spy == []


@_needs_windows
def test_the_same_layout_with_real_stubs_starts_the_child(stop_spy, tmp_path):
    _real_stub(tmp_path / ".uv" / "uv.exe")
    _real_stub(tmp_path / ".venv" / "Scripts" / "localm.exe")
    with pytest.raises(_Stop):
        subprocess.run(["cmd", "/c", r".\setup.bat"], cwd=tmp_path)
    assert len(stop_spy) == 1


@_needs_windows
def test_allow_window_skips_the_fake_exe_scan(stop_spy, tmp_path):
    _fake(tmp_path / "uv.exe")
    with nw.allow_window():
        with pytest.raises(_Stop):
            subprocess.run(["cmd", "/c", "exit 0"], cwd=tmp_path)
    assert len(stop_spy) == 1


@_needs_windows
def test_write_exe_stub_makes_a_file_the_guard_accepts(stop_spy, tmp_path):
    exe = write_exe_stub(tmp_path / ".uv" / "uv.exe")
    assert exe.read_bytes()[:2] == nw.PE_MAGIC
    with pytest.raises(_Stop):
        subprocess.run(["cmd", "/c", "exit 0"], cwd=tmp_path)
    assert len(stop_spy) == 1
