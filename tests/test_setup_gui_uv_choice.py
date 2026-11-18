# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup-gui.bat: which uv opens the window, with no question in a console.

The launcher asks nothing. It uses the folder's own .\\.uv\\uv.exe, reuses the
per-user uv for an earlier Shared install (an install record and no .\\.uv), and
otherwise installs uv into .\\.uv. The window's first page asks Portable or
Shared; when Shared is chosen it exits 46 (the per-user uv was there) or 47 (it
installed it), and the launcher removes what this run put into the folder and
opens the window again on the per-user uv.

The real script runs from a throwaway clone with the uv installer and the window
each replaced by a stub batch file: the installer stub drops a uv where the
real one would; the window stub reports what it was given, creates .\\.python
and .\\.cache like `uv run` does, and exits with the next code of
WINDOW_CODES. No network is touched. LOCALM_SETUP_GUI_INPLACE=1 keeps the
script in this console; the hidden-start tests stand a recording stub in for
the console host and replace the message box and "getting ready" window with
echo lines.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from tests._exe_stub import write_exe_stub

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(os.name != "nt", reason="setup-gui.bat runs only on Windows")

_UV_RUN = '"%UVEXE%" run --no-project --isolated --python 3.12 python "installer\\gui.py"'
_INSTALL_PREFIX = 'powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference'
_BOX_PREFIX = 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName System.Windows.Forms; $m='
_SPLASH_PREFIX = 'start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-Type'
_CONHOST = 'start "" "%SystemRoot%\\System32\\conhost.exe" --headless '

_STUB_INSTALLER = (
    "@echo off\r\n"
    "call :say\r\n"
    'if defined STUB_RC exit /b %STUB_RC%\r\n'
    'if not "%STUB_CREATES%"=="1" exit /b 0\r\n'
    'mkdir "%UV_INSTALL_DIR%" 2>nul\r\n'
    'type nul > "%UV_INSTALL_DIR%\\uv.exe"\r\n'
    "exit /b 0\r\n"
    ":say\r\n"
    "setlocal EnableDelayedExpansion\r\n"
    "echo INSTALLER-RAN INSTALL_DIR=[!UV_INSTALL_DIR!] UNMANAGED=[!UV_UNMANAGED_INSTALL!]\r\n"
    "endlocal\r\n"
    "goto :eof\r\n")
_STUB_WINDOW = (
    "@echo off\r\n"
    "call :fields\r\n"
    'if defined STUB_REPORT call :fields >> "%STUB_REPORT%"\r\n'
    'if defined UV_PYTHON_INSTALL_DIR mkdir "%UV_PYTHON_INSTALL_DIR%" 2>nul\r\n'
    'if defined UV_CACHE_DIR mkdir "%UV_CACHE_DIR%" 2>nul\r\n'
    'set "CODE=0"\r\n'
    'for /f "tokens=1*" %%a in ("%WINDOW_CODES%") do set "CODE=%%a"& set "WINDOW_CODES=%%b"\r\n'
    'if "%CODE%"=="47" mkdir "%USERPROFILE%\\.local\\bin" 2>nul\r\n'
    'if "%CODE%"=="47" type nul > "%USERPROFILE%\\.local\\bin\\uv.exe"\r\n'
    'if not defined WAIT_FOR_GO goto stub_exit\r\n'
    'if not "%CODE%"=="46" goto stub_exit\r\n'
    'type nul > "%UV_PYTHON_INSTALL_DIR%\\held"\r\n'
    'type nul > "%WAIT_FOR_GO%.ready"\r\n'
    ":stub_wait\r\n"
    'if exist "%WAIT_FOR_GO%" goto stub_exit\r\n'
    "ping -n 2 127.0.0.1 >nul\r\n"
    "goto stub_wait\r\n"
    ":stub_exit\r\n"
    "exit /b %CODE%\r\n"
    ":fields\r\n"
    "setlocal EnableDelayedExpansion\r\n"
    "echo WINDOW-UV=!UVEXE!\r\n"
    "echo WINDOW-STORE=!LOCALM_GUI_STORE!\r\n"
    "echo WINDOW-PYDIR=!UV_PYTHON_INSTALL_DIR!\r\n"
    "echo WINDOW-CACHEDIR=!UV_CACHE_DIR!\r\n"
    "echo WINDOW-GUIUV=!LOCALM_GUI_UV!\r\n"
    "echo WINDOW-INSTALLED=!LOCALM_GUI_UV_INSTALLED!\r\n"
    "echo WINDOW-BOOTSTRAP=!LOCALM_GUI_BOOTSTRAP!\r\n"
    "echo WINDOW-REPOINTED=!LOCALM_GUI_REPOINTED!\r\n"
    "echo WINDOW-SPLASH=!LOCALM_GUI_SPLASH!\r\n"
    "echo WINDOW-LOG=!LOCALM_GUI_LOG!\r\n"
    "endlocal\r\n"
    "goto :eof\r\n")
_STUB_CONHOST = (
    "@echo off\r\n"
    "echo CONHOST-RAN %*\r\n"
    "%*\r\n"
    "exit /b 0\r\n")


def _probe_text(hidden: bool = False) -> str:
    text = (ROOT / "setup-gui.bat").read_text(encoding="utf-8")
    assert text.count(_UV_RUN) == 1, "the window launch line moved"
    lines = text.splitlines()
    for prefix, stand_in in ((_INSTALL_PREFIX, 'call "%STUB_INSTALLER%"'),
                             (_BOX_PREFIX, "echo BOX-%LOCALM_MSG_ICON%& set LOCALM_MSG"),
                             (_SPLASH_PREFIX, "echo SPLASH-START=%LOCALM_GUI_SPLASH%")):
        found = [i for i, line in enumerate(lines) if line.startswith(prefix)]
        assert len(found) == 1, f"{prefix[:60]} moved"
        lines[found[0]] = stand_in
    conhost = [i for i, line in enumerate(lines) if line.startswith(_CONHOST)]
    assert len(conhost) == 1, "the start without a console moved"
    if hidden:
        lines[conhost[0]] = lines[conhost[0]].replace(_CONHOST, 'call "%STUB_CONHOST%" ')
    text = "\n".join(lines).replace(_UV_RUN, 'call "%STUB_WINDOW%"')
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


def _run(tmp_path: Path, *, system_uv: bool = False, contained_uv: bool = False,
         installer_creates_uv: bool = True, installer_rc: str = "", codes: str = "0",
         record: bool = False, existing=(), hidden: bool = False, clone_name: str = "clone",
         marker: str = ""):
    clone = tmp_path / clone_name
    clone.mkdir()
    (clone / "probe.bat").write_text(_probe_text(hidden), encoding="utf-8", newline="")
    if contained_uv:
        (clone / ".uv").mkdir()
        write_exe_stub(clone / ".uv" / "uv.exe")
    if record:
        (clone / ".localm-install.json").write_text("{}", encoding="utf-8")
    if marker:
        (clone / ".localm-gui-bootstrap").write_text(marker + "\n", encoding="ascii")
    for name in existing:
        (clone / name).mkdir()
        (clone / name / "keep").write_bytes(b"x")
    home = tmp_path / "home"
    home.mkdir()
    stubs = {}
    for name, body in (("installer", _STUB_INSTALLER), ("window", _STUB_WINDOW),
                       ("conhost", _STUB_CONHOST)):
        stubs[name] = tmp_path / f"stub-{name}.bat"
        stubs[name].write_text(body, encoding="ascii", newline="")
    system32 = Path(os.environ["SystemRoot"]) / "System32"
    path = str(system32)
    sysuv = tmp_path / "systemuv"
    if system_uv:
        sysuv.mkdir()
        (sysuv / "uv.cmd").write_text("@exit /b 1\r\n", encoding="ascii")
        path = str(sysuv) + os.pathsep + path
    env = dict(os.environ, PATH=path, USERPROFILE=str(home), HOMEDRIVE=home.drive,
               HOMEPATH=str(home)[len(home.drive):], STUB_INSTALLER=str(stubs["installer"]),
               STUB_WINDOW=str(stubs["window"]), STUB_CONHOST=str(stubs["conhost"]),
               STUB_CREATES="1" if installer_creates_uv else "0", WINDOW_CODES=codes)
    for var in ("LOCALM_GUI_STORE", "LOCALM_GUI_UV", "LOCALM_GUI_UV_INSTALLED",
                "LOCALM_GUI_BOOTSTRAP", "LOCALM_GUI_REPOINTED", "LOCALM_GUI_SPLASH",
                "LOCALM_SETUP_GUI_HIDDEN", "LOCALM_SETUP_GUI_INPLACE", "XDG_BIN_HOME",
                "UV_PYTHON_INSTALL_DIR", "UV_CACHE_DIR", "UV_INSTALL_DIR",
                "UV_UNMANAGED_INSTALL", "STUB_RC"):
        env.pop(var, None)
    if installer_rc:
        env["STUB_RC"] = installer_rc
    if not hidden:
        env["LOCALM_SETUP_GUI_INPLACE"] = "1"
    report = tmp_path / "window-report.txt"
    env["STUB_REPORT"] = str(report)
    out = subprocess.run(["cmd", "/c", str(clone / "probe.bat")], cwd=str(clone), env=env,
                         capture_output=True, text=True, timeout=120,
                         stdin=subprocess.DEVNULL)
    if hidden and report.is_file():
        out.stdout += report.read_text(encoding="utf-8", errors="replace")
    return clone, sysuv, home, out


def _fields(out, name: str) -> list:
    return [line.split("=", 1)[1].strip() for line in out.stdout.splitlines()
            if line.startswith(name + "=")]


def _field(out, name: str, n: int = 0) -> str:
    found = _fields(out, name)
    assert len(found) > n, f"{name} not reported:\n{out.stdout}\n{out.stderr}"
    return found[n]


def _installer_line(out) -> str:
    return next(ln for ln in out.stdout.splitlines() if ln.startswith("INSTALLER-RAN"))


def test_a_fresh_folder_installs_uv_into_the_folder_without_asking(tmp_path):
    clone, _, _, out = _run(tmp_path, system_uv=True)
    dot_uv = str(clone / ".uv")
    assert _installer_line(out) == f"INSTALLER-RAN INSTALL_DIR=[{dot_uv}] UNMANAGED=[{dot_uv}]"
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv.exe")
    assert _field(out, "WINDOW-GUIUV") == str(clone / ".uv" / "uv.exe")
    assert _field(out, "WINDOW-STORE") == "portable"
    assert _field(out, "WINDOW-PYDIR") == str(clone / ".python")
    assert _field(out, "WINDOW-CACHEDIR") == str(clone / ".cache")
    assert _field(out, "WINDOW-BOOTSTRAP").split() == ["uv", "python", "cache"]
    assert _field(out, "WINDOW-REPOINTED") == ""
    assert "Pick 1 or 2" not in out.stdout and "[Y/n]" not in out.stdout
    assert out.returncode == 0, (out.stdout, out.stderr)


def test_a_contained_uv_is_used_and_not_named_as_put_there_by_this_run(tmp_path):
    clone, _, _, out = _run(tmp_path, system_uv=True, contained_uv=True,
                            existing=(".python",))
    assert "INSTALLER-RAN" not in out.stdout
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv.exe")
    assert _field(out, "WINDOW-BOOTSTRAP").split() == ["cache"]
    assert out.returncode == 0, (out.stdout, out.stderr)


def test_an_earlier_shared_install_reuses_the_per_user_uv(tmp_path):
    clone, sysuv, _, out = _run(tmp_path, system_uv=True, record=True)
    assert "INSTALLER-RAN" not in out.stdout and not (clone / ".uv").exists()
    assert _field(out, "WINDOW-UV") == str(sysuv / "uv.cmd")
    assert _field(out, "WINDOW-STORE") == "shared"
    assert _field(out, "WINDOW-PYDIR") == "", "Shared must not point uv's Python folder into the clone"
    assert _field(out, "WINDOW-BOOTSTRAP") == ""


def test_an_install_record_with_no_per_user_uv_still_installs_into_the_folder(tmp_path):
    clone, _, _, out = _run(tmp_path, record=True)
    assert "INSTALLER-RAN" in out.stdout
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv.exe")
    assert _field(out, "WINDOW-STORE") == "portable"


def test_shared_removes_what_this_run_put_in_the_folder_and_reopens_on_the_per_user_uv(tmp_path):
    clone, sysuv, _, out = _run(tmp_path, system_uv=True, codes="46 0")
    assert _field(out, "WINDOW-UV", 0) == str(clone / ".uv" / "uv.exe")
    assert _field(out, "WINDOW-UV", 1) == str(sysuv / "uv.cmd")
    assert _field(out, "WINDOW-GUIUV", 1) == str(sysuv / "uv.cmd")
    assert _field(out, "WINDOW-STORE", 1) == "shared"
    assert _field(out, "WINDOW-REPOINTED", 1) == "1"
    assert _field(out, "WINDOW-PYDIR", 1) == ""
    assert _field(out, "WINDOW-CACHEDIR", 1) == ""
    assert _field(out, "WINDOW-BOOTSTRAP", 1) == ""
    assert _field(out, "WINDOW-INSTALLED", 1) == ""
    for name in (".uv", ".python", ".cache"):
        assert not (clone / name).exists(), f"{name} was left behind\n{out.stdout}"
    assert out.returncode == 0, (out.stdout, out.stderr)


def test_shared_keeps_folders_this_run_did_not_create(tmp_path):
    clone, _, _, out = _run(tmp_path, system_uv=True, contained_uv=True,
                            existing=(".python", ".cache"), codes="46 0")
    assert _field(out, "WINDOW-BOOTSTRAP", 0) == ""
    for name in (".uv", ".python", ".cache"):
        assert (clone / name).is_dir(), f"{name} was removed but this run did not create it"
    assert _field(out, "WINDOW-STORE", 1) == "shared"


def test_shared_with_a_uv_the_window_installed_says_so_to_the_window(tmp_path):
    clone, _, home, out = _run(tmp_path, codes="47 0")
    per_user = str(home / ".local" / "bin" / "uv.exe")
    assert _field(out, "WINDOW-UV", 1) == per_user
    assert _field(out, "WINDOW-INSTALLED", 1) == "1"
    assert _field(out, "WINDOW-STORE", 1) == "shared"
    assert not (clone / ".uv").exists()
    assert out.returncode == 0, (out.stdout, out.stderr)


def test_shared_with_no_per_user_uv_after_all_says_uv_is_not_callable(tmp_path):
    clone, _, _, out = _run(tmp_path, codes="46 0")
    assert len(_fields(out, "WINDOW-UV")) == 1, "the window reopened on no uv"
    assert "uv still is not callable" in out.stdout
    assert "per-user uv" in out.stdout
    assert (clone / ".uv" / "uv.exe").is_file(), "the folder's uv was removed with nothing to use"
    assert out.returncode == 1, (out.stdout, out.stderr)


def test_what_this_run_put_in_the_folder_is_recorded_for_a_later_run(tmp_path):
    clone, _, _, out = _run(tmp_path)
    record = (clone / ".localm-gui-bootstrap").read_text(encoding="ascii").split()
    assert record == ["uv", "python", "cache"], out.stdout


def test_shared_also_removes_what_an_earlier_run_put_in_the_folder(tmp_path):
    clone, sysuv, _, out = _run(tmp_path, system_uv=True, contained_uv=True,
                                existing=(".python", ".cache"), marker="uv python cache",
                                codes="46 0")
    assert _field(out, "WINDOW-BOOTSTRAP", 0).split() == ["uv", "python", "cache"]
    for name in (".uv", ".python", ".cache", ".localm-gui-bootstrap"):
        assert not (clone / name).exists(), f"{name} was left behind\n{out.stdout}"
    assert _field(out, "WINDOW-UV", 1) == str(sysuv / "uv.cmd")
    assert out.returncode == 0, (out.stdout, out.stderr)


def test_a_record_naming_folders_that_are_gone_is_dropped(tmp_path):
    clone, sysuv, _, out = _run(tmp_path, system_uv=True, record=True, marker="uv python cache")
    assert _field(out, "WINDOW-UV") == str(sysuv / "uv.cmd")
    assert _field(out, "WINDOW-BOOTSTRAP") == ""
    assert not (clone / ".localm-gui-bootstrap").exists()


def test_the_window_asking_for_shared_twice_is_a_failure_not_a_loop(tmp_path):
    _clone, _, _, out = _run(tmp_path, system_uv=True, codes="46 46 0")
    assert len(_fields(out, "WINDOW-UV")) == 2
    assert "The setup window could not run (exit 46)" in out.stdout
    assert out.returncode == 46


def test_a_folder_that_cannot_be_removed_is_named_and_setup_goes_on(tmp_path, monkeypatch):
    """The first window waits until this test holds a file in .\\.python open,
    then asks for Shared: the folder cannot go, the launcher says so and opens
    the window again on the per-user uv."""
    go = tmp_path / "go"
    monkeypatch.setenv("WAIT_FOR_GO", str(go))
    real_run = subprocess.run
    held = []

    def run_holding(argv, **kw):
        kw.pop("capture_output")
        kw.pop("timeout")
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)
        ready = Path(str(go) + ".ready")
        import time
        deadline = time.monotonic() + 60
        while not ready.exists() and time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.1)
        assert ready.exists(), "the first window never got ready"
        held.append(open(Path(kw["cwd"]) / ".python" / "held", "rb"))
        go.write_bytes(b"")
        stdout, stderr = proc.communicate(timeout=120)
        return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)

    monkeypatch.setattr(subprocess, "run", run_holding)
    try:
        clone, sysuv, _, out = _run(tmp_path, system_uv=True, codes="46 0")
    finally:
        for fh in held:
            fh.close()
        monkeypatch.setattr(subprocess, "run", real_run)
    assert "Could not remove .\\.python from this folder." in out.stdout, out.stdout
    assert not (clone / ".uv").exists() and not (clone / ".cache").exists()
    assert _field(out, "WINDOW-UV", 1) == str(sysuv / "uv.cmd")
    assert out.returncode == 0, (out.stdout, out.stderr)


def test_portable_never_falls_back_to_the_system_uv_when_the_install_failed(tmp_path):
    _clone, _, _, out = _run(tmp_path, system_uv=True, installer_creates_uv=False)
    assert "INSTALLER-RAN" in out.stdout
    assert "WINDOW-UV" not in out.stdout, "the window opened on the system uv"
    assert "uv still is not callable" in out.stdout
    assert out.returncode == 1


@pytest.mark.parametrize("rc, says", [
    ("61", "Could not download the uv"),
    ("62", "did not match its expected checksum"),
    ("63", "Could not create a temporary folder"),
])
def test_a_refused_uv_download_says_why_and_what_to_do(tmp_path, rc, says):
    clone, _, _, out = _run(tmp_path, installer_rc=rc)
    assert says in out.stdout, out.stdout
    assert "uv was not installed, so the graphical setup cannot start." in out.stdout
    assert "winget install astral-sh.uv" in out.stdout
    assert "WINDOW-UV" not in out.stdout
    assert not (clone / ".uv").exists()
    assert out.returncode == 1


def test_a_folder_with_a_bang_in_its_path_still_installs(tmp_path):
    clone, _, _, out = _run(tmp_path, clone_name="bang!dir", codes="0")
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv.exe")
    assert _field(out, "WINDOW-PYDIR") == str(clone / ".python")


# --------------------------------------------------------------------------- #
#  Started by a double-click: again with no console, messages in boxes          #
# --------------------------------------------------------------------------- #

def test_a_double_click_starts_again_without_a_console_and_ends_at_once(tmp_path):
    clone, _, _, out = _run(tmp_path, hidden=True)
    ran = [ln for ln in out.stdout.splitlines() if ln.startswith("CONHOST-RAN")]
    assert len(ran) == 1, out.stdout
    assert "/d /v:off /c \"\"" in ran[0] and str(clone / "probe.bat") in ran[0], ran[0]
    assert len(_fields(out, "WINDOW-UV")) == 1, "the window opened more or less than once"
    assert out.returncode == 0, (out.stdout, out.stderr)


def test_the_real_start_without_a_console_runs_setup_from_an_awkward_folder(tmp_path):
    """The real `start "" conhost --headless` line (no stand-in): the started
    copy ends at once, and the copy under the windowless console host finds
    its folder and opens the window, from a folder whose name holds cmd's
    special characters."""
    import time
    report = tmp_path / "window-report.txt"
    clone_name = "a&b %PATH% !x ^(y)"
    text = _probe_text(hidden=False)
    clone = tmp_path / clone_name
    clone.mkdir()
    (clone / "probe.bat").write_text(text, encoding="utf-8", newline="")
    stubs = {}
    for name, body in (("installer", _STUB_INSTALLER), ("window", _STUB_WINDOW)):
        stubs[name] = tmp_path / f"stub-{name}.bat"
        stubs[name].write_text(body, encoding="ascii", newline="")
    home = tmp_path / "home"
    home.mkdir()
    system32 = Path(os.environ["SystemRoot"]) / "System32"
    env = dict(os.environ, PATH=str(system32), USERPROFILE=str(home),
               STUB_INSTALLER=str(stubs["installer"]), STUB_WINDOW=str(stubs["window"]),
               STUB_CREATES="1", WINDOW_CODES="0", STUB_REPORT=str(report))
    for var in ("LOCALM_SETUP_GUI_HIDDEN", "LOCALM_SETUP_GUI_INPLACE", "LOCALM_GUI_STORE",
                "LOCALM_GUI_UV", "UV_PYTHON_INSTALL_DIR", "UV_CACHE_DIR", "STUB_RC"):
        env.pop(var, None)
    started = time.monotonic()
    out = subprocess.run(["cmd", "/c", ".\\probe.bat"], cwd=str(clone), env=env,
                         capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert out.returncode == 0 and time.monotonic() - started < 30, (out.stdout, out.stderr)
    assert "WINDOW-UV=" not in out.stdout, "the window ran in the double-clicked copy"
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline and not (
            report.is_file() and "WINDOW-LOG=" in report.read_text(encoding="utf-8", errors="replace")
            and not (clone / ".localm-setup-gui.log").exists()):
        time.sleep(0.25)
    said = report.read_text(encoding="utf-8", errors="replace") if report.is_file() else ""
    assert f"WINDOW-UV={clone / '.uv' / 'uv.exe'}" in said, said
    assert f"WINDOW-LOG={clone / '.localm-setup-gui.log'}" in said, said
    assert not (clone / ".localm-setup-gui.log").exists()


def test_without_a_console_the_getting_ready_window_shows_and_its_file_is_removed(tmp_path):
    clone, _, _, out = _run(tmp_path, hidden=True)
    splash = _field(out, "SPLASH-START")
    assert Path(splash).parent == clone and Path(splash).name.startswith(".localm-setup-starting-")
    assert _field(out, "WINDOW-SPLASH") == splash
    assert not list(clone.glob(".localm-setup-starting-*")), "the getting-ready file was left"


def test_without_a_console_a_failed_window_names_its_output_and_shows_its_end(tmp_path):
    clone, _, _, out = _run(tmp_path, hidden=True, codes="3")
    log = clone / ".localm-setup-gui.log"
    assert _field(out, "WINDOW-LOG") == str(log)
    assert log.is_file() and "WINDOW-UV=" in log.read_text(encoding="utf-8", errors="replace")
    assert re.search(r"^LOCALM_MSG3=What it printed is in .*\.localm-setup-gui\.log", out.stdout, re.M)
    assert re.search(r"^LOCALM_MSG_TAIL=.*\.localm-setup-gui\.log$", out.stdout, re.M), out.stdout


def test_without_a_console_a_window_that_ended_well_leaves_no_output_file(tmp_path):
    clone, _, _, out = _run(tmp_path, hidden=True, codes="0")
    assert _field(out, "WINDOW-LOG") == str(clone / ".localm-setup-gui.log")
    assert not (clone / ".localm-setup-gui.log").exists()


def test_in_this_console_the_window_output_is_not_redirected(tmp_path):
    _clone, _, _, out = _run(tmp_path)
    assert _field(out, "WINDOW-LOG") == ""


def test_without_a_console_a_refused_download_is_shown_in_a_message_box(tmp_path):
    _clone, _, _, out = _run(tmp_path, hidden=True, installer_rc="61")
    assert "BOX-Error" in out.stdout, out.stdout
    assert re.search(r"^LOCALM_MSG1=\[!\] Could not download the uv", out.stdout, re.M), out.stdout
    assert re.search(r"^LOCALM_MSG3=.*winget install astral-sh\.uv", out.stdout, re.M)
    assert out.returncode == 0, "the double-clicked instance itself ends at once"


@pytest.mark.parametrize("code, icon, says", [
    ("44", "Warning", "were not"),
    ("45", "Error", "The uninstall did not finish"),
    ("3", "Error", "The setup window could not run (exit 3)"),
])
def test_without_a_console_every_ending_is_shown_in_a_message_box(tmp_path, code, icon, says):
    _clone, _, _, out = _run(tmp_path, hidden=True, codes=code)
    assert f"BOX-{icon}" in out.stdout, out.stdout
    assert says in out.stdout


def test_in_this_console_no_message_box_is_shown(tmp_path):
    _clone, _, _, out = _run(tmp_path, installer_rc="61")
    assert "BOX-" not in out.stdout and "SPLASH-START" not in out.stdout
    assert "CONHOST-RAN" not in out.stdout


# --------------------------------------------------------------------------- #
#  The launcher asks nothing                                                   #
# --------------------------------------------------------------------------- #

def test_the_launcher_asks_no_question_and_never_pauses():
    lines = (ROOT / "setup-gui.bat").read_text(encoding="utf-8").splitlines()
    code = [ln.strip() for ln in lines if not ln.strip().lower().startswith("rem")]
    assert not [ln for ln in code if re.search(r"\bset\s+/p\b", ln, re.I)]
    assert not [ln for ln in code if re.match(r"(?i)^pause\b", ln)]
    assert not [ln for ln in code if re.search(r"(?i)\bchoice\b\s+/", ln)]
    assert not [ln for ln in lines if "SETUP-Q:" in ln]


def test_the_start_without_a_console_comes_before_any_work():
    lines = [ln.strip() for ln in (ROOT / "setup-gui.bat").read_text(encoding="utf-8").splitlines()]
    start = next(i for i, ln in enumerate(lines) if ln.startswith(_CONHOST))
    inplace = next(i for i, ln in enumerate(lines) if "LOCALM_SETUP_GUI_INPLACE%\"==\"1\"" in ln)
    work = min(i for i, ln in enumerate(lines)
               if ln.startswith(("powershell", '"%UVEXE%"', "call :splash_start", "start \"\" /b")))
    assert inplace < start < work
    assert lines[start + 1].startswith("if errorlevel 1 goto"), "a failed start must not end setup"
    assert lines[start + 2] == "exit /b 0"
