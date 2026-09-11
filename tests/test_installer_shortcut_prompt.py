# SPDX-License-Identifier: AGPL-3.0-or-later
r"""The end of setup must describe what the user actually chose. Four properties of
the desktop-shortcut screen:

1. "launcher" names ONE thing. The branded LocaLM.exe built by `make-launcher`
   and localm-launcher.bat (the mode picker) are distinguished.
2. The web-GUI label does not assert a browser: `localm gui` opens a NATIVE
   WINDOW when the desktop extra is installed, which an earlier prompt in this
   same script decides.
3. The closing line names the shortcut that was made, not a fixed command.
4. The closing line and the install manifest key on whether the shortcut was
   actually CREATED, not on what was ASKED FOR (SCPICK), so a failed .lnk write
   does not produce "start it from the shortcut" or record a .lnk for uninstall
   to hunt.

These run against the REAL setup.bat, so they fail if the wording regresses.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BAT = ROOT / "setup.bat"

# The whole PowerShell -Command payload of a shortcut block, from the literal
# start of the invocation through the closing quote of its last fragment.
_POWERSHELL_BODY_RE = re.compile(r'powershell -NoProfile -Command \^.*?Write-Output \$p"', re.DOTALL)


@pytest.fixture(scope="module")
def bat():
    return BAT.read_text(encoding="utf-8", errors="replace")


def _shortcut_blocks(bat_text):
    """The two `if "%SCPICK%"=="N" ( ... )` shortcut blocks, sliced out of the
    real setup.bat text by their own boundaries."""
    b1 = bat_text[bat_text.index('if "%SCPICK%"=="1" ('):bat_text.index('if "%SCPICK%"=="2" (')]
    b2 = bat_text[bat_text.index('if "%SCPICK%"=="2" ('):bat_text.index('if "%SCPICK%"=="3"')]
    return b1, b2


def test_window_mode_is_captured_where_it_is_chosen(bat):
    """WINMODE must be set from WPICK, before anything describes the GUI."""
    assert 'set "WINMODE=in your browser"' in bat
    assert 'if "%WPICK%"=="2" set "WINMODE=in its own app window"' in bat
    assert bat.index('set "WINMODE=') < bat.index("Create a desktop shortcut?"), \
        "WINMODE must be set before the shortcut prompt uses it"


def test_shortcut_options_say_what_they_do(bat):
    """Neither option may be a bare noun the user has to guess at."""
    assert "echo    [1] Launcher\r\n" not in bat, \
        "bare '[1] Launcher' is back - it never says what the launcher IS"
    assert "[2] Web GUI directly" not in bat, \
        "'Web GUI directly' is wrong for anyone who picked the app window"
    assert "[1] LocaLM launcher - choose GUI / chat / server / coder each time" in bat
    assert "[2] Straight to the GUI - skips that menu, opens %WINMODE%" in bat


def test_the_build_step_does_not_also_call_itself_a_launcher(bat):
    """Only ONE thing in this script may be called 'the launcher'."""
    assert "Building the LocaLM app launcher" not in bat, \
        "two different things are called 'launcher' again"
    assert "Branding the app executable" in bat
    assert "make-launcher --force --quiet" in bat, \
        "setup must pass --quiet so this step stops printing a competing " \
        "'Launch it:' instruction before the user has picked a shortcut"


def test_closing_lines_follow_the_choice_and_only_claim_a_real_shortcut(bat):
    """Gated on SCMADE (it worked), never on SCPICK (it was requested)."""
    tail = bat[bat.index("echo  Done. Setup complete."):]
    assert "if not defined SCMADE echo  Run localm-launcher.bat to start." in tail
    assert 'if defined SCMADE if "%SCPICK%"=="1" echo  Start it from the LocaLM ' \
           "shortcut on your desktop." in tail
    assert '%WINMODE%' in tail, "the [2] closing line must name the real window mode"
    # The old unconditional line must be gone: it ignored the answer entirely.
    assert "\r\necho  Run localm-launcher.bat to start." not in bat, \
        "the closing instruction is unconditional again"
    for line in tail.splitlines():
        s = line.strip()
        if s.startswith("echo  Start it from") or s.startswith("echo  Or run"):
            pytest.fail(f"ungated claim about a shortcut that may not exist: {s}")


def test_manifest_records_no_shortcut_when_none_was_created(bat):
    """A failed .lnk write must not leave SCPATH set for the manifest."""
    assert 'if not defined SCMADE set "SCPATH="' in bat
    assert bat.index('if not defined SCMADE set "SCPATH="') < bat.index("--shortcut"), \
        "SCPATH must be cleared BEFORE the manifest records it"
    assert bat.count('if defined SCPATH set "SCMADE=1"') == 2, \
        "both shortcut branches must derive SCMADE from the path PowerShell " \
        "actually wrote back, never from errorlevel"


def test_shortcut_path_is_not_a_second_guess(bat):
    """SCPATH must come from what PowerShell actually wrote, not a hardcoded
    %USERPROFILE%\\Desktop literal that can diverge from a redirected Desktop
    (OneDrive Known Folder Move, Folder Redirection, a moved Desktop)."""
    assert "SCPATH=%USERPROFILE%\\Desktop" not in bat


def test_shortcut_powershell_writes_the_path_back(bat):
    """Both shortcut blocks must read back the real path, the same way
    installer/gui.py already does, instead of assuming one."""
    for block in _shortcut_blocks(bat):
        assert "Write-Output $p" in block


def test_shortcut_powershell_stops_on_the_first_error(bat):
    """A thrown .Save() must not still reach Write-Output and claim success."""
    for block in _shortcut_blocks(bat):
        assert "$ErrorActionPreference = 'Stop'" in block
        assert block.index("$ErrorActionPreference = 'Stop'") < block.index(".Save()")


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe only")
class TestShortcutBlockDerivesScmadeFromThePath:
    """Drives the real %SCPICK%==1 block, with its PowerShell body replaced by
    a harmless placeholder and `powershell` shadowed by a stub, so SCMADE's
    derivation is proven under a real cmd.exe instead of merely read as text.
    Nothing here invokes real PowerShell, writes a .lnk, or touches the
    registry, the Desktop, %USERPROFILE%, or any real LocaLM install."""

    def _stubbed_block(self, bat):
        block, _ = _shortcut_blocks(bat)
        substituted, n = _POWERSHELL_BODY_RE.subn(
            'powershell -NoProfile -Command "x"', block, count=1)
        assert n == 1, "the placeholder regex no longer matches the shipped block"
        assert "WScript.Shell" not in substituted
        assert ".Save()" not in substituted
        assert '-Command "x"' in substituted
        return substituted

    def _write_probe(self, tmp_path, block, seed):
        bat_path = tmp_path / "probe.bat"
        bat_path.write_text(
            "@echo off\r\nsetlocal\r\n"
            'set "SCPICK=1"\r\nset "SCPATH="\r\nset "SCMADE="\r\n'
            "call :seterr {seed}\r\n"
            "{block}\r\n"
            'if not defined SCMADE set "SCPATH="\r\n'
            'echo RESULT SCPATH=[%SCPATH%] SCMADE=[%SCMADE%]\r\n'
            "exit /b 0\r\n:seterr\r\nexit /b %1\r\n".format(seed=seed, block=block),
            encoding="utf-8")
        return bat_path

    def _write_stub_powershell(self, directory, succeed):
        directory.mkdir(parents=True, exist_ok=True)
        stub = directory / "powershell.bat"
        if succeed:
            stub.write_text(
                "@echo off\r\necho C:\\Redirected\\OneDrive\\Desktop\\LocaLM.lnk\r\nexit /b 0\r\n",
                encoding="utf-8")
        else:
            stub.write_text("@echo off\r\nexit /b 1\r\n", encoding="utf-8")
        return directory

    def _run(self, probe_bat, stub_dir):
        env = dict(os.environ)
        env["PATH"] = str(stub_dir) + ";" + env.get("PATH", "")
        return subprocess.run(["cmd", "/c", str(probe_bat)], capture_output=True,
                              text=True, stdin=subprocess.DEVNULL, timeout=15, env=env)

    def _result(self, stdout):
        m = re.search(r"RESULT SCPATH=\[(.*?)\] SCMADE=\[(.*?)\]", stdout)
        assert m, "RESULT line not found: {!r}".format(stdout)
        return m.group(1), m.group(2)

    def test_a_succeeding_write_is_captured_at_the_default_menu_answer(self, bat, tmp_path):
        """set /p leaves errorlevel 1 on the common default-Enter path, which
        a naive `for /f` + `if not errorlevel 1` capture reads as failure."""
        block = self._stubbed_block(bat)
        stub_dir = self._write_stub_powershell(tmp_path / "ok", succeed=True)
        probe = self._write_probe(tmp_path, block, seed=1)
        out = self._run(probe, stub_dir)
        scpath, scmade = self._result(out.stdout)
        assert scpath == "C:\\Redirected\\OneDrive\\Desktop\\LocaLM.lnk", (out.stdout, out.stderr)
        assert scmade == "1", (out.stdout, out.stderr)

    def test_a_failing_write_records_nothing(self, bat, tmp_path):
        block = self._stubbed_block(bat)
        stub_dir = self._write_stub_powershell(tmp_path / "fail", succeed=False)
        probe = self._write_probe(tmp_path, block, seed=0)
        out = self._run(probe, stub_dir)
        scpath, scmade = self._result(out.stdout)
        assert scpath == "", (out.stdout, out.stderr)
        assert scmade == "", (out.stdout, out.stderr)


def test_cd_bootstrap_precedes_delayed_expansion(bat):
    """`cd /d "%~dp0"` must run BEFORE delayed expansion is enabled: a `!` in
    the install path is silently dropped by cmd's delayed-expansion scanner
    when this line runs under EnableDelayedExpansion, and `cd /d` then fails
    outright ("The system cannot find the path specified")."""
    assert bat.index('cd /d "%~dp0"') < bat.index("setlocal EnableDelayedExpansion")


def test_shortcut_blocks_isolate_the_bang_hazard_and_close_it_on_failure(bat):
    """Each shortcut block's FOR /F embeds `%CD%` several times in one
    PowerShell command; a literal `!` in the install path is silently eaten
    by cmd's delayed-expansion scanner once those are substituted in, unless
    delayed expansion is disabled for that FOR /F. The scope must also close
    when the loop captures nothing (a failed write), or delayed expansion
    stays disabled for the rest of the script - see
    TestShortcutSurvivesBangInInstallPath for the executing proof of both."""
    for block in _shortcut_blocks(bat):
        assert "setlocal DisableDelayedExpansion" in block
        assert 'set "SC_STILL_OPEN=1"' in block
        assert 'endlocal & set "SCPATH=%%p"' in block
        assert "if defined SC_STILL_OPEN endlocal" in block


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe only")
class TestBootstrapSurvivesBangInInstallPath:
    """Drives the REAL bootstrap lines of setup.bat (through `set
    LOCALM_SETUP=1`) from a directory whose name contains a literal `!`,
    proving `cd /d` actually lands there rather than merely reading the
    source order."""

    def _bootstrap_lines(self, bat):
        start = bat.index('cd /d "%~dp0"')
        end = bat.index("set LOCALM_SETUP=1") + len("set LOCALM_SETUP=1")
        return bat[start:end]

    def test_cd_succeeds_and_lands_in_the_bang_directory(self, bat, tmp_path):
        bangdir = tmp_path / "bang!dir"
        bangdir.mkdir()
        probe = bangdir / "probe.bat"
        probe.write_text(
            "@echo off\r\n" + self._bootstrap_lines(bat) + "\r\n"
            'echo RC=[%errorlevel%]\r\n'
            'setlocal DisableDelayedExpansion\r\n'
            'echo TRUE_CD=[%CD%]\r\n'
            "exit /b 0\r\n",
            encoding="utf-8")
        out = subprocess.run(["cmd", "/c", str(probe)], capture_output=True,
                              text=True, stdin=subprocess.DEVNULL, timeout=15)
        assert "RC=[0]" in out.stdout, (out.stdout, out.stderr)
        assert "TRUE_CD=[{}]".format(bangdir) in out.stdout, (out.stdout, out.stderr)


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe only")
class TestShortcutSurvivesBangInInstallPath:
    """A literal `!` in the install directory name must not corrupt the
    PowerShell command text (cmd's delayed-expansion scanner silently eats
    exclamation marks out of already-substituted `%CD%` text), and a FAILED
    write must still leave delayed expansion enabled for the rest of the
    script. `powershell` is shadowed by a stub capturing its own argv so the
    exact text cmd.exe hands it can be inspected directly, instead of
    inferring correctness from a shortcut file nothing here writes."""

    def _write_argv_capturing_stub(self, directory, argv_file, succeed):
        directory.mkdir(parents=True, exist_ok=True)
        stub = directory / "powershell.bat"
        body = '@echo off\r\necho ARGV=[%*]>>"{}"\r\n'.format(argv_file)
        body += ("echo C:\\FakeDesktop\\LocaLM.lnk\r\nexit /b 0\r\n" if succeed
                 else "exit /b 1\r\n")
        stub.write_text(body, encoding="utf-8")
        return directory

    def _write_probe(self, directory, block, scpick):
        probe = directory / "probe.bat"
        probe.write_text(
            "@echo off\r\nsetlocal EnableDelayedExpansion\r\n"
            'set "SCPICK={scpick}"\r\nset "SCPATH="\r\nset "SCMADE="\r\n'
            "{block}\r\n"
            'echo RESULT SCPATH=[%SCPATH%] SCMADE=[%SCMADE%]\r\n'
            'set "PROBEVAR=still-here"\r\n'
            'echo DELAYED_EXPANSION_OK=[!PROBEVAR!]\r\n'
            "exit /b 0\r\n".format(scpick=scpick, block=block),
            encoding="utf-8")
        return probe

    def _run(self, probe, stub_dir, cwd):
        env = dict(os.environ)
        env["PATH"] = str(stub_dir) + ";" + env.get("PATH", "")
        return subprocess.run(["cmd", "/c", str(probe)], capture_output=True,
                              text=True, stdin=subprocess.DEVNULL, timeout=15,
                              env=env, cwd=str(cwd))

    def _result(self, stdout):
        m = re.search(r"RESULT SCPATH=\[(.*?)\] SCMADE=\[(.*?)\]", stdout)
        assert m, "RESULT line not found: {!r}".format(stdout)
        return m.group(1), m.group(2)

    @pytest.mark.parametrize("scpick,block_index", [("1", 0), ("2", 1)])
    def test_command_text_survives_a_bang_in_the_install_path(
            self, bat, tmp_path, scpick, block_index):
        block = _shortcut_blocks(bat)[block_index]
        bangdir = tmp_path / "bang!dir"
        bangdir.mkdir()
        argv_file = tmp_path / "argv.txt"
        stub_dir = self._write_argv_capturing_stub(tmp_path / "stub", argv_file, succeed=True)
        probe = self._write_probe(bangdir, block, scpick)
        out = self._run(probe, stub_dir, cwd=bangdir)
        assert argv_file.exists(), (out.stdout, out.stderr)
        argv = argv_file.read_text(encoding="utf-8")
        bang = str(bangdir)
        # Each fragment embedding %CD% must survive as its OWN intact,
        # separately-quoted argument with the `!` preserved - not merged
        # into a neighbour by a corrupted delayed-expansion scan.
        assert "\"$s.WorkingDirectory = '{}';\"".format(bang) in argv, argv
        assert "\"$s.IconLocation = '{}\\assets\\localm.ico';\"".format(bang) in argv, argv
        scpath, scmade = self._result(out.stdout)
        assert scmade == "1", (out.stdout, out.stderr)

    def test_scpick_2_targetpath_fragment_survives_a_bang_with_two_cd_occurrences(
            self, bat, tmp_path):
        """SCPICK==2's TargetPath fragment embeds %CD% TWICE in one PowerShell
        statement (the LocaLM.exe branch and its Scripts\\localm.exe
        fallback) - the highest-risk shape, since two exclamation marks from
        one fragment are exactly what can pair up and consume everything
        between them."""
        block = _shortcut_blocks(bat)[1]
        bangdir = tmp_path / "bang!dir"
        bangdir.mkdir()
        argv_file = tmp_path / "argv.txt"
        stub_dir = self._write_argv_capturing_stub(tmp_path / "stub", argv_file, succeed=True)
        probe = self._write_probe(bangdir, block, "2")
        self._run(probe, stub_dir, cwd=bangdir)
        argv = argv_file.read_text(encoding="utf-8")
        bang = str(bangdir)
        assert "$exe = '{}\\.venv\\localm-app\\LocaLM.exe'".format(bang) in argv, argv
        assert "$s.TargetPath = '{}\\.venv\\Scripts\\localm.exe'".format(bang) in argv, argv

    @pytest.mark.parametrize("scpick,block_index", [("1", 0), ("2", 1)])
    def test_a_failed_write_in_a_bang_path_leaves_delayed_expansion_enabled(
            self, bat, tmp_path, scpick, block_index):
        """The nested `setlocal DisableDelayedExpansion` scope must close
        even when the FOR /F loop captures zero lines (a failed write) - a
        transport that only closes it inside the loop's own do-body leaves
        the scope open, silently disabling delayed expansion for every line
        the rest of the script runs afterward."""
        block = _shortcut_blocks(bat)[block_index]
        bangdir = tmp_path / "bang!dir"
        bangdir.mkdir()
        argv_file = tmp_path / "argv.txt"
        stub_dir = self._write_argv_capturing_stub(tmp_path / "stub", argv_file, succeed=False)
        probe = self._write_probe(bangdir, block, scpick)
        out = self._run(probe, stub_dir, cwd=bangdir)
        scpath, scmade = self._result(out.stdout)
        assert scpath == "" and scmade == "", (out.stdout, out.stderr)
        assert "DELAYED_EXPANSION_OK=[still-here]" in out.stdout, (
            "delayed expansion was left disabled after a failed shortcut "
            "write in a bang path: {}".format(out.stdout))


def test_make_launcher_quiet_prints_no_competing_start_instruction(monkeypatch):
    """--quiet keeps the notes and the failure path, drops the hints."""
    import sys
    from click.testing import CliRunner
    from localm import applaunch
    from localm.cli import maintenance

    fake_path = (Path("X:/clone/.venv/localm-app/LocaLM.exe") if sys.platform == "win32"
                else Path("/clone/.venv/localm-app/LocaLM"))
    fake = applaunch.LauncherResult(
        ok=True, path=fake_path,
        notes=["built LocaLM.exe from python.exe + 4 runtime DLL(s)"])
    monkeypatch.setattr(applaunch, "make_launcher", lambda force=False: fake)

    loud = CliRunner().invoke(maintenance.make_launcher_cmd, [])
    quiet = CliRunner().invoke(maintenance.make_launcher_cmd, ["--quiet"])

    assert loud.exit_code == 0 and quiet.exit_code == 0
    # The note (real work done) survives both; the hints only appear when loud.
    assert "built LocaLM.exe" in loud.output and "built LocaLM.exe" in quiet.output
    # "Launch it:" is a Windows-only hint (maintenance.py gates it on
    # sys.platform == "win32"); POSIX's equivalent affordance is the desktop
    # entry line, which this fake result does not set.
    if sys.platform == "win32":
        assert "Launch it:" in loud.output
    assert "Launch it:" not in quiet.output, \
        "setup must not be handed a competing way to start localm"
    assert "Launcher ready:" not in loud.output, \
        "'Launcher ready' collides with localm-launcher.bat; name the executable"
    assert "App executable ready:" in loud.output


def test_make_launcher_quiet_still_reports_failure(monkeypatch):
    """--quiet silences hints, never problems (we do not hide problems)."""
    from click.testing import CliRunner
    from localm import applaunch
    from localm.cli import maintenance

    fake = applaunch.LauncherResult(ok=False, notes=["could not build LocaLM.exe: boom"])
    monkeypatch.setattr(applaunch, "make_launcher", lambda force=False: fake)

    res = CliRunner().invoke(maintenance.make_launcher_cmd, ["--quiet"])
    assert res.exit_code == 1, "a failed build must still exit non-zero under --quiet"
    assert "could not build LocaLM.exe: boom" in res.output
    assert "Could not build the native launcher" in res.output
