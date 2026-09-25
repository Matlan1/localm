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
import json
import os
import re
import shutil
import subprocess
import sys
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


def _manifest_record_block(bat_text):
    """The final `setlocal DisableDelayedExpansion` / install-manifest-record /
    `endlocal` wrapper (the one at the end of setup, not the early record after
    the environment is created), sliced out of the real setup.bat text by its
    own boundaries."""
    section = bat_text.index("rem ---- record what we installed")
    start = bat_text.index(
        'setlocal DisableDelayedExpansion\n.venv\\Scripts\\python -m localm.install_manifest record',
        section)
    end = bat_text.index('\nif errorlevel 1 echo  [^^!] Could not record the install manifest', start)
    return bat_text[start:end]


def _early_record_block(bat_text):
    """The record right after the environment is created (`:venv_create_ok`),
    from its `setlocal DisableDelayedExpansion` through its `endlocal`."""
    ok = bat_text.index(":venv_create_ok")
    start = bat_text.index("setlocal DisableDelayedExpansion\n", ok)
    end = bat_text.index("\nendlocal", start) + len("\nendlocal")
    block = bat_text[start:end]
    assert "localm.install_manifest record" in block, "the early record moved"
    assert bat_text.index(":venv_done", ok) > end, "the early record left :venv_create_ok"
    return block


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


def test_manifest_record_line_isolates_the_bang_hazard(bat):
    """The install-manifest record line embeds %CD% directly twice (--venv,
    --lib-dir) on one top-level line, outside any block - the same multi-
    occurrence delayed-expansion hazard the shortcut blocks had - so it must
    be isolated the same way: setlocal DisableDelayedExpansion immediately
    before the command, endlocal immediately after, before the pre-existing
    errorlevel check runs. See TestManifestRecordSurvivesBangInInstallPath
    for the executing proof."""
    block = _manifest_record_block(bat)
    lines = block.splitlines()
    assert lines[0] == "setlocal DisableDelayedExpansion"
    assert lines[-1] == "endlocal"
    assert lines[1].startswith(".venv\\Scripts\\python -m localm.install_manifest record")
    assert bat.index(block) < bat.index("if errorlevel 1 echo  [^^!] Could not record the install manifest")


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe only")
class TestManifestRecordSurvivesBangInInstallPath:
    """The install-manifest record line embeds %CD% directly twice (--venv,
    --lib-dir) on one top-level line, outside any block - the same multi-
    occurrence delayed-expansion hazard the shortcut blocks had, where cmd's
    scanner pairs up 2+ literal `!` characters left on one already-
    substituted line and merges or drops whatever text sits between them.
    `echo` replaces the real `.venv\\Scripts\\python -m
    localm.install_manifest record` invocation so the substituted text can
    be inspected directly: a stub .bat invoked here without `call` would
    hand control to the stub and never return to this probe - a harness
    artefact unrelated to this defect, since the real command is an .exe,
    not a .bat."""

    def _echoed_block(self, bat):
        block = _manifest_record_block(bat)
        target = ".venv\\Scripts\\python -m localm.install_manifest record"
        assert target in block, "the manifest-record invocation text moved; update this test"
        assert block.endswith(" >nul 2>nul\nendlocal"), \
            "the manifest-record line's shape changed; update this test"
        echoed = block[: -len(" >nul 2>nul\nendlocal")] + "\nendlocal"
        return echoed.replace(target, "echo MANIFEST_ARGS", 1)

    def _run(self, directory, block_text, extra_setup):
        probe = directory / "probe.bat"
        probe.write_text(
            "@echo off\r\nsetlocal EnableDelayedExpansion\r\n"
            + extra_setup +
            "{block}\r\n"
            'echo AFTER_MARK\r\n'
            'set "PROBEVAR=still-here"\r\n'
            'echo DELAYED_EXPANSION_OK=[!PROBEVAR!]\r\n'
            "exit /b 0\r\n".format(block=block_text),
            encoding="utf-8")
        return subprocess.run(["cmd", "/c", str(probe)], capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=15, cwd=str(directory))

    def test_command_text_survives_a_bang_in_the_install_path_flags_set(self, bat, tmp_path):
        """Every optional field populated (a global+contained install with a
        shortcut) - the highest-risk shape, since it puts the most text
        between the two %CD% occurrences."""
        echoed = self._echoed_block(bat)
        bangdir = tmp_path / "bang!dir"
        bangdir.mkdir()
        extra_setup = (
            'set "SCPATH=C:\\FakeDesktop\\LocaLM.lnk"\r\nset "RCFLAG=--runtime-contained"\r\n'
            'set "PYDIR=C:\\FakePython"\r\nset "CACHEDIR=C:\\FakeCache"\r\n'
            'set "UVDIR=C:\\FakeUv"\r\nset "UVSHARED=--uv-shared-installed"\r\n'
            'set "PATHDIR=C:\\FakeBin"\r\n'
            'set "CMDSHIM=C:\\FakeBin\\localm.cmd"\r\nset "PATHMOD=--path-modified"\r\n')
        out = self._run(bangdir, echoed, extra_setup)
        expected = (
            'MANIFEST_ARGS --root . --venv "{bang}\\.venv" --lib-dir '
            '"{bang}\\runtime\\localm_llama_runtime\\lib" '
            '--shortcut "C:\\FakeDesktop\\LocaLM.lnk" --runtime-contained '
            '--python-dir "C:\\FakePython" --cache-dir "C:\\FakeCache" --uv-dir "C:\\FakeUv" '
            '--uv-shared-installed '
            '--path-dir "C:\\FakeBin" --command-shim "C:\\FakeBin\\localm.cmd" --path-modified'
        ).format(bang=bangdir)
        assert expected in out.stdout, (out.stdout, out.stderr)
        assert "AFTER_MARK" in out.stdout, (out.stdout, out.stderr)
        assert "DELAYED_EXPANSION_OK=[still-here]" in out.stdout, (out.stdout, out.stderr)

    def test_command_text_survives_a_bang_in_the_install_path_flags_empty(self, bat, tmp_path):
        """The common case: no global install, no contained runtime, no
        shortcut - CRD/RCFLAG/PYDIR/CACHEDIR/PATHDIR/CMDSHIM/PATHMOD are all
        empty, so the substituted line carries several adjacent-space and
        empty-quote gaps around the two %CD% occurrences."""
        echoed = self._echoed_block(bat)
        bangdir = tmp_path / "bang!dir"
        bangdir.mkdir()
        extra_setup = (
            'set "SCPATH=C:\\FakeDesktop\\LocaLM.lnk"\r\nset "RCFLAG="\r\n'
            'set "PYDIR="\r\nset "CACHEDIR="\r\n'
            'set "UVDIR=C:\\FakeUv"\r\nset "UVSHARED="\r\nset "PATHDIR="\r\n'
            'set "CMDSHIM="\r\nset "PATHMOD="\r\n')
        out = self._run(bangdir, echoed, extra_setup)
        expected = (
            'MANIFEST_ARGS --root . --venv "{bang}\\.venv" --lib-dir '
            '"{bang}\\runtime\\localm_llama_runtime\\lib" '
            '--shortcut "C:\\FakeDesktop\\LocaLM.lnk"  --python-dir "" --cache-dir "" '
            '--uv-dir "C:\\FakeUv"  --path-dir "" --command-shim "" '
        ).format(bang=bangdir)
        assert expected in out.stdout, (out.stdout, out.stderr)
        assert "AFTER_MARK" in out.stdout, (out.stdout, out.stderr)
        assert "DELAYED_EXPANSION_OK=[still-here]" in out.stdout, (out.stdout, out.stderr)

    def test_the_early_record_survives_a_bang_in_the_install_path(self, bat, tmp_path):
        """The record right after the environment is created embeds %CD% and
        the runtime folders on one line too, so it carries the same wrap."""
        block = _early_record_block(bat)
        target = ".venv\\Scripts\\python -m localm.install_manifest record"
        assert block.endswith(" >nul 2>nul\nendlocal"), "the early record's shape changed"
        echoed = (block[: -len(" >nul 2>nul\nendlocal")] + "\nendlocal").replace(
            target, "echo MANIFEST_ARGS", 1)
        bangdir = tmp_path / "bang!dir"
        bangdir.mkdir()
        extra_setup = (
            'set "RCFLAG=--runtime-contained"\r\nset "PYDIR=%CD:!=^!%\\.python"\r\n'
            'set "CACHEDIR=%CD:!=^!%\\.cache"\r\nset "UVDIR=%CD:!=^!%\\.uv"\r\n'
            'set "UVSHARED="\r\n')
        out = self._run(bangdir, echoed, extra_setup)
        expected = (
            'MANIFEST_ARGS --root . --venv "{bang}\\.venv" --runtime-contained '
            '--python-dir "{bang}\\.python" --cache-dir "{bang}\\.cache" '
            '--uv-dir "{bang}\\.uv" '
        ).format(bang=bangdir)
        assert expected in out.stdout, (out.stdout, out.stderr)
        assert "DELAYED_EXPANSION_OK=[still-here]" in out.stdout, (out.stdout, out.stderr)

    @pytest.mark.parametrize("exit_code,expect_flagged", [(0, False), (3, True)])
    def test_errorlevel_survives_the_disabled_expansion_scope(
            self, bat, tmp_path, exit_code, expect_flagged):
        """The pre-existing `if errorlevel 1 echo ...` guard right after this
        block must still see the wrapped command's real exit code, not one
        reset by entering or leaving the new setlocal scope."""
        block = _manifest_record_block(bat)
        lines = block.splitlines()
        assert lines[0] == "setlocal DisableDelayedExpansion"
        assert lines[-1] == "endlocal"
        substituted = "\r\n".join([lines[0], "cmd /c exit {}".format(exit_code), lines[-1]])
        bangdir = tmp_path / "bang!dir"
        bangdir.mkdir()
        probe = bangdir / "probe.bat"
        probe.write_text(
            "@echo off\r\nsetlocal EnableDelayedExpansion\r\n"
            + substituted + "\r\n"
            'if errorlevel 1 (echo FLAGGED) else (echo NOT_FLAGGED)\r\n'
            'set "PROBEVAR=still-here"\r\n'
            'echo DELAYED_EXPANSION_OK=[!PROBEVAR!]\r\n'
            "exit /b 0\r\n",
            encoding="utf-8")
        out = subprocess.run(["cmd", "/c", str(probe)], capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=15, cwd=str(bangdir))
        lines_out = out.stdout.splitlines()
        assert ("FLAGGED" in lines_out) == expect_flagged, (out.stdout, out.stderr)
        assert ("NOT_FLAGGED" in lines_out) == (not expect_flagged), (out.stdout, out.stderr)
        assert "DELAYED_EXPANSION_OK=[still-here]" in out.stdout, (out.stdout, out.stderr)


def _top_install_message_block(bat_text):
    """The `setlocal DisableDelayedExpansion` / echo / `endlocal` wrapper
    around the very first line that names the install directory."""
    literal = ('setlocal DisableDelayedExpansion\n'
               'echo  LocaLM setup - self-contained install in: %CD%\n'
               'endlocal')
    assert literal in bat_text, "the top install-directory message wrapper moved"
    i = bat_text.index(literal)
    return bat_text[i:i + len(literal)]


def _uv_dirs_block(bat_text):
    """The `if "%STOREPICK%"=="1" ( ... ) else ( ... )` block that provisions
    uv's managed-Python-install and cache dirs."""
    start = bat_text.index('if "%STOREPICK%"=="1" (')
    end = bat_text.index('rem ---- uv is required', start)
    return bat_text[start:end]


def _uv_check_portable_block(bat_text):
    """The `:uv_check_portable` reuse-an-existing-portable-uv block."""
    start = bat_text.index(':uv_check_portable')
    end = bat_text.index(':uv_missing', start)
    return bat_text[start:end]


def _uv_missing_contained_block(bat_text):
    """The CONTAINED-mode `if "%CONTAINED%"=="1" ( ... )` block inside
    `:uv_missing` that confines uv's own binary under .\\.uv."""
    marker = bat_text.index("rem  Portable was picked: confine uv's OWN binary")
    start = bat_text.rindex('if "%CONTAINED%"=="1" (', 0, marker)
    end = bat_text.index('powershell -NoProfile -ExecutionPolicy Bypass', start)
    return bat_text[start:end]


def _uv_missing_full_sequence(bat_text):
    """The CONTAINED-mode block through the `where uv` / `goto uv_ready`
    check that follows the (stubbed-out, by the caller) Astral installer
    call - covers the UVDIRS/PATH rebuild that consumes UV_INSTALL_DIR a few
    lines after it is set."""
    marker = bat_text.index("rem  Portable was picked: confine uv's OWN binary")
    start = bat_text.rindex('if "%CONTAINED%"=="1" (', 0, marker)
    end = bat_text.index('if not errorlevel 1 goto uv_ready', start)
    end = bat_text.index('\n', end)
    return bat_text[start:end]


def _pathdir_cmdshim_lines(bat_text):
    """The four single-line `if ... set ...` PATHDIR/CMDSHIM lines."""
    start = bat_text.index('if "!GCRC!"=="0" set "PATHMOD=--path-modified"')
    marker = 'if "!GCRC!"=="20" set "CMDSHIM='
    end = bat_text.index('\n', bat_text.index(marker, start))
    return bat_text[start:end]


def _pydir_cachedir_block(bat_text):
    """The CONTAINED-mode `if "%CONTAINED%"=="1" ( ... )` block that computes
    PYDIR/CACHEDIR for the install manifest. Ends at its own closing `)`, NOT
    at the manifest-record line - that line's own `setlocal
    DisableDelayedExpansion` wrap (a separate fix) sits between the two and
    must not be swept into this block, or it leaks into the caller as an
    unmatched setlocal."""
    marker = bat_text.index('set "RCFLAG="\nset "PYDIR="\nset "CACHEDIR="')
    start = bat_text.index('if "%CONTAINED%"=="1" (', marker)
    end = bat_text.index('\n)\n', start) + len('\n)')
    return bat_text[start:end]


def _data_folder_subroutines(bat_text):
    """`:do_custom_home`, `:custom_home_blank` and `:portable_home` verbatim,
    from the `:do_custom_home` label LINE through `:portable_home`'s closing
    `exit /b 0`. Anchored on the label line via regex: `call :do_custom_home`
    and a `rem` line both contain the text earlier in the file."""
    m = re.search(r"(?m)^:do_custom_home\s*$", bat_text)
    assert m, "the :do_custom_home label moved or was renamed"
    p = re.search(r"(?m)^:portable_home\s*$", bat_text)
    assert p and p.start() > m.start(), ":portable_home moved"
    end = bat_text.index("exit /b 0", bat_text.index("created .\\home anyway", p.start()))
    return bat_text[m.start():end + len("exit /b 0")]


def _flush_block(bat_text):
    """The real `:flush` subroutine, verbatim. `:do_custom_home` calls it
    twice, so any probe driving that block whole needs a real target for
    `call :flush` to jump to."""
    m = re.search(r"(?m)^:flush\s*$", bat_text)
    assert m, "the :flush label moved or was renamed"
    end = bat_text.index('goto :eof', m.start()) + len('goto :eof')
    return bat_text[m.start():end]


def _uninstall_header_block(bat_text):
    """The `:uninstall` label's clone-path banner."""
    literal = 'setlocal DisableDelayedExpansion\necho    %CD%\nendlocal'
    assert literal in bat_text, "the uninstall banner's wrapper moved"
    i = bat_text.index(literal)
    return bat_text[i:i + len(literal)]


def test_cd_derived_set_statements_escape_the_bang_before_delayed_expansion_scans_it(bat):
    """Every plain `set VAR=%CD%\\...` statement outside a FOR loop or the
    PowerShell shortcut blocks (each of those already has its own coverage)
    has exactly ONE %CD% occurrence on its line, so cmd's delayed-expansion
    scanner silently drops the lone `!` a bang-bearing install path
    substitutes in - corrupting the variable at the point it is set, before
    anything downstream (including the install manifest) ever reads it.

    `%CD:!=^!%` replaces a literal `!` with the caret-escaped form as PART of
    the same %-expansion pass that inserts %CD%'s value, so the delayed-
    expansion scan that runs immediately after sees an escaped `^!`, not a
    bare `!`, and resolves it back to a literal `!` in the value actually
    stored - regardless of whether delayed expansion is enabled at that
    line. No setlocal/endlocal is needed for these sites (unlike the
    PowerShell shortcut blocks, nothing here needs to survive an `endlocal`
    - see TestCdDerivedVarsSurviveBangInInstallPath for why that specific
    transport does NOT preserve a bang, and the module docstring's note on
    scope). The two direct, immediate `%CD%`-naming echoes (the opening
    banner and the uninstall banner) are a different case - the escape alone
    does not protect an unquoted echo - and keep their
    setlocal-DisableDelayedExpansion/endlocal wrap instead. See
    TestCdDerivedVarsSurviveBangInInstallPath for the executing proof of
    both shapes."""
    for extractor in (_top_install_message_block, _uninstall_header_block):
        block = extractor(bat)
        assert "setlocal DisableDelayedExpansion" in block, extractor.__name__
        assert "endlocal" in block, extractor.__name__

    for extractor in (
            _uv_dirs_block,
            _uv_check_portable_block,
            _uv_missing_contained_block,
            _pathdir_cmdshim_lines,
            _pydir_cachedir_block,
    ):
        block = extractor(bat)
        assert "%CD:!=^!%" in block, extractor.__name__
        assert "%CD%" not in block, extractor.__name__


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe only")
class TestCdDerivedVarsSurviveBangInInstallPath:
    """Drives each isolated block above through real cmd.exe from a directory
    whose name contains a literal `!`, proving the recorded value keeps the
    bang. Read back two ways per variable: `!VAR!` (delayed-expansion syntax,
    proven safe for a value that already holds a genuine `!` - it is not
    re-scanned the way an ordinary %-substitution is) as a quick sanity
    check, and a %-substitution wrapped in its own
    setlocal-DisableDelayedExpansion/endlocal - the shape the real
    install-manifest-record consumer needs (that line's own protection is a
    separate, already-tracked fix; this proves the value it will read is
    correct once it has one)."""

    def _run(self, directory, preamble, block_text, readback):
        probe = directory / "probe.bat"
        probe.write_text(
            "@echo off\r\nsetlocal EnableDelayedExpansion\r\n"
            + preamble +
            "{block}\r\n"
            "{readback}"
            "exit /b 0\r\n".format(block=block_text, readback=readback),
            encoding="utf-8")
        return subprocess.run(["cmd", "/c", str(probe)], capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=15, cwd=str(directory))

    @staticmethod
    def _protected_readback(pairs):
        """pairs: [(label, varname), ...] -> a setlocal-DisableDelayedExpansion-
        wrapped block of `echo LABEL=[%VAR%]` lines, the shape the real
        manifest-record consumer will use once it is itself protected."""
        lines = ["setlocal DisableDelayedExpansion\r\n"]
        for label, varname in pairs:
            lines.append('echo {}=[%{}%]\r\n'.format(label, varname))
        lines.append("endlocal\r\n")
        return "".join(lines)

    @staticmethod
    def _bang_readback(pairs):
        return "".join('echo {}_BANG=[!{}!]\r\n'.format(label, varname)
                        for label, varname in pairs)

    def test_top_install_message_names_the_real_bang_path(self, bat, tmp_path):
        block = _top_install_message_block(bat)
        bangdir = tmp_path / "bang!dir"
        bangdir.mkdir()
        out = self._run(bangdir, "", block, "")
        assert "self-contained install in: {}".format(bangdir) in out.stdout, out.stdout

    def test_uv_dirs_survive_a_bang_in_the_install_path(self, bat, tmp_path):
        block = _uv_dirs_block(bat)
        bangdir = tmp_path / "bang!dir"
        bangdir.mkdir()
        pairs = [("UPI", "UV_PYTHON_INSTALL_DIR"), ("UCD", "UV_CACHE_DIR")]
        out = self._run(bangdir, 'set "STOREPICK=1"\r\n', block,
                         self._protected_readback(pairs) + self._bang_readback(pairs))
        assert "UPI=[{}\\.python]".format(bangdir) in out.stdout, out.stdout
        assert "UCD=[{}\\.cache]".format(bangdir) in out.stdout, out.stdout
        assert "UPI_BANG=[{}\\.python]".format(bangdir) in out.stdout, out.stdout
        assert "UCD_BANG=[{}\\.cache]".format(bangdir) in out.stdout, out.stdout

    def test_uv_check_portable_dirs_survive_a_bang_in_the_install_path(self, bat, tmp_path):
        block = _uv_check_portable_block(bat)
        bangdir = tmp_path / "bang!dir"
        bangdir.mkdir()
        (bangdir / ".uv").mkdir()
        (bangdir / ".uv" / "uv.exe").write_text("stub", encoding="utf-8")
        pairs = [("UVDIR", "UVDIR")]
        probe = bangdir / "probe.bat"
        probe.write_text(
            "@echo off\r\nsetlocal EnableDelayedExpansion\r\n"
            "{block}\r\n"
            'echo NOT_REACHED_IF_GOTO_FAILED\r\n'
            ":uv_ready\r\n"
            "{protected}"
            "{bang}"
            'echo PATH_HAS=[%PATH%]\r\n'
            "exit /b 0\r\n".format(
                block=block, protected=self._protected_readback(pairs),
                bang=self._bang_readback(pairs)),
            encoding="utf-8")
        out = subprocess.run(["cmd", "/c", str(probe)], capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=15, cwd=str(bangdir))
        assert "NOT_REACHED_IF_GOTO_FAILED" not in out.stdout, out.stdout
        assert "{}\\.uv".format(bangdir) in out.stdout, out.stdout
        assert "UVDIR=[{}\\.uv]".format(bangdir) in out.stdout, out.stdout
        assert "UVDIR_BANG=[{}\\.uv]".format(bangdir) in out.stdout, out.stdout

    def test_uv_missing_contained_dirs_survive_a_bang_in_the_install_path(self, bat, tmp_path):
        block = _uv_missing_contained_block(bat)
        bangdir = tmp_path / "bang!dir"
        bangdir.mkdir()
        pairs = [("UID", "UV_INSTALL_DIR"), ("UVDIR", "UVDIR")]
        out = self._run(bangdir, 'set "CONTAINED=1"\r\n', block,
                         self._protected_readback(pairs) + self._bang_readback(pairs))
        assert "UID=[{}\\.uv]".format(bangdir) in out.stdout, out.stdout
        assert "UVDIR=[{}\\.uv]".format(bangdir) in out.stdout, out.stdout
        assert "UID_BANG=[{}\\.uv]".format(bangdir) in out.stdout, out.stdout
        assert "UVDIR_BANG=[{}\\.uv]".format(bangdir) in out.stdout, out.stdout

    def test_uv_install_dir_is_actually_findable_on_path_after_a_bang_install(
            self, bat, tmp_path):
        """UV_INSTALL_DIR surviving its OWN `set` (the test above) is not
        enough: a few lines later setup.bat rebuilds PATH from it so `where
        uv` can find the freshly-installed binary. Reading UV_INSTALL_DIR
        back there via an ordinary %-substitution corrupts it AGAIN, the
        same way a bare %CD% would - confirmed directly: the first draft of
        this fix read `%UV_INSTALL_DIR%` at that point and `where uv` failed
        to find a real, just-created uv.exe at the correct (bang-preserving)
        location. The PATH entry is prepended fresh from %CD:!=^!% instead.
        A minimal PATH (just enough to resolve `where`/`powershell`) rules
        out a real, pre-existing system `uv` masking the check."""
        block = _uv_missing_full_sequence(bat)
        install_target = "powershell -NoProfile -ExecutionPolicy Bypass -Command \"irm https://astral.sh/uv/install.ps1 | iex\""
        assert install_target in block, "the Astral install invocation text moved; update this test"
        stub_install = (
            'powershell -NoProfile -Command '
            '"New-Item -ItemType Directory -Force $env:UV_INSTALL_DIR | Out-Null; '
            'New-Item -ItemType File -Force (Join-Path $env:UV_INSTALL_DIR \'uv.exe\') | Out-Null"'
        )
        block = block.replace(install_target, stub_install, 1)
        bangdir = tmp_path / "bang!dir"
        bangdir.mkdir()
        system_root = Path(r"C:\Windows\System32")
        minimal_path = "{};{}".format(
            system_root, system_root / "WindowsPowerShell" / "v1.0")
        probe = bangdir / "probe.bat"
        probe.write_text(
            "@echo off\r\nsetlocal EnableDelayedExpansion\r\n"
            'set "CONTAINED=1"\r\n'
            'set "PATH={minimal_path}"\r\n'
            "{block}\r\n"
            'echo NOT_REACHED_IF_GOTO_FAILED\r\n'
            ":uv_ready\r\n"
            'where uv\r\n'
            "exit /b 0\r\n".format(block=block, minimal_path=minimal_path),
            encoding="utf-8")
        out = subprocess.run(["cmd", "/c", str(probe)], capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=15, cwd=str(bangdir))
        assert "NOT_REACHED_IF_GOTO_FAILED" not in out.stdout, out.stdout
        expected_uv = str(bangdir / ".uv" / "uv.exe")
        assert expected_uv in out.stdout.splitlines(), (
            "where uv did not find the just-installed binary at the real "
            "bang-preserving location: {}".format(out.stdout), out.stderr)

    @pytest.mark.parametrize("gcrc", ["0", "20"])
    def test_pathdir_cmdshim_survive_a_bang_in_the_install_path(self, bat, tmp_path, gcrc):
        block = _pathdir_cmdshim_lines(bat)
        bangdir = tmp_path / "bang!dir"
        bangdir.mkdir()
        pairs = [("PATHDIR", "PATHDIR"), ("CMDSHIM", "CMDSHIM")]
        out = self._run(
            bangdir,
            'set "GCRC={}"\r\nset "PATHDIR="\r\nset "CMDSHIM="\r\n'.format(gcrc),
            block, self._protected_readback(pairs) + self._bang_readback(pairs))
        assert "PATHDIR=[{}\\bin]".format(bangdir) in out.stdout, out.stdout
        assert "CMDSHIM=[{}\\bin\\localm.cmd]".format(bangdir) in out.stdout, out.stdout
        assert "PATHDIR_BANG=[{}\\bin]".format(bangdir) in out.stdout, out.stdout
        assert "CMDSHIM_BANG=[{}\\bin\\localm.cmd]".format(bangdir) in out.stdout, out.stdout

    def test_pathdir_cmdshim_stay_empty_when_gcrc_matches_neither(self, bat, tmp_path):
        block = _pathdir_cmdshim_lines(bat)
        bangdir = tmp_path / "bang!dir"
        bangdir.mkdir()
        pairs = [("PATHDIR", "PATHDIR"), ("CMDSHIM", "CMDSHIM")]
        out = self._run(bangdir, 'set "GCRC=99"\r\nset "PATHDIR="\r\nset "CMDSHIM="\r\n',
                         block, self._protected_readback(pairs))
        assert "PATHDIR=[]" in out.stdout, out.stdout
        assert "CMDSHIM=[]" in out.stdout, out.stdout

    def test_pydir_cachedir_survive_a_bang_in_the_install_path(self, bat, tmp_path):
        block = _pydir_cachedir_block(bat)
        bangdir = tmp_path / "bang!dir"
        bangdir.mkdir()
        pairs = [("PYDIR", "PYDIR"), ("CACHEDIR", "CACHEDIR")]
        out = self._run(bangdir, 'set "CONTAINED=1"\r\nset "PYDIR="\r\nset "CACHEDIR="\r\n',
                         block, self._protected_readback(pairs) + self._bang_readback(pairs))
        assert "PYDIR=[{}\\.python]".format(bangdir) in out.stdout, out.stdout
        assert "CACHEDIR=[{}\\.cache]".format(bangdir) in out.stdout, out.stdout
        assert "PYDIR_BANG=[{}\\.python]".format(bangdir) in out.stdout, out.stdout
        assert "CACHEDIR_BANG=[{}\\.cache]".format(bangdir) in out.stdout, out.stdout

    def test_uninstall_header_names_the_real_bang_path(self, bat, tmp_path):
        block = _uninstall_header_block(bat)
        bangdir = tmp_path / "bang!dir"
        bangdir.mkdir()
        out = self._run(bangdir, "", block, "")
        assert str(bangdir) in out.stdout, out.stdout

    def test_manifest_record_line_reads_every_cd_derived_var_correctly_once_protected(
            self, bat, tmp_path):
        """End-to-end: every fixed site's value, fed into the REAL manifest-
        record line's own argument text (not a paraphrase), wrapped in the
        same setlocal-DisableDelayedExpansion/endlocal shape the manifest
        line itself needs (tracked separately) - proving the two fixes
        compose correctly rather than merely each looking right alone."""
        bangdir = tmp_path / "bang!dir"
        bangdir.mkdir()
        block = _manifest_record_block(bat)
        target = ".venv\\Scripts\\python -m localm.install_manifest record"
        assert target in block, "the manifest-record invocation text moved; update this test"
        assert block.endswith(" >nul 2>nul\nendlocal"), \
            "the manifest-record line's shape changed; update this test"
        echoed = (block[: -len(" >nul 2>nul\nendlocal")] + "\nendlocal").replace(
            target, "echo MANIFEST_ARGS", 1)
        script = (
            "@echo off\r\nsetlocal EnableDelayedExpansion\r\n"
            'set "STOREPICK=1"\r\n' + _uv_dirs_block(bat) + "\r\n"
            'set "GCRC=0"\r\nset "PATHDIR="\r\nset "CMDSHIM="\r\n'
            + _pathdir_cmdshim_lines(bat) + "\r\n"
            'set "CONTAINED=1"\r\nset "PYDIR="\r\nset "CACHEDIR="\r\n'
            + _pydir_cachedir_block(bat) + "\r\n"
            'set "SCPATH=C:\\FakeDesktop\\LocaLM.lnk"\r\n'
            'set "UVDIR=%CD:!=^!%\\.uv"\r\nset "PATHMOD=--path-modified"\r\n'
            + echoed + "\r\n"
            "exit /b 0\r\n")
        probe = bangdir / "probe.bat"
        probe.write_text(script, encoding="utf-8")
        out = subprocess.run(["cmd", "/c", str(probe)], capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=15, cwd=str(bangdir))
        assert "MANIFEST_ARGS" in out.stdout, (out.stdout, out.stderr)
        bang = str(bangdir)
        for flag, suffix in [
                ("--python-dir", "\\.python"),
                ("--cache-dir", "\\.cache"),
                ("--path-dir", "\\bin"),
                ("--command-shim", "\\bin\\localm.cmd"),
                ("--uv-dir", "\\.uv"),
        ]:
            expected = '{} "{}{}"'.format(flag, bang, suffix)
            assert expected in out.stdout, (expected, out.stdout, out.stderr)


@pytest.fixture(scope="module")
def venv_template(tmp_path_factory):
    """A real, pip-less venv, copied into a probe directory as .venv so the
    data-folder subroutines run the real install_manifest prepare-data."""
    if os.name != "nt":
        pytest.skip("cmd.exe only")
    d = tmp_path_factory.mktemp("venv-template") / ".venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(d)],
                   check=True, capture_output=True, timeout=300)
    return d


def _clone_with_venv(directory, template):
    directory.mkdir(parents=True)
    shutil.copytree(template, directory / ".venv")
    return directory


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe only")
class TestDataFolderSubroutinesThroughCmd:
    """The real `:do_custom_home` / `:custom_home_blank` / `:portable_home`
    (plus the real `:flush`), driven through cmd.exe with a real venv Python
    running the worktree's install_manifest prepare-data.

    `set /p` stores typed text without re-parsing it, and `!CUSTOMHOME!` is
    expanded once, so a typed path with bangs reaches prepare-data intact;
    every read site below checks that.

    The answers are fed from a FILE: `set /p` reading a pipe takes the whole
    buffer and keeps only its first line, while from a file each line reaches
    its own prompt."""

    def _run(self, bat, clone, stdin_text, entry):
        probe = clone / "probe.bat"
        probe.write_text(
            "@echo off\r\nsetlocal EnableDelayedExpansion\r\n"
            "call :{entry}\r\n"
            "echo EXITED=[%errorlevel%]\r\n"
            "exit /b 0\r\n".format(entry=entry)
            + _data_folder_subroutines(bat) + "\r\n"
            + _flush_block(bat) + "\r\n",
            encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT)
        answers = clone / "answers.txt"
        answers.write_text(stdin_text, encoding="utf-8", newline="")
        with open(answers, "rb") as fh:
            return subprocess.run(["cmd", "/c", str(probe)], stdin=fh,
                                  capture_output=True, text=True, timeout=120,
                                  cwd=str(clone), env=env)

    @staticmethod
    def _record(clone):
        return json.loads((clone / ".localm-install.json").read_text(encoding="utf-8"))

    @pytest.mark.parametrize("dirname", [
        "bang!dir", "ba!ng!dir", "!bangfirst", "banglast!", "a!b!c!d!e",
    ])
    def test_a_typed_path_with_bangs_reaches_every_site_intact(
            self, bat, tmp_path, venv_template, dirname):
        clone = _clone_with_venv(tmp_path / "clone", venv_template)
        target = tmp_path / dirname / "sub"
        out = self._run(bat, clone, "{}\r\nY\r\n".format(target), "do_custom_home")
        assert "EXITED=[0]" in out.stdout, (out.stdout, out.stderr)
        # the confirm prompt names the real path back
        assert "Use '{}'? [Y/n]:".format(target) in out.stdout, out.stdout
        # localm-home.cfg holds the literal path, UTF-8
        cfg = (clone / "localm-home.cfg").read_bytes().decode("utf-8")
        assert cfg.strip() == str(target), out.stdout
        # the folder exists
        assert target.is_dir(), (out.stdout, out.stderr)
        # the install record names it, as a folder setup created
        rec = self._record(clone)
        assert Path(rec["data_dir"]) == target and rec["data_created"] is True
        # the closing lines name it
        assert "Data directory: {}".format(target) in out.stdout, out.stdout
        assert "(recorded in localm-home.cfg)" in out.stdout, out.stdout

    def test_an_existing_folder_is_recorded_with_what_was_in_it(
            self, bat, tmp_path, venv_template):
        clone = _clone_with_venv(tmp_path / "clone", venv_template)
        shared = tmp_path / "AI models"
        (shared / "models").mkdir(parents=True)
        out = self._run(bat, clone, "{}\r\nY\r\n".format(shared), "do_custom_home")
        rec = self._record(clone)
        assert rec["data_created"] is False, (out.stdout, out.stderr)
        assert rec["data_preexisting"] == ["models"]

    def test_a_bang_clone_records_its_portable_home(self, bat, tmp_path, venv_template):
        clone = _clone_with_venv(tmp_path / "bang!clone", venv_template)
        (clone / "localm-home.cfg").write_text("X:\\old", encoding="utf-8")
        out = self._run(bat, clone, "", "portable_home")
        assert (clone / "home" / ".localm-data").is_file(), (out.stdout, out.stderr)
        assert not (clone / "localm-home.cfg").exists()
        assert Path(self._record(clone)["data_dir"]) == clone / "home"
        assert "Data directory: {}".format(clone / "home") in out.stdout, out.stdout

    def test_a_blank_path_falls_back_to_portable(self, bat, tmp_path, venv_template):
        clone = _clone_with_venv(tmp_path / "clone", venv_template)
        out = self._run(bat, clone, "\r\n", "do_custom_home")
        assert "No path given" in out.stdout, out.stdout
        assert (clone / "home").is_dir()
        assert not (clone / "localm-home.cfg").exists()

    def test_a_relative_path_is_refused_and_asked_again(self, bat, tmp_path, venv_template):
        clone = _clone_with_venv(tmp_path / "clone", venv_template)
        target = tmp_path / "second try"
        out = self._run(bat, clone, "models\r\nY\r\n{}\r\nY\r\n".format(target),
                        "do_custom_home")
        assert "Cannot use that data folder" in out.stdout, out.stdout
        assert not (clone / "models").exists()
        assert target.is_dir(), out.stdout
        assert Path(self._record(clone)["data_dir"]) == target

    def test_answering_no_asks_for_the_path_again(self, bat, tmp_path, venv_template):
        clone = _clone_with_venv(tmp_path / "clone", venv_template)
        wrong, right = tmp_path / "wrong", tmp_path / "right"
        out = self._run(bat, clone, "{}\r\nn\r\n{}\r\nY\r\n".format(wrong, right),
                        "do_custom_home")
        assert not wrong.exists(), out.stdout
        assert right.is_dir(), out.stdout
        assert Path(self._record(clone)["data_dir"]) == right


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
