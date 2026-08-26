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
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BAT = ROOT / "setup.bat"


@pytest.fixture(scope="module")
def bat():
    return BAT.read_text(encoding="utf-8", errors="replace")


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
    assert bat.count('if not errorlevel 1 set "SCMADE=1"') == 2, \
        "both shortcut branches must record whether they actually succeeded"


def test_make_launcher_quiet_prints_no_competing_start_instruction(monkeypatch):
    """--quiet keeps the notes and the failure path, drops the hints."""
    from click.testing import CliRunner
    from localm import applaunch
    from localm.cli import maintenance

    fake = applaunch.LauncherResult(
        ok=True, path=Path("X:/clone/.venv/localm-app/LocaLM.exe"),
        notes=["built LocaLM.exe from python.exe + 4 runtime DLL(s)"])
    monkeypatch.setattr(applaunch, "make_launcher", lambda force=False: fake)

    loud = CliRunner().invoke(maintenance.make_launcher_cmd, [])
    quiet = CliRunner().invoke(maintenance.make_launcher_cmd, ["--quiet"])

    assert loud.exit_code == 0 and quiet.exit_code == 0
    # The note (real work done) survives both; the hints only appear when loud.
    assert "built LocaLM.exe" in loud.output and "built LocaLM.exe" in quiet.output
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
