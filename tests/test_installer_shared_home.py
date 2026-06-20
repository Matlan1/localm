# SPDX-License-Identifier: AGPL-3.0-or-later
"""B8: in shared mode the installers tried to remove ./home with a NON-recursive
remover and suppressed stderr, so a non-empty ./home survived while setup printed
"(shared)" - and config.py then silently kept using the leftover portable ./home,
ignoring the user's choice. Both installers must now surface that instead of
failing silently (without destructively deleting a home that may hold models).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _shared_branch(name, start_marker, end_marker):
    text = (ROOT / name).read_text(encoding="utf-8")
    i = text.index(start_marker)
    j = text.index(end_marker, i)
    return text[i:j]


def test_setup_bat_warns_when_home_blocks_shared():
    # NEGATIVE: pre-fix the shared branch only did `rd "home" 2>nul` and printed
    # "(shared)" unconditionally, with no warning.
    branch = _shared_branch("setup.bat", 'if "%DATAPICK%"=="2"', 'if "%DATAPICK%"=="3"')
    assert "NOT take effect" in branch


def test_setup_sh_warns_when_home_blocks_shared():
    branch = _shared_branch("setup.sh", "dpick", "application menu entry")
    assert "NOT take effect" in branch


def test_both_installers_share_the_warning_wording():
    bat = (ROOT / "setup.bat").read_text(encoding="utf-8")
    sh = (ROOT / "setup.sh").read_text(encoding="utf-8")
    assert "shared mode will NOT take effect" in bat
    assert "shared mode will NOT take effect" in sh
