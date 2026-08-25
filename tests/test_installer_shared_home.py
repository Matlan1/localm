# SPDX-License-Identifier: AGPL-3.0-or-later
"""The installers must NEVER use ~/.localm as a data directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_setup_bat_never_uses_localm_as_data_dir():
    text = (ROOT / "setup.bat").read_text(encoding="utf-8")
    # No data-dir assignment to the per-user ~/.localm (the old shared default).
    assert "%USERPROFILE%\\.localm" not in text
    # The default data dir is the contained ./home.
    assert 'set "DATADIR=%CD%\\home"' in text


def test_setup_sh_never_uses_localm_as_data_dir():
    text = (ROOT / "setup.sh").read_text(encoding="utf-8")
    assert "$HOME/.localm" not in text
    assert 'DATA_DIR="$(pwd)/home"' in text


def test_installers_offer_portable_default_and_custom_only():
    bat = (ROOT / "setup.bat").read_text(encoding="utf-8")
    sh = (ROOT / "setup.sh").read_text(encoding="utf-8")
    # The old "Shared (~/.localm)" data option is gone from both.
    assert "Shared (%USERPROFILE%\\.localm)" not in bat
    assert "Shared per-user (~/.localm)" not in sh
    # Portable ./home is the offered default in both.
    assert "Portable (.\\home)" in bat
    assert "Portable (./home)" in sh
