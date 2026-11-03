# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.bat/setup.sh must never install uv itself outside the clone without
asking. The portable/shared choice is asked BEFORE uv is ever installed, and a
Portable pick sets Astral's own UV_INSTALL_DIR override so uv itself lands
inside the clone too, tracked in the install manifest for a symmetric
uninstall. setup.bat installs localm's own uv without asking whether to: a
Shared pick puts it in localm's Shared folder."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _bat() -> str:
    return (ROOT / "setup.bat").read_text(encoding="utf-8")


def _sh() -> str:
    return (ROOT / "setup.sh").read_text(encoding="utf-8")


_BAT_INSTALLER = '"%PSEXE%" -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference'


def test_bat_asks_portable_vs_shared_before_installing_uv():
    text = _bat()
    ask_idx = text.index("Keep localm's Python tooling")
    assert ask_idx < text.index(_BAT_INSTALLER), (
        "setup.bat must ask where to keep uv/Python tooling BEFORE installing uv, "
        "so a Portable pick can confine uv's own binary to the clone")
    assert "Install it now with Astral's official installer" not in text


def test_sh_asks_portable_vs_shared_before_installing_uv():
    text = _sh()
    ask_idx = text.index("Keep localm's Python tooling")
    install_idx = text.index('fetch_uv_installer "$UVTMP/uv-installer.sh"')
    assert ask_idx < install_idx, (
        "setup.sh must ask where to keep uv/Python tooling BEFORE installing uv, "
        "so the pick decides the folder uv itself lands in")


def test_bat_sets_uv_install_dir_before_the_installer_when_contained():
    text = _bat()
    # See TestCdDerivedVarsSurviveBangInInstallPath.
    installer_idx = text.index(_BAT_INSTALLER)
    for line in ('set "UV_INSTALL_DIR=!UVDIR!"', 'set "UV_UNMANAGED_INSTALL=!UVDIR!"'):
        assert text.index(line) < installer_idx, (
            f"{line} must come before Astral's installer, or the store choice has "
            "no effect on where uv itself lands")
    contained = text[text.index('if "%CONTAINED%"=="1" (\n'):]
    assert contained.index('set "UVDIR=!CD!\\.uv"') < contained.index(") else (")


def test_sh_sets_uv_install_dir_before_the_installer_in_both_modes():
    text = _sh()
    assert 'STORE_UV="$ROOT/.uv"' in text and 'STORE_UV="$SHARED_STORE/uv"' in text
    set_idx = text.index('export UV_INSTALL_DIR="$STORE_UV"')
    unmanaged_idx = text.index('export UV_UNMANAGED_INSTALL="$STORE_UV"')
    installer_idx = text.index('sh "$UVTMP/uv-installer.sh" || true')
    assert set_idx < installer_idx and unmanaged_idx < installer_idx, (
        "UV_INSTALL_DIR and UV_UNMANAGED_INSTALL must be exported before invoking "
        "Astral's installer, or uv lands outside localm's own store")


def test_bat_records_and_passes_uv_dir_to_the_manifest():
    text = _bat()
    assert "--uv-dir" in text
    record_line = next(
        line for line in text.splitlines() if "install_manifest record" in line)
    assert "--uv-dir" in record_line


def test_sh_records_and_passes_uv_dir_to_the_manifest():
    text = _sh()
    assert "--uv-dir" in text


def test_bat_shared_installs_uv_into_localm_own_shared_folder():
    text = _bat()
    shared = text[text.index('if "%CONTAINED%"=="1" (\n'):]
    shared = shared[shared.index(") else (\n"):shared.index("\n)\n")]
    assert 'set "UVDIR=!LOCALAPPDATA!\\localm\\shared\\uv"' in shared
