# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.bat/setup.sh must never install uv itself outside the clone without asking."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _bat() -> str:
    return (ROOT / "setup.bat").read_text(encoding="utf-8")


def _sh() -> str:
    return (ROOT / "setup.sh").read_text(encoding="utf-8")


def test_bat_asks_portable_vs_shared_before_installing_uv():
    text = _bat()
    ask_idx = text.index("Keep localm's Python tooling")
    install_prompt_idx = text.index(
        "uv (the Python package manager localm builds on) is not installed")
    assert ask_idx < install_prompt_idx, (
        "setup.bat must ask where to keep uv/Python tooling BEFORE offering to "
        "install uv, so a Portable pick can confine uv's own binary to the clone")


def test_sh_asks_portable_vs_shared_before_installing_uv():
    text = _sh()
    ask_idx = text.index("Keep localm's Python tooling")
    install_prompt_idx = text.index(
        "uv (the Python package manager localm builds on) is not installed")
    assert ask_idx < install_prompt_idx, (
        "setup.sh must ask where to keep uv/Python tooling BEFORE offering to "
        "install uv, so a Portable pick can confine uv's own binary to the clone")


def test_bat_sets_uv_install_dir_before_the_installer_when_contained():
    text = _bat()
    contained_set_idx = text.index('set "UV_INSTALL_DIR=%CD%\\.uv"')
    # The real invocation, not the "install it yourself" messages shown when the
    # user declines or the bootstrap fails - those `echo` the identical URL as
    # plain text, so match on the unique `-NoProfile ... -Command` prefix only
    # the real invocation carries.
    installer_idx = text.index(
        'powershell -NoProfile -ExecutionPolicy Bypass -Command '
        '"irm https://astral.sh/uv/install.ps1')
    assert contained_set_idx < installer_idx, (
        "UV_INSTALL_DIR must be set before invoking Astral's installer, or a "
        "Portable pick has no effect on where uv itself lands")
    # Only set inside the CONTAINED branch, not unconditionally.
    assert 'if "%CONTAINED%"=="1" (' in text


def test_sh_sets_uv_install_dir_before_the_installer_when_contained():
    text = _sh()
    contained_set_idx = text.index('export UV_INSTALL_DIR="$(pwd)/.uv"')
    # The real invocation, not the "install it yourself" messages shown when the
    # user declines or the bootstrap fails - those contain the identical command
    # as plain text and appear earlier/later in the file, so match on the `|| true`
    # suffix that ONLY the real invocation carries.
    installer_idx = text.index(
        "curl -LsSf https://astral.sh/uv/install.sh | sh || true")
    assert contained_set_idx < installer_idx, (
        "UV_INSTALL_DIR must be exported before invoking Astral's installer, or "
        "a Portable pick has no effect on where uv itself lands")
    assert 'if [ "$CONTAINED" = 1 ]; then' in text


def test_bat_records_and_passes_uv_dir_to_the_manifest():
    text = _bat()
    assert "--uv-dir" in text
    record_line = next(
        line for line in text.splitlines() if "install_manifest record" in line)
    assert "--uv-dir" in record_line


def test_sh_records_and_passes_uv_dir_to_the_manifest():
    text = _sh()
    assert "--uv-dir" in text


def test_bat_uv_install_dir_is_a_documented_astral_override_not_invented():
    # UV_INSTALL_DIR is already read elsewhere in the SAME file (the PATH-prepend
    # after installing) - confirms it is Astral's real, already-relied-upon
    # override, not a made-up variable name.
    text = _bat()
    assert "if defined UV_INSTALL_DIR" in text
