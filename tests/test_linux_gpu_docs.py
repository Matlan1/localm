# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Linux GPU install route must be documented as first-class, with the
pip-extra caveat and WSL2/VM caveats (the `[gpu]` extra cannot embed an index
URL, so Linux users need setup.sh / a manual --index-url install).
"""

from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "linux-setup.md"


def test_linux_doc_has_gpu_install_route():
    text = DOC.read_text(encoding="utf-8")
    assert "--index-url" in text                     # the manual install command


def test_linux_doc_notes_pip_extra_is_windows_only():
    text = DOC.read_text(encoding="utf-8").lower()
    assert "localm[gpu]" in text
    assert "windows-only" in text


def test_linux_doc_has_wsl2_and_vm_caveats():
    text = DOC.read_text(encoding="utf-8").lower()
    assert "wsl2" in text
    assert "passthrough" in text
