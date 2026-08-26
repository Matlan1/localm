# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup-llama must ship the upstream MIT license text alongside the bundled
llama.cpp/ggml binaries: _copy_binaries captures the archive's license, or a
bundled MIT fallback, into the runtime lib/.
"""

import sys

from localm.setup_llama import _copy_binaries


def _fake_binary(d):
    # Create a binary matching the current platform's _is_wanted rule.
    name = ("llama.dll" if sys.platform == "win32"
            else "libllama.dylib" if sys.platform == "darwin"
            else "libllama.so")
    (d / name).write_bytes(b"\x00binary")


def _license_files(target):
    return [p for p in target.iterdir()
            if p.name.lower().startswith(("license", "licence", "copying", "notice"))]


def test_upstream_license_is_copied(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _fake_binary(src)
    (src / "LICENSE").write_text(
        "MIT License\n\nPermission is hereby granted, free of charge ...\n",
        encoding="utf-8")
    target = tmp_path / "lib"
    target.mkdir()

    assert _copy_binaries(src, target) == 1          # one binary copied
    lics = _license_files(target)
    assert lics, "no license file shipped alongside the binary"
    assert "Permission is hereby granted" in lics[0].read_text(encoding="utf-8")


def test_fallback_mit_notice_when_archive_has_none(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _fake_binary(src)                                  # binary, but NO license file
    target = tmp_path / "lib"
    target.mkdir()

    _copy_binaries(src, target)
    notice = target / "LICENSE.llama-cpp"
    assert notice.is_file()
    text = notice.read_text(encoding="utf-8")
    assert "MIT" in text and "ggml authors" in text


def test_no_binaries_means_no_license_litter(tmp_path):
    # An empty/no-binary source must not drop a stray license file.
    src = tmp_path / "src"
    src.mkdir()
    (src / "readme.txt").write_text("not a binary", encoding="utf-8")
    target = tmp_path / "lib"
    target.mkdir()

    assert _copy_binaries(src, target) == 0
    assert _license_files(target) == []


def test_notices_doc_names_llama_and_mit():
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1] / "THIRD-PARTY-NOTICES.md").read_text(
        encoding="utf-8")
    assert "llama.cpp" in doc and "MIT" in doc
