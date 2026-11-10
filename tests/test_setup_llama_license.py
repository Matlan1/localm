# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup-llama must ship the upstream MIT license text alongside the bundled
llama.cpp/ggml binaries: _copy_binaries captures the archive's license, or a
bundled MIT fallback, into the runtime lib/.
"""

import sys

import pytest

from localm.setup_llama import _LLAMA_CPP_MIT_NOTICE, _copy_binaries, _license_dest_name


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


def test_a_qualified_license_keeps_its_own_name_beside_the_mit_notice(tmp_path):
    """Upstream's Windows zips carry only LICENSE-LLVM-OpenMP: it is kept as
    LICENSE.llvm-openmp, and LICENSE.llama-cpp still holds llama.cpp's MIT text."""
    src = tmp_path / "src"
    src.mkdir()
    _fake_binary(src)
    (src / "LICENSE-LLVM-OpenMP").write_bytes(b"Apache License v2.0 with LLVM Exceptions")
    target = tmp_path / "lib"
    target.mkdir()

    _copy_binaries(src, target)

    assert (target / "LICENSE.llvm-openmp").read_bytes() == \
        b"Apache License v2.0 with LLVM Exceptions"
    assert (target / "LICENSE.llama-cpp").read_text(encoding="utf-8") == _LLAMA_CPP_MIT_NOTICE


def test_a_second_archive_without_a_license_keeps_the_first_ones(tmp_path):
    """The Windows CUDA runtime bundle carries no license file; placing it after
    the build leaves the build's LICENSE.llama-cpp as it was."""
    target = tmp_path / "lib"
    target.mkdir()
    build = tmp_path / "build"
    build.mkdir()
    _fake_binary(build)
    (build / "LICENSE").write_bytes(b"UPSTREAM MIT TEXT")
    cudart = tmp_path / "cudart"
    cudart.mkdir()
    _fake_binary(cudart)

    _copy_binaries(build, target)
    _copy_binaries(cudart, target)

    assert (target / "LICENSE.llama-cpp").read_bytes() == b"UPSTREAM MIT TEXT"
    assert sorted(p.name for p in _license_files(target)) == ["LICENSE.llama-cpp"]


def test_a_second_archive_never_overwrites_an_existing_license_file(tmp_path):
    target = tmp_path / "lib"
    target.mkdir()
    for name, text in (("first", b"FIRST TEXT"), ("second", b"SECOND TEXT"),
                       ("third", b"FIRST TEXT")):
        src = tmp_path / name
        src.mkdir()
        _fake_binary(src)
        (src / "LICENSE").write_bytes(text)
        _copy_binaries(src, target)

    assert (target / "LICENSE.llama-cpp").read_bytes() == b"FIRST TEXT"
    assert (target / "LICENSE.llama-cpp.1").read_bytes() == b"SECOND TEXT"
    assert sorted(p.name for p in _license_files(target)) == [
        "LICENSE.llama-cpp", "LICENSE.llama-cpp.1"]


def test_the_top_level_license_is_llama_cpps(tmp_path):
    """A nested vendor LICENSE (in a folder that sorts before it) does not take
    llama.cpp's name from the archive's own; it is named after its folder."""
    src = tmp_path / "src"
    (src / "Abseil").mkdir(parents=True)
    _fake_binary(src)
    (src / "Abseil" / "LICENSE").write_bytes(b"VENDOR TEXT")
    (src / "LICENSE").write_bytes(b"TOP TEXT")
    target = tmp_path / "lib"
    target.mkdir()

    _copy_binaries(src, target)

    assert (target / "LICENSE.llama-cpp").read_bytes() == b"TOP TEXT"
    assert (target / "LICENSE.abseil").read_bytes() == b"VENDOR TEXT"


def test_a_nested_vendor_license_alone_still_gets_the_mit_notice(tmp_path):
    src = tmp_path / "src"
    (src / "_deps" / "json").mkdir(parents=True)
    _fake_binary(src)
    (src / "_deps" / "json" / "LICENSE.txt").write_bytes(b"VENDOR TEXT")
    target = tmp_path / "lib"
    target.mkdir()

    _copy_binaries(src, target)

    assert (target / "LICENSE.json").read_bytes() == b"VENDOR TEXT"
    assert (target / "LICENSE.llama-cpp").read_text(encoding="utf-8") == _LLAMA_CPP_MIT_NOTICE


def test_a_tarball_license_beside_its_libraries_is_llama_cpps(tmp_path):
    """Upstream's tarballs wrap everything in one folder: llama-<tag>/LICENSE sits
    at the same depth as the libraries."""
    src = tmp_path / "src"
    (src / "llama-b1").mkdir(parents=True)
    _fake_binary(src / "llama-b1")
    (src / "llama-b1" / "LICENSE").write_bytes(b"UPSTREAM MIT")
    target = tmp_path / "lib"
    target.mkdir()

    _copy_binaries(src, target)

    assert (target / "LICENSE.llama-cpp").read_bytes() == b"UPSTREAM MIT"


@pytest.mark.parametrize("name, expected", [
    ("LICENSE", "LICENSE.llama-cpp"),
    ("LICENSE.txt", "LICENSE.llama-cpp"),
    ("license.md", "LICENSE.llama-cpp"),
    ("COPYING", "LICENSE.llama-cpp"),
    ("LICENSE-LLVM-OpenMP", "LICENSE.llvm-openmp"),
    ("LICENSE_THIRD_PARTY.txt", "LICENSE.third-party"),
    ("NOTICE", "NOTICE.llama-cpp"),
    ("NOTICE-foo.txt", "NOTICE.foo"),
])
def test_license_dest_name(name, expected):
    assert _license_dest_name(name) == expected
