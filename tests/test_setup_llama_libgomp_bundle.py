# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bundling libgomp.so.1 into the Linux runtime dir.

Upstream's Linux `cpu`/`vulkan` llama.cpp release tarballs link OpenMP
dynamically (libggml-base.so.0 and every libggml-cpu-*.so NEED
libgomp.so.1) and ship no copy of their own, so a bare/minimal Linux image
without a system libgomp cannot load the runtime - see
_bundle_missing_native_deps's own docstring in setup_llama.py for the full
mechanism.

Covers, each against the REAL code path rather than a mock of it:
  - _read_ar_archive / _extract_libgomp_from_deb: a real ar+tar(.gz) .deb
    fixture built in-test, including the real-file-vs-symlink distinction
    Debian's own packaging uses.
  - _bundle_missing_native_deps: fires only when an extracted .so actually
    needs libgomp.so.1 and nothing already provides it (item 19's both-arms
    check: the fixture can express BOTH the triggering and non-triggering
    case), and never raises on a fetch/checksum failure.
  - _name_missing_shared_lib: the plain-words message defect 4 asks for.
"""

from __future__ import annotations

import hashlib
import io
import struct
import tarfile

import pytest

from localm import setup_llama as sl


# --------------------------------------------------------------------------- #
#  Minimal real ELF64 .so builder (shared shape with test_elf_deps.py, kept   #
#  local and small rather than importing test internals across files)         #
# --------------------------------------------------------------------------- #

def _minimal_elf64_needing(*needed: str) -> bytes:
    strtab = b"\x00"
    offsets = []
    for name in needed:
        offsets.append(len(strtab))
        strtab += name.encode("ascii") + b"\x00"
    dynamic = b"".join(struct.pack("<qQ", 1, o) for o in offsets)
    dynamic += struct.pack("<qQ", 0, 0)

    strtab_off = 64
    dynamic_off = strtab_off + len(strtab)
    shoff = dynamic_off + len(dynamic)

    ehdr = bytearray(64)
    ehdr[0:4] = b"\x7fELF"
    ehdr[4], ehdr[5], ehdr[6] = 2, 1, 1
    struct.pack_into("<Q", ehdr, 0x28, shoff)
    struct.pack_into("<H", ehdr, 0x34, 64)
    struct.pack_into("<H", ehdr, 0x3A, 64)
    struct.pack_into("<H", ehdr, 0x3C, 3)
    struct.pack_into("<H", ehdr, 0x3E, 0)

    def shdr(sh_type, off, size, link=0):
        return struct.pack("<IIQQQQIIQQ", 0, sh_type, 0, 0, off, size, link, 0, 0, 0)

    sections = shdr(0, 0, 0) + shdr(3, strtab_off, len(strtab)) + shdr(6, dynamic_off, len(dynamic), link=1)
    return bytes(ehdr) + strtab + dynamic + sections


# --------------------------------------------------------------------------- #
#  A real Debian-.deb-shaped fixture: ar archive containing a gzip data.tar   #
# --------------------------------------------------------------------------- #

def _ar_member_header(name: str, size: int) -> bytes:
    # Matches _read_ar_archive's own reader: 16-byte name, then padding
    # fields it does not use, then a 10-byte size field, all ASCII.
    header = name.ljust(16)
    header += "0".ljust(12)      # mtime
    header += "0".ljust(6)       # uid
    header += "0".ljust(6)       # gid
    header += "100644".ljust(8)  # mode
    header += str(size).ljust(10)
    header += "\x60\n"
    assert len(header) == 60
    return header.encode("ascii")


def _build_ar_archive(members: dict) -> bytes:
    out = bytearray(b"!<arch>\n")
    for name, content in members.items():
        out += _ar_member_header(name, len(content))
        out += content
        if len(content) % 2:
            out += b"\n"
    return bytes(out)


def _build_fake_deb(*, so_content: bytes = b"FAKE-LIBGOMP-BYTES",
                    include_symlink: bool = True) -> bytes:
    """A real ar(deb) + tar(gz) archive shaped like Debian's libgomp1: a
    REGULAR FILE libgomp.so.1.0.0 plus (optionally) a SYMLINK libgomp.so.1
    pointing at it - exactly Debian's own layout, so the "pick the regular
    file, not the symlink" logic is exercised against a real tar, not an
    assumption about it."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        ti = tarfile.TarInfo("./usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0")
        ti.size = len(so_content)
        tf.addfile(ti, io.BytesIO(so_content))
        if include_symlink:
            link = tarfile.TarInfo("./usr/lib/x86_64-linux-gnu/libgomp.so.1")
            link.type = tarfile.SYMTYPE
            link.linkname = "libgomp.so.1.0.0"
            tf.addfile(link)
        # An unrelated file, so a naive "first regular file" picker would fail.
        other = tarfile.TarInfo("./usr/share/doc/libgomp1/changelog.gz")
        other.size = 5
        tf.addfile(other, io.BytesIO(b"noise"))
    data_tar_gz = buf.getvalue()
    return _build_ar_archive({
        "debian-binary": b"2.0\n",
        "control.tar.gz": b"",   # never read by the extractor
        "data.tar.gz": data_tar_gz,
    })


# --------------------------------------------------------------------------- #
#  _read_ar_archive / _extract_libgomp_from_deb                               #
# --------------------------------------------------------------------------- #

def test_read_ar_archive_recovers_every_member_byte_for_byte(tmp_path):
    deb = tmp_path / "pkg.deb"
    deb.write_bytes(_build_ar_archive({"a": b"hello", "bb": b"world!!"}))
    members = sl._read_ar_archive(deb)
    assert members["a"] == b"hello"
    assert members["bb"] == b"world!!"


def test_read_ar_archive_rejects_a_non_ar_file(tmp_path):
    p = tmp_path / "not-a-deb.deb"
    p.write_bytes(b"PK\x03\x04 this is actually a zip file, not an ar archive")
    with pytest.raises(sl.ArtifactError, match="not an ar archive"):
        sl._read_ar_archive(p)


def test_extract_libgomp_from_deb_picks_the_real_file_not_the_symlink(tmp_path):
    deb = tmp_path / "libgomp1.deb"
    deb.write_bytes(_build_fake_deb(so_content=b"REAL-LIBRARY-BYTES", include_symlink=True))
    out = sl._extract_libgomp_from_deb(deb, tmp_path)
    assert out.read_bytes() == b"REAL-LIBRARY-BYTES"


def test_extract_libgomp_from_deb_works_without_the_symlink_too(tmp_path):
    """THE NEGATIVE CASE for the symlink itself: real Debian packaging always
    ships one, but the extractor's contract is "find the regular file",
    which must not silently depend on a symlink also being present."""
    deb = tmp_path / "libgomp1.deb"
    deb.write_bytes(_build_fake_deb(so_content=b"NO-SYMLINK-HERE", include_symlink=False))
    out = sl._extract_libgomp_from_deb(deb, tmp_path)
    assert out.read_bytes() == b"NO-SYMLINK-HERE"


def test_extract_libgomp_from_deb_raises_when_no_data_tar_member(tmp_path):
    deb = tmp_path / "empty.deb"
    deb.write_bytes(_build_ar_archive({"debian-binary": b"2.0\n"}))
    with pytest.raises(sl.ArtifactError, match="data.tar"):
        sl._extract_libgomp_from_deb(deb, tmp_path)


def test_extract_libgomp_from_deb_raises_when_libgomp_absent_from_data_tar(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        ti = tarfile.TarInfo("./usr/lib/x86_64-linux-gnu/libssl.so.3")
        ti.size = 3
        tf.addfile(ti, io.BytesIO(b"abc"))
    deb = tmp_path / "wrong-package.deb"
    deb.write_bytes(_build_ar_archive({"data.tar.gz": buf.getvalue()}))
    with pytest.raises(sl.ArtifactError, match="libgomp"):
        sl._extract_libgomp_from_deb(deb, tmp_path)


# --------------------------------------------------------------------------- #
#  _bundle_missing_native_deps: both arms (item 19 - the fixture must be able #
#  to express BOTH "needed" and "not needed")                                #
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_download(monkeypatch):
    """Redirects sl._download to write a real fake-.deb fixture instead of
    touching the network, keyed to sl._LIBGOMP_DEB_SHA256 so the checksum
    gate is exercised for real. Also pins sys.platform to "linux": bundling
    is Linux-only, and these tests run on whatever platform CI/the dev box
    actually is (this test suite runs on Windows too) - callers that want to
    test the platform gate itself override sl.sys.platform again afterward."""
    monkeypatch.setattr(sl.sys, "platform", "linux")
    calls = []

    def _install(deb_bytes: bytes, sha_override: str | None = None):
        def fake(url, dest):
            calls.append(url)
            dest.write_bytes(deb_bytes)
            return sl._DownloadResult(bytes_received=len(deb_bytes),
                                      content_length=len(deb_bytes),
                                      content_type="application/x-debian-package",
                                      final_url=url)
        monkeypatch.setattr(sl, "_download", fake)
        monkeypatch.setattr(sl, "_LIBGOMP_DEB_SHA256",
                            sha_override or hashlib.sha256(deb_bytes).hexdigest())
        monkeypatch.setattr(sl, "_LIBGOMP_DEB_MIN_BYTES", 1)
        return calls

    return _install


def test_bundles_libgomp_when_an_extracted_so_needs_it(tmp_path, monkeypatch, fake_download):
    target = tmp_path / "runtime"
    target.mkdir()
    (target / "libggml-base.so.0").write_bytes(_minimal_elf64_needing("libgomp.so.1", "libc.so.6"))
    (target / "libllama.so").write_bytes(_minimal_elf64_needing("libggml-base.so.0"))

    deb_bytes = _build_fake_deb(so_content=b"THE-REAL-LIBGOMP")
    calls = fake_download(deb_bytes)

    sl._bundle_missing_native_deps(target)

    assert calls, "libgomp1 must actually have been fetched"
    assert (target / "libgomp.so.1").read_bytes() == b"THE-REAL-LIBGOMP"
    assert (target / "LICENSE.libgomp").exists()


def test_does_not_bundle_when_nothing_needs_libgomp(tmp_path, monkeypatch, fake_download):
    """THE OTHER ARM: a backend whose .so files never mention libgomp.so.1
    must not trigger a fetch at all - the fixture space includes BOTH
    outcomes, so this test could not pass by accident (see item 19)."""
    target = tmp_path / "runtime"
    target.mkdir()
    (target / "libggml-base.so.0").write_bytes(_minimal_elf64_needing("libc.so.6", "libm.so.6"))

    calls = fake_download(_build_fake_deb())

    sl._bundle_missing_native_deps(target)

    assert calls == [], "no fetch when nothing needs libgomp"
    assert not (target / "libgomp.so.1").exists()
    assert not (target / "LICENSE.libgomp").exists()


def test_skips_when_libgomp_is_already_present(tmp_path, monkeypatch, fake_download):
    target = tmp_path / "runtime"
    target.mkdir()
    (target / "libggml-base.so.0").write_bytes(_minimal_elf64_needing("libgomp.so.1"))
    (target / "libgomp.so.1").write_bytes(b"already here")

    calls = fake_download(_build_fake_deb())

    sl._bundle_missing_native_deps(target)

    assert calls == [], "already-provided means no fetch"
    assert (target / "libgomp.so.1").read_bytes() == b"already here", "must not overwrite"


def test_wrong_checksum_is_refused_and_never_raises(tmp_path, monkeypatch, fake_download, caplog):
    """A tampered/wrong download must never be installed - AND must never
    abort a provision that would otherwise succeed (the docstring's 'never
    raises' contract): a warning is logged instead."""
    target = tmp_path / "runtime"
    target.mkdir()
    (target / "libggml-base.so.0").write_bytes(_minimal_elf64_needing("libgomp.so.1"))

    fake_download(_build_fake_deb(), sha_override="0" * 64)   # deliberately wrong

    sl._bundle_missing_native_deps(target)   # must not raise

    assert not (target / "libgomp.so.1").exists()
    assert any("could not bundle" in r.message for r in caplog.records)


def test_platform_gated_off_on_windows_and_macos(tmp_path, monkeypatch, fake_download):
    target = tmp_path / "runtime"
    target.mkdir()
    (target / "libggml-base.so.0").write_bytes(_minimal_elf64_needing("libgomp.so.1"))
    calls = fake_download(_build_fake_deb())
    for plat in ("win32", "darwin"):
        monkeypatch.setattr(sl.sys, "platform", plat)
        sl._bundle_missing_native_deps(target)
    assert calls == []


# --------------------------------------------------------------------------- #
#  _name_missing_shared_lib                                                   #
# --------------------------------------------------------------------------- #

def test_names_a_known_missing_library_with_its_package():
    msg = sl._name_missing_shared_lib(
        "RuntimeError: Failed to load libllama.so from /x/libllama.so: "
        "libgomp.so.1: cannot open shared object file: No such file or directory")
    assert msg is not None
    assert "libgomp.so.1" in msg
    assert "libgomp1" in msg


def test_names_an_unknown_missing_library_generically():
    msg = sl._name_missing_shared_lib(
        "RuntimeError: Failed to load libllama.so: "
        "libsome-weird-vendor-lib.so.7: cannot open shared object file")
    assert msg is not None
    assert "libsome-weird-vendor-lib.so.7" in msg
    assert "install the OS package" in msg


def test_returns_none_when_detail_does_not_match_the_dlopen_shape():
    assert sl._name_missing_shared_lib("no compute backends are loaded") is None
    assert sl._name_missing_shared_lib("") is None
    assert sl._name_missing_shared_lib(None) is None
