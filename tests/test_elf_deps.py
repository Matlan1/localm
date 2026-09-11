# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.elf_deps.needed_libraries: a pure-stdlib ELF64-LE DT_NEEDED reader.

The fixture below is a hand-built, minimal-but-real ELF64 file (a null
section, a .dynstr-shaped STRTAB, and a .dynamic section whose sh_link
points at that strtab - exactly the structure the parser is documented to
rely on) rather than a mock of the parser's own internals, so a change to
the parsing logic itself is what these tests actually exercise.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

from localm import elf_deps

_SHT_NULL = 0
_SHT_DYNAMIC = 6
_SHT_STRTAB = 3
_DT_NEEDED = 1
_DT_NULL = 0


def _shdr(sh_type: int, sh_offset: int, sh_size: int, sh_link: int = 0) -> bytes:
    return struct.pack("<IIQQQQIIQQ", 0, sh_type, 0, 0, sh_offset, sh_size,
                       sh_link, 0, 0, 0)


def _build_elf64(needed: tuple, *, ei_class: int = 2, ei_data: int = 1) -> bytes:
    """A minimal ELF64 whose .dynamic section declares DT_NEEDED for each
    name in *needed*, resolved through a real .dynstr-shaped string table via
    the .dynamic section's own sh_link - not a hardcoded section index."""
    strtab = b"\x00"
    name_offsets = []
    for name in needed:
        name_offsets.append(len(strtab))
        strtab += name.encode("ascii") + b"\x00"

    dynamic = b""
    for off in name_offsets:
        dynamic += struct.pack("<qQ", _DT_NEEDED, off)
    dynamic += struct.pack("<qQ", _DT_NULL, 0)

    header_size = 64
    strtab_off = header_size
    dynamic_off = strtab_off + len(strtab)
    shoff = dynamic_off + len(dynamic)

    ehdr = bytearray(64)
    ehdr[0:4] = b"\x7fELF"
    ehdr[4] = ei_class
    ehdr[5] = ei_data
    ehdr[6] = 1  # EI_VERSION
    struct.pack_into("<H", ehdr, 0x10, 3)          # e_type = ET_DYN
    struct.pack_into("<H", ehdr, 0x12, 62)         # e_machine = EM_X86_64
    struct.pack_into("<I", ehdr, 0x14, 1)          # e_version
    struct.pack_into("<Q", ehdr, 0x28, shoff)      # e_shoff
    struct.pack_into("<H", ehdr, 0x34, 64)         # e_ehsize
    struct.pack_into("<H", ehdr, 0x3A, 64)         # e_shentsize
    struct.pack_into("<H", ehdr, 0x3C, 3)          # e_shnum: null, strtab, dynamic
    struct.pack_into("<H", ehdr, 0x3E, 0)          # e_shstrndx (unused by the parser)

    sections = (
        _shdr(_SHT_NULL, 0, 0)
        + _shdr(_SHT_STRTAB, strtab_off, len(strtab))
        + _shdr(_SHT_DYNAMIC, dynamic_off, len(dynamic), sh_link=1)
    )
    return bytes(ehdr) + strtab + dynamic + sections


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


# --------------------------------------------------------------------------- #
#  The real, wired-together property                                          #
# --------------------------------------------------------------------------- #

def test_reads_needed_libraries_in_file_order(tmp_path):
    p = _write(tmp_path, "libfake.so", _build_elf64(
        ("libgomp.so.1", "libc.so.6", "libm.so.6")))
    assert elf_deps.needed_libraries(p) == ["libgomp.so.1", "libc.so.6", "libm.so.6"]


def test_no_dt_needed_entries_returns_empty_list(tmp_path):
    p = _write(tmp_path, "libnodeps.so", _build_elf64(()))
    assert elf_deps.needed_libraries(p) == []


def test_a_decoy_strtab_ahead_of_dynstr_is_not_mistaken_for_it(tmp_path):
    """The parser's whole reason to use .dynamic's own sh_link rather than
    'the first STRTAB section found': a binary with an EARLIER, unrelated
    STRTAB section (e.g. an unstripped .strtab) must not be mistaken for
    .dynstr. Built from scratch (not via _build_elf64) so the layout is
    explicit rather than spliced."""
    decoy_strtab = b"\x00wrong-name.so.9\x00"      # offset 1 decodes to garbage
    real_strtab = b"\x00libreal.so.1\x00"          # offset 1 is the real name
    dynamic = struct.pack("<qQ", _DT_NEEDED, 1) + struct.pack("<qQ", _DT_NULL, 0)

    header_size = 64
    decoy_off = header_size
    real_off = decoy_off + len(decoy_strtab)
    dynamic_off = real_off + len(real_strtab)
    shoff = dynamic_off + len(dynamic)

    ehdr = bytearray(64)
    ehdr[0:4] = b"\x7fELF"
    ehdr[4], ehdr[5], ehdr[6] = 2, 1, 1
    struct.pack_into("<Q", ehdr, 0x28, shoff)
    struct.pack_into("<H", ehdr, 0x34, 64)
    struct.pack_into("<H", ehdr, 0x3A, 64)
    struct.pack_into("<H", ehdr, 0x3C, 4)   # null, decoy strtab, real strtab, dynamic
    struct.pack_into("<H", ehdr, 0x3E, 0)

    sections = (
        _shdr(_SHT_NULL, 0, 0)
        + _shdr(_SHT_STRTAB, decoy_off, len(decoy_strtab))   # index 1: decoy
        + _shdr(_SHT_STRTAB, real_off, len(real_strtab))     # index 2: real .dynstr
        + _shdr(_SHT_DYNAMIC, dynamic_off, len(dynamic), sh_link=2)
    )
    data = bytes(ehdr) + decoy_strtab + real_strtab + dynamic + sections
    p = _write(tmp_path, "libdecoy.so", data)
    assert elf_deps.needed_libraries(p) == ["libreal.so.1"]


# --------------------------------------------------------------------------- #
#  Shapes the function must NOT choke on - every one returns [], never raises #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("data", [
    b"",
    b"not an elf file at all, just some padding text 0123456789",
    b"\x7fELF" + b"\x00" * 10,           # truncated right after the magic
])
def test_non_elf_or_truncated_input_returns_empty_list(tmp_path, data):
    p = _write(tmp_path, "x", data)
    assert elf_deps.needed_libraries(p) == []


def test_32_bit_elf_returns_empty_list(tmp_path):
    p = _write(tmp_path, "lib32.so", _build_elf64(("libx.so",), ei_class=1))
    assert elf_deps.needed_libraries(p) == []


def test_big_endian_elf_returns_empty_list(tmp_path):
    p = _write(tmp_path, "libbe.so", _build_elf64(("libx.so",), ei_data=2))
    assert elf_deps.needed_libraries(p) == []


def test_missing_file_returns_empty_list(tmp_path):
    assert elf_deps.needed_libraries(tmp_path / "does-not-exist.so") == []


def test_elf_with_no_dynamic_section_returns_empty_list(tmp_path):
    ehdr = bytearray(64)
    ehdr[0:4] = b"\x7fELF"
    ehdr[4] = 2
    ehdr[5] = 1
    struct.pack_into("<Q", ehdr, 0x28, 64)   # e_shoff points right after header
    struct.pack_into("<H", ehdr, 0x3A, 64)   # e_shentsize
    struct.pack_into("<H", ehdr, 0x3C, 1)    # e_shnum: just the null section
    sections = _shdr(_SHT_NULL, 0, 0)
    p = _write(tmp_path, "libnodyn.so", bytes(ehdr) + sections)
    assert elf_deps.needed_libraries(p) == []


# --------------------------------------------------------------------------- #
#  Cross-check against the real, upstream-built runtime, when one is present  #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(sys.platform != "linux", reason="the provisioned runtime is ELF only on Linux")
def test_matches_a_real_provisioned_runtime_lib_when_present():
    """Not a fixture: if this box has a real Linux llama.cpp runtime
    provisioned (from a prior `localm setup-llama`), parse an actual shipped
    .so and sanity-check the result rather than only ever trusting the
    hand-built fixture above."""
    from localm.inference.backends.llamacpp import _loader
    binary_dir = _loader.runtime_binary_dir()
    if binary_dir is None:
        pytest.skip("no native runtime provisioned on this box")
    candidates = sorted(binary_dir.glob("libggml-base.so*"))
    if not candidates:
        pytest.skip("no libggml-base.so in the provisioned runtime")
    needed = elf_deps.needed_libraries(candidates[0])
    assert needed, "a real shared object must declare at least libc"
    assert any(n.startswith("libc.so") for n in needed)
