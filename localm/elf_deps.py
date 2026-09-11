# SPDX-License-Identifier: AGPL-3.0-or-later
"""Minimal ELF64 little-endian ``DT_NEEDED`` reader, stdlib only.

Lets a caller ask "what shared libraries does this .so declare it needs"
without shelling out to ``readelf``/``objdump``/``ldd`` - none of which is
guaranteed present on every Linux distribution localm runs on."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import List

_ELF_MAGIC = b"\x7fELF"
_ELFCLASS64 = 2
_ELFDATA2LSB = 1
_SHT_DYNAMIC = 6
_DT_NULL = 0
_DT_NEEDED = 1


def needed_libraries(path: Path) -> List[str]:
    """The ``DT_NEEDED`` sonames a 64-bit little-endian ELF shared object at
    *path* declares, in file order.

    Returns ``[]`` for anything that is not shaped like one - a truncated
    file, a 32-bit or big-endian ELF (this project ships neither), or a
    non-ELF file entirely - rather than raising, so a caller scanning a
    directory of mixed files needs no filter of its own. The one thing this
    does NOT do is resolve whether a needed name is actually satisfied
    anywhere; that is a separate question for the caller.

    Uses each ``SHT_DYNAMIC`` section's own ``sh_link`` to find its string
    table, per the ELF spec, rather than guessing which ``SHT_STRTAB``
    section is ``.dynstr`` - a binary that also carries an unstripped
    ``.strtab`` symbol-name table would make that guess wrong."""
    try:
        data = path.read_bytes()
    except OSError:
        return []
    if len(data) < 64 or data[:4] != _ELF_MAGIC:
        return []
    if data[4] != _ELFCLASS64 or data[5] != _ELFDATA2LSB:
        return []

    try:
        e_shoff, = struct.unpack_from("<Q", data, 0x28)
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    except struct.error:
        return []
    if not e_shoff or not e_shnum or not e_shentsize:
        return []

    def section(i: int):
        off = e_shoff + i * e_shentsize
        if off + 0x40 > len(data):
            return None
        sh_type, = struct.unpack_from("<I", data, off + 0x04)
        sh_offset, sh_size = struct.unpack_from("<QQ", data, off + 0x18)
        sh_link, = struct.unpack_from("<I", data, off + 0x28)
        return sh_type, sh_offset, sh_size, sh_link

    dynamic = None
    dyn_link = None
    for i in range(e_shnum):
        s = section(i)
        if s is None:
            return []
        sh_type, sh_offset, sh_size, sh_link = s
        if sh_type == _SHT_DYNAMIC:
            dynamic = (sh_offset, sh_size)
            dyn_link = sh_link
            break
    if dynamic is None or dyn_link is None:
        return []
    strtab_section = section(dyn_link)
    if strtab_section is None:
        return []
    _, strtab_off, strtab_size, _ = strtab_section

    d_off, d_size = dynamic
    entry_size = 16  # Elf64_Dyn: Sxword d_tag; union { Xword d_val; Addr d_ptr; }
    needed_offsets = []
    pos = d_off
    end = d_off + d_size
    while pos + entry_size <= end and pos + entry_size <= len(data):
        d_tag, d_val = struct.unpack_from("<qQ", data, pos)
        if d_tag == _DT_NULL:
            break
        if d_tag == _DT_NEEDED:
            needed_offsets.append(d_val)
        pos += entry_size

    names = []
    for rel in needed_offsets:
        start = strtab_off + rel
        if start >= strtab_off + strtab_size or start >= len(data):
            continue
        term = data.find(b"\x00", start, min(strtab_off + strtab_size, len(data)))
        if term == -1:
            continue
        names.append(data[start:term].decode("utf-8", "replace"))
    return names
