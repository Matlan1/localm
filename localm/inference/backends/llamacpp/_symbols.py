# SPDX-License-Identifier: AGPL-3.0-or-later
"""Find a llama.cpp function that is exported with C++ linkage.

Most of llama.cpp's public surface is `extern "C"`, so `getattr(lib, name)`
finds it. A handful of functions are declared in the internal `src/llama-ext.h`
without `extern "C"` and are therefore exported under a mangled name. The MTP
draft-head API is the whole of that group localm cares about:

    llama_set_embeddings_nextn      llama_get_embeddings_nextn
    llama_get_embeddings_nextn_ith  llama_set_nextn_layer_offset
    llama_get_ctx_other

`getattr(lib, "llama_set_embeddings_nextn")` raises AttributeError for these,
which is indistinguishable from the runtime not having them at all. Reading that
as absence is a real error this repo has made: it produced a written conclusion
that driving an MTP head was impossible against the shipped runtime, when in
fact `llama.dll` exports all five and every one of them can be called.

So resolution goes: the plain name first, in case upstream promotes the API to
`extern "C"`, then a scan of the binary's own export/symbol table for the name as
each compiler spells it - MSVC `?name@@<sig>`, Itanium `_Z<len(name)><name><sig>`.
Matching per scheme rather than by containment is load-bearing in both
directions: containment cannot separate `llama_get_embeddings_nextn` from
`llama_get_embeddings_nextn_ith`, and an Itanium symbol brackets the identifier
with a digit and a letter, so any "non-identifier character on each side" rule
rejects every GCC and Clang build.

Returning None is a legitimate answer and callers must treat it as "this runtime
cannot do that", never as an error. See tests/test_llamacpp_symbols.py.
"""

from __future__ import annotations

import ctypes
import struct
from pathlib import Path
from typing import List, Optional

_cache: dict = {}


def _pe_exports(data: bytes) -> List[str]:
    """Exported names from a PE image, or [] when there is no export table."""
    if data[:2] != b"MZ":
        return []
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        return []
    nsec, = struct.unpack_from("<H", data, pe + 6)
    optsz, = struct.unpack_from("<H", data, pe + 20)
    magic, = struct.unpack_from("<H", data, pe + 24)
    dirs = pe + 24 + (112 if magic == 0x20B else 96)
    edir_rva, _edir_sz = struct.unpack_from("<II", data, dirs)
    if not edir_rva:
        return []
    sections = []
    sec_off = pe + 24 + optsz
    for i in range(nsec):
        base = sec_off + i * 40
        vsize, vaddr, _rsize, raddr = struct.unpack_from("<IIII", data, base + 8)
        sections.append((vaddr, max(vsize, 1), raddr))

    def to_offset(rva: int) -> Optional[int]:
        for vaddr, vsize, raddr in sections:
            if vaddr <= rva < vaddr + vsize:
                return raddr + (rva - vaddr)
        return None

    edir = to_offset(edir_rva)
    if edir is None:
        return []
    _nfuncs, nnames = struct.unpack_from("<II", data, edir + 20)
    names_rva, = struct.unpack_from("<I", data, edir + 32)
    names_off = to_offset(names_rva)
    if names_off is None:
        return []
    out = []
    for i in range(nnames):
        rva, = struct.unpack_from("<I", data, names_off + 4 * i)
        off = to_offset(rva)
        if off is None:
            continue
        end = data.find(b"\0", off)
        if end < 0:
            continue
        out.append(data[off:end].decode("ascii", "replace"))
    return out


def _elf_dynsym(data: bytes) -> List[str]:
    """Names from an ELF `.dynstr`, which is where a shared object's exports live."""
    if data[:4] != b"\x7fELF":
        return []
    is64 = data[4] == 2
    if not is64:
        return []
    endian = "<" if data[5] == 1 else ">"
    e_shoff, = struct.unpack_from(endian + "Q", data, 0x28)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(endian + "HHH", data, 0x3A)
    if not e_shoff or not e_shnum:
        return []
    def section(i):
        base = e_shoff + i * e_shentsize
        sh_name, sh_type = struct.unpack_from(endian + "II", data, base)
        sh_offset, sh_size = struct.unpack_from(endian + "QQ", data, base + 0x18)
        return sh_name, sh_type, sh_offset, sh_size
    shstr_off = section(e_shstrndx)[2]
    out = []
    for i in range(e_shnum):
        sh_name, sh_type, sh_offset, sh_size = section(i)
        end = data.find(b"\0", shstr_off + sh_name)
        name = data[shstr_off + sh_name:end].decode("ascii", "replace")
        if name == ".dynstr" and sh_size:
            blob = data[sh_offset:sh_offset + sh_size]
            out = [s.decode("ascii", "replace") for s in blob.split(b"\0") if s]
            break
    return out


def _macho_symbols(data: bytes) -> List[str]:
    """Names from a Mach-O string table (LC_SYMTAB)."""
    magic, = struct.unpack_from("<I", data, 0)
    if magic not in (0xFEEDFACF, 0xCFFAEDFE):
        return []
    ncmds, = struct.unpack_from("<I", data, 16)
    off = 32
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", data, off)
        if cmd == 0x2:  # LC_SYMTAB
            _symoff, _nsyms, stroff, strsize = struct.unpack_from("<IIII", data, off + 8)
            blob = data[stroff:stroff + strsize]
            return [s.decode("ascii", "replace") for s in blob.split(b"\0") if s]
        off += cmdsize
    return []


def exported_names(image: Path) -> List[str]:
    """Every exported symbol name in *image*, for whichever format it is."""
    key = ("names", str(image))
    if key in _cache:
        return _cache[key]
    try:
        data = image.read_bytes()
    except OSError:
        names: List[str] = []
    else:
        names = _pe_exports(data) or _elf_dynsym(data) or _macho_symbols(data)
    _cache[key] = names
    return names


def mangled_name(image: Path, plain: str) -> Optional[str]:
    """The exported symbol for *plain*, or None when the image has no such function.

    Prefers an exact `extern "C"` match, then the shortest containing mangled
    name. Shortest is the right tie-break: a longer candidate that also contains
    the identifier is a different, more decorated symbol.
    """
    names = exported_names(image)
    if plain in names:
        return plain
    # Match each scheme on its own terms rather than by containment. Containment
    # cannot separate `..._nextn` from `..._nextn_ith`, and an Itanium symbol
    # brackets the identifier with a digit and a letter, so a "non-identifier
    # character on each side" rule rejects every GCC and Clang build.
    #   MSVC     ?name@@<signature>
    #   Itanium  _Z<len(name)><name><signature>
    msvc = "?" + plain + "@@"
    itanium = "_Z%d%s" % (len(plain), plain)
    hits = [n for n in names
            if n.startswith(msvc) or n.startswith(itanium)]
    return min(hits, key=len) if hits else None


def resolve(lib: ctypes.CDLL, plain: str) -> Optional[object]:
    """The callable for *plain*, or None when this runtime does not export it.

    None is an answer, not a failure: MTP is simply unavailable on a runtime
    without these, and the caller reports that rather than raising.
    """
    key = ("fn", id(lib), plain)
    if key in _cache:
        return _cache[key]
    fn = getattr(lib, plain, None)
    if fn is None:
        image = getattr(lib, "_name", None)
        symbol = mangled_name(Path(image), plain) if image else None
        if symbol:
            try:
                fn = getattr(lib, symbol)
            except AttributeError:
                fn = None
    _cache[key] = fn
    return fn
