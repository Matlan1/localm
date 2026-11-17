# SPDX-License-Identifier: AGPL-3.0-or-later
"""Minimal Windows PE reader, stdlib only: the DLL names an image imports, its
file version, and which of its imports cannot be found.

The Windows counterpart of :mod:`localm.elf_deps`. Pure Python, so it reads a
Windows binary on any platform and needs no ``dumpbin`` or ``pefile``."""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

_VS_FIXEDFILEINFO_SIGNATURE = b"\xbd\x04\xef\xfe"

# Import names the OS resolves without a file of that name: API sets map to
# real system DLLs through the loader's schema.
_API_SET_PREFIXES = ("api-ms-", "ext-ms-")


def imported_dlls(source: Union[Path, bytes]) -> list[str]:
    """The DLL names a PE32/PE32+ image imports, normal and delay-load, in table
    order without duplicates. *source* is a path or the image's bytes. ``[]``
    for anything that is not a readable PE image."""
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        try:
            data = Path(source).read_bytes()
        except OSError:
            return []
    try:
        if data[:2] != b"MZ":
            return []
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe:pe + 4] != b"PE\0\0":
            return []
        n_sections = struct.unpack_from("<H", data, pe + 6)[0]
        opt_size = struct.unpack_from("<H", data, pe + 20)[0]
        opt = pe + 24
        magic = struct.unpack_from("<H", data, opt)[0]
        dirs = opt + (112 if magic == 0x20B else 96 if magic == 0x10B else -1)
        if dirs < opt:
            return []
        n_dirs = struct.unpack_from("<I", data, dirs - 4)[0]
        sections = []
        sec = opt + opt_size
        for i in range(n_sections):
            vsize, va, rsize, raw = struct.unpack_from("<IIII", data, sec + 40 * i + 8)
            sections.append((va, max(vsize, rsize), raw))

        def offset(rva: int) -> int:
            for va, size, raw in sections:
                if va <= rva < va + size:
                    return raw + rva - va
            raise ValueError(f"rva {rva:#x} outside every section")

        def cstr(rva: int) -> str:
            start = offset(rva)
            return data[start:data.index(b"\0", start)].decode("ascii", "replace")

        names: list[str] = []
        for index, size, name_at in ((1, 20, 12), (13, 32, 4)):
            if index >= n_dirs:
                continue
            rva = struct.unpack_from("<I", data, dirs + 8 * index)[0]
            if not rva:
                continue
            try:
                at = offset(rva)
            except ValueError:
                continue
            while True:
                desc = data[at:at + size]
                if len(desc) < size or not any(desc):
                    break
                name_rva = struct.unpack_from("<I", desc, name_at)[0]
                try:
                    name = cstr(name_rva) if name_rva else ""
                except ValueError:
                    name = ""
                if name and name not in names:
                    names.append(name)
                at += size
        return names
    except struct.error:
        return []


def file_version(source: Union[Path, bytes]) -> Optional[tuple[int, int, int, int]]:
    """The file version from an image's ``VS_FIXEDFILEINFO`` resource, as four
    integers, or None when it has none or cannot be read."""
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        try:
            data = Path(source).read_bytes()
        except OSError:
            return None
    at = data.find(_VS_FIXEDFILEINFO_SIGNATURE)
    if at < 0 or len(data) < at + 16:
        return None
    ms, ls = struct.unpack_from("<II", data, at + 8)
    return (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)


def _is_loaded(name: str) -> bool:
    """Whether a module called *name* is already loaded in this process, which
    the loader reuses for an import of that name."""
    if sys.platform != "win32":
        return False
    import ctypes
    try:
        get = ctypes.windll.kernel32.GetModuleHandleW
        get.restype = ctypes.c_void_p
        return bool(get(name))
    except (AttributeError, OSError):
        return False


def unresolved_imports(roots: Iterable[Path], search_dirs: Iterable[Path],
                       loaded: Callable[[str], bool] = _is_loaded) -> dict[str, list[str]]:
    """``{missing dll name (lower case): [names of the images that import it]}``
    for the images in *roots* and every import of theirs found in *search_dirs*.

    An import counts as found when a file of that name is in one of
    *search_dirs*, when *loaded* says the process already has it, or when it is
    an API set. Found imports in *search_dirs* are followed, so a dependency of
    a dependency is checked too."""
    dirs = [Path(d) for d in search_dirs]
    present: dict[str, Path] = {}
    for d in dirs:
        try:
            for f in d.iterdir():
                present.setdefault(f.name.lower(), f)
        except OSError:
            continue
    missing: dict[str, list[str]] = {}
    queue = [Path(r) for r in roots]
    seen: set[str] = set()
    while queue:
        image = queue.pop(0)
        key = image.name.lower()
        if key in seen:
            continue
        seen.add(key)
        for name in imported_dlls(image):
            low = name.lower()
            if low.startswith(_API_SET_PREFIXES):
                continue
            found = present.get(low)
            if found is not None:
                queue.append(found)
            elif not loaded(name):
                missing.setdefault(low, [])
                if image.name not in missing[low]:
                    missing[low].append(image.name)
    return missing
