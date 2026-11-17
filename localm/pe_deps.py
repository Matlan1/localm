# SPDX-License-Identifier: AGPL-3.0-or-later
"""Minimal Windows PE reader, stdlib only: the DLL names an image imports, its
machine type and file version, and which of its imports cannot be found.

The Windows counterpart of :mod:`localm.elf_deps`. Pure Python, so it reads a
Windows binary on any platform and needs no ``dumpbin`` or ``pefile``. A file
is memory-mapped rather than read whole, since a runtime folder holds DLLs of
several hundred MB."""

from __future__ import annotations

import contextlib
import mmap
import struct
import sys
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Union

_VS_FIXEDFILEINFO_SIGNATURE = b"\xbd\x04\xef\xfe"

# Import names the OS resolves without a file of that name: API sets map to
# real system DLLs through the loader's schema.
_API_SET_PREFIXES = ("api-ms-", "ext-ms-")

Source = Union[Path, str, bytes, bytearray]


@contextlib.contextmanager
def _opened(source: Source) -> Iterator[Optional[Union[bytes, mmap.mmap]]]:
    """*source*'s bytes: the bytes themselves, a read-only map of the file, or
    None when the file cannot be read or is empty."""
    if isinstance(source, (bytes, bytearray)):
        yield bytes(source)
        return
    try:
        f = open(source, "rb")
    except OSError:
        yield None
        return
    with f:
        try:
            m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except (OSError, ValueError):
            yield None
            return
        with m:
            yield m


def _header(data) -> Optional[tuple[int, int]]:
    """(offset of the PE signature, machine) of a PE image, else None."""
    if len(data) < 0x40 or data[:2] != b"MZ":
        return None
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        return None
    return pe, struct.unpack_from("<H", data, pe + 4)[0]


def machine(source: Source) -> Optional[int]:
    """The IMAGE_FILE_HEADER machine of a PE image (0x8664 x64, 0xAA64 ARM64),
    or None when *source* is not one."""
    with _opened(source) as data:
        if data is None:
            return None
        try:
            head = _header(data)
        except struct.error:
            return None
        return head[1] if head else None


def imported_dlls(source: Source) -> list[str]:
    """The DLL names a PE32/PE32+ image imports, normal and delay-load, in table
    order without duplicates. *source* is a path or the image's bytes. ``[]``
    for anything that is not a readable PE image."""
    with _opened(source) as data:
        if data is None:
            return []
        try:
            return _imports(data)
        except struct.error:
            return []


def _imports(data) -> list[str]:
    head = _header(data)
    if head is None:
        return []
    pe = head[0]
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
        end = data.find(b"\0", start)
        if end < 0:
            raise ValueError("unterminated name")
        return bytes(data[start:end]).decode("ascii", "replace")

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
            desc = bytes(data[at:at + size])
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


def file_version(source: Source) -> Optional[tuple[int, int, int, int]]:
    """The file version from an image's ``VS_FIXEDFILEINFO`` resource, as four
    integers, or None when it has none or cannot be read."""
    with _opened(source) as data:
        if data is None:
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
                       loaded: Optional[Callable[[str], bool]] = None,
                       follow: Optional[Iterable[Path]] = None) -> dict[str, list[str]]:
    """``{missing dll name (lower case): [names of the images that import it]}``
    for the images in *roots* and the imports of theirs this follows.

    An import counts as found when a file of that name is in one of
    *search_dirs*, when *loaded* (default: whether this process has a module of
    that name) says so, or when it is an API set. A found import in one of
    *follow* (default: every search dir) is checked in turn, so a dependency of
    a dependency is covered."""
    is_loaded = loaded if loaded is not None else _is_loaded
    dirs = [Path(d) for d in search_dirs]
    followed = {str(Path(d)).lower() for d in (follow if follow is not None else dirs)}
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
                if str(found.parent).lower() in followed:
                    queue.append(found)
            elif not is_loaded(name):
                missing.setdefault(low, [])
                if image.name not in missing[low]:
                    missing[low].append(image.name)
    return missing
