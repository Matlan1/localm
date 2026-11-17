# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Microsoft Visual C++ runtime next to a native runtime on Windows.

The Windows builds of llama.cpp and stable-diffusion.cpp are built with MSVC
and import its runtime (msvcp140.dll, vcruntime140.dll, vcomp140.dll, ...),
which their release archives do not include and a clean Windows does not have.
:func:`place_app_local` puts the DLLs a runtime folder needs into that folder,
taken from Microsoft's own Visual C++ Redistributable package, downloaded from
Microsoft at a pinned version and checked against a pinned sha256. The package
is unpacked here (its WiX bundle and the cabinets inside it), never run, so no
administrator prompt appears and nothing is registered with the system.

Windows loads a DLL's imports from the DLL's own folder first, so the copies
serve exactly the runtime they sit beside and nothing else on the machine."""

from __future__ import annotations

import hashlib
import os
import platform
import struct
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Callable, Iterable, Optional

from localm import pe_deps

# Every DLL of the Visual C++ v14 runtime package (the files of its runtime
# cabinet), lower case.
RUNTIME_DLLS = frozenset({
    "concrt140.dll", "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll", "vcamp140.dll",
    "vccorlib140.dll", "vcomp140.dll", "vcruntime140.dll", "vcruntime140_1.dll",
    "vcruntime140_threads.dll",
})

REDIST_VERSION = (14, 52, 37029, 0)

# (machine as platform.machine() reports it, lower case) -> (url, sha256, size,
# the suffix the runtime cabinet gives its files). The URLs are Microsoft's
# versioned download locations behind https://aka.ms/vs/18/release/.
_REDIST = {
    "amd64": (
        "https://download.visualstudio.microsoft.com/download/pr/"
        "78d061c5-9ef3-42ab-98ed-3309d1095dc4/"
        "9C02C9ED120F671B900E6B9471D890E3B2414117A7E53F4AE2E855466FA2B7C9/"
        "VC_redist.x64.exe",
        "9c02c9ed120f671b900e6b9471d890e3b2414117a7e53f4ae2e855466fa2b7c9",
        18863624, "amd64"),
    "arm64": (
        "https://download.visualstudio.microsoft.com/download/pr/"
        "78d061c5-9ef3-42ab-98ed-3309d1095dc4/"
        "3C81BCF5B06156858362400D7066F65ECE38E8ACCB5E687FC65BEF5735F31FFC/"
        "VC_redist.arm64.exe",
        "3c81bcf5b06156858362400d7066f65ece38e8accb5e687fc65bef5735f31ffc",
        11967280, "arm64"),
}
_MACHINE_ALIASES = {"x86_64": "amd64", "x64": "amd64", "aarch64": "arm64"}

REDIST_PAGE = "https://aka.ms/vc14/vc_redist.x64.exe"

LICENSE_NAME = "LICENSE.msvc-runtime"

_LICENSE_NOTICE = """\
{names}
are files of the Microsoft Visual C++ v14 Redistributable {version}, unpacked
from Microsoft's own package
{url}
(sha256 {sha256}) on this machine and placed beside the runtime that needs
them. They are Microsoft's, under the Microsoft Visual C++ Redistributable and
Runtime license terms: https://aka.ms/VCRedistLicense
"""

# Upper bound on what one cabinet folder may expand to. The runtime cabinet
# holds about 3 MB; the bound stops a malformed archive from filling memory.
_MAX_FOLDER_BYTES = 256 * 1024 * 1024


class RedistError(Exception):
    """The redistributable package could not be fetched or unpacked."""


def redist_machine(machine: Optional[str] = None) -> Optional[str]:
    """The key of :data:`_REDIST` for *machine* (default: this machine), or None
    when no package is pinned for it."""
    m = (machine if machine is not None else platform.machine()).lower()
    m = _MACHINE_ALIASES.get(m, m)
    return m if m in _REDIST else None


def redist_url(machine: Optional[str] = None) -> str:
    """Where a person downloads the Visual C++ Redistributable for *machine*."""
    m = redist_machine(machine)
    return f"https://aka.ms/vc14/vc_redist.{'arm64' if m == 'arm64' else 'x64'}.exe"


def needed(folder: Path) -> set[str]:
    """The runtime DLL names (lower case) that a ``.dll`` or ``.exe`` directly
    in *folder* imports and *folder* does not hold."""
    have: set[str] = set()
    images: list[Path] = []
    try:
        for f in folder.iterdir():
            have.add(f.name.lower())
            if f.suffix.lower() in (".dll", ".exe") and f.is_file():
                images.append(f)
    except OSError:
        return set()
    wanted: set[str] = set()
    for image in images:
        wanted.update(n.lower() for n in pe_deps.imported_dlls(image))
    return (wanted & RUNTIME_DLLS) - have


def _system_dir() -> Optional[Path]:
    if sys.platform != "win32":
        return None
    import ctypes
    buf = ctypes.create_unicode_buffer(512)
    try:
        n = ctypes.windll.kernel32.GetSystemDirectoryW(buf, len(buf))
    except (AttributeError, OSError):
        return None
    return Path(buf.value) if n else None


def provided_by_system(names: Iterable[str], system_dir: Optional[Path]) -> set[str]:
    """The *names* the system folder already has at :data:`REDIST_VERSION` or
    newer. Those are left to the system copy, which Windows Update services."""
    out: set[str] = set()
    if system_dir is None:
        return out
    for name in names:
        version = pe_deps.file_version(system_dir / name)
        if version is not None and version >= REDIST_VERSION:
            out.add(name)
    return out


# --------------------------------------------------------------------------- #
#  Unpacking the redistributable                                               #
# --------------------------------------------------------------------------- #

def _burn_containers(exe: bytes) -> list[bytes]:
    """The containers of a WiX Burn bundle, in order: the UX container first,
    then the attached ones. Read from the bundle's ``.wixburn`` section."""
    try:
        pe = struct.unpack_from("<I", exe, 0x3C)[0]
        if exe[:2] != b"MZ" or exe[pe:pe + 4] != b"PE\0\0":
            raise RedistError("the package is not a Windows executable")
        n_sections = struct.unpack_from("<H", exe, pe + 6)[0]
        opt_size = struct.unpack_from("<H", exe, pe + 20)[0]
        sec = pe + 24 + opt_size
        header = None
        for i in range(n_sections):
            at = sec + 40 * i
            if exe[at:at + 8].rstrip(b"\0") == b".wixburn":
                raw = struct.unpack_from("<I", exe, at + 20)[0]
                header = raw
                break
        if header is None:
            raise RedistError("the package has no WiX bundle section")
        magic, version = struct.unpack_from("<II", exe, header)
        if magic != 0x00F14300 or version != 2:
            raise RedistError(f"unknown WiX bundle header ({magic:#x} v{version})")
        (stub, _checksum, sig_offset, sig_size, fmt,
         count) = struct.unpack_from("<IIIIII", exe, header + 24)
        if fmt != 1 or not 1 <= count <= 16:
            raise RedistError(f"unknown WiX container layout (format {fmt}, {count})")
        sizes = struct.unpack_from("<" + "I" * count, exe, header + 48)
    except struct.error as e:
        raise RedistError(f"the WiX bundle header is truncated ({e})") from e
    out = []
    at = stub
    for i, size in enumerate(sizes):
        if i == 1 and sig_size:
            at = sig_offset + sig_size
        blob = exe[at:at + size]
        if len(blob) != size or blob[:4] != b"MSCF":
            raise RedistError(f"WiX container {i} is not a cabinet")
        out.append(blob)
        at += size
    return out


def cabinet_files(cab: bytes) -> dict[str, bytes]:
    """``{name: content}`` of a single-volume Microsoft cabinet whose folders are
    stored or MSZIP-compressed. Raises RedistError for anything else."""
    try:
        if cab[:4] != b"MSCF":
            raise RedistError("not a cabinet")
        coff_files = struct.unpack_from("<I", cab, 16)[0]
        n_folders, n_files, flags = struct.unpack_from("<HHH", cab, 26)
        if flags & 0x3:
            raise RedistError("multi-volume cabinets are not supported")
        at = 36
        folder_reserve = data_reserve = 0
        if flags & 0x4:
            header_reserve, folder_reserve, data_reserve = struct.unpack_from("<HBB", cab, 36)
            at = 40 + header_reserve
        folders: list[bytes] = []
        for _ in range(n_folders):
            first, n_blocks, kind = struct.unpack_from("<IHH", cab, at)
            at += 8 + folder_reserve
            folders.append(_folder_bytes(cab, first, n_blocks, kind & 0xF, data_reserve))
        files: dict[str, bytes] = {}
        at = coff_files
        for _ in range(n_files):
            size, start, index = struct.unpack_from("<IIH", cab, at)
            attribs = struct.unpack_from("<H", cab, at + 14)[0]
            end = cab.index(b"\0", at + 16)
            raw_name = cab[at + 16:end]
            at = end + 1
            name = raw_name.decode("utf-8" if attribs & 0x80 else "latin-1")
            if index >= len(folders):
                raise RedistError(f"{name}: spans cabinets, which is not supported")
            data = folders[index][start:start + size]
            if len(data) != size:
                raise RedistError(f"{name}: truncated in the cabinet")
            files[name] = data
        return files
    except (struct.error, ValueError) as e:
        raise RedistError(f"the cabinet is malformed ({e})") from e


def _folder_bytes(cab: bytes, at: int, n_blocks: int, kind: int, reserve: int) -> bytes:
    if kind not in (0, 1):
        raise RedistError(f"cabinet compression type {kind} is not supported")
    out = bytearray()
    for _ in range(n_blocks):
        n_in, n_out = struct.unpack_from("<HH", cab, at + 4)
        block = cab[at + 8 + reserve:at + 8 + reserve + n_in]
        at += 8 + reserve + n_in
        if kind == 0:
            out += block
        else:
            if block[:2] != b"CK":
                raise RedistError("MSZIP block without its signature")
            inflater = zlib.decompressobj(-15, zdict=bytes(out[-32768:]))
            piece = inflater.decompress(block[2:], n_out) + inflater.flush()
            if len(piece) != n_out:
                raise RedistError("MSZIP block expanded to the wrong size")
            out += piece
        if len(out) > _MAX_FOLDER_BYTES:
            raise RedistError("cabinet folder is larger than expected")
    return bytes(out)


def runtime_files(exe: bytes, suffix: str) -> dict[str, bytes]:
    """``{dll name (lower case): content}`` of the runtime cabinet in the
    redistributable package *exe*: the cabinet whose files carry *suffix*
    (``msvcp140.dll_amd64``)."""
    containers = _burn_containers(exe)
    if len(containers) < 2:
        raise RedistError("the package has no attached container")
    for payload in cabinet_files(containers[1]).values():
        if payload[:4] != b"MSCF":
            continue
        found: dict[str, bytes] = {}
        for name, data in cabinet_files(payload).items():
            low = name.lower()
            if low.endswith("_" + suffix) and low[:-len(suffix) - 1] in RUNTIME_DLLS:
                found[low[:-len(suffix) - 1]] = data
        if "msvcp140.dll" in found:
            return found
    raise RedistError(f"the package holds no runtime cabinet for {suffix}")


def _closure(names: set[str], files: dict[str, bytes]) -> set[str]:
    """*names* plus every runtime DLL they import, transitively."""
    out = set(names)
    todo = list(names)
    while todo:
        data = files.get(todo.pop())
        if data is None:
            continue
        for dep in pe_deps.imported_dlls(data):
            low = dep.lower()
            if low in RUNTIME_DLLS and low not in out:
                out.add(low)
                todo.append(low)
    return out


# --------------------------------------------------------------------------- #
#  Placing the DLLs                                                            #
# --------------------------------------------------------------------------- #

_SYSTEM = object()


def place_app_local(folder: Path, download: Callable[[str, Path], object],
                    say: Callable[[str], None] = lambda _m: None, *,
                    machine: Optional[str] = None,
                    system_dir: object = _SYSTEM,
                    on_windows: Optional[bool] = None) -> list[str]:
    """Put the Visual C++ runtime DLLs the binaries in *folder* need, and the
    system folder does not already provide at the pinned version or newer, into
    *folder*. Returns the names placed (empty when nothing was needed).
    *system_dir* defaults to the Windows system folder; None means there is none.

    *download(url, dest)* fetches a URL to a file; the caller passes its own,
    so its network policy applies. Raises RedistError when the package cannot
    be fetched, does not match its pin, or cannot be unpacked; *folder* is then
    unchanged."""
    if not (sys.platform == "win32" if on_windows is None else on_windows):
        return []
    missing = needed(folder)
    sysdir = (_system_dir() if system_dir is _SYSTEM
              else system_dir if isinstance(system_dir, Path) else None)
    missing -= provided_by_system(missing, sysdir)
    if not missing:
        return []
    key = redist_machine(machine)
    if key is None:
        raise RedistError(f"no Visual C++ runtime package is pinned for "
                          f"{machine or platform.machine()}")
    url, sha256, size, suffix = _REDIST[key]
    say(f"Placing the Microsoft Visual C++ runtime ({', '.join(sorted(missing))}) next to "
        "the runtime: this Windows does not have it.")
    with tempfile.TemporaryDirectory(prefix=".vcredist-", dir=folder) as tmp:
        exe_path = Path(tmp) / url.rsplit("/", 1)[-1]
        download(url, exe_path)
        exe = exe_path.read_bytes()
    if len(exe) != size:
        raise RedistError(f"the Visual C++ runtime package is {len(exe)} bytes, "
                          f"expected {size}")
    got = hashlib.sha256(exe).hexdigest()
    if got != sha256:
        raise RedistError(f"the Visual C++ runtime package's sha256 is {got}, expected "
                          f"{sha256}: refusing a possibly tampered or wrong file")
    files = runtime_files(exe, suffix)
    wanted = _closure(missing, files) - {f.name.lower() for f in folder.iterdir()}
    wanted -= provided_by_system(wanted - missing, sysdir)
    absent = sorted(n for n in wanted if n not in files)
    if absent:
        raise RedistError(f"the Visual C++ runtime package has no {', '.join(absent)}")
    for name in sorted(wanted):
        version = pe_deps.file_version(files[name])
        if version != REDIST_VERSION:
            raise RedistError(f"{name} in the package is version {version}, expected "
                              f"{REDIST_VERSION}")
    placed: list[str] = []
    try:
        for name in sorted(wanted):
            part = folder / (name + ".part")
            part.write_bytes(files[name])
            os.replace(part, folder / name)
            placed.append(name)
        (folder / LICENSE_NAME).write_text(_LICENSE_NOTICE.format(
            names=", ".join(placed), version=".".join(map(str, REDIST_VERSION[:3])),
            url=url, sha256=sha256), encoding="utf-8")
    except OSError:
        for name in placed:
            try:
                (folder / name).unlink()
            except OSError:
                pass
        raise
    return placed


# --------------------------------------------------------------------------- #
#  Naming what is missing after a failed load                                  #
# --------------------------------------------------------------------------- #

def describe_missing(missing: dict[str, list[str]]) -> Optional[str]:
    """One clause naming the DLLs in *missing* (as
    :func:`pe_deps.unresolved_imports` returns it), saying which belong to the
    Visual C++ runtime. None when *missing* is empty. The text starts with the
    DLL names and "is missing" or "are missing", which
    :func:`localm.setup_llama.load_probe._name_missing_shared_lib` reads back."""
    if not missing:
        return None
    vc = sorted(n for n in missing if n in RUNTIME_DLLS)
    other = sorted(n for n in missing if n not in RUNTIME_DLLS)
    parts = []
    for names, what in ((vc, "the Microsoft Visual C++ runtime"), (other, None)):
        if not names:
            continue
        users = sorted({u for n in names for u in missing[n]})
        verb = "is" if len(names) == 1 else "are"
        needed_by = ", ".join(users[:3]) + (", ..." if len(users) > 3 else "")
        label = f"{what}; " if what else ""
        parts.append(f"{', '.join(names)} {verb} missing ({label}needed by {needed_by})")
    return "; ".join(parts)
