# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bundling a native dependency that upstream's Linux builds link but do not
ship (libgomp.so.1).
"""

from __future__ import annotations

import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

from localm import elf_deps
from localm.debuglog import logger
from localm.setup_llama._common import console
from localm.setup_llama.download import _sha256_file, ArtifactError
import localm.setup_llama as _sl

# Bundles libgomp.so.1: upstream's Linux release tarballs link OpenMP
# dynamically and ship no copy of their own. Pinned by sha256 to an
# immutable snapshot.debian.org URL - Debian, not Ubuntu, whose pool
# compresses a .deb's data member with zstd (unreadable by this venv's
# Python without a third-party module; Debian uses xz). See
# _extract_libgomp_from_deb.
_LIBGOMP_SONAME = "libgomp.so.1"


_LIBGOMP_DEB_URL = "https://snapshot.debian.org/file/855f73e203af87b85693b43b807f0ba1d6bb410e"


_LIBGOMP_DEB_SHA256 = "4530c95aefa48e33fd8cf4acbe5c4b559dbe7bdf4c56469986c83a203982cef1"


_LIBGOMP_DEB_MIN_BYTES = 20_000  # catches an HTML/error substitute for the real file


_LIBGOMP_LICENSE_NOTICE = """\
libgomp.so.1 (GCC's OpenMP runtime) is bundled here from Debian's libgomp1
package, licensed GPL-3.0-or-later WITH the GCC Runtime Library Exception
3.1 <https://www.gnu.org/licenses/gcc-exception-3.1.html>. That exception's
Grant of Additional Permission (section 1) permits combining the Runtime
Library with Independent Modules such as the llama.cpp/ggml binaries it
ships alongside here, under terms of your choice.

Source package: https://snapshot.debian.org/package/gcc-10/10.2.1-6/
Full GPLv3 text: https://www.gnu.org/licenses/gpl-3.0.txt
"""


def _read_ar_archive(path: Path) -> dict:
    """``{member_name: bytes}`` for every member of a plain ``ar`` archive -
    the container format a Debian ``.deb`` uses (``!<arch>\\n`` then one
    60-byte header per member, each content block padded to an even byte
    count). No decompression at this layer; a ``.deb``'s members
    (``debian-binary``, ``control.tar.*``, ``data.tar.*``) are themselves
    separately-compressed tarballs. Raises ArtifactError if *path* is not an
    ``ar`` archive at all."""
    data = path.read_bytes()
    if data[:8] != b"!<arch>\n":
        raise ArtifactError(f"{path.name} is not an ar archive")
    members: dict = {}
    pos = 8
    while pos + 60 <= len(data):
        header = data[pos:pos + 60]
        name = header[0:16].decode("ascii", "replace").strip().rstrip("/")
        size = int(header[48:58].decode("ascii", "replace").strip())
        pos += 60
        members[name] = data[pos:pos + size]
        pos += size + (size & 1)   # members are 2-byte aligned
    return members


def _extract_libgomp_from_deb(deb_path: Path, workdir: Path) -> Path:
    """Pull the real ``libgomp.so.1*`` file out of a Debian ``.deb`` at
    *deb_path* using only the stdlib (``ar`` reader + :mod:`tarfile`'s xz
    support), writing it into *workdir* and returning its path. Picks the
    REGULAR FILE member (Debian ships ``libgomp.so.1 -> libgomp.so.1.0.0`` as
    a symlink to it), so the destination filename this returns is never the
    ``libgomp.so.1`` the caller will rename it to - callers must not assume
    otherwise."""
    members = _read_ar_archive(deb_path)
    data_name = next((n for n in members if n.startswith("data.tar")), None)
    if data_name is None:
        raise ArtifactError(f"{deb_path.name} has no data.tar member")
    data_tar_path = workdir / data_name
    data_tar_path.write_bytes(members[data_name])
    with tarfile.open(data_tar_path) as tf:
        member = next((m for m in tf.getmembers()
                       if m.isfile() and Path(m.name).name.startswith("libgomp.so.1")),
                      None)
        if member is None:
            raise ArtifactError(f"{deb_path.name} contains no libgomp.so.1 file")
        extracted = workdir / "extracted-libgomp"
        src = tf.extractfile(member)
        if src is None:
            raise ArtifactError(f"{deb_path.name}: could not read {member.name}")
        with src, open(extracted, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return extracted


def _bundle_missing_native_deps(target: Path) -> None:
    """After a Linux backend is extracted into *target*, provide any native
    dependency the extracted ``.so`` files need but neither the archive nor
    this runtime dir already supplies. Currently handles only
    ``libgomp.so.1`` (see the comment above _LIBGOMP_SONAME); a silent no-op
    for every other missing dependency and for Windows/macOS.

    Never raises. A failure to bundle is logged as a warning, never silently
    treated as success."""
    if sys.platform in ("win32", "darwin"):
        return
    try:
        so_files = [f for f in target.iterdir()
                   if f.is_file() and not f.is_symlink() and ".so" in f.name]
        provided = {f.name for f in so_files}
        needed: set = set()
        for f in so_files:
            needed.update(elf_deps.needed_libraries(f))
    except OSError as e:
        logger.warning("could not scan %s for native dependencies: %s", target, e)
        return
    if _LIBGOMP_SONAME not in needed or _LIBGOMP_SONAME in provided:
        return
    console.print(f"[dim]Bundling {_LIBGOMP_SONAME} (OpenMP runtime; upstream's Linux "
                  "builds link it dynamically and ship no copy of their own)[/dim]")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            deb = Path(tmp) / "libgomp1.deb"
            _sl._download(_LIBGOMP_DEB_URL, deb)
            size = deb.stat().st_size
            if size < _sl._LIBGOMP_DEB_MIN_BYTES:
                raise ArtifactError(f"libgomp1 download too small ({size} bytes)")
            got = _sha256_file(deb)
            if got != _sl._LIBGOMP_DEB_SHA256:
                raise ArtifactError(
                    f"libgomp1 download sha256 mismatch (expected "
                    f"{_sl._LIBGOMP_DEB_SHA256}, got {got}) - refusing to bundle a "
                    "possibly tampered or wrong file")
            so_path = _extract_libgomp_from_deb(deb, Path(tmp))
            shutil.copy2(so_path, target / _LIBGOMP_SONAME)
            (target / "LICENSE.libgomp").write_text(_LIBGOMP_LICENSE_NOTICE, encoding="utf-8")
    except Exception as e:
        logger.warning("could not bundle %s into %s (%s) - if the runtime still "
                       "does not load, the reported cause will name what is "
                       "missing", _LIBGOMP_SONAME, target, e)
