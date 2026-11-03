# SPDX-License-Identifier: AGPL-3.0-or-later
"""Source trees at a pinned commit, without a git program.

localm's managed ComfyUI and its pinned custom nodes are fetched as GitHub
source archives (``codeload.github.com/<owner>/<repo>/zip/<commit>``). An
archive is used only when its zip comment names the requested commit (git
writes the commit id there when it builds a zip of a commit) and, when
:data:`ARCHIVE_SHA256` lists the commit, when its sha256 matches. The files a
tree holds are written to :data:`SOURCE_RECORD` beside it, so a later update
can replace exactly those files and leave everything else (the venv, models,
outputs, installed custom nodes) alone. A tree made by ``git clone`` before
this module existed is read through its ``.git`` folder (HEAD and the index)
with no git program.

:func:`apply_tree` writes a new tree over an old one file by file and returns a
:class:`TreeChange` that can put back every file it overwrote or removed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

SOURCE_RECORD = ".localm-source.json"

# sha256 of the GitHub zip archive of each pinned commit, keyed by commit.
ARCHIVE_SHA256 = {
    "426f5eea52a7ece1dd599fff4cb6cd1f352ddd24":
        "01f381c8330f5c5bec9bd3d4c651e8b53b45ffb70020b9e1e947fc63f93a1548",
    "6ea2651e7df66d7585f6ffee804b20e92fb38b8a":
        "aad273a0b774684285b4496f6edff7e2293bd29d71ef89926d6861bfdf4947ac",
}

_GITHUB = re.compile(r"^https://github\.com/(?P<owner>[A-Za-z0-9._-]+)/"
                     r"(?P<repo>[A-Za-z0-9._-]+?)(?:\.git)?/?$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DOWNLOAD_TIMEOUT_S = 120

Progress = Optional[Callable[[str], None]]


class ArchiveError(Exception):
    """A source archive could not be fetched, verified or unpacked; the message
    says which and why."""


def archive_url(repo: str, commit: str) -> str:
    """The GitHub zip archive URL of *commit* in the GitHub repository *repo*
    (``https://github.com/<owner>/<repo>[.git]``). Raises ArchiveError for any
    other repository URL or a commit that is not a full 40-hex sha."""
    m = _GITHUB.match(repo or "")
    if not m:
        raise ArchiveError(f"{repo!r} is not a GitHub repository URL")
    if not _COMMIT.match(commit or ""):
        raise ArchiveError(f"{commit!r} is not a full commit sha")
    return f"https://codeload.github.com/{m['owner']}/{m['repo']}/zip/{commit}"


def download(url: str, dest: Path) -> None:
    """Download *url* to *dest* through localm's network policy and TLS
    verification. Raises ArchiveError on any failure."""
    from localm import netpolicy
    from localm.http_ssl import verified_urlopen
    from localm.model_manager.pull import _ssrf_resolve_final_url
    try:
        final = _ssrf_resolve_final_url(url)
        req = urllib.request.Request(final, headers={"User-Agent": "localm"})
        with verified_urlopen(req, timeout=_DOWNLOAD_TIMEOUT_S) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f, 256 * 1024)
    except netpolicy.NetworkPolicyError as e:
        raise ArchiveError(f"the download was refused by the network policy: {e}") from None
    except (OSError, ValueError) as e:
        raise ArchiveError(f"downloading {url} failed: {e}") from None


def _long(path: Path) -> Path:
    """*path* made absolute, with the Windows extended-length prefix on Windows
    so a file deeper than 260 characters can be written."""
    path = Path(os.path.abspath(path))
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        text = str(path)
        return Path("\\\\?\\UNC\\" + text[2:] if text.startswith("\\\\") else "\\\\?\\" + text)
    return path


def _safe_name(name: str) -> Optional[str]:
    """*name* (a path inside the archive, below its top folder) as a relative
    POSIX path, or None when it is empty, absolute or climbs out."""
    p = PurePosixPath(name)
    if not name or p.is_absolute() or ".." in p.parts or ":" in p.parts[0]:
        return None
    return p.as_posix()


def _unpack(archive: Path, dest: Path, commit: str) -> list[str]:
    """Unpack the GitHub archive *archive* of *commit* into the new folder *dest*
    and return the files written (relative POSIX paths). Raises ArchiveError when
    the zip comment does not name *commit* or an entry is unsafe."""
    try:
        zf = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as e:
        raise ArchiveError(f"the source archive is not a readable zip file: {e}") from None
    with zf:
        named = zf.comment.decode("ascii", "replace").strip()
        if named != commit:
            raise ArchiveError(f"the source archive is of commit {named or '(none)'!r}, "
                               f"not {commit}")
        files = []
        dest.mkdir(parents=True)
        base = _long(dest)
        for info in zf.infolist():
            top, _, rest = info.filename.partition("/")
            if info.is_dir() or not rest:
                continue
            rel = _safe_name(rest)
            if rel is None:
                raise ArchiveError(f"the source archive holds an unsafe path {info.filename!r}")
            out = base.joinpath(*rel.split("/"))
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            mode = info.external_attr >> 16
            if os.name == "posix" and mode & 0o111:
                out.chmod(0o755)
            files.append(rel)
    return sorted(files)


def fetch_tree(repo: str, commit: str, dest: Path, *, on_progress: Progress = None,
               fetch: Optional[Callable[[str, Path], None]] = None) -> list[str]:
    """Fetch *commit* of *repo* into the new folder *dest* and write its
    :data:`SOURCE_RECORD`. Returns the files of the tree. *fetch* replaces
    :func:`download` (tests). The archive is downloaded next to *dest* and
    removed afterwards; *dest* is left absent when anything fails. Raises
    ArchiveError."""
    dest = Path(dest)
    if dest.exists():
        raise ArchiveError(f"{dest} already exists")
    url = archive_url(repo, commit)
    dest.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".localm-src-", dir=dest.parent))
    try:
        archive = work / "source.zip"
        if on_progress:
            on_progress(f"Downloading {url} ...")
        (fetch or download)(url, archive)
        expected = ARCHIVE_SHA256.get(commit)
        if expected is not None:
            got = hashlib.sha256(archive.read_bytes()).hexdigest()
            if got != expected:
                raise ArchiveError(f"the source archive of {commit[:12]} has sha256 {got}, "
                                   f"not the pinned {expected}")
        staging = work / "tree"
        files = _unpack(archive, staging, commit)
        write_record(staging, repo, commit, files)
        os.replace(staging, dest)
        return files
    finally:
        shutil.rmtree(_long(work), ignore_errors=True)


def write_record(folder: Path, repo: str, commit: str, files: list[str]) -> None:
    """Write *folder*'s :data:`SOURCE_RECORD`: where its tree came from and the
    files it holds."""
    (Path(folder) / SOURCE_RECORD).write_text(
        json.dumps({"repo": repo, "commit": commit, "files": sorted(files)}, indent=1) + "\n",
        encoding="utf-8")


def read_record(folder: Path) -> Optional[dict]:
    """*folder*'s :data:`SOURCE_RECORD`, or None when it is absent or unreadable."""
    try:
        data = json.loads((Path(folder) / SOURCE_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    if not isinstance(data, dict) or not _COMMIT.match(str(data.get("commit") or "")):
        return None
    if not isinstance(data.get("files"), list):
        return None
    return data


# --------------------------------------------------------------------------- #
#  A tree made by git clone, read without a git program                        #
# --------------------------------------------------------------------------- #

def _git_dir(folder: Path) -> Optional[Path]:
    """The git folder of a checkout whose top is *folder* (``.git`` itself, or
    the folder a ``.git`` file points at), or None."""
    dot = Path(folder) / ".git"
    if dot.is_dir():
        return dot
    try:
        text = dot.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None
    if text.startswith("gitdir:"):
        target = Path(text[len("gitdir:"):].strip())
        target = target if target.is_absolute() else (Path(folder) / target)
        return target if target.is_dir() else None
    return None


def _common_dir(git_dir: Path) -> Path:
    try:
        rel = (git_dir / "commondir").read_text(encoding="utf-8").strip()
    except OSError:
        return git_dir
    p = Path(rel)
    return p if p.is_absolute() else (git_dir / p)


def git_head(folder: Path) -> Optional[str]:
    """The commit a git checkout whose top is *folder* has checked out, read
    from its ``HEAD``, loose refs and ``packed-refs``; None when *folder* is not
    the top of a git checkout or the commit cannot be read."""
    gd = _git_dir(folder)
    if gd is None:
        return None
    try:
        head = (gd / "HEAD").read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None
    if _COMMIT.match(head):
        return head
    if not head.startswith("ref:"):
        return None
    ref = head[len("ref:"):].strip()
    for base in (gd, _common_dir(gd)):
        try:
            value = (base / ref).read_text(encoding="utf-8").strip()
        except (OSError, ValueError):
            continue
        if _COMMIT.match(value):
            return value
    try:
        packed = (_common_dir(gd) / "packed-refs").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    for line in packed.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == ref and _COMMIT.match(parts[0]):
            return parts[0]
    return None


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    """Git's offset varint (index v4 path compression) at *pos*: (value, next)."""
    b = data[pos]
    pos += 1
    value = b & 0x7F
    while b & 0x80:
        b = data[pos]
        pos += 1
        value = ((value + 1) << 7) | (b & 0x7F)
    return value, pos


def git_tracked_files(folder: Path) -> Optional[list[str]]:
    """The files the git index of the checkout whose top is *folder* tracks
    (index versions 2, 3 and 4), as relative POSIX paths; None when there is no
    readable index."""
    gd = _git_dir(folder)
    if gd is None:
        return None
    try:
        data = (gd / "index").read_bytes()
    except OSError:
        return None
    if len(data) < 12 or data[:4] != b"DIRC":
        return None
    version, count = struct.unpack(">II", data[4:12])
    if version not in (2, 3, 4):
        return None
    pos, prev, out = 12, b"", []
    try:
        for _ in range(count):
            start = pos
            flags = struct.unpack(">H", data[pos + 60:pos + 62])[0]
            pos += 62
            if version >= 3 and flags & 0x4000:
                pos += 2
            if version == 4:
                strip, pos = _varint(data, pos)
                end = data.index(b"\0", pos)
                name = prev[:len(prev) - strip] + data[pos:end]
                pos = end + 1
            else:
                end = data.index(b"\0", pos)
                name = data[pos:end]
                pos = start + ((end - start + 8) // 8) * 8
            prev = name
            if (flags >> 12) & 0x3 == 0:
                out.append(name.decode("utf-8", "surrogateescape"))
    except (struct.error, ValueError, IndexError):
        return None
    return sorted(set(out))


def tree_commit(folder: Path) -> Optional[str]:
    """The commit of the source tree in *folder*: its :data:`SOURCE_RECORD`, else
    its git checkout's HEAD; None when neither is readable."""
    rec = read_record(folder)
    return rec["commit"] if rec else git_head(folder)


def tree_files(folder: Path) -> Optional[list[str]]:
    """The files of the source tree in *folder*: its :data:`SOURCE_RECORD`, else
    its git index; None when neither is readable."""
    rec = read_record(folder)
    if rec:
        return [f for f in rec["files"] if isinstance(f, str) and _safe_name(f)]
    return git_tracked_files(folder)


# --------------------------------------------------------------------------- #
#  Replacing a tree's files, with a way back                                   #
# --------------------------------------------------------------------------- #

@dataclass
class TreeChange:
    """What :func:`apply_tree` changed in *root*: files it created, and a backup
    of every file it overwrote or removed (``backup/<rel>``)."""
    root: Path
    backup: Path
    created: list = field(default_factory=list)
    saved: list = field(default_factory=list)

    def rollback(self) -> list[str]:
        """Put *root* back as it was before the change. Returns the files that
        could not be restored (empty on success)."""
        failed = []
        for rel in self.created:
            try:
                (self.root / rel).unlink(missing_ok=True)
            except OSError:
                failed.append(rel)
        for rel in self.saved:
            src, dst = self.backup / rel, self.root / rel
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            except OSError:
                failed.append(rel)
        return failed

    def discard(self) -> None:
        """Remove the backup."""
        shutil.rmtree(self.backup, ignore_errors=True)


def apply_tree(new: Path, root: Path, old_files: Optional[list[str]],
               backup: Path) -> TreeChange:
    """Make the source files of *root* those of the tree in *new*: every file of
    *new* is written into *root*, and every file of *old_files* (the files of the
    tree *root* held) that *new* lacks is removed. Files that are in neither (a
    venv, models, outputs, installed custom nodes) are not touched. Each file
    overwritten or removed is first copied to *backup*, which must not exist.
    Raises OSError after putting back what it changed when a write fails."""
    new_files = tree_files(new) or []
    new, root, backup = _long(new), _long(root), _long(backup)
    change = TreeChange(root=root, backup=backup)
    backup.mkdir(parents=True)
    wanted = set(new_files) | {SOURCE_RECORD}
    removed = [f for f in (old_files or []) if f not in wanted and _safe_name(f)]
    written = list(new_files) + [SOURCE_RECORD]
    try:
        for rel in written + removed:
            target = root / rel
            if target.is_file():
                keep = backup / rel
                keep.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, keep)
                change.saved.append(rel)
            elif rel not in removed:
                change.created.append(rel)
        for rel in written:
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(new / rel, target)
        for rel in removed:
            (root / rel).unlink(missing_ok=True)
    except OSError:
        change.rollback()
        raise
    return change
