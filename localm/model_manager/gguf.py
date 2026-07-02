# SPDX-License-Identifier: AGPL-3.0-or-later
"""GGUF file-format helpers: split-part detection, the gguf magic check,
safe model filenames, and SHA256 hashing. Leaf utilities, no registry/pull deps."""

import localm.model_manager as _mm  # read package-patchable names at call time

import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Callable
from typing import List
from typing import Optional
from rich.progress import BarColumn
from rich.progress import DownloadColumn
from rich.progress import Progress
from rich.progress import TextColumn
from rich.progress import TimeRemainingColumn
from ..debuglog import logger
from ._shared import console




# ------------------------------------------------------------------ #
#  Split GGUF (multi-part *-00001-of-00003.gguf files)                 #
# ------------------------------------------------------------------ #

# llama.cpp split naming convention: <stem>-00001-of-00003.gguf
_SPLIT_GGUF_RE = re.compile(
    r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE
)




def split_gguf_parts(filename: str) -> Optional[List[str]]:
    """
    If *filename* follows the llama.cpp split convention
    (``model-00001-of-00003.gguf``), return the full ordered list of part
    filenames. Returns None for regular single-file GGUFs.
    """
    m = _SPLIT_GGUF_RE.match(Path(filename).name)
    if not m:
        return None
    total = int(m.group("total"))
    if total < 2:
        return None
    stem = m.group("stem")
    return [f"{stem}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]




def first_split_part(filename: str) -> str:
    """Return the first-part filename for a split GGUF (llama.cpp wants this one)."""
    parts = split_gguf_parts(filename)
    return parts[0] if parts else filename




def missing_split_parts(first_part: Path) -> List[Path]:
    """
    Given the path of any part of a split GGUF, return sibling part paths
    that are missing on disk. Empty list means all parts present (or the
    file is not a split GGUF at all).
    """
    parts = split_gguf_parts(first_part.name)
    if not parts:
        return []
    return [first_part.parent / p for p in parts
            if not (first_part.parent / p).is_file()]




def _sha256_file(
    path: Path,
    progress: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Return the hex SHA256 digest of a file.

    When *progress* is given it is called ``progress(bytes_done, total_bytes)``
    after each block so a caller can drive a progress bar; *total_bytes* is the
    file size (0 if it cannot be stat'd). The callback must not raise."""
    h = hashlib.sha256()
    total = 0
    if progress is not None:
        try:
            total = path.stat().st_size
        except OSError:
            total = 0
    done = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
            if progress is not None:
                done += len(block)
                progress(done, total)
    return h.hexdigest()




def _sha256_file_bytes(data: bytes) -> str:
    """Return the hex SHA256 digest of an in-memory byte string."""
    return hashlib.sha256(data).hexdigest()




def _safe_models_filename(filename: str) -> Optional[str]:
    """Return a single-component filename confined to ``MODELS_DIR``.

    A model download must never write outside the models folder. ``filename`` is
    derived from untrusted input (a URL path or an ``owner/repo:file`` spec), so
    a value like ``../../evil.gguf`` or ``sub/dir/evil.gguf`` must be rejected
    rather than used as a destination. Returns the bare filename when it is a
    single, non-traversing path component, else ``None`` (GAP-CLI-2).
    """
    if not filename:
        return None
    # Reject anything that is not a single path component (no separators, no
    # drive/absolute prefixes, no '.'/'..').
    name = Path(filename).name
    if name != filename or name in ("", ".", ".."):
        return None
    if "/" in filename or "\\" in filename or os.sep in filename:
        return None
    dest = (_mm.MODELS_DIR / name).resolve()
    try:
        if dest.parent != _mm.MODELS_DIR.resolve():
            return None
    except OSError:
        return None
    return name




# Files at or below this size hash inline (silently); larger ones get the
# off-main-thread progress bar - the threading/UI overhead isn't worth it for
# small files. ~0.5 GB, matching the prior notice threshold.
_HASH_PROGRESS_MIN_BYTES = 512 * 1024 * 1024




def _hash_with_progress(path: Path) -> Optional[str]:
    """SHA256 a model file, showing live progress for large files.

    For files larger than ``_HASH_PROGRESS_MIN_BYTES`` the hash runs in a
    background worker thread so the main thread stays responsive and can render a
    progress bar (a one-time cost stored in the registry for duplicate
    detection); smaller files hash inline without any UI. Returns None for
    directories (HF models are identified by path only)."""
    if not path.is_file():
        return None
    try:
        size = path.stat().st_size
    except OSError:
        size = 0

    if size <= _mm._HASH_PROGRESS_MIN_BYTES:
        return _mm._sha256_file(path)

    from rich.progress import (SpinnerColumn)

    result: dict = {}

    def _worker(report):
        try:
            result["digest"] = _mm._sha256_file(path, progress=report)
        except Exception as exc:                      # surface on the main thread
            result["error"] = exc

    with Progress(
        SpinnerColumn(),
        TextColumn("[dim]Hashing {task.description} for duplicate detection[/dim]"),
        BarColumn(),
        DownloadColumn(),
        TimeRemainingColumn(),
        transient=True,
        console=console,
    ) as prog:
        task = prog.add_task(path.name, total=size)
        worker = threading.Thread(
            target=_worker,
            args=(lambda done, total: prog.update(task, completed=done),),
            daemon=True,
        )
        worker.start()
        worker.join()

    if "error" in result:
        raise result["error"]
    return result.get("digest")




# Conservative size floor for a plausible GGUF. The fixed header alone (magic +
# version + tensor count + KV count) is 24 bytes, and every real model carries a
# metadata KV block (general.architecture, tokenizer, ...) plus tensor infos well
# past 1 KiB before the first weight byte; even the tiniest test GGUFs are tens of
# KB. Kept deliberately low so no legitimate model can ever be rejected.
_GGUF_MIN_BYTES = 1024


def _has_gguf_magic(path: Path) -> bool:
    """True when *path* begins with the GGUF magic ``b"GGUF"`` and is at least
    ``_GGUF_MIN_BYTES`` long.

    Auto-registration (sync_models_dir) keys on the ``.gguf`` extension alone, so
    a foreign file renamed ``.gguf``, a 0-byte placeholder, or a partial copy that
    never received its header would otherwise be registered and then crash a later
    load - in the worst case wedging the app if it became the active model (R45,
    "copying a file into models/ broke the whole app"). A real GGUF always starts
    with this 4-byte magic; an unreadable file is treated as not-a-GGUF and
    skipped. The size floor additionally rejects a header-only truncated copy or
    placeholder that got just the magic, which would pass the magic check and then
    fail a later load with an opaque ggml error. (A mid-copy of a *valid* GGUF that
    already passed the floor stays a best-effort gap.)"""
    floor = _mm._GGUF_MIN_BYTES
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                return False
        size = path.stat().st_size
    except OSError:
        return False
    if size < floor:
        logger.debug(
            "skipping %s: GGUF magic but only %d bytes (< %d) - truncated or "
            "placeholder, not a usable model",
            path.name, size, floor,
        )
        return False
    return True




def _gguf_first_parts(d: Path, max_depth: int = 3) -> List[Path]:
    """First-part GGUF files inside *d*, scanning up to *max_depth* levels deep.

    *max_depth* counts the filename as level 1 (depth 1 = files directly in *d*,
    depth 3 = up to two subfolders down), so batch imports of models organised in
    subdirectories are picked up. Split GGUFs (``model-00001-of-00003.gguf``)
    contribute only their first part - llama.cpp finds the siblings on its own;
    loose single-file GGUFs contribute themselves. Mirrors the first-part filter
    in sync_models_dir.
    """
    out: List[Path] = []
    for f in sorted(d.rglob("*.gguf")):
        try:
            if not f.is_file():
                continue
            depth = len(f.relative_to(d).parts)
        except (OSError, ValueError):
            continue
        if depth > max_depth:
            continue
        parts = split_gguf_parts(f.name)
        if parts and f.name != parts[0]:
            continue   # non-first split part -> registered via its first part
        out.append(f)
    return out

