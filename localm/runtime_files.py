# SPDX-License-Identifier: AGPL-3.0-or-later
"""The list of the native runtime's files, written when the runtime is installed and
compared by the integrity check (``localm.integrity``).

``<runtime folder>/.localm-files.json``::

    {"format": "localm-runtime-files/1",
     "files": {"<path in the folder>": {"size": <bytes>, "mtime_ns": <int>,
                                        "sha256": "<hex>"}, ...}}

It lists every regular file of the folder, its own records (:data:`OWN`) and
``__pycache__`` aside. :func:`compare` reports a file of another size without reading
it, reads a file whose modification time changed, and reads every file when asked
for a full comparison. Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

NAME = ".localm-files.json"
FORMAT = "localm-runtime-files/1"
# Files of the runtime folder that describe the runtime instead of being part of it.
OWN = frozenset({".localm-backend", ".localm-backend-choice", NAME, ".gitignore",
                 ".gitkeep", ".localm-runtime-kept.json"})


def sha256(path) -> str:
    """The sha256 of the file at *path*, read in full. Raises OSError."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def listed(folder) -> list:
    """The paths (POSIX, relative to *folder*) of the runtime files in *folder*."""
    folder = Path(folder)
    out = []
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder).as_posix()
        if "__pycache__" in rel.split("/") or (p.parent == folder and p.name in OWN):
            continue
        if p.is_file() and not p.is_symlink():
            out.append(rel)
    return out


def write(folder) -> int:
    """Write the list of the runtime files in *folder* and return how many it lists.
    Raises OSError."""
    folder = Path(folder)
    files = {}
    for rel in listed(folder):
        p = folder.joinpath(*rel.split("/"))
        st = os.stat(p)
        files[rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha256(p)}
    tmp = folder / (NAME + ".tmp")
    tmp.write_text(json.dumps({"format": FORMAT, "files": files}, sort_keys=True),
                   encoding="utf-8")
    os.replace(tmp, folder / NAME)
    return len(files)


def compare(folder, *, full: bool = False) -> dict:
    """``{"present", "listed", "full", "checked", "differ", "missing", "hashed"}`` for
    the runtime in *folder*: whether it holds runtime files, whether it has a readable
    list, how many listed files were compared, those of another size or content and
    those absent, and how many were read in full."""
    folder = Path(folder)
    out = {"present": False, "listed": False, "full": full, "checked": 0,
           "differ": [], "missing": [], "hashed": 0}
    if not folder.is_dir() or not listed(folder):
        return out
    out["present"] = True
    try:
        data = json.loads((folder / NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return out
    if not (isinstance(data, dict) and data.get("format") == FORMAT
            and isinstance(data.get("files"), dict)):
        return out
    out["listed"] = True
    for rel, entry in sorted(data["files"].items()):
        if not isinstance(entry, dict):
            continue
        out["checked"] += 1
        p = folder.joinpath(*rel.split("/"))
        try:
            st = os.stat(p)
        except (FileNotFoundError, NotADirectoryError):
            out["missing"].append(rel)
            continue
        if st.st_size != entry.get("size"):
            out["differ"].append(rel)
            continue
        if not full and st.st_mtime_ns == entry.get("mtime_ns"):
            continue
        out["hashed"] += 1
        try:
            same = sha256(p) == entry.get("sha256")
        except OSError:
            same = False
        if not same:
            out["differ"].append(rel)
    return out
