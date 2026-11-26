# SPDX-License-Identifier: AGPL-3.0-or-later
"""The list of the native runtime's files, written when the runtime is installed and
compared by the integrity check (``localm.integrity``).

``<runtime folder>/.localm-files.json``::

    {"format": "localm-runtime-files/1",
     "files": {"<path in the folder>": {"size": <bytes>, "mtime_ns": <int>,
                                        "sha256": "<hex>"}, ...}}

It lists every regular file of the folder except localm's own records about the
runtime (every top-level entry named ``.localm-*``, :func:`own`), the placeholders
``.gitignore`` and ``.gitkeep``, and ``__pycache__``. :func:`compare` reports a file
of another size without reading it, reads a file whose modification time changed,
and reads every file when asked for a full comparison. A CPU tier library that
``cpu_backend_select`` set aside under :data:`PRUNED_PREFIX` counts as the listed
file of its own name. Imports nothing from localm.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

NAME = ".localm-files.json"
FORMAT = "localm-runtime-files/1"
# The name start of every record localm keeps in the runtime folder about the runtime.
RECORD_PREFIX = ".localm-"
# Files a source checkout keeps in the empty runtime folder.
PLACEHOLDERS = frozenset({".gitignore", ".gitkeep"})
# The name start ``cpu_backend_select`` gives a CPU tier library it sets aside.
PRUNED_PREFIX = "_unused-"


def sha256(path) -> str:
    """The sha256 of the file at *path*, read in full. Raises OSError."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def own(rel: str) -> bool:
    """Whether the path *rel* (POSIX, relative to the runtime folder) is not a runtime
    file: a record of localm's (a top-level entry ``.localm-*`` and everything in it),
    a placeholder, or a ``__pycache__``."""
    parts = rel.split("/")
    return (parts[0].startswith(RECORD_PREFIX) or "__pycache__" in parts
            or (len(parts) == 1 and parts[0] in PLACEHOLDERS))


def listed(folder) -> list:
    """The paths (POSIX, relative to *folder*) of the runtime files in *folder*; an
    entry whose kind cannot be read counts as a file. Raises OSError when the folder
    cannot be read."""
    folder = Path(folder)
    out = []
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder).as_posix()
        if own(rel):
            continue
        try:
            plain = p.is_file() and not p.is_symlink()
        except OSError:
            plain = True
        if plain:
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


def _stat(folder: Path, rel: str):
    """``os.stat`` of the listed file *rel*, or of the same file set aside under
    :data:`PRUNED_PREFIX` when only that one is there. Raises OSError."""
    p = folder.joinpath(*rel.split("/"))
    try:
        return p, os.stat(p)
    except (FileNotFoundError, NotADirectoryError):
        aside = p.with_name(PRUNED_PREFIX + p.name)
        try:
            return aside, os.stat(aside)
        except (FileNotFoundError, NotADirectoryError):
            pass
        raise


def compare(folder, *, full: bool = False) -> dict:
    """``{"present", "listed", "full", "checked", "differ", "missing", "unreadable",
    "hashed", "error"}`` for the runtime in *folder*: whether it holds runtime files,
    whether it has a readable list, how many listed files were compared, those of
    another size or content, those absent and those that could not be read, how many
    were read in full, and why the folder or its list could not be read (None when
    they could). Never raises."""
    folder = Path(folder)
    out = {"present": False, "listed": False, "full": full, "checked": 0,
           "differ": [], "missing": [], "unreadable": [], "hashed": 0, "error": None}
    try:
        if not folder.is_dir() or not listed(folder):
            return out
        out["present"] = True
        text = (folder / NAME).read_text(encoding="utf-8")
    except FileNotFoundError:
        return out
    except (OSError, UnicodeDecodeError) as e:
        detail = getattr(e, "strerror", None) or getattr(e, "reason", None) or "unreadable"
        out["error"] = f"{type(e).__name__}: {detail}"
        return out
    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as e:
        out["error"] = f"the list {NAME} cannot be read: {e}"
        return out
    if not (isinstance(data, dict) and data.get("format") == FORMAT
            and isinstance(data.get("files"), dict)):
        out["error"] = f"the list {NAME} is not in the format {FORMAT}"
        return out
    out["listed"] = True
    for rel, entry in sorted(data["files"].items()):
        if not isinstance(entry, dict):
            continue
        out["checked"] += 1
        try:
            p, st = _stat(folder, rel)
        except (FileNotFoundError, NotADirectoryError):
            out["missing"].append(rel)
            continue
        except OSError:
            out["unreadable"].append(rel)
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
            out["unreadable"].append(rel)
            continue
        if not same:
            out["differ"].append(rel)
    return out
