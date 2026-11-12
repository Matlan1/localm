# SPDX-License-Identifier: AGPL-3.0-or-later
"""The update package statement: what a localm update package changes.

A release build carries ``update-package.json`` at its root, added by the
release build (:func:`stamp_zip`) as its last member before it is signed, so
the signature covers it::

    {"format": "localm-update-package/1",
     "version": "<the build's VERSION>",
     "files": {"<path in the build>": {"sha256": "<hex>", "size": <bytes>}, ...},
     "components": {"<component>": {"version": "<pin>",
                                    "assets": {"<platform/backend or asset>": "<sha256>"}}}}

``files`` lists every file of localm itself in the package, the statement
excepted. ``components`` lists every machine-specific component the build pins,
read by ``updater_components.stated_components`` (the one list of components an
update can carry is ``updater_components.STATED``): its version and, per
platform and backend, the checksum of what is downloaded for it.

:func:`stamp_zip` refuses to finish a build whose statement, read back from the
written zip, does not match the zip. :func:`check` runs the same comparison at
update time, after the signature check and before anything is replaced. A build
newer than :data:`LAST_WITHOUT` must carry a statement; an older one is reached
only by choosing that version and is applied from its file tree.

Imports nothing from localm but the version helpers: the component readers are
passed in by the caller.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, Optional

NAME = "update-package.json"
FORMAT = "localm-update-package/1"
# The newest version whose builds carry no statement.
LAST_WITHOUT = "0.3.4b2"
# Largest statement accepted, in bytes.
MAX_BYTES = 32 * 1024 * 1024

_HEX = frozenset("0123456789abcdef")


class PackageError(ValueError):
    """A statement that is malformed or does not match the package."""


class MissingStatement(PackageError):
    """A build that must carry a statement carries none."""


def is_sha256(v) -> bool:
    """True for a 64-character lowercase hex string."""
    return isinstance(v, str) and len(v) == 64 and set(v) <= _HEX


def required_for(version: str) -> bool:
    """Whether a build of *version* must carry a statement."""
    from localm import _version
    return _version.is_newer(version, LAST_WITHOUT)


# ------------------------------- the files ------------------------------------

def _file_entry(path: Path) -> dict:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
            size += len(block)
    return {"sha256": h.hexdigest(), "size": size}


def files(root) -> dict:
    """``{posix path: {"sha256", "size"}}`` for every file under *root*, the
    statement itself excepted."""
    root = Path(root)
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if rel != NAME:
                out[rel] = _file_entry(p)
    return out


def tops(manifest: dict) -> set:
    """The top-level entries the stated files belong to, plus the statement."""
    return {rel.split("/", 1)[0] for rel in manifest["files"]} | {NAME}


# ------------------------------- reading --------------------------------------

def _strict_json(text: str):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise PackageError(f"duplicate field {k!r}")
            out[k] = v
        return out

    def constant(name):
        raise PackageError(f"{name} is not allowed")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except RecursionError:
        raise PackageError(f"{NAME} is nested too deeply") from None


def _fields(obj, where: str, required: set) -> None:
    if not isinstance(obj, dict):
        raise PackageError(f"{where} is not an object")
    missing = required - obj.keys()
    if missing:
        raise PackageError(f"{where} lacks {', '.join(sorted(missing))}")
    extra = obj.keys() - required
    if extra:
        raise PackageError(f"{where} has unknown field(s) {', '.join(sorted(map(str, extra)))}")


def _safe_path(rel) -> bool:
    if not isinstance(rel, str) or not rel or rel == NAME or "\\" in rel:
        return False
    if rel.startswith("/") or (len(rel) >= 2 and rel[1] == ":"):
        return False
    return all(part not in ("", ".", "..") for part in rel.split("/"))


def parse(text: str) -> dict:
    """The statement in *text*, checked for shape. Raises PackageError."""
    if len(text) > MAX_BYTES:
        raise PackageError(f"{NAME} is larger than {MAX_BYTES} bytes")
    try:
        obj = _strict_json(text)
    except (ValueError, RecursionError) as e:
        if isinstance(e, PackageError):
            raise
        raise PackageError(f"{NAME} is not valid JSON: {e}") from None
    _fields(obj, NAME, {"format", "version", "files", "components"})
    if obj["format"] != FORMAT:
        raise PackageError(f"{NAME} has format {str(obj['format'])[:40]!r}, not {FORMAT!r}")
    if not isinstance(obj["version"], str) or not obj["version"].strip():
        raise PackageError(f"{NAME} names no version")
    if not isinstance(obj["files"], dict) or not obj["files"]:
        raise PackageError(f"{NAME} states no files")
    for rel, entry in obj["files"].items():
        if not _safe_path(rel):
            raise PackageError(f"{NAME} states an unsafe path {str(rel)[:80]!r}")
        _fields(entry, f"the entry for {rel}", {"sha256", "size"})
        if not is_sha256(entry["sha256"]):
            raise PackageError(f"the entry for {rel} has no sha256")
        size = entry["size"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise PackageError(f"the entry for {rel} has no size")
    if not isinstance(obj["components"], dict):
        raise PackageError(f"{NAME} components is not an object")
    for name, entry in obj["components"].items():
        _fields(entry, f"component {name}", {"version", "assets"})
        if not isinstance(entry["version"], str) or not entry["version"]:
            raise PackageError(f"component {name} names no version")
        if not isinstance(entry["assets"], dict) or not all(
                isinstance(k, str) and k and is_sha256(v) for k, v in entry["assets"].items()):
            raise PackageError(f"component {name} has an asset without a sha256")
    return obj


def _first(items) -> str:
    items = sorted(items)
    return f"{len(items)} file(s), first {items[0]}"


def check(root, version: str, components_of: Callable) -> Optional[dict]:
    """Check the statement of the extracted build at *root*, whose VERSION is
    *version*, against the build, and return it; None for a build that need not
    carry one (:func:`required_for`) and does not. *components_of* returns the
    components the tree at a given root pins, in the statement's shape.

    Raises :class:`MissingStatement` when a build that must carry one does not,
    and :class:`PackageError` when the statement is malformed, names another
    version, states a file the build lacks or that differs from it, misses a file
    the build holds, or states the components differently from what the build's
    own code pins. Reads only; changes nothing."""
    root = Path(root)
    path = root / NAME
    if not path.is_file():
        if not required_for(version):
            return None
        raise MissingStatement(f"{NAME} is missing from the {version} build")
    try:
        if path.stat().st_size > MAX_BYTES:
            raise PackageError(f"{NAME} is larger than {MAX_BYTES} bytes")
        manifest = parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as e:
        raise PackageError(f"{NAME} cannot be read: {e}") from None
    if manifest["version"] != version:
        raise PackageError(f"{NAME} is for {manifest['version'][:40]}, the build is {version}")
    have = files(root)
    stated = manifest["files"]
    missing = stated.keys() - have.keys()
    if missing:
        raise PackageError(f"the package is incomplete: {_first(missing)} stated but absent")
    extra = have.keys() - stated.keys()
    if extra:
        raise PackageError(f"the package holds {_first(extra)} it does not state")
    differ = [rel for rel in stated if stated[rel] != have[rel]]
    if differ:
        raise PackageError(f"{_first(differ)} differ from what the package states (a damaged "
                           "or cut-off file)")
    pinned = components_of(root)
    wrong = sorted(n for n in set(pinned) | set(manifest["components"])
                   if pinned.get(n) != manifest["components"].get(n))
    if wrong:
        raise PackageError("the package states these components differently from what its "
                           "code pins: " + ", ".join(map(str, wrong)))
    return manifest


# ------------------------------- stamping -------------------------------------

def _extract_safe(z: zipfile.ZipFile, dest: Path) -> None:
    for info in z.infolist():
        if info.is_dir() or not (info.filename == NAME or _safe_path(info.filename)):
            continue
        out = dest.joinpath(*info.filename.split("/"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(z.read(info))


def stamp_zip(target, components_of: Callable) -> dict:
    """Add the statement to the build zip *target* (a path or a seekable binary
    file object) and return it, then read the zip back and check it
    (:func:`check`). *components_of* is as for :func:`check`. The zip must be
    flat (``VERSION`` at its root), must not already hold a statement and may hold
    only safe member names. Raises PackageError when any of this fails, or when
    the written zip does not match its statement."""
    with zipfile.ZipFile(target) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        names = [i.filename for i in infos]
        if NAME in names:
            raise PackageError(f"the build already holds {NAME}")
        if "VERSION" not in names:
            raise PackageError("the build has no VERSION at its root")
        unsafe = [n for n in names if not _safe_path(n)]
        if unsafe:
            raise PackageError(f"the build holds an unsafe member name {unsafe[0][:80]!r}")
        if len(set(names)) != len(names):
            raise PackageError("the build holds a member name twice")
        stated = {}
        for info in infos:
            data = z.read(info)
            stated[info.filename] = {"sha256": hashlib.sha256(data).hexdigest(),
                                     "size": len(data)}
        version = z.read("VERSION").decode("utf-8").strip()
        with tempfile.TemporaryDirectory() as tmp:
            _extract_safe(z, Path(tmp))
            comps = components_of(Path(tmp))
    manifest = {"format": FORMAT, "version": version,
                "files": dict(sorted(stated.items())), "components": comps}
    with zipfile.ZipFile(target, "a", zipfile.ZIP_DEFLATED) as z:
        z.writestr(NAME, json.dumps(manifest, indent=1, sort_keys=True))
    if hasattr(target, "seek"):
        target.seek(0)
    with zipfile.ZipFile(target) as z, tempfile.TemporaryDirectory() as tmp:
        _extract_safe(z, Path(tmp))
        readback = check(Path(tmp), version, components_of)
    if readback != manifest:
        raise PackageError(f"{NAME} read back from the build differs from what was written")
    return manifest


def stamped(data: bytes, components_of: Callable) -> bytes:
    """The build zip bytes *data* with the statement added (:func:`stamp_zip`)."""
    buf = io.BytesIO(data)
    stamp_zip(buf, components_of)
    return buf.getvalue()
