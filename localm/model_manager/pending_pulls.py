# SPDX-License-Identifier: AGPL-3.0-or-later
"""Records of model downloads that have not finished.

Every remote pull writes one JSON record to ``<data dir>/pending-pulls/<id>.json``
when its transfer starts: the request (spec and options), the exact source
(repository, revision, URLs, expected sizes and digests) and where its partial
data lives. The record is removed when the pull finishes. A pull that stops
without finishing keeps its record when partial data is on disk, so a later
``localm pull`` or the GUI can offer to resume it, delete its partial data, or
hide it.

A record whose ``ended`` field is False belongs to a pull that may still run:
it counts as interrupted only once its owner process is proven gone (pid
liveness plus the process start identity, never elapsed time).
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Optional

import localm.model_manager as _mm
from localm import instances

from ..debuglog import logger

RECORDS_DIR_NAME = "pending-pulls"
_VERSION = 1
_ID_RE = re.compile(r"\A[0-9a-f]{32}\Z")
_ORPHAN_ID_RE = re.compile(r"\Aorphan-[0-9a-f]{24}\Z")
_REQUEST_KEYS = ("spec", "name", "sha256", "mmproj", "type", "redownload",
                 "dest_dir", "register")


class UnknownPull(KeyError):
    """No interrupted download has the given id."""


class _Active:
    """The record of the pull running in this thread, before and after its
    first write."""

    def __init__(self, request: dict) -> None:
        self.id = uuid.uuid4().hex
        self.written = False
        self.done_target: Optional[tuple] = None
        self.rec: dict[str, Any] = {
            "version": _VERSION,
            "id": self.id,
            "created": time.time(),
            "ended": False,
            "dismissed": False,
            "request": {k: request.get(k) for k in _REQUEST_KEYS},
            "owner": _owner_record(),
            "source": None,
            "repo": None,
            "revision": None,
            "commit": None,
            "url": None,
            "sha256": None,
            "files": [],
            "parts": [],
            "snapshot": None,
        }


_CURRENT: contextvars.ContextVar[_Active | None] = contextvars.ContextVar(
    "localm_pending_pull", default=None)
_LIVE_IDS: set[str] = set()
_LIVE_LOCK = threading.Lock()
_WRITE_LOCK = threading.Lock()


def _snapshot_target(rec: dict) -> Optional[tuple]:
    """``(repo, revision, folder)`` that identifies the full-repo download of
    *rec* whatever options requested it, or None for any other record."""
    snap = rec.get("snapshot")
    if (rec.get("source") != "hf-repo" or not isinstance(snap, dict)
            or not isinstance(snap.get("dir"), str) or not isinstance(rec.get("repo"), str)):
        return None
    return (rec["repo"].casefold(), rec.get("revision") or None,
            os.path.normcase(os.path.abspath(snap["dir"])))


def _same_pull(old: dict, request: dict, target: Optional[tuple]) -> bool:
    """True when the record *old* is of the pull of *request*, or of the
    full-repo download *target*."""
    return old["request"] == request or (target is not None
                                         and _snapshot_target(old) == target)


def records_dir() -> Path:
    """The folder holding the records: beside the models folder."""
    return _mm.MODELS_DIR.parent / RECORDS_DIR_NAME


def _record_path(pull_id: str) -> Path:
    return records_dir() / f"{pull_id}.json"


def _owner_record() -> dict:
    """This process as the owner of a record, in the shape
    ``pull._record_holder_is_gone`` reads."""
    where = records_dir()
    return {"pid": os.getpid(), "space": instances.pid_space_id(),
            "host": instances.host_id(),
            "local": instances.path_is_local(where if where.exists() else where.parent),
            "start": instances.process_start_identity(os.getpid())}


def _write(rec: dict) -> None:
    """Write *rec* to its file atomically. Raises OSError on failure."""
    from ..storekit import atomic_write
    d = records_dir()
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(_record_path(rec["id"]), json.dumps(rec, indent=1))


def _remove(pull_id: str) -> None:
    """Delete the record file of *pull_id*; a missing file is not an error."""
    try:
        _record_path(pull_id).unlink(missing_ok=True)
    except OSError as e:
        logger.warning("could not remove the download record %s: %s",
                       _record_path(pull_id), e)


def _read(path: Path) -> Optional[dict]:
    """The record at *path*, or None when it is missing, unreadable or not a
    record of this format."""
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, RecursionError) as e:
        logger.warning("ignoring the unreadable download record %s: %s", path, e)
        return None
    if (not isinstance(rec, dict) or rec.get("version") != _VERSION
            or rec.get("id") != path.stem or not _ID_RE.match(path.stem)
            or not isinstance(rec.get("request"), dict)
            or not isinstance(rec.get("request", {}).get("spec"), str)):
        logger.warning("ignoring the malformed download record %s", path)
        return None
    for key in ("files", "parts"):
        if not isinstance(rec.get(key), list):
            rec[key] = []
    return rec


def _save_current(act: _Active) -> None:
    """Write the record of *act*; on its first write, records of the same
    request or of the same full-repo target (see :func:`_snapshot_target`)
    that are not running are replaced by it. A failed write is logged and the
    download goes on without a record."""
    with _WRITE_LOCK:
        first = not act.written
        if first:
            target = _snapshot_target(act.rec)
            for _path, old in _iter_records():
                if old["id"] == act.id or not _same_pull(old, act.rec["request"], target):
                    continue
                if _is_running(old):
                    continue
                snap = old.get("snapshot")
                mine = act.rec.get("snapshot")
                if (isinstance(snap, dict) and snap.get("created")
                        and isinstance(mine, dict) and mine.get("dir") == snap.get("dir")):
                    mine["created"] = True
                act.rec.setdefault("superseded", []).append(old["id"])
                _remove(old["id"])
        try:
            _write(act.rec)
        except OSError as e:
            logger.warning("could not record the download of %s (%s); if it is "
                           "interrupted it will not be offered for resuming",
                           act.rec["request"]["spec"], e)
            return
        act.written = True


@contextlib.contextmanager
def tracking(request: dict) -> Iterator[dict]:
    """Track the pull described by *request* (the keys of ``_REQUEST_KEYS``)
    for the block. Yields a dict whose ``"ok"`` key the caller sets to True
    when the pull succeeded. Nothing is written until a ``note_*`` call marks
    the start of a transfer. On exit: success removes the record; any other
    exit (a failure, an exception, Ctrl+C) keeps it, marked ended, when
    partial data of it is on disk, and removes it otherwise. Success also
    removes the records of other pulls of the same request or of the same
    full-repo target."""
    act = _Active(request)
    token = _CURRENT.set(act)
    with _LIVE_LOCK:
        _LIVE_IDS.add(act.id)
    outcome = {"ok": False}
    try:
        yield outcome
    finally:
        _CURRENT.reset(token)
        try:
            _finish(act, bool(outcome.get("ok")))
        finally:
            with _LIVE_LOCK:
                _LIVE_IDS.discard(act.id)


def _finish(act: _Active, ok: bool) -> None:
    if ok:
        target = act.done_target or _snapshot_target(act.rec)
        for _p, old in _iter_records():
            if old["id"] != act.id and _same_pull(old, act.rec["request"], target) \
                    and not _is_running(old):
                _remove(old["id"])
    if not act.written:
        return
    if ok:
        _remove(act.id)
        return
    try:
        have = bytes_on_disk(act.rec) or int(bool(_locate(act.rec).outside))
    except Exception as e:
        logger.warning("could not measure the partial data of %s: %s",
                       act.rec["request"]["spec"], e)
        have = 1
    if have <= 0:
        _remove(act.id)
        return
    act.rec["ended"] = True
    act.rec["ended_at"] = time.time()
    with _WRITE_LOCK:
        try:
            _write(act.rec)
        except OSError as e:
            logger.warning("could not mark the download of %s as stopped: %s",
                           act.rec["request"]["spec"], e)


def current_request() -> Optional[dict]:
    """The request of the pull tracked in this thread, or None."""
    act = _CURRENT.get()
    return dict(act.rec["request"]) if act is not None else None


def note_source(**fields: Any) -> None:
    """Merge source facts (``source``, ``repo``, ``revision``, ``commit``,
    ``url``, ``sha256``) into the tracked record. A field already set keeps
    its value, so a projector fetched after the model does not replace the
    model's source. Rewritten when already written. No effect outside
    :func:`tracking`."""
    act = _CURRENT.get()
    if act is None:
        return
    for k, v in fields.items():
        if v is not None and act.rec.get(k) is None:
            act.rec[k] = v
    if act.written:
        _save_current(act)


def note_hf_files(base: Path, files: list[dict]) -> None:
    """Record that the Hugging Face files *files* (each ``rel``, ``url``,
    ``check_url``, ``size``, ``etag``) are about to download into *base* (a
    huggingface_hub local dir) and write the record. ``url`` is the address
    downloaded from; ``check_url`` names the file at the requested branch or
    revision, which :func:`check_upstream` reads. No effect outside
    :func:`tracking`."""
    act = _CURRENT.get()
    if act is None:
        return
    known = {(f.get("base"), f.get("rel")) for f in act.rec["files"]}
    for f in files:
        key = (str(base), f["rel"])
        if key in known:
            continue
        known.add(key)
        act.rec["files"].append({"base": str(base), "rel": f["rel"],
                                 "url": f.get("url"), "check_url": f.get("check_url"),
                                 "size": f.get("size") or None, "etag": f.get("etag")})
    _save_current(act)


def note_snapshot(folder: Path, name: str, created: bool, size: Optional[int]) -> None:
    """Record that a full-repo download into the models folder *folder*
    (registered as *name*) starts; *created* is True when this download
    created the folder. No effect outside :func:`tracking`."""
    act = _CURRENT.get()
    if act is None:
        return
    act.rec["snapshot"] = {"dir": str(folder), "name": name,
                           "created": bool(created), "size": size or None}
    _save_current(act)


def note_snapshot_complete(folder: Path, repo: str, revision: Optional[str]) -> None:
    """Record that the full-repo download of *repo* at *revision* (None: the
    default branch) in *folder* was already complete, so no transfer ran.
    Success of this pull then removes the records of that download. No effect
    outside :func:`tracking`."""
    act = _CURRENT.get()
    if act is None:
        return
    act.done_target = (repo.casefold(), revision or None,
                       os.path.normcase(os.path.abspath(str(folder))))


def note_part(part_file: Path, lock_name: str, size: Optional[int],
              etag: Optional[str] = None) -> None:
    """Record that a ``.part`` download into *part_file* (serialised by the
    part lock *lock_name*) starts. No effect outside :func:`tracking`."""
    act = _CURRENT.get()
    if act is None:
        return
    if any(p.get("path") == str(part_file) for p in act.rec["parts"]):
        return
    act.rec["parts"].append({"path": str(part_file), "lock": lock_name,
                             "size": size or None, "etag": etag})
    _save_current(act)


# ------------------------------------------------------------------ #
#  Reading, liveness, sizes                                            #
# ------------------------------------------------------------------ #

def _iter_records() -> Iterator[tuple[Path, dict]]:
    d = records_dir()
    try:
        paths = sorted(d.glob("*.json"))
    except OSError as e:
        logger.warning("could not list the download records in %s: %s", d, e)
        return
    for p in paths:
        rec = _read(p)
        if rec is not None:
            yield p, rec


def _is_running(rec: dict) -> bool:
    """True unless the pull of *rec* is proven not to be running: it ended,
    or its owner process is proven gone. Every uncertainty counts as
    running."""
    if rec.get("ended"):
        return False
    owner = rec.get("owner")
    if not isinstance(owner, dict):
        return True
    try:
        pid = int(owner.get("pid", -1))
    except (TypeError, ValueError):
        return True
    from .pull import _record_holder_is_gone, _same_pid_space
    if owner.get("space") is not None and not _same_pid_space(owner, records_dir()):
        return True
    if pid == os.getpid():
        with _LIVE_LOCK:
            return rec.get("id") in _LIVE_IDS
    return not _record_holder_is_gone(owner, records_dir())


class _HfPaths(NamedTuple):
    metadata_path: Path
    lock_path: Path


def _as_extended_path(path: Path, max_length: int) -> Path:
    """*path* in its Windows extended-length form when it is longer than
    *max_length*, as huggingface_hub names it; unchanged elsewhere."""
    if os.name != "nt":
        return path
    absolute = os.path.abspath(str(path))
    if len(absolute) <= max_length or absolute.startswith("\\\\?\\"):
        return path
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def _hf_layout(base: Path, rel: str) -> Optional[_HfPaths]:
    """The metadata and lock paths huggingface_hub uses for *rel* under the
    local dir *base*, or None when *rel* is not a plain relative repo path.
    Creates nothing on disk (huggingface_hub's own helper makes folders and
    writes marker files)."""
    from .pull import _REPO_PATH_PART_RE
    comps = rel.split("/")
    if not comps or any(c in ("", ".", "..") or not _REPO_PATH_PART_RE.match(c)
                        for c in comps):
        return None
    try:
        metadata = Path(base) / ".cache" / "huggingface" / "download" / (
            os.path.join(*comps) + ".metadata")
        return _HfPaths(_as_extended_path(metadata, 247),
                        _as_extended_path(metadata.with_suffix(".lock"), 247))
    except Exception as e:
        logger.debug("cannot compute the download paths of %s: %s", rel, e)
        return None


def _hf_temp_files(base: Path, rel: str, etag: Optional[str] = None) -> list[Path]:
    """The temp files of the downloads of *rel* under *base*: those of
    *etag* only, or of every etag when *etag* is None."""
    paths = _hf_layout(base, rel)
    if paths is None:
        return []
    from huggingface_hub._local_folder import _short_hash
    folder = paths.metadata_path.parent
    prefix = _short_hash(paths.metadata_path.name) + "."
    if etag:
        prefix += etag + "."
    try:
        return [f for f in folder.glob("*.incomplete")
                if f.name.startswith(prefix) and f.is_file()]
    except OSError:
        return []


def _size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _snapshot_bytes(folder: Path, created: bool) -> int:
    from .pull import _snapshot_bytes_on_disk
    if not folder.is_dir():
        return 0
    if created:
        return _snapshot_bytes_on_disk(folder)
    pending = folder / ".cache" / "huggingface" / "download"
    try:
        return sum(_size(f) for f in pending.rglob("*.incomplete"))
    except OSError:
        return 0


class _Located(NamedTuple):
    """The places a record names that its pull could have written to.

    ``files``: (resolved base folder, repo path, etag, base folder as
    recorded) of Hugging Face files;
    ``parts``: (``.part`` file, lock name); ``snapshot``: (resolved folder,
    model name, created by the pull) of a full-repo download or None;
    ``outside``: the recorded places that are not under the models folder or
    the record's own destination, which nothing ever reads or deletes."""
    files: list
    parts: list
    snapshot: Optional[tuple]
    outside: list


def _locate(rec: dict) -> _Located:
    """Check every path *rec* names against the folders its pull may use (see
    :func:`_allowed_bases`). Only a resolved path is looked at before it
    passes; a ``.part.json`` identity is read for a CivitAI partial whose
    folder is not an allowed base."""
    from .pull import _safe_models_filename
    allowed = _allowed_bases(rec)
    files: list = []
    parts: list = []
    outside: list = []
    for f in rec.get("files") or []:
        if not (isinstance(f, dict) and isinstance(f.get("base"), str)
                and isinstance(f.get("rel"), str)):
            continue
        base = _resolved(Path(f["base"]))
        if base is None or base not in allowed or _hf_layout(base, f["rel"]) is None:
            outside.append(f"{f['base']}/{f['rel']}")
            continue
        files.append((base, f["rel"], _etag(f), Path(f["base"])))
    for p in rec.get("parts") or []:
        if not (isinstance(p, dict) and isinstance(p.get("path"), str)
                and isinstance(p.get("lock"), str)):
            continue
        part, lock = Path(p["path"]), p["lock"]
        parent = _resolved(part.parent)
        if (part.name.endswith(".part") and part.name == lock + ".part"
                and _safe_models_filename(lock, part.parent) is not None
                and parent is not None
                and (parent in allowed or _is_civitai_dest(rec, parent))):
            parts.append((part, lock))
        else:
            outside.append(str(part))
    snapshot = None
    snap = rec.get("snapshot")
    if isinstance(snap, dict) and isinstance(snap.get("dir"), str):
        rfolder = _resolved(Path(snap["dir"]))
        models = _resolved(_mm.MODELS_DIR)
        name = snap.get("name")
        if (isinstance(name, str) and rfolder is not None and models is not None
                and rfolder.parent == models and rfolder.name.lower() == name.lower()):
            snapshot = (rfolder, name, bool(snap.get("created")))
        else:
            outside.append(snap["dir"])
    return _Located(files, parts, snapshot, outside)


def bytes_on_disk(rec: dict, located: Optional[_Located] = None) -> int:
    """Bytes of partial data the pull of *rec* left on disk, counting only the
    places :func:`_locate` accepts."""
    loc = located or _locate(rec)
    total = 0
    for base, rel, etag, _raw in loc.files:
        total += max((_size(t) for t in _hf_temp_files(base, rel, etag)), default=0)
    for part, _lock in loc.parts:
        total += _size(part)
    if loc.snapshot is not None:
        total += _snapshot_bytes(loc.snapshot[0], loc.snapshot[2])
    return total


def _expected_size(rec: dict) -> Optional[int]:
    sizes = [f.get("size") for f in rec.get("files") or [] if isinstance(f, dict)]
    sizes += [p.get("size") for p in rec.get("parts") or [] if isinstance(p, dict)]
    snap = rec.get("snapshot")
    if isinstance(snap, dict):
        sizes.append(snap.get("size"))
    known = [s for s in sizes if isinstance(s, int) and s > 0]
    return sum(known) if known else None


def _source_urls(rec: dict) -> list[str]:
    urls = [f.get("url") for f in rec.get("files") or [] if isinstance(f, dict)]
    if rec.get("url"):
        urls.append(rec["url"])
    repo = rec.get("repo")
    if rec.get("source") == "hf-repo" and isinstance(repo, str) and repo:
        from .pull import _HF_ENDPOINT
        rev = rec.get("revision")
        urls.append(f"{_HF_ENDPOINT}/{repo}" + (f"/tree/{rev}" if isinstance(rev, str) and rev else ""))
    elif rec.get("source") == "civitai" and isinstance(repo, str) and repo.startswith("civitai:"):
        from .sources import CIVITAI_API
        urls.append(f"{CIVITAI_API}/api/v1/model-versions/{repo[len('civitai:'):]}")
    out: list[str] = []
    for u in urls:
        if isinstance(u, str) and u and u not in out:
            out.append(u)
    return out


def resumable(rec: dict, located: Optional[_Located] = None) -> bool:
    """True when re-running the recorded pull continues from its partial
    data. Every pull source localm has resumes; a request without a remote
    spec does not, and neither does a record that names a place outside the
    folders its pull may use."""
    spec = str(rec["request"].get("spec") or "")
    return (bool(spec) and rec.get("source") in ("hf-file", "hf-repo", "url", "civitai")
            and not (located or _locate(rec)).outside)


def describe(rec: dict) -> dict:
    """The client view of *rec*. ``invalid`` is True when the record names a
    place outside the folders its pull may use; nothing there is read or
    counted."""
    req = rec["request"]
    loc = _locate(rec)
    return {
        "id": rec["id"],
        "spec": req.get("spec"),
        "name": req.get("name"),
        "options": {k: req.get(k) for k in _REQUEST_KEYS if k not in ("spec", "dest_dir")},
        "comfy_dest": bool(req.get("dest_dir")),
        "source": rec.get("source"),
        "repo": rec.get("repo"),
        "revision": rec.get("revision"),
        "commit": rec.get("commit"),
        "urls": _source_urls(rec),
        "files": [f.get("rel") for f in rec.get("files") or [] if isinstance(f, dict)],
        "bytes_on_disk": bytes_on_disk(rec, loc),
        "expected_size": _expected_size(rec),
        "sha256": rec.get("sha256"),
        "created": rec.get("created"),
        "ended_at": rec.get("ended_at"),
        "dismissed": bool(rec.get("dismissed")),
        "resumable": resumable(rec, loc),
        "invalid": bool(loc.outside),
    }


def list_interrupted(include_dismissed: bool = False) -> list[dict]:
    """Records of pulls that are not running, oldest first, as
    :func:`describe` views. Dismissed ones only with *include_dismissed*."""
    out = []
    for _p, rec in _iter_records():
        if _is_running(rec):
            continue
        view = describe(rec)
        if view["bytes_on_disk"] <= 0 and not view["invalid"]:
            logger.info("removing the record of %s: none of its partial data is "
                        "on disk any more", rec["request"].get("spec"))
            _remove(rec["id"])
            continue
        if rec.get("dismissed") and not include_dismissed:
            continue
        out.append(view)
    out.sort(key=lambda r: r.get("created") or 0)
    return out


def _load(pull_id: str) -> dict:
    if not isinstance(pull_id, str) or not _ID_RE.match(pull_id):
        raise UnknownPull(pull_id)
    rec = _read(_record_path(pull_id))
    if rec is None:
        raise UnknownPull(pull_id)
    return rec


def get_interrupted(pull_id: str) -> dict:
    """The record *pull_id* of a pull that is not running. Raises
    :class:`UnknownPull` when there is none and ``pull.PullInFlight`` when
    that pull is still running."""
    rec = _load(pull_id)
    if _is_running(rec):
        from .pull import PullInFlight
        raise PullInFlight("that download is still running")
    return rec


def set_dismissed(pull_id: str, dismissed: bool = True) -> dict:
    """Hide (or show again) the interrupted download *pull_id*; its files and
    record stay. Returns the updated :func:`describe` view."""
    with _WRITE_LOCK:
        rec = get_interrupted(pull_id)
        rec["dismissed"] = bool(dismissed)
        _write(rec)
    return describe(rec)


def resume_args(rec: dict) -> list[str]:
    """The ``localm pull`` arguments that repeat the recorded request."""
    req = rec["request"]
    args: list[str] = []
    if req.get("name"):
        args += ["--name", str(req["name"])]
    if req.get("mmproj"):
        args += ["--mmproj", str(req["mmproj"])]
    if req.get("sha256"):
        args += ["--sha256", str(req["sha256"])]
    if req.get("type") and req.get("type") != "auto":
        args += ["--type", str(req["type"])]
    if req.get("redownload"):
        args += ["--redownload"]
    if req.get("dest_dir"):
        args += ["--comfy-dest-dir", str(req["dest_dir"])]
    if req.get("register") is False:
        args += ["--no-register"]
    return args + ["--", str(req["spec"])]


# ------------------------------------------------------------------ #
#  Deleting                                                            #
# ------------------------------------------------------------------ #

def _allowed_bases(rec: dict) -> set[Path]:
    from .pull import _REPO_FOLDER_STAGING
    bases = {_mm.MODELS_DIR, _mm.MODELS_DIR / _REPO_FOLDER_STAGING}
    dest = rec["request"].get("dest_dir")
    if isinstance(dest, str) and dest:
        bases |= {Path(dest), Path(dest) / _REPO_FOLDER_STAGING}
    out = set()
    for b in bases:
        try:
            out.add(b.resolve())
        except OSError:
            continue
    return out


def _resolved(p: Path) -> Optional[Path]:
    try:
        return p.resolve()
    except OSError:
        return None


def _snapshot_is_registered(folder: Path) -> bool:
    """True when a registry entry's path is *folder* or inside it, or when
    the registry cannot be read."""
    try:
        reg = _mm.load_registry()
    except Exception as e:
        logger.warning("cannot read the registry to check %s: %s", folder, e)
        return True
    target = _resolved(folder)
    for entry in reg.values():
        p = _mm._entry_path(entry) if isinstance(entry, dict) else None
        if p is None:
            continue
        rp = _resolved(Path(p))
        if rp is not None and target is not None and (rp == target or target in rp.parents):
            return True
    return False


class DeleteFailed(OSError):
    """Partial data that should have been deleted is still on disk."""


class DownloadComplete(Exception):
    """The download is finished (only unregistered), so deleting it would
    remove a complete model."""


_ETAG_RE = re.compile(r"\A[A-Za-z0-9._-]{1,200}\Z")


def _etag(f: dict) -> Optional[str]:
    """The recorded etag of the file entry *f* when it can name a temp file,
    else None."""
    e = f.get("etag")
    return e if isinstance(e, str) and _ETAG_RE.match(e) else None


def _unlink_checked(path: Path, left: list[Path]) -> int:
    """Delete *path* (and an owner sidecar beside it); returns its size, or 0
    with *path* appended to *left* when it is still there afterwards."""
    from .pull import _partial_owner_path
    size = _size(path)
    try:
        path.unlink(missing_ok=True)
    except OSError as e:
        logger.warning("could not delete the partial download %s: %s", path, e)
    if path.exists():
        left.append(path)
        return 0
    try:
        _partial_owner_path(path).unlink(missing_ok=True)
    except OSError as e:
        logger.warning("could not delete %s: %s", _partial_owner_path(path), e)
    return size


def _snapshot_looks_complete(folder: Path, expected: Optional[int]) -> bool:
    """True when the full-repo download in *folder* has no temp file left and
    holds at least *expected* bytes, so it is a finished download and not a
    partial one. With *expected* unknown, any finished file and no temp file
    counts as finished, so a folder is never taken for partial on missing
    information."""
    from .pull import _snapshot_bytes_on_disk
    if not folder.is_dir():
        return False
    pending = folder / ".cache" / "huggingface" / "download"
    try:
        if any(pending.rglob("*.incomplete")):
            return False
    except OSError:
        return False
    have = _snapshot_bytes_on_disk(folder)
    if not isinstance(expected, int) or expected <= 0:
        return have > 0
    return have >= expected


def _delete_partials(rec: dict) -> tuple[int, list[Path]]:
    """Delete the partial data of *rec* under the locks its pull takes, only
    in the places :func:`_locate` accepts.
    Raises ``pull.PullInFlight`` when a download holds one of those locks and
    :class:`DownloadComplete` when the folder it would remove holds a finished
    download. Returns ``(bytes freed, files that could not be deleted)``."""
    from .pull import (_SNAPSHOT_LOCK_WAIT_SECONDS, _delete_hf_temp_files,
                       _hold_download_lock, _part_lock, _part_record_path,
                       _snapshot_bytes_on_disk, _snapshot_dest_lock)
    freed = 0
    left: list[Path] = []
    loc = _locate(rec)
    for place in loc.outside:
        logger.warning("not deleting partial data outside the folders of its download: %s",
                       place)
    for base, rel, etag, _raw in loc.files:
        paths = _hf_layout(base, rel)
        if paths is None or not paths.lock_path.parent.is_dir():
            continue
        with _hold_download_lock(paths.lock_path, wait=_SNAPSHOT_LOCK_WAIT_SECONDS):
            for t in _hf_temp_files(base, rel, etag):
                freed += _unlink_checked(t, left)

    for part, lock in loc.parts:
        with _part_lock(lock):
            try:
                _part_record_path(part).unlink(missing_ok=True)
            except OSError as e:
                logger.warning("could not delete %s: %s", _part_record_path(part), e)
            freed += _unlink_checked(part, left)

    if loc.snapshot is not None:
        folder, name, created = loc.snapshot
        models = _resolved(_mm.MODELS_DIR)
        with _snapshot_dest_lock(name):
            current = _resolved(folder)
            if (current != folder or models is None or current.parent != models
                    or current.name.lower() != name.lower()):
                logger.warning("not deleting %s: it is no longer a folder in the "
                               "models folder", folder)
            elif created and folder.is_dir() and not _snapshot_is_registered(folder):
                if _snapshot_looks_complete(folder, (rec.get("snapshot") or {}).get("size")):
                    raise DownloadComplete(
                        f"{name} is a complete download that is not registered; "
                        "resume it to register it, or dismiss it. Its files were not deleted.")
                size = _snapshot_bytes_on_disk(folder)
                shutil.rmtree(folder)
                freed += size
            elif folder.is_dir():
                freed += _delete_hf_temp_files(folder, None)
    return freed, left


def _is_civitai_dest(rec: dict, parent: Path) -> bool:
    """True when *parent* is the folder the CivitAI pull of *rec* wrote its
    ``.part`` into, proven by the ``.part.json`` identity of a CivitAI
    partial there."""
    if rec.get("source") != "civitai":
        return False
    for p in rec.get("parts") or []:
        if not (isinstance(p, dict) and isinstance(p.get("path"), str)):
            continue
        part = Path(p["path"])
        if _resolved(part.parent) != parent:
            continue
        try:
            ident = json.loads(part.with_name(part.name + ".json").read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            return False
        return isinstance(ident, dict) and ident.get("source") == "civitai"
    return False


def delete_interrupted(pull_id: str) -> int:
    """Delete the partial data of the interrupted download *pull_id* and its
    record. Returns the bytes freed. Raises :class:`UnknownPull`;
    ``pull.PullInFlight`` when it or another download of the same files is
    running; :class:`DeleteFailed` when a file could not be deleted;
    :class:`DownloadComplete` when the download is finished. The record is
    kept whenever one of these is raised."""
    rec = get_interrupted(pull_id)
    freed, left = _delete_partials(rec)
    if left:
        raise DeleteFailed("could not delete " + ", ".join(str(p) for p in left)
                           + f" ({freed} bytes of other files were deleted)")
    _remove(pull_id)
    return freed


def forget(pull_id: str) -> None:
    """Remove the record *pull_id* without touching any file."""
    if isinstance(pull_id, str) and _ID_RE.match(pull_id):
        _remove(pull_id)


def records_for_request(spec: str, name: Optional[str]) -> list[dict]:
    """Interrupted records whose spec and name equal *spec* and *name*."""
    return [r for _p, r in _iter_records()
            if not _is_running(r) and r["request"].get("spec") == spec
            and (r["request"].get("name") or None) == (name or None)]


# ------------------------------------------------------------------ #
#  Upstream check before a resume                                       #
# ------------------------------------------------------------------ #

def check_upstream(rec: dict) -> dict:
    """Compare what the source serves now with what the record holds.

    Returns ``{"changed": True|False|None, "detail": str}``: True when the
    partial data cannot be continued safely (a recorded file's etag or size,
    a direct URL's size or etag, or a full repo's commit differs now, or a
    direct URL with no expected digest has neither size nor etag to compare),
    None when the check could not be made. Raises ``netpolicy.NetworkPolicyError``
    when the network policy refuses the request."""
    from localm import netpolicy
    from .pull import _etag_from_head, _policy_head
    changed: list[str] = []
    unchecked: list[str] = []
    for f in rec.get("files") or []:
        url, etag, size = f.get("check_url") or f.get("url"), f.get("etag"), f.get("size")
        if not url:
            unchecked.append(str(f.get("rel")))
            continue
        try:
            head = _policy_head(url)
        except netpolicy.NetworkPolicyError:
            raise
        except Exception as e:
            logger.debug("upstream HEAD of %s failed: %s", url, e)
            unchecked.append(str(f.get("rel")))
            continue
        now_etag = _etag_from_head(head)
        try:
            now_size = int(head.headers.get("content-length", 0)) or None
        except (TypeError, ValueError):
            now_size = None
        if (etag and now_etag and etag != now_etag) or (size and now_size and size != now_size):
            changed.append(f"{f.get('rel')} changed at the source")
    if rec.get("source") == "url" and rec.get("url"):
        part = (rec.get("parts") or [{}])[0]
        try:
            from .pull import _ssrf_resolve_final_url
            final = _ssrf_resolve_final_url(rec["url"])
            head = netpolicy.pinned_request("HEAD", final, allow_redirects=False, timeout=10)
            now_size = int(head.headers.get("content-length", 0)) or None
            now_etag = head.headers.get("etag")
            if ((part.get("size") and now_size and part["size"] != now_size)
                    or (part.get("etag") and now_etag and part["etag"] != now_etag)):
                changed.append(f"{rec['url']} changed at the source")
            elif not ((part.get("size") and now_size) or (part.get("etag") and now_etag)):
                target = changed if not rec.get("sha256") else unchecked
                target.append(rec["url"] + " cannot be compared (the server reports no "
                              "size or etag)" if target is changed else rec["url"])
        except netpolicy.NetworkPolicyError:
            raise
        except Exception as e:
            logger.debug("upstream HEAD of %s failed: %s", rec["url"], e)
            unchecked.append(rec["url"])
    snap = rec.get("snapshot")
    if rec.get("source") == "hf-repo" and isinstance(snap, dict) and rec.get("commit"):
        from .pull import _hf_endpoint_refusal
        refusal = _hf_endpoint_refusal()
        if refusal:
            logger.info("not checking the commit of %s: %s", rec.get("repo"), refusal)
            unchecked.append(f"{rec.get('repo')} (network policy: {refusal})")
        else:
            try:
                from .pull import _hf_api
                kw = {"revision": rec["revision"]} if rec.get("revision") else {}
                now = getattr(_hf_api().model_info(rec["repo"], **kw), "sha", None)
                if now and now != rec["commit"]:
                    changed.append(f"{rec['repo']} changed at the source (now at commit {now[:12]})")
            except netpolicy.NetworkPolicyError:
                raise
            except Exception as e:
                logger.debug("upstream commit check of %s failed: %s", rec.get("repo"), e)
                unchecked.append(str(rec.get("repo")))
    if changed:
        return {"changed": True,
                "detail": "cannot be continued: " + "; ".join(changed)}
    if unchecked:
        return {"changed": None,
                "detail": "could not check the source of: " + ", ".join(unchecked)}
    return {"changed": False, "detail": ""}


# ------------------------------------------------------------------ #
#  Partial files no record names                                       #
# ------------------------------------------------------------------ #

def _orphan_id(path: Path) -> str:
    return "orphan-" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:24]


def _claimed_paths() -> tuple[set[Path], set[Path]]:
    """Files and folders named by any record (running or not)."""
    files: set[Path] = set()
    folders: set[Path] = set()
    for _p, rec in _iter_records():
        loc = _locate(rec)
        for _base, rel, etag, raw in loc.files:
            files.update(_hf_temp_files(raw, rel, etag))
        for p in rec.get("parts") or []:
            if isinstance(p, dict) and isinstance(p.get("path"), str):
                files.add(Path(p["path"]))
        snap = rec.get("snapshot")
        if isinstance(snap, dict) and isinstance(snap.get("dir"), str):
            folders.add(Path(snap["dir"]))
        if loc.snapshot is not None:
            folders.add(loc.snapshot[0])
    return files, folders


def _lock_for_temp(f: Path) -> Optional[Path]:
    """huggingface_hub's download lock for the temp file *f*, found by
    matching its short-hash prefix against the lock files beside it."""
    from huggingface_hub._local_folder import _short_hash
    prefix = f.name.split(".", 1)[0]
    try:
        for lock in f.parent.glob("*.lock"):
            if _short_hash(lock.name[:-len(".lock")] + ".metadata") == prefix:
                return lock
    except OSError:
        return None
    return None


def _temp_is_unowned(f: Path) -> bool:
    """True when nothing is proven to be writing the temp file *f*."""
    from .pull import _lock_file_is_free, _partial_owner_is_gone, _read_partial_owner
    if _read_partial_owner(f) is not None:
        return _partial_owner_is_gone(f, lock_held=False)
    lock = _lock_for_temp(f)
    return lock is not None and _lock_file_is_free(lock)


def _temp_roots() -> list[Path]:
    from .pull import _REPO_FOLDER_STAGING
    roots = [_mm.MODELS_DIR, _mm.MODELS_DIR / _REPO_FOLDER_STAGING]
    try:
        roots += [d for d in _mm.MODELS_DIR.iterdir()
                  if d.is_dir() and not d.name.startswith(".")]
    except OSError:
        pass
    return [r / ".cache" / "huggingface" / "download" for r in roots]


def _is_url_partial(p: Path) -> bool:
    """True when the ``.part.json`` beside *p* records a direct-URL or CivitAI
    download."""
    try:
        ident = json.loads(p.with_name(p.name + ".json").read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return False
    return isinstance(ident, dict) and ident.get("source") in ("url", "civitai")


def _scan_orphans() -> list[dict]:
    files, folders = _claimed_paths()
    out: list[dict] = []
    for root in _temp_roots():
        if not root.is_dir() or any(f in root.parents for f in folders):
            continue
        try:
            temps = [f for f in root.rglob("*.incomplete") if f.is_file()]
        except OSError:
            continue
        for f in temps:
            if f in files or not _temp_is_unowned(f):
                continue
            out.append({"id": _orphan_id(f), "path": f, "kind": "hf-temp",
                        "size": _size(f)})
    from .pull import _lock_is_reclaimable, _part_lock_dir
    try:
        parts = [p for p in _mm.MODELS_DIR.glob("*.part") if p.is_file()]
    except OSError:
        parts = []
    for p in parts:
        lock_dir = _part_lock_dir(p.name[:-len(".part")])
        if (p.name.startswith(".") or p in files or not _is_url_partial(p)
                or (lock_dir.exists() and not _lock_is_reclaimable(lock_dir))):
            continue
        out.append({"id": _orphan_id(p), "path": p, "kind": "part", "size": _size(p)})
    return out


def list_orphans() -> list[dict]:
    """Partial download files no record names and no running download is
    writing, as ``{"id", "file", "size"}`` (file relative to the models
    folder)."""
    out = []
    for o in _scan_orphans():
        try:
            rel = o["path"].relative_to(_mm.MODELS_DIR).as_posix()
        except ValueError:
            rel = o["path"].name
        out.append({"id": o["id"], "file": rel, "size": o["size"]})
    return out


def delete_orphan(orphan_id: str) -> int:
    """Delete the orphan partial *orphan_id* (found by a fresh scan, never by
    a path from the caller). Returns the bytes freed. Raises
    :class:`UnknownPull` when no such orphan exists now and
    ``pull.PullInFlight`` when a download took it over."""
    from .pull import (PullInFlight, _hold_download_lock, _part_lock,
                       _partial_owner_is_gone, _read_partial_owner, _unlink_quiet)
    if not isinstance(orphan_id, str) or not _ORPHAN_ID_RE.match(orphan_id):
        raise UnknownPull(orphan_id)
    match = next((o for o in _scan_orphans() if o["id"] == orphan_id), None)
    if match is None:
        raise UnknownPull(orphan_id)
    path: Path = match["path"]
    left: list[Path] = []
    if match["kind"] == "part":
        from .pull import _part_record_path
        with _part_lock(path.name[:-len(".part")]):
            size = _unlink_checked(path, left)
            if not left:
                _unlink_quiet(_part_record_path(path))
        if left:
            raise DeleteFailed(f"could not delete {path}")
        return size
    lock = _lock_for_temp(path)
    with contextlib.ExitStack() as held:
        if lock is not None:
            held.enter_context(_hold_download_lock(lock))
        if _read_partial_owner(path) is not None and not _partial_owner_is_gone(
                path, lock_held=lock is not None):
            raise PullInFlight("a download is writing that file")
        if lock is None and _read_partial_owner(path) is None:
            raise PullInFlight("cannot tell whether a download is writing that file")
        size = _unlink_checked(path, left)
    if left:
        raise DeleteFailed(f"could not delete {path}")
    return size
