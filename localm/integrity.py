# SPDX-License-Identifier: AGPL-3.0-or-later
"""Whether localm's files are the files of the installed release, and the repair
that puts the differing ones back from the signed release.

The installed release's statement (``update-package.json`` at the install root,
:mod:`localm.update_package`) lists every file of localm with its sha256 and size.
:func:`check` compares the install with it: a file of another size differs, a file
of the stated size is hashed in full unless the record of the last check
(``<data dir>/updates/integrity-record.json``) holds it with the same size and
modification time, and then the recorded checksum stands for it. Files the user
chose to keep (:func:`keep`) are listed apart for as long as the release states
the same content for them. Files the statement does not name are not reported.

The native runtime in ``runtime/localm_llama_runtime/lib`` is not part of the
release. When it is installed, :func:`write_runtime_list` writes the list of its
files with size, modification time and sha256 beside them (``runtime_files``);
the check compares size and time, and hashes a file only when its time changed or a
full check is asked for.

:func:`startup` runs the check in a background thread of a server, after
:data:`START_DELAY_S`, when the setting ``integrity_check_at_start`` is on, and
reads nothing at all when it is off. ``integrity_notice`` decides whether a
difference it finds is announced. The result is kept in memory (:func:`state`); the
record of the last check is written at startup only where automatic writes are
allowed (``audit.automatic_write_allowed``), and on every check a person asks for.

:func:`repair` downloads the installed version's build through the updater, checks
its signature, and writes exactly the named files from it.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

RECORD = "integrity-record.json"
RECORD_FORMAT = "localm-integrity-record/1"
KEPT = "integrity-kept.json"
# Seconds a server waits after its start before the startup check begins.
START_DELAY_S = 30.0
# The commands whose start runs the check (a server).
SERVER_COMMANDS = frozenset({"gui", "serve"})
# Seconds a repair keeps trying to replace a file another program holds.
_HELD_S = 3.0

_lock = threading.Lock()
_last: Optional[dict] = None
_pending = False
_running = False


class IntegrityError(Exception):
    """A check or repair that cannot be carried out, with the reason."""


# ------------------------------------------------------------------ helpers

def _root(root) -> Path:
    if root is not None:
        return Path(root)
    from localm import updater
    return updater.repo_root()


def _updates(create: bool) -> Path:
    from localm import updater
    return updater._updates_dir(create=create)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    from localm import runtime_files
    return runtime_files.sha256(path)


def _path(root: Path, rel: str) -> Path:
    return root.joinpath(*rel.split("/"))


def _skipped(rel: str) -> bool:
    """Whether *rel* lies where an update never compares files (``NEVER_TOUCH``,
    ``PRESERVE_WITHIN``)."""
    from localm import _apply_update as au
    if rel.split("/", 1)[0] in au.NEVER_TOUCH:
        return True
    return any(rel == p or rel.startswith(p + "/") for p in au.PRESERVE_WITHIN)


def statement(root=None) -> tuple:
    """``(statement, reason)`` of the install at *root*: the parsed
    ``update-package.json`` and None, or None and why there is nothing to compare
    with (no statement, one that cannot be read, one for another version)."""
    from localm import update_package
    root = _root(root)
    path = root / update_package.NAME
    if not path.is_file():
        return None, ("this install has no list of its release's files (a git clone, "
                      "or an install of a version older than 0.3.4b3)")
    try:
        manifest = update_package.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, update_package.PackageError) as e:
        return None, f"the list of the release's files cannot be read: {e}"
    try:
        version = (root / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        version = None
    if version is not None and version != manifest["version"]:
        return None, (f"the list of the release's files is for {manifest['version'][:40]}, "
                      f"but this install is {version[:40]}")
    return manifest, None


def kept() -> dict:
    """``{path: stated sha256}`` of the files the user keeps as their own."""
    data = _read_json(_updates(False) / KEPT)
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}


def _record(version: str) -> dict:
    data = _read_json(_updates(False) / RECORD)
    if not (isinstance(data, dict) and data.get("format") == RECORD_FORMAT
            and data.get("version") == version and isinstance(data.get("files"), dict)):
        return {}
    return {k: v for k, v in data["files"].items()
            if isinstance(v, list) and len(v) == 3}


# ------------------------------------------------------------------- runtime

def write_runtime_list(target) -> int:
    """Write the list of the runtime files in *target* (``runtime_files.write``) and
    return how many it lists. Raises OSError."""
    from localm import runtime_files
    return runtime_files.write(target)


def _check_runtime(root: Path, full: bool) -> dict:
    from localm import _apply_update as au
    from localm import runtime_files
    return runtime_files.compare(_path(root, au.RUNTIME_LIB), full=full)


# --------------------------------------------------------------------- check

def check(root=None, *, full: bool = False, write_record: bool = True) -> dict:
    """Compare the install at *root* (default the running one) with its release.

    Returns ``{"available", "reason", "version", "checked", "differ", "missing",
    "unreadable", "kept", "runtime", "count", "hashed", "full", "at"}``:
    ``available`` False with a ``reason`` when there is nothing to compare with;
    ``differ`` and ``missing`` the stated paths whose file is not the stated one or
    is absent, ``unreadable`` those that could not be read, ``kept`` the differing
    ones the user keeps (:func:`keep`), ``runtime`` the runtime's result
    (:func:`_check_runtime`), ``count`` how many files differ or are missing, kept
    ones aside, ``hashed`` how many release files were read in full. With *full* the
    record of the last check is not used. With *write_record* the identities found
    are recorded for the next check. Never raises for a file it cannot read."""
    root = _root(root)
    result = {"available": False, "reason": None, "version": None, "checked": 0,
              "differ": [], "missing": [], "unreadable": [], "kept": [], "hashed": 0,
              "runtime": None, "count": 0, "full": full, "at": time.time()}
    manifest, reason = statement(root)
    if manifest is None:
        result["reason"] = reason
        return result
    version = manifest["version"]
    result.update(available=True, version=version)
    before = {} if full else _record(version)
    mine = kept()
    found = {}
    for rel, entry in sorted(manifest["files"].items()):
        if _skipped(rel):
            continue
        result["checked"] += 1
        p = _path(root, rel)
        try:
            st = os.stat(p)
        except (FileNotFoundError, NotADirectoryError):
            result["missing"].append(rel)
            continue
        except OSError:
            result["unreadable"].append(rel)
            continue
        sha = None
        if st.st_size == entry["size"]:
            was = before.get(rel)
            if was and was[0] == st.st_size and was[1] == st.st_mtime_ns:
                sha = was[2]
            else:
                try:
                    sha = _sha256(p)
                except OSError:
                    result["unreadable"].append(rel)
                    continue
                result["hashed"] += 1
        found[rel] = [st.st_size, st.st_mtime_ns, sha]
        if sha != entry["sha256"]:
            (result["kept"] if mine.get(rel) == entry["sha256"] else result["differ"]).append(rel)
    result["runtime"] = _check_runtime(root, full)
    rt = result["runtime"]
    result["count"] = (len(result["differ"]) + len(result["missing"])
                       + len(rt["differ"]) + len(rt["missing"]))
    if write_record:
        try:
            _write_json(_updates(True) / RECORD,
                        {"format": RECORD_FORMAT, "version": version, "files": found})
        except OSError as e:
            from localm.debuglog import logger
            logger.warning("integrity: could not write the record of the check: %s", e)
    return result


def lines(result: dict) -> list:
    """One line per finding of a :func:`check` result, for a person."""
    if not result.get("available"):
        return [result.get("reason") or "nothing to compare with"]
    out = [f"differs from the release: {p}" for p in result["differ"]]
    out += [f"missing: {p}" for p in result["missing"]]
    out += [f"could not be read: {p}" for p in result["unreadable"]]
    rt = result.get("runtime") or {}
    out += [f"native runtime file differs: {p}" for p in rt.get("differ", [])]
    out += [f"native runtime file missing: {p}" for p in rt.get("missing", [])]
    return out


def summary(result: Optional[dict]) -> str:
    """The one-line outcome of a :func:`check` result."""
    if result is None:
        return "not checked"
    if not result.get("available"):
        return f"not checked: {result.get('reason')}"
    n = result["count"]
    if not n:
        tail = f"; {len(result['kept'])} kept as yours" if result["kept"] else ""
        return f"every file matches localm {result['version']}{tail}"
    return f"{n} file(s) differ from localm {result['version']}"


# ---------------------------------------------------------------- keep, state

def keep(paths, root=None) -> list:
    """Record the differing files *paths* as the user's own, so the check lists them
    apart while the release states the same content for them. Returns the paths
    kept. Raises IntegrityError for a path the release does not state."""
    manifest, reason = statement(root)
    if manifest is None:
        raise IntegrityError(reason)
    mine = kept()
    done = []
    for rel in paths:
        entry = manifest["files"].get(rel)
        if entry is None:
            raise IntegrityError(f"{rel} is not a file of the release")
        mine[rel] = entry["sha256"]
        done.append(rel)
    _write_json(_updates(True) / KEPT, mine)
    return done


def _set(result: Optional[dict] = None, *, pending=None, running=None) -> None:
    global _last, _pending, _running
    with _lock:
        if result is not None:
            _last = result
            from localm.bugreport import diagnostics
            diagnostics.note_integrity(_counts(result))
        if pending is not None:
            _pending = pending
        if running is not None:
            _running = running


def run(*, full: bool = False, root=None, write_record: bool = True) -> dict:
    """:func:`check`, with its result kept for :func:`state`."""
    _set(running=True)
    try:
        result = check(root, full=full, write_record=write_record)
        _set(result)
        return result
    finally:
        _set(running=False, pending=False)


def _settings() -> tuple:
    from localm.config import load_config
    cfg = load_config()
    return (cfg.get("integrity_check_at_start", True) is not False,
            cfg.get("integrity_notice", True) is not False)


def state() -> dict:
    """``{"at_start", "notice", "pending", "running", "result"}``: the two settings,
    whether the startup check is still to run or running, and the last result in
    this process (None when no check ran)."""
    at_start, notice = _settings()
    with _lock:
        return {"at_start": at_start, "notice": notice, "pending": _pending,
                "running": _running, "result": _last}


def _counts(r: dict) -> dict:
    if not r.get("available"):
        return {"checked": False}
    rt = r.get("runtime") or {}
    return {"checked": True, "differ": len(r["differ"]), "missing": len(r["missing"]),
            "kept": len(r["kept"]),
            "runtime_differ": len(rt.get("differ", [])) + len(rt.get("missing", []))}


def startup(command, *, say: Optional[Callable] = None,
            delay: float = START_DELAY_S) -> Optional[threading.Thread]:
    """Start the startup check of a server (*command* in :data:`SERVER_COMMANDS`) in a
    daemon thread that waits *delay* seconds first, and return the thread. None, with
    nothing read, for another command or when ``integrity_check_at_start`` is off.
    With ``integrity_notice`` on, a difference found is passed to *say* as one
    line."""
    if command not in SERVER_COMMANDS:
        return None
    try:
        at_start, notice = _settings()
    except Exception as e:
        from localm.debuglog import logger
        logger.warning("integrity: the settings could not be read, so no check ran: %s", e)
        return None
    if not at_start:
        return None
    _set(pending=True)

    def work():
        time.sleep(delay)
        try:
            from localm.audit import automatic_write_allowed
            result = run(write_record=automatic_write_allowed("server"))
        except Exception as e:
            from localm.debuglog import logger
            logger.warning("integrity: the check at startup failed: %s", e)
            return
        from localm.debuglog import logger
        logger.info("integrity: %s (%d of %d read in full)", summary(result),
                    result["hashed"], result["checked"])
        if result["count"] and notice and say is not None:
            say(f"{summary(result)}. 'localm integrity' lists them; "
                "'localm integrity --repair' puts them back.")

    thread = threading.Thread(target=work, name="localm-integrity", daemon=True)
    thread.start()
    return thread


# -------------------------------------------------------------------- repair

def _replace(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".localm-repair")
    tmp.write_bytes(data)
    deadline = time.monotonic() + _HELD_S
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.2)


def repair(paths=None, *, root=None, say: Optional[Callable] = None,
           opener=None) -> dict:
    """Write the release files *paths* (default every differing and missing one, kept
    ones aside) again from the signed build of the installed version: the build is
    downloaded through the updater, its signature checked, and each file is taken
    from it only when the build's own statement gives it the content the installed
    statement does. A repaired file stops being kept. Returns ``{"version",
    "repaired", "failed"}`` (``failed``: ``[{"path", "reason"}]``). Raises
    IntegrityError when nothing can be repaired (no statement, a path the release
    does not state, an update in progress) and LocalmError when the build cannot be
    fetched or does not verify. *opener* is the updater's download opener."""
    from localm import update_package, updater
    from localm.bugreport import LocalmError
    from localm.update_mechanism import job
    say = say or (lambda _m: None)
    root = _root(root)
    manifest, reason = statement(root)
    if manifest is None:
        raise IntegrityError(reason)
    version = manifest["version"]
    if paths is None:
        found = check(root)
        paths = found["differ"] + found["missing"]
    paths = list(dict.fromkeys(paths))
    for rel in paths:
        if rel not in manifest["files"] or _skipped(rel):
            raise IntegrityError(f"{rel} is not a file of the release")
    if not paths:
        return {"version": version, "repaired": [], "failed": []}
    if job.record_path(root).exists() or job.locked(root):
        raise IntegrityError("an update of this install is in progress; repair it once "
                             "the update has finished")
    scratch = _updates(True) / "integrity-repair"
    with updater._apply_lock("repair"):
        say(f"Fetching the signed build of localm {version}...")
        meta = updater.fetch_release(version)
        asset = meta.get("asset") or {}
        if not asset.get("id"):
            raise LocalmError(f"the release {version} has no build to repair from",
                              reason="unzip that release over the install folder instead")
        try:
            scratch.mkdir(parents=True, exist_ok=True)
            zip_path = scratch / "build.zip"
            updater.download(asset["id"], zip_path, opener=opener)
            data = zip_path.read_bytes()
            updater.verify_signature(data, meta.get("signature"))
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                names = set(z.namelist())
                if update_package.NAME not in names:
                    raise LocalmError(f"the build of {version} lists no files",
                                      reason="it cannot tell which content is right")
                signed = update_package.parse(z.read(update_package.NAME).decode("utf-8"))
                if signed["version"] != version:
                    raise LocalmError(f"the update server sent {signed['version'][:40]} "
                                      f"for {version}", reason="nothing was repaired")
                repaired, failed = [], []
                for rel in paths:
                    entry = signed["files"].get(rel)
                    if entry != manifest["files"][rel] or rel not in names:
                        failed.append({"path": rel, "reason": "the signed build states "
                                       "other content for it than this install's list"})
                        continue
                    content = z.read(rel)
                    if (hashlib.sha256(content).hexdigest() != entry["sha256"]
                            or len(content) != entry["size"]):
                        failed.append({"path": rel, "reason": "the build's copy differs "
                                       "from its own list"})
                        continue
                    try:
                        _replace(_path(root, rel), content)
                    except OSError as e:
                        failed.append({"path": rel, "reason": f"could not be written: {e}"})
                        continue
                    repaired.append(rel)
                    say(f"Repaired {rel}")
        finally:
            import shutil
            try:
                shutil.rmtree(scratch)
            except OSError as e:
                from localm.debuglog import logger
                logger.warning("integrity: the downloaded build in %s was not removed: %s",
                               scratch, e)
    mine = kept()
    if any(rel in mine for rel in repaired):
        for rel in repaired:
            mine.pop(rel, None)
        _write_json(_updates(True) / KEPT, mine)
    return {"version": version, "repaired": repaired, "failed": failed}
