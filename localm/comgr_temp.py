# SPDX-License-Identifier: AGPL-3.0-or-later
"""Removes the empty temp folders AMD's HIP runtime leaves behind.

The kernel compiler of the HIP runtime (comgr) makes ``comgr-<pid>-...`` folders
in the temp folder of the process that uses it and does not remove them when
that process is ended from outside, as localm ends its GPU workers.

A process that is about to load a HIP runtime calls :func:`register`: it records
its pid in ``<data dir>/run/comgr-temp/``. The localm process that starts workers
(:func:`watch`) removes the folders of each recorded process once that process is
gone, and :func:`finish` does it for the last ones before a localm process ends.
Records a killed localm left are handled at the next start.

A folder is removed only when it is in this process's temp folder, which every
localm process of one data folder shares, its name carries a recorded pid, no
process has that pid, the folder was last changed between the record's start and
the time the process was last seen (the record's own change time), and it holds
no file; nothing is followed through a link."""

from __future__ import annotations

import importlib
import json
import logging
import os
import secrets
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from localm import procid

logger = logging.getLogger("localm")

_POLL_SECONDS = 2.0
_IDLE_POLLS = 150
_FINISH_WAIT_SECONDS = 3.0
_SEEN_MARGIN_SECONDS = 30.0
_FOREIGN_RECORD_SECONDS = 86400.0

# The process that starts the GPU workers, handed to them by environment.
_STARTER_VAR = "LOCALM_COMGR_STARTER"

# A record is data from a folder anything may write to: what is larger, or not of
# this shape, is removed unread.
_MAX_RECORD_BYTES = 4096

_lock = threading.Lock()
_watcher: Optional[threading.Thread] = None
_registered_pid: Optional[int] = None
_exit_hooked = False


_home: Optional[Callable[[], Path]] = None


def set_home(home: Callable[[], Path]) -> None:
    """Set by ``localm.config``: the callable that gives the data folder."""
    global _home
    _home = home


def records_dir() -> Path:
    if _home is None:
        raise OSError("the data folder is not known yet")
    return _home() / "run" / "comgr-temp"


def _temp_folder() -> str:
    """The folder native code of this process gets as its temp folder."""
    names = (("TMP", "TEMP", "USERPROFILE") if sys.platform == "win32"
             else ("TMPDIR", "TMP", "TEMP", "TEMPDIR"))
    return next((os.path.abspath(os.environ[n]) for n in names if os.environ.get(n)), "/tmp")


def _record(pid: int, since: float, starter: str) -> None:
    d = records_dir()
    d.mkdir(parents=True, exist_ok=True)
    data = {"pid": pid, "since": since, "start": procid.process_start_identity(pid),
            "space": procid.pid_space_id(), "starter": starter}
    tmp = d / f".{pid}-{secrets.token_hex(4)}.tmp"
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, d / f"{pid}-{secrets.token_hex(4)}.json")


def register() -> None:
    """Record this process as one that loads a HIP runtime, once. Never raises: a
    record that cannot be written leaves its empty folders behind."""
    global _registered_pid
    if _registered_pid == os.getpid():
        return
    _registered_pid = os.getpid()
    try:
        if _home is None:
            importlib.import_module("localm.config")
        _record(os.getpid(), time.time(), os.environ.get(_STARTER_VAR, ""))
    except Exception as e:
        logger.debug("comgr temp: could not record process %s (%s)", os.getpid(), e)


def register_if_hip(runtime_dir) -> None:
    """:func:`register` when the native runtime in *runtime_dir* uses HIP."""
    try:
        d = Path(runtime_dir)
        hip = "rocm" in d.name.lower() or any(
            "hip" in p.name.lower() for p in d.iterdir())
    except OSError:
        hip = False
    if hip:
        register()


def adopt_tree(pid: int) -> None:
    """Record the running process *pid* and its descendants (a ComfyUI localm
    started and is about to end). Never raises."""
    try:
        import psutil
        root = psutil.Process(pid)
        for p in [root, *root.children(recursive=True)]:
            _record(p.pid, p.create_time(), str(os.getpid()))
    except Exception as e:
        logger.debug("comgr temp: could not record the processes of %s (%s)", pid, e)


def _read(path: Path) -> Optional[dict]:
    """The record in *path*, or None for anything that is not one: not a plain
    file, too large, not JSON, or a pid or start of the wrong type or range."""
    try:
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode) or st.st_size > _MAX_RECORD_BYTES:
            return None
        data = json.loads(path.read_bytes()[:_MAX_RECORD_BYTES].decode("utf-8"))
        if not procid.is_pid(data["pid"]) or type(data["since"]) not in (int, float):
            return None
        return {**data, "since": float(data["since"])}
    except (OSError, ValueError, KeyError, TypeError, OverflowError, RecursionError):
        return None


def _is_plain_dir(st: os.stat_result) -> bool:
    reparse = getattr(st, "st_file_attributes", 0) & getattr(
        stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISDIR(st.st_mode) and not reparse


def _folders_in(path: str) -> Optional[list[str]]:
    """*path* and every folder below it, deepest last, or None when one of them is a
    link or holds anything but folders."""
    if not _is_plain_dir(os.lstat(path)):
        return None
    found = [path]
    for child in os.listdir(path):
        below = _folders_in(os.path.join(path, child))
        if below is None:
            return None
        found += below
    return found


def _remove_folders(dead: dict) -> int:
    """Remove the empty comgr folders of the *dead* processes: pid to the
    ``(since, last seen)`` times of its records. Returns how many."""
    removed = 0
    with os.scandir(_temp_folder()) as it:
        found = [(e.path, e.name.split("-")[1]) for e in it
                 if e.name.startswith("comgr-") and e.name.count("-") >= 2]
    for path, pid in found:
        try:
            st = os.lstat(path)
            if hasattr(os, "getuid") and st.st_uid != os.getuid():
                continue
            if not any(since - 2 <= st.st_mtime <= seen + _SEEN_MARGIN_SECONDS
                       for since, seen in dead.get(pid, ())):
                continue
            folders = _folders_in(path)
            for folder in reversed(folders or ()):
                os.rmdir(folder)
            removed += folders is not None
        except (OSError, RecursionError) as e:
            logger.debug("comgr temp: %s stays (%s)", path, e)
    return removed


def sweep() -> list:
    """Remove the folders of every recorded process that is gone, its record, and
    what is no record. Returns the starters of the recorded processes that still
    run. Never raises."""
    running: list = []
    dead: dict = {}
    done: list = []
    try:
        paths = list(records_dir().iterdir())
    except OSError:
        return running   # See test_no_records_folder_is_nothing_to_do.
    for path in paths:
        try:
            seen = path.stat().st_mtime
            rec = _read(path)
            if rec is None or rec.get("space") != procid.pid_space_id():
                limit = _SEEN_MARGIN_SECONDS if rec is None else _FOREIGN_RECORD_SECONDS
                if time.time() - seen > limit:
                    done.append(path)
            elif procid.pid_liveness(rec["pid"]) is False:
                dead.setdefault(str(rec["pid"]), []).append((rec["since"], seen))
                done.append(path)
            elif not procid.start_identity_differs(
                    rec.get("start"), procid.process_start_identity(rec["pid"])):
                running.append(rec.get("starter"))
                os.utime(path)
        except OSError as e:
            logger.debug("comgr temp: record %s stays (%s)", path, e)
    try:
        removed = _remove_folders(dead) if dead else 0
        for path in done:
            if path.is_file():
                path.unlink(missing_ok=True)
        if removed:
            logger.debug("comgr temp: removed %d empty folder(s) of ended GPU processes", removed)
    except OSError as e:
        logger.debug("comgr temp: cleanup failed (%s)", e)
    return running


def watch() -> None:
    """Remove the folders of recorded processes as they end, from a background
    thread that stops once none has been running for a while. Called where localm
    starts a process that may load a HIP runtime; also removes what is left when
    this process ends normally, unless it is itself a worker, whose parent does.
    Never raises."""
    global _watcher
    with _lock:
        os.environ[_STARTER_VAR] = str(os.getpid())
        _hook_exit()
        if _watcher is not None and _watcher.is_alive():
            return
        _watcher = threading.Thread(target=_watch_loop, name="localm-comgr-temp", daemon=True)
        try:
            _watcher.start()
        except RuntimeError as e:
            logger.debug("comgr temp: no watcher thread (%s)", e)
            _watcher = None


def _watch_loop() -> None:
    idle = 0
    while idle < _IDLE_POLLS:
        idle = 0 if sweep() else idle + 1
        time.sleep(_POLL_SECONDS)


def _hook_exit() -> None:
    global _exit_hooked
    if _exit_hooked:
        return
    _exit_hooked = True
    from multiprocessing import parent_process, util
    if parent_process() is None:
        util.Finalize(None, finish, exitpriority=-100)


def finish(wait: float = _FINISH_WAIT_SECONDS) -> None:
    """The last cleanup of a localm process: stop this process's VRAM probe
    daemon, wait at most *wait* seconds for the recorded processes it started to
    end, and remove their folders. Never raises."""
    try:
        loader = sys.modules.get("localm.inference.backends.llamacpp._loader")
        if loader is not None:
            loader.stop_probe_daemon(wait=1.0)
        deadline = time.monotonic() + wait
        while str(os.getpid()) in sweep() and time.monotonic() < deadline:
            time.sleep(0.1)
    except Exception as e:   # noqa: BLE001 - never blocks an exit
        logger.debug("comgr temp: final cleanup failed (%s)", e)
