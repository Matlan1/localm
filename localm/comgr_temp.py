# SPDX-License-Identifier: AGPL-3.0-or-later
"""Removes the empty temp folders AMD's HIP runtime leaves behind.

The kernel compiler of the HIP runtime (comgr) makes ``comgr-<pid>-...`` folders
in the temp folder of the process that uses it and does not remove them when
that process is ended from outside, as localm ends its GPU workers.

A process that is about to load a HIP runtime calls :func:`register`: it records
its pid, start identity and temp folder in ``<data dir>/run/comgr-temp/``. The
localm process that starts workers (:func:`watch`) removes the folders of each
recorded process once that process is gone, and :func:`finish` does it for the
last ones before a localm process ends. Records a killed localm left are handled
at the next start.

A folder is removed only when its name carries a recorded pid, the recorded
process is gone, the folder was last changed while that process was running, and
it holds no file; nothing is followed through a link. A record's own change time
is when its process was last seen running."""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import secrets
import stat
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from localm import procid

logger = logging.getLogger("localm")

_POLL_SECONDS = 2.0
_IDLE_POLLS = 150
_FINISH_WAIT_SECONDS = 3.0
_FINISH_LIMIT_SECONDS = 8.0
_SEEN_MARGIN_SECONDS = 30.0

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
    if sys.platform == "win32":
        import ctypes
        buf = ctypes.create_unicode_buffer(32768)
        if ctypes.windll.kernel32.GetTempPathW(len(buf), buf):
            return os.path.abspath(buf.value)
        return tempfile.gettempdir()
    for var in ("TMPDIR", "TMP", "TEMP", "TEMPDIR"):
        if os.environ.get(var):
            return os.path.abspath(os.environ[var])
    return "/tmp"


def _record(pid: int, temp: str) -> None:
    d = records_dir()
    d.mkdir(parents=True, exist_ok=True)
    data = {"pid": pid, "start": procid.process_start_identity(pid),
            "space": procid.pid_space_id(), "created": _started_at(pid), "temp": temp}
    tmp = d / f".{pid}-{secrets.token_hex(4)}.tmp"
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, d / f"{pid}-{secrets.token_hex(4)}.json")


def _started_at(pid: int) -> Optional[float]:
    try:
        import psutil
        return float(psutil.Process(pid).create_time())
    except Exception as e:
        logger.debug("comgr temp: start time of process %s unknown (%s)", pid, e)
        return None


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
        _record(os.getpid(), _temp_folder())
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
    started and is about to end), assuming they use this process's temp folder.
    Never raises."""
    try:
        import psutil
        root = psutil.Process(pid)
        for p in [root, *root.children(recursive=True)]:
            _record(p.pid, _temp_folder())
    except Exception as e:
        logger.debug("comgr temp: could not record the processes of %s (%s)", pid, e)


def _read(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (isinstance(data, dict) and type(data.get("pid")) is int and data["pid"] > 0
            and isinstance(data.get("temp"), str) and os.path.isabs(data["temp"])):
        return data
    return None


def _is_gone(rec: dict) -> Optional[bool]:
    """True when the recorded process has ended, also when another process has its
    pid by now; False while it runs; None when that cannot be told."""
    if rec.get("space") != procid.pid_space_id():
        return None
    alive = procid.pid_liveness(rec["pid"])
    if not alive:
        return True if alive is False else None
    return procid.start_identity_differs(rec.get("start"),
                                         procid.process_start_identity(rec["pid"]))


def _is_plain_dir(st: os.stat_result) -> bool:
    reparse = getattr(st, "st_file_attributes", 0) & getattr(
        stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISDIR(st.st_mode) and not reparse


def _remove_empty_tree(path: str) -> bool:
    """Remove the folder *path* when it and every folder in it hold no file and no
    link. Returns whether it is gone."""
    try:
        if not _is_plain_dir(os.lstat(path)):
            return False
        with os.scandir(path) as it:
            names = [e.path for e in it]
        for child in names:
            if not _remove_empty_tree(child):
                return False
        os.rmdir(path)
        return True
    except FileNotFoundError:
        return True
    except OSError as e:
        logger.debug("comgr temp: %s stays (%s)", path, e)
        return False


def _remove_folders(rec: dict, last_seen: float) -> int:
    """Remove the empty comgr folders the recorded process, last seen running at
    *last_seen*, made. Returns how many."""
    name = re.compile(rf"comgr-{rec['pid']}-[^/\\]*")
    created = rec.get("created")
    until = last_seen + _SEEN_MARGIN_SECONDS
    holder = _started_at(rec["pid"]) if procid.pid_liveness(rec["pid"]) else None
    if holder is not None:
        until = min(until, holder)
    removed = 0
    try:
        with os.scandir(rec["temp"]) as it:
            found = [e for e in it if name.fullmatch(e.name)]
    except OSError as e:
        logger.debug("comgr temp: cannot list %s (%s)", rec["temp"], e)
        return 0
    for e in found:
        try:
            st = os.lstat(e.path)
        except OSError:
            continue
        if not _is_plain_dir(st):
            continue
        if st.st_mtime > until or (isinstance(created, (int, float))
                                   and st.st_mtime < created - 2):
            continue
        if hasattr(os, "getuid") and st.st_uid != os.getuid():
            continue
        removed += _remove_empty_tree(e.path)
    return removed


def sweep() -> int:
    """Remove the folders of every recorded process that is gone, and its record.
    Returns how many records of running processes are left. Never raises."""
    running = 0
    try:
        paths = list(records_dir().glob("*.json"))
    except OSError:
        return 0
    removed = 0
    for path in paths:
        rec = _read(path)
        if rec is None:
            continue
        gone = _is_gone(rec)
        try:
            if gone is False:
                running += 1
                os.utime(path)
            if gone is not True:
                continue
            removed += _remove_folders(rec, path.stat().st_mtime)
            path.unlink()
        except OSError as e:
            logger.debug("comgr temp: record %s stays (%s)", path, e)
    if removed:
        logger.debug("comgr temp: removed %d empty folder(s) of ended GPU processes", removed)
    return running


def watch() -> None:
    """Remove the folders of recorded processes as they end, from a background
    thread that stops once none has been running for a while. Called where localm
    starts a process that may load a HIP runtime; also removes what is left when
    this process ends normally, unless it is itself a worker, whose parent does.
    Never raises."""
    global _watcher
    with _lock:
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
    daemon, wait at most *wait* seconds for the recorded processes below this one
    to end, and remove their folders. Returns after ``_FINISH_LIMIT_SECONDS`` at
    the latest. Never raises."""
    t = threading.Thread(target=_finish, args=(wait,), name="localm-comgr-finish",
                         daemon=True)
    try:
        t.start()
    except RuntimeError as e:
        logger.debug("comgr temp: no final cleanup (%s)", e)
        return
    t.join(_FINISH_LIMIT_SECONDS)


def _finish(wait: float) -> None:
    try:
        mine = _descendants()
        loader = sys.modules.get("localm.inference.backends.llamacpp._loader")
        if loader is not None:
            loader.stop_probe_daemon(wait=1.0)
        deadline = time.monotonic() + wait
        while _recorded_running(mine) and time.monotonic() < deadline:
            time.sleep(0.1)
        sweep()
    except Exception as e:   # noqa: BLE001 - never blocks an exit
        logger.debug("comgr temp: final cleanup failed (%s)", e)


def _descendants() -> set:
    import psutil
    return {p.pid for p in psutil.Process().children(recursive=True)}


def _recorded_running(pids: set) -> bool:
    for path in records_dir().glob("*.json"):
        rec = _read(path)
        if rec is not None and rec["pid"] in pids and _is_gone(rec) is False:
            return True
    return False
