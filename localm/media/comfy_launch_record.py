# SPDX-License-Identifier: AGPL-3.0-or-later
"""On-disk record of a ComfyUI process localm launched, readable by any later
localm process in the same data dir.

``comfy_client._launch_and_wait`` keeps the live ``Popen`` handle in the
process-local ``_spawned_procs`` and calls :func:`write_record`, which stores,
per api_url, the launched process's pid, start time, executable and command
line, plus the pid and start time of the localm process that launched it.
:func:`adopt` turns a record back into a handle ``stop_comfy`` accepts, only
when the process under that pid still has the recorded start time, executable
and command line. :func:`drop` removes a record; callers drop one once its
process has exited or its pid belongs to a different process.

Needs ``psutil`` both to write and to check a record. Without it nothing is
recorded and :func:`adopt` answers ``UNVERIFIABLE``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

RECORD_GLOB = "comfy-launched-*.json"

MATCH = "match"
GONE = "gone"
MISMATCH = "mismatch"
UNVERIFIABLE = "unverifiable"

_START_TIME_TOLERANCE_S = 1.0


def record_path(api_url: str) -> Path:
    """The record file for *api_url*: one per api_url, in the data dir,
    matching :data:`RECORD_GLOB`."""
    from localm.config import home_dir
    suffix = hashlib.sha256(api_url.rstrip("/").encode("utf-8")).hexdigest()[:12]
    return home_dir() / f"comfy-launched-{suffix}.json"


def _identity(pid: int) -> dict:
    """``{"pid", "create_time", "exe", "cmdline"}`` of process *pid*. Raises
    ImportError without psutil and ``psutil.Error`` when it cannot be read."""
    import psutil
    proc = psutil.Process(pid)
    with proc.oneshot():
        return {"pid": int(pid), "create_time": float(proc.create_time()),
                "exe": str(proc.exe()), "cmdline": [str(a) for a in proc.cmdline()]}


def write_record(api_url: str, pid: int) -> bool:
    """Record process *pid* as the ComfyUI this localm process launched for
    *api_url*. Returns whether the record was written; a failure to read the
    process or write the file is logged at WARNING. Never raises."""
    import os

    from localm.debuglog import logger
    from localm.storekit import atomic_write
    api_url = api_url.rstrip("/")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        logger.debug("not recording the ComfyUI launched for %s: pid %r is not a pid",
                     api_url, pid)
        return False
    try:
        launched = _identity(pid)
        owner = _identity(os.getpid())
        data = {"api_url": api_url, **launched,
                "owner_pid": owner["pid"], "owner_create_time": owner["create_time"]}
        atomic_write(record_path(api_url), json.dumps(data))
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("could not record the ComfyUI localm launched for %s (pid %s): %s; "
                       "localm comfy stop from another localm process cannot stop it",
                       api_url, pid, e)
        return False


def _parse(text: str) -> Optional[dict]:
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("cmdline"), list):
        return None
    try:
        record = {
            "api_url": str(data["api_url"]).rstrip("/"),
            "pid": int(data["pid"]),
            "create_time": float(data["create_time"]),
            "exe": str(data["exe"]),
            "cmdline": [str(a) for a in data["cmdline"]],
            "owner_pid": int(data["owner_pid"]),
            "owner_create_time": float(data["owner_create_time"]),
        }
    except (KeyError, TypeError, ValueError):
        return None
    if not record["api_url"] or record["pid"] <= 0:
        return None
    return record


def read_records() -> list[dict]:
    """Every readable record in the data dir. A file that cannot be read or
    parsed is skipped and logged at debug level. Never raises."""
    from localm.config import home_dir
    from localm.debuglog import logger
    records = []
    try:
        paths = sorted(home_dir().glob(RECORD_GLOB))
    except OSError as e:
        logger.debug("could not list ComfyUI launch records: %s", e)
        return records
    for path in paths:
        try:
            record = _parse(path.read_text(encoding="utf-8"))
        except OSError as e:
            logger.debug("could not read ComfyUI launch record %s: %s", path, e)
            continue
        if record is None:
            logger.debug("ignoring malformed ComfyUI launch record %s", path)
            continue
        if record_path(record["api_url"]) != path:
            logger.debug("ignoring ComfyUI launch record %s: it names another api_url", path)
            continue
        records.append(record)
    return records


def drop(record: dict) -> None:
    """Remove *record*'s file when it still names the same pid. Never raises."""
    from localm.debuglog import logger
    path = record_path(record["api_url"])
    try:
        current = _parse(path.read_text(encoding="utf-8"))
        if current is not None and current["pid"] == record["pid"]:
            path.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.debug("could not remove ComfyUI launch record %s: %s", path, e)


def _same_start(a: float, b: float) -> bool:
    return abs(a - b) <= _START_TIME_TOLERANCE_S


def owner_alive(record: dict) -> bool:
    """True when the localm process that wrote *record* is still running, or
    when that cannot be read (psutil missing or access denied)."""
    try:
        now = _identity(record["owner_pid"])
    except ImportError:
        return True
    except Exception as e:  # noqa: BLE001
        import psutil
        if isinstance(e, psutil.NoSuchProcess):
            return False
        return True
    return _same_start(now["create_time"], record["owner_create_time"])


@dataclass
class RecordedComfyProcess:
    """A ``Popen``-shaped handle on a recorded ComfyUI process, for
    ``stop_comfy`` and ``_kill_process_tree``: ``pid``, ``poll``,
    ``terminate``, ``kill`` and ``wait``."""
    pid: int
    _proc: object

    def poll(self) -> Optional[int]:
        """None while the process runs (or when that cannot be read), else 0."""
        import psutil
        try:
            if self._proc.is_running() and self._proc.status() != psutil.STATUS_ZOMBIE:
                return None
        except psutil.NoSuchProcess:
            return 0
        except psutil.Error:
            return None
        return 0

    def terminate(self) -> None:
        self._proc.terminate()

    def kill(self) -> None:
        self._proc.kill()

    def wait(self, timeout: Optional[float] = None):
        return self._proc.wait(timeout)

    def gone_within(self, seconds: float) -> bool:
        """Wait up to *seconds* for the process to exit; True once it has."""
        import psutil
        try:
            self._proc.wait(seconds)
        except psutil.TimeoutExpired:
            return self.poll() is not None
        except psutil.Error:
            return self.poll() is not None
        return True


def adopt(record: dict) -> tuple[str, Optional[RecordedComfyProcess]]:
    """``(MATCH, handle)`` when the process under *record*'s pid has the
    recorded start time, executable and command line. Otherwise ``(GONE,
    None)`` when no process has that pid, ``(MISMATCH, None)`` when the pid
    belongs to a different process, and ``(UNVERIFIABLE, None)`` when psutil is
    missing or the process cannot be read. Signals nothing. Never raises."""
    try:
        import psutil
    except ImportError:
        return UNVERIFIABLE, None
    try:
        proc = psutil.Process(record["pid"])
        with proc.oneshot():
            create_time = float(proc.create_time())
            exe = str(proc.exe())
            cmdline = [str(a) for a in proc.cmdline()]
    except psutil.NoSuchProcess:
        return GONE, None
    except psutil.Error:
        return UNVERIFIABLE, None
    if (not _same_start(create_time, record["create_time"])
            or exe != record["exe"] or cmdline != record["cmdline"]):
        return MISMATCH, None
    return MATCH, RecordedComfyProcess(pid=record["pid"], _proc=proc)
