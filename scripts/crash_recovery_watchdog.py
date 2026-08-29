#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Detached crash-recovery watchdog: relaunch the server once if it dies
without a clean shutdown, without waiting for a human to notice and restart
it by hand.

``localm.portmux.run_server()`` spawns this DETACHED once, right after arming
the crash guard (``bugreport.arm_crash_guard``). It is stdlib-only, with ZERO
``localm`` package imports, so a broken localm install cannot break the thing
meant to recover from one - the same reasoning as ``update_watchdog.py``.

It watches exactly one pid/instance and returns as soon as ONE of these is
true, and never follows a transition to a different pid:

* the watched pid answers ``GET /whoami`` with an instance_id other than its
  own within the grace window after exiting: an ordinary restart or update
  was already in flight. Nothing to do - that instance's own run_server()
  call already spawned its own watchdog.
* nothing answers, and the watched instance's own crash marker file (see
  ``localm.bugreport._crash_marker_path`` - the naming is duplicated here
  rather than imported, and pinned by
  test_crash_watchdog_marker_path_matches_bugreport) is gone: a clean,
  intentional shutdown (disarm_crash_guard already ran). Nothing to recover.
* nothing answers, and the marker is still present: the process died without
  reaching its own clean-shutdown path. Relaunch it with the saved command
  line, then return - the relaunched process's own startup spawns a fresh
  watchdog for itself, so this one does not keep watching afterward. Two
  watchdogs both adopting the same replacement would race to relaunch it
  twice on a later crash; retiring on handoff avoids that entirely.

Relaunches are capped within a rolling window (_STORM_LIMIT/_STORM_WINDOW_S)
so a process that crashes on every startup is not retried forever. Because
each watchdog is a short-lived, one-shot process, that history cannot live in
its own memory - it is carried in the LOCALM_CRASH_WATCHDOG_HISTORY
environment variable, set on the relaunched child so its own next watchdog
inherits it, mirroring localm.inference._hang_alarm's in-process restart
history but threaded across real process boundaries instead of an execv.

This does not touch privacy mode: it writes no new persistent file by default
(--log-file is opt-in, mirroring update_watchdog.py), never reads or logs
request/chat content, and a relaunch is an action on an already-running
server, not a new disclosure.

Run standalone for manual testing:
    python scripts/crash_recovery_watchdog.py --pid 12345 --host 127.0.0.1 \\
        --port 8642 --instance-id abc123 --crash-dir /path/to/home/run \\
        --relaunch-argv '["python", "-m", "localm", "gui", "-p", "8642"]'
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

EXIT_OK = 0
EXIT_STORM_CAPPED = 1
EXIT_RELAUNCH_FAILED = 2

DEFAULT_POLL_INTERVAL_S = 3.0
DEFAULT_GRACE_S = 25.0
DEFAULT_REQUEST_TIMEOUT_S = 3.0
_STORM_LIMIT = 4
_STORM_WINDOW_S = 300.0
_HISTORY_ENV = "LOCALM_CRASH_WATCHDOG_HISTORY"


def _log(log_path: Optional[Path], msg: str) -> None:
    """Timestamped trace line to stdout and, best-effort, *log_path*."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def pid_alive(pid: int) -> bool:
    """True if *pid* names a live process. Windows and POSIX both handled
    without a third-party dependency."""
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.wintypes.DWORD()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(
                handle, ctypes.byref(code))
            return bool(ok) and code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    else:
        import errno

        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError as e:
            return e.errno != errno.ESRCH
        return True


def marker_path(crash_dir: Path, instance_id: str) -> Path:
    """Mirrors localm.bugreport._crash_marker_path's naming exactly - pinned
    by test_crash_watchdog_marker_path_matches_bugreport so the two cannot
    silently drift apart."""
    return crash_dir / f"server-crash.{instance_id}.marker"


def read_marker_pid(marker: Path) -> Optional[int]:
    """The pid recorded in *marker*, or None if it is missing/unreadable."""
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        pid = data.get("pid")
        return int(pid) if pid is not None else None
    except (OSError, ValueError, TypeError):
        return None


def _insecure_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def probe_whoami(host: str, port: int, scheme: str, timeout: float) -> Optional[dict]:
    """One GET against the instance's own /whoami. None on any failure - a
    refused connection, a timeout, a non-200, or an unparseable body all just
    mean "nothing answering yet". localm's own local CA is self-signed, so an
    https probe never verifies the certificate - the watchdog only needs to
    know the instance answered, not to validate the LAN-exposure cert chain."""
    _host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    url = f"{scheme}://{_host}:{port}/whoami"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "localm-crash-watchdog"})
        handlers = [urllib.request.ProxyHandler({})]
        if url.startswith("https://"):
            handlers.append(urllib.request.HTTPSHandler(context=_insecure_ssl_context()))
        opener = urllib.request.build_opener(*handlers)
        with opener.open(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def poll_for_new_instance(host: str, port: int, scheme: str, old_instance_id: str,
                          crash_dir: Path, *, deadline_s: float, poll_interval: float,
                          request_timeout: float, log_path: Optional[Path]
                          ) -> Optional[tuple]:
    """Poll /whoami until it answers with an instance_id other than
    *old_instance_id*, or *deadline_s* elapses. Returns (new_instance_id,
    new_pid) once that instance's own crash marker names a pid (arm_crash_guard
    writes the marker before the server can answer /whoami at all, so a
    responding new instance always has one), else None on timeout."""
    deadline = time.monotonic() + deadline_s
    while True:
        body = probe_whoami(host, port, scheme, request_timeout)
        if body is not None:
            new_id = body.get("instance_id")
            if new_id and new_id != old_instance_id:
                new_pid = read_marker_pid(marker_path(crash_dir, new_id))
                if new_pid is not None:
                    return (new_id, new_pid)
        if time.monotonic() >= deadline:
            return None
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))


def restart_history_from_string(raw: str) -> list:
    """Parse the comma-separated epoch-seconds list carried in
    LOCALM_CRASH_WATCHDOG_HISTORY. An unparseable entry is dropped rather
    than failing the whole history, since a corrupt env var must never block
    a genuine crash recovery."""
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            try:
                out.append(float(part))
            except ValueError:
                continue
    return out


def storm_active(history: list, now_epoch: float) -> bool:
    recent = [t for t in history if now_epoch - t < _STORM_WINDOW_S]
    return len(recent) >= _STORM_LIMIT


def relaunch(argv: list, *, restart_history: list, now_epoch: float,
            log_path: Optional[Path]) -> Optional[int]:
    """Start *argv* detached, matching the creation flags
    localm.updater.spawn_health_watchdog already uses for a process meant to
    outlive its spawner. The updated restart history is set on the child's
    environment (a plain Popen, unlike os.execv, does not inherit implicitly)
    so the watchdog THAT process spawns for itself can see how many
    relaunches already happened in this window. Returns the new process's
    pid, or None on failure."""
    kept = [t for t in restart_history if now_epoch - t < _STORM_WINDOW_S]
    kept.append(now_epoch)
    env = os.environ.copy()
    env[_HISTORY_ENV] = ",".join(f"{t:.3f}" for t in kept)
    kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                  stderr=subprocess.DEVNULL, close_fds=True, env=env)
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200))
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(argv, **kwargs)
    except OSError as e:
        _log(log_path, f"relaunch failed to start: {e}")
        return None
    _log(log_path, f"relaunched (pid {proc.pid}): {argv!r}")
    return proc.pid


def run(*, pid: int, host: str, port: int, scheme: str, instance_id: str,
       crash_dir: Path, relaunch_argv: list, restart_history: list,
       poll_interval: float, grace_s: float, request_timeout: float,
       log_path: Optional[Path]) -> int:
    """Watch exactly one pid/instance until ONE of three things happens, then
    return - this watchdog never follows a transition to a new pid. Whatever
    replaces the watched instance (a relaunch this call performs, or an
    ordinary restart/update already in flight) goes through run_server()
    again on its own, which spawns ITS OWN fresh watchdog - so two watchdogs
    racing to watch the same replacement is avoided by each one retiring as
    soon as it sees a replacement exists, rather than adopting it."""
    _log(log_path,
        f"watching pid {pid} (instance {instance_id!r}) on "
        f"{scheme}://{host}:{port}")

    while pid_alive(pid):
        time.sleep(poll_interval)

    _log(log_path, f"pid {pid} is gone - waiting up to {grace_s}s for a "
                   "replacement to come up on its own")
    replacement = poll_for_new_instance(
        host, port, scheme, instance_id, crash_dir, deadline_s=grace_s,
        poll_interval=poll_interval, request_timeout=request_timeout,
        log_path=log_path)
    if replacement is not None:
        new_id, new_pid = replacement
        _log(log_path,
            f"replacement instance {new_id!r} (pid {new_pid}) is already "
            "up with its own watchdog - nothing more for this one to do")
        return EXIT_OK

    marker = marker_path(crash_dir, instance_id)
    if not marker.exists():
        _log(log_path,
            f"no replacement appeared and {marker.name} is gone - this was "
            "a clean, intentional shutdown. Nothing to recover.")
        return EXIT_OK

    now = time.time()
    if storm_active(restart_history, now):
        _log(log_path,
            f"{marker.name} is still present (no clean shutdown) but "
            f"{_STORM_LIMIT} relaunches already happened in the last "
            f"{_STORM_WINDOW_S:.0f}s - not relaunching again. Giving up; "
            "the crash marker and any trace file are left in place for "
            "the next manual start to report.")
        return EXIT_STORM_CAPPED

    _log(log_path,
        f"{marker.name} is still present - the process died without a "
        "clean shutdown. Relaunching.")
    new_os_pid = relaunch(relaunch_argv, restart_history=restart_history,
                          now_epoch=now, log_path=log_path)
    if new_os_pid is None:
        return EXIT_RELAUNCH_FAILED
    _log(log_path,
        f"relaunch (pid {new_os_pid}) started - it will spawn its own "
        "watchdog on startup. This watchdog's job is done.")
    return EXIT_OK


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Watch a localm server pid; relaunch it once if it "
                    "dies without a clean shutdown.")
    ap.add_argument("--pid", required=True, type=int)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", required=True, type=int)
    ap.add_argument("--scheme", default="http", choices=("http", "https"))
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--crash-dir", required=True, type=Path)
    ap.add_argument("--relaunch-argv", required=True,
                    help="JSON-encoded list, e.g. '[\"python\",\"-m\",\"localm\"]'")
    ap.add_argument("--restart-history", default="",
                    help="Comma-separated epoch-seconds list, normally set "
                        "via LOCALM_CRASH_WATCHDOG_HISTORY rather than by hand.")
    ap.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_S)
    ap.add_argument("--grace", type=float, default=DEFAULT_GRACE_S)
    ap.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    ap.add_argument("--log-file", type=Path, default=None)
    args = ap.parse_args(argv)

    try:
        relaunch_argv = json.loads(args.relaunch_argv)
        if not isinstance(relaunch_argv, list) or not relaunch_argv:
            raise ValueError("relaunch-argv must be a non-empty JSON list")
    except ValueError as e:
        _log(args.log_file, f"bad --relaunch-argv: {e}")
        return EXIT_RELAUNCH_FAILED

    restart_history = restart_history_from_string(
        args.restart_history or os.environ.get(_HISTORY_ENV, ""))

    return run(pid=args.pid, host=args.host, port=args.port, scheme=args.scheme,
              instance_id=args.instance_id, crash_dir=args.crash_dir.resolve(),
              relaunch_argv=relaunch_argv, restart_history=restart_history,
              poll_interval=args.poll_interval, grace_s=args.grace,
              request_timeout=args.request_timeout, log_path=args.log_file)


if __name__ == "__main__":
    raise SystemExit(main())
