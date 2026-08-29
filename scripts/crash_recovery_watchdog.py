#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Detached crash-recovery watchdog: relaunch the server if it dies without a
clean shutdown, without waiting for a human to notice and restart it by hand.

``localm.portmux.run_server()`` spawns this DETACHED once, right after arming
the crash guard (``bugreport.arm_crash_guard``). It is stdlib-only, with ZERO
``localm`` package imports, so a broken localm install cannot break the thing
meant to recover from one - the same reasoning as ``update_watchdog.py``.

It watches one PID at a time. When that PID exits, it gives the replacement a
grace window to show up on its own (the ordinary restart path re-execs a new
process on the same port), by polling that port's own unauthenticated
``GET /whoami``:

* a NEW instance_id answers within the grace window: an ordinary restart or
  update already in progress. Read that instance's own crash marker for its
  pid and resume watching it.
* nothing answers, and the watched instance's own crash marker file (see
  ``localm.bugreport._crash_marker_path`` - the naming is duplicated here
  rather than imported, and pinned by
  test_crash_watchdog_marker_path_matches_bugreport) is still present: the
  process died without reaching its own clean-shutdown path. Relaunch it with
  the saved command line.
* nothing answers, and the marker is gone: a clean, intentional shutdown
  (disarm_crash_guard already ran). Exit; there is nothing left to watch.

Relaunches are capped within a rolling window (see _STORM_LIMIT/_STORM_WINDOW_S)
so a process that crashes on every startup cannot loop forever; once capped,
this watchdog logs and exits rather than retrying silently.

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
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

EXIT_WATCHDOG_STOPPED_CLEAN = 0
EXIT_STORM_CAPPED = 1
EXIT_RELAUNCH_FAILED = 2

DEFAULT_POLL_INTERVAL_S = 3.0
DEFAULT_GRACE_S = 25.0
DEFAULT_REQUEST_TIMEOUT_S = 3.0
_STORM_LIMIT = 4
_STORM_WINDOW_S = 300.0


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
        import os
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


def relaunch(argv: list, *, log_path: Optional[Path]) -> Optional[int]:
    """Start *argv* detached, matching the creation flags
    localm.updater.spawn_health_watchdog already uses for a process meant to
    outlive its spawner. Returns the new process's pid, or None on failure."""
    kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                  stderr=subprocess.DEVNULL, close_fds=True)
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
       crash_dir: Path, relaunch_argv: list, poll_interval: float,
       grace_s: float, request_timeout: float, log_path: Optional[Path]) -> int:
    _log(log_path,
        f"watching pid {pid} (instance {instance_id!r}) on "
        f"{scheme}://{host}:{port}")
    restart_history: list = []

    while True:
        if pid_alive(pid):
            time.sleep(poll_interval)
            continue

        _log(log_path, f"pid {pid} is gone - waiting up to {grace_s}s for a "
                       "replacement to come up on its own")
        replacement = poll_for_new_instance(
            host, port, scheme, instance_id, crash_dir, deadline_s=grace_s,
            poll_interval=poll_interval, request_timeout=request_timeout,
            log_path=log_path)
        if replacement is not None:
            instance_id, pid = replacement
            _log(log_path,
                f"replacement instance {instance_id!r} (pid {pid}) is "
                "already up - resuming watch, no relaunch needed")
            continue

        marker = marker_path(crash_dir, instance_id)
        if not marker.exists():
            _log(log_path,
                f"no replacement appeared and {marker.name} is gone - this "
                "was a clean, intentional shutdown. Nothing to recover.")
            return EXIT_WATCHDOG_STOPPED_CLEAN

        now = time.monotonic()
        restart_history = [t for t in restart_history if now - t < _STORM_WINDOW_S]
        if len(restart_history) >= _STORM_LIMIT:
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
        new_os_pid = relaunch(relaunch_argv, log_path=log_path)
        if new_os_pid is None:
            return EXIT_RELAUNCH_FAILED
        restart_history.append(now)
        pid = new_os_pid
        learned = poll_for_new_instance(
            host, port, scheme, instance_id, crash_dir, deadline_s=grace_s,
            poll_interval=poll_interval, request_timeout=request_timeout,
            log_path=log_path)
        if learned is not None:
            instance_id, learned_pid = learned
            _log(log_path,
                f"relaunch confirmed as instance {instance_id!r} "
                f"(pid {learned_pid})")
        else:
            _log(log_path,
                f"relaunch (pid {pid}) did not answer /whoami within "
                f"{grace_s}s - watching its process directly under the "
                "previous instance id until it does")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Watch a localm server pid; relaunch it if it dies "
                    "without a clean shutdown.")
    ap.add_argument("--pid", required=True, type=int)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", required=True, type=int)
    ap.add_argument("--scheme", default="http", choices=("http", "https"))
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--crash-dir", required=True, type=Path)
    ap.add_argument("--relaunch-argv", required=True,
                    help="JSON-encoded list, e.g. '[\"python\",\"-m\",\"localm\"]'")
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

    return run(pid=args.pid, host=args.host, port=args.port, scheme=args.scheme,
              instance_id=args.instance_id, crash_dir=args.crash_dir.resolve(),
              relaunch_argv=relaunch_argv, poll_interval=args.poll_interval,
              grace_s=args.grace, request_timeout=args.request_timeout,
              log_path=args.log_file)


if __name__ == "__main__":
    raise SystemExit(main())
