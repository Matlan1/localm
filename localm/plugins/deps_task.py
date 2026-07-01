# SPDX-License-Identifier: AGPL-3.0-or-later
"""Background dependency-install task + the host-local guard for its routes.

The GUI kicks off a plugin's pip-extra install as a background task and follows
its progress over SSE, so a multi-minute install never blocks the request. The
task keeps a full line buffer so a viewer that connects late (or reconnects)
replays the whole log, and it keeps running even if the viewer navigates away.

``is_local_request`` is the security boundary: only the local operator (a
loopback request) may trigger a server-side pip. A remote client is refused.
"""

from __future__ import annotations

import ipaddress
import threading

from localm.debuglog import logger


class DepInstallTask:
    """One plugin's in-flight (or finished) dependency install."""

    def __init__(self, name: str):
        self.name = name
        self.lines: list = []              # full progress history (replayable)
        self.status = "running"            # running | done | error
        self.result = None                 # deps.InstallResult once finished
        self._lock = threading.Lock()

    def emit(self, line: str) -> None:
        with self._lock:
            self.lines.append(line)

    def snapshot(self) -> list:
        with self._lock:
            return list(self.lines)

    def finish(self, result) -> None:
        self.result = result
        self.status = "done" if result.ok else "error"

    def end_event(self) -> dict:
        r = self.result
        return {
            "type": "end",
            "ok": bool(r and r.ok),
            "installed": list(r.installed) if r else [],
            "failed": list(r.failed) if r else [],
            "error": (r.error if r else "") or "",
        }


def run_dep_install(manager, name: str, task: DepInstallTask) -> None:
    """Body of the background thread: install *name*'s extras, feeding progress
    into *task*. Any unexpected error is captured as a failed result rather than
    crashing the worker thread (AGENTS: surface, do not hide)."""
    from localm.plugins import deps
    try:
        result = manager.install_plugin_deps(name, on_progress=task.emit)
    except Exception as e:               # never let the worker die silently
        logger.debug("dep install worker crashed for %s", name, exc_info=True)
        task.emit(f"internal error: {e}")
        result = deps.InstallResult(ok=False, error=str(e))
    task.finish(result)


def is_local_request(request) -> bool:
    """True when the request comes from the local host (a loopback peer). This
    is what gates the host-only pip install. Unknown/parse-failure is treated as
    NOT local (fail closed)."""
    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client else None
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
