# SPDX-License-Identifier: AGPL-3.0-or-later
"""Background dependency-install task + the host-local guard for its routes.

The GUI kicks off a plugin's pip-extra install as a background task and follows
its progress over SSE, so a multi-minute install never blocks the request. The
task keeps a full line buffer so a viewer that connects late (or reconnects)
replays the whole log, and it keeps running even if the viewer navigates away.

``host_pip_allowed`` is the security boundary: only the local operator (a
loopback bind) may trigger a server-side pip. A remote client is refused.
"""

from __future__ import annotations

import threading

from localm.bindhost import is_loopback_host  # noqa: F401  (re-export for back-compat)
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
        # Set result before status, both under the lock, so the SSE reader (which
        # checks status then reads result) never sees a finished status with a
        # not-yet-set result.
        with self._lock:
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
    crashing the worker thread."""
    from localm.plugins import deps
    try:
        result = manager.install_plugin_deps(name, on_progress=task.emit)
    except Exception as e:               # never let the worker die silently
        logger.debug("dep install worker crashed for %s", name, exc_info=True)
        task.emit(f"internal error: {e}")
        result = deps.InstallResult(ok=False, error=str(e))
    task.finish(result)


def host_pip_allowed(app) -> bool:
    """Whether the HTTP dependency-install path may run pip on this host.

    Decided from the server's BIND host, never the request peer: the GUI runs
    behind portmux, which relays every connection through an internal loopback
    socket, so ``request.client.host`` reads as 127.0.0.1 even for a genuinely
    REMOTE client. Only a loopback BIND means every client is truly on this
    machine. A network bind (e.g. -H 0.0.0.0) is refused, and an unknown bind
    host is denied."""
    state = getattr(app, "state", None)
    bind_host = getattr(state, "bind_host", None)
    return is_loopback_host(bind_host or "")
