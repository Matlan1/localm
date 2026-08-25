# SPDX-License-Identifier: AGPL-3.0-or-later
"""Background dependency-install task + the host-local guard for its routes."""

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
    """Body of the background thread: install *name*'s extras, feeding progress into *task*."""
    from localm.plugins import deps
    try:
        result = manager.install_plugin_deps(name, on_progress=task.emit)
    except Exception as e:               # never let the worker die silently
        logger.debug("dep install worker crashed for %s", name, exc_info=True)
        task.emit(f"internal error: {e}")
        result = deps.InstallResult(ok=False, error=str(e))
    task.finish(result)


def host_pip_allowed(app) -> bool:
    """Whether the HTTP dependency-install path may run pip on this host."""
    state = getattr(app, "state", None)
    bind_host = getattr(state, "bind_host", None)
    return is_loopback_host(bind_host or "")
