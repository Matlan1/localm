# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host-side plugin dependency-install routes (pip extras).

Carries top-level fastapi imports, keeping engine.py fastapi-free at import time
(it imports fastapi lazily).

Security: a remote client must NEVER trigger a server-side pip. Both endpoints
fail closed with 403 whenever the server is on a NETWORK bind - decided from the
bind host, not the request peer, because portmux relays every connection through
an internal loopback socket (see deps_task.host_pip_allowed). This is on top of
the PLUGINS_ADMIN scope already required.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse

from localm import scopes
from localm.inference.http_server import require_scope
from localm.plugins import deps_task


def register_dep_routes(app, manager) -> None:
    @app.post("/api/plugins/{name}/install-deps",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def install_plugin_deps_ep(name: str):
        if not deps_task.host_pip_allowed(app):
            raise HTTPException(
                403, "Dependency install runs on the host only, and this server "
                     "is on a network bind. Install on the machine running localm "
                     f"(e.g. localm plugin install-deps {name}).")
        if manager._spec_for(name) is None:
            raise HTTPException(404, f"No such plugin: {name}")
        task = manager.start_dep_install(name)
        return {"status": task.status, "name": name,
                "missing": manager.plugin_missing_deps(name),
                "lines": task.snapshot()}

    @app.get("/api/plugins/{name}/install-deps/events",
             dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def install_plugin_deps_events(name: str):
        if not deps_task.host_pip_allowed(app):
            raise HTTPException(403, "Dependency install runs on the host only.")
        task = manager.get_dep_task(name)
        if task is None:
            raise HTTPException(404, f"No dependency install for {name}")

        async def _stream():
            # Replay the full buffer, then poll for new lines until the task
            # finishes, so a late or reconnecting viewer sees the whole log.
            idx = 0
            ticks = 0
            while True:
                lines = task.snapshot()
                while idx < len(lines):
                    yield ("data: " + json.dumps(
                        {"type": "log", "line": lines[idx]},
                        ensure_ascii=False) + "\n\n")
                    idx += 1
                if task.status != "running":
                    yield ("data: " + json.dumps(
                        task.end_event(), ensure_ascii=False) + "\n\n")
                    return
                ticks += 1
                if ticks % 30 == 0:            # ~keepalive every few seconds
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.2)

        return StreamingResponse(
            _stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
