# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI model routes, packages group: listing what a registered Hugging Face
model's own code needs that localm can add, then installing exactly that list
after the user agreed to it.

Both routes need ``plugins:admin`` and refuse on a network bind
(``deps_task.host_pip_allowed``): only the local operator may install code into
localm's environment. An install installs only a plan this server showed, by id.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from localm import scopes
from localm.debuglog import logger
from localm.inference.http_server import require_scope
from localm.plugins import deps_task
from localm.plugins.gui.routes.models._context import (ModelRouteContext,
                                                       _require_registered)

_NETWORK_BIND = ("Packages are installed on the machine running localm only, and this "
                 "server is on a network bind. Run  localm install-packages <model>  there.")


class PackagesPlanRequest(BaseModel):
    model: str


class PackagesInstallRequest(BaseModel):
    plan_id: str


_task_lock = threading.Lock()
_task: dict = {"current": None}


def current_task() -> Optional[deps_task.DepInstallTask]:
    """The running or last finished package install, or None."""
    with _task_lock:
        return _task["current"]


def _run(plan, task: deps_task.DepInstallTask) -> None:
    from localm.model_manager import model_packages
    try:
        result = model_packages.install(plan, on_progress=task.emit)
    except Exception as e:  # the worker thread never dies without a result
        logger.warning("model package install crashed: %s", e, exc_info=True)
        task.emit(f"internal error: {e}")
        result = model_packages.InstallResult(ok=False, error=str(e))
    if not result.ok:
        task.emit(result.error)
    task.finish(result)


def start_install(plan) -> deps_task.DepInstallTask:
    """Start installing *plan* in a background thread. Raises HTTPException 409
    while another install runs."""
    with _task_lock:
        running = _task["current"]
        if running is not None and running.status == "running":
            raise HTTPException(409, "Another package install is running.")
        task = deps_task.DepInstallTask("model-packages")
        _task["current"] = task
    threading.Thread(target=_run, args=(plan, task), name="localm-model-packages",
                     daemon=True).start()
    return task


def register(app: FastAPI, context: ModelRouteContext) -> None:
    del context

    @app.post("/api/models/packages/plan",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def plan_model_packages(req: PackagesPlanRequest):
        if not deps_task.host_pip_allowed(app):
            raise HTTPException(403, _NETWORK_BIND)
        from localm.model_manager import model_packages
        from localm.model_manager.missing_packages import refusal_for
        from localm.model_manager.registry import get_model_path

        def _plan():
            _require_registered(req.model)
            folder = get_model_path(req.model)
            if folder is None or not folder.is_dir():
                raise HTTPException(409, f"{req.model} is not a Hugging Face model folder.")
            state = model_packages.folder_state(folder)
            if not state.trusted:
                raise HTTPException(409, "localm runs a model's own code only with "
                                         "hf_trust_remote_code on.")
            if not state.missing:
                raise HTTPException(409, f"{req.model} needs no packages this install lacks.")
            if state.shipped or state.blocked:
                raise HTTPException(409, refusal_for(req.model, state.missing))
            try:
                p = model_packages.plan(state.addable)
            except model_packages.PlanError as e:
                raise HTTPException(409, str(e)) from e
            model_packages.remember(p)
            return {"model": req.model, **p.as_dict()}

        return await asyncio.get_running_loop().run_in_executor(None, _plan)

    @app.post("/api/models/packages/install",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def install_model_packages(req: PackagesInstallRequest):
        if not deps_task.host_pip_allowed(app):
            raise HTTPException(403, _NETWORK_BIND)
        from localm.model_manager import model_packages
        p = model_packages.take(req.plan_id)
        if p is None:
            raise HTTPException(404, "This install list has expired. Open it again.")
        task = start_install(p)
        return {"status": task.status, "lines": task.snapshot()}

    @app.get("/api/models/packages/install/events",
             dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def model_packages_events():
        if not deps_task.host_pip_allowed(app):
            raise HTTPException(403, _NETWORK_BIND)
        task = current_task()
        if task is None:
            raise HTTPException(404, "No package install has run.")

        async def _stream():
            idx = 0
            ticks = 0
            while True:
                lines = task.snapshot()
                while idx < len(lines):
                    yield ("data: " + json.dumps({"type": "log", "line": lines[idx]},
                                                 ensure_ascii=False) + "\n\n")
                    idx += 1
                if task.status != "running":
                    yield "data: " + json.dumps(task.end_event(), ensure_ascii=False) + "\n\n"
                    return
                ticks += 1
                if ticks % 30 == 0:
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.2)

        return StreamingResponse(_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
