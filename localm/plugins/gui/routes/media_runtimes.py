# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI routes for the native media runtimes a finished job leaves loaded (the
stable-diffusion.cpp worker of the Images and Video pages, the KoboldCpp
runtime of the Music page), so the model sidebar can show them and free them.

  GET  /api/media/runtimes         -> ``{"runtimes": [{"name", "state",
                                      "idle_unload_seconds"}]}``, one row per
                                      runtime this process holds loaded; state
                                      is "busy" (a job uses it), "idle" or
                                      "unknown".
  POST /api/media/runtimes/unload  -> stop every idle one: ``{"outcomes":
                                      [{"name", "outcome"}]}`` with outcome
                                      "stopped", "busy" (left running) or
                                      "failed".
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from localm import scopes
from localm.inference._threadpool_timeout import ThreadCallTimeout, run_in_threadpool_bounded
from localm.inference.http_server import require_scope

_READ_TIMEOUT_S = 10.0
_UNLOAD_TIMEOUT_S = 60.0


def _idle_seconds() -> dict:
    """``{display name: seconds}`` after which each idle runtime stops itself."""
    from localm import vram
    from localm.media.koboldcpp import server as kserver
    from localm.media.sdcpp import shared
    by_key = {"sdcpp": shared.IDLE_UNLOAD_SECONDS, "koboldcpp": kserver.IDLE_SECONDS}
    return {name: by_key[key] for key, name in vram._NATIVE_RUNTIME_NAME.items()}


def _states() -> list:
    from localm import vram
    idle = _idle_seconds()
    return [{"name": name, "state": state, "idle_unload_seconds": idle.get(name)}
            for name, state in vram.native_media_runtime_states().items()]


def _unload() -> list:
    from localm import vram
    return [{"name": name, "outcome": outcome}
            for name, outcome in vram.stop_idle_native_media().items()]


def register(app: FastAPI, ctx) -> None:
    @app.get("/api/media/runtimes", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def media_runtimes():
        """The native media runtimes this process holds loaded and whether a
        job is using each."""
        try:
            return {"runtimes": await run_in_threadpool_bounded(_states,
                                                                timeout=_READ_TIMEOUT_S)}
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Reading the media runtimes timed out: {e}") from e

    @app.post("/api/media/runtimes/unload",
              dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def media_runtimes_unload():
        """Stop every loaded native media runtime no job is using; a busy one
        is left running and reported as such."""
        try:
            return {"outcomes": await run_in_threadpool_bounded(_unload,
                                                                timeout=_UNLOAD_TIMEOUT_S)}
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Stopping the media runtimes timed out: {e}") from e
