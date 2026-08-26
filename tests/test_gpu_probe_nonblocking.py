# SPDX-License-Identifier: AGPL-3.0-or-later
"""A GUI request that reads GPU state must NOT run the (blocking) GPU driver probe
on the event loop.

The probe is bounded by its own deadline, and the GUI handlers offload it to the
plugin executor. This test covers the offload end to end: while the probe is
blocked in a worker thread, the event loop keeps running coroutines."""

import asyncio
import threading
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from localm.plugins.gui.web import attach_gui


@pytest.fixture
def gui_app(tmp_path, monkeypatch):
    """GUI mounted on a throwaway home with an owner key so MODELS_READ passes."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import attach_engine
    app = FastAPI()
    attach_engine(app)
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None, active_model=lambda: "model-a")
    return app


def _hdr():
    return {"Authorization": "Bearer ownersecret"}


@pytest.mark.anyio
async def test_gpus_endpoint_probe_does_not_block_event_loop(gui_app, monkeypatch):
    probe_entered = threading.Event()
    release = threading.Event()

    def _blocking_probe():
        # Simulate a slow/wedged GPU driver call. If this ran on the event loop,
        # the whole server would freeze here until release.
        probe_entered.set()
        release.wait(3)
        return [{"index": 0, "name": "X", "total": 8, "free": 8}]

    monkeypatch.setattr("localm.discover._list_gpus_probe", _blocking_probe)

    transport = httpx.ASGITransport(app=gui_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        req = asyncio.ensure_future(client.get("/api/gpus", headers=_hdr()))
        try:
            # These awaits only make progress if the event loop is FREE. If the
            # probe ran inline on the loop, the loop would be blocked inside
            # release.wait() and this sleep-loop could not advance until release.
            loop_ran = False
            for _ in range(200):
                if probe_entered.is_set():
                    loop_ran = True
                    break
                await asyncio.sleep(0.01)
            assert loop_ran, "probe never started (executor not reached)"
            assert not req.done(), (
                "the /api/gpus request completed while its probe was still "
                "blocked - the probe was NOT offloaded")
        finally:
            release.set()
        resp = await req

    assert resp.status_code == 200, resp.text
    assert resp.json()["gpus"] == [
        {"index": 0, "name": "X", "total": 8, "free": 8}]
