# SPDX-License-Identifier: AGPL-3.0-or-later
"""The three /comfy-models picker endpoints must not run their blocking ComfyUI
/object_info fetch on the event loop.

``imagine_comfy_models`` (image/plug.py), ``music_comfy_models`` and
``video_comfy_models`` are ``async def`` handlers whose backend call reaches
``comfy_client.comfy_object_info()``, a synchronous
``urllib.request.urlopen('/object_info', timeout=10.0)`` that reads and
json.loads the entire body - commonly several MB. Run inline, that read stalls
the whole uvicorn event loop for up to the 10s timeout.

Oracle: ``asyncio.get_running_loop()`` succeeds only on the event-loop thread and
raises RuntimeError in a threadpool worker, so this is structural - no sleeps, no
timing, nothing load-sensitive.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.gui.web import attach_gui

# (plugin name, route) for the three identical pickers.
PICKERS = [
    ("image", "/api/imagine/comfy-models"),
    ("music", "/api/music/comfy-models"),
    ("video", "/api/video/comfy-models"),
]


def _installed_backend(plugin: str):
    """The backend module the INSTALLED plugin actually uses.

    PluginManager loads a plugin under a synthetic package name
    (``_localm_plugin_image``), so ``_localm_plugin_image.backend`` is a
    DIFFERENT module object from ``localm.plugins.builtin.image.backend``.
    Patching the canonical path does nothing: the real _comfy_model_slots then
    runs and returns None (no ComfyUI in the test env), so an "unreachable"
    assertion would pass for the wrong reason.
    """
    import sys
    mod = sys.modules.get(f"_localm_plugin_{plugin}.backend")
    assert mod is not None, (
        f"the installed {plugin} plugin's backend module was not found - the "
        "plugin loader's naming changed and this test would silently patch nothing")
    return mod


def _media_app(tmp_path, monkeypatch, plugin):
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install(plugin)

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app


@pytest.mark.parametrize("plugin,route", PICKERS, ids=[p[0] for p in PICKERS])
def test_comfy_models_does_not_fetch_on_the_event_loop(tmp_path, monkeypatch,
                                                       plugin, route):
    """The blocking /object_info fetch must not run on the loop thread."""
    app = _media_app(tmp_path, monkeypatch, plugin)
    backend = _installed_backend(plugin)

    seen: dict = {}

    def _probing_slots(s):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True      # ON the event-loop thread
        except RuntimeError:
            seen["on_loop"] = False     # off-loop (threadpool worker)
        return ["unet: flux1-dev.safetensors"]

    monkeypatch.setattr(backend, "_comfy_model_slots", _probing_slots)

    with TestClient(app) as client:
        r = client.get(route)
        assert r.status_code == 200, r.text
        assert r.json()["reachable"] is True

    assert seen.get("on_loop") is False, (
        f"{route} ran ComfyUI's multi-MB /object_info fetch ON the event loop: "
        "a slow or hanging ComfyUI would freeze every concurrent request")


@pytest.mark.parametrize("plugin,route", PICKERS, ids=[p[0] for p in PICKERS])
def test_comfy_models_still_reports_unreachable_honestly(tmp_path, monkeypatch,
                                                         plugin, route):
    """The offload does not change the contract: None from the backend still
    means a "not running" answer, never a silently-empty picker."""
    app = _media_app(tmp_path, monkeypatch, plugin)
    backend = _installed_backend(plugin)
    called: list = []
    monkeypatch.setattr(backend, "_comfy_model_slots",
                        lambda s: called.append(1) or None)

    with TestClient(app) as client:
        r = client.get(route)
        assert r.status_code == 200, r.text
        body = r.json()
    assert called, "the patched backend was never reached - this test proved nothing"
    assert body["reachable"] is False
    assert body["slots"] == []
    assert "not running" in body["message"]


@pytest.mark.parametrize("plugin,route", PICKERS, ids=[p[0] for p in PICKERS])
def test_comfy_models_returns_the_slots_it_resolved(tmp_path, monkeypatch,
                                                    plugin, route):
    """What the backend resolved survives the offload unchanged.

    Asserted as a SUBSET rather than by equality, because the route also
    annotates each slot with its localm model_type and the plugin role it
    fills. The annotation itself is pinned separately."""
    app = _media_app(tmp_path, monkeypatch, plugin)
    backend = _installed_backend(plugin)
    slots = [{"node": "4", "input": "ckpt_name", "value": "m.safetensors",
              "options": ["m.safetensors", "n.safetensors"]}]
    monkeypatch.setattr(backend, "_comfy_model_slots", lambda s: slots)

    with TestClient(app) as client:
        body = client.get(route).json()
    assert body["reachable"] is True
    assert len(body["slots"]) == len(slots)
    for got, sent in zip(body["slots"], slots):
        assert {k: got[k] for k in sent} == sent, "the offload altered the payload"
        assert {"model_type", "role_id", "role_label", "installed"} <= set(got)
    assert body["api_url"]
