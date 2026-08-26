# SPDX-License-Identifier: AGPL-3.0-or-later
"""NEW-COMFY-URL-SANITIZE-DNS-ON-THE-LOOP: settings()/default_api_url()/
comfy_models_dest_dir() reach sanitize_comfy_url's blocking getaddrinfo
(comfy_client._host_is_link_local), on 11 ``async def`` routes across the
image/music/video plugins and the GUI models routes. Not reachable with the
DEFAULT comfy_api_url (an IP literal, which ``ipaddress.ip_address`` parses
without touching DNS) - only a HOSTNAME value reaches it, so every test here
configures one.

Oracle: ``asyncio.get_running_loop()`` succeeds only on the event-loop thread
and raises RuntimeError anywhere else (a threadpool worker, or plain
synchronous code with no loop at all) - the same structural, non-timing oracle
``tests/test_comfy_models_offloaded_638.py`` and
``tests/test_grammar_validation_offload.py`` already use for this defect
class. ``_host_is_link_local`` is patched once, at its single canonical
module, rather than per-plugin: unlike a media backend (which a
PluginManager-installed plugin loads under a synthetic module name, see
``_installed_backend`` below), it is never re-imported into a plugin's own
namespace - every caller looks it up in ``comfy_client``'s own globals at call
time, so one patch there is reached from image, music, video and the GUI
routes alike.

Every test asserts the probe fired BEFORE asserting where, so a test that
never reached the sink cannot pass vacuously.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _sink_probe(monkeypatch):
    """Patch comfy_client._host_is_link_local to record, per call, whether it
    ran on a running asyncio event loop. Returns the calls list. Always
    answers "not link-local" so the caller's URL passes through unchanged.

    Some backends' settings() sanitize twice per call (default_api_url()'s own
    guard, then settings()'s "sanitize the resolved value" defense-in-depth
    step - see music/video backend.py, which never checks the legacy global
    comfy_api_url directly and so always falls through default_api_url()).
    That is a pre-existing, harmless redundancy in settings() itself, not a
    property of WHERE it runs - so callers assert every recorded call is
    off-loop, never an exact count.
    """
    from localm.media import comfy_client as cc

    calls: list[bool] = []

    def _probe(host):
        try:
            asyncio.get_running_loop()
            calls.append(True)
        except RuntimeError:
            calls.append(False)
        return False

    monkeypatch.setattr(cc, "_host_is_link_local", _probe)
    return calls


# --------------------------------------------------------------------------- #
#  imagine / music / video: settings() must be deferred into the job thread    #
# --------------------------------------------------------------------------- #

class _FakeJob:
    def __init__(self):
        self.lines: list = []

    def push(self, ev):
        self.lines.append(ev)


class _FakeJobs:
    def __init__(self):
        self.captured = {}

    def start_fn(self, kind, fn, *, result_path=None, owner=None, label=None):
        self.captured["fn"] = fn
        return MagicMock(id="job1")


def _fake_request():
    request = MagicMock()
    request.app.state.jobs = _FakeJobs()
    request.app.state.self_url = "http://127.0.0.1:8642/v1"
    request.app.state.instance_token = None
    request.headers = {}
    request.cookies = {}
    return request


def _settings_reaching_the_sink(api_url="http://offload-probe.example:8188"):
    """A stand-in for backend.settings() that calls the REAL sanitize step (so
    the sink probe above actually fires) without needing the rest of
    settings()'s machinery (media_config.resolve_config, swap-policy,
    vram-estimate) to tolerate a bare-bones config dict."""
    def _settings(cfg):
        from localm.media import comfy_client as cc
        sanitized, _warning = cc.sanitize_comfy_url_checked(api_url)
        return {"reload_after": False, "warning": "", "api_url": sanitized}
    return _settings


_GENERATE_CASES = [
    ("image", "imagine", "ImagineRequest", {"prompt": "a cat"}),
    ("music", "music", "MusicRequest", {"tags": "lofi"}),
    ("video", "video", "VideoRequest", {"prompt": "a cat"}),
]


@pytest.mark.parametrize("plugin,handler_name,req_cls_name,req_kwargs",
                         _GENERATE_CASES, ids=[c[0] for c in _GENERATE_CASES])
def test_generate_route_defers_settings_to_the_job_thread(
        monkeypatch, plugin, handler_name, req_cls_name, req_kwargs):
    plug = __import__(f"localm.plugins.builtin.{plugin}.plug",
                      fromlist=["plug"])
    calls = _sink_probe(monkeypatch)
    monkeypatch.setattr(plug._backend, "settings", _settings_reaching_the_sink())
    # Stop _generate immediately after settings() resolves - ensure_available
    # returning False takes the closure's early-return path (job.push + return
    # False), so nothing past it (vram/placement/generate) needs mocking.
    monkeypatch.setattr(plug._backend, "ensure_available",
                        lambda s, on_progress=None: (False, "stopping here"))

    request = _fake_request()
    req_cls = getattr(plug, req_cls_name)
    handler = getattr(plug, handler_name)
    asyncio.run(handler(req_cls(**req_kwargs), request))

    assert calls == [], (
        f"{handler_name}() called settings() (and so sanitize_comfy_url) "
        "directly on its own coroutine, before the job even started - a slow "
        "or hanging DNS lookup here stalls the whole server, not just this "
        "request")

    # Prove the probe is not vacuous: run the captured job function for real
    # (outside any event loop, exactly like a real JobManager worker thread)
    # and confirm the sink DOES fire there.
    request.app.state.jobs.captured["fn"](_FakeJob())
    assert calls == [False], (
        "the sink was never reached at all by the job function either - this "
        "test could not have failed on the assertion above")


# --------------------------------------------------------------------------- #
#  {imagine,music,video}/comfy-models and comfy-launch: settings() itself is  #
#  now offloaded alongside the calls REG-638 already offloaded                #
# --------------------------------------------------------------------------- #

def _installed_backend(plugin: str):
    """The backend module the INSTALLED plugin actually uses (see
    test_comfy_models_offloaded_638.py's identical helper and its docstring:
    PluginManager loads a plugin under a synthetic package name, so patching
    the canonical localm.plugins.builtin.<x>.backend silently patches nothing
    the installed plugin's own route code sees)."""
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
    cfg = _cfg.load_config()
    cfg["comfy_api_url"] = "http://offload-probe.example:8188"
    _cfg.save_config(cfg)

    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install(plugin)

    async def switch_model(name):
        pass

    from localm.plugins.gui.web import attach_gui
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app


_MODELS_ROUTES = [
    ("image", "/api/imagine/comfy-models"),
    ("music", "/api/music/comfy-models"),
    ("video", "/api/video/comfy-models"),
]
_LAUNCH_ROUTES = [
    ("image", "/api/imagine/comfy-launch"),
    ("music", "/api/music/comfy-launch"),
    ("video", "/api/video/comfy-launch"),
]


@pytest.mark.parametrize("plugin,route", _MODELS_ROUTES, ids=[p[0] for p in _MODELS_ROUTES])
def test_comfy_models_route_offloads_settings_dns_lookup(
        tmp_path, monkeypatch, plugin, route):
    app = _media_app(tmp_path, monkeypatch, plugin)
    backend = _installed_backend(plugin)
    calls = _sink_probe(monkeypatch)
    monkeypatch.setattr(backend, "_comfy_model_roles",
                        lambda s, roles: {"reachable": False, "slots": [], "roles": []})
    if hasattr(backend, "_comfy_lora_options"):    # image only
        monkeypatch.setattr(backend, "_comfy_lora_options", lambda s: [])

    with TestClient(app) as client:
        r = client.get(route)
    assert r.status_code == 200, r.text

    assert calls, (
        f"{route} never reached sanitize_comfy_url at all - this test cannot "
        "say anything about where it runs")
    assert all(c is False for c in calls), (
        f"{route}'s settings() call ran sanitize_comfy_url's blocking "
        f"getaddrinfo ON the event loop: {calls}")


@pytest.mark.parametrize("plugin,route", _LAUNCH_ROUTES, ids=[p[0] for p in _LAUNCH_ROUTES])
def test_comfy_launch_route_offloads_settings_dns_lookup(
        tmp_path, monkeypatch, plugin, route):
    app = _media_app(tmp_path, monkeypatch, plugin)
    backend = _installed_backend(plugin)
    calls = _sink_probe(monkeypatch)
    monkeypatch.setattr(backend, "ensure_available",
                        lambda s, on_progress=None: (True, "ComfyUI is up."))

    with TestClient(app) as client:
        r = client.post(route)
    assert r.status_code == 200, r.text

    assert calls, (
        f"{route} never reached sanitize_comfy_url at all - this test cannot "
        "say anything about where it runs")
    assert all(c is False for c in calls), (
        f"{route}'s settings() call ran sanitize_comfy_url's blocking "
        f"getaddrinfo ON the event loop: {calls}")


# --------------------------------------------------------------------------- #
#  GUI routes: /api/media/{kind}/preflight and /api/models/pull-comfy-source  #
# --------------------------------------------------------------------------- #

@pytest.fixture
def scoped_app(tmp_path, monkeypatch):
    """Mirrors test_gui_comfy_pull_routes.py's fixture of the same name."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import attach_engine
    from localm.plugins.gui.web import attach_gui
    app = FastAPI()
    attach_engine(app)

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app


def _set_hostname_comfy_config(tmp_path, *, workdir_name="external-comfy"):
    import localm.config as _cfg
    cfg = _cfg.load_config()
    cfg["comfy_api_url"] = "http://offload-probe.example:8188"
    cfg["comfy_workdir"] = str(tmp_path / workdir_name)
    _cfg.save_config(cfg)


def test_preflight_route_offloads_settings_dns_lookup(
        scoped_app, tmp_path, monkeypatch):
    """The missing slot must resolve to a CURATED source (not the null-source
    case tests/test_gui_comfy_pull_routes.py also covers): comfy_models_dest_dir()
    - the previously-buggy call - only runs inside the `if source is not None`
    branch. A fixture with no curated match would never reach it at all,
    passing whether or not the offload fix is present (proven by fires-control:
    an earlier version of this test used the uncurated fixture and stayed
    green with the fix reverted)."""
    _set_hostname_comfy_config(tmp_path)
    calls = _sink_probe(monkeypatch)

    fake_info = {
        "UnetLoaderGGUF": {
            "input": {"required": {"unet_name": [["other.gguf"], {}]}}
        }
    }
    fake_wf = tmp_path / "wf.json"
    fake_wf.write_text(json.dumps({
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": "flux1-dev-Q8_0.gguf"}}
    }))

    from localm.media import comfy_client as cc
    with patch.object(cc, "comfy_object_info", return_value=fake_info), \
        patch("localm.image_gen.comfy.workflow_path", return_value=fake_wf):
        with TestClient(scoped_app) as c:
            r = c.post("/api/media/image/preflight", json={})
    assert r.status_code == 200, r.text
    missing = r.json()["missing"]
    assert missing and missing[0]["source"] is not None, (
        "the fixture did not resolve a curated source - this test cannot "
        "reach comfy_models_dest_dir() at all, the call this test exists to "
        "check")

    assert calls, (
        "preflight never reached sanitize_comfy_url at all - this test "
        "cannot say anything about where it runs")
    assert all(c is False for c in calls), (
        "preflight's resolve_comfy_target() call ran sanitize_comfy_url's "
        f"blocking getaddrinfo ON the event loop: {calls}")


def test_pull_comfy_source_route_offloads_settings_dns_lookup(
        scoped_app, tmp_path, monkeypatch):
    _set_hostname_comfy_config(tmp_path)
    calls = _sink_probe(monkeypatch)

    with TestClient(scoped_app) as c:
        r = c.post("/api/models/pull-comfy-source",
                   json={"filename": "ae.safetensors"})
    assert r.status_code == 200, r.text
    assert "job_id" in r.json()

    assert calls, (
        "pull-comfy-source never reached sanitize_comfy_url at all - this "
        "test cannot say anything about where it runs")
    assert all(c is False for c in calls), (
        "pull-comfy-source's comfy_models_dest_dir() call ran "
        f"sanitize_comfy_url's blocking getaddrinfo ON the event loop: {calls}")
