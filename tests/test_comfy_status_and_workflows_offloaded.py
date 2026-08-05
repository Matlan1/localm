# SPDX-License-Identifier: AGPL-3.0-or-later
"""GET /v1/comfy/status and GET/POST/DELETE /api/{media}/workflows ran
synchronous blocking I/O directly on the event-loop thread - the same REG-638
class already fixed for the /comfy-models pickers (see
test_comfy_models_offloaded_638.py's docstring for the mechanism).

get_comfy_status called _comfy_alive(url, timeout=1.0) - a synchronous
urllib.request.urlopen - inline. Reachable live: bind a real TCP listener that
accepts a connection and never answers, point comfy_api_url at it, and a
concurrent unrelated request stalls for the same ~1s the probe waits out
(verified live against this exact mechanism before writing this test).

The workflows routes called list_workflows()/selected_name() inline: an
uncached directory glob + a stat() per file, plus a config.json read (on
Windows, load_config can hit a documented ~1s antivirus/indexer retry -
config.py's _replace_atomic docstring). selected_name() was ALSO called
twice per request (once inside list_workflows, once again in the route),
doubling that cost.

Oracle (same as test_comfy_models_offloaded_638.py): asyncio.get_running_loop()
succeeds only on the event-loop thread and raises RuntimeError in a
threadpool worker. A probe planted at the blocking call site records which
thread the real work actually ran on - deterministic, no timing, no sleeps.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
#  GET /v1/comfy/status
# --------------------------------------------------------------------------- #

def _keyless_app(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.inference.http_server import create_app
    return create_app(None)


def test_comfy_status_does_not_probe_on_the_event_loop(tmp_path, monkeypatch):
    from localm.image_gen import comfy as ic

    seen: dict = {}

    def _probing_alive(url, timeout=1.0):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True      # ON the event-loop thread: the defect
        except RuntimeError:
            seen["on_loop"] = False     # off-loop (threadpool worker): correct
        return False

    # image_gen.comfy imports _comfy_alive as its OWN module-level name (`from
    # localm.media.comfy_client import _comfy_alive`), a binding SEPARATE from
    # comfy_client's own attribute - the route resolves it via `from
    # localm.image_gen.comfy import _comfy_alive`, so THIS is the name that
    # must be patched (patching comfy_client._comfy_alive is a silent no-op
    # here, confirmed live - see test_stopcomfy_2026_07_01.py's fix).
    monkeypatch.setattr(ic, "_comfy_alive", _probing_alive)

    app = _keyless_app(tmp_path, monkeypatch)
    client = TestClient(app)
    tok = {"Authorization": f"Bearer {app.state.shell_token}"}
    r = client.get("/v1/comfy/status", headers=tok)
    assert r.status_code == 200, r.text
    assert seen.get("on_loop") is False, (
        "GET /v1/comfy/status ran the ComfyUI reachability probe ON the "
        "event loop: a hung/firewalled ComfyUI would freeze every "
        "concurrent request for the full probe timeout")


def test_comfy_status_still_reports_the_probe_result(tmp_path, monkeypatch):
    """Offloading must not change the contract."""
    from localm.image_gen import comfy as ic
    monkeypatch.setattr(ic, "_comfy_alive", lambda url, timeout=1.0: True)
    app = _keyless_app(tmp_path, monkeypatch)
    client = TestClient(app)
    tok = {"Authorization": f"Bearer {app.state.shell_token}"}
    r = client.get("/v1/comfy/status", headers=tok)
    assert r.status_code == 200, r.text
    assert r.json()["alive"] is True


# --------------------------------------------------------------------------- #
#  GET/POST/select/DELETE /api/{media}/workflows
# --------------------------------------------------------------------------- #

def _media_app(tmp_path, monkeypatch, plugin="image"):
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
    from localm.plugins.gui.web import attach_gui
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install(plugin)

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app


def test_list_workflows_route_does_not_read_on_the_event_loop(tmp_path, monkeypatch):
    import localm.media_workflows as mw

    seen: dict = {}
    real = mw._list_and_selected

    def _probing(media):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        return real(media)

    monkeypatch.setattr(mw, "_list_and_selected", _probing)
    app = _media_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        r = client.get("/api/image/workflows")
    assert r.status_code == 200, r.text
    assert seen.get("on_loop") is False, (
        "GET /api/image/workflows ran its directory glob + config.json read "
        "ON the event loop: a slow disk / AV-locked config would freeze "
        "every concurrent request for the duration")


def test_list_workflows_route_only_loads_config_once(tmp_path, monkeypatch):
    """The double load_config() this fix removed, proven directly: exactly
    one selected_name() call (and so one load_config() call THROUGH THIS
    MODULE) for a GET, not two.

    Counts calls to media_workflows.selected_name specifically, not a global
    load_config() total - auth (require_auth_enabled), the plugin engine
    (_enabled_set), and comfy's own default_api_url all call load_config()
    independently of this fix on the same request (measured: 3 such calls on
    a fresh app), and asserting a global count would make this test fail on
    any unrelated future change to those, not just a regression in the
    property this test actually exists to pin - the single load inside
    _list_and_selected."""
    import localm.media_workflows as mw

    calls = []
    real_selected_name = mw.selected_name

    def _counting(media):
        calls.append(media)
        return real_selected_name(media)

    monkeypatch.setattr(mw, "selected_name", _counting)
    app = _media_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        r = client.get("/api/image/workflows")
    assert r.status_code == 200, r.text
    assert len(calls) == 1, (
        f"expected exactly one selected_name() call for one GET, got "
        f"{len(calls)} - the redundant double-read regressed (list_workflows "
        f"re-resolving `active` itself instead of using the value "
        f"_list_and_selected already computed)")


def test_select_workflow_route_does_not_write_on_the_event_loop(tmp_path, monkeypatch):
    import localm.media_workflows as mw

    seen: dict = {}
    real = mw.select_workflow

    def _probing(media, name):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        return real(media, name)

    monkeypatch.setattr(mw, "select_workflow", _probing)
    app = _media_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        r = client.post("/api/image/workflows/select", json={"name": None})
    assert r.status_code == 200, r.text
    assert seen.get("on_loop") is False, (
        "POST /api/image/workflows/select ran select_workflow() ON the "
        "event loop")


def test_delete_workflow_route_does_not_read_on_the_event_loop(tmp_path, monkeypatch):
    import localm.media_workflows as mw

    seen: dict = {}

    def _probing(media, name):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        raise ValueError(f"no such workflow: {name}")   # avoid touching disk

    monkeypatch.setattr(mw, "delete_workflow", _probing)
    app = _media_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        r = client.delete("/api/image/workflows/nope.json")
    assert r.status_code == 400, r.text
    assert seen.get("on_loop") is False, (
        "DELETE /api/image/workflows/{name} ran delete_workflow() ON the "
        "event loop")


def test_upload_workflow_route_does_not_write_on_the_event_loop(tmp_path, monkeypatch):
    import localm.media_workflows as mw

    seen: dict = {}
    real = mw.save_workflow

    def _probing(media, name, content):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        return real(media, name, content)

    monkeypatch.setattr(mw, "save_workflow", _probing)
    app = _media_app(tmp_path, monkeypatch)

    workflow = {"1": {"class_type": "CheckpointLoaderSimple",
                       "inputs": {"ckpt_name": "m.safetensors"}}}
    with TestClient(app) as client:
        r = client.post("/api/image/workflows",
                        json={"name": "test.json", "workflow": workflow})
    assert r.status_code == 200, r.text
    assert seen.get("on_loop") is False, (
        "POST /api/image/workflows ran save_workflow() ON the event loop")


def test_routes_serialize_concurrent_delete_and_list_via_real_http(tmp_path, monkeypatch):
    """The exact scenario an adversarial review confirmed live (2026-08-05): a
    concurrent DELETE racing a GET's listing raised an unhandled
    FileNotFoundError -> 500, because run_in_threadpool let the two requests'
    bodies genuinely interleave on separate OS threads where before they were
    atomic relative to each other on the single event loop.

    Drives the REAL router through REAL concurrent HTTP requests (real
    threads hitting the same TestClient), not a scripted single-threaded
    interleaving - proving the routes themselves acquire _lock_for, not just
    that the helper function does when used correctly in isolation."""
    import threading

    import localm.media_workflows as mw

    app = _media_app(tmp_path, monkeypatch)
    d = mw.workflows_dir("image")
    d.mkdir(parents=True, exist_ok=True)
    names = [f"wf{i}.json" for i in range(15)]
    for n in names:
        (d / n).write_bytes(b'{"1": {"class_type": "X"}}')

    statuses = []
    errors = []
    lock = threading.Lock()

    def _get(client):
        try:
            r = client.get("/api/image/workflows")
            with lock:
                statuses.append(r.status_code)
        except Exception as e:  # noqa: BLE001 - the property under test is "never raises"
            with lock:
                errors.append(e)

    def _delete(client, name):
        try:
            r = client.delete(f"/api/image/workflows/{name}")
            with lock:
                statuses.append(r.status_code)
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(e)

    with TestClient(app, raise_server_exceptions=False) as client:
        threads = [threading.Thread(target=_get, args=(client,)) for _ in range(15)]
        threads += [threading.Thread(target=_delete, args=(client, n)) for n in names]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    assert not errors, f"a request raised instead of returning a response: {errors!r}"
    assert 500 not in statuses, (
        f"a concurrent delete/list race produced a 500 - the routes are not "
        f"actually serialized: {statuses}")
