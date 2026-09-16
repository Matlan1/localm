# SPDX-License-Identifier: AGPL-3.0-or-later
"""The contract of the GUI model routes that ``localm.plugins.gui.routes.models``
registers: the exact ``(method, path)`` set, the scope each route carries, the
host-filesystem gate on the routes that reach the server's disk, the status
code of each refusal, and the field set of each success response.

Every check here is behavioural (a request goes in, a status and a body come
out) or structural on the mounted FastAPI routes; none names a handler or a
private helper, so the routes can move between modules without touching this
file.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from localm import scopes as S
from localm.discover import FREE_SCOPE_DEVICE, GPU_PROBE_OK, GPU_PROBE_TIMEOUT
from localm.inference import http_server as _hs
from localm.plugins.gui.routes import models as models_routes
from tests.conftest import probe_double

_GB = 1024 ** 3

# (method, path) -> the scope its Depends(require_scope(...)) must carry.
EXPECTED_ROUTES = {
    ("GET", "/api/models"): S.MODELS_READ,
    ("POST", "/api/models/scan"): S.MODELS_WRITE,
    ("GET", "/api/models/roles"): S.MODELS_READ,
    ("GET", "/api/models/shortcuts"): S.MODELS_READ,
    ("POST", "/api/models/load"): S.MODELS_WRITE,
    ("POST", "/api/models/unload"): S.MODELS_WRITE,
    ("POST", "/api/embedding/warmup"): S.MODELS_WRITE,
    ("GET", "/api/vram-estimate"): S.MODELS_READ,
    ("GET", "/api/gpus"): S.MODELS_READ,
    ("POST", "/api/models/pull-token/redeem"): S.MODELS_WRITE,
    ("POST", "/api/models/pull"): S.MODELS_WRITE,
    ("POST", "/api/media/{kind}/preflight"): S.MODELS_WRITE,
    ("POST", "/api/models/pull-comfy-source"): S.MODELS_WRITE,
    ("POST", "/api/models/remove"): S.MODELS_WRITE,
    ("POST", "/api/models/alias"): S.MODELS_WRITE,
    ("POST", "/api/models/rename"): S.MODELS_WRITE,
    ("POST", "/api/models/type"): S.MODELS_WRITE,
    ("POST", "/api/models/relocate"): S.MODELS_WRITE,
    ("GET", "/api/discover/search"): S.MODELS_READ,
    ("GET", "/api/discover/files"): S.MODELS_READ,
}

# Routes that ALSO require host filesystem access (an inline require_fs_host
# after the scope dependency), because they read or write a server-side path.
HOST_FS_ROUTES = {
    ("POST", "/api/models/scan"),
    ("POST", "/api/models/relocate"),
    ("POST", "/api/models/pull-comfy-source"),
}

_REGISTRY = {
    "m-a": {"path": "Z:/nonexistent/a.gguf", "source": "local"},
    "m-b": {"path": "Z:/nonexistent/b.gguf", "source": "hf", "model_type": "llm"},
}

_ROW_KEYS = {"name", "source", "size_bytes", "mtime", "active", "loaded",
             "model_type", "architecture", "expert_count"}


class FakeJobs:
    """Records what the routes ask the job manager to start, without spawning
    anything. ``running`` is what snapshot() reports."""

    def __init__(self):
        self.cli_calls = []
        self.fn_calls = []
        self.running = []

    def start_cli(self, kind, cli_args, **kw):
        self.cli_calls.append((kind, list(cli_args), kw))
        return SimpleNamespace(id=f"job-{len(self.cli_calls)}")

    def start_fn(self, kind, fn, **kw):
        self.fn_calls.append((kind, fn, kw))
        return SimpleNamespace(id=f"fnjob-{len(self.fn_calls)}")

    def snapshot(self, visible=None):
        return list(self.running)


class FakeEngine:
    def __init__(self, loaded=True, split=None):
        self.loaded = loaded
        self.active_requests = 0
        self._backend = SimpleNamespace(applied_gpu_split=split)


@pytest.fixture
def engines():
    saved = dict(_hs._engines)
    _hs._engines.clear()
    yield _hs._engines
    _hs._engines.clear()
    _hs._engines.update(saved)


@pytest.fixture
def harness(monkeypatch, engines):
    """A bare app with ONLY the model routes mounted, in open mode (no key
    configured, so every scope and the host-filesystem gate pass), with the
    registry pinned to _REGISTRY and the collaborators recorded."""
    app = FastAPI()
    state = SimpleNamespace(active="m-a", switch_calls=[], switch_result=None,
                            switch_error=None)

    async def switch_model(name, **kw):
        state.switch_calls.append((name, kw))
        if state.switch_error is not None:
            raise state.switch_error
        return state.switch_result

    jobs = FakeJobs()
    registry = {k: dict(v) for k, v in _REGISTRY.items()}
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    ctx = SimpleNamespace(active_model=lambda: state.active,
                          switch_model=switch_model, jobs=jobs)
    models_routes.register(app, ctx)
    return SimpleNamespace(app=app, state=state, jobs=jobs, registry=registry)


def _api_routes(app):
    return [r for r in app.routes if isinstance(r, APIRoute)]


# --------------------------------------------------------------------------- #
#  Route set and scope dependencies                                            #
# --------------------------------------------------------------------------- #

class TestRouteSet:
    def test_register_mounts_exactly_the_expected_routes_once_each(self):
        app = FastAPI()
        ctx = SimpleNamespace(active_model=lambda: "", switch_model=None, jobs=None)
        models_routes.register(app, ctx)
        mounted = [(m, r.path) for r in _api_routes(app) for m in sorted(r.methods)]
        assert sorted(mounted) == sorted(EXPECTED_ROUTES), (
            "the set of (method, path) the model group registers changed")
        assert len(mounted) == len(set(mounted)), "a route is registered twice"

    def test_every_route_carries_exactly_its_scope_dependency(self):
        app = FastAPI()
        ctx = SimpleNamespace(active_model=lambda: "", switch_model=None, jobs=None)
        models_routes.register(app, ctx)
        seen = {}
        for route in _api_routes(app):
            gates = [dep.call for dep in route.dependant.dependencies
                     if getattr(dep.call, "__qualname__", "") == "require_scope.<locals>.dep"]
            for method in route.methods:
                assert len(gates) == 1, (
                    f"{method} {route.path} carries {len(gates)} require_scope "
                    "dependencies, expected exactly one")
                seen[(method, route.path)] = inspect.getclosurevars(gates[0]).nonlocals["scope"]
        assert seen == EXPECTED_ROUTES


# --------------------------------------------------------------------------- #
#  Scope and host-filesystem enforcement, through the real auth stack          #
# --------------------------------------------------------------------------- #

def _hdr(key):
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def scoped(tmp_path, monkeypatch, engines):
    """The full GUI surface on a throwaway home with no owner key, so a created
    scoped key is the only credential in effect. Network, GPU and subprocess
    collaborators are stubbed so a request that passes its gate is answered on
    its merits without leaving the process."""
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

    async def switch_model(name, **kw):
        return {"status": "loaded", "model": name}

    attach_gui(app, self_url="http://127.0.0.1:9/v1", switch_model=switch_model,
               active_model=lambda: "m-a")
    jobs = FakeJobs()
    monkeypatch.setattr("localm.plugins.gui.jobs.JobManager.start_cli",
                        lambda self, *a, **kw: jobs.start_cli(*a, **kw))
    monkeypatch.setattr("localm.plugins.gui.jobs.JobManager.start_fn",
                        lambda self, *a, **kw: jobs.start_fn(*a, **kw))
    monkeypatch.setattr("localm.config.load_registry", lambda: dict(_REGISTRY))
    monkeypatch.setattr("localm.discover.hf_search", lambda *a, **kw: [])
    monkeypatch.setattr("localm.discover.hf_gguf_files", lambda *a, **kw: [])
    monkeypatch.setattr("localm.discover.vram_capacity",
                        probe_double({"total": 8 * _GB}))
    monkeypatch.setattr("localm.discover.list_gpus", probe_double([]))
    monkeypatch.setattr("localm.discover._native_backend_has_vulkan", lambda: False)
    monkeypatch.setattr("localm.inference.embedder.loaded_dim", lambda: None)
    monkeypatch.setattr("localm.media.comfy_client.comfy_object_info",
                        lambda *a, **kw: None)
    return SimpleNamespace(app=app, jobs=jobs)


def _requests_for(app):
    """One well-formed request per route. The redeem request carries a freshly
    minted grant so a 403 can only come from a gate, never from the token."""
    from localm.plugins.gui.web import mint_pull_grant
    token = mint_pull_grant(app, "owner/repo")
    return [
        ("GET", "/api/models", None),
        ("POST", "/api/models/scan", {"dry_run": True}),
        ("GET", "/api/models/roles", None),
        ("GET", "/api/models/shortcuts", None),
        ("POST", "/api/models/load", {"model": "m-b"}),
        ("POST", "/api/models/unload", {"model": "m-b"}),
        ("POST", "/api/embedding/warmup", {}),
        ("GET", "/api/vram-estimate", None),
        ("GET", "/api/gpus", None),
        ("POST", "/api/models/pull-token/redeem", {"spec": "owner/repo", "token": token}),
        ("POST", "/api/models/pull", {"spec": "owner/repo"}),
        ("POST", "/api/media/image/preflight", {}),
        ("POST", "/api/models/pull-comfy-source", {"filename": "not-curated.gguf"}),
        ("POST", "/api/models/remove", {"model": "nope"}),
        ("POST", "/api/models/alias", {"model": "nope", "alias": "x"}),
        ("POST", "/api/models/rename", {"model": "nope", "new_name": "x"}),
        ("POST", "/api/models/type", {"model": "nope", "model_type": "llm"}),
        ("POST", "/api/models/relocate", {"model": "nope", "new_path": "Z:/x.gguf"}),
        ("GET", "/api/discover/search?q=x", None),
        ("GET", "/api/discover/files?repo=a/b", None),
    ]


def _send(client, method, path, body, key):
    kw = {"headers": _hdr(key)}
    if body is not None:
        kw["json"] = body
    return getattr(client, method.lower())(path, **kw)


def _route_key(method, path):
    return (method, path.split("?", 1)[0].replace("/image/", "/{kind}/"))


class TestScopeEnforcement:
    def test_every_route_covered_by_one_request(self, scoped):
        assert {_route_key(m, p) for m, p, _ in _requests_for(scoped.app)} == set(EXPECTED_ROUTES)

    def test_key_without_models_scope_is_403_everywhere_and_no_handler_runs(self, scoped):
        from localm import auth
        narrow = auth.create_key("narrow", [S.MCP], fs_access="host")["key"]
        with TestClient(scoped.app) as c:
            for method, path, body in _requests_for(scoped.app):
                r = _send(c, method, path, body, narrow)
                assert r.status_code == 403, f"{method} {path}: {r.status_code} {r.text}"
        # The minted grant is still unconsumed and no job was started: every
        # refusal happened before its handler ran.
        assert len(scoped.app.state.pull_grants) == 1
        assert scoped.jobs.cli_calls == [] and scoped.jobs.fn_calls == []

    def test_models_read_reaches_read_routes_only(self, scoped):
        from localm import auth
        reader = auth.create_key("reader", [S.MODELS_READ], fs_access="host")["key"]
        with TestClient(scoped.app) as c:
            for method, path, body in _requests_for(scoped.app):
                r = _send(c, method, path, body, reader)
                if EXPECTED_ROUTES[_route_key(method, path)] == S.MODELS_READ:
                    assert r.status_code != 403, f"{method} {path}: {r.text}"
                else:
                    assert r.status_code == 403, f"{method} {path}: {r.status_code}"

    def test_models_write_with_host_fs_reaches_every_write_route(self, scoped):
        from localm import auth
        writer = auth.create_key("writer", [S.MODELS_WRITE], fs_access="host")["key"]
        with TestClient(scoped.app) as c:
            for method, path, body in _requests_for(scoped.app):
                if EXPECTED_ROUTES[_route_key(method, path)] != S.MODELS_WRITE:
                    continue
                r = _send(c, method, path, body, writer)
                assert r.status_code != 403, f"{method} {path}: {r.status_code} {r.text}"
                assert r.status_code < 500, f"{method} {path}: {r.status_code} {r.text}"

    def test_host_fs_routes_refuse_a_write_key_without_host_access(self, scoped):
        from localm import auth
        writer = auth.create_key("writer", [S.MODELS_WRITE])["key"]
        with TestClient(scoped.app) as c:
            for method, path, body in _requests_for(scoped.app):
                key = _route_key(method, path)
                if EXPECTED_ROUTES[key] != S.MODELS_WRITE:
                    continue
                r = _send(c, method, path, body, writer)
                if key in HOST_FS_ROUTES:
                    assert r.status_code == 403, f"{method} {path}: {r.status_code}"
                else:
                    assert r.status_code != 403, f"{method} {path}: {r.status_code}"
        assert "model-scan" not in [k for k, _, _ in scoped.jobs.fn_calls], (
            "a scan ran for a key without host access")

    def test_pull_of_a_host_path_spec_needs_host_fs_access_a_remote_spec_does_not(self, scoped):
        from localm import auth
        writer = auth.create_key("writer", [S.MODELS_WRITE])["key"]
        with TestClient(scoped.app) as c:
            for spec in ("Z:/models/x.gguf", "/srv/models/x.gguf", "~/x.gguf",
                         "../x.gguf", "//srv/share/x.gguf"):
                r = c.post("/api/models/pull", json={"spec": spec}, headers=_hdr(writer))
                assert r.status_code == 403, f"{spec}: {r.status_code}"
            assert scoped.jobs.cli_calls == []
            r = c.post("/api/models/pull", json={"spec": "owner/repo:file.gguf"},
                       headers=_hdr(writer))
        assert r.status_code == 200 and set(r.json()) == {"job_id"}


# --------------------------------------------------------------------------- #
#  Inventory and runtime                                                       #
# --------------------------------------------------------------------------- #

class TestModelsList:
    def test_shape_rows_and_missing_markers(self, harness):
        with TestClient(harness.app) as c:
            data = c.get("/api/models").json()
        assert set(data) == {"models", "active"}
        assert data["active"] == "m-a"
        rows = {row["name"]: row for row in data["models"]}
        assert list(rows) == ["m-a", "m-b"]
        # Neither file exists: both rows are marked missing and carry the
        # registry path back; the one with no recorded type flags its default.
        assert set(rows["m-a"]) == _ROW_KEYS | {"missing", "last_path", "model_type_recorded"}
        assert set(rows["m-b"]) == _ROW_KEYS | {"missing", "last_path"}
        for name, row in rows.items():
            assert row["missing"] is True
            assert row["last_path"] == _REGISTRY[name]["path"]
            assert row["size_bytes"] is None and row["mtime"] is None
            assert row["loaded"] is False
            assert row["model_type"] == "llm"
            assert row["architecture"] is None and row["expert_count"] is None
            assert row["source"] == _REGISTRY[name]["source"]
        assert rows["m-a"]["active"] is True and rows["m-b"]["active"] is False
        assert rows["m-a"]["model_type_recorded"] is False

    def test_type_filter(self, harness):
        with TestClient(harness.app) as c:
            assert [m["name"] for m in c.get("/api/models?type=llm").json()["models"]] == ["m-a", "m-b"]
            assert c.get("/api/models?type=embedding").json()["models"] == []

    def test_loaded_engine_and_active_split_are_reported(self, harness, engines):
        engines["m-a"] = FakeEngine(loaded=True, split=[0.75, 0.25])
        with TestClient(harness.app) as c:
            data = c.get("/api/models").json()
        rows = {row["name"]: row for row in data["models"]}
        assert rows["m-a"]["loaded"] is True and rows["m-b"]["loaded"] is False
        assert data["active_gpu_split"] == [0.75, 0.25]

    def test_resumable_is_reported_only_without_an_active_model(self, harness, monkeypatch):
        monkeypatch.setattr(_hs, "_resolve_unnamed_model_name", lambda: "m-b")
        with TestClient(harness.app) as c:
            assert "resumable" not in c.get("/api/models").json()
            harness.state.active = ""
            data = c.get("/api/models").json()
        assert data["active"] == "" and data["resumable"] == "m-b"


class TestScan:
    def test_dry_run_previews_and_registers_nothing(self, harness, monkeypatch):
        preview = SimpleNamespace(method="folders", counts={"lora": 2, "vae": 1},
                                  already_registered=4)
        seen = {}

        def _preview(workdir=None):
            seen["workdir"] = workdir
            return preview
        monkeypatch.setattr("localm.model_manager.scan.preview_comfy_models", _preview)
        with TestClient(harness.app) as c:
            r = c.post("/api/models/scan", json={"dry_run": True, "workdir": "Z:/comfy"})
        assert r.status_code == 200
        assert r.json() == {"dry_run": True, "method": "folders",
                            "counts": {"lora": 2, "vae": 1},
                            "already_registered": 4, "total_new": 3}
        assert seen["workdir"] == "Z:/comfy"
        assert harness.jobs.fn_calls == []

    def test_dry_run_failure_is_500(self, harness, monkeypatch):
        def _boom(workdir=None):
            raise RuntimeError("no folder")
        monkeypatch.setattr("localm.model_manager.scan.preview_comfy_models", _boom)
        with TestClient(harness.app) as c:
            r = c.post("/api/models/scan", json={"dry_run": True})
        assert r.status_code == 500

    @pytest.mark.parametrize("body", [None, {}, {"workdir": "Z:/comfy"}])
    def test_real_scan_starts_a_job(self, harness, body):
        with TestClient(harness.app) as c:
            r = c.post("/api/models/scan", **({"json": body} if body is not None else {}))
        assert r.status_code == 200
        assert set(r.json()) == {"job_id"}
        assert [k for k, _, _ in harness.jobs.fn_calls] == ["model-scan"]


class TestRolesAndShortcuts:
    def test_roles_without_a_plugin_manager(self, harness):
        with TestClient(harness.app) as c:
            assert c.get("/api/models/roles").json() == {"roles": []}

    def test_roles_come_from_the_plugin_manager(self, harness):
        roles = [{"plugin": "rag", "role": "embedding"}]
        harness.app.state.plugin_manager = SimpleNamespace(get_all_model_roles=lambda: roles)
        with TestClient(harness.app) as c:
            assert c.get("/api/models/roles").json() == {"roles": roles}

    def test_shortcuts_list_every_curated_alias(self, harness):
        from localm.model_manager import MODEL_SHORTCUTS, _SHORTCUT_SIZES
        with TestClient(harness.app) as c:
            data = c.get("/api/models/shortcuts").json()
        assert set(data) == {"shortcuts"}
        assert [s["alias"] for s in data["shortcuts"]] == list(MODEL_SHORTCUTS)
        for s in data["shortcuts"]:
            assert set(s) == {"alias", "spec", "size"}
            assert s["spec"] == MODEL_SHORTCUTS[s["alias"]]
            assert s["size"] == _SHORTCUT_SIZES.get(s["alias"], "")


class TestLoad:
    def test_unregistered_is_404_and_never_switches(self, harness):
        with TestClient(harness.app) as c:
            r = c.post("/api/models/load", json={"model": "nope"})
        assert r.status_code == 404
        assert harness.state.switch_calls == []

    def test_switch_status_is_passed_through(self, harness):
        harness.state.switch_result = {"status": "superseded", "model": "m-b"}
        with TestClient(harness.app) as c:
            r = c.post("/api/models/load", json={"model": "m-b"})
        assert r.status_code == 200
        assert r.json() == {"status": "superseded", "model": "m-b"}
        assert harness.state.switch_calls == [("m-b", {})]

    def test_force_is_forwarded_only_when_asked(self, harness):
        with TestClient(harness.app) as c:
            c.post("/api/models/load", json={"model": "m-b", "force": True})
            c.post("/api/models/load", json={"model": "m-b", "force": False})
        assert harness.state.switch_calls == [("m-b", {"force": True}), ("m-b", {})]

    def test_a_status_less_switch_still_reports_loaded(self, harness):
        harness.state.switch_result = None
        with TestClient(harness.app) as c:
            r = c.post("/api/models/load", json={"model": "m-b"})
        assert r.json() == {"status": "loaded", "model": "m-b"}

    def test_switch_failure_is_500(self, harness):
        harness.state.switch_error = RuntimeError("no vram")
        with TestClient(harness.app) as c:
            r = c.post("/api/models/load", json={"model": "m-b"})
        assert r.status_code == 500


class TestUnload:
    def test_unregistered_is_404(self, harness):
        with TestClient(harness.app) as c:
            assert c.post("/api/models/unload", json={"model": "nope"}).status_code == 404

    def test_registered_but_not_loaded_is_a_noop_success(self, harness):
        with TestClient(harness.app) as c:
            r = c.post("/api/models/unload", json={"model": "m-b"})
        assert r.status_code == 200
        assert r.json() == {"status": "already_unloaded", "model": "m-b"}

    def test_no_model_unloads_everything(self, harness, monkeypatch):
        monkeypatch.setattr("localm.discover.vram_info",
                            probe_double({"free": 4 * _GB, "total": 8 * _GB}))
        with TestClient(harness.app) as c:
            r = c.post("/api/models/unload", json={})
        assert r.status_code == 200
        data = r.json()
        assert data["unloaded_models"] == []
        assert {"status", "model", "unloaded_models", "embedder_unloaded",
                "vram_freed", "vram_before_bytes", "vram_after_bytes"} <= set(data)


class TestEmbeddingWarmup:
    @pytest.mark.parametrize("dim", [None, 384])
    def test_starts_a_warmup_job_whether_or_not_already_warm(self, harness, monkeypatch, dim):
        monkeypatch.setattr("localm.inference.embedder.loaded_dim", lambda: dim)
        with TestClient(harness.app) as c:
            r = c.post("/api/embedding/warmup")
        assert r.status_code == 200
        assert set(r.json()) == {"job_id"}
        assert [k for k, _, _ in harness.jobs.fn_calls] == ["embedding-warmup"]


class TestVramEstimate:
    def test_shape_with_a_trusted_reading(self, harness, monkeypatch):
        monkeypatch.setattr("localm.discover.vram_capacity", probe_double(
            {"free": 6 * _GB, "total": 8 * _GB, "free_scope": FREE_SCOPE_DEVICE}))
        with TestClient(harness.app) as c:
            r = c.get("/api/vram-estimate", params={"model": "m-b", "n_ctx": 2048,
                                                     "n_gpu_layers": 99})
        assert r.status_code == 200
        data = r.json()
        assert set(data) == {"model", "model_bytes", "weights", "kv_cache", "overhead",
                             "needed", "free", "total", "fits", "approximate"}
        assert data["model"] == "m-b" and data["model_bytes"] == 0
        assert data["free"] == 6 * _GB and data["total"] == 8 * _GB
        assert data["fits"] is True and data["approximate"] is True

    def test_defaults_to_the_active_model_and_withholds_a_stale_free(self, harness, monkeypatch):
        monkeypatch.setattr("localm.discover.vram_capacity", probe_double(
            {"free": 6 * _GB, "total": 8 * _GB, "free_scope": FREE_SCOPE_DEVICE},
            status=GPU_PROBE_TIMEOUT))
        with TestClient(harness.app) as c:
            data = c.get("/api/vram-estimate").json()
        assert data["model"] == "m-a"
        assert data["free"] is None and data["fits"] is None and data["total"] == 8 * _GB


class TestGpus:
    def test_shape_from_the_generic_probe(self, harness, monkeypatch):
        gpus = [{"index": 0, "name": "GPU", "total": 8 * _GB, "free": 4 * _GB}]
        monkeypatch.setattr("localm.discover._native_backend_has_vulkan", lambda: False)
        monkeypatch.setattr("localm.discover.list_gpus", probe_double(gpus))
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": 0, "gpu_split_indices": [0, 1]})
        with TestClient(harness.app) as c:
            data = c.get("/api/gpus").json()
        assert data == {"gpus": gpus, "probe_status": GPU_PROBE_OK,
                        "main_gpu_index": 0, "gpu_split_indices": [0, 1]}

    def test_inconclusive_probe_is_labelled(self, harness, monkeypatch):
        monkeypatch.setattr("localm.discover._native_backend_has_vulkan", lambda: False)
        monkeypatch.setattr("localm.discover.list_gpus",
                            probe_double([], status=GPU_PROBE_TIMEOUT))
        with TestClient(harness.app) as c:
            data = c.get("/api/gpus").json()
        assert data["gpus"] == [] and data["probe_status"] == GPU_PROBE_TIMEOUT
        assert "index_space" not in data

    def test_vulkan_build_reports_the_native_index_space(self, harness, monkeypatch):
        native = [{"index": 0, "name": "Vulkan GPU", "total": 8 * _GB}]
        monkeypatch.setattr("localm.discover._native_backend_has_vulkan", lambda: True)
        monkeypatch.setattr("localm.discover.native_gpu_devices", lambda: native)
        monkeypatch.setattr("localm.discover.list_gpus",
                            lambda **kw: pytest.fail("list_gpus must not run"))
        with TestClient(harness.app) as c:
            data = c.get("/api/gpus").json()
        assert data["gpus"] == native and data["probe_status"] == GPU_PROBE_OK
        assert data["index_space"] == "native"


# --------------------------------------------------------------------------- #
#  Acquisition                                                                 #
# --------------------------------------------------------------------------- #

class TestPullTokenRedeem:
    def test_valid_grant_redeems_once(self, harness):
        from localm.plugins.gui.web import mint_pull_grant
        token = mint_pull_grant(harness.app, "owner/repo")
        with TestClient(harness.app) as c:
            r = c.post("/api/models/pull-token/redeem",
                       json={"spec": "owner/repo", "token": token})
            assert r.status_code == 200 and r.json() == {"ok": True}
            r = c.post("/api/models/pull-token/redeem",
                       json={"spec": "owner/repo", "token": token})
            assert r.status_code == 403

    def test_wrong_spec_or_forged_token_is_403(self, harness):
        from localm.plugins.gui.web import mint_pull_grant
        token = mint_pull_grant(harness.app, "owner/repo")
        with TestClient(harness.app) as c:
            assert c.post("/api/models/pull-token/redeem",
                          json={"spec": "other/repo", "token": token}).status_code == 403
            assert c.post("/api/models/pull-token/redeem",
                          json={"spec": "owner/repo", "token": "forged"}).status_code == 403


class TestPull:
    @pytest.mark.parametrize("spec", ["", "   ", "-", "--"])
    def test_empty_or_dash_only_spec_is_400(self, harness, spec):
        with TestClient(harness.app) as c:
            assert c.post("/api/models/pull", json={"spec": spec}).status_code == 400
        assert harness.jobs.cli_calls == []

    def test_invalid_store_and_type_are_400(self, harness):
        with TestClient(harness.app) as c:
            assert c.post("/api/models/pull",
                          json={"spec": "o/r", "store": "link"}).status_code == 400
            assert c.post("/api/models/pull",
                          json={"spec": "o/r", "model_type": "bogus"}).status_code == 400
        assert harness.jobs.cli_calls == []

    def test_a_running_pull_of_the_same_spec_is_409(self, harness):
        harness.jobs.running = [{"kind": "pull", "status": "running",
                                 "label": "Model pull o/r"}]
        with TestClient(harness.app) as c:
            assert c.post("/api/models/pull", json={"spec": "o/r"}).status_code == 409
            assert c.post("/api/models/pull", json={"spec": "o/other"}).status_code == 200

    def test_starts_a_pull_job_with_the_cli_argv(self, harness):
        with TestClient(harness.app) as c:
            r = c.post("/api/models/pull", json={
                "spec": " o/r:file.gguf ", "name": "nick", "mmproj": "mm.gguf",
                "sha256": "ab" * 32, "store": "copy", "model_type": "llm"})
        assert r.status_code == 200 and set(r.json()) == {"job_id"}
        kind, args, kw = harness.jobs.cli_calls[0]
        assert kind == "pull"
        assert args == ["pull", "--name", "nick", "--mmproj", "mm.gguf",
                        "--sha256", "ab" * 32, "--store", "copy", "--type", "llm",
                        "--", "o/r:file.gguf"]
        assert kw["host_label"] == "Model pull o/r:file.gguf"
        assert kw["extra_env"] == {"LOCALM_PROGRESS_JSON": "1",
                                   "HF_HUB_DISABLE_PROGRESS_BARS": "1"}

    def test_a_flag_shaped_spec_travels_after_the_separator(self, harness):
        with TestClient(harness.app) as c:
            c.post("/api/models/pull", json={"spec": "-h"})
        assert harness.jobs.cli_calls[0][1] == ["pull", "--", "-h"]


class TestMediaPreflight:
    def test_unknown_kind_is_404(self, harness):
        with TestClient(harness.app) as c:
            assert c.post("/api/media/bogus/preflight", json={}).status_code == 404

    @pytest.mark.parametrize("bad", ["../x.safetensors", "\\\\srv\\share\\x", "a/../b"])
    def test_unsafe_image_lora_name_is_400(self, harness, bad):
        with TestClient(harness.app) as c:
            r = c.post("/api/media/image/preflight", json={"lora_name": bad})
        assert r.status_code == 400

    @pytest.mark.parametrize("kind", ["image", "video", "music"])
    def test_a_check_that_cannot_run_is_reported_as_unavailable(
            self, harness, monkeypatch, tmp_path, kind):
        mod = {"image": "localm.image_gen.comfy", "video": "localm.video_gen.comfy",
               "music": "localm.music_gen.comfy"}[kind]
        monkeypatch.setattr(f"{mod}.workflow_path", lambda: tmp_path / "missing.json")
        with TestClient(harness.app) as c:
            r = c.post(f"/api/media/{kind}/preflight", json={})
        assert r.status_code == 200
        assert r.json() == {"status": "unavailable", "missing": [],
                            "warning": f"Could not check {kind} models before generating."}

    def test_verified_with_nothing_missing(self, harness, monkeypatch):
        monkeypatch.setattr("localm.media.comfy_client.comfy_object_info",
                            lambda *a, **kw: None)
        with TestClient(harness.app) as c:
            r = c.post("/api/media/image/preflight", json={})
        assert r.status_code == 200
        assert r.json() == {"status": "verified", "missing": [], "warning": ""}

    def test_missing_entries_carry_a_curated_source_or_null(self, harness, monkeypatch):
        slots = [SimpleNamespace(class_type="UnetLoaderGGUF", input_name="unet_name",
                                 filename="flux1-dev-Q8_0.gguf"),
                 SimpleNamespace(class_type="VAELoader", input_name="vae_name",
                                 filename="homemade.safetensors")]
        monkeypatch.setattr("localm.media.comfy_client.describe_missing_models",
                            lambda workflow, api_url: slots)
        with TestClient(harness.app) as c:
            r = c.post("/api/media/image/preflight", json={})
        assert r.status_code == 200
        data = r.json()
        assert set(data) == {"status", "missing", "warning"}
        assert data["status"] == "verified" and data["warning"] == ""
        curated, homemade = data["missing"]
        for entry in (curated, homemade):
            assert set(entry) == {"class_type", "input_name", "filename", "source", "dest_dir"}
        assert curated["filename"] == "flux1-dev-Q8_0.gguf"
        assert set(curated["source"]) == {"repo", "file", "size_bytes", "model_type"}
        assert curated["source"]["repo"] == "city96/FLUX.1-dev-gguf"
        assert curated["source"]["file"] == "flux1-dev-Q8_0.gguf"
        assert homemade["source"] is None and homemade["dest_dir"] is None


class TestPullComfySource:
    def test_uncurated_filename_is_400(self, harness):
        with TestClient(harness.app) as c:
            r = c.post("/api/models/pull-comfy-source", json={"filename": "homemade.gguf"})
        assert r.status_code == 400
        assert harness.jobs.cli_calls == []

    def test_no_destination_folder_is_400(self, harness, monkeypatch):
        monkeypatch.setattr("localm.media.managed_comfy.comfy_models_dest_dir",
                            lambda subfolder, plugin=None: None)
        with TestClient(harness.app) as c:
            r = c.post("/api/models/pull-comfy-source",
                       json={"filename": "flux1-dev-Q8_0.gguf"})
        assert r.status_code == 400
        assert harness.jobs.cli_calls == []

    def test_curated_download_starts_an_unregistered_pull_into_the_folder(
            self, harness, monkeypatch, tmp_path):
        dest = tmp_path / "comfy" / "models" / "unet"
        seen = {}

        def _dest(subfolder, plugin=None):
            seen["subfolder"], seen["plugin"] = subfolder, plugin
            return dest
        monkeypatch.setattr("localm.media.managed_comfy.comfy_models_dest_dir", _dest)
        with TestClient(harness.app) as c:
            r = c.post("/api/models/pull-comfy-source",
                       json={"filename": " flux1-dev-Q8_0.gguf ", "plugin": "image"})
        assert r.status_code == 200 and set(r.json()) == {"job_id"}
        assert seen == {"subfolder": "unet", "plugin": "image"}
        kind, args, kw = harness.jobs.cli_calls[0]
        assert kind == "pull"
        assert args == ["pull", "--type", "diffusion-unet", "--comfy-dest-dir", str(dest),
                        "--no-register", "--", "city96/FLUX.1-dev-gguf:flux1-dev-Q8_0.gguf"]
        assert kw["host_label"] == "Model pull city96/FLUX.1-dev-gguf:flux1-dev-Q8_0.gguf"

    def test_an_unknown_plugin_selector_falls_back_to_no_plugin(self, harness, monkeypatch, tmp_path):
        seen = {}

        def _dest(subfolder, plugin=None):
            seen["plugin"] = plugin
            return tmp_path
        monkeypatch.setattr("localm.media.managed_comfy.comfy_models_dest_dir", _dest)
        with TestClient(harness.app) as c:
            c.post("/api/models/pull-comfy-source",
                   json={"filename": "flux1-dev-Q8_0.gguf", "plugin": "bogus"})
        assert seen == {"plugin": None}


# --------------------------------------------------------------------------- #
#  Registry mutation                                                           #
# --------------------------------------------------------------------------- #

class TestRemove:
    def test_unregistered_is_404(self, harness):
        with TestClient(harness.app) as c:
            assert c.post("/api/models/remove", json={"model": "nope"}).status_code == 404

    def test_active_model_is_409(self, harness):
        with TestClient(harness.app) as c:
            assert c.post("/api/models/remove", json={"model": "m-a"}).status_code == 409
        assert harness.jobs.cli_calls == []

    def test_loaded_model_is_409(self, harness, engines):
        engines["m-b"] = FakeEngine(loaded=True)
        with TestClient(harness.app) as c:
            assert c.post("/api/models/remove", json={"model": "m-b"}).status_code == 409
        assert harness.jobs.cli_calls == []

    def test_a_live_engine_holding_the_file_under_another_name_is_409(self, harness, monkeypatch):
        monkeypatch.setattr(_hs, "loaded_engine_holding_model_file",
                            lambda model, registry=None: SimpleNamespace(key="alias-of-b", reason=None))
        with TestClient(harness.app) as c:
            r = c.post("/api/models/remove", json={"model": "m-b"})
        assert r.status_code == 409
        assert "alias-of-b" in r.json()["detail"]

    def test_removable_model_starts_a_remove_job(self, harness):
        with TestClient(harness.app) as c:
            r = c.post("/api/models/remove", json={"model": "m-b"})
        assert r.status_code == 200 and set(r.json()) == {"job_id"}
        assert harness.jobs.cli_calls[0][:2] == ("remove", ["rm", "m-b", "--yes"])


class TestAlias:
    def test_unregistered_is_404(self, harness):
        with TestClient(harness.app) as c:
            r = c.post("/api/models/alias", json={"model": "nope", "alias": "x"})
        assert r.status_code == 404

    def test_taken_name_is_409(self, harness):
        with TestClient(harness.app) as c:
            r = c.post("/api/models/alias", json={"model": "m-a", "alias": "m-b"})
        assert r.status_code == 409

    def test_success_reports_the_sanitized_alias(self, harness, monkeypatch):
        from localm.model_manager import _sanitize_name
        calls = []
        monkeypatch.setattr("localm.model_manager.alias_model",
                            lambda model, alias: calls.append((model, alias)) or True)
        with TestClient(harness.app) as c:
            r = c.post("/api/models/alias", json={"model": "m-a", "alias": "my alias/1"})
        assert r.status_code == 200
        assert r.json() == {"status": "aliased", "model": "m-a",
                            "alias": _sanitize_name("my alias/1")}
        assert calls == [("m-a", "my alias/1")]

    def test_alias_failure_is_400(self, harness, monkeypatch):
        def _boom(model, alias):
            raise ValueError("bad alias")
        monkeypatch.setattr("localm.model_manager.alias_model", _boom)
        with TestClient(harness.app) as c:
            r = c.post("/api/models/alias", json={"model": "m-a", "alias": "x"})
        assert r.status_code == 400

    def test_a_lost_race_reports_which_precheck_it_lost(self, harness, monkeypatch):
        # Both prechecks passed, so a False from alias_model means a concurrent
        # writer won: the name is now taken (409) or the model is gone (404).
        monkeypatch.setattr("localm.model_manager.alias_model", lambda model, alias: False)
        with TestClient(harness.app) as c:
            assert c.post("/api/models/alias",
                          json={"model": "m-a", "alias": "x"}).status_code == 409

        def _vanish(model, alias):
            harness.registry.pop(model)
            return False
        monkeypatch.setattr("localm.model_manager.alias_model", _vanish)
        with TestClient(harness.app) as c:
            assert c.post("/api/models/alias",
                          json={"model": "m-b", "alias": "x"}).status_code == 404


class TestRename:
    def test_unregistered_is_404(self, harness):
        with TestClient(harness.app) as c:
            r = c.post("/api/models/rename", json={"model": "nope", "new_name": "x"})
        assert r.status_code == 404

    def test_delegates_to_the_single_rename_helper_and_passes_its_body_through(
            self, harness, monkeypatch):
        calls = []

        async def _rename(model, new_name):
            calls.append((model, new_name))
            return {"status": "renamed", "model": model, "new_name": new_name,
                    "notes": ["coder config still names m-a"]}
        monkeypatch.setattr(_hs, "rename_registered_model", _rename)
        with TestClient(harness.app) as c:
            r = c.post("/api/models/rename", json={"model": "m-a", "new_name": "fresh"})
        assert r.status_code == 200
        assert r.json() == {"status": "renamed", "model": "m-a", "new_name": "fresh",
                            "notes": ["coder config still names m-a"]}
        assert calls == [("m-a", "fresh")]


class TestSetType:
    def test_unregistered_is_404_and_bad_type_is_400(self, harness):
        with TestClient(harness.app) as c:
            assert c.post("/api/models/type",
                          json={"model": "nope", "model_type": "llm"}).status_code == 404
            assert c.post("/api/models/type",
                          json={"model": "m-a", "model_type": "bogus"}).status_code == 400

    def test_success_and_failure_shapes(self, harness, monkeypatch):
        calls = []
        monkeypatch.setattr("localm.model_manager.set_model_type",
                            lambda model, mtype: calls.append((model, mtype)) or True)
        with TestClient(harness.app) as c:
            r = c.post("/api/models/type", json={"model": "m-a", "model_type": "embedding"})
            assert r.status_code == 200
            assert r.json() == {"status": "typed", "model": "m-a", "model_type": "embedding"}
            monkeypatch.setattr("localm.model_manager.set_model_type",
                                lambda model, mtype: False)
            assert c.post("/api/models/type",
                          json={"model": "m-a", "model_type": "embedding"}).status_code == 400
        assert calls == [("m-a", "embedding")]


class TestRelocate:
    def test_unregistered_is_404(self, harness):
        with TestClient(harness.app) as c:
            r = c.post("/api/models/relocate", json={"model": "nope", "new_path": "Z:/x.gguf"})
        assert r.status_code == 404

    @pytest.mark.parametrize("bad", ["\\\\srv\\share\\m.gguf", "\\\\?\\C:\\m.gguf"])
    def test_lexically_unsafe_path_is_400_before_any_filesystem_call(self, harness, monkeypatch, bad):
        touched = []
        monkeypatch.setattr("localm.model_manager.registry.relocate_target",
                            lambda p: touched.append(p) or (None, "reached the filesystem"))
        with TestClient(harness.app) as c:
            r = c.post("/api/models/relocate", json={"model": "m-a", "new_path": bad})
        assert touched == [], "the path reached a filesystem call"
        assert r.status_code == 400

    def test_success_reports_the_resolved_path(self, harness, monkeypatch, tmp_path):
        target = tmp_path / "moved.gguf"
        target.write_bytes(b"GGUF")
        monkeypatch.setattr("localm.model_manager.registry.relocate_target",
                            lambda p: (Path(p), None))
        calls = []
        monkeypatch.setattr("localm.model_manager.registry.relocate_model",
                            lambda model, p: calls.append((model, p)) or True)
        with TestClient(harness.app) as c:
            r = c.post("/api/models/relocate", json={"model": "m-a", "new_path": str(target)})
        assert r.status_code == 200
        assert r.json() == {"status": "relocated", "model": "m-a",
                            "path": str(target.resolve())}
        assert calls == [("m-a", str(target))]

    def test_an_unusable_target_is_400_with_its_reason(self, harness, monkeypatch):
        monkeypatch.setattr("localm.model_manager.registry.relocate_target",
                            lambda p: (None, "not a GGUF file"))
        with TestClient(harness.app) as c:
            r = c.post("/api/models/relocate", json={"model": "m-a", "new_path": "Z:/x.gguf"})
        assert r.status_code == 400 and r.json()["detail"] == "not a GGUF file"


# --------------------------------------------------------------------------- #
#  Discovery                                                                   #
# --------------------------------------------------------------------------- #

@pytest.fixture
def trusted_vram(monkeypatch):
    monkeypatch.setattr("localm.discover.vram_capacity", probe_double(
        {"free": 6 * _GB, "total": 8 * _GB, "free_scope": FREE_SCOPE_DEVICE}))


class TestDiscoverSearch:
    def test_hf_shape_with_fit_badges(self, harness, monkeypatch, trusted_vram):
        seen = {}

        def _search(q, limit=20, formats=("gguf",), model_types=None):
            seen.update(q=q, limit=limit, formats=formats, model_types=model_types)
            return [{"repo": "a/b", "size_bytes": 2 * _GB}, {"repo": "c/d"}]
        monkeypatch.setattr("localm.discover.hf_search", _search)
        monkeypatch.setattr("localm.discover.hf_backend_available", lambda: False)
        with TestClient(harness.app) as c:
            r = c.get("/api/discover/search", params={"q": "llama", "limit": 5,
                                                      "formats": "gguf, hf", "types": "LLM,"})
        assert r.status_code == 200
        data = r.json()
        assert set(data) == {"query", "source", "results", "vram", "hf_backend_available"}
        assert data["query"] == "llama" and data["source"] == "hf"
        assert data["hf_backend_available"] is False
        assert data["vram"] == {"total": 8 * _GB, "free": 6 * _GB}
        assert seen == {"q": "llama", "limit": 5, "formats": ["gguf", "hf"],
                        "model_types": ["llm"]}
        assert "fit" in data["results"][0] and "fit" not in data["results"][1]

    def test_civitai_shape(self, harness, monkeypatch):
        seen = {}

        def _search(q, limit=20, types=None, nsfw=False):
            seen.update(q=q, limit=limit, types=types, nsfw=nsfw)
            return {"items": [{"id": 1}], "next_cursor": "c2"}
        monkeypatch.setattr("localm.model_manager.sources.civitai_search", _search)
        with TestClient(harness.app) as c:
            r = c.get("/api/discover/search", params={"q": "x", "source": "civitai",
                                                      "types": "Checkpoint,LORA",
                                                      "nsfw": "true"})
        assert r.status_code == 200
        assert r.json() == {"query": "x", "source": "civitai", "results": [{"id": 1}],
                            "next_cursor": "c2"}
        assert seen == {"q": "x", "limit": 20, "types": ["Checkpoint", "LORA"], "nsfw": True}

    @pytest.mark.parametrize("message,off,status", [
        ("Network access is disabled (net_mode=off).", True, 403),
        ("HuggingFace request failed: timeout", False, 502),
        ("no files match the requested formats", False, 422),
    ])
    def test_discovery_errors_map_to_their_status(self, harness, monkeypatch,
                                                   message, off, status):
        from localm.discover import DiscoverError

        def _search(*a, **kw):
            raise DiscoverError(message, off=off)
        monkeypatch.setattr("localm.discover.hf_search", _search)
        with TestClient(harness.app) as c:
            r = c.get("/api/discover/search", params={"q": "x"})
        assert r.status_code == status
        if off:
            assert "Settings" in r.json()["detail"]
        else:
            assert r.json()["detail"] == message

    def test_civitai_refusal_maps_the_same_way(self, harness, monkeypatch):
        from localm.model_manager.sources import ModelSourceError

        def _search(*a, **kw):
            raise ModelSourceError("net_mode=off", off=True)
        monkeypatch.setattr("localm.model_manager.sources.civitai_search", _search)
        with TestClient(harness.app) as c:
            r = c.get("/api/discover/search", params={"q": "x", "source": "civitai"})
        assert r.status_code == 403


class TestDiscoverFiles:
    def test_hf_shape_splits_projectors_and_badges_fit(self, harness, monkeypatch, trusted_vram):
        files = [{"file": "m-Q4.gguf", "size_bytes": 2 * _GB},
                 {"file": "mmproj-f16.gguf", "size_bytes": 512 * 1024 ** 2}]
        monkeypatch.setattr("localm.discover.hf_gguf_files", lambda repo: [dict(f) for f in files])
        with TestClient(harness.app) as c:
            r = c.get("/api/discover/files", params={"repo": " a/b/ "})
        assert r.status_code == 200
        data = r.json()
        assert set(data) == {"repo", "files", "mmprojs", "vram"}
        assert data["repo"] == "a/b"
        assert [f["file"] for f in data["files"]] == ["m-Q4.gguf"]
        assert [f["file"] for f in data["mmprojs"]] == ["mmproj-f16.gguf"]
        assert all("fit" in f for f in data["files"] + data["mmprojs"])
        assert data["vram"] == {"total": 8 * _GB, "free": 6 * _GB}

    def test_untrusted_reading_withholds_free(self, harness, monkeypatch):
        monkeypatch.setattr("localm.discover.vram_capacity", probe_double(
            {"free": 6 * _GB, "total": 8 * _GB, "free_scope": FREE_SCOPE_DEVICE},
            status=GPU_PROBE_TIMEOUT))
        monkeypatch.setattr("localm.discover.hf_gguf_files", lambda repo: [])
        with TestClient(harness.app) as c:
            data = c.get("/api/discover/files", params={"repo": "a/b"}).json()
        assert data["vram"] == {"total": 8 * _GB}

    def test_civitai_shape(self, harness, monkeypatch):
        seen = {}

        def _files(version_id, include_legacy_formats=False):
            seen.update(version_id=version_id, legacy=include_legacy_formats)
            return [{"file": "x.safetensors", "scan": "clean"}]
        monkeypatch.setattr("localm.model_manager.sources.civitai_list_files", _files)
        with TestClient(harness.app) as c:
            r = c.get("/api/discover/files", params={"repo": " 123 ", "source": "civitai",
                                                     "legacy_formats": "true"})
        assert r.status_code == 200
        assert r.json() == {"repo": "123", "source": "civitai",
                            "files": [{"file": "x.safetensors", "scan": "clean"}]}
        assert seen == {"version_id": " 123 ", "legacy": True}

    def test_unreachable_hub_is_502(self, harness, monkeypatch):
        from localm.discover import DiscoverError

        def _down(repo):
            raise DiscoverError("HuggingFace request failed: timeout")
        monkeypatch.setattr("localm.discover.hf_gguf_files", _down)
        with TestClient(harness.app) as c:
            assert c.get("/api/discover/files", params={"repo": "a/b"}).status_code == 502
