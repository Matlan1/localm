# SPDX-License-Identifier: AGPL-3.0-or-later
"""Targeted per-model unload (`POST /v1/models/unload?model=...` and the GUI's
`/api/models/unload`), alongside the existing unload-everything behavior.

localm can hold multiple models loaded simultaneously (`_engines`). These tests
cover the single-model path used by the GUI's per-row Unload button and
`localm unload`, plus the no-body "unload everything" behavior.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.inference import embedder as emb
from localm.inference import http_server as hs
from localm.plugins.gui.web import attach_gui
from tests.conftest import probe_double


class FakeEngine:
    def __init__(self, display_name):
        self.display_name = display_name
        self._loaded = False
        self.active_requests = 0
        self.supports_images = False
        self.model_path = f"Z:/models/{display_name}.gguf"

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def set_load_cancel(self, cancel):
        pass

    def count_tokens(self, text):
        return len(text.split())

    def count_messages_tokens(self, messages):
        return 100

    def context_capacity(self):
        return 4096

    def chat_stream(self, messages, **gen_kwargs):
        yield "Hello from " + self.display_name

    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def setup_multi_model(monkeypatch):
    fake_registry = {
        "model-a": {"path": "Z:/models/model-a.gguf", "source": "local"},
        "model-b": {"path": "Z:/models/model-b.gguf", "source": "local"},
    }
    monkeypatch.setattr("localm.config.load_registry", lambda: fake_registry)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (f"Z:/models/{name}.gguf", "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
    # Plenty of (static, not usage-aware) free VRAM so loading a second model
    # never triggers eviction of the first - both stay resident.
    monkeypatch.setattr("localm.discover.vram_info",
                        probe_double({"free": 10 * 1024 ** 3, "total": 16 * 1024 ** 3}))
    monkeypatch.setattr(hs, "_engine_factory", lambda name: FakeEngine(name))

    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._last_activity_per_model.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None


def _authed_client(app):
    """A TestClient carrying the in-process loopback shell token: the model
    load/unload and GUI model-list routes are MODELS_WRITE/MODELS_READ
    management routes, which in open mode (no LOCALM_API_KEY) require it -
    see test_bearer_auth.py's test_model_unload_with_shell_token."""
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {app.state.shell_token}"
    return client


def _load(client, name):
    r = client.post("/v1/chat/completions", json={
        "model": name, "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert name in hs._engines and hs._engines[name].loaded


def test_unload_targeted_leaves_other_models_loaded(setup_multi_model):
    app = hs.create_app(None)
    client = _authed_client(app)
    _load(client, "model-a")
    _load(client, "model-b")

    r = client.post("/v1/models/unload", params={"model": "model-a"})
    assert r.status_code == 200
    assert r.json()["status"] == "unloaded"
    assert r.json()["model"] == "model-a"

    assert hs._engines["model-a"].loaded is False
    assert hs._engines["model-b"].loaded is True  # untouched


def test_unload_targeted_active_model_clears_active_pointer(setup_multi_model):
    app = hs.create_app(None)
    client = _authed_client(app)
    _load(client, "model-a")
    assert hs._active_model_name == "model-a"

    r = client.post("/v1/models/unload", params={"model": "model-a"})
    assert r.status_code == 200
    assert r.json()["was_active"] is True
    assert hs._active_model_name is None
    assert hs._engine is None


def test_unload_targeted_non_active_model_preserves_active_pointer(setup_multi_model):
    app = hs.create_app(None)
    client = _authed_client(app)
    _load(client, "model-a")
    _load(client, "model-b")  # model-b is now the active one
    assert hs._active_model_name == "model-b"

    r = client.post("/v1/models/unload", params={"model": "model-a"})
    assert r.status_code == 200
    assert r.json()["was_active"] is False
    # Unloading a background (non-active) model must never disturb the one
    # actually serving requests.
    assert hs._active_model_name == "model-b"
    assert hs._engine is not None and hs._engine.display_name == "model-b"


def test_unload_targeted_registered_but_not_loaded_is_idempotent_noop(setup_multi_model):
    app = hs.create_app(None)
    client = _authed_client(app)
    # model-b is registered (in the fake registry) but never loaded.
    r = client.post("/v1/models/unload", params={"model": "model-b"})
    assert r.status_code == 200
    assert r.json()["status"] == "already_unloaded"


def test_unload_targeted_unregistered_model_404s(setup_multi_model):
    app = hs.create_app(None)
    client = _authed_client(app)
    r = client.post("/v1/models/unload", params={"model": "no-such-model"})
    assert r.status_code == 404


def test_unload_no_model_param_still_unloads_everything(setup_multi_model):
    """The no-body/no-param call (used by the image/video/music plugins'
    unload-before-media-gen calls) unloads every loaded model."""
    app = hs.create_app(None)
    client = _authed_client(app)
    _load(client, "model-a")
    _load(client, "model-b")

    r = client.post("/v1/models/unload")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "unloaded"
    assert set(data["unloaded_models"]) == {"model-a", "model-b"}
    assert hs._engines["model-a"].loaded is False
    assert hs._engines["model-b"].loaded is False
    assert hs._active_model_name is None


@pytest.fixture
def gui_app_with_engines(setup_multi_model):
    """A bare attach_gui() app (the GUI plugin route surface, unauthenticated
    like the existing gui_app fixture in test_gui.py) with real FakeEngine
    instances pre-loaded directly into hs._engines - there is no
    /v1/chat/completions route on this bare app to load through, unlike the
    full hs.create_app(None) used for the core /v1/models/unload tests above."""
    app = FastAPI()
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
              switch_model=lambda name: None, active_model=lambda: hs._active_model_name)

    def _load_direct(name):
        hs._engines[name] = hs._engine_factory(name)
        hs._engines[name].load()
        hs._engines_lru.append(name)
        hs._active_model_name = name

    return app, _load_direct


def test_gui_unload_route_targeted(gui_app_with_engines):
    """The GUI's /api/models/unload (JSON body, not query param) mirrors the
    same targeted/all behavior as the core /v1/models/unload."""
    app, load = gui_app_with_engines
    load("model-a")
    load("model-b")
    with patch("localm.config.load_registry", return_value={
        "model-a": {"path": "Z:/models/model-a.gguf"},
        "model-b": {"path": "Z:/models/model-b.gguf"},
    }):
        with TestClient(app) as client:
            r = client.post("/api/models/unload", json={"model": "model-a"})
    assert r.status_code == 200
    assert hs._engines["model-a"].loaded is False
    assert hs._engines["model-b"].loaded is True


def test_gui_unload_route_all(gui_app_with_engines):
    app, load = gui_app_with_engines
    load("model-a")
    load("model-b")
    with TestClient(app) as client:
        r = client.post("/api/models/unload", json={})
    assert r.status_code == 200
    assert set(r.json()["unloaded_models"]) == {"model-a", "model-b"}


def test_gui_unload_route_unregistered_model_404s(gui_app_with_engines):
    app, _load_direct = gui_app_with_engines
    with patch("localm.config.load_registry", return_value={}):
        with TestClient(app) as client:
            r = client.post("/api/models/unload", json={"model": "no-such-model"})
    assert r.status_code == 404


def test_gui_models_list_exposes_loaded_independent_of_active(gui_app_with_engines):
    app, load = gui_app_with_engines
    load("model-a")
    load("model-b")  # active is now model-b; model-a stays loaded
    with patch("localm.config.load_registry", return_value={
        "model-a": {"path": "Z:/models/model-a.gguf"},
        "model-b": {"path": "Z:/models/model-b.gguf"},
    }):
        with TestClient(app) as client:
            r = client.get("/api/models")
    assert r.status_code == 200
    by_name = {m["name"]: m for m in r.json()["models"]}
    assert by_name["model-a"]["active"] is False
    assert by_name["model-a"]["loaded"] is True    # loaded-but-not-active, visible
    assert by_name["model-b"]["active"] is True
    assert by_name["model-b"]["loaded"] is True


# --------------------------------------------------------------------------- #
#  The shared embedder's separate lifecycle also gets released                #
#                                                                               #
#  The embedder (localm.inference.embedder) is loaded by get_embedder(), never #
#  through switch_engine/_engines, so unload_all_models() and                  #
#  unload_one_model() release it explicitly.                                   #
# --------------------------------------------------------------------------- #

def test_unload_all_releases_loaded_embedder(setup_multi_model, monkeypatch):
    """Unload all must also release the shared embedder, not report 'unloaded'
    while its VRAM stays resident."""
    app = hs.create_app(None)
    client = _authed_client(app)
    _load(client, "model-a")

    monkeypatch.setattr(emb, "loaded_dim", lambda: 384)
    calls = []
    monkeypatch.setattr(emb, "reset_embedder", lambda force=True: (calls.append(1), True)[1])

    r = client.post("/v1/models/unload")
    assert r.status_code == 200
    data = r.json()
    assert data["embedder_unloaded"] is True
    assert calls == [1]
    assert hs._engines["model-a"].loaded is False


def test_unload_all_status_unloaded_when_only_embedder_loaded(setup_multi_model, monkeypatch):
    """With no chat model loaded and only the embedder resident, the status is
    'unloaded': iterating only _engines (empty) would report 'already_unloaded'
    while the embedder stayed fully resident."""
    app = hs.create_app(None)
    client = _authed_client(app)

    monkeypatch.setattr(emb, "loaded_dim", lambda: 384)
    calls = []
    monkeypatch.setattr(emb, "reset_embedder", lambda force=True: (calls.append(1), True)[1])

    r = client.post("/v1/models/unload")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "unloaded"
    assert data["embedder_unloaded"] is True
    assert calls == [1]


def test_unload_all_reports_already_unloaded_when_nothing_loaded(setup_multi_model, monkeypatch):
    """Negative control: no chat model AND no embedder -> the original honest
    no-op status is preserved (must not always claim 'unloaded')."""
    app = hs.create_app(None)
    client = _authed_client(app)
    monkeypatch.setattr(emb, "loaded_dim", lambda: None)

    r = client.post("/v1/models/unload")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "already_unloaded"
    assert data["embedder_unloaded"] is False


def test_unload_one_model_releases_matching_embedder(setup_multi_model, monkeypatch):
    """The GUI's per-row Unload button, targeted at a registered model that IS
    the resident embedder (get_embedder() loaded it - never through _engines,
    so the plain _engines.get(name) lookup finds nothing). Matched by resolved
    path, since embedding_model config may resolve via a different route than
    the registry name the row displays."""
    app = hs.create_app(None)
    client = _authed_client(app)

    monkeypatch.setattr(emb, "loaded_path", lambda: "Z:/models/model-a.gguf")
    calls = []
    monkeypatch.setattr(emb, "reset_embedder", lambda force=True: (calls.append(1), True)[1])

    r = client.post("/v1/models/unload", params={"model": "model-a"})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "unloaded"
    assert data["model"] == "model-a"
    assert calls == [1]


def test_unload_one_model_leaves_non_matching_registered_model_alone(setup_multi_model, monkeypatch):
    """A registered-but-never-loaded model whose path does NOT match the
    resident embedder stays an idempotent no-op: the embedder is released only
    when the NAME the caller targeted resolves to it."""
    app = hs.create_app(None)
    client = _authed_client(app)

    monkeypatch.setattr(emb, "loaded_path", lambda: "Z:/models/some-other-embedder.gguf")
    calls = []
    monkeypatch.setattr(emb, "reset_embedder", lambda force=True: (calls.append(1), True)[1])

    r = client.post("/v1/models/unload", params={"model": "model-a"})
    assert r.status_code == 200
    assert r.json()["status"] == "already_unloaded"
    assert calls == []


def test_gui_models_list_reports_embedder_loaded_via_path_match(gui_app_with_engines, monkeypatch):
    """A registered model that is the resident embedder (never appears in
    _engines) must show loaded:true on the Models page - and only that one,
    not a differently-pathed registered model."""
    app, _load_direct = gui_app_with_engines
    monkeypatch.setattr(emb, "loaded_path", lambda: "Z:/models/model-a.gguf")
    with patch("localm.config.load_registry", return_value={
        "model-a": {"path": "Z:/models/model-a.gguf"},
        "model-b": {"path": "Z:/models/model-b.gguf"},
    }):
        with TestClient(app) as client:
            r = client.get("/api/models")
    assert r.status_code == 200
    by_name = {m["name"]: m for m in r.json()["models"]}
    assert by_name["model-a"]["loaded"] is True     # resolved-path match to the embedder
    assert by_name["model-b"]["loaded"] is False     # different path, not loaded anywhere


# --------------------------------------------------------------------------- #
#  CLI: `localm unload [MODEL]`                                               #
# --------------------------------------------------------------------------- #

def _fake_response(status_code=200, payload=None):
    r = MagicMock()
    r.ok = status_code < 400
    r.status_code = status_code
    r.json.return_value = payload or {}
    r.text = ""
    return r


def test_cli_unload_all_uses_discovered_instance(cli_runner, monkeypatch):
    from localm.cli import main as cli_main

    fake_entry = {"host": "127.0.0.1", "port": 8642, "scheme": "http"}
    monkeypatch.setattr("localm.instances.find_attachable", lambda home, root: fake_entry)
    captured = {}

    def fake_post(url, headers=None, params=None, timeout=None, verify=None):
        captured.update(url=url, params=params)
        return _fake_response(payload={"status": "unloaded",
                                       "unloaded_models": ["model-a", "model-b"]})

    with patch("requests.post", side_effect=fake_post), \
         patch("localm.tls.requests_verify", return_value=True):
        r = cli_runner.invoke(cli_main, ["unload"])

    assert r.exit_code == 0, r.output
    assert captured["url"] == "http://127.0.0.1:8642/v1/models/unload"
    assert captured["params"] is None
    assert "model-a" in r.output and "model-b" in r.output


def test_cli_unload_targeted_passes_model_param(cli_runner, monkeypatch):
    from localm.cli import main as cli_main

    fake_entry = {"host": "127.0.0.1", "port": 8642, "scheme": "http"}
    monkeypatch.setattr("localm.instances.find_attachable", lambda home, root: fake_entry)
    captured = {}

    def fake_post(url, headers=None, params=None, timeout=None, verify=None):
        captured.update(params=params)
        return _fake_response(payload={"status": "unloaded",
                                       "unloaded_models": ["model-a"]})

    with patch("requests.post", side_effect=fake_post), \
         patch("localm.tls.requests_verify", return_value=True):
        r = cli_runner.invoke(cli_main, ["unload", "model-a"])

    assert r.exit_code == 0, r.output
    assert captured["params"] == {"model": "model-a"}


def test_cli_unload_targeted_noop_reports_honestly_not_as_success(cli_runner, monkeypatch):
    """A targeted unload of a registered-but-not-loaded model is an idempotent
    no-op server-side ("already_unloaded"): the CLI must say nothing was
    unloaded, never claim a fake success."""
    from localm.cli import main as cli_main

    fake_entry = {"host": "127.0.0.1", "port": 8642, "scheme": "http"}
    monkeypatch.setattr("localm.instances.find_attachable", lambda home, root: fake_entry)

    def fake_post(url, headers=None, params=None, timeout=None, verify=None):
        return _fake_response(payload={"status": "already_unloaded", "model": "model-a"})

    with patch("requests.post", side_effect=fake_post), \
         patch("localm.tls.requests_verify", return_value=True):
        r = cli_runner.invoke(cli_main, ["unload", "model-a"])

    assert r.exit_code == 0, r.output
    assert "unloaded" not in r.output.lower() or "not loaded" in r.output.lower()
    assert "nothing to do" in r.output.lower()


def test_cli_unload_no_running_instance_errors_cleanly(cli_runner, monkeypatch):
    from localm.cli import main as cli_main

    monkeypatch.setattr("localm.instances.find_attachable", lambda home, root: None)
    with patch("requests.post") as post:
        r = cli_runner.invoke(cli_main, ["unload"])
    post.assert_not_called()
    assert r.exit_code != 0
    assert "no running localm server" in r.output.lower()


def test_cli_unload_unauthorized_reports_clearly(cli_runner, monkeypatch):
    from localm.cli import main as cli_main

    fake_entry = {"host": "127.0.0.1", "port": 8642, "scheme": "http"}
    monkeypatch.setattr("localm.instances.find_attachable", lambda home, root: fake_entry)
    with patch("requests.post", return_value=_fake_response(status_code=401)), \
         patch("localm.tls.requests_verify", return_value=True):
        r = cli_runner.invoke(cli_main, ["unload"])
    assert r.exit_code != 0
    assert "unauthorized" in r.output.lower()
