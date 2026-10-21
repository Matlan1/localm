# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Knowledge write routes that open an existing collection (add, upload,
re-embed, repair, remove-document) do it on a worker thread: on a snapshot-cache
miss the open parses chunks.jsonl and vectors.json. The credential write of
PATCH /v1/config takes a cross-process lock that waits in a blocking sleep, so it
runs on a worker thread too."""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app
from localm.rag.store import Collection

OWNER_KEY = "owner-key-rag-write-routes-off-loop"


def _owner():
    return {"Authorization": f"Bearer {OWNER_KEY}"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setenv("LOCALM_API_KEY", OWNER_KEY)
    app = create_app(None)
    from localm.plugins.engine import PluginManager
    PluginManager(app, external_root=tmp_path / "noplugins").install("rag")
    with TestClient(app) as c:
        yield c, home


def _cold_collection(home):
    base = home / "rag"
    coll = Collection("cold", base=base).create()
    coll._chunks = [{"text": "wind energy", "source": "f.txt", "pos": 0}]
    coll._vectors = [[1.0, 0.0, 0.0]]
    coll._save()
    meta_path = base / "cold" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.pop("_stats_cache", None)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return base


@pytest.fixture
def load_spy(monkeypatch):
    calls = []
    orig = Collection._load

    def spy(self, *args, **kwargs):
        try:
            asyncio.get_running_loop()
            on_loop = True
        except RuntimeError:
            on_loop = False
        calls.append(on_loop)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(Collection, "_load", spy)
    return calls


@pytest.mark.parametrize("path,body", [
    ("add", {"paths": []}),
    ("upload", {"files": []}),
    ("reembed", None),
    ("repair", {"embed": False, "confirm": True}),
    ("remove-doc", {"path": "nowhere.txt"}),
])
def test_opening_a_cold_collection_runs_off_the_event_loop(client, load_spy, path, body):
    c, home = client
    _cold_collection(home)
    load_spy.clear()
    kwargs = {"headers": _owner()}
    if body is not None:
        kwargs["json"] = body
    r = c.post(f"/api/rag/collections/cold/{path}", **kwargs)
    assert r.status_code in (200, 400, 404), r.text
    assert load_spy, f"{path}: the cold collection was never opened"
    assert not any(load_spy), f"{path}: Collection._load ran on the event loop thread"


def test_the_credential_write_of_a_config_save_runs_off_the_event_loop(client, monkeypatch):
    c, _home = client
    seen = []

    def spy(updates):
        try:
            asyncio.get_running_loop()
            seen.append("loop")
        except RuntimeError:
            seen.append("thread")

    monkeypatch.setattr("localm.model_source_credentials.set_credentials", spy)
    r = c.patch("/v1/config", headers=_owner(), json={"hf_token": "hf_abcdefghijklmnop"})
    assert r.status_code == 200, r.text
    assert seen == ["thread"]
