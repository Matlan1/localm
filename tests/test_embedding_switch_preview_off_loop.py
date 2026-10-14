# SPDX-License-Identifier: AGPL-3.0-or-later
"""The embedding-switch dry-run preview must not parse a collection on the event
loop. ``collection_provenance_report`` falls back to a full ``Collection`` load
for a collection whose ``_stats_cache`` is missing, so both callers in async
handlers (``POST /api/rag/embedding`` and ``PATCH /v1/config``) run it on a
worker thread."""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app
from localm.rag.store import Collection

OWNER_KEY = "owner-key-embedding-preview-offloop"


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
    with TestClient(create_app(None)) as c:
        yield c, home


def _cold_collection(home):
    """A collection with vectors whose meta.json has no ``_stats_cache``, so
    ``peek_stats`` answers None and the report must load it."""
    base = home / "rag"
    coll = Collection("cold", base=base).create()
    coll._chunks = [{"text": "wind energy", "source": "f.txt", "pos": 0}]
    coll._vectors = [[1.0, 0.0, 0.0]]
    coll._save()
    meta_path = base / "cold" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.pop("_stats_cache", None)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert Collection.peek_stats("cold", base=base) is None
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


def test_rag_embedding_set_preview_loads_cold_collection_off_the_loop(client, load_spy):
    c, home = client
    _cold_collection(home)
    load_spy.clear()
    r = c.post("/api/rag/embedding", headers=_owner(), json={"model": "some-other-model"})
    assert r.status_code == 200, r.text
    assert r.json().get("needs_confirm") is True
    assert load_spy, "the cold collection was never loaded; the preview did not reach it"
    assert not any(load_spy), "Collection._load ran on the event loop thread"


def test_patch_config_embedding_preview_loads_cold_collection_off_the_loop(client, load_spy):
    c, home = client
    _cold_collection(home)
    load_spy.clear()
    r = c.patch("/v1/config", headers=_owner(), json={"embedding_model": "some-other-model"})
    assert r.status_code == 200, r.text
    assert r.json().get("needs_confirm") is True
    assert load_spy, "the cold collection was never loaded; the preview did not reach it"
    assert not any(load_spy), "Collection._load ran on the event loop thread"
