# SPDX-License-Identifier: AGPL-3.0-or-later
"""/v1/config: the network access mode in force, the embedding_model value
check shared with the MCP setup_embeddings tool, and the owner-only refusal
text for keys that run code."""
from __future__ import annotations

import json

import pytest

from tests.test_owner_only_settings import _owner, _scoped, app_env  # noqa: F401


def _write_registry(tmp_path, entries: dict) -> None:
    (tmp_path / ".localm" / "registry.json").write_text(json.dumps(entries),
                                                        encoding="utf-8")


# ------------------------------------------------------------ net mode

def test_get_config_reports_the_env_net_mode_over_the_stored_one(app_env, monkeypatch):
    c, _scoped_key, _ = app_env
    assert c.patch("/v1/config", headers=_owner(),
                   json={"net_mode": "allow"}).status_code == 200
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    cfg = c.get("/v1/config", headers=_owner()).json()
    assert cfg["net_mode"] == "allow"
    assert cfg["effective_net_mode"] == "off"
    assert cfg["net_mode_from_env"] is True


def test_get_config_without_the_env_reports_the_stored_net_mode(app_env, monkeypatch):
    c, _scoped_key, _ = app_env
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    assert c.patch("/v1/config", headers=_owner(),
                   json={"net_mode": "allow"}).status_code == 200
    cfg = c.get("/v1/config", headers=_owner()).json()
    assert cfg["effective_net_mode"] == "allow"
    assert cfg["net_mode_from_env"] is False


def test_patch_ignores_the_read_only_net_mode_extras(app_env):
    c, _scoped_key, _ = app_env
    r = c.patch("/v1/config", headers=_owner(),
                json={"effective_net_mode": "off", "net_mode_from_env": True})
    assert r.status_code == 200, r.text


# ------------------------------------------------------- embedding_model

def test_embedding_model_naming_nothing_is_refused_and_not_saved(app_env, tmp_path):
    c, _scoped_key, _ = app_env
    missing = str(tmp_path / "no-such" / "model.gguf")
    r = c.patch("/v1/config", headers=_owner(), json={"embedding_model": missing})
    assert r.status_code == 400
    assert "names no registered model" in r.json()["detail"]
    assert c.get("/v1/config", headers=_owner()).json().get("embedding_model") != missing


def test_embedding_model_naming_a_chat_model_is_refused(app_env, tmp_path):
    c, _scoped_key, _ = app_env
    _write_registry(tmp_path, {"chatty": {"path": str(tmp_path / "c.gguf"),
                                          "model_type": "llm"}})
    r = c.patch("/v1/config", headers=_owner(), json={"embedding_model": "chatty"})
    assert r.status_code == 400
    assert "registered as a llm model" in r.json()["detail"]
    assert "localm set-type chatty embedding" in r.json()["detail"]


def test_embedding_model_known_key_and_embedding_entry_are_accepted(app_env, tmp_path):
    c, _scoped_key, _ = app_env
    weights = tmp_path / "e.gguf"
    weights.write_bytes(b"not a real gguf")
    _write_registry(tmp_path, {"emb": {"path": str(weights),
                                       "model_type": "embedding", "reranker": False}})
    for value in ("bge-small-en-v1.5", "emb"):
        r = c.patch("/v1/config", headers=_owner(),
                    json={"embedding_model": value, "confirm": True})
        assert r.status_code == 200, r.text


def test_embedding_model_existing_gguf_path_is_accepted(app_env, tmp_path, monkeypatch):
    c, _scoped_key, _ = app_env
    weights = tmp_path / "mine.gguf"
    weights.write_bytes(b"x")
    import localm.model_manager.registry as reg
    monkeypatch.setattr(reg, "gguf_reranker_state", lambda path: False)
    r = c.patch("/v1/config", headers=_owner(),
                json={"embedding_model": str(weights), "confirm": True})
    assert r.status_code == 200, r.text


def test_mcp_setup_embeddings_refuses_a_chat_model(app_env, tmp_path, monkeypatch):
    _c, _scoped_key, _ = app_env
    _write_registry(tmp_path, {"chatty": {"path": str(tmp_path / "c.gguf"),
                                          "model_type": "llm"}})
    import localm.config_write as cw
    writes: list = []
    monkeypatch.setattr(cw, "set_global_config_key",
                        lambda *a, **k: writes.append((a, k)))
    from localm.plugins.mcpserver.server import EngineCache
    from localm.plugins.mcpserver.tools import models
    tool = models.build(EngineCache())["setup_embeddings"]["handler"]
    result = tool({"model": "chatty"})
    assert result.get("isError") is True
    assert "registered as a llm model" in result["content"][0]["text"]
    assert writes == []


# ---------------------------------------------------------- 403 text

def test_owner_only_refusal_names_code_the_server_may_run(app_env):
    c, scoped, _ = app_env
    r = c.patch("/v1/config", headers=_scoped(scoped),
                json={"hf_trust_remote_code": True})
    assert r.status_code == 403
    assert "programs and model code it may run" in r.json()["detail"]
