# SPDX-License-Identifier: AGPL-3.0-or-later
"""A download nobody asked for starts only under net_mode=allow.

A background task that needs the embedding model (memory consolidation, recall,
a scheduled job) never downloads it under ``ask`` or ``off``; it runs without
vectors and logs one note naming the model and the action that downloads it.
The user's own one-time download (``allow_download=True``) still works under
``ask``, and ``allow`` still downloads on first use.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from localm import netpolicy
from localm.inference import embedder as emb
from localm.plugins.builtin.memory import plug
from tests._net_config import set_config


@pytest.fixture(autouse=True)
def _fresh_embedder(monkeypatch):
    emb.reset_embedder()
    monkeypatch.setattr(emb, "_DOWNLOAD_SKIP_NOTED", set())
    yield
    emb.reset_embedder()


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """Counts every Hugging Face file download. ``write`` makes a call land a
    file, otherwise it raises like an unreachable hub. The policy's endpoint
    check is answered "allowed" so a download that the mode permits always
    reaches the counter, whatever this machine's DNS does."""
    import huggingface_hub
    state = {"calls": [], "write": False}

    def _download(repo_id, filename, *, local_dir=None, **kw):
        state["calls"].append((repo_id, filename))
        if not state["write"]:
            raise RuntimeError("no network in this test")
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"GGUF")
        return str(out)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    monkeypatch.setattr(netpolicy, "download_refusal", lambda url: None)
    monkeypatch.setattr(emb, "_embeddings_dir", lambda: tmp_path / "embeddings")
    return state


def _memory_home(monkeypatch, tmp_path):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path / "home")
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    monkeypatch.setenv("LOCALM_MODE", "log")


def _consolidation_model(prompt: str) -> str:
    if "Decide the single best action" in prompt:
        return json.dumps({"decision": "ADD", "confidence": 0.9})
    return json.dumps({"facts": [{"fact": "User's name is Ada", "confidence": 0.9}]})


@pytest.mark.parametrize("mode", ["allow", "ask", "off"])
def test_automatic_download_allowed_only_under_allow(monkeypatch, mode):
    set_config(monkeypatch, {"net_mode": mode, "net_allow_model_downloads": True})
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    assert netpolicy.automatic_download_allowed() is (mode == "allow")


def test_automatic_consolidation_under_ask_makes_no_download_attempt(
        monkeypatch, tmp_path, hub):
    set_config(monkeypatch, {"net_mode": "ask"})
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    _memory_home(monkeypatch, tmp_path)

    res = plug.synthesize_memory(_consolidation_model,
                                 conversation="User: my name is Ada")

    assert hub["calls"] == []
    assert any("Ada" in r.text for r in plug._chat_store().all())
    assert res["status"] != "error"
    err = emb.last_error() or ""
    assert "bge-small-en-v1.5" in err and "setup-embeddings" in err


def test_ask_skip_is_noted_once_per_model(monkeypatch, hub):
    set_config(monkeypatch, {"net_mode": "ask"})
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    notes = []
    real = emb.logger

    class _Recorder:
        def __getattr__(self, name):
            return getattr(real, name)

        def info(self, msg, *a, **k):
            notes.append(msg % a if a else msg)

    monkeypatch.setattr(emb, "logger", _Recorder())

    assert emb.get_embedder() is None
    assert emb.get_embedder() is None

    assert hub["calls"] == []
    skip = [n for n in notes if "not auto-downloading" in n]
    assert len(skip) == 1
    assert "bge-small-en-v1.5" in skip[0] and "Download button" in skip[0]


def test_background_task_under_off_never_downloads_even_with_the_exemption(
        monkeypatch, hub):
    set_config(monkeypatch, {"net_mode": "off", "net_allow_model_downloads": True})
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)

    assert emb.get_embedder() is None
    assert hub["calls"] == []


def test_allow_still_downloads_on_first_use(monkeypatch, hub):
    set_config(monkeypatch, {"net_mode": "allow"})
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)

    assert emb.get_embedder() is None
    assert emb.get_embedder() is None
    assert len(hub["calls"]) == 1


def test_user_triggered_download_still_works_under_ask(monkeypatch, hub, tmp_path):
    set_config(monkeypatch, {"net_mode": "ask"})
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.setattr(emb, "_EMBEDDING_PINS", {})
    hub["write"] = True
    repo, filename = emb.KNOWN_EMBEDDING_MODELS["bge-small-en-v1.5"]

    path = emb.resolve_embedding_model_path(allow_download=True)

    assert hub["calls"] == [(repo, filename)]
    assert path == str(tmp_path / "embeddings" / filename)
    assert Path(path).is_file()
