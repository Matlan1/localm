# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model search and file listing send the saved HF token and CivitAI key, from
every entry point (CLI, MCP tool, the helpers the GUI routes call), and send
none when nothing is saved or the caller passes ""."""

import pytest
from click.testing import CliRunner

from localm import discover
from localm.model_manager import sources
from localm.model_source_credentials import set_credentials


@pytest.fixture(autouse=True)
def _online(monkeypatch):
    monkeypatch.setenv("LOCALM_NET_MODE", "ask")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("CIVITAI_API_KEY", raising=False)


@pytest.fixture
def headers(monkeypatch):
    """Every request that would leave the machine, as its extra_headers."""
    seen = []

    def fake(url, **kw):
        seen.append(kw.get("extra_headers"))
        if "civitai" in url:
            body = b'{"items": [], "metadata": {}, "files": []}'
        elif "/tree/" in url:
            body = b'[{"type": "file", "path": "m-Q4_K_M.gguf", "size": 1}]'
        else:
            body = b"[]"
        return url, "application/json", body

    monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", fake)
    return seen


_HF = {"Authorization": "Bearer hf_saved"}
_CIV = {"Authorization": "Bearer civ_saved"}


def _save():
    set_credentials({"hf_token": "hf_saved", "civitai_api_key": "civ_saved"})


def test_hf_search_sends_the_saved_token(headers):
    _save()
    discover.hf_search("qwen")
    assert headers and all(h == _HF for h in headers)


def test_hf_gguf_files_sends_the_saved_token(headers):
    _save()
    discover.hf_gguf_files("org/repo")
    assert headers == [_HF]


def test_civitai_search_and_file_list_send_the_saved_key(headers):
    _save()
    sources.civitai_search("x")
    sources.civitai_list_files(135867)
    assert headers == [_CIV, _CIV]


def test_an_empty_token_sends_none_even_with_one_saved(headers):
    _save()
    discover.hf_search("qwen", token="")
    discover.hf_gguf_files("org/repo", token="")
    sources.civitai_search("x", api_key="")
    assert headers and all(h is None for h in headers)


def test_nothing_saved_sends_no_header(headers):
    discover.hf_search("qwen")
    discover.hf_gguf_files("org/repo")
    sources.civitai_search("x")
    assert headers and all(h is None for h in headers)


def test_the_cli_search_sends_the_saved_credentials(headers):
    from localm.cli import main
    _save()
    CliRunner().invoke(main, ["search", "qwen"])
    CliRunner().invoke(main, ["search", "org/repo", "--files"])
    CliRunner().invoke(main, ["search", "x", "--source", "civitai"])
    assert _HF in headers and _CIV in headers
    assert None not in headers


def test_the_mcp_search_tools_send_the_saved_token(headers):
    from localm.plugins.mcpserver.tools import models as mcp_models
    _save()
    tools = mcp_models.build(None)
    tools["search_models"]["handler"]({"query": "qwen"})
    tools["list_model_files"]["handler"]({"repo": "org/repo"})
    assert headers and all(h == _HF for h in headers)


@pytest.fixture
def gui_client(tmp_path, monkeypatch):
    """The GUI app on a throwaway home with an owner key."""
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import localm.config as cfg
    from localm.plugins.engine import attach_engine
    from localm.plugins.gui.web import attach_gui
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setattr(discover, "vram_info",
                        lambda *a, **k: ({}, discover.GPU_PROBE_OK) if k.get("return_status") else {})
    app = FastAPI()
    attach_engine(app)
    attach_gui(app, self_url="http://127.0.0.1:9/v1", switch_model=lambda name: None,
               active_model=lambda: "model-a")
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("path", [
    "/api/discover/search?q=qwen",
    "/api/discover/files?repo=org/repo",
    "/api/discover/search?q=x&source=civitai",
    "/api/discover/files?repo=135867&source=civitai",
])
def test_the_gui_routes_send_the_saved_credentials_for_the_owner_only(
        headers, gui_client, path):
    from localm import auth, scopes
    _save()
    reader = auth.create_key("reader", [scopes.MODELS_READ])["key"]
    want = _CIV if "civitai" in path else _HF

    owner = gui_client.get(path, headers={"Authorization": "Bearer ownersecret"})
    assert owner.status_code == 200, owner.text
    assert headers and all(h == want for h in headers)

    headers.clear()
    other = gui_client.get(path, headers={"Authorization": f"Bearer {reader}"})
    assert other.status_code == 200, other.text
    assert headers and all(h is None for h in headers)
