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
        body = b'{"items": [], "metadata": {}, "files": []}' if "civitai" in url else b"[]"
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
