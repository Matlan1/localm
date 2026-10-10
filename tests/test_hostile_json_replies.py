# SPDX-License-Identifier: AGPL-3.0-or-later
"""Replies from a server (the CLI's server, an embeddings endpoint, ComfyUI, the
coder's model server) that hold over-nested or over-long-integer JSON: the client
reports it like any other unusable reply instead of dying on ``RecursionError``.
Each test serves the reply from a real loopback HTTP server and runs the real
client code."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from localm import jsonreply
from tests._hostile_json import BIG_INT, DEEP, HOSTILE, serve


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _outcome(fn):
    try:
        return "returned", fn()
    except Exception as e:
        return type(e).__name__, None


# ------------------------------------------------------------ the helper

@pytest.mark.parametrize("doc", [DEEP, BIG_INT], ids=["deep", "bigint"])
def test_loads_raises_value_error_for_hostile_text(doc):
    with pytest.raises(ValueError):
        jsonreply.loads(doc)


def test_loads_still_parses_ordinary_json():
    assert jsonreply.loads('{"a": [1, 2]}') == {"a": [1, 2]}


@HOSTILE
def test_response_json_raises_value_error_for_a_hostile_body(doc):
    import requests
    with serve(doc) as base:
        with pytest.raises(ValueError):
            jsonreply.response_json(requests.get(base, timeout=10))


# ------------------------------------------------------------ CLI commands

@HOSTILE
def test_unload_with_a_hostile_reply_says_the_outcome_is_unknown(
        doc, cli_env, monkeypatch):
    from localm.cli import main
    with serve({"/v1/models/unload": doc}) as base:
        monkeypatch.setenv("LOCALM_URL", base)
        result = CliRunner().invoke(main, ["unload"])
    assert result.exit_code == 1
    assert "not valid JSON" in result.output


@HOSTILE
def test_rename_with_a_hostile_reply_still_reports_the_rename(
        doc, cli_env, monkeypatch, capsys):
    from localm.cli import models as cli_models
    with serve({"/v1/models/rename": doc}) as base:
        monkeypatch.setenv("LOCALM_URL", base)
        assert cli_models._rename_on_running_server("a", "b") is True
    assert "Renamed" in capsys.readouterr().out


@HOSTILE
def test_rag_cli_embedder_with_a_hostile_reply_raises_value_error(
        doc, cli_env):
    from localm.cli.rag import _cli_rag_embed_fn
    with serve({"/v1/embeddings": doc}) as base:
        embed = _cli_rag_embed_fn(base)
        with pytest.raises(ValueError):
            embed(["text"])


# ------------------------------------------------- plugin server clients

@HOSTILE
def test_self_embed_with_a_hostile_reply_raises_value_error(doc, cli_env):
    from localm.plugins.builtin.rag.plug import _make_self_embed
    with serve({"/v1/embeddings": doc}) as base:
        embed = _make_self_embed(base + "/v1", lambda: "m")
        with pytest.raises(ValueError):
            embed(["text"])


@HOSTILE
def test_coder_load_model_with_a_hostile_reply_raises_value_error(doc):
    from localm.plugins.coder.backends.http import HTTPBackend
    with serve({"/v1/models/load": doc}) as base:
        backend = HTTPBackend(base + "/v1", "m", localm_server=True)
        with pytest.raises(ValueError):
            backend.load_model("m")


@HOSTILE
def test_coder_chat_with_a_hostile_reply_raises_value_error(doc):
    from localm.plugins.coder.backends.http import HTTPBackend
    with serve({"/v1/chat/completions": doc}) as base:
        backend = HTTPBackend(base + "/v1", "m", localm_server=True)
        with pytest.raises(ValueError):
            backend.chat([{"role": "user", "content": "hi"}])


@HOSTILE
def test_comfy_upload_with_a_hostile_reply_raises_value_error(doc, tmp_path):
    from localm.media import comfy_client
    image = tmp_path / "in.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    with serve({"/upload/image": doc}) as base:
        with pytest.raises(ValueError):
            comfy_client._upload_image(image, base)
