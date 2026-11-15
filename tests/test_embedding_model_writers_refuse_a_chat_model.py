# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every writer of the ``embedding_model`` setting refuses, through
``embedder.embedding_model_refusal``, a model that cannot be the embedding
model: ``PATCH /v1/config``, ``POST /api/rag/embedding``,
``localm setup-embeddings``, ``localm config`` and the MCP ``setup_embeddings``
tool. A chat-model GGUF given by path is refused; an embedding GGUF given by
path is accepted."""
# ruff: noqa: F811
from __future__ import annotations

import ast
import struct
from pathlib import Path

import pytest

from tests.test_config_admin_gating import _owner as _rag_owner
from tests.test_config_admin_gating import rag_app_env  # noqa: F401
from tests.test_owner_only_settings import _owner, app_env  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
_REFUSED = "not an embedding model"


def _gguf(path: Path, arch: str, pooling=None) -> Path:
    """A GGUF header with ``general.architecture`` = *arch* and, when given, an
    ``<arch>.pooling_type`` = *pooling*; no tensors."""
    def kv(key, vtype, value):
        k = key.encode()
        out = struct.pack("<Q", len(k)) + k + struct.pack("<I", vtype)
        if vtype == 8:
            v = value.encode()
            return out + struct.pack("<Q", len(v)) + v
        return out + struct.pack("<I", value)

    kvs = [("general.architecture", 8, arch)]
    if pooling is not None:
        kvs.append((f"{arch}.pooling_type", 4, pooling))
    body = b"".join(kv(*x) for x in kvs)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + body)
    return path


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_a_chat_model_gguf_path_is_refused_and_an_embedding_gguf_path_is_not(tmp_path):
    from localm.inference.embedder import embedding_model_refusal
    from localm.model_manager.registry import gguf_file_model_type

    chat = _gguf(tmp_path / "chat.gguf", "llama")
    emb = _gguf(tmp_path / "emb.gguf", "bert", pooling=1)
    other = tmp_path / "weights.bin"
    other.write_bytes(b"x" * 64)

    assert gguf_file_model_type(chat) == "llm"
    assert gguf_file_model_type(emb) == "embedding"
    assert gguf_file_model_type(other) is None
    why = embedding_model_refusal(str(chat))
    assert _REFUSED in why and "chat.gguf" in why and "llm" in why
    assert embedding_model_refusal(str(emb)) == ""
    assert embedding_model_refusal(str(other)) == ""


def test_patch_config_refuses_a_chat_model_gguf_path(app_env, tmp_path):
    c, _scoped_key, _ = app_env
    chat = _gguf(tmp_path / "chat.gguf", "llama")
    before = c.get("/v1/config", headers=_owner()).json()["embedding_model"]
    r = c.patch("/v1/config", headers=_owner(),
                json={"embedding_model": str(chat), "confirm": True})
    assert r.status_code == 400, r.text
    assert _REFUSED in r.json()["detail"]
    assert c.get("/v1/config", headers=_owner()).json()["embedding_model"] == before


def test_rag_embedding_route_refuses_a_chat_model_on_the_dry_run_and_the_switch(
        rag_app_env):  # noqa: F811
    from localm.config import load_config
    c, _rag_key, tmp_path = rag_app_env
    chat = _gguf(tmp_path / "chat.gguf", "llama")
    before = load_config().get("embedding_model")
    for confirm in (False, True):
        r = c.post("/api/rag/embedding", headers=_rag_owner(),
                   json={"model": str(chat), "confirm": confirm})
        assert r.status_code == 400, r.text
        assert _REFUSED in r.json()["detail"]
    assert load_config().get("embedding_model") == before


def test_rag_embedding_route_refuses_a_registered_chat_model(rag_app_env):  # noqa: F811
    from localm.config import load_config
    from localm.model_manager import update_registry
    c, _rag_key, tmp_path = rag_app_env
    chat = _gguf(tmp_path / "chat.gguf", "llama")
    update_registry(lambda reg: reg.__setitem__("my-chat", {
        "path": str(chat), "source": "local", "model_type": "llm"}))
    before = load_config().get("embedding_model")
    r = c.post("/api/rag/embedding", headers=_rag_owner(),
               json={"model": "my-chat", "confirm": True})
    assert r.status_code == 400, r.text
    assert "registered as a llm model" in r.json()["detail"]
    assert load_config().get("embedding_model") == before


def _no_install(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must refuse before resolving or downloading anything")
    monkeypatch.setattr("localm.inference.embedder.resolve_embedding_model_path", boom)


@pytest.mark.parametrize("registered", [False, True], ids=["by-path", "registered"])
def test_setup_embeddings_refuses_a_chat_model(cli_runner, monkeypatch, tmp_path, registered):
    from localm.cli.maintenance import setup_embeddings
    from localm.config import load_config
    from localm.model_manager import update_registry
    _no_install(monkeypatch)
    chat = _gguf(tmp_path / "chat.gguf", "llama")
    spec = str(chat)
    if registered:
        update_registry(lambda reg: reg.__setitem__("my-chat", {
            "path": str(chat), "source": "local", "model_type": "llm"}))
        spec = "my-chat"
    before = load_config().get("embedding_model")

    result = cli_runner.invoke(setup_embeddings, ["--model", spec, "--yes"])

    assert result.exit_code == 1, result.output
    out = _flat(result.output)
    assert _REFUSED in out, out
    assert "Embedding model ready" not in out
    assert load_config().get("embedding_model") == before


def test_localm_config_refuses_a_chat_model_as_the_embedding_model(cli_runner, tmp_path):
    from localm.cli.models import config_cmd
    from localm.config import load_config
    chat = _gguf(tmp_path / "chat.gguf", "llama")
    before = load_config().get("embedding_model")

    result = cli_runner.invoke(config_cmd, ["embedding_model", str(chat)])

    assert result.exit_code != 0, result.output
    assert _REFUSED in _flat(result.output)
    assert load_config().get("embedding_model") == before


def test_localm_config_accepts_an_embedding_gguf_path(cli_runner, tmp_path):
    from localm.cli.models import config_cmd
    from localm.config import load_config
    emb = _gguf(tmp_path / "emb.gguf", "bert", pooling=1)

    result = cli_runner.invoke(config_cmd, ["embedding_model", str(emb)])

    assert result.exit_code == 0, result.output
    assert load_config().get("embedding_model") == str(emb)


# Modules that name the embedding_model key next to a config-writing call
# without writing that key: the default value and the settings-schema field.
_NOT_WRITERS = {"localm/config.py", "localm/settings_schema.py"}
_WRITER_NAMES = {"update_config", "set_global_config", "set_global_config_key",
                 "save_config", "validate_update"}


def _embedding_model_writer_modules() -> dict:
    """Modules under localm/ holding the string literal ``"embedding_model"``
    and a config-writing call, mapped to whether they name
    ``embedding_model_refusal``."""
    found = {}
    for path in sorted((REPO_ROOT / "localm").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = list(ast.walk(tree))
        if not any(isinstance(n, ast.Constant) and n.value == "embedding_model"
                   for n in nodes):
            continue
        names = ({n.id for n in nodes if isinstance(n, ast.Name)}
                 | {n.attr for n in nodes if isinstance(n, ast.Attribute)}
                 | {a.name for n in nodes if isinstance(n, ast.ImportFrom)
                    for a in n.names})
        if names & _WRITER_NAMES:
            rel = path.relative_to(REPO_ROOT).as_posix()
            found[rel] = "embedding_model_refusal" in names
    return found


def test_every_writer_of_embedding_model_calls_the_one_refusal():
    found = _embedding_model_writer_modules()
    expected_writers = {
        "localm/cli/maintenance.py", "localm/cli/models.py",
        "localm/inference/routes/config.py",
        "localm/plugins/builtin/rag/plug.py",
        "localm/plugins/mcpserver/tools/models.py",
    }
    assert expected_writers <= set(found), sorted(found)
    missing = sorted(rel for rel, calls in found.items()
                     if not calls and rel not in _NOT_WRITERS)
    assert missing == [], (
        f"these modules write embedding_model without embedding_model_refusal: {missing}")


def test_the_writer_scan_finds_a_module_without_the_refusal(tmp_path, monkeypatch):
    fake_root = tmp_path / "repo"
    pkg = fake_root / "localm"
    pkg.mkdir(parents=True)
    (pkg / "writer.py").write_text(
        "from localm.config import update_config\n"
        "def f(x):\n"
        "    update_config(lambda c: c.update({'embedding_model': x}))\n",
        encoding="utf-8")
    monkeypatch.setitem(globals(), "REPO_ROOT", fake_root)
    assert _embedding_model_writer_modules() == {"localm/writer.py": False}
