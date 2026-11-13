# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm setup-embeddings --model` and `localm config embedding_model` run the
same checks: a model that cannot embed is refused, a switch that would
invalidate collections asks first, and setup-embeddings keeps the setting
unchanged when the model cannot be installed, loaded or tested."""

from localm.cli import main
from localm.config import load_config


class _Embedder:
    def __init__(self, vec):
        self.vec = vec

    def embed(self, texts):
        return [list(self.vec) for _ in texts]


def _register(name, model_type):
    from localm.model_manager import update_registry
    update_registry(lambda reg: reg.__setitem__(name, {
        "path": "/nowhere/x.gguf", "source": "local", "model_type": model_type}))


def _affected_collection():
    from localm.rag.store import Collection, rag_dir
    c = Collection("docs", base=rag_dir()).create()
    c._chunks = [{"source": "doc0.txt", "pos": 0, "text": "alpha"}]
    c._vectors = [[0.1] * 8]
    c._meta["embedding_model"] = "old-model"
    c._save()


def test_setup_embeddings_refuses_a_chat_model(cli_runner, monkeypatch):
    _register("chatty", "llm")
    before = load_config().get("embedding_model")

    res = cli_runner.invoke(main, ["setup-embeddings", "--model", "chatty", "--yes"])

    assert res.exit_code == 1, res.output
    assert "Not switching the embedding model" in res.output
    assert load_config().get("embedding_model") == before


def test_a_failed_install_keeps_the_setting_and_says_why(cli_runner, monkeypatch):
    before = load_config().get("embedding_model")
    monkeypatch.setattr("localm.inference.embedder.resolve_embedding_model_path",
                        lambda **k: None)
    monkeypatch.setattr("localm.inference.embedder.last_error",
                        lambda: "downloads are off (net_mode is 'off')")

    res = cli_runner.invoke(main, ["setup-embeddings", "--model", "no-such-model", "--yes"])

    flat = " ".join(res.output.split())
    assert res.exit_code == 1, res.output
    assert "downloads are off (net_mode is 'off')" in flat
    assert "The embedding model setting stays" in flat
    assert load_config().get("embedding_model") == before


def test_a_model_that_returns_no_vectors_is_not_kept(cli_runner, monkeypatch, tmp_path):
    fake = tmp_path / "m.gguf"
    fake.write_bytes(b"x")
    before = load_config().get("embedding_model")
    monkeypatch.setattr("localm.inference.embedder.resolve_embedding_model_path",
                        lambda **k: str(fake))
    monkeypatch.setattr("localm.inference.embedder.get_embedder",
                        lambda *a, **k: _Embedder([]))

    res = cli_runner.invoke(main, ["setup-embeddings", "--model", "some-model", "--yes"])

    assert res.exit_code == 1, res.output
    assert "did not produce embeddings" in " ".join(res.output.split())
    assert "Embedding model ready" not in res.output
    assert load_config().get("embedding_model") == before


def test_config_embedding_model_refuses_a_chat_model(cli_runner):
    _register("chatty", "llm")
    before = load_config().get("embedding_model")

    res = cli_runner.invoke(main, ["config", "embedding_model", "chatty"])

    assert res.exit_code == 1, res.output
    assert load_config().get("embedding_model") == before


def test_config_embedding_model_names_the_collections_and_asks(cli_runner):
    _affected_collection()
    before = load_config().get("embedding_model")

    declined = cli_runner.invoke(main, ["config", "embedding_model", "new-model"], input="n\n")
    assert declined.exit_code == 1
    assert "docs" in declined.output
    assert load_config().get("embedding_model") == before

    accepted = cli_runner.invoke(main, ["config", "embedding_model", "new-model", "--yes"])
    assert accepted.exit_code == 0, accepted.output
    assert load_config().get("embedding_model") == "new-model"


def test_config_pooling_names_the_collections_with_old_vectors(cli_runner):
    _affected_collection()
    current = load_config().get("embedding_pooling") or ""
    new = "cls" if current != "cls" else "mean"

    res = cli_runner.invoke(main, ["config", "embedding_pooling", new])

    assert res.exit_code == 0, res.output
    flat = " ".join(res.output.split())
    assert "localm rag reembed" in flat
    assert "docs" in flat
