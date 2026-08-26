# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm rag query` is lexical-only by default; `--embed` opts into hybrid
retrieval by embedding the query via a running localm server, the same as the
GUI. The wiring under test: --embed passes an embed_fn to Collection.query; the
default does not.
"""

import localm.rag.store as store


def _patch_query(monkeypatch):
    captured = {}
    monkeypatch.setattr(store.Collection, "exists", lambda self: True)
    monkeypatch.setattr(
        store.Collection, "query",
        lambda self, text, k=4, embed_fn=None: captured.update(embed_fn=embed_fn) or [])
    return captured


def test_rag_query_embed_passes_embed_fn(cli_runner, monkeypatch):
    # --embed is a known option, so the invocation exits zero.
    from localm.cli import main
    captured = _patch_query(monkeypatch)
    r = cli_runner.invoke(
        main, ["rag", "query", "kb", "hello", "--embed", "--url",
               "http://127.0.0.1:9999"])
    assert r.exit_code == 0, r.output
    assert callable(captured["embed_fn"])


def test_rag_query_default_is_lexical(cli_runner, monkeypatch):
    from localm.cli import main
    captured = _patch_query(monkeypatch)
    r = cli_runner.invoke(main, ["rag", "query", "kb", "hello"])
    assert r.exit_code == 0, r.output
    assert captured["embed_fn"] is None


def test_cli_embed_fn_uses_persisted_key(monkeypatch):
    """`rag add/query --embed` authenticates to /v1/embeddings with the
    persisted owner key (auth.key), not with LOCALM_API_KEY only."""
    import requests

    from localm import auth
    from localm.cli.rag import _cli_rag_embed_fn

    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    auth.set_api_key("cli-file-key-123")   # writes <throwaway home>/auth.key
    captured = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"embedding": [0.1, 0.2]}]}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    embed_fn = _cli_rag_embed_fn("http://127.0.0.1:8642/v1")
    out = embed_fn(["hello"])
    assert out == [[0.1, 0.2]]
    assert captured["headers"]["Authorization"] == "Bearer cli-file-key-123"


def _capture_embed_post(monkeypatch):
    import requests
    captured = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"embedding": [0.1]}]}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return _Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    return captured


def test_cli_embed_fn_sends_the_configured_model_name(monkeypatch):
    """`rag add/query --embed` sends the CONFIGURED embedding model name, the
    same as the GUI's server-side self-embed, not the sentinel "localm".
    /v1/embeddings routes to the dedicated embedder only when the model name
    matches the configured `embedding_model` or a registered embedder, and
    "localm" matches neither."""
    from localm.cli.rag import _cli_rag_embed_fn
    monkeypatch.setattr(
        "localm.config.load_config",
        lambda: {"embedding_model": "my-custom-embedder", "port": 8642})
    captured = _capture_embed_post(monkeypatch)
    embed_fn = _cli_rag_embed_fn("http://127.0.0.1:8642/v1")
    embed_fn(["hello"])
    assert captured["json"]["model"] == "my-custom-embedder"
    assert captured["json"]["model"] != "localm"


def test_cli_embed_fn_defaults_to_the_internal_embedder_not_localm(monkeypatch):
    """With no explicit embedding_model set, the CLI sends the internal DEFAULT
    embedder name (bge-small-en-v1.5), which /v1/embeddings can route, rather
    than the bare "localm" sentinel that falls through to the chat path."""
    from localm.cli.rag import _cli_rag_embed_fn
    from localm.inference.embedder import DEFAULT_EMBEDDING_MODEL
    monkeypatch.setattr("localm.config.load_config", lambda: {"port": 8642})
    captured = _capture_embed_post(monkeypatch)
    embed_fn = _cli_rag_embed_fn("http://127.0.0.1:8642/v1")
    embed_fn(["hello"])
    assert captured["json"]["model"] == DEFAULT_EMBEDDING_MODEL
    assert captured["json"]["model"] != "localm"
