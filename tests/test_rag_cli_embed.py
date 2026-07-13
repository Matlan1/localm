# SPDX-License-Identifier: AGPL-3.0-or-later
"""B10: `localm rag query` is lexical-only by default; `--embed` opts into
hybrid retrieval by embedding the query via a running localm server (matching
the GUI). We verify the wiring: --embed passes an embed_fn to Collection.query;
the default does not.
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
    # NEGATIVE: pre-fix `--embed` is an unknown option -> nonzero exit.
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
    """`rag add/query --embed` must authenticate to /v1/embeddings with the
    persisted owner key (auth.key), not the env var only. Pre-fix _cli_rag_embed_fn
    read LOCALM_API_KEY only, so against a keyed server (localm key generate, key in
    auth.key, not the env) every embed call 401'd and the CLI silently indexed
    lexical-only (memory-audit 2026-07-02 cluster 19)."""
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
