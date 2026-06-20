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
