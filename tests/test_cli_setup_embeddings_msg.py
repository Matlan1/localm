# SPDX-License-Identifier: AGPL-3.0-or-later
"""cli-2: `localm setup-embeddings` must not claim existing RAG collections "will
now use semantic search" - they stay lexical (BM25) until re-indexed WITH
embeddings and queried with --embed against a running server. Memory uses it now;
the message must state the capability and name the RAG re-index step."""

from click.testing import CliRunner


def test_setup_embeddings_message_scopes_the_rag_claim(monkeypatch, tmp_path):
    from localm.cli.maintenance import setup_embeddings

    fake = tmp_path / "bge-small.gguf"
    fake.write_bytes(b"x")
    # Stub the DOWNLOADER (the environment), not the message logic under test.
    monkeypatch.setattr(
        "localm.inference.embedder.resolve_embedding_model_path",
        lambda allow_download=True: str(fake))

    result = CliRunner().invoke(setup_embeddings, [])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    low = flat.lower()

    # The success line is still printed.
    assert "Embedding model ready" in flat
    # The overclaim is gone: no accomplished-fact "RAG will now use semantic search".
    assert "RAG will now use semantic search" not in flat
    assert "Memory and RAG will now use semantic search" not in flat
    # The RAG caveat + the concrete re-index step are named.
    assert "--embed" in flat
    assert ("lexical" in low) or ("re-index" in low) or ("re-add" in low)
    # Memory's capability is stated (it is the part that IS true now).
    assert "memory" in low
