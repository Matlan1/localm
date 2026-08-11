# SPDX-License-Identifier: AGPL-3.0-or-later
"""cli-2: `localm setup-embeddings` must not claim existing RAG collections "will
now use semantic search" - they stay lexical (BM25) until re-embedded. Memory uses
it now; the message must state the capability and name the RAG re-embed step.

The re-embed step itself was `rag add ... --embed` (re-reads the original source
files) until `rag reembed <name>` existed (works from stored chunk text alone, no
source files needed) - the message now points at the latter, the actual fix for
the common case where a source file has moved, been deleted, or arrived only as
an upload."""

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
    # The RAG caveat + the concrete, no-source-files-needed re-embed step are named.
    assert "rag reembed" in low
    assert ("lexical" in low) or ("re-embed" in low)
    # The stale advice (re-reading originals, which reembed's whole point is to
    # avoid needing) must not still be there alongside the new one.
    assert "rag add" not in low
    # Memory's capability is stated (it is the part that IS true now).
    assert "memory" in low
