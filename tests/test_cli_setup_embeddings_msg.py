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


# --------------------------------------------------------------------------- #
#  NEW-RAG-DIM-NO-REEMBED (2026-08-12 follow-up): `setup-embeddings --model`  #
#  is the third writer of embedding_model, alongside POST /api/rag/embedding  #
#  and PATCH /v1/config - both of which already warn what a switch is about   #
#  to invalidate BEFORE it happens instead of only in a post-switch note.     #
#  Uses the `cli_runner` fixture (conftest.py) for real per-test isolation:   #
#  unlike the test above, these assert on load_config() afterward, so a       #
#  session-shared home is not good enough.                                    #
# --------------------------------------------------------------------------- #

def _stub_install(monkeypatch, tmp_path, name="new-model.gguf"):
    fake = tmp_path / name
    fake.write_bytes(b"x")
    monkeypatch.setattr(
        "localm.inference.embedder.resolve_embedding_model_path",
        lambda allow_download=True: str(fake))
    return fake


def _make_affected_collection():
    """A collection with real vectors under this test's isolated rag_dir(),
    recorded as built with 'old-model' - the thing a switch would invalidate."""
    from localm.rag.store import Collection, rag_dir
    c = Collection("docs", base=rag_dir()).create()
    c._chunks = [{"source": "doc0.txt", "pos": 0, "text": "alpha"}]
    c._vectors = [[0.1] * 768]
    c._meta["embedding_model"] = "old-model"
    c._save()
    return c


class TestSetupEmbeddingsPreSwitchConfirm:
    def test_change_with_affected_collections_reports_before_prompting(
            self, cli_runner, monkeypatch, tmp_path):
        from localm.cli.maintenance import setup_embeddings
        _stub_install(monkeypatch, tmp_path)
        _make_affected_collection()

        result = cli_runner.invoke(setup_embeddings, ["--model", "new-model"],
                                   input="n\n")
        assert "may invalidate" in result.output
        assert "docs" in result.output
        assert "old-model" in result.output

    def test_declining_aborts_without_writing_the_config(
            self, cli_runner, monkeypatch, tmp_path):
        from localm.cli.maintenance import setup_embeddings
        from localm.config import load_config
        _stub_install(monkeypatch, tmp_path)
        _make_affected_collection()
        before = load_config().get("embedding_model")

        result = cli_runner.invoke(setup_embeddings, ["--model", "new-model"],
                                   input="n\n")
        assert result.exit_code != 0
        assert load_config().get("embedding_model") == before, \
            "declining must not switch the model"

    def test_explicit_yes_answer_proceeds_and_writes(
            self, cli_runner, monkeypatch, tmp_path):
        from localm.cli.maintenance import setup_embeddings
        from localm.config import load_config
        _stub_install(monkeypatch, tmp_path)
        _make_affected_collection()

        result = cli_runner.invoke(setup_embeddings, ["--model", "new-model"],
                                   input="y\n")
        assert result.exit_code == 0, result.output
        assert load_config().get("embedding_model") == "new-model"

    def test_yes_flag_skips_the_prompt_entirely(
            self, cli_runner, monkeypatch, tmp_path):
        from localm.cli.maintenance import setup_embeddings
        from localm.config import load_config
        _stub_install(monkeypatch, tmp_path)
        _make_affected_collection()

        # No `input=` at all: if the code prompted, CliRunner would hit EOF
        # (a different, already-covered path) rather than silently answering
        # yes - so this also proves --yes never reaches click.confirm.
        result = cli_runner.invoke(setup_embeddings,
                                   ["--model", "new-model", "--yes"])
        assert result.exit_code == 0, result.output
        assert load_config().get("embedding_model") == "new-model"
        assert "Continue with the switch?" not in result.output

    def test_noninteractive_eof_proceeds_rather_than_aborting(
            self, cli_runner, monkeypatch, tmp_path):
        # REG-589's shape (mirrors test_rag_reg589_repair_noninteractive.py):
        # run from cron/CI/a script with no stdin. Unlike `rag repair` (which
        # would DESTROY embeddings by proceeding), nothing here is destroyed -
        # a collection's chunk text and existing vectors stay on disk either
        # way - so EOF must proceed rather than abort, and say so (rule 5).
        from localm.cli.maintenance import setup_embeddings
        from localm.config import load_config
        _stub_install(monkeypatch, tmp_path)
        _make_affected_collection()

        result = cli_runner.invoke(setup_embeddings, ["--model", "new-model"],
                                   input="")   # "" == EOF == no stdin
        assert result.exit_code == 0, result.output
        assert load_config().get("embedding_model") == "new-model"
        low = result.output.lower()
        assert "not an interactive" in low or "--yes" in low, result.output

    def test_no_affected_collections_skips_the_report_and_prompt(
            self, cli_runner, monkeypatch, tmp_path):
        # No collection created - nothing at risk, so this must behave
        # exactly as before this change: one command, no prompt.
        from localm.cli.maintenance import setup_embeddings
        from localm.config import load_config
        _stub_install(monkeypatch, tmp_path)

        result = cli_runner.invoke(setup_embeddings, ["--model", "new-model"])
        assert result.exit_code == 0, result.output
        assert load_config().get("embedding_model") == "new-model"
        assert "may invalidate" not in result.output
        assert "Continue with the switch?" not in result.output

    def test_same_model_skips_the_report(
            self, cli_runner, monkeypatch, tmp_path):
        from localm.cli.maintenance import setup_embeddings
        from localm.config import update_config
        _stub_install(monkeypatch, tmp_path)
        update_config(lambda c: c.update({"embedding_model": "new-model"}))
        _make_affected_collection()

        result = cli_runner.invoke(setup_embeddings, ["--model", "new-model"])
        assert result.exit_code == 0, result.output
        assert "may invalidate" not in result.output


# --------------------------------------------------------------------------- #
#  An unreadable memory namespace must be NAMED in the banner, not silently   #
#  absorbed as "nothing to embed". Reproduces the CLI-layer cost of the       #
#  backfill.py defect: pending == 0 for a root whose only namespace could not #
#  be opened, which used to skip the whole memory block with no note at all.  #
# --------------------------------------------------------------------------- #

class TestSetupEmbeddingsUnreadableMemoryNamespace:
    def test_unreadable_namespace_is_named_in_the_banner(
            self, cli_runner, monkeypatch, tmp_path):
        from localm.cli.maintenance import setup_embeddings
        _stub_install(monkeypatch, tmp_path)

        ns_dir = tmp_path / ".localm" / "memory" / "chat"
        ns_dir.mkdir(parents=True, exist_ok=True)
        (ns_dir / ("0" * 16 + ".jsonl")).write_bytes(
            b'{"id":"a"}\n\xff\xfe not utf-8\n')

        result = cli_runner.invoke(setup_embeddings, [])
        assert result.exit_code == 0, result.output
        low = result.output.lower()
        assert "could not be read" in low, (
            "an unreadable namespace must be named in the banner, not silently "
            f"treated as nothing to embed: {result.output!r}")
        assert "so recall is semantic for those too" not in low, (
            "must never claim success while a namespace went unread")
