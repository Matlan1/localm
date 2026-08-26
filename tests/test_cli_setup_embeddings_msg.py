# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm setup-embeddings` must not claim existing RAG collections "will now
use semantic search" - they stay lexical (BM25) until re-embedded. Memory uses
it immediately; the message must state that capability and name the RAG
re-embed step.

The step it names is `rag reembed <name>`, which works from stored chunk text
alone and needs no source files, not `rag add ... --embed`, which re-reads the
original source files."""

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
    # No accomplished-fact claim that RAG will now use semantic search.
    assert "RAG will now use semantic search" not in flat
    assert "Memory and RAG will now use semantic search" not in flat
    # The RAG caveat + the concrete, no-source-files-needed re-embed step are named.
    assert "rag reembed" in low
    assert ("lexical" in low) or ("re-embed" in low)
    # The advice to re-read originals is not present.
    assert "rag add" not in low
    # Memory's capability is stated.
    assert "memory" in low


# --------------------------------------------------------------------------- #
#  `setup-embeddings --model` warns what an embedding-model switch is about   #
#  to invalidate before it happens. Uses the `cli_runner` fixture for real    #
#  per-test isolation: these assert on load_config() afterward.               #
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
        # rather than silently answering yes.
        result = cli_runner.invoke(setup_embeddings,
                                   ["--model", "new-model", "--yes"])
        assert result.exit_code == 0, result.output
        assert load_config().get("embedding_model") == "new-model"
        assert "Continue with the switch?" not in result.output

    def test_noninteractive_eof_proceeds_rather_than_aborting(
            self, cli_runner, monkeypatch, tmp_path):
        # Run from cron/CI/a script with no stdin: EOF proceeds rather than
        # aborts, and says so.
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
        # No collection created: one command, no prompt.
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
#  absorbed as "nothing to embed".                                            #
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
