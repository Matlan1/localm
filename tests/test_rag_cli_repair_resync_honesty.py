# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm rag repair`, `rag resync` and `rag query` say what happened: an unknown
collection is an error, vanished sources and still-missing documents are named,
a repair that cannot compute embeddings changes nothing, and -k must be 1 or
more. Real collections, real CLI."""

from localm.cli import main
from localm.rag import Collection


def _docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.txt").write_text("alpha content about turbines", encoding="utf-8")
    (d / "b.txt").write_text("bravo content about bread", encoding="utf-8")
    return d


def test_repair_of_an_unknown_collection_is_an_error(cli_runner):
    r = cli_runner.invoke(main, ["rag", "repair", "nope", "--yes"])
    assert r.exit_code == 1, r.output
    assert "No such collection" in r.output


def test_repair_names_a_source_file_that_is_gone(cli_runner, tmp_path):
    d = _docs(tmp_path)
    Collection("kb").create().add_paths([d])
    (d / "b.txt").unlink()

    r = cli_runner.invoke(main, ["rag", "repair", "kb", "--yes"])

    flat = " ".join(r.output.split())
    assert r.exit_code == 0, r.output
    assert "1 source file(s) no longer exist" in flat
    assert "b.txt" in flat
    assert "repaired: 1 re-indexed" in flat


def test_a_repair_that_cannot_embed_changes_nothing(cli_runner, tmp_path):
    d = _docs(tmp_path)
    Collection("kb").create().add_paths([d], embed_fn=lambda texts: [[1.0, 0.0]] * len(texts),
                                        model_name="test-embedder")
    assert Collection("kb").stats()["has_vectors"]

    r = cli_runner.invoke(main, ["rag", "repair", "kb", "--url", "http://127.0.0.1:9"])

    assert r.exit_code == 1, r.output
    assert "nothing was changed" in " ".join(r.output.split())
    assert Collection("kb").stats()["has_vectors"]


def test_a_second_resync_still_names_a_missing_document(cli_runner, tmp_path):
    d = _docs(tmp_path)
    Collection("kb").create().add_paths([d])
    (d / "b.txt").unlink()

    first = cli_runner.invoke(main, ["rag", "resync", "kb"])
    second = cli_runner.invoke(main, ["rag", "resync", "kb"])

    assert first.exit_code == 0 and second.exit_code == 0, second.output
    assert "no longer on disk" in " ".join(first.output.split())
    flat = " ".join(second.output.split())
    assert "still not on disk" in flat
    assert "b.txt" in flat


def test_query_k_must_be_at_least_1(cli_runner, tmp_path):
    Collection("kb").create().add_paths([_docs(tmp_path)])
    for k in ("0", "-1"):
        r = cli_runner.invoke(main, ["rag", "query", "kb", "turbines", "-k", k])
        assert r.exit_code == 2, r.output
