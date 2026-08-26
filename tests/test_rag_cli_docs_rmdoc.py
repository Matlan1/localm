# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm rag docs` and `localm rag rm-doc`: per-document CLI verbs.

They wrap the same `Collection.docs()` / `Collection.remove_doc()` the GUI's
`GET /api/rag/collections/{name}` and `POST .../remove-doc` routes call, so
these tests exercise the REAL store - create a collection, index real files,
remove and resync for real - rather than a mocked Collection.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    """Own copy rather than importing the one in
    test_rag_reg589_repair_noninteractive.py, which is module-private."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    # rich.console.Console() reads COLUMNS at construction time; without it, a
    # non-tty width default (80) hard-wraps mid-word inside the long pytest
    # basetemp paths these tests assert on.
    monkeypatch.setenv("COLUMNS", "300")
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


@pytest.fixture
def ragcli():
    from localm.cli import rag as ragcli
    return ragcli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def docs_dir(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "alpha.txt").write_text(
        "alpha content about turbines and gearboxes", encoding="utf-8")
    (d / "beta.txt").write_text(
        "beta content about compressors and bearings", encoding="utf-8")
    return d


def _real_docs(name: str):
    """Read a collection's documents straight from the store, bypassing the
    CLI - the ground truth the CLI's own output is checked against."""
    from localm.rag import Collection
    return {d["path"]: d for d in Collection(name).docs()}


class TestRagDocsUnknownOrEmptyCollection:
    def test_unknown_collection_is_a_red_error_not_a_traceback(self, runner, ragcli):
        r = runner.invoke(ragcli.rag_group, ["docs", "nope"])
        assert r.exit_code == 1, r.output
        assert "No such collection" in r.output

    def test_an_existing_but_empty_collection_says_so(self, runner, ragcli):
        from localm.rag import Collection
        Collection("empty").create()
        r = runner.invoke(ragcli.rag_group, ["docs", "empty"])
        assert r.exit_code == 0, r.output
        assert "no indexed documents" in r.output


class TestRagDocsListsRealDocuments:
    def test_lists_each_document_with_its_real_chunk_count(
            self, runner, ragcli, docs_dir):
        add = runner.invoke(ragcli.rag_group, ["add", "kb", str(docs_dir)])
        assert add.exit_code == 0, add.output

        truth = _real_docs("kb")
        assert len(truth) == 2, "setup did not index both files"

        r = runner.invoke(ragcli.rag_group, ["docs", "kb"])
        assert r.exit_code == 0, r.output
        for path, info in truth.items():
            assert path in r.output, r.output
            n = info["chunks"]
            expect = f"{n} chunk" + ("s" if n != 1 else "")
            assert expect in r.output, (
                f"expected the REAL chunk count ({expect}) for {path}: {r.output}")

    def test_a_document_flagged_missing_by_a_real_resync_is_marked(
            self, runner, ragcli, docs_dir):
        from localm.rag import Collection
        add = runner.invoke(ragcli.rag_group, ["add", "kb", str(docs_dir)])
        assert add.exit_code == 0, add.output

        (docs_dir / "alpha.txt").unlink()          # the file really vanishes
        Collection("kb").resync(policy=None)        # real reconcile, not a stub

        r = runner.invoke(ragcli.rag_group, ["docs", "kb"])
        assert r.exit_code == 0, r.output
        lines = r.output.splitlines()
        alpha_line = next(l for l in lines if "alpha.txt" in l)
        beta_line = next(l for l in lines if "beta.txt" in l)
        assert "missing" in alpha_line, r.output
        assert "missing" not in beta_line, (
            f"the untouched document must not be marked missing: {r.output}")

    def test_an_uploaded_document_is_marked_and_listed_by_its_upload_key(
            self, runner, ragcli):
        from localm.rag import Collection
        coll = Collection("kb").create()
        result = coll.add_uploads(
            [{"filename": "notes.txt", "data": b"uploaded content about pumps"}])
        assert result["added"] == 1, result

        r = runner.invoke(ragcli.rag_group, ["docs", "kb"])
        assert r.exit_code == 0, r.output
        assert "upload:notes.txt" in r.output, r.output
        upload_line = next(l for l in r.output.splitlines() if "notes.txt" in l)
        assert "uploaded" in upload_line, r.output


class TestRagRmDoc:
    def test_unknown_collection_is_a_red_error(self, runner, ragcli):
        r = runner.invoke(ragcli.rag_group, ["rm-doc", "nope", "x.txt"])
        assert r.exit_code == 1, r.output
        assert "No such collection" in r.output

    def test_a_path_not_in_the_collection_is_a_red_error_and_changes_nothing(
            self, runner, ragcli, docs_dir):
        add = runner.invoke(ragcli.rag_group, ["add", "kb", str(docs_dir)])
        assert add.exit_code == 0, add.output
        before = _real_docs("kb")

        r = runner.invoke(ragcli.rag_group, ["rm-doc", "kb", "/never/indexed.txt"])
        assert r.exit_code == 1, r.output
        assert "Not in this collection" in r.output
        assert _real_docs("kb") == before, "a failed removal must not mutate the index"

    def test_removes_only_the_named_document_the_rest_survive(
            self, runner, ragcli, docs_dir):
        add = runner.invoke(ragcli.rag_group, ["add", "kb", str(docs_dir)])
        assert add.exit_code == 0, add.output
        before = _real_docs("kb")
        assert len(before) == 2
        target = next(p for p in before if p.endswith("alpha.txt"))
        other = next(p for p in before if p.endswith("beta.txt"))

        r = runner.invoke(ragcli.rag_group, ["rm-doc", "kb", target])
        assert r.exit_code == 0, r.output
        assert "Removed" in r.output

        after = _real_docs("kb")
        assert target not in after, "the removed document is still in the index"
        assert other in after, (
            "removing one document must not touch the rest of the collection")
        assert after[other] == before[other], (
            "the surviving document's own record must be byte-for-byte unchanged")

    def test_removing_an_uploaded_document_by_its_upload_key_works(
            self, runner, ragcli):
        from localm.rag import Collection
        coll = Collection("kb").create()
        coll.add_uploads(
            [{"filename": "notes.txt", "data": b"uploaded content about pumps"}])
        assert "upload:notes.txt" in _real_docs("kb")

        r = runner.invoke(ragcli.rag_group, ["rm-doc", "kb", "upload:notes.txt"])
        assert r.exit_code == 0, r.output
        assert "upload:notes.txt" not in _real_docs("kb")

    def test_a_locked_collection_is_reported_not_left_to_escape(
            self, runner, ragcli, docs_dir, monkeypatch):
        """`rm-doc` translates a CollectionLockedError through
        _refuse_if_locked, the same way every other write command here does,
        rather than letting it escape. This covers rm-doc's wiring only; the
        lock mechanism itself is covered elsewhere.

        The discriminator: a caught error that reaches sys.exit(1) leaves
        r.exception as a bare SystemExit; an error left to escape leaves
        r.exception as the original exception type. Both give exit_code == 1,
        so exit_code alone cannot tell the two apart."""
        from localm.rag import Collection, CollectionLockedError
        add = runner.invoke(ragcli.rag_group, ["add", "kb", str(docs_dir)])
        assert add.exit_code == 0, add.output
        target = next(iter(_real_docs("kb")))

        def _locked(self, source):
            raise CollectionLockedError("kb", None, 1.0)
        monkeypatch.setattr(Collection, "remove_doc", _locked)

        r = runner.invoke(ragcli.rag_group, ["rm-doc", "kb", target])
        assert r.exit_code == 1, r.output
        assert isinstance(r.exception, SystemExit), (
            "a CollectionLockedError must be caught and reported via "
            f"sys.exit, not left to escape as {type(r.exception).__name__}: "
            f"{r.output!r}")
        assert "is being written by" in r.output, r.output
