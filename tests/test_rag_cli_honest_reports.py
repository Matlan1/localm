# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm rag`: nothing is lost, hidden or misreported without the person being told.

Every case drives the real store and the real click commands against a collection
under a throwaway LOCALM_HOME; only the embedder (a server call) is replaced, by
a deterministic function.

  * repair of a collection that does not exist fails like every other command;
  * repair, resync and add say which named sources are gone;
  * a second resync repeats the missing-document warning;
  * `rag add --force` asks before dropping a collection's embeddings;
  * a touched file whose content is unchanged is skipped, not re-read;
  * one document / one chunk is not pluralised.
"""

from __future__ import annotations

import os

import pytest
from click.testing import CliRunner


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setenv("COLUMNS", "400")
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


def _fake_embed(texts):
    return [[1.0, 0.5] for _ in texts]


@pytest.fixture
def fake_server(ragcli, monkeypatch):
    """`--embed` and the non-interactive keep-the-embeddings path reach an
    embedder here instead of a server, and count the texts they embed."""
    calls: list = []

    def _make(url):
        def _embed(texts):
            calls.append(list(texts))
            return _fake_embed(texts)
        return _embed

    monkeypatch.setattr(ragcli, "_cli_rag_embed_fn", _make)
    return calls


def _coll(name):
    from localm.rag import Collection
    return Collection(name, cache=False)


def _hybrid(tmp_path, name="emb", fname="g.md"):
    f = tmp_path / fname
    f.write_text("gamma document text", encoding="utf-8")
    from localm.rag import Collection
    c = Collection(name)
    c.create()
    c.add_paths([f], embed_fn=_fake_embed, model_name="fake")
    assert _coll(name).stats()["has_vectors"]
    return f


class TestRepairUnknownCollection:
    def test_repair_of_a_mistyped_name_is_an_error(self, runner, ragcli):
        r = runner.invoke(ragcli.rag_group, ["repair", "no-such-collection"])
        assert r.exit_code == 1, r.output
        assert "No such collection" in r.output

    def test_reembed_of_a_mistyped_name_says_the_same(self, runner, ragcli):
        r = runner.invoke(ragcli.rag_group, ["reembed", "no-such-collection", "-y"])
        assert r.exit_code == 1, r.output
        assert "No such collection" in r.output

    def test_repair_of_an_existing_empty_collection_still_exits_zero(self, runner, ragcli):
        _coll("empty").create()
        r = runner.invoke(ragcli.rag_group, ["repair", "empty"])
        assert r.exit_code == 0, r.output
        assert "no indexed documents" in r.output


class TestVanishedSourcesAreNamed:
    def test_repair_names_a_deleted_source_and_keeps_its_chunks(
            self, runner, ragcli, tmp_path):
        f = tmp_path / "f.md"
        f.write_text("turbine manual text", encoding="utf-8")
        c = _coll("v").create()
        c.add_paths([f])
        f.unlink()
        r = runner.invoke(ragcli.rag_group, ["repair", "v"])
        assert r.exit_code == 0, r.output
        assert "no longer exist" in r.output
        assert str(f) in r.output
        assert _coll("v").stats()["n_chunks"] == 1

    def test_add_paths_reports_the_named_path_that_is_not_there(self, tmp_path):
        f = tmp_path / "f.md"
        f.write_text("turbine manual text", encoding="utf-8")
        c = _coll("v2").create()
        c.add_paths([f])
        f.unlink()
        res = c.add_paths([f], force=True)
        assert res["not_found"] == [str(f.resolve())] or res["not_found"] == [str(f)]
        assert res["updated"] == 0

    def test_a_present_source_is_not_reported(self, tmp_path):
        f = tmp_path / "f.md"
        f.write_text("turbine manual text", encoding="utf-8")
        c = _coll("v3").create()
        assert c.add_paths([f])["not_found"] == []
        assert c.add_paths([f], force=True)["not_found"] == []


class TestSecondResyncRepeatsTheMissingWarning:
    def _folder(self, tmp_path, n=2):
        d = tmp_path / "fold"
        d.mkdir()
        for i in range(n):
            (d / f"doc{i}.md").write_text(f"document number {i} about ships",
                                          encoding="utf-8")
        return d

    def test_missing_line_is_printed_on_every_run_while_the_file_is_gone(
            self, runner, ragcli, tmp_path):
        d = self._folder(tmp_path)
        c = _coll("fold").create()
        c.add_paths([d])
        (d / "doc1.md").unlink()
        first = runner.invoke(ragcli.rag_group, ["resync", "fold"])
        second = runner.invoke(ragcli.rag_group, ["resync", "fold"])
        assert first.exit_code == 0 and second.exit_code == 0
        for r in (first, second):
            assert "no longer on disk" in r.output, r.output
            assert "doc1.md" in r.output, r.output

    def test_prune_missing_stops_the_warning(self, runner, ragcli, tmp_path):
        d = self._folder(tmp_path)
        c = _coll("fold").create()
        c.add_paths([d])
        (d / "doc1.md").unlink()
        runner.invoke(ragcli.rag_group, ["resync", "fold"])
        r = runner.invoke(ragcli.rag_group, ["resync", "fold", "--prune-missing"])
        assert "no longer on disk" not in r.output
        assert "pruned 1 entry" in r.output

    def test_more_than_ten_missing_documents_are_counted_not_cut_off(
            self, runner, ragcli, tmp_path):
        d = self._folder(tmp_path, n=12)
        c = _coll("fold").create()
        c.add_paths([d])
        for p in d.glob("*.md"):
            p.unlink()
        r = runner.invoke(ragcli.rag_group, ["resync", "fold"])
        assert "12 document(s) are no longer on disk" in r.output, r.output
        assert "and 2 more" in r.output, r.output


class TestForceAsksBeforeDroppingEmbeddings:
    def test_no_answer_keeps_the_embeddings_by_re_embedding(
            self, runner, ragcli, tmp_path, fake_server):
        f = _hybrid(tmp_path)
        r = runner.invoke(ragcli.rag_group, ["add", "emb", str(f), "--force"], input="")
        assert r.exit_code == 0, r.output
        assert "REMOVE the existing embeddings" in r.output
        assert fake_server, "no embedding was computed"
        assert _coll("emb").stats()["has_vectors"]

    def test_answering_no_aborts_and_changes_nothing(
            self, runner, ragcli, tmp_path, fake_server):
        f = _hybrid(tmp_path)
        before = _coll("emb").stats()
        r = runner.invoke(ragcli.rag_group, ["add", "emb", str(f), "--force"], input="n\n")
        assert r.exit_code != 0, r.output
        after = _coll("emb").stats()
        assert after["has_vectors"] and after["n_chunks"] == before["n_chunks"]
        assert not fake_server

    def test_yes_accepts_the_drop(self, runner, ragcli, tmp_path, fake_server):
        f = _hybrid(tmp_path)
        r = runner.invoke(ragcli.rag_group, ["add", "emb", str(f), "--force", "--yes"])
        assert r.exit_code == 0, r.output
        assert not _coll("emb").stats()["has_vectors"]

    def test_embed_flag_needs_no_question(self, runner, ragcli, tmp_path, fake_server):
        f = _hybrid(tmp_path)
        r = runner.invoke(ragcli.rag_group, ["add", "emb", str(f), "--force", "--embed"])
        assert r.exit_code == 0, r.output
        assert "REMOVE" not in r.output
        assert _coll("emb").stats()["has_vectors"]

    def test_collection_without_embeddings_is_not_asked(
            self, runner, ragcli, tmp_path, fake_server):
        f = tmp_path / "p.md"
        f.write_text("plain lexical text", encoding="utf-8")
        _coll("plain").create().add_paths([f])
        r = runner.invoke(ragcli.rag_group, ["add", "plain", str(f), "--force"])
        assert r.exit_code == 0, r.output
        assert "REMOVE" not in r.output

    def test_changed_file_added_without_embed_says_the_vectors_are_missing(
            self, runner, ragcli, tmp_path, fake_server):
        f = _hybrid(tmp_path)
        f.write_text("gamma document text, now much longer than before", encoding="utf-8")
        r = runner.invoke(ragcli.rag_group, ["add", "emb", str(f)])
        assert r.exit_code == 0, r.output
        assert "without embeddings" in r.output
        assert "rag repair emb --embed" in r.output

    def test_one_and_the_same_message_for_repair_and_add(
            self, runner, ragcli, tmp_path, fake_server):
        f = _hybrid(tmp_path)
        a = runner.invoke(ragcli.rag_group, ["add", "emb", str(f), "--force"], input="n\n")
        b = runner.invoke(ragcli.rag_group, ["repair", "emb"], input="n\n")
        assert "will REMOVE the existing embeddings" in a.output
        assert "will REMOVE the existing embeddings" in b.output


class TestTouchedUnchangedFileIsSkipped:
    def test_touch_then_add_skips_without_re_reading_or_re_embedding(
            self, runner, ragcli, tmp_path, fake_server):
        f = tmp_path / "f.md"
        f.write_text("hello world", encoding="utf-8")
        r1 = runner.invoke(ragcli.rag_group, ["add", "c", str(f), "--embed"])
        assert r1.exit_code == 0, r1.output
        embedded_once = len(fake_server)
        st = f.stat()
        os.utime(f, (st.st_atime, st.st_mtime + 100))
        r2 = runner.invoke(ragcli.rag_group, ["add", "c", str(f), "--embed"])
        assert "0 added, 0 updated, 1 unchanged" in r2.output, r2.output
        assert "reading" not in r2.output
        assert len(fake_server) == embedded_once

    def test_touched_file_with_new_content_is_still_re_indexed(self, tmp_path):
        f = tmp_path / "f.md"
        f.write_text("hello world", encoding="utf-8")
        c = _coll("c2").create()
        c.add_paths([f])
        f.write_text("hallo world", encoding="utf-8")        # same size
        st = f.stat()
        os.utime(f, (st.st_atime, st.st_mtime + 100))
        assert c.add_paths([f])["updated"] == 1

    def test_skipping_a_returned_file_clears_its_missing_flag(self, tmp_path):
        f = tmp_path / "f.md"
        f.write_text("hello world", encoding="utf-8")
        c = _coll("c3").create()
        c.add_paths([f])
        meta_doc = c._meta["docs"][str(f.resolve())]
        meta_doc["missing"] = True
        meta_doc["missing_since"] = 1.0
        c._save_meta()
        st = f.stat()
        os.utime(f, (st.st_atime, st.st_mtime + 100))
        res = _coll("c3").add_paths([f])
        assert res["skipped"] == 1
        assert _coll("c3").stats()["n_missing"] == 0


class TestSingularNouns:
    def test_list_add_and_docs_say_one_doc_one_chunk(self, runner, ragcli, tmp_path):
        f = tmp_path / "one.md"
        f.write_text("single line", encoding="utf-8")
        add = runner.invoke(ragcli.rag_group, ["add", "x", str(f)])
        assert "(1 chunk)" in add.output and "1 chunks" not in add.output, add.output
        assert "- 1 chunk in 'x'" in add.output
        lst = runner.invoke(ragcli.rag_group, ["list"])
        assert "1 doc · 1 chunk ·" in lst.output, lst.output
        assert "1 docs" not in lst.output

    def test_plural_is_kept_for_other_counts(self, runner, ragcli, tmp_path):
        d = tmp_path / "two"
        d.mkdir()
        (d / "a.md").write_text("first", encoding="utf-8")
        (d / "b.md").write_text("second", encoding="utf-8")
        runner.invoke(ragcli.rag_group, ["add", "y", str(d)])
        lst = runner.invoke(ragcli.rag_group, ["list"])
        assert "2 docs · 2 chunks ·" in lst.output, lst.output
        zero = _coll("z").create()
        assert zero.stats()["n_docs"] == 0
        assert "0 docs · 0 chunks" in runner.invoke(ragcli.rag_group, ["list"]).output
