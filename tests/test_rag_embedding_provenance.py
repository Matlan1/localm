# SPDX-License-Identifier: AGPL-3.0-or-later
"""A collection built the ORDINARY way (add_paths/add_uploads/resync) records which
embedding model built it, not only one that went through reembed(). That value
is what fills the "built with X" clause in the dimension-mismatch message
(_dim_mismatch_message).

No mocks of the thing under test: every case drives Collection.add_paths /
add_uploads / resync against a tmp_path collection dir and reloads from disk to
prove the value was actually persisted, not just left on the live instance.
"""
from localm.rag.store import Collection


def _vecs(dim):
    def embed(texts):
        return [[1.0] * dim for _ in texts]
    return embed


class TestAddPathsRecordsModel:
    def test_records_the_model_on_first_embed(self, tmp_path):
        c = Collection("kb", base=tmp_path).create()
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.txt").write_text("alpha beta gamma", encoding="utf-8")
        c.add_paths([d], embed_fn=_vecs(3), model_name="bge-small-en-v1.5")

        reloaded = Collection("kb", base=tmp_path)
        assert reloaded.embedding_model() == "bge-small-en-v1.5"

    def test_no_model_name_records_nothing(self, tmp_path):
        # Same as reembed()'s own `if model_name:` guard: a caller that embeds
        # without naming the model leaves the field unset rather than writing
        # a false empty string.
        c = Collection("kb", base=tmp_path).create()
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.txt").write_text("alpha beta gamma", encoding="utf-8")
        c.add_paths([d], embed_fn=_vecs(3))

        reloaded = Collection("kb", base=tmp_path)
        assert reloaded.embedding_model() is None

    def test_model_name_without_embed_fn_records_nothing(self, tmp_path):
        # Passing model_name is harmless when nothing is actually embedded
        # (lexical-only indexing): the first-embed branch is never reached.
        c = Collection("kb", base=tmp_path).create()
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.txt").write_text("alpha beta gamma", encoding="utf-8")
        c.add_paths([d], model_name="bge-small-en-v1.5")   # no embed_fn

        reloaded = Collection("kb", base=tmp_path)
        assert reloaded.embedding_model() is None
        assert reloaded.stats()["has_vectors"] is False

    def test_a_later_add_does_not_overwrite_the_recorded_model(self, tmp_path):
        # The write is gated on the "first-embed" branch (self._vec_dim is None),
        # the same site _vec_dim itself is first set - once a collection has
        # vectors, only reembed() changes which model built it.
        c = Collection("kb", base=tmp_path).create()
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.txt").write_text("alpha beta gamma", encoding="utf-8")
        c.add_paths([d], embed_fn=_vecs(3), model_name="model-one")

        (d / "b.txt").write_text("delta epsilon zeta", encoding="utf-8")
        c.add_paths([d], embed_fn=_vecs(3), model_name="model-two")

        reloaded = Collection("kb", base=tmp_path)
        assert reloaded.embedding_model() == "model-one"


class TestAddUploadsRecordsModel:
    def test_records_the_model_on_first_embed(self, tmp_path):
        c = Collection("kb", base=tmp_path).create()
        c.add_uploads(
            [{"filename": "a.txt", "data": b"alpha beta gamma"}],
            embed_fn=_vecs(3), model_name="nomic-embed-text-v1.5")

        reloaded = Collection("kb", base=tmp_path)
        assert reloaded.embedding_model() == "nomic-embed-text-v1.5"

    def test_no_model_name_records_nothing(self, tmp_path):
        c = Collection("kb", base=tmp_path).create()
        c.add_uploads(
            [{"filename": "a.txt", "data": b"alpha beta gamma"}],
            embed_fn=_vecs(3))

        reloaded = Collection("kb", base=tmp_path)
        assert reloaded.embedding_model() is None


class TestMixedProvenanceMarker:
    """embedding_model() names only the FIRST model (see
    test_a_later_add_does_not_overwrite_the_recorded_model above). A later
    add/upload under a DIFFERENT named model at the same dimension must still
    be recorded somewhere, so collection_provenance_report() does not call
    the collection safe for either model."""

    def test_add_paths_same_dim_different_model_marks_mixed(self, tmp_path):
        c = Collection("kb", base=tmp_path).create()
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.txt").write_text("alpha beta gamma", encoding="utf-8")
        c.add_paths([d], embed_fn=_vecs(3), model_name="model-one")
        assert c.embedding_model_mixed() is False

        c.add_paths([d], force=True, embed_fn=_vecs(3), model_name="model-two")

        reloaded = Collection("kb", base=tmp_path)
        assert reloaded.embedding_model() == "model-one"
        assert reloaded.embedding_model_mixed() is True

    def test_add_uploads_same_dim_different_model_marks_mixed(self, tmp_path):
        c = Collection("kb", base=tmp_path).create()
        c.add_uploads(
            [{"filename": "a.txt", "data": b"alpha beta gamma"}],
            embed_fn=_vecs(3), model_name="model-one")
        c.add_uploads(
            [{"filename": "a.txt", "data": b"alpha beta gamma"}],
            force=True, embed_fn=_vecs(3), model_name="model-two")

        reloaded = Collection("kb", base=tmp_path)
        assert reloaded.embedding_model() == "model-one"
        assert reloaded.embedding_model_mixed() is True

    def test_reindexing_under_the_same_model_again_does_not_mark_mixed(self, tmp_path):
        c = Collection("kb", base=tmp_path).create()
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.txt").write_text("alpha beta gamma", encoding="utf-8")
        c.add_paths([d], embed_fn=_vecs(3), model_name="model-one")

        (d / "a.txt").write_text("alpha beta gamma updated", encoding="utf-8")
        c.add_paths([d], embed_fn=_vecs(3), model_name="model-one")

        reloaded = Collection("kb", base=tmp_path)
        assert reloaded.embedding_model_mixed() is False

    def test_reembed_clears_the_mixed_marker(self, tmp_path):
        c = Collection("kb", base=tmp_path).create()
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.txt").write_text("alpha beta gamma", encoding="utf-8")
        c.add_paths([d], embed_fn=_vecs(3), model_name="model-one")
        c.add_paths([d], force=True, embed_fn=_vecs(3), model_name="model-two")
        assert c.embedding_model_mixed() is True

        c.reembed(embed_fn=_vecs(3), model_name="model-one")

        assert c.embedding_model_mixed() is False
        reloaded = Collection("kb", base=tmp_path)
        assert reloaded.embedding_model_mixed() is False


class TestResyncRecordsModel:
    def test_records_the_model_on_first_embed(self, tmp_path):
        base = tmp_path / "collections"
        base.mkdir()
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.txt").write_text("alpha beta gamma", encoding="utf-8")

        coll = Collection("kb", base=base).create()
        coll.add_paths([docs])                    # lexical-only initial index
        assert coll.embedding_model() is None

        (docs / "b.txt").write_text("delta epsilon zeta", encoding="utf-8")
        coll.resync(embed_fn=_vecs(3), model_name="bge-small-en-v1.5")

        reloaded = Collection("kb", base=base)
        assert reloaded.embedding_model() == "bge-small-en-v1.5"
