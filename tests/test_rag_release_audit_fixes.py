# SPDX-License-Identifier: AGPL-3.0-or-later
"""RAG fixes from the 0.3.0 audit.

* The relevance floors of a known embedding model apply to a collection that
  records the model under another registered name (matched by file sha256).
* A chunks.jsonl with bytes that are not UTF-8 loads, flagged corrupt, instead
  of raising out of the collection load.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from localm.rag.store import RELEVANCE_FLOORS, Collection

GPU = "Install the ROCm runtime and set the GPU layers for the gfx1030 card."
WEB = "The web search reads the top result pages and quotes them as evidence."
SYNC = "Knowledge collections are rebuilt with resync when files change on disk."
EXTRA = "Release notes are published with every tagged version of the package."
AXES = {GPU: 0, WEB: 1, SYNC: 2, EXTRA: 3}

BGE = "bge-small-en-v1.5"
BGE_STRONG = RELEVANCE_FLOORS[BGE][0]
NOMIC = "nomic-embed-text-v1.5"


def _embedder(queries: dict):
    def embed(texts):
        out = []
        for t in texts:
            if t in AXES:
                v = [0.0] * 5
                v[AXES[t]] = 1.0
                out.append(v)
            else:
                out.append(queries[t])
        return out
    return embed


def _collection(tmp_path, model, queries=None):
    coll = Collection("kb", base=tmp_path / "rag").create()
    coll.add_uploads(
        [{"filename": f"{n}.md", "data": text.encode()}
         for n, text in (("gpu", GPU), ("web", WEB), ("sync", SYNC), ("extra", EXTRA))],
        embed_fn=_embedder(queries or {}), model_name=model)
    coll = Collection("kb", base=tmp_path / "rag")
    assert len(coll._chunks) == 4 and coll._vectors and all(coll._vectors)
    assert coll.embedding_model() == model
    return coll


def _pin_sha(key: str) -> str:
    from localm.inference import embedder
    return embedder._EMBEDDING_PINS[embedder.KNOWN_EMBEDDING_MODELS[key]][1]


QUERY = "zebra quartz xylophone"


def _strong_query():
    cos = BGE_STRONG + 0.02
    return {QUERY: [cos, 0.0, 0.0, 0.0, math.sqrt(1.0 - cos ** 2)]}


class TestFloorsFollowTheFileNotTheName:
    def test_a_known_model_registered_under_another_name_keeps_its_floors(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr("localm.config.load_registry",
                            lambda: {"my-bge": {"sha256": _pin_sha(BGE)}})
        queries = _strong_query()
        coll = _collection(tmp_path, "my-bge", queries)
        hits = coll.query(QUERY, k=4, embed_fn=_embedder(queries), relevant_only=True)
        assert [h["text"] for h in hits] == [GPU]

    def test_the_sha_match_is_case_insensitive(self, tmp_path, monkeypatch):
        monkeypatch.setattr("localm.config.load_registry",
                            lambda: {"my-bge": {"sha256": _pin_sha(BGE).upper()}})
        queries = _strong_query()
        coll = _collection(tmp_path, "my-bge", queries)
        assert coll.query(QUERY, k=4, embed_fn=_embedder(queries), relevant_only=True)

    def test_a_different_file_under_an_unknown_name_has_no_floors(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr("localm.config.load_registry",
                            lambda: {"other": {"sha256": "0" * 64}})
        queries = _strong_query()
        coll = _collection(tmp_path, "other", queries)
        assert coll.query(QUERY, k=4, embed_fn=_embedder(queries),
                          relevant_only=True) == []

    def test_a_name_with_no_registry_entry_has_no_floors(self, tmp_path, monkeypatch):
        monkeypatch.setattr("localm.config.load_registry", lambda: {})
        queries = _strong_query()
        coll = _collection(tmp_path, "unregistered", queries)
        assert coll.query(QUERY, k=4, embed_fn=_embedder(queries),
                          relevant_only=True) == []

    def test_the_known_name_still_matches_by_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr("localm.config.load_registry", lambda: {})
        queries = _strong_query()
        coll = _collection(tmp_path, BGE, queries)
        assert coll.query(QUERY, k=4, embed_fn=_embedder(queries), relevant_only=True)

    def test_each_known_model_resolves_to_its_own_key(self):
        from localm.inference.embedder import known_embedding_model_for_sha256
        assert known_embedding_model_for_sha256(_pin_sha(BGE)) == BGE
        assert known_embedding_model_for_sha256(_pin_sha(NOMIC)) == NOMIC
        assert known_embedding_model_for_sha256("") is None
        assert known_embedding_model_for_sha256("f" * 64) is None


class TestUndecodableChunksFile:
    def _collection(self, tmp_path):
        coll = Collection("kb", base=tmp_path / "rag").create()
        coll.add_uploads(
            [{"filename": "a.md", "data": b"alpha turbines about energy"},
             {"filename": "b.md", "data": b"beta generators about power"}])
        return Path(coll.dir) / "chunks.jsonl"

    def test_a_non_utf8_byte_flags_the_collection_and_keeps_the_other_chunks(
            self, tmp_path):
        chunks_file = self._collection(tmp_path)
        lines = chunks_file.read_bytes().split(b"\n")
        assert len([ln for ln in lines if ln.strip()]) == 2
        lines[0] = lines[0].replace(b"alpha", b"alp\xffha")
        chunks_file.write_bytes(b"\n".join(lines))

        coll = Collection("kb", base=tmp_path / "rag")
        stats = coll.stats()
        assert stats["corrupt"] is True
        assert stats["chunks_bad_lines"] == 1
        assert [c["text"] for c in coll._chunks] == ["beta generators about power"]

    def test_the_whole_file_not_utf8_loads_with_nothing_kept(self, tmp_path):
        chunks_file = self._collection(tmp_path)
        chunks_file.write_bytes(b"\xff\xfe\x00\x01 not a chunk\n\xc3\x28\n")
        coll = Collection("kb", base=tmp_path / "rag")
        assert coll.stats()["corrupt"] is True
        assert coll._chunks == []

    def test_an_unreadable_file_raises_and_a_write_never_replaces_it(
            self, tmp_path, monkeypatch):
        chunks_file = self._collection(tmp_path)
        before = chunks_file.read_bytes()
        real = Path.read_text

        def read_text(self, *args, **kwargs):
            if self.name == "chunks.jsonl":
                raise PermissionError("denied")
            return real(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read_text)
        with pytest.raises(PermissionError):
            Collection("kb", base=tmp_path / "rag").add_uploads(
                [{"filename": "c.md", "data": b"gamma third document"}])
        assert chunks_file.read_bytes() == before
        monkeypatch.undo()
        again = Collection("kb", base=tmp_path / "rag")
        assert again.stats()["corrupt"] is False
        assert len(again._chunks) == 2

    def test_a_clean_file_with_crlf_lines_and_blank_lines_is_unchanged(self, tmp_path):
        chunks_file = self._collection(tmp_path)
        data = chunks_file.read_bytes()
        chunks_file.write_bytes(data.replace(b"\n", b"\r\n") + b"\r\n\r\n")
        coll = Collection("kb", base=tmp_path / "rag")
        assert coll.stats()["corrupt"] is False
        assert len(coll._chunks) == 2

    @pytest.mark.parametrize("text", ["café 日本語", "emoji \U0001f600"])
    def test_valid_multibyte_text_survives(self, tmp_path, text):
        coll = Collection("kb", base=tmp_path / "rag").create()
        coll.add_uploads([{"filename": "a.md", "data": text.encode("utf-8")}])
        again = Collection("kb", base=tmp_path / "rag")
        assert again.stats()["corrupt"] is False
        assert [c["text"] for c in again._chunks] == [text]


class TestUncalibratedModelIsSaid:
    def test_the_missing_floors_are_noted_once_per_collection(
            self, tmp_path, monkeypatch, caplog):
        import logging

        from localm.rag._store import search
        monkeypatch.setattr("localm.config.load_registry", lambda: {})
        monkeypatch.setattr(search, "_WARNED_UNCALIBRATED", set())
        queries = _strong_query()
        coll = _collection(tmp_path, "unregistered", queries)
        with caplog.at_level(logging.INFO, logger="localm"):
            for _ in range(3):
                coll.query(QUERY, k=4, embed_fn=_embedder(queries), relevant_only=True)
        notes = [r.getMessage() for r in caplog.records
                 if "no measured relevance floors" in r.getMessage()]
        assert len(notes) == 1 and "unregistered" in notes[0]

    def test_a_calibrated_model_is_not_noted(self, tmp_path, monkeypatch, caplog):
        import logging

        from localm.rag._store import search
        monkeypatch.setattr("localm.config.load_registry", lambda: {})
        monkeypatch.setattr(search, "_WARNED_UNCALIBRATED", set())
        queries = _strong_query()
        coll = _collection(tmp_path, BGE, queries)
        with caplog.at_level(logging.INFO, logger="localm"):
            coll.query(QUERY, k=4, embed_fn=_embedder(queries), relevant_only=True)
        assert not [r for r in caplog.records
                    if "no measured relevance floors" in r.getMessage()]
