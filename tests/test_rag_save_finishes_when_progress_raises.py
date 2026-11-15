# SPDX-License-Identifier: AGPL-3.0-or-later
"""A progress callback that raises (a scheduled knowledge run whose key was
revoked) never cuts a collection's save in half: a raise while the index is
being saved lets the save finish and is raised after it, and a raise while files
are being read leaves the collection on disk as it was."""

from __future__ import annotations

from localm.rag import Collection


def _docs(tmp_path, n=3):
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(n):
        (docs / f"d{i}.txt").write_text(f"document number {i}", encoding="utf-8")
    return docs


def _embed(texts):
    return [[1.0, 0.0] for _ in texts]


def test_a_raise_while_saving_the_index_lets_the_save_finish(tmp_path, monkeypatch):
    from localm.rag._store import files
    monkeypatch.setattr(files, "_REPORT_VECTORS_COMPONENTS", 1)
    monkeypatch.setattr(files, "_REPORT_EVERY_S", 0.0)
    base = tmp_path / "collections"
    base.mkdir()
    docs = _docs(tmp_path)
    Collection("c", base=base).create()
    saving_calls = []

    def stop_while_saving(text, **_kw):
        if text.startswith("saving the index"):
            saving_calls.append(text)
            raise PermissionError("the key was revoked")

    caught = None
    try:
        Collection("c", base=base, load=False, cache=False).add_paths(
            [docs], embed_fn=_embed, on_progress=stop_while_saving)
    except PermissionError as e:
        caught = e

    saved = Collection("c", base=base, cache=False)
    assert len(saved.documents()) == 3
    assert len(saved._chunks) == 3
    assert saved._vectors is not None and len(saved._vectors) == 3
    assert all(saved._vectors)
    assert not saved.corrupt
    assert len(saving_calls) == 1, "the callback is not called again after it raised"
    assert isinstance(caught, PermissionError) and "revoked" in str(caught)


def test_a_raise_while_reading_files_saves_nothing_from_the_run(tmp_path):
    base = tmp_path / "collections"
    base.mkdir()
    docs = _docs(tmp_path)
    Collection("c", base=base).create()

    def stop_while_reading(text, **_kw):
        if "reading" in text:
            raise PermissionError("the key was revoked")

    caught = None
    try:
        Collection("c", base=base, load=False, cache=False).add_paths(
            [docs], embed_fn=_embed, on_progress=stop_while_reading)
    except PermissionError as e:
        caught = e

    saved = Collection("c", base=base, cache=False)
    assert saved.documents() == []
    assert saved._chunks == []
    assert isinstance(caught, PermissionError)
