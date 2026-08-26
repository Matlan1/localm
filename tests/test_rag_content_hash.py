# SPDX-License-Identifier: AGPL-3.0-or-later
"""RAG change-detection must be content-aware.

(mtime, size) alone misses same-size edits whose mtime is unchanged (coarse
mtime FS, cp -p / rsync --times, git restore, restore-from-backup), leaving the
index stale. add_paths also compares a content hash, supports force=True
(repair), and Collection.documents() lists indexed paths.
"""

import os

from localm.rag import Collection


def _write(p, text):
    p.write_text(text, encoding="utf-8")


def test_same_size_same_mtime_edit_is_reindexed(tmp_path):
    # NEGATIVE: mtime and size both match, so only the content hash catches it.
    base = tmp_path / "rag"
    docs = tmp_path / "docs"
    docs.mkdir()
    f = docs / "note.txt"
    _write(f, "AAAAA original")          # 14 bytes
    c = Collection("kb", base=base).create()
    c.add_paths([docs])
    st = f.stat()

    _write(f, "BBBBB modified")          # also 14 bytes
    os.utime(f, (st.st_atime, st.st_mtime))   # restore the original mtime
    assert f.stat().st_size == st.st_size
    assert f.stat().st_mtime == st.st_mtime

    res = c.add_paths([docs])
    assert res["updated"] == 1 and res["skipped"] == 0
    assert "modified" in c.query("modified", k=1)[0]["text"]
    assert all("original" not in h["text"] for h in c.query("original", k=5))


def test_unchanged_file_is_skipped(tmp_path):
    base = tmp_path / "rag"
    docs = tmp_path / "docs"
    docs.mkdir()
    _write(docs / "a.txt", "hello world")
    c = Collection("kb", base=base).create()
    c.add_paths([docs])
    again = c.add_paths([docs])
    assert again["skipped"] == 1 and again["updated"] == 0


def test_force_reindexes_unchanged_file(tmp_path):
    base = tmp_path / "rag"
    docs = tmp_path / "docs"
    docs.mkdir()
    _write(docs / "a.txt", "hello world")
    c = Collection("kb", base=base).create()
    c.add_paths([docs])
    forced = c.add_paths([docs], force=True)
    assert forced["updated"] == 1 and forced["skipped"] == 0


def test_repair_via_documents_and_force(tmp_path):
    # Mirrors `localm rag repair`: re-add the indexed doc list with force.
    base = tmp_path / "rag"
    docs = tmp_path / "docs"
    docs.mkdir()
    _write(docs / "a.txt", "alpha")
    _write(docs / "b.txt", "beta")
    c = Collection("kb", base=base).create()
    c.add_paths([docs])
    paths = c.documents()
    assert len(paths) == 2
    res = c.add_paths(paths, force=True)
    assert res["updated"] == 2 and res["skipped"] == 0
