# SPDX-License-Identifier: AGPL-3.0-or-later
"""RAG data-robustness hardening found by an adversarial hostile-input sweep.

Seven distinct bugs:

B1  _extract_docx `<w:t>` run extraction backtracked quadratically (a tiny docx
    with many unclosed <w:t> openers pinned a CPU).
B2  _extract_tar_members called sorted(getmembers()), materialising every member of
    a compressed tarball BEFORE the member cap applied, so a .tgz of hundreds of
    thousands of empty members ran far under the size cap.
B3  Collection._expand walked with rglob(), which follows Windows NTFS junctions
    (is_symlink() is False for them), so a self-referential junction looped forever.
B4  _extract_ipynb caught only json.JSONDecodeError, so deeply-nested JSON raised
    RecursionError past the guard and answered HTTP 500.
B5  Collection._load appended chunk lines with no shape check, so a non-dict or
    missing-"text" line crashed query()/remove_doc()/add_paths() and stats() lied.
B6  A non-finite (NaN/inf) embedding component made _cosine return nan; the blended
    score dropped the chunk from results with no surfaced degrade reason.
B7  chunk_text recorded a paragraph's pos one line too low when preceded by an odd
    number of blank lines, so the citation pointed at a blank line.
"""

import io
import json
import math
import subprocess
import sys
import time
import zipfile

import pytest

from localm.rag import Collection
from localm.rag import extract
from localm.rag.chunk import chunk_text
from localm.rag.extract import ExtractError, extract_bytes


# --------------------------------------------------------------------------- #
#  docx <w:t> extraction must be linear, not O(n^2)                            #
# --------------------------------------------------------------------------- #
def _docx(document_xml: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", document_xml)
    return buf.getvalue()


def test_docx_many_unclosed_wt_openers_is_fast():
    # 100k <w:t> openers with NO closers. The linear `[^<]*` finishes fast; with
    # no closers there is genuinely no text, so a clean ExtractError is the
    # correct fast outcome.
    body = "<w:document><w:body><w:p>" + ("<w:t>x" * 100_000) + "</w:p></w:body></w:document>"
    t0 = time.time()
    try:
        extract_bytes(_docx(body), "bomb.docx")
    except ExtractError:
        pass
    elapsed = time.time() - t0
    assert elapsed < 3.0, f"docx <w:t> extraction is not linear ({elapsed:.1f}s)"


def test_docx_many_closed_wt_runs_is_fast_and_correct():
    # 60k CLOSED runs: must extract ALL of them and stay linear.
    body = "<w:document><w:body><w:p>" + ("<w:t>a</w:t>" * 60_000) + "</w:p></w:body></w:document>"
    t0 = time.time()
    out = extract_bytes(_docx(body), "big.docx")
    elapsed = time.time() - t0
    assert elapsed < 3.0, f"docx run extraction is not linear ({elapsed:.1f}s)"
    assert out.count("a") == 60_000, "every closed run must be extracted"


def test_docx_normal_runs_still_extract():
    body = ("<w:document><w:body>"
            "<w:p><w:t>Hello </w:t><w:t>world</w:t></w:p>"
            "<w:p><w:t>Second paragraph</w:t></w:p>"
            "</w:body></w:document>")
    out = extract_bytes(_docx(body), "doc.docx")
    assert "Hello world" in out
    assert "Second paragraph" in out


def test_docx_run_text_with_gt_and_entities():
    body = ("<w:document><w:body><w:p>"
            "<w:t>a &gt; b and c &lt; d</w:t>"
            "</w:p></w:body></w:document>")
    out = extract_bytes(_docx(body), "doc.docx")
    assert "a > b and c < d" in out


# --------------------------------------------------------------------------- #
#  compressed-tar member-count bomb must be bounded, not materialised          #
# --------------------------------------------------------------------------- #
def _tgz_many_empty_members(n: int) -> bytes:
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for i in range(n):
            ti = tarfile.TarInfo(f"m{i}.txt")
            ti.size = 0
            tf.addfile(ti, io.BytesIO(b""))
    return buf.getvalue()


def test_compressed_tar_member_scan_is_capped(monkeypatch):
    # Counts the per-member header reads (TarFile.next): the lazy scan stops at
    # the cap rather than getmembers()-ing the whole archive. The cap is shrunk
    # so the test stays tiny.
    import tarfile
    monkeypatch.setattr(extract, "MAX_ARCHIVE_MEMBERS", 50)
    data = _tgz_many_empty_members(5_000)
    calls = {"n": 0}
    real_next = tarfile.TarFile.next

    def counting_next(self):
        calls["n"] += 1
        return real_next(self)

    monkeypatch.setattr(tarfile.TarFile, "next", counting_next)
    out = extract_bytes(data, "bomb.tgz")
    assert isinstance(out, str)
    # Bounded to about the cap (a couple past it for the break check), nowhere
    # near the 5000 members getmembers() would have parsed.
    assert calls["n"] <= 200, (
        f"scanned {calls['n']} member headers for a {50}-cap - getmembers() "
        "materialised the whole archive")


def test_small_tarball_still_extracts():
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, text in (("a.txt", b"alpha content"), ("b.txt", b"beta content")):
            ti = tarfile.TarInfo(name)
            ti.size = len(text)
            tf.addfile(ti, io.BytesIO(text))
    out = extract_bytes(buf.getvalue(), "bundle.tgz")
    assert "alpha content" in out and "beta content" in out


# --------------------------------------------------------------------------- #
#  the indexing walk must not loop on a Windows junction                       #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(sys.platform != "win32", reason="NTFS junctions are Windows-only")
def test_indexing_walk_skips_junction_loops(tmp_path, monkeypatch):
    # BRANCHING junctions: a single self-referential junction is depth-capped by
    # Windows itself, so it does not exercise the walk. Two junctions per directory
    # make rglob's descent exponential; _walk_files skips reparse points and
    # scandirs each real dir exactly once.
    import os as _os
    from pathlib import Path
    d = tmp_path / "docs"
    d.mkdir()
    (d / "real.txt").write_text("genuine indexable content", encoding="utf-8")
    for jn in ("loop1", "loop2"):
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(d / jn), str(d)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip(f"could not create junction: {r.stderr.strip()}")

    import localm.rag.store as store_mod
    calls = {"n": 0}
    real_scandir = _os.scandir

    def counting_scandir(p):
        calls["n"] += 1
        return real_scandir(p)

    monkeypatch.setattr(store_mod.os, "scandir", counting_scandir)
    c = Collection("kb", base=tmp_path / "rag").create()
    files = c._expand([str(d)])
    # Each REAL directory is scanned once; the junctions are never descended.
    assert calls["n"] <= 5, f"walk scandir'd {calls['n']} dirs - junctions were followed"
    assert any(Path(f).name == "real.txt" for f in files), "real file must be indexed"


# --------------------------------------------------------------------------- #
#  deeply-nested JSON in a notebook must not escape as RecursionError          #
# --------------------------------------------------------------------------- #
def test_deeply_nested_ipynb_json_is_clean_error():
    payload = ('{"a":' * 6000 + "1" + "}" * 6000).encode()
    with pytest.raises(ExtractError):
        extract_bytes(payload, "evil.ipynb")


def test_normal_ipynb_still_extracts():
    nb = {"cells": [{"cell_type": "code", "source": ["print('hi')\n"]},
                    {"cell_type": "markdown", "source": "# Title"}]}
    out = extract_bytes(json.dumps(nb).encode(), "nb.ipynb")
    assert "print('hi')" in out and "Title" in out


# --------------------------------------------------------------------------- #
#  a malformed chunks.jsonl line must not brick the collection                #
# --------------------------------------------------------------------------- #
def _seed(tmp_path, text="the quick brown fox jumps over the lazy dog"):
    base = tmp_path / "rag"
    d = tmp_path / "docs"
    d.mkdir()
    f = d / "a.txt"
    f.write_text((text + "\n") * 10, encoding="utf-8")
    c = Collection("kb", base=base).create()
    c.add_paths([str(f)])
    return base


@pytest.mark.parametrize("bad_line", ["42", '"just a string"', "[1, 2, 3]", "null", "true"])
def test_non_dict_chunk_line_does_not_crash(tmp_path, bad_line):
    base = _seed(tmp_path)
    chunks_file = base / "kb" / "chunks.jsonl"
    with chunks_file.open("a", encoding="utf-8") as fh:
        fh.write("\n" + bad_line + "\n")
    c = Collection("kb", base=base)
    # query, remove_doc and add_paths(force) must all survive the poison line
    assert isinstance(c.query("fox"), list)
    assert c.query("fox"), "the real chunk must still be queryable"
    assert c.stats()["corrupt"] is True, "malformed chunk line must be surfaced, not hidden"
    docs = c.documents()
    assert c.add_paths(docs, force=True)["chunks"] >= 1   # repair must not crash


def test_malformed_chunk_line_warning_logs_once_per_process(tmp_path, caplog):
    """_load() runs from __init__, and every /api/rag request builds a FRESH
    Collection - so a warning with no dedup guard fires on essentially every
    request for a collection with any corrupt lines at all. Must warn on the
    first Collection() for this directory and stay silent on a second one for
    the SAME directory/count, matching the sibling _note_vector_degrade
    warn-once pattern. self.corrupt must still be set on EVERY load regardless -
    only the duplicate LOG LINE is suppressed, not the actual state."""
    base = _seed(tmp_path)
    chunks_file = base / "kb" / "chunks.jsonl"
    with chunks_file.open("a", encoding="utf-8") as fh:
        fh.write("\nnot valid json\n")

    with caplog.at_level("WARNING", logger="localm"):
        c1 = Collection("kb", base=base)
        caplog.clear()
        c2 = Collection("kb", base=base)

    assert c1.corrupt is True and c2.corrupt is True, (
        "the malformed-line state must be set on every load, dedup is only "
        "about the LOG line")
    assert "malformed" not in caplog.text, (
        f"a second Collection() for the same directory re-logged the "
        f"warning instead of being deduped: {caplog.text!r}")


def test_malformed_chunk_line_warning_still_fires_for_a_different_collection(tmp_path, caplog):
    """The dedup key must include the collection directory - two DIFFERENT
    collections each having their own corruption must both be reported, never
    collapsed into "one warning covers every collection", which would hide a
    genuinely broken SECOND collection behind the first one's warning."""
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    base1 = _seed(tmp_path / "one")
    base2 = _seed(tmp_path / "two")
    for base in (base1, base2):
        (base / "kb" / "chunks.jsonl").open("a", encoding="utf-8").write("\nnot valid json\n")

    with caplog.at_level("WARNING", logger="localm"):
        Collection("kb", base=base1)
        first_text = caplog.text
        caplog.clear()
        Collection("kb", base=base2)
        second_text = caplog.text

    assert "malformed" in first_text
    assert "malformed" in second_text, (
        "a different collection's identical-shaped corruption must still "
        "warn - the dedup key collapsed two distinct collections into one")


def test_chunk_line_missing_text_key_does_not_crash(tmp_path):
    base = _seed(tmp_path)
    chunks_file = base / "kb" / "chunks.jsonl"
    with chunks_file.open("a", encoding="utf-8") as fh:
        fh.write('\n{"source": "x", "pos": 1}\n')   # dict but no "text"
    c = Collection("kb", base=base)
    assert isinstance(c.query("fox"), list)
    assert c.stats()["corrupt"] is True


def test_valid_meta_docs_map_preserved_on_chunk_corruption(tmp_path):
    # meta.json is VALID; only a chunk line is corrupt. The real docs map (with
    # hash/mtime/size) must be kept, NOT overwritten by the chunk-source
    # reconstruction, which is gated on META corruption specifically.
    base = _seed(tmp_path)
    meta_before = json.loads((base / "kb" / "meta.json").read_text(encoding="utf-8"))
    doc_key = next(iter(meta_before["docs"]))
    assert "hash" in meta_before["docs"][doc_key]
    with (base / "kb" / "chunks.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("\n999\n")
    c = Collection("kb", base=base)
    assert c.stats()["corrupt"] is True
    assert c._meta["docs"].get(doc_key, {}).get("hash") == \
        meta_before["docs"][doc_key]["hash"], "valid docs map must not be clobbered"


# --------------------------------------------------------------------------- #
#  a non-finite embedding must degrade to BM25, never drop a chunk             #
# --------------------------------------------------------------------------- #
def test_nan_embedding_does_not_silently_drop_chunk(tmp_path):
    base = tmp_path / "rag"
    d = tmp_path / "docs"
    d.mkdir()
    docs = {"alpha.txt": "alpha unicorn magical creature",
            "beta.txt": "beta ordinary horse animal",
            "gamma.txt": "gamma plain donkey beast"}
    for name, text in docs.items():
        (d / name).write_text(text, encoding="utf-8")

    # embed_fn: the doc that actually matches the query gets a NaN component.
    def embed_fn(texts):
        out = []
        for t in texts:
            if "unicorn" in t or "alpha" in t:
                out.append([float("nan"), 0.0, 0.0])
            else:
                out.append([0.0, 1.0, 0.0])
        return out

    c = Collection("kb", base=base).create()
    c.add_paths([str(d / n) for n in docs], embed_fn=embed_fn)

    # A finite query embedding; the matching doc has a NaN stored vector.
    hits = c.query("alpha unicorn", embed_fn=lambda ts: [[0.0, 0.0, 1.0]])
    sources = [h["source"] for h in hits]
    assert any("alpha" in s for s in sources), (
        "the lexically-matching chunk was silently dropped by a NaN cosine")
    for h in hits:
        assert math.isfinite(h["score"]), f"non-finite score leaked: {h}"


def test_nan_in_vectors_json_degrades_with_reason(tmp_path):
    base = tmp_path / "rag"
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.txt").write_text("mitochondria powerhouse of the cell", encoding="utf-8")
    c = Collection("kb", base=base).create()
    c.add_paths([str(d / "a.txt")], embed_fn=lambda ts: [[1.0, 0.0, 0.0] for _ in ts])
    # Corrupt vectors.json with a NaN component.
    vf = base / "kb" / "vectors.json"
    vf.write_text('{"dim": 3, "vectors": [[1.0, NaN, 0.0]]}', encoding="utf-8")
    c2 = Collection("kb", base=base)
    # Must still answer (degraded to BM25) and SURFACE why, not silently
    # score NaN.
    assert c2.query("mitochondria"), "must still answer lexically"
    assert c2.stats()["vector_degrade_reason"], "non-finite vectors must be surfaced"


def test_nan_query_embedding_degrades_not_empties(tmp_path):
    # A NaN/inf QUERY embedding must not empty the result set (every cosine would
    # be nan, nan !> 0); it degrades to BM25 with a surfaced reason.
    base = tmp_path / "rag"
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.txt").write_text("photosynthesis converts sunlight into energy", encoding="utf-8")
    c = Collection("kb", base=base).create()
    c.add_paths([str(d / "a.txt")], embed_fn=lambda ts: [[1.0, 0.0, 0.0] for _ in ts])
    hits = c.query("photosynthesis", embed_fn=lambda ts: [[float("nan"), 0.0, 0.0]])
    assert hits, "a NaN query embedding must not empty results (BM25 must answer)"
    assert c.stats()["vector_degrade_reason"], "non-finite query embedding must be surfaced"


# --------------------------------------------------------------------------- #
#  chunk pos must be the true 1-based start line, incl. odd blank gaps         #
# --------------------------------------------------------------------------- #
def _section(name):
    # >CHUNK_CHARS so each section lands in its OWN chunk carrying its own pos
    # (small paragraphs would pack into one chunk with a single pos).
    return f"SECTION {name} " + ("z" * 1300)


def _pos_of(chunks, marker):
    for ch in chunks:
        if f"SECTION {marker}" in ch["text"]:
            return ch["pos"]
    return None


def test_chunk_pos_correct_with_odd_blank_line_gaps():
    # ALPHA on line 1; BETA after a 3-newline gap -> line 4; GAMMA after another
    # 3-newline gap -> line 7.
    text = _section("ALPHA") + "\n\n\n" + _section("BETA") + "\n\n\n" + _section("GAMMA")
    chunks = chunk_text(text)
    assert _pos_of(chunks, "ALPHA") == 1
    assert _pos_of(chunks, "BETA") == 4, f"BETA pos wrong: {_pos_of(chunks, 'BETA')}"
    assert _pos_of(chunks, "GAMMA") == 7, f"GAMMA pos wrong: {_pos_of(chunks, 'GAMMA')}"


def test_chunk_pos_correct_with_even_blank_gaps():
    # Even-numbered blank gaps: FIRST on line 1; SECOND after a 2-nl gap -> line 3;
    # THIRD after a 4-nl gap -> line 7.
    text = _section("FIRST") + "\n\n" + _section("SECOND") + "\n\n\n\n" + _section("THIRD")
    chunks = chunk_text(text)
    assert _pos_of(chunks, "FIRST") == 1
    assert _pos_of(chunks, "SECOND") == 3, f"SECOND pos wrong: {_pos_of(chunks, 'SECOND')}"
    assert _pos_of(chunks, "THIRD") == 7, f"THIRD pos wrong: {_pos_of(chunks, 'THIRD')}"
