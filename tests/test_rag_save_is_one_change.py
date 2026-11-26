# SPDX-License-Identifier: AGPL-3.0-or-later
"""A collection's meta.json, chunks.jsonl and vectors.json are saved as one
change: a process that ends, a disk that fills or a file another program holds
at any step of a save leaves the collection as it was or as saved, never a mix
(vectors beside other chunks, a document listed but not indexed, a model name
over another model's vectors)."""

from __future__ import annotations

import ast
import errno
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import localm.rag.store as store
from localm.rag import Collection
from localm.rag._store import commit

FILES = {"meta.json", "chunks.jsonl", "vectors.json"}


class _Killed(BaseException):
    """Stands in for the end of the process."""


def _embed(texts, salt=b""):
    return [[b / 255 for b in hashlib.sha256(salt + t.encode()).digest()[:8]]
            for t in texts]


def _embed_b(texts):
    return _embed(texts, b"B")


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "lh"))
    return tmp_path / "rag"


@pytest.fixture
def docs(tmp_path):
    return _write_docs(tmp_path / "docs")


def _write_docs(d):
    d.mkdir(exist_ok=True)
    for old in d.iterdir():
        old.unlink()
    for n in ("alpha", "beta", "gamma"):
        (d / f"{n[0]}.txt").write_text(f"{n} text", encoding="utf-8")
    return d


def _build(base, docs):
    if (base / "kb").exists():
        store.delete_collection("kb", base=base)
    _write_docs(docs)
    Collection("kb", base=base).create().add_paths(
        [docs], embed_fn=_embed, model_name="A")


def _state(base):
    """What a fresh reader gets: docs, model, and each chunk with whether its
    vector is the one its own text gives under the recorded model."""
    c = Collection("kb", base=base, cache=False)
    fn = _embed_b if c.embedding_model() == "B" else _embed
    rows = []
    for i, chunk in enumerate(c._chunks):
        vec = None if c._vectors is None else c._vectors[i]
        right = fn([chunk["text"]])[0]
        if not vec:
            says = "none"
        elif all(abs(x - y) < 1e-6 for x, y in zip(vec, right, strict=True)):
            says = "own"
        else:
            says = "WRONG"
        rows.append((Path(chunk["source"]).name, chunk["text"], says))
    return (sorted((Path(p).name, e["chunks"], e["hash"])
                   for p, e in c._meta["docs"].items()),
            c.embedding_model(), rows, c.vector_degrade_reason)


def _edit(c, docs):
    (docs / "a.txt").write_text("alpha NEW text", encoding="utf-8")
    c.add_paths([docs / "a.txt"], embed_fn=_embed, model_name="A")


def _new(c, docs):
    (docs / "d.txt").write_text("delta text", encoding="utf-8")
    c.add_paths([docs / "d.txt"], embed_fn=_embed, model_name="A")


def _remove(c, docs):
    assert c.remove_doc(str(docs / "a.txt"))


def _reembed(c, docs):
    c.reembed(embed_fn=_embed_b, model_name="B")


def _drop(c, docs):
    c.add_paths([docs], force=True, allow_vector_loss=True)


def _upload(c, docs):
    c.add_uploads([{"filename": "u.txt", "data": b"upload text"}],
                  embed_fn=_embed, model_name="A")


OPS = {"edit": _edit, "new": _new, "remove": _remove, "reembed": _reembed,
       "drop": _drop, "upload": _upload}


def _killed_op(monkeypatch, base, docs, name, nth):
    """Run op *name*, ended (as far as the disk can tell) after its *nth* step
    that changes a file. (the steps taken, whether it ran to its end)."""
    steps: list = []

    def counted(fn, name_of):
        def run(*args, **kwargs):
            result = fn(*args, **kwargs)
            steps.append(name_of(*args))
            if len(steps) == nth:
                raise _Killed
            return result
        return run

    with monkeypatch.context() as m:
        m.setattr(store, "_storekit_atomic_write", counted(
            store._storekit_atomic_write, lambda path, _d: f"write {Path(path).name}"))
        m.setattr(commit, "replace_atomic", counted(
            commit.replace_atomic, lambda _s, dst: f"move {Path(dst).name}"))
        m.setattr(commit, "_unlink", counted(
            commit._unlink, lambda path: f"remove {Path(path).name}"))
        m.setattr(commit.StagedSave, "discard", lambda self: None)
        try:
            OPS[name](Collection("kb", base=base), docs)
        except _Killed:
            return steps, False
        return steps, True


@pytest.mark.parametrize("name", sorted(OPS))
class TestAProcessThatEndsDuringASave:
    def test_leaves_the_collection_as_it_was_or_as_saved(
            self, base, docs, monkeypatch, name):
        _build(base, docs)
        before = _state(base)
        OPS[name](Collection("kb", base=base), docs)
        after = _state(base)
        assert before != after, "the operation changes something"
        seen = set()
        finished = False
        for nth in range(1, 30):
            _build(base, docs)
            steps, finished = _killed_op(monkeypatch, base, docs, name, nth)
            if finished:
                break
            got = _state(base)
            committed = "write save.pending" in steps
            assert got == (after if committed else before), (nth, steps)
            seen.add(committed)
        assert finished and seen == {False, True}
        assert _state(base) == after

    def test_the_next_write_finishes_it_and_leaves_no_stray_file(
            self, base, docs, monkeypatch, name):
        _build(base, docs)
        OPS[name](Collection("kb", base=base), docs)
        after = _state(base)
        finished = False
        for nth in range(1, 30):
            _build(base, docs)
            steps, finished = _killed_op(monkeypatch, base, docs, name, nth)
            if finished:
                break
            if "write save.pending" in steps:
                Collection("kb", base=base).remove_doc("not there")
            else:
                OPS[name](Collection("kb", base=base), docs)
            assert _state(base) == after, (nth, steps)
            names = {p.name for p in (base / "kb").iterdir()}
            assert names <= FILES, (nth, steps, names)
        assert finished


_CHILD = """
import os, sys
sys.path.insert(0, sys.argv[4])
sys.path.insert(0, sys.argv[5])
from pathlib import Path
import localm.rag.store as store
from localm.rag import Collection
from localm.rag._store import commit
import test_rag_save_is_one_change as t
real = {what}
def end(*a, **k):
    real(*a, **k)
    if Path(a[{arg}]).name.startswith(sys.argv[3]):
        os._exit(9)
{what} = end
t._edit(Collection("kb", base=Path(sys.argv[1])), Path(sys.argv[2]))
os._exit(0)
"""


@pytest.mark.parametrize("what, arg, at, committed", [
    ("store._storekit_atomic_write", 0, "chunks.jsonl", False),
    ("store._storekit_atomic_write", 0, "save.pending", True),
    ("commit.replace_atomic", 1, "chunks.jsonl", True),
])
def test_a_real_process_that_exits_during_a_save(base, docs, tmp_path, what, arg,
                                                 at, committed):
    _build(base, docs)
    before = _state(base)
    root = Path(__file__).resolve().parent
    r = subprocess.run(
        [sys.executable, "-c", _CHILD.format(what=what, arg=arg), str(base),
         str(docs), at, str(root.parent), str(root)],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "LOCALM_HOME": str(tmp_path / "lh")},
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert r.returncode == 9, r.stderr
    got = _state(base)
    if committed:
        assert got[2] == [("b.txt", "beta text", "own"), ("g.txt", "gamma text", "own"),
                          ("a.txt", "alpha NEW text", "own")]
    else:
        assert got == before


@pytest.mark.parametrize("stage", ["chunks.jsonl.", "vectors.json.", "meta.json.",
                                   "save.pending"])
def test_a_full_disk_leaves_the_collection_as_it_was(base, docs, monkeypatch, stage):
    _build(base, docs)
    before = _state(base)
    real = store._storekit_atomic_write

    def full(path, data):
        if Path(path).name.startswith(stage):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(path, data)

    with monkeypatch.context() as m:
        m.setattr(store, "_storekit_atomic_write", full)
        with pytest.raises(OSError):
            _edit(Collection("kb", base=base), docs)
    assert _state(base) == before
    assert {p.name for p in (base / "kb").iterdir()} == FILES
    _edit(Collection("kb", base=base), docs)
    assert ("a.txt", "alpha NEW text", "own") in _state(base)[2]


def _refuse_move(monkeypatch, refused):
    real = commit.replace_atomic

    def move(src, dst):
        if Path(dst).name in refused:
            raise PermissionError(13, "held by another program")
        return real(src, dst)

    monkeypatch.setattr(commit, "replace_atomic", move)


@pytest.mark.parametrize("held", sorted(FILES))
def test_a_file_another_program_holds_does_not_mix_the_save(
        base, docs, monkeypatch, held):
    _build(base, docs)
    _edit(Collection("kb", base=base), docs)
    after = _state(base)
    _build(base, docs)
    with monkeypatch.context() as m:
        _refuse_move(m, {held})
        _edit(Collection("kb", base=base), docs)
        assert (base / "kb" / commit.MARKER).is_file()
        assert _state(base) == after
    Collection("kb", base=base).remove_doc("not there")
    assert _state(base) == after
    assert {p.name for p in (base / "kb").iterdir()} == FILES


_windows_share_modes = pytest.mark.skipif(
    os.name != "nt", reason="file share modes are a Windows behaviour")


def _hold(path, seconds):
    """Another process that holds *path* open, sharing it with nobody, for
    *seconds*: every open of *path* meanwhile is a sharing violation, as an
    open is while another localm process replaces the file."""
    code = ("import ctypes, sys, time\n"
            "from ctypes import wintypes\n"
            "k = ctypes.WinDLL('kernel32'); k.CreateFileW.restype = wintypes.HANDLE\n"
            "k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, "
            "wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]\n"
            "h = k.CreateFileW(sys.argv[1], 0x80000000, 0, None, 3, 0, None)\n"
            "assert h != wintypes.HANDLE(-1).value\n"
            "print('held', flush=True)\n"
            f"time.sleep({seconds})\n")
    p = subprocess.Popen([sys.executable, "-I", "-S", "-c", code, str(path)],
                         stdout=subprocess.PIPE,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert p.stdout is not None and p.stdout.readline().strip() == b"held"
    return p


def _let_go(p):
    p.kill()
    p.wait(timeout=60)
    assert p.stdout is not None
    p.stdout.close()


@_windows_share_modes
@pytest.mark.parametrize("held", sorted(FILES))
def test_a_read_while_another_process_has_the_file_waits_for_it(base, docs, held):
    _build(base, docs)
    expected = _state(base)
    holder = _hold(base / "kb" / held, 0.4)
    try:
        got = _state(base)
    finally:
        _let_go(holder)
    assert got == expected


@_windows_share_modes
def test_a_file_held_past_the_wait_is_an_error_not_an_empty_collection(base, docs):
    _build(base, docs)
    holder = _hold(base / "kb" / "chunks.jsonl", 60)
    try:
        with pytest.raises(PermissionError):
            Collection("kb", base=base, cache=False)
    finally:
        _let_go(holder)


def test_an_interrupt_after_the_save_took_effect_keeps_it(base, docs, monkeypatch):
    _build(base, docs)
    _edit(Collection("kb", base=base), docs)
    after = _state(base)
    _build(base, docs)

    def interrupted(src, dst):
        raise KeyboardInterrupt

    with monkeypatch.context() as m:
        m.setattr(commit, "replace_atomic", interrupted)
        with pytest.raises(KeyboardInterrupt):
            _edit(Collection("kb", base=base), docs)
    assert _state(base) == after


class TestWhileASaveIsPending:
    @pytest.fixture
    def pending(self, base, docs, monkeypatch):
        """A save that took effect and whose files are all still to be moved."""
        _build(base, docs)
        with monkeypatch.context() as m:
            _refuse_move(m, FILES)
            _new(Collection("kb", base=base), docs)
        assert (base / "kb" / commit.MARKER).is_file()
        return base

    def test_a_write_that_cannot_finish_it_refuses(self, pending, docs, monkeypatch):
        now = _state(pending)
        with monkeypatch.context() as m:
            _refuse_move(m, FILES)
            with pytest.raises(OSError):
                Collection("kb", base=pending).remove_doc(str(docs / "a.txt"))
        assert _state(pending) == now

    @_windows_share_modes
    @pytest.mark.parametrize("held", sorted(FILES))
    def test_a_read_while_another_process_has_a_staged_file_waits_for_it(
            self, pending, held):
        expected = _state(pending)
        save = commit.pending_save(pending / "kb")
        assert save is not None
        holder = _hold(pending / "kb" / f"{held}.{save[0]}.next", 0.4)
        try:
            got = _state(pending)
        finally:
            _let_go(holder)
        assert got == expected

    def test_the_meta_only_paths_do_not_answer_from_the_old_files(self, pending):
        assert Collection.peek_stats("kb", pending) is None
        assert Collection.peek_detail("kb", pending) is None

    def test_a_backfill_finishes_it(self, pending):
        c = Collection.load_and_maybe_backfill("kb", pending)
        assert c.stats()["n_docs"] == 4
        assert not (pending / "kb" / commit.MARKER).exists()
        assert Collection.peek_stats("kb", pending)["n_docs"] == 4

    def test_confinement_is_judged_on_the_documents_that_are_served(
            self, pending):
        keys = [k for k in Collection("kb", base=pending).documents()
                if not k.endswith("d.txt")]
        assert len(keys) == 3
        assert Collection.confined_to("kb", keys, pending) is False

    def test_a_backfill_that_cannot_finish_it_writes_nothing(
            self, pending, monkeypatch):
        old = (pending / "kb" / "meta.json").read_bytes()
        with monkeypatch.context() as m:
            _refuse_move(m, FILES)
            c = Collection.load_and_maybe_backfill("kb", pending)
            assert c.stats()["n_docs"] == 4
        assert (pending / "kb" / "meta.json").read_bytes() == old

    def test_recording_model_refs_finishes_it_first(self, pending):
        assert store.record_embedding_model_refs(pending) == []
        assert not (pending / "kb" / commit.MARKER).exists()
        assert Collection("kb", base=pending).stats()["n_docs"] == 4

    def test_recording_model_refs_skips_one_it_cannot_finish(
            self, pending, monkeypatch):
        with monkeypatch.context() as m:
            _refuse_move(m, FILES)
            assert store.record_embedding_model_refs(pending) == ["kb"]

    def test_a_meta_only_save_is_not_lost_behind_it(self, pending):
        c = Collection("kb", base=pending)
        c._meta["note"] = 1
        c._save_meta()
        got = Collection("kb", base=pending, cache=False)
        assert got._meta.get("note") == 1 and got.stats()["n_docs"] == 4

    def test_a_cached_reader_of_another_process_sees_the_save(
            self, base, docs, monkeypatch):
        _build(base, docs)
        old = store._collection_cache_fingerprint(base / "kb")
        with monkeypatch.context() as m:
            _refuse_move(m, FILES)
            _new(Collection("kb", base=base), docs)
        assert store._collection_cache_fingerprint(base / "kb") != old


def test_a_save_without_vectors_removes_the_old_ones(base, docs, monkeypatch):
    _build(base, docs)
    with monkeypatch.context() as m:
        _refuse_move(m, FILES)
        _drop(Collection("kb", base=base), docs)
    marker = json.loads((base / "kb" / commit.MARKER).read_text(encoding="utf-8"))
    assert sorted(marker["files"]) == ["chunks.jsonl", "meta.json"]
    assert (base / "kb" / "vectors.json").is_file()
    assert {row[2] for row in _state(base)[2]} == {"none"}
    Collection("kb", base=base).remove_doc("not there")
    assert not (base / "kb" / "vectors.json").exists()


@pytest.mark.parametrize("content", [
    "", "not json", "[]", "{}", '{"id": "../../x", "files": ["meta.json"]}',
    '{"id": 5, "files": []}',
    '{"id": "%s", "files": ["auth.key"]}' % ("0" * 32),
    '{"id": "%s", "files": [["meta.json"]]}' % ("0" * 32),
    "[" * 100_000], ids=range(9))
def test_a_marker_that_describes_no_save_changes_nothing(base, docs, content):
    _build(base, docs)
    before = _state(base)
    d = base / "kb"
    (d / commit.MARKER).write_text(content, encoding="utf-8")
    staged = d / f"chunks.jsonl.{'0' * 32}.next"
    staged.write_text("", encoding="utf-8")
    assert _state(base) == before
    commit.finish_save(d)
    assert staged.exists()
    _edit(Collection("kb", base=base), docs)
    assert not (d / commit.MARKER).exists()
    assert ("a.txt", "alpha NEW text", "own") in _state(base)[2]


def test_a_marker_that_cannot_be_read_stops_a_write(base, docs, monkeypatch):
    _build(base, docs)
    (base / "kb" / commit.MARKER).write_text("{}", encoding="utf-8")
    real = Path.read_text

    def held(self, *args, **kwargs):
        if self.name == commit.MARKER:
            raise PermissionError(13, "held by another program")
        return real(self, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(Path, "read_text", held)
        with pytest.raises(PermissionError):
            commit.finish_save(base / "kb")


def test_what_an_interrupted_write_left_is_removed_and_nothing_else(base, docs):
    _build(base, docs)
    d = base / "kb"
    gone = [f"chunks.jsonl.{'a' * 32}.next", f"vectors.json.{'a' * 32}.next.12.34.tmp",
            "meta.json.12.34.tmp", "save.pending.12.34.tmp"]
    kept = ["vectors.json.rejected", "vectors.json.rejected.2", "notes.tmp",
            "chunks.jsonl.bak"]
    for n in gone + kept:
        (d / n).write_text("x", encoding="utf-8")
    Collection("kb", base=base).remove_doc("not there")
    assert {p.name for p in d.iterdir()} == FILES | set(kept)


def test_no_module_writes_the_files_past_a_save():
    """chunks.jsonl and vectors.json are only ever written through a save, and a
    function that writes meta.json alone finishes a pending save first."""
    root = Path(store.__file__).parent
    meta_writers = set()
    for f in sorted([root / "store.py", *(root / "_store").glob("*.py")]):
        for fn in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if not isinstance(fn, ast.FunctionDef):
                continue
            calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)]
            names = [getattr(c.func, "attr", getattr(c.func, "id", "")) for c in calls]
            for call, name in zip(calls, names, strict=True):
                if name not in ("_atomic_write", "_storekit_atomic_write"):
                    continue
                text = "".join(ast.unparse(a) for a in call.args)
                assert "chunks.jsonl" not in text and "vectors.json" not in text, (
                    f.name, fn.name)
                if "meta.json" in text:
                    meta_writers.add(fn.name)
                    assert "finish_save" in names, (f.name, fn.name)
    assert meta_writers == {"_save_meta", "load_and_maybe_backfill",
                            "record_embedding_model_refs"}
