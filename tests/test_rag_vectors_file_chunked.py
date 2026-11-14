# SPDX-License-Identifier: AGPL-3.0-or-later
"""vectors.json is written and read a few rows per C call.

A single json.dumps / json.loads over a large collection's vectors holds the
GIL for seconds, which freezes the event loop and every other thread of the
server. The chunked writer must produce the exact bytes the single call did,
the chunked reader must return exactly what json.loads returns (and fail the
same way on a broken file), and a save plus a load of a large collection must
never keep another thread from running for long.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from localm.rag import Collection
from localm.rag._store import vecfile
from localm.rag._store.vecfile import load_vectors_json, vectors_json_pieces


def _rows(n, dim, seed=7, holes=()):
    rnd = random.Random(seed)
    return [None if i in holes else [rnd.uniform(-1, 1) for _ in range(dim)]
            for i in range(n)]


@pytest.mark.parametrize("dim, rows", [
    (3, []),
    (None, []),
    (3, [[0.1, 0.2, 0.3]]),
    (4, _rows(200, 4, holes={0, 5, 199})),
    (2, [(1.0, -0.0), None, [1e-300, 1e300]]),
    (1, [[float("nan")], [float("inf")], [-float("inf")]]),
    (3, _rows(130, 3)),
])
def test_pieces_concatenate_to_the_single_dumps_bytes(dim, rows):
    assert "".join(vectors_json_pieces(dim, rows)) == json.dumps(
        {"dim": dim, "vectors": rows})


def _equal(a, b):
    """Equality that treats NaN as equal to NaN."""
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


_VALID = [
    json.dumps({"dim": 3, "vectors": _rows(50, 3, holes={3})}),
    json.dumps({"dim": None, "vectors": []}),
    json.dumps({"dim": 2, "vectors": [[1, 2], [3.5, -4e10]]}),
    json.dumps({"dim": 1, "vectors": [[float("nan")]]}),
    '\n {\n "dim" : 2 ,\n\t"vectors":[ [1,2] ,\r\n null , [3,4] ]\n}\n ',
    '{"vectors": [[1, 2]], "dim": 2}',
    '{"dim": 2, "vectors": [[1, 2]], "extra": true}',
    '{"dim": 2, "vectors": [[1, 2]], "vectors": [[9, 9]]}',
    '{"dim": 2.0, "vectors": [[1, 2]]}',
    '{"dim": 2, "vectors": [1, "x", {"a": [1]}, [[1]], true]}',
    '{"dim": -0, "vectors": [[1e400]]}',
    '{"dim": 1, "vectors": [1.5, 12e3, 1.25, -0.5e-2, 1234567, null, true]}',
    '{"dim": 1, "vectors": [1.5 , 12e3\n, 7]}',
    '[]',
    '{}',
    # Longer than the reader's first read, so rows cross read boundaries.
    '{"dim": 1, "vectors": [' + ", ".join(["1.5", "12e3", "-0.5e-2", "1234567", "null"] * 500) + "]}",
    json.dumps({"dim": 3, "vectors": _rows(300, 3, holes={1, 150})}),
    '{"dim": 2, "vectors": [' + " ,\n ".join(["[1.5, 2]"] * 900) + ' ]\n}\n',
]


@pytest.mark.parametrize("read_bytes", [1, 7, 1 << 20])
@pytest.mark.parametrize("text", _VALID, ids=[str(i) for i in range(len(_VALID))])
def test_reader_returns_what_json_loads_returns(tmp_path, monkeypatch, text, read_bytes):
    monkeypatch.setattr(vecfile, "_READ_BYTES", read_bytes)
    p = tmp_path / "vectors.json"
    p.write_text(text, encoding="utf-8")
    assert _equal(load_vectors_json(p), json.loads(p.read_text(encoding="utf-8")))


_INVALID = [
    '{"dim": 3, "vectors": [[1, 2, 3], [4, 5',
    '{"dim": 3, "vectors": [[1, 2, 3], ]}',
    '{"dim": 3, "vectors": [[1, 2, 3] [4, 5, 6]]}',
    '{"dim": 3, "vectors": [[1, 2, 3]]',
    '{"dim": 3, "vectors": [[1, 2, 3]]} x',
    '{"dim": 3, "vectors": [nul]}',
    '{"dim": 2, "vectors": [' + ", ".join(["[1.5, 2]"] * 900) + ', [3, ',
    '{"dim": 2, "vectors": [' + ", ".join(["[1.5, 2]"] * 900) + ' [3, 4]]}',
    '﻿{"dim": 3, "vectors": []}',
    '',
]


@pytest.mark.parametrize("read_bytes", [1, 1 << 20])
@pytest.mark.parametrize("text", _INVALID, ids=[str(i) for i in range(len(_INVALID))])
def test_reader_rejects_what_json_loads_rejects(tmp_path, monkeypatch, text, read_bytes):
    monkeypatch.setattr(vecfile, "_READ_BYTES", read_bytes)
    p = tmp_path / "vectors.json"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        json.loads(p.read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        load_vectors_json(p)


def test_reader_rejects_invalid_utf8(tmp_path):
    p = tmp_path / "vectors.json"
    p.write_bytes(b'{"dim": 1, "vectors": [[1], "\xff"]}')
    with pytest.raises(UnicodeDecodeError):
        load_vectors_json(p)


_DEEP = "[" * 200_000 + "]" * 200_000


@pytest.mark.parametrize("text", [
    '{"dim": 1, "vectors": [' + _DEEP + "]}",
    '{"vectors": [' + _DEEP + '], "dim": 1}',
], ids=["written-shape", "other-shape"])
def test_hostile_nesting_degrades_the_collection_instead_of_raising(tmp_path, text):
    base = tmp_path / "collections"
    base.mkdir()
    coll = Collection("deep", base=base, cache=False)
    coll.create()
    coll._load(use_cache=False)
    coll._chunks = [{"text": "t", "source": "upload:a", "format": "text"}]
    coll._meta["docs"]["upload:a"] = {"size": 1, "hash": "h", "chunks": 1, "uploaded": True}
    coll._save()
    (base / "deep" / "vectors.json").write_text(text, encoding="utf-8")
    with pytest.raises(RecursionError):
        load_vectors_json(base / "deep" / "vectors.json")
    reopened = Collection("deep", base=base, cache=False)
    assert reopened._vectors is None
    assert "unreadable (RecursionError)" in reopened.vector_degrade_reason


_DIM = 384
_ROWS = 6000


@pytest.fixture
def big_collection(tmp_path):
    """A collection with _ROWS vectors of _DIM components, saved through the
    real Collection API."""
    base = tmp_path / "collections"
    base.mkdir()
    coll = Collection("big", base=base, cache=False)
    coll.create()
    coll._load(use_cache=False)
    coll._chunks = [{"text": f"chunk {i}", "source": f"upload:s{i // 50}.txt",
                     "format": "text"} for i in range(_ROWS)]
    for i in range(0, _ROWS, 50):
        coll._meta["docs"][f"upload:s{i // 50}.txt"] = {"size": 1, "hash": str(i),
                                                       "chunks": 50, "uploaded": True}
    coll._vectors = _rows(_ROWS, _DIM)
    coll._vec_dim = _DIM
    coll._save()
    return base


class _Starvation:
    """A thread that sleeps 10 ms at a time and records its worst lateness: how
    long it could not run beyond its own sleep."""

    def __init__(self) -> None:
        self.worst = 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            time.sleep(0.01)
            self.worst = max(self.worst, time.perf_counter() - t0 - 0.01)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(5)


class _RunnerStall:
    """A separate process that sleeps 10 ms at a time while the block runs and
    reports its worst lateness: how long the machine itself kept a process
    from running. Nothing in this process can make it late."""

    def __init__(self) -> None:
        self.worst = 0.0
        self._proc = None

    def __enter__(self):
        code = ("import sys, time, threading\n"
                "w = [0.0]\n"
                "def run():\n"
                "    while True:\n"
                "        t = time.perf_counter(); time.sleep(0.01)\n"
                "        w[0] = max(w[0], time.perf_counter() - t - 0.01)\n"
                "threading.Thread(target=run, daemon=True).start()\n"
                "sys.stdin.read()\n"
                "print(w[0])\n")
        self._proc = subprocess.Popen(
            [sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        out, _ = self._proc.communicate("", timeout=10)
        self.worst = float(out.strip() or 0.0)


_MAX_STARVED_S = 0.5


def test_indexing_into_a_large_collection_never_starves_another_thread(big_collection, tmp_path):
    """add_paths loads the whole collection and saves it again; neither may keep
    another thread from running for more than _MAX_STARVED_S."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "new.txt").write_text("a new document about compressors", encoding="utf-8")
    vec = [0.5] * _DIM
    with _RunnerStall() as runner, _Starvation() as s:
        result = Collection("big", base=big_collection, load=False, cache=False).add_paths(
            [docs], embed_fn=lambda texts: [vec for _ in texts])
    assert result["added"] == 1
    reread = Collection("big", base=big_collection, cache=False)
    assert len(reread._vectors) == result["chunks"] > _ROWS
    assert reread._vectors[-1] == vec
    assert s.worst - runner.worst < _MAX_STARVED_S, (
        f"another thread was starved {s.worst:.2f} s ({runner.worst:.2f} s of it the machine's)")


_MAX_HEALTH_S = 0.75


def test_health_answers_while_a_large_collection_is_indexed(big_collection, tmp_path, monkeypatch):
    """A request served by the event loop answers within _MAX_HEALTH_S during
    the whole of add_paths: its load, its embedding and its save."""
    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "new.txt").write_text("a new document about gearboxes", encoding="utf-8")
    result = {}

    def index():
        result.update(Collection("big", base=big_collection, load=False, cache=False).add_paths(
            [docs], embed_fn=lambda texts: [[0.5] * _DIM for _ in texts]))

    worker = threading.Thread(target=index)
    worst = 0.0
    answers = 0
    with _RunnerStall() as runner, TestClient(create_app(None)) as client:
        client.get("/health")
        worker.start()
        while worker.is_alive():
            t0 = time.perf_counter()
            assert client.get("/health").status_code in (200, 503)
            worst = max(worst, time.perf_counter() - t0)
            answers += 1
            time.sleep(0.01)
    worker.join()
    assert result["added"] == 1
    assert answers > 10
    assert worst - runner.worst < _MAX_HEALTH_S, (
        f"/health took {worst:.2f} s during indexing ({runner.worst:.2f} s of it the machine's)")


def test_the_saved_file_is_exactly_what_a_single_dumps_wrote(big_collection):
    coll = Collection("big", base=big_collection, cache=False)
    text = (Path(big_collection) / "big" / "vectors.json").read_text(encoding="utf-8")
    assert text == json.dumps({"dim": _DIM, "vectors": coll._vectors})


def test_a_large_index_reports_its_load_and_save(big_collection, tmp_path, monkeypatch):
    from localm.rag import store
    from localm.rag._store import files
    monkeypatch.setattr(files, "_REPORT_EVERY_S", 0.0)
    monkeypatch.setattr(store, "_REPORT_VECTORS_BYTES", 1)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "new.txt").write_text("another document", encoding="utf-8")
    calls = []
    Collection("big", base=big_collection, load=False, cache=False).add_paths(
        [docs], embed_fn=lambda texts: [[0.1] * _DIM for _ in texts],
        on_progress=lambda text, **kw: calls.append((text, kw)))
    texts = [t for t, _ in calls]
    assert any(t.startswith("loading the collection (") for t in texts)
    assert f"saving the index ({_ROWS + 1} vectors)..." in texts
    structured = [kw for _, kw in calls if kw.get("phase") == "saving the index"]
    assert structured and structured[-1]["total"] == _ROWS + 1
    assert all(kw["unit"] == "vectors" for kw in structured)
    dones = [kw["done"] for kw in structured]
    assert dones == sorted(dones) and dones[-1] == _ROWS + 1


def test_a_large_save_always_ends_on_its_total(big_collection, monkeypatch):
    from localm.rag._store import files
    monkeypatch.setattr(files, "_REPORT_EVERY_S", 3600.0)
    calls = []
    coll = Collection("big", base=big_collection, cache=False)
    coll._save(lambda text, **kw: calls.append(kw))
    structured = [kw for kw in calls if kw]
    assert [(kw["done"], kw["total"]) for kw in structured] == [(_ROWS, _ROWS)]


def test_a_small_index_reports_neither(tmp_path):
    base = tmp_path / "collections"
    base.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    Collection("small", base=base).create()
    calls = []
    Collection("small", base=base).add_paths(
        [docs], embed_fn=lambda texts: [[1.0, 0.0] for _ in texts],
        on_progress=lambda text, **kw: calls.append(text))
    assert not [t for t in calls if t.startswith(("loading the collection", "saving the index"))]


def test_unloaded_collection_reads_nothing_until_a_locked_method_loads(big_collection):
    coll = Collection("big", base=big_collection, load=False)
    assert coll.exists()
    assert coll._chunks == [] and coll._vectors is None
