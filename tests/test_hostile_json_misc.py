# SPDX-License-Identifier: AGPL-3.0-or-later
"""Session logs, RAG collection files, process output and sidecars that hold an
over-nested or over-long-integer JSON document: each reader keeps going on its
documented fallback instead of raising ``RecursionError`` / the digit-limit
``ValueError``."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from localm import cpu_backend_select, netname
from localm.plugins.builtin.memory import plug as memory_plug
from localm.rag import Collection
from tests._hostile_json import HOSTILE

USER_TURN = json.dumps({"type": "user", "data": {"content": "hello there"}})


# ------------------------------------------------------- memory plugin

@HOSTILE
def test_session_text_skips_a_hostile_line_and_keeps_the_rest(doc, tmp_path):
    log = tmp_path / "s.jsonl"
    log.write_text(doc + "\n" + USER_TURN + "\n", encoding="utf-8")
    assert memory_plug._session_text(log) == "User: hello there"


@HOSTILE
def test_recent_sessions_skip_a_hostile_line_and_keep_the_rest(
        doc, tmp_path, monkeypatch):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "a.jsonl").write_text(doc + "\n" + USER_TURN + "\n", encoding="utf-8")
    monkeypatch.setattr(memory_plug, "_home", lambda: tmp_path)
    assert memory_plug._recent_sessions_text() == "User: hello there"


@HOSTILE
def test_episodic_watermark_that_is_hostile_means_process_from_scratch(doc, tmp_path):
    store = SimpleNamespace(path=tmp_path / "episodic.jsonl")
    memory_plug._episodic_watermark_path(store).write_text(doc, encoding="utf-8")
    assert memory_plug._read_episodic_watermark(store) == 0.0
    assert memory_plug._read_episodic_stems(store) == set()


# ----------------------------------------------------------------- RAG

@pytest.fixture
def collection(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("Sourdough starter needs flour and water.\n" * 4,
                      encoding="utf-8")
    base = tmp_path / "rag"
    coll = Collection("kb", base=base).create()
    coll.add_paths([source])
    assert coll.stats()["n_chunks"] > 0
    return base, coll.stats()["n_chunks"]


@HOSTILE
def test_rag_hostile_meta_json_flags_the_collection_and_keeps_its_chunks(
        doc, collection):
    base, n_chunks = collection
    (base / "kb" / "meta.json").write_text(doc, encoding="utf-8")
    reopened = Collection("kb", base=base)
    assert reopened.corrupt is True
    assert reopened.stats()["n_chunks"] == n_chunks


@HOSTILE
def test_rag_hostile_chunk_line_is_counted_and_the_rest_load(doc, collection):
    base, n_chunks = collection
    with open(base / "kb" / "chunks.jsonl", "a", encoding="utf-8") as fh:
        fh.write("\n" + doc + "\n")
    reopened = Collection("kb", base=base)
    assert reopened.stats()["n_chunks"] == n_chunks
    assert reopened.chunks_bad_lines == 1


@HOSTILE
def test_rag_hostile_vectors_json_degrades_to_lexical_retrieval(doc, collection):
    base, n_chunks = collection
    (base / "kb" / "vectors.json").write_text(doc, encoding="utf-8")
    reopened = Collection("kb", base=base)
    assert reopened.vector_degrade_reason is not None
    assert reopened.stats()["n_chunks"] == n_chunks
    assert reopened.query("sourdough flour")


# ------------------------------------------------- process output boundary

def _completed(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["x"], returncode=0, stdout=stdout,
                                       stderr="")


@HOSTILE
def test_tailscale_status_that_is_hostile_is_unreadable(doc, monkeypatch):
    monkeypatch.setattr(netname, "_tailscale_cli", lambda: "tailscale")
    monkeypatch.setattr(netname.subprocess, "run",
                        lambda *a, **k: _completed(doc))
    assert netname._run_tailscale_status() is None


@HOSTILE
def test_cpu_tier_probe_with_a_hostile_verdict_scores_none(doc, monkeypatch):
    monkeypatch.setattr(cpu_backend_select.subprocess, "run",
                        lambda *a, **k: _completed("@@VERDICT@@" + doc + "\n"))
    assert cpu_backend_select._probe_score(Path("cand.so"), Path(".")) is None


@pytest.mark.parametrize("verdict", ["[1]", "5", "null"])
def test_cpu_tier_probe_with_a_wrong_shape_verdict_scores_none(verdict, monkeypatch):
    monkeypatch.setattr(cpu_backend_select.subprocess, "run",
                        lambda *a, **k: _completed("@@VERDICT@@" + verdict + "\n"))
    assert cpu_backend_select._probe_score(Path("cand.so"), Path(".")) is None


@pytest.mark.parametrize("vectors", ["[1]", "5", "null"])
def test_rag_vectors_json_that_is_not_an_object_degrades_to_lexical(
        vectors, collection):
    base, n_chunks = collection
    (base / "kb" / "vectors.json").write_text(vectors, encoding="utf-8")
    reopened = Collection("kb", base=base)
    assert reopened.vector_degrade_reason is not None
    assert reopened.stats()["n_chunks"] == n_chunks


@HOSTILE
def test_cpu_tier_marker_that_is_hostile_is_no_marker(doc, tmp_path):
    cpu_backend_select._marker_path(tmp_path).write_text(doc, encoding="utf-8")
    assert cpu_backend_select._read_marker(tmp_path) is None
