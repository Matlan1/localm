# SPDX-License-Identifier: AGPL-3.0-or-later
"""The user-data stores (memory, episodes, jobs, RAG, gallery index, TTS settings)
read JSON a user can hand-edit or a crash can truncate. An over-nested or
over-long-integer document is treated exactly like any other corrupt document:
the store's documented fallback, never an uncaught ``RecursionError`` or digit
limit ``ValueError``."""

from __future__ import annotations

import json

import pytest

import localm.config as config
from localm.media import gallery
from localm.memory.store import MemoryStore
from localm.plugins.builtin.jobs.store import JobStore
from localm.plugins.builtin.tts import settings as tts_settings
from localm.plugins.coder.episodes import EpisodeStore
from localm.plugins.coder.indexer import ProjectMap
from localm.rag._store.introspect import _CollectionIntrospection
from tests._hostile_json import HOSTILE

GOOD_EPISODE = {"task": "t", "lesson": "l"}


# ------------------------------------------------------------------ memory

@pytest.fixture
def memory(tmp_path):
    return MemoryStore("owner", "chat", "scope", root=tmp_path)


@HOSTILE
def test_memory_record_line_that_is_hostile_is_skipped(doc, memory):
    memory.path.parent.mkdir(parents=True, exist_ok=True)
    memory.path.write_text(doc + "\n", encoding="utf-8")
    memory._load()
    assert memory._records == []


@HOSTILE
def test_memory_vector_sidecar_that_is_hostile_leaves_no_vectors(doc, memory):
    memory._vec_file().parent.mkdir(parents=True, exist_ok=True)
    memory._vec_file().write_text(doc, encoding="utf-8")
    memory._load()
    assert memory._vectors == {}


@HOSTILE
def test_memory_forgotten_archive_skips_a_hostile_line_and_keeps_the_rest(doc, memory):
    memory._forgotten_file().parent.mkdir(parents=True, exist_ok=True)
    memory._forgotten_file().write_text(
        doc + "\n" + json.dumps({"id": "kept"}) + "\n", encoding="utf-8")
    assert memory._load_forgotten() == [{"id": "kept"}]


@HOSTILE
def test_memory_pending_corrections_skip_a_hostile_line(doc, memory):
    memory._corrections_file().parent.mkdir(parents=True, exist_ok=True)
    memory._corrections_file().write_text(doc + "\n", encoding="utf-8")
    assert memory._load_corrections() == []


@HOSTILE
def test_memory_dismissed_set_that_is_hostile_reads_as_empty(doc, memory):
    memory._dismissed_file().parent.mkdir(parents=True, exist_ok=True)
    memory._dismissed_file().write_text(doc, encoding="utf-8")
    assert memory._load_dismissed() == set()


# ---------------------------------------------------------------- episodes

@pytest.fixture
def episodes(tmp_path):
    return EpisodeStore(tmp_path / "project", root=tmp_path / "eps")


@HOSTILE
def test_episode_log_skips_a_hostile_line_and_keeps_the_rest(doc, episodes):
    episodes.path.parent.mkdir(parents=True, exist_ok=True)
    episodes.path.write_text(
        doc + "\n" + json.dumps(GOOD_EPISODE) + "\n", encoding="utf-8")
    assert [e.task for e in episodes.all()] == ["t"]


@HOSTILE
def test_episode_archive_skips_a_hostile_line_and_keeps_the_rest(doc, episodes):
    episodes.archive_path.parent.mkdir(parents=True, exist_ok=True)
    episodes.archive_path.write_text(
        doc + "\n" + json.dumps({"task": "gone"}) + "\n", encoding="utf-8")
    assert [r["task"] for r in episodes.forgotten()] == ["gone"]
    assert episodes.last_forgotten_ok is True


@HOSTILE
def test_episode_vector_sidecar_that_is_hostile_is_recomputed(doc, episodes):
    sidecar = episodes.path.with_suffix(".vec.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(doc, encoding="utf-8")
    vecs = episodes._vectors(["a", "b"], lambda texts: [[1.0] for _ in texts])
    assert vecs == [[1.0], [1.0]]


# -------------------------------------------------------------------- jobs

@HOSTILE
def test_jobs_definitions_that_are_hostile_are_quarantined_and_read_as_empty(
        doc, tmp_path):
    store = JobStore(tmp_path)
    (tmp_path / "jobs.json").write_text(doc, encoding="utf-8")
    assert store._read_all() == {}
    backups = list(tmp_path.glob("jobs.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == doc


@HOSTILE
def test_job_results_skip_a_hostile_file_and_keep_the_rest(doc, tmp_path):
    store = JobStore(tmp_path)
    results = store._result_dir("job1")
    results.mkdir(parents=True)
    (results / "a.json").write_text(doc, encoding="utf-8")
    (results / "b.json").write_text(json.dumps({"output": "ok"}), encoding="utf-8")
    assert store.list_results("job1") == [{"output": "ok"}]


# --------------------------------------------------------------------- rag

@HOSTILE
def test_rag_peek_meta_of_a_hostile_meta_json_falls_back_to_the_full_load(
        doc, tmp_path):
    coll = tmp_path / "docs"
    coll.mkdir()
    (coll / "meta.json").write_text(doc, encoding="utf-8")
    assert _CollectionIntrospection._peek_meta("docs", tmp_path) is None


# ----------------------------------------------------------------- gallery

@HOSTILE
def test_gallery_index_that_is_hostile_fails_closed(doc, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "home_dir", lambda: tmp_path)
    index = tmp_path / "gallery_index" / "image.json"
    index.parent.mkdir(parents=True)
    index.write_text(doc, encoding="utf-8")
    with pytest.raises(gallery.GalleryIndexUnreadable):
        gallery._read_index("image")


# --------------------------------------------------------------------- tts

@HOSTILE
def test_tts_defaults_that_are_hostile_read_as_empty(doc, tmp_path, monkeypatch):
    template = tmp_path / "tts.example.json"
    template.write_text(doc, encoding="utf-8")
    monkeypatch.setattr(tts_settings, "_TEMPLATE", template)
    assert tts_settings.defaults() == {}


@HOSTILE
def test_tts_voices_that_are_hostile_read_as_empty(doc, tmp_path, monkeypatch):
    voices = tmp_path / "voices.json"
    voices.write_text(doc, encoding="utf-8")
    monkeypatch.setattr(tts_settings, "_VOICES", voices)
    assert tts_settings.voices() == []


# ----------------------------------------------------------------- indexer

@HOSTILE
def test_project_map_cache_that_is_hostile_is_a_cache_miss(doc, tmp_path):
    cache = tmp_path / "map.json"
    cache.write_text(doc, encoding="utf-8")
    assert ProjectMap.load_cached_and_reconcile(tmp_path, cache) is None


# ------------------------------------------- non-UTF-8 is not "corrupt JSON"

def test_jobs_definitions_that_are_not_utf8_are_left_intact(tmp_path):
    store = JobStore(tmp_path)
    raw = b'{"jobs": []}\n\xff\xfe tail'
    (tmp_path / "jobs.json").write_bytes(raw)
    with pytest.raises(UnicodeDecodeError):
        store._read_all()
    assert (tmp_path / "jobs.json").read_bytes() == raw
    assert not list(tmp_path.glob("jobs.json.corrupt-*"))


# ------------------------------------------- valid JSON of the wrong shape

@pytest.mark.parametrize("line", ["[1]", "5", "null", '"text"'])
def test_episode_log_skips_a_wrong_shape_line(line, episodes):
    episodes.path.parent.mkdir(parents=True, exist_ok=True)
    episodes.path.write_text(
        line + "\n" + json.dumps(GOOD_EPISODE) + "\n", encoding="utf-8")
    assert [e.task for e in episodes.all()] == ["t"]


def test_episode_vector_sidecar_that_is_a_list_is_recomputed(episodes):
    sidecar = episodes.path.with_suffix(".vec.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("[1]", encoding="utf-8")
    assert episodes._vectors(["a"], lambda texts: [[2.0]]) == [[2.0]]
