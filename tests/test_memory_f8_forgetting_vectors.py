# SPDX-License-Identifier: AGPL-3.0-or-later
"""Forgetting and semantic recall in real operation.

- prune (decay + size cap) must run even when a consolidation extracts zero
  facts, not only through a fact-producing run;
- vectors must backfill so semantic recall turns on retroactively after an
  embedding model is installed later;
- the vector path must be used on a small store (< TINY_CORPUS), where the
  lexical signal is skipped but cosine is not noisy.
"""

from __future__ import annotations

import pytest

from localm.memory import MemoryRecord, MemoryStore, run_consolidation
from localm.memory.store import PRUNE_FLOOR, TINY_CORPUS

NOW = 1_700_000_000.0
DAY = 86400.0


_MEASURE_TERMS = ("metric", "unit", "measure", "measurement", "system")
_EDITOR_TERMS = ("vim", "editor", "edit")


def _fake_embed(texts):
    """Deterministic topic-clustered vectors: measurement-related words (metric,
    unit, measure, measurement, system) load axis 0, editor words load axis 1.
    A paraphrased query about measurements therefore cosine-matches the 'metric'
    record even with no shared exact tokens (stand-in for real semantics)."""
    out = []
    for t in texts:
        lo = t.lower()
        v = [1.0 if any(w in lo for w in _MEASURE_TERMS) else 0.05,
             1.0 if any(w in lo for w in _EDITOR_TERMS) else 0.05,
             0.3]
        out.append(v)
    return out


@pytest.fixture
def allow_writes(monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "log")


# ------------------------------------------------- prune decoupled from facts #

def test_prune_runs_when_consolidation_extracts_nothing(tmp_path, allow_writes):
    s = MemoryStore("owner", "chat", root=tmp_path)
    # A stale, low-importance synth record that prune should forget.
    stale = s.add(MemoryRecord(text="a trivial passing thought", source="synth",
                               importance=0.05, last_used=NOW - 400 * DAY))
    assert s._decayed(stale, NOW) < PRUNE_FLOOR

    # Consolidation that extracts NO facts (model returns no facts).
    res = run_consolidation(s, "User: hello there", lambda p: '{"facts": []}',
                            now=NOW)
    assert res["added"] == 0
    # The stale record is forgotten even on a zero-fact consolidation.
    assert s.get(stale.id) is None, "prune did not run on a zero-fact consolidation"


def test_zero_fact_prune_surfaces_user_eviction(tmp_path, allow_writes):
    from localm.memory.store import N_MAX
    s = MemoryStore("owner", "chat", root=tmp_path)
    for i in range(N_MAX + 3):
        s.add(MemoryRecord(text=f"user fact {i}", source="user",
                           importance=0.8, last_used=NOW - i * DAY), save=False)
    s._save()
    res = run_consolidation(s, "User: hi", lambda p: '{"facts": []}', now=NOW)
    assert res.get("evicted_user", 0) >= 1
    assert "warning" in res


# --------------------------------------------------------- vector backfill -- #

def test_backfill_embeds_records_without_vectors(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    # Stored WITHOUT an embedder (as if before setup-embeddings).
    for t in ("User prefers metric units", "User uses Vim", "User lives in Ghent"):
        s.add(MemoryRecord(text=t, source="user"), embed_fn=None)
    assert s._vectors == {}

    n = s.backfill_vectors(_fake_embed)
    assert n == 3
    # Persisted and reloadable.
    s2 = MemoryStore("owner", "chat", root=tmp_path)
    assert len(s2._vectors) == 3


def test_backfill_is_bounded_and_resumable(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    for i in range(10):
        s.add(MemoryRecord(text=f"fact number {i} about units", source="user"),
              embed_fn=None)
    first = s.backfill_vectors(_fake_embed, limit=4)
    assert first == 4
    second = s.backfill_vectors(_fake_embed, limit=4)
    assert second == 4
    third = s.backfill_vectors(_fake_embed, limit=4)
    assert third == 2                        # only the remaining 2
    assert s.backfill_vectors(_fake_embed) == 0   # nothing left


def test_backfill_turns_on_semantic_recall_retroactively(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    # 8+ records so the store is not "tiny" and BM25 runs; a paraphrased query
    # shares no tokens, so only semantic recall can surface the right fact.
    s.add(MemoryRecord(text="User prefers metric units for measurements",
                       source="user"), embed_fn=None)
    for i in range(8):
        s.add(MemoryRecord(text=f"unrelated note about topic {i}", source="user"),
              embed_fn=None)
    # No vectors yet -> lexical only; the paraphrase misses.
    s.backfill_vectors(_fake_embed)
    hits = s.recall("what measurement system do I like", k=1, embed_fn=_fake_embed)
    assert hits and "metric" in hits[0].text


# ------------------------------------------------- tiny-corpus vector path -- #

def test_small_store_uses_vectors_when_present(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    assert 3 < TINY_CORPUS
    s.add(MemoryRecord(text="User prefers metric units", source="synth",
                       importance=0.5), embed_fn=_fake_embed)
    s.add(MemoryRecord(text="User uses Vim as an editor", source="synth",
                       importance=0.5), embed_fn=_fake_embed)
    s.add(MemoryRecord(text="User lives somewhere", source="synth",
                       importance=0.5), embed_fn=_fake_embed)
    # Paraphrased query about units: on a tiny store the vector path still ranks
    # the metric record first.
    hits = s.recall("which unit system", k=1, embed_fn=_fake_embed)
    assert hits and "metric" in hits[0].text


def test_small_store_without_embedder_still_works(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User prefers metric units", source="user",
                       importance=0.9))
    # No embedder: a relevant (content-word) query still recalls via lexical match,
    # ranked by recency+importance, and never crashes.
    hits = s.recall("what metric units", k=3, embed_fn=None)
    assert hits and hits[0].text == "User prefers metric units"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
