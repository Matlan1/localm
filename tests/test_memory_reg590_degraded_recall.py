# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recall must not fall silent on a paraphrase in the DEFAULT (no-embedder)
config, leaving the user's own saved facts out of the model's context.

The precision gate is lexical-OR-semantic. In a default install there is no
embedding model (it is opt-in via ``localm setup-embeddings``), so the semantic
branch is dead and the gate degrades to EXACT content-word intersection:

    'what is my name'  vs  'User is called Sam'  ->  {} -> ineligible -> recall() == []

The degraded path is not only "no embedder": ``_vector_status`` also reports
``no_vectors``, ``low_coverage`` (fewer than VEC_COVERAGE carry a vector, and
backfill is bounded at 64/pass, so a few-hundred-record store is degraded for
several passes even WITH an embedder installed) and ``dim_mismatch``.

The behaviour pinned here: when the semantic signal is degraded, recall promotes
at most TRUST_FALLBACK_K user-authored (TRUSTED_SOURCES) facts that failed the
lexical gate, in score order. user and import are modelled as stable profile
facts - recall exempts them from recency decay and prune exempts them from decay
eviction.

That is a BOUNDED TRUSTED-FACT FALLBACK, NOT paraphrase recall. With no semantic
signal there is no relevance signal, so the promotion is importance-ordered and
on a store with many user facts it can surface 2 unrelated ones and still miss
the fact the query was about (pinned by
test_fallback_is_importance_ordered_not_relevance_ordered).
"""

from __future__ import annotations

from localm.memory import MemoryRecord, MemoryStore


def _topic_embed(texts):
    """3-axis one-hot topic stub (name / drink / other) so distinct topics are
    ORTHOGONAL: a paraphrased name query cosine-matches the name fact, while an
    off-topic query lands on its own axis and matches nothing."""
    name = ("name", "called", "sam")
    drink = ("tea", "coffee", "drink")
    out = []
    for t in texts:
        lo = t.lower()
        if any(w in lo for w in name):
            out.append([1.0, 0.0, 0.0])
        elif any(w in lo for w in drink):
            out.append([0.0, 1.0, 0.0])
        else:
            out.append([0.0, 0.0, 1.0])
    return out


# ----------------------------------------------------------- with no embedder #

def test_paraphrased_query_recalls_user_fact_without_embedder(tmp_path):
    """THE bug: no embedder + zero content-word overlap -> the fact must still surface."""
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User is called Sam", source="user"))
    out = s.recall("what is my name", embed_fn=None)
    assert [r.text for r in out] == ["User is called Sam"], (
        "a user-authored profile fact must survive a paraphrased query when the "
        "semantic signal is unavailable (REG-590)")


def test_other_reported_paraphrases_also_surface(tmp_path):
    """Several paraphrases, none sharing a content word with the stored fact.
    What is asserted: the wanted fact is AMONG the promoted set, not that it is
    singled out - the fallback does not rank by relevance (see the limitation
    test below), so with 2 facts and K=2 both surface."""
    from localm.memory.store import TRUST_FALLBACK_K
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User prefers tea over coffee", source="user"))
    s.add(MemoryRecord(text="User lives in Berlin", source="user"))
    out = [r.text for r in s.recall("what do i drink", embed_fn=None)]
    assert "User prefers tea over coffee" in out
    assert len(out) <= TRUST_FALLBACK_K


def test_offtopic_query_stays_silent_even_though_it_is_also_a_lexical_miss(tmp_path):
    """THE discriminator test.

    'recommend a pasta recipe' vs 'User was born in 1990' is a
    zero-content-word-overlap miss against a trusted fact, identical in signal
    to 'what is my name' vs 'User is called Sam'. The precision gate requires
    the first to stay silent and the paraphrase case requires the second to
    recall; only the first-person reference separates them. A fallback that
    fires here puts an unrelated profile fact into every off-topic turn."""
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User was born in 1990", source="user", importance=0.9))
    s.add(MemoryRecord(text="User lives in Ghent", source="user", importance=0.9))
    assert s.recall("recommend a pasta recipe", embed_fn=None) == []
    assert s.recall("what is the capital of France", embed_fn=None) == []


def test_self_referential_variants_all_fire(tmp_path):
    """Every paraphrase repro carries a first-person pronoun; all are stopwords,
    hence invisible to the lexical gate."""
    from localm.memory.store import _is_self_referential
    for q in ("what is my name", "who am i", "what do i drink", "where do i live",
              "should I use miles or kilometers", "tell me about myself"):
        assert _is_self_referential(q), q
    for q in ("recommend a pasta recipe", "what is the capital of France",
              "how does a diesel engine work"):
        assert not _is_self_referential(q), q


def test_fallback_is_importance_ordered_not_relevance_ordered(tmp_path):
    """PINS A KNOWN LIMITATION; it does not fix it.

    With the semantic branch dead there is no relevance signal: ``rel`` is 0 for
    a zero-overlap record and ``rec`` is pinned to 1.0 for TRUSTED_SOURCES (the
    decay exemption). Every zero-overlap user fact therefore scores
    W_REC*1.0 + W_IMP*importance and they differ ONLY by importance, so the
    fallback promotes the most IMPORTANT trusted facts, which need not include
    the one the query was about.

    Making the degraded path relevance-aware fails this test."""
    from localm.memory.store import TRUST_FALLBACK_K
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User is called Sam", source="user", importance=0.5))
    for i in range(4):
        s.add(MemoryRecord(text=f"User unrelated preference {i}", source="user",
                           importance=0.9))
    out = [r.text for r in s.recall("what is my name", embed_fn=None)]
    assert len(out) == TRUST_FALLBACK_K
    assert "User is called Sam" not in out, (
        "if this now passes, the degraded path became relevance-aware - update this "
        "test and the docs deliberately rather than deleting the assertion")


# ------------------------------------------------------- the fix stays BOUNDED ---- #

def test_degraded_fallback_is_bounded_not_a_topk_revert(tmp_path):
    """A SELF-REFERENTIAL query with many user facts must promote at most
    TRUST_FALLBACK_K - NOT the ungated top-k that [10] removed (a naive revert would
    return all 5). The off-topic case now promotes ZERO and is asserted separately in
    test_offtopic_query_stays_silent_even_though_it_is_also_a_lexical_miss; this test
    guards the BOUND on the path where the fallback actually fires."""
    # Imported inside the test: a module-level import of a missing constant
    # would fail collection for the whole file.
    from localm.memory.store import TRUST_FALLBACK_K
    s = MemoryStore("owner", "chat", root=tmp_path)
    for i in range(5):
        s.add(MemoryRecord(text=f"User profile detail number {i}", source="user"))
    # know is the only content word and shares nothing with the records, so the
    # lexical path misses; me makes it self-referential, reaching the fallback.
    out = s.recall("what do you know about me", embed_fn=None, k=6)
    assert len(out) == TRUST_FALLBACK_K, (
        f"degraded recall must promote exactly {TRUST_FALLBACK_K} trusted facts, "
        f"not 0 (the REG-590 silence) and not all 5 (a top-k revert); got {len(out)}")


def test_synth_memory_is_still_gated_when_degraded(tmp_path):
    """The gate must still silence the numerous/noisy classes: only user/import are
    promoted. A synth memory failing the lexical gate stays out."""
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="Assistant observed the sky was blue", source="synth"))
    assert s.recall("what is my name", embed_fn=None) == []


# --------------------------------------------- the healthy semantic path is intact - #

def test_strict_gate_holds_when_semantic_signal_is_usable(tmp_path):
    """With vectors usable, the fallback must NOT fire: an off-topic query still
    injects nothing. Guards against the fix leaking into the good path."""
    s = MemoryStore("owner", "chat", root=tmp_path)
    for t in ("User is called Sam", "User prefers tea over coffee"):
        s.add(MemoryRecord(text=t, source="user"), embed_fn=_topic_embed)
    out = s.recall("completely unrelated aardvark topic", embed_fn=_topic_embed, k=6)
    assert out == [], ("with a usable semantic signal the strict gate must still "
                       "silence an off-topic turn (no trust fallback)")


def test_semantic_paraphrase_still_matches_when_usable(tmp_path):
    """Sanity: the semantic half genuinely handles the paraphrase, so the fallback is
    only ever needed on the degraded path."""
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User is called Sam", source="user"), embed_fn=_topic_embed)
    out = s.recall("what is my name", embed_fn=_topic_embed)
    assert [r.text for r in out] == ["User is called Sam"]


# ------------------------------------------- the degradation is surfaced ----- #

def test_degraded_fallback_is_reported_in_diagnostics(tmp_path):
    """The degrade reason and the fallback count must both be observable, so a
    degraded miss is distinguishable from "you have no memories"."""
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User is called Sam", source="user"))
    diag: dict = {}
    out = s.recall("what is my name", embed_fn=None, diagnostics=diag)
    assert len(out) == 1
    assert diag["degrade_reason"] == "no_embedder"
    assert diag["trust_fallback"] == 1, (
        "the trust fallback must be reported, not applied invisibly")


def test_no_fallback_reported_when_lexical_hits(tmp_path):
    """A normal lexical hit must not be counted as a fallback promotion."""
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User name is Sam", source="user"))
    diag: dict = {}
    out = s.recall("my name", embed_fn=None, diagnostics=diag)
    assert [r.text for r in out] == ["User name is Sam"]
    assert diag["trust_fallback"] == 0
