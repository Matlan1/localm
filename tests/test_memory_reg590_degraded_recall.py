# SPDX-License-Identifier: AGPL-3.0-or-later
"""REG-590 (regression audit 2026-07-14): recall falls silent on a paraphrase in the DEFAULT (no-embedder) config, so the user's own saved facts never reach the model."""

from __future__ import annotations

from localm.memory import MemoryRecord, MemoryStore


def _topic_embed(texts):
    """3-axis one-hot topic stub (name / drink / other) so distinct topics are ORTHOGONAL: a paraphrased name query cosine-matches the name fact, while an off-topic query lands on its own axis and matches nothing."""
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


# ---------------------------------------------- the regression itself (no embedder) #

def test_paraphrased_query_recalls_user_fact_without_embedder(tmp_path):
    """THE bug: no embedder + zero content-word overlap -> the fact must still surface."""
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User is called Sam", source="user"))
    out = s.recall("what is my name", embed_fn=None)
    assert [r.text for r in out] == ["User is called Sam"], (
        "a user-authored profile fact must survive a paraphrased query when the "
        "semantic signal is unavailable (REG-590)")


def test_other_reported_paraphrases_also_surface(tmp_path):
    """The audit and the peer session reproduced several; none share a content word."""
    from localm.memory.store import TRUST_FALLBACK_K
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User prefers tea over coffee", source="user"))
    s.add(MemoryRecord(text="User lives in Berlin", source="user"))
    out = [r.text for r in s.recall("what do i drink", embed_fn=None)]
    assert "User prefers tea over coffee" in out
    assert len(out) <= TRUST_FALLBACK_K


def test_offtopic_query_stays_silent_even_though_it_is_also_a_lexical_miss(tmp_path):
    """THE discriminator test, and the reason REG-590 was hard."""
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User was born in 1990", source="user", importance=0.9))
    s.add(MemoryRecord(text="User lives in Ghent", source="user", importance=0.9))
    assert s.recall("recommend a pasta recipe", embed_fn=None) == []
    assert s.recall("what is the capital of France", embed_fn=None) == []


def test_self_referential_variants_all_fire(tmp_path):
    """Every REG-590 repro (mine and the peer session's) carries a first-person pronoun; all are stopwords, hence invisible to the lexical gate."""
    from localm.memory.store import _is_self_referential
    for q in ("what is my name", "who am i", "what do i drink", "where do i live",
              "should I use miles or kilometers", "tell me about myself"):
        assert _is_self_referential(q), q
    for q in ("recommend a pasta recipe", "what is the capital of France",
              "how does a diesel engine work"):
        assert not _is_self_referential(q), q


def test_fallback_is_importance_ordered_not_relevance_ordered(tmp_path):
    """DOCUMENTS A KNOWN LIMITATION (raised in peer review), it does not fix it."""
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
    """A SELF-REFERENTIAL query with many user facts must promote at most TRUST_FALLBACK_K - NOT the ungated top-k that [10] removed (a naive revert would return all 5)."""
    # Imported inside the test on purpose: the constant does not exist pre-fix, and a
    # module-level import would make the whole file fail to COLLECT, hiding whether
    # the other tests fail for the real behavioural reason.
    from localm.memory.store import TRUST_FALLBACK_K
    s = MemoryStore("owner", "chat", root=tmp_path)
    for i in range(5):
        s.add(MemoryRecord(text=f"User profile detail number {i}", source="user"))
    # "know" is the only content word; shares nothing with the records -> lexical miss,
    # and "me" makes it self-referential so the fallback is reached.
    out = s.recall("what do you know about me", embed_fn=None, k=6)
    assert len(out) == TRUST_FALLBACK_K, (
        f"degraded recall must promote exactly {TRUST_FALLBACK_K} trusted facts, "
        f"not 0 (the REG-590 silence) and not all 5 (a top-k revert); got {len(out)}")


def test_synth_memory_is_still_gated_when_degraded(tmp_path):
    """The gate must still silence the numerous/noisy classes: only user/import are promoted."""
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="Assistant observed the sky was blue", source="synth"))
    assert s.recall("what is my name", embed_fn=None) == []


# --------------------------------------------- the healthy semantic path is intact - #

def test_strict_gate_holds_when_semantic_signal_is_usable(tmp_path):
    """With vectors usable, the fallback must NOT fire: an off-topic query still injects nothing."""
    s = MemoryStore("owner", "chat", root=tmp_path)
    for t in ("User is called Sam", "User prefers tea over coffee"):
        s.add(MemoryRecord(text=t, source="user"), embed_fn=_topic_embed)
    out = s.recall("completely unrelated aardvark topic", embed_fn=_topic_embed, k=6)
    assert out == [], ("with a usable semantic signal the strict gate must still "
                       "silence an off-topic turn (no trust fallback)")


def test_semantic_paraphrase_still_matches_when_usable(tmp_path):
    """Sanity: the semantic half genuinely handles the paraphrase, so the fallback is only ever needed on the degraded path."""
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User is called Sam", source="user"), embed_fn=_topic_embed)
    out = s.recall("what is my name", embed_fn=_topic_embed)
    assert [r.text for r in out] == ["User is called Sam"]


# ------------------------------------------------------- rule 5: surface, not hide - #

def test_degraded_fallback_is_reported_in_diagnostics(tmp_path):
    """The miss used to be SILENT (indistinguishable from 'you have no memories')."""
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
