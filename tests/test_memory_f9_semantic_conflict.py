# SPDX-License-Identifier: AGPL-3.0-or-later
"""A PARAPHRASED contradiction must reach the ADD/UPDATE/DELETE decision instead
of blind-accumulating.

Matching candidates against existing records by difflib text ratio alone puts
'User lives in Berlin' vs 'User moved to Munich' at ratio ~0.45, which blind-ADDs
a second, conflicting record. With an embedder present, the semantic matcher
routes cosine-near candidates to the LLM decide step.
"""

from __future__ import annotations

import json

import pytest

from localm.memory import MemoryRecord, MemoryStore, run_consolidation
from localm.memory.consolidate import _nearest


# Topic-clustered fake embedder: location words share an axis, food another,
# so paraphrased same-topic facts are cosine-near despite few shared tokens.
_LOCATION = ("berlin", "munich", "ghent", "live", "lives", "moved", "city")
_FOOD = ("tea", "coffee", "drink", "vegetarian", "meat", "eat")


def _fake_embed(texts):
    out = []
    for t in texts:
        lo = t.lower()
        out.append([
            1.0 if any(w in lo for w in _LOCATION) else 0.05,
            1.0 if any(w in lo for w in _FOOD) else 0.05,
            0.2,
        ])
    return out


@pytest.fixture
def allow_writes(monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "log")


def test_paraphrased_contradiction_is_lexically_missed():
    # Establishes the premise: difflib alone does NOT catch this pair.
    from localm.memory.consolidate import MATCH_THRESHOLD
    recs = [MemoryRecord(text="User lives in Berlin")]
    _idx, ratio = _nearest("User moved to Munich", recs)
    assert ratio < MATCH_THRESHOLD, f"expected a lexical miss, got {ratio}"


def test_semantic_match_routes_contradiction_to_decide(tmp_path, allow_writes):
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User lives in Berlin", source="synth"),
          embed_fn=_fake_embed)

    decided = []

    def complete(prompt: str) -> str:
        if "durable" in prompt:                        # extraction
            return json.dumps({"facts": [
                {"fact": "User moved to Munich", "confidence": 0.9}]})
        # decide step: record that it was ASKED, and say UPDATE.
        decided.append(prompt)
        return json.dumps({"decision": "UPDATE", "confidence": 0.9})

    res = run_consolidation(s, "User: I moved to Munich", complete,
                            embed_fn=_fake_embed)
    # The decide step ran (semantic match), and the store did NOT keep both.
    assert decided, "paraphrased contradiction never reached the decide step"
    texts = [r.text for r in s.all()]
    assert "User moved to Munich" in texts
    assert "User lives in Berlin" not in texts, \
        f"both contradictory facts coexist: {texts}"
    assert res["updated"] == 1


def test_unrelated_fact_still_blind_adds(tmp_path, allow_writes):
    # A semantically UNRELATED new fact must NOT trigger a decide call (cost) and
    # must simply ADD alongside.
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User lives in Berlin", source="synth"),
          embed_fn=_fake_embed)
    decided = []

    def complete(prompt: str) -> str:
        if "durable" in prompt:
            return json.dumps({"facts": [
                {"fact": "User drinks tea not coffee", "confidence": 0.9}]})
        decided.append(prompt)
        return json.dumps({"decision": "NO_OP", "confidence": 0.9})

    res = run_consolidation(s, "User: I drink tea", complete, embed_fn=_fake_embed)
    assert decided == [], "an unrelated fact was needlessly sent to the LLM"
    assert res["added"] == 1
    assert len(s.all()) == 2


def test_no_embedder_keeps_lexical_only_behavior(tmp_path, allow_writes):
    # Without an embedder the paraphrase is not caught, and nothing crashes: it
    # blind-ADDs.
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User lives in Berlin", source="synth"))

    def complete(prompt: str) -> str:
        if "durable" in prompt:
            return json.dumps({"facts": [
                {"fact": "User moved to Munich", "confidence": 0.9}]})
        return json.dumps({"decision": "NO_OP", "confidence": 0.9})

    res = run_consolidation(s, "User: I moved", complete, embed_fn=None)
    assert res["added"] == 1                      # blind ADD, no decide


def test_synth_cannot_semantically_override_user_fact(tmp_path, allow_writes):
    # The user-fact guardrail holds through the semantic path: a synth candidate
    # that semantically matches a user fact cannot silently UPDATE or DELETE it.
    # A HIGH-confidence contradiction is surfaced as a PENDING CORRECTION and the
    # trusted record is left untouched.
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User lives in Berlin", source="user",
                       importance=0.8), embed_fn=_fake_embed)

    def complete(prompt: str) -> str:
        if "durable" in prompt:
            return json.dumps({"facts": [
                {"fact": "User moved to Munich", "confidence": 0.9}]})
        return json.dumps({"decision": "UPDATE", "confidence": 0.9})

    res = run_consolidation(s, "User: I moved to Munich", complete, embed_fn=_fake_embed)
    texts = [r.text for r in s.all()]
    # The user's Berlin fact is protected (unchanged, not overwritten, not deleted)
    # and the synth Munich claim is NOT blind-added alongside it.
    assert texts == ["User lives in Berlin"]
    # ...but the contradiction is surfaced for review, not swallowed.
    assert res["proposed"] == 1
    assert len(s.corrections()) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
