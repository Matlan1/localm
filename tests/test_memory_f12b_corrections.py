# SPDX-License-Identifier: AGPL-3.0-or-later
"""A stale user fact must be CORRECTABLE by the system, not permanently
immortal.

Consolidation must not downgrade an UPDATE/DELETE against a source=user record
to a silent NO_OP: that drops the user's own later words ("actually I moved to
Munich") and leaves the earlier typed fact ("I live in Berlin") in place
forever. A high-confidence contradiction is recorded as a PENDING CORRECTION the
user accepts or rejects; nothing is silently lost and the trusted fact is never
auto-overwritten by a possibly-hallucinated synth candidate.
"""

from __future__ import annotations

import json

import pytest

from localm.memory import MemoryRecord, MemoryStore, PendingCorrection, run_consolidation


# Topic-clustered fake embedder: location words share an axis, food another, so
# paraphrased same-topic facts are cosine-near.
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


def _update_completer(conf=0.9):
    def complete(prompt: str) -> str:
        if "durable" in prompt:                        # extraction
            return json.dumps({"facts": [
                {"fact": "User moved to Munich", "confidence": 0.9}]})
        return json.dumps({"decision": "UPDATE", "confidence": conf})
    return complete


# --------------------------------------------------------------------------- #
#  consolidation: propose instead of silent NO_OP                              #
# --------------------------------------------------------------------------- #

def test_high_conf_contradiction_proposes_correction(tmp_path, allow_writes):
    s = MemoryStore("owner", "chat", root=tmp_path)
    berlin = s.add(MemoryRecord(text="User lives in Berlin", source="user",
                                importance=0.8), embed_fn=_fake_embed)

    res = run_consolidation(s, "User: I moved to Munich", _update_completer(0.9),
                            embed_fn=_fake_embed)

    # Surfaced, not silent: exactly one proposal, and the run reports it.
    assert res["proposed"] == 1, res
    # The trusted fact is NOT auto-overwritten and NOT dropped, and the synth
    # Munich claim is NOT added alongside ("keep both" is the wrong outcome too).
    texts = [r.text for r in s.all()]
    assert texts == ["User lives in Berlin"], texts
    # The proposal is pending and points at the Berlin record.
    corrs = s.corrections()
    assert len(corrs) == 1
    assert corrs[0].target_id == berlin.id
    assert corrs[0].action == "update"
    assert corrs[0].proposed_text == "User moved to Munich"
    assert corrs[0].target_text == "User lives in Berlin"


def test_low_conf_contradiction_makes_no_proposal(tmp_path, allow_writes):
    # A weak contradiction against a trusted fact is ignored (NO_OP), not surfaced.
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User lives in Berlin", source="user"),
          embed_fn=_fake_embed)

    res = run_consolidation(s, "User: hmm maybe Munich", _update_completer(0.4),
                            embed_fn=_fake_embed)
    assert res["proposed"] == 0
    assert s.corrections() == []
    assert [r.text for r in s.all()] == ["User lives in Berlin"]


def test_delete_contradiction_proposes_a_delete(tmp_path, allow_writes):
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User lives in Berlin", source="user"),
          embed_fn=_fake_embed)

    def complete(prompt: str) -> str:
        if "durable" in prompt:
            return json.dumps({"facts": [
                {"fact": "User no longer lives in Berlin", "confidence": 0.9}]})
        return json.dumps({"decision": "DELETE", "confidence": 0.9})

    res = run_consolidation(s, "User: I don't live in Berlin anymore", complete,
                            embed_fn=_fake_embed)
    assert res["proposed"] == 1
    corrs = s.corrections()
    assert corrs[0].action == "delete"
    assert corrs[0].proposed_text == ""            # a delete carries no new text
    assert [r.text for r in s.all()] == ["User lives in Berlin"]


def test_import_source_is_protected_like_user(tmp_path, allow_writes):
    # An imported (migrated) fact is trusted too: a synth contradiction is proposed,
    # never a silent overwrite.
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User lives in Berlin", source="import"),
          embed_fn=_fake_embed)
    res = run_consolidation(s, "User: I moved to Munich", _update_completer(0.9),
                            embed_fn=_fake_embed)
    assert res["proposed"] == 1
    assert [r.text for r in s.all()] == ["User lives in Berlin"]


def test_synth_vs_synth_still_updates_no_proposal(tmp_path, allow_writes):
    # A synth target is superseded directly (no review); only trusted facts get
    # the propose-and-review treatment.
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User lives in Berlin", source="synth"),
          embed_fn=_fake_embed)
    res = run_consolidation(s, "User: I moved to Munich", _update_completer(0.9),
                            embed_fn=_fake_embed)
    assert res["proposed"] == 0
    assert res["updated"] == 1
    assert s.corrections() == []
    assert [r.text for r in s.all()] == ["User moved to Munich"]


def test_proposal_dedups_across_runs(tmp_path, allow_writes):
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User lives in Berlin", source="user"),
          embed_fn=_fake_embed)
    run_consolidation(s, "User: I moved to Munich", _update_completer(0.9),
                      embed_fn=_fake_embed)
    run_consolidation(s, "User: I moved to Munich", _update_completer(0.9),
                      embed_fn=_fake_embed)
    # The same contradiction distilled twice is one pending suggestion, not two.
    assert len(s.corrections()) == 1


# --------------------------------------------------------------------------- #
#  resolving a correction                                                      #
# --------------------------------------------------------------------------- #

def _seed_pending(tmp_path, action="update", proposed="User moved to Munich"):
    s = MemoryStore("owner", "chat", root=tmp_path)
    target = s.add(MemoryRecord(text="User lives in Berlin", source="user"),
                   embed_fn=_fake_embed)
    s.propose_corrections([PendingCorrection(
        target_id=target.id, action=action, proposed_text=proposed,
        target_text=target.text, confidence=0.9)])
    return s, target


def _forgotten_texts(store):
    ff = store.path.with_suffix(".forgotten.jsonl")
    if not ff.is_file():
        return []
    return [json.loads(ln)["text"]
            for ln in ff.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_accept_update_applies_and_archives_old(tmp_path, allow_writes):
    s, target = _seed_pending(tmp_path)
    cid = s.corrections()[0].id
    out = s.resolve_correction(cid, accept=True, embed_fn=_fake_embed)
    assert out["status"] == "updated"

    fresh = MemoryStore("owner", "chat", root=tmp_path)      # reload from disk
    assert [r.text for r in fresh.all()] == ["User moved to Munich"]
    assert fresh.corrections() == []                          # pending cleared
    assert "User lives in Berlin" in _forgotten_texts(fresh)  # recoverable
    # The record kept its id and user provenance, only its text changed.
    assert fresh.all()[0].id == target.id
    assert fresh.all()[0].source == "user"
    # The vector tracks the NEW text (cosine to a Munich query is high).
    idx, score = fresh.semantic_nearest("User moved to Munich", fresh.all(), _fake_embed)
    assert idx == 0 and score > 0.9


def test_accept_delete_removes_and_archives(tmp_path, allow_writes):
    s, target = _seed_pending(tmp_path, action="delete", proposed="")
    cid = s.corrections()[0].id
    out = s.resolve_correction(cid, accept=True)
    assert out["status"] == "deleted"

    fresh = MemoryStore("owner", "chat", root=tmp_path)
    assert fresh.all() == []
    assert fresh.corrections() == []
    assert "User lives in Berlin" in _forgotten_texts(fresh)


def test_reject_keeps_record_and_clears_pending(tmp_path, allow_writes):
    s, target = _seed_pending(tmp_path)
    before_updated = s.all()[0].updated
    cid = s.corrections()[0].id
    out = s.resolve_correction(cid, accept=False, now=before_updated + 100.0)
    assert out["status"] == "rejected"

    fresh = MemoryStore("owner", "chat", root=tmp_path)
    assert [r.text for r in fresh.all()] == ["User lives in Berlin"]   # kept
    assert fresh.corrections() == []                                   # cleared
    assert fresh.all()[0].updated > before_updated                     # confirmed
    assert _forgotten_texts(fresh) == []                               # nothing archived


def test_corrections_drops_stale_target(tmp_path, allow_writes):
    s, target = _seed_pending(tmp_path)
    assert len(s.corrections()) == 1
    s.delete(target.id)                                # target gone
    assert s.corrections() == []                       # stale proposal dropped
    # And the sidecar was cleaned so it is not shown next time either.
    fresh = MemoryStore("owner", "chat", root=tmp_path)
    assert fresh.corrections() == []


def test_resolve_unknown_id_returns_none(tmp_path, allow_writes):
    s, _target = _seed_pending(tmp_path)
    assert s.resolve_correction("deadbeefdeadbeef", accept=True) is None
    assert len(s.corrections()) == 1                   # untouched


def test_resolve_target_gone_clears_stale_entry(tmp_path, allow_writes):
    # Accepting a correction whose target was deleted/evicted after it was proposed
    # must not raise; it drops the stale pending entry and archives nothing.
    s, target = _seed_pending(tmp_path)
    cid = s.corrections()[0].id
    s.delete(target.id)                                # target vanishes
    out = s.resolve_correction(cid, accept=True)
    assert out["status"] == "target_gone"
    fresh = MemoryStore("owner", "chat", root=tmp_path)
    assert fresh.corrections() == []                   # stale entry cleared
    assert _forgotten_texts(fresh) == []               # nothing archived


def test_accept_aborts_and_reports_when_archive_fails(tmp_path, allow_writes,
                                                      monkeypatch):
    # If the recoverable-archive step fails, accept must NOT destroy the trusted
    # record and must NOT report success: the record and the pending correction
    # both stay intact.
    s, _target = _seed_pending(tmp_path)
    cid = s.corrections()[0].id
    monkeypatch.setattr(MemoryStore, "_archive_forgotten",
                        lambda self, recs: False)      # simulate an archive failure
    out = s.resolve_correction(cid, accept=True)
    assert out["status"] == "archive_failed"
    fresh = MemoryStore("owner", "chat", root=tmp_path)
    assert [r.text for r in fresh.all()] == ["User lives in Berlin"]   # intact
    assert len(fresh.corrections()) == 1                               # still pending


def test_rejected_correction_is_not_reproposed(tmp_path, allow_writes):
    # A dismissed suggestion must NOT come back every consolidation while the same
    # contradicting session is still in the recent window.
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User lives in Berlin", source="user"),
          embed_fn=_fake_embed)
    res1 = run_consolidation(s, "User: I moved to Munich", _update_completer(0.9),
                             embed_fn=_fake_embed)
    assert res1["proposed"] == 1
    cid = s.corrections()[0].id
    s.resolve_correction(cid, accept=False)            # user clicks "Keep as is"
    assert s.corrections() == []

    # Re-run consolidation over the identical content: it must NOT resurrect it.
    res2 = run_consolidation(s, "User: I moved to Munich", _update_completer(0.9),
                             embed_fn=_fake_embed)
    assert res2["proposed"] == 0, "a dismissed correction was re-proposed"
    assert s.corrections() == []
    assert [r.text for r in s.all()] == ["User lives in Berlin"]


def test_proposal_for_size_cap_evicted_target_is_not_falsely_counted(tmp_path,
                                                                     allow_writes):
    # prune's SIZE cap can evict an old low-value trusted record; a proposal
    # against a target the same run then evicts must not report a false proposed
    # count and then be dropped by corrections(). The eviction is surfaced
    # separately.
    from localm.memory.store import N_MAX

    s = MemoryStore("owner", "chat", root=tmp_path)
    # An OLD, weak trusted fact - the one the contradiction targets, and the one the
    # size cap will drop first (lowest decayed score).
    old = 1_000_000.0
    s.add(MemoryRecord(text="User lives in Berlin", source="user", importance=0.3,
                       created=old, last_used=old), embed_fn=_fake_embed)
    # Fill to the cap with fresh synth facts so the store is over-cap after this run
    # and Berlin is the weakest (evicted).
    now = 2_000_000.0
    for i in range(N_MAX):
        s.add(MemoryRecord(text=f"User fact number {i} about topic {i}",
                           source="synth", importance=0.9, created=now,
                           last_used=now), embed_fn=_fake_embed, save=False)
    s._save()

    def complete(prompt: str) -> str:
        if "durable" in prompt:
            return json.dumps({"facts": [
                {"fact": "User moved to Munich", "confidence": 0.9}]})
        return json.dumps({"decision": "UPDATE", "confidence": 0.9})

    res = run_consolidation(s, "User: I moved to Munich", complete,
                            embed_fn=_fake_embed, now=now)
    texts = [r.text for r in s.all()]
    # Berlin was evicted by the size cap (surfaced), so the proposal against it is
    # NOT persisted and NOT counted - no false success, no stale sidecar entry.
    assert "User lives in Berlin" not in texts
    assert res["proposed"] == 0, "proposal counted against an evicted target"
    assert s.corrections() == []
    assert res.get("evicted_user", 0) >= 1             # the eviction IS surfaced


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
