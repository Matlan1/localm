# SPDX-License-Identifier: AGPL-3.0-or-later
"""A consolidation UPDATE must re-embed the record's NEW text. A run_consolidation
that mutates the text in place while store.replace re-embeds only vectorless
ids leaves the updated record on the OLD text's vector, so semantic recall
points at the contradicted content."""

from __future__ import annotations

import json

import pytest

from localm.memory import MemoryRecord, MemoryStore, run_consolidation


def _fake_embed(texts):
    """Deterministic, text-dependent 4-dim unit-ish vectors."""
    out = []
    for t in texts:
        h = abs(hash(t.lower()))
        v = [((h >> s) % 97) / 97.0 + 0.01 for s in (0, 7, 14, 21)]
        out.append(v)
    return out


@pytest.fixture
def allow_writes(monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "log")


def _vec_sidecar(store):
    return json.loads(store.path.with_suffix(".vec.json")
                      .read_text(encoding="utf-8"))["vectors"]


def test_update_reembeds_new_text(tmp_path, allow_writes):
    store = MemoryStore("owner", "chat", root=tmp_path)
    rec = store.add(MemoryRecord(text="User prefers Vim for editing code",
                                 source="synth"),
                    embed_fn=_fake_embed)
    old_vec = _vec_sidecar(store)[rec.id]
    assert old_vec == _fake_embed([rec.text])[0]

    new_text = "User now prefers VS Code for editing code"

    def complete(prompt: str) -> str:
        if "durable" in prompt:                        # extract
            return json.dumps({"facts": [
                {"fact": new_text, "confidence": 0.9}]})
        return json.dumps({"decision": "UPDATE", "confidence": 0.9})

    res = run_consolidation(store, "User: I switched to VS Code",
                            complete, embed_fn=_fake_embed)
    assert res["updated"] == 1

    updated = store.get(rec.id)
    assert updated is not None and updated.text == new_text
    fresh_vec = _vec_sidecar(store)[rec.id]
    assert fresh_vec == _fake_embed([new_text])[0], \
        "UPDATEd record still carries the old text's vector (audit F3 regression)"
    assert fresh_vec != old_vec


def test_invalidate_vectors_drops_only_given_ids(tmp_path):
    store = MemoryStore("owner", "chat", root=tmp_path)
    a = store.add(MemoryRecord(text="alpha"), embed_fn=_fake_embed)
    b = store.add(MemoryRecord(text="beta"), embed_fn=_fake_embed)
    store.invalidate_vectors({a.id})
    # replace() persists and re-embeds only the invalidated id
    store.replace(store.all(), embed_fn=_fake_embed)
    vecs = _vec_sidecar(store)
    assert a.id in vecs and b.id in vecs
    assert vecs[b.id] == _fake_embed(["beta"])[0]


def test_update_without_embedder_keeps_store_consistent(tmp_path, allow_writes):
    """No embedder configured: the UPDATE applies and the stale vector is gone
    (no vector at all beats a wrong one; recall then scores it lexically)."""
    store = MemoryStore("owner", "chat", root=tmp_path)
    rec = store.add(MemoryRecord(text="User prefers Vim for editing code",
                                 source="synth"),
                    embed_fn=_fake_embed)

    def complete(prompt: str) -> str:
        if "durable" in prompt:
            return json.dumps({"facts": [
                {"fact": "User now prefers VS Code for editing code",
                 "confidence": 0.9}]})
        return json.dumps({"decision": "UPDATE", "confidence": 0.9})

    res = run_consolidation(store, "User: I switched", complete, embed_fn=None)
    assert res["updated"] == 1
    vf = store.path.with_suffix(".vec.json")
    if vf.is_file():
        assert rec.id not in json.loads(
            vf.read_text(encoding="utf-8"))["vectors"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
