# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vectors must actually get written, not merely be writable.

`MemoryStore.backfill_vectors` is bounded (64/call) and had exactly ONE caller:
the consolidation pass. Its own docstring claimed "a regular background pass
calls this so coverage climbs" - there was no such pass. On an install where
auto-consolidation never ran, records written before an embedder existed kept no
vector forever, `_vector_status` stayed `low_coverage`, and recall fell back to
promoting profile facts by IMPORTANCE rather than relevance.

That is what produced the 2026-08-14 report: "Greet my friend Memo" answered by
recalling an unrelated person, with the log reading

    memory recall: injected 2 record(s), degrade=low_coverage

and `setup-embeddings` had told the user "Memory now retrieves semantically".
"""

from pathlib import Path

from localm.memory.backfill import _namespaces, backfill_all, vectorless_total
from localm.memory.record import MemoryRecord
from localm.memory.store import MemoryStore


def _fake_embed(texts):
    one = not isinstance(texts, list)
    out = [[0.1, 0.2, 0.3, 0.4] for _ in ([texts] if one else texts)]
    return out[0] if one else out


def _seed(root: Path, principal: str, n: int) -> None:
    st = MemoryStore(principal, "chat", "", root=root)
    for i in range(n):
        st.add(MemoryRecord(text=f"fact {i} for {principal}", source="user"),
               embed_fn=None)                      # no embedder: the reported state


def test_backfill_reaches_every_namespace_including_key_scoped(tmp_path):
    """A key-scoped namespace's principal is a bearer-key hash that cannot be
    reconstructed from disk, so a backfill that rebuilt paths from known
    principals would silently skip it. This walks the root instead."""
    root = tmp_path / "memory"
    _seed(root, "owner", 5)
    _seed(root, "abc123keyhash", 3)

    assert len(_namespaces(root)) == 2
    assert vectorless_total(root) == 8

    res = backfill_all(root, _fake_embed)

    assert res["namespaces"] == 2
    assert res["embedded"] == 8
    assert res["remaining"] == 0
    assert vectorless_total(root) == 0, "every record must end up with a vector"


def test_backfill_runs_to_completion_past_the_per_call_bound(tmp_path):
    """backfill_vectors caps at 64 per call on purpose. The point of this module
    is that a caller wanting COMPLETION gets it, rather than 64 and a promise."""
    root = tmp_path / "memory"
    _seed(root, "owner", 150)                       # > 2 bounded passes

    assert vectorless_total(root) == 150
    res = backfill_all(root, _fake_embed)
    assert res["embedded"] == 150
    assert vectorless_total(root) == 0


def test_no_embedder_is_a_no_op_not_a_claim(tmp_path):
    """With no embedder there is nothing to do, and it must not report having
    done it."""
    root = tmp_path / "memory"
    _seed(root, "owner", 4)
    res = backfill_all(root, None)
    assert res == {"namespaces": 0, "embedded": 0, "remaining": 0}
    assert vectorless_total(root) == 4


def test_vectors_clear_the_low_coverage_degrade(tmp_path):
    """The whole point: below VEC_COVERAGE the semantic gate is unusable and the
    importance-ordered fallback opens. After a backfill it must not be."""
    root = tmp_path / "memory"
    _seed(root, "owner", 10)
    st = MemoryStore.open_file(_namespaces(root)[0])
    usable_before, reason_before = st._vector_status(_fake_embed)
    assert usable_before is False and reason_before, (
        "precondition: a vectorless store must report a degrade")

    backfill_all(root, _fake_embed)

    st2 = MemoryStore.open_file(_namespaces(root)[0])
    usable_after, reason_after = st2._vector_status(_fake_embed)
    assert usable_after is True, (
        f"the semantic gate must be usable after a backfill, got {reason_after!r}")
