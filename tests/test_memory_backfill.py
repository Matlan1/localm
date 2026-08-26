# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vectors must actually get written, not merely be writable.

`MemoryStore.backfill_vectors` is bounded (64/call). With only the consolidation
pass calling it, an install where auto-consolidation never runs keeps records
written before an embedder existed without a vector forever: `_vector_status`
stays `low_coverage`, and recall falls back to promoting profile facts by
IMPORTANCE rather than relevance.
"""

from pathlib import Path

import pytest

from localm.memory.backfill import (
    _namespaces, backfill_all, vectorless_scan, vectorless_total,
)
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
    """backfill_vectors caps at 64 per call. The point of this module
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
    assert res == {"namespaces": 0, "embedded": 0, "remaining": 0, "unreadable": 0}
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


# --------------------------------------------------------------------------- #
#  An unreadable namespace is COUNTED, never absorbed as a clean pass: a       #
#  namespace MemoryStore.open_file cannot open contributes to its own counter, #
#  not zero to every counter like a fully embedded one.                        #
# --------------------------------------------------------------------------- #

def _write_unreadable_namespace(root: Path, agent: str = "chat") -> Path:
    """A namespace file whose bytes are not valid UTF-8, so MemoryStore._load's
    ``self._file.read_text(encoding="utf-8")`` raises UnicodeDecodeError before
    the per-line tolerant JSON parsing (which only guards json.loads) ever
    runs."""
    ns_dir = root / agent
    ns_dir.mkdir(parents=True, exist_ok=True)
    bad_path = ns_dir / ("0" * 16 + ".jsonl")
    bad_path.write_bytes(b'{"id":"a"}\n\xff\xfe not utf-8\n')
    return bad_path


def test_unreadable_namespace_fault_actually_fires(tmp_path):
    """Precondition for the tests below: prove the corrupt fixture really makes
    MemoryStore.open_file raise, rather than silently decoding. A fixture that
    happens to decode injects no fault at all, and every test below would pass
    with the bug fully in place."""
    root = tmp_path / "memory"
    bad_path = _write_unreadable_namespace(root)
    with pytest.raises(UnicodeDecodeError):
        MemoryStore.open_file(bad_path)


def test_backfill_all_counts_an_unreadable_namespace_not_silently(tmp_path):
    """A root whose only namespace cannot be opened must report that shortfall,
    not read as a clean pass with nothing left to do."""
    root = tmp_path / "memory"
    _write_unreadable_namespace(root)

    res = backfill_all(root, _fake_embed)

    assert res["unreadable"] == 1, (
        f"an unopenable namespace must be counted, not silently skipped: {res!r}")
    assert res["namespaces"] == 0
    assert res["embedded"] == 0
    assert res["remaining"] == 0


def test_backfill_all_embeds_the_good_namespace_despite_a_bad_sibling(tmp_path):
    """One namespace failing to open must not stop the rest of the root from
    being backfilled in full."""
    root = tmp_path / "memory"
    _seed(root, "owner", 5)
    _write_unreadable_namespace(root)

    res = backfill_all(root, _fake_embed)

    assert res["namespaces"] == 1
    assert res["embedded"] == 5
    assert res["remaining"] == 0
    assert res["unreadable"] == 1


def test_vectorless_scan_reports_unreadable_separately_from_total(tmp_path):
    """vectorless_scan is what lets a caller distinguish 'nothing left to
    embed' from 'could not even check' - vectorless_total alone cannot."""
    root = tmp_path / "memory"
    _seed(root, "owner", 3)
    _write_unreadable_namespace(root)

    total, unreadable = vectorless_scan(root)

    assert total == 3, "the good namespace's real vectorless count must still show"
    assert unreadable == 1
