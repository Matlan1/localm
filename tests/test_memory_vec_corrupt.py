# SPDX-License-Identifier: AGPL-3.0-or-later
"""A corrupt/unreadable vector sidecar (.vec.json) must degrade to lexical recall
WITHOUT hiding the fault. An absent sidecar is a normal cold start; a
present-but-unparseable one warns, mirroring the records-path warning in the
same loader. The stored records themselves are never lost; only the vector
cache is dropped, so recall falls back to BM25.
"""

from __future__ import annotations

import logging
from pathlib import Path

from localm.memory import MemoryRecord, MemoryStore


def _seed(root: Path) -> MemoryStore:
    s = MemoryStore("owner", "chat", root=root)
    # No embed_fn -> no vectors stored -> no vec sidecar written yet.
    s.add(MemoryRecord(text="the quick brown fox jumps", source="user"))
    return s


def test_corrupt_vec_sidecar_warns_and_degrades(tmp_path, caplog):
    root = tmp_path / "mem"
    s = _seed(root)
    vf = s._vec_file()
    vf.write_text("not valid json {{{", encoding="utf-8")   # present but corrupt
    with caplog.at_level(logging.WARNING, logger="localm"):
        s2 = MemoryStore("owner", "chat", root=root)
    # Degraded, not crashed: records still load; vectors dropped -> lexical.
    assert len(s2) == 1
    assert s2._vectors == {}
    # SURFACED, not silent.
    assert any("vector sidecar" in r.message.lower() for r in caplog.records), \
        "corrupt vector sidecar was reset silently (no warning)"


def test_non_object_vec_sidecar_warns_not_crashes(tmp_path, caplog):
    # A valid-JSON-but-not-an-object sidecar (e.g. a bare list) degrades and
    # warns like any other corrupt sidecar.
    root = tmp_path / "mem"
    s = _seed(root)
    s._vec_file().write_text("[1, 2, 3]", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="localm"):
        s2 = MemoryStore("owner", "chat", root=root)   # must not raise
    assert len(s2) == 1 and s2._vectors == {}
    assert any("vector sidecar" in r.message.lower() for r in caplog.records)


def test_absent_vec_sidecar_is_silent(tmp_path, caplog):
    # NEGATIVE / distinctness: an absent sidecar (the normal cold-start case)
    # must NOT warn.
    root = tmp_path / "mem"
    _seed(root)
    assert not MemoryStore("owner", "chat", root=root)._vec_file().is_file()
    with caplog.at_level(logging.WARNING, logger="localm"):
        MemoryStore("owner", "chat", root=root)
    assert not any("vector sidecar" in r.message.lower() for r in caplog.records)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
