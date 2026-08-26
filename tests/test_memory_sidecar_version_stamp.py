# SPDX-License-Identifier: AGPL-3.0-or-later
"""The correction sidecars (`.corrections.jsonl`, `.forgotten.jsonl`,
`.corrections-dismissed.json`) carry the same `"v": FORMAT_VERSION` stamp the
main record store does, and load tolerates its absence exactly like the main
store tolerates a pre-stamp file.
"""

from __future__ import annotations

import json

import pytest

from localm.memory import MemoryRecord, MemoryStore, PendingCorrection
from localm.memory.store import FORMAT_VERSION


@pytest.fixture(autouse=True)
def _allow_writes(monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "log")


# --------------------------------------------------------------------- #
#  new writes are stamped                                                #
# --------------------------------------------------------------------- #

def test_corrections_sidecar_is_stamped(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    target = s.add(MemoryRecord(text="User lives in Berlin", source="user"))
    s.propose_corrections([PendingCorrection(
        target_id=target.id, action="update", proposed_text="User moved to Munich",
        target_text=target.text, confidence=0.9)])
    lines = [json.loads(ln) for ln in
             s._corrections_file().read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    assert len(lines) == 1
    assert lines[0]["v"] == FORMAT_VERSION


def test_forgotten_sidecar_is_stamped(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    r = s.add(MemoryRecord(text="a fact", source="user"))
    assert s._archive_forgotten([r]) is True
    lines = [json.loads(ln) for ln in
             s._forgotten_file().read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    assert len(lines) == 1
    assert lines[0]["v"] == FORMAT_VERSION


def test_dismissed_sidecar_is_stamped(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    s._dismissed_file().parent.mkdir(parents=True, exist_ok=True)
    s._save_dismissed({("t1", "update", "x")})
    data = json.loads(s._dismissed_file().read_text(encoding="utf-8"))
    assert data["v"] == FORMAT_VERSION
    assert data["keys"] == [["t1", "update", "x"]]


# --------------------------------------------------------------------- #
#  old (unstamped) files still load                                      #
# --------------------------------------------------------------------- #

def test_unstamped_pending_correction_line_still_loads(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    target = s.add(MemoryRecord(text="User lives in Berlin", source="user"))
    legacy = {"id": "abcd1234abcd1234", "target_id": target.id, "action": "update",
              "proposed_text": "User moved to Munich", "target_text": target.text,
              "confidence": 0.9, "source": "synth", "created": 1_700_000_000.0}
    s._corrections_file().write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    corrs = s.corrections()
    assert len(corrs) == 1
    assert corrs[0].proposed_text == "User moved to Munich"


def test_unstamped_forgotten_line_still_loads(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    legacy = {"id": "abcd1234abcd1234", "kind": "semantic", "text": "an old fact",
              "importance": 0.5, "created": 1_700_000_000.0,
              "updated": 1_700_000_000.0, "last_used": 1_700_000_000.0, "uses": 0,
              "source": "user", "meta": {}, "forgotten_at": 1_700_000_100.0}
    s._forgotten_file().parent.mkdir(parents=True, exist_ok=True)
    s._forgotten_file().write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    items = s.forgotten()
    assert len(items) == 1
    assert items[0]["text"] == "an old fact"
    # And it is restorable despite having no "v" tag - from_dict filters unknown
    # keys, so a pre-stamp archive entry reconstructs a valid record.
    restored = s.restore_forgotten("abcd1234abcd1234")
    assert restored is not None and restored.text == "an old fact"


def test_unstamped_dismissed_bare_array_still_loads(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    # Legacy shape: a bare JSON array, not the {"v":.., "keys":[...]} wrapper.
    s._dismissed_file().parent.mkdir(parents=True, exist_ok=True)
    s._dismissed_file().write_text(json.dumps([["t1", "update", "x"]]),
                                   encoding="utf-8")
    assert s._load_dismissed() == {("t1", "update", "x")}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
