# SPDX-License-Identifier: AGPL-3.0-or-later
"""A present-but-UNREADABLE sidecar must NOT be collapsed into "absent" and then
rewritten to a lesser state.

``except OSError: return []`` treats a file that EXISTS but cannot be read (a
transient Windows AV/indexer lock - see storekit.py) exactly like a file that is
simply ABSENT. When the empty result then feeds a full-sidecar REWRITE, every
prior entry is silently wiped while the caller is told it succeeded.
sessions.py:_load re-raises on an unreadable store so lookup fails CLOSED.

Fault is injected only at the DISK BOUNDARY (a per-path ``Path.read_text`` that
raises, or an embedded-NUL config root that ``resolve()`` rejects) - the real
store/rag code under test runs unmocked.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from localm.memory import MemoryRecord, MemoryStore, PendingCorrection
from localm.rag.store import indexing_policy


@pytest.fixture
def allow_writes(monkeypatch):
    # Memory writes are gated on a non-privacy mode.
    monkeypatch.setenv("LOCALM_MODE", "log")


def _lock_reads_of(monkeypatch, target: Path):
    """Make BOTH ``Path.read_text`` and ``Path.read_bytes`` raise for *target* only
    (an existing file that is unreadable by any method), leaving every other path
    readable. Simulates a transient exclusive lock: a real lock fails every read, and
    the store reads different sidecars via different methods (corrections/forgotten via
    read_text, dismissed via read_bytes), so both must be faulted to be faithful."""
    real_text = Path.read_text
    real_bytes = Path.read_bytes

    def locked_text(self, *a, **k):
        if self == target:
            raise PermissionError(f"simulated transient lock on {self}")
        return real_text(self, *a, **k)

    def locked_bytes(self, *a, **k):
        if self == target:
            raise PermissionError(f"simulated transient lock on {self}")
        return real_bytes(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", locked_text)
    monkeypatch.setattr(Path, "read_bytes", locked_bytes)

    def restore():
        monkeypatch.setattr(Path, "read_text", real_text)
        monkeypatch.setattr(Path, "read_bytes", real_bytes)
    return restore


# --------------------------------------------------------------------------- #
#  corrections sidecar: unreadable must not wipe pending                       #
# --------------------------------------------------------------------------- #

def test_absent_corrections_file_is_empty_not_crash(tmp_path, allow_writes):
    s = MemoryStore("owner", "chat", root=tmp_path)
    assert not s._corrections_file().is_file()
    assert s._load_corrections() == []            # absent -> empty, no raise
    t = s.add(MemoryRecord(text="User lives in Berlin", source="user"))
    assert s.propose_corrections([PendingCorrection(
        target_id=t.id, action="update", proposed_text="User moved to Munich")]) == 1


def test_unreadable_corrections_file_does_not_wipe_pending(tmp_path, allow_writes,
                                                           monkeypatch):
    s = MemoryStore("owner", "chat", root=tmp_path)
    t = s.add(MemoryRecord(text="User lives in Berlin", source="user"))
    assert s.propose_corrections([
        PendingCorrection(target_id=t.id, action="update",
                          proposed_text="User moved to Munich"),
        PendingCorrection(target_id=t.id, action="delete", proposed_text=""),
    ]) == 2
    cf = s._corrections_file()
    assert cf.is_file() and len(s._load_corrections()) == 2

    # Sidecar becomes unreadable mid-life: a new propose pass must SKIP, not
    # rewrite the file down to only the new proposal.
    restore = _lock_reads_of(monkeypatch, cf)
    added = s.propose_corrections([PendingCorrection(
        target_id=t.id, action="update", proposed_text="User moved to Ghent")])
    assert added == 0                              # aborted, not wiped
    restore()

    survived = {c.dedup_key()[2] for c in s._load_corrections()}
    assert survived == {"user moved to munich", ""}   # both originals intact


def test_unreadable_corrections_file_warns(tmp_path, allow_writes, monkeypatch,
                                           caplog):
    s = MemoryStore("owner", "chat", root=tmp_path)
    t = s.add(MemoryRecord(text="User lives in Berlin", source="user"))
    s.propose_corrections([PendingCorrection(
        target_id=t.id, action="update", proposed_text="User moved to Munich")])
    _lock_reads_of(monkeypatch, s._corrections_file())
    with caplog.at_level(logging.WARNING, logger="localm"):
        assert s.propose_corrections([PendingCorrection(
            target_id=t.id, action="update", proposed_text="X")]) == 0
    assert any("correction" in r.getMessage().lower()
               and "unreadable" in r.getMessage().lower()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


# --------------------------------------------------------------------------- #
#  dismissed sidecar: unreadable must not wipe prior dismissals on reject      #
# --------------------------------------------------------------------------- #

def test_unreadable_dismissed_file_does_not_wipe_on_reject(tmp_path, allow_writes,
                                                           monkeypatch):
    s = MemoryStore("owner", "chat", root=tmp_path)
    t = s.add(MemoryRecord(text="User lives in Berlin", source="user"))
    # Seed one dismissed key by rejecting a first correction.
    s.propose_corrections([PendingCorrection(
        target_id=t.id, action="update", proposed_text="User moved to Munich")])
    s.resolve_correction(s.corrections()[0].id, accept=False)
    df = s._dismissed_file()
    assert df.is_file() and len(s._load_dismissed()) == 1
    munich_key = PendingCorrection(
        target_id=t.id, action="update",
        proposed_text="User moved to Munich").dedup_key()

    # Reject a SECOND correction while the dismissed file is unreadable: the
    # reject still succeeds, but the prior dismissal must NOT be wiped.
    s.propose_corrections([PendingCorrection(
        target_id=t.id, action="update", proposed_text="User moved to Ghent")])
    cid2 = s.corrections()[0].id
    restore = _lock_reads_of(monkeypatch, df)
    out = s.resolve_correction(cid2, accept=False)
    assert out["status"] == "rejected"             # record confirmed, pending cleared
    restore()

    assert munich_key in s._load_dismissed()       # prior dismissal survived
    assert s.corrections() == []                    # pending was still cleared


# --------------------------------------------------------------------------- #
#  the other guards: resolve/corrections unreadable, corrupt-content split     #
# --------------------------------------------------------------------------- #

def test_unreadable_corrections_file_resolve_returns_none_not_500(tmp_path, allow_writes,
                                                                 monkeypatch, caplog):
    # resolve_correction's OWN guard (distinct from the dismissed-file guard above):
    # an unreadable corrections sidecar must skip to a non-destructive None (a 404
    # upstream), never propagate an OSError out to plug.py (which does not catch it).
    s = MemoryStore("owner", "chat", root=tmp_path)
    t = s.add(MemoryRecord(text="User lives in Berlin", source="user"))
    s.propose_corrections([PendingCorrection(
        target_id=t.id, action="update", proposed_text="User moved to Munich")])
    cid = s.corrections()[0].id
    restore = _lock_reads_of(monkeypatch, s._corrections_file())
    with caplog.at_level(logging.WARNING, logger="localm"):
        out = s.resolve_correction(cid, accept=False)       # must NOT raise
    assert out is None
    assert any("unreadable" in r.getMessage().lower() for r in caplog.records)
    restore()
    # The pending entry survived the locked attempt; a retry after the lock clears works.
    assert len(s.corrections()) == 1
    assert s.resolve_correction(cid, accept=False)["status"] == "rejected"


def test_unreadable_corrections_file_corrections_lists_none_not_500(tmp_path, allow_writes,
                                                                   monkeypatch, caplog):
    # corrections() is the route-facing read (GET /api/memory): an unreadable sidecar
    # must return [] (warned), never propagate an OSError, and never run the stale-prune
    # save over a phantom-empty list (which would wipe the file).
    s = MemoryStore("owner", "chat", root=tmp_path)
    t = s.add(MemoryRecord(text="User lives in Berlin", source="user"))
    s.propose_corrections([PendingCorrection(
        target_id=t.id, action="update", proposed_text="User moved to Munich")])
    restore = _lock_reads_of(monkeypatch, s._corrections_file())
    with caplog.at_level(logging.WARNING, logger="localm"):
        got = s.corrections()                                # must NOT raise
    assert got == []
    assert any("unreadable" in r.getMessage().lower() for r in caplog.records)
    restore()
    assert len(s.corrections()) == 1                         # not wiped


@pytest.mark.parametrize("bad", [b"{ this is not valid json", b"\xff\xfe\x00bad-bytes"])
def test_corrupt_dismissed_content_is_empty_not_raise(tmp_path, allow_writes, caplog, bad):
    # CORRUPT CONTENT (bad JSON, or invalid UTF-8) is distinct from an UNREADABLE
    # file: it is unrecoverable, so it self-heals to an empty set with a warning
    # and must NOT raise. read_bytes()+decode keeps a UnicodeDecodeError (a
    # ValueError, not OSError) from escaping past the reject route's OSError guard.
    s = MemoryStore("owner", "chat", root=tmp_path)
    t = s.add(MemoryRecord(text="User lives in Berlin", source="user"))
    s.propose_corrections([PendingCorrection(
        target_id=t.id, action="update", proposed_text="User moved to Munich")])
    s.resolve_correction(s.corrections()[0].id, accept=False)   # seed a real dismissed key
    s._dismissed_file().write_bytes(bad)                        # corrupt the content
    with caplog.at_level(logging.WARNING, logger="localm"):
        got = s._load_dismissed()                               # must NOT raise
    assert got == set()
    assert any("dismissed" in r.getMessage().lower() and "unparseable" in r.getMessage().lower()
               for r in caplog.records), [r.getMessage() for r in caplog.records]
    # Self-heals: a subsequent reject writes a fresh valid file (no crash on the route).
    s.propose_corrections([PendingCorrection(
        target_id=t.id, action="delete", proposed_text="")])
    assert s.resolve_correction(s.corrections()[0].id, accept=False)["status"] == "rejected"
    assert len(s._load_dismissed()) >= 1                        # valid again


def test_corrupt_utf8_corrections_line_is_skipped_not_500(tmp_path, allow_writes, caplog):
    # A corrections sidecar is JSONL: an invalid-UTF-8 (torn multibyte) line must be
    # skipped as corrupt content and the VALID lines salvaged - never an uncaught
    # UnicodeDecodeError (a ValueError, not OSError) escaping past the callers' OSError
    # guard to 500 the memory routes. read_text() would have raised on the first bad
    # byte; read_bytes()+per-line-decode isolates the damage to the bad line.
    s = MemoryStore("owner", "chat", root=tmp_path)
    t = s.add(MemoryRecord(text="User lives in Berlin", source="user"))
    s.propose_corrections([
        PendingCorrection(target_id=t.id, action="update", proposed_text="User moved to Munich"),
        PendingCorrection(target_id=t.id, action="delete", proposed_text=""),
    ])
    with open(s._corrections_file(), "ab") as fh:               # append a bad-UTF-8 line
        fh.write(b"\xff\xfe\x00 not valid utf8 or json\n")
    with caplog.at_level(logging.WARNING, logger="localm"):
        got = s._load_corrections()                             # must NOT raise
    assert {c.dedup_key()[2] for c in got} == {"user moved to munich", ""}  # both salvaged
    assert any("correction" in r.getMessage().lower() and "skipped" in r.getMessage().lower()
               for r in caplog.records), [r.getMessage() for r in caplog.records]
    # Route-facing calls keep working (no 500), and the bad line does not wipe the good ones.
    assert len(s.corrections()) == 2
    assert s.resolve_correction(s.corrections()[0].id, accept=False)["status"] == "rejected"


# --------------------------------------------------------------------------- #
#  forgotten archive: unreadable must warn, not silently claim empty           #
# --------------------------------------------------------------------------- #

def test_unreadable_forgotten_archive_warns_and_returns_empty(tmp_path, allow_writes,
                                                              monkeypatch, caplog):
    s = MemoryStore("owner", "chat", root=tmp_path)
    t = s.add(MemoryRecord(text="User lives in Berlin", source="user"))
    # Accept a DELETE correction -> the record is archived to the forgotten sidecar.
    s.propose_corrections([PendingCorrection(
        target_id=t.id, action="delete", proposed_text="")])
    s.resolve_correction(s.corrections()[0].id, accept=True)
    ff = s._forgotten_file()
    assert ff.is_file() and len(s._load_forgotten()) == 1

    _lock_reads_of(monkeypatch, ff)
    with caplog.at_level(logging.WARNING, logger="localm"):
        got = s.forgotten()
    assert got == []                                # non-destructive empty
    assert any("forgotten" in r.getMessage().lower()
               and "unreadable" in r.getMessage().lower()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


def test_absent_forgotten_archive_is_empty(tmp_path, allow_writes):
    s = MemoryStore("owner", "chat", root=tmp_path)
    assert not s._forgotten_file().is_file()
    assert s.forgotten() == []                      # absent -> empty, no raise


def test_corrupt_utf8_forgotten_line_is_skipped_not_500(tmp_path, allow_writes, caplog):
    # The forgotten archive is JSONL too: an invalid-UTF-8 line must be skipped (the
    # valid records still list), never an uncaught UnicodeDecodeError that would 500
    # the forgotten-list / restore routes.
    s = MemoryStore("owner", "chat", root=tmp_path)
    t = s.add(MemoryRecord(text="User lives in Berlin", source="user"))
    s.propose_corrections([PendingCorrection(
        target_id=t.id, action="delete", proposed_text="")])
    s.resolve_correction(s.corrections()[0].id, accept=True)   # archive one real record
    with open(s._forgotten_file(), "ab") as fh:                # append a bad-UTF-8 line
        fh.write(b"\xff\xfe\x00 torn line\n")
    with caplog.at_level(logging.WARNING, logger="localm"):
        got = s.forgotten()                                    # must NOT raise
    assert [e["text"] for e in got] == ["User lives in Berlin"]   # valid record salvaged
    assert any("forgotten" in r.getMessage().lower() and "skipped" in r.getMessage().lower()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


# --------------------------------------------------------------------------- #
#  rag denied root that fails to resolve must be WARNED, not silently dropped  #
# --------------------------------------------------------------------------- #

def test_unresolvable_denied_root_is_warned(caplog):
    # An embedded NUL makes Path().expanduser().resolve() raise ValueError, so the
    # denied root cannot be resolved. It must be warned rather than silently
    # dropped, which would fail-OPEN the deny (confine_index_path could no longer
    # refuse a path inside it).
    cfg = {"rag_indexing_mode": "blacklist",
           "rag_denied_roots": ["bad\x00root"],
           "rag_allowed_roots": []}
    with caplog.at_level(logging.WARNING, logger="localm"):
        policy = indexing_policy(cfg)
    assert policy["denied"] == []                   # unresolvable -> not enforceable
    msgs = [r.getMessage().lower() for r in caplog.records]
    assert any("denied" in m and "not being enforced" in m for m in msgs), msgs


def test_resolvable_roots_still_resolve_without_warning(tmp_path, caplog):
    d = tmp_path / "denied_dir"
    d.mkdir()
    cfg = {"rag_indexing_mode": "blacklist",
           "rag_denied_roots": [str(d)], "rag_allowed_roots": []}
    with caplog.at_level(logging.WARNING, logger="localm"):
        policy = indexing_policy(cfg)
    assert policy["denied"] == [d.resolve()]        # good root resolves normally
    assert not any("not being enforced" in r.getMessage().lower()
                   for r in caplog.records)
