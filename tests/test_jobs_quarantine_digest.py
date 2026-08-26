# SPDX-License-Identifier: AGPL-3.0-or-later
"""A quarantined jobs.json copy must not keep the owner key digest.

When jobs.json is corrupt, JobStore copies it aside as
``jobs.json.corrupt-<ts>`` so the user's scheduled jobs are not silently lost.
A VERBATIM copy carries every job's ``owner`` field - the same sha256 the
keystore stores (``http_server.principal_id`` returns ``key_hash``). The copy is
owner-only on disk, but permission-restricted is not the same as digest-free,
and nothing prunes these, so one legacy bare-sha256 digest could outlive a KDF
migration indefinitely.

Two independent halves, both covered here:

* REDACTION fixes copies written from now on. It must redact to a NON-NULL
  sentinel: ``job_owner_ok`` (http_server.py) returns True when ``owner is
  None`` - "a job with NO recorded owner is unrestricted" - so dropping the
  field or nulling it would make every job in a restored copy reachable by any
  caller.
* A RETENTION CAP bounds the copies already on disk, which redaction can never
  reach, and stays correct for any future credential-shaped field.

The copy must stay usable for recovery either way: schedule, prompt, model and
name are what it exists to preserve, and only the access-control field goes.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest

# The digest shape principal_id() produces: 64 lowercase hex chars.
DIGEST_RE = re.compile(r"\b[0-9a-f]{64}\b")
OWNER_A = "a1" * 32          # 64 hex chars, valid digest shape
OWNER_B = "b2" * 32


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "h"))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "h")
    from localm.plugins.builtin.jobs.store import Job, JobStore
    s = JobStore(root=tmp_path / "h" / "jobs")
    s.add(Job(name="nightly", task_kind="chat", prompt="summarise the inbox",
              schedule=3600, owner=OWNER_A))
    s.add(Job(name="weekly", task_kind="chat", prompt="weekly digest",
              schedule=86400, owner=OWNER_B))
    return s


def _corrupt_keeping_content(store) -> None:
    """Corrupt the defs file WITHOUT discarding it.

    Overwriting it with garbage would leave the quarantine copy holding only
    garbage, so the assertions below would pass while proving nothing about
    digests. Appending keeps every digest in the file the copy is made from.
    """
    f = store._defs_file
    f.write_text(f.read_text(encoding="utf-8") + "\n{trailing garbage",
                 encoding="utf-8")


def _backups(store) -> list:
    return sorted(store._defs_file.parent.glob("jobs.json.corrupt-*"))


class TestQuarantineRedactsTheDigest:
    def test_premise_the_live_file_really_holds_the_digests(self, store):
        """Fires-control. If the source file never contained a digest, every
        'no digest in the copy' assertion below is vacuously true."""
        raw = store._defs_file.read_text(encoding="utf-8")
        assert OWNER_A in raw and OWNER_B in raw
        assert DIGEST_RE.search(raw)

    def test_copy_carries_no_owner_digest(self, store):
        # NEGATIVE: neither owner digest survives into the quarantine copy.
        _corrupt_keeping_content(store)
        store._read_all()
        backups = _backups(store)
        assert backups, "no quarantine copy was written"
        text = backups[0].read_text(encoding="utf-8")
        assert OWNER_A not in text, "owner digest survived into the copy"
        assert OWNER_B not in text
        assert not DIGEST_RE.search(text), (
            f"a 64-hex digest remains: {DIGEST_RE.search(text).group(0)[:16]}...")

    def test_owner_is_redacted_to_a_NON_NULL_sentinel(self, store):
        """The whole point. `job_owner_ok` treats owner=None as UNRESTRICTED,
        so a null/absent owner would make a restored job reachable by any
        caller - a privilege escalation dressed as a scrub."""
        _corrupt_keeping_content(store)
        store._read_all()
        text = _backups(store)[0].read_text(encoding="utf-8")
        data = json.loads(text.split("\n{trailing garbage")[0])
        owners = [j.get("owner") for j in data["jobs"]]
        assert owners, "no jobs in the copy"
        for o in owners:
            assert o is not None, "owner=None makes a restored job UNRESTRICTED"
            assert isinstance(o, str) and o != ""
            # Must be un-equalable by principal_id(), which returns 64-hex.
            assert not DIGEST_RE.fullmatch(o), f"sentinel {o!r} has digest shape"

    def test_recovery_data_survives(self, store):
        """It exists so a corrupt store does not lose the user's scheduled
        jobs. Only the access-control field goes."""
        _corrupt_keeping_content(store)
        store._read_all()
        text = _backups(store)[0].read_text(encoding="utf-8")
        for expected in ("nightly", "weekly", "summarise the inbox",
                         "weekly digest", "3600", "86400"):
            assert expected in text, f"recovery data lost: {expected!r}"

    def test_still_owner_only_on_disk(self, store):
        """The quarantine copy's owner-only ACL is not regressed by the redaction change."""
        import os
        import subprocess
        _corrupt_keeping_content(store)
        store._read_all()
        b = _backups(store)[0]
        if os.name == "nt":
            out = subprocess.run(["icacls", str(b)], capture_output=True,
                                 text=True, check=False).stdout
            user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
            assert f"{user}:(F)" in out, out
            assert "BUILTIN\\Users" not in out, out
        else:
            assert (b.stat().st_mode & 0o077) == 0


class TestQuarantineRetentionCap:
    def test_old_copies_are_pruned(self, store, monkeypatch):
        """Redaction cannot reach copies already on disk; the cap bounds them."""
        from localm.plugins.builtin.jobs import store as store_mod
        assert store_mod._QUARANTINE_KEEP >= 1
        keep = store_mod._QUARANTINE_KEEP
        # Distinct stamps: the filename is int(time.time()), so several
        # quarantines in one second would otherwise collide on one name.
        t = [1_700_000_000]
        monkeypatch.setattr(store_mod.time, "time", lambda: t[0])
        for _ in range(keep + 3):
            t[0] += 1
            store._defs_file.write_text('{"version": 1, "jobs": [{"owner": "'
                                        + OWNER_A + '"}]} trailing',
                                        encoding="utf-8")
            store._read_all()
        backups = _backups(store)
        assert len(backups) == keep, (
            f"expected {keep} copies retained, found {len(backups)}")

    def test_the_newest_copies_are_the_ones_kept(self, store, monkeypatch):
        """Pruning the wrong end would discard the copy most likely to matter."""
        from localm.plugins.builtin.jobs import store as store_mod
        keep = store_mod._QUARANTINE_KEEP
        t = [1_700_000_000]
        monkeypatch.setattr(store_mod.time, "time", lambda: t[0])
        stamps = []
        for _ in range(keep + 2):
            t[0] += 1
            stamps.append(t[0])
            store._defs_file.write_text('{"version": 1, "jobs": []} trailing',
                                        encoding="utf-8")
            store._read_all()
        names = {b.name for b in _backups(store)}
        for s in stamps[-keep:]:
            assert f"jobs.json.corrupt-{s}" in names, f"newest {s} was pruned"
        for s in stamps[:-keep]:
            assert f"jobs.json.corrupt-{s}" not in names, f"stale {s} kept"


class TestQuarantineStillDoesItsJob:
    def test_a_corrupt_store_still_warns_and_does_not_discard(self, store, caplog):
        """The data-loss risk must stay visible, and the copy must still be
        written. Redaction must not turn this into a silent path."""
        _corrupt_keeping_content(store)
        with caplog.at_level("WARNING"):
            jobs = store._read_all()
        assert jobs == {}, "a corrupt store must not resolve to phantom jobs"
        assert _backups(store), "the copy must still be written"
        assert any("corrupt" in r.message.lower() for r in caplog.records), \
            "the corrupt-store warning was lost"


class TestResidualDigestIsWarningVisible:
    """`_redact_owner_digests` reports a surviving digest-shaped value at DEBUG,
    where the operator never sees it, and the outer "backed up to ... your
    scheduled jobs are preserved" WARNING says nothing about a residual
    credential. A digest that dodges the regex (odd key casing or whitespace in
    an already-malformed file) must surface at WARNING, the level an operator is
    actually watching."""

    def test_case_mismatched_owner_key_survives_and_warns(self, caplog):
        from localm.plugins.builtin.jobs.store import _redact_owner_digests

        digest = "c3" * 32  # 64 hex chars: valid digest shape
        # "Owner" (capital O) does not match the regex's literal "owner", so
        # this digest is not touched by the substitution.
        raw = '{"version": 1, "jobs": [{"Owner": "%s", "name": "n"}]}' % digest
        with caplog.at_level(logging.DEBUG, logger="localm"):
            out = _redact_owner_digests(raw)

        assert digest in out, (
            "premise: the case-mismatched field must survive redaction "
            "untouched, or this test proves nothing about the leftover check")
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, (
            "a digest-shaped value survived redaction but nothing was logged "
            "at WARNING - it is only visible at DEBUG, which an operator "
            "reading the outer 'backed up to ...' WARNING will never see")


class TestQuarantineBackupPermsRetryOnFailure:
    """`restrict_file_perms(backup)` must check its return value and retry when
    the first attempt fails, like `_write_all` does. The quarantine copy holds
    the same owner digests as the live file and deserves the same robustness."""

    def test_first_failure_is_retried(self, store, monkeypatch):
        import localm.config as cfg

        calls = []

        def flaky_restrict(path):
            calls.append(Path(path))
            return len(calls) > 1  # fail the first call, succeed on retry

        monkeypatch.setattr(cfg, "restrict_file_perms", flaky_restrict)
        _corrupt_keeping_content(store)
        store._read_all()

        backups = _backups(store)
        assert backups, "no quarantine copy was written"
        assert len(calls) >= 2, (
            "restrict_file_perms was not retried after the first call "
            "reported failure - the backup got only one best-effort attempt")
        assert all(c == backups[0] for c in calls), (
            "the retry targeted a different path than the original attempt")


class TestQuarantineBackupFailureMessageStatesReality:
    """The "refusing to silently discard it" wording would claim a refusal that
    never happens - the code logs a WARNING and returns normally, and the caller
    proceeds exactly as on success. The message must describe what actually
    occurs."""

    def test_message_does_not_claim_a_refusal_that_never_happens(
            self, store, monkeypatch, caplog):
        real_write_text = Path.write_text

        def flaky_write_text(self, *a, **kw):
            if self.name.startswith("jobs.json.corrupt-"):
                raise OSError("disk full")
            return real_write_text(self, *a, **kw)

        monkeypatch.setattr(Path, "write_text", flaky_write_text)
        _corrupt_keeping_content(store)
        with caplog.at_level("WARNING"):
            jobs = store._read_all()

        assert jobs == {}, "still starts empty on unbacked-up corruption"
        messages = [r.message for r in caplog.records]
        assert any("could not be backed" in m for m in messages), (
            "the backup-failure warning was not emitted")
        assert not any("refusing to silently discard" in m for m in messages), (
            "the message still claims a refusal that never actually happens "
            "(the caller proceeds with an empty job list exactly as on success)")
