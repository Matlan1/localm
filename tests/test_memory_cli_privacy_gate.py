# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm memory` must refuse every durable write while privacy mode is active,
exactly like the routes already do, and must not spread that refusal to the
deletion verbs, which this change deliberately leaves untouched."""

from __future__ import annotations

import json as _json
from pathlib import Path

import pytest
from click.testing import CliRunner


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """A throwaway LOCALM_HOME in privacy mode (the default), which the CLI and
    the store both resolve to."""
    h = tmp_path / ".localm"
    h.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", h)
    monkeypatch.setattr(_cfg, "home_dir", lambda: h)
    return h


def _run(*args, expect_ok=True):
    from localm.plugins.builtin.memory.cli import main
    res = CliRunner().invoke(main, list(args), catch_exceptions=False)
    if expect_ok:
        assert res.exit_code == 0, f"{args} -> {res.exit_code}\n{res.output}"
    return res


def _cli_store():
    from localm.plugins.builtin.memory.cli import _store
    return _store()


def _safe_read_bytes(p):
    try:
        return p.read_bytes()
    except OSError:
        return b""


def _assert_privacy_mode():
    from localm.audit import SessionMode, effective_mode
    assert effective_mode("chat") is SessionMode.PRIVACY, (
        "precondition: this fixture must resolve privacy mode, or the test proves nothing")


# --------------------------------------------------------------------------- #
#  add                                                                         #
# --------------------------------------------------------------------------- #

def test_add_refuses_and_leaves_nothing_on_disk_in_privacy_mode(home):
    from localm.audit import SessionMode, effective_mode
    assert effective_mode("chat") is SessionMode.PRIVACY, (
        "precondition: this fixture must resolve privacy mode, or the test proves nothing")

    canary = "canary sentence 4QF7 that must never reach disk"
    res = _run("add", canary, expect_ok=False)

    # DATA FIRST. The property is that no durable trace exists, not that a message was printed.
    hits = [str(p) for p in home.rglob("*")
            if p.is_file() and canary.encode() in _safe_read_bytes(p)]
    assert hits == [], f"privacy mode wrote the fact to disk: {hits}"

    assert res.exit_code != 0
    assert "privacy mode" in res.output


def test_add_still_writes_when_the_mode_allows_it(home, monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "log")
    from localm.audit import SessionMode, effective_mode
    assert effective_mode("chat") is not SessionMode.PRIVACY
    _run("add", "the user drinks tea")
    assert [r.text for r in _cli_store().all()] == ["the user drinks tea"]


# --------------------------------------------------------------------------- #
#  restore                                                                     #
# --------------------------------------------------------------------------- #

def test_restore_refuses_and_leaves_the_record_archived(home):
    _assert_privacy_mode()
    from localm.memory.store import MemoryRecord
    store = _cli_store()
    for i in range(3):
        store.add(MemoryRecord(text=f"fact number {i}", kind="semantic",
                               source="user", importance=0.8))
    store.prune(n_max=1)
    archived = store.forgotten()
    assert archived, "precondition: something is archived"
    gone_id = archived[0]["id"]

    res = _run("restore", gone_id, expect_ok=False)

    fresh = _cli_store()
    assert gone_id not in [r.id for r in fresh.all()], (
        "privacy mode restored the record into the live store anyway")
    assert gone_id in [row["id"] for row in fresh.forgotten()], (
        "the record is no longer archived after a refused restore")
    assert res.exit_code != 0
    assert "privacy mode" in res.output


# --------------------------------------------------------------------------- #
#  accept / reject                                                             #
# --------------------------------------------------------------------------- #

def test_accept_refuses_and_leaves_the_target_text_unchanged(home):
    _assert_privacy_mode()
    from localm.memory.corrections import PendingCorrection
    from localm.memory.store import MemoryRecord
    store = _cli_store()
    target = store.add(MemoryRecord(text="the user drinks coffee", kind="semantic",
                                    source="user", importance=0.8))
    store.propose_corrections([PendingCorrection(
        target_id=target.id, action="replace", proposed_text="the user drinks tea",
        target_text=target.text, confidence=0.9, source="consolidation",
        id="corr-accept-test")])

    res = _run("accept", "corr-accept-test", expect_ok=False)

    fresh = _cli_store().get(target.id)
    assert fresh is not None and fresh.text == "the user drinks coffee", (
        f"privacy mode applied the correction anyway: text is now {fresh and fresh.text!r}")
    assert res.exit_code != 0
    assert "privacy mode" in res.output


def test_reject_refuses_and_creates_no_dismissal_sidecar(home):
    _assert_privacy_mode()
    from localm.memory.corrections import PendingCorrection
    from localm.memory.store import MemoryRecord
    store = _cli_store()
    target = store.add(MemoryRecord(text="the user drinks coffee", kind="semantic",
                                    source="user", importance=0.8))
    store.propose_corrections([PendingCorrection(
        target_id=target.id, action="replace", proposed_text="the user drinks tea",
        target_text=target.text, confidence=0.9, source="consolidation",
        id="corr-reject-test")])
    dismissed_file = store.path.with_suffix(".corrections-dismissed.json")

    res = _run("reject", "corr-reject-test", expect_ok=False)

    assert not dismissed_file.exists(), (
        "privacy mode rejected the correction anyway, writing the dismissal sidecar")
    assert res.exit_code != 0
    assert "privacy mode" in res.output


# --------------------------------------------------------------------------- #
#  corrections (read command, bespoke refusal - see cli.py)                   #
# --------------------------------------------------------------------------- #

def test_corrections_prints_the_privacy_note_and_leaves_the_sidecar_untouched(home):
    _assert_privacy_mode()
    from localm.memory.corrections import PendingCorrection
    from localm.memory.store import MemoryRecord
    store = _cli_store()
    live = store.add(MemoryRecord(text="a live fact", kind="semantic", source="user",
                                  importance=0.8))
    stale_target = store.add(MemoryRecord(text="a fact about to be deleted",
                                          kind="semantic", source="user",
                                          importance=0.8))
    store.propose_corrections([
        PendingCorrection(target_id=live.id, action="replace",
                          proposed_text="a live fact, revised",
                          target_text=live.text, confidence=0.9,
                          source="consolidation", id="corr-live"),
        PendingCorrection(target_id=stale_target.id, action="replace",
                          proposed_text="a stale proposal",
                          target_text=stale_target.text, confidence=0.9,
                          source="consolidation", id="corr-stale"),
    ])
    store.delete(stale_target.id)          # hard delete: leaves this correction's target gone

    corrections_file = store.path.with_suffix(".corrections.jsonl")
    assert corrections_file.exists(), "precondition: a pending correction is on disk"
    before = _safe_read_bytes(corrections_file)
    assert before, "precondition: the sidecar has content"

    res = _run("corrections")
    assert "privacy mode" in res.output
    assert "No pending corrections." not in res.output
    assert _safe_read_bytes(corrections_file) == before, (
        "listing corrections in privacy mode rewrote the sidecar")

    res_json = _run("corrections", "--json")
    assert _json.loads(res_json.stdout) == []
    assert "privacy mode" in res_json.stderr
    assert _safe_read_bytes(corrections_file) == before, (
        "listing corrections --json in privacy mode rewrote the sidecar")


# --------------------------------------------------------------------------- #
#  scope pin                                                                   #
# --------------------------------------------------------------------------- #

def test_the_gate_did_not_spread_to_the_deletion_verbs(home):
    """Characterizes the scope of this change: forget and clear are untouched by it."""
    _assert_privacy_mode()
    from localm.memory.store import MemoryRecord
    store = _cli_store()
    rec = store.add(MemoryRecord(text="delete me", kind="semantic", source="user",
                                 importance=0.8))

    res = _run("forget", rec.id, "--yes")
    assert res.exit_code == 0
    assert _cli_store().all() == []

    _cli_store().add(MemoryRecord(text="clear me", kind="semantic", source="user",
                                  importance=0.8))
    res2 = _run("clear", "--yes")
    assert res2.exit_code == 0
    assert _cli_store().all() == []
