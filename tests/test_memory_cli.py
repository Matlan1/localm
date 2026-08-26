# SPDX-License-Identifier: AGPL-3.0-or-later
"""O5 (parity board): the `localm memory` command group.

The memory plugin shipped no `cli` manifest key, so installing it added ZERO CLI
commands and every memory operation was GUI-only - while `localm job add --memory`
could already schedule the consolidation that CREATES corrections the terminal then
had no way to read, accept, reject or undo.

The properties these tests exist to pin are the ones that fail SILENTLY:

  - the CLI and the routes must resolve the SAME namespace, or the CLI shows an
    empty list and the user concludes localm has learned nothing about them;
  - `MemoryRecord.__post_init__` silently COERCES an unknown kind to "semantic" and
    an unknown source to "synth", so a CLI inventing its own values files the user's
    own assertion as machine-synthesised at a lower weight, with no error anywhere;
  - `delete()` does not archive, so promising a restore would be a promise `restore`
    cannot keep;
  - `clear()` does not touch the forgotten sidecar, so reporting the memory cleared
    without taking it would leave remembered text on disk under a false claim.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """A throwaway LOCALM_HOME the CLI and the store both resolve to."""
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


# --------------------------------------------------------------------------- #
#  The namespace                                                               #
# --------------------------------------------------------------------------- #

def test_cli_and_routes_resolve_the_same_store(home):
    """Pinned against the ROUTE's own helper, not against a constant.

    A CLI that opened a different agent, scope_key or root would pass every other
    test in this file while showing the user an empty memory. Asserting both
    against a hardcoded path would not catch it either - only comparing the two
    live resolutions does.
    """
    from localm.plugins.builtin.memory.cli import _store as cli_store
    from localm.plugins.builtin.memory.plug import _chat_store as route_store
    assert cli_store().path == route_store().path


def test_a_fact_written_by_the_route_is_visible_to_the_cli(home):
    """The end-to-end form of the same property, through the real surfaces."""
    from localm.memory.store import MemoryRecord
    from localm.plugins.builtin.memory.plug import _chat_store
    _chat_store().add(MemoryRecord(text="written by the route", kind="semantic",
                                   source="user", importance=0.8))
    out = _run("list").output
    assert "written by the route" in out

    # ...and the reverse.
    _run("add", "written by the cli")
    texts = [r.text for r in _chat_store().all()]
    assert "written by the cli" in texts


# --------------------------------------------------------------------------- #
#  add                                                                         #
# --------------------------------------------------------------------------- #

def test_add_produces_the_same_record_the_append_route_does(home):
    """kind semantic, source user, importance 0.8 - matching
    `POST /api/memory/append` exactly.

    Both wrong values are SILENTLY corrected by `MemoryRecord.__post_init__`
    (an invalid kind becomes "semantic", an invalid source becomes "synth"), so a
    mismatch here produces no error at all - it just files the user's own
    assertion as machine-synthesised, at a lower weight, where prune's user-fact
    eviction reporting stops seeing it.
    """
    _run("add", "the user drinks tea")
    rec = _cli_store().all()[0]
    assert rec.text == "the user drinks tea"
    assert rec.kind == "semantic"
    assert rec.source == "user", (
        "a fact the human typed is user-sourced; 'synth' is what an invalid value "
        "silently degrades to")
    assert rec.importance == 0.8


def test_add_rejects_an_invalid_kind_instead_of_silently_coercing_it(home):
    """The choices are constrained precisely because the coercion is silent."""
    res = _run("add", "x", "--kind", "preference", expect_ok=False)
    assert res.exit_code != 0
    assert "preference" in res.output or "choice" in res.output.lower()


def test_add_refuses_an_empty_fact(home):
    res = _run("add", "   ", expect_ok=False)
    assert res.exit_code == 1
    assert "empty" in res.output.lower()


def test_add_refuses_past_the_cap_like_the_route_does(home, monkeypatch):
    """Refuse rather than accept a fact the next prune would silently evict -
    the same guard the append route applies."""
    import localm.memory.store as _st
    monkeypatch.setattr(_st, "N_MAX", 2)
    _run("add", "one")
    _run("add", "two")
    res = _run("add", "three", expect_ok=False)
    assert res.exit_code == 1
    assert "cap" in res.output.lower()
    assert len(_cli_store().all()) == 2, "nothing was added past the cap"


# --------------------------------------------------------------------------- #
#  forget / forgotten / restore                                                #
# --------------------------------------------------------------------------- #

def test_forget_deletes_and_does_not_claim_it_is_recoverable(home):
    """`delete()` never writes the forgotten sidecar, so the CLI must not claim
    the record is recoverable."""
    _run("add", "the user drinks tea")
    mem_id = _cli_store().all()[0].id
    out = _run("forget", mem_id, "--yes").output
    assert _cli_store().all() == []
    assert _cli_store().forgotten() == [], (
        "a hand forget is a hard delete - if this ever archives, the help text "
        "and the forgotten/restore wording must change with it")
    assert "restore" not in out.lower(), (
        "must not offer a restore that cannot work")


def test_forget_refuses_an_unknown_id_without_touching_anything(home):
    """Guarded twice - the get() lookup and delete()'s own False. Both the exit
    code and the surviving record are asserted; the status alone would not show
    that the OTHER fact was left alone."""
    _run("add", "keep me")
    res = _run("forget", "nope", "--yes", expect_ok=False)
    assert res.exit_code == 1
    assert "nope" in res.output
    assert [r.text for r in _cli_store().all()] == ["keep me"]


def test_forgotten_lists_what_localm_dropped_itself_and_restore_brings_it_back(home,
                                                                              monkeypatch):
    """The archive is filled by prune eviction, not by `forget`."""
    import localm.memory.store as _st
    for i in range(3):
        _run("add", f"fact number {i}")
    store = _cli_store()
    store.prune(n_max=1)
    assert len(store.forgotten()) == 2

    out = _run("forgotten").output
    assert "archived fact(s)" in out
    gone_id = _cli_store().forgotten()[0]["id"]
    assert gone_id in out
    # The rows carry `forgotten_at`, NOT `reason` - that key belongs to the
    # coder's episode archive.
    assert "just now" in out, (
        "the archived-at column rendered '?' - it is reading a key the row does "
        "not have")

    _run("restore", gone_id)
    assert gone_id in [r.id for r in _cli_store().all()]
    assert _st.N_MAX  # the module really is the one we patched in the cap test


def test_restore_refuses_an_id_that_was_never_archived(home):
    _run("add", "live fact")
    live_id = _cli_store().all()[0].id
    res = _run("restore", live_id, expect_ok=False)
    assert res.exit_code == 1
    assert "forgotten" in res.output.lower()


# --------------------------------------------------------------------------- #
#  clear                                                                       #
# --------------------------------------------------------------------------- #

def test_clear_takes_the_archive_too_so_the_claim_is_true(home):
    """A plain `clear()` leaves every archived record readable in the
    `.forgotten.jsonl` sidecar, so the CLI must take the archive too before it
    reports the memory cleared."""
    for i in range(3):
        _run("add", f"secret fact {i}")
    store = _cli_store()
    store.prune(n_max=1)
    assert store.forgotten(), "precondition: something is archived"
    forgotten_file = store.path.with_suffix(".forgotten.jsonl")
    assert forgotten_file.exists()

    out = _run("clear", "--yes").output
    assert "Erased" in out
    fresh = _cli_store()
    assert fresh.all() == []
    assert fresh.forgotten() == []
    assert not forgotten_file.exists(), (
        "remembered text survived a clear that reported success")


def test_clear_on_an_empty_store_says_so_rather_than_claiming_an_erase(home):
    out = _run("clear", "--yes").output
    assert "Nothing to clear" in out


def test_clear_reports_a_partial_erase_as_a_failure(home, monkeypatch):
    """The one command with no undo: a partial erase reports as a failure."""
    _run("add", "stubborn")
    import localm.memory.store as _st
    monkeypatch.setattr(_st.MemoryStore, "clear",
                        lambda self, *, include_forgotten=False: None)
    res = _run("clear", "--yes", expect_ok=False)
    assert res.exit_code == 1
    assert "NOT reported as cleared" in res.output


# --------------------------------------------------------------------------- #
#  corrections                                                                 #
# --------------------------------------------------------------------------- #

def _propose(store, target_id, text):
    from localm.memory.corrections import PendingCorrection
    store.propose_corrections([PendingCorrection(
        target_id=target_id, action="replace", proposed_text=text,
        target_text="", confidence=0.9, source="consolidation")])


def test_corrections_are_listed_and_can_be_accepted(home):
    _run("add", "the user drinks coffee")
    store = _cli_store()
    target = store.all()[0].id
    _propose(store, target, "the user drinks tea")

    out = _run("corrections").output
    assert "pending correction" in out
    corr_id = _cli_store().corrections()[0].id
    assert corr_id in out

    _run("accept", corr_id)
    texts = [r.text for r in _cli_store().all()]
    assert "the user drinks tea" in texts
    assert _cli_store().corrections() == []


def test_a_rejected_correction_leaves_the_memory_alone(home):
    _run("add", "the user drinks coffee")
    store = _cli_store()
    _propose(store, store.all()[0].id, "the user drinks tea")
    corr_id = _cli_store().corrections()[0].id

    _run("reject", corr_id)
    texts = [r.text for r in _cli_store().all()]
    assert texts == ["the user drinks coffee"]
    assert _cli_store().corrections() == []


def test_an_unresolvable_correction_does_not_assert_it_does_not_exist(home):
    """`resolve_correction` returns None for TWO reasons - unknown id, and an
    unreadable corrections file, which it treats as non-destructive and warns
    about. `corrections()` returns [] for that second case too, so neither can
    disambiguate. Asserting "no such correction" would be a clean negative for a
    step that may simply have failed."""
    res = _run("accept", "nosuchid", expect_ok=False)
    assert res.exit_code == 1
    low = res.output.lower()
    assert "could not be read" in low, "must name the second possibility"
    assert "nothing was changed" in low


# --------------------------------------------------------------------------- #
#  clear vs the correction sidecars - NEW-MEMORY-CLEAR-LEAVES-TEXT             #
# --------------------------------------------------------------------------- #

def test_clear_erases_the_corrections_sidecars_too(home):
    """MEASURED before this was written (reproduces the ticket exactly): `clear()`
    took the forgotten archive but left two other sidecars holding the user's own
    words completely untouched - `.corrections.jsonl` (`target_text` /
    `proposed_text`) and `.corrections-dismissed.json` (a rejected proposal's text,
    casefolded, inside its dedup key). `localm memory clear -y` still reported
    "Erased N remembered and M forgotten fact(s)." and exited 0 while a grep of
    the home directory returned the sentence verbatim - exactly what AGENTS.md
    rule 5 forbids for a privacy step."""
    _run("add", "My bank PIN reminder is my dog Rex birth year")
    store = _cli_store()
    target = store.all()[0].id
    _propose(store, target, "a first proposed correction naming Rex")
    first_id = store.corrections()[0].id
    _run("reject", first_id)                      # populates the dismissed sidecar
    _propose(store, target, "a second still-pending correction naming Rex")

    corrections_file = store.path.with_suffix(".corrections.jsonl")
    dismissed_file = store.path.with_suffix(".corrections-dismissed.json")
    assert corrections_file.exists(), "precondition: a pending correction is on disk"
    assert dismissed_file.exists(), "precondition: a dismissed correction is on disk"

    out = _run("clear", "--yes").output
    assert "Erased" in out
    assert not corrections_file.exists(), (
        "verbatim correction text survived a clear that reported success")
    assert not dismissed_file.exists(), (
        "a rejected proposal's casefolded text survived a clear that reported "
        "success")
    fresh = _cli_store()
    assert fresh.remnants() == []


def test_clear_does_not_wave_through_an_orphaned_corrections_sidecar(home):
    """The fast path ("Nothing to clear") used to gate only on `all()` and
    `forgotten()`. A pending correction whose target record was hard-deleted
    (`forget`, not archived) is invisible to both: `all()` no longer has the
    record and `forget` never writes the forgotten archive. So a namespace could
    reach 0 live and 0 forgotten while `.corrections.jsonl` and
    `.corrections-dismissed.json` still held verbatim text on disk, and `clear -y`
    would print "Nothing to clear" and exit without even trying to erase it -
    the same false-success bug through a second door."""
    _run("add", "a rare sentence that must not survive")
    store = _cli_store()
    target = store.all()[0].id
    _propose(store, target, "first proposal about the rare sentence")
    first_id = store.corrections()[0].id
    _run("reject", first_id)                      # populates the dismissed sidecar
    _propose(store, target, "second, still-pending proposal about the rare sentence")
    _run("forget", target, "--yes")                # hard delete: no forgotten archive

    fresh = _cli_store()
    assert fresh.all() == [], "precondition: nothing live"
    assert fresh.forgotten() == [], "precondition: nothing archived (hard delete)"
    corrections_file = fresh.path.with_suffix(".corrections.jsonl")
    dismissed_file = fresh.path.with_suffix(".corrections-dismissed.json")
    assert corrections_file.exists(), (
        "precondition: the orphaned pending correction is still on disk")
    assert dismissed_file.exists(), (
        "precondition: the dismissed correction is still on disk")

    out = _run("clear", "--yes").output
    assert "Nothing to clear" not in out, (
        "claimed there was nothing to clear while correction sidecars remained")
    assert "Erased" in out
    assert not corrections_file.exists()
    assert not dismissed_file.exists()
