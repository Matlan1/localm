# SPDX-License-Identifier: AGPL-3.0-or-later
"""The settings help-text budget: docs/gui-design.md rule 9, enforced.

Rule 9 says a setting's help states what the control does and what changes if
you alter it; rationale, threat models and history live in a code comment or the
docs. A character count is the only part of that which a machine can check, and
it is fully objective - so it cannot generate the false positives that get a
gate switched off, and it binds every field added from now on rather than
decaying the moment this cleanup ships.

Measured before the cleanup that added this file: 30 of 116 fields were over
200 characters, 3,205 characters over budget in total, with the worst single
field at 559. Median was 136, so most of the page was already fine.
"""

import re

import pytest

from localm import settings_schema as ss

# docs/gui-design.md rule 9.
MAX_HELP_CHARS = 200

# admin_only fields whose help is effectively a threat model go here while
# awaiting the MAINTAINER's sign-off on the exact replacement wording (decision
# D8): tightening prose is safe to do unilaterally, deciding which risk a
# warning communicates is not. Decision D2 still caps them at 200 like every
# other field once signed off.
#
# EMPTY as of 2026-08-13: the seven fields that lived here
# (coder_reviewer, keep_diagnostics, coder_untrusted_provenance,
# rag_indexing_mode, embedding_model, hf_trust_remote_code,
# update_allow_prerelease) were all signed off and rewritten - see
# dev-notes/settings-ia-2026-08-13/09-ADMIN-COPY-PROPOSALS.md for the approved
# text and dev-notes/ for the applied why-comments. The mechanism stays for the
# next trust-boundary field that needs the same treatment.
#
# This is a shrinking list, not a permanent carve-out, and
# test_pending_signoff_list_is_exact below is what stops it rotting into one -
# an entry that has been trimmed but not deleted from here FAILS.
PENDING_SIGNOFF = frozenset()

# A positional reference ("the maximum below", "the global mode above") is false
# the moment anything moves, and .settings-fields is a TWO-COLUMN grid, so the
# next field renders to the RIGHT rather than below - which made several of
# these false while standing still. Decision D3: name the setting instead.
_POSITIONAL = re.compile(
    r"\b(above|below|beside|to the left|to the right|preceding|following)\b",
    re.IGNORECASE)


def _all_fields():
    """Every field the settings page can render, as (list name, field)."""
    return ([("CORE_FIELDS", f) for f in ss.CORE_FIELDS]
            + [("MEDIA_PLUGIN_FIELDS", f) for f in ss.MEDIA_PLUGIN_FIELDS]
            + [("TTS_FIELDS", f) for f in ss.TTS_FIELDS])


def test_every_help_string_is_within_the_budget():
    """Rule 9's hard cap, over every list the settings page renders."""
    over = [
        f"{name}[{f.key}] {len(f.help)} chars"
        for name, f in _all_fields()
        if f.key not in PENDING_SIGNOFF and len(f.help or "") > MAX_HELP_CHARS
    ]
    assert not over, (
        f"{len(over)} help string(s) exceed {MAX_HELP_CHARS} characters "
        f"(docs/gui-design.md rule 9): " + "; ".join(over))


def test_the_budget_covers_hidden_fields_too():
    """A HIDDEN field renders no control today, so it is tempting to exempt it.

    It is not exempt, deliberately: HIDDEN is a rendering decision that gets
    reversed (gpu_split_indices and main_gpu_index are HIDDEN only because a
    dedicated Live-tuning control renders them instead), and a field that
    becomes visible must not bring a wall of text in with it. This test states
    that the coverage above is total rather than leaving it to be inferred.
    """
    hidden = [f for _, f in _all_fields() if f.widget == ss.Widget.HIDDEN]
    assert hidden, "no HIDDEN fields left - this test has lost its subject"
    over = [f.key for f in hidden
            if f.key not in PENDING_SIGNOFF and len(f.help or "") > MAX_HELP_CHARS]
    assert not over, f"HIDDEN fields over budget: {over}"


def test_pending_signoff_list_is_exact():
    """The ratchet. Every exemption must still be earning its place.

    Three ways this goes red, and each is the correct outcome:
      - a key here no longer exists in the schema (renamed or removed);
      - a key here is no longer admin_only, so D8's trust-boundary reasoning
        does not apply to it and it should just be trimmed;
      - a key here is ALREADY within budget, i.e. someone did the rewrite and
        left the exemption behind. That is exactly how an exemption list decays
        into a permanent carve-out, so it fails loudly.
    """
    by_key = {f.key: f for _, f in _all_fields()}

    missing = sorted(PENDING_SIGNOFF - set(by_key))
    assert not missing, (
        f"PENDING_SIGNOFF names fields that no longer exist: {missing}")

    not_admin = sorted(k for k in PENDING_SIGNOFF if not by_key[k].admin_only)
    assert not not_admin, (
        "PENDING_SIGNOFF exists for trust-boundary copy (D8), but these are not "
        f"admin_only and so need no sign-off - just trim them: {not_admin}")

    already_fine = sorted(
        k for k in PENDING_SIGNOFF if len(by_key[k].help or "") <= MAX_HELP_CHARS)
    assert not already_fine, (
        "these were rewritten but left in PENDING_SIGNOFF - remove them from "
        f"the list so the exemption keeps shrinking: {already_fine}")


def test_no_positional_references_in_labels_or_help():
    """Decision D3: name the setting you mean, never its position."""
    hits = []
    for name, f in _all_fields():
        if f.key in PENDING_SIGNOFF:
            continue
        for what, text in (("help", f.help or ""), ("label", f.label or "")):
            for m in _POSITIONAL.finditer(text):
                hits.append(f"{name}[{f.key}].{what} says {m.group()!r}")
    assert not hits, (
        f"{len(hits)} positional reference(s) - a two-column grid puts the next "
        "field to the RIGHT, and the group-first move set changed which nav tab "
        "a field is on, so these are false: " + "; ".join(hits))


@pytest.mark.parametrize("key", sorted(PENDING_SIGNOFF))
def test_pending_signoff_fields_are_admin_only_and_over_budget(key):
    """Pin WHY each pending field is exempt, one assertion each, so a future
    edit that quietly adds one has to change this file and say why. Currently
    parametrizes over zero keys (PENDING_SIGNOFF is empty) - that is expected,
    not a gap: this activates again the moment a new entry is added."""
    by_key = {f.key: f for _, f in _all_fields()}
    assert by_key[key].admin_only
    assert len(by_key[key].help or "") > MAX_HELP_CHARS


def test_pending_signoff_count_is_tracked():
    """A count, so adding an exemption is a visible, deliberate act.

    All seven fields that once lived in PENDING_SIGNOFF were signed off and
    rewritten 2026-08-13 (dev-notes/settings-ia-2026-08-13/09-ADMIN-COPY-
    PROPOSALS.md), so the tracked count is now 0. Anyone adding a new
    trust-boundary exemption must bump this assertion in the same diff and
    say why in the PENDING_SIGNOFF comment above - a silent bump is exactly
    what this test exists to prevent.
    """
    assert len(PENDING_SIGNOFF) == 0
