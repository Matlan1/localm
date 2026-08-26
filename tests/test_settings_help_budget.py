# SPDX-License-Identifier: AGPL-3.0-or-later
"""The settings help-text budget: docs/gui-design.md rule 9, enforced.

Rule 9 says a setting's help states what the control does and what changes if
you alter it; rationale, threat models and history live in a code comment or the
docs. A character count is the part of that a machine can check.
"""

import re

import pytest

from localm import settings_schema as ss

MAX_HELP_CHARS = 200

# admin_only fields exempted from MAX_HELP_CHARS while their replacement wording
# is pending sign-off. Currently empty; test_pending_signoff_list_is_exact below
# fails on an entry that no longer needs the exemption.
PENDING_SIGNOFF = frozenset()

# A positional reference ("the maximum below", "the global mode above") is false
# the moment anything moves, and .settings-fields is a TWO-COLUMN grid, so the
# next field renders to the RIGHT rather than below. Name the setting instead.
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
    """A HIDDEN field is NOT exempt from the budget.

    HIDDEN is a rendering decision that gets reversed: gpu_split_indices and
    main_gpu_index are HIDDEN only because a dedicated Live-tuning control
    renders them instead.
    """
    hidden = [f for _, f in _all_fields() if f.widget == ss.Widget.HIDDEN]
    assert hidden, "no HIDDEN fields left - this test has lost its subject"
    over = [f.key for f in hidden
            if f.key not in PENDING_SIGNOFF and len(f.help or "") > MAX_HELP_CHARS]
    assert not over, f"HIDDEN fields over budget: {over}"


def test_pending_signoff_list_is_exact():
    """The ratchet. Every exemption must still be earning its place.

    Three ways this goes red:
      - a key here no longer exists in the schema (renamed or removed);
      - a key here is no longer admin_only, so the trust-boundary reasoning does
        not apply to it and it should be trimmed;
      - a key here is ALREADY within budget, so the rewrite happened and the
        exemption was left behind.
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
    """A count, so adding an exemption is a visible act.

    The tracked count is 0. Adding a trust-boundary exemption means bumping this
    assertion in the same diff and saying why in the PENDING_SIGNOFF comment
    above.
    """
    assert len(PENDING_SIGNOFF) == 0
