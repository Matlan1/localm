# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Settings key-minter's scope checkbox list offers EVERY scope `localm.scopes.PRIVILEGED_SCOPES` marks privileged, and its owner-only dimming set names exactly that same set - no more, no less."""

from __future__ import annotations

import re

from localm import scopes as S
from localm.plugins.gui.web import STATIC_DIR

_SETTINGS_JS = STATIC_DIR / "pages" / "settings.js"


def _js_text() -> str:
    return _SETTINGS_JS.read_text(encoding="utf-8")


def _key_scope_values() -> "list[str]":
    """The scope VALUES (not labels) offered as checkboxes, in document order."""
    text = _js_text()
    m = re.search(r"export const KEY_SCOPES = \[(.*?)\n\];", text, re.S)
    assert m, "settings.js no longer has an `export const KEY_SCOPES = [...]` array"
    return re.findall(r'\["([^"]+)",', m.group(1))


def _privileged_key_scopes() -> "set[str]":
    text = _js_text()
    m = re.search(
        r"export const PRIVILEGED_KEY_SCOPES = new Set\(\[(.*?)\]\);", text, re.S)
    assert m, "settings.js no longer has a `PRIVILEGED_KEY_SCOPES` Set"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def test_every_privileged_scope_has_a_checkbox():
    offered = set(_key_scope_values())
    missing = S.PRIVILEGED_SCOPES - offered
    assert not missing, (
        f"these privileged scopes have no checkbox in the GUI key minter at all: "
        f"{sorted(missing)}")


def test_privileged_key_scopes_constant_matches_the_python_source_of_truth():
    """The set settings.js uses to dim/disable owner-only checkboxes must name EXACTLY scopes.PRIVILEGED_SCOPES - not a hand-picked subset (which would silently leave a real privileged scope enabled for a non-owner) and not a superset (which would dim a scope the server never actually restricts)."""
    assert _privileged_key_scopes() == S.PRIVILEGED_SCOPES


def test_every_offered_scope_is_a_real_known_scope():
    for scope in _key_scope_values():
        assert S.is_valid_scope(scope), f"{scope!r} is not a scope localm.scopes knows"


def test_no_duplicate_checkboxes():
    values = _key_scope_values()
    assert len(values) == len(set(values)), values
