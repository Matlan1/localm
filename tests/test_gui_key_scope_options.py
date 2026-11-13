# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Settings key-minter's scope checkbox list offers EVERY scope localm
ships (`localm.scopes.all_known_scopes`), each with an English label, and its
owner-only dimming set names exactly `localm.scopes.PRIVILEGED_SCOPES` - no
more, no less.

`POST /v1/keys` refuses a privileged scope for a non-owner key regardless of what
the GUI offers (auth.create_key's allow_privileged gate), so the SERVER boundary
does not depend on this. What can drift is the GUI's own bookkeeping: settings.js
hand-maintains a KEY_SCOPES checkbox list and a PRIVILEGED_KEY_SCOPES set used to
dim and disable those checkboxes for a non-owner minter, and neither is derived
from localm.scopes at runtime. A missing checkbox hides a scope entirely; a
missing entry in the dimming set lets a non-owner check one and hit a 403 on
submit, which applyOwnerGate exists to avoid.

These checks read the real settings.js, not a description of it."""

from __future__ import annotations

import re

from localm import scopes as S
from localm.plugins.gui.web import STATIC_DIR

_SETTINGS_JS = STATIC_DIR / "pages" / "settings.js"


def _js_text() -> str:
    return _SETTINGS_JS.read_text(encoding="utf-8")


def _key_scope_values() -> list[str]:
    """The scope VALUES (not labels) offered as checkboxes, in document order."""
    text = _js_text()
    m = re.search(r"export const KEY_SCOPES = \[(.*?)\n\];", text, re.S)
    assert m, "settings.js no longer has an `export const KEY_SCOPES = [...]` array"
    return re.findall(r'\["([^"]+)",', m.group(1))


def _privileged_key_scopes() -> set[str]:
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
    """The set settings.js uses to dim and disable owner-only checkboxes names
    EXACTLY scopes.PRIVILEGED_SCOPES: a subset would leave a real privileged
    scope enabled for a non-owner, a superset would dim one the server never
    restricts."""
    assert _privileged_key_scopes() == S.PRIVILEGED_SCOPES


def test_every_offered_scope_is_a_real_known_scope():
    for scope in _key_scope_values():
        assert S.is_valid_scope(scope), f"{scope!r} is not a scope localm.scopes knows"


def test_no_duplicate_checkboxes():
    values = _key_scope_values()
    assert len(values) == len(set(values)), values


def test_every_scope_localm_ships_has_a_checkbox():
    """A scope with no box can be minted only by the CLI or the API."""
    missing = S.all_known_scopes() - set(_key_scope_values())
    assert not missing, (
        f"these scopes have no checkbox in the GUI key minter: {sorted(missing)}")


def test_every_offered_scope_has_an_english_label():
    catalog = (STATIC_DIR / "app" / "i18n-en.js").read_text(encoding="utf-8")
    m = re.search(r"export const KEY_SCOPES = \[(.*?)\n\];", _js_text(), re.S)
    labels = re.findall(r'\["[^"]+", "([^"]+)"\]', m.group(1))
    assert len(labels) == len(_key_scope_values())
    missing = [k for k in labels if f'"{k}":' not in catalog]
    assert not missing, f"scope labels with no English text: {missing}"
