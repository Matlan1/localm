# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Settings "Inference runtime" backend picker offers EXACTLY the backends
`localm setup-llama --backend` accepts.

POST /api/runtime/update validates its `backend` against
setup_llama.BACKENDS, and the CLI's own click.Choice is built from that same
tuple. The GUI is the third statement of the same fact: index.html hardcodes the
<option> list, so that a card can offer a backend without a network read having
succeeded first.

The drift is one-directional: a backend added to setup_llama leaves the picker
offering the old set, and a renamed one makes the picker send a value the route
400s. This test reads BOTH REAL ARTEFACTS - the shipped index.html and the live
constant.
"""

from __future__ import annotations

import re

from localm import setup_llama as sl
from localm.plugins.gui.web import STATIC_DIR

# The empty value is the picker's "Keep the installed backend" entry. It is NOT
# a backend: it means "send no backend at all", which the route reads as "keep
# what is installed, or auto-detect when nothing is".
_KEEP_INSTALLED = ""


def _runtime_backend_options() -> "list[str]":
    """The <option> values of #runtime-backend, in document order."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    m = re.search(r'<select id="runtime-backend".*?</select>', html, re.S)
    assert m, "index.html no longer has a #runtime-backend <select>"
    return re.findall(r'<option value="([^"]*)"', m.group(0))


def test_picker_offers_exactly_the_cli_backends():
    opts = _runtime_backend_options()
    assert opts[0] == _KEEP_INSTALLED, (
        "the first entry must stay the no-op 'keep the installed backend' one, "
        f"got {opts[0]!r}")
    assert tuple(opts[1:]) == sl.BACKENDS, (
        "the Settings backend picker has drifted from setup_llama.BACKENDS.\n"
        f"  picker: {opts[1:]}\n"
        f"  BACKENDS: {list(sl.BACKENDS)}")


def test_every_offered_backend_is_labelled_not_bare():
    """A bare value tells a user nothing about which hardware it is for, and
    this list is the one place localm asks someone to choose a GPU backend
    without a terminal in front of them."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    m = re.search(r'<select id="runtime-backend".*?</select>', html, re.S)
    pairs = re.findall(r'<option value="([^"]*)">([^<]*)</option>', m.group(0))
    assert len(pairs) == len(sl.BACKENDS) + 1, pairs
    for value, label in pairs:
        if value == _KEEP_INSTALLED:
            continue
        assert label.strip() != value, f"{value!r} is offered with no explanation"
        assert label.strip().startswith(value), (
            f"the label for {value!r} must lead with the value the CLI uses, so "
            f"the two surfaces are recognisably the same choice; got {label!r}")


def test_auto_is_offered():
    """'auto' is what a bare `localm setup-llama` resolves through, and a real
    member of BACKENDS rather than a sentinel, so it must be selectable."""
    assert "auto" in _runtime_backend_options()
