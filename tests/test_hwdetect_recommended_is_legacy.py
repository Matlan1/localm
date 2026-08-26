# SPDX-License-Identifier: AGPL-3.0-or-later
"""`Detection.recommended` is a LEGACY field, and reaching for it is a real bug.

It only ever holds "vulkan" or "cpu" and cannot express "cuda", "amd-rocm" or
"metal". It answers "which of the two universally-safe backends applies", not
"the backend this machine should install". Those agree on most hardware and
diverge on the vendor-optimised paths.

`recommended_install_backend()` is the answer to the second question, and is
what every caller uses.

The first test pins the DIVERGENCE, so the two are never collapsed into one;
the second proves nothing in `localm/` reads the field at all.
"""

from __future__ import annotations

import ast
import pathlib
import sys

from localm import hwdetect


# --------------------------------------------------------------------------- #
#  The divergence stays visible                                                #
# --------------------------------------------------------------------------- #

def test_the_legacy_field_cannot_express_the_amd_answer(monkeypatch):
    """On Windows + AMD the field says "vulkan" while the policy says
    "amd-rocm", so reading the field would downgrade a ROCm install."""
    monkeypatch.setattr(sys, "platform", "win32")
    det = hwdetect.Detection(vendors=["amd"], recommended="vulkan",
                             source="test", gpu_names="amd radeon rx 6900 xt")

    assert hwdetect.recommended_install_backend(det) == "amd-rocm"
    assert det.recommended == "vulkan"
    assert det.recommended != hwdetect.recommended_install_backend(det)


def test_the_legacy_field_cannot_express_the_nvidia_answer(monkeypatch):
    """Not an AMD quirk: the field has no way to say "cuda" either."""
    monkeypatch.setattr(sys, "platform", "win32")
    det = hwdetect.Detection(vendors=["nvidia"], recommended="vulkan",
                             source="test", gpu_names="nvidia geforce rtx 4090")

    assert hwdetect.recommended_install_backend(det) == "cuda"
    assert det.recommended != hwdetect.recommended_install_backend(det)


def test_they_agree_where_they_are_allowed_to(monkeypatch):
    """The reason this survived three times: on most hardware the two DO agree,
    so a wrong call site looks correct everywhere the author tested it."""
    monkeypatch.setattr(sys, "platform", "linux")
    det = hwdetect.Detection(vendors=["intel"], recommended="vulkan",
                             source="test", gpu_names="intel arc a770")

    assert hwdetect.recommended_install_backend(det) == "vulkan"
    assert det.recommended == hwdetect.recommended_install_backend(det)


# --------------------------------------------------------------------------- #
#  Nothing may read it                                                         #
# --------------------------------------------------------------------------- #

def _recommended_attribute_reads(root: pathlib.Path):
    """Every `<something>.recommended` attribute access under *root*.

    AST, not grep, for two reasons: it cannot see COMMENTS (and every surviving
    mention of this field in the tree is a comment warning against it, which a
    grep would flag as a false positive), and it does not match the
    `Detection(recommended=...)` keyword argument, which is a legitimate write
    rather than a read."""
    hits = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "recommended":
                hits.append(f"{path}:{node.lineno}")
    return hits


def test_the_scan_can_actually_detect_a_read():
    """Fires-control for the instrument, run BEFORE trusting its clean result.
    A scan that cannot find a planted read would report an empty tree forever
    and the test below would be worthless."""
    planted = ast.parse("value = det.recommended\n")
    found = [n for n in ast.walk(planted)
             if isinstance(n, ast.Attribute) and n.attr == "recommended"]

    assert found, "the AST scan cannot see an attribute read; it proves nothing"


def test_the_keyword_argument_form_is_not_counted_as_a_read():
    """The write inside hwdetect's own factory is legitimate and must not trip
    this. Asserted rather than assumed, since a scan that flagged it would be
    turned off within a day."""
    written = ast.parse("d = Detection(vendors=[], recommended='cpu')\n")
    found = [n for n in ast.walk(written)
             if isinstance(n, ast.Attribute) and n.attr == "recommended"]

    assert found == []


def test_nothing_in_localm_reads_the_legacy_field():
    """The real assertion. Every consumer was migrated to
    recommended_install_backend(); this keeps it that way."""
    root = pathlib.Path(hwdetect.__file__).resolve().parent
    hits = _recommended_attribute_reads(root)

    assert hits == [], (
        "Detection.recommended is a legacy field that can only ever be "
        "'vulkan' or 'cpu' - call hwdetect.recommended_install_backend() "
        f"instead. Read at: {hits}")
