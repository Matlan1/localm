# SPDX-License-Identifier: AGPL-3.0-or-later
"""`Detection.recommended` is a LEGACY field, and reaching for it is a real bug."""

from __future__ import annotations

import ast
import pathlib
import sys

from localm import hwdetect


# --------------------------------------------------------------------------- #
#  The divergence is real and must stay visible                                #
# --------------------------------------------------------------------------- #

def test_the_legacy_field_cannot_express_the_amd_answer(monkeypatch):
    """MEASURED live on this hardware class (Windows AMD RX 6900 XT, gfx1030): the field says 'vulkan' while the policy says 'amd-rocm'."""
    monkeypatch.setattr(sys, "platform", "win32")
    det = hwdetect.Detection(vendors=["amd"], recommended="vulkan",
                             source="test", gpu_names="amd radeon rx 6900 xt")

    assert hwdetect.recommended_install_backend(det) == "amd-rocm"
    assert det.recommended == "vulkan"
    assert det.recommended != hwdetect.recommended_install_backend(det)


def test_the_legacy_field_cannot_express_the_nvidia_answer(monkeypatch):
    """Not an AMD quirk: the field has no way to say 'cuda' either."""
    monkeypatch.setattr(sys, "platform", "win32")
    det = hwdetect.Detection(vendors=["nvidia"], recommended="vulkan",
                             source="test", gpu_names="nvidia geforce rtx 4090")

    assert hwdetect.recommended_install_backend(det) == "cuda"
    assert det.recommended != hwdetect.recommended_install_backend(det)


def test_they_agree_where_they_are_allowed_to(monkeypatch):
    """The reason this survived three times: on most hardware the two DO agree, so a wrong call site looks correct everywhere the author tested it."""
    monkeypatch.setattr(sys, "platform", "linux")
    det = hwdetect.Detection(vendors=["intel"], recommended="vulkan",
                             source="test", gpu_names="intel arc a770")

    assert hwdetect.recommended_install_backend(det) == "vulkan"
    assert det.recommended == hwdetect.recommended_install_backend(det)


# --------------------------------------------------------------------------- #
#  Nothing may read it - the guard that would have caught all three            #
# --------------------------------------------------------------------------- #

def _recommended_attribute_reads(root: pathlib.Path):
    """Every `<something>.recommended` attribute access under *root*."""
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
    """Fires-control for the instrument, run BEFORE trusting its clean result."""
    planted = ast.parse("value = det.recommended\n")
    found = [n for n in ast.walk(planted)
             if isinstance(n, ast.Attribute) and n.attr == "recommended"]

    assert found, "the AST scan cannot see an attribute read; it proves nothing"


def test_the_keyword_argument_form_is_not_counted_as_a_read():
    """The write inside hwdetect's own factory is legitimate and must not trip this."""
    written = ast.parse("d = Detection(vendors=[], recommended='cpu')\n")
    found = [n for n in ast.walk(written)
             if isinstance(n, ast.Attribute) and n.attr == "recommended"]

    assert found == []


def test_nothing_in_localm_reads_the_legacy_field():
    """The real assertion."""
    root = pathlib.Path(hwdetect.__file__).resolve().parent
    hits = _recommended_attribute_reads(root)

    assert hits == [], (
        "Detection.recommended is a legacy field that can only ever be "
        "'vulkan' or 'cpu' - call hwdetect.recommended_install_backend() "
        f"instead. Read at: {hits}")
