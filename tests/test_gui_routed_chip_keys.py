# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every capability routing can report as a gap has a label in both GUI
catalogs, so the routed chip never shows a raw i18n key."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from localm.model_manager import capabilities as caps

_STATIC = Path(__file__).resolve().parent.parent / "localm" / "plugins" / "gui" / "static"
_GAPS = (*caps.BOOLEAN_CAPABILITIES, caps.CONTEXT_LENGTH)


@pytest.mark.parametrize("cap", _GAPS)
def test_the_english_catalog_labels_every_routing_gap(cap):
    en = (_STATIC / "app" / "i18n-en.js").read_text(encoding="utf-8")
    assert re.search(rf'"chat\.routed\.cap\.{re.escape(cap)}":\s*"[^"]+"', en)


@pytest.mark.parametrize("cap", _GAPS)
def test_the_german_catalog_labels_every_routing_gap(cap):
    de = json.loads((_STATIC / "i18n" / "de.json").read_text(encoding="utf-8"))
    assert de.get(f"chat.routed.cap.{cap}")
