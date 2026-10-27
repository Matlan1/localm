# SPDX-License-Identifier: AGPL-3.0-or-later
"""The REPL banner names the session mode by its value."""

from __future__ import annotations

import pytest
from rich.console import Console

from localm.plugins.coder import display
from localm.plugins.coder.audit import SessionMode


@pytest.mark.parametrize("mode, word", [
    (SessionMode.PRIVACY, "privacy"),
    (SessionMode.LOG, "log"),
    (SessionMode.FULL, "full"),
    ("log", "log"),
])
def test_banner_shows_the_mode_value(monkeypatch, tmp_path, mode, word):
    con = Console(record=True, width=200, highlight=False)
    monkeypatch.setattr(display, "console", con)
    display.print_banner("m", tmp_path, session_mode=mode)
    text = con.export_text()
    assert f"mode: {word}" in text, text
    assert "SessionMode" not in text, text
