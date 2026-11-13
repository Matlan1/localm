# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every test that sets capability routing's GPU reading does it through
tests._routing_fit.pin_vram_reading, which first drops the reading the server
cached for planning; otherwise a test gets an earlier test's reading."""

from __future__ import annotations

from pathlib import Path


def test_no_test_sets_the_routing_reading_but_through_pin_vram_reading():
    tests_dir = Path(__file__).resolve().parent
    allowed = {tests_dir / "_routing_fit.py", Path(__file__).resolve()}
    offenders = sorted(
        str(p.relative_to(tests_dir)) for p in tests_dir.rglob("*.py")
        if p.resolve() not in allowed
        and '"routing_vram_reading"' in p.read_text(encoding="utf-8", errors="replace"))
    assert offenders == []
