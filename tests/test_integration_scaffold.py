# SPDX-License-Identifier: AGPL-3.0-or-later
"""Integration-test scaffold. Beyond the coarse `integration` marker, real
end-to-end paths are tagged with a resource-specific marker - `real_gguf`,
`real_comfy`, `real_browser` - that conftest gates: a test carrying one is
skipped (not failed) unless its resource is actually available.

This module holds the always-runnable oracle (the markers are registered) plus
gated skeletons for the comfy and browser paths; the gguf path is the existing
tests/test_gguf_smoke_integration.py, now also tagged `real_gguf`.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_real_resource_markers_registered():
    """The three resource markers are declared in pyproject so `-m real_gguf`
    (etc.) works and pytest does not warn about unknown markers."""
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    markers = cfg["tool"]["pytest"]["ini_options"]["markers"]
    names = {m.split(":", 1)[0].strip() for m in markers}
    for m in ("real_gguf", "real_comfy", "real_browser"):
        assert m in names, f"{m} marker is not registered in pyproject.toml"


def test_gguf_smoke_is_tagged_real_gguf():
    """The existing GGUF end-to-end smoke test carries the fine-grained marker so
    it can be selected (and gated) as a real_gguf test, not just `integration`."""
    text = (ROOT / "tests" / "test_gguf_smoke_integration.py").read_text(encoding="utf-8")
    assert "real_gguf" in text, "the GGUF smoke test should be tagged real_gguf"


# --------------------------------------------------------------------------- #
#  Gated skeletons. conftest skips these unless the resource is present, so in  #
#  CI they appear as skips, never failures.                                     #
# --------------------------------------------------------------------------- #

@pytest.mark.integration
@pytest.mark.real_comfy
def test_comfy_reachable():
    """A real ComfyUI (set LOCALM_TEST_COMFY_URL) answers its stats endpoint."""
    import urllib.request

    base = os.environ["LOCALM_TEST_COMFY_URL"].rstrip("/")
    with urllib.request.urlopen(f"{base}/system_stats", timeout=10) as r:
        assert r.status == 200


@pytest.mark.integration
@pytest.mark.real_browser
def test_gui_loads_in_a_real_browser():
    """The GUI shell loads its static index in a real browser (Playwright)."""
    from playwright.sync_api import sync_playwright

    index = ROOT / "localm" / "plugins" / "gui" / "static" / "index.html"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(index.as_uri())
            assert page.title() is not None
        finally:
            browser.close()
