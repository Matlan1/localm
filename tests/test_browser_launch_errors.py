# SPDX-License-Identifier: AGPL-3.0-or-later
"""A failed browser launch must name the command that fixes it.

The pip extra installs the playwright PACKAGE. It does not bring the Chromium
build playwright drives: that is a separate download, one build per version. So
an install that followed the documented steps can still fail at launch, and the
raw playwright error is not something a user can act on.
"""

from __future__ import annotations

import asyncio

import pytest

from localm.browser import session as bsession


class _FakeChromium:
    def __init__(self, exc):
        self._exc = exc

    async def launch(self, **kw):
        raise self._exc


class _FakePlaywright:
    def __init__(self, exc):
        self.chromium = _FakeChromium(exc)

    async def stop(self):
        return None


def _patched(monkeypatch, exc):
    fake = _FakePlaywright(exc)

    class _Factory:
        async def start(self):
            return fake

    monkeypatch.setattr(bsession, "_require_playwright", lambda: (lambda: _Factory()))
    return fake


def test_bundled_launch_failure_names_the_chromium_download(monkeypatch):
    _patched(monkeypatch, RuntimeError("Executable doesn't exist at ...ms-playwright"))
    sess = bsession.BrowserSession("t-bundled", engine="bundled")

    with pytest.raises(bsession.BrowserUnavailableError) as ei:
        asyncio.run(sess._launch())

    msg = str(ei.value)
    assert "playwright install chromium" in msg, (
        f"a bundled-engine launch failure must name the download command: {msg}")


def test_system_launch_failure_still_points_at_the_system_browser(monkeypatch):
    # The system engine keeps its own, different remedy: install Chrome or go
    # back to the bundled engine. It must not be given the chromium-download
    # message, which would be wrong for it.
    _patched(monkeypatch, RuntimeError("channel chrome not found"))
    sess = bsession.BrowserSession("t-system", engine="system")

    with pytest.raises(bsession.BrowserUnavailableError) as ei:
        asyncio.run(sess._launch())

    msg = str(ei.value)
    assert "Google Chrome" in msg
    assert "playwright install chromium" not in msg
