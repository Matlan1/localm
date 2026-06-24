# SPDX-License-Identifier: AGPL-3.0-or-later
"""Seamless updates: the GUI shell and its assets must be served with a
revalidation cache policy so a new app.js / index.html / sw.js is picked up
automatically, without the user (or a tester) clearing the browser cache.

Starlette's StaticFiles sends an ETag but NO Cache-Control, so browsers fall back
to heuristic caching and can serve stale code. `Cache-Control: no-cache` keeps
caching (cheap 304s via the ETag) but forces a revalidation every load, so the
server - not the user - delivers current code.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.gui.web import attach_gui


def _client():
    app = FastAPI()
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None, active_model=lambda: "m")
    return TestClient(app)


def test_assets_revalidate():
    """app.js / style.css / pages.js / sw.js must carry Cache-Control: no-cache."""
    with _client() as c:
        for path in ("/app.js", "/style.css", "/pages.js", "/sw.js"):
            r = c.get(path)
            assert r.status_code == 200, path
            assert "no-cache" in r.headers.get("cache-control", ""), (
                f"{path} must revalidate (Cache-Control: no-cache); got "
                f"{r.headers.get('cache-control')!r}")


def test_shell_revalidates():
    """The shell document (/ and /index.html) must never be served stale."""
    with _client() as c:
        for path in ("/", "/index.html"):
            r = c.get(path)
            assert r.status_code == 200, path
            assert "no-cache" in r.headers.get("cache-control", ""), (
                f"the shell ({path}) must revalidate")


def test_unchanged_asset_can_304():
    """no-cache is not no-store: an unchanged asset still revalidates to a cheap
    304 (the ETag round-trip), so caching efficiency is preserved."""
    with _client() as c:
        r1 = c.get("/app.js")
        etag = r1.headers.get("etag")
        assert etag, "StaticFiles should still send an ETag for conditional requests"
        r2 = c.get("/app.js", headers={"If-None-Match": etag})
        assert r2.status_code == 304
