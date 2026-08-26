# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GUI shell and its assets are served with a revalidation cache policy, so
a new app.js / index.html / sw.js is picked up without clearing the browser
cache.

Starlette's StaticFiles sends an ETag but NO Cache-Control, so browsers fall
back to heuristic caching and can serve stale code. `Cache-Control: no-cache`
keeps caching (cheap 304s via the ETag) while forcing a revalidation on every
load.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.gui.web import attach_gui


def _client():
    app = FastAPI()
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None, active_model=lambda: "m")
    return TestClient(app)


_STATIC = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "gui" / "static"


def _a_served_js() -> str:
    """A served GUI JS module path that exists, found by walking static/.
    Excludes the service worker (special-cased separately) and vendored
    third-party libs."""
    for p in sorted(_STATIC.rglob("*.js")):
        if "vendor" in p.parts or p.name == "sw.js":
            continue
        return "/" + p.relative_to(_STATIC).as_posix()
    raise AssertionError("no served GUI JS module found under static/")


def test_assets_revalidate():
    """JS/CSS assets carry Cache-Control: no-cache, so new code is picked up
    without a manual cache clear. Covers a per-module JS file discovered under
    static/, so the policy is checked on a subdirectory asset as well as
    top-level files."""
    with _client() as c:
        for path in ("/style.css", "/sw.js", _a_served_js()):
            r = c.get(path)
            assert r.status_code == 200, path
            assert "no-cache" in r.headers.get("cache-control", ""), (
                f"{path} must revalidate (Cache-Control: no-cache); got "
                f"{r.headers.get('cache-control')!r}")


def test_shell_revalidates():
    """The shell document (/ and /index.html) is never served stale."""
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
        asset = _a_served_js()
        r1 = c.get(asset)
        etag = r1.headers.get("etag")
        assert etag, "StaticFiles should still send an ETag for conditional requests"
        r2 = c.get(asset, headers={"If-None-Match": etag})
        assert r2.status_code == 304
