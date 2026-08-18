# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GUI remote-image proxy (localm/plugins/gui/routes/imgproxy.py).

A model reply can link a remote image. The shell CSP refuses it, so it renders
broken - the one place localm showed less than the comparable UIs. Those render
it by letting the BROWSER fetch it, which leaks the user's IP/User-Agent/referrer
to the remote host. This route fetches server-side instead, off by default.

These tests pin the refusals, because that is where the value is: the feature is
one route whose whole job is to say no in five different ways before it says yes.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from localm.plugins.gui.routes import imgproxy


def _client(monkeypatch, *, enabled, fetch=None):
    """A GUI app with the proxy route mounted and config forced on/off.

    load_config is patched where the ROUTE imports it (inside the handler, from
    localm.config), not where it is defined, so the patch is on the path the code
    under test actually takes.
    """
    from fastapi import FastAPI
    app = FastAPI()
    imgproxy.register(app, MagicMock())
    monkeypatch.setattr("localm.config.load_config",
                        lambda *a, **k: {"gui_proxy_remote_images": enabled})
    if fetch is not None:
        monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", fetch)
    return TestClient(app)


PNG = b"\x89PNG\r\n\x1a\n" + b"payload-bytes"


def _ok_fetch(url, **kw):
    return (url, "image/png", PNG)


def test_refused_by_default_and_nothing_is_fetched(monkeypatch):
    """The default install must behave exactly as it does today."""
    called = []

    def _spy(url, **kw):
        called.append(url)
        return (url, "image/png", PNG)

    c = _client(monkeypatch, enabled=False, fetch=_spy)
    r = c.get("/api/image-proxy", params={"url": "https://example.com/a.png"})
    assert r.status_code == 403, r.text
    # The refusal must come BEFORE any outbound request. Asserting only on the
    # status code would pass on a version that fetched first and refused after,
    # which is the failure that would actually matter here.
    assert called == [], f"an outbound fetch was made while the feature was OFF: {called}"


def test_proxies_the_bytes_when_enabled(monkeypatch):
    c = _client(monkeypatch, enabled=True, fetch=_ok_fetch)
    r = c.get("/api/image-proxy", params={"url": "https://example.com/a.png"})
    assert r.status_code == 200, r.text
    assert r.content == PNG
    assert r.headers["content-type"].startswith("image/png")
    # Attacker-choosable bytes served from our own origin stay inert.
    assert "default-src 'none'" in r.headers.get("content-security-policy", "")
    assert r.headers.get("x-content-type-options") == "nosniff"


def test_the_response_is_never_cached(monkeypatch):
    """Turning the feature OFF must take effect immediately.

    MEASURED: with `private, max-age=300` the browser kept serving an
    already-fetched image for five minutes after the owner switched the setting
    off, so the control they had just used appeared to do nothing - and
    model-influenced bytes sat in the on-disk HTTP cache of an offline-first
    product. The client keeps its own in-page blob cache, so a streaming
    re-render still does not refetch; the HTTP cache was buying almost nothing.
    """
    c = _client(monkeypatch, enabled=True, fetch=_ok_fetch)
    r = c.get("/api/image-proxy", params={"url": "https://example.com/a.png"})
    assert r.status_code == 200
    cache = r.headers.get("cache-control", "")
    assert "no-store" in cache, f"cache-control={cache!r}"
    assert "max-age" not in cache, (
        f"a max-age here delays the off-switch by exactly that long: {cache!r}")


def test_ssrf_refusal_from_netpolicy_is_surfaced_not_swallowed(monkeypatch):
    """The SSRF guard lives in netpolicy; this pins that the route lets it refuse.

    A loopback target is the canonical case: without the guard, the proxy would
    be a confused deputy able to read this machine's own private services.
    """
    from localm.netpolicy import NetworkPolicyError

    def _refuse(url, **kw):
        raise NetworkPolicyError("Refusing to fetch a loopback address.")

    c = _client(monkeypatch, enabled=True, fetch=_refuse)
    r = c.get("/api/image-proxy", params={"url": "http://127.0.0.1:8080/secret.png"})
    assert r.status_code == 403, r.text
    assert "loopback" in r.text.lower()


@pytest.mark.parametrize("scheme_url", [
    "file:///etc/passwd",
    "data:image/png;base64,AAAA",
    "ftp://example.com/a.png",
    "",
])
def test_non_http_schemes_never_reach_the_fetch_layer(monkeypatch, scheme_url):
    called = []

    def _spy(url, **kw):
        called.append(url)
        return (url, "image/png", PNG)

    c = _client(monkeypatch, enabled=True, fetch=_spy)
    r = c.get("/api/image-proxy", params={"url": scheme_url})
    assert r.status_code == 400, r.text
    assert called == [], f"{scheme_url!r} reached the fetch layer"


@pytest.mark.parametrize("ctype", [
    "text/html",
    "application/json",
    "text/plain",
    "application/octet-stream",
    "",
])
def test_non_image_content_types_are_refused(monkeypatch, ctype):
    """Otherwise the proxy is a general-purpose fetch-anything endpoint."""
    c = _client(monkeypatch, enabled=True,
                fetch=lambda url, **kw: (url, ctype, b"whatever"))
    r = c.get("/api/image-proxy", params={"url": "https://example.com/a"})
    assert r.status_code == 415, r.text


def test_svg_is_refused_even_though_it_is_an_image_type(monkeypatch):
    """The sharpest edge in the feature, so it gets its own test.

    image/svg+xml passes any naive `startswith("image/")` check. Served from our
    OWN origin it renders as a document when opened directly, and SVG can carry
    script - so allowing it would turn a model-chosen URL into script execution
    on localm's origin. If someone ever "fixes" the allowlist by loosening it to
    a prefix match, this is the test that must stop them.
    """
    c = _client(monkeypatch, enabled=True,
                fetch=lambda url, **kw: (url, "image/svg+xml",
                                         b"<svg xmlns='http://www.w3.org/2000/svg'></svg>"))
    r = c.get("/api/image-proxy", params={"url": "https://example.com/x.svg"})
    assert r.status_code == 415, r.text
    assert "image/svg+xml" not in r.headers.get("content-type", "")


def test_content_type_parameters_do_not_defeat_the_allowlist(monkeypatch):
    """`image/png; charset=binary` is still image/png."""
    c = _client(monkeypatch, enabled=True,
                fetch=lambda url, **kw: (url, "image/PNG; charset=binary", PNG))
    r = c.get("/api/image-proxy", params={"url": "https://example.com/a.png"})
    assert r.status_code == 200, r.text
    assert r.content == PNG


def test_the_byte_cap_is_passed_to_the_fetch_layer(monkeypatch):
    """The cap is netpolicy's to enforce; this pins that the route asks for one.

    Without it the default (1 MB) or no bound at all would apply, and one rendered
    reply could make this server pull an unbounded body.
    """
    seen = {}

    def _capture(url, **kw):
        seen.update(kw)
        return (url, "image/png", PNG)

    c = _client(monkeypatch, enabled=True, fetch=_capture)
    c.get("/api/image-proxy", params={"url": "https://example.com/a.png"})
    assert seen.get("max_bytes") == imgproxy._MAX_BYTES
    assert imgproxy._MAX_BYTES <= 25_000_000, "a display image should not be unbounded"


def test_a_fetch_failure_is_a_502_not_a_500(monkeypatch):
    """A dead remote host is not a localm bug and must not read as one."""
    def _boom(url, **kw):
        raise OSError("connection refused")

    c = _client(monkeypatch, enabled=True, fetch=_boom)
    r = c.get("/api/image-proxy", params={"url": "https://example.com/a.png"})
    assert r.status_code == 502, r.text


def test_the_setting_defaults_to_off_in_the_shipped_config():
    """The route's refusal is only a default if the CONFIG default agrees."""
    from localm.config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["gui_proxy_remote_images"] is False


def test_the_setting_is_owner_only_and_visible_in_the_schema():
    # admin_only_keys() is the PUBLIC accessor routes/config.py itself gates on,
    # so asserting through it checks the gate rather than a parallel reading of
    # the dataclass field.
    from localm.settings_schema import CORE_FIELDS, admin_only_keys
    assert "gui_proxy_remote_images" in admin_only_keys(), (
        "a non-owner config:write key must not be able to switch on outbound "
        "fetching triggered by rendered model content")
    f = next(f for f in CORE_FIELDS if f.key == "gui_proxy_remote_images")
    assert f.group == "Network"
    assert f.widget == "toggle", f.widget


def test_the_schema_still_shows_the_toggle_in_a_keyless_install():
    """admin_only must not make the toggle unreachable for the ordinary user.

    routes/config.py computes `is_owner = held is None or ADMIN in held`, so open
    mode counts as owner. If that ever changes, admin_only would hide this control
    from the default single-user GUI and the feature would be unusable rather than
    merely off.
    """
    from localm.settings_schema import schema_json
    keys = {f["key"] for f in schema_json(values={}, is_owner=True)}
    assert "gui_proxy_remote_images" in keys
    hidden = {f["key"] for f in schema_json(values={}, is_owner=False)}
    assert "gui_proxy_remote_images" not in hidden, (
        "expected the field to be owner-gated in the schema for a scoped key")
