# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GUI remote-image proxy (localm/plugins/gui/routes/imgproxy.py).

A model reply can link a remote image, which the shell CSP refuses, so it
renders broken. This route fetches such an image server-side instead, off by
default, so the remote host never sees the browser's IP, User-Agent or referrer.

These tests pin the refusals the route makes before it serves anything.
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
    # The refusal must come BEFORE any outbound request.
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
    """Turning the feature OFF takes effect immediately: any max-age would let
    the browser keep serving an already-fetched image for that long after the
    owner switched the setting off, and would leave model-influenced bytes in
    the on-disk HTTP cache. The client keeps its own in-page blob cache, so a
    streaming re-render still does not refetch.
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
    """image/svg+xml passes any naive `startswith("image/")` check. Served from
    localm's OWN origin it renders as a document when opened directly, and SVG
    can carry script, so the allowlist names exact types rather than a prefix.
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


def test_a_truncated_oversize_image_is_REFUSED_not_served_corrupt(monkeypatch):
    """safe_fetch_bytes truncates at the cap; serving that would be a false success.

    netpolicy breaks out of its chunk loop at max_bytes and returns what it has.
    For an IMAGE that is a corrupt file, and returning it with a 200 and a valid
    image/* type is indistinguishable to the client from a small image.
    """
    truncated = b"IMGDATA" * ((imgproxy._MAX_BYTES // 7) + 2)
    c = _client(monkeypatch, enabled=True,
                fetch=lambda url, **kw: (url, "image/png", truncated[:imgproxy._MAX_BYTES]))
    r = c.get("/api/image-proxy", params={"url": "https://example.com/huge.png"})
    assert r.status_code == 413, (
        f"an image that hit the byte cap was served as {r.status_code} rather than "
        "refused, so a truncated/corrupt body reaches the page as a success")


def test_an_image_just_under_the_cap_is_still_served(monkeypatch):
    """The refusal above must not swallow legitimate large-but-complete images."""
    body = b"z" * (imgproxy._MAX_BYTES - 100)
    c = _client(monkeypatch, enabled=True,
                fetch=lambda url, **kw: (url, "image/png", body))
    r = c.get("/api/image-proxy", params={"url": "https://example.com/big.png"})
    assert r.status_code == 200, r.text
    assert r.content == body


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


# --------------------------------------------------------------------------- #
#  The fetch must not run ON the event loop.                                   #
#                                                                             #
#  safe_fetch_bytes is a blocking urlopen with a 15s connect and a 15s read,   #
#  and the URL comes from a model-authored <img src>. The check below holds a  #
#  real blocking call and requires an unrelated coroutine to still get its     #
#  turn.                                                                       #
# --------------------------------------------------------------------------- #

def test_the_fetch_does_not_block_the_event_loop(monkeypatch):
    import asyncio
    import time
    from fastapi import FastAPI

    BLOCK_S = 2.0

    def _slow_fetch(url, **kw):
        time.sleep(BLOCK_S)          # stands in for a connect to a dead host
        return (url, "image/png", PNG)

    app = FastAPI()
    imgproxy.register(app, MagicMock())
    monkeypatch.setattr("localm.config.load_config",
                        lambda *a, **k: {"gui_proxy_remote_images": True})
    monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", _slow_fetch)
    endpoint = next(r.endpoint for r in app.routes
                    if getattr(r, "path", None) == "/api/image-proxy")

    async def _drive():
        trivial_done = []

        async def _trivial():
            for _ in range(3):
                await asyncio.sleep(0)
            trivial_done.append(time.monotonic())

        t0 = time.monotonic()
        trivial = asyncio.ensure_future(_trivial())
        main = asyncio.ensure_future(endpoint(url="https://example.com/a.png"))
        try:
            await asyncio.wait_for(trivial, timeout=BLOCK_S * 0.5)
        except asyncio.TimeoutError:
            main.cancel()
            raise AssertionError(
                "a concurrent trivial coroutine never got to run while the "
                "image fetch was in flight - the fetch is on the event loop, "
                "so ONE slow image URL freezes the whole server")
        elapsed = trivial_done[0] - t0
        resp = await asyncio.wait_for(main, timeout=BLOCK_S + 10)
        return elapsed, resp

    elapsed, resp = asyncio.run(_drive())
    assert elapsed < BLOCK_S * 0.5, (
        f"an unrelated coroutine took {elapsed:.2f}s while a {BLOCK_S}s fetch "
        "was in flight - the event loop was blocked")
    # The route still does its job; the offload changes who waits, not the answer.
    assert resp.status_code == 200
    assert resp.body == PNG


def test_the_fetch_budget_stays_above_netpolicy_s_own_worst_case():
    """The offload's deadline must never fire for a slow-but-WORKING fetch, only
    for a wedged one - so it has to sit above safe_fetch_bytes' real worst case.

    That case is not one timeout: netpolicy applies its timeout separately to
    the connect and to each read, and follows redirects manually, so every hop
    re-pays both. Asserted as a RELATION rather than a number, since the two
    bounds live in netpolicy and the arithmetic lives in imgproxy."""
    from localm import netpolicy
    worst_case = (netpolicy._MAX_REDIRECTS + 1) * 2 * netpolicy._DEFAULT_TIMEOUT
    assert imgproxy._fetch_budget_s() > worst_case, (
        f"the offload budget ({imgproxy._fetch_budget_s()}s) is below "
        f"safe_fetch_bytes' own legitimate worst case ({worst_case}s: "
        f"{netpolicy._MAX_REDIRECTS + 1} hops x connect+read x "
        f"{netpolicy._DEFAULT_TIMEOUT}s), so a slow but working image would 504")
