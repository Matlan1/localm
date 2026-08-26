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


def _client(monkeypatch, *, mode, fetch=None):
    """A GUI app with the proxy route mounted and gui_proxy_remote_images forced
    to *mode* ("off" / "ask" / "on").

    load_config is patched where the ROUTE imports it (inside the handler, from
    localm.config), not where it is defined, so the patch is on the path the code
    under test actually takes. The value goes in RAW, so passing the pre-3-state
    True/False here also exercises the legacy coercion the route inherits from
    config.remote_image_mode.
    """
    from fastapi import FastAPI
    app = FastAPI()
    imgproxy.register(app, MagicMock())
    monkeypatch.setattr("localm.config.load_config",
                        lambda *a, **k: {"gui_proxy_remote_images": mode})
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

    c = _client(monkeypatch, mode="off", fetch=_spy)
    r = c.get("/api/image-proxy", params={"url": "https://example.com/a.png"})
    assert r.status_code == 403, r.text
    # The refusal must come BEFORE any outbound request. Asserting only on the
    # status code would pass on a version that fetched first and refused after,
    # which is the failure that would actually matter here.
    assert called == [], f"an outbound fetch was made while the feature was OFF: {called}"


def test_proxies_the_bytes_when_enabled(monkeypatch):
    c = _client(monkeypatch, mode="on", fetch=_ok_fetch)
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
    c = _client(monkeypatch, mode="on", fetch=_ok_fetch)
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

    c = _client(monkeypatch, mode="on", fetch=_refuse)
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

    c = _client(monkeypatch, mode="on", fetch=_spy)
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
    c = _client(monkeypatch, mode="on",
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
    c = _client(monkeypatch, mode="on",
                fetch=lambda url, **kw: (url, "image/svg+xml",
                                         b"<svg xmlns='http://www.w3.org/2000/svg'></svg>"))
    r = c.get("/api/image-proxy", params={"url": "https://example.com/x.svg"})
    assert r.status_code == 415, r.text
    assert "image/svg+xml" not in r.headers.get("content-type", "")


def test_content_type_parameters_do_not_defeat_the_allowlist(monkeypatch):
    """`image/png; charset=binary` is still image/png."""
    c = _client(monkeypatch, mode="on",
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

    c = _client(monkeypatch, mode="on", fetch=_capture)
    c.get("/api/image-proxy", params={"url": "https://example.com/a.png"})
    assert seen.get("max_bytes") == imgproxy._MAX_BYTES
    assert imgproxy._MAX_BYTES <= 25_000_000, "a display image should not be unbounded"


def test_a_truncated_oversize_image_is_REFUSED_not_served_corrupt(monkeypatch):
    """safe_fetch_bytes truncates at the cap; serving that would be a false success.

    netpolicy breaks out of its chunk loop at max_bytes and returns what it has.
    For a text fetch a clipped body is degraded but usable. For an IMAGE it is a
    corrupt file, and returning it with a 200 and a valid image/* type reports
    success for a step that failed (AGENTS.md rule 5) - the browser renders a
    half-decoded strip or nothing, and the client cannot tell that apart from a
    small image.
    """
    truncated = b"IMGDATA" * ((imgproxy._MAX_BYTES // 7) + 2)
    c = _client(monkeypatch, mode="on",
                fetch=lambda url, **kw: (url, "image/png", truncated[:imgproxy._MAX_BYTES]))
    r = c.get("/api/image-proxy", params={"url": "https://example.com/huge.png"})
    assert r.status_code == 413, (
        f"an image that hit the byte cap was served as {r.status_code} rather than "
        "refused, so a truncated/corrupt body reaches the page as a success")


def test_an_image_just_under_the_cap_is_still_served(monkeypatch):
    """The refusal above must not swallow legitimate large-but-complete images."""
    body = b"z" * (imgproxy._MAX_BYTES - 100)
    c = _client(monkeypatch, mode="on",
                fetch=lambda url, **kw: (url, "image/png", body))
    r = c.get("/api/image-proxy", params={"url": "https://example.com/big.png"})
    assert r.status_code == 200, r.text
    assert r.content == body


def test_a_fetch_failure_is_a_502_not_a_500(monkeypatch):
    """A dead remote host is not a localm bug and must not read as one."""
    def _boom(url, **kw):
        raise OSError("connection refused")

    c = _client(monkeypatch, mode="on", fetch=_boom)
    r = c.get("/api/image-proxy", params={"url": "https://example.com/a.png"})
    assert r.status_code == 502, r.text


def test_the_setting_defaults_to_off_in_the_shipped_config():
    """The route's refusal is only a default if the CONFIG default agrees.

    "ask" would ALSO be a closed channel, and is still the wrong default: it
    would start prompting on every install that never opted in to anything.
    """
    from localm.config import DEFAULT_CONFIG, REMOTE_IMAGE_OFF
    assert DEFAULT_CONFIG["gui_proxy_remote_images"] == REMOTE_IMAGE_OFF


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
    assert f.widget == "select", f.widget


def test_the_modes_match_config_s_own_constants():
    """settings_schema spells the options out (its localm.config imports are all
    deliberately lazy), so the copy needs a gate or it drifts. A drift here is
    silent in both directions: an option config cannot store, or a stored value
    the form cannot show."""
    from localm.config import REMOTE_IMAGE_LEGACY_BOOL, REMOTE_IMAGE_MODES
    from localm.settings_schema import CORE_FIELDS
    f = next(f for f in CORE_FIELDS if f.key == "gui_proxy_remote_images")
    assert tuple(f.options) == REMOTE_IMAGE_MODES
    assert f.legacy_bool == REMOTE_IMAGE_LEGACY_BOOL


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
#  QA 2026-08-20 (#6): the fetch must not run ON the event loop.               #
#                                                                             #
#  safe_fetch_bytes is a blocking urlopen with a 15s connect and a 15s read.   #
#  Called inline from this `async def` handler it stopped the WHOLE server for #
#  the length of the fetch - MEASURED, an unrelated GET /api/activity went     #
#  from 0.018s to 13.76s with one in-flight proxy fetch to an unroutable host, #
#  and the hang alarm fired independently with this frame on top. The URL      #
#  comes from a model-authored <img src>, so a rendered reply chose how long   #
#  the server stalled.                                                        #
#                                                                             #
#  Behavioural, not structural: it holds a real blocking call and requires an  #
#  unrelated coroutine to still get its turn, so it fails for any handler that #
#  stops offloading regardless of how the offload was written.                #
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

    That case is not one timeout. netpolicy applies its timeout separately to
    the connect and to each read, and follows redirects manually, so every hop
    re-pays both. This asserts the RELATION rather than the number, because the
    two bounds live in netpolicy and the arithmetic lives in imgproxy - the shape
    that breaks quietly when someone retunes one side. A literal 40.0 (the first
    version of this) covered a single connect+read pair and would have expired
    part-way down a legitimate three-hop CDN chain, turning a working image into
    a 504."""
    from localm import netpolicy
    worst_case = (netpolicy._MAX_REDIRECTS + 1) * 2 * netpolicy._DEFAULT_TIMEOUT
    assert imgproxy._fetch_budget_s() > worst_case, (
        f"the offload budget ({imgproxy._fetch_budget_s()}s) is below "
        f"safe_fetch_bytes' own legitimate worst case ({worst_case}s: "
        f"{netpolicy._MAX_REDIRECTS + 1} hops x connect+read x "
        f"{netpolicy._DEFAULT_TIMEOUT}s), so a slow but working image would 504")


# --------------------------------------------------------------------------- #
#  The `ask` state: per-origin consent (NEW-GUI-EXFIL-CHANNEL-REMOTE-IMAGES).  #
#                                                                             #
#  `on` moves the request from the browser to this machine. It does not stop   #
#  it, and the URL IS the exfiltration payload. `ask` is the state that stops   #
#  it for a host the reader has not agreed to, so what these pin is that the    #
#  refusal happens BEFORE any outbound fetch, and that nothing except the       #
#  reader's own answer can get past it.                                        #
# --------------------------------------------------------------------------- #

def test_ask_refuses_with_428_and_nothing_is_fetched(monkeypatch):
    called = []

    def _spy(url, **kw):
        called.append(url)
        return (url, "image/png", PNG)

    c = _client(monkeypatch, mode="ask", fetch=_spy)
    r = c.get("/api/image-proxy", params={"url": "https://example.com/a.png"})
    assert r.status_code == 428, r.text
    # The whole point of the state. A status-code-only assertion would pass on a
    # version that fetched first and refused after, which is the failure that
    # would actually matter: by then the address has already reached the host.
    assert called == [], f"an outbound fetch was made before the reader agreed: {called}"
    # 428 rather than 403 is load-bearing for the client, which uses the code to
    # tell "there is nothing you can do" from "ask, then retry".
    assert r.status_code != 403


def test_ask_fetches_once_the_request_carries_the_reader_s_consent(monkeypatch):
    c = _client(monkeypatch, mode="ask", fetch=_ok_fetch)
    r = c.get("/api/image-proxy",
              params={"url": "https://example.com/a.png", "consent": "1"})
    assert r.status_code == 200, r.text
    assert r.content == PNG
    assert r.headers["content-type"].startswith("image/png")


def test_consent_cannot_switch_the_feature_on_while_it_is_off(monkeypatch):
    """consent states the READER's answer for one origin. It is not a capability,
    and it must never reach past what `on` would have reached."""
    called = []

    def _spy(url, **kw):
        called.append(url)
        return (url, "image/png", PNG)

    c = _client(monkeypatch, mode="off", fetch=_spy)
    r = c.get("/api/image-proxy",
              params={"url": "https://example.com/a.png", "consent": "1"})
    assert r.status_code == 403, r.text
    assert called == [], f"consent reached past an OFF setting: {called}"


def test_a_url_carrying_its_own_consent_parameter_is_still_refused(monkeypatch):
    """The URL is the one part of this a MODEL chooses, so it is the one part
    that must not be able to answer for the reader.

    The client builds the query with encodeURIComponent, which escapes `&` and
    `=`; this asserts the server side of that, i.e. that a `consent=1` sitting
    inside the url value is read as part of the URL and never as the parameter.
    """
    called = []

    def _spy(url, **kw):
        called.append(url)
        return (url, "image/png", PNG)

    c = _client(monkeypatch, mode="ask", fetch=_spy)
    # params= encodes it exactly as the client's encodeURIComponent would.
    r = c.get("/api/image-proxy",
              params={"url": "https://evil.example/p.png?x=1&consent=1"})
    assert r.status_code == 428, r.text
    assert called == [], f"a model-chosen URL answered for the reader: {called}"


def test_a_config_still_holding_the_pre_three_state_boolean_keeps_working(monkeypatch):
    """There is no config migration step in this project: load_config() is the
    defaults plus the stored delta, so an install that switched the old boolean
    on still has `true` on disk. It must keep meaning "on"."""
    c = _client(monkeypatch, mode=True, fetch=_ok_fetch)
    assert c.get("/api/image-proxy",
                 params={"url": "https://example.com/a.png"}).status_code == 200

    c_off = _client(monkeypatch, mode=False, fetch=_ok_fetch)
    assert c_off.get("/api/image-proxy",
                     params={"url": "https://example.com/a.png"}).status_code == 403


def test_an_unreadable_setting_refuses_rather_than_fetches(monkeypatch):
    """A hand-edited or newer-version config.json must fail CLOSED here: this key
    decides whether rendering a reply makes an outbound request at all."""
    called = []

    def _spy(url, **kw):
        called.append(url)
        return (url, "image/png", PNG)

    c = _client(monkeypatch, mode="sometimes", fetch=_spy)
    r = c.get("/api/image-proxy", params={"url": "https://example.com/a.png"})
    assert r.status_code == 403, r.text
    assert called == []


def test_load_config_canonicalises_the_legacy_boolean_and_says_when_it_cannot(
        tmp_path, monkeypatch, capsys):
    """The read-side half, which is the entire migration: every consumer (the
    route, the settings schema, the CLI, the bug report) goes through
    load_config(), so normalising there is what makes a legacy file readable
    everywhere at once - and save_config() then heals the file."""
    import json
    from localm import config as cfgmod

    monkeypatch.setattr(cfgmod, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfgmod, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfgmod, "ensure_dirs", lambda *a, **k: None)

    (tmp_path / "config.json").write_text(
        json.dumps({"gui_proxy_remote_images": True}), encoding="utf-8")
    assert cfgmod.load_config()["gui_proxy_remote_images"] == "on"

    (tmp_path / "config.json").write_text(
        json.dumps({"gui_proxy_remote_images": "ask"}), encoding="utf-8")
    assert cfgmod.load_config()["gui_proxy_remote_images"] == "ask"

    # Unreadable: falls back to off AND says so. A silent fallback here would
    # turn a feature the owner had switched on back off with nothing to read.
    cfgmod._warned_bad_remote_image_mode.clear()
    (tmp_path / "config.json").write_text(
        json.dumps({"gui_proxy_remote_images": "sometimes"}), encoding="utf-8")
    assert cfgmod.load_config()["gui_proxy_remote_images"] == "off"
    err = capsys.readouterr().err
    assert "gui_proxy_remote_images" in err and "sometimes" in err, err


def test_patch_config_still_accepts_the_boolean_an_older_client_sends():
    """The write-side half. `localm config gui_proxy_remote_images true`, a shell
    script, and any API client written before the third state exists all still
    send a boolean; rejecting it would break them for no gain."""
    from localm.settings_schema import validate_update
    assert validate_update({"gui_proxy_remote_images": True}) == {
        "gui_proxy_remote_images": "on"}
    assert validate_update({"gui_proxy_remote_images": False}) == {
        "gui_proxy_remote_images": "off"}
    # An exact mode always wins over the boolean reading of the same word.
    assert validate_update({"gui_proxy_remote_images": "ask"}) == {
        "gui_proxy_remote_images": "ask"}
    with pytest.raises(ValueError):
        validate_update({"gui_proxy_remote_images": "sometimes"})
