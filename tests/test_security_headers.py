# SPDX-License-Identifier: AGPL-3.0-or-later
"""Defense in depth: every response carries X-Content-Type-Options: nosniff, and
the GUI carries an ENFORCING Content-Security-Policy whose script-src is pinned
to a per-request nonce.
"""

import re
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from localm.inference.http_server import create_app

# A <script ...> open tag that carries no src= attribute, i.e. an INLINE script.
# Those are the ones an enforcing script-src blocks without a nonce.
_INLINE_SCRIPT_OPEN = re.compile(
    r"<script\b(?![^>]*\bsrc=)[^>]*>", re.IGNORECASE)
_NONCE_ATTR = re.compile(r"""\bnonce=["']([^"']*)["']""", re.IGNORECASE)


def _engine():
    e = MagicMock()
    e.display_name = "test-model"
    e.loaded = True
    return e


def _shell_app():
    """The PRODUCTION wiring: create_app installs the _security_headers
    middleware that mints the nonce, attach_gui mounts the shell route that has
    to stamp it. create_app alone 404s on "/" - the shell is the GUI plugin's -
    and attach_gui alone has no middleware, so only the pair can express this
    property at all.
    """
    from localm.plugins.gui.web import attach_gui

    async def _switch(name):
        pass

    app = create_app(_engine())
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=_switch, active_model=lambda: "")
    return app


def _csp_nonce(resp):
    """The nonce the response's own CSP header advertises."""
    csp = resp.headers.get("Content-Security-Policy", "")
    m = re.search(r"'nonce-([^']+)'", csp)
    assert m, f"no nonce in script-src: {csp!r}"
    return m.group(1)


def test_nosniff_on_every_response():
    with TestClient(create_app(_engine())) as c:
        r = c.get("/health")
    assert r.headers.get("X-Content-Type-Options") == "nosniff"


def test_csp_enforcing_and_locked_down():
    with TestClient(create_app(_engine())) as c:
        r = c.get("/health")
    csp = r.headers.get("Content-Security-Policy", "")
    assert "default-src 'self'" in csp
    assert "object-src 'none'" in csp
    assert "base-uri 'none'" in csp
    assert "frame-ancestors 'none'" in csp
    # It ENFORCES: the report-only header must be gone, or a browser that honours
    # only the report-only one would silently go back to not blocking anything.
    assert r.headers.get("Content-Security-Policy-Report-Only") is None
    # script-src is nonce-pinned and must NOT re-admit inline scripts wholesale.
    assert re.search(r"script-src[^;]*'nonce-[^']+'", csp), csp
    assert "'unsafe-inline'" not in csp.split("script-src")[1].split(";")[0]
    # External deps the GUI legitimately needs stay allowed so enforcing does not
    # break TTS (HF + the onnx CDN).
    assert "huggingface.co" in csp


def _directive(csp: str, name: str) -> str:
    """The source list for one CSP directive, or "" when it is absent.

    Membership has to be tested PER DIRECTIVE: `"x" in csp` is true when x sits
    in any directive at all.
    """
    for part in csp.split(";"):
        part = part.strip()
        if part == name or part.startswith(name + " "):
            return part[len(name):].strip()
    return ""


def test_no_cdn_origin_is_granted_anywhere_in_the_policy():
    """The onnxruntime runtime is VENDORED, so no CDN origin belongs in the CSP.

    Asserted per directive rather than as a substring of the whole header. The
    tts plugin's Kokoro bundle pulls the onnxruntime backend with a dynamic
    import(), which is a MODULE SCRIPT and is therefore governed by script-src,
    not connect-src, so an origin sitting in connect-src alone still refuses the
    import ("no available backend found") while fetch() of the same URL
    succeeds.

    Vendoring the runtime removed the need for either grant, so this asserts the
    ABSENCE: an origin nothing uses is pure attack surface, and re-adding one is
    the cheap-looking "fix" whenever a TTS load error appears.

    Checked in EVERY directive, not just the two that carried it.
    """
    with TestClient(create_app(_engine())) as c:
        r = c.get("/health")
    csp = r.headers.get("Content-Security-Policy", "")
    assert csp, "no enforcing Content-Security-Policy at all"
    script_src = _directive(csp, "script-src")
    assert script_src, f"no script-src at all: {csp!r}"
    assert "cdn.jsdelivr.net" not in csp, (
        "a CDN origin is back in the CSP. The onnxruntime runtime ships under "
        "localm/plugins/builtin/tts/static/vendor/onnxruntime/ and is served "
        f"from 'self'; nothing needs a third-party script origin. csp={csp!r}"
    )
    # Only the model weights may come from off-box, and only over fetch/XHR,
    # which is connect-src's job. script-src must name no remote origin at all.
    for token in script_src.split():
        assert not token.startswith("http"), (
            "script-src must not admit any remote origin; the shell's own "
            f"scripts are same-origin and nonce-gated. offending token={token!r}"
        )
    connect_src = _directive(csp, "connect-src")
    assert "https://huggingface.co" in connect_src, connect_src


def test_script_src_still_admits_wasm_compilation_and_blob_workers():
    """Vendoring moved the runtime, it did not remove WebAssembly.

    Both grants are required and are unrelated to where the runtime is hosted:
    'wasm-unsafe-eval' because compiling any WebAssembly needs an explicit grant
    (onnxruntime-web is WebAssembly on both its wasm and webgpu paths), and
    blob: because a cross-origin-isolated page gives onnxruntime
    SharedArrayBuffer, so it loads its THREADED build whose worker arrives as a
    blob: module - and a dynamic import of that blob is governed by script-src,
    not worker-src.
    """
    with TestClient(create_app(_engine())) as c:
        r = c.get("/health")
    csp = r.headers.get("Content-Security-Policy", "")
    script_src = _directive(csp, "script-src")
    assert "'wasm-unsafe-eval'" in script_src, script_src
    assert "blob:" in script_src, script_src
    assert "blob:" in _directive(csp, "worker-src"), csp


def test_cross_origin_isolation_headers_are_present():
    """Both are needed, on the DOCUMENT, or the page is not isolated at all.

    Without isolation SharedArrayBuffer is unavailable and onnxruntime-web drops
    to numThreads=1, which is the difference between neural TTS stalling on
    every sentence and streaming ahead of playback.

    COOP alone or COEP alone does NOT isolate, which is why this asserts both
    rather than either.
    """
    with TestClient(create_app(_engine())) as c:
        r = c.get("/health")
    assert r.headers.get("Cross-Origin-Opener-Policy") == "same-origin"
    assert r.headers.get("Cross-Origin-Embedder-Policy") == "credentialless"


def test_coep_is_credentialless_not_require_corp():
    """require-corp would demand a CORP header on every cross-origin subresource.

    We do not control huggingface.co or the onnx CDN, so require-corp would break
    the model and backend downloads outright. credentialless is sufficient for
    isolation and correct here: none of localm's cross-origin fetches are
    authenticated, they are public downloads.
    """
    with TestClient(create_app(_engine())) as c:
        r = c.get("/health")
    assert r.headers.get("Cross-Origin-Embedder-Policy") != "require-corp"


def test_script_src_allows_webassembly_compilation():
    """onnxruntime-web is WebAssembly, and compiling it needs its own CSP grant.

    With the backend downloadable but WebAssembly compilation still refused, the
    load dies with

        CompileError: WebAssembly.instantiate() violates the following Content
        Security policy directive ... is not an allowed source of script

    so NO backend can start - neither wasm nor webgpu, since onnxruntime-web is
    WebAssembly on both paths. The narrow CSP3 token permits WebAssembly
    compilation WITHOUT permitting dynamic JavaScript evaluation, so it must be
    the one present and the broader token must NOT be.
    """
    with TestClient(create_app(_engine())) as c:
        r = c.get("/health")
    csp = r.headers.get("Content-Security-Policy", "")
    script_src = _directive(csp, "script-src")
    assert "'wasm-unsafe-eval'" in script_src, (
        "WebAssembly cannot be compiled without this grant, so every ONNX "
        f"backend fails to start. script-src={script_src!r}"
    )
    # The broader grant would also re-admit dynamic JS evaluation, so the narrow
    # one is used.
    assert "'unsafe-eval'" not in script_src.replace("'wasm-unsafe-eval'", ""), (
        f"script-src must not widen to full dynamic evaluation: {script_src!r}"
    )


def test_csp_nonce_is_per_request_not_per_process():
    """A nonce reused across requests is worth little: an attacker who learns it
    once could reuse it forever."""
    with TestClient(create_app(_engine())) as c:
        first = _csp_nonce(c.get("/health"))
        second = _csp_nonce(c.get("/health"))
    assert first != second, "CSP nonce must be minted per request"
    assert len(first) >= 16, f"nonce too short to be unguessable: {first!r}"


def test_every_inline_script_in_the_served_shell_carries_the_nonce():
    """The mechanical nonce-coverage check.

    Parses the SERVED shell rather than trusting the placeholder substitution,
    so a newly added inline <script> that nobody remembered to mark is caught
    here instead of white-screening the GUI. One uncovered inline script is the
    whole failure mode.
    """
    with TestClient(_shell_app()) as c:
        r = c.get("/")
    assert r.status_code == 200, r.status_code
    nonce = _csp_nonce(r)
    body = r.text

    opens = _INLINE_SCRIPT_OPEN.findall(body)
    assert opens, "found no inline <script> in the shell at all - check the fixture"

    uncovered = []
    for tag in opens:
        m = _NONCE_ATTR.search(tag)
        if not m or m.group(1) != nonce:
            uncovered.append(tag)
    assert not uncovered, (
        f"{len(uncovered)} inline <script> tag(s) in the served shell do not "
        f"carry this response's CSP nonce {nonce!r}: {uncovered}")

    # The placeholder must never survive into a served page.
    assert "{{LOCALM_CSP_NONCE}}" not in body, \
        "the CSP nonce placeholder reached the client unsubstituted"


def test_substitution_does_not_eat_the_nonce_global():
    """The placeholder must not eat the JS global of the same name.

    The placeholder is spelled __LOCALM_CSP_NONCE__, which is ALSO the name of
    the JS global the shell publishes for the artifact canvas, so a plain
    str.replace rewrites

        window.__LOCALM_CSP_NONCE__ = ...   ->   window.<nonce> = ...

    a subtraction on the left of an assignment: "SyntaxError: Invalid left-hand
    side in assignment", which kills the entire bootstrap script. A check that
    only looks for a surviving placeholder cannot see this, so this asserts the
    global itself survives.
    """
    with TestClient(_shell_app()) as c:
        body = c.get("/").text
    assert "window.__LOCALM_CSP_NONCE__ =" in body, (
        "the nonce substitution clobbered the JS global it is supposed to "
        "populate; the shell bootstrap will not parse")


def test_shell_token_snippet_is_nonced_too():
    """The open-mode shell-token snippet is injected server-side, so it is not
    covered by the placeholder in index.html and has to be nonced at the
    injection site. It is also the one inline script that carries a secret, so a
    silent failure here is the most expensive one."""
    from localm.plugins.gui import web as gui_web

    html = gui_web._index_html_with_shell_token("tok-abc", "N0NCE")
    assert "__LOCALM_SHELL_TOKEN__" in html
    snippet = html[html.index("<script"):html.index("__LOCALM_SHELL_TOKEN__")]
    assert 'nonce="N0NCE"' in snippet, snippet
    # And every inline script in that same document agrees on the value.
    for tag in _INLINE_SCRIPT_OPEN.findall(html):
        m = _NONCE_ATTR.search(tag)
        assert m and m.group(1) == "N0NCE", tag


def test_media_and_fetch_of_blob_urls_are_permitted():
    """The GUI mints blob: URLs from its OWN responses and then plays or reads
    them back, and two directives have to allow that.

    With `media-src` absent ENTIRELY, media falls back to `default-src 'self'`
    and every <video>/<audio> fed a blob: URL dies with MEDIA_ELEMENT_ERROR
    "Media load rejected by URL safety check". With no blob: in `connect-src`,
    "send to chat" and "copy image" fail with "Failed to fetch".

    The media half is the dangerous one because it is SILENT: a blocked src
    fires an error EVENT on the element rather than throwing, so the try/catch
    around the assignment cannot see it and the player sits there dead with no
    toast. That is a step failing while reporting success.

    Asserted on the HEADER rather than through a browser, so it cannot regress
    silently.
    """
    with TestClient(create_app(_engine())) as c:
        r = c.get("/health")
    csp = r.headers.get("Content-Security-Policy", "")

    assert "media-src" in csp, (
        "no media-src directive: media falls back to default-src 'self', which "
        f"has no blob:, and every gallery player dies silently. csp={csp!r}"
    )
    media_src = _directive(csp, "media-src")
    assert "blob:" in media_src, (
        f"<video>/<audio> cannot play a blob: URL. media-src={media_src!r}"
    )
    connect_src = _directive(csp, "connect-src")
    assert "blob:" in connect_src, (
        "fetch() of a blob: URL is refused, so 'send to chat' and 'copy image' "
        f"fail. connect-src={connect_src!r}"
    )
    # The grant stays scoped to blob:, which is same-origin-minted and cannot
    # point at a remote origin.
    assert "*" not in connect_src.replace("https://*.hf.co", ""), (
        f"connect-src must not carry a wildcard origin: {connect_src!r}"
    )


def test_form_action_is_locked_down_because_it_has_no_default_src_fallback():
    """A model-authored <form> survives sanitisation; form-action is what stops it.

    DOMPurify's default ALLOWED_TAGS includes `form`, so a reply rendering
        <form action="https://elsewhere/" method="post"><input name="apikey">
    reaches the DOM intact, the action resolves to that remote origin, and
    submitting raises no CSP violation of its own. No script is involved, so
    neither DOMPurify nor the script-src nonce sits in that path, and for an
    offline-first product that is a way for a rendered reply to post what the
    user typed into it off the machine.

    This asserts PRESENCE rather than a value: form-action is a NAVIGATION
    directive with no default-src fallback, so `default-src 'self'` looks like
    it covers submissions and does not. Omitting it allows submission anywhere.
    """
    with TestClient(create_app(_engine())) as c:
        r = c.get("/health")
    csp = r.headers.get("Content-Security-Policy", "")
    form_action = _directive(csp, "form-action")
    assert form_action, (
        "no form-action directive. default-src does NOT cover it (it is a "
        "navigation directive with no fallback), so form submission is "
        f"unrestricted and a sanitiser-surviving <form> can post off-box. csp={csp!r}"
    )
    assert form_action == "'none'", (
        "form-action must stay 'none': nothing in the GUI submits a form (there "
        "is no <form> element in static/ at all), so 'none' costs nothing and "
        "also closes the same-origin CSRF shape against localm's own /api. "
        f"form-action={form_action!r}"
    )
