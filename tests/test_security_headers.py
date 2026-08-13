# SPDX-License-Identifier: AGPL-3.0-or-later
"""R41 defense-in-depth: every response carries X-Content-Type-Options: nosniff,
and the GUI carries an ENFORCING Content-Security-Policy (R41 D1) whose
script-src is pinned to a per-request nonce. See dev-notes/SECURITY-xss-render-
review and dev-notes/W3-CSP-ENFORCING-2026-08-13.md.
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

    Membership has to be tested PER DIRECTIVE. `"x" in csp` is true when x sits
    in any directive at all, which is exactly how the outage below survived a
    green test.
    """
    for part in csp.split(";"):
        part = part.strip()
        if part == name or part.startswith(name + " "):
            return part[len(name):].strip()
    return ""


def test_script_src_admits_the_onnx_runtime_origin_not_just_connect_src():
    """The onnxruntime backend is a MODULE SCRIPT, so it needs script-src.

    REGRESSION, live 2026-08-13: cdn.jsdelivr.net was listed in connect-src only.
    The tts plugin's Kokoro bundle pulls the backend with a dynamic import(),
    which is governed by script-src, so the browser refused it and Kokoro died
    with "no available backend found" - neural TTS was completely dead. Measured
    from a live page: fetch() of that exact URL returned 200 (connect-src) while
    import() of the same URL was blocked (script-src).

    The pre-existing assertion above could not catch it: `"huggingface.co" in csp`
    is true no matter which directive the host sits in, and says nothing about
    jsdelivr - yet its comment claimed this very property ("so enforcing does not
    break TTS"). Hence a directive-scoped check.
    """
    with TestClient(create_app(_engine())) as c:
        r = c.get("/health")
    csp = r.headers.get("Content-Security-Policy", "")
    script_src = _directive(csp, "script-src")
    assert script_src, f"no script-src at all: {csp!r}"
    assert "https://cdn.jsdelivr.net" in script_src, (
        "the onnxruntime backend origin must be in script-src, not only "
        f"connect-src - a dynamic import() is a module script. script-src={script_src!r}"
    )
    # And the fetch side must keep working too: the model weights come over
    # fetch/XHR, which is connect-src's job.
    connect_src = _directive(csp, "connect-src")
    assert "https://huggingface.co" in connect_src, connect_src


def test_cross_origin_isolation_headers_are_present():
    """Both are needed, on the DOCUMENT, or the page is not isolated at all.

    Without isolation SharedArrayBuffer is unavailable and onnxruntime-web drops
    to numThreads=1. Measured on a 12-core box: 12883 ms vs 4762 ms for the same
    6.3 s of audio (2.04x vs 0.76x realtime), i.e. neural TTS went from stalling
    on every sentence to streaming ahead of playback.

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

    SECOND, INDEPENDENT block found live 2026-08-13 only after fixing the origin
    above: with the backend downloadable but WebAssembly compilation still
    refused, the load died with

        CompileError: WebAssembly.instantiate() violates the following Content
        Security policy directive ... is not an allowed source of script

    so NO backend could start - neither wasm nor webgpu, since onnxruntime-web is
    WebAssembly on both paths. The narrow CSP3 token for this permits WebAssembly
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
    # The broader grant would also make WebAssembly work, and would additionally
    # re-admit dynamic JS evaluation - which is most of what an enforcing CSP is
    # for. Keep the narrow one.
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
    """O2, the mechanical nonce-coverage check.

    Parses the SERVED shell rather than trusting the placeholder substitution,
    so a newly added inline <script> that nobody remembered to mark is caught
    here instead of white-screening the GUI. One uncovered inline script is the
    whole failure mode of this change.
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

    # The placeholder must never survive into a served page: if it did, the
    # substitution silently failed and every one of those scripts is dead.
    assert "{{LOCALM_CSP_NONCE}}" not in body, \
        "the CSP nonce placeholder reached the client unsubstituted"


def test_substitution_does_not_eat_the_nonce_global():
    """REGRESSION, and it was a live-browser find, not a unit-test one.

    The placeholder was first spelled __LOCALM_CSP_NONCE__, which is ALSO the
    name of the JS global the shell publishes for the artifact canvas. A plain
    str.replace therefore rewrote

        window.__LOCALM_CSP_NONCE__ = ...   ->   window.<nonce> = ...

    a subtraction on the left of an assignment: "SyntaxError: Invalid left-hand
    side in assignment", which killed the entire bootstrap script. Every unit
    test still passed, because the placeholder HAD been substituted - just in
    one place too many - so nothing that only looks for a surviving placeholder
    can see this. Assert the global itself survives.
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
