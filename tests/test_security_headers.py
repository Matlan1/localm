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
    assert "LOCALM_CSP_NONCE__" not in body, \
        "the CSP nonce placeholder reached the client unsubstituted"


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
