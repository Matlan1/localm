# SPDX-License-Identifier: AGPL-3.0-or-later
"""R41 defense-in-depth: every response carries X-Content-Type-Options: nosniff,
and the GUI carries a Content-Security-Policy (report-only, so it never breaks the
app) as a backstop behind DOMPurify. See dev-notes/SECURITY-xss-render-review.
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from localm.inference.http_server import create_app


def _engine():
    e = MagicMock()
    e.display_name = "test-model"
    e.loaded = True
    return e


def test_nosniff_on_every_response():
    with TestClient(create_app(_engine())) as c:
        r = c.get("/health")
    assert r.headers.get("X-Content-Type-Options") == "nosniff"


def test_csp_report_only_present_and_locked_down():
    with TestClient(create_app(_engine())) as c:
        r = c.get("/health")
    csp = r.headers.get("Content-Security-Policy-Report-Only", "")
    # Report-Only (never blocks) so it cannot break the GUI; assert the key
    # directives are present and the dangerous ones are closed.
    assert "default-src 'self'" in csp
    assert "object-src 'none'" in csp
    assert "base-uri 'none'" in csp
    assert "frame-ancestors 'none'" in csp
    # It must NOT be the enforcing header yet (enforce needs the inline-script nonce).
    assert r.headers.get("Content-Security-Policy") is None
    # External deps the GUI legitimately needs must be allowed so a future enforce
    # does not silently break TTS (HF + the onnx CDN).
    assert "huggingface.co" in csp
