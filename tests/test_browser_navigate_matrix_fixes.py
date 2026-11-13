# SPDX-License-Identifier: AGPL-3.0-or-later
"""Navigation in the automated browser: an address typed without a scheme
gets one, and a destination the gate refuses before loading is recorded with
the other refused requests."""

import pytest

from localm.browser import netgate
from localm.browser.session import BrowserSession
from tests.test_browser_netgate import _set, cfg_home  # noqa: F401


@pytest.mark.parametrize("typed, expected", [
    ("example.com", "https://example.com"),
    ("example.com/path?q=1", "https://example.com/path?q=1"),
    ("localhost:8771", "http://localhost:8771"),
    ("127.0.0.1:8080/x", "http://127.0.0.1:8080/x"),
    ("[::1]:9000", "http://[::1]:9000"),
    ("localhost.example.com", "https://localhost.example.com"),
    ("https://example.com", "https://example.com"),
    ("http://example.com", "http://example.com"),
    ("about:blank", "about:blank"),
    ("data:text/html,x", "data:text/html,x"),
    ("file:///etc/passwd", "file:///etc/passwd"),
    ("  example.com  ", "https://example.com"),
    ("", ""),
    ("mailto:a@example.com", "mailto:a@example.com"),
    ("foo:bar@evil.com", "foo:bar@evil.com"),
    ("javascript:alert(1)", "javascript:alert(1)"),
    ("view-source:example.com", "view-source:example.com"),
    ("C:\\x", "C:\\x"),
    ("file:C:/x", "file:C:/x"),
    ("127.evil.com", "https://127.evil.com"),
    ("127.0.0.1.nip.io", "https://127.0.0.1.nip.io"),
    ("example.com:8443/x", "https://example.com:8443/x"),
])
def test_with_default_scheme(typed, expected):
    assert netgate.with_default_scheme(typed) == expected


def test_a_scheme_given_as_written_still_meets_the_gate(cfg_home):  # noqa: F811
    assert "scheme 'file'" in netgate.decide(
        netgate.with_default_scheme("file:///etc/passwd"))


def test_a_pre_refused_destination_is_recorded_and_named(cfg_home):  # noqa: F811
    _set(net_mode="off")
    sess = BrowserSession("t-prerefused")
    res = sess.navigate("example.com")
    assert res["ok"] is False
    assert res["url"] == "https://example.com"
    assert res["refused"]
    assert sess.blocked_requests() == [{"url": "https://example.com",
                                        "reason": res["refused"]}]
