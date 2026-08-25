# SPDX-License-Identifier: AGPL-3.0-or-later
"""A failed bug-report upload must diagnose WHERE it failed (offline/DNS, server unreachable, TLS, timeout, server-rejected, rate-limited) from the actual attempt - never by contacting a third-party host - so the user is told what went wrong and the report is kept for a retry or manual send."""

import socket
import ssl
import urllib.error
import urllib.request

import pytest

from localm import bugreport
from tests._fake_https import patch_https_transport


@pytest.mark.parametrize("exc,expected", [
    (socket.gaierror(11001, "getaddrinfo failed"), "offline_or_dns"),
    (ssl.SSLError("handshake failed"), "tls"),
    (TimeoutError("timed out"), "timeout"),
    (ConnectionRefusedError("refused"), "unreachable"),
    (OSError("network is unreachable"), "offline_or_dns"),
    (OSError("some odd socket error"), "unreachable"),
    (ValueError("not a network error"), "unknown"),
])
def test_classify_url_error_stages(exc, expected):
    stage, hint = bugreport._classify_url_error(exc)
    assert stage == expected
    assert hint and isinstance(hint, str)      # always an actionable message


def test_classify_unwraps_urlerror_reason():
    # A urllib URLError wraps the real cause in .reason - the classifier must see
    # through it to the gaierror, not report "unknown".
    err = urllib.error.URLError(socket.gaierror(11001, "getaddrinfo failed"))
    stage, _ = bugreport._classify_url_error(err)
    assert stage == "offline_or_dns"


def test_upload_report_network_error_carries_stage(monkeypatch):
    def _boom(req, timeout=None, context=None):
        raise urllib.error.URLError(socket.gaierror(11001, "getaddrinfo failed"))
    patch_https_transport(monkeypatch, _boom)
    with pytest.raises(bugreport.LocalmError) as ei:
        bugreport.upload_report("t", "b", url="https://proxy.example/report")
    assert ei.value.stage == "offline_or_dns"
    assert ei.value.hint


def test_upload_report_server_rejected_stage():
    # A custom opener returning a 500 exercises the status-based rejection path.
    def opener(u, data, hdrs, to):
        return 500, "boom"
    with pytest.raises(bugreport.LocalmError) as ei:
        bugreport.upload_report("t", "b", url="https://proxy.example/report",
                                opener=opener)
    assert ei.value.stage == "server_rejected"
    assert "HTTP 500" in ei.value.reason


def test_upload_report_rate_limited_stage():
    def opener(u, data, hdrs, to):
        return 429, '{"retry_after": 12}'
    with pytest.raises(bugreport.RateLimitedError) as ei:
        bugreport.upload_report("t", "b", url="https://proxy.example/report",
                                opener=opener)
    assert ei.value.stage == "rate_limited"
    assert ei.value.retry_after == 12
    assert ei.value.hint


def test_upload_report_no_endpoint_stage(monkeypatch):
    monkeypatch.setattr(bugreport, "upload_config", lambda: (None, None))
    with pytest.raises(bugreport.LocalmError) as ei:
        bugreport.upload_report("t", "b")
    assert ei.value.stage == "no_endpoint"
