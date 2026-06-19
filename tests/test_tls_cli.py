# SPDX-License-Identifier: AGPL-3.0-or-later
"""TLS wiring in the CLIs (NET-1 unit 2): the _resolve_tls decision and that the
serve/http_server entry points thread ssl params to uvicorn.

LOCALM_HOME is pinned per-test by the autouse conftest fixture, so the local CA
+ leaf land under a throwaway home.
"""

import os

import click
import pytest

from localm.cli import _resolve_tls


def test_loopback_is_plain_http():
    assert _resolve_tls("127.0.0.1", no_tls=False, tls_cert=None, tls_key=None) == (None, None)
    assert _resolve_tls("localhost", no_tls=False, tls_cert=None, tls_key=None) == (None, None)
    assert _resolve_tls("::1", no_tls=False, tls_cert=None, tls_key=None) == (None, None)


def test_network_bind_generates_cert():
    cert, key = _resolve_tls("0.0.0.0", no_tls=False, tls_cert=None, tls_key=None)
    assert cert and key
    assert os.path.isfile(cert) and os.path.isfile(key)


def test_no_tls_forces_plain_http_even_on_network_bind():
    assert _resolve_tls("0.0.0.0", no_tls=True, tls_cert=None, tls_key=None) == (None, None)


def test_user_supplied_pair_overrides(tmp_path):
    cert = tmp_path / "c.pem"
    key = tmp_path / "k.pem"
    cert.write_text("cert")
    key.write_text("key")
    # An explicit pair is honoured verbatim, even on loopback.
    assert _resolve_tls("127.0.0.1", no_tls=False,
                        tls_cert=str(cert), tls_key=str(key)) == (str(cert), str(key))


def test_half_specified_pair_is_a_usage_error(tmp_path):
    cert = tmp_path / "c.pem"
    cert.write_text("cert")
    with pytest.raises(click.UsageError):
        _resolve_tls("0.0.0.0", no_tls=False, tls_cert=str(cert), tls_key=None)
    with pytest.raises(click.UsageError):
        _resolve_tls("0.0.0.0", no_tls=False, tls_cert=None, tls_key=str(cert))


def test_http_server_serve_passes_ssl_to_uvicorn(monkeypatch):
    """serve() must forward ssl_certfile/ssl_keyfile to uvicorn.run."""
    import uvicorn
    from localm.inference import http_server

    captured = {}

    def fake_run(app, **kwargs):
        captured.update(kwargs)
        captured["bind_host"] = getattr(app.state, "bind_host", None)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(http_server, "create_app", lambda engine: _FakeApp())

    http_server.serve(object(), host="0.0.0.0", port=9443,
                      ssl_certfile="/x/server.crt", ssl_keyfile="/x/server.key")
    assert captured["ssl_certfile"] == "/x/server.crt"
    assert captured["ssl_keyfile"] == "/x/server.key"
    assert captured["bind_host"] == "0.0.0.0"


class _FakeApp:
    class _State:
        pass

    def __init__(self):
        self.state = _FakeApp._State()
