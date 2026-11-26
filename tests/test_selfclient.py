# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.selfclient: the self-authenticated loopback HTTP helper shared by
rag/plug.py's three _make_self_* factories and vram.py's
unload_chat_for_media/reload_chat_after_media. Every one of those call sites
must go through this single implementation of the auth-header + TLS-verify
setup.
"""

import requests

from localm.selfclient import self_request


class _FakeResp:
    ok = True


def test_requires_base_url():
    import pytest
    with pytest.raises(ValueError):
        self_request("GET", "/health")


def test_builds_auth_header_from_env_key(monkeypatch):
    monkeypatch.setenv("LOCALM_API_KEY", "sekret123")
    captured = {}

    def fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResp()

    monkeypatch.setattr(requests, "request", fake_request)
    resp = self_request("POST", "/models/unload", timeout=300,
                        base_url="http://127.0.0.1:8642/v1")
    assert resp.ok
    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:8642/v1/models/unload"
    assert captured["headers"]["Authorization"] == "Bearer sekret123"
    assert captured["timeout"] == 300


def test_builds_auth_header_from_persisted_key_file(monkeypatch):
    """A keyed server whose owner key lives in auth.key (``localm key generate`` /
    the launcher) but NOT in the environment must still authenticate its OWN
    loopback self-calls: the key is resolved via ``auth.get_api_key()`` (env,
    then the persisted auth.key)."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    from localm import auth
    auth.set_api_key("file-only-key-123")   # writes <throwaway home>/auth.key
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(kwargs)
        return _FakeResp()

    monkeypatch.setattr(requests, "request", fake_request)
    self_request("POST", "/embeddings", base_url="http://127.0.0.1:8642/v1")
    assert captured["headers"]["Authorization"] == "Bearer file-only-key-123"


def test_env_key_wins_over_persisted_file(monkeypatch):
    """The env var still takes precedence over the persisted file (the launcher's
    one-run override), matching auth.get_api_key()'s own precedence."""
    from localm import auth
    auth.set_api_key("file-key-000")
    monkeypatch.setenv("LOCALM_API_KEY", "env-key-999")
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(kwargs)
        return _FakeResp()

    monkeypatch.setattr(requests, "request", fake_request)
    self_request("GET", "/health", base_url="http://127.0.0.1:8642/v1")
    assert captured["headers"]["Authorization"] == "Bearer env-key-999"


def test_no_auth_header_in_open_mode(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(kwargs)
        return _FakeResp()

    monkeypatch.setattr(requests, "request", fake_request)
    self_request("GET", "/health", base_url="http://127.0.0.1:8642/v1")
    assert "Authorization" not in captured["headers"]


def test_open_mode_uses_instance_token_when_no_key_configured(monkeypatch):
    """An open (keyless) server's own self-call (the chat<->media VRAM handover)
    must carry an Authorization header, or http_server.py's open-mode management
    gate 403s it. Mirrors read_activity's instance_token fallback."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(kwargs)
        return _FakeResp()

    monkeypatch.setattr(requests, "request", fake_request)
    self_request("POST", "/models/unload", base_url="http://127.0.0.1:8642/v1",
                instance_token="inst-token-abc")
    assert captured["headers"]["Authorization"] == "Bearer inst-token-abc"


def test_owner_key_still_wins_over_instance_token(monkeypatch):
    """A protected-mode server (an owner key configured) must keep using the
    real key even when a caller also passes instance_token - the token is a
    keyless-mode fallback only, never a second credential that could shadow
    the real one."""
    monkeypatch.setenv("LOCALM_API_KEY", "sekret123")
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(kwargs)
        return _FakeResp()

    monkeypatch.setattr(requests, "request", fake_request)
    self_request("POST", "/models/unload", base_url="http://127.0.0.1:8642/v1",
                instance_token="inst-token-abc")
    assert captured["headers"]["Authorization"] == "Bearer sekret123"


def test_passes_json_payload_through(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(kwargs)
        return _FakeResp()

    monkeypatch.setattr(requests, "request", fake_request)
    payload = {"input": ["a", "b"], "model": "localm"}
    self_request("POST", "/embeddings", json=payload,
                base_url="http://127.0.0.1:8642/v1")
    assert captured["json"] is payload


def test_verify_resolved_via_tls_module_for_loopback_https(monkeypatch, tmp_path):
    """A loopback HTTPS self-call must trust this install's own local CA (not
    plain True/system trust), resolved via localm.tls.requests_verify."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    ca = tmp_path / "ca.crt"
    ca.write_text("fake-ca")

    from localm import tls as _tls
    monkeypatch.setattr(_tls, "ca_cert_path", lambda home: ca)
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)

    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(kwargs)
        return _FakeResp()

    monkeypatch.setattr(requests, "request", fake_request)
    self_request("GET", "/health", base_url="https://127.0.0.1:8642/v1")
    assert captured["verify"] == str(ca)


def test_a_request_to_this_machines_own_server_never_goes_through_a_proxy(monkeypatch):
    """With a proxy in the environment and no ``no_proxy``, a call to a loopback address
    (``self_request``, ``read_activity`` and the restart of another server for an update,
    ``updater_handoff.restart_servers``) still reaches the server on this machine; any
    other address keeps the proxies of the environment."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from pathlib import Path

    from localm import selfclient, updater_handoff

    asked = []

    class Answer(BaseHTTPRequestHandler):
        def _answer(self):
            asked.append((self.command, self.path))
            body = json.dumps({"operations": []}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = _answer

        def log_message(self, *_a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Answer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    proxy = "http://192.0.2.1:9"
    for name in ("http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, proxy)
        monkeypatch.setenv(name.upper(), proxy)
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOCALM_API_KEY", "sekret123")
    try:
        for host in ("127.0.0.1", "localhost"):
            reply = self_request("GET", "/health", timeout=3,
                                 base_url=f"http://{host}:{port}/v1")
            assert reply.status_code == 200
        assert selfclient.read_activity("http", port) == ("ok", {"operations": []})
        prepared = updater_handoff.Prepared(
            Path("."), "0" * 32, "1", "2", None, True, None, False, None,
            servers={"9-9": {"url": f"http://127.0.0.1:{port}/v1", "token": None}})
        assert updater_handoff.restart_servers(prepared, say=lambda _t: None) == []
    finally:
        server.shutdown()
        server.server_close()
    assert ("POST", "/v1/server/restart") in asked
    for url in ("http://127.0.0.1:1/", "http://[::1]:1/", "https://localhost/",
                "http://127.5.6.7/"):
        assert selfclient._proxies(url) == {"http": None, "https": None, "all": None}
    assert selfclient._proxies("http://192.0.2.5:8/") is None
    assert selfclient._proxies("http://localhost.example/") is None
