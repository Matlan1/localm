# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression: the media-gen VRAM unload must authenticate and verify TLS.

The chat model is unloaded before a ComfyUI media gen by POSTing
``/v1/models/unload`` on the local server. That endpoint requires the
models-write scope, so the call must carry the LOCALM_API_KEY bearer token;
and on a built-in-TLS (https) loopback self-call it must trust the install's
own CA. An earlier version used a bare ``urllib`` POST with neither, so the
unload silently 401'd / SSL-failed, the chat model stayed resident, and the
media model loaded on top of it and hung the GPU driver (AMD TDR).
"""

from unittest.mock import MagicMock, patch

from localm.image_gen import comfy


def _resp(ok=True, payload=None):
    r = MagicMock()
    r.ok = ok
    r.json.return_value = payload if payload is not None else {"status": "unloaded"}
    return r


def test_unload_sends_bearer_and_tls_verify(monkeypatch):
    monkeypatch.setenv("LOCALM_API_KEY", "secret-key")
    captured = {}

    def fake_post(url, headers=None, timeout=None, verify=None):
        captured.update(url=url, headers=headers or {}, verify=verify)
        return _resp(payload={"status": "unloaded", "vram_freed": True,
                              "vram_before_bytes": 1, "vram_after_bytes": 2})

    with patch("requests.post", side_effect=fake_post), \
         patch("localm.tls.requests_verify", return_value="/ca/bundle.pem"):
        out = comfy._localm_unload("https://127.0.0.1:8642/v1")

    assert captured["url"] == "https://127.0.0.1:8642/v1/models/unload"
    assert captured["headers"].get("Authorization") == "Bearer secret-key"
    # TLS verification must use the install CA bundle, never be disabled.
    assert captured["verify"] == "/ca/bundle.pem"
    assert out and out["status"] == "unloaded"


def test_unload_no_key_omits_auth_header(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    captured = {}

    def fake_post(url, headers=None, timeout=None, verify=None):
        captured.update(headers=headers or {})
        return _resp()

    with patch("requests.post", side_effect=fake_post), \
         patch("localm.tls.requests_verify", return_value=True):
        comfy._localm_unload("http://127.0.0.1:8642/v1")

    assert "Authorization" not in captured["headers"]


def test_unload_noop_without_url(monkeypatch):
    monkeypatch.delenv("LOCALM_URL", raising=False)
    with patch("requests.post") as post:
        assert comfy._localm_unload(None) is None
        post.assert_not_called()


def test_unload_swallows_failure(monkeypatch):
    monkeypatch.setenv("LOCALM_API_KEY", "k")
    with patch("requests.post", side_effect=RuntimeError("boom")), \
         patch("localm.tls.requests_verify", return_value=True):
        assert comfy._localm_unload("http://127.0.0.1:8642/v1") is None
