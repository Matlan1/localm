# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for HTTP server hardening: CORS lockdown and exposed-bind warning."""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from localm.cli import _exposed_bind_warning
from localm.inference.http_server import create_app


def _make_engine():
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.count_tokens.return_value = 5
    engine.chat_stream.side_effect = lambda messages, **kw: iter(["ok"])
    type(engine).loaded = property(lambda self: True)
    return engine


def _client(cors_cfg=None):
    os.environ.pop("LOCALM_API_KEY", None)
    with patch(
        "localm.config.load_config",
        return_value={"cors_origins": cors_cfg},
    ):
        app = create_app(_make_engine())
    return TestClient(app)


CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


class TestCorsLockdown:
    def test_foreign_origin_not_allowed_by_default(self):
        with _client() as c:
            r = c.post("/v1/chat/completions", json=CHAT,
                       headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in r.headers

    @pytest.mark.parametrize("origin", ["http://localhost:5173", "http://127.0.0.1:8642"])
    def test_default_local_origin_allowed(self, origin):
        with _client() as c:
            r = c.post("/v1/chat/completions", json=CHAT,
                       headers={"Origin": origin})
        assert r.headers.get("access-control-allow-origin") == origin

    def test_wildcard_opt_in_allows_everything(self):
        with _client(cors_cfg="*") as c:
            r = c.post("/v1/chat/completions", json=CHAT,
                       headers={"Origin": "https://evil.example"})
        assert r.headers.get("access-control-allow-origin") == "*"

    def test_explicit_origin_list_respected(self):
        with _client(cors_cfg=["https://app.example"]) as c:
            allowed = c.post("/v1/chat/completions", json=CHAT,
                             headers={"Origin": "https://app.example"})
            denied = c.post("/v1/chat/completions", json=CHAT,
                            headers={"Origin": "https://other.example"})
        assert allowed.headers.get("access-control-allow-origin") == "https://app.example"
        assert "access-control-allow-origin" not in denied.headers

    def test_lookalike_host_not_matched_by_regex(self):
        """localhost.evil.example must not slip through the default regex."""
        with _client() as c:
            r = c.post("/v1/chat/completions", json=CHAT,
                       headers={"Origin": "http://localhost.evil.example"})
        assert "access-control-allow-origin" not in r.headers


class TestExposedBindWarning:
    def test_loopback_hosts_are_silent(self, monkeypatch):
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        for host in ("127.0.0.1", "localhost", "::1"):
            assert _exposed_bind_warning(host) is None

    def test_lan_bind_without_key_warns(self, monkeypatch):
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        warning = _exposed_bind_warning("0.0.0.0")
        assert warning is not None
        assert "LOCALM_API_KEY" in warning

    def test_lan_bind_with_key_is_silent(self, monkeypatch):
        # A strong key (>= MIN_KEY_LEN) silences the warning.
        monkeypatch.setenv("LOCALM_API_KEY", "a-strong-secret")
        assert _exposed_bind_warning("0.0.0.0") is None

    def test_lan_bind_with_weak_key_warns(self, monkeypatch):
        # A too-short key (via the env var, bypassing the set-time floor) does
        # NOT silence the warning.
        monkeypatch.setenv("LOCALM_API_KEY", "1")
        w = _exposed_bind_warning("0.0.0.0")
        assert w is not None and "WEAK" in w
