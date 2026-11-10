# SPDX-License-Identifier: AGPL-3.0-or-later
"""LOCALM_URL is read the same way by every command: a server root or a /v1
base, with or without a trailing slash. A media unload that a server refuses
prints one notice naming the status. A LOCALM_URL that is this home's own
registered instance authenticates with that instance's attach token."""

from __future__ import annotations

import http.server
import json
import threading

import pytest

from localm import localm_url
from localm.media import comfy_client


class _Handler(http.server.BaseHTTPRequestHandler):
    seen: list = []
    auth: list = []
    want_token = "inst-tok"

    def do_POST(self):
        type(self).seen.append(self.path)
        type(self).auth.append(self.headers.get("Authorization"))
        if not self.path.startswith("/v1/models/unload"):
            code, body = 404, {"detail": "Not Found"}
        elif self.headers.get("Authorization") != f"Bearer {self.want_token}":
            code, body = 403, {"detail": "Forbidden"}
        else:
            code, body = 200, {"status": "unloaded", "model": "m",
                               "unloaded_models": ["m"], "unloaded_other": [],
                               "embedder_unloaded": False}
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


@pytest.fixture
def server(monkeypatch):
    _Handler.seen, _Handler.auth = [], []
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr("localm.auth.get_api_key", lambda: None)
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.mark.parametrize("given,root,v1", [
    ("http://h:1", "http://h:1", "http://h:1/v1"),
    ("http://h:1/", "http://h:1", "http://h:1/v1"),
    ("http://h:1/v1", "http://h:1", "http://h:1/v1"),
    ("http://h:1/v1/", "http://h:1", "http://h:1/v1"),
    (" http://h:1/v1 ", "http://h:1", "http://h:1/v1"),
    ("", None, None), (None, None, None), ("  ", None, None), ("/v1", None, None)])
def test_both_forms_resolve_to_one_root(given, root, v1):
    assert localm_url.server_root(given) == root
    assert localm_url.v1_base(given) == v1
    assert comfy_client.localm_v1_base(given) == v1


@pytest.mark.parametrize("suffix", ["", "/", "/v1", "/v1/"])
def test_env_root_strips_the_v1_suffix(monkeypatch, suffix):
    monkeypatch.setenv("LOCALM_URL", "http://127.0.0.1:9" + suffix)
    assert localm_url.env_root() == "http://127.0.0.1:9"


def test_env_root_unset_is_empty(monkeypatch):
    monkeypatch.delenv("LOCALM_URL", raising=False)
    assert localm_url.env_root() == ""


@pytest.mark.parametrize("suffix", ["", "/v1", "/v1/"])
def test_cli_unload_with_a_v1_localm_url_reaches_the_route(server, monkeypatch, suffix):
    from click.testing import CliRunner

    from localm.cli import main
    monkeypatch.setattr(localm_url, "registry_token", lambda url, home=None: "inst-tok")
    monkeypatch.setenv("LOCALM_URL", f"http://127.0.0.1:{server.server_port}{suffix}")
    res = CliRunner().invoke(main, ["unload"])
    assert _Handler.seen == ["/v1/models/unload"], res.output
    assert res.exit_code == 0, res.output


def test_refused_media_unload_names_the_status(server):
    lines = []
    out = comfy_client._localm_unload(f"http://127.0.0.1:{server.server_port}", None,
                                      media=False, on_failure=lines.append,
                                      label="video")
    assert out is None
    assert _Handler.seen == ["/v1/models/unload?media=false"]
    assert lines == [comfy_client.unload_failed_notice("HTTP 403", "video")]
    assert "HTTP 403" in lines[0] and "LOCALM_API_KEY" in lines[0]


def test_unreachable_media_unload_names_the_error():
    lines = []
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_port
    srv.server_close()
    assert comfy_client._localm_unload(f"http://127.0.0.1:{port}", "t",
                                       on_failure=lines.append) is None
    assert len(lines) == 1 and lines[0].startswith("Could not unload the chat model (")


def test_unset_url_reports_nothing(monkeypatch):
    monkeypatch.delenv("LOCALM_URL", raising=False)
    lines = []
    assert comfy_client._localm_unload(None, on_failure=lines.append) is None
    assert lines == []


def test_media_unload_uses_the_registry_token_for_its_own_address(server, monkeypatch):
    port = server.server_port
    seen = []
    monkeypatch.setattr("localm.instances.list_entries", lambda home: [
        {"instance_id": "x1", "host": "127.0.0.1", "port": port + 1, "scheme": "http",
         "token": "other-tok"},
        {"instance_id": "x2", "host": "127.0.0.1", "port": port, "scheme": "http",
         "token": "inst-tok"}])
    monkeypatch.setattr("localm.instances.default_probe",
                        lambda entry, **k: seen.append(entry["instance_id"]) or True)
    lines = []
    out = comfy_client._localm_unload(f"http://127.0.0.1:{port}/v1", None,
                                      on_failure=lines.append)
    assert out and out["status"] == "unloaded", lines
    assert _Handler.auth == ["Bearer inst-tok"]
    assert seen == ["x2"]


def test_registry_token_needs_the_identity_handshake(monkeypatch):
    monkeypatch.setattr("localm.instances.list_entries", lambda home: [
        {"instance_id": "x", "host": "127.0.0.1", "port": 9, "scheme": "http",
         "token": "tok"}])
    monkeypatch.setattr("localm.instances.default_probe", lambda entry, **k: False)
    assert localm_url.registry_token("http://127.0.0.1:9", home=".") is None
    monkeypatch.setattr("localm.instances.default_probe", lambda entry, **k: True)
    assert localm_url.registry_token("http://127.0.0.1:9/v1/", home=".") == "tok"
    assert localm_url.registry_token("http://127.0.0.1:10", home=".") is None
    assert localm_url.registry_token("https://127.0.0.1:9", home=".") is None
