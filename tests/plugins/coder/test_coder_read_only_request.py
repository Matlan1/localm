# SPDX-License-Identifier: AGPL-3.0-or-later
"""A read-only request ("summarise it, do not edit anything") sent through the
real `localm coder --no-server -u` command to a loopback OpenAI-compatible
server is answered and reported finished, with no request to write a file."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from click.testing import CliRunner

from localm.plugins.coder.cli import _main

_READ_CALL = ('<tool_call>\n{"name": "read_file", "args": {"path": "m00.py"}}\n'
              "</tool_call>")
_SUMMARY = "Summary: m00.py defines one function, add, that returns a + b."


@pytest.fixture
def proj(tmp_path, monkeypatch):
    """An active coder plugin under a throwaway HOME, and a one-file project."""
    home = tmp_path / "home"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    PluginManager(None).set_installed_state("coder", True)
    p = tmp_path / "proj"
    p.mkdir()
    (p / "m00.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return p


@pytest.fixture
def stub_server():
    """A loopback chat server that answers with a read_file call, then a
    summary; yields its base URL and the list of chat request bodies."""
    replies = [_READ_CALL, _SUMMARY]
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):                                     # noqa: N802
            self._send_json({"data": [{"id": "plain-model"}]})

        def do_POST(self):                                    # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            requests.append(body)
            text = replies[min(len(requests) - 1, len(replies) - 1)]
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for delta, finish in (({"content": text}, None), ({}, "stop")):
                    chunk = {"choices": [{"index": 0, "delta": delta,
                                          "finish_reason": finish}]}
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                return
            self._send_json({"choices": [{"index": 0, "finish_reason": "stop",
                                          "message": {"role": "assistant",
                                                      "content": text}}]})

        def _send_json(self, obj):
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1", requests
    finally:
        srv.shutdown()
        srv.server_close()


def _last_user_text(body: dict) -> str:
    users = [m for m in body.get("messages", []) if m.get("role") == "user"]
    return str(users[-1].get("content", "")) if users else ""


def test_a_do_not_edit_summary_request_ends_finished_with_no_write_nudge(proj, stub_server):
    url, requests = stub_server
    res = CliRunner().invoke(_main.main, [
        "--no-server", "-u", url, "-m", "plain-model", "-y", "--cwd", str(proj),
        "--output-format", "jsonl",
        "Read m00.py with read_file and summarise what it contains. Do not edit anything."])
    assert res.exit_code == 0, res.output + res.stderr
    assert not any("[nothing written]" in _last_user_text(b) for b in requests)
    assert len(requests) == 2
    final = [json.loads(ln) for ln in res.stdout.splitlines()][-1]
    assert final["type"] == "final"
    assert final["response"].startswith(_SUMMARY)
    assert "[not finished" not in final["response"]
    assert final["success"] is True
    assert (proj / "m00.py").read_text(encoding="utf-8") == "def add(a, b):\n    return a + b\n"
