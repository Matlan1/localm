# SPDX-License-Identifier: AGPL-3.0-or-later
"""A coder agent that runs inside the server (it has a confirm_handler) fails a
turn on a 401 from its backend instead of asking for a key on the server's
console; the terminal coder (no confirm_handler) still asks."""

from __future__ import annotations

import getpass
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from localm.plugins.coder.backends.http import CoderAuthError, HTTPBackend

GOOD_KEY = "good-key"


@pytest.fixture
def stub():
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):                                    # noqa: N802
            self.rfile.read(int(self.headers["Content-Length"]))
            auth = self.headers.get("Authorization", "")
            hits.append(auth)
            if auth != f"Bearer {GOOD_KEY}":
                payload = json.dumps({"detail": "bad key"}).encode()
                self.send_response(401)
            else:
                payload = json.dumps({
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1", hits
    srv.shutdown()
    srv.server_close()


def _agent(tmp_path: Path, url: str, **kw):
    from localm.plugins.coder.agent import Agent
    backend = HTTPBackend(url, "remote-model", api_key="wrong-key")
    with patch("localm.plugins.coder.agent.ProjectMap") as pm, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        pm.build.return_value.file_count.return_value = 0
        agent = Agent(backend=backend, cwd=tmp_path, **kw)
    return agent


@pytest.fixture
def console(monkeypatch):
    prompts: list[str] = []

    def fake_getpass(prompt=""):
        prompts.append(prompt)
        return GOOD_KEY

    monkeypatch.setattr(getpass, "getpass", fake_getpass)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.delenv("CI", raising=False)
    return prompts


def test_a_server_session_fails_the_turn_without_prompting(tmp_path, stub, console):
    url, hits = stub
    agent = _agent(tmp_path, url, on_event=lambda e: None, confirm_handler=lambda *a, **k: True)

    with pytest.raises(CoderAuthError) as exc:
        agent._call_llm([{"role": "user", "content": "hi"}], interactive=False)

    assert console == []
    assert len(hits) == 1
    assert url in str(exc.value)
    assert "remote-model" in str(exc.value)


def test_the_terminal_coder_still_asks_for_a_key(tmp_path, stub, console):
    url, hits = stub
    agent = _agent(tmp_path, url)

    result = agent._call_llm([{"role": "user", "content": "hi"}], interactive=False)

    assert result == "ok"
    assert console == ["Enter API key: "]
    assert hits == ["Bearer wrong-key", f"Bearer {GOOD_KEY}"]
