# SPDX-License-Identifier: AGPL-3.0-or-later
"""A coder agent that runs inside the server (a GUI session or a scheduled job) and
its sub-agents fail a turn on a 401 from the backend instead of asking for a key on
the server's console; the terminal coder and its sub-agents still ask."""

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
    agent = _agent(tmp_path, url, on_event=lambda e: None,
                   confirm_handler=lambda *a, **k: True, server_side=True)

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


def _child(parent, tmp_path: Path, url: str):
    from localm.plugins.coder.agent import Agent
    from localm.plugins.coder.tools.agents import (
        _inherited_confirm_handler, inherited_child_kwargs)
    backend = HTTPBackend(url, "other-model", api_key="wrong-key")
    with patch("localm.plugins.coder.agent.ProjectMap") as pm, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        pm.build.return_value.file_count.return_value = 0
        return Agent(**inherited_child_kwargs(
            parent, backend=backend, cwd=tmp_path, name="child", max_turns=3,
            confirm_handler=_inherited_confirm_handler(parent)))


def test_a_terminal_sub_agent_still_asks_for_a_key(tmp_path, stub, console):
    url, hits = stub
    parent = _agent(tmp_path, url)
    parent._interactive = True
    child = _child(parent, tmp_path, url)
    assert child.confirm_handler is not None

    result = child._call_llm([{"role": "user", "content": "hi"}], interactive=False)

    assert result == "ok"
    assert console == ["Enter API key: "]
    assert hits == ["Bearer wrong-key", f"Bearer {GOOD_KEY}"]


def test_a_server_session_sub_agent_fails_the_turn_without_prompting(tmp_path, stub, console):
    url, hits = stub
    parent = _agent(tmp_path, url, on_event=lambda e: None,
                    confirm_handler=lambda *a, **k: True, server_side=True)
    child = _child(parent, tmp_path, url)

    with pytest.raises(CoderAuthError):
        child._call_llm([{"role": "user", "content": "hi"}], interactive=False)

    assert console == []
    assert len(hits) == 1


def test_a_scheduled_coder_job_fails_on_a_401_and_the_next_job_still_runs(
        tmp_path, stub, console, monkeypatch):
    url, hits = stub
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "home"))
    from localm.plugins.builtin.jobs import runner
    from localm.plugins.builtin.jobs.store import Job

    keys = iter(["wrong-key", GOOD_KEY])
    monkeypatch.setattr(
        runner, "_coder_backend",
        lambda job: HTTPBackend(url, "remote-model", api_key=next(keys)))

    def job():
        return Job(name="j", task_kind="coder", prompt="hi", cwd=str(tmp_path),
                   schedule_kind="interval", schedule=60)

    first = runner.run_job(job())
    second = runner.run_job(job())

    assert first["status"] == "error"
    assert "CoderAuthError" in first["error"]
    assert console == []
    assert second["status"] == "ok"
    assert hits == ["Bearer wrong-key", f"Bearer {GOOD_KEY}"]
