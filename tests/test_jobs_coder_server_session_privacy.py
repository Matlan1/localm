# SPDX-License-Identifier: AGPL-3.0-or-later
"""A scheduled coder job run through the real jobs routes on a real server
records its exchange with that server no less privately than the job's own
session mode: with the global mode full and the project's
``.localcoder/config.toml`` set to ``mode = "privacy"``, nothing of the run
(prompt, tool call, file content, answer) reaches any file under the data
folder other than the job's own result. The inverse (global privacy, project
full) is the control."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import uvicorn
from fastapi.testclient import TestClient

PROMPT_MARK = "PRV10PROMPTMARK"
FILE_MARK = "PRV10FILECONTENTMARK"
ANSWER_MARK = "PRV10ANSWERMARK"
MARKS = (PROMPT_MARK, FILE_MARK, ANSWER_MARK)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    (h / "models").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    for var in ("LOCALM_MODE", "LOCALM_API_KEY", "LOCALM_REQUIRE_AUTH",
                "LOCALM_SELF_URL"):
        monkeypatch.delenv(var, raising=False)
    import localm.audit as audit
    import localm.config as cfg
    import localm.model_manager as mm
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    monkeypatch.setattr(mm, "MODELS_DIR", h / "models")
    monkeypatch.setattr(audit, "_SESSIONS_DIR", h / "sessions")
    return h


def _install_jobs(home: Path) -> None:
    from fastapi import FastAPI

    from localm.plugins.engine import PluginManager
    store_root = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "builtin"
    PluginManager(FastAPI(), store_root=store_root,
                  installed_root=home / "plugins").install("jobs")


def _scripted_engine(calls: list):
    """An engine whose first reply reads notes.txt and whose later replies
    answer with ANSWER_MARK."""
    def _stream(messages, **kw):
        calls.append(messages)
        seen = json.dumps(messages, default=str)
        if FILE_MARK in seen:
            text = f"The notes say {ANSWER_MARK}."
        else:
            text = ('<tool_call>\n' + json.dumps(
                {"name": "read_file", "args": {"path": "notes.txt"}})
                + '\n</tool_call>')
        yield text

    engine = MagicMock()
    engine.display_name = "test-model"
    type(engine).loaded = property(lambda self: True)
    engine.chat_stream.side_effect = _stream
    engine.count_tokens.side_effect = lambda text: max(1, len(str(text).split()))
    return engine


def _start(app):
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    port = lsock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
    th = threading.Thread(target=lambda: asyncio.run(server.serve(sockets=[lsock])),
                          daemon=True)
    th.start()
    deadline = time.time() + 15.0
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn did not start"
    return port, server, th


def _files_with_marks(root: Path, skip: tuple[Path, ...]) -> dict:
    hits: dict = {}
    for p in root.rglob("*"):
        if not p.is_file() or any(s in p.parents for s in skip):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        found = [m for m in MARKS if m in text]
        if found:
            hits[str(p.relative_to(root))] = found
    return hits


def _run_job(home: Path, tmp_path: Path, monkeypatch, *, global_mode: str,
             project_mode: str):
    (home / "config.json").write_text(json.dumps({"mode": global_mode}),
                                      encoding="utf-8")
    proj = tmp_path / "proj"
    (proj / ".localcoder").mkdir(parents=True)
    (proj / ".localcoder" / "config.toml").write_text(
        f'mode = "{project_mode}"\n', encoding="utf-8")
    (proj / "notes.txt").write_text(f"{FILE_MARK}\n", encoding="utf-8")
    _install_jobs(home)

    calls: list = []
    from localm.inference.http_server import create_app
    app = create_app(_scripted_engine(calls))
    port, server, th = _start(app)
    monkeypatch.setenv("LOCALM_SELF_URL", f"http://127.0.0.1:{port}/v1")
    try:
        with TestClient(app) as c:
            c.headers["Authorization"] = f"Bearer {app.state.shell_token}"
            r = c.post("/api/jobs", json={
                "name": "prv10", "task_kind": "coder",
                "prompt": f"Read notes.txt and tell me what it says. {PROMPT_MARK}",
                "cwd": str(proj), "schedule_kind": "interval", "schedule": 3600})
            assert r.status_code == 200, r.text
            jid = r.json()["id"]
            r = c.post(f"/api/jobs/{jid}/run")
            assert r.status_code == 200, r.text
            result = r.json()
    finally:
        server.should_exit = True
        th.join(timeout=10.0)
    from localm.plugins.builtin.jobs.store import jobs_dir
    return result, calls, _files_with_marks(home, (jobs_dir().resolve(),
                                                   jobs_dir()))


def test_a_privacy_project_keeps_a_job_out_of_a_full_server_session_log(
        home, tmp_path, monkeypatch):
    result, calls, hits = _run_job(home, tmp_path, monkeypatch,
                                   global_mode="full", project_mode="privacy")

    assert result["status"] == "ok", result
    assert ANSWER_MARK in result["output"]
    assert any(FILE_MARK in json.dumps(m, default=str) for m in calls), \
        "the run never sent the file content to the server"
    assert hits == {}
    assert not list((home / "sessions").glob("*_server.md"))
    records = [json.loads(line)
               for p in sorted((home / "sessions").glob("*_server.jsonl"))
               for line in p.read_text(encoding="utf-8").splitlines() if line]
    assert {r["type"] for r in records} <= {"system"}, records


def test_a_full_project_under_a_privacy_server_persists_nothing_either(
        home, tmp_path, monkeypatch):
    result, calls, hits = _run_job(home, tmp_path, monkeypatch,
                                   global_mode="privacy", project_mode="full")

    assert result["status"] == "ok", result
    assert ANSWER_MARK in result["output"]
    assert any(FILE_MARK in json.dumps(m, default=str) for m in calls)
    assert hits == {}


def test_without_a_project_pin_a_full_server_records_the_job(
        home, tmp_path, monkeypatch):
    result, calls, hits = _run_job(home, tmp_path, monkeypatch,
                                   global_mode="full", project_mode="full")

    assert result["status"] == "ok", result
    server_hits = {k: v for k, v in hits.items() if "_server." in k}
    assert server_hits, hits
    assert any(FILE_MARK in v for v in server_hits.values())


@pytest.mark.parametrize("current, mode, expected", [
    (None, "privacy", "privacy"),
    (None, "full", "full"),
    ("full", "privacy", "privacy"),
    ("privacy", "full", "privacy"),
    ("log", "full", "log"),
    ("full", "log", "log"),
    ("loud", "full", "privacy"),
    (None, "loud", "privacy"),
])
def test_cap_backend_session_mode_only_tightens(current, mode, expected):
    from types import SimpleNamespace

    from localm.audit import SessionMode
    from localm.plugins.coder.agent.core import cap_backend_session_mode
    backend = SimpleNamespace(session_mode=current)
    cap_backend_session_mode(backend, SessionMode(mode) if mode != "loud" else mode)
    assert backend.session_mode == expected


def test_cap_backend_session_mode_leaves_a_backend_without_the_attribute():
    from localm.audit import SessionMode
    from localm.plugins.coder.agent.core import cap_backend_session_mode

    class _Plain:
        pass

    backend = _Plain()
    cap_backend_session_mode(backend, SessionMode.PRIVACY)
    assert not hasattr(backend, "session_mode")


def test_an_agent_sends_its_own_mode_to_a_localm_server(home, tmp_path):
    from localm.audit import SESSION_MODE_HEADER, SessionMode
    from localm.plugins.coder.agent import Agent
    from localm.plugins.coder.backends.http import HTTPBackend
    backend = HTTPBackend("http://127.0.0.1:9/v1", model="m", localm_server=True)
    backend.session_mode = "full"
    agent = Agent(backend, tmp_path, mode=SessionMode.PRIVACY, server_side=True)
    try:
        assert backend._headers()[SESSION_MODE_HEADER] == "privacy"
    finally:
        agent.close()
