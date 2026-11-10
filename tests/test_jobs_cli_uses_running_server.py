# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm job run`` hands the run to the localm server already running for
the data folder and runs it in-process only when no server has the jobs route."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


class _FakeServer:
    """A real HTTP server on 127.0.0.1 that answers every POST with a fixed
    status and JSON body, and records each request's path and headers."""

    def __init__(self, status: int, body: dict):
        self.requests: list[tuple[str, dict]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                outer.requests.append((self.path, dict(self.headers)))
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _setup(home, monkeypatch, status, body):
    from localm import instances
    from localm.plugins.builtin.jobs import cli
    from localm.plugins.builtin.jobs.store import Job, JobStore
    import localm.plugins.builtin.jobs.runner as runner

    store = JobStore(home / "jobs")
    job = store.add(Job(name="t", task_kind="chat", prompt="hi",
                        schedule_kind="interval", schedule=3600))
    monkeypatch.setattr(cli, "_store", lambda: store)
    local_runs = []
    monkeypatch.setattr(runner, "run_job", lambda j, engine=None: local_runs.append(j) or {
        "status": "ok", "output": "ran locally", "error": None,
        "started": 0, "finished": 0})
    srv = _FakeServer(status, body)
    entry = {"scheme": "http", "host": "127.0.0.1", "port": srv.port,
             "token": "instance-tok"}
    seen = {}

    def fake_find(h, root_dir, *a, **k):
        seen["home"] = h
        return entry
    monkeypatch.setattr(instances, "find_attachable", fake_find)
    return cli, store, job, srv, local_runs, seen


def test_job_run_posts_to_the_running_server_and_loads_nothing_locally(home, monkeypatch):
    from click.testing import CliRunner
    cli, store, job, srv, local_runs, seen = _setup(
        home, monkeypatch, 200,
        {"result_id": "r1", "status": "ok", "output": "from the server",
         "error": None})
    try:
        res = CliRunner().invoke(cli.main, ["run", job.id])
    finally:
        srv.close()
    assert local_runs == []
    assert res.exit_code == 0, res.output
    assert "from the server" in res.output
    assert [p for p, _ in srv.requests] == [f"/api/jobs/{job.id}/run"]
    headers = {k.lower(): v for k, v in srv.requests[0][1].items()}
    assert headers.get("authorization") == "Bearer instance-tok"
    assert seen["home"] == home
    assert store.get(job.id).claim is None


def test_job_run_reports_a_busy_server_and_does_not_run_locally(home, monkeypatch):
    from click.testing import CliRunner
    cli, store, job, srv, local_runs, _ = _setup(
        home, monkeypatch, 409,
        {"detail": "A job run is already in progress; try again once it finishes."})
    try:
        res = CliRunner().invoke(cli.main, ["run", job.id])
    finally:
        srv.close()
    assert local_runs == []
    assert res.exit_code == 1
    assert "Another job is running in the localm server" in res.output
    assert store.get(job.id).claim is None


def test_job_run_reports_a_failed_server_run(home, monkeypatch):
    from click.testing import CliRunner
    cli, store, job, srv, local_runs, _ = _setup(
        home, monkeypatch, 200,
        {"result_id": "r1", "status": "error", "output": "", "error": "boom"})
    try:
        res = CliRunner().invoke(cli.main, ["run", job.id])
    finally:
        srv.close()
    assert local_runs == []
    assert res.exit_code == 1
    assert "Job failed: boom" in res.output


def test_job_run_falls_back_to_this_process_without_a_jobs_route(home, monkeypatch):
    from click.testing import CliRunner
    cli, store, job, srv, local_runs, _ = _setup(
        home, monkeypatch, 404, {"detail": "Not Found"})
    try:
        res = CliRunner().invoke(cli.main, ["run", job.id])
    finally:
        srv.close()
    assert len(local_runs) == 1
    assert res.exit_code == 0, res.output
    assert "ran locally" in res.output


def test_job_run_without_a_server_runs_in_this_process(home, monkeypatch):
    from click.testing import CliRunner

    from localm import instances
    from localm.plugins.builtin.jobs import cli
    from localm.plugins.builtin.jobs.store import Job, JobStore
    import localm.plugins.builtin.jobs.runner as runner

    store = JobStore(home / "jobs")
    job = store.add(Job(name="t", task_kind="chat", prompt="hi",
                        schedule_kind="interval", schedule=3600))
    monkeypatch.setattr(cli, "_store", lambda: store)
    calls = []
    monkeypatch.setattr(runner, "run_job", lambda j, engine=None: calls.append(engine) or {
        "status": "ok", "output": "x", "error": None, "started": 0, "finished": 0})
    monkeypatch.setattr(instances, "find_attachable", lambda *a, **k: None)
    res = CliRunner().invoke(cli.main, ["run", job.id])
    assert res.exit_code == 0, res.output
    assert calls == [None]
