# SPDX-License-Identifier: AGPL-3.0-or-later
"""A chat-surface privacy setting (``chat_mode = privacy`` under ``mode = log``)
reaches media generated from the GUI chat and the bug report of a failed
``localm run``.

The GUI chat's media commands send ``X-Client-Id: chat``; the request-privacy
middleware runs such a request inside ``audit.request_privacy``, and every
media job carries that into its worker thread."""

import io
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def chat_private(monkeypatch):
    from localm.config import update_config
    monkeypatch.delenv("LOCALM_MODE", raising=False)
    monkeypatch.delenv("LOCALM_KEEP_DIAGNOSTICS", raising=False)
    update_config(lambda c: c.update(mode="log", chat_mode="privacy",
                                     keep_diagnostics=False))


def test_media_session_is_private_follows_request_privacy(chat_private):
    from localm.audit import request_privacy
    from localm.plugins.media_config import media_session_is_private
    assert media_session_is_private() is False
    with request_privacy(True):
        assert media_session_is_private() is True


def test_media_session_is_private_in_server_privacy_mode(monkeypatch):
    from localm.plugins.media_config import media_session_is_private
    monkeypatch.setenv("LOCALM_MODE", "privacy")
    assert media_session_is_private() is True


def test_diagnostics_are_off_inside_a_private_request(chat_private):
    from localm.audit import diagnostics_allowed, request_privacy
    assert diagnostics_allowed() is True
    with request_privacy(True):
        assert diagnostics_allowed() is False


def test_keep_diagnostics_still_wins_inside_a_private_request(chat_private, monkeypatch):
    from localm.audit import diagnostics_allowed, request_privacy
    monkeypatch.setenv("LOCALM_KEEP_DIAGNOSTICS", "1")
    with request_privacy(True):
        assert diagnostics_allowed() is True


def _png_writer(calls):
    def fake_generate_image(prompt, out, **kwargs):
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, "PNG")
        Path(out).write_bytes(buf.getvalue())
        calls.append(kwargs)
        return True, f"Image saved to {out}"
    return fake_generate_image


@pytest.fixture
def server(monkeypatch, tmp_path):
    from localm.inference.http_server import create_app
    from localm.plugins.engine import PluginManager
    monkeypatch.setattr("localm.plugins.lookup.is_available", lambda name: True)
    monkeypatch.setattr("localm.media.backend_choice.comfy_is_set_up", lambda *a, **k: True)
    monkeypatch.setattr("localm.image_gen.comfy.ensure_comfy", lambda *a, **k: (True, "up"))
    monkeypatch.setattr("localm.image_gen.comfy.free_comfy_vram", lambda *a, **k: True)
    monkeypatch.setattr("localm.vram.decide_media_swap", lambda s: False)
    monkeypatch.setattr("localm.vram.media_single_device_shortfall", lambda s: None)
    engine = MagicMock()
    engine.display_name = "test-model"
    app = create_app(engine)
    PluginManager(app, external_root=tmp_path / "noplugins").install("image")
    app.state.self_url = "http://127.0.0.1:8642/v1"
    with TestClient(app) as client:
        client.headers["Authorization"] = f"Bearer {app.state.shell_token}"
        yield app, client


def _imagine(app, client, headers):
    calls: list = []
    with patch("localm.image_gen.comfy.generate_image", _png_writer(calls)):
        r = client.post("/api/imagine", json={"prompt": "a private cat"}, headers=headers)
        assert r.status_code == 200, r.text
        job = app.state.jobs.get(r.json()["job_id"])
        deadline = time.monotonic() + 30
        while job.status not in ("done", "failed", "cancelled") and time.monotonic() < deadline:
            time.sleep(0.05)
    assert job.status == "done", job.status
    assert len(calls) == 1
    return calls[0], job


def test_gui_chat_imagine_follows_the_chat_privacy_setting(chat_private, server):
    app, client = server
    kwargs, _job = _imagine(app, client, {"X-Client-Id": "chat"})
    assert kwargs["write_sidecar"] is False
    assert kwargs["delete_outputs"] is True


def test_imagine_without_the_chat_header_follows_the_server_mode(chat_private, server):
    app, client = server
    kwargs, _job = _imagine(app, client, {})
    assert kwargs["write_sidecar"] is True


def _activity_text(root=None):
    from localm.jobmanager import activity_dir
    d = Path(root) if root is not None else activity_dir()
    if not d.is_dir():
        return ""
    return "".join(f.read_text(encoding="utf-8") for f in d.glob("*.json"))


def _wait_finished(job):
    deadline = time.monotonic() + 30
    while job.finished_at is None and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.1)
    assert job.finished_at is not None


def test_a_private_chat_job_keeps_no_finished_row(chat_private, server):
    app, client = server
    _kwargs, private_job = _imagine(app, client, {"X-Client-Id": "chat"})
    _wait_finished(private_job)
    assert private_job.id not in _activity_text()
    _kwargs, open_job = _imagine(app, client, {})
    _wait_finished(open_job)
    text = _activity_text()
    assert open_job.id in text
    assert private_job.id not in text


@pytest.fixture
def jobs(tmp_path):
    from localm.jobmanager import JobManager, _ActivityStore
    return JobManager(store=_ActivityStore(tmp_path / "activity"), reconcile=False)


def test_a_job_started_in_a_private_request_runs_private(chat_private, jobs):
    from localm.audit import request_is_privacy, request_privacy
    seen = []

    def body(job):
        seen.append(request_is_privacy())
        return True
    with request_privacy(True):
        job = jobs.start_fn("imagine", body)
    _wait_finished(job)
    assert seen == [True]
    assert job.private is True


def test_a_later_job_does_not_write_a_private_jobs_finished_row(chat_private, jobs,
                                                                tmp_path):
    from localm.audit import request_privacy
    with request_privacy(True):
        private_job = jobs.start_fn("imagine", lambda job: True)
    _wait_finished(private_job)
    open_job = jobs.start_fn("pull", lambda job: True)
    _wait_finished(open_job)
    text = _activity_text(tmp_path / "activity")
    assert open_job.id in text
    assert private_job.id not in text


def test_keep_diagnostics_keeps_a_private_jobs_finished_row(chat_private, jobs,
                                                           tmp_path, monkeypatch):
    from localm.audit import request_privacy
    monkeypatch.setenv("LOCALM_KEEP_DIAGNOSTICS", "1")
    with request_privacy(True):
        private_job = jobs.start_fn("imagine", lambda job: True)
    _wait_finished(private_job)
    assert private_job.id in _activity_text(tmp_path / "activity")


def test_a_private_job_stays_private_across_its_record():
    from localm.jobmanager import Job
    job = Job(id="abc", kind="imagine", argv=[])
    job.private = True
    assert Job.from_record(job.to_record()).private is True
    job.private = False
    assert "private" not in job.to_record()
    assert Job.from_record(job.to_record()).private is False


def test_a_self_request_from_a_private_request_says_privacy(chat_private):
    import http.server
    import threading
    from localm.audit import request_privacy, request_privacy_headers
    from localm.selfclient import self_request
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append((self.headers.get("X-Localm-Session-Mode"),
                         self.headers.get("Authorization")))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
        self_request("POST", "/models/load", base_url=base, timeout=10,
                     headers=request_privacy_headers())
        with request_privacy(True):
            self_request("POST", "/models/load", base_url=base, timeout=10,
                         headers=request_privacy_headers())
        self_request("POST", "/models/load", base_url=base, timeout=10,
                     headers={"Authorization": "Bearer not-the-key"})
    finally:
        srv.shutdown()
        srv.server_close()
    assert seen == [(None, None), ("privacy", None), (None, None)]


def test_a_private_jobs_requests_to_the_server_say_privacy():
    from localm.jobmanager import Job
    from localm.vram import _job_headers
    job = Job(id="abc", kind="imagine", argv=[])
    assert _job_headers(job) == {}
    job.private = True
    assert _job_headers(job) == {"X-Localm-Session-Mode": "privacy"}
    assert _job_headers(MagicMock()) is None


def test_every_self_request_of_vram_and_rag_names_its_headers():
    """Each ``self_request`` call in ``localm/vram.py`` and the RAG plugin
    passes ``headers=``, so a private request's mode reaches the server."""
    import ast
    root = Path(__file__).resolve().parent.parent / "localm"
    found = []
    for rel in ("vram.py", "plugins/builtin/rag/plug.py"):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "self_request"):
                found.append((f"{rel}:{node.lineno}",
                              any(k.arg == "headers" for k in node.keywords)))
    assert len(found) >= 5
    assert [where for where, named in found if not named] == []


def test_a_model_load_inside_a_private_request_runs_private(chat_private, monkeypatch):
    import asyncio
    from localm import audit
    from localm.inference import http_server as hs
    seen = []

    class Engine:
        display_name = "x"

        def load(self):
            seen.append(audit.request_is_privacy())
    monkeypatch.setattr(hs._routing_latch, "fingerprint", lambda name: None)
    monkeypatch.setattr(hs._routing_latch, "record_success", lambda name: None)

    async def go():
        loop = asyncio.get_running_loop()
        await hs._switch_load(loop, "x", Engine(), preempt=False)
        with audit.request_privacy(True):
            await hs._switch_load(loop, "x", Engine(), preempt=False)
    asyncio.run(go())
    assert seen == [False, True]


def test_every_media_job_carries_the_request_privacy():
    """Each ``jobs.start_fn`` in a media plugin route runs its job body through
    ``carry_request_privacy``, so the job thread sees the request's mode."""
    import ast
    root = Path(__file__).resolve().parent.parent / "localm" / "plugins" / "builtin"
    found = []
    for kind in ("image", "video", "music"):
        path = root / kind / "plug.py"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "start_fn"):
                body = node.args[1]
                carried = (isinstance(body, ast.Call) and isinstance(body.func, ast.Name)
                           and body.func.id == "carry_request_privacy")
                found.append((f"{kind}/plug.py:{node.lineno}", carried))
    assert len(found) >= 5
    assert [where for where, carried in found if not carried] == []


def _report_failure(command):
    from localm.cli._core import _GracefulGroup
    ctx = MagicMock()
    ctx.invoked_subcommand = command
    with patch("localm.bugreport.stdin_is_interactive", return_value=False):
        _GracefulGroup._report_failure(ctx, RuntimeError("boom"))


def _saved_reports():
    from localm.config import home_dir
    return list((home_dir() / "bug-reports").glob("bug-*.md"))


def test_a_failed_run_saves_no_report_when_chat_is_private(chat_private):
    _report_failure("run")
    assert _saved_reports() == []


def test_another_failed_command_still_saves_its_report(chat_private):
    _report_failure("pull")
    assert len(_saved_reports()) == 1

def test_a_private_jobs_own_write_keeps_other_jobs_rows(chat_private, jobs, tmp_path):
    from localm.audit import request_privacy
    open_job = jobs.start_fn("pull", lambda job: True)
    _wait_finished(open_job)
    with request_privacy(True):
        private_job = jobs.start_fn("imagine", lambda job: True)
    _wait_finished(private_job)
    text = _activity_text(tmp_path / "activity")
    assert open_job.id in text
    assert private_job.id not in text


def test_a_reload_a_private_job_left_to_a_later_job_says_privacy(monkeypatch):
    from localm import vram
    from localm.jobmanager import Job
    sent = []

    class _Resp:
        ok = True
        status_code = 200

    def self_request(method, path, **kw):
        sent.append((path, kw.get("headers")))
        return _Resp()
    monkeypatch.setattr("localm.selfclient.self_request", self_request)
    monkeypatch.setattr(vram, "_handover_deferred_headers", {})
    monkeypatch.setattr(vram, "_handover_deferred_backends", [])
    monkeypatch.setattr(vram, "_handover_deferred", None)
    backend = MagicMock()
    backend.free_vram.return_value = True
    s = {"reload_after": True}
    private_job = Job(id="a", kind="imagine", argv=[])
    private_job.private = True
    open_job = Job(id="b", kind="imagine", argv=[])
    monkeypatch.setattr(vram, "_handover_jobs", 2)
    vram.reload_chat_after_media(private_job, "http://127.0.0.1:9/v1", s, backend, "image")
    assert sent == []
    monkeypatch.setattr(vram, "_handover_jobs", 1)
    vram.reload_chat_after_media(open_job, "http://127.0.0.1:9/v1", s, backend, "image")
    assert sent == [("/models/load", {"X-Localm-Session-Mode": "privacy"})]
    vram.reload_chat_after_media(open_job, "http://127.0.0.1:9/v1", s, backend, "image")
    assert sent[-1] == ("/models/load", {})

def test_two_jobs_deferring_in_parallel_both_reach_the_reload(monkeypatch):
    import threading
    from localm import vram
    from localm.jobmanager import Job
    sent = []

    class _Resp:
        ok = True
        status_code = 200

    def self_request(method, path, **kw):
        sent.append((path, kw.get("headers")))
        return _Resp()
    monkeypatch.setattr("localm.selfclient.self_request", self_request)
    monkeypatch.setattr(vram, "_job_headers", lambda job: {f"X-Job-{job.id}": "1"})
    monkeypatch.setattr(vram, "_handover_deferred_headers", {})
    monkeypatch.setattr(vram, "_handover_deferred_backends", [])
    monkeypatch.setattr(vram, "_handover_deferred", None)
    backend = MagicMock()
    backend.free_vram.return_value = True
    s = {"reload_after": True}
    monkeypatch.setattr(vram, "_handover_jobs", 3)
    start = threading.Barrier(2)

    def defer(job_id):
        start.wait()
        vram.reload_chat_after_media(Job(id=job_id, kind="imagine", argv=[]),
                                     "http://127.0.0.1:9/v1", s, backend, "image")
    threads = [threading.Thread(target=defer, args=(i,)) for i in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert sent == []
    monkeypatch.setattr(vram, "_handover_jobs", 1)
    vram.reload_chat_after_media(Job(id="c", kind="imagine", argv=[]),
                                 "http://127.0.0.1:9/v1", s, backend, "image")
    assert sent == [("/models/load", {"X-Job-a": "1", "X-Job-b": "1", "X-Job-c": "1"})]
