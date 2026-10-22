# SPDX-License-Identifier: AGPL-3.0-or-later
"""A client that leaves while a media route waits on its job, on the real
create_app() stack: the route cancels the job and raises CancelledError, and the
request is answered 499 (counted 499 in /metrics) with no ERROR log, while a
route that crashes is still a logged 500. Driven as raw ASGI so the client can
disconnect after sending its body."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import threading

import pytest

from localm.inference import metrics
from localm.inference.app_assembly.transport import ClientLeftMiddleware
from localm.inference.http_server import _DISCONNECT_POLL_KEY, create_app


@pytest.fixture(autouse=True)
def _metrics_off_after():
    yield
    metrics.configure(False)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setenv("LOCALM_TMPDIR", str(tmp_path / "scratch"))
    (tmp_path / "scratch").mkdir()
    monkeypatch.setenv("LOCALM_METRICS", "1")
    for name in ("LOCALM_API_KEY", "LOCALM_REQUIRE_AUTH", "LOCALM_MODE", "LOCALM_DEBUG"):
        monkeypatch.delenv(name, raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _drive(app, method, path, body=b"", *, leave_when=None):
    """Run one request through *app* as ASGI. Once ``leave_when()`` is true,
    every receive after the body reports ``http.disconnect``; until then (and
    always, with no *leave_when*) a receive waits. Returns
    ``(sent messages, exception the app raised or None)``."""
    sent = []

    async def run():
        state = {"body_sent": False}

        async def receive():
            if not state["body_sent"]:
                state["body_sent"] = True
                return {"type": "http.request", "body": body, "more_body": False}
            if leave_when is not None and leave_when():
                return {"type": "http.disconnect"}
            await asyncio.Event().wait()

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": method, "scheme": "http", "path": path,
            "raw_path": path.encode(), "query_string": b"", "root_path": "",
            "headers": [(b"host", b"127.0.0.1:8642"),
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode())],
            "client": ("127.0.0.1", 50123), "server": ("127.0.0.1", 8642),
        }
        try:
            await asyncio.wait_for(app(scope, receive, send), timeout=20)
        except BaseException as exc:  # noqa: BLE001 - the test inspects it
            return exc
        return None

    return sent, asyncio.run(run())


def _status(sent):
    starts = [m for m in sent if m["type"] == "http.response.start"]
    return starts[0]["status"] if starts else None


def _count(route, status):
    m = re.search(r'localm_http_requests_total\{[^}]*route="%s"[^}]*status="%s"[^}]*\} (\d+)'
                  % (re.escape(route), status), metrics.render())
    return int(m.group(1)) if m else 0


def _errors(caplog):
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


# --------------------------------------------------------------------------- #
# The real image route
# --------------------------------------------------------------------------- #

@pytest.fixture
def image_app(home, monkeypatch):
    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.jobs import JobManager
    app = create_app(None)
    PluginManager(app, external_root=home / "noplugins").install("image")
    app.state.jobs = JobManager()
    app.state.self_url = "http://127.0.0.1:8642/v1"
    release = threading.Event()
    started = threading.Event()

    def generate(s, prompt, out_path, **kw):
        started.set()
        release.wait(10)
        return False, "stopped"

    mod = sys.modules["_localm_plugin_image.backend"]
    monkeypatch.setattr(mod, "settings", lambda cfg: {
        "reload_after": True, "warning": "", "api_url": "http://127.0.0.1:8188"})
    monkeypatch.setattr(mod, "ensure_available", lambda s, on_progress=None: (True, "up"))
    monkeypatch.setattr(mod, "generate", generate)
    monkeypatch.setattr("localm.vram.decide_media_swap", lambda s: False)
    monkeypatch.setattr("localm.vram.media_single_device_shortfall", lambda s: None)
    plug = sys.modules["_localm_plugin_image"]
    monkeypatch.setattr(plug, "JOB_POLL_SECONDS", 0.05)
    monkeypatch.setattr(plug, "CANCEL_SETTLE_SECONDS", 0.3)
    app.state.test_started = started
    yield app
    release.set()


def test_a_client_that_leaves_an_image_job_is_a_499_with_no_error_log(image_app, caplog):
    body = json.dumps({"prompt": "a red apple", "response_format": "b64_json"}).encode()
    with caplog.at_level(logging.DEBUG, logger="localm"):
        sent, raised = _drive(image_app, "POST", "/v1/images/generations", body,
                              leave_when=image_app.state.test_started.is_set)
    assert image_app.state.test_started.is_set(), "the job never started"
    assert raised is None, f"the app raised {raised!r}"
    assert _status(sent) == 499
    assert not _errors(caplog), [r.getMessage() for r in _errors(caplog)]
    assert "unhandled error" not in caplog.text
    assert caplog.text.count("the client left before the response") == 1
    assert _count("/v1/images/generations", 499) == 1
    assert _count("/v1/images/generations", 500) == 0


# --------------------------------------------------------------------------- #
# The real speech route
# --------------------------------------------------------------------------- #

@pytest.fixture
def speech_app(home, monkeypatch):
    from localm.inference import speech
    from localm.inference.backends.llamacpp.mtmd_gen import SpeechCancelled
    from localm.plugins.builtin.tts import speech_route
    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.jobs import JobManager
    models = home / "models"
    models.mkdir()
    (models / "voice-1.gguf").write_bytes(b"GGUF")
    (models / "mmproj-voice-1.gguf").write_bytes(b"GGUF")
    (home / "registry.json").write_text(json.dumps({"voice-1": {
        "path": str(models / "voice-1.gguf"), "source": "local", "model_type": "tts",
        "mmproj": str(models / "mmproj-voice-1.gguf")}}), encoding="utf-8")
    started = threading.Event()

    def slow(model, text, *, on_progress=None, should_cancel=None, **kw):
        started.set()
        for _ in range(1000):
            if should_cancel():
                raise SpeechCancelled()
            threading.Event().wait(0.01)
        raise AssertionError("never cancelled")

    monkeypatch.setattr(speech, "synthesize", slow)
    monkeypatch.setattr(speech_route, "_JOB_POLL_SECONDS", 0.05)
    monkeypatch.setattr(speech_route, "_CANCEL_SETTLE_SECONDS", 5.0)
    app = create_app(None)
    PluginManager(app, external_root=home / "noplugins").install("tts")
    if getattr(app.state, "jobs", None) is None:
        app.state.jobs = JobManager()
    app.state.test_started = started
    return app


def test_a_client_that_leaves_a_speech_job_is_a_499_with_no_error_log(speech_app, caplog):
    body = json.dumps({"input": "Hello there."}).encode()
    with caplog.at_level(logging.DEBUG, logger="localm"):
        sent, raised = _drive(speech_app, "POST", "/v1/audio/speech", body,
                              leave_when=speech_app.state.test_started.is_set)
    assert speech_app.state.test_started.is_set(), "the job never started"
    assert raised is None, f"the app raised {raised!r}"
    assert _status(sent) == 499
    assert not _errors(caplog), [r.getMessage() for r in _errors(caplog)]
    assert _count("/v1/audio/speech", 499) == 1
    assert _count("/v1/audio/speech", 500) == 0


# --------------------------------------------------------------------------- #
# A route that crashes is still a logged 500
# --------------------------------------------------------------------------- #

def test_a_route_that_crashes_is_still_a_500_with_an_error_log(home, caplog):
    app = create_app(None)

    @app.get("/boom-for-client-left")
    async def _boom():
        raise RuntimeError("route bug")

    with caplog.at_level(logging.DEBUG, logger="localm"):
        sent, raised = _drive(app, "GET", "/boom-for-client-left")
    assert isinstance(raised, RuntimeError) and str(raised) == "route bug"
    assert _status(sent) == 500
    assert any("unhandled error: GET /boom-for-client-left" in r.getMessage()
               for r in _errors(caplog))
    assert _count("/boom-for-client-left", 500) == 1
    assert _count("/boom-for-client-left", 499) == 0


def test_a_cancelled_error_while_the_client_is_still_there_is_not_a_499(home, caplog):
    app = create_app(None)

    @app.get("/cancel-for-client-left")
    async def _cancel():
        raise asyncio.CancelledError()

    with caplog.at_level(logging.DEBUG, logger="localm"):
        sent, raised = _drive(app, "GET", "/cancel-for-client-left")
    assert raised is not None
    assert _status(sent) != 499
    assert "the client left before the response" not in caplog.text
    assert _count("/cancel-for-client-left", 499) == 0


# --------------------------------------------------------------------------- #
# ClientLeftMiddleware re-raises what is not a client that left
# --------------------------------------------------------------------------- #

def _gone_scope(gone):
    async def poll():
        return gone
    return {"type": "http", "method": "POST", "path": "/x", _DISCONNECT_POLL_KEY: poll}


async def _noop_receive():
    return {"type": "http.request", "body": b"", "more_body": False}


def _middleware_run(inner, scope):
    sent = []

    async def send(message):
        sent.append(message)

    async def run():
        try:
            await ClientLeftMiddleware(inner)(scope, _noop_receive, send)
        except BaseException as exc:  # noqa: BLE001 - the test inspects it
            return exc
        return None

    return sent, asyncio.run(run())


async def _raises_cancelled(scope, receive, send):
    raise asyncio.CancelledError()


def test_middleware_answers_499_when_the_client_left():
    sent, raised = _middleware_run(_raises_cancelled, _gone_scope(True))
    assert raised is None
    assert [m["type"] for m in sent] == ["http.response.start", "http.response.body"]
    assert sent[0]["status"] == 499


def test_middleware_reraises_when_the_client_is_still_connected():
    sent, raised = _middleware_run(_raises_cancelled, _gone_scope(False))
    assert isinstance(raised, asyncio.CancelledError) and sent == []


def test_middleware_reraises_without_a_disconnect_poll():
    sent, raised = _middleware_run(_raises_cancelled,
                                   {"type": "http", "method": "POST", "path": "/x"})
    assert isinstance(raised, asyncio.CancelledError) and sent == []


def test_middleware_reraises_after_a_response_started():
    async def started_then_cancelled(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise asyncio.CancelledError()

    sent, raised = _middleware_run(started_then_cancelled, _gone_scope(True))
    assert isinstance(raised, asyncio.CancelledError)
    assert [m["status"] for m in sent if m["type"] == "http.response.start"] == [200]


def test_middleware_lets_a_real_task_cancellation_through():
    sent = []
    entered = asyncio.Event

    async def run():
        ready = entered()

        async def waits_forever(scope, receive, send):
            ready.set()
            await asyncio.Event().wait()

        async def send(message):
            sent.append(message)

        task = asyncio.ensure_future(
            ClientLeftMiddleware(waits_forever)(_gone_scope(True), _noop_receive, send))
        await ready.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return "cancelled"
        return "swallowed"

    assert asyncio.run(run()) == "cancelled"
    assert sent == []
