# SPDX-License-Identifier: AGPL-3.0-or-later
"""A request that carries privacy mode to a server in full mode.

``localm run MODEL --mode privacy`` (and a coder session in privacy mode)
attaches to a running server and sends ``X-Localm-Session-Mode: privacy``. The
server may itself be in full mode with the debug log on. Nothing of that
request (prompt, reply, raw model output) may reach the debug log, and memory
must treat the request as privacy mode: nothing stored, nothing recalled.

Every test here drives a real request through ``create_app`` with the real
``debug_content_enabled()``; only the engine is a stand-in.
"""

from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm import audit, debuglog
from localm.audit import SESSION_MODE_HEADER
from localm.inference.http_server import create_app
from tests._shared_model_guard import (  # noqa: F401
    no_embedder_download,
    no_shared_model_left_resident,
)

SECRET = "my bank PIN hint is my mother's maiden name Vandermeulen"
REPLY = ("zebra", " quartz")
PRIVACY = {SESSION_MODE_HEADER: "privacy"}

_ROUTES = [
    pytest.param("/v1/chat/completions", True, id="chat-stream"),
    pytest.param("/v1/chat/completions", False, id="chat-full"),
    pytest.param("/v1/completions", True, id="completion-stream"),
    pytest.param("/v1/completions", False, id="completion-full"),
]


def _engine(seen: dict | None = None):
    """An engine whose ``chat_stream`` records, on the thread that generates,
    what the backend's content gate and the request scope say there."""
    engine = MagicMock()

    def _chat_stream(messages, **kwargs):
        if seen is not None:
            seen["content_allowed"] = debuglog.debug_content_enabled()
            seen["request_is_privacy"] = audit.request_is_privacy()
            seen["thread"] = threading.get_ident()
        yield from REPLY

    engine.chat_stream.side_effect = _chat_stream
    engine.count_tokens.return_value = 3
    engine.count_messages_tokens.return_value = 3
    engine.context_capacity.return_value = None
    engine.display_name = "test-model"
    engine.supports_images = False
    engine.can_be_multimodal = False
    engine.last_finish_reason = "stop"
    type(engine).loaded = property(lambda self: True)
    return engine


def _body(path: str, stream: bool) -> dict:
    if path == "/v1/completions":
        return {"model": "test-model", "prompt": SECRET, "stream": stream}
    return {"model": "test-model", "stream": stream,
            "messages": [{"role": "user", "content": SECRET}]}


@pytest.fixture
def full_mode_debug(monkeypatch, caplog):
    """The server is in full mode on every surface and the debug log is on.
    Closes the session recorder the test's app left open."""
    monkeypatch.setenv("LOCALM_MODE", "full")
    monkeypatch.setenv("LOCALM_DEBUG", "1")
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    caplog.set_level(logging.DEBUG, logger="localm")
    assert debuglog.debug_content_enabled() is True
    yield caplog
    from localm.inference import http_server as hs
    if hs._session_recorder is not None:
        hs._session_recorder.close()


def _blob(caplog) -> str:
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert blob.strip(), "nothing was captured from the 'localm' logger"
    return blob


def _post(client, path, stream, headers=None):
    r = client.post(path, json=_body(path, stream), headers=headers or {})
    assert r.status_code == 200, r.text
    assert "zebra" in r.text
    return r


@pytest.mark.parametrize("path,stream", _ROUTES)
def test_a_privacy_request_leaves_no_content_in_the_debug_log(
        full_mode_debug, path, stream):
    seen: dict = {}
    with TestClient(create_app(_engine(seen))) as client:
        _post(client, path, stream, PRIVACY)
    blob = _blob(full_mode_debug)
    assert SECRET not in blob, f"the prompt reached the debug log:\n{blob}"
    assert "Vandermeulen" not in blob
    assert "zebra quartz" not in blob, f"the reply reached the debug log:\n{blob}"
    assert "assembled chat prompt" not in blob
    assert "chat completion reply" not in blob
    assert seen["thread"] != threading.get_ident()
    assert seen["request_is_privacy"] is True
    assert seen["content_allowed"] is False, (
        "the backend's raw model output line would have been written")


@pytest.mark.parametrize("path,stream", _ROUTES)
def test_the_same_request_without_the_header_is_logged(
        full_mode_debug, path, stream):
    seen: dict = {}
    with TestClient(create_app(_engine(seen))) as client:
        _post(client, path, stream)
    blob = _blob(full_mode_debug)
    assert SECRET in blob
    assert "zebra quartz" in blob
    assert seen["request_is_privacy"] is False
    assert seen["content_allowed"] is True


def test_content_is_logged_again_once_the_privacy_request_has_ended(full_mode_debug):
    with TestClient(create_app(_engine())) as client:
        _post(client, "/v1/chat/completions", True, PRIVACY)
        assert audit.any_coder_session_is_privacy() is False
        assert SECRET not in _blob(full_mode_debug)
        _post(client, "/v1/chat/completions", True)
    assert SECRET in _blob(full_mode_debug)


def test_an_unknown_mode_in_the_header_counts_as_privacy(full_mode_debug):
    seen: dict = {}
    with TestClient(create_app(_engine(seen))) as client:
        _post(client, "/v1/chat/completions", False, {SESSION_MODE_HEADER: "bogus"})
    assert SECRET not in _blob(full_mode_debug)
    assert seen["content_allowed"] is False


def test_a_request_whose_mode_cannot_be_resolved_counts_as_privacy(
        full_mode_debug, monkeypatch):
    import localm.inference.http_server as hs

    def _boom(request):
        raise RuntimeError("mode lookup failed")

    monkeypatch.setattr(hs, "request_session_mode", _boom)
    seen: dict = {}
    with TestClient(create_app(_engine(seen))) as client:
        _post(client, "/v1/chat/completions", False, PRIVACY)
    blob = _blob(full_mode_debug)
    assert SECRET not in blob
    assert "could not resolve the session mode of a request" in blob
    assert seen["request_is_privacy"] is True


@pytest.mark.parametrize("who", ["scoped", "owner"])
def test_only_the_owners_header_counts_on_a_keyed_server(full_mode_debug, who):
    from localm import auth
    owner = "owner-key-request-privacy-0123456789"
    auth.set_api_key(owner)
    key = owner
    if who == "scoped":
        key = auth.create_key("chatter", ["chat"], allow_privileged=False)["key"]
    seen: dict = {}
    with TestClient(create_app(_engine(seen))) as client:
        _post(client, "/v1/chat/completions", False,
              {**PRIVACY, "Authorization": f"Bearer {key}"})
    assert seen["request_is_privacy"] is (who == "owner")
    assert (SECRET in _blob(full_mode_debug)) is (who == "scoped")


def test_a_failed_generation_ends_the_privacy_scope(full_mode_debug):
    engine = _engine()

    def _fails(messages, **kwargs):
        raise RuntimeError("backend exploded")
        yield ""   # pragma: no cover

    engine.chat_stream.side_effect = _fails
    with TestClient(create_app(engine), raise_server_exceptions=False) as client:
        client.post("/v1/chat/completions", headers=PRIVACY,
                    json=_body("/v1/chat/completions", False))
        client.post("/v1/chat/completions", headers=PRIVACY,
                    json=_body("/v1/chat/completions", True))
    assert audit.any_coder_session_is_privacy() is False
    assert SECRET not in _blob(full_mode_debug)


# --------------------------------------------------------------------------- #
#  The scope itself                                                           #
# --------------------------------------------------------------------------- #

def test_the_scope_is_seen_process_wide_and_nests(monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "full")
    monkeypatch.setenv("LOCALM_DEBUG", "1")
    other: dict = {}

    def _look(name):
        other[name] = (audit.request_is_privacy(), debuglog.debug_content_enabled())

    with audit.request_privacy(True):
        with audit.request_privacy(True):
            t = threading.Thread(target=_look, args=("plain thread",))
            t.start()
            t.join()
        assert audit.any_coder_session_is_privacy() is True
        carried = audit.carry_request_privacy(lambda: _look("carried"))
    # A plain thread is outside the scope but still gets no content while a
    # privacy request is in flight anywhere in the process.
    assert other["plain thread"] == (False, False)
    assert audit.any_coder_session_is_privacy() is False
    t = threading.Thread(target=carried)
    t.start()
    t.join()
    assert other["carried"] == (True, False)
    assert audit.any_coder_session_is_privacy() is False
    assert debuglog.debug_content_enabled() is True


def test_a_false_scope_changes_nothing(monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "full")
    monkeypatch.setenv("LOCALM_DEBUG", "1")

    def fn():
        return 1

    with audit.request_privacy(False):
        assert audit.request_is_privacy() is False
        assert audit.any_coder_session_is_privacy() is False
        assert debuglog.debug_content_enabled() is True
        assert audit.carry_request_privacy(fn) is fn


# --------------------------------------------------------------------------- #
#  Memory follows the request's mode                                          #
# --------------------------------------------------------------------------- #

def _echo_engine(captured: dict):
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.count_tokens.return_value = 5

    def _stream(messages, **kw):
        captured["messages"] = [dict(m) for m in messages]
        yield "ok"

    engine.chat_stream.side_effect = _stream
    type(engine).loaded = property(lambda self: True)
    engine.supports_images = True
    return engine


@pytest.fixture
def memory_app(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setenv("LOCALM_MODE", "full")
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    captured: dict = {}
    app = create_app(_echo_engine(captured))
    mgr = app.state.plugin_manager
    mgr.install("memory")
    mgr.enable("memory")
    shell = getattr(app.state, "shell_token", None)
    hdr = {"Authorization": f"Bearer {shell}"} if shell else {}
    return app, captured, hdr


def _system_text(captured) -> str:
    return "\n".join(m.get("content") or "" for m in captured.get("messages", [])
                     if m.get("role") == "system")


def _chat(client, text, headers=None):
    r = client.post("/v1/chat/completions", headers=headers or {}, json={
        "model": "m", "messages": [{"role": "user", "content": text}]})
    assert r.status_code == 200, r.text
    return r


def test_a_privacy_request_recalls_nothing_from_memory(memory_app):
    app, captured, hdr = memory_app
    with TestClient(app) as c:
        assert c.post("/api/memory/append", headers=hdr,
                      json={"text": "User prefers Rust and cargo"}).status_code == 200
        _chat(c, "help me fix my rust cargo build", PRIVACY)
        assert "Rust and cargo" not in _system_text(captured)
        _chat(c, "help me fix my rust cargo build")
        assert "Rust and cargo" in _system_text(captured)


def test_a_privacy_request_stores_nothing_in_memory(memory_app):
    app, _captured, hdr = memory_app

    def _stored(c) -> str:
        r = c.get("/api/memory", headers=hdr)
        assert r.status_code == 200, r.text
        return r.text

    with TestClient(app) as c:
        _chat(c, "remember that I love RPG-style games", PRIVACY)
        assert "RPG-style" not in _stored(c)
        _chat(c, "remember that I love RPG-style games")
        assert "RPG-style" in _stored(c)


def test_a_privacy_request_cannot_write_through_the_memory_routes(memory_app):
    app, _captured, hdr = memory_app
    with TestClient(app) as c:
        r = c.post("/api/memory/append", headers={**hdr, **PRIVACY},
                   json={"text": "User prefers Rust and cargo"})
        assert r.status_code == 403, r.text
        assert "Rust and cargo" not in c.get("/api/memory", headers=hdr).text


def test_the_memory_outlet_runs_inside_the_request_scope(memory_app, monkeypatch):
    import sys
    app, _captured, _hdr = memory_app
    plug = sys.modules["localm.plugins.builtin.memory.plug"]
    seen: list = []
    monkeypatch.setattr(plug, "_maybe_auto_consolidate",
                        lambda principal=None: seen.append(
                            (audit.request_is_privacy(), plug._persist_enabled())))
    with TestClient(app) as c:
        _chat(c, "hello there", PRIVACY)
        _chat(c, "hello there")
    assert seen == [(True, False), (False, True)]


# --------------------------------------------------------------------------- #
#  Other work a request hands to a thread                                     #
# --------------------------------------------------------------------------- #

def test_compaction_generates_inside_the_request_scope(full_mode_debug, monkeypatch):
    import asyncio

    import localm.inference.compact as compact
    import localm.inference.http_server as hs
    seen: dict = {}

    def _compact(messages, generate, **kwargs):
        seen["request_is_privacy"] = audit.request_is_privacy()
        seen["content_allowed"] = debuglog.debug_content_enabled()
        seen["thread"] = threading.get_ident()
        return messages, False

    monkeypatch.setattr(compact, "compact_messages", _compact)
    engine = _engine()
    engine.context_capacity.return_value = 4096

    async def _run(private: bool):
        with audit.request_privacy(private):
            await hs._compact_for_capacity(
                engine, [{"role": "user", "content": SECRET}])

    asyncio.run(_run(True))
    assert seen["thread"] != threading.get_ident()
    assert (seen["request_is_privacy"], seen["content_allowed"]) == (True, False)
    asyncio.run(_run(False))
    assert (seen["request_is_privacy"], seen["content_allowed"]) == (False, True)
    assert audit.any_coder_session_is_privacy() is False


def test_a_job_run_on_demand_runs_inside_the_request_scope(
        full_mode_debug, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    import localm.plugins.builtin.jobs.runner as runner
    seen: list = []

    def _run_job(job, engine=None):
        seen.append((audit.request_is_privacy(), debuglog.debug_content_enabled()))
        return {"status": "ok", "output": "ran", "error": None,
                "started": 0.0, "finished": 0.0}

    monkeypatch.setattr(runner, "run_job", _run_job)
    app = create_app(_engine())
    mgr = app.state.plugin_manager
    mgr.install("jobs")
    mgr.enable("jobs")
    shell = getattr(app.state, "shell_token", None)
    hdr = {"Authorization": f"Bearer {shell}"} if shell else {}
    with TestClient(app) as c:
        r = c.post("/api/jobs", headers=hdr, json={
            "name": "j", "prompt": "hi", "schedule_kind": "interval",
            "schedule": 3600})
        assert r.status_code == 200, r.text
        jid = r.json()["id"]
        assert c.post(f"/api/jobs/{jid}/run",
                      headers={**hdr, **PRIVACY}).status_code == 200
        assert c.post(f"/api/jobs/{jid}/run", headers=hdr).status_code == 200
    assert seen == [(True, False), (False, True)]


def test_a_client_that_goes_away_mid_stream_leaves_no_content_and_no_scope(
        full_mode_debug):
    """The response's ``send`` fails after the first body chunk, as it does for
    a closed connection, while the model is still generating."""
    import asyncio
    import json
    import time

    gate: list = []
    done = threading.Event()

    def _slow(messages, **kwargs):
        try:
            for i in range(200):
                gate.append(debuglog.debug_content_enabled())
                time.sleep(0.01)
                yield f"tok{i} "
        finally:
            gate.append(debuglog.debug_content_enabled())
            done.set()

    engine = _engine()
    engine.chat_stream.side_effect = _slow
    app = create_app(engine)
    body = json.dumps(_body("/v1/chat/completions", True)).encode()
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions", "query_string": b"", "root_path": "",
        "headers": [(b"host", b"testserver"),
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (SESSION_MODE_HEADER.lower().encode(), b"privacy")],
        "client": ("127.0.0.1", 50000), "server": ("testserver", 80),
    }
    sent: list = []

    async def _drive():
        got = [{"type": "http.request", "body": body, "more_body": False}]

        async def receive():
            if got:
                return got.pop()
            await asyncio.sleep(3600)

        async def send(message):
            if message["type"] == "http.response.body":
                sent.append(message.get("body", b""))
                if b"tok" in b"".join(sent):
                    raise OSError("connection closed by the client")

        try:
            await app(scope, receive, send)
        except Exception:   # noqa: BLE001
            pass

    asyncio.run(_drive())
    assert b"tok" in b"".join(sent), "the stream never started"
    assert done.wait(30), "the generation thread never ended"
    assert gate and not any(gate), "content was allowed while the thread generated"
    deadline = time.monotonic() + 10
    while audit.any_coder_session_is_privacy() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert audit.any_coder_session_is_privacy() is False
    blob = _blob(full_mode_debug)
    assert SECRET not in blob
    assert "tok0" not in blob
