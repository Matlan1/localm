# SPDX-License-Identifier: AGPL-3.0-or-later
"""GET /metrics: the opt-in Prometheus endpoint.

Pins what the endpoint exposes, who may read it, and (most importantly) what it
can never carry: prompts, replies, model names and request paths.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from unittest.mock import MagicMock

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from localm import scopes
from localm.inference import http_server as hs
from localm.inference import metrics
from localm.inference.app_assembly import metrics as assembly
from localm.inference.http_server import create_app

SECRET = "ZQXJ-distinctive-7731"


def _engine(display_name: str = "test-model"):
    engine = MagicMock()
    state = {"loaded": True}
    engine.display_name = display_name

    def _chat_stream(messages, **kw):
        yield f"reply-{SECRET} "
        yield "more"

    engine.chat_stream.side_effect = _chat_stream
    engine.count_tokens.return_value = 7
    engine.count_messages_tokens.return_value = 5
    engine.context_capacity.return_value = 4096
    type(engine).loaded = property(lambda self: state["loaded"])
    return engine


@pytest.fixture(autouse=True)
def _metrics_off_after():
    yield
    metrics.configure(False)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("LOCALM_METRICS", "1")
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)


def _client(engine=None, *, bind_host=None):
    app = create_app(engine or _engine())
    if bind_host is not None:
        app.state.bind_host = bind_host
    return app, TestClient(app)


def _scrape(app, client, **kw):
    return client.get("/metrics",
                      headers={"Authorization": f"Bearer {app.state.shell_token}"},
                      **kw)


def _chat(client, content="hello", *, stream=False):
    return client.post("/v1/chat/completions", json={
        "model": "test-model", "stream": stream,
        "messages": [{"role": "user", "content": content}]})


# --------------------------------------------------------------------------- #
# Off by default
# --------------------------------------------------------------------------- #


def test_default_config_leaves_metrics_off():
    from localm.config import DEFAULT_CONFIG
    from localm.settings_schema import CORE_FIELDS
    assert DEFAULT_CONFIG["metrics_enabled"] is False
    field = next(f for f in CORE_FIELDS if f.key == "metrics_enabled")
    assert field.admin_only


def test_disabled_registers_no_route_and_no_middleware(monkeypatch):
    monkeypatch.delenv("LOCALM_METRICS", raising=False)
    app, client = _client()
    assert "/metrics" not in {getattr(r, "path", None) for r in app.routes}
    assert "MetricsMiddleware" not in {m.cls.__name__ for m in app.user_middleware}
    assert _scrape(app, client).status_code == 404
    unknown = client.get("/no-such-route")
    for headers in ({}, {"Origin": "http://localhost:5173"}):
        r = client.get("/metrics", headers=headers)
        assert (r.status_code, r.json()) == (unknown.status_code, unknown.json())
    assert not metrics.is_enabled()


def test_config_key_turns_it_on(monkeypatch):
    monkeypatch.delenv("LOCALM_METRICS", raising=False)
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"metrics_enabled": True})
    assert assembly.metrics_enabled()
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"metrics_enabled": False})
    assert not assembly.metrics_enabled()


def test_unreadable_config_reads_as_off(monkeypatch):
    monkeypatch.delenv("LOCALM_METRICS", raising=False)

    def boom():
        raise OSError("config unreadable")
    monkeypatch.setattr("localm.config.load_config", boom)
    assert not assembly.metrics_enabled()


# --------------------------------------------------------------------------- #
# Access control
# --------------------------------------------------------------------------- #


def test_route_requires_the_admin_scope(enabled):
    app = create_app(_engine())
    route = next(r for r in app.routes
                 if isinstance(r, APIRoute) and r.path == "/metrics")
    gates = [d.call for d in route.dependant.dependencies]
    assert [inspect.getclosurevars(g).nonlocals["scope"] for g in gates
            if g.__qualname__ == "require_scope.<locals>.dep"] == [scopes.ADMIN]


def test_keyless_loopback_needs_the_shell_token(enabled):
    app, client = _client()
    assert client.get("/metrics").status_code == 403
    assert _scrape(app, client).status_code == 200


def test_keyless_non_loopback_bind_is_not_served(enabled):
    app, client = _client(bind_host="0.0.0.0")
    assert client.get("/metrics").status_code == 404
    assert _scrape(app, client).status_code == 404


def test_cross_origin_scrape_is_refused(enabled):
    app, client = _client()
    r = client.get("/metrics", headers={
        "Authorization": f"Bearer {app.state.shell_token}",
        "Origin": "http://localhost:5173"})
    assert r.status_code == 403


def test_keyed_cross_origin_scrape_is_refused(enabled, monkeypatch):
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    app, client = _client()
    headers = {"Authorization": "Bearer ownersecret"}
    assert client.get("/metrics", headers=headers).status_code == 200
    r = client.get("/metrics", headers={**headers,
                                        "Origin": "http://localhost:5173"})
    assert r.status_code == 403


def test_keyed_server_needs_an_admin_key(enabled, monkeypatch):
    from localm import auth
    narrow = auth.create_key("narrow", [scopes.MODELS_READ])["key"]
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    app, client = _client(bind_host="0.0.0.0")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={
        "Authorization": f"Bearer {narrow}"}).status_code == 403
    ok = client.get("/metrics", headers={"Authorization": "Bearer ownersecret"})
    assert ok.status_code == 200
    assert ok.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert ok.headers["cache-control"] == "no-store"


# --------------------------------------------------------------------------- #
# What it reports
# --------------------------------------------------------------------------- #


def test_chat_shows_up_in_requests_tokens_and_ttft(enabled):
    app, client = _client()
    assert _chat(client, stream=True).status_code == 200
    assert _chat(client, stream=False).status_code == 200
    body = _scrape(app, client).text
    assert re.search(r'localm_http_requests_total\{method="POST",'
                     r'route="/v1/chat/completions",status="200"\} 2\b', body)
    assert 'localm_http_request_duration_seconds_bucket{method="POST",' in body
    assert re.search(r"localm_generated_tokens_total 14\b", body)
    assert re.search(r"localm_prompt_tokens_total \d+", body)
    assert "localm_time_to_first_token_seconds_count 2" in body
    assert "# TYPE localm_http_request_duration_seconds histogram" in body
    assert re.search(r"localm_models_loaded 1\b", body)
    assert re.search(r"localm_inference_queue_depth 0\b", body)
    assert re.search(r"localm_http_requests_in_flight 1\b", body)


@pytest.mark.parametrize("path, body", [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                              "stream": True}),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/completions", {"prompt": "hi", "stream": True}),
], ids=["chat-stream", "chat-full", "completions-stream"])
def test_every_generation_path_reports_its_tokens(enabled, path, body):
    app, client = _client()
    assert client.post(path, json={"model": "test-model", **body}).status_code == 200
    text = _scrape(app, client).text
    assert re.search(r"^localm_generated_tokens_total 7$", text, re.M)
    assert "localm_time_to_first_token_seconds_count 1" in text


def test_error_status_is_counted(enabled):
    app, client = _client()
    assert client.post("/v1/chat/completions", json={}).status_code == 422
    body = _scrape(app, client).text
    assert 'route="/v1/chat/completions",status="422"} 1' in body


def test_vram_gauges_follow_the_cached_reading(enabled, monkeypatch):
    from localm import sysstats
    monkeypatch.setattr(sysstats, "_vram", lambda: {
        "vram": {"used": 1000, "total": 4000, "percent": 25.0}})
    app, client = _client()
    body = _scrape(app, client).text
    assert re.search(r"localm_vram_used_bytes 1000\b", body)
    assert re.search(r"localm_vram_total_bytes 4000\b", body)

    monkeypatch.setattr(sysstats, "_vram", lambda: {"vram": {"total": 4000}})
    body = _scrape(app, client).text
    assert "localm_vram_used_bytes" not in body
    assert re.search(r"localm_vram_total_bytes 4000\b", body)

    monkeypatch.setattr(sysstats, "_vram", lambda: {})
    assert "localm_vram" not in _scrape(app, client).text


def test_queue_depth_counts_a_real_waiter(monkeypatch):
    async def scenario():
        sem = asyncio.Semaphore(1)
        monkeypatch.setattr(hs, "_inference_sems", {"m": sem})
        monkeypatch.setattr(hs, "_inference_sem", sem)
        await sem.acquire()
        waiter = asyncio.create_task(sem.acquire())
        await asyncio.sleep(0)
        depth = assembly._queue_depth()
        waiter.cancel()
        return depth

    assert asyncio.run(scenario()) == 1


def test_queue_depth_is_left_out_when_waiters_are_not_exposed(monkeypatch):
    class Opaque:
        pass
    monkeypatch.setattr(hs, "_inference_sems", {"m": Opaque()})
    monkeypatch.setattr(hs, "_inference_sem", None)
    assert assembly._queue_depth() is None


# --------------------------------------------------------------------------- #
# Privacy: no content in any label or value
# --------------------------------------------------------------------------- #


def test_scrape_after_a_chat_carries_none_of_the_content(enabled):
    engine = _engine(display_name=f"model-{SECRET}")
    app, client = _client(engine)
    assert client.post("/v1/chat/completions", json={
        "model": engine.display_name, "stream": True,
        "messages": [{"role": "user", "content": f"prompt {SECRET}"}]
    }).status_code == 200
    assert client.post("/v1/chat/completions", json={
        "model": engine.display_name,
        "messages": [{"role": "user", "content": f"prompt {SECRET}"}]
    }).status_code == 200
    client.get(f"/v1/models/{SECRET}")
    client.get(f"/no/such/{SECRET}/path?q={SECRET}")
    client.post(f"/api/{SECRET}", json={"x": SECRET})

    body = _scrape(app, client).text
    assert "localm_generated_tokens_total" in body
    assert 'route="other"' in body
    assert SECRET not in body
    assert "reply-" not in body and "prompt " not in body
    assert "test-model" not in body


# --------------------------------------------------------------------------- #
# The collector itself
# --------------------------------------------------------------------------- #


def test_collection_is_inert_until_configured():
    metrics.configure(False)
    metrics.request_started()
    metrics.request_finished("GET", "/health", 200, 0.1)
    metrics.observe_generation(5, 5, 100.0, 10.0)
    assert metrics.render() == "\n".join(
        ["# HELP localm_http_requests_in_flight HTTP requests currently being served.",
         "# TYPE localm_http_requests_in_flight gauge",
         "localm_http_requests_in_flight 0"]) + "\n"


def test_histogram_buckets_are_cumulative():
    metrics.configure(True)
    for seconds in (0.004, 0.2, 0.2, 400.0):
        metrics.request_started()
        metrics.request_finished("GET", "/health", 200, seconds)
    text = metrics.render()
    bucket = lambda le: int(re.search(  # noqa: E731
        rf'duration_seconds_bucket\{{method="GET",route="/health",le="{re.escape(le)}"\}} (\d+)',
        text).group(1))
    assert bucket("0.005") == 1
    assert bucket("0.25") == 3
    assert bucket("300") == 3
    assert bucket("+Inf") == 4
    assert 'duration_seconds_count{method="GET",route="/health"} 4' in text


def test_unmeasured_generation_figures_are_not_recorded():
    metrics.configure(True)
    metrics.observe_generation(None, 0, None, None)
    text = metrics.render()
    assert "localm_generated_tokens_total" not in text
    assert "localm_time_to_first_token_seconds" not in text
    assert "localm_tokens_per_second" not in text


def test_label_values_are_escaped():
    metrics.configure(True)
    metrics.request_started()
    metrics.request_finished("GET", 'a"b\\c\nd', 200, 0.1)
    assert '\\"' in metrics.render() and "\\n" in metrics.render()


@pytest.mark.parametrize("template, expected", [
    ("/v1/models/{model_id}", "/v1/models/{model_id}"),
    ("/api/files/{path:path}", "/api/files/{path:path}"),
    ("/has space", "other"),
    ("/emoji/é", "other"),
    ("/" + "a" * 200, "other"),
    ("", "other"),
])
def test_route_label_only_admits_url_template_characters(template, expected):
    route = MagicMock()
    route.path = template
    assert metrics.route_label(route) == expected


def test_route_label_for_a_non_route_is_other():
    assert metrics.route_label(None) == "other"
    assert metrics.route_label(object()) == "other"


@pytest.mark.parametrize("method, expected", [
    ("GET", "GET"), ("PROPFIND", "OTHER"), (None, "OTHER"), ("get", "OTHER")])
def test_method_label_is_a_closed_set(method, expected):
    assert metrics.method_label(method) == expected
