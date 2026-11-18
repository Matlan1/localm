# SPDX-License-Identifier: AGPL-3.0-or-later
# ruff: noqa: F811
"""Server admin fixes: the api-mode landing redirect, a removed model leaving
/health and /v1/models, a superseded load that stops evicting, and the
no-model answer of /api/vram-estimate."""

import asyncio

from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from localm.inference import switch_admission
from tests.test_model_unload_targeted import (  # noqa: F401
    FakeEngine, _authed_client, _load, setup_multi_model)


def _registry_without(monkeypatch, *names):
    reg = {n: {"path": f"Z:/models/{n}.gguf", "source": "local"}
           for n in ("model-a", "model-b") if n not in names}
    monkeypatch.setattr("localm.config.load_registry", lambda: reg)


def test_api_mode_app_redirects_root_to_docs(setup_multi_model):
    client = TestClient(hs.create_app(None, api_landing=True), follow_redirects=False)
    r = client.get("/")
    assert r.status_code == 307
    assert r.headers["location"] == "/docs"


def test_gui_cli_builds_the_app_with_the_landing_in_api_mode(monkeypatch):
    from localm.plugins.gui import cli
    seen = {}

    def fake_create_app(engine, **kw):
        seen.update(kw)
        raise RuntimeError("stop")
    monkeypatch.setattr(hs, "create_app", fake_create_app)
    for api_mode in (True, False):
        seen.clear()
        try:
            cli._build_app(None, None, None, api_mode=api_mode)
        except RuntimeError:
            pass
        assert seen == {"api_landing": api_mode}


def test_a_removed_model_whose_file_remains_leaves_health_models_and_chat(
        setup_multi_model, monkeypatch, tmp_path):
    kept_file = tmp_path / "model-a.gguf"
    kept_file.write_bytes(b"x")
    engine = FakeEngine("model-a")
    engine.model_path = str(kept_file)
    app = hs.create_app(engine)
    client = _authed_client(app)
    assert "model-a" in hs._registered_seen
    assert client.post("/v1/models/unload", params={"model": "model-a"}).status_code == 200
    _registry_without(monkeypatch, "model-a")

    health = client.get("/health")
    assert "model-a" not in health.text
    listed = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert "model-a" not in listed
    tags = [m["name"] for m in client.get("/api/tags").json()["models"]]
    assert not any(t.startswith("model-a") for t in tags)
    chat = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}]})
    assert chat.status_code in (400, 503), chat.text
    assert "Model files not found" not in chat.text
    assert hs._default_model_name is None and "model-a" not in hs._engines


def test_a_startup_model_given_by_path_is_kept_while_its_file_exists(
        setup_multi_model, monkeypatch, tmp_path):
    kept_file = tmp_path / "direct.gguf"
    kept_file.write_bytes(b"x")
    engine = FakeEngine("direct")
    engine.model_path = str(kept_file)
    app = hs.create_app(engine)
    client = _authed_client(app)
    assert "direct" not in hs._registered_seen
    assert client.post("/v1/models/unload", params={"model": "direct"}).status_code == 200
    assert client.get("/health").json()["model"] == "direct"
    assert "direct" in [m["id"] for m in client.get("/v1/models").json()["data"]]


def test_a_renamed_then_removed_model_does_not_stay_as_the_health_model(
        setup_multi_model, monkeypatch, tmp_path):
    kept_file = tmp_path / "model-a.gguf"
    kept_file.write_bytes(b"x")
    engine = FakeEngine("model-a")
    engine.model_path = str(kept_file)
    app = hs.create_app(engine)
    client = _authed_client(app)
    hs.rekey_loaded_model("model-a", "renamed")
    assert client.post("/v1/models/unload", params={"model": "renamed"}).status_code == 200
    _registry_without(monkeypatch, "model-a")
    assert "renamed" not in client.get("/health").text
    assert "renamed" not in [m["id"] for m in client.get("/v1/models").json()["data"]]


def _budget(name):
    return switch_admission.LoadBudget(
        name=name, vram_required=100, headroom=0, resident_cap=None,
        pinned=frozenset(), check_split_fit=False)


def _evicting_make_room(monkeypatch, victims, on_free):
    """_switch_make_room against a probe that never admits and a decision that
    names the next idle victim; *on_free* runs after each victim is freed."""
    freed = []

    async def probe(loop, budget, reading=None):
        return None

    def decide(probe_, budget, lru, live):
        return switch_admission.AdmissionDecision(
            switch_admission.EVICT, False, False, victim=next(
                (v for v in victims if v not in freed), None))

    async def free_victim(loop, victim, engine, probe_, needed=None, extra_wait=True):
        freed.append(victim)
        on_free(victim)
        return hs.VictimRelease(victim, True, None, None, 0.0)

    monkeypatch.setattr(hs, "_switch_probe_vram", probe)
    monkeypatch.setattr(switch_admission, "decide_admission", decide)
    monkeypatch.setattr(hs, "_switch_free_victim", free_victim)
    monkeypatch.setattr(hs, "_switch_detach_victim", lambda *a, **k: False)
    monkeypatch.setattr(switch_admission, "effective_unload_policy",
                        lambda *a, **k: switch_admission.UNLOAD_ALWAYS)
    monkeypatch.setattr(hs, "_switch_confirm_before_eviction",
                        lambda *a, **k: _none())
    return freed


async def _none():
    return None


async def _make_room(budget):
    return await hs._switch_make_room(asyncio.get_running_loop(), budget, preempt=True,
                                force=False, activate=True)


def test_a_load_superseded_before_eviction_evicts_nothing(monkeypatch):
    engines = {n: FakeEngine(n) for n in ("v1", "v2")}
    for e in engines.values():
        e._loaded = True
    monkeypatch.setattr(hs, "_engines", engines)
    monkeypatch.setattr(hs, "_switch_desired", "other")
    freed = _evicting_make_room(monkeypatch, ["v1", "v2"], lambda v: None)
    result = asyncio.run(_make_room(_budget("big")))
    assert result == {"status": "superseded", "model": "big", "by": "other"}
    assert freed == []


def test_a_load_superseded_mid_eviction_stops_at_the_next_victim(monkeypatch):
    engines = {n: FakeEngine(n) for n in ("v1", "v2", "v3")}
    for e in engines.values():
        e._loaded = True
    monkeypatch.setattr(hs, "_engines", engines)
    monkeypatch.setattr(hs, "_switch_desired", "big")

    def select_other(victim):
        monkeypatch.setattr(hs, "_switch_desired", "m05")
    freed = _evicting_make_room(monkeypatch, ["v1", "v2", "v3"], select_other)
    result = asyncio.run(_make_room(_budget("big")))
    assert result == {"status": "superseded", "model": "big", "by": "m05"}
    assert freed == ["v1"]


def test_vram_estimate_with_no_model_says_so_and_claims_no_fit(setup_multi_model):
    app = hs.create_app(None)
    from localm.plugins.gui.web import attach_gui
    attach_gui(app, self_url="http://127.0.0.1:1/v1", switch_model=None,
               active_model=lambda: "")
    client = _authed_client(app)
    r = client.get("/api/vram-estimate")
    assert r.status_code == 200
    body = r.json()
    assert body["no_model"] is True
    assert body["model"] == "" and body["model_bytes"] == 0
    assert body["fits"] is None
    assert "No model is selected" in body["note"]


def test_an_unreadable_registry_does_not_make_the_held_model_look_removed(
        setup_multi_model, monkeypatch, tmp_path):
    engine = FakeEngine("model-a")
    engine.model_path = str(tmp_path / "model-a.gguf")
    app = hs.create_app(engine)
    client = _authed_client(app)
    assert client.post("/v1/models/unload", params={"model": "model-a"}).status_code == 200
    monkeypatch.setattr("localm.config.load_registry", lambda: {})
    monkeypatch.setattr("localm.storekit.read_json_checked", lambda path, default: ({}, False))
    client.get("/health")
    assert "model-a" in hs._engines and hs._default_model_name == "model-a"


def test_health_with_an_empty_registry_answers_while_registry_reads_are_blocked(
        setup_multi_model, monkeypatch):
    import threading
    import time

    from localm import config, storekit
    app = hs.create_app(None)
    monkeypatch.setattr("localm.config.load_registry", lambda: {})
    release = threading.Event()
    real = storekit.read_json_checked

    def blocked(path, default):
        release.wait(30)
        return real(path, default)
    monkeypatch.setattr(storekit, "read_json_checked", blocked)
    monkeypatch.setattr(hs, "load_registry_checked", lambda: ({}, True))
    try:
        with TestClient(app) as client:
            started = time.monotonic()
            r = client.get("/health")
            took = time.monotonic() - started
    finally:
        release.set()
    assert r.status_code in (200, 503)
    assert took < 2 * config.LOOP_READ_WAIT_S + 1.0, took
