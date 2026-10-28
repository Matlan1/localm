# SPDX-License-Identifier: AGPL-3.0-or-later
"""`force` on every unload entry point, for every kind of resident model.

`POST /v1/models/unload` (query `force`) and the Ollama `keep_alive: 0` unload
(body `"force": true`) leave a busy model loaded and ask without force, and cut
it off with force. A forced Unload all also stops a busy embedding, reranker or
speech model: its worker is killed, the request in flight on it fails, nothing
respawns it, and the response counts it in `unloaded_other`.
"""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from localm.inference import _embedder_runner as emb_runner_mod
from localm.inference import _speech_runner as speech_runner_mod
from localm.inference import embedder as emb
from localm.inference import http_server as hs
from localm.inference import reranker as rr
from localm.inference import residency
from localm.inference import speech


class FakeEngine:
    def __init__(self, name, active=0):
        self.display_name = name
        self.model_path = f"Z:/models/{name}.gguf"
        self._loaded = True
        self.active_requests = active
        self.unload_calls = 0

    @property
    def loaded(self):
        return self._loaded

    def unload(self):
        self.unload_calls += 1
        self._loaded = False


@pytest.fixture
def isolated(monkeypatch):
    monkeypatch.setattr("localm.discover.vram_info",
                        lambda: {"free": 10 * 1024 ** 3, "total": 16 * 1024 ** 3})
    monkeypatch.setattr("localm.vram.wait_for_vram_release",
                        lambda free_fn, before_bytes=None: (0, before_bytes))
    for d in (hs._engines, hs._engines_lru, hs._inference_sems,
              hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._engine = None
    hs._inference_sem = None
    hs._coder_session_manager = None
    real_wait = hs._wait_for_pin_clear
    monkeypatch.setattr(hs, "_wait_for_pin_clear",
                        lambda engine, **kw: real_wait(engine, timeout=0.05,
                                                       poll_interval=0.01))
    yield


def _install(name, active=0):
    e = FakeEngine(name, active)
    hs._engines[name] = e
    hs._engines_lru.append(name)
    hs._inference_sems[name] = asyncio.Semaphore(1)
    if hs._active_model_name is None:
        hs._active_model_name = name
        hs._engine = e
    return e


def _registered_reply(name):
    reply = residency.GenerationCancel()
    residency.register_cancel(name, reply)
    return reply


# --------------------------------------------------------------------------- #
#  POST /v1/models/unload takes force                                          #
# --------------------------------------------------------------------------- #

@pytest.fixture
def api(isolated, monkeypatch):
    monkeypatch.setattr("localm.config.load_registry", lambda: {
        "busy": {"path": "Z:/models/busy.gguf"},
        "idle": {"path": "Z:/models/idle.gguf"}})
    monkeypatch.setattr(emb, "loaded_path", lambda: None)
    monkeypatch.setattr(rr, "reranker_info", lambda: None)
    monkeypatch.setattr(speech, "speech_info", lambda: None)
    app = hs.create_app(None)
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {app.state.shell_token}"
    return client


def test_api_unload_of_a_busy_model_asks_without_force_and_cuts_it_with_force(api):
    engine = _install("busy", active=1)
    reply = _registered_reply("busy")
    try:
        asked = api.post("/v1/models/unload", params={"model": "busy"})
        assert asked.status_code == 200, asked.text
        assert asked.json()["status"] == "confirm_required", asked.json()
        assert "cuts off the reply" in asked.json()["detail"]
        assert not reply.is_set(), "the reply was cut before the client confirmed"
        assert engine.loaded and engine.unload_calls == 0

        forced = api.post("/v1/models/unload", params={"model": "busy", "force": "true"})
        assert forced.status_code == 200, forced.text
        assert forced.json()["status"] == "unloaded", forced.json()
        assert reply.is_set() and reply.by_server, "force did not stop the reply"
        assert not engine.loaded and engine.unload_calls == 1
    finally:
        residency.unregister_cancel("busy", reply)


def test_api_unload_all_of_a_busy_model_asks_without_force_and_cuts_it_with_force(api):
    engine = _install("busy", active=1)
    idle = _install("idle", active=0)
    reply = _registered_reply("busy")
    try:
        asked = api.post("/v1/models/unload").json()
        assert asked["skipped_in_use"] == ["busy"], asked
        assert asked["confirm_required"] == {"busy": "1 other active request"}
        assert asked["unloaded_models"] == ["idle"] and not idle.loaded
        assert not reply.is_set() and engine.loaded

        forced = api.post("/v1/models/unload", params={"force": "true"}).json()
        assert forced["unloaded_models"] == ["busy"], forced
        assert "skipped_in_use" not in forced and "confirm_required" not in forced
        assert reply.is_set() and reply.by_server
        assert not engine.loaded
    finally:
        residency.unregister_cancel("busy", reply)


def test_api_unload_rejects_a_force_that_is_not_a_boolean(api):
    engine = _install("busy", active=1)
    r = api.post("/v1/models/unload", params={"model": "busy", "force": "maybe"})
    assert r.status_code == 422
    assert engine.loaded and engine.unload_calls == 0


# --------------------------------------------------------------------------- #
#  The Ollama keep_alive: 0 unload takes "force": true                         #
# --------------------------------------------------------------------------- #

def test_ollama_unload_of_a_busy_model_is_a_409_without_force_and_cuts_it_with_force(
        isolated, monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    engine = MagicMock()
    state = {"loaded": True}
    engine.unload.side_effect = lambda: state.update(loaded=False)
    type(engine).loaded = property(lambda self: state["loaded"])
    engine.display_name = "test-model"
    engine.model_path = ""
    engine.active_requests = 1
    app = hs.create_app(engine)
    reply = _registered_reply("test-model")
    body = {"model": "test-model", "messages": [], "keep_alive": 0}
    try:
        with TestClient(app, headers={"Authorization": f"Bearer {app.state.shell_token}"}) as c:
            asked = c.post("/api/chat", json=body)
            assert asked.status_code == 409, asked.text
            assert '"force": true' in asked.json()["error"]
            assert not reply.is_set() and state["loaded"]

            forced = c.post("/api/chat", json={**body, "force": True})
        assert forced.status_code == 200, forced.text
        assert forced.json()["done_reason"] == "unload"
        assert reply.is_set() and reply.by_server
        assert not state["loaded"]
    finally:
        residency.unregister_cancel("test-model", reply)


# --------------------------------------------------------------------------- #
#  A forced Unload all stops a busy embedding / reranker / speech model         #
# --------------------------------------------------------------------------- #

def _hung_isolated_embedder(monkeypatch):
    """An IsolatedEmbedder whose real worker process hangs on its first
    command, as a wedged embed would."""
    monkeypatch.setenv(emb_runner_mod._FAULT_ENV, "hang")
    e = emb.IsolatedEmbedder.__new__(emb.IsolatedEmbedder)
    e.model_path = "Z:/models/embed.gguf"
    e.n_gpu_layers = 99
    e.gpu_fallback_reason = None
    e.effective_pooling = None
    e.dim = 384
    e.active_requests = 0
    e._rpc_lock = threading.RLock()
    reloads = []
    e._reload = lambda: reloads.append(1)
    runner = emb_runner_mod.EmbedderRunner()
    runner._spawn()
    e._runner = runner
    return e, runner, reloads


def _start_in_flight(call):
    outcome = {}

    def _run():
        try:
            outcome["result"] = call()
        except BaseException as exc:
            outcome["error"] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t, outcome


def _wait_until(predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.02)


def test_forced_unload_all_stops_a_busy_embedder_and_counts_it(isolated, monkeypatch):
    e, runner, reloads = _hung_isolated_embedder(monkeypatch)
    proc = runner._proc
    monkeypatch.setattr(emb, "_EMBEDDER", e)
    t, outcome = _start_in_flight(lambda: e.embed(["hello"]))
    try:
        _wait_until(lambda: e.active_requests == 1)

        asked = asyncio.run(hs.unload_all_models())
        assert asked["status"] == "in_use", asked
        assert asked["skipped_in_use"] == ["embedding model"]
        assert asked["confirm_required"] == {"embedding model": "1 active request"}
        assert asked["unloaded_other"] == [] and asked["embedder_unloaded"] is False
        assert emb._EMBEDDER is e and proc.is_alive(), "asking must not stop it"

        forced = asyncio.run(hs.unload_all_models(force=True))
        t.join(timeout=30)
        assert not t.is_alive(), "the in-flight embed never returned"
    finally:
        runner.shutdown(grace=0)

    assert outcome.get("error") is not None, outcome
    assert isinstance(outcome["error"], RuntimeError)
    assert "was unloaded while this request was running" in str(outcome["error"])
    assert forced["status"] == "unloaded", forced
    assert forced["unloaded_other"] == ["embedding model"]
    assert forced["embedder_unloaded"] is True
    assert "skipped_in_use" not in forced and "confirm_required" not in forced
    assert len(forced["unloaded_models"]) + len(forced["unloaded_other"]) == 1
    assert emb._EMBEDDER is None
    assert not proc.is_alive()
    assert reloads == [], "a stopped embedder respawned its worker"
    assert e.gpu_fallback_reason is None and e.n_gpu_layers == 99, \
        "the stop was taken for a GPU crash and fell back to CPU"
    assert e._runner is None


def test_a_call_queued_behind_a_stopped_embedder_never_respawns_it(isolated, monkeypatch):
    e, runner, reloads = _hung_isolated_embedder(monkeypatch)
    monkeypatch.setattr(emb, "_EMBEDDER", e)
    first, first_out = _start_in_flight(lambda: e.embed(["a"]))
    try:
        _wait_until(lambda: e.active_requests == 1)
        second, second_out = _start_in_flight(lambda: e.embed(["b"]))
        _wait_until(lambda: e.active_requests == 2)
        assert emb.reset_embedder(force=True) is True
        first.join(timeout=30)
        second.join(timeout=30)
    finally:
        runner.shutdown(grace=0)
    for out in (first_out, second_out):
        assert isinstance(out.get("error"), RuntimeError), out
        assert "was unloaded" in str(out["error"])
    assert reloads == []
    assert e.active_requests == 0


def test_forced_unload_all_stops_a_busy_gguf_reranker_and_counts_it(isolated, monkeypatch):
    e, runner, reloads = _hung_isolated_embedder(monkeypatch)
    e.effective_pooling = emb._POOLING_RANK
    e.cls_labels = []
    proc = runner._proc
    monkeypatch.setattr(emb, "loaded_dim", lambda: None)
    monkeypatch.setattr(rr, "_RERANKER", e)
    monkeypatch.setattr(rr, "_RERANKER_KEY", (("z", 0, 0), rr._RERANK))
    t, outcome = _start_in_flight(lambda: e.rerank([("q", "d")]))
    try:
        _wait_until(lambda: e.active_requests >= 1)
        asked = asyncio.run(hs.unload_all_models())
        assert asked["skipped_in_use"] == ["reranker model"], asked
        assert proc.is_alive()

        forced = asyncio.run(hs.unload_all_models(force=True))
        t.join(timeout=30)
    finally:
        runner.shutdown(grace=0)
    assert forced["unloaded_other"] == ["reranker model"], forced
    assert "skipped_in_use" not in forced
    assert rr._RERANKER is None and not proc.is_alive()
    assert isinstance(outcome.get("error"), RuntimeError), outcome
    assert reloads == []


def _hung_speech_engine(monkeypatch):
    """A SpeechEngine whose real worker process hangs inside 'speak' and does
    not answer a cancel."""
    monkeypatch.setenv(speech_runner_mod._FAULT_ENV, "speak-hang")
    eng = speech.SpeechEngine.__new__(speech.SpeechEngine)
    eng.model = SimpleNamespace(name="voice", path="Z:/models/voice.gguf")
    eng._rpc_lock = threading.Lock()
    eng._count_lock = threading.Lock()
    eng.active_requests = 0
    eng.broken = False
    runner = speech_runner_mod.SpeechRunner()
    runner._spawn()
    eng._runner = runner
    return eng, runner


def _speak(eng, text):
    eng.claim()
    try:
        return eng.speak(text)
    finally:
        eng.release()


def test_forced_unload_all_stops_a_busy_speech_model_and_counts_it(isolated, monkeypatch):
    from localm.inference.backends.llamacpp.mtmd_gen import SpeechCancelled
    eng, runner = _hung_speech_engine(monkeypatch)
    proc = runner._proc
    monkeypatch.setattr(speech, "_STOP_CANCEL_GRACE", 0.5)
    monkeypatch.setattr(emb, "loaded_dim", lambda: None)
    monkeypatch.setattr(rr, "is_loaded", lambda: False)
    monkeypatch.setattr(speech, "_ENGINE", eng)
    monkeypatch.setattr(speech, "_ENGINE_KEY", ("voice",))
    first, first_out = _start_in_flight(lambda: _speak(eng, "one"))
    try:
        _wait_until(lambda: eng.active_requests == 1 and eng._rpc_lock.locked())
        second, second_out = _start_in_flight(lambda: _speak(eng, "two"))
        _wait_until(lambda: eng.active_requests == 2)

        asked = asyncio.run(hs.unload_all_models())
        assert asked["skipped_in_use"] == ["speech model"], asked
        assert asked["confirm_required"] == {"speech model": "2 active requests"}
        assert proc.is_alive()

        forced = asyncio.run(hs.unload_all_models(force=True))
        first.join(timeout=30)
        second.join(timeout=30)
    finally:
        runner.shutdown(grace=0)
    assert forced["unloaded_other"] == ["speech model"], forced
    assert "skipped_in_use" not in forced
    assert speech._ENGINE is None and not proc.is_alive()
    assert isinstance(first_out.get("error"), RuntimeError), first_out
    assert isinstance(second_out.get("error"), SpeechCancelled), second_out
    assert "was unloaded" in str(second_out["error"])


class _BlockingHFBackend:
    """The HFBackend surface HFReranker uses, with a rerank that blocks until
    its runner is killed and then raises as a dead HF worker does."""

    def __init__(self):
        self.killed = threading.Event()
        self.unload_calls = 0
        self._runner = SimpleNamespace(kill=self.killed.set)

    @property
    def loaded(self):
        return not self.killed.is_set()

    def rerank(self, pairs):
        assert self.killed.wait(timeout=30)
        raise RuntimeError("The hf model process crashed (exit code 1)")

    def unload(self):
        self.unload_calls += 1


def test_a_forced_reset_stops_a_busy_hf_reranker(monkeypatch):
    model = rr.HFReranker.__new__(rr.HFReranker)
    model.model_path = "Z:/models/hf-rerank"
    model.active_requests = 0
    model._call_lock = threading.Lock()
    model._backend = _BlockingHFBackend()
    model._runner = model._backend._runner
    monkeypatch.setattr(rr, "_RERANKER", model)
    monkeypatch.setattr(rr, "_RERANKER_KEY", (("z", 0, 0), rr._RERANK))
    t, outcome = _start_in_flight(lambda: model.rerank([("q", "d")]))
    _wait_until(lambda: model.active_requests == 1 and model._call_lock.locked())
    assert rr.reset_reranker(force=True) is True
    t.join(timeout=30)
    assert isinstance(outcome.get("error"), rr.RerankerUnavailableError), outcome
    assert "was unloaded" in str(outcome["error"])
    assert model._backend.unload_calls == 1
    assert rr._RERANKER is None
    with pytest.raises(rr.RerankerUnavailableError):
        model.rerank([("q", "d")])


# --------------------------------------------------------------------------- #
#  Single-model unload of a busy embedding model honours force                  #
# --------------------------------------------------------------------------- #

def test_unload_one_busy_embedder_is_in_use_without_force_and_stopped_with_it(
        isolated, monkeypatch):
    monkeypatch.setattr("localm.config.load_registry",
                        lambda: {"embed": {"path": "Z:/models/embed.gguf"}})
    monkeypatch.setattr(emb, "loaded_path", lambda: "Z:/models/embed.gguf")
    monkeypatch.setattr(emb, "active_requests", lambda: 1)
    resets = []
    monkeypatch.setattr(emb, "reset_embedder",
                        lambda force=True: (resets.append(force), True)[1])
    with patch("localm.vram._vram_free_reading", return_value=(None, True, None)):
        asked = asyncio.run(hs.unload_one_model("embed"))
        forced = asyncio.run(hs.unload_one_model("embed", force=True))
    assert asked["status"] == "in_use" and resets == [True]
    assert forced["status"] == "unloaded"


# --------------------------------------------------------------------------- #
#  A reply cut off by a forced unload still ends with its usage block          #
# --------------------------------------------------------------------------- #

class _CutOffEngine:
    def __init__(self, loaded):
        self.loaded = loaded

    def count_tokens(self, text):
        raise RuntimeError("The model was unloaded while handling 'count_tokens'.")


def test_usage_of_a_reply_cut_off_by_an_unload_is_estimated():
    text = "x" * 40
    assert asyncio.run(hs._count_streamed_tokens(_CutOffEngine(loaded=False), text)) == 10
    assert asyncio.run(hs._count_streamed_tokens(_CutOffEngine(loaded=False), "")) == 0


def test_a_count_failure_on_a_loaded_engine_still_raises():
    with pytest.raises(RuntimeError, match="count_tokens"):
        asyncio.run(hs._count_streamed_tokens(_CutOffEngine(loaded=True), "abc"))
