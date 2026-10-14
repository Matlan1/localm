# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every unload path takes its before-unload VRAM reading off the event loop.

``localm.vram._vram_free_reading`` re-probes the GPU on every call and can take
up to ``discover._GPU_PROBE_DEADLINE`` seconds, so a call on the loop thread
stalls every other request and stream for that long."""
from __future__ import annotations

import asyncio
import threading

import pytest

from localm.inference import embedder as emb
from localm.inference import http_server as hs
from localm.inference import reranker as rr
from localm.inference import speech as sp

PATH = "Z:/models/target.gguf"


class _Engine:
    def __init__(self):
        self.display_name = "chat"
        self.loaded = True
        self.active_requests = 0
        self.unloading = False
        self.unloading_at_read = None
        self.gate_held_at_read = None

    def unload(self):
        self.loaded = False


@pytest.fixture
def reading(monkeypatch):
    """Patches the VRAM reading and records, per call, whether it ran on the
    thread driving the event loop."""
    calls = []
    engine = _Engine()

    def _reading():
        calls.append(threading.get_ident())
        engine.unloading_at_read = engine.unloading
        gate = hs._inference_sems.get("target")
        engine.gate_held_at_read = gate is not None and gate._exclusive
        return None, True, None

    monkeypatch.setattr("localm.vram._vram_free_reading", _reading)
    monkeypatch.setattr("localm.vram.wait_for_vram_release",
                        lambda free_fn, before_bytes=None: (0, before_bytes))
    monkeypatch.setattr("localm.config.load_registry",
                        lambda: {"target": {"path": PATH}})
    for d in (hs._engines, hs._engines_lru, hs._inference_sems,
              hs._last_activity_per_model):
        d.clear()
    monkeypatch.setattr(hs, "_active_model_name", None)
    monkeypatch.setattr(hs, "_engine", None)
    monkeypatch.setattr(hs, "_inference_sem", None)
    monkeypatch.setattr(emb, "loaded_dim", lambda: None)
    monkeypatch.setattr(emb, "loaded_path", lambda: None)
    monkeypatch.setattr(rr, "is_loaded", lambda: False)
    monkeypatch.setattr(rr, "reranker_info", lambda: None)
    monkeypatch.setattr(sp, "is_loaded", lambda: False)
    monkeypatch.setattr(sp, "speech_info", lambda: None)
    yield calls, engine
    for d in (hs._engines, hs._engines_lru, hs._inference_sems):
        d.clear()


def _run(coro_fn):
    loop_thread = []

    async def _main():
        loop_thread.append(threading.get_ident())
        return await coro_fn()

    result = asyncio.run(_main())
    return result, loop_thread[0]


def _assert_off_loop(calls, loop_thread):
    assert calls, "the unload path took no VRAM reading"
    assert loop_thread not in calls, "the VRAM reading ran on the event loop thread"


def test_unload_all_reads_off_the_loop(reading):
    calls, _ = reading
    _result, loop_thread = _run(hs.unload_all_models)
    _assert_off_loop(calls, loop_thread)


def test_unload_one_chat_engine_reads_off_the_loop_with_the_engine_flagged(reading):
    calls, engine = reading
    hs._engines["target"] = engine
    result, loop_thread = _run(lambda: hs.unload_one_model("target"))
    assert result["status"] == "unloaded"
    _assert_off_loop(calls, loop_thread)
    assert engine.unloading_at_read is True
    assert engine.gate_held_at_read is True, (
        "a request could take the model's gate during the reading's await")
    assert engine.unloading is False and engine.loaded is False


def test_unload_one_embedder_reads_off_the_loop(reading, monkeypatch):
    calls, _ = reading
    monkeypatch.setattr(emb, "loaded_path", lambda: PATH)
    monkeypatch.setattr(emb, "active_requests", lambda: 0)
    monkeypatch.setattr(emb, "reset_embedder", lambda force=True: True)
    result, loop_thread = _run(lambda: hs.unload_one_model("target"))
    assert result["status"] == "unloaded"
    _assert_off_loop(calls, loop_thread)


def test_unload_one_reranker_reads_off_the_loop(reading, monkeypatch):
    calls, _ = reading
    monkeypatch.setattr(rr, "reranker_info", lambda: {"path": PATH, "labels": []})
    monkeypatch.setattr(rr, "active_requests", lambda: 0)
    monkeypatch.setattr(rr, "reset_reranker", lambda force=True: True)
    result, loop_thread = _run(lambda: hs.unload_one_model("target"))
    assert result["status"] == "unloaded"
    _assert_off_loop(calls, loop_thread)


def test_unload_one_speech_model_reads_off_the_loop(reading, monkeypatch):
    calls, _ = reading
    monkeypatch.setattr(sp, "speech_info", lambda: {"path": PATH})
    monkeypatch.setattr(sp, "active_requests", lambda: 0)
    monkeypatch.setattr(sp, "reset_speech", lambda force=True: True)
    result, loop_thread = _run(lambda: hs.unload_one_model("target"))
    assert result["status"] == "unloaded"
    _assert_off_loop(calls, loop_thread)
