# SPDX-License-Identifier: AGPL-3.0-or-later
"""switch_engine() does its registry, config and model-directory reads, and
builds the engine, on a worker thread.

Those reads walk the model directory with a stat per entry, so running them
inline stalls every other request on the event loop for as long as the disk
takes. Each test records the thread the blocking work really ran on and
compares it with the event loop's own thread.
"""

import asyncio
import threading

import pytest

from localm.inference import http_server as hs
from localm.inference import residency
from tests.test_eviction_victim_race import GatedEngine
from tests.test_switch_engine_transition_matrix import _install, _seat


@pytest.fixture(autouse=True)
def _fresh_switch_state(monkeypatch):
    def _clear():
        for d in (hs._engines, hs._engines_lru, hs._inference_sems,
                  hs._last_activity_per_model, hs._evicting_names):
            d.clear()

    _clear()
    for name, value in (("_active_model_name", None),
                        ("_last_active_model_name", None),
                        ("_engine", None),
                        ("_inference_sem", None),
                        ("_switch_desired", None),
                        ("_switch_loading", None),
                        ("_switch_cancel", None),
                        ("_hang_alarm_instance", None),
                        ("_gpu_coord", None),
                        ("_engine_factory", hs._engine_factory),
                        ("_INCONCLUSIVE_LOAD_RETRY_DELAY", 0)):
        monkeypatch.setattr(hs, name, value)
    yield
    _clear()


def _record_threads(monkeypatch, env, seen):
    """Record the thread each blocking step runs on, into *seen*."""
    real_footprint = residency.model_footprint_bytes

    def _footprint(path):
        seen.setdefault("footprint", []).append(threading.get_ident())
        return real_footprint(path)

    monkeypatch.setattr(residency, "model_footprint_bytes", _footprint)

    inner = env.factory

    def _factory(name):
        seen.setdefault("factory", []).append(threading.get_ident())
        return inner(name)

    return _factory


def test_a_load_reads_the_model_directory_and_builds_the_engine_off_the_loop(monkeypatch):
    b = GatedEngine("model-b")
    env = _install(monkeypatch, {"model-b": b})
    seen = {}
    factory = _record_threads(monkeypatch, env, seen)

    async def _go():
        loop_thread = threading.get_ident()
        res = await hs.switch_engine("model-b", factory, preempt=False)
        return loop_thread, res

    loop_thread, res = asyncio.run(_go())

    assert res["status"] == "loaded"
    assert seen["footprint"], "the model directory was never sized"
    assert seen["factory"], "the engine was never built"
    assert loop_thread not in seen["footprint"], "the directory walk ran on the event loop"
    assert loop_thread not in seen["factory"], "the engine was built on the event loop"


@pytest.mark.parametrize("install_kwargs", [
    {"suffix": ".safetensors"},
    {"registered": ()},
], ids=["not-a-gguf", "empty-registry"])
def test_a_load_without_a_split_check_builds_the_engine_off_the_loop(
        monkeypatch, install_kwargs):
    b = GatedEngine("model-b")
    env = _install(monkeypatch, {"model-b": b}, **install_kwargs)
    seen = {}
    factory = _record_threads(monkeypatch, env, seen)

    async def _go():
        loop_thread = threading.get_ident()
        res = await hs.switch_engine("model-b", factory, preempt=False)
        return loop_thread, res

    loop_thread, res = asyncio.run(_go())

    assert res["status"] == "loaded"
    assert env.built == ["model-b"]
    assert loop_thread not in seen["factory"], "the engine was built on the event loop"


def test_a_load_that_needs_the_split_check_builds_once_and_off_the_loop(monkeypatch):
    b = GatedEngine("model-b")
    env = _install(monkeypatch, {"model-b": b})
    seen = {}
    factory = _record_threads(monkeypatch, env, seen)

    async def _go():
        loop_thread = threading.get_ident()
        res = await hs.switch_engine("model-b", factory, preempt=False)
        return loop_thread, res

    loop_thread, res = asyncio.run(_go())

    assert res["status"] == "loaded"
    assert env.built == ["model-b"], "the engine was built more than once"
    assert loop_thread not in seen["factory"]


def test_a_placement_reload_sizes_the_model_off_the_loop(monkeypatch):
    b = GatedEngine("model-b")
    env = _install(monkeypatch, {"model-b": b})
    _seat("model-b", b)
    seen = {}
    _record_threads(monkeypatch, env, seen)
    monkeypatch.setattr(hs, "_placement_heal_due", lambda name, engine: True)

    async def _go():
        loop_thread = threading.get_ident()
        await hs.switch_engine("model-b", env.factory, preempt=False)
        return loop_thread

    loop_thread = asyncio.run(_go())

    assert seen.get("footprint"), "the heal path never sized the model"
    assert loop_thread not in seen["footprint"], \
        "the heal path walked the model directory on the event loop"
