# SPDX-License-Identifier: AGPL-3.0-or-later
"""switch_engine() tries the shared embedder and a busy resident model at most
once per load attempt.

Each test sets up the case where repeating the step would change what
happens: an embedder that reports itself loaded again right after every
reset, and a second busy model left over after the first was evicted.
"""

import asyncio

import pytest

from localm.inference import http_server as hs
from tests.test_eviction_victim_race import GatedEngine
from tests.test_switch_engine_transition_matrix import GB, _install, _seat


@pytest.fixture(autouse=True)
def _fresh_switch_state(monkeypatch):
    """Every test starts from an empty server and leaves one behind."""
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


def test_an_embedder_that_reloads_after_its_reset_is_reset_only_once(monkeypatch):
    b = GatedEngine("model-b")
    env = _install(monkeypatch, {"model-b": b}, total=3 * GB)
    resets = []

    def _reset_embedder(force=True):
        resets.append(force)
        return len(resets) <= 5

    monkeypatch.setattr("localm.inference.embedder.loaded_dim", lambda: 768)
    monkeypatch.setattr("localm.inference.embedder.reset_embedder", _reset_embedder)

    res = asyncio.run(hs.switch_engine("model-b", env.factory, preempt=False))

    assert resets == [False], "the embedder was reset more than once in one load"
    assert res == {"status": "loaded", "model": "model-b"}
    assert len(env.release_waits) == 1
    assert env.probes == 2


def test_an_explicit_switch_cancels_and_evicts_only_one_busy_model(monkeypatch):
    a = GatedEngine("model-a")
    b = GatedEngine("model-b")
    c = GatedEngine("model-c")
    env = _install(monkeypatch, {"model-a": a, "model-b": b, "model-c": c})
    _seat("model-a", a, busy=1)
    _seat("model-c", c, active=True, busy=1)

    res = asyncio.run(hs.switch_engine("model-b", env.factory, force=True))

    assert env.cancelled == ["model-a"], "a second busy model was interrupted"
    assert c.loaded and c.unload_calls == 0 and c.active_requests == 1
    assert "model-c" in hs._engines
    assert a.unload_calls == 1 and "model-a" not in hs._engines
    assert res == {"status": "loaded", "model": "model-b"}
    assert env.probes == 2
