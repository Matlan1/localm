# SPDX-License-Identifier: AGPL-3.0-or-later
"""The explicit-unload paths must honor the in-flight-request pin (AUDIT-CRIT-1).

A request pins its engine (`active_requests > 0`) for its whole lifetime; the
VRAM-eviction loop and the idle-unload loop both skip a pinned engine so it is
never unloaded out from under an in-flight request. `unload_all_models` and
`unload_one_model` (the owner `/v1/models/unload` route and the sibling-instance
cooperate-unload) did NOT check the pin - they would unload a model a streaming
request was mid-generation on, report the VRAM as "freed" (a lie: the request
immediately reloads it), and race a use-after-unload. These tests pin that they
now skip a pinned engine, consistent with the other eviction paths.
"""

import asyncio

import pytest

from localm.inference import http_server as hs


class FakeEngine:
    def __init__(self, name, active=0):
        self.display_name = name
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
    monkeypatch.setattr(hs, "_gpu_registry_sync", lambda: None)
    for d in (hs._engines, hs._engines_lru, hs._inference_sems,
              hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None
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


def test_unload_one_model_skips_pinned(isolated):
    e = _install("m", active=1)          # an in-flight request pinned it
    res = asyncio.run(hs.unload_one_model("m"))
    assert e.loaded, "a pinned engine must NOT be unloaded"
    assert e.unload_calls == 0
    assert res.get("status") == "in_use", res


def test_unload_one_model_unloads_idle(isolated):
    e = _install("m", active=0)
    res = asyncio.run(hs.unload_one_model("m"))
    assert not e.loaded and e.unload_calls == 1
    assert res.get("status") == "unloaded", res


def test_unload_all_models_skips_only_pinned(isolated):
    busy = _install("busy", active=1)
    idle = _install("idle", active=0)
    res = asyncio.run(hs.unload_all_models())
    assert busy.loaded and busy.unload_calls == 0, "pinned engine must survive unload-all"
    assert not idle.loaded and idle.unload_calls == 1, "idle engine should be unloaded"
    assert "idle" in res.get("unloaded_models", []), res
    assert "busy" not in res.get("unloaded_models", []), res
