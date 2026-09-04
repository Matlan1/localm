# SPDX-License-Identifier: AGPL-3.0-or-later
"""An engine resolves its config-derived load parameters (context window, GPU
layers, MoE split, VRAM overhead) when it is CONSTRUCTED, not when it loads.

switch_engine keeps an engine in ``_engines`` after an idle unload. Reusing that
object for the next load of the same name re-applies the parameters it was built
with, so a setting saved in between never reaches the model - while the Settings
page states that engine values apply on the next model load. These pin that a
cached engine which is neither loaded nor mid-unload is rebuilt from the factory,
and that a loaded or unloading one is still reused.
"""

import asyncio

import pytest

import localm.inference.http_server as hs


class _StubEngine:
    """Records the config value visible when it was CONSTRUCTED, the way a real
    Engine captures n_ctx through create_backend at construction time."""

    def __init__(self, name, ctx_at_build):
        self.display_name = name
        self.ctx_at_build = ctx_at_build
        self.loaded = False
        self.unloading = False
        self.gpu_placement = None
        self.load_calls = 0

    def load(self):
        self.load_calls += 1
        self.loaded = True

    def set_load_cancel(self, ev):
        self._cancel = ev


@pytest.fixture()
def engine_bed(monkeypatch):
    """Isolate switch_engine's module globals and give it a factory whose
    engines capture a mutable 'config' value at construction."""
    cfg = {"n_ctx": 5150}
    built = []

    def factory(name):
        eng = _StubEngine(name, cfg["n_ctx"])
        built.append(eng)
        return eng

    monkeypatch.setattr(hs, "_engines", {}, raising=False)
    monkeypatch.setattr(hs, "_engines_lru", [], raising=False)
    monkeypatch.setattr(hs, "_inference_sems", {}, raising=False)
    monkeypatch.setattr(hs, "_last_activity_per_model", {}, raising=False)
    monkeypatch.setattr(hs, "_evicting_names", set(), raising=False)
    monkeypatch.setattr(hs, "_active_model_name", None, raising=False)
    monkeypatch.setattr(hs, "_engine", None, raising=False)
    # The registry-driven eviction/VRAM block is skipped for an empty registry.
    monkeypatch.setattr("localm.config.load_registry", lambda: {})
    monkeypatch.setattr("localm.model_manager.get_model_info", lambda n: None)
    monkeypatch.setattr(hs, "_gpu_registry_sync", lambda: None, raising=False)
    return cfg, built, factory


def test_unloaded_engine_is_rebuilt_so_config_changes_apply(engine_bed):
    cfg, built, factory = engine_bed

    first = asyncio.run(hs.switch_engine("m", factory))
    assert first["status"] == "loaded"
    assert built[0].ctx_at_build == 5150

    # Idle unload leaves the engine in _engines, exactly as the server does.
    built[0].loaded = False

    cfg["n_ctx"] = 3000
    second = asyncio.run(hs.switch_engine("m", factory))
    assert second["status"] == "loaded"

    # THE PROPERTY: the load that just happened used the CURRENT config.
    assert hs._engines["m"].ctx_at_build == 3000, (
        "reloading an idle-unloaded model reused the engine built under the old "
        f"config (ctx_at_build={hs._engines['m'].ctx_at_build}, want 3000)")
    assert len(built) == 2, "the engine should have been rebuilt, not reused"


def test_a_loaded_engine_is_still_reused(engine_bed):
    """The fast path must not start rebuilding engines that are resident: the
    point of the cache is the loaded weights."""
    cfg, built, factory = engine_bed

    asyncio.run(hs.switch_engine("m", factory))
    cfg["n_ctx"] = 3000
    again = asyncio.run(hs.switch_engine("m", factory))

    assert again["status"] == "already_active"
    assert len(built) == 1
    assert hs._engines["m"].load_calls == 1


def test_an_unloading_engine_is_not_replaced_mid_flight(engine_bed):
    """A rebuild while the old object is still freeing its VRAM would put two
    copies of one model in flight."""
    cfg, built, factory = engine_bed

    asyncio.run(hs.switch_engine("m", factory))
    hs._engines["m"].loaded = False
    hs._engines["m"].unloading = True

    cfg["n_ctx"] = 3000
    asyncio.run(hs.switch_engine("m", factory))

    assert len(built) == 1, "an unloading engine must be reused, not rebuilt"


def test_a_model_the_factory_cannot_rebuild_keeps_its_object(engine_bed):
    """A model served by direct path is not in the registry, so the factory
    raises for it and there is no rebuild to be had. It must keep the object it
    already has rather than losing the ability to reload at all."""
    cfg, built, factory = engine_bed

    asyncio.run(hs.switch_engine("served-by-path", factory))
    kept = hs._engines["served-by-path"]
    kept.loaded = False

    def exploding_factory(name):
        raise ValueError(f"Model not found: {name}")

    res = asyncio.run(hs.switch_engine("served-by-path", exploding_factory))

    assert res["status"] == "loaded"
    assert hs._engines["served-by-path"] is kept, (
        "an engine the factory cannot rebuild must keep its kept object")
    assert kept.load_calls == 2
