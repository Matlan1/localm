# SPDX-License-Identifier: AGPL-3.0-or-later
"""An engine sizes its NEXT load, not its last one.

After a first load, ``Engine._load_locked`` builds a new backend for every
reload, so a setting changed in the config since then (here ``n_cpu_moe``)
applies to the reload. The figures switch admission reads before that reload
(``gpu_offload_needs`` and its siblings) come from a backend built the same
way, so they describe the load being admitted.

The model is a real GGUF file built byte by byte; the sizing runs for real
against it and only the VRAM readings and the native load are patched.
"""

import asyncio
from unittest.mock import patch

import pytest

from localm.config import update_config
from localm.inference import http_server as hs
from localm.inference import switch_admission
from localm.inference.backends.gguf import GgufBackend
from localm.inference.engine import Engine
from tests.test_moe_auto_placement import _moe_model, _Vram

FREE = 10**9


def _set_n_cpu_moe(value: int) -> None:
    def mutate(cfg):
        cfg["n_cpu_moe"] = value
    update_config(mutate)


def _loaded_once(path) -> Engine:
    """An engine whose backend went through one load (native load patched)
    and is unloaded again, as one stays in the server after an unload."""
    engine = Engine(str(path))
    with patch.object(GgufBackend, "load", lambda self: None):
        engine._load_locked()
    assert engine._backend_fresh is False
    return engine


def _fresh_needs(path):
    with _Vram(FREE):
        return Engine(str(path)).gpu_offload_needs()


@pytest.mark.parametrize("before,after", [(2, 0), (0, 2)])
def test_offload_needs_follow_the_config_of_the_next_load(tmp_path, before, after):
    path = _moe_model(tmp_path)
    _set_n_cpu_moe(after)
    want = _fresh_needs(path)
    _set_n_cpu_moe(before)
    stale = _fresh_needs(path)
    assert want != stale
    engine = _loaded_once(path)
    _set_n_cpu_moe(after)
    with _Vram(FREE):
        assert engine.gpu_offload_needs() == want
        assert engine.full_offload_vram_bytes() == want[0]
        assert engine.predict_placement()["n_cpu_moe"] == after


def test_a_backend_never_loaded_is_sized_as_built(tmp_path):
    path = _moe_model(tmp_path)
    _set_n_cpu_moe(2)
    engine = Engine(str(path))
    built = engine._backend
    _set_n_cpu_moe(0)
    with _Vram(FREE):
        assert engine.gpu_offload_needs() == built.gpu_offload_needs()
    assert engine._backend is built


def test_sizing_leaves_the_engines_backend_in_place(tmp_path):
    path = _moe_model(tmp_path)
    engine = _loaded_once(path)
    backend = engine._backend
    _set_n_cpu_moe(2)
    with _Vram(FREE):
        engine.gpu_offload_needs()
        engine.predict_placement()
        engine.configured_gpu_layers()
        engine.context_reserve_limit()
    assert engine._backend is backend
    assert engine._backend_fresh is False


@pytest.mark.parametrize("before,after", [(2, 0), (0, 2)])
def test_switch_admission_bar_is_the_next_loads_need(tmp_path, before, after):
    path = _moe_model(tmp_path)
    _set_n_cpu_moe(after)
    want = _fresh_needs(path)
    _set_n_cpu_moe(before)
    engine = _loaded_once(path)
    _set_n_cpu_moe(after)
    budget = switch_admission.LoadBudget(
        name="moe", vram_required=1, headroom=0, resident_cap=1,
        pinned=frozenset(), check_split_fit=True)

    async def admit():
        return await hs._switch_with_backend_need(asyncio.get_running_loop(),
                                                  budget, engine)

    with _Vram(FREE):
        sized = asyncio.run(admit())
    assert sized.backend_need == want[0]
