# SPDX-License-Identifier: AGPL-3.0-or-later
"""The RAM-aware mmap decision for a GGUF load.

A load with every layer on the GPU reads its host-resident weights (the input
layer and any routed experts n_cpu_moe pins to system RAM) into memory when
they fit available RAM, and keeps the runtime's mmap default when they do not,
so a model whose CPU-resident part is larger than RAM pages from the file
instead of failing or swapping. A configured use_mmap of on or off always wins.

The decision runs against REAL GGUF files built byte by byte; only the system
RAM reading is patched. The load-param layer drives LlamaCpp.__init__ with the
real llama_model_params structs of every layout.
"""

from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp import llama as llama_mod
from localm.inference.backends.llamacpp._sizing import (
    HOST_RAM_HEADROOM_BYTES, MmapDecision, decide_use_mmap)
from localm.inference.backends.llamacpp._structs import (
    LLAMA_LOAD_MODE_AUTO, LLAMA_LOAD_MODE_MMAP, LLAMA_LOAD_MODE_NONE,
    LlamaModelParamsV1, LlamaModelParamsV2, LlamaModelParamsV3)
from tests.test_moe_auto_placement import EXPERT_BYTES, _dense_model, _moe_model

GIB = 1024 ** 3
PINNED_TWO = EXPERT_BYTES[0] + EXPERT_BYTES[1]


# --------------------------------------------------------------------------- #
#  decide_use_mmap: the rule itself                                           #
# --------------------------------------------------------------------------- #

class TestDecideUseMmap:
    @pytest.mark.parametrize("setting", ["on", "ON", " On "])
    @pytest.mark.parametrize("full", [True, False])
    def test_on_forces_mmap_whatever_the_ram(self, setting, full):
        d = decide_use_mmap(setting, full, 1, 64 * GIB, 64 * GIB)
        assert (d.use_mmap, d.reason) == (True, "user_on")

    @pytest.mark.parametrize("setting", ["off", "OFF"])
    @pytest.mark.parametrize("full", [True, False])
    def test_off_forces_no_mmap_even_when_the_weights_exceed_ram(self, setting, full):
        d = decide_use_mmap(setting, full, 100 * GIB, 8 * GIB, 1 * GIB)
        assert (d.use_mmap, d.reason) == (False, "user_off")

    def test_auto_full_offload_that_fits_reads_the_weights_into_memory(self):
        d = decide_use_mmap("auto", True, 4 * GIB, 32 * GIB, 16 * GIB)
        assert d == MmapDecision(False, "fits_ram", 4 * GIB, 32 * GIB, 16 * GIB)

    def test_auto_full_offload_that_exceeds_keeps_the_runtime_default(self):
        d = decide_use_mmap("auto", True, 40 * GIB, 32 * GIB, 16 * GIB)
        assert d == MmapDecision(None, "exceeds_ram", 40 * GIB, 32 * GIB, 16 * GIB)

    def test_the_headroom_boundary_is_inclusive(self):
        avail = 10 * GIB
        fits = decide_use_mmap("auto", True, avail - HOST_RAM_HEADROOM_BYTES, None, avail)
        over = decide_use_mmap("auto", True, avail - HOST_RAM_HEADROOM_BYTES + 1, None, avail)
        assert (fits.use_mmap, fits.reason) == (False, "fits_ram")
        assert (over.use_mmap, over.reason) == (None, "exceeds_ram")

    def test_unknown_ram_is_never_read_as_plenty(self):
        d = decide_use_mmap("auto", True, 1, None, None)
        assert (d.use_mmap, d.reason) == (None, "ram_unknown")

    def test_unknown_host_bytes_keep_the_runtime_default(self):
        d = decide_use_mmap("auto", True, None, 64 * GIB, 64 * GIB)
        assert (d.use_mmap, d.reason) == (None, "host_unknown")

    def test_auto_partial_offload_keeps_the_runtime_default(self):
        d = decide_use_mmap("auto", False, 100 * GIB, 8 * GIB, 1 * GIB)
        assert (d.use_mmap, d.reason) == (None, "partial_offload")

    @pytest.mark.parametrize("setting", [None, True, False, "", "maybe", 1])
    def test_anything_but_on_or_off_reads_as_auto(self, setting):
        d = decide_use_mmap(setting, True, 4 * GIB, 32 * GIB, 16 * GIB)
        assert (d.use_mmap, d.reason) == (False, "fits_ram")


# --------------------------------------------------------------------------- #
#  The sizing layer: host-resident bytes from the real file, RAM read once    #
# --------------------------------------------------------------------------- #

def _backend(path, **kw):
    return GgufBackend(str(path), n_ctx=64, **kw)


def _ram(total, available):
    return patch("localm.sysstats.system_ram", MagicMock(return_value=(total, available)))


class TestHostResidentBytes:
    def test_a_dense_full_offload_keeps_only_the_input_layer_in_ram(self, tmp_path):
        host = _backend(_dense_model(tmp_path))._host_resident_bytes()
        assert 5_000 <= host < 5_000 + 32   # token_embd, padded to the alignment

    def test_pinned_experts_add_exactly_their_bytes(self, tmp_path):
        path = _moe_model(tmp_path)
        none_pinned = _backend(path)._host_resident_bytes()
        two_pinned = _backend(path, n_cpu_moe=2)._host_resident_bytes()
        assert two_pinned - none_pinned == PINNED_TWO

    def test_an_auto_chosen_n_cpu_moe_counts_like_a_configured_one(self, tmp_path):
        b = _backend(_moe_model(tmp_path))
        b.effective_n_cpu_moe = 2
        assert b._host_resident_bytes() == _backend(
            _moe_model(tmp_path), n_cpu_moe=2)._host_resident_bytes()

    def test_pinned_experts_with_an_unreadable_header_are_unknown(self, tmp_path):
        bad = tmp_path / "bad.gguf"
        bad.write_bytes(b"\0" * 4096)
        assert _backend(bad, n_cpu_moe=2)._host_resident_bytes() is None

    def test_a_missing_file_is_unknown(self, tmp_path):
        assert _backend(tmp_path / "gone.gguf")._host_resident_bytes() is None


class TestResolveUseMmap:
    def test_moe_experts_that_fit_ram_load_unmapped(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        host = b._host_resident_bytes()
        with _ram(64 * GIB, host + HOST_RAM_HEADROOM_BYTES) as probe:
            d = b._resolve_use_mmap(99)
        assert d == MmapDecision(False, "fits_ram", host, 64 * GIB,
                                 host + HOST_RAM_HEADROOM_BYTES)
        assert b.last_mmap_decision == d
        assert probe.call_count == 1

    def test_moe_experts_larger_than_ram_keep_mmap(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        host = b._host_resident_bytes()
        with _ram(64 * GIB, host + HOST_RAM_HEADROOM_BYTES - 1):
            d = b._resolve_use_mmap(99)
        assert (d.use_mmap, d.reason, d.host_bytes) == (None, "exceeds_ram", host)

    def test_an_unreadable_ram_reading_keeps_mmap(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        with _ram(None, None):
            d = b._resolve_use_mmap(99)
        assert (d.use_mmap, d.reason) == (None, "ram_unknown")

    def test_a_dense_full_offload_with_room_stays_unmapped(self, tmp_path):
        b = _backend(_dense_model(tmp_path))
        with _ram(16 * GIB, 8 * GIB):
            d = b._resolve_use_mmap(99)
        assert (d.use_mmap, d.reason) == (False, "fits_ram")

    def test_a_dense_full_offload_on_a_full_box_keeps_mmap(self, tmp_path):
        b = _backend(_dense_model(tmp_path))
        with _ram(16 * GIB, 1 * GIB):
            d = b._resolve_use_mmap(99)
        assert (d.use_mmap, d.reason) == (None, "exceeds_ram")

    @pytest.mark.parametrize("setting, want", [("on", True), ("off", False)])
    def test_an_explicit_setting_wins_and_reads_no_ram(self, tmp_path, setting, want):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        b.use_mmap = setting
        free = 1 if setting == "off" else 1024 * GIB
        with _ram(64 * GIB, free) as probe:
            d = b._resolve_use_mmap(99)
        assert d.use_mmap is want
        assert probe.call_count == 0

    @pytest.mark.parametrize("layers", [0, 24, 98])
    def test_a_partial_offload_keeps_the_default_and_reads_no_ram(self, tmp_path, layers):
        b = _backend(_moe_model(tmp_path))
        with _ram(64 * GIB, 1) as probe:
            d = b._resolve_use_mmap(layers)
        assert (d.use_mmap, d.reason) == (None, "partial_offload")
        assert probe.call_count == 0

    def test_a_failing_host_byte_count_never_raises(self, tmp_path):
        b = _backend(_moe_model(tmp_path))
        with patch.object(GgufBackend, "_host_resident_bytes",
                          side_effect=ValueError("corrupt")), _ram(64 * GIB, 64 * GIB) as probe:
            d = b._resolve_use_mmap(99)
        assert (d.use_mmap, d.reason) == (None, "host_unknown")
        assert probe.call_count == 0


# --------------------------------------------------------------------------- #
#  GgufBackend._load_native: the decision reaches the worker, the report back #
# --------------------------------------------------------------------------- #

def _load(backend, meta_mmap):
    seen = {}

    def _spawn(params, **_kw):
        seen.update(params)
        return {"n_layers": 4, "kv_bytes_per_token": 0, "supports_images": False,
                "weight_placement": [], "moe_skip_reason": None, "mmap": meta_mmap}

    with patch("localm.discover.list_gpus", return_value=([], "ok")), \
         patch("localm.inference.backends.llamacpp._runner.ModelRunner."
               "spawn_and_load", side_effect=_spawn):
        backend._load_native()
    return seen


class TestLoadNativeWiring:
    def test_experts_larger_than_ram_reach_the_worker_as_the_default(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        with _ram(64 * GIB, 1 * GIB):
            params = _load(b, True)
        assert "use_mmap" in params and params["use_mmap"] is None
        assert b.effective_use_mmap is True
        assert b.mmap_forced_by_ram is True

    def test_experts_that_fit_reach_the_worker_as_forced_off(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        with _ram(64 * GIB, 32 * GIB):
            params = _load(b, None)
        assert params["use_mmap"] is False
        assert b.effective_use_mmap is False
        assert b.mmap_forced_by_ram is False

    def test_the_worker_report_wins_over_the_requested_mode(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        b.use_mmap = "on"
        with _ram(64 * GIB, 32 * GIB):
            params = _load(b, False)
        assert params["use_mmap"] is True
        assert b.effective_use_mmap is False
        assert b.mmap_forced_by_ram is False

    def test_an_unreported_default_stays_unknown(self, tmp_path):
        b = _backend(_moe_model(tmp_path), n_cpu_moe=2)
        with _ram(64 * GIB, 1 * GIB):
            _load(b, None)
        assert b.effective_use_mmap is None
        assert b.mmap_forced_by_ram is False

    def test_a_mapped_load_with_an_unknown_ram_reading_is_not_flagged(self, tmp_path):
        b = _backend(_moe_model(tmp_path))
        with _ram(None, None):
            _load(b, True)
        assert b.effective_use_mmap is True
        assert b.mmap_forced_by_ram is False


# --------------------------------------------------------------------------- #
#  LlamaCpp.__init__: the load mode written into the real struct              #
# --------------------------------------------------------------------------- #

def _native_default(cls):
    """A llama_model_params of *cls* with the native default load mode: V1
    use_mmap true, V2 (b10270) MMAP, V3 (b11118) AUTO."""
    mp = cls()
    if cls is LlamaModelParamsV1:
        mp.use_mmap = True
    elif cls is LlamaModelParamsV2:
        mp.load_mode = LLAMA_LOAD_MODE_MMAP
    else:
        mp.load_mode = LLAMA_LOAD_MODE_AUTO
    return mp


def _load_params(monkeypatch, cls, **kw):
    """The params LlamaCpp.__init__ hands llama_load_model_from_file for *kw*
    (the native load is faked to return NULL, ending __init__)."""
    seen = []
    monkeypatch.setattr(llama_mod.api, "llama_backend_init", lambda: None)
    monkeypatch.setattr(llama_mod.api, "llama_model_default_params",
                        lambda: _native_default(cls))
    monkeypatch.setattr(llama_mod.api, "llama_load_model_from_file",
                        lambda path, params: seen.append(params))
    with patch("localm.discover.apply_main_gpu", lambda mp, **_k: None), \
         patch("localm.discover.apply_gpu_split", lambda mp, ratios_override=None: None):
        with pytest.raises(RuntimeError, match="Failed to load model"):
            llama_mod.LlamaCpp("m.gguf", n_ctx=64, verbose=False, **kw)
    (mp,) = seen
    return mp


def _mode(mp):
    return mp.use_mmap if isinstance(mp, LlamaModelParamsV1) else mp.load_mode


LAYOUTS = [LlamaModelParamsV1, LlamaModelParamsV2, LlamaModelParamsV3]


class TestLoadParamLayer:
    @pytest.mark.parametrize("cls", LAYOUTS)
    def test_false_writes_no_mmap(self, monkeypatch, cls):
        mp = _load_params(monkeypatch, cls, n_gpu_layers=99, use_mmap=False)
        assert _mode(mp) == (False if cls is LlamaModelParamsV1 else LLAMA_LOAD_MODE_NONE)

    @pytest.mark.parametrize("cls", LAYOUTS)
    def test_true_writes_mmap(self, monkeypatch, cls):
        mp = _load_params(monkeypatch, cls, n_gpu_layers=24, use_mmap=True)
        assert _mode(mp) == (True if cls is LlamaModelParamsV1 else LLAMA_LOAD_MODE_MMAP)

    @pytest.mark.parametrize("cls", LAYOUTS)
    def test_none_at_full_offload_keeps_the_native_default(self, monkeypatch, cls):
        mp = _load_params(monkeypatch, cls, n_gpu_layers=99, use_mmap=None)
        assert _mode(mp) == _mode(_native_default(cls))


# --------------------------------------------------------------------------- #
#  _CapturedStderr.mapped: what the native load log says                      #
# --------------------------------------------------------------------------- #

def _mapped(tmp_path, text):
    p = tmp_path / "captured.log"
    p.write_text(text, encoding="utf-8")
    return llama_mod._CapturedStderr(str(p)).mapped()


class TestMappedFromTheLoadLog:
    @pytest.mark.parametrize("mode, want", [
        ("mmap", True), ("mmap+mlock", True),
        ("none", False), ("mlock", False), ("dio", False),
    ])
    def test_the_reported_load_mode(self, tmp_path, mode, want):
        line = (f"load_tensors: loading model tensors, this can take a while... "
                f"(load_mode = {mode})\n")
        assert _mapped(tmp_path, line) is want

    @pytest.mark.parametrize("value, want", [("true", True), ("false", False)])
    def test_the_older_mmap_flag_report(self, tmp_path, value, want):
        line = f"load_tensors: loading model tensors, this can take a while... (mmap = {value})\n"
        assert _mapped(tmp_path, line) is want

    def test_auto_falls_back_to_a_mapped_cpu_buffer(self, tmp_path):
        text = ("load_tensors: loading model tensors, this can take a while... "
                "(load_mode = auto)\n"
                "load_tensors:   CPU_Mapped model buffer size =   200.00 MiB\n")
        assert _mapped(tmp_path, text) is True

    def test_an_unsupported_platform_is_unmapped_whatever_was_asked(self, tmp_path):
        text = ("llama_model_loader: mmap is not supported on this platform\n"
                "load_tensors: loading model tensors, this can take a while... "
                "(load_mode = mmap)\n")
        assert _mapped(tmp_path, text) is False

    def test_nothing_reported_is_unknown(self, tmp_path):
        text = "load_tensors:        ROCm0 model buffer size =   3.35 MiB\n"
        assert _mapped(tmp_path, text) is None

    def test_a_missing_capture_is_unknown(self, tmp_path):
        assert llama_mod._CapturedStderr(str(tmp_path / "gone.log")).mapped() is None

    def test_adversarial_input_stays_linear_time(self, tmp_path):
        import time
        text = "(load_mode = " + "mmap" * 200_000 + "(mmap = " * 50_000
        start = time.perf_counter()
        assert _mapped(tmp_path, text) is None
        assert time.perf_counter() - start < 2.0
