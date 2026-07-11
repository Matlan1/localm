# SPDX-License-Identifier: AGPL-3.0-or-later
"""BUG-2: a large GGUF that fills VRAM must keep generating, not crash.

A model whose weights nearly fill the card was loaded with its KV cache in VRAM,
leaving no room for the first decode's compute buffers; on the Vulkan backend that
faults with a native C++ crash (0xe06d7363) instead of spilling to RAM (as ROCm
does). Root cause: a file-size KV heuristic under-counted the real KV cache ~2.6x,
so the load "fit" per the estimate yet overflowed VRAM for real.

These tests cover the fix, WITHOUT the native DLL (the api module / metadata are
mocked): the architecture-accurate per-token KV size, the load-time offload_kqv
decision (keep KV in system RAM when it will not fit alongside compute), that the
placement is sticky across a grow, and that the grow check now reasons with the
accurate size. The real-runtime symbol export + real generation are covered by the
@integration tests at the bottom and by the on-hardware Vulkan run.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.llamacpp import _api
from localm.inference.backends.llamacpp.llama import LlamaCpp
from tests._fake_batch import fake_batch_init

_LOAD_LIB = "localm.inference.backends.llamacpp._api.load_lib"
_API = "localm.inference.backends.llamacpp.llama.api"


# --------------------------------------------------------------------------- #
#  _api bindings: has_kv_head_api() + the two accessors (load_lib mocked)
# --------------------------------------------------------------------------- #

class TestHasKvHeadApi:
    def test_true_when_both_symbols_present(self):
        fake_lib = MagicMock(spec=["llama_model_n_head", "llama_model_n_head_kv"])
        with patch(_LOAD_LIB, return_value=fake_lib):
            assert _api.has_kv_head_api() is True

    def test_false_when_head_kv_absent(self):
        # A build that exports n_head but not n_head_kv must NOT be treated as
        # having the API (a partial export is unusable for the KV formula).
        fake_lib = MagicMock(spec=["llama_model_n_head"])
        with patch(_LOAD_LIB, return_value=fake_lib):
            assert _api.has_kv_head_api() is False

    def test_false_when_both_absent(self):
        fake_lib = MagicMock(spec=[])
        with patch(_LOAD_LIB, return_value=fake_lib):
            assert _api.has_kv_head_api() is False


class TestKvHeadBindings:
    def test_n_head_calls_through_and_returns_int(self):
        fake_fn = MagicMock(return_value=32)
        fake_lib = MagicMock(spec=["llama_model_n_head"])
        fake_lib.llama_model_n_head = fake_fn
        with patch(_LOAD_LIB, return_value=fake_lib):
            assert _api.llama_model_n_head(1234) == 32
        fake_fn.assert_called_once_with(1234)

    def test_n_head_kv_calls_through_and_returns_int(self):
        fake_fn = MagicMock(return_value=8)
        fake_lib = MagicMock(spec=["llama_model_n_head_kv"])
        fake_lib.llama_model_n_head_kv = fake_fn
        with patch(_LOAD_LIB, return_value=fake_lib):
            assert _api.llama_model_n_head_kv(1234) == 8
        fake_fn.assert_called_once_with(1234)


# --------------------------------------------------------------------------- #
#  _read_kv_bytes_per_token(): the architecture-accurate formula
# --------------------------------------------------------------------------- #

def _bare_model(n_layers=32, model_ptr=111) -> LlamaCpp:
    llm = LlamaCpp.__new__(LlamaCpp)
    llm._model_ptr = model_ptr
    llm.n_layers = n_layers
    return llm


def _fake_api(*, has=True, n_embd=4096, n_head=32, n_head_kv=8):
    m = MagicMock()
    m.has_kv_head_api.return_value = has
    m.llama_model_n_embd.return_value = n_embd
    m.llama_model_n_head.return_value = n_head
    m.llama_model_n_head_kv.return_value = n_head_kv
    return m


class TestReadKvBytesPerToken:
    def test_computes_from_architecture(self):
        # 32 layers, n_embd 4096, n_head 32 -> head_dim 128, n_head_kv 8 (GQA).
        # per token = 32 * 8 * 128 * 2 (K+V) * 2 (f16 bytes) = 131072 = 128 KiB.
        llm = _bare_model(n_layers=32)
        with patch(_API, _fake_api(n_embd=4096, n_head=32, n_head_kv=8)):
            assert llm._read_kv_bytes_per_token() == 131072

    def test_gqa_is_smaller_than_full_multi_head(self):
        # n_head_kv < n_head (grouped-query) yields a proportionally smaller KV;
        # the whole point of using n_head_kv and not n_head.
        llm = _bare_model(n_layers=40)
        gqa = _fake_api(n_embd=5120, n_head=40, n_head_kv=8)
        mha = _fake_api(n_embd=5120, n_head=40, n_head_kv=40)
        with patch(_API, gqa):
            small = llm._read_kv_bytes_per_token()
        with patch(_API, mha):
            big = llm._read_kv_bytes_per_token()
        assert small * 5 == big          # 8 vs 40 KV heads

    def test_zero_when_api_absent(self):
        llm = _bare_model()
        with patch(_API, _fake_api(has=False)):
            assert llm._read_kv_bytes_per_token() == 0

    def test_zero_when_layers_unknown(self):
        llm = _bare_model(n_layers=None)
        with patch(_API, _fake_api()):
            assert llm._read_kv_bytes_per_token() == 0

    def test_zero_on_bogus_metadata(self):
        # A build that returns 0 for a head count (unreadable) must fall back to
        # the estimate (0), not divide-by-zero or produce a nonsense KV size.
        llm = _bare_model()
        with patch(_API, _fake_api(n_head=0)):
            assert llm._read_kv_bytes_per_token() == 0

    def test_zero_when_probe_raises(self):
        m = MagicMock()
        m.has_kv_head_api.side_effect = RuntimeError("native probe exploded")
        llm = _bare_model()
        with patch(_API, m):
            assert llm._read_kv_bytes_per_token() == 0


# --------------------------------------------------------------------------- #
#  _initial_offload_kqv(): where the INITIAL context's KV cache lives
# --------------------------------------------------------------------------- #

def _bare_offload(kv_per_token, free_fn) -> LlamaCpp:
    llm = LlamaCpp.__new__(LlamaCpp)
    llm.kv_bytes_per_token = kv_per_token
    llm._free_vram_fn = free_fn
    return llm


GB = 1024 ** 3


class TestInitialOffloadKqv:
    @pytest.fixture(autouse=True)
    def _force_vulkan(self, monkeypatch):
        # The preemptive offload is gated on the Vulkan backend (the only one that
        # crashes); these tests exercise the offload MATH, so force the gate on.
        # test_non_vulkan_backend_keeps_kv_in_vram overrides it to check the gate.
        from localm.inference.backends.llamacpp import _loader
        monkeypatch.setattr(_loader, "gpu_backend_is_vulkan", lambda: True)

    def test_keeps_vram_when_kv_size_unknown(self):
        # No accurate KV size (stripped build): keep the VRAM default rather than
        # needlessly slowing a model that may well fit.
        llm = _bare_offload(0, lambda: 200_000)
        assert llm._initial_offload_kqv(4096) is True

    def test_keeps_vram_when_no_free_reader(self):
        llm = _bare_offload(200_000, None)
        assert llm._initial_offload_kqv(4096) is True

    def test_keeps_vram_when_free_unmeasurable(self):
        llm = _bare_offload(200_000, lambda: None)
        assert llm._initial_offload_kqv(4096) is True

    def test_keeps_vram_when_reader_raises(self):
        def _boom():
            raise RuntimeError("torch fell over")
        llm = _bare_offload(200_000, _boom)
        assert llm._initial_offload_kqv(4096) is True

    def test_moves_kv_to_ram_when_model_fills_vram(self):
        # The crash case: gemma12b-scale KV (~320 KiB/token) at 4096 tokens is
        # ~1.25 GB; plus the ~3 GB compute reserve that far exceeds the 0.2 GB
        # free after weights -> KV cache goes to system RAM (return False).
        llm = _bare_offload(327_680, lambda: int(0.2 * GB))
        assert llm._initial_offload_kqv(4096) is False

    def test_keeps_kv_in_vram_for_a_model_that_fits(self):
        # A ~7B with plenty of headroom: whole KV (~0.8 GB) + 3 GB reserve fits
        # comfortably in 11 GB free -> KV stays in VRAM (full speed), no needless
        # RAM offload.
        llm = _bare_offload(200_000, lambda: int(11 * GB))
        assert llm._initial_offload_kqv(4096) is True

    def test_boundary_reserve_is_respected(self):
        # whole_kv + reserve must be strictly greater than free to move to RAM.
        llm = _bare_offload(200_000, None)
        llm._free_vram_fn = lambda: 4096 * 200_000 + LlamaCpp._COMPUTE_RESERVE_BYTES
        assert llm._initial_offload_kqv(4096) is True          # exactly fits -> VRAM
        llm._free_vram_fn = lambda: 4096 * 200_000 + LlamaCpp._COMPUTE_RESERVE_BYTES - 1
        assert llm._initial_offload_kqv(4096) is False         # one byte short -> RAM

    def test_logs_a_hint_when_moving_to_ram(self):
        llm = _bare_offload(327_680, lambda: int(0.2 * GB))
        fake_logger = MagicMock()
        with patch("localm.debuglog.logger", fake_logger):
            assert llm._initial_offload_kqv(4096) is False
        assert fake_logger.warning.called
        msg = fake_logger.warning.call_args[0][0]
        assert "system RAM" in msg

    def test_non_vulkan_backend_keeps_kv_in_vram(self, monkeypatch):
        # ROCm/CUDA/Metal never had the Vulkan compute-buffer crash; even a model
        # that fills VRAM must KEEP its KV cache in VRAM (return True) so it runs
        # full-speed as before - the fix must not regress those backends.
        from localm.inference.backends.llamacpp import _loader
        monkeypatch.setattr(_loader, "gpu_backend_is_vulkan", lambda: False)
        llm = _bare_offload(327_680, lambda: int(0.2 * GB))   # would offload on Vulkan
        assert llm._initial_offload_kqv(4096) is True


class TestGpuBackendIsVulkan:
    def test_false_when_lib_not_loaded(self, monkeypatch):
        from localm.inference.backends.llamacpp import _loader
        monkeypatch.setattr(_loader, "_loaded_lib", None)
        assert _loader.gpu_backend_is_vulkan() is False

    def test_true_for_vulkan_device(self, monkeypatch):
        from localm.inference.backends.llamacpp import _loader
        monkeypatch.setattr(_loader, "_loaded_lib", object())
        monkeypatch.setattr(_loader, "compute_devices",
                            lambda: [("CPU", 0), ("Vulkan0", 1)])
        assert _loader.gpu_backend_is_vulkan() is True

    def test_false_for_rocm_device(self, monkeypatch):
        from localm.inference.backends.llamacpp import _loader
        monkeypatch.setattr(_loader, "_loaded_lib", object())
        monkeypatch.setattr(_loader, "compute_devices",
                            lambda: [("ROCm0", 1), ("CPU", 0)])
        assert _loader.gpu_backend_is_vulkan() is False


# --------------------------------------------------------------------------- #
#  Sticky placement: a RAM-placed KV cache stays in RAM across a grow
# --------------------------------------------------------------------------- #

def _bare_grow(offload_kqv, vram_check) -> LlamaCpp:
    llm = LlamaCpp.__new__(LlamaCpp)
    llm._n_ctx = 4096
    llm._n_ctx_max = None
    llm._n_ctx_grow = 4096
    llm._cached_tokens = []
    llm._ctx_capacity = 4096
    llm._ctx_ptr = 222
    llm._model_ptr = 111
    llm._tokenizer = MagicMock()
    llm._vram_check = vram_check
    llm._offload_kqv = offload_kqv
    return llm


def _grow_api():
    m = MagicMock()
    cp = MagicMock()
    m.llama_context_default_params.return_value = cp
    m.llama_init_from_model.return_value = 444
    m.llama_decode.return_value = 0
    m.llama_batch_init.side_effect = fake_batch_init
    return m, cp


class TestStickyRamOnGrow:
    def test_ram_placement_sticks_and_skips_the_hook(self):
        # KV already in system RAM (a big model). A grow must KEEP it in RAM and
        # must NOT re-consult the hook (a larger context cannot suddenly fit VRAM;
        # moving it back risks the very crash the RAM offload avoids).
        hook = MagicMock(return_value=True)   # would say "VRAM fits" if asked
        llm = _bare_grow(offload_kqv=False, vram_check=hook)
        m, cp = _grow_api()
        with patch(_API, m):
            llm._prefill_fresh_context(list(range(100)), needed=5000)
        assert cp.offload_kqv is False        # stayed in RAM
        assert llm._offload_kqv is False      # and remembered
        hook.assert_not_called()              # not reconsidered

    def test_vram_placement_still_consults_the_hook(self):
        # KV in VRAM: the shipped behaviour is unchanged - the hook decides, and a
        # False flips it to RAM and is then remembered.
        hook = MagicMock(return_value=False)
        llm = _bare_grow(offload_kqv=True, vram_check=hook)
        m, cp = _grow_api()
        with patch(_API, m):
            llm._prefill_fresh_context(list(range(100)), needed=5000)
        hook.assert_called_once_with(8192, 4096)
        assert cp.offload_kqv is False
        assert llm._offload_kqv is False      # transition VRAM->RAM remembered

    def test_vram_stays_vram_when_hook_approves(self):
        hook = MagicMock(return_value=True)
        llm = _bare_grow(offload_kqv=True, vram_check=hook)
        m, cp = _grow_api()
        with patch(_API, m):
            llm._prefill_fresh_context(list(range(100)), needed=5000)
        assert cp.offload_kqv is True
        assert llm._offload_kqv is True


# --------------------------------------------------------------------------- #
#  Grow check reasons with the accurate KV size (GgufBackend._check_context_fit)
# --------------------------------------------------------------------------- #

class _StubLlm:
    def __init__(self, kv):
        self.kv_bytes_per_token = kv


def _gguf_backend(tmp_path, size_bytes, n_ctx=4096, n_gpu_layers=99):
    from localm.inference.backends.gguf import GgufBackend
    f = tmp_path / "model.gguf"
    with open(f, "wb") as fh:
        fh.truncate(size_bytes)
    return GgufBackend(str(f), n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)


class TestGrowCheckUsesAccurateKv:
    """The accurate per-token KV (LlamaCpp.kv_bytes_per_token) must OVERRIDE the
    file-size heuristic in _check_context_fit, so a KV cache that really would
    overflow VRAM is placed in system RAM (False) even when the under-counting
    heuristic would have called it a fit (True)."""

    def test_accurate_kv_flips_a_heuristic_fit_to_ram(self, tmp_path):
        # 9 GB model -> heuristic per_token 90_000; delta(4096->8192)=~369 MB.
        # Free = 500 MB: the heuristic alone says True (the shipped behaviour).
        b = _gguf_backend(tmp_path, size_bytes=9_000_000_000, n_ctx=4096)
        with patch.object(type(b), "_free_vram_bytes", return_value=500_000_000):
            b._llm = None                              # no accurate value yet
            assert b._check_context_fit(8192, current_ctx=4096) is True
            # The real model reports a much larger KV: 200_000 B/token ->
            # delta = 4096 * 200_000 = ~819 MB > 500 MB free -> RAM (False).
            b._llm = _StubLlm(200_000)
            assert b._check_context_fit(8192, current_ctx=4096) is False

    def test_accurate_kv_used_when_present(self, tmp_path):
        # Even when free would cover the heuristic, the accurate (larger) value is
        # the one charged, so the decision reflects real KV cost.
        b = _gguf_backend(tmp_path, size_bytes=2_000_000_000, n_ctx=4096)
        b._llm = _StubLlm(300_000)                     # accurate, large
        # delta = 4096 * 300_000 = ~1.23 GB. Free 1.0 GB -> does not fit -> RAM.
        with patch.object(type(b), "_free_vram_bytes", return_value=1_000_000_000):
            assert b._check_context_fit(8192, current_ctx=4096) is False
        # Free 2.0 GB -> fits -> VRAM.
        with patch.object(type(b), "_free_vram_bytes", return_value=2_000_000_000):
            assert b._check_context_fit(8192, current_ctx=4096) is True


# --------------------------------------------------------------------------- #
#  Backend-view free-VRAM signal (loader.gpu_memory) + _free_vram_bytes prefers it
# --------------------------------------------------------------------------- #

class TestGpuMemory:
    """loader.gpu_memory() reads free VRAM from the ACTIVE ggml backend
    (ggml_backend_dev_memory) - the runtime that allocates the model - so the
    offload decision works without torch and matches the backend's own budget."""

    def test_none_when_lib_not_loaded(self, monkeypatch):
        from localm.inference.backends.llamacpp import _loader
        # Do NOT force-load the native lib just to measure (a preflight before the
        # first load falls back to torch).
        monkeypatch.setattr(_loader, "_loaded_lib", None)
        assert _loader.gpu_memory() is None

    def test_queries_backend_when_resolved(self, monkeypatch):
        from localm.inference.backends.llamacpp import _loader
        monkeypatch.setattr(_loader, "_loaded_lib", object())   # lib "loaded"

        def fake_mem(dev, free_p, total_p):
            free_p._obj.value = 3 * 1024 ** 3       # byref(x)._obj is x
            total_p._obj.value = 16 * 1024 ** 3

        # Pre-resolved cache: (device handle, bound ggml_backend_dev_memory).
        monkeypatch.setattr(_loader, "_gpu_mem_cache", (object(), fake_mem))
        assert _loader.gpu_memory() == (3 * 1024 ** 3, 16 * 1024 ** 3)

    def test_none_when_unavailable_sentinel(self, monkeypatch):
        from localm.inference.backends.llamacpp import _loader
        monkeypatch.setattr(_loader, "_loaded_lib", object())
        monkeypatch.setattr(_loader, "_gpu_mem_cache", False)   # resolved: no GPU
        assert _loader.gpu_memory() is None


class TestFreeVramBytesPrefersBackend:
    def test_prefers_backend_over_torch(self, monkeypatch):
        from localm.inference.backends.llamacpp import _loader
        from localm.inference.backends.gguf import GgufBackend
        monkeypatch.setattr(_loader, "gpu_memory",
                            lambda: (5 * 1024 ** 3, 16 * 1024 ** 3))
        # torch would say something different; the backend view must win.
        monkeypatch.setattr(GgufBackend, "_free_total_vram_bytes",
                            staticmethod(lambda: (99, 99)))
        assert GgufBackend._free_vram_bytes() == 5 * 1024 ** 3

    def test_falls_back_to_torch_when_backend_unavailable(self, monkeypatch):
        from localm.inference.backends.llamacpp import _loader
        from localm.inference.backends.gguf import GgufBackend
        monkeypatch.setattr(_loader, "gpu_memory", lambda: None)
        monkeypatch.setattr(GgufBackend, "_free_total_vram_bytes",
                            staticmethod(lambda: (7 * 1024 ** 3, 16 * 1024 ** 3)))
        assert GgufBackend._free_vram_bytes() == 7 * 1024 ** 3


# --------------------------------------------------------------------------- #
#  Integration: the real runtime must export the symbols the fix relies on
# --------------------------------------------------------------------------- #

@pytest.mark.integration
@pytest.mark.real_gguf
def test_real_runtime_exports_kv_head_api():
    """The bundled llama runtime (any provisioned backend: cpu/vulkan/amd-rocm)
    must export llama_model_n_head + llama_model_n_head_kv, or the accurate KV
    size silently falls back to the under-counting heuristic and the Vulkan crash
    returns. Skips cleanly when the native runtime is not provisioned."""
    try:
        from localm.inference.backends.llamacpp._loader import load_lib
        load_lib()
    except Exception as e:
        pytest.skip(f"native llama runtime not provisioned: {e}")
    assert _api.has_kv_head_api() is True


@pytest.mark.integration
@pytest.mark.real_gguf
def test_real_runtime_gpu_memory_query(monkeypatch):
    """On a GPU build, loader.gpu_memory() must return a plausible (free, total)
    from the backend itself - the free-VRAM signal the offload decision relies on,
    with no torch involved. Skips on a CPU-only build (no GPU device)."""
    from localm.inference.backends.llamacpp import _loader
    try:
        _loader.load_lib()
    except Exception as e:
        pytest.skip(f"native llama runtime not provisioned: {e}")
    # The autouse conftest fixture neutralises the cache for hermetic unit tests;
    # clear it so this integration test exercises the REAL resolution + query.
    monkeypatch.setattr(_loader, "_gpu_mem_cache", None)
    gpus = [d for d in _loader.compute_devices() if d[1] != 0]
    mem = _loader.gpu_memory()
    if len(gpus) != 1:
        pytest.skip(f"gpu_memory() targets a single GPU; found {len(gpus)}")
    assert mem is not None, "single GPU present but gpu_memory() returned None"
    free, total = mem
    assert 0 < free <= total and total > 512 * 1024 ** 2   # sane, > 512 MiB card
