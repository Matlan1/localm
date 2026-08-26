# SPDX-License-Identifier: AGPL-3.0-or-later
"""A large GGUF that fills VRAM must keep generating, not crash.

A model whose weights nearly fill the card, loaded with its KV cache in VRAM,
leaves no room for the first decode's compute buffers; on the Vulkan backend that
faults with a native C++ crash (0xe06d7363) instead of spilling to RAM as ROCm
does. A file-size KV heuristic under-counts the real KV cache by roughly 2.6x, so
such a load "fits" per the estimate and overflows VRAM for real.

These tests run WITHOUT the native DLL (the api module and metadata are mocked)
and cover the architecture-accurate per-token KV size, the load-time offload_kqv
decision (keep KV in system RAM when it will not fit alongside compute), that the
placement is sticky across a grow, and that the grow check reasons with the
accurate size. The real-runtime symbol export and real generation are covered by
the @integration tests at the bottom.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import os

import pytest

from localm.inference.backends.llamacpp import _api
from localm.inference.backends.llamacpp.llama import LlamaCpp

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


def _fake_api(*, has=True, n_embd=4096, n_head=32, n_head_kv=8,
              has_hybrid=True, hybrid=False, recurrent=False):
    """A UNIFORM stack by default. hybrid/recurrent must be set explicitly, and
    must never be left to MagicMock's auto-attribute: a bare MagicMock returns a
    truthy Mock for llama_model_is_hybrid(), which would make every test here
    silently exercise the hybrid refusal path instead of the formula."""
    m = MagicMock()
    m.has_kv_head_api.return_value = has
    m.has_hybrid_api.return_value = has_hybrid
    m.llama_model_is_hybrid.return_value = hybrid
    m.llama_model_is_recurrent.return_value = recurrent
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
        # n_head_kv < n_head (grouped-query) yields a proportionally smaller KV.
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
#  Vulkan dedicated-VRAM flag (loader._force_vulkan_dedicated_vram)
# --------------------------------------------------------------------------- #

class TestForceVulkanDedicatedVram:
    """On a Windows Vulkan build, load_lib sets GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM
    so ggml-vulkan keeps model weights in DEDICATED VRAM - WDDM otherwise backs the
    host-visible allocation with shared system RAM (~13x slower). Respect an explicit
    user value; do nothing on non-Windows or a non-Vulkan build."""

    def _run(self, tmp_path, monkeypatch, *, platform, files, preset=None):
        from localm.inference.backends.llamacpp import _loader
        for f in files:
            (tmp_path / f).write_bytes(b"x")
        monkeypatch.setattr(_loader.sys, "platform", platform)
        monkeypatch.delenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", raising=False)
        if preset is not None:
            monkeypatch.setenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", preset)
        _loader._force_vulkan_dedicated_vram(tmp_path)
        return os.environ.get("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM")

    def test_sets_flag_on_windows_vulkan_build(self, tmp_path, monkeypatch):
        assert self._run(tmp_path, monkeypatch, platform="win32",
                         files=["llama.dll", "ggml-vulkan.dll"]) == "1"

    def test_no_flag_on_windows_rocm_build(self, tmp_path, monkeypatch):
        assert self._run(tmp_path, monkeypatch, platform="win32",
                         files=["llama.dll", "ggml-hip.dll"]) is None

    def test_no_flag_off_windows(self, tmp_path, monkeypatch):
        assert self._run(tmp_path, monkeypatch, platform="linux",
                         files=["libllama.so", "libggml-vulkan.so"]) is None

    @pytest.mark.parametrize("preset", ["0", "false", "FALSE", "off", "no", "", " 0 "])
    def test_optout_unsets_the_var_because_ggml_switches_on_presence(
            self, tmp_path, monkeypatch, preset):
        """An opt-out must reach GGML, not merely survive in os.environ.

        ggml reads `getenv(...) != nullptr`, so "0" disables host-visible
        vidmem exactly as "1" does. ABSENCE is the property: it is the only
        value of this variable that ggml reads as "off".
        """
        assert self._run(tmp_path, monkeypatch, platform="win32",
                         files=["llama.dll", "ggml-vulkan.dll"],
                         preset=preset) is None

    @pytest.mark.parametrize("preset", ["1", "2", "yes", "true"])
    def test_an_explicit_non_falsey_value_is_left_exactly_as_set(
            self, tmp_path, monkeypatch, preset):
        """The fix must not become "always unset when the user set anything".
        A user who set the var to enable the guard keeps their own value
        verbatim - we neither overwrite it with "1" nor remove it."""
        assert self._run(tmp_path, monkeypatch, platform="win32",
                         files=["llama.dll", "ggml-vulkan.dll"],
                         preset=preset) == preset


# --------------------------------------------------------------------------- #
#  Grow check reasons with the accurate KV size (GgufBackend._check_context_fit)
# --------------------------------------------------------------------------- #

class _StubLlm:
    def __init__(self, kv):
        self.kv_bytes_per_token = kv


def _gguf_backend(tmp_path, size_bytes, n_ctx=4096, n_gpu_layers=99):
    # A tiny REAL file (so is_file()/stat work), with the multi-GB "on disk" size
    # FAKED via _model_bytes. NEVER truncate() to the real size here: Windows
    # truncate() is not sparse and allocates REAL disk. The size is only ever
    # READ back through _model_bytes().
    from localm.inference.backends.gguf import GgufBackend
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * 4096)
    b = GgufBackend(str(f), n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
    b._model_bytes = lambda: size_bytes
    return b


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


class TestRamResidentKvChargesFullTarget:
    """Once a prior grow placed the KV cache in SYSTEM RAM (offload_kqv=False),
    the GPU holds no KV, so free VRAM reads large. A delta-only charge would then
    say a further grow fits VRAM, offload_kqv would flip back to True, and the
    FULL target KV would overflow VRAM -> llama_init_from_model NULLs ->
    _prefill_fresh_context raises and truncates a reply that was generating fine.
    When the resident KV is already in RAM, the FULL target is charged."""

    def test_ram_resident_kv_charges_full_target_stays_ram(self, tmp_path):
        b = _gguf_backend(tmp_path, size_bytes=2_000_000_000, n_ctx=4096)
        b._llm = _StubLlm(300_000)              # accurate per-token KV
        b._llm._offload_kqv = False             # prior grow put the KV in system RAM
        # Grow 8192 -> 12288: delta = 4096*300k = 1.23 GB (fits 2 GB free), but the
        # FULL target 12288*300k = 3.69 GB does NOT. With the KV already in RAM there
        # is no VRAM KV to reclaim, so the honest answer is: still does not fit -> RAM.
        with patch.object(type(b), "_free_vram_bytes", return_value=2_000_000_000):
            assert b._check_context_fit(12288, current_ctx=8192) is False

    def test_ram_resident_kv_returns_to_vram_when_full_target_fits(self, tmp_path):
        b = _gguf_backend(tmp_path, size_bytes=2_000_000_000, n_ctx=4096)
        b._llm = _StubLlm(300_000)
        b._llm._offload_kqv = False             # currently in RAM
        # Now free VRAM is large enough for the WHOLE target (12288*300k = 3.69 GB):
        # moving back to VRAM genuinely fits, so it should.
        with patch.object(type(b), "_free_vram_bytes", return_value=4_000_000_000):
            assert b._check_context_fit(12288, current_ctx=8192) is True

    def test_vram_resident_kv_still_uses_net_delta(self, tmp_path):
        # When the current KV is in VRAM (the normal case), the net delta is
        # charged (the old VRAM KV IS reclaimed on recreation).
        b = _gguf_backend(tmp_path, size_bytes=2_000_000_000, n_ctx=4096)
        b._llm = _StubLlm(300_000)
        b._llm._offload_kqv = True               # current KV in VRAM
        with patch.object(type(b), "_free_vram_bytes", return_value=2_000_000_000):
            # delta = 4096*300k = 1.23 GB <= 2.0 GB free -> VRAM, even though the full
            # 3.69 GB target would not fit (the reclaimed old KV covers the difference).
            assert b._check_context_fit(12288, current_ctx=8192) is True


class TestRamOffloadHintFiresOnce:
    """The RAM-offload notice is a one-time hint: a conversation that keeps
    growing past free VRAM explains the slowdown ONCE rather than on every grow.
    A card-filling model with the default grow step overflows on EVERY grow."""

    def test_hint_fires_once_across_repeated_ram_grows(self, tmp_path, caplog):
        import logging
        b = _gguf_backend(tmp_path, size_bytes=2_000_000_000, n_ctx=4096)
        b._llm = _StubLlm(300_000)                     # accurate, large KV
        # Free stays tiny, so each grow's delta overflows -> RAM (False) every time.
        with patch.object(type(b), "_free_vram_bytes", return_value=100_000_000):
            with caplog.at_level(logging.WARNING, logger="localm"):
                assert b._check_context_fit(8192, current_ctx=4096) is False
                assert b._check_context_fit(12288, current_ctx=8192) is False
                assert b._check_context_fit(16384, current_ctx=12288) is False
        hits = [r for r in caplog.records
                if "large context" in r.getMessage()
                and "kept in system RAM" in r.getMessage()]
        assert len(hits) == 1, (
            f"RAM-offload hint should fire once, fired {len(hits)} times")

    def test_load_resets_the_hint_for_a_reloaded_instance(self, tmp_path, monkeypatch):
        # The SAME backend instance can be reloaded (Engine.chat_stream auto-reload),
        # so load() must clear the guard - otherwise the hint would fire once EVER
        # instead of once per loaded-model session. load() resets it on its first
        # line, before any native work; bail right after via missing_split_parts.
        import localm.model_manager as mm
        b = _gguf_backend(tmp_path, size_bytes=1_000_000_000)
        b._ram_kv_hint_shown = True                       # simulate "already shown"
        monkeypatch.setattr(mm, "missing_split_parts",
                            lambda p: [tmp_path / "phantom.gguf"])
        with pytest.raises(FileNotFoundError):
            b.load()
        assert b._ram_kv_hint_shown is False


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
    """_free_vram_bytes()'s source of truth for the KV-placement decision this
    file covers: torch.cuda is ALWAYS preferred when it can answer, and the
    direct native loader.gpu_memory() call is never made from here at all - only
    its crash-safe, subprocess-isolated wrapper gpu_memory_isolated(), as a
    fallback when torch cannot answer."""

    def test_prefers_torch_over_isolated_fallback(self, monkeypatch):
        from localm.inference.backends.llamacpp import _loader
        from localm.inference.backends.gguf import GgufBackend
        monkeypatch.setattr(_loader, "gpu_memory_isolated",
                            lambda: (5 * 1024 ** 3, 16 * 1024 ** 3))
        # the isolated fallback would say something different; torch must win.
        monkeypatch.setattr(GgufBackend, "_free_total_vram_bytes",
                            staticmethod(lambda: (7 * 1024 ** 3, 16 * 1024 ** 3)))
        # _free_vram_bytes applies a device-global correction on a Windows +
        # ROCm/HIP box, which would replace the faked free above with a REAL
        # measurement. Patch it to None so this test asserts source SELECTION
        # (torch vs isolated), not the correction step.
        monkeypatch.setattr(GgufBackend, "_device_global_free_bytes",
                            staticmethod(lambda total: None))
        assert GgufBackend._free_vram_bytes() == 7 * 1024 ** 3

    def test_falls_back_to_isolated_probe_when_torch_unavailable(self, monkeypatch):
        from localm.inference.backends.llamacpp import _loader
        from localm.inference.backends.gguf import GgufBackend
        monkeypatch.setattr(_loader, "gpu_memory_isolated",
                            lambda: (5 * 1024 ** 3, 16 * 1024 ** 3))
        monkeypatch.setattr(GgufBackend, "_free_total_vram_bytes",
                            staticmethod(lambda: (None, None)))
        # Isolate from the device-global correction so this asserts fallback
        # SELECTION, not the correction.
        monkeypatch.setattr(GgufBackend, "_device_global_free_bytes",
                            staticmethod(lambda total: None))
        assert GgufBackend._free_vram_bytes() == 5 * 1024 ** 3


# --------------------------------------------------------------------------- #
#  Integration: the real runtime exports the symbols this code relies on
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


# --------------------------------------------------------------------------- #
#  n_layers * n_head_kv on a HYBRID stack                                      #
#                                                                              #
#  llama_model_n_head_kv reports LAYER 0 only - upstream llama_hparams::       #
#  n_head_kv() takes an il parameter defaulting to 0, and the exported wrapper  #
#  passes nothing. On a uniform stack layer 0 speaks for every layer; on a      #
#  hybrid one (Qwen3-Next, Granite 4 H, LFM2, Jamba ...) most layers keep a     #
#  fixed-size recurrent state and hold no KV cache at all, so multiplying by    #
#  the layer count over-charges and there is nothing here to sum over.         #
# --------------------------------------------------------------------------- #

class TestHasHybridApi:
    def test_true_when_both_symbols_present(self):
        fake_lib = MagicMock(spec=["llama_model_is_recurrent",
                                   "llama_model_is_hybrid"])
        with patch(_LOAD_LIB, return_value=fake_lib):
            assert _api.has_hybrid_api() is True

    def test_false_when_one_is_absent(self):
        # A partial export is unusable: without both, a hybrid stack cannot be
        # told apart from a uniform one.
        fake_lib = MagicMock(spec=["llama_model_is_recurrent"])
        with patch(_LOAD_LIB, return_value=fake_lib):
            assert _api.has_hybrid_api() is False

    def test_false_when_both_absent(self):
        with patch(_LOAD_LIB, return_value=MagicMock(spec=[])):
            assert _api.has_hybrid_api() is False


class TestHybridBindings:
    def test_is_hybrid_calls_through_and_returns_a_real_bool(self):
        fake_fn = MagicMock(return_value=1)
        fake_lib = MagicMock(spec=["llama_model_is_hybrid"])
        fake_lib.llama_model_is_hybrid = fake_fn
        with patch(_LOAD_LIB, return_value=fake_lib):
            got = _api.llama_model_is_hybrid(1234)
        assert got is True                     # a real bool, not ctypes' 1
        fake_fn.assert_called_once_with(1234)

    def test_is_recurrent_calls_through_and_returns_a_real_bool(self):
        fake_fn = MagicMock(return_value=0)
        fake_lib = MagicMock(spec=["llama_model_is_recurrent"])
        fake_lib.llama_model_is_recurrent = fake_fn
        with patch(_LOAD_LIB, return_value=fake_lib):
            got = _api.llama_model_is_recurrent(1234)
        assert got is False
        fake_fn.assert_called_once_with(1234)


class TestHybridStackIsRefused:
    def test_hybrid_returns_zero_instead_of_the_over_charged_product(self):
        # A Qwen3-Next shape: 48 layers, only 12 of which attend. The formula
        # would charge all 48 and be 4x high; 0 means "no signal", which sends
        # the caller to the GGUF header probe that can read the per-layer truth.
        llm = _bare_model(n_layers=48)
        with patch(_API, _fake_api(n_embd=2048, n_head=16, n_head_kv=2,
                                   hybrid=True)):
            assert llm._read_kv_bytes_per_token() == 0

    def test_recurrent_returns_zero(self):
        # Mamba/RWKV: no growing KV cache anywhere in the stack.
        llm = _bare_model(n_layers=48)
        with patch(_API, _fake_api(n_embd=2048, n_head=16, n_head_kv=2,
                                   recurrent=True)):
            assert llm._read_kv_bytes_per_token() == 0

    def test_the_same_shape_still_computes_when_the_stack_is_uniform(self):
        # The control: a uniform stack still computes a nonzero per-token KV.
        llm = _bare_model(n_layers=48)
        with patch(_API, _fake_api(n_embd=2048, n_head=16, n_head_kv=2)):
            assert llm._read_kv_bytes_per_token() == 48 * 2 * 128 * 2 * 2

    def test_a_build_without_the_predicates_keeps_the_old_answer(self):
        # A build that cannot answer "is this hybrid?" degrades to the previous
        # behaviour rather than refusing everything.
        llm = _bare_model(n_layers=48)
        with patch(_API, _fake_api(n_embd=2048, n_head=16, n_head_kv=2,
                                   has_hybrid=False, hybrid=True)):
            assert llm._read_kv_bytes_per_token() == 48 * 2 * 128 * 2 * 2

    def test_the_hybrid_check_is_not_consulted_before_the_kv_head_api(self):
        # Ordering guard: has_kv_head_api() gates the whole block, so a build
        # without the head accessors must answer 0 without ever probing hybrid.
        api_mock = _fake_api(has=False, hybrid=True)
        llm = _bare_model(n_layers=48)
        with patch(_API, api_mock):
            assert llm._read_kv_bytes_per_token() == 0
        api_mock.llama_model_is_hybrid.assert_not_called()
