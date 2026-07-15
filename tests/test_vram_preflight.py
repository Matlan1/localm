# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the VRAM pre-flight warning in GgufBackend.load()."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.gguf import GgufBackend


def _backend(tmp_path, size_bytes=80_000_000, n_gpu_layers=99, n_ctx=4096):
    f = tmp_path / "model.gguf"
    # Sparse-ish: just truncate to size without writing real bytes
    with open(f, "wb") as fh:
        fh.truncate(size_bytes)
    return GgufBackend(str(f), n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)


@pytest.fixture(autouse=True)
def _small_overhead(monkeypatch):
    """Scale the fixed KV/buffer overhead down to match the MB-scale files."""
    monkeypatch.setattr(GgufBackend, "_VRAM_OVERHEAD_BYTES", 15_000_000)


@pytest.fixture(autouse=True)
def _reset_torch_broken_flag():
    """_torch_rocm_init_broken is a deliberate process-lifetime cache (see its
    docstring in _sizing.py) - correct for a real server process, but poison
    for a test session where many unrelated tests share one process. Several
    tests in this module (e.g. TestVramPreflight.test_load_failure_mentions_
    vram_when_low) mock only _free_vram_bytes and leave GgufBackend's OTHER
    real call, _total_vram_bytes(), unmocked; in an environment without torch
    installed at all (no [gpu] extra - the common case for NVIDIA/Linux/macOS/
    Vulkan/Metal, and for a quick dev venv on this AMD-ROCm-only project too),
    that unmocked call hits a genuine `import torch` failure and latches the
    flag True for the rest of the process - which then makes EVERY later test
    that expects a real torch.cuda read (via patch.dict(sys.modules, ...)) see
    None instead, regardless of its mock, since the flag check short-circuits
    before `import torch` is even reached. Reset around every test in this
    module so one test's incidental unmocked call can never leak into another
    - this is a test-isolation fix only; the flag's process-wide production
    behavior (_sizing.py) is untouched."""
    from localm.inference.backends.llamacpp._sizing import VramSizingMixin
    saved = VramSizingMixin._torch_rocm_init_broken
    VramSizingMixin._torch_rocm_init_broken = False
    yield
    VramSizingMixin._torch_rocm_init_broken = saved


class TestVramPreflight:
    def test_warns_when_model_exceeds_free_vram(self, tmp_path, capsys):
        b = _backend(tmp_path, size_bytes=80_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=40_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16_000_000_000):
            b._check_vram()
        out = capsys.readouterr().out
        assert "Low VRAM" in out
        assert "-g 0" in out          # actionable advice present

    def test_silent_when_model_fits(self, tmp_path, capsys):
        b = _backend(tmp_path, size_bytes=40_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=300_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16_000_000_000):
            b._check_vram()
        assert "Low VRAM" not in capsys.readouterr().out

    def test_silent_when_vram_not_measurable(self, tmp_path, capsys):
        b = _backend(tmp_path)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=None):
            b._check_vram()
        assert capsys.readouterr().out == ""

    def test_silent_for_cpu_only_run(self, tmp_path, capsys):
        b = _backend(tmp_path, n_gpu_layers=0)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=0):
            b._check_vram()
        assert capsys.readouterr().out == ""

    def test_model_bytes_sums_split_parts(self, tmp_path):
        for i in (1, 2):
            f = tmp_path / f"m-0000{i}-of-00002.gguf"
            with open(f, "wb") as fh:
                fh.truncate(1_000_000)
        b = GgufBackend(str(tmp_path / "m-00001-of-00002.gguf"))
        assert b._model_bytes() == 2_000_000

    def test_load_failure_mentions_vram_when_low(self, tmp_path):
        # No subprocess fallback: a native load failure raises, and the message
        # includes the low-VRAM hint and points at setup-llama.
        b = _backend(tmp_path, size_bytes=80_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=20_000_000), \
             patch.object(b, "_load_native", side_effect=RuntimeError("alloc failed")):
            with pytest.raises(RuntimeError) as exc:
                b.load()
        msg = str(exc.value)
        assert "low on memory" in msg
        assert "setup-llama" in msg
        assert b.loaded is False           # never claims to be loaded

    def test_load_failure_no_vram_hint_when_plenty_free(self, tmp_path):
        b = _backend(tmp_path, size_bytes=20_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=120_000_000), \
             patch.object(b, "_load_native", side_effect=RuntimeError("bad dll")):
            with pytest.raises(RuntimeError) as exc:
                b.load()
        assert "low on memory" not in str(exc.value)


class TestKvCacheAwarePreflight:
    """CHK-KVCACHE-OVERFLOW: _check_vram()'s ``need`` estimate must include the
    KV cache for the context this load will actually create (n_ctx), not just
    model weights. A large -c/n_ctx request can need many GB of KV cache on
    top of small weights (e.g. a 10GB Q6_K model with -c 131072 needs a ~20GB
    KV cache - see dev-notes/ for the real-hardware repro), and llama.cpp
    allocates that KV cache as part of context construction, AFTER this
    preflight runs. A weights-only estimate stayed silent for exactly this
    case, letting the load reach the native context-creation call with no
    warning at all - which on ROCm has been observed to either silently spill
    into slow system memory or crash the GPU driver ("unspecified launch
    failure") with nothing surfaced to the user.
    """

    def test_raises_when_kv_cache_pushes_need_past_total_vram(self, tmp_path):
        # 80MB weights (bytes-per-token floors at 16_000 regardless of model
        # size) with a 2,000,000-token context needs ~32GB of KV cache alone -
        # far more than even a clean, empty 16GB card could ever hold. This must
        # be refused BEFORE the native context-creation call, not discovered by
        # letting the driver attempt it.
        b = _backend(tmp_path, size_bytes=80_000_000, n_ctx=2_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=15_000_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16_000_000_000):
            with pytest.raises(RuntimeError) as exc:
                b._check_vram()
        msg = str(exc.value)
        assert "too large" in msg.lower()
        assert "2,000,000" in msg            # names the actual context size
        assert "-c " in msg                  # actionable: how to shrink it

    def test_no_regression_when_context_genuinely_fits(self, tmp_path, capsys):
        # A normal, modest context on the same small model must neither raise
        # nor warn - the new KV-cache term must not make a genuinely-fitting
        # load look oversized.
        b = _backend(tmp_path, size_bytes=80_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=10_000_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16_000_000_000):
            b._check_vram()   # must not raise
        assert "Low VRAM" not in capsys.readouterr().out

    def test_warns_but_does_not_raise_when_only_free_is_short(self, tmp_path, capsys):
        # need exceeds FREE (something else is using the GPU) but comfortably
        # fits under TOTAL (a clean card would hold it) - this is the
        # recoverable "free some VRAM" case and must stay a warning, not a
        # hard refusal.
        b = _backend(tmp_path, size_bytes=80_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=100_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16_000_000_000):
            b._check_vram()   # must not raise
        assert "Low VRAM" in capsys.readouterr().out

    def test_raise_check_skipped_when_total_not_measurable(self, tmp_path, capsys):
        # total unmeasurable (e.g. no torch / registry-fallback tier): the hard
        # refusal must be skipped rather than crash on a None comparison, even
        # for a context that would have tripped it - falls through to the
        # ordinary (here: silent, since free comfortably covers it) free-vram
        # check instead of raising.
        b = _backend(tmp_path, size_bytes=80_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=15_000_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=None):
            b._check_vram()   # must not raise
        assert "Low VRAM" not in capsys.readouterr().out


class TestCheckContextFit:
    """GgufBackend._check_context_fit() is the vram_check callback wired into
    LlamaCpp: it decides WHERE a growing context's KV cache must live. Returns True
    to keep the KV cache in VRAM (offload_kqv - fast), False to place it in system
    RAM (slower, but keeps the FULL window - a degrade, not a shrink or an abort),
    or None when VRAM is unmeasurable / irrelevant.

    It NEVER raises and NEVER charges weights (already resident). Because
    _prefill_fresh_context frees the live context (current_ctx KV) BEFORE allocating
    the bigger one, only the NET KV growth over the resident context is charged; the
    resident compute-buffer overhead is not re-charged. Regression: the old version
    charged the WHOLE target KV + a 1.5 GB overhead against free VRAM measured while
    the old KV was still resident, and hard-raised - false-aborting the very first
    prompt on a model that fills the card yet generates fine once the old KV is
    reclaimed (vram-grow-fail-rootcause.md)."""

    def test_uses_ram_when_kv_growth_does_not_fit_vram(self, tmp_path):
        # A grow whose NET KV growth cannot fit VRAM keeps the FULL window but puts
        # the KV cache in system RAM (return False) - never shrinks, never aborts.
        b = _backend(tmp_path, size_bytes=80_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=50_000_000):
            assert b._check_context_fit(2_000_000, current_ctx=4096) is False

    def test_charges_only_net_growth_over_resident_context(self, tmp_path):
        # The old context's KV is reclaimed before the new is allocated, so only the
        # DELTA (target - current) is charged. Free here holds the delta but NOT the
        # whole target KV + overhead: the OLD hard check would have false-refused;
        # the delta check passes -> KV fits VRAM (return True).
        # per_token ~= 90_000; delta(8192 from 4096) ~= 368 MB; whole target KV
        # ~= 737 MB, + 1.5 GB overhead ~= 2.2 GB. Free = 500 MB.
        b = _backend(tmp_path, size_bytes=9_000_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=500_000_000):
            assert b._check_context_fit(8192, current_ctx=4096) is True

    def test_does_not_double_count_resident_weights(self, tmp_path):
        b = _backend(tmp_path, size_bytes=9_000_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=4_000_000_000):
            assert b._check_context_fit(8192, current_ctx=4096) is True   # KV fits VRAM

    def test_returns_none_when_vram_not_measurable(self, tmp_path):
        b = _backend(tmp_path, size_bytes=80_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=None):
            assert b._check_context_fit(2_000_000) is None   # unmeasurable -> keep default (VRAM)

    def test_returns_none_for_cpu_only_run(self, tmp_path):
        b = _backend(tmp_path, size_bytes=80_000_000, n_gpu_layers=0)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=0):
            assert b._check_context_fit(2_000_000) is None   # CPU-only -> KV already in RAM

    # test_load_native_wires_check_context_fit_into_llamacpp moved to
    # tests/test_gguf_worker.py (TestLoad.test_load_wires_check_context_fit_and_returns_metadata):
    # the real LlamaCpp construction (and the vram_check wiring) now happens
    # in GgufWorker, inside the isolated worker process, not in
    # GgufBackend._load_native() - see llamacpp/_runner.py.


class TestVramReport:
    """The post-load VRAM line must use driver-level numbers (mem_get_info),
    never torch allocator counters - llama.dll allocates outside torch, so
    memory_allocated() reads 0.00 GB no matter what the model occupies."""

    def _fake_torch(self, free_total_per_device):
        import sys
        from unittest.mock import MagicMock
        fake = MagicMock()
        fake.cuda.is_available.return_value = True
        fake.cuda.device_count.return_value = len(free_total_per_device)
        fake.cuda.mem_get_info.side_effect = \
            lambda i: free_total_per_device[i]
        return patch.dict(sys.modules, {"torch": fake})

    def test_vram_levels_returns_driver_numbers(self):
        with self._fake_torch([(4_000_000_000, 16_000_000_000)]):
            levels = GgufBackend._vram_levels()
        assert levels == [(4_000_000_000, 16_000_000_000)]

    def test_vram_levels_empty_without_torch(self):
        import sys
        # Simulate torch being absent (import raises)
        with patch.dict(sys.modules, {"torch": None}):
            assert GgufBackend._vram_levels() == []

    def test_load_reports_usage_delta(self, tmp_path, capsys):
        """End to end through _load_native with a stubbed ModelRunner: the
        printed line shows in-use/total and the delta this load consumed.
        The VRAM-delta report stays parent-side (it reads driver-global
        state, correct from either process) even though the real model load
        now happens in an isolated worker - see llamacpp/_runner.py."""
        b = _backend(tmp_path, size_bytes=1_000_000)
        # 12 GiB free before the load, 4 GiB free after -> 8 GiB this load.
        # Sizes are displayed in binary units (GiB labelled "GB").
        GIB = 1024 ** 3
        levels = iter([[(12 * GIB, 16 * GIB)],
                       [(4 * GIB, 16 * GIB)]])
        with patch.object(GgufBackend, "_vram_levels",
                          side_effect=lambda: next(levels)), \
             patch("localm.inference.backends.llamacpp._runner.ModelRunner.spawn_and_load",
                   return_value={"n_layers": None, "kv_bytes_per_token": 0,
                                 "supports_images": False}):
            b._load_native()
        out = capsys.readouterr().out
        assert "12.00 GB in use / 16.00 GB total" in out
        assert "+8.00 GB this load" in out
        assert "0.00 GB allocated" not in out      # the old, wrong line


class TestFreeVramBytesDeviceSelection:
    """_free_vram_bytes() must read the CONFIGURED main GPU device, not always
    device 0, once a multi-GPU main_gpu_index is set."""

    def _fake_torch(self, per_device_free_total):
        fake = MagicMock()
        fake.cuda.is_available.return_value = True
        fake.cuda.device_count.return_value = len(per_device_free_total)
        fake.cuda.mem_get_info.side_effect = lambda i: per_device_free_total[i]
        return patch.dict(sys.modules, {"torch": fake})

    def test_reads_configured_device(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": 1})
        monkeypatch.setattr("localm.discover.list_gpus",
                            lambda: [{"index": 0}, {"index": 1}])
        with self._fake_torch([(1_000, 2_000), (5_000, 8_000)]):
            free = GgufBackend._free_vram_bytes()
        assert free == 5_000

    def test_defaults_to_device_zero_when_unconfigured(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": None})
        with self._fake_torch([(1_000, 2_000), (5_000, 8_000)]):
            free = GgufBackend._free_vram_bytes()
        assert free == 1_000

    def test_invalid_configured_index_falls_back_to_device_zero(self, monkeypatch):
        # Pin non-Vulkan so membership validation actually runs, regardless of
        # what native backend is provisioned in the ambient environment (see
        # test_discover.py::TestResolveMainGpuIndex).
        monkeypatch.setattr("localm.discover._native_backend_has_vulkan", lambda: False)
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": 9})
        monkeypatch.setattr("localm.discover.list_gpus",
                            lambda: [{"index": 0}])
        with self._fake_torch([(1_000, 2_000)]):
            free = GgufBackend._free_vram_bytes()
        assert free == 1_000


class TestFreeVramBytesUsesIsolatedNativeFallback:
    """_free_vram_bytes() must prefer torch.cuda.mem_get_info, and must NEVER
    call the DIRECT, abort-prone loader.gpu_memory() - only the crash-safe
    loader.gpu_memory_isolated() (subprocess-isolated) as a fallback when torch
    cannot answer.

    Root cause this guards: loader.gpu_memory()'s mem_fn() ctypes call
    (ggml_backend_dev_memory) is not exception-safe - llama.cpp's own CUDA/HIP
    error macro aborts the WHOLE PROCESS on a transient driver failure. Confirmed
    live three times on this exact call, and a documented, recurring class of
    issue on ROCm/HIP generally (ollama/ollama#3840; ROCm/ROCm#5378). This
    project's own torch dependency (pyproject.toml) is an AMD-ROCm, Windows-only
    wheel - there is no NVIDIA/Linux/macOS torch wired in at all - so the
    fallback path is the COMMON case for NVIDIA/Linux/macOS/Vulkan/Metal users,
    not a rare edge, which is why it must actually work (not just degrade to
    None) rather than simply avoiding the direct native call."""

    def _fake_torch_ok(self):
        fake = MagicMock()
        fake.cuda.is_available.return_value = True
        fake.cuda.device_count.return_value = 1
        fake.cuda.mem_get_info.return_value = (7_000, 9_000)
        return patch.dict(sys.modules, {"torch": fake})

    def test_isolated_fallback_never_called_when_torch_answers(self, monkeypatch):
        sentinel = MagicMock(side_effect=AssertionError(
            "gpu_memory_isolated() must not be called when torch already answered"))
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.gpu_memory_isolated", sentinel)
        with self._fake_torch_ok():
            free = GgufBackend._free_vram_bytes()
        assert free == 7_000
        sentinel.assert_not_called()

    def test_direct_native_ggml_query_never_called(self, monkeypatch):
        """The DIRECT, abort-prone native call must never be reached from
        _free_vram_bytes under any condition - only the isolated wrapper."""
        fake = MagicMock()
        fake.cuda.is_available.return_value = False
        sentinel = MagicMock(side_effect=AssertionError(
            "loader.gpu_memory() (direct, abort-prone) must never be called "
            "from _free_vram_bytes - only gpu_memory_isolated()"))
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.gpu_memory", sentinel)
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.gpu_memory_isolated",
            lambda: (3_000, 5_000))
        with patch.dict(sys.modules, {"torch": fake}):
            free = GgufBackend._free_vram_bytes()
        assert free == 3_000
        sentinel.assert_not_called()

    def test_falls_back_to_isolated_probe_when_torch_unmeasurable(self, monkeypatch):
        """Vulkan/Metal/NVIDIA-without-torch/Linux/macOS (no CUDA/ROCm torch):
        the isolated probe must be consulted and its answer used - VRAM
        measurement must keep working, not just degrade to None."""
        fake = MagicMock()
        fake.cuda.is_available.return_value = False
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.gpu_memory_isolated",
            lambda: (3_000, 5_000))
        with patch.dict(sys.modules, {"torch": fake}):
            free = GgufBackend._free_vram_bytes()
        assert free == 3_000

    def test_none_when_neither_torch_nor_isolated_probe_can_answer(self, monkeypatch):
        fake = MagicMock()
        fake.cuda.is_available.return_value = False
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.gpu_memory_isolated",
            lambda: None)
        with patch.dict(sys.modules, {"torch": fake}):
            free = GgufBackend._free_vram_bytes()
        assert free is None

    def test_none_when_torch_itself_raises(self, monkeypatch):
        """A torch import/query failure (e.g. the observed ROCm-init hiccup) must
        degrade via _free_total_vram_bytes' own try/except and still try the
        isolated fallback, not propagate."""
        class _BoomModule:
            def __getattr__(self, name):
                raise RuntimeError("simulated torch ROCm-init failure")
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.gpu_memory_isolated",
            lambda: (3_000, 5_000))
        with patch.dict(sys.modules, {"torch": _BoomModule()}):
            free = GgufBackend._free_vram_bytes()
        assert free == 3_000

    def test_torch_import_failure_is_cached_and_never_retried(self, monkeypatch):
        """The real bug this guards (found live, reproduced on demand - see
        _sizing.py's VramSizingMixin._free_total_vram_bytes docstring): once
        llama.cpp's bundled HIP runtime is loaded in-process (as GgufWorker's
        isolated worker process does for every real load), a LATER `import
        torch` hits a genuine DLL entry-point conflict in torch's separate
        rocm_sdk ROCm runtime. A module that faults during import is evicted
        from sys.modules, so an uncached failure would retry - and re-fault -
        on EVERY subsequent VRAM check for the rest of the process's life.
        `sys.modules[name] = None` is the standard way to simulate `import
        torch` itself raising (as opposed to _BoomModule above, which
        simulates torch importing fine but a later attribute access failing
        - a different, NOT cached, failure mode)."""
        monkeypatch.setitem(sys.modules, "torch", None)
        assert GgufBackend._free_total_vram_bytes() == (None, None)
        from localm.inference.backends.llamacpp._sizing import VramSizingMixin
        assert VramSizingMixin._torch_rocm_init_broken is True

        # Second call must be answered from the cached flag WITHOUT
        # re-attempting the import - proven by making torch "recover" to a
        # working fake and confirming it is still never consulted.
        fake = MagicMock()
        fake.cuda.is_available.return_value = True
        fake.cuda.mem_get_info.return_value = (1, 2)
        monkeypatch.setitem(sys.modules, "torch", fake)
        assert GgufBackend._free_total_vram_bytes() == (None, None), (
            "a cached broken-import must not be retried, even once torch "
            "would now succeed")
        fake.cuda.is_available.assert_not_called()


class _FakeStream:
    """A minimal write()/flush() or readline() double for a fake daemon proc's
    stdin/stdout, scripted with a fixed queue of responses."""

    def __init__(self, lines=None):
        self._lines = list(lines or [])
        self.written = []

    def write(self, s):
        self.written.append(s)

    def flush(self):
        pass

    def readline(self):
        return self._lines.pop(0) if self._lines else ""     # "" == EOF


class _FakeDaemonProc:
    """A minimal subprocess.Popen double: scripted stdout lines, a `killed`
    flag that flips poll() from None (running) to an exit code (dead)."""

    def __init__(self, response_lines):
        self.stdin = _FakeStream()
        self.stdout = _FakeStream(response_lines)
        self._killed = False

    def poll(self):
        return -9 if self._killed else None

    def kill(self):
        self._killed = True


class TestGpuMemoryIsolated:
    """loader.gpu_memory_isolated() must degrade to None on every failure mode
    of the daemon subprocess, and never raise - this is what makes it safe to
    call unconditionally from _free_vram_bytes. Every test resets the module-
    level daemon singleton (_PROBE_PROC) before and after, since it is shared
    global state across calls by design (that is the whole point of the
    daemon - reuse across calls within one process)."""

    def _loader(self):
        from localm.inference.backends.llamacpp import _loader
        return _loader

    @pytest.fixture(autouse=True)
    def _isolate_probe_singleton(self):
        loader = self._loader()
        saved = loader._PROBE_PROC
        loader._PROBE_PROC = None
        yield
        loader._PROBE_PROC = saved

    def _patch_readline_passthrough(self, monkeypatch, loader):
        """Replace the real threaded timeout wrapper with a direct readline()
        call for these higher-level tests (the timeout mechanism itself has its
        own dedicated tests below) - keeps these fast and deterministic."""
        monkeypatch.setattr(
            loader, "_readline_with_timeout",
            lambda stream, timeout: (stream.readline() or None))

    def test_first_call_spawns_daemon_and_parses_response(self, monkeypatch):
        loader = self._loader()
        self._patch_readline_passthrough(monkeypatch, loader)
        fake = _FakeDaemonProc(["123456 789012\n"])
        monkeypatch.setattr(loader, "_spawn_probe_daemon", lambda: fake)
        assert loader.gpu_memory_isolated() == (123456, 789012)
        assert fake.stdin.written == ["q\n"]

    def test_second_call_reuses_running_daemon_without_respawning(self, monkeypatch):
        loader = self._loader()
        self._patch_readline_passthrough(monkeypatch, loader)
        fake = _FakeDaemonProc(["100 200\n", "300 400\n"])
        spawn_calls = []
        monkeypatch.setattr(
            loader, "_spawn_probe_daemon",
            lambda: spawn_calls.append(1) or fake)
        assert loader.gpu_memory_isolated() == (100, 200)
        assert loader.gpu_memory_isolated() == (300, 400)
        assert len(spawn_calls) == 1, "the second call must reuse the daemon, not respawn"

    def test_daemon_err_reply_returns_none_but_keeps_daemon_alive(self, monkeypatch):
        """A daemon that is alive and answered "unmeasurable" is not the same
        as a crashed daemon - it must not be killed/respawned for that alone."""
        loader = self._loader()
        self._patch_readline_passthrough(monkeypatch, loader)
        fake = _FakeDaemonProc(["ERR\n", "50 100\n"])
        monkeypatch.setattr(loader, "_spawn_probe_daemon", lambda: fake)
        assert loader.gpu_memory_isolated() is None
        assert not fake._killed
        assert loader.gpu_memory_isolated() == (50, 100)

    def test_dead_daemon_triggers_respawn(self, monkeypatch):
        loader = self._loader()
        self._patch_readline_passthrough(monkeypatch, loader)
        dead = _FakeDaemonProc([])
        dead.kill()                          # already dead before any query
        loader._PROBE_PROC = dead            # simulate a daemon that died between calls
        fresh = _FakeDaemonProc(["10 20\n"])
        monkeypatch.setattr(loader, "_spawn_probe_daemon", lambda: fresh)
        assert loader.gpu_memory_isolated() == (10, 20)

    def test_eof_on_read_kills_and_clears_daemon_for_next_respawn(self, monkeypatch):
        """The exact scenario this exists for: the daemon hard-aborts mid-query
        (native crash) - the OS closes its pipes, so the next read returns EOF,
        never an exception that could propagate into the parent."""
        loader = self._loader()
        self._patch_readline_passthrough(monkeypatch, loader)
        crashing = _FakeDaemonProc([])       # readline() -> "" == EOF immediately
        monkeypatch.setattr(loader, "_spawn_probe_daemon", lambda: crashing)
        assert loader.gpu_memory_isolated() is None
        assert loader._PROBE_PROC is None, "a crashed daemon must be cleared, not reused"

    def test_garbage_response_kills_and_clears_daemon(self, monkeypatch):
        loader = self._loader()
        self._patch_readline_passthrough(monkeypatch, loader)
        fake = _FakeDaemonProc(["not a valid pair of ints\n"])
        monkeypatch.setattr(loader, "_spawn_probe_daemon", lambda: fake)
        assert loader.gpu_memory_isolated() is None
        assert fake._killed
        assert loader._PROBE_PROC is None

    def test_spawn_failure_degrades_to_none(self, monkeypatch):
        loader = self._loader()

        def _raise():
            raise OSError("could not spawn python")

        monkeypatch.setattr(loader, "_spawn_probe_daemon", _raise)
        assert loader.gpu_memory_isolated() is None
        assert loader._PROBE_PROC is None


class TestReadlineWithTimeout:
    """The low-level timeout wrapper itself: a genuinely blocking stream must
    not hang the caller past `timeout`, and a normal read must pass through."""

    def _loader(self):
        from localm.inference.backends.llamacpp import _loader
        return _loader

    def test_normal_read_passes_through(self):
        loader = self._loader()
        stream = _FakeStream(["hello\n"])
        assert loader._readline_with_timeout(stream, timeout=2.0) == "hello\n"

    def test_eof_returns_none(self):
        loader = self._loader()
        stream = _FakeStream([])
        assert loader._readline_with_timeout(stream, timeout=2.0) is None

    def test_genuinely_blocking_read_times_out(self):
        """A stream whose readline() never returns must not hang this call past
        the timeout - uses a small timeout so the test itself stays fast."""
        import threading

        loader = self._loader()

        class _HangingStream:
            def readline(self):
                threading.Event().wait()      # blocks forever

        import time
        t0 = time.time()
        result = loader._readline_with_timeout(_HangingStream(), timeout=0.2)
        elapsed = time.time() - t0
        assert result is None
        assert elapsed < 2.0, f"timeout wrapper did not return promptly: {elapsed:.2f}s"
