# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GGUF backend's per-load sizing/preflight must budget an applied
multi-GPU split against the split's COMBINED free/total, not the main GPU
alone (VramSizingMixin._split_free_total_bytes + its four consumers:
_auto_gpu_layers, _check_vram, _auto_ctx_max, _check_context_fit - see
llamacpp/_sizing.py), with vram_capacity(combined_only=True) as the single
summing source (discover.py).

The regression these tests pin (functional audit 2026-07-21, gpu-split scope):
the admission gate (vram_capacity + gpu_split_shortfall) learned to sum across
a split long ago, but the backend's own deeper preflight stayed split-blind -
on a simulated 2x16 GB box with a 20 GB model and gpu_split_indices=[0, 1],
_auto_gpu_layers picked a silent partial offload (21/32 layers), _check_vram
hard-refused a pinned load with a factually wrong "cannot fit regardless", and
_auto_ctx_max collapsed the growth ceiling to n_ctx despite ~11 GB combined
headroom.

Style mirrors tests/test_auto_gpu_layers.py and tests/test_gpu_split_wiring.py:
the REAL methods run; only the VRAM readings (localm.discover.list_gpus /
the single-device probe pair), localm.config.load_config, and the model
size/layer count are faked.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from localm.discover import GPU_PROBE_OK, GPU_PROBE_TIMEOUT, vram_capacity
from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp import _loader, _sizing


GB = 1024 ** 3


def _model(tmp_path, size_bytes, *, n_gpu_layers=99, auto=True, n_ctx=4096,
           ctx_auto=False):
    # A tiny REAL file (so is_file() works), with the multi-GB "on disk" size
    # faked via _model_bytes. NEVER truncate() to GB sizes here: Windows
    # truncate() is not sparse and allocates real disk.
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * 4096)
    b = GgufBackend(str(f), n_gpu_layers=n_gpu_layers, n_gpu_layers_auto=auto,
                    n_ctx=n_ctx, ctx_auto=ctx_auto)
    b._model_bytes = lambda: size_bytes
    return b


def _vram(free, total):
    """Deterministic SINGLE-DEVICE reading, all three real paths patched
    together - copied from tests/test_auto_gpu_layers.py (see its docstring
    for why each of the three patches is required on a GPU-equipped box)."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch.object(
        GgufBackend, "_free_total_vram_bytes", return_value=(free, total)))
    stack.enter_context(patch.object(
        _loader, "gpu_memory_isolated",
        return_value=(None if free is None else (free, total))))
    stack.enter_context(patch.object(
        GgufBackend, "_device_global_free_bytes", return_value=None))
    return stack


def _gpus_double(gpus, status=GPU_PROBE_OK):
    """A list_gpus stand-in honouring the status-aware contract the combined
    path opts into (return_status=True, wait_for_inflight=True)."""
    def fake(**kw):
        return (list(gpus), status) if kw.get("return_status") else list(gpus)
    return fake


def _split_box(monkeypatch, per_gpu, *, indices=(0, 1), status=GPU_PROBE_OK):
    """A box with gpu_split_indices configured and list_gpus() seeing the
    given [(free, total), ...] devices. Also pins native_lib_loaded() False
    (we model the PARENT process; an earlier test in this worker could have
    loaded the native lib, which would honestly - but nondeterministically -
    disable the combined path) and gives every listed device an index/name."""
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"gpu_split_indices": list(indices)})
    gpus = [{"index": i, "name": f"GPU {i}", "free": f, "total": t}
            for i, (f, t) in enumerate(per_gpu)]
    monkeypatch.setattr("localm.discover.list_gpus", _gpus_double(gpus, status))
    monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)
    return gpus


def _forbidden_list_gpus(**kw):
    raise AssertionError("list_gpus must not be probed on this path")


# The audit's repro box: 2 x 16 GB, ~15.5 GB free each, a 20 GB model split
# over both. Combined: 31 GB free / 32 GB total.
AUDIT_BOX = [(int(15.5 * GB), 16 * GB), (int(15.5 * GB), 16 * GB)]
MODEL_20GB = 20 * GB


# --------------------------------------------------------------------------- #
#  _split_free_total_bytes (the shared combined-budget source)                 #
# --------------------------------------------------------------------------- #

class TestSplitFreeTotalBytes:
    def test_combined_sum_on_a_detected_split(self, tmp_path, monkeypatch):
        _split_box(monkeypatch, AUDIT_BOX)
        b = _model(tmp_path, MODEL_20GB)
        free, total, devices = b._split_free_total_bytes()
        assert free == 31 * GB
        assert total == 32 * GB
        assert devices == 2

    def test_no_probe_when_no_split_configured(self, tmp_path, monkeypatch):
        # The common single-GPU case must stay probe-free (config answers it).
        monkeypatch.setattr("localm.config.load_config", lambda: {})
        monkeypatch.setattr("localm.discover.list_gpus", _forbidden_list_gpus)
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)
        b = _model(tmp_path, MODEL_20GB)
        assert b._split_free_total_bytes() == (None, None, 0)

    def test_no_probe_inside_the_worker(self, tmp_path, monkeypatch):
        # GgufWorker (native lib loaded in-process): list_gpus would import
        # torch - the documented DLL-conflict class - and a GPU probe has no
        # place in the decode path. The helper must decline WITHOUT probing.
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"gpu_split_indices": [0, 1]})
        monkeypatch.setattr("localm.discover.list_gpus", _forbidden_list_gpus)
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: True)
        b = _model(tmp_path, MODEL_20GB)
        assert b._split_free_total_bytes() == (None, None, 0)

    def test_stale_probe_yields_no_combined_reading(self, tmp_path, monkeypatch):
        # A TIMEOUT/BUSY probe serves a frozen last-known-good list; sizing or
        # refusing from stale data is the rule-5 gap the admission gate's
        # freshness contract prevents - mirror it.
        _split_box(monkeypatch, AUDIT_BOX, status=GPU_PROBE_TIMEOUT)
        b = _model(tmp_path, MODEL_20GB)
        assert b._split_free_total_bytes() == (None, None, 0)

    def test_degraded_split_yields_no_combined_reading(self, tmp_path, monkeypatch):
        # Only one of the configured devices is detected: nothing honest to
        # sum, fall back to the single-device behavior.
        _split_box(monkeypatch, [(15 * GB, 16 * GB)], indices=(0, 1))
        b = _model(tmp_path, MODEL_20GB)
        assert b._split_free_total_bytes() == (None, None, 0)

    def test_tolerates_a_no_kwarg_vram_capacity_double(self, tmp_path, monkeypatch):
        # The codebase's standing test-double contract (_vram_free_reading's
        # posture): a patched vram_capacity that rejects the opt-in kwargs
        # means "no combined reading", never a crash.
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"gpu_split_indices": [0, 1]})
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)
        monkeypatch.setattr("localm.discover.vram_capacity",
                            lambda config=None: {"total": 16 * GB, "free": 15 * GB})
        b = _model(tmp_path, MODEL_20GB)
        assert b._split_free_total_bytes() == (None, None, 0)

    def test_plain_dict_double_without_devices_is_not_combined(
            self, tmp_path, monkeypatch):
        # A kwargs-accepting double that returns the classic single-GPU dict
        # (no "devices" key) must not be mistaken for a combined figure.
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"gpu_split_indices": [0, 1]})
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)
        monkeypatch.setattr("localm.discover.vram_capacity",
                            lambda config=None, **kw: {"total": 16 * GB,
                                                       "free": 15 * GB})
        b = _model(tmp_path, MODEL_20GB)
        assert b._split_free_total_bytes() == (None, None, 0)


# --------------------------------------------------------------------------- #
#  _auto_gpu_layers: repro (a) - the silent partial offload                    #
# --------------------------------------------------------------------------- #

class TestAutoGpuLayersSplitAware:
    def test_full_offload_when_the_model_fits_combined(self, tmp_path, monkeypatch):
        # The audit repro: 20 GB model, 31 GB combined free. Split-blind sizing
        # picked 21/32 layers (11 on CPU); combined budgeting must say "all".
        _split_box(monkeypatch, AUDIT_BOX)
        b = _model(tmp_path, MODEL_20GB)
        assert b._auto_gpu_layers() == 99

    def test_partial_sized_against_the_combined_budget(self, tmp_path, monkeypatch):
        # Even combined is tight (8+8 GB free for 20 GB): a partial offload is
        # right, but its fraction must come from the COMBINED budget.
        _split_box(monkeypatch, [(8 * GB, 16 * GB), (8 * GB, 16 * GB)])
        b = _model(tmp_path, MODEL_20GB)
        n = b._auto_gpu_layers()
        kv = 4096 * GgufBackend._bytes_per_token(MODEL_20GB)
        budget = 16 * GB - kv - GgufBackend._VRAM_OVERHEAD_BYTES
        expected = int(min(max(budget / MODEL_20GB, 0.0), 1.0) * 32)
        assert n == expected
        assert 0 < n < 99

    def test_falls_back_to_single_device_on_a_stale_probe(self, tmp_path, monkeypatch):
        # Combined unmeasurable (TIMEOUT): today's single-device sizing stands.
        _split_box(monkeypatch, AUDIT_BOX, status=GPU_PROBE_TIMEOUT)
        b = _model(tmp_path, 8 * GB)
        with _vram(6 * GB, 16 * GB):
            n = b._auto_gpu_layers()
        assert 0 < n < 99   # the pre-fix partial sizing, unchanged

    def test_effective_layers_full_fit_no_scary_notice(self, tmp_path, monkeypatch,
                                                       capsys):
        # End of the audit's case (a): with auto on, the resolved count is a
        # quiet full offload, not a "model too big, offloading N/32" notice.
        _split_box(monkeypatch, AUDIT_BOX)
        b = _model(tmp_path, MODEL_20GB, auto=True)
        assert b._effective_gpu_layers() == 99
        assert "gpu layers auto" not in capsys.readouterr().out.lower()


# --------------------------------------------------------------------------- #
#  _check_vram: repro (b) - the wrong "cannot fit regardless" refusal          #
# --------------------------------------------------------------------------- #

class TestCheckVramSplitAware:
    def test_pinned_full_load_that_fits_combined_is_not_refused(
            self, tmp_path, monkeypatch, capsys):
        # n_gpu_layers_auto=false on the audit box: the pre-fix code raised
        # "this GPU only has 16.0 GB total ... cannot fit regardless" - false,
        # it fits across the configured split.
        _split_box(monkeypatch, AUDIT_BOX)
        b = _model(tmp_path, MODEL_20GB, n_gpu_layers=99, auto=False)
        b._check_vram()   # must not raise
        assert "Low VRAM" not in capsys.readouterr().out   # 31 GB free covers it

    def test_refusal_when_too_big_even_combined_names_the_split(
            self, tmp_path, monkeypatch):
        # 40 GB model on 32 GB combined: refusing is right, but the wording
        # must name the split's combined ceiling, not "this GPU".
        _split_box(monkeypatch, AUDIT_BOX)
        b = _model(tmp_path, 40 * GB, n_gpu_layers=99, auto=False)
        with pytest.raises(RuntimeError) as exc:
            b._check_vram()
        msg = str(exc.value)
        assert "2 GPUs in the configured split" in msg
        assert "combined" in msg
        assert "cannot fit across this split" in msg
        assert "this GPU only has" not in msg
        assert "n_gpu_layers_auto" in msg   # the options stay actionable

    def test_low_vram_warning_uses_combined_figures(self, tmp_path, monkeypatch,
                                                    capsys):
        # Fits the combined TOTAL but not the combined FREE: warn (not refuse),
        # quoting the combined free and naming the split.
        _split_box(monkeypatch, [(4 * GB, 16 * GB), (4 * GB, 16 * GB)])
        b = _model(tmp_path, MODEL_20GB, n_gpu_layers=99, auto=False)
        b._check_vram()   # warns, never raises (freeing VRAM could still fix it)
        out = capsys.readouterr().out
        assert "Low VRAM" in out
        assert "across the 2 GPUs in the configured split" in out

    def test_single_gpu_refusal_wording_unchanged(self, tmp_path):
        # No split configured: the established message survives verbatim.
        b = _model(tmp_path, MODEL_20GB, n_gpu_layers=99, auto=False)
        with patch.object(GgufBackend, "_split_free_total_bytes",
                          return_value=(None, None, 0)), \
             patch.object(GgufBackend, "_free_vram_bytes", return_value=15 * GB), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16 * GB):
            with pytest.raises(RuntimeError, match="cannot fit regardless"):
                b._check_vram()


# --------------------------------------------------------------------------- #
#  _auto_ctx_max: repro (c) - the growth ceiling collapsing to n_ctx           #
# --------------------------------------------------------------------------- #

class TestAutoCtxMaxSplitAware:
    def test_ceiling_derived_from_the_combined_budget(self, tmp_path, monkeypatch):
        # 31 GB combined free - 20 GB weights - overhead: ~10 GB of KV headroom.
        # The split-blind reading (15.5 GB free on one card) put the budget
        # underwater and collapsed the ceiling to n_ctx (4096).
        _split_box(monkeypatch, AUDIT_BOX)
        monkeypatch.setattr(_sizing, "embedder_ctx_reservation_bytes", lambda: 0)
        b = _model(tmp_path, MODEL_20GB)
        auto = b._auto_ctx_max()
        budget = 31 * GB - MODEL_20GB - GgufBackend._VRAM_OVERHEAD_BYTES
        expected = (budget // GgufBackend._bytes_per_token(MODEL_20GB)) // 1024 * 1024
        expected = int(max(4096, min(65536, expected)))
        assert auto == expected
        assert auto > 4096   # the collapse is gone

    def test_falls_back_to_single_device_when_split_degraded(
            self, tmp_path, monkeypatch):
        # One detected device: the combined path declines, and the ceiling is
        # sized exactly as before the fix from the single-device reading.
        _split_box(monkeypatch, [(int(15.5 * GB), 16 * GB)], indices=(0, 1))
        monkeypatch.setattr(_sizing, "embedder_ctx_reservation_bytes", lambda: 0)
        b = _model(tmp_path, MODEL_20GB)
        with _vram(int(15.5 * GB), 16 * GB):
            auto = b._auto_ctx_max()
        assert auto == max(b.n_ctx, b._AUTO_CTX_MIN)   # budget <= 0 -> floor


# --------------------------------------------------------------------------- #
#  _check_context_fit: the mid-generation grow check (worker-resident)         #
# --------------------------------------------------------------------------- #

class TestCheckContextFitSplitAware:
    def _grown(self, tmp_path, per_token, *, offload_kqv=True):
        b = _model(tmp_path, MODEL_20GB, n_gpu_layers=99, auto=False)
        b.effective_gpu_layers = 99
        b._llm = SimpleNamespace(kv_bytes_per_token=per_token,
                                 _offload_kqv=offload_kqv)
        return b

    def test_combined_free_admits_a_grow_one_card_could_not(
            self, tmp_path, monkeypatch):
        # The KV delta fits the split's combined free but not one card's: the
        # KV cache must STAY in VRAM (True). The single-device reading is
        # patched too small on purpose - a split-blind check would say False.
        per_token = 100_000
        delta = (8192 - 4096) * per_token          # ~0.4 GB of new KV
        _split_box(monkeypatch, [(delta, 16 * GB), (delta, 16 * GB)])
        b = self._grown(tmp_path, per_token)
        with patch.object(GgufBackend, "_free_vram_bytes",
                          return_value=delta // 2):
            assert b._check_context_fit(8192, current_ctx=4096) is True

    def test_combined_free_too_small_moves_kv_to_ram(self, tmp_path, monkeypatch):
        per_token = 100_000
        delta = (8192 - 4096) * per_token
        _split_box(monkeypatch, [(delta // 8, 16 * GB), (delta // 8, 16 * GB)])
        b = self._grown(tmp_path, per_token)
        assert b._check_context_fit(8192, current_ctx=4096) is False
        assert b._ram_kv_hint_shown is True

    def test_worker_never_probes_and_stays_honest(self, tmp_path, monkeypatch):
        # Inside the worker (native lib loaded): no list_gpus probe may run,
        # and with the isolated native probe declining (2+ devices) the check
        # honestly answers None (keep the default) - today's behavior.
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"gpu_split_indices": [0, 1]})
        monkeypatch.setattr("localm.discover.list_gpus", _forbidden_list_gpus)
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: True)
        monkeypatch.setattr(_loader, "gpu_memory_isolated", lambda: None)
        b = self._grown(tmp_path, 100_000)
        assert b._check_context_fit(8192, current_ctx=4096) is None


# --------------------------------------------------------------------------- #
#  load() end to end on the audit box                                          #
# --------------------------------------------------------------------------- #

class TestLoadEndToEndSplit:
    def test_load_resolves_full_offload_and_does_not_refuse(
            self, tmp_path, monkeypatch):
        # The whole pre-fix failure chain at once: auto on, 20 GB model,
        # 2 x 16 GB split box -> load() must resolve 99 layers and reach
        # _load_native (stubbed), not refuse or partial-offload.
        _split_box(monkeypatch, AUDIT_BOX)
        b = _model(tmp_path, MODEL_20GB, auto=True)
        with patch.object(GgufBackend, "_load_native", lambda self: None):
            b.load()
        assert b.effective_gpu_layers == 99


# --------------------------------------------------------------------------- #
#  vram_capacity(combined_only=True) - the summing source's own contract       #
# --------------------------------------------------------------------------- #

class TestVramCapacityCombinedOnly:
    CFG = {"gpu_split_indices": [0, 1]}

    def test_no_split_is_probe_free_and_empty(self, monkeypatch):
        monkeypatch.setattr("localm.discover.list_gpus", _forbidden_list_gpus)
        assert vram_capacity({}, combined_only=True) == {}
        info, status = vram_capacity({}, combined_only=True, return_status=True)
        assert info == {} and status == GPU_PROBE_OK

    def test_summed_dict_carries_the_device_count(self, monkeypatch):
        monkeypatch.setattr(
            "localm.discover.list_gpus",
            _gpus_double([{"index": 0, "name": "A", "free": 3 * GB, "total": 8 * GB},
                          {"index": 1, "name": "B", "free": 5 * GB, "total": 8 * GB}]))
        info = vram_capacity(self.CFG, combined_only=True)
        assert info == {"total": 16 * GB, "free": 8 * GB, "devices": 2}

    def test_degraded_split_returns_empty_not_single_gpu(self, monkeypatch):
        monkeypatch.setattr(
            "localm.discover.list_gpus",
            _gpus_double([{"index": 0, "name": "A", "free": 3 * GB,
                           "total": 8 * GB}]))
        # The classic call degrades to the single-GPU vram_info() figure; the
        # combined_only caller must get NOTHING instead of that stand-in.
        assert vram_capacity(self.CFG, combined_only=True) == {}

    def test_classic_shape_has_no_devices_key(self, monkeypatch):
        monkeypatch.setattr(
            "localm.discover.list_gpus",
            _gpus_double([{"index": 0, "name": "A", "free": 3 * GB, "total": 8 * GB},
                          {"index": 1, "name": "B", "free": 5 * GB, "total": 8 * GB}]))
        info = vram_capacity(self.CFG)
        assert "devices" not in info
        assert info["total"] == 16 * GB
