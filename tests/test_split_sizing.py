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


def _implicit_box(monkeypatch, per_gpu, *, vulkan=False, status=GPU_PROBE_OK,
                  dev_types=None):
    """A box with NO gpu_split_indices, where llama.cpp's own default layer
    split spreads the load over the given [(free, total), ...] devices.

    Pins _native_backend_has_vulkan explicitly rather than letting it read this
    machine: the vulkan branch reads the ggml registry (native_gpu_devices) and
    the other reads torch's view (list_gpus), so an unpinned value silently
    decides WHICH double is consulted - and on a box where the other one is
    live, a test can pass without its own fixture ever being read."""
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)
    monkeypatch.setattr("localm.discover._native_backend_has_vulkan",
                        lambda: vulkan)
    # type defaults to GGML_DEV_TYPE_GPU (discrete). It is carried even on the
    # list_gpus path, where it is ignored, so the two fixtures stay comparable.
    # A fixture WITHOUT a type cannot fail the discrete-only filter at all -
    # the field that decides right from wrong has to be in the data.
    devices = [{"index": i, "name": f"GPU {i}", "free": f, "total": t,
                "type": (dev_types[i] if dev_types else 1)}
               for i, (f, t) in enumerate(per_gpu)]
    if vulkan:
        monkeypatch.setattr("localm.discover.native_gpu_devices",
                            lambda: list(devices))
        monkeypatch.setattr("localm.discover.list_gpus", _forbidden_list_gpus)
    else:
        monkeypatch.setattr("localm.discover.list_gpus",
                            _gpus_double(devices, status))
    return devices


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

    def test_one_device_and_no_split_configured_yields_no_combined_reading(
            self, tmp_path, monkeypatch):
        # No gpu_split_indices AND a single detected GPU: nothing to combine,
        # so the single-device reading stands. This is the single-GPU majority
        # and it must be untouched by the implicit-split work.
        #
        # It USED to assert the stronger "and it never probes at all", which
        # stopped being true once we learned llama.cpp splits implicitly (an
        # unset split is not a single-GPU load, so the device COUNT has to be
        # looked up). That assertion was also dead: it raised AssertionError
        # from a list_gpus double, and _split_free_total_bytes' own
        # `except Exception` swallowed it, so it passed either way.
        _implicit_box(monkeypatch, [(8 * GB, 16 * GB)])
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
        # 2x the overhead: each device that holds layers reserves its own
        # compute buffer, so an N-device split reserves N of them (see
        # _split_overhead_bytes). The COMBINED FREE is what this test is about,
        # and that is unchanged - only the overhead term is per-device.
        budget = 16 * GB - kv - 2 * GgufBackend._VRAM_OVERHEAD_BYTES
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
        # 2x the overhead - one compute buffer per device holding layers; see
        # _split_overhead_bytes. The combined 31 GB free is the subject here.
        budget = 31 * GB - MODEL_20GB - 2 * GgufBackend._VRAM_OVERHEAD_BYTES
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


# --------------------------------------------------------------------------- #
#  The IMPLICIT split: no gpu_split_indices, and llama.cpp splits anyway       #
# --------------------------------------------------------------------------- #
#
# Field finding (BLACKWELL, 2026-08-11): a three-GPU box loaded 39.24 GB across
# its cards and left 21.5 GB idle, because the sizing budgeted the whole load
# against ONE card. It did that whenever gpu_split_indices was unset, on the
# premise that an unconfigured box runs single-GPU. That premise is false.
# llama_model_default_params() sets split_mode = LLAMA_SPLIT_MODE_LAYER with
# tensor_split = NULL, llama.cpp narrows to main_gpu only under
# LLAMA_SPLIT_MODE_NONE (which localm never sets), and a NULL tensor_split takes
# the "default split, by free memory" branch across every registered GPU.
# See dev-notes/MULTI-GPU-SIZING-split-policy-2026-08-11.md for the source.

# Deliberately UNEVEN, and free != total on every card. A fixture of identical
# GPUs cannot fail on the overcommit question at all (every candidate budget
# agrees when the cards agree), so this set is what makes the smallest-card
# assertion below able to go red - diff-review-discipline item 19.
#   free  22 + 23 +  6 = 51 GB      total  24 + 24 + 16 = 64 GB
UNEVEN_BOX = [(22 * GB, 24 * GB), (23 * GB, 24 * GB), (6 * GB, 16 * GB)]
MODEL_45GB = 45 * GB


def _shares(free_by_device):
    """llama.cpp's own default split fractions: splits[i] = free_i, cumulative,
    normalised (llama-model.cpp, the all_zero branch). So device i receives
    free_i / SUM(free) of the offloaded layers, and their KV with them - the KV
    cache follows its layer's device."""
    total = sum(free_by_device)
    return [f / total for f in free_by_device]


class TestImplicitSplitSizing:
    def test_combined_budget_spans_every_card_not_just_one(self, tmp_path,
                                                           monkeypatch):
        # The headline regression. No split configured, 3 GPUs present.
        _implicit_box(monkeypatch, UNEVEN_BOX)
        b = _model(tmp_path, MODEL_45GB)
        free, total, devices = b._split_free_total_bytes()
        assert free == 51 * GB          # NOT 22 GB, the main card alone
        assert total == 64 * GB
        assert devices == 3

    def test_layer_budget_accounts_for_the_full_set(self, tmp_path, monkeypatch):
        # _auto_gpu_layers must size the offload from the combined 51 GB, not
        # the 22 GB main card. Assert the NUMBER, and that it beats what the
        # one-card budget produces - that difference IS the defect.
        _implicit_box(monkeypatch, UNEVEN_BOX)
        b = _model(tmp_path, MODEL_45GB)
        n = b._auto_gpu_layers()

        kv = 4096 * GgufBackend._bytes_per_token(MODEL_45GB)
        combined = 51 * GB - kv - 3 * GgufBackend._VRAM_OVERHEAD_BYTES
        expected = int(min(max(combined / MODEL_45GB, 0.0), 1.0) * 32)
        assert n == expected
        assert 0 < n < 99               # still a partial offload, honestly sized

        one_card = 22 * GB - kv - GgufBackend._VRAM_OVERHEAD_BYTES
        split_blind = int(min(max(one_card / MODEL_45GB, 0.0), 1.0) * 32)
        assert n > split_blind          # the idle-VRAM bug, in one assertion

    def test_context_budget_accounts_for_the_full_set(self, tmp_path, monkeypatch):
        # The finding named context sizing too, not only layers: the KV cache
        # lands on the same devices as the layers it belongs to.
        _implicit_box(monkeypatch, UNEVEN_BOX)
        monkeypatch.setattr(_sizing, "embedder_ctx_reservation_bytes", lambda: 0)
        b = _model(tmp_path, MODEL_20GB)
        auto = b._auto_ctx_max(capped=False)   # uncapped: the raw arithmetic

        budget = 51 * GB - MODEL_20GB - 3 * GgufBackend._VRAM_OVERHEAD_BYTES
        expected = (budget // GgufBackend._bytes_per_token(MODEL_20GB)) // 1024 * 1024
        assert auto == int(max(4096, expected))

        # The one-card budget is not underwater here, it is merely tiny: 22 - 20
        # - 1.5 leaves 0.6 GB of KV headroom against 26.8 GB combined. So assert
        # the CEILING, not the sign - a ~40x difference in usable context.
        one_card = 22 * GB - MODEL_20GB - GgufBackend._VRAM_OVERHEAD_BYTES
        split_blind = ((one_card // GgufBackend._bytes_per_token(MODEL_20GB))
                       // 1024 * 1024)
        assert auto > int(max(4096, split_blind))

    def test_smallest_card_is_not_overcommitted(self, tmp_path, monkeypatch):
        # THE SAFETY DIRECTION. Summing free is only correct because llama.cpp
        # weights the split BY FREE MEMORY: device i gets free_i/SUM(free) of
        # the layers, so a SUM(free) budget places exactly free_i on device i.
        # Assert that per device, on a set where the cards genuinely disagree.
        #
        # This is what kills the plausible-but-wrong budgets: SUM(total), 64 GB,
        # would hand the 6 GB card 64 * 6/51 = 7.5 GB, and a "3x the main card's
        # free" shortcut, 66 GB, would hand it 7.8 GB. Both overcommit the one
        # card that cannot take it, which is an OOM on a user's box.
        _implicit_box(monkeypatch, UNEVEN_BOX)
        b = _model(tmp_path, MODEL_45GB)
        free, _total, devices = b._split_free_total_bytes()

        per_device_free = [f for f, _t in UNEVEN_BOX]
        # What actually gets placed: the budget net of the per-device overhead.
        placed = free - b._split_overhead_bytes(devices)
        for share, card_free in zip(_shares(per_device_free), per_device_free):
            assert placed * share <= card_free

        smallest = min(per_device_free)
        assert placed * _shares(per_device_free)[-1] < smallest   # strictly under
        assert smallest == 6 * GB                # the fixture really is uneven

    def test_vulkan_reads_the_native_registry_not_torch(self, tmp_path,
                                                        monkeypatch):
        # GPU-SPLIT-VKINDEX: on the vulkan build list_gpus speaks torch's index
        # space and is structurally blind to ggml's. The sum must come from the
        # space that actually receives the layers. _implicit_box makes list_gpus
        # explode here, so reaching the right number proves which one was read.
        _implicit_box(monkeypatch, UNEVEN_BOX, vulkan=True)
        b = _model(tmp_path, MODEL_45GB)
        assert b._split_free_total_bytes() == (51 * GB, 64 * GB, 3)

    def test_igpu_alongside_a_discrete_card_is_not_summed(self, tmp_path,
                                                          monkeypatch):
        # THE OOM CASE, and an ordinary one: any laptop, or any desktop CPU with
        # integrated graphics, on the vulkan build. The native registry reports
        # every non-CPU device, but llama.cpp appends integrated GPUs ONLY when
        # no discrete GPU was found - so it places the whole load on the two
        # discrete cards. Summing the iGPU's memory in would budget 8 GB that
        # llama.cpp never uses, and over-budgeting is the direction that OOMs.
        # GGML_DEV_TYPE: CPU 0, GPU 1, IGPU 2.
        _implicit_box(monkeypatch,
                      [(22 * GB, 24 * GB), (23 * GB, 24 * GB), (8 * GB, 8 * GB)],
                      vulkan=True, dev_types=[1, 1, 2])
        b = _model(tmp_path, MODEL_45GB)
        # 45 GB free / 48 GB total over the two DISCRETE cards - the iGPU's
        # 8 GB is excluded, and it does not count toward the device total.
        assert b._split_free_total_bytes() == (45 * GB, 48 * GB, 2)

    def test_one_discrete_card_plus_an_igpu_is_a_single_gpu_box(self, tmp_path,
                                                                monkeypatch):
        # Same filter, the commoner shape: llama.cpp puts everything on the one
        # discrete card, so there is no combined budget at all and the
        # single-device reading must stand.
        _implicit_box(monkeypatch, [(22 * GB, 24 * GB), (8 * GB, 8 * GB)],
                      vulkan=True, dev_types=[1, 2])
        b = _model(tmp_path, MODEL_45GB)
        assert b._split_free_total_bytes() == (None, None, 0)

    def test_untyped_native_devices_decline_rather_than_assume_discrete(
            self, tmp_path, monkeypatch):
        # The probe did not report a device class. We cannot tell a discrete
        # card from an iGPU, and guessing "discrete" is the OOM direction, so
        # the combined budget is declined.
        monkeypatch.setattr("localm.config.load_config", lambda: {})
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)
        monkeypatch.setattr("localm.discover._native_backend_has_vulkan",
                            lambda: True)
        monkeypatch.setattr("localm.discover.native_gpu_devices", lambda: [
            {"index": 0, "name": "A", "free": 22 * GB, "total": 24 * GB},
            {"index": 1, "name": "B", "free": 23 * GB, "total": 24 * GB}])
        b = _model(tmp_path, MODEL_45GB)
        assert b._split_free_total_bytes() == (None, None, 0)

    def test_a_blind_device_declines_rather_than_undercounting(self, tmp_path,
                                                               monkeypatch):
        # All-or-nothing: one device with no free reading must not be summed as
        # 0 (under-count) nor assumed empty (over-count, an OOM).
        box = [{"index": 0, "name": "A", "free": 22 * GB, "total": 24 * GB},
               {"index": 1, "name": "B", "total": 24 * GB}]
        monkeypatch.setattr("localm.config.load_config", lambda: {})
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)
        monkeypatch.setattr("localm.discover._native_backend_has_vulkan",
                            lambda: False)
        monkeypatch.setattr("localm.discover.list_gpus", _gpus_double(box))
        b = _model(tmp_path, MODEL_45GB)
        assert b._split_free_total_bytes() == (None, None, 0)

    def test_stale_probe_declines(self, tmp_path, monkeypatch):
        # A non-OK probe serves a frozen last-known-good list; sizing from it is
        # the rule-5 gap the admission gate's freshness contract closes.
        _implicit_box(monkeypatch, UNEVEN_BOX, status=GPU_PROBE_TIMEOUT)
        b = _model(tmp_path, MODEL_45GB)
        assert b._split_free_total_bytes() == (None, None, 0)


class TestSingleGpuPathUnchanged:
    """The regression that matters most: one card is the field's commonest
    configuration and it works today. Every number below is the FLAT-overhead
    arithmetic that shipped before the implicit-split change."""

    def test_layer_sizing_is_byte_identical_on_one_card(self, tmp_path,
                                                        monkeypatch):
        _implicit_box(monkeypatch, [(10 * GB, 16 * GB)])
        b = _model(tmp_path, MODEL_20GB)
        with _vram(10 * GB, 16 * GB):
            n = b._auto_gpu_layers()
        kv = 4096 * GgufBackend._bytes_per_token(MODEL_20GB)
        budget = 10 * GB - kv - GgufBackend._VRAM_OVERHEAD_BYTES   # FLAT, x1
        assert n == int(min(max(budget / MODEL_20GB, 0.0), 1.0) * 32)
        assert 0 < n < 99

    def test_context_sizing_is_byte_identical_on_one_card(self, tmp_path,
                                                          monkeypatch):
        _implicit_box(monkeypatch, [(30 * GB, 32 * GB)])
        monkeypatch.setattr(_sizing, "embedder_ctx_reservation_bytes", lambda: 0)
        b = _model(tmp_path, MODEL_20GB)
        with _vram(30 * GB, 32 * GB):
            auto = b._auto_ctx_max(capped=False)
        budget = 30 * GB - MODEL_20GB - GgufBackend._VRAM_OVERHEAD_BYTES  # FLAT
        expected = (budget // GgufBackend._bytes_per_token(MODEL_20GB)) // 1024 * 1024
        assert auto == int(max(4096, expected))

    def test_overhead_is_the_flat_constant_for_one_device(self, tmp_path):
        b = _model(tmp_path, MODEL_20GB)
        assert b._split_overhead_bytes(1) == GgufBackend._VRAM_OVERHEAD_BYTES
        # 0 means "no combined reading" - the single-device fallback.
        assert b._split_overhead_bytes(0) == GgufBackend._VRAM_OVERHEAD_BYTES
        assert b._split_overhead_bytes(3) == 3 * GgufBackend._VRAM_OVERHEAD_BYTES

    def test_check_vram_refusal_wording_unchanged_on_one_card(self, tmp_path,
                                                              monkeypatch):
        _implicit_box(monkeypatch, [(15 * GB, 16 * GB)])
        b = _model(tmp_path, MODEL_45GB, n_gpu_layers=99, auto=False)
        with _vram(15 * GB, 16 * GB):
            with pytest.raises(RuntimeError, match="cannot fit regardless"):
                b._check_vram()


class TestImplicitSplitDecisionIsRecorded:
    """The combined budget must leave a trace naming the figures it used.

    Rule 5, surface the decision - the same contract resolve_auto_split_ratios
    states for its own INFO line. Until this existed only the DECLINE path
    logged, so a bug report about a wrongly-sized load could not distinguish a
    budget taken across the whole board from one taken against a single card,
    which is exactly the defect implicit_split_capacity was added to fix.
    """

    def test_combined_budget_logs_the_devices_and_the_figures(
            self, tmp_path, monkeypatch, caplog):
        import logging
        _implicit_box(monkeypatch, UNEVEN_BOX)
        b = _model(tmp_path, MODEL_45GB)
        with caplog.at_level(logging.INFO, logger="localm"):
            assert b._split_free_total_bytes() == (51 * GB, 64 * GB, 3)
        line = "\n".join(r.getMessage() for r in caplog.records)
        assert "implicit GPU split" in line
        assert "3 devices" in line
        # The COMBINED figure, and the per-device readings behind it - a line
        # naming neither could not answer the question a bug report asks.
        assert "51.0 GB free" in line
        assert "64.0 GB total" in line
        assert "22.0 GB free" in line and "6.0 GB free" in line

    def test_single_gpu_box_stays_silent(self, tmp_path, monkeypatch, caplog):
        import logging
        _implicit_box(monkeypatch, [(8 * GB, 16 * GB)])
        b = _model(tmp_path, MODEL_20GB)
        with caplog.at_level(logging.INFO, logger="localm"):
            assert b._split_free_total_bytes() == (None, None, 0)
        assert "implicit GPU split" not in "\n".join(
            r.getMessage() for r in caplog.records)
