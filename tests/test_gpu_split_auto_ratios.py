# SPDX-License-Identifier: AGPL-3.0-or-later
"""Automatic free-VRAM-proportional GPU split distribution.

Feature under test: when ``gpu_split_indices`` names 2+ devices and
``gpu_split_ratios`` is UNSET (None or []), localm distributes the model
across the split devices proportionally to each device's CURRENT free VRAM
instead of the historical equal split. An explicitly configured
``gpu_split_ratios`` is never overridden; every unmeasurable case falls back
to the equal split, byte-identical to the pre-feature behavior.

The decision is made ONCE, parent-side (discover.resolve_auto_split_ratios),
and PINNED into the isolated worker via the load params
(gguf.py -> GgufWorker -> LlamaCpp -> apply_gpu_split(ratios_override=...);
embedder: IsolatedEmbedder._reload -> GGUFEmbedder). The worker never probes:
on Windows + AMD a torch import inside a native-runtime process hits a DLL
identity conflict, and only the parent has the device-global corrected
readings anyway. On the vulkan build the per-device reading comes from
discover.native_gpu_devices() - the ONLY source in ggml-vulkan's own index
space, the space tensor_split actually consumes.

gpu_split_shortfall computes its per-device shares with the same auto ratios
(from its own fresh reading), which makes the asymmetric-occupancy refusal
(one device short while the aggregate fits) structurally impossible in auto
mode: needed_i = R * free_i / total_free <= free_i whenever R <= total_free.
switch_engine's shortfall branch therefore only fires in auto mode when the
COMBINED estimate is short, where it defers to the backend's split-aware
sizing exactly like the single-GPU path, while pinned ratios keep the hard
per-device 503 (sizing budgets the split's COMBINED capacity, never one pinned
share, so the gate is the only protection there).

Mocking style: mocked ctypes api with a REAL apply_gpu_split, and TestClient +
FakeEngine + probe_double - the functions under test always run for real.
"""

import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from localm import discover
from localm.discover import resolve_auto_split_ratios
from localm.inference import http_server as hs
from tests.conftest import probe_double
from tests.test_vram_eviction_safety import FakeEngine, _chat

GB = 1024 ** 3


# --------------------------------------------------------------------------- #
#  resolve_auto_split_ratios (the parent-side decision)                         #
# --------------------------------------------------------------------------- #

class TestResolveAutoSplitRatios:
    # free_scope=device on every entry: this class's fixtures represent a TRUSTED
    # reading throughout. The untrusted-scope cases live in TestScopeTrust below
    # and build their own fixtures rather than mutating this shared one.
    _GPUS = [
        {"index": 0, "name": "A", "total": 16 * GB, "free": 12 * GB,
         "free_scope": discover.FREE_SCOPE_DEVICE},
        {"index": 1, "name": "B", "total": 16 * GB, "free": 4 * GB,
         "free_scope": discover.FREE_SCOPE_DEVICE},
    ]

    def _no_vulkan(self, monkeypatch):
        monkeypatch.setattr(discover, "_native_backend_has_vulkan", lambda: False)

    def test_proportional_from_free(self, monkeypatch):
        self._no_vulkan(monkeypatch)
        monkeypatch.setattr(discover, "list_gpus", probe_double(self._GPUS))
        ratios = resolve_auto_split_ratios(
            {"gpu_split_indices": [0, 1], "gpu_split_ratios": None})
        assert ratios == pytest.approx([0.75, 0.25])

    def test_ratios_align_with_configured_index_order(self, monkeypatch):
        """The returned list pairs by POSITION with gpu_split_indices, exactly
        the contract resolve_gpu_split's re-pairing relies on - so a config
        listing the small card first gets the small share first."""
        self._no_vulkan(monkeypatch)
        monkeypatch.setattr(discover, "list_gpus", probe_double(self._GPUS))
        ratios = resolve_auto_split_ratios(
            {"gpu_split_indices": [1, 0], "gpu_split_ratios": None})
        assert ratios == pytest.approx([0.25, 0.75])

    def test_pinned_ratios_disable_auto(self, monkeypatch):
        """Never override an explicit user choice: a configured
        gpu_split_ratios means auto must not even compute."""
        self._no_vulkan(monkeypatch)
        called = {"probe": False}

        def _probe(*a, **k):
            called["probe"] = True
            return self._GPUS

        monkeypatch.setattr(discover, "list_gpus", _probe)
        assert resolve_auto_split_ratios(
            {"gpu_split_indices": [0, 1], "gpu_split_ratios": [2.0, 1.0]}) is None
        assert not called["probe"], "auto must not probe when ratios are pinned"

    def test_empty_list_ratios_count_as_unset(self, monkeypatch):
        """[] has always meant "unset" (resolve_gpu_split's own falsy check),
        so it engages auto exactly like None."""
        self._no_vulkan(monkeypatch)
        monkeypatch.setattr(discover, "list_gpus", probe_double(self._GPUS))
        ratios = resolve_auto_split_ratios(
            {"gpu_split_indices": [0, 1], "gpu_split_ratios": []})
        assert ratios == pytest.approx([0.75, 0.25])

    @pytest.mark.parametrize("indices", [None, [], [0]])
    def test_fewer_than_two_indices_no_auto_and_no_probe(self, monkeypatch, indices):
        self._no_vulkan(monkeypatch)
        called = {"probe": False}

        def _probe(*a, **k):
            called["probe"] = True
            return self._GPUS

        monkeypatch.setattr(discover, "list_gpus", _probe)
        assert resolve_auto_split_ratios(
            {"gpu_split_indices": indices, "gpu_split_ratios": None}) is None
        assert not called["probe"], "no split -> auto must answer from config alone"

    def test_missing_free_on_one_device_all_or_nothing(self, monkeypatch):
        """A split where one device's free is unmeasurable cannot be
        distributed honestly - guessing a share for the blind device could
        overload it. All-or-nothing, mirroring vram_capacity's 'free' key."""
        self._no_vulkan(monkeypatch)
        gpus = [
            {"index": 0, "name": "A", "total": 16 * GB, "free": 12 * GB,
             "free_scope": discover.FREE_SCOPE_DEVICE},
            {"index": 1, "name": "B", "total": 16 * GB},   # no "free" key
        ]
        monkeypatch.setattr(discover, "list_gpus", probe_double(gpus))
        assert resolve_auto_split_ratios(
            {"gpu_split_indices": [0, 1], "gpu_split_ratios": None}) is None

    def test_unknown_configured_index_disables_auto(self, monkeypatch):
        self._no_vulkan(monkeypatch)
        monkeypatch.setattr(discover, "list_gpus", probe_double(self._GPUS))
        assert resolve_auto_split_ratios(
            {"gpu_split_indices": [0, 5], "gpu_split_ratios": None}) is None

    @staticmethod
    def _stale_probe(gpus, status):
        """A status-CAPABLE list_gpus double with a NAMED return_status param:
        _list_gpus_reading's signature inspection treats a **kwargs-only double
        as the historical bare contract (= completed probe), so simulating a
        non-OK probe needs the explicit signature."""
        def fake(*a, return_status=False, **k):
            return (gpus, status) if return_status else gpus
        return fake

    def test_stale_probe_disables_auto(self, monkeypatch):
        """A TIMEOUT/BUSY probe serves a frozen last-known-good reading;
        distributing by a stale snapshot is the rule-5 gap the probe-status
        contract exists to prevent (same posture as gpu_split_shortfall)."""
        self._no_vulkan(monkeypatch)
        monkeypatch.setattr(
            discover, "list_gpus",
            self._stale_probe(self._GPUS, discover.GPU_PROBE_TIMEOUT))
        assert resolve_auto_split_ratios(
            {"gpu_split_indices": [0, 1], "gpu_split_ratios": None}) is None

    def test_zero_free_device_keeps_a_positive_share(self, monkeypatch):
        """resolve_gpu_split discards the WHOLE ratio list on any entry <= 0,
        which would silently hand a completely full card an EQUAL share - the
        exact overload auto exists to avoid. A 0-free device therefore gets a
        tiny positive share (~nothing lands on it), never a 0 that would
        invalidate the list."""
        self._no_vulkan(monkeypatch)
        gpus = [
            {"index": 0, "name": "A", "total": 16 * GB, "free": 8 * GB,
             "free_scope": discover.FREE_SCOPE_DEVICE},
            {"index": 1, "name": "B", "total": 16 * GB, "free": 0,
             "free_scope": discover.FREE_SCOPE_DEVICE},
        ]
        monkeypatch.setattr(discover, "list_gpus", probe_double(gpus))
        ratios = resolve_auto_split_ratios(
            {"gpu_split_indices": [0, 1], "gpu_split_ratios": None})
        assert ratios is not None
        assert all(r > 0 for r in ratios)
        assert ratios[0] == pytest.approx(1.0, abs=1e-6)
        assert ratios[1] < 1e-6

    def test_injected_gpus_skip_the_probe(self, monkeypatch):
        """gpu_split_shortfall already holds a fresh GPU_PROBE_OK reading; the
        gpus= injection lets it reuse that snapshot instead of paying (and
        possibly disagreeing with) a second probe."""
        self._no_vulkan(monkeypatch)

        def _boom(*a, **k):
            raise AssertionError("list_gpus must not be called when gpus= is given")

        monkeypatch.setattr(discover, "list_gpus", _boom)
        ratios = resolve_auto_split_ratios(
            {"gpu_split_indices": [0, 1], "gpu_split_ratios": None},
            gpus=self._GPUS)
        assert ratios == pytest.approx([0.75, 0.25])

    def test_non_integer_indices_disable_auto(self, monkeypatch):
        self._no_vulkan(monkeypatch)
        monkeypatch.setattr(discover, "list_gpus", probe_double(self._GPUS))
        assert resolve_auto_split_ratios(
            {"gpu_split_indices": ["x", 1], "gpu_split_ratios": None}) is None

    def test_success_logged_at_info(self, monkeypatch, caplog):
        """The distribution decision must reach the INFO+ ring buffer, so a bug
        report about a lopsided split shows WHAT was decided and from WHICH
        readings."""
        self._no_vulkan(monkeypatch)
        monkeypatch.setattr(discover, "list_gpus", probe_double(self._GPUS))
        with caplog.at_level("INFO"):
            resolve_auto_split_ratios(
                {"gpu_split_indices": [0, 1], "gpu_split_ratios": None})
        assert any("auto" in r.message and "split" in r.message
                   for r in caplog.records)

    def test_unmeasurable_split_fallback_logged_at_info(self, monkeypatch, caplog):
        """A user who configured a split should be able to see WHY it fell
        back to the equal distribution: the reason is surfaced, not hidden."""
        self._no_vulkan(monkeypatch)
        monkeypatch.setattr(
            discover, "list_gpus",
            self._stale_probe(self._GPUS, discover.GPU_PROBE_TIMEOUT))
        with caplog.at_level("INFO"):
            resolve_auto_split_ratios(
                {"gpu_split_indices": [0, 1], "gpu_split_ratios": None})
        assert any("equal" in r.message for r in caplog.records)


class TestScopeTrust:
    """The scope half of this function's trustworthiness audit: unlike
    gpu_split_shortfall's refuse-only use of the same reading, a PROPORTIONAL
    split cannot accept a PROCESS-scoped (or untagged) free-VRAM figure on the
    list_gpus() branch - a reading equally blind on every device makes an empty
    card and a nearly-full one look equally free, steering too much of a real
    split onto the full one. That is a materially wrong allocation, not merely
    an imprecise refusal, so it fails toward the safe equal-split fallback."""

    def _no_vulkan(self, monkeypatch):
        monkeypatch.setattr(discover, "_native_backend_has_vulkan", lambda: False)

    _GPUS_DEVICE = [
        {"index": 0, "name": "A", "total": 16 * GB, "free": 12 * GB,
         "free_scope": discover.FREE_SCOPE_DEVICE},
        {"index": 1, "name": "B", "total": 16 * GB, "free": 4 * GB,
         "free_scope": discover.FREE_SCOPE_DEVICE},
    ]

    def test_one_process_scoped_device_disables_auto(self, monkeypatch):
        """A SINGLE process-scoped device corrupts the WHOLE proportional
        comparison (ratios are computed by comparing devices against each
        other), so any one failing scope must decline the entire split, not
        just that device's share."""
        self._no_vulkan(monkeypatch)
        gpus = [
            self._GPUS_DEVICE[0],
            {**self._GPUS_DEVICE[1], "free_scope": discover.FREE_SCOPE_PROCESS},
        ]
        monkeypatch.setattr(discover, "list_gpus", probe_double(gpus))
        assert resolve_auto_split_ratios(
            {"gpu_split_indices": [0, 1], "gpu_split_ratios": None}) is None

    def test_untagged_free_scope_disables_auto(self, monkeypatch):
        """No free_scope key at all never happens from real list_gpus()
        (_apply_device_global_free tags every entry, every platform) - it is
        what a synthetic/legacy double looks like, and must be rejected the
        same as an explicit PROCESS tag rather than silently trusted."""
        self._no_vulkan(monkeypatch)
        gpus = [dict(self._GPUS_DEVICE[0]), dict(self._GPUS_DEVICE[1])]
        del gpus[1]["free_scope"]
        monkeypatch.setattr(discover, "list_gpus", probe_double(gpus))
        assert resolve_auto_split_ratios(
            {"gpu_split_indices": [0, 1], "gpu_split_ratios": None}) is None

    def test_both_device_scoped_succeeds(self, monkeypatch):
        """The trusted case still succeeds: the scope check must not regress the
        healthy path."""
        self._no_vulkan(monkeypatch)
        monkeypatch.setattr(discover, "list_gpus", probe_double(self._GPUS_DEVICE))
        ratios = resolve_auto_split_ratios(
            {"gpu_split_indices": [0, 1], "gpu_split_ratios": None})
        assert ratios == pytest.approx([0.75, 0.25])

    def test_scope_decline_logged_at_info(self, monkeypatch, caplog):
        self._no_vulkan(monkeypatch)
        gpus = [
            self._GPUS_DEVICE[0],
            {**self._GPUS_DEVICE[1], "free_scope": discover.FREE_SCOPE_PROCESS},
        ]
        monkeypatch.setattr(discover, "list_gpus", probe_double(gpus))
        with caplog.at_level("INFO"):
            resolve_auto_split_ratios(
                {"gpu_split_indices": [0, 1], "gpu_split_ratios": None})
        assert any("equal" in r.message for r in caplog.records)
        assert any("device-global" in r.message for r in caplog.records)

    def test_vulkan_branch_is_not_scope_gated(self, monkeypatch):
        """Deliberate asymmetry (see the docstring's TRUSTWORTHINESS section):
        no measurement in this codebase shows ggml-vulkan's own
        ggml_backend_dev_memory query is cross-process blind the way torch's
        mem_get_info / llama.cpp's bundled HIP runtime are. So a platform
        gpu_usage flags as HIP-blind must NOT stop the vulkan branch from
        computing real ratios; asserting an unmeasured blindness would be a
        problem in the other direction."""
        monkeypatch.setattr(discover, "_native_backend_has_vulkan", lambda: True)
        monkeypatch.setattr(discover, "native_gpu_devices", lambda: [
            {"index": 0, "name": "Radeon", "total": 16 * GB, "free": 8 * GB},
            {"index": 1, "name": "llvmpipe", "total": 32 * GB, "free": 24 * GB},
        ])
        monkeypatch.setattr(
            "localm.gpu_usage.raw_reading_is_process_scoped", lambda: True)
        ratios = resolve_auto_split_ratios(
            {"gpu_split_indices": [0, 1], "gpu_split_ratios": None})
        assert ratios == pytest.approx([0.25, 0.75])


class TestResolveAutoSplitRatiosVulkan:
    """On the vulkan build list_gpus() (torch/nvidia-smi) is structurally
    blind to the real split devices AND speaks a different index space, so
    auto must read per-device free from native_gpu_devices() - the
    crash-isolated probe daemon's view of ggml's own registry - and never from
    list_gpus()."""

    _NATIVE = [
        {"index": 0, "name": "Radeon", "total": 16 * GB, "free": 8 * GB},
        {"index": 1, "name": "llvmpipe", "total": 32 * GB, "free": 24 * GB},
    ]

    def _vulkan(self, monkeypatch):
        monkeypatch.setattr(discover, "_native_backend_has_vulkan", lambda: True)

        def _boom(*a, **k):
            raise AssertionError(
                "list_gpus must not be consulted on the vulkan build - its "
                "index space is not tensor_split's (GPU-SPLIT-VKINDEX)")

        monkeypatch.setattr(discover, "list_gpus", _boom)

    def test_vulkan_uses_native_devices(self, monkeypatch):
        self._vulkan(monkeypatch)
        monkeypatch.setattr(discover, "native_gpu_devices", lambda: self._NATIVE)
        ratios = resolve_auto_split_ratios(
            {"gpu_split_indices": [0, 1], "gpu_split_ratios": None})
        assert ratios == pytest.approx([0.25, 0.75])

    def test_vulkan_daemon_unavailable_disables_auto(self, monkeypatch):
        self._vulkan(monkeypatch)
        monkeypatch.setattr(discover, "native_gpu_devices", lambda: None)
        assert resolve_auto_split_ratios(
            {"gpu_split_indices": [0, 1], "gpu_split_ratios": None}) is None

    def test_vulkan_missing_free_disables_auto(self, monkeypatch):
        self._vulkan(monkeypatch)
        native = [
            {"index": 0, "name": "Radeon", "total": 16 * GB, "free": 8 * GB},
            {"index": 1, "name": "llvmpipe"},   # registry reported no bytes
        ]
        monkeypatch.setattr(discover, "native_gpu_devices", lambda: native)
        assert resolve_auto_split_ratios(
            {"gpu_split_indices": [0, 1], "gpu_split_ratios": None}) is None

    def test_vulkan_unknown_index_disables_auto(self, monkeypatch):
        self._vulkan(monkeypatch)
        monkeypatch.setattr(discover, "native_gpu_devices", lambda: self._NATIVE)
        assert resolve_auto_split_ratios(
            {"gpu_split_indices": [0, 3], "gpu_split_ratios": None}) is None

    def test_vulkan_absent_device_is_not_reported_as_unmeasurable(
            self, monkeypatch, caplog):
        """An index past the end of the device list and a device that reported
        no VRAM are DIFFERENT problems and must not share a message.

        native_gpu_devices yields llama.cpp's own device list (integrated GPUs
        and accelerators removed, the rest renumbered), so the ABSENT case is
        ordinary - typically a split saved before that filtering existed.
        Calling it "reported no free-VRAM figure" would send a reader hunting a
        driver fault instead of a stale setting."""
        self._vulkan(monkeypatch)
        monkeypatch.setattr(discover, "native_gpu_devices", lambda: self._NATIVE)
        with caplog.at_level("INFO"):
            assert resolve_auto_split_ratios(
                {"gpu_split_indices": [0, 3], "gpu_split_ratios": None}) is None
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "device 3 is not one of the 2 device(s)" in msgs
        assert "free-VRAM figure" not in msgs

    def test_vulkan_present_but_unmeasurable_device_keeps_the_vram_wording(
            self, monkeypatch, caplog):
        """The other arm, and what makes the test above discriminating: a
        device that IS in the list but reported no bytes must still say so.
        Without this pair, one message covering both cases passes either test
        alone."""
        self._vulkan(monkeypatch)
        native = [
            {"index": 0, "name": "Radeon", "total": 16 * GB, "free": 8 * GB},
            {"index": 1, "name": "llvmpipe"},   # present, registry gave no bytes
        ]
        monkeypatch.setattr(discover, "native_gpu_devices", lambda: native)
        with caplog.at_level("INFO"):
            assert resolve_auto_split_ratios(
                {"gpu_split_indices": [0, 1], "gpu_split_ratios": None}) is None
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "device 1 reported no free-VRAM figure" in msgs
        assert "is not one of" not in msgs


# --------------------------------------------------------------------------- #
#  apply_gpu_split(ratios_override=...) - the worker-side consumption           #
# --------------------------------------------------------------------------- #

class TestApplyGpuSplitRatiosOverride:
    def _mp(self):
        return SimpleNamespace(main_gpu=0, split_mode=1, tensor_split=None)

    def _values(self, mp, count):
        ptr = ctypes.cast(mp.tensor_split, ctypes.POINTER(ctypes.c_float))
        return [ptr[i] for i in range(count)]

    def test_override_written_to_tensor_split(self, monkeypatch):
        monkeypatch.setattr(discover, "list_gpus",
                            lambda: [{"index": 0}, {"index": 1}])
        mp = self._mp()
        keepalive = discover.apply_gpu_split(
            mp, config={"gpu_split_indices": [0, 1], "gpu_split_ratios": None},
            ratios_override=[0.75, 0.25])
        assert keepalive is not None
        values = self._values(mp, 2)
        assert values[0] == pytest.approx(0.75)
        assert values[1] == pytest.approx(0.25)

    def test_override_none_keeps_config_ratios(self, monkeypatch):
        """ratios_override=None is 'no parent decision arrived' - the config
        path (pinned or equal) must behave byte-identically to before the
        kwarg existed."""
        monkeypatch.setattr(discover, "list_gpus",
                            lambda: [{"index": 0}, {"index": 1}])
        mp = self._mp()
        discover.apply_gpu_split(
            mp, config={"gpu_split_indices": [0, 1], "gpu_split_ratios": [3.0, 1.0]},
            ratios_override=None)
        values = self._values(mp, 2)
        assert values == [pytest.approx(3.0), pytest.approx(1.0)]

    def test_override_beats_config_ratios(self, monkeypatch):
        """The parent resolved the effective ratios for THIS load; a config
        value read in the worker (possibly newer/edited mid-flight) must not
        produce a split the parent's admission gate never checked."""
        monkeypatch.setattr(discover, "list_gpus",
                            lambda: [{"index": 0}, {"index": 1}])
        mp = self._mp()
        discover.apply_gpu_split(
            mp, config={"gpu_split_indices": [0, 1], "gpu_split_ratios": [9.0, 1.0]},
            ratios_override=[0.5, 0.5])
        values = self._values(mp, 2)
        assert values == [pytest.approx(0.5), pytest.approx(0.5)]

    def test_override_length_mismatch_degrades_to_equal_with_warning(
            self, monkeypatch, caplog):
        """Same degradation contract as a misconfigured gpu_split_ratios: a
        malformed override falls back to the equal split with a WARNING, never
        a crash or a silent truncation."""
        monkeypatch.setattr(discover, "list_gpus",
                            lambda: [{"index": 0}, {"index": 1}])
        mp = self._mp()
        with caplog.at_level("WARNING"):
            discover.apply_gpu_split(
                mp, config={"gpu_split_indices": [0, 1], "gpu_split_ratios": None},
                ratios_override=[0.5])
        values = self._values(mp, 2)
        assert values == [pytest.approx(1.0), pytest.approx(1.0)]
        assert any("gpu_split_ratios" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
#  gpu_split_shortfall in auto mode                                             #
# --------------------------------------------------------------------------- #

class TestShortfallAutoShares:
    """With unset ratios the gate must judge each device by the share the
    loader will ACTUALLY give it (the auto free-proportional ratio), not the
    historical equal split - otherwise it refuses exactly the loads the
    feature makes possible."""

    _GPUS = [
        {"index": 0, "name": "A", "total": 16 * GB, "free": 2 * GB,
         "free_scope": discover.FREE_SCOPE_DEVICE},
        {"index": 1, "name": "B", "total": 16 * GB, "free": 14 * GB,
         "free_scope": discover.FREE_SCOPE_DEVICE},
    ]

    def _no_vulkan(self, monkeypatch):
        monkeypatch.setattr(discover, "_native_backend_has_vulkan", lambda: False)

    def test_auto_shares_absorb_asymmetric_occupancy(self, monkeypatch):
        """THE headline case: 8 GB ask, devices at 2/14 GB free. The equal
        split flagged GPU 0 (4 GB share vs 2 GB free) even though the load
        fits combined; auto gives GPU 0 only its proportional 1 GB share, so
        nothing is short."""
        self._no_vulkan(monkeypatch)
        monkeypatch.setattr(discover, "list_gpus", probe_double(self._GPUS))
        assert discover.gpu_split_shortfall(
            8 * GB, {"gpu_split_indices": [0, 1], "gpu_split_ratios": None}) == []

    def test_pinned_equal_ratio_still_flags(self, monkeypatch):
        """An EXPLICIT [1, 1] must keep today's behavior exactly - the user
        pinned the shares, so the loader will not adapt, and the gate must
        still catch the device that cannot hold its pinned share."""
        self._no_vulkan(monkeypatch)
        monkeypatch.setattr(discover, "list_gpus", probe_double(self._GPUS))
        result = discover.gpu_split_shortfall(
            8 * GB, {"gpu_split_indices": [0, 1], "gpu_split_ratios": [1.0, 1.0]})
        assert result == [{"index": 0, "needed": 4 * GB, "free": 2 * GB}]

    def test_auto_aggregate_short_flags_proportionally(self, monkeypatch):
        """When even the COMBINED free cannot cover the ask, every device is
        short by its proportional share - the honest per-device figures for
        the caller's message (and switch_engine's cue to defer, in auto
        mode, to the backend's combined-aware sizing)."""
        self._no_vulkan(monkeypatch)
        monkeypatch.setattr(discover, "list_gpus", probe_double(self._GPUS))
        result = discover.gpu_split_shortfall(
            20 * GB, {"gpu_split_indices": [0, 1], "gpu_split_ratios": None})
        assert {d["index"] for d in result} == {0, 1}

    def test_auto_unmeasurable_falls_back_to_equal_shares(self, monkeypatch):
        """One device without a 'free' reading: auto declines (all-or-nothing)
        and the gate keeps the historical equal-split math - which then skips
        the unmeasurable device exactly as before."""
        self._no_vulkan(monkeypatch)
        gpus = [
            {"index": 0, "name": "A", "total": 16 * GB},   # no "free"
            {"index": 1, "name": "B", "total": 16 * GB, "free": 3 * GB},
        ]
        monkeypatch.setattr(discover, "list_gpus", probe_double(gpus))
        result = discover.gpu_split_shortfall(
            8 * GB, {"gpu_split_indices": [0, 1], "gpu_split_ratios": None})
        # Equal split: device 1's 4 GB share vs 3 GB free is short; device 0 is
        # skipped as unmeasurable.
        assert result == [{"index": 1, "needed": 4 * GB, "free": 3 * GB}]


class TestShortfallSharesAdaptiveFlag:
    """return_shares_adaptive: the refuse-vs-defer caller (switch_engine) must
    know which math produced the shares - True only when the live auto ratios
    were actually used. Keying the defer on the CONFIG shape instead admits the
    declined-auto case (equal fallback, per-device hazard fully live)."""

    _GPUS = [
        {"index": 0, "name": "A", "total": 16 * GB, "free": 2 * GB,
         "free_scope": discover.FREE_SCOPE_DEVICE},
        {"index": 1, "name": "B", "total": 16 * GB, "free": 14 * GB,
         "free_scope": discover.FREE_SCOPE_DEVICE},
    ]

    def _no_vulkan(self, monkeypatch):
        monkeypatch.setattr(discover, "_native_backend_has_vulkan", lambda: False)

    def test_true_when_auto_shares_in_effect(self, monkeypatch):
        self._no_vulkan(monkeypatch)
        monkeypatch.setattr(discover, "list_gpus", probe_double(self._GPUS))
        shortfall, adaptive = discover.gpu_split_shortfall(
            8 * GB, {"gpu_split_indices": [0, 1], "gpu_split_ratios": None},
            return_shares_adaptive=True)
        assert shortfall == [] and adaptive is True

    def test_false_when_auto_declines_on_a_stale_index(self, monkeypatch):
        """A configured index no longer detected: auto declines all-or-nothing,
        the surviving devices get the equal fallback, and the flag must say
        static - a non-empty result here is the pre-feature hazard."""
        self._no_vulkan(monkeypatch)
        monkeypatch.setattr(discover, "list_gpus", probe_double(self._GPUS))
        shortfall, adaptive = discover.gpu_split_shortfall(
            8 * GB, {"gpu_split_indices": [0, 1, 5], "gpu_split_ratios": None},
            return_shares_adaptive=True)
        assert adaptive is False
        # Equal fallback across the two survivors: device 0's 4 GB share is short
        # of its 2 GB free, which the 503 still catches.
        assert shortfall == [{"index": 0, "needed": 4 * GB, "free": 2 * GB}]

    def test_false_for_pinned_ratios(self, monkeypatch):
        self._no_vulkan(monkeypatch)
        monkeypatch.setattr(discover, "list_gpus", probe_double(self._GPUS))
        shortfall, adaptive = discover.gpu_split_shortfall(
            8 * GB,
            {"gpu_split_indices": [0, 1], "gpu_split_ratios": [1.0, 1.0]},
            return_shares_adaptive=True)
        assert adaptive is False
        assert shortfall == [{"index": 0, "needed": 4 * GB, "free": 2 * GB}]

    def test_false_on_the_no_split_early_return(self, monkeypatch):
        self._no_vulkan(monkeypatch)
        shortfall, adaptive = discover.gpu_split_shortfall(
            8 * GB, {"gpu_split_indices": None},
            return_shares_adaptive=True)
        assert shortfall == [] and adaptive is False

    def test_composes_with_return_status(self, monkeypatch):
        self._no_vulkan(monkeypatch)
        monkeypatch.setattr(discover, "list_gpus", probe_double(self._GPUS))
        shortfall, status, adaptive = discover.gpu_split_shortfall(
            8 * GB, {"gpu_split_indices": [0, 1], "gpu_split_ratios": None},
            return_status=True, return_shares_adaptive=True)
        assert shortfall == []
        assert status == discover.GPU_PROBE_OK
        assert adaptive is True

    def test_bare_call_shape_unchanged(self, monkeypatch):
        self._no_vulkan(monkeypatch)
        monkeypatch.setattr(discover, "list_gpus", probe_double(self._GPUS))
        result = discover.gpu_split_shortfall(
            8 * GB, {"gpu_split_indices": [0, 1], "gpu_split_ratios": None})
        assert result == []   # a plain list, not a tuple


class TestWaitForInflightForwarding:
    """resolve_auto_split_ratios(wait_for_inflight=True): the load-path
    callers must JOIN a concurrent heartbeat probe instead of taking an
    instant BUSY that silently declines auto into the equal fallback on
    exactly the asymmetric box the feature targets."""

    def test_forwarded_to_a_production_signature_probe(self, monkeypatch):
        monkeypatch.setattr(discover, "_native_backend_has_vulkan", lambda: False)
        seen = {}

        def fake(*a, return_status=False, wait_for_inflight=False, **k):
            seen["wfi"] = wait_for_inflight
            gpus = [{"index": 0, "free": 2 * GB, "total": 4 * GB,
                     "free_scope": discover.FREE_SCOPE_DEVICE},
                    {"index": 1, "free": 6 * GB, "total": 8 * GB,
                     "free_scope": discover.FREE_SCOPE_DEVICE}]
            return (gpus, discover.GPU_PROBE_OK) if return_status else gpus

        monkeypatch.setattr(discover, "list_gpus", fake)
        ratios = resolve_auto_split_ratios(
            {"gpu_split_indices": [0, 1], "gpu_split_ratios": None},
            wait_for_inflight=True)
        assert ratios == pytest.approx([0.25, 0.75])
        assert seen["wfi"] is True

    def test_not_forced_on_a_double_without_the_kwarg(self, monkeypatch):
        """A status-capable double lacking wait_for_inflight (and **kwargs)
        must not be handed a kwarg it never agreed to accept - same tolerance
        contract as _list_gpus_reading's return_status inspection."""
        monkeypatch.setattr(discover, "_native_backend_has_vulkan", lambda: False)

        def fake(*a, return_status=False):
            gpus = [{"index": 0, "free": 2 * GB, "total": 4 * GB,
                     "free_scope": discover.FREE_SCOPE_DEVICE},
                    {"index": 1, "free": 6 * GB, "total": 8 * GB,
                     "free_scope": discover.FREE_SCOPE_DEVICE}]
            return (gpus, discover.GPU_PROBE_OK) if return_status else gpus

        monkeypatch.setattr(discover, "list_gpus", fake)
        ratios = resolve_auto_split_ratios(
            {"gpu_split_indices": [0, 1], "gpu_split_ratios": None},
            wait_for_inflight=True)
        assert ratios == pytest.approx([0.25, 0.75])


# --------------------------------------------------------------------------- #
#  Parent -> worker pinning: the chat chain                                     #
# --------------------------------------------------------------------------- #

class TestChatParamChain:
    def _backend(self, tmp_path):
        from localm.inference.backends.gguf import GgufBackend
        f = tmp_path / "model.gguf"
        f.write_bytes(b"\0" * 4096)
        return GgufBackend(str(f), n_gpu_layers=99, n_gpu_layers_auto=False,
                           n_ctx=512)

    def test_load_native_pins_auto_ratios_into_worker_params(
            self, tmp_path, monkeypatch):
        seen = {}

        def _resolve(*a, **k):
            seen.update(k)
            return [0.6, 0.4]

        monkeypatch.setattr(discover, "resolve_auto_split_ratios", _resolve)
        b = self._backend(tmp_path)
        with patch("localm.discover.list_gpus", return_value=([], "ok")), \
             patch("localm.inference.backends.llamacpp._runner.ModelRunner."
                   "spawn_and_load",
                   return_value={"n_layers": 1, "kv_bytes_per_token": 0,
                                 "supports_images": False}) as spawn:
            b._load_native()
        params = spawn.call_args.args[0]
        assert params["gpu_split_ratios"] == [0.6, 0.4]
        # Off-loop load path: must join a concurrent heartbeat probe rather
        # than decline auto into the equal fallback on an instant BUSY.
        assert seen.get("wait_for_inflight") is True

    def test_load_native_passes_none_when_auto_declines(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(discover, "resolve_auto_split_ratios",
                            lambda *a, **k: None)
        b = self._backend(tmp_path)
        with patch("localm.discover.list_gpus", return_value=([], "ok")), \
             patch("localm.inference.backends.llamacpp._runner.ModelRunner."
                   "spawn_and_load",
                   return_value={"n_layers": 1, "kv_bytes_per_token": 0,
                                 "supports_images": False}) as spawn:
            b._load_native()
        params = spawn.call_args.args[0]
        assert params["gpu_split_ratios"] is None

    def test_gguf_worker_forwards_ratios_to_llamacpp(self):
        from localm.inference.backends.llamacpp._worker import GgufWorker
        import localm.inference.backends.llamacpp as llamacpp_pkg
        captured = {}

        class _StubLlama:
            supports_images = False

            def __init__(self, **kwargs):
                captured.update(kwargs)

        worker = GgufWorker(
            model_path="m.gguf", mmproj_path=None, n_ctx=512, n_gpu_layers=99,
            n_ctx_max=None, n_ctx_grow=4096, gpu_split_ratios=[0.7, 0.3])
        with patch("localm.inference.backends.llamacpp._loader.load_lib"), \
             patch.object(llamacpp_pkg, "LlamaCpp", _StubLlama):
            worker.load()
        assert captured.get("gpu_split_ratios") == [0.7, 0.3]

    def test_llamacpp_ctor_override_reaches_tensor_split(self, monkeypatch):
        """End of the chain: LlamaCpp built with pinned ratios writes them
        into mp.tensor_split (real apply_gpu_split, mocked ctypes api) even
        though the worker-side config carries NO ratios - proving the load
        uses the parent's decision, not a worker-side guess."""
        from localm.inference.backends.llamacpp.llama import LlamaCpp
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 1], "gpu_split_ratios": None})
        monkeypatch.setattr("localm.discover.list_gpus",
                            lambda: [{"index": 0}, {"index": 1}])
        mock_api = MagicMock()
        mock_api.llama_model_default_params.return_value = SimpleNamespace(
            main_gpu=0, n_gpu_layers=0, use_mmap=True, split_mode=1,
            tensor_split=None)
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            llm = LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=99, verbose=True,
                           gpu_split_ratios=[0.8, 0.2])
            llm.close()
        mp = mock_api.llama_model_default_params.return_value
        ptr = ctypes.cast(mp.tensor_split, ctypes.POINTER(ctypes.c_float))
        assert ptr[0] == pytest.approx(0.8)
        assert ptr[1] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
#  Parent -> worker pinning: the embedder chain                                 #
# --------------------------------------------------------------------------- #

class TestEmbedderParamChain:
    def test_reload_pins_auto_ratios_into_worker_params(self, monkeypatch):
        from localm.inference import embedder as emb
        monkeypatch.setattr(discover, "resolve_auto_split_ratios",
                            lambda *a, **k: [0.6, 0.4])
        captured = {}

        class _StubRunner:
            def spawn_and_load(self, params, timeout=None):
                captured.update(params)
                return {"dim": 384, "declared_pooling": None,
                        "effective_pooling": 1, "n_ctx": 512}

            def shutdown(self, grace=5.0):
                pass

        monkeypatch.setattr(
            "localm.inference._embedder_runner.EmbedderRunner", _StubRunner)
        with patch.object(emb.IsolatedEmbedder, "_preflight_vram"):
            emb.IsolatedEmbedder("embed.gguf", n_gpu_layers=99)
        assert captured.get("gpu_split_ratios") == [0.6, 0.4]

    def test_ggufembedder_ctor_override_reaches_tensor_split(self, monkeypatch):
        from localm.inference.embedder import GGUFEmbedder
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 1], "gpu_split_ratios": None})
        monkeypatch.setattr("localm.discover.list_gpus",
                            lambda: [{"index": 0}, {"index": 1}])
        mock_api = MagicMock()
        mock_api.llama_model_default_params.return_value = SimpleNamespace(
            main_gpu=0, n_gpu_layers=0, use_mmap=True, split_mode=1,
            tensor_split=None)
        mock_api.llama_model_n_embd.return_value = 768
        mock_api.has_embeddings_api.return_value = True
        mock_api.has_memory_api.return_value = False
        with patch("localm.inference.backends.llamacpp._api", mock_api):
            GGUFEmbedder("embed.gguf", gpu_split_ratios=[0.8, 0.2])
        mp = mock_api.llama_model_default_params.return_value
        ptr = ctypes.cast(mp.tensor_split, ctypes.POINTER(ctypes.c_float))
        assert ptr[0] == pytest.approx(0.8)
        assert ptr[1] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
#  switch_engine: auto mode defers a combined-short load to the backend         #
# --------------------------------------------------------------------------- #

class TestSwitchEngineAutoDefer:
    """In auto mode the only way shortfall is non-empty is a combined-short
    estimate (see TestShortfallAutoShares), where the backend's own sizing is
    the accurate judge (_auto_gpu_layers/_check_vram budget the split's
    COMBINED capacity) and partial offload can still make the load work, so
    switch_engine defers instead of refusing. Pinned ratios keep the hard
    refusal: sizing never checks one pinned share, so the gate remains the only
    protection against a per-device abort there."""

    def _install(self, monkeypatch, tmp_path, *, gpus, gpu_split_ratios=None,
                 gpu_split_indices=(0, 1), fails_to_fit=False):
        model_file = tmp_path / "model-a.gguf"
        fake_registry = {"model-a": {"path": str(model_file), "source": "local"}}
        monkeypatch.setattr("localm.config.load_registry", lambda: fake_registry)
        monkeypatch.setattr("localm.model_manager.get_model_info",
                            lambda name: (str(model_file), "hint"))
        monkeypatch.setattr("localm.model_manager.get_model_mmproj",
                            lambda name: None)
        monkeypatch.setattr(discover, "_native_backend_has_vulkan", lambda: False)
        from localm.config import load_config as real_load_config
        base_cfg = real_load_config()

        def _cfg():
            return {**base_cfg, "gpu_split_indices": list(gpu_split_indices),
                    "gpu_split_ratios": gpu_split_ratios}

        monkeypatch.setattr("localm.config.load_config", _cfg)
        monkeypatch.setattr("localm.discover.list_gpus", probe_double(gpus))
        monkeypatch.setattr(
            hs, "_engine_factory",
            lambda name: FakeEngine(name, fails_to_fit=fails_to_fit))
        hs._engines.clear()
        hs._engines_lru.clear()
        hs._inference_sems.clear()
        hs._last_activity_per_model.clear()
        hs._active_model_name = None
        hs._default_model_name = None
        hs._engine = None
        hs._inference_sem = None

    # Unregistered-on-disk model file -> the documented 4 GB default size, so
    # vram_required ~= 5.15 GiB and the aggregate threshold ~= 6.15 GiB.
    _TIGHT_GPUS = [
        {"index": 0, "name": "A", "total": 16 * GB, "free": 2 * GB,
         "free_scope": discover.FREE_SCOPE_DEVICE},
        {"index": 1, "name": "B", "total": 16 * GB, "free": 3 * GB,
         "free_scope": discover.FREE_SCOPE_DEVICE},
    ]

    def test_auto_combined_short_defers_to_backend(self, monkeypatch, tmp_path):
        """5 GiB combined free vs the ~6.15 GiB estimate: pre-feature this was
        a hard 503; in auto mode the backend's split-aware sizing decides -
        and (like a real partial offload would) it makes the load work."""
        self._install(monkeypatch, tmp_path, gpus=self._TIGHT_GPUS)
        app = hs.create_app(None)
        client = TestClient(app)
        r = _chat(client, "model-a")
        assert r.status_code == 200, (
            f"auto-mode combined-short must defer to the backend's own "
            f"combined-aware sizing (partial offload), not 503: {r.text}")
        assert hs._engines["model-a"].loaded

    def test_auto_combined_short_backend_refusal_stays_clean(
            self, monkeypatch, tmp_path):
        """Deferral must not turn a genuinely impossible load into a native
        abort: the backend's own _check_vram-style hard refusal still comes
        back as the same clean 503 shape."""
        self._install(monkeypatch, tmp_path, gpus=self._TIGHT_GPUS,
                      fails_to_fit=True)
        app = hs.create_app(None)
        client = TestClient(app)
        r = _chat(client, "model-a")
        assert r.status_code == 503
        assert "cannot fit" in r.text

    def test_pinned_ratios_keep_the_hard_refusal(self, monkeypatch, tmp_path):
        """Explicit ratios: aggregate fits (32 GiB) but GPU 0 cannot hold its
        pinned equal share - the per-device 503 must fire exactly as today,
        naming the short device."""
        gpus = [
            {"index": 0, "name": "A", "total": 16 * GB, "free": 2 * GB},
            {"index": 1, "name": "B", "total": 32 * GB, "free": 30 * GB},
        ]
        self._install(monkeypatch, tmp_path, gpus=gpus,
                      gpu_split_ratios=[1.0, 1.0])
        app = hs.create_app(None)
        client = TestClient(app)
        r = _chat(client, "model-a")
        assert r.status_code == 503
        assert "configured split" in r.text
        assert "GPU 0" in r.text

    def test_declined_auto_keeps_the_hard_refusal(self, monkeypatch, tmp_path):
        """Ratios UNSET but a configured index no longer detected: auto declines
        all-or-nothing and the LOADER applies the equal fallback across the
        survivors, so the per-device hazard is fully live and keying the defer
        on the config shape alone would remove this 503. The gate keys on
        whether ADAPTIVE shares were actually in effect: here they were not, so
        the pre-feature refusal fires, naming the short device."""
        gpus = [
            {"index": 0, "name": "A", "total": 16 * GB, "free": 2 * GB,
             "free_scope": discover.FREE_SCOPE_DEVICE},
            {"index": 1, "name": "B", "total": 32 * GB, "free": 30 * GB,
             "free_scope": discover.FREE_SCOPE_DEVICE},
        ]
        self._install(monkeypatch, tmp_path, gpus=gpus,
                      gpu_split_indices=(0, 1, 2))   # device 2 vanished
        app = hs.create_app(None)
        client = TestClient(app)
        r = _chat(client, "model-a")
        assert r.status_code == 503, (
            f"auto declined (stale index) -> the loader will equal-split the "
            f"survivors -> GPU 0 cannot hold its equal share -> the hard 503 "
            f"must fire exactly as pre-feature: {r.text}")
        assert "configured split" in r.text
        assert "GPU 0" in r.text
