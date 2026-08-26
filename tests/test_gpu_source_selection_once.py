# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GPU source selection is resolved once per process, not once per probe.

``list_gpus`` deliberately re-probes on every call so the ``free`` reading is
never stale, and the GUI polls it continuously. The DECISION about which VRAM
source to read is not a reading: its inputs (a resident bundled HIP runtime, an
installed ``rocm_sdk``) cannot change while the process runs. These tests pin
that the inputs are evaluated once and the decision is announced once.

Each announcement test has a work-counting sibling, because a fix that only
latched the log lines would satisfy the announcement assertions while still
running the directory glob and the ``sys.path`` walk on every probe.
"""

import importlib.util
import sys
from types import SimpleNamespace

import pytest

from localm import discover, gpu_usage


SKIP_MESSAGE = "skipping the torch GPU probe"


def _arm_doomed(monkeypatch, tmp_path, *, native_loaded=True, hip_dll=True,
                rocm_sdk_installed=True, torch_resident=False):
    """Arrange the known-doomed combination and return counters recording how
    many times each expensive input was actually evaluated.

    The runtime glob runs for real over a real directory: only the DLL name is
    faked, so the detection logic itself is exercised rather than replaced.
    """
    counts = {"runtime_dir": 0, "rocm_sdk": 0}

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        "localm.inference.backends.llamacpp._loader.native_lib_loaded",
        lambda: native_loaded)

    rt = tmp_path / "native-runtime"
    rt.mkdir()
    (rt / ("ggml-hip.dll" if hip_dll else "ggml-vulkan.dll")).write_bytes(b"")

    def _runtime_binary_dir():
        counts["runtime_dir"] += 1
        return rt

    monkeypatch.setattr(
        "localm.inference.backends.llamacpp._loader.runtime_binary_dir",
        _runtime_binary_dir)

    real_find_spec = importlib.util.find_spec

    def _fake_find_spec(name, *args, **kwargs):
        if name == "rocm_sdk":
            counts["rocm_sdk"] += 1
            return object() if rocm_sdk_installed else None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)

    if torch_resident:
        monkeypatch.setitem(
            sys.modules, "torch",
            SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    else:
        monkeypatch.delitem(sys.modules, "torch", raising=False)

    return counts


class TestSourceSelectionResolvedOnce:
    """discover: the two inputs that cost anything are evaluated once."""

    def test_doomed_answer_is_still_true_every_call(self, monkeypatch, tmp_path):
        """The latch must not change the answer, only how often it is derived."""
        _arm_doomed(monkeypatch, tmp_path)
        assert [discover._torch_gpu_probe_known_doomed() for _ in range(5)] == \
            [True] * 5

    def test_runtime_glob_runs_once_across_many_probes(self, monkeypatch, tmp_path):
        counts = _arm_doomed(monkeypatch, tmp_path)
        for _ in range(10):
            discover._torch_gpu_probe_known_doomed()
        assert counts["runtime_dir"] == 1

    def test_rocm_sdk_lookup_runs_once_across_many_probes(self, monkeypatch,
                                                          tmp_path):
        counts = _arm_doomed(monkeypatch, tmp_path)
        for _ in range(10):
            discover._torch_gpu_probe_known_doomed()
        assert counts["rocm_sdk"] == 1

    def test_resident_hip_answer_is_latched_true(self, monkeypatch, tmp_path):
        """Once the native lib has been seen loaded, the answer stays True: a
        loaded library is never unloaded, so re-globbing could only ever agree."""
        _arm_doomed(monkeypatch, tmp_path)
        assert discover.native_hip_runtime_resident() is True
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.native_lib_loaded",
            lambda: False)
        assert discover.native_hip_runtime_resident() is True

    def test_not_resident_is_not_latched(self, monkeypatch, tmp_path):
        """False is re-derived: the lib can still load later in the process."""
        _arm_doomed(monkeypatch, tmp_path, native_loaded=False)
        assert discover.native_hip_runtime_resident() is False
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.native_lib_loaded",
            lambda: True)
        assert discover.native_hip_runtime_resident() is True


class TestSkipAnnouncedOnce:
    """discover: the skip is logged when decided, and again only if it changes."""

    def test_announced_once_across_many_probes(self, monkeypatch, tmp_path,
                                               caplog):
        _arm_doomed(monkeypatch, tmp_path)
        with caplog.at_level("DEBUG", logger="localm"):
            for _ in range(10):
                discover._torch_gpu_probe_known_doomed()
        said = [r for r in caplog.records if SKIP_MESSAGE in r.getMessage()]
        assert len(said) == 1

    def test_announced_again_when_the_decision_changes(self, monkeypatch,
                                                       tmp_path, caplog):
        """A decision that stops and later resumes must re-announce, so a fresh
        cause is never hidden behind an earlier one."""
        _arm_doomed(monkeypatch, tmp_path)
        with caplog.at_level("DEBUG", logger="localm"):
            assert discover._torch_gpu_probe_known_doomed() is True
            monkeypatch.setitem(
                sys.modules, "torch",
                SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
            assert discover._torch_gpu_probe_known_doomed() is False
            monkeypatch.delitem(sys.modules, "torch", raising=False)
            assert discover._torch_gpu_probe_known_doomed() is True
        said = [r for r in caplog.records if SKIP_MESSAGE in r.getMessage()]
        assert len(said) == 2

    def test_reset_re_arms_the_announcement(self, monkeypatch, tmp_path, caplog):
        """The reset the test suite runs around every test must hand back a cold
        source selection, or one test's decision would decide the next one's."""
        _arm_doomed(monkeypatch, tmp_path)
        with caplog.at_level("DEBUG", logger="localm"):
            discover._torch_gpu_probe_known_doomed()
            discover._reset_gpu_probe_cache()
            discover._torch_gpu_probe_known_doomed()
        said = [r for r in caplog.records if SKIP_MESSAGE in r.getMessage()]
        assert len(said) == 2

    def test_reset_re_arms_the_expensive_inputs(self, monkeypatch, tmp_path):
        counts = _arm_doomed(monkeypatch, tmp_path)
        discover._torch_gpu_probe_known_doomed()
        discover._reset_gpu_probe_cache()
        discover._torch_gpu_probe_known_doomed()
        assert counts["runtime_dir"] == 2
        assert counts["rocm_sdk"] == 2


class TestNoticeOnce:
    """gpu_usage: one notice per distinct source selection, per process."""

    def test_repeat_is_suppressed_and_a_different_key_is_not(self):
        assert gpu_usage._notice_once("site", "a") is True
        assert gpu_usage._notice_once("site", "a") is False
        assert gpu_usage._notice_once("site", "b") is True

    def test_sites_do_not_share_a_key_space(self):
        assert gpu_usage._notice_once("one", "k") is True
        assert gpu_usage._notice_once("two", "k") is True

    def test_cap_stops_the_site_and_says_so(self, caplog):
        with caplog.at_level("DEBUG", logger="localm"):
            for i in range(gpu_usage._NOTICE_KEY_CAP + 3):
                gpu_usage._notice_once("site", i)
        announced = [r for r in caplog.records
                     if "distinct source selections" in r.getMessage()]
        assert len(announced) == 1

    def test_reset_re_arms_every_site(self):
        assert gpu_usage._notice_once("site", "a") is True
        gpu_usage._reset_source_selection_notices()
        assert gpu_usage._notice_once("site", "a") is True


class TestGpuUsageNoticesOncePerProcess:
    """gpu_usage: the two torch-consultability notices fire once, not per poll."""

    def test_process_scoped_notice_fires_once_per_reason(self, monkeypatch,
                                                         tmp_path, caplog):
        _arm_doomed(monkeypatch, tmp_path)
        with caplog.at_level("DEBUG", logger="localm"):
            for _ in range(10):
                assert gpu_usage.raw_reading_is_process_scoped() is True
        said = [r for r in caplog.records
                if "are process-scoped" in r.getMessage()]
        assert len(said) == 1

    def test_missing_pci_bus_notice_fires_once_per_device(self, monkeypatch,
                                                          tmp_path, caplog):
        _arm_doomed(monkeypatch, tmp_path)
        with caplog.at_level("DEBUG", logger="localm"):
            for _ in range(10):
                assert gpu_usage._torch_pci_bus(0) is None
            assert gpu_usage._torch_pci_bus(1) is None
        said = [r for r in caplog.records
                if "no pci_bus_id for device" in r.getMessage()]
        # Once for device 0, once for device 1: a different device is a
        # different selection and stays visible.
        assert len(said) == 2

    @pytest.mark.parametrize("reason", ["first cause", "second cause"])
    def test_a_different_reason_is_still_announced(self, monkeypatch, tmp_path,
                                                   caplog, reason):
        _arm_doomed(monkeypatch, tmp_path)
        with caplog.at_level("DEBUG", logger="localm"):
            gpu_usage._known_blind_without_torch("first cause")
            gpu_usage._known_blind_without_torch(reason)
        said = [r for r in caplog.records
                if "are process-scoped" in r.getMessage()]
        assert len(said) == (1 if reason == "first cause" else 2)
