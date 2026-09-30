# SPDX-License-Identifier: AGPL-3.0-or-later
"""A bug report from a multi-GPU box must say which GPUs there are, and must
never leave an environment field silently blank.

A 5-GPU report once named only the first NVIDIA card and left the driver and
CUDA-capability fields empty, with no way to tell a parse miss from a missing
driver. The report now lists every GPU from the last reading the process took
(the same index space as the load-time "implicit GPU split" log line), writes
"not detected" for a probed field that came back empty, carries the settings
that shape a multi-GPU load, and the log digest keeps llama.cpp's own
allocation-failure lines.
"""

from unittest import mock

from localm import _log_digest as ld
from localm import bugreport, discover

GiB = 1024 ** 3


def _polling_line(ts: str) -> str:
    return f"{ts},000 DEBUG   localm: GET /api/stats -> 200 (7 ms, loop_lag=0.2s)"


class TestGpuInventory:
    def test_every_gpu_is_listed_with_its_memory(self):
        gpus = [{"index": 0, "name": "GPU A", "free": int(13.7 * GiB),
                 "total": int(15.6 * GiB)},
                {"index": 4, "name": "GPU E", "free": int(2.7 * GiB),
                 "total": int(5.8 * GiB), "source": "nvidia-smi"}]
        with mock.patch.object(discover, "last_gpu_reading", return_value=gpus):
            lines = bugreport._gpu_inventory()
        assert lines == ["0: GPU A (13.7 of 15.6 GB free)",
                         "4: GPU E (2.7 of 5.8 GB free via nvidia-smi)"]

    def test_no_reading_yet_is_stated_not_blank(self):
        with mock.patch.object(discover, "last_gpu_reading", return_value=None):
            assert bugreport._gpu_inventory() == ["not measured in this process"]

    def test_an_empty_reading_is_not_detected(self):
        with mock.patch.object(discover, "last_gpu_reading", return_value=[]):
            assert bugreport._gpu_inventory() == ["not detected"]

    def test_missing_free_is_named(self):
        gpus = [{"index": 0, "name": "", "total": 8 * GiB}]
        with mock.patch.object(discover, "last_gpu_reading", return_value=gpus):
            assert bugreport._gpu_inventory() == [
                "0: not detected (8.0 GB total, free not detected)"]

    def test_last_gpu_reading_never_probes(self):
        discover._reset_gpu_probe_cache()
        with mock.patch.object(discover, "_list_gpus_probe",
                               side_effect=AssertionError("must not probe")):
            assert discover.last_gpu_reading() is None

    def test_the_report_renders_a_gpus_line(self):
        gpus = [{"index": 0, "name": "GPU A", "free": GiB, "total": 2 * GiB}]
        with mock.patch.object(discover, "last_gpu_reading", return_value=gpus):
            text = bugreport.build_report("x")
        assert "- GPUs: 0: GPU A (1.0 of 2.0 GB free)" in text


class TestNvidiaFieldsNeverBlank:
    def test_unparsed_driver_fields_read_not_detected(self, monkeypatch):
        from localm import hwdetect, setup_llama
        det = hwdetect.Detection(vendors=["nvidia"], recommended="cuda",
                                 source="test", gpu_names="nvidia")
        monkeypatch.setattr(hwdetect, "detect", lambda: det)
        monkeypatch.setattr(
            setup_llama, "nvidia_preflight",
            lambda: setup_llama.NvidiaInfo(present=True,
                                           gpu_name="NVIDIA GeForce GTX 1660",
                                           compute_capability="7.5"))
        diag = bugreport.collect_diagnostics({})
        assert diag["nvidia_gpu"] == "NVIDIA GeForce GTX 1660"
        assert diag["nvidia_driver"] == "not detected"
        assert diag["nvidia_cuda_capability"] == "not detected"
        assert diag["nvidia_compute_capability"] == "7.5"


class TestSplitSettingsInTheConfigSubset:
    def test_multi_gpu_load_settings_are_reported(self, monkeypatch):
        cfg = {"mtp_enabled": True, "n_cpu_moe": 0, "main_gpu_index": 1,
               "gpu_split_indices": [0, 1], "gpu_split_ratios": [0.5, 0.5],
               "n_ctx_grow": 4096, "api_key": "SECRET-CANARY"}
        monkeypatch.setattr("localm.config.load_config", lambda: cfg)
        out = bugreport._safe_config_subset()
        for key in ("mtp_enabled", "n_cpu_moe", "main_gpu_index",
                    "gpu_split_indices", "gpu_split_ratios", "n_ctx_grow"):
            assert out[key] == cfg[key]
        assert "SECRET-CANARY" not in repr(out)


class TestNativeAllocationFailureSurvivesTheDigest:
    """llama.cpp's own allocation-failure lines arrive as unleveled native
    stderr glued to a routine record. They name the device and the size that
    failed, so they must survive a busy log like any crash line."""

    def test_cudamalloc_out_of_memory_line_survives_dense_polling(self):
        line = ("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1940.00 "
                "MiB on device 4: cudaMalloc failed: out of memory")
        lines = [_polling_line(f"2026-09-25 14:08:{i:02d}") for i in range(40)]
        lines.insert(20, line)
        assert line in ld.build_digest("\n".join(lines))

    def test_failed_compute_buffer_and_context_lines_are_errors(self):
        for text in ("graph_reserve: failed to allocate compute buffers",
                     "llama_init_from_model: failed to initialize the context: "
                     "failed to allocate compute pp buffers"):
            rec = {"level": "DEBUG", "logger": "localm", "lines": [
                "2026-09-25 14:08:12,000 DEBUG   localm: GET /api/stats -> 200",
                text]}
            assert ld.is_error_record(rec), text

    def test_a_buffer_size_report_is_still_benign(self):
        rec = {"level": "DEBUG", "logger": "localm", "lines": [
            "2026-09-25 14:08:12,000 DEBUG   localm: GET /api/stats -> 200",
            "sched_reserve:      CUDA4 compute buffer size =  1940.00 MiB"]}
        assert not ld.is_error_record(rec)
