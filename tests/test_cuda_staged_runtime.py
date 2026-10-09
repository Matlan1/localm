# SPDX-License-Identifier: AGPL-3.0-or-later
"""CUDA runtimes fetched without a GPU (``setup-llama --cuda-line``): the
staging command, the record it leaves, and the start check that tests the
runtime on the GPU host."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from localm import setup_llama as sl


def _gpu(name="NVIDIA GeForce RTX 4090", compute="8.9", cuda="12.9", driver="575.51"):
    return sl.NvidiaInfo(present=True, gpu_name=name, driver_version=driver,
                         cuda_capability=cuda, compute_capability=compute)


@pytest.fixture
def staged(tmp_path):
    def make(line="cuda-12"):
        sl.record_staged_cuda(tmp_path, line)
        return tmp_path
    return make


@pytest.fixture
def probes(monkeypatch):
    """Replace the GPU query and the load test, counting the load test's calls."""
    state = {"info": sl.NvidiaInfo(present=False), "loads": (True, ""), "load_calls": 0}
    monkeypatch.setattr(sl, "nvidia_preflight", lambda: state["info"])

    def loads():
        state["load_calls"] += 1
        return state["loads"]

    monkeypatch.setattr(sl, "_native_gpu_loads_ok", loads)
    return state


class TestStagedRecord:
    def test_round_trips_each_line(self, tmp_path):
        for line in ("cuda-12", "cuda-13"):
            sl.record_staged_cuda(tmp_path, line)
            assert sl.staged_cuda_line(tmp_path) == line

    def test_no_note_is_not_a_staged_runtime(self, tmp_path):
        assert sl.staged_cuda_line(tmp_path) is None

    def test_an_unknown_line_is_not_a_staged_runtime(self, tmp_path):
        (tmp_path / sl.STAGED_NOTE).write_text("cuda-99\n", encoding="utf-8")
        assert sl.staged_cuda_line(tmp_path) is None

    def test_every_line_has_an_image_tag(self):
        assert sl.IMAGE_TAG_FOR_LINE == {"cuda-12": "cuda", "cuda-13": "cuda13"}
        assert set(sl.IMAGE_TAG_FOR_LINE) == set(sl._MIN_DRIVER_CUDA)


class TestStartCheck:
    def test_a_runtime_that_was_not_staged_passes_without_probing(self, tmp_path, probes):
        assert sl.check_staged_cuda_runtime(tmp_path) == (True, [])
        assert probes["load_calls"] == 0

    def test_no_gpu_is_refused_and_says_how_to_attach_one(self, staged, probes):
        ok, lines = sl.check_staged_cuda_runtime(staged())
        assert ok is False
        text = " ".join(lines)
        assert "no NVIDIA GPU is visible" in text
        assert "--gpus all" in text
        assert "LOCALM_ALLOW_NO_GPU=1" in text
        assert probes["load_calls"] == 0

    def test_no_gpu_passes_when_allowed_and_says_the_runtime_is_unused(self, staged, probes):
        ok, lines = sl.check_staged_cuda_runtime(staged(), allow_no_gpu=True)
        assert ok is True
        assert "not in use" in " ".join(lines)
        assert probes["load_calls"] == 0

    def test_a_blackwell_gpu_on_the_cuda_12_image_is_pointed_at_the_cuda13_tag(self, staged, probes):
        probes["info"] = _gpu(name="NVIDIA GeForce RTX 5090", compute="12.0", cuda="13.4")
        ok, lines = sl.check_staged_cuda_runtime(staged("cuda-12"))
        assert ok is False
        text = " ".join(lines)
        assert "cuda13" in text and "RTX 5090" in text and "cuda-12" in text
        assert probes["load_calls"] == 0

    def test_an_older_gpu_on_the_cuda13_image_is_pointed_at_the_cuda_tag(self, staged, probes):
        probes["info"] = _gpu()
        ok, lines = sl.check_staged_cuda_runtime(staged("cuda-13"))
        assert ok is False
        assert "Use the cuda image tag" in " ".join(lines)
        assert probes["load_calls"] == 0

    def test_a_driver_older_than_the_staged_line_needs_is_refused(self, staged, probes):
        probes["info"] = _gpu(cuda="12.2", driver="535.1")
        ok, lines = sl.check_staged_cuda_runtime(staged("cuda-12"))
        assert ok is False
        text = " ".join(lines)
        assert "12.2" in text and "12.4" in text and "535.1" in text
        assert probes["load_calls"] == 0

    def test_a_runtime_that_does_not_load_is_refused_with_the_cause(self, staged, probes):
        probes["info"] = _gpu()
        probes["loads"] = (False, "libcublas.so.12: cannot open shared object file")
        ok, lines = sl.check_staged_cuda_runtime(staged("cuda-12"))
        assert ok is False
        assert "libcublas.so.12: cannot open shared object file" in " ".join(lines)
        assert probes["load_calls"] == 1

    def test_a_runtime_that_loads_on_a_matching_gpu_passes_and_names_both(self, staged, probes):
        probes["info"] = _gpu()
        ok, lines = sl.check_staged_cuda_runtime(staged("cuda-12"))
        assert ok is True
        assert "cuda-12" in " ".join(lines) and "RTX 4090" in " ".join(lines)
        assert probes["load_calls"] == 1

    def test_blackwell_passes_on_the_cuda_13_line(self, staged, probes):
        probes["info"] = _gpu(name="NVIDIA GeForce RTX 5090", compute="12.0", cuda="13.4")
        ok, _ = sl.check_staged_cuda_runtime(staged("cuda-13"))
        assert ok is True

    def test_a_failing_gpu_query_is_a_refusal_not_a_crash(self, staged, monkeypatch):
        def broken():
            raise RuntimeError("nvidia-smi exploded")
        monkeypatch.setattr(sl, "nvidia_preflight", broken)
        ok, lines = sl.check_staged_cuda_runtime(staged())
        assert ok is False
        assert "nvidia-smi exploded" in " ".join(lines)


class TestContainerHook:
    @pytest.fixture
    def hook(self, tmp_path, monkeypatch, probes):
        monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
        monkeypatch.delenv("LOCALM_ALLOW_NO_GPU", raising=False)
        monkeypatch.delenv("LOCALM_IMAGE_BACKEND", raising=False)
        return tmp_path

    @pytest.mark.parametrize("tag, line", [("cuda", "cuda-12"), ("cuda13", "cuda-13")])
    def test_a_cuda_image_without_its_record_is_refused_before_any_probe(
            self, hook, probes, monkeypatch, capsys, tag, line):
        monkeypatch.setenv("LOCALM_IMAGE_BACKEND", tag)
        assert sl.cuda_container_check() == 3
        err = capsys.readouterr().err
        assert "refusing to start" in err and "missing or unreadable" in err and line in err
        assert probes["load_calls"] == 0

    def test_a_cuda_image_holding_the_other_lines_record_is_refused(
            self, hook, staged, probes, monkeypatch, capsys):
        staged("cuda-13")
        monkeypatch.setenv("LOCALM_IMAGE_BACKEND", "cuda")
        assert sl.cuda_container_check() == 3
        assert "is for cuda-13, expected cuda-12" in capsys.readouterr().err

    def test_a_cuda_image_with_its_record_goes_on_to_the_gpu_check(
            self, hook, staged, probes, monkeypatch, capsys):
        staged("cuda-12")
        probes["info"] = _gpu()
        monkeypatch.setenv("LOCALM_IMAGE_BACKEND", "cuda")
        assert sl.cuda_container_check() == 0
        assert "loaded for NVIDIA GeForce RTX 4090" in capsys.readouterr().err

    @pytest.mark.parametrize("tag", ["", "cpu", "vulkan"])
    def test_other_images_need_no_record(self, hook, monkeypatch, tag):
        monkeypatch.setenv("LOCALM_IMAGE_BACKEND", tag)
        assert sl.cuda_container_check() == 0

    def test_exits_with_the_documented_status_when_refusing(self, hook, staged, capsys):
        staged()
        assert sl.cuda_container_check() == sl.CUDA_CHECK_FAILED == 3
        err = capsys.readouterr().err
        assert err.startswith("localm: refusing to start: no NVIDIA GPU is visible")

    def test_the_allow_variable_lets_a_gpuless_container_start(self, hook, staged, monkeypatch, capsys):
        staged()
        monkeypatch.setenv("LOCALM_ALLOW_NO_GPU", "1")
        assert sl.cuda_container_check() == 0
        assert "not in use" in capsys.readouterr().err

    @pytest.mark.parametrize("value", ["", "0", "true", "yes"])
    def test_only_the_value_1_allows_it(self, hook, staged, monkeypatch, value):
        staged()
        monkeypatch.setenv("LOCALM_ALLOW_NO_GPU", value)
        assert sl.cuda_container_check() == 3

    def test_a_loaded_runtime_is_logged_on_start(self, hook, staged, probes, capsys):
        staged()
        probes["info"] = _gpu()
        assert sl.cuda_container_check() == 0
        assert "CUDA runtime (cuda-12) loaded for NVIDIA GeForce RTX 4090" in capsys.readouterr().err

    def test_a_directory_with_no_staged_runtime_prints_nothing(self, hook, capsys):
        assert sl.cuda_container_check() == 0
        assert capsys.readouterr().err == ""


class TestStagingCommand:
    @pytest.fixture
    def stage(self, tmp_path, monkeypatch):
        """Run ``setup-llama`` with the network, the wheel install and the
        probes replaced; the clearing, the lock, the marker and the staged note
        are the real ones. Any call to the GPU dialogue or the load test fails
        the test through the counters."""
        target = tmp_path / "lib"
        monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: target)
        monkeypatch.setattr(sl.sys, "platform", "linux")
        monkeypatch.setattr(sl, "_install_runtime_wheel", lambda pkg_dir: True)
        monkeypatch.setattr(sl, "_bundle_missing_native_deps", lambda tgt: None)
        calls = {"provision": [], "forbidden": []}

        def provision(backend, tgt, sha256, with_cudart, cuda_line=sl._CUDA_LINE, tag=None):
            calls["provision"].append((backend, with_cudart, cuda_line, sha256))
            (tgt / sl._lib_name()).write_bytes(b"stub")
            (tgt / "libcudart.so.12").write_bytes(b"stub")
            return "bSTAGED"

        def forbid(name):
            def fn(*a, **k):
                calls["forbidden"].append(name)
                return (True, "")
            return fn

        monkeypatch.setattr(sl, "_provision_backend", provision)
        for name in ("nvidia_preflight", "_cuda_setup_dialogue", "_native_loads_ok", "_verify"):
            monkeypatch.setattr(sl, name, forbid(name))

        def run(*args):
            return CliRunner().invoke(sl.main, ["--backend", "cuda", *args])

        run.target = target
        run.calls = calls
        run.provision = provision
        return run

    @pytest.mark.parametrize("line", ["cuda-12", "cuda-13"])
    def test_stages_the_line_without_the_driver_check_or_the_load_test(self, stage, line):
        result = stage("--cuda-line", line, "--yes")
        assert result.exit_code == 0, result.output
        assert stage.calls["provision"] == [("cuda", True, line, None)]
        assert stage.calls["forbidden"] == []
        assert sl.staged_cuda_line(stage.target) == line
        assert sl._provisioned_backend(stage.target) == "cuda"
        assert sl._provisioned_build(stage.target) == "bSTAGED"

    def test_says_plainly_that_nothing_was_load_tested(self, stage):
        result = stage("--cuda-line", "cuda-12", "--yes")
        assert "without a GPU" in result.output
        assert "NOT" in result.output and "load-tested" in result.output
        assert "Native runtime ready" not in result.output

    def test_a_failed_fetch_exits_non_zero_without_falling_back_or_recording(self, stage, monkeypatch):
        def fails(*a, **k):
            stage.calls["provision"].append(a[0])
            raise sl.ArtifactError("download failed")
        monkeypatch.setattr(sl, "_provision_backend", fails)
        result = stage("--cuda-line", "cuda-12", "--yes")
        assert result.exit_code == 1
        assert "Staging cuda-12 failed" in result.output
        assert stage.calls["provision"] == ["cuda"]
        assert sl.staged_cuda_line(stage.target) is None
        assert sl._provisioned_backend(stage.target) is None

    def test_an_unresolvable_asset_is_reported_not_replaced_by_vulkan(self, stage, monkeypatch):
        import click

        def no_asset(*a, **k):
            stage.calls["provision"].append(a[0])
            raise click.ClickException("no Linux CUDA build found for llama.cpp tag 'b1'")
        monkeypatch.setattr(sl, "_provision_backend", no_asset)
        result = stage("--cuda-line", "cuda-13", "--yes")
        assert result.exit_code == 1
        assert "no Linux CUDA build found" in result.output
        assert stage.calls["provision"] == ["cuda"]

    def test_an_archive_without_the_library_is_an_error(self, stage, monkeypatch):
        monkeypatch.setattr(sl, "_provision_backend", lambda *a, **k: "bX")
        result = stage("--cuda-line", "cuda-12", "--yes")
        assert result.exit_code == 1
        assert "did not contain" in result.output
        assert sl.staged_cuda_line(stage.target) is None

    def test_a_later_normal_provision_removes_the_staged_note(self, stage):
        assert stage("--cuda-line", "cuda-12", "--yes").exit_code == 0
        assert sl.staged_cuda_line(stage.target) == "cuda-12"
        sl._clear_target(stage.target)
        assert sl.staged_cuda_line(stage.target) is None

    @pytest.mark.parametrize("args", [
        ["--backend", "vulkan", "--cuda-line", "cuda-12"],
        ["--backend", "auto", "--cuda-line", "cuda-12"],
    ])
    def test_needs_the_cuda_backend(self, stage, args):
        result = CliRunner().invoke(sl.main, args)
        assert result.exit_code == 2
        assert "needs --backend cuda" in result.output
        assert stage.calls["provision"] == []

    def test_is_linux_only(self, stage, monkeypatch):
        monkeypatch.setattr(sl.sys, "platform", "win32")
        result = stage("--cuda-line", "cuda-12")
        assert result.exit_code == 2
        assert "only available on Linux" in result.output
        assert stage.calls["provision"] == []

    @pytest.mark.parametrize("extra", [["--url", "https://example.invalid/a.tar.gz"], ["--rollback"]])
    def test_cannot_be_combined_with_another_source(self, stage, extra):
        result = stage("--cuda-line", "cuda-12", *extra)
        assert result.exit_code == 2
        assert "cannot be combined" in result.output
        assert stage.calls["provision"] == []

    def test_rejects_an_unknown_line(self, stage):
        result = stage("--cuda-line", "cuda-11")
        assert result.exit_code == 2
        assert stage.calls["provision"] == []


DRIVER = textwrap.dedent("""\
    import sys
    from localm.inference.backends.llamacpp import _loader
    devices = {devices!r}
    _loader.load_lib = lambda: None
    _loader.compute_devices = lambda: devices
    exec(compile(sys.argv[1], "<probe>", "exec"))
""")


class TestGpuLoadProbe:
    """The probe code the start check runs, executed in a child interpreter with
    the loader reporting a chosen device list."""

    @staticmethod
    def _run(devices):
        proc = subprocess.run(
            [sys.executable, "-c", DRIVER.format(devices=devices), sl._GPU_LOAD_PROBE_CODE],
            capture_output=True, text=True, timeout=120,
            env={**os.environ,
                 "PYTHONPATH": str(Path(sl.__file__).resolve().parents[2])})
        return proc

    def test_a_gpu_device_passes(self):
        from localm.inference.backends.llamacpp import _loader
        proc = self._run([("CPU", _loader.GGML_DEV_TYPE_CPU), ("CUDA0", _loader.GGML_DEV_TYPE_GPU)])
        assert proc.returncode == 0, proc.stderr

    def test_only_the_cpu_device_fails_and_is_named(self):
        from localm.inference.backends.llamacpp import _loader
        proc = self._run([("CPU", _loader.GGML_DEV_TYPE_CPU)])
        assert proc.returncode == sl._PROBE_NO_GPU_DEVICE
        assert proc.stdout.strip() == "CPU"

    def test_no_device_at_all_fails(self):
        proc = self._run([])
        assert proc.returncode == sl._PROBE_NO_GPU_DEVICE
        assert proc.stdout.strip() == "none"


class TestGpuLoadVerdict:
    @staticmethod
    def _fake_run(monkeypatch, returncode, stdout="", stderr=""):
        monkeypatch.setattr(sl.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
            a, returncode, stdout=stdout, stderr=stderr))

    def test_passes_on_exit_0(self, monkeypatch):
        self._fake_run(monkeypatch, 0)
        assert sl._native_gpu_loads_ok() == (True, "")

    def test_a_cpu_only_runtime_is_a_failure_that_names_what_registered(self, monkeypatch):
        self._fake_run(monkeypatch, sl._PROBE_NO_GPU_DEVICE, stdout="CPU\n")
        ok, detail = sl._native_gpu_loads_ok()
        assert ok is False
        assert "registered no GPU device (registered: CPU)" in detail

    def test_an_abi_rejection_keeps_the_abi_prefix(self, monkeypatch):
        self._fake_run(monkeypatch, sl._PROBE_ABI_MISMATCH, stderr="layout drift")
        ok, detail = sl._native_gpu_loads_ok()
        assert ok is False and sl._is_abi_rejection(detail)

    def test_a_loader_crash_reports_its_error_line(self, monkeypatch):
        self._fake_run(monkeypatch, 1, stderr="Traceback\nRuntimeError: libcuda.so.1: cannot open")
        ok, detail = sl._native_gpu_loads_ok()
        assert ok is False and "libcuda.so.1: cannot open" in detail

    def test_a_probe_that_cannot_start_is_a_failure(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("no interpreter")
        monkeypatch.setattr(sl.subprocess, "run", boom)
        assert sl._native_gpu_loads_ok() == (False, "no interpreter")
