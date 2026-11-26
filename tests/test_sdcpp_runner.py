# SPDX-License-Identifier: AGPL-3.0-or-later
"""The stable-diffusion.cpp worker as a process: a crash, a hang, a cancel that
the native side never honours, and a runtime that will not load all come back
as Python results with the worker stopped. These spawn real worker processes;
the native faults come from the worker's fault-injection hook. The end-to-end
generation test needs an installed runtime and a model and is marked
integration."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from localm.media.sdcpp import runner as runner_mod
from localm.media.sdcpp.runner import SdCancelled, SdRunner, SdWorkerError

_FAULT = "LOCALM_SDCPP_FAULT_FOR_TEST"


@pytest.fixture
def sd_runner():
    r = SdRunner()
    yield r
    r.shutdown(grace=0)


def test_a_worker_that_dies_is_reported_and_reaped(sd_runner, monkeypatch, tmp_path):
    monkeypatch.setenv(_FAULT, "exit")
    with pytest.raises(SdWorkerError, match="crashed") as ei:
        sd_runner.probe(tmp_path)
    assert "The server stayed up" in str(ei.value)
    assert not sd_runner.is_alive()


def test_a_hung_worker_times_out_and_is_killed(sd_runner, monkeypatch, tmp_path):
    monkeypatch.setenv(_FAULT, "hang")
    t0 = time.monotonic()
    with pytest.raises(SdWorkerError, match="timed out"):
        sd_runner.probe(tmp_path, timeout=3)
    assert time.monotonic() - t0 < 30
    assert not sd_runner.is_alive()


def test_a_cancel_the_native_side_ignores_kills_the_worker(sd_runner, monkeypatch):
    monkeypatch.setenv(_FAULT, "hang")
    monkeypatch.setattr(runner_mod, "CANCEL_GRACE", 1.0)
    sd_runner._spawn()
    pid = sd_runner.pid
    t0 = time.monotonic()
    with pytest.raises(SdCancelled, match="worker was stopped"):
        sd_runner.generate_image({"prompt": "x", "width": 64, "height": 64},
                                 timeout=60, cancel_check=lambda: True)
    assert time.monotonic() - t0 < 30
    assert not sd_runner.is_alive()
    assert pid is not None


def test_a_runtime_directory_without_the_library_is_a_clear_error(sd_runner, tmp_path):
    with pytest.raises(SdWorkerError, match="could not load the stable-diffusion.cpp runtime"):
        sd_runner.probe(tmp_path)
    assert not sd_runner.is_alive()


def test_requests_without_a_worker_are_refused(sd_runner):
    with pytest.raises(SdWorkerError, match="not running"):
        sd_runner.generate_image({"prompt": "x", "width": 64, "height": 64})


def test_generate_before_load_is_an_error_and_keeps_the_worker(sd_runner):
    sd_runner._spawn()
    with pytest.raises(SdWorkerError, match="no model is loaded"):
        sd_runner.generate_image({"prompt": "x", "width": 64, "height": 64}, timeout=60)
    assert sd_runner.is_alive()
    sd_runner.shutdown()
    assert not sd_runner.is_alive()


def _real_setup():
    rt = os.environ.get("LOCALM_SDCPP_TEST_RUNTIME")
    model = os.environ.get("LOCALM_SDCPP_TEST_MODEL")
    if not rt or not model:
        pytest.skip("LOCALM_SDCPP_TEST_RUNTIME and LOCALM_SDCPP_TEST_MODEL are not set")
    from localm.media.sdcpp import runtime
    extra = runtime.rocm_library_dirs() if "rocm" in Path(rt).name else []
    return Path(rt), model, extra


@pytest.mark.integration
def test_real_generation_sizes_progress_and_cancel(sd_runner):
    rt, model, extra = _real_setup()
    info = sd_runner.ensure_loaded(("m", model), rt, {"model_path": model}, extra_dirs=extra)
    assert info["image"] is True
    for w, h in ((512, 512), (768, 512)):
        events = []
        res = sd_runner.generate_image(
            {"prompt": "a lighthouse", "width": w, "height": h, "steps": 2,
             "cfg_scale": 1.0, "seed": 5}, on_event=events.append)
        assert (res["width"], res["height"], res["channel"]) == (w, h, 3)
        assert len(res["data"]) == w * h * 3
        assert any(e[0] == "progress" and e[1]["phase"] == "sampling" for e in events)
    seen = []
    with pytest.raises(SdCancelled):
        sd_runner.generate_image(
            {"prompt": "a forest", "width": 512, "height": 512, "steps": 60,
             "cfg_scale": 1.0, "seed": 1},
            on_event=lambda e: seen.append(e), cancel_check=lambda: bool(seen))
    assert sd_runner.is_alive()
    res = sd_runner.generate_image({"prompt": "a cat", "width": 512, "height": 512,
                                    "steps": 1, "cfg_scale": 1.0, "seed": 2})
    assert len(res["data"]) == 512 * 512 * 3


def test_a_loaded_runtime_with_no_compute_device_refuses_the_model_load(monkeypatch):
    """A runtime directory that still holds the main library but has lost its
    compute-backend modules loads and lists NO device. install() refuses that at
    install time; the model load must refuse it too instead of building a context
    on nothing."""
    from localm.media.sdcpp import _binding as b, _child

    monkeypatch.setattr(_child._State, "lib", object(), raising=False)
    monkeypatch.setattr(b, "list_devices", lambda lib: [])
    with pytest.raises(RuntimeError, match="no compute device"):
        _child._do_load({"runtime_dir": "X", "ctx": {"model_path": "m.gguf"}})


def test_the_no_device_refusal_does_not_fire_when_a_device_is_present(monkeypatch):
    """Control for the test above: with a device listed, the load gets past that
    check and fails later for an unrelated reason."""
    from localm.media.sdcpp import _binding as b, _child

    monkeypatch.setattr(_child._State, "lib", object(), raising=False)
    monkeypatch.setattr(b, "list_devices", lambda lib: [("CPU", "a cpu")])
    with pytest.raises(Exception) as excinfo:
        _child._do_load({"runtime_dir": "X", "ctx": {"model_path": "m.gguf"}})
    assert "no compute device" not in str(excinfo.value)


@pytest.mark.integration
def test_real_runtime_without_its_backend_modules_refuses_the_load(sd_runner, tmp_path):
    import shutil

    rt, model, extra = _real_setup()
    copy = tmp_path / "damaged"
    shutil.copytree(rt, copy)
    removed = 0
    for pattern in ("ggml-cpu-*.dll", "libggml-cpu*.so", "ggml-cpu*.dylib",
                    "ggml-vulkan*", "ggml-hip*", "ggml-cuda*", "ggml-metal*"):
        for f in copy.glob(pattern):
            f.unlink()
            removed += 1
    if not removed:
        pytest.skip("this runtime ships no separate compute-backend modules")
    with pytest.raises(SdWorkerError, match="no compute device"):
        sd_runner.ensure_loaded(("m", model), copy, {"model_path": model},
                                extra_dirs=extra)


# --------------------------------------------------------------------------- #
#  What the native library prints while it loads stays off the console         #
# --------------------------------------------------------------------------- #

class _QuietLib:
    def sd_set_log_callback(self, cb, data):
        pass

    def sd_set_progress_callback(self, cb, data):
        pass


def _probe_in_process(monkeypatch, load_lib):
    """Run one "probe" through _child.worker_main in this process (a thread),
    with *load_lib* standing in for the library load, and return its reply."""
    import queue
    import threading

    import localm._mp_spawn as spawn
    import localm.debuglog as debuglog
    from localm.media.sdcpp import _child

    monkeypatch.setattr(_child._State, "lib", None)
    monkeypatch.setattr(_child, "_arm_crash_trace", lambda path: None)
    monkeypatch.setattr(_child, "_use_c_ctype_locale", lambda: None)
    for name in ("install_parent_death_watchdog", "ignore_interrupt_signals",
                 "suppress_native_error_dialogs"):
        monkeypatch.setattr(spawn, name, lambda: None)
    monkeypatch.setattr(debuglog, "attach_child_logging", lambda: None)
    monkeypatch.setattr(_child, "_load_lib", load_lib)

    def probe(payload):
        os.write(2, b"ggml_vulkan: 0 = Test GPU (driver) | uma: 0\n")
        return {"commit": "c", "devices": [("Vulkan0", "Test GPU")], "system_info": ""}

    monkeypatch.setattr(_child, "_do_probe", probe)
    req, resp = queue.Queue(), queue.Queue()
    worker = threading.Thread(target=_child.worker_main, args=(req, resp, threading.Event()),
                              daemon=True)
    worker.start()
    req.put(("probe", {"runtime_dir": "rt", "extra_dirs": []}))
    reply = resp.get(timeout=30)
    req.put(None)
    worker.join(timeout=30)
    return reply


def test_the_probe_keeps_native_load_lines_off_the_console(monkeypatch, capfd):
    def load_lib(payload):
        os.write(2, b"ggml_vulkan: Found 1 Vulkan devices:\n")
        os.write(1, b"load_backend: loaded Vulkan backend from staging\\ggml-vulkan.dll\n")
        return _QuietLib()

    reply = _probe_in_process(monkeypatch, load_lib)
    out, err = capfd.readouterr()
    assert reply[0] == "ok", reply
    assert reply[1]["devices"] == [("Vulkan0", "Test GPU")]
    assert "ggml_vulkan" not in out + err
    assert "load_backend" not in out + err


def test_a_load_failure_carries_the_captured_native_lines(monkeypatch, capfd):
    def load_lib(payload):
        os.write(2, b"ggml_vulkan: vkCreateInstance failed\n")
        raise RuntimeError("could not load the stable-diffusion.cpp runtime from rt")

    reply = _probe_in_process(monkeypatch, load_lib)
    out, err = capfd.readouterr()
    assert reply[0] == "error"
    assert "could not load the stable-diffusion.cpp runtime from rt" in reply[1]
    assert "ggml_vulkan: vkCreateInstance failed" in reply[1]
    assert "ggml_vulkan" not in out + err


def test_native_output_survives_a_process_that_dies_in_the_capture(tmp_path):
    import subprocess
    import sys
    keep = tmp_path / "trace.native.log"
    code = ("import os\n"
            "from localm.debuglog import capture_native_stdio\n"
            f"with capture_native_stdio(keep_path={str(keep)!r}):\n"
            "    os.write(2, b'GGML_ASSERT(buf != NULL) failed\\n')\n"
            "    os._exit(3)\n")
    env = dict(os.environ)
    env.pop("LOCALM_DEBUG", None)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=120, env=env,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert r.returncode == 3, r.stderr
    assert "GGML_ASSERT" not in r.stderr
    assert "GGML_ASSERT(buf != NULL) failed" in keep.read_text(encoding="utf-8")


def test_in_debug_mode_native_output_goes_straight_to_the_debug_log(tmp_path, monkeypatch,
                                                                     capfd):
    from localm import debuglog
    log = tmp_path / "debug.log"
    log.write_text("earlier line\n", encoding="utf-8")
    monkeypatch.setattr(debuglog, "log_file_path", lambda: log)
    with debuglog.capture_native_stdio(keep_path=str(tmp_path / "unused.log")) as cap:
        os.write(2, b"load_backend: loaded CPU backend\n")
        assert cap.tail() == "load_backend: loaded CPU backend"
    assert "load_backend" not in "".join(capfd.readouterr())
    assert log.read_text(encoding="utf-8").endswith("load_backend: loaded CPU backend\n")
    assert not (tmp_path / "unused.log").exists()


def test_the_parent_reports_the_native_output_of_a_worker_that_died(tmp_path):
    from localm.media.sdcpp import _child
    r = SdRunner()
    r._crash_trace_path = tmp_path / "sdcpp-worker.trace"
    native = Path(_child.native_output_path(r._crash_trace_path))
    native.write_text("ggml_vulkan: Found 1 Vulkan devices:\nGGML_ASSERT(x) failed\n",
                      encoding="utf-8")
    detail = r._crash_detail()
    assert "Native output: GGML_ASSERT(x) failed." in detail
    assert not native.exists()
