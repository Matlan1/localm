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
