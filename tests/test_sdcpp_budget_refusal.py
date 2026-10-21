# SPDX-License-Identifier: AGPL-3.0-or-later
"""A stable-diffusion.cpp generation the runtime refuses to run within the
``max_vram`` budget of its load: the worker recognises the refusal from the
runtime's own log lines, the runner raises ``SdBudgetRefused``, and the image
and video backends load the model once more without the budget and generate."""

from __future__ import annotations

import queue
import threading

import pytest

from localm import discover
from localm.media.sdcpp import _binding as b
from localm.media.sdcpp import _child, shared
from localm.media.sdcpp import budget as sd_budget
from localm.media.sdcpp import runtime as sd_runtime
from localm.media.sdcpp.logrelay import RecoveredWarnings
from localm.media.sdcpp.runner import SdBudgetRefused, SdRunner, SdWorkerError
from localm.plugins.builtin.image.backends import native as image_native
from localm.plugins.builtin.video.backends import native as video_native

GIB = 1024 ** 3

CAPACITY_WARN = ("model manager cannot make enough memory available on Vulkan0: need 882.28 MB "
                 "device / 370.28 MB budget, available 14260.36 MB device / 256.00 MB budget "
                 "--- model_manager.cpp:2018\n")
SEGMENT_REFUSED = ("unet segment 1/22 (unet.input_blocks.1) failed during weight preparation "
                   "--- ggml_runner.cpp:899\n")
SAMPLING_FAILED = ("diffusion model compute failed --- diffusion_engine.cpp:2751\n",
                   "Diffusion model sampling failed --- diffusion_engine.cpp:2972\n",
                   "sampling for image 1/1 failed after 0.01s --- image.cpp:907\n")

REFUSED_LINES = ([(b.SD_LOG_WARN, CAPACITY_WARN), (b.SD_LOG_ERROR, SEGMENT_REFUSED)]
                 + [(b.SD_LOG_ERROR, t) for t in SAMPLING_FAILED])


@pytest.fixture
def child_log(monkeypatch):
    monkeypatch.setattr(_child._State, "redact", ())
    monkeypatch.setattr(_child._State, "paths", ())
    monkeypatch.setattr(_child._State, "model_desc", "")
    _child._reset_generation_state()
    q = queue.Queue()
    log_cb, _progress, _preview = _child._make_callbacks(q)
    yield lambda level, text: log_cb(level, text.encode("utf-8"), None)
    _child._reset_generation_state()


def _failure_after(child_log, lines):
    for level, text in lines:
        child_log(level, text)
    return _child._generation_failed("stable-diffusion.cpp generation failed")


def test_a_segment_refused_after_a_capacity_warning_is_a_budget_refusal(child_log):
    err = _failure_after(child_log, REFUSED_LINES)
    assert isinstance(err, _child._BudgetRefused)
    assert "sampling for image 1/1 failed" in str(err)


def test_a_segment_failure_without_the_capacity_warning_is_an_ordinary_failure(child_log):
    err = _failure_after(child_log, [(b.SD_LOG_ERROR, SEGMENT_REFUSED)]
                         + [(b.SD_LOG_ERROR, t) for t in SAMPLING_FAILED])
    assert type(err) is RuntimeError


def test_a_capacity_warning_before_another_failure_is_an_ordinary_failure(child_log):
    workspace = SEGMENT_REFUSED.replace("weight preparation", "workspace preparation")
    err = _failure_after(child_log, [(b.SD_LOG_WARN, CAPACITY_WARN),
                                     (b.SD_LOG_ERROR, workspace)]
                         + [(b.SD_LOG_ERROR, t) for t in SAMPLING_FAILED])
    assert type(err) is RuntimeError


def test_the_next_generation_starts_without_the_previous_refusal(child_log):
    child_log(b.SD_LOG_WARN, CAPACITY_WARN)
    child_log(b.SD_LOG_ERROR, SEGMENT_REFUSED)
    _child._reset_generation_state()
    err = _failure_after(child_log, [(b.SD_LOG_ERROR, t) for t in SAMPLING_FAILED])
    assert type(err) is RuntimeError


def test_the_worker_replies_budget_refused_for_a_refused_generation(monkeypatch):
    from localm import _mp_spawn, debuglog
    for name in ("install_parent_death_watchdog", "ignore_interrupt_signals",
                 "suppress_native_error_dialogs"):
        monkeypatch.setattr(_mp_spawn, name, lambda: None)
    monkeypatch.setattr(debuglog, "attach_child_logging", lambda: None)
    monkeypatch.setattr(_child, "_use_c_ctype_locale", lambda: None)
    monkeypatch.delenv(_child._FAULT_ENV, raising=False)

    def refused(payload, cancel_event):
        raise _child._BudgetRefused("stable-diffusion.cpp generation failed: x")
    monkeypatch.setattr(_child, "_do_generate_image", refused)
    req_q, resp_q = queue.Queue(), queue.Queue()
    req_q.put(("generate_image", {}))
    req_q.put(None)
    _child.worker_main(req_q, resp_q, threading.Event())
    assert resp_q.get_nowait() == ("budget_refused", "stable-diffusion.cpp generation failed: x")


class _Proc:
    def is_alive(self):
        return True


def test_the_runner_raises_budget_refused_for_that_reply():
    r = SdRunner()
    r._proc, r._resp_q, r._cancel = _Proc(), queue.Queue(), threading.Event()
    r._resp_q.put(("budget_refused", "generation failed: x"))
    with pytest.raises(SdBudgetRefused, match="generation failed: x") as info:
        r._wait("generate_image", 5)
    assert isinstance(info.value, SdWorkerError)


# --------------------------------------------------------------------------- #
#  backends                                                                   #
# --------------------------------------------------------------------------- #

class _Runner:
    """A worker that refuses every generation of a load with a ``max_vram``
    budget, as the runtime does when a segment does not fit it."""

    def __init__(self, video=False, refuse_budgeted=True):
        self.video = video
        self.refuse_budgeted = refuse_budgeted
        self.loaded_key = None
        self.loaded_ctx = None
        self.alive = False
        self.pid = 1
        self.loads, self.generates, self.shutdowns = [], [], 0

    def is_alive(self):
        return self.alive

    def ensure_loaded(self, key, runtime_dir, ctx, **kw):
        info = {"version": "Flux", "image": True, "video": self.video}
        if self.alive and key == self.loaded_key:
            return info
        self.loads.append((key, dict(ctx)))
        self.loaded_key, self.loaded_ctx, self.alive = key, dict(ctx), True
        return info

    def _refuse(self, params, on_event):
        self.generates.append((params, dict(self.loaded_ctx)))
        if self.refuse_budgeted and self.loaded_ctx.get("max_vram"):
            for level, text in REFUSED_LINES:
                on_event(("log", level, text.split(" --- ")[0]))
            raise SdBudgetRefused("stable-diffusion.cpp generation failed: diffusion model "
                                  "compute failed")

    def generate_image(self, params, *, on_event=None, cancel_check=None, timeout=None):
        self._refuse(params, on_event)
        w, h = params["width"], params["height"]
        return {"width": w, "height": h, "channel": 3, "data": bytes([9, 9, 9]) * (w * h),
                "seed": params["seed"]}

    def generate_video(self, params, *, on_event=None, cancel_check=None, timeout=None):
        self._refuse(params, on_event)
        w, h = params["width"], params["height"]
        return {"width": w, "height": h, "channel": 3,
                "frames": [bytes([9, 9, 9]) * (w * h)] * params["video_frames"],
                "fps": params["fps"], "audio": None, "seed": params["seed"]}

    def shutdown(self, grace=5.0):
        self.shutdowns += 1
        self.alive = False
        self.loaded_key = self.loaded_ctx = None


def _gpu(free):
    return {"index": 0, "name": "AMD Radeon RX 6900 XT", "free": free, "total": 16 * GIB,
            "free_scope": discover.FREE_SCOPE_DEVICE}


@pytest.fixture
def media_env(tmp_path, monkeypatch):
    runner = _Runner()
    monkeypatch.setattr(shared, "runner", runner)
    monkeypatch.setattr(shared, "arm_idle_timer", lambda seconds=None: None)
    backend = {"name": "vulkan", "platform": "win32"}
    monkeypatch.setattr(sd_runtime, "resolve", lambda choice="auto": sd_runtime.Runtime(
        backend=backend["name"], path=tmp_path / "rt"))
    monkeypatch.setattr(sd_budget, "_platform", lambda: backend["platform"])
    monkeypatch.setattr(discover, "list_gpus",
                        lambda **kw: ([_gpu(int(0.6 * GIB))], discover.GPU_PROBE_OK))
    model = tmp_path / "model.gguf"
    model.write_bytes(b"weights")
    s = {"backend": "native", "native": {"model": str(model)}}
    return runner, s, backend


def test_a_refused_budget_reloads_without_it_and_saves_the_image(media_env, tmp_path):
    runner, s, _backend = media_env
    said = []
    out = tmp_path / "a.png"
    ok, msg = image_native.generate(s, "p", out, write_sidecar=False, width=64, height=64,
                                    on_progress=said.append)
    assert out.is_file() and out.stat().st_size > 0
    assert ok, msg
    (key1, ctx1), (key2, ctx2) = runner.loads
    assert ctx1["max_vram"] == "0.25"
    assert "max_vram" not in ctx2
    assert key1 == key2
    assert {k: v for k, v in ctx1.items() if k != "max_vram"} == ctx2
    assert [c.get("max_vram") for _p, c in runner.generates] == ["0.25", None]
    assert sd_budget.UNBUDGETED_RETRY in said
    assert not [line for line in said if line.startswith("stable-diffusion.cpp:")]


def test_the_next_job_keeps_the_model_loaded_without_the_budget(media_env, tmp_path):
    runner, s, _backend = media_env
    for name in ("a.png", "b.png"):
        ok, msg = image_native.generate(s, "p", tmp_path / name, write_sidecar=False,
                                        width=64, height=64)
        assert ok, msg
    assert len(runner.loads) == 2
    assert [c.get("max_vram") for _p, c in runner.generates] == ["0.25", None, None]


def test_every_image_of_a_batch_runs_after_the_retry(media_env, tmp_path):
    runner, s, _backend = media_env
    ok, msg = image_native.generate(s, "p", tmp_path / "a.png", write_sidecar=False,
                                    width=64, height=64, batch=3)
    assert ok, msg
    assert len(runner.loads) == 2
    assert [c.get("max_vram") for _p, c in runner.generates] == ["0.25", None, None, None]


def test_a_refusal_of_a_load_without_a_budget_is_reported_not_retried(media_env, tmp_path,
                                                                      monkeypatch):
    runner, s, backend = media_env
    backend["platform"] = "linux"
    monkeypatch.setattr(runner, "refuse_budgeted", False)

    def always(params, *, on_event=None, cancel_check=None, timeout=None):
        runner.generates.append((params, dict(runner.loaded_ctx)))
        raise SdBudgetRefused("stable-diffusion.cpp generation failed: diffusion model "
                              "compute failed")
    monkeypatch.setattr(runner, "generate_image", always)
    said = []
    out = tmp_path / "a.png"
    ok, msg = image_native.generate(s, "p", out, write_sidecar=False, width=64, height=64,
                                    on_progress=said.append)
    assert not out.exists()
    assert not ok
    assert "diffusion model compute failed" in msg
    assert len(runner.loads) == 1 and len(runner.generates) == 1
    assert sd_budget.UNBUDGETED_RETRY not in said


def test_a_refused_video_budget_reloads_without_it(tmp_path, monkeypatch):
    runner = _Runner(video=True)
    monkeypatch.setattr(shared, "runner", runner)
    monkeypatch.setattr(shared, "arm_idle_timer", lambda seconds=None: None)
    monkeypatch.setattr(sd_runtime, "resolve", lambda choice="auto": sd_runtime.Runtime(
        backend="vulkan", path=tmp_path / "rt"))
    monkeypatch.setattr(sd_budget, "_platform", lambda: "win32")
    monkeypatch.setattr(discover, "list_gpus",
                        lambda **kw: ([_gpu(int(0.6 * GIB))], discover.GPU_PROBE_OK))
    files = {}
    for k in ("dit", "t5", "vae"):
        files[k] = tmp_path / f"{k}.gguf"
        files[k].write_bytes(b"w")
    s = {"backend": "native", "native": {"model": str(files["dit"]), "t5xxl": str(files["t5"]),
                                         "vae": str(files["vae"])}}
    from localm.media import backend_choice, comfy_client
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    monkeypatch.setattr(backend_choice, "managed_comfy_active", lambda cfg=None: False)
    monkeypatch.setattr(comfy_client, "is_comfy_confirmed", lambda api_url=None: False)
    monkeypatch.setattr(comfy_client, "_comfy_alive", lambda url, timeout=3.0: False)
    monkeypatch.setattr(video_native, "_pyav_problem", lambda: None)
    said = []
    out = tmp_path / "v.mp4"
    video_native.generate(s, "p", out, write_sidecar=False, seconds=0.1, fps=8,
                          width=64, height=64, on_progress=said.append)
    assert [c.get("max_vram") for _p, c in runner.generates] == ["0.25", None]
    (_k1, ctx1), (_k2, ctx2) = runner.loads
    assert ctx1["max_vram"] == "0.25" and "max_vram" not in ctx2
    assert sd_budget.UNBUDGETED_RETRY in said


# --------------------------------------------------------------------------- #
#  job log lines                                                              #
# --------------------------------------------------------------------------- #

def _relayed(lines, retry):
    said = []
    relay = RecoveredWarnings(said.append)
    for _level, text in lines:
        relay.line(text.split(" --- ")[0])
    if retry:
        relay.budget_retry()
    relay.flush()
    return said


def test_a_retried_refusal_leaves_no_raw_lines():
    assert _relayed(REFUSED_LINES, retry=True) == []


def test_a_refusal_that_is_not_retried_shows_every_line_in_order():
    said = _relayed(REFUSED_LINES, retry=False)
    assert said == [f"stable-diffusion.cpp: {t.split(' --- ')[0]}" for _l, t in REFUSED_LINES]


def test_a_failed_batched_pass_is_still_shown():
    said = _relayed([(b.SD_LOG_ERROR, "batched diffusion model compute failed "
                                      "--- diffusion_engine.cpp:2805")], retry=False)
    assert said == ["stable-diffusion.cpp: batched diffusion model compute failed"]


def _progress(phase, step, steps, secs=0.0):
    return ("progress", {"phase": phase, "step": step, "steps": steps, "secs": secs})


STREAMED_JOB = ([_progress("loading", 0, 3), _progress("loading", 3, 3)]
                + [e for n in (1, 2) for e in (_progress("sampling", n, 2, 5.0),
                                              _progress("loading", 0, 4),
                                              _progress("loading", 4, 4))]
                + [_progress("decoding", 0, 1), _progress("decoding", 1, 1)])


@pytest.mark.parametrize("module", [image_native, video_native], ids=["image", "video"])
def test_weights_streamed_in_per_step_add_no_load_lines(module):
    said = []
    relay = module._event_relay(said.append)
    for event in STREAMED_JOB:
        relay(event)
    loads = [i for i, line in enumerate(said) if line.startswith("Loading weights")]
    steps = [i for i, line in enumerate(said) if line.startswith("Step ")]
    assert len(steps) == 2
    assert loads and all(i < steps[0] for i in loads)
    assert any(line.startswith("Decoding") for line in said[steps[-1]:])
