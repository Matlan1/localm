# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native media loads: the VRAM budget a stable-diffusion.cpp load gets, the wait
for the previous model's VRAM on a switch, diffusion flash attention on the ROCm
runtime, and the runtime in the worker's load key."""

from __future__ import annotations

import pytest

from localm import discover
from localm.media.sdcpp import shared
from localm.media.sdcpp import budget as sd_budget
from localm.media.sdcpp import runtime as sd_runtime
from localm.plugins.builtin.image.backends import native as image_native
from localm.plugins.builtin.video.backends import native as video_native

GIB = 1024 ** 3


def _gpu(index=0, free=10 * GIB, total=16 * GIB, scope=discover.FREE_SCOPE_DEVICE):
    return {"index": index, "name": "GPU", "free": free, "total": total, "free_scope": scope}


# --------------------------------------------------------------------------- #
#  vram_budget                                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("backend", ["rocm", "vulkan"])
def test_one_gpu_gets_its_free_vram_less_the_margin(backend):
    got = sd_budget.vram_budget(backend, gpus=[_gpu(free=10 * GIB)],
                                status=discover.GPU_PROBE_OK, platform="win32")
    assert got == sd_budget.Budget(setting="9.50", free=10 * GIB, total=16 * GIB)


@pytest.mark.parametrize("backend", ["cpu", "cuda", "metal"])
def test_a_runtime_with_its_own_device_wide_reading_gets_none(backend):
    assert sd_budget.vram_budget(backend, gpus=[_gpu()], status=discover.GPU_PROBE_OK,
                                 platform="win32") is None


def _unmeasured(got, reason):
    assert got is not None and got.setting is None and reason in got.reason


def test_off_windows_there_is_no_budget():
    assert sd_budget.vram_budget("rocm", gpus=[_gpu()], status=discover.GPU_PROBE_OK,
                                 platform="linux") is None


@pytest.mark.parametrize("status", [discover.GPU_PROBE_TIMEOUT, discover.GPU_PROBE_BUSY,
                                    discover.GPU_PROBE_INCONCLUSIVE])
def test_a_probe_that_did_not_complete_gives_no_budget(status):
    _unmeasured(sd_budget.vram_budget("rocm", gpus=[_gpu()], status=status,
                                      platform="win32"), "did not complete")


def test_a_process_scoped_reading_gives_no_budget():
    gpus = [_gpu(scope=discover.FREE_SCOPE_PROCESS)]
    _unmeasured(sd_budget.vram_budget("rocm", gpus=gpus, status=discover.GPU_PROBE_OK,
                                      platform="win32"), "only this process")


def test_one_unmeasured_gpu_among_several_gives_no_budget():
    gpus = [_gpu(0), {**_gpu(1), "free": None}]
    _unmeasured(sd_budget.vram_budget("rocm", gpus=gpus, status=discover.GPU_PROBE_OK,
                                      platform="win32"), "incomplete")


def test_a_boolean_reading_is_not_a_number():
    gpus = [{**_gpu(), "free": True}]
    _unmeasured(sd_budget.vram_budget("rocm", gpus=gpus, status=discover.GPU_PROBE_OK,
                                      platform="win32"), "incomplete")


def test_no_gpu_gives_no_budget():
    _unmeasured(sd_budget.vram_budget("rocm", gpus=[], status=discover.GPU_PROBE_OK,
                                      platform="win32"), "no GPU")


def test_several_rocm_gpus_get_one_assignment_each():
    gpus = [_gpu(1, free=4 * GIB, total=8 * GIB), _gpu(0, free=12 * GIB)]
    got = sd_budget.vram_budget("rocm", gpus=gpus, status=discover.GPU_PROBE_OK,
                                platform="win32")
    assert got.setting == "rocm0=11.50,rocm1=3.50"
    assert (got.free, got.total) == (16 * GIB, 24 * GIB)


def test_several_vulkan_gpus_give_no_budget():
    gpus = [_gpu(0), _gpu(1)]
    _unmeasured(sd_budget.vram_budget("vulkan", gpus=gpus, status=discover.GPU_PROBE_OK,
                                      platform="win32"), "several GPUs")


def test_an_almost_full_card_gets_the_floor_not_zero():
    got = sd_budget.vram_budget("rocm", gpus=[_gpu(free=100 * 1024 ** 2)],
                                status=discover.GPU_PROBE_OK, platform="win32")
    assert got.setting == "0.25"


def test_a_failing_probe_gives_no_budget(monkeypatch):
    def boom(**kw):
        raise RuntimeError("driver gone")
    monkeypatch.setattr(sd_budget, "_platform", lambda: "win32")
    monkeypatch.setattr(discover, "list_gpus", boom)
    _unmeasured(sd_budget.vram_budget("rocm"), "driver gone")


def test_the_live_probe_is_asked_for_a_fresh_reading(monkeypatch):
    calls = []

    def probe(**kw):
        calls.append(kw)
        return [_gpu(free=6 * GIB)], discover.GPU_PROBE_OK
    monkeypatch.setattr(sd_budget, "_platform", lambda: "win32")
    monkeypatch.setattr(discover, "list_gpus", probe)
    assert sd_budget.vram_budget("vulkan").setting == "5.50"
    assert calls == [{"return_status": True, "wait_for_inflight": True}]


def test_with_budget_sets_max_vram_and_says_the_free_vram(monkeypatch):
    monkeypatch.setattr(sd_budget, "vram_budget",
                        lambda backend: sd_budget.Budget("5.50", 6 * GIB, 16 * GIB))
    said = []
    ctx = {"diffusion_model_path": "m.gguf"}
    got = sd_budget.with_budget(ctx, "rocm", said.append)
    assert got == {"diffusion_model_path": "m.gguf", "max_vram": "5.50"}
    assert ctx == {"diffusion_model_path": "m.gguf"}
    assert said == ["Free VRAM: 6.0 of 16.0 GB."]


def test_an_unmeasured_reading_is_said_and_logged_not_silent(monkeypatch):
    warned = []

    class _Log:
        def warning(self, fmt, *args):
            warned.append(fmt % args)

        def debug(self, fmt, *args):
            pass

        info = error = debug
    monkeypatch.setattr("localm.debuglog.logger", _Log())
    monkeypatch.setattr(sd_budget, "vram_budget",
                        lambda backend: sd_budget.Budget(None, reason="the GPU probe did "
                                                         "not complete (timeout)"))
    said = []
    ctx = {"diffusion_model_path": "m.gguf"}
    assert sd_budget.with_budget(ctx, "rocm", said.append) is ctx
    assert len(said) == 1 and "could not be measured" in said[0] and "timeout" in said[0]
    assert len(warned) == 1 and "timeout" in warned[0]


def test_with_budget_leaves_ctx_alone_without_a_budget(monkeypatch):
    monkeypatch.setattr(sd_budget, "vram_budget", lambda backend: None)
    said = []
    ctx = {"diffusion_model_path": "m.gguf"}
    assert sd_budget.with_budget(ctx, "cpu", said.append) is ctx
    assert said == []


# --------------------------------------------------------------------------- #
#  image backend loads                                                        #
# --------------------------------------------------------------------------- #

class _Runner:
    def __init__(self, video=False):
        self.video = video
        self.loaded_key = None
        self.alive = False
        self.pid = 1
        self.loads, self.generates, self.shutdowns = [], [], 0
        self.events = []

    def is_alive(self):
        return self.alive

    def ensure_loaded(self, key, runtime_dir, ctx, **kw):
        if self.alive and key == self.loaded_key:
            return {"version": "Flux", "image": True, "video": self.video}
        self.loads.append((key, ctx))
        self.loaded_key, self.alive = key, True
        return {"version": "Flux", "image": True, "video": self.video}

    def generate_image(self, params, *, on_event=None, cancel_check=None, timeout=None):
        self.generates.append(params)
        for event in self.events:
            on_event(event)
        w, h = params["width"], params["height"]
        return {"width": w, "height": h, "channel": 3, "data": bytes([9, 9, 9]) * (w * h),
                "seed": params["seed"]}

    def generate_video(self, params, *, on_event=None, cancel_check=None, timeout=None):
        self.generates.append(params)
        w, h = params["width"], params["height"]
        return {"width": w, "height": h, "channel": 3,
                "frames": [bytes([9, 9, 9]) * (w * h)] * params["video_frames"],
                "fps": params["fps"], "audio": None, "seed": params["seed"]}

    def shutdown(self, grace=5.0):
        self.shutdowns += 1
        self.alive = False


@pytest.fixture
def image_env(tmp_path, monkeypatch):
    runner = _Runner()
    monkeypatch.setattr(shared, "runner", runner)
    monkeypatch.setattr(shared, "arm_idle_timer", lambda seconds=None: None)
    backend = {"name": "rocm"}
    monkeypatch.setattr(sd_runtime, "resolve", lambda choice="auto": sd_runtime.Runtime(
        backend=backend["name"], path=tmp_path / "rt"))
    probes = []

    def probe(**kw):
        probes.append(kw)
        return [_gpu(free=3 * GIB if runner.alive else 13 * GIB)], discover.GPU_PROBE_OK
    monkeypatch.setattr(sd_budget, "_platform", lambda: "win32")
    monkeypatch.setattr(discover, "list_gpus", probe)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"weights")
    s = {"backend": "native", "native": {"model": str(model)}}
    return runner, s, backend, probes


def test_a_rocm_load_gets_flash_attention_and_the_budget(image_env, tmp_path):
    runner, s, _backend, probes = image_env
    said = []
    ok, msg = image_native.generate(s, "p", tmp_path / "a.png", write_sidecar=False,
                                    width=64, height=64, on_progress=said.append)
    assert ok, msg
    (key, ctx), = runner.loads
    assert ctx["diffusion_flash_attn"] is True
    assert ctx["max_vram"] == "12.50"
    assert key[1] == "rocm"
    assert "Free VRAM: 13.0 of 16.0 GB." in said
    assert len(probes) == 1


def test_a_vulkan_load_gets_the_budget_without_flash_attention(image_env, tmp_path):
    runner, s, backend, _probes = image_env
    backend["name"] = "vulkan"
    ok, msg = image_native.generate(s, "p", tmp_path / "a.png", write_sidecar=False,
                                    width=64, height=64)
    assert ok, msg
    (_key, ctx), = runner.loads
    assert "diffusion_flash_attn" not in ctx
    assert ctx["max_vram"] == "12.50"


def test_a_loaded_model_is_reused_without_a_new_probe(image_env, tmp_path):
    runner, s, _backend, probes = image_env
    for name in ("a.png", "b.png"):
        ok, msg = image_native.generate(s, "p", tmp_path / name, write_sidecar=False,
                                        width=64, height=64)
        assert ok, msg
    assert len(runner.loads) == 1 and len(probes) == 1


def test_a_runtime_change_reloads_the_same_model(image_env, tmp_path):
    runner, s, backend, probes = image_env
    image_native.generate(s, "p", tmp_path / "a.png", write_sidecar=False, width=64, height=64)
    backend["name"] = "vulkan"
    ok, msg = image_native.generate(s, "p", tmp_path / "b.png", write_sidecar=False,
                                    width=64, height=64)
    assert ok, msg
    assert [k[1] for k, _ctx in runner.loads] == ["rocm", "vulkan"]
    assert [ctx["max_vram"] for _k, ctx in runner.loads] == ["12.50", "12.50"]
    assert len(probes) == 3


def _switch_env(image_env, monkeypatch, release_after):
    """A worker holding another model (3 GiB free while it runs); after it stops,
    the reading shows 13 GiB from the *release_after*-th read on."""
    runner, s, _backend, _probes = image_env
    runner.alive, runner.loaded_key = True, ("something else", "rocm")
    order = []

    def probe(**kw):
        order.append(runner.alive)
        stopped_reads = sum(1 for alive in order if not alive)
        free = 3 * GIB if runner.alive or stopped_reads < release_after else 13 * GIB
        return [_gpu(free=free)], discover.GPU_PROBE_OK
    monkeypatch.setattr(discover, "list_gpus", probe)
    return runner, s, order


def test_a_switch_reads_before_the_stop_and_budgets_after_the_release(image_env, tmp_path,
                                                                       monkeypatch):
    runner, s, order = _switch_env(image_env, monkeypatch, release_after=3)
    said = []
    ok, msg = image_native.generate(s, "p", tmp_path / "a.png", write_sidecar=False,
                                    width=64, height=64, on_progress=said.append)
    assert ok, msg
    assert order[0] is True and all(alive is False for alive in order[1:])
    (_key, ctx), = runner.loads
    assert ctx["max_vram"] == "12.50"
    assert "Free VRAM: 13.0 of 16.0 GB." in said


def test_a_release_that_never_shows_logs_and_still_loads(image_env, tmp_path, monkeypatch):
    runner, s, _order = _switch_env(image_env, monkeypatch, release_after=10 ** 6)
    monkeypatch.setattr(sd_budget, "RELEASE_TIMEOUT_S", 0.05)
    warned = []

    class _Log:
        def warning(self, fmt, *args):
            warned.append(fmt % args)

        def debug(self, fmt, *args):
            pass

        info = error = debug
    monkeypatch.setattr("localm.debuglog.logger", _Log())
    ok, msg = image_native.generate(s, "p", tmp_path / "a.png", write_sidecar=False,
                                    width=64, height=64)
    assert ok, msg
    (_key, ctx), = runner.loads
    assert ctx["max_vram"] == "2.50"
    assert any("did not rise" in w for w in warned)


@pytest.mark.parametrize("when", ["before the stop", "after the stop"])
def test_a_cancel_during_a_switch_loads_nothing(image_env, tmp_path, monkeypatch, when):
    runner, s, _order = _switch_env(image_env, monkeypatch, release_after=10 ** 6)
    monkeypatch.setattr(sd_budget, "RELEASE_TIMEOUT_S", 30.0)

    def cancel_check():
        return True if when == "before the stop" else not runner.alive
    ok, msg = image_native.generate(s, "p", tmp_path / "a.png", write_sidecar=False,
                                    width=64, height=64, cancel_check=cancel_check)
    assert (ok, msg) == (False, "Cancelled.")
    assert runner.loads == []


def test_a_failed_final_reading_falls_back_to_the_last_good_one(image_env, tmp_path,
                                                                monkeypatch):
    runner, s, _backend, _probes = image_env
    runner.alive, runner.loaded_key = True, ("something else", "rocm")
    reads = []

    def probe(**kw):
        reads.append(runner.alive)
        stopped = sum(1 for alive in reads if not alive)
        if runner.alive or stopped == 1:
            return [_gpu(free=3 * GIB)], discover.GPU_PROBE_OK
        return [], discover.GPU_PROBE_TIMEOUT
    monkeypatch.setattr(discover, "list_gpus", probe)
    monkeypatch.setattr(sd_budget, "RELEASE_TIMEOUT_S", 0.05)
    said = []
    ok, msg = image_native.generate(s, "p", tmp_path / "a.png", write_sidecar=False,
                                    width=64, height=64, on_progress=said.append)
    assert ok, msg
    (_key, ctx), = runner.loads
    assert ctx["max_vram"] == "2.50"
    assert sum(1 for alive in reads if not alive) >= 2
    assert not any("could not be measured" in line for line in said)


def test_a_first_load_does_not_wait_for_a_release(image_env, tmp_path, monkeypatch):
    runner, s, _backend, probes = image_env
    waited = []
    import localm.vram as vram
    monkeypatch.setattr(vram, "wait_for_vram_release",
                        lambda *a, **kw: waited.append(1) or (True, None))
    ok, msg = image_native.generate(s, "p", tmp_path / "a.png", write_sidecar=False,
                                    width=64, height=64)
    assert ok, msg
    assert waited == [] and len(probes) == 1


# --------------------------------------------------------------------------- #
#  video backend loads                                                        #
# --------------------------------------------------------------------------- #

def test_a_video_load_gets_the_budget_and_keeps_its_flash_attention(tmp_path, monkeypatch):
    runner = _Runner(video=True)
    monkeypatch.setattr(shared, "runner", runner)
    monkeypatch.setattr(shared, "arm_idle_timer", lambda seconds=None: None)
    monkeypatch.setattr(sd_runtime, "resolve", lambda choice="auto": sd_runtime.Runtime(
        backend="rocm", path=tmp_path / "rt"))
    monkeypatch.setattr(sd_budget, "_platform", lambda: "win32")
    monkeypatch.setattr(discover, "list_gpus",
                        lambda **kw: ([_gpu(free=9 * GIB)], discover.GPU_PROBE_OK))
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
    video_native.generate(s, "p", tmp_path / "v.mp4", write_sidecar=False,
                          seconds=0.1, fps=8, width=64, height=64)
    (key, ctx), = runner.loads
    assert ctx["max_vram"] == "8.50" and ctx["diffusion_flash_attn"] is True
    assert key[1] == "rocm"
