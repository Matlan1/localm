# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native image jobs on the Vulkan runtime: without flash attention the
diffusion model's attention builds a score matrix that must fit the device's
buffer limit (else ggml runs it on the CPU); with flash attention it is one GPU
job that must finish within the graphics driver's reset timeout. The backend
keeps the loaded attention when it is safe, reloads with flash attention set
the safe way when one is, and refuses when neither is."""

from __future__ import annotations

import queue

import pytest

from localm.media.sdcpp import _binding as b
from localm.media.sdcpp import _child, attention, gpureset
from localm.media.sdcpp.runner import SdWorkerError

from tests.test_sdcpp_gpu_reset_limit import ON_PCI, THREAD, FakeHost, dgpu, linux


@pytest.fixture(autouse=True)
def _reference_box(monkeypatch):
    from tests.test_sdcpp_gpu_reset_limit import pin_rx6900xt
    pin_rx6900xt(monkeypatch)


GIB = 1024 ** 3
RX6900XT_VULKAN_RATE = 0.455e12
FLUX = attention.image_attention("flux")


def test_image_shapes_by_family():
    assert (FLUX.heads, FLUX.head_dim, FLUX.stride, FLUX.text_tokens) == (24, 128, 16, 512)
    assert FLUX.tokens(1024, 1024) == 64 * 64 + 512
    assert FLUX.tokens(1024, 1024, 2) == 3 * 64 * 64 + 512
    sd1 = attention.image_attention("sd1")
    assert sd1.tokens(512, 512) == 64 * 64 and sd1.batch == 2
    assert sd1.score_bytes(512, 512) == 8 * 2 * 4096 ** 2 * 4
    assert attention.image_attention("sdxl").tokens(1024, 1024) == 64 * 64
    edit = attention.image_attention("qwen-image")
    assert edit.tokens(1024, 1024, 1) == 2 * 4096 + 400 + 512


def test_a_load_log_shape_replaces_the_family_default():
    klein = attention.image_attention("flux2", 3072, 24)
    assert (klein.heads, klein.head_dim, klein.stride) == (24, 128, 16)
    assert attention.image_attention("flux2").heads == 48


def test_every_image_family_has_a_shape():
    from localm.media.sdcpp import families
    for family in families.IMAGE_FAMILIES:
        assert attention.image_attention(family) is not None, family
    assert attention.image_attention("something-new") is None
    assert attention.image_attention(None) is None


def test_largest_square():
    assert attention.largest_square(lambda s: s <= 1000) == 960
    assert attention.largest_square(lambda s: False) == 0


@pytest.mark.parametrize("line,expected", [
    (b"flash attention auto: Vulkan0 -> off (probe: normal 0.634 ms, flash 5.892 ms)", False),
    (b"flash attention auto: Vulkan0 -> on (Vulkan NVIDIA RTX)", True),
])
def test_the_worker_reads_the_flash_attention_decision(monkeypatch, line, expected):
    monkeypatch.setattr(_child._State, "flash_auto_by_name", {})
    log_cb, _progress, _preview = _child._make_callbacks(queue.Queue())
    log_cb(b.SD_LOG_INFO, line + b" --- flash_attn.cpp:95", None)
    assert _child._State.flash_auto_by_name == {"vulkan0": expected}
    assert _child._module_flash({}, "diffusion", ["Vulkan0"]) is expected


def test_the_worker_reads_the_flux_shape_from_the_load_log(monkeypatch):
    monkeypatch.setattr(_child._State, "dit_hidden", None)
    monkeypatch.setattr(_child._State, "dit_heads", None)
    log_cb, _progress, _preview = _child._make_callbacks(queue.Queue())
    log_cb(b.SD_LOG_VERBOSE, b"flux: depth = 5, depth_single_blocks = 20, guidance_embed = false, "
           b"context_in_dim = 7680, hidden_size = 3072, num_heads = 24 --- flux.hpp:185", None)
    assert (_child._State.dit_hidden, _child._State.dit_heads) == (3072, 24)


@pytest.mark.parametrize("ctx,auto,expected", [
    ({"flash_attn_mode": "diffusion=on"}, False, True),
    ({"flash_attn_mode": "diffusion=off", "diffusion_flash_attn": True}, True, False),
    ({"diffusion_flash_attn": True}, False, True),
    ({}, False, False),
    ({}, None, None),
])
def test_the_load_info_says_whether_the_model_runs_flash_attention(monkeypatch, ctx, auto,
                                                                    expected):
    monkeypatch.setattr(_child._State, "flash_auto_by_name",
                        {} if auto is None else {"vulkan0": auto})
    assert _child._module_flash(ctx, "diffusion") is expected


def test_the_worker_passes_the_flash_mode_to_the_library():
    assert "flash_attn_mode" in _child._CTX_STRING_FIELDS


class _ImageRunner:
    def __init__(self, info, rate, loaded_key=None, loaded_ctx=None):
        self.info, self.rate = info, rate
        self.loaded_key, self.loaded_ctx = loaded_key, loaded_ctx
        self.alive = loaded_key is not None
        self.pid = 1
        self.loads, self.params, self.asked = [], [], []

    def is_alive(self):
        return self.alive

    def ensure_loaded(self, key, runtime_dir, ctx, **kw):
        if not (self.alive and key == self.loaded_key):
            self.loads.append((key, dict(ctx)))
        self.loaded_key, self.loaded_ctx, self.alive = key, dict(ctx), True
        info = dict(self.info)
        if "flash_attn_mode" in ctx:
            info["flash_attn"] = _child._module_flash(ctx, "diffusion")
            if ctx.get("control_net_path"):
                info["controlnet_flash_attn"] = _child._module_flash(ctx, "controlnet")
        return info

    def attention_rate(self, runtime_dir, head_dim=128, timeout=None, devices=None):
        self.asked.append(head_dim)
        self.devices = devices
        if isinstance(self.rate, Exception):
            raise self.rate
        return self.rate

    def generate_image(self, params, *, on_event=None, cancel_check=None, timeout=None):
        self.params.append(params)
        w, h = params["width"], params["height"]
        return {"width": w, "height": h, "channel": 3, "data": bytes([9, 9, 9]) * (w * h),
                "seed": params["seed"]}

    def shutdown(self, grace=5.0):
        self.alive = False


def _image(monkeypatch, tmp_path, size, *, flash=False, rate=RX6900XT_VULKAN_RATE,
           backend="vulkan", family="flux", held=None, cfg=1.0, info=None, host=None,
           control=False, **kw):
    from localm import discover
    from localm.media.sdcpp import budget as sd_budget
    from localm.media.sdcpp import runtime as sd_runtime
    from localm.media.sdcpp import shared
    from localm.plugins.builtin.image.backends import native as image_native
    models = {"key": ("m",), "ctx": {"diffusion_model_path": "m.gguf"}, "label": "Model",
              "recommended": None, "family": family, "components": {}, "notes": [],
              "defaults": None}
    if control:
        models["control_net"] = {"path": "cn.gguf", "name": "cn"}
        models["ctx"]["control_net_path"] = "cn.gguf"
        models["capabilities"] = {"controlnet": True}
        kw["control_image"] = _png(size)
    if host is not None:
        monkeypatch.setattr(gpureset, "_HOST", host)
    monkeypatch.setattr(image_native, "resolve_models", lambda s: dict(models))
    info = {"version": "Flux", "image": True, "max_tensor_bytes": 2 * GIB,
            "vae_decoder_ch": 128, "flash_attn": flash, **(info or {})}
    loaded_key = loaded_ctx = None
    if held:
        loaded_key = (("m",), backend, held)
        loaded_ctx = {"diffusion_model_path": "m.gguf", "flash_attn_mode": f"diffusion={held}"}
    runner = _ImageRunner(info, rate, loaded_key, loaded_ctx)
    monkeypatch.setattr(shared, "runner", runner)
    monkeypatch.setattr(shared, "arm_idle_timer", lambda seconds=None: None)
    monkeypatch.setattr(sd_runtime, "resolve", lambda choice="auto": sd_runtime.Runtime(
        backend=backend, path=tmp_path / "rt"))
    monkeypatch.setattr(sd_budget, "_platform", lambda: "linux")
    monkeypatch.setattr(discover, "list_gpus", lambda **k: ([], discover.GPU_PROBE_OK))
    said = []
    ok, msg = image_native.generate({"backend": "native", "native": {}}, "p", tmp_path / "a.png",
                                    write_sidecar=False, on_progress=said.append, width=size,
                                    height=size, cfg=cfg, **kw)
    return ok, msg, runner, said


def test_an_image_whose_scores_fit_runs_as_loaded_without_measuring(monkeypatch, tmp_path):
    ok, msg, runner, said = _image(monkeypatch, tmp_path, 1024)
    assert ok, msg
    assert len(runner.loads) == 1 and runner.asked == []
    assert not [line for line in said if "Reloading" in line]


def test_an_image_whose_scores_do_not_fit_reloads_with_flash_attention(monkeypatch, tmp_path):
    ok, msg, runner, said = _image(monkeypatch, tmp_path, 1152)
    assert ok, msg
    assert runner.asked == [128]
    assert len(runner.loads) == 2
    assert runner.loads[1][1]["flash_attn_mode"] == "diffusion=on"
    assert runner.loads[1][0] == (("m",), "vulkan", "on")
    assert any(line.startswith("Reloading Model with flash attention on") for line in said)
    assert runner.params and runner.params[0]["width"] == 1152


def test_the_next_job_keeps_the_flash_mode_it_holds(monkeypatch, tmp_path):
    ok, msg, runner, said = _image(monkeypatch, tmp_path, 1024, held="on")
    assert ok, msg
    assert runner.loads == []
    assert not [line for line in said if "Reloading" in line]


def test_a_slow_auto_flash_attention_is_turned_off_when_the_scores_fit(monkeypatch, tmp_path):
    ok, msg, runner, said = _image(monkeypatch, tmp_path, 1024, flash=True, rate=1e11)
    assert ok, msg
    assert runner.loads[-1][1]["flash_attn_mode"] == "diffusion=off"
    assert any(line.startswith("Reloading Model with flash attention off") for line in said)


def test_an_unknown_flash_state_counts_as_off(monkeypatch, tmp_path):
    ok, msg, runner, _said = _image(monkeypatch, tmp_path, 1024, flash=None, rate=1e11)
    assert ok, msg
    assert len(runner.loads) == 1 and runner.asked == []
    ok, msg, runner, _said = _image(monkeypatch, tmp_path, 1152, flash=None)
    assert ok, msg
    assert runner.loads[-1][1]["flash_attn_mode"] == "diffusion=on"


def test_an_unknown_family_keeps_the_loaded_attention(monkeypatch, tmp_path):
    ok, msg, runner, _said = _image(monkeypatch, tmp_path, 1536, family=None)
    assert ok, msg
    assert runner.asked == [] and len(runner.loads) == 1


def test_an_image_that_fits_neither_way_is_refused(monkeypatch, tmp_path):
    ok, msg, runner, _said = _image(monkeypatch, tmp_path, 1536)
    assert not ok
    assert runner.params == [] and len(runner.loads) == 1
    assert "1536x1536 is too large for Model on the Vulkan runtime" in msg
    assert "about 1152x1152" in msg and "about 2.6 s" in msg
    assert "Windows resets this GPU when one job runs longer than 2 s" in msg
    assert msg.endswith(", or pick the ROCm runtime in Settings > Images.")


@pytest.mark.parametrize("rate", [None, SdWorkerError("worker died")])
def test_an_unmeasured_rate_refuses_an_image_that_needs_flash(monkeypatch, tmp_path, rate):
    ok, msg, runner, _said = _image(monkeypatch, tmp_path, 1152, rate=rate)
    assert not ok and runner.params == []
    assert "could not be measured" in msg and "about 1024x1024" in msg


def test_a_hires_target_is_what_must_fit(monkeypatch, tmp_path):
    ok, msg, runner, _said = _image(monkeypatch, tmp_path, 768,
                                    hires={"upscaler": "lanczos", "scale": 2.0, "steps": 1,
                                           "denoise": 0.5})
    assert not ok and runner.params == []
    assert "1536x1536 is too large" in msg


def test_cfg_batches_two_images_in_a_unet_graph(monkeypatch, tmp_path):
    ok, _msg, runner, _said = _image(monkeypatch, tmp_path, 512, family="sd1", cfg=7.0)
    assert ok and runner.asked == []
    ok, msg, _runner, _said = _image(monkeypatch, tmp_path, 768, family="sd1", cfg=7.0, rate=1e9)
    assert not ok and "about 576x576" in msg
    ok, msg, _runner, _said = _image(monkeypatch, tmp_path, 768, family="sd1", cfg=1.0, rate=1e9)
    assert not ok and "about 704x704" in msg


@pytest.mark.parametrize("backend", ["rocm", "cuda"])
def test_other_runtimes_keep_their_image_attention(monkeypatch, tmp_path, backend):
    ok, msg, runner, _said = _image(monkeypatch, tmp_path, 1536, backend=backend)
    assert ok, msg
    assert runner.asked == [] and len(runner.loads) == 1


def test_the_rate_is_measured_at_the_models_head_size(monkeypatch, tmp_path):
    ok, msg, runner, _said = _image(monkeypatch, tmp_path, 1536, family="sdxl", cfg=7.0,
                                    rate=4e12)
    assert ok, msg
    assert runner.asked == [64]
    assert runner.loads[-1][1]["flash_attn_mode"] == "diffusion=on"


def _png(size):
    import io

    from PIL import Image
    out = io.BytesIO()
    Image.new("RGB", (size, size), (10, 20, 30)).save(out, format="PNG")
    return out.getvalue()


# --------------------------------------------------------------------------- #
#  The time limit follows what can reset the device                            #
# --------------------------------------------------------------------------- #

def test_a_gpu_windows_pauses_mid_dispatch_takes_flash_attention_at_any_size(monkeypatch,
                                                                            tmp_path):
    host = FakeHost(adapters=[dgpu(THREAD)])
    ok, msg, runner, _said = _image(monkeypatch, tmp_path, 1536, rate=1e9, host=host)
    assert ok, msg
    assert runner.asked == []
    assert runner.loads[-1][1]["flash_attn_mode"] == "diffusion=on"


def test_a_slow_auto_flash_attention_stays_on_without_a_time_limit(monkeypatch, tmp_path):
    host = FakeHost(adapters=[dgpu(THREAD)])
    ok, msg, runner, _said = _image(monkeypatch, tmp_path, 1024, flash=True, rate=1e9, host=host)
    assert ok, msg
    assert len(runner.loads) == 1 and runner.asked == []


def test_a_longer_driver_timeout_lets_a_larger_image_take_flash_attention(monkeypatch, tmp_path):
    host = FakeHost(**linux("amdgpu", release="6.8.0"))
    ok, msg, runner, _said = _image(monkeypatch, tmp_path, 1536,
                                    info={"compute_devices": [ON_PCI]}, host=host)
    assert ok, msg
    assert runner.devices == ["Vulkan0"]
    assert runner.loads[-1][1]["flash_attn_mode"] == "diffusion=on"


def test_an_unreadable_limit_is_said_when_the_image_asks_it(monkeypatch, tmp_path):
    host = FakeHost(platform="darwin")
    ok, msg, _runner, said = _image(monkeypatch, tmp_path, 1152, host=host)
    assert ok, msg
    assert any(line.startswith("localm could not read how long this GPU may run one job")
               for line in said)
    ok, msg, _runner, said = _image(monkeypatch, tmp_path, 1024, host=host)
    assert ok, msg
    assert not [line for line in said if "could not read" in line]


def test_the_image_refusal_offers_no_runtime_the_machine_lacks(monkeypatch, tmp_path):
    from localm.media.sdcpp import runtime as sd_runtime
    monkeypatch.setattr(sd_runtime, "other_gpu_runtimes", lambda det=None: [])
    ok, msg, _runner, _said = _image(monkeypatch, tmp_path, 1536)
    assert not ok and "runtime in Settings" not in msg
    assert msg.endswith("The largest square image that fits is about 1152x1152.")


# --------------------------------------------------------------------------- #
#  SD3 shapes from the load log                                                #
# --------------------------------------------------------------------------- #

def test_the_worker_reads_the_mmdit_shape_from_the_load_log(monkeypatch):
    monkeypatch.setattr(_child._State, "dit_hidden", None)
    monkeypatch.setattr(_child._State, "dit_heads", None)
    log_cb, _progress, _preview = _child._make_callbacks(queue.Queue())
    log_cb(b.SD_LOG_VERBOSE, b"mmdit: num_layers = 24, num_mmdit_x_layers = 13, hidden_size = "
           b"1536, patch_size = 2, in_channels = 16, out_channels = 16, context_size = 4096, "
           b"adm_in_channels = 2048, qk_norm = rms --- mmdit.hpp:129", None)
    assert (_child._State.dit_hidden, _child._State.dit_heads) == (1536, 24)


def test_sd3_medium_is_sized_by_its_logged_heads():
    medium = attention.image_attention("sd3", 1536, 24)
    large = attention.image_attention("sd3")
    assert (medium.heads, medium.head_dim) == (24, 64)
    assert (large.heads, large.head_dim) == (38, 64)
    assert medium.score_bytes(1024, 1024) <= 2 * GIB < large.score_bytes(1024, 1024)


def test_sd3_medium_at_1024_keeps_its_loaded_attention(monkeypatch, tmp_path):
    ok, msg, runner, _said = _image(monkeypatch, tmp_path, 1024, family="sd3",
                                    info={"dit_hidden": 1536, "dit_heads": 24})
    assert ok, msg
    assert len(runner.loads) == 1 and runner.asked == []
    ok, msg, runner, _said = _image(monkeypatch, tmp_path, 1024, family="sd3")
    assert ok, msg
    assert runner.loads[-1][1]["flash_attn_mode"] == "diffusion=on"


# --------------------------------------------------------------------------- #
#  A ControlNet gets the diffusion model's flash attention                     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ctx,auto,expected", [
    ({"flash_attn_mode": "diffusion=on,controlnet=on"}, False, True),
    ({"flash_attn_mode": "diffusion=on,control_net=off"}, True, False),
    ({"flash_attn_mode": "diffusion=on"}, False, False),
    ({"flash_attn_mode": "on"}, False, True),
    ({"diffusion_flash_attn": True}, False, False),
    ({"flash_attn": True}, False, True),
    ({}, None, None),
])
def test_the_load_info_says_whether_the_controlnet_runs_flash_attention(monkeypatch, ctx, auto,
                                                                         expected):
    monkeypatch.setattr(_child._State, "flash_auto_by_name",
                        {} if auto is None else {"vulkan0": auto})
    assert _child._module_flash(ctx, "controlnet") is expected


def test_a_controlnet_is_reloaded_with_the_diffusion_models_flash_attention(monkeypatch,
                                                                           tmp_path):
    ok, msg, runner, _said = _image(monkeypatch, tmp_path, 768, family="sd1", cfg=7.0,
                                    control=True)
    assert ok, msg
    assert runner.loads[-1][1]["flash_attn_mode"] == "diffusion=on,controlnet=on"


def test_a_controlnet_left_without_flash_attention_forces_the_reload(monkeypatch, tmp_path):
    from localm.plugins.builtin.image.backends import native as image_native
    rt = type("Rt", (), {"backend": "vulkan", "path": tmp_path})()
    runner = _ImageRunner({}, RX6900XT_VULKAN_RATE)
    models = {"family": "sd1", "label": "SD", "control_net": {"path": "cn.gguf"}}
    info = {"max_tensor_bytes": 2 * GIB, "flash_attn": True, "controlnet_flash_attn": None}
    assert image_native._vulkan_attention_mode(runner, rt, info, models, 768, 768, None, 0,
                                               7.0) == ("on", None)
    info["controlnet_flash_attn"] = True
    assert image_native._vulkan_attention_mode(runner, rt, info, models, 768, 768, None, 0,
                                               7.0) == (None, None)
    models.pop("control_net")
    info["controlnet_flash_attn"] = None
    assert image_native._vulkan_attention_mode(runner, rt, info, models, 768, 768, None, 0,
                                               7.0) == (None, None)


def test_a_slow_flash_attention_is_turned_off_for_the_controlnet_too(monkeypatch, tmp_path):
    from localm.plugins.builtin.image.backends import native as image_native
    rt = type("Rt", (), {"backend": "vulkan", "path": tmp_path})()
    runner = _ImageRunner({}, 1e9)
    models = {"family": "sd1", "label": "SD", "control_net": {"path": "cn.gguf"}}
    info = {"max_tensor_bytes": 2 * GIB, "flash_attn": False, "controlnet_flash_attn": True}
    assert image_native._vulkan_attention_mode(runner, rt, info, models, 512, 512, None, 0,
                                               7.0) == ("off", None)


def _auto_lines(monkeypatch, lines):
    monkeypatch.setattr(_child._State, "flash_auto_by_name", {})
    log_cb, _progress, _preview = _child._make_callbacks(queue.Queue())
    for line in lines:
        log_cb(b.SD_LOG_INFO, line + b" --- flash_attn.cpp:95", None)


def test_the_auto_decision_is_the_one_for_the_models_device(monkeypatch):
    _auto_lines(monkeypatch, [b"flash attention auto: Vulkan0 -> on (probe: normal 9 ms, flash 1 ms)",
                              b"flash attention auto: CPU -> off (probe: normal 1 ms, flash 2 ms)"])
    assert _child._module_flash({}, "diffusion", ["Vulkan0"]) is True
    assert _child._module_flash({}, "controlnet", ["Vulkan0"]) is True
    assert _child._module_flash({}, "diffusion", ["CPU"]) is False
    assert _child._module_flash({}, "diffusion", ["Vulkan0", "CPU"]) is False
    assert _child._module_flash({}, "diffusion", ["Vulkan1"]) is None
    assert _child._module_flash({}, "diffusion") is None


def test_a_module_whose_devices_disagree_runs_without_flash_attention(monkeypatch):
    _auto_lines(monkeypatch, [b"flash attention auto: Vulkan0 -> on (probe)",
                              b"flash attention auto: Vulkan1 -> off (probe)",
                              b"flash attention auto: diffusion -> off (compute devices disagree)"])
    assert _child._module_flash({}, "diffusion", ["Vulkan0"]) is False
    assert _child._module_flash({}, "controlnet", ["Vulkan0"]) is True

