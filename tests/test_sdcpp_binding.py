# SPDX-License-Identifier: AGPL-3.0-or-later
"""The stable-diffusion.cpp ABI check: it must accept a library whose struct
layouts and commit match the binding and refuse one whose layout or commit
differs. The fake library below writes the pinned commit's init defaults through
a chosen layout, the way the real ``*_init`` functions do."""

from __future__ import annotations

import ctypes
import importlib.util
import os
from ctypes import Structure, c_int, c_int64
from pathlib import Path

import pytest

from localm.media.sdcpp import _binding as b
from localm.media.sdcpp import pins, sampling


def _fill_sample(sp):
    sp.guidance.txt_cfg = 7.0
    sp.guidance.img_cfg = float("inf")
    sp.guidance.distilled_guidance = 3.5
    sp.guidance.slg.layer_start = 0.01
    sp.guidance.slg.layer_end = 0.2
    sp.scheduler = b._SCHEDULER_COUNT
    sp.sample_method = b._SAMPLE_METHOD_COUNT
    sp.sample_steps = 20
    sp.eta = float("inf")
    sp.flow_shift = float("inf")


def _fill_cache(c):
    c.reuse_threshold = float("inf")
    c.start_percent, c.end_percent, c.error_decay_rate = 0.15, 0.95, 1.0
    c.use_relative_threshold = c.reset_error_on_compute = c.scm_policy_dynamic = True
    c.Fn_compute_blocks, c.residual_diff_threshold, c.max_warmup_steps = 8, 0.08, 8
    c.max_cached_steps = c.max_continuous_cached_steps = -1
    c.taylorseer_n_derivatives = c.taylorseer_skip_interval = 1
    c.spectrum_w, c.spectrum_m, c.spectrum_lam = 0.40, 3, 1.0
    c.spectrum_window_size, c.spectrum_flex_window = 2, 0.50
    c.spectrum_warmup_steps, c.spectrum_stop_percent = 4, 0.9


def _fill_hires(h):
    h.upscaler = b._SD_HIRES_UPSCALER_LATENT
    h.scale, h.denoising_strength, h.upscale_tile_size = 2.0, 0.7, 128


def _fill_img(s):
    _fill_sample(s.sample_params)
    s.clip_skip, s.width, s.height, s.seed, s.batch_count = -1, 512, 512, -1, 1
    s.ref_image_args = b""
    s.strength, s.control_strength, s.ip_adapter_strength = 0.75, 0.9, 1.0
    s.qwen_image_layers = 3
    s.pm_params.style_strength = 20.0
    s.pulid_params.id_weight = 1.0
    s.vae_tiling_params.target_overlap = 0.5
    _fill_cache(s.cache)
    _fill_hires(s.hires)


def _fill_vid(s):
    _fill_sample(s.sample_params)
    _fill_sample(s.high_noise_sample_params)
    s.high_noise_sample_params.sample_steps = -1
    s.width, s.height, s.seed, s.video_frames, s.fps = 512, 512, -1, 6, 16
    s.strength, s.moe_boundary, s.vace_strength = 0.75, 0.875, 1.0
    s.vae_tiling_params.target_overlap = 0.5
    _fill_cache(s.cache)
    _fill_hires(s.hires)


def _fill_ctx(s):
    s.n_threads, s.conditioning_cache_size = 6, 4
    s.wtype, s.rng_type, s.sampler_rng_type = b._SD_TYPE_COUNT, b._CUDA_RNG, b._RNG_TYPE_COUNT
    s.prediction, s.vae_format, s.auto_fit = b._PREDICTION_COUNT, b._SD_VAE_FORMAT_AUTO, True
    s.batched_cfg = True


def _with_extra_int_before(struct_type, field_name):
    """*struct_type* with an 8-byte field inserted before *field_name*, so every
    later field moves (a 4-byte one can land in alignment padding and move
    nothing)."""
    fields = []
    for f in struct_type._fields_:
        if f[0] == field_name:
            fields.append(("_inserted", c_int64))
        fields.append(f)
    return type(f"Shifted_{struct_type.__name__}", (Structure,), {"_fields_": fields})


_PINNED_SHORT = pins.COMMIT[:7].encode()


class _FakeLib:
    """Implements the three ``*_init`` functions and ``sd_commit``."""

    def __init__(self, commit=_PINNED_SHORT, ctx_type=b.sd_ctx_params_t,
                 img_type=b.sd_img_gen_params_t, vid_type=b.sd_vid_gen_params_t):
        self.commit = commit
        self.types = {"ctx": ctx_type, "img": img_type, "vid": vid_type}
        self.keep = []

    def _write(self, kind, ptr, fill):
        s = self.types[kind].from_address(ptr.value)
        fill(s)
        self.keep.append(s)

    def sd_ctx_params_init(self, ptr):
        self._write("ctx", ptr, _fill_ctx)

    def sd_img_gen_params_init(self, ptr):
        self._write("img", ptr, _fill_img)

    def sd_vid_gen_params_init(self, ptr):
        self._write("vid", ptr, _fill_vid)

    def sd_commit(self):
        return self.commit

    names = {"sample": list(sampling.SAMPLE_METHODS), "sched": list(sampling.SCHEDULERS),
             "hires": ["None", *sampling.HIRES_UPSCALER_LIBRARY_NAMES.values()]}

    def sd_sample_method_name(self, i):
        return self.names["sample"][i].encode()

    def sd_scheduler_name(self, i):
        return self.names["sched"][i].encode()

    def sd_hires_upscaler_name(self, i):
        return self.names["hires"][i].encode()


def test_matching_layout_and_commit_pass():
    lib = _FakeLib()
    assert b.check_layouts(lib) == []
    b.verify_abi(lib)


@pytest.mark.parametrize("kind,struct_type,field,expect", [
    ("img", b.sd_img_gen_params_t, "width", "img.width"),
    ("img", b.sd_img_gen_params_t, "cache", "img.cache"),
    ("ctx", b.sd_ctx_params_t, "conditioning_cache_size", "ctx.conditioning_cache_size"),
    ("ctx", b.sd_ctx_params_t, "n_threads", "ctx."),
    ("vid", b.sd_vid_gen_params_t, "video_frames", "vid.video_frames"),
    ("vid", b.sd_vid_gen_params_t, "sample_params", "vid."),
])
def test_a_shifted_field_is_reported(kind, struct_type, field, expect):
    shifted = _with_extra_int_before(struct_type, field)
    lib = _FakeLib(**{f"{kind}_type": shifted})
    bad = b.check_layouts(lib)
    assert bad, f"inserting an int before {field} went undetected"
    assert any(expect in item for item in bad), bad
    with pytest.raises(b.AbiMismatch, match="struct layout does not match"):
        b.verify_abi(lib)


def test_a_renamed_sampler_is_refused_with_the_abi_check():
    lib = _FakeLib()
    lib.names = dict(lib.names, sample=["renamed", *lib.names["sample"][1:]])
    assert b.check_layouts(lib) == ["sample method names"]
    with pytest.raises(b.AbiMismatch, match="sample method names"):
        b.verify_abi(lib)


def test_a_library_struct_larger_than_the_binding_keeps_its_trailing_defaults():
    grown = type("Grown", (Structure,), {"_fields_": list(b.sd_img_gen_params_t._fields_)
                                         + [("tail", c_int)]})

    def fill(s):
        _fill_img(s)
        s.tail = 1234

    lib = _FakeLib(img_type=grown)
    lib.sd_img_gen_params_init = lambda ptr: lib._write("img", ptr, fill)
    view = b._init_struct(lib, "sd_img_gen_params_init", b.sd_img_gen_params_t)
    view.width = 640
    raw = ctypes.string_at(ctypes.addressof(view), ctypes.sizeof(grown))
    back = grown.from_buffer_copy(raw)
    assert back.width == 640
    assert back.tail == 1234


@pytest.mark.parametrize("commit", [b"deadbee", b"", None, pins.COMMIT[:4].encode()])
def test_a_different_commit_is_refused(commit):
    lib = _FakeLib(commit=commit)
    with pytest.raises(b.AbiMismatch, match=f"localm binds commit {pins.COMMIT[:7]}"):
        b.verify_abi(lib)


# sd_ctx_params_t as include/stable-diffusion.h declares it at the pinned commit.
_CTX_HEADER = """
typedef struct {
    const char* model_path;
    const char* clip_l_path;
    const char* clip_g_path;
    const char* clip_vision_path;
    const char* t5xxl_path;
    const char* llm_path;
    const char* llm_vision_path;
    const char* diffusion_model_path;
    const char* high_noise_diffusion_model_path;
    const char* uncond_diffusion_model_path;
    const char* embeddings_connectors_path;
    const char* vae_path;
    const char* audio_vae_path;
    const char* audio_encoder_path;
    const char* taesd_path;
    const char* control_net_path;
    const char* ip_adapter_path;
    const char* motion_module_path;
    const sd_embedding_t* embeddings;
    uint32_t embedding_count;
    const char* photo_maker_path;
    const char* pulid_weights_path;
    const char* tensor_type_rules;
    int n_threads;
    enum sd_type_t wtype;
    enum rng_type_t rng_type;
    enum rng_type_t sampler_rng_type;
    enum prediction_t prediction;
    enum lora_apply_mode_t lora_apply_mode;
    bool enable_mmap;
    bool flash_attn;
    bool diffusion_flash_attn;
    bool tae_preview_only;
    bool diffusion_conv_direct;
    bool vae_conv_direct;
    bool force_sdxl_vae_conv_scale;
    enum sd_vae_format_t vae_format;
    const char* max_vram;
    bool disable_prefetch;
    bool eager_load;
    const char* backend;
    const char* params_backend;
    const char* split_mode;
    bool auto_fit;
    const char* rpc_servers;
    const char* model_args;
    bool disable_segmented_compute;
    bool batched_cfg;
    float linear_scale;
    float attn_scale;
    const char* tokenizer;
    bool sage_attn;
    int conditioning_cache_size;
    const char* flash_attn_mode;
    const char* conv_direct_mode;
} sd_ctx_params_t;
"""


def _confirm_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "confirm_sdcpp_runtime.py"
    spec = importlib.util.spec_from_file_location("_confirm_sdcpp_for_binding", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ctx_params_match_the_header_declaration_field_for_field():
    cf = _confirm_script()
    header = cf.parse_header_structs(_CTX_HEADER)
    bound = cf.binding_structs(b)
    assert cf.compare_layout(header, {"sd_ctx_params_t": bound["sd_ctx_params_t"]}) == []
    names = [f for f, _t in b.sd_ctx_params_t._fields_]
    assert names.index("batched_cfg") == names.index("disable_segmented_compute") + 1
    assert names[-2:] == ["flash_attn_mode", "conv_direct_mode"]


@pytest.mark.parametrize("field,value,expect", [
    ("batched_cfg", False, "ctx.batched_cfg"),
    ("flash_attn_mode", b"auto", "ctx string pointers not null"),
    ("conv_direct_mode", b"on", "ctx string pointers not null"),
])
def test_a_ctx_default_other_than_the_pinned_init_is_reported(field, value, expect):
    lib = _FakeLib()

    def fill(s):
        _fill_ctx(s)
        setattr(s, field, value)

    lib.sd_ctx_params_init = lambda ptr: lib._write("ctx", ptr, fill)
    assert expect in b.check_layouts(lib)
    with pytest.raises(b.AbiMismatch, match="struct layout does not match"):
        b.verify_abi(lib)


class _LoadLib(_FakeLib):
    """A library the worker's load request can run against; records the
    ``sd_ctx_params_t`` that ``new_sd_ctx`` receives."""

    def __init__(self):
        super().__init__()
        self.received = None
        self.received_model_path = None

    def sd_list_devices(self, buf, size):
        text = b"CPU\tfake cpu\n"
        if buf is not None:
            ctypes.memmove(buf, text, min(len(text), size))
        return len(text)

    def new_sd_ctx(self, ref):
        params = ctypes.cast(ref, ctypes.POINTER(b.sd_ctx_params_t)).contents
        self.received = b.sd_ctx_params_t.from_buffer_copy(params)
        self.received_model_path = params.model_path
        return 1

    def free_sd_ctx(self, ctx):
        pass

    def sd_get_model_version_name(self, ctx):
        return b"SD 1.x"

    def sd_ctx_supports_image_generation(self, ctx):
        return True

    def sd_ctx_supports_video_generation(self, ctx):
        return False

    def sd_get_default_sample_method(self, ctx):
        return 0

    def sd_get_default_scheduler(self, ctx, method):
        return 0


@pytest.mark.parametrize("ctx,want", [({}, True), ({"batched_cfg": False}, False),
                                      ({"batched_cfg": True}, True)])
def test_the_worker_load_passes_batched_cfg_to_the_library(monkeypatch, ctx, want):
    from localm.media.sdcpp import _child
    lib = _LoadLib()
    monkeypatch.setattr(_child._State, "lib", lib)
    for name in ("ctx", "last_errors", "model_desc", "paths"):
        monkeypatch.setattr(_child._State, name, getattr(_child._State, name))
    monkeypatch.setattr(_child._State, "ctx", None)
    _child._do_load({"ctx": {"model_path": "m.safetensors", **ctx}})
    assert lib.received is not None
    assert lib.received.batched_cfg is want
    assert lib.received_model_path == b"m.safetensors"
    assert b._raw_ptr(lib.received, "flash_attn_mode") == 0
    assert b._raw_ptr(lib.received, "conv_direct_mode") == 0


def test_the_pinned_commit_matches_the_tag():
    assert pins.TAG.endswith(pins.COMMIT[:7])


class _FakeGgmlLib:
    """A loaded library exposing ggml's backend registry functions."""

    def __init__(self, devices=0):
        self.devices = devices
        self.loaded_from = []

        def dev_count():
            return self.devices

        def load_all_from_path(path):
            self.loaded_from.append(path)
            self.devices += 1

        self.ggml_backend_dev_count = dev_count
        self.ggml_backend_load_all_from_path = load_all_from_path


def _no_cdll(path, mode=0):
    raise AssertionError(f"loaded {path} by path")


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_backends_register_through_the_main_library_handle(tmp_path, monkeypatch, platform):
    for name in ("libggml.so", "libggml.so.0", "libggml.dylib"):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(b.sys, "platform", platform)
    monkeypatch.setattr(b.ctypes, "CDLL", _no_cdll)
    lib = _FakeGgmlLib()
    b._register_backends(lib, tmp_path)
    assert lib.loaded_from == [str(tmp_path).encode("utf-8")]
    assert lib.devices == 1


def test_windows_registers_through_ggml_dll_beside_the_library(tmp_path, monkeypatch):
    (tmp_path / "ggml.dll").write_bytes(b"")
    ggml = _FakeGgmlLib()
    opened = []

    def cdll(path, mode=0):
        opened.append(path)
        return ggml

    monkeypatch.setattr(b.sys, "platform", "win32")
    monkeypatch.setattr(b.ctypes, "CDLL", cdll)
    lib = _FakeGgmlLib()
    b._register_backends(lib, tmp_path)
    assert opened == [str(tmp_path / "ggml.dll")]
    assert ggml.loaded_from == [str(tmp_path).encode("utf-8")]
    assert lib.loaded_from == []


def test_windows_without_ggml_dll_registers_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(b.sys, "platform", "win32")
    monkeypatch.setattr(b.ctypes, "CDLL", _no_cdll)
    lib = _FakeGgmlLib()
    b._register_backends(lib, tmp_path)
    assert lib.loaded_from == []


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_backends_already_registered_are_not_loaded_again(tmp_path, monkeypatch, platform):
    (tmp_path / "ggml.dll").write_bytes(b"")
    lib = _FakeGgmlLib(devices=1)
    monkeypatch.setattr(b.sys, "platform", platform)
    monkeypatch.setattr(b.ctypes, "CDLL", lambda path, mode=0: lib)
    b._register_backends(lib, tmp_path)
    assert lib.loaded_from == []


@pytest.mark.integration
def test_layouts_match_a_real_runtime():
    """Set LOCALM_SDCPP_TEST_RUNTIME to an installed runtime directory. The
    library is loaded in a worker process, never in the test process."""
    d = os.environ.get("LOCALM_SDCPP_TEST_RUNTIME")
    if not d:
        pytest.skip("LOCALM_SDCPP_TEST_RUNTIME is not set")
    from localm.media.sdcpp.runner import SdRunner
    info = SdRunner().probe(Path(d))
    assert info["commit"] == pins.COMMIT[:7]
    assert info["devices"]
