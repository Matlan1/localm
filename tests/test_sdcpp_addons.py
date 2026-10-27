# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native image extras: masks (inpaint/outpaint), reference images for edit
models, ControlNet and textual-inversion embeddings. Classification from file
headers, the registry types, request validation, the native backend's payload
and refusals, the worker's ctypes structs, and the routes. Model files are
synthetic headers (tensor names and shapes only); the worker is a fake."""

from __future__ import annotations

import base64
import ctypes
import io
import json
from pathlib import Path

import pytest

from localm.media import native_models as nm
from localm.media.sdcpp import families, pairing
from localm.plugins.builtin.image.backends import native as image_native
from tests.test_sdcpp_families import (  # noqa: F401
    FLUX_DEV, QWEN_IMAGE, SD1_UNET, SDXL_UNET, Z_IMAGE, _finish, _register, flux_set, media_app, runner,
    write_gguf, write_safetensors,
)

P = "model.diffusion_model."

CN_SD1 = {"control_model.input_hint_block.0.weight": (16, 3, 3, 3),
          "control_model.zero_convs.0.0.weight": (320, 320, 1, 1),
          "control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": (320, 768),
          "control_model.middle_block_out.0.weight": (1280, 1280, 1, 1)}
CN_SD1_BARE = {k[len("control_model."):]: v for k, v in CN_SD1.items()}
CN_SDXL = {"input_hint_block.0.weight": (16, 3, 3, 3), "zero_convs.0.0.weight": (320, 320, 1, 1),
           "label_emb.0.0.weight": (1280, 2816)}
CN_SD1_DIFFUSERS = {
    "controlnet_cond_embedding.conv_in.weight": (16, 3, 3, 3),
    "controlnet_down_blocks.0.weight": (320, 320, 1, 1),
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight": (320, 768)}
CN_SDXL_DIFFUSERS = dict(CN_SD1_DIFFUSERS, **{"add_embedding.linear_1.weight": (1280, 2816)})
TI_SD1 = {"emb_params": (8, 768)}
TI_SD2 = {"emb_params": (4, 1024)}
TI_SDXL = {"clip_l": (6, 768), "clip_g": (6, 1280)}
SD1_CKPT = {P + "input_blocks.0.0.weight": (320, 4, 3, 3),
            P + "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": (320, 768),
            "first_stage_model.decoder.conv_in.weight": (512, 4, 3, 3),
            "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight":
                (49408, 768)}
SD1_INPAINT_CKPT = dict(SD1_CKPT, **{P + "input_blocks.0.0.weight": (320, 9, 3, 3)})
SD1_PIX2PIX = dict(SD1_UNET, **{P + "input_blocks.0.0.weight": (320, 8, 3, 3)})
FLUX_FILL = dict(FLUX_DEV, **{"img_in.weight": (3072, 384)})
FLUX_CONTROLS = dict(FLUX_DEV, **{"img_in.weight": (3072, 128)})
FLEX_2 = dict(FLUX_DEV, **{"img_in.weight": (3072, 196)})
Z_IMAGE_L2P = dict(Z_IMAGE, **{"local_decoder.out_conv.weight": (3, 64, 3, 3)})
QWEN_EDIT_2511 = dict(QWEN_IMAGE, **{"transformer_blocks.0.__index_timestep_zero__": (1,)})


def _info(tmp_path, shapes, name="f.safetensors"):
    return families.inspect_file(write_safetensors(tmp_path / name, shapes))


def _png(size=(64, 64), color=(10, 200, 30), mode="RGB", fmt="PNG") -> bytes:
    from PIL import Image
    out = io.BytesIO()
    Image.new(mode, size, color).save(out, format=fmt)
    return out.getvalue()


def _url(data: bytes, mime="image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def _mask(size=(64, 64), box=(16, 16, 48, 48)) -> bytes:
    from PIL import Image, ImageDraw
    im = Image.new("L", size, 0)
    if box is not None:
        ImageDraw.Draw(im).rectangle(box, fill=255)
    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


# --------------------------------------------------------------------------- #
#  classification                                                             #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("shapes,kind,family,layout", [
    (CN_SD1, "controlnet", "sd1", "original"),
    (CN_SD1_BARE, "controlnet", "sd1", "original"),
    (CN_SDXL, "controlnet", "sdxl", "original"),
    (CN_SD1_DIFFUSERS, "controlnet", "sd1", "diffusers"),
    (CN_SDXL_DIFFUSERS, "controlnet", "sdxl", "diffusers"),
    (TI_SD1, "textual_inversion", "sd1", None),
    (TI_SD2, "textual_inversion", "sd2", None),
    (TI_SDXL, "textual_inversion", "sdxl", None),
])
def test_controlnets_and_embeddings_are_classified_from_their_headers(tmp_path, shapes, kind,
                                                                      family, layout):
    info = _info(tmp_path, shapes)
    assert (info.kind, info.family) == (kind, family)
    assert info.detail.get("layout") == layout


def test_a_bare_sd1_controlnet_is_not_read_as_an_sd1_unet(tmp_path):
    info = _info(tmp_path, CN_SD1_BARE)
    assert info.kind == "controlnet"
    assert families.capabilities(info)["controlnet"] is False


@pytest.mark.parametrize("shapes", [
    {"a": (8, 768), "b": (8, 768), "c": (8, 768), "d": (8, 768), "e": (8, 768)},
    {"emb_params": (8, 777)},
    {"emb_params": (4096, 768)},
    {"emb_params": (2, 8, 768)},
    {"lora_unet_down_blocks_2_attentions_0_transformer_blocks_2_attn1_to_k.lora_down.weight":
         (64, 1280)},
    {"text_model.final_layer_norm.weight": (768,)},
])
def test_other_small_files_are_not_embeddings(tmp_path, shapes):
    assert _info(tmp_path, shapes).kind != "textual_inversion"


@pytest.mark.parametrize("shapes,fmt,want", [
    (FLUX_DEV, "gguf", {"references": True, "controlnet": False, "embeddings": False,
                        "inpaint": False, "edit": False}),
    (SD1_CKPT, "st", {"references": False, "controlnet": True, "embeddings": True,
                      "inpaint": False, "edit": False}),
    (SD1_INPAINT_CKPT, "st", {"references": False, "controlnet": True, "embeddings": True,
                              "inpaint": True, "edit": False}),
    (SD1_PIX2PIX, "st", {"references": True, "controlnet": True, "embeddings": True,
                         "inpaint": False, "edit": True}),
    (SDXL_UNET, "st", {"references": False, "controlnet": True, "embeddings": True,
                       "inpaint": False, "edit": False}),
    (FLUX_FILL, "st", {"references": False, "controlnet": False, "embeddings": False,
                       "inpaint": True, "edit": False}),
    (FLUX_CONTROLS, "st", {"references": False, "controlnet": False, "embeddings": False,
                           "inpaint": False, "edit": False}),
    (FLEX_2, "gguf", {"references": False, "controlnet": False, "embeddings": False,
                      "inpaint": False, "edit": False}),
    (Z_IMAGE, "st", {"references": True, "controlnet": False, "embeddings": False,
                     "inpaint": False, "edit": False}),
    (Z_IMAGE_L2P, "st", {"references": False, "controlnet": False, "embeddings": False,
                         "inpaint": False, "edit": False}),
    (QWEN_IMAGE, "st", {"references": True, "controlnet": False, "embeddings": False,
                        "inpaint": False, "edit": False}),
    (QWEN_EDIT_2511, "st", {"references": True, "controlnet": False, "embeddings": False,
                            "inpaint": False, "edit": True}),
])
def test_model_capabilities_follow_the_header(tmp_path, shapes, fmt, want):
    path = (write_gguf(tmp_path / "m.gguf", shapes, "flux") if fmt == "gguf"
            else write_safetensors(tmp_path / "m.safetensors", shapes))
    assert families.capabilities(families.inspect_file(path)) == want


def test_an_addon_fits_only_its_own_unet_family_and_layout(tmp_path):
    sd1 = _info(tmp_path, CN_SD1, "a.safetensors")
    sd1_diffusers = _info(tmp_path, CN_SD1_DIFFUSERS, "b.safetensors")
    xl_diffusers = _info(tmp_path, CN_SDXL_DIFFUSERS, "c.safetensors")
    ti = _info(tmp_path, TI_SD1, "d.safetensors")
    assert families.fits_unet_addon(sd1, "sd1") and not families.fits_unet_addon(sd1, "sdxl")
    assert not families.fits_unet_addon(sd1_diffusers, "sd1")
    assert families.fits_unet_addon(xl_diffusers, "sdxl")
    assert families.fits_unet_addon(ti, "sd1") and not families.fits_unet_addon(ti, "flux")


@pytest.mark.parametrize("shapes,want", [(CN_SD1, "controlnet"), (TI_SD1, "textual-inversion")])
def test_a_lone_addon_safetensors_registers_with_its_type(tmp_path, shapes, want):
    from localm.model_manager.registry import _media_component_type
    assert _media_component_type(write_safetensors(tmp_path / "x.safetensors", shapes),
                                 None) == want


def test_civitai_and_comfy_place_controlnets_and_embeddings_under_their_types():
    from localm.media.comfy_client import model_type_for_node
    from localm.model_manager.registry import MODEL_TYPES
    from localm.model_manager.sources import CIVITAI_TYPE_MAP
    assert CIVITAI_TYPE_MAP["TextualInversion"][0] == "textual-inversion"
    assert CIVITAI_TYPE_MAP["Controlnet"][0] == "controlnet"
    assert {"controlnet", "textual-inversion"} <= MODEL_TYPES
    assert model_type_for_node("ControlNetLoader") == "controlnet"
    assert model_type_for_node("DiffControlNetLoader") == "controlnet"


# --------------------------------------------------------------------------- #
#  request validation                                                         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,needle", [
    ("C:/x/y.png", "data URL"),
    ("data:image/gif;base64,R0lGODlh", "data URL"),
    ("data:image/png;base64,@@@@", "base64"),
    (_url(b"not an image"), "could not be read"),
    (_url(_png((4, 4))), "each side"),
    (_url(_png((64, 64), fmt="GIF", mode="P", color=1)), "PNG, JPEG or WebP"),
])
def test_bad_image_data_is_refused_with_a_reason(raw, needle):
    with pytest.raises(ValueError) as ei:
        nm.decode_image_data(raw, "mask")
    assert needle in str(ei.value)


def test_good_image_data_decodes_to_its_bytes():
    data = _png()
    assert nm.decode_image_data(_url(data), "x") == data
    jpeg = _png(fmt="JPEG")
    assert nm.decode_image_data(_url(jpeg, "image/jpeg"), "x") == jpeg


@pytest.mark.parametrize("kw,needle", [
    ({"mask": _url(_mask())}, "needs a base image"),
    ({"input_image": "a.png", "input_image_data": _url(_png())}, "not both"),
    ({"ref_images": [_url(_png())] * 5}, "At most 4"),
    ({"control": nm.ControlChoice(model="cn", image=_url(_png()), preprocess="depth")},
     "Unknown control preprocess"),
])
def test_image_input_combinations_are_checked(kw, needle):
    args = {"input_image": None, "input_image_data": None, "mask": None, "ref_images": None,
            "control": None, **kw}
    with pytest.raises(ValueError) as ei:
        nm.check_image_inputs(**args)
    assert needle in str(ei.value)


def test_addons_must_be_registered_names_of_their_type():
    reg = {"cn": {"model_type": "controlnet", "path": "x"},
           "ti": {"model_type": "textual-inversion", "path": "y"},
           "lo": {"model_type": "lora", "path": "z"}}
    img = _url(_png())
    out = nm.validate_addons(nm.ControlChoice(model="cn", image=img, strength=0.5),
                             ["ti", "ti"], reg)
    assert out == {"control_net": "cn", "control_strength": 0.5, "control_preprocess": "none",
                   "embeddings": ["ti"]}
    for control, emb, needle in [
        (nm.ControlChoice(model="lo", image=img), None, "not a registered ControlNet"),
        (nm.ControlChoice(model="D:/models/cn.safetensors", image=img), None,
         "not a registered ControlNet"),
        (None, ["cn"], "not a registered embedding"),
        (None, ["x"] * 17, "At most 16"),
    ]:
        with pytest.raises(ValueError) as ei:
            nm.validate_addons(control, emb, reg)
        assert needle in str(ei.value)


# --------------------------------------------------------------------------- #
#  the native backend                                                         #
# --------------------------------------------------------------------------- #

@pytest.fixture
def addon_set(flux_set, tmp_path):  # noqa: F811
    d = tmp_path / "addons"
    d.mkdir()
    files = {
        "sd15": (write_safetensors(d / "sd15.safetensors", SD1_CKPT), "diffusion-unet"),
        "sd15-inpaint": (write_safetensors(d / "sd15_inpaint.safetensors", SD1_INPAINT_CKPT),
                         "diffusion-unet"),
        "flux-canny": (write_gguf(d / "flux1-canny-dev-q8_0.gguf", FLUX_CONTROLS, "flux"),
                       "diffusion-unet"),
        "canny15": (write_safetensors(d / "control_v11p_sd15_canny.safetensors", CN_SD1),
                    "controlnet"),
        "canny-xl": (write_safetensors(d / "canny_xl.safetensors", CN_SDXL), "controlnet"),
        "old-scan-cn": (write_safetensors(d / "old_cn.safetensors", CN_SD1_BARE), "unknown"),
        "EasyNeg": (write_safetensors(d / "EasyNegative.safetensors", TI_SD1),
                    "textual-inversion"),
        "xl-emb": (write_safetensors(d / "xl_emb.safetensors", TI_SDXL), "textual-inversion"),
    }
    (d / "dup").mkdir()
    files["dup-emb"] = (write_safetensors(d / "dup" / "EasyNegative.safetensors", TI_SD1),
                        "textual-inversion")
    for name, (path, mtype) in files.items():
        _register(name, path, mtype)
    return {**flux_set, **{k: v[0] for k, v in files.items()}}


def _gen(runner, tmp_path, native: dict, prompt="a fox", **kw):  # noqa: F811
    lines: list = []
    out = tmp_path / "o.png"
    ok, msg = image_native.generate({"backend": "native", "native": native}, prompt, out,
                                    on_progress=lines.append, **kw)
    side = out.with_suffix(".png.json")
    record = json.loads(side.read_text(encoding="utf-8")) if side.exists() else None
    return ok, msg, lines, record


def test_a_controlnet_and_embeddings_load_with_the_model_and_drive_the_job(addon_set, runner,  # noqa: F811
                                                                         tmp_path):
    ok, msg, lines, rec = _gen(
        runner, tmp_path,
        {"model": "sd15", "control_net": "canny15", "embeddings": ["EasyNeg"]},
        prompt="a fox", negative_prompt="EasyNegative, blurry", width=128, height=96, seed=3,
        control_image=_png((300, 200)), control_strength=0.6, control_preprocess="canny")
    assert ok, msg
    ctx = runner.loads[-1]
    assert ctx["control_net_path"] == str(addon_set["canny15"])
    assert ctx["embeddings"] == (("EasyNegative", str(addon_set["EasyNeg"])),)
    params = runner.generates[-1]
    assert {k: params["control_image"][k] for k in ("width", "height", "channel")} == \
        {"width": 128, "height": 96, "channel": 3}
    assert len(params["control_image"]["data"]) == 128 * 96 * 3
    assert (params["control_strength"], params["control_canny"]) == (0.6, True)
    assert rec["control"] == {"model": "canny15", "strength": 0.6, "preprocess": "canny"}
    assert rec["embeddings"] == ["EasyNeg"]
    assert "control_net_path" not in json.dumps(rec)
    assert any(line.startswith("ControlNet: canny15 (0.6, canny)") for line in lines)
    assert not any("no effect" in line for line in lines)


def test_an_embedding_missing_from_the_prompt_is_noted(addon_set, runner, tmp_path):  # noqa: F811
    ok, _, lines, _ = _gen(runner, tmp_path, {"model": "sd15", "embeddings": ["EasyNeg"]},
                           width=64, height=64)
    assert ok
    assert any("no effect" in line and "EasyNegative" in line for line in lines)


def test_a_changed_controlnet_or_embedding_set_is_a_new_load_key(addon_set):
    plain = image_native.resolve_models({"native": {"model": "sd15"}})
    with_cn = image_native.resolve_models({"native": {"model": "sd15",
                                                      "control_net": "canny15"}})
    with_emb = image_native.resolve_models({"native": {"model": "sd15",
                                                       "embeddings": ["EasyNeg"]}})
    assert len({plain["key"], with_cn["key"], with_emb["key"]}) == 3
    assert "control_net_path" not in plain["ctx"] and "embeddings" not in plain["ctx"]


def test_an_unknown_typed_controlnet_from_an_old_scan_is_accepted_by_its_header(addon_set):
    m = image_native.resolve_models({"native": {"model": "sd15", "control_net": "old-scan-cn"}})
    assert m["ctx"]["control_net_path"] == str(addon_set["old-scan-cn"])


@pytest.mark.parametrize("native,needle", [
    ({"model": "sd15", "control_net": "canny-xl"}, "made for SDXL, not Stable Diffusion 1.x"),
    ({"model": "fluxunchained", "control_net": "canny15"}, "FLUX.1 models do not take"),
    ({"model": "sd15", "control_net": "EasyNeg"}, "not a registered ControlNet"),
    ({"model": "sd15", "embeddings": ["xl-emb"]}, "made for SDXL"),
    ({"model": "sd15", "embeddings": ["EasyNeg", "dup-emb"]}, "both use the trigger word"),
    ({"model": "canny15"}, "is a ControlNet, not an image model"),
])
def test_an_addon_that_cannot_work_is_refused_before_any_load(addon_set, runner, tmp_path,  # noqa: F811
                                                             native, needle):
    ok, msg, _, _ = _gen(runner, tmp_path, native, width=64, height=64,
                         control_image=_png() if native.get("control_net") else None)
    assert not ok and needle in msg, msg
    assert runner.loads == []


def test_reference_images_reach_an_edit_family_and_are_refused_elsewhere(addon_set, runner,  # noqa: F811
                                                                       tmp_path):
    refs = [_png((300, 150)), _png((40, 80))]
    ok, msg, _, rec = _gen(runner, tmp_path, {"model": "fluxunchained"}, width=64, height=64,
                           ref_images=refs)
    assert ok, msg
    got = runner.generates[-1]["ref_images"]
    assert [(r["width"], r["height"], r["channel"]) for r in got] == [(296, 144, 3), (64, 80, 3)]
    assert rec["reference_images"] == 2
    ok, msg, _, _ = _gen(runner, tmp_path, {"model": "sd15"}, width=64, height=64,
                         ref_images=refs)
    assert not ok and "does not take reference images" in msg
    assert image_native.input_refusal({"native": {"model": "sd15"}}, ref_images=refs) == msg


def test_a_mask_repaints_on_the_base_image_and_is_recorded(addon_set, runner, tmp_path):  # noqa: F811
    ok, msg, lines, rec = _gen(runner, tmp_path, {"model": "sd15-inpaint"},
                               init_image_data=_png((96, 64)), mask_image=_mask((48, 32),
                                                                                (0, 0, 24, 32)))
    assert ok, msg
    params = runner.generates[-1]
    assert (params["init_image"]["width"], params["init_image"]["height"]) == (96, 64)
    mask = params["mask_image"]
    assert (mask["width"], mask["height"], mask["channel"]) == (96, 64, 1)
    assert mask["data"][0] == 255 and mask["data"][95] == 0
    assert params["strength"] == 0.75
    assert rec["mask"] is True and rec["input_image_data"] is True
    assert rec.get("input_image") is None
    assert "Inpainting with a dedicated inpainting model." in lines


@pytest.mark.parametrize("kw,needle", [
    ({"mask_image": _mask()}, "needs a base image"),
    ({"mask_image": _mask(box=None), "init_image_data": _png()}, "mask is empty"),
    ({"mask_image": _mask(), "init_image_data": _png()[:60]}, "base image could not be read"),
    ({"init_image_data": _png(), "control_image": _png()[:60]},
     "control image could not be read"),
])
def test_a_mask_that_cannot_work_fails_with_a_reason(addon_set, runner, tmp_path, kw, needle):  # noqa: F811
    ok, msg, _, _ = _gen(runner, tmp_path, {"model": "sd15"}, **kw)
    assert not ok and needle in msg, msg
    assert runner.loads == [] and runner.generates == []


def test_a_control_image_without_a_controlnet_and_the_reverse_are_refused(addon_set, runner,  # noqa: F811
                                                                       tmp_path):
    ok, msg, _, _ = _gen(runner, tmp_path, {"model": "sd15"}, control_image=_png())
    assert not ok and "needs a ControlNet" in msg
    ok, msg, _, _ = _gen(runner, tmp_path, {"model": "sd15", "control_net": "canny15"})
    assert not ok and "needs a control image" in msg
    assert runner.loads == []


# --------------------------------------------------------------------------- #
#  the worker                                                                 #
# --------------------------------------------------------------------------- #

def _px(w, h, c, fill=7):
    return {"width": w, "height": h, "channel": c, "data": bytes([fill]) * (w * h * c)}


def test_the_worker_sets_mask_reference_and_control_structs():
    from localm.media.sdcpp import _binding as b
    from localm.media.sdcpp import _child
    p = b.sd_img_gen_params_t()
    keep = _child._set_image_inputs(b, p, {
        "mask_image": _px(8, 4, 1, 255), "ref_images": [_px(4, 4, 3, 1), _px(2, 6, 3, 2)],
        "control_image": _px(8, 4, 3, 9), "control_strength": 0.4, "control_canny": True})
    assert (p.mask_image.width, p.mask_image.height, p.mask_image.channel) == (8, 4, 1)
    assert p.mask_image.data[0] == 255
    assert p.ref_images_count == 2
    assert (p.ref_images[1].width, p.ref_images[1].height, p.ref_images[1].data[0]) == (2, 6, 2)
    assert (p.control_image.channel, p.control_image.data[0]) == (3, 9)
    assert abs(p.control_strength - 0.4) < 1e-6
    assert p.image_preprocess.rules == b"target=control,canny=true"
    assert keep
    q = b.sd_img_gen_params_t()
    assert _child._set_image_inputs(b, q, {}) == []
    assert q.ref_images_count == 0 and not q.mask_image.data and q.image_preprocess.rules is None


@pytest.mark.parametrize("img,channels", [({"width": 4, "height": 4, "channel": 3,
                                            "data": b"\0" * 47}, (3,)),
                                          (_px(4, 4, 3), (1,))])
def test_the_worker_refuses_an_image_buffer_that_does_not_match(img, channels):
    from localm.media.sdcpp import _binding as b
    from localm.media.sdcpp import _child
    with pytest.raises(ValueError):
        _child._image_struct(b, img, channels)


class _FakeLib:
    """The library functions ``_child._do_load`` calls, recording the ctx
    params ``new_sd_ctx`` receives."""

    def __init__(self):
        self.seen = None

    def sd_ctx_params_init(self, ptr):
        pass

    def sd_list_devices(self, buf, size):
        text = b"ROCm0\tfake\n"
        if buf is not None:
            ctypes.memmove(buf, text, len(text))
        return len(text)

    def new_sd_ctx(self, ref):
        params = ref._obj
        embs = [(params.embeddings[i].name, params.embeddings[i].path)
                for i in range(params.embedding_count)]
        self.seen = {"control_net_path": params.control_net_path,
                     "model_path": params.model_path, "embeddings": embs}
        return 1234

    def sd_get_model_version_name(self, ctx):
        return b"SD 1.x"

    def sd_get_default_sample_method(self, ctx):
        return 0

    def sd_get_default_scheduler(self, ctx, method):
        return 0

    def sd_ctx_supports_image_generation(self, ctx):
        return True

    def sd_ctx_supports_video_generation(self, ctx):
        return False

    def sd_sample_method_name(self, v):
        return b"euler"

    def sd_scheduler_name(self, v):
        return b"discrete"


def test_the_worker_load_passes_the_controlnet_and_embeddings(monkeypatch):
    from localm.media.sdcpp import _child
    lib = _FakeLib()
    monkeypatch.setattr(_child._State, "lib", lib)
    monkeypatch.setattr(_child._State, "ctx", None)
    out = _child._do_load({"runtime_dir": "x", "ctx": {
        "model_path": "D:/m/sd15.safetensors", "control_net_path": "D:/m/cn.safetensors",
        "embeddings": (("EasyNegative", "D:/m/EasyNegative.safetensors"),
                       ("Other", "D:/m/o.safetensors"))}})
    assert out["image"] is True and "ref_images" not in out
    assert lib.seen == {"control_net_path": b"D:/m/cn.safetensors",
                        "model_path": b"D:/m/sd15.safetensors",
                        "embeddings": [(b"EasyNegative", b"D:/m/EasyNegative.safetensors"),
                                       (b"Other", b"D:/m/o.safetensors")]}
    assert "D:/m/o.safetensors" in _child._State.paths
    assert "D:/m" not in _child._redacted("embedding D:/m/o.safetensors failed")
    monkeypatch.setattr(_child._State, "ctx", None)


# --------------------------------------------------------------------------- #
#  routes                                                                     #
# --------------------------------------------------------------------------- #

def test_imagine_route_runs_mask_controlnet_and_embeddings(media_app, addon_set, runner):  # noqa: F811
    from fastapi.testclient import TestClient
    with TestClient(media_app) as c:
        r = c.post("/api/imagine", json={
            "prompt": "a fox", "negative_prompt": "EasyNegative",
            "native_models": {"model": "sd15"},
            "input_image_data": _url(_png((64, 64))), "mask": _url(_mask()),
            "control": {"model": "canny15", "image": _url(_png((32, 32))), "strength": 0.7,
                        "preprocess": "canny"},
            "embeddings": ["EasyNeg"]})
        assert r.status_code == 200, r.text
        job = _finish(media_app, r.json()["job_id"])
    lines = [e.get("text", "") for e in job._history if e.get("type") == "line"]
    assert job.status == "done", lines
    ctx = runner.loads[-1]
    assert ctx["control_net_path"] == str(addon_set["canny15"])
    assert ctx["embeddings"] == (("EasyNegative", str(addon_set["EasyNeg"])),)
    params = runner.generates[-1]
    assert params["mask_image"]["channel"] == 1 and params["control_canny"] is True
    assert params["control_image"]["width"] == 64 and params["control_strength"] == 0.7


@pytest.mark.parametrize("body,needle", [
    ({"mask": "data:image/png;base64,AAAA"}, "needs a base image"),
    ({"input_image_data": "C:/Windows/win.ini"}, "data URL"),
    ({"control": {"model": "C:/x/cn.safetensors", "image": "data:image/png;base64,AA"}},
     "not a registered ControlNet"),
    ({"embeddings": ["../../EasyNegative"]}, "not a registered embedding"),
    ({"ref_images": ["data:image/png;base64,AA"] * 5}, ""),
])
def test_imagine_route_refuses_bad_image_inputs_up_front(media_app, addon_set, runner, body,  # noqa: F811
                                                         needle):
    from fastapi.testclient import TestClient
    with TestClient(media_app) as c:
        r = c.post("/api/imagine", json={"prompt": "p", "native_models": {"model": "sd15"},
                                         **body})
    assert r.status_code in (400, 422), r.text
    assert needle in r.text
    assert runner.loads == []


def test_imagine_route_refuses_references_for_a_model_without_them_before_any_load(
        media_app, addon_set, runner):  # noqa: F811
    from fastapi.testclient import TestClient
    with TestClient(media_app) as c:
        r = c.post("/api/imagine", json={"prompt": "p", "native_models": {"model": "sd15"},
                                         "ref_images": [_url(_png())]})
        assert r.status_code == 200, r.text
        job = _finish(media_app, r.json()["job_id"])
    lines = [e.get("text", "") for e in job._history if e.get("type") == "line"]
    assert job.status != "done"
    assert any("does not take reference images" in line for line in lines)
    assert not any("backend ready" in line for line in lines), lines
    assert runner.loads == []


def test_imagine_route_refuses_an_empty_mask_before_the_backend_is_prepared(
        media_app, addon_set, runner):  # noqa: F811
    from fastapi.testclient import TestClient
    with TestClient(media_app) as c:
        r = c.post("/api/imagine", json={"prompt": "p", "native_models": {"model": "sd15"},
                                         "input_image_data": _url(_png()),
                                         "mask": _url(_mask(box=None))})
        assert r.status_code == 200, r.text
        job = _finish(media_app, r.json()["job_id"])
    lines = [e.get("text", "") for e in job._history if e.get("type") == "line"]
    assert any("mask is empty" in line for line in lines), lines
    assert not any("backend ready" in line for line in lines), lines
    assert runner.loads == []


def test_a_camera_jpeg_with_several_frames_is_accepted():
    from PIL import Image
    out = io.BytesIO()
    Image.new("RGB", (32, 24), (1, 2, 3)).save(
        out, format="MPO", save_all=True, append_images=[Image.new("RGB", (32, 24))])
    data = out.getvalue()
    assert Image.open(io.BytesIO(data)).format == "MPO"
    assert nm.decode_image_data(_url(data, "image/jpeg"), "x") == data


def test_exif_orientation_is_applied_to_input_images():
    from PIL import Image
    im = Image.new("RGB", (160, 80), (200, 0, 0))
    exif = im.getexif()
    exif[0x0112] = 6
    out = io.BytesIO()
    im.save(out, format="JPEG", exif=exif.tobytes())
    data = out.getvalue()
    got = image_native._load_rgb(data)
    assert (got["width"], got["height"]) == (80, 160)
    init = image_native._load_init_image(data, None, None)
    assert (init["width"], init["height"]) == (80, 160)


def test_an_embedding_from_an_older_civitai_download_still_works(addon_set, runner,  # noqa: F811
                                                                  tmp_path):
    from localm.media.sdcpp import pairing as pr
    path = write_safetensors(tmp_path / "old_style.safetensors", TI_SD1)
    _register("old-civitai-ti", path, "embedding")
    assert nm.validate_addons(None, ["old-civitai-ti"])["embeddings"] == ["old-civitai-ti"]
    m = image_native.resolve_models({"native": {"model": "sd15",
                                                "embeddings": ["old-civitai-ti"]}})
    assert m["ctx"]["embeddings"] == (("old_style", str(path)),)
    opts = pr.native_options("image", family="sd1")["embeddings"]
    assert any(o["name"] == "old-civitai-ti" and o["fits"] for o in opts)
    assert not any(c.name == "old-civitai-ti" and c.info.kind == "llm"
                   for c in pr.registry_candidates())


def test_the_same_embeddings_in_another_order_keep_the_load_key(addon_set, tmp_path):
    _register("second", write_safetensors(tmp_path / "Second.safetensors", TI_SD1),
              "textual-inversion")
    a = image_native.resolve_models({"native": {"model": "sd15",
                                                "embeddings": ["EasyNeg", "second"]}})
    b = image_native.resolve_models({"native": {"model": "sd15",
                                                "embeddings": ["second", "EasyNeg"]}})
    assert a["key"] == b["key"]


def test_native_models_route_lists_controlnets_embeddings_and_capabilities(media_app,  # noqa: F811
                                                                         addon_set):
    from fastapi.testclient import TestClient
    with TestClient(media_app) as c:
        data = c.get("/api/imagine/native-models", params={"model": "sd15"}).json()
        flux = c.get("/api/imagine/native-models", params={"model": "fluxunchained"}).json()
    cns = data["controlnets"]
    assert cns[0]["fits"] is True and {c["name"] for c in cns if c["fits"]} == {
        "canny15", "old-scan-cn"}
    assert next(c for c in cns if c["name"] == "canny-xl")["fits"] is False
    emb = {e["name"]: e for e in data["embeddings"]}
    assert emb["EasyNeg"]["trigger"] == "EasyNegative" and emb["EasyNeg"]["fits"] is True
    assert emb["xl-emb"]["fits"] is False
    assert data["preview"]["capabilities"]["controlnet"] is True
    assert flux["preview"]["capabilities"]["references"] is True
    assert all(c["fits"] is False for c in flux["controlnets"])
    shown = json.dumps(data).replace("\\\\", "/")
    assert str(Path(addon_set["canny15"]).parent).replace("\\", "/") not in shown


def test_a_comfyui_job_refuses_native_image_inputs(media_app, addon_set, runner, monkeypatch):  # noqa: F811
    from fastapi.testclient import TestClient

    from localm.config import update_config
    update_config(lambda cfg: cfg["plugins"]["image"].update({"backend": "comfy"}))
    with TestClient(media_app) as c:
        r = c.post("/api/imagine", json={"prompt": "p", "embeddings": ["EasyNeg"]})
        assert r.status_code == 200, r.text
        job = _finish(media_app, r.json()["job_id"])
    lines = [e.get("text", "") for e in job._history if e.get("type") == "line"]
    assert any("apply to the native backend" in line for line in lines), lines
    assert job.status != "done"


# --------------------------------------------------------------------------- #
#  the Qwen LLM vision encoder (reference images for Qwen-Image-Edit)          #
# --------------------------------------------------------------------------- #

QWEN_MMPROJ = {"v.blk.0.attn_q.weight": (1280, 1280), "v.patch_embd.weight": (1280, 3, 14, 14),
               "mm.0.weight": (5120, 5120), "mm.2.weight": (3584, 5120)}
OTHER_MMPROJ = dict(QWEN_MMPROJ, **{"mm.2.weight": (2560, 5120)})
QWEN25VL_GGUF = {"token_embd.weight": (152064, 3584), "blk.0.attn_q.weight": (3584, 3584)}
QWEN_VAE = {"decoder.head.0.gamma": (96, 1, 1, 1), "decoder.upsamples.0.residual.0.gamma": (384,),
            "encoder.downsamples.0.residual.0.gamma": (96,), "conv2.weight": (16, 16, 1, 1, 1)}


@pytest.fixture
def qwen_set(addon_set, tmp_path):
    d = tmp_path / "qwen"
    d.mkdir()
    files = {
        "qwen-edit": (write_safetensors(d / "qwen_image_edit.safetensors", QWEN_IMAGE),
                      "diffusion-unet"),
        "qwen25vl": (write_gguf(d / "Qwen2.5-VL-7B.gguf", QWEN25VL_GGUF, "qwen2vl"),
                     "text-encoder"),
        "qwen_vae": (write_safetensors(d / "qwen_image_vae.safetensors", QWEN_VAE), "vae"),
    }
    for name, (path, mtype) in files.items():
        _register(name, path, mtype)
    return {**addon_set, **{k: v[0] for k, v in files.items()}, "dir": d}


def test_an_llm_mmproj_reads_as_a_vision_encoder_with_its_llm_width(tmp_path):
    info = families.inspect_file(write_gguf(tmp_path / "mmproj.gguf", QWEN_MMPROJ, "clip"))
    assert (info.kind, info.detail.get("hidden")) == ("llm_vision", 3584)
    assert families.accepts("llm_vision", "qwen-image", info)
    other = families.inspect_file(write_gguf(tmp_path / "o.gguf", OTHER_MMPROJ, "clip"))
    assert not families.accepts("llm_vision", "qwen-image", other)


def test_the_qwen_vision_encoder_is_optional_and_picked_when_registered(qwen_set):
    without = image_native.resolve_models({"native": {"model": "qwen-edit"}})
    assert "llm_vision_path" not in without["ctx"]
    prev = pairing.preview("image", "qwen-edit")
    assert "llm_vision" not in prev["needs"] and prev["optional"] == ["llm_vision"]
    assert prev["error"] is None and prev["missing"] == []
    _register("other-mmproj", write_gguf(qwen_set["dir"] / "o.gguf", OTHER_MMPROJ, "clip"),
              "mmproj")
    assert "llm_vision_path" not in image_native.resolve_models(
        {"native": {"model": "qwen-edit"}})["ctx"]
    mm = write_gguf(qwen_set["dir"] / "mmproj-qwen.gguf", QWEN_MMPROJ, "clip")
    _register("qwen-mmproj", mm, "mmproj")
    got = image_native.resolve_models({"native": {"model": "qwen-edit"}})
    assert got["ctx"]["llm_vision_path"] == str(mm)
    assert got["components"]["llm_vision"] == {"name": "qwen-mmproj", "auto": True}


def test_qwen_reference_images_without_a_vision_encoder_say_so(qwen_set, runner, tmp_path):  # noqa: F811
    ok, msg, lines, _ = _gen(runner, tmp_path, {"model": "qwen-edit"}, width=64, height=64,
                             ref_images=[_png()])
    assert ok, msg
    assert image_native.NO_LLM_VISION in lines
    _register("qwen-mmproj", write_gguf(qwen_set["dir"] / "mm.gguf", QWEN_MMPROJ, "clip"),
              "mmproj")
    ok, msg, lines, _ = _gen(runner, tmp_path, {"model": "qwen-edit"}, width=64, height=64,
                             ref_images=[_png()])
    assert ok, msg
    assert image_native.NO_LLM_VISION not in lines
    assert runner.loads[-1]["llm_vision_path"].endswith("mm.gguf")


def test_a_flux_structural_control_model_is_refused_reference_images_before_the_load(
        addon_set, runner, tmp_path):  # noqa: F811
    ok, msg, _, _ = _gen(runner, tmp_path, {"model": "flux-canny"}, width=64, height=64,
                         ref_images=[_png()])
    assert not ok and "does not take reference images" in msg
    assert runner.loads == [] and runner.generates == []
    ok, msg, _, _ = _gen(runner, tmp_path, {"model": "flux-canny"}, width=64, height=64)
    assert ok, msg



def test_clip_only_models_get_non_ascii_letters_lowercased_like_clip(addon_set, runner,  # noqa: F811
                                                                      tmp_path):
    d = tmp_path / "Zusätze Öß"
    d.mkdir()
    _register("oel-stil", write_safetensors(d / "Öl-Stil.safetensors", TI_SD1),
              "textual-inversion")
    ok, msg, lines, rec = _gen(runner, tmp_path, {"model": "sd15", "embeddings": ["oel-stil"]},
                               prompt="Öl-Stil, Ein großer ÖLBAUM über dem Fluß",
                               negative_prompt="ÄRGER", width=64, height=64)
    assert ok, msg
    params = runner.generates[-1]
    assert params["prompt"] == "öl-Stil, Ein großer öLBAUM über dem Fluß"
    assert params["negative_prompt"] == "äRGER"
    assert runner.loads[-1]["embeddings"] == (("öl-Stil", str(d / "Öl-Stil.safetensors")),)
    assert rec["prompt"] == "Öl-Stil, Ein großer ÖLBAUM über dem Fluß"
    assert not any("no effect" in line for line in lines)


def test_a_model_with_a_case_sensitive_encoder_gets_the_prompt_unchanged(addon_set, runner,  # noqa: F811
                                                                         tmp_path):
    ok, msg, _, _ = _gen(runner, tmp_path, {"model": "fluxunchained"},
                         prompt="Ein großer ÖLBAUM", width=64, height=64)
    assert ok, msg
    assert runner.generates[-1]["prompt"] == "Ein großer ÖLBAUM"
