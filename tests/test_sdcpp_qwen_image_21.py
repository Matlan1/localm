# SPDX-License-Identifier: AGPL-3.0-or-later
"""Qwen-Image 2.1 in the native image backend: detection from the header,
component pairing, the refusal of an image model of unknown family before
stable-diffusion.cpp, and the refusal of a LoRA made for another family.

Synthetic headers stand in for the real files: the tensor names and shapes
are those of the published Qwen-Image 2.1 GGUF, VAE, Qwen3-VL-8B text encoder,
its mmproj and the Qwen-Image 2.1 turbo LoRA."""

from __future__ import annotations

import json

import pytest

from localm.media.sdcpp import families, pairing
from localm.plugins.builtin.image.backends import native as image_native
from tests import test_sdcpp_families as _base
from tests.test_sdcpp_families import (FLUX_LORA, QWEN_IMAGE, QWEN_IMAGE_LORA, WAN21_VAE,
                                       _register, write_gguf, write_safetensors)

_reference_box = _base._reference_box
runner = _base.runner

QWEN21 = {
    "img_in.weight": (4096, 64),
    "modulation.1.weight": (16384, 4096),
    "time_text_embed.timestep_embedder.linear_1.weight": (4096, 256),
    "transformer_blocks.0.attn.norm_k.weight": (128,),
    "transformer_blocks.0.attn.to_q.weight": (4096, 4096),
    "transformer_blocks.0.img_mlp.gate_layer.weight": (12288, 4096),
    "transformer_blocks.0.img_mlp.proj.weight": (12288, 4096),
    "transformer_blocks.0.img_mlp.out.weight": (4096, 12288),
    "txt_in.in_layer.weight": (4096, 4096),
    "txt_in.out_layer.weight": (4096, 4096),
    "txt_in.text_norm.weight": (4096,),
    "norm_out.linear.weight": (4096, 4096),
    "proj_out.weight": (64, 4096),
}
QWEN21_VAE = {"conv1.weight": (128, 128, 1, 1, 1), "conv2.weight": (64, 64, 1, 1, 1),
              "decoder.conv1.weight": (1152, 64, 1, 3, 3),
              "decoder.head.2.weight": (4, 144, 1, 3, 3),
              "decoder.upsamples.0.residual.0.gamma": (1152, 1, 1, 1),
              "encoder.conv1.weight": (96, 4, 1, 3, 3),
              "encoder.downsamples.0.residual.0.gamma": (96, 1, 1, 1)}
QWEN3_VL = {"token_embd.weight": (151936, 4096), "blk.0.attn_k_norm.weight": (128,),
            "blk.0.attn_k.weight": (1024, 4096)}
QWEN25_VL = {"token_embd.weight": (152064, 3584), "blk.0.attn_k.weight": (512, 3584)}
QWEN3_VL_MMPROJ = {"v.blk.0.attn_qkv.weight": (3456, 1152), "mm.0.weight": (4608, 4608),
                   "mm.2.weight": (4096, 4608), "v.deepstack.8.fc1.weight": (4608, 4608)}
QWEN35_MMPROJ = {k: v for k, v in QWEN3_VL_MMPROJ.items() if not k.startswith("v.deepstack")}
QWEN21_TURBO_LORA = {
    "diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_down.weight": (178, 4096),
    "diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_up.weight": (24576, 178),
    "diffusion_model.transformer_blocks.0.attn.to_q.lora_down.weight": (178, 4096),
    "diffusion_model.txt_in.in_layer.lora_down.weight": (178, 4096),
}
QWEN21_ATTN_ONLY_LORA = {
    "diffusion_model.transformer_blocks.0.attn.to_q.lora_down.weight": (16, 4096),
    "diffusion_model.transformer_blocks.0.attn.to_q.lora_up.weight": (4096, 16),
}


def test_the_published_gguf_reads_as_qwen_image_21(tmp_path):
    info = families.inspect_file(write_gguf(tmp_path / "q.gguf", QWEN21, "qwen_image21"))
    assert (info.kind, info.family, info.detail) == ("diffusion", "qwen-image-2.1", {"latent": 64})
    assert families.FAMILY_LABELS[info.family] == "Qwen-Image 2.1"
    assert info.family in families.IMAGE_FAMILIES


def test_detection_is_by_tensors_not_the_file_name_nor_only_the_architecture(tmp_path):
    named_old = write_safetensors(tmp_path / "qwen-image-Q8_0.safetensors",
                                  {"model.diffusion_model." + k: v for k, v in QWEN21.items()})
    assert families.inspect_file(named_old).family == "qwen-image-2.1"
    old = write_gguf(tmp_path / "qwen-image-2.1-really-old.gguf", QWEN_IMAGE, "qwen_image")
    assert families.inspect_file(old).family == "qwen-image"
    no_tensors_matched = write_gguf(tmp_path / "x.gguf", {"foo.weight": (8, 8)}, "qwen_image21")
    assert families.inspect_file(no_tensors_matched).family == "qwen-image-2.1"


@pytest.mark.parametrize("arch", ["qwen_image21", None])
def test_the_gguf_registers_as_an_image_model_not_a_chat_model(tmp_path, arch):
    from localm.model_manager.registry import _detect_local_model_type
    f = write_gguf(tmp_path / "qwen-image-2.1-UC-Q6_K.gguf", QWEN21, arch)
    assert _detect_local_model_type(f, is_gguf=True, is_hf=False)[0] == "diffusion-unet"


def test_a_qwen_image_gguf_with_its_architecture_registers_as_an_image_model(tmp_path):
    from localm.model_manager.registry import _detect_local_model_type
    f = write_gguf(tmp_path / "Qwen_Image-Q2_K.gguf", QWEN_IMAGE, "qwen_image")
    assert _detect_local_model_type(f, is_gguf=True, is_hf=False)[0] == "diffusion-unet"
    assert families.inspect_file(f).family == "qwen-image"


def test_qwen_image_21_needs_qwen3_vl_and_its_own_vae_and_takes_reference_images(tmp_path):
    info = families.inspect_file(write_gguf(tmp_path / "q.gguf", QWEN21, "qwen_image21"))
    reqs = families.requirements(info)
    assert [(r.slot, r.required) for r in reqs] == [("llm", True), ("vae", True),
                                                   ("llm_vision", False)]
    caps = families.capabilities(info)
    assert caps["references"] and not caps["controlnet"] and not caps["embeddings"]
    assert families.defaults_for(info) == families.Defaults(40, 6.0, 2048, 2048)


def test_components_are_classified_from_their_headers(tmp_path):
    vae = families.inspect_file(write_safetensors(tmp_path / "v.safetensors", QWEN21_VAE))
    assert (vae.kind, vae.family, vae.detail) == ("vae", "qwen-image-2.1",
                                                  {"vae_dims": "3d", "latent": 64})
    te = families.inspect_file(write_gguf(tmp_path / "te.gguf", QWEN3_VL, "qwen3vl"))
    assert (te.kind, te.detail) == ("llm", {"arch": "qwen3vl", "hidden": 4096})
    mm = families.inspect_file(write_gguf(tmp_path / "mm.gguf", QWEN3_VL_MMPROJ, "clip"))
    assert (mm.kind, mm.detail) == ("llm_vision", {"hidden": 4096, "deepstack": True})


@pytest.fixture
def qwen_sets(tmp_path):
    """A registry with Qwen-Image 2.1 and every Qwen-Image (2.5-VL, 16-channel)
    part, the 2.1 parts registered under misleading names."""
    d = tmp_path / "models"
    d.mkdir()
    files = {
        "qwen-image-2.1-UC-Q6_K": (write_gguf(d / "qwen-image-2.1-UC-Q6_K.gguf", QWEN21,
                                              "qwen_image21"), "diffusion-unet"),
        "qwen_image_vae": (write_safetensors(d / "qwen_image_vae.safetensors", WAN21_VAE),
                           "vae"),
        "Qwen2.5-VL-7B-Instruct-Q4_K_M": (write_gguf(d / "Qwen2.5-VL-7B.gguf", QWEN25_VL,
                                                     "qwen2vl"), "text-encoder"),
        "mmproj-qwen35": (write_gguf(d / "mmproj-F16.gguf", QWEN35_MMPROJ, "clip"), "mmproj"),
        "a-vae": (write_safetensors(d / "a.safetensors", QWEN21_VAE), "vae"),
        "b-encoder": (write_gguf(d / "b.gguf", QWEN3_VL, "qwen3vl"), "text-encoder"),
        "aidma-flux": (write_safetensors(d / "aidmaNSFWunlock-FLUX-V0.2.safetensors", FLUX_LORA),
                       "lora"),
        "qwen-image-lora": (write_safetensors(d / "qi_lora.safetensors", QWEN_IMAGE_LORA),
                            "lora"),
        "qwen21-turbo": (write_safetensors(d / "turbo.safetensors", QWEN21_TURBO_LORA), "lora"),
        "qwen21-attn": (write_safetensors(d / "attn.safetensors", QWEN21_ATTN_ONLY_LORA), "lora"),
    }
    for name, (path, mtype) in files.items():
        _register(name, path, mtype)
    return {k: v[0] for k, v in files.items()}


def _settings(**native):
    return {"backend": "native", "native": {"model": "qwen-image-2.1-UC-Q6_K", **native}}


def test_pairing_picks_qwen3_vl_and_the_21_vae_never_the_qwen_image_parts(qwen_sets):
    models = image_native.resolve_models(_settings())
    ctx = models["ctx"]
    assert ctx["diffusion_model_path"] == str(qwen_sets["qwen-image-2.1-UC-Q6_K"])
    assert ctx["llm_path"] == str(qwen_sets["b-encoder"])
    assert ctx["vae_path"] == str(qwen_sets["a-vae"])
    assert "llm_vision_path" not in ctx
    assert "model_path" not in ctx
    assert models["family"] == "qwen-image-2.1"


def test_the_qwen3_vl_mmproj_is_picked_and_a_qwen35_one_never(qwen_sets, tmp_path):
    assert "llm_vision_path" not in image_native.resolve_models(_settings())["ctx"]
    mm = write_gguf(tmp_path / "models" / "zz-mm.gguf", QWEN3_VL_MMPROJ, "clip")
    _register("zz-qwen3vl-mmproj", mm, "mmproj")
    assert image_native.resolve_models(_settings())["ctx"]["llm_vision_path"] == str(mm)


def test_without_its_parts_the_downloads_named_are_the_21_ones(qwen_sets):
    from localm.config import REGISTRY_FILE
    reg = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    for name in ("a-vae", "b-encoder"):
        del reg[name]
    REGISTRY_FILE.write_text(json.dumps(reg), encoding="utf-8")
    with pytest.raises(pairing.MissingComponents) as caught:
        image_native.resolve_models(_settings())
    got = {m["slot"]: m["download"] for m in caught.value.missing}
    assert got["llm"]["spec"] == "Qwen/Qwen3-VL-8B-Instruct-GGUF:Qwen3VL-8B-Instruct-Q4_K_M.gguf"
    assert got["vae"]["spec"] == ("Comfy-Org/Qwen-Image-2.1:"
                                  "vae/qwen_image_2.1_vae_bf16.safetensors")
    assert all(len(d["sha256"]) == 64 for d in got.values())
    assert "Qwen2.5" not in str(caught.value) and "qwen_image_vae" not in str(caught.value)


def test_curated_downloads_classify_as_the_slot_they_fill():
    dl = {d.spec: d for d in pairing.all_downloads()}
    assert pairing.QWEN3_VL_8B_MMPROJ.spec in dl
    assert pairing.suggested_download("llm_vision", "qwen-image-2.1").model_type == "mmproj"
    assert pairing.suggested_download("llm", "qwen-image").spec.startswith("unsloth/Qwen2.5-VL")


def test_an_explicit_qwen_image_encoder_is_kept_but_noted(qwen_sets):
    models = image_native.resolve_models(_settings(llm="Qwen2.5-VL-7B-Instruct-Q4_K_M"))
    assert any("does not look like one Qwen-Image 2.1 models use" in n for n in models["notes"])


# ------------------------------------------------------ unknown families --

UNKNOWN_DIT = {"img_in.weight": (3072, 128), "blocks.0.attn.qkv.weight": (9216, 3072)}


@pytest.mark.parametrize("arch, shapes, shown", [
    ("qwen_image99", UNKNOWN_DIT, "GGUF architecture 'qwen_image99' and 128 input channels"),
    (None, UNKNOWN_DIT, "its header shows 128 input channels"),
    ("wan", {"foo.weight": (8, 8)}, "GGUF architecture 'wan'"),
    (None, {"foo.weight": (8, 8)}, "no model layers localm knows"),
])
def test_an_image_model_of_unknown_family_is_refused_before_any_load(
        tmp_path, runner, arch, shapes, shown):
    path = write_gguf(tmp_path / "mystery.gguf", shapes, arch)
    _register("mystery", path, "diffusion-unet")
    clip = write_gguf(tmp_path / "te.gguf", QWEN3_VL, "qwen3vl")
    _register("te", clip, "text-encoder")
    for settings in ({"model": "mystery"}, {"model": "mystery", "llm": "te"}):
        ok, msg = image_native.generate({"backend": "native", "native": settings}, "a cat",
                                        tmp_path / "o.png", width=64, height=64)
        assert not ok
        assert msg.startswith("'mystery' is not an image model localm can identify (")
        assert shown in msg and "is not passed to stable-diffusion.cpp" in msg
        assert msg.count(". ") == 0 and msg.endswith("backend.")
    assert runner.loads == [] and runner.generates == []
    ok, msg = image_native.ensure_available({"backend": "native",
                                             "native": {"model": "mystery"}})
    assert not ok and "localm can identify" in msg


def test_an_unidentified_file_of_a_format_localm_does_not_read_keeps_the_old_rule(tmp_path):
    ckpt = tmp_path / "v1-5.ckpt"
    ckpt.write_bytes(b"PK\x03\x04" + b"\0" * 64)
    built = pairing.build_context(ckpt, "v1-5.ckpt", {}, where="Settings > Images",
                                  media="image")
    assert built["ctx"] == {"model_path": str(ckpt)}


def test_the_preview_shows_the_unknown_family_reason(tmp_path):
    path = write_gguf(tmp_path / "mystery.gguf", UNKNOWN_DIT, "qwen_image99")
    _register("mystery", path, "diffusion-unet")
    out = pairing.preview("image", "mystery")
    assert out["error"].startswith("'mystery' is not an image model localm can identify")


# ----------------------------------------------------------------- LoRAs --

def test_lora_families_tell_qwen_image_21_from_qwen_image(qwen_sets):
    fam = {n: families.inspect_file(qwen_sets[n]).family
           for n in ("qwen21-turbo", "qwen-image-lora", "aidma-flux", "qwen21-attn")}
    assert fam == {"qwen21-turbo": "qwen-image-2.1", "qwen-image-lora": "qwen-image",
                   "aidma-flux": "flux", "qwen21-attn": None}


@pytest.mark.parametrize("lora, made_for", [("aidma-flux", "FLUX.1"),
                                            ("qwen-image-lora", "Qwen-Image")])
def test_a_lora_of_another_family_is_refused_before_any_load(qwen_sets, runner, tmp_path,
                                                             lora, made_for):
    ok, msg = image_native.generate(_settings(), "a cat", tmp_path / "o.png", width=64,
                                    height=64, lora_name=lora)
    assert not ok
    assert msg == (f"The LoRA '{lora}' is made for {made_for}, not Qwen-Image 2.1, so it is "
                   "not applied. Remove it, or choose a Qwen-Image 2.1 LoRA.")
    assert runner.loads == []


def test_a_qwen_image_21_lora_is_applied_without_a_note(qwen_sets, runner, tmp_path):
    lines = []
    ok, msg = image_native.generate(_settings(), "a cat", tmp_path / "o.png", width=64,
                                    height=64, lora_name="qwen21-turbo",
                                    on_progress=lines.append)
    assert ok, msg
    assert runner.generates[0]["loras"] == [{"path": str(qwen_sets["qwen21-turbo"]),
                                             "multiplier": 1.0}]
    assert not any("could not be checked" in line for line in lines)
    side = json.loads((tmp_path / "o.png.json").read_text(encoding="utf-8"))
    assert side["loras"] == [{"name": "qwen21-turbo", "strength": 1.0}]


def test_a_lora_of_unknown_family_is_applied_and_says_it_could_not_be_checked(
        qwen_sets, runner, tmp_path):
    lines = []
    ok, msg = image_native.generate(_settings(), "a cat", tmp_path / "o.png", width=64,
                                    height=64, lora_name="qwen21-attn", on_progress=lines.append)
    assert ok, msg
    assert runner.generates[0]["loras"][0]["path"] == str(qwen_sets["qwen21-attn"])
    assert ("The LoRA 'qwen21-attn' could not be checked: localm cannot tell from its file "
            "whether it suits Qwen-Image 2.1.") in lines
    side = json.loads((tmp_path / "o.png.json").read_text(encoding="utf-8"))
    assert side["loras"] == [{"name": "qwen21-attn", "strength": 1.0, "family_checked": False}]


# ------------------------------------------------------- reference images --

def test_reference_images_need_the_qwen3_vl_vision_encoder(qwen_sets, runner, tmp_path):
    from io import BytesIO

    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (64, 64), (200, 10, 10)).save(buf, "PNG")
    ref = buf.getvalue()
    ok, msg = image_native.generate(_settings(), "make it blue", tmp_path / "o.png", width=64,
                                    height=64, ref_images=[ref])
    assert not ok and msg == image_native.NO_LLM_VISION_21
    assert "mmproj-Qwen3VL-8B-Instruct-F16.gguf --type mmproj --sha256 " in msg
    assert runner.loads == []
    assert image_native.input_refusal(_settings(), ref_images=[ref]) == msg
    mm = write_gguf(tmp_path / "models" / "zz-mm.gguf", QWEN3_VL_MMPROJ, "clip")
    _register("zz-qwen3vl-mmproj", mm, "mmproj")
    ok, msg = image_native.generate(_settings(), "make it blue", tmp_path / "o2.png", width=64,
                                    height=64, ref_images=[ref])
    assert ok, msg
    assert runner.loads[-1]["llm_vision_path"] == str(mm)
    assert len(runner.generates[-1]["ref_images"]) == 1
