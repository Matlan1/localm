# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native media model files: classification from headers
(``localm.media.sdcpp.families``), companion pairing from the registry and
LoRA resolution (``localm.media.sdcpp.pairing``), and how the native image and
video backends use them. Files are synthetic headers laid out like the
published checkpoints (tensor names and shapes only)."""

from __future__ import annotations

import ctypes
import json
import struct
from pathlib import Path

import pytest

from localm.media.sdcpp import families, pairing
from localm.media.sdcpp.files import ModelFileError
from localm.plugins.builtin.image.backends import native as image_native
from localm.plugins.builtin.video.backends import native as video_native


@pytest.fixture(autouse=True)
def _reference_box(monkeypatch):
    from tests.test_sdcpp_gpu_reset_limit import pin_rx6900xt
    pin_rx6900xt(monkeypatch)


def write_safetensors(path: Path, shapes: dict) -> Path:
    header = {name: {"dtype": "F16", "shape": list(shape), "data_offsets": [0, 0]}
              for name, shape in shapes.items()}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 16)
    return path


def _gguf_string(s: str) -> bytes:
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path: Path, shapes: dict, arch=None) -> Path:
    """A GGUF whose tensors have *shapes* (PyTorch order; stored as ggml ne)."""
    kvs, n_kv = b"", 0
    if arch:
        kvs += _gguf_string("general.architecture") + struct.pack("<I", 8) + _gguf_string(arch)
        n_kv = 1
    infos = b""
    for i, (name, shape) in enumerate(shapes.items()):
        ne = tuple(reversed(shape))
        infos += _gguf_string(name) + struct.pack("<I", len(ne))
        infos += struct.pack(f"<{len(ne)}Q", *ne) + struct.pack("<I", 0)
        infos += struct.pack("<Q", i * 32)
    body = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(shapes), n_kv) + kvs + infos
    body += b"\0" * ((32 - len(body) % 32) % 32) + b"\0" * 64
    path.write_bytes(body)
    return path


FLUX_DEV = {"double_blocks.0.img_attn.qkv.weight": (9216, 3072),
            "single_blocks.0.linear1.weight": (21504, 3072),
            "guidance_in.in_layer.weight": (3072, 256), "img_in.weight": (3072, 64)}
FLUX_SCHNELL = {k: v for k, v in FLUX_DEV.items() if not k.startswith("guidance_in")}
WAN21_T2V = {"model.diffusion_model.patch_embedding.weight": (1536, 16, 1, 2, 2),
             "model.diffusion_model.blocks.0.cross_attn.k.weight": (1536, 1536),
             "model.diffusion_model.blocks.0.cross_attn.norm_k.weight": (1536,)}
WAN21_I2V = dict(WAN21_T2V, **{"model.diffusion_model.patch_embedding.weight": (5120, 36, 1, 2, 2),
                              "model.diffusion_model.img_emb.proj.1.weight": (5120, 1280)})
WAN22_5B = dict(WAN21_T2V, **{"model.diffusion_model.patch_embedding.weight": (3072, 48, 1, 2, 2)})
SD3 = {"model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight": (4608, 1536)}
SDXL_UNET = {"model.diffusion_model.input_blocks.0.0.weight": (320, 4, 3, 3),
             "model.diffusion_model.label_emb.0.0.weight": (1280, 2816)}
SDXL_UNET_DIFFUSERS = {"conv_in.weight": (320, 4, 3, 3),
                       "time_embedding.linear_1.weight": (1280, 320),
                       "add_embedding.linear_1.weight": (1280, 2816),
                       "down_blocks.0.resnets.0.norm1.weight": (320,)}
SDXL_CKPT = {"model.diffusion_model.input_blocks.0.0.weight": (320, 4, 3, 3),
             "model.diffusion_model.label_emb.0.0.weight": (1280, 2816),
             "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight":
                 (49408, 768),
             "conditioner.embedders.1.model.ln_final.weight": (1280,),
             "first_stage_model.decoder.conv_in.weight": (512, 4, 3, 3)}
QWEN_IMAGE = {"transformer_blocks.0.img_mod.1.weight": (18432, 3072)}
Z_IMAGE = {"cap_embedder.0.weight": (2560,), "layers.0.attention.qkv.weight": (11520, 3840)}
SD1_UNET_DIFFUSERS = {
    "conv_in.weight": (320, 4, 3, 3),
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight": (320, 768),
    "mid_block.attentions.0.norm.weight": (1280,)}
SD2_UNET_DIFFUSERS = dict(SD1_UNET_DIFFUSERS, **{
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight": (320, 1024)})
SD1_UNET = {"model.diffusion_model.input_blocks.0.0.weight": (320, 4, 3, 3),
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight":
                (320, 768)}
DIFFUSERS_UNET_WITHOUT_CROSS_ATTENTION = {
    "conv_in.weight": (320, 4, 3, 3), "down_blocks.0.resnets.0.norm1.weight": (320,)}
SD2_CKPT = {"model.diffusion_model.input_blocks.0.0.weight": (320, 4, 3, 3),
            "first_stage_model.decoder.conv_in.weight": (512, 4, 3, 3),
            "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight":
                (49408, 1024)}
CLIP_L = {"text_model.embeddings.token_embedding.weight": (49408, 768),
          "text_model.encoder.layers.0.layer_norm1.weight": (768,)}
CLIP_L_FULL = dict(CLIP_L, **{"vision_model.encoder.layers.0.mlp.fc1.weight": (4096, 1024),
                              "text_projection": (768, 768)})
CLIP_G = {"text_model.embeddings.token_embedding.weight": (49408, 1280)}
CLIP_H = {"text_model.embeddings.token_embedding.weight": (49408, 1024),
          "text_model.encoder.layers.0.layer_norm1.weight": (1024,)}
CLIP_VISION = {"vision_model.encoder.layers.0.mlp.fc1.weight": (5120, 1280)}
T5XXL = {"shared.weight": (32128, 4096),
         "encoder.block.0.layer.0.SelfAttention.k.weight": (4096, 4096)}
UMT5_GGUF = {"token_embd.weight": (256384, 4096), "enc.blk.0.attn_k.weight": (4096, 4096)}
FLUX_AE = {"decoder.conv_in.weight": (512, 16, 3, 3), "encoder.conv_in.weight": (128, 3, 3, 3)}
SD_VAE = {"decoder.conv_in.weight": (512, 4, 3, 3), "encoder.conv_in.weight": (128, 3, 3, 3)}
WAN21_VAE = {"conv1.weight": (32, 32, 1, 1, 1), "conv2.weight": (16, 16, 1, 1, 1),
             "decoder.head.0.gamma": (96, 1, 1, 1), "decoder.upsamples.0.residual.0.gamma": (384,),
             "encoder.downsamples.0.residual.0.gamma": (96,)}
WAN22_VAE = dict(WAN21_VAE, **{"conv2.weight": (48, 48, 1, 1, 1)})
FLUX_LORA = {"lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": (2, 3072),
             "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight": (3072, 2)}
XLABS_FLUX_LORA = {
    "double_blocks.0.processor.proj_lora1.down.weight": (16, 3072),
    "double_blocks.0.processor.proj_lora1.up.weight": (3072, 16),
    "double_blocks.0.processor.proj_lora2.down.weight": (16, 3072),
    "double_blocks.0.processor.proj_lora2.up.weight": (3072, 16),
    "double_blocks.0.processor.qkv_lora1.down.weight": (16, 3072),
    "double_blocks.0.processor.qkv_lora1.up.weight": (9216, 16)}
SDXL_LORA = {"lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": (4, 768),
             "lora_unet_input_blocks_4_1_proj_in.lora_down.weight": (4, 640)}
SDXL_UNET_LORA_CROSS_ATTN = {
    "lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight": (64, 320, 3, 3),
    "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight":
        (64, 2048)}
SDXL_UNET_LORA_DEPTH = {
    "lora_unet_down_blocks_2_attentions_0_transformer_blocks_2_attn1_to_k.lora_down.weight":
        (64, 1280)}
SD1_UNET_LORA = {
    "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight":
        (4, 768)}
QWEN_IMAGE_LORA = {"transformer_blocks.0.attn.add_k_proj.lora_down.weight": (64, 3072),
                   "transformer_blocks.0.img_mlp.net.0.proj.lora_down.weight": (64, 3072),
                   "transformer_blocks.0.txt_mlp.net.0.proj.lora_down.weight": (64, 3072)}
Z_IMAGE_LORA = {"diffusion_model.layers.0.attention.to_k.lora_A.weight": (32, 3840),
                "diffusion_model.layers.0.feed_forward.w1.lora_A.weight": (32, 3840),
                "diffusion_model.layers.0.adaLN_modulation.0.lora_A.weight": (32, 256)}
WAN22_5B_LORA = {"diffusion_model.blocks.0.cross_attn.k.lora_down.weight": (128, 3072),
                 "diffusion_model.blocks.0.ffn.0.lora_down.weight": (128, 3072)}


@pytest.mark.parametrize("shapes,fmt,arch,kind,family", [
    (FLUX_DEV, "gguf", "flux", "diffusion", "flux"),
    (FLUX_DEV, "st", None, "diffusion", "flux"),
    (WAN21_T2V, "st", None, "diffusion", "wan2.1"),
    (WAN22_5B, "st", None, "diffusion", "wan2.2"),
    (SD3, "st", None, "diffusion", "sd3"),
    (SDXL_UNET, "st", None, "diffusion", "sdxl"),
    (SDXL_UNET_DIFFUSERS, "gguf", "stable-diffusion-xl-unet", "diffusion", "sdxl"),
    (SD1_UNET_DIFFUSERS, "gguf", "stable-diffusion-unet", "diffusion", "sd1"),
    (SDXL_UNET_DIFFUSERS, "st", None, "diffusion", "sdxl"),
    (SDXL_CKPT, "st", None, "checkpoint", "sdxl"),
    (QWEN_IMAGE, "st", None, "diffusion", "qwen-image"),
    (Z_IMAGE, "st", None, "diffusion", "z-image"),
    (SD2_CKPT, "gguf", None, "checkpoint", "sd2"),
    (SD1_UNET_DIFFUSERS, "st", None, "diffusion", "sd1"),
    (SD2_UNET_DIFFUSERS, "st", None, "diffusion", "sd2"),
    (SD1_UNET, "st", None, "diffusion", "sd1"),
    (DIFFUSERS_UNET_WITHOUT_CROSS_ATTENTION, "st", None, "unknown", None),
    (CLIP_H, "st", None, "clip_h", None),
    (CLIP_L, "st", None, "clip_l", None),
    (CLIP_L_FULL, "st", None, "clip_l", None),
    (CLIP_G, "st", None, "clip_g", None),
    (CLIP_VISION, "st", None, "clip_vision", None),
    (T5XXL, "st", None, "t5xxl", None),
    (UMT5_GGUF, "gguf", "t5encoder", "umt5", None),
    (FLUX_AE, "st", None, "vae", "flux"),
    (SD_VAE, "st", None, "vae", "sd"),
    (WAN21_VAE, "st", None, "vae", "wan2.1"),
    (WAN22_VAE, "st", None, "vae", "wan2.2"),
    (FLUX_LORA, "st", None, "lora", "flux"),
    (XLABS_FLUX_LORA, "st", None, "lora", "flux"),
    (SDXL_LORA, "st", None, "lora", "sdxl"),
    (SDXL_UNET_LORA_CROSS_ATTN, "st", None, "lora", "sdxl"),
    (SDXL_UNET_LORA_DEPTH, "st", None, "lora", "sdxl"),
    (SD1_UNET_LORA, "st", None, "lora", "sd1"),
    (QWEN_IMAGE_LORA, "st", None, "lora", "qwen-image"),
    (Z_IMAGE_LORA, "st", None, "lora", "z-image"),
    (WAN22_5B_LORA, "st", None, "lora", "wan"),
])
def test_files_are_classified_from_their_headers(tmp_path, shapes, fmt, arch, kind, family):
    if fmt == "gguf":
        f = write_gguf(tmp_path / "m.gguf", shapes, arch)
    else:
        f = write_safetensors(tmp_path / "m.safetensors", shapes)
    info = families.inspect_file(f)
    assert (info.kind, info.family) == (kind, family)


def test_llm_text_encoders_are_known_by_gguf_architecture(tmp_path):
    f = write_gguf(tmp_path / "q.gguf", {"token_embd.weight": (151936, 2560)}, "qwen3")
    info = families.inspect_file(f)
    assert info.kind == "llm" and info.detail["arch"] == "qwen3"


@pytest.mark.parametrize("data", [b"", b"GGUF", b"\x00" * 3, b"\xff" * 64,
                                  struct.pack("<Q", 10 ** 12) + b"{}"])
def test_unreadable_headers_are_unknown_and_never_raise(tmp_path, data):
    for suffix in (".gguf", ".safetensors"):
        f = tmp_path / f"bad{suffix}"
        f.write_bytes(data)
        assert families.inspect_file(f) is families.UNKNOWN
    assert families.inspect_file(tmp_path / "absent.gguf") is families.UNKNOWN


def test_a_deeply_nested_safetensors_header_is_unreadable_not_a_crash(tmp_path):
    raw = b"[" * 200_000 + b"]" * 200_000
    f = tmp_path / "nested.safetensors"
    f.write_bytes(struct.pack("<Q", len(raw)) + raw)
    assert families._read_safetensors(f) is None
    assert families.read_tensors(f) is None
    assert families.inspect_file(f) is families.UNKNOWN


def test_a_rewritten_file_is_classified_again(tmp_path):
    f = write_safetensors(tmp_path / "x.safetensors", CLIP_L)
    assert families.inspect_file(f).kind == "clip_l"
    write_safetensors(f, dict(FLUX_AE, pad=(1, 2, 3)))
    assert families.inspect_file(f).kind == "vae"


@pytest.mark.parametrize("shapes,slots", [
    (FLUX_DEV, ["clip_l", "t5xxl", "vae"]),
    (SD3, ["clip_l", "clip_g", "t5xxl", "vae"]),
    (WAN21_T2V, ["t5xxl", "vae"]),
    (WAN21_I2V, ["t5xxl", "vae", "clip_vision"]),
    (QWEN_IMAGE, ["llm", "vae"]),
])
def test_each_family_needs_its_companions(tmp_path, shapes, slots):
    info = families.inspect_file(write_safetensors(tmp_path / "m.safetensors", shapes))
    assert [r.slot for r in families.requirements(info) if r.required] == slots


def test_flux_schnell_defaults_to_four_steps_and_dev_to_more(tmp_path):
    dev = families.inspect_file(write_gguf(tmp_path / "d.gguf", FLUX_DEV, "flux"))
    schnell = families.inspect_file(write_gguf(tmp_path / "s.gguf", FLUX_SCHNELL, "flux"))
    assert families.defaults_for(schnell).steps == 4
    assert families.defaults_for(dev).steps > 4
    assert families.defaults_for(dev).cfg_scale == 1.0


# --------------------------------------------------------------------------- #
#  registry pairing                                                           #
# --------------------------------------------------------------------------- #

def _register(name: str, path: Path, model_type: str, **extra) -> None:
    from localm.config import REGISTRY_FILE
    reg = json.loads(REGISTRY_FILE.read_text(encoding="utf-8")) if REGISTRY_FILE.exists() else {}
    reg[name] = {"path": str(path), "model_type": model_type, "source": "test", **extra}
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(reg), encoding="utf-8")


@pytest.fixture
def flux_set(tmp_path):
    """A registry laid out like a ComfyUI import: a Flux GGUF, its encoders,
    two VAEs of different families, a CLIP-L fine-tune, the Wan set and a
    LoRA."""
    d = tmp_path / "models"
    d.mkdir()
    files = {
        "fluxunchained": (write_gguf(d / "fluxunchained-dev-q8-0-v2.gguf", FLUX_DEV, "flux"),
                          "diffusion-unet"),
        "ViT-L-14-TEXT": (write_safetensors(d / "ViT-L-14-TEXT.safetensors", CLIP_L_FULL),
                          "text-encoder"),
        "clip_l": (write_safetensors(d / "clip_l.safetensors", CLIP_L), "text-encoder"),
        "t5xxl_fp8_e4m3fn": (write_safetensors(d / "t5xxl_fp8_e4m3fn.safetensors", T5XXL),
                             "text-encoder"),
        "umt5-xxl-encoder-Q4_K_M": (write_gguf(d / "umt5.gguf", UMT5_GGUF, "t5encoder"),
                                    "text-encoder"),
        "sd-vae": (write_safetensors(d / "sd_vae.safetensors", SD_VAE), "vae"),
        "ae": (write_safetensors(d / "ae.safetensors", FLUX_AE), "vae"),
        "wan_2.1_vae": (write_safetensors(d / "wan_2.1_vae.safetensors", WAN21_VAE), "vae"),
        "wan2.1_t2v": (write_safetensors(d / "wan2.1_t2v_1.3B_fp16.safetensors", WAN21_T2V),
                       "diffusion-unet"),
        "aidma-flux": (write_safetensors(d / "aidmaNSFWunlock-FLUX-V0.2.safetensors", FLUX_LORA),
                       "lora"),
        "sdxl-style": (write_safetensors(d / "sdxl_style.safetensors", SDXL_LORA), "lora"),
        "xlabs-realism": (write_safetensors(d / "xlabs_realism.safetensors", XLABS_FLUX_LORA),
                          "lora"),
    }
    for name, (path, mtype) in files.items():
        _register(name, path, mtype)
    return {k: v[0] for k, v in files.items()}


def test_the_xlabs_flux_lora_is_offered_as_a_lora_not_as_a_flux_model(flux_set):
    from localm.media.sdcpp import pairing
    opts = pairing.native_options("image")
    assert "xlabs-realism" not in {m["name"] for m in opts["models"]}
    assert "xlabs-realism" in {m["name"] for m in opts["loras"]}


def test_a_flux_model_alone_gets_its_encoders_and_vae_from_the_registry(flux_set):
    m = image_native.resolve_models({"native": {"model": "fluxunchained"}})
    assert m["ctx"] == {"diffusion_model_path": str(flux_set["fluxunchained"]),
                        "clip_l_path": str(flux_set["clip_l"]),
                        "t5xxl_path": str(flux_set["t5xxl_fp8_e4m3fn"]),
                        "vae_path": str(flux_set["ae"])}
    assert m["family"] == "flux"
    assert all(c["auto"] for c in m["components"].values())
    assert m["defaults"].cfg_scale == 1.0


def test_an_explicit_component_wins_over_the_automatic_pick(flux_set):
    m = image_native.resolve_models({"native": {"model": "fluxunchained",
                                                "clip_l": "ViT-L-14-TEXT"}})
    assert m["ctx"]["clip_l_path"] == str(flux_set["ViT-L-14-TEXT"])
    assert m["components"]["clip_l"] == {"name": "ViT-L-14-TEXT", "auto": False}
    assert m["components"]["vae"]["auto"] is True


def test_an_explicit_component_of_the_wrong_kind_is_kept_and_noted(flux_set):
    m = image_native.resolve_models({"native": {"model": "fluxunchained", "vae": "sd-vae"}})
    assert m["ctx"]["vae_path"] == str(flux_set["sd-vae"])
    assert any("sd-vae" in n and "VAE" in n for n in m["notes"])


def test_a_missing_companion_names_a_download(flux_set):
    from localm.config import REGISTRY_FILE
    reg = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    del reg["t5xxl_fp8_e4m3fn"]
    REGISTRY_FILE.write_text(json.dumps(reg), encoding="utf-8")
    with pytest.raises(pairing.MissingComponents) as ei:
        image_native.resolve_models({"native": {"model": "fluxunchained"}})
    assert [m["slot"] for m in ei.value.missing] == ["t5xxl"]
    dl = ei.value.missing[0]["download"]
    assert dl["spec"] == "comfyanonymous/flux_text_encoders:t5xxl_fp8_e4m3fn.safetensors"
    assert (f"localm pull {dl['spec']} --type text-encoder --sha256 {dl['sha256']}"
            in str(ei.value))
    st = image_native.status({"native": {"model": "fluxunchained"}})
    assert st["missing_components"][0]["slot"] == "t5xxl" and st["family"] == "flux"


def test_wan_umt5_is_never_given_to_flux_nor_t5xxl_to_wan(flux_set):
    from localm.config import REGISTRY_FILE
    reg = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    del reg["t5xxl_fp8_e4m3fn"]
    REGISTRY_FILE.write_text(json.dumps(reg), encoding="utf-8")
    with pytest.raises(pairing.MissingComponents):
        image_native.resolve_models({"native": {"model": "fluxunchained"}})
    m = video_native.resolve_models({"native": {"model": "wan2.1_t2v"}})
    assert m["ctx"]["t5xxl_path"] == str(flux_set["umt5-xxl-encoder-Q4_K_M"])
    assert m["ctx"]["vae_path"] == str(flux_set["wan_2.1_vae"])
    assert m["ctx"]["diffusion_flash_attn"] is True
    assert m["defaults"].flow_shift == 3.0


def test_a_component_chosen_as_the_model_is_refused_with_a_reason(flux_set):
    with pytest.raises(ModelFileError, match="is a VAE, not an image model"):
        image_native.resolve_models({"native": {"model": "ae"}})


def test_the_vae_pick_prefers_the_family_name(tmp_path):
    a = write_safetensors(tmp_path / "flux_vae_a.safetensors", FLUX_AE)
    b = write_safetensors(tmp_path / "ae.safetensors", FLUX_AE)
    cands = [pairing.Candidate("zz-other", a, families.inspect_file(a), "vae"),
             pairing.Candidate("ae", b, families.inspect_file(b), "vae")]
    assert pairing.fitting("vae", "flux", cands)[0].name == "ae"


def test_wan22_5b_needs_the_wan22_vae(tmp_path):
    v21 = write_safetensors(tmp_path / "wan_2.1_vae.safetensors", WAN21_VAE)
    v22 = write_safetensors(tmp_path / "wan2.2_vae.safetensors", WAN22_VAE)
    cands = [pairing.Candidate(p.stem, p, families.inspect_file(p), "vae") for p in (v21, v22)]
    assert [c.name for c in pairing.fitting("vae", "wan2.2", cands, "5b")] == ["wan2.2_vae"]
    assert [c.name for c in pairing.fitting("vae", "wan2.1", cands)] == ["wan_2.1_vae"]


def test_chat_models_are_llm_candidates_only_with_a_known_architecture(tmp_path):
    q = write_gguf(tmp_path / "q.gguf", {"token_embd.weight": (10, 10)}, "qwen3")
    _register("qwen3-chat", q, "llm", architecture="qwen3")
    _register("mystery-chat", q, "llm")
    names = [c.name for c in pairing.registry_candidates()]
    assert "qwen3-chat" in names and "mystery-chat" not in names


def test_unreadable_and_network_entries_are_not_candidates(tmp_path, monkeypatch):
    _register("gone", tmp_path / "gone.safetensors", "vae")
    _register("unc", r"\\server\share\vae.safetensors", "vae")
    assert pairing.registry_candidates() == []


# --------------------------------------------------------------------------- #
#  LoRAs                                                                      #
# --------------------------------------------------------------------------- #

def test_loras_resolve_by_registry_name_and_by_file_name(flux_set):
    out, notes = pairing.resolve_loras(
        [{"name": "aidma-flux", "strength": 0.8},
         {"name": "aidmaNSFWunlock-FLUX-V0.2.safetensors"}], "flux")
    assert [x["path"] for x in out] == [str(flux_set["aidma-flux"])] * 2
    assert [x["multiplier"] for x in out] == [0.8, 1.0]
    assert notes == []


def test_a_lora_for_another_family_is_refused(flux_set):
    with pytest.raises(ModelFileError, match="made for SDXL, not FLUX.1, so it is not applied"):
        pairing.resolve_loras([{"name": "sdxl-style"}], "flux")


def test_a_flux_style_lora_is_not_noted_on_chroma_nor_a_chroma_lora_on_flux(flux_set):
    out, notes = pairing.resolve_loras([{"name": "aidma-flux"}], "chroma")
    assert len(out) == 1 and notes == []
    out, notes = pairing.resolve_loras([{"name": "aidma-flux"}], "flux")
    assert notes == []
    with pytest.raises(ModelFileError, match="made for FLUX.1, not Qwen-Image,"):
        pairing.resolve_loras([{"name": "aidma-flux"}], "qwen-image")


@pytest.mark.parametrize("name", ["not-registered", "fluxunchained", "../x.safetensors"])
def test_a_name_that_is_not_a_registered_lora_is_refused(flux_set, name):
    with pytest.raises(ModelFileError, match="not a registered LoRA"):
        pairing.resolve_loras([{"name": name}], "flux")


def test_the_worker_receives_lora_structs():
    from localm.media.sdcpp import _binding as b
    from localm.media.sdcpp import _child
    p = b.sd_img_gen_params_t()
    keep = _child._set_loras(b, p, {"loras": [{"path": "C:/x/a.safetensors", "multiplier": 0.5},
                                             {"path": "/y/b.safetensors"}]})
    assert p.lora_count == 2
    assert p.loras[0].path == b"C:/x/a.safetensors" and p.loras[0].multiplier == 0.5
    assert p.loras[1].multiplier == 1.0 and p.loras[1].is_high_noise is False
    assert "a.safetensors" in _child._redacted("loading C:/x/a.safetensors") \
        and "C:/x" not in _child._redacted("loading C:/x/a.safetensors")
    assert keep is not None
    q = b.sd_img_gen_params_t()
    assert _child._set_loras(b, q, {}) is None and q.lora_count == 0


# --------------------------------------------------------------------------- #
#  generate                                                                   #
# --------------------------------------------------------------------------- #

class _Runner:
    def __init__(self):
        self.loads, self.generates, self.alive, self.loaded_key = [], [], False, None
        self.loaded_ctx = None
        self.fill = b"\x10"

    def is_alive(self):
        return self.alive

    def ensure_loaded(self, key, runtime_dir, ctx, **kw):
        self.loads.append(ctx)
        self.loaded_key, self.loaded_ctx, self.alive = key, dict(ctx), True
        return {"version": "Flux", "image": True, "video": True}

    def attention_rate(self, runtime_dir, head_dim=128, timeout=None, devices=None):
        return 1e15

    def generate_image(self, params, **kw):
        self.generates.append(params)
        w, h = params["width"], params["height"]
        return {"width": w, "height": h, "channel": 3, "data": self.fill * (w * h * 3),
                "seed": params["seed"]}

    def shutdown(self, grace=5.0):
        self.alive = False


@pytest.fixture
def runner(monkeypatch, tmp_path):
    from localm.media.sdcpp import runtime as sd_runtime
    from localm.media.sdcpp import shared
    r = _Runner()
    monkeypatch.setattr(shared, "runner", r)
    monkeypatch.setattr(shared, "arm_idle_timer", lambda seconds=None: None)
    rt = sd_runtime.Runtime(backend="vulkan", path=tmp_path / "rt")
    monkeypatch.setattr(sd_runtime, "resolve", lambda choice="auto": rt)
    return r


def test_flux_generates_with_its_family_defaults_loras_and_records_them(flux_set, runner,
                                                                         tmp_path):
    lines = []
    out = tmp_path / "o.png"
    ok, msg = image_native.generate({"backend": "native",
                                     "native": {"model": "fluxunchained"}}, "a cat", out,
                                    width=64, height=64, seed=7, lora_name="aidma-flux",
                                    lora_strength_model=0.7, on_progress=lines.append)
    assert ok, msg
    params = runner.generates[0]
    assert (params["steps"], params["cfg_scale"]) == (20, 1.0)
    assert params["loras"] == [{"path": str(flux_set["aidma-flux"]), "multiplier": 0.7}]
    assert runner.loads[0]["vae_path"] == str(flux_set["ae"])
    assert any(line.startswith("Using ") and "(auto)" in line for line in lines)
    side = json.loads(out.with_suffix(".png.json").read_text(encoding="utf-8"))
    assert side["components"] == {"clip_l": "clip_l", "t5xxl": "t5xxl_fp8_e4m3fn", "vae": "ae"}
    assert side["loras"] == [{"name": "aidma-flux", "strength": 0.7}]


def test_a_pure_white_result_is_reported_as_a_failure_and_not_saved(flux_set, runner, tmp_path):
    runner.fill = b"\xff"
    out = tmp_path / "white.png"
    ok, msg = image_native.generate({"backend": "native", "native": {"model": "fluxunchained"}},
                                    "a cat", out, width=64, height=64, seed=7)
    assert not ok and "pure white" in msg and "VAE" in msg
    assert not out.exists()
    runner.fill = b"\x10"
    ok, msg = image_native.generate({"backend": "native", "native": {"model": "fluxunchained"}},
                                    "a cat", out, width=64, height=64, seed=7)
    assert ok and out.exists()


def test_an_unregistered_lora_fails_before_any_load(flux_set, runner, tmp_path):
    ok, msg = image_native.generate({"backend": "native", "native": {"model": "fluxunchained"}},
                                    "p", tmp_path / "o.png", lora_name="nope")
    assert not ok and "not a registered LoRA" in msg
    assert runner.loads == []


def test_the_flux_default_size_is_1024(flux_set, runner, tmp_path):
    ok, _ = image_native.generate({"backend": "native", "native": {"model": "fluxunchained"}},
                                  "p", tmp_path / "o.png", write_sidecar=False)
    assert ok
    assert (runner.generates[0]["width"], runner.generates[0]["height"]) == (1024, 1024)


def test_ctypes_lora_array_matches_the_c_layout():
    from localm.media.sdcpp import _binding as b
    assert ctypes.sizeof(b.sd_lora_t) == ctypes.sizeof(ctypes.c_void_p) * 2


# --------------------------------------------------------------------------- #
#  registering a single .safetensors component                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("shapes,expected", [
    (FLUX_AE, "vae"), (CLIP_L, "text-encoder"), (T5XXL, "text-encoder"),
    (FLUX_LORA, "lora"), (FLUX_DEV, "diffusion-unet"), (WAN21_T2V, "diffusion-unet"),
])
def test_a_single_media_safetensors_registers_with_its_detected_type(tmp_path, shapes, expected):
    import localm.model_manager as mm
    f = write_safetensors(tmp_path / "part.safetensors", shapes)
    assert mm.add_local(str(f), no_hash=True) is True
    entry = mm.load_registry()["part"]
    assert entry["model_type"] == expected and Path(entry["path"]) == f


def test_an_explicit_media_type_registers_a_file_the_header_cannot_tell(tmp_path):
    import localm.model_manager as mm
    f = tmp_path / "odd.safetensors"
    f.write_bytes(b"\0" * 32)
    assert mm.add_local(str(f), no_hash=True, model_type="vae") is True
    assert mm.load_registry()["odd"]["model_type"] == "vae"


@pytest.mark.parametrize("model_type", ["upscaler", "controlnet", "textual-inversion", "lora"])
def test_an_explicit_media_type_registers_a_lone_safetensors(tmp_path, model_type):
    import localm.model_manager as mm
    d = tmp_path / "Zusätze Öß"
    d.mkdir()
    f = d / "Hochskalierer-Ö.safetensors"
    f.write_bytes(b"\0" * 32)
    assert mm.add_local(str(f), no_hash=True, model_type=model_type) is True
    entry = mm.load_registry()["Hochskalierer-Ö"]
    assert entry["model_type"] == model_type and Path(entry["path"]) == f


def test_an_unrecognised_lone_safetensors_is_still_refused(tmp_path):
    import localm.model_manager as mm
    f = write_safetensors(tmp_path / "w.safetensors",
                          {"model.layers.0.self_attn.q_proj.weight": (8, 8)})
    assert mm.add_local(str(f), no_hash=True) is False
    assert mm.load_registry() == {}


def test_a_safetensors_inside_an_hf_folder_still_registers_the_folder(tmp_path):
    import localm.model_manager as mm
    d = tmp_path / "t5-encoder"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"architectures": ["T5EncoderModel"]}),
                                   encoding="utf-8")
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    write_safetensors(d / "model.safetensors", T5XXL)
    assert mm.add_local(str(d / "model.safetensors"), no_hash=True) is True
    reg = mm.load_registry()
    assert Path(reg["t5-encoder"]["path"]) == d


# --------------------------------------------------------------------------- #
#  review follow-ups                                                          #
# --------------------------------------------------------------------------- #

def test_hunyuan_video_is_not_read_as_flux(tmp_path):
    shapes = {"double_blocks.0.img_attn_qkv.weight": (9216, 3072),
              "txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1.weight": (6144, 3072)}
    info = families.inspect_file(write_safetensors(tmp_path / "hy.safetensors", shapes))
    assert (info.kind, info.family) == ("diffusion", "hunyuan-video")


@pytest.mark.parametrize("shape", [[[1]], [None], [{}], [1e400], [True], "x"])
def test_malformed_shapes_never_raise(tmp_path, shape):
    f = tmp_path / "bad.safetensors"
    raw = json.dumps({"a": {"dtype": "F16", "shape": shape, "data_offsets": [0, 0]}})
    raw = raw.replace("Infinity", "1e400")
    f.write_bytes(struct.pack("<Q", len(raw)) + raw.encode() + b"\0" * 8)
    assert families.inspect_file(f) is families.UNKNOWN


def test_a_chat_model_is_never_picked_automatically_as_an_llm_encoder(tmp_path):
    zi = write_safetensors(tmp_path / "z_image.safetensors", Z_IMAGE)
    vae = write_safetensors(tmp_path / "ae.safetensors", FLUX_AE)
    chat = write_gguf(tmp_path / "Qwen3-8B.gguf", {"token_embd.weight": (151936, 4096)}, "qwen3")
    _register("z", zi, "diffusion-unet")
    _register("ae", vae, "vae")
    _register("Qwen3-8B-Instruct", chat, "llm", architecture="qwen3")
    with pytest.raises(pairing.MissingComponents) as ei:
        image_native.resolve_models({"native": {"model": "z"}})
    assert [m["slot"] for m in ei.value.missing] == ["llm"]
    assert ei.value.missing[0]["download"]["filename"] == "qwen_3_4b.safetensors"


def test_an_llm_encoder_of_the_wrong_size_does_not_fit(tmp_path):
    big = write_gguf(tmp_path / "q8b.gguf", {"token_embd.weight": (151936, 4096)}, "qwen3")
    ok = write_gguf(tmp_path / "q4b.gguf", {"token_embd.weight": (151936, 2560)}, "qwen3")
    cands = [pairing.Candidate(p.stem, p, families.inspect_file(p), "text-encoder")
             for p in (big, ok)]
    assert [c.name for c in pairing.fitting("llm", "z-image", cands, auto=True)] == ["q4b"]


ACESTEP_DIT = {"decoder.condition_embedder.weight": (2048, 2048),
               "decoder.layers.0.self_attn.q_proj.weight": (2048, 2048),
               "decoder.layers.0.cross_attn.k_proj.weight": (1024, 2048),
               "decoder.layers.0.mlp.gate_proj.weight": (6144, 2048)}
ACESTEP_TEXT_ENC = {"embed_tokens.weight": (151669, 1024),
                    "layers.0.self_attn.q_proj.weight": (2048, 1024),
                    "layers.0.mlp.gate_proj.weight": (3072, 1024)}
QWEN3_4B_GGUF = {"token_embd.weight": (151936, 2560), "blk.0.attn_q.weight": (4096, 2560)}


@pytest.mark.parametrize("shapes,arch", [(ACESTEP_DIT, "acestep-dit"),
                                         (ACESTEP_TEXT_ENC, "acestep-text-enc")])
def test_an_acestep_music_component_never_reads_as_an_image_component(tmp_path, shapes, arch):
    info = families.inspect_file(write_gguf(tmp_path / f"{arch}.gguf", shapes, arch))
    assert info.kind == "music"
    assert not any(families.accepts(slot, fam, info)
                   for slot in families.SLOTS for fam in ("z-image", "flux2", "qwen-image"))


@pytest.fixture
def z_image_with_music(tmp_path):
    """A Z-Image model and its VAE next to the ACE-Step music DiT that
    ``localm setup-music`` registers."""
    d = tmp_path / "zimg"
    d.mkdir()
    _register("z", write_safetensors(d / "z_image.safetensors", Z_IMAGE), "diffusion-unet")
    _register("ae", write_safetensors(d / "ae.safetensors", FLUX_AE), "vae")
    _register("acestep-v15-turbo-Q4_K_M",
              write_gguf(d / "acestep-v15-turbo-Q4_K_M.gguf", ACESTEP_DIT, "acestep-dit"),
              "diffusion-unet", architecture="acestep-dit")
    return d


def test_z_image_pairs_the_text_encoder_and_not_the_music_dit(z_image_with_music):
    te = write_gguf(z_image_with_music / "z_te.gguf", QWEN3_4B_GGUF, "qwen3")
    _register("z-te", te, "text-encoder", architecture="qwen3")
    m = image_native.resolve_models({"native": {"model": "z"}})
    assert m["ctx"]["llm_path"] == str(te)
    assert m["components"]["llm"] == {"name": "z-te", "auto": True}


def test_z_image_with_only_the_music_dit_reports_the_missing_text_encoder(z_image_with_music):
    with pytest.raises(pairing.MissingComponents) as ei:
        image_native.resolve_models({"native": {"model": "z"}})
    assert [m["slot"] for m in ei.value.missing] == ["llm"]
    assert ei.value.missing[0]["label"] == "LLM text encoder"
    assert ei.value.missing[0]["download"]["filename"] == "qwen_3_4b.safetensors"
    assert "LLM text encoder" in str(ei.value)
    out = pairing.preview("image", "z")
    assert [m["slot"] for m in out["missing"]] == ["llm"]
    assert "llm" not in out["components"]


def test_the_music_dit_chosen_as_the_image_model_is_refused(z_image_with_music):
    with pytest.raises(ModelFileError, match="is a music generation model, not an image model"):
        image_native.resolve_models({"native": {"model": "acestep-v15-turbo-Q4_K_M"}})


@pytest.mark.parametrize("shapes,want", [
    (FLUX_SCHNELL, "diffusion-unet"),
    (CLIP_L, "text-encoder"),
    (T5XXL, "text-encoder"),
    (FLUX_AE, "vae"),
    ({"blk.0.attn_q.weight": (2048, 2048), "token_embd.weight": (32000, 2048)}, "llm"),
])
def test_a_gguf_without_architecture_is_typed_by_its_tensors(tmp_path, shapes, want):
    from localm.model_manager.registry import _detect_local_model_type
    f = write_gguf(tmp_path / "file.gguf", shapes)
    assert _detect_local_model_type(f, is_gguf=True, is_hf=False)[0] == want
    from localm.model_manager import gguf_media_component_type
    assert gguf_media_component_type(f) == (None if want == "llm" else want)


def test_a_gguf_that_declares_an_architecture_is_not_retyped_by_its_tensors(tmp_path):
    from localm.model_manager import gguf_media_component_type
    assert gguf_media_component_type(write_gguf(tmp_path / "c.gguf", CLIP_L, "llama")) is None


def test_a_page_choice_of_another_model_drops_the_settings_components():
    from localm.media.native_models import apply_choice
    s = {"native": {"model": "flux", "t5xxl": "t5", "vae": "ae", "steps": 20}}
    out = apply_choice(s, {"model": "sdxl-unet"})
    assert out["native"] == {"model": "sdxl-unet", "steps": 20}
    same = apply_choice(s, {"model": "flux", "clip_l": "vit"})
    assert same["native"]["t5xxl"] == "t5" and same["native"]["clip_l"] == "vit"
    assert s["native"]["t5xxl"] == "t5"


def test_a_slot_the_family_does_not_use_is_left_out_with_a_note(flux_set):
    m = video_native.resolve_models({"native": {"model": "wan2.1_t2v", "llm": "clip_l"}})
    assert "llm_path" not in m["ctx"]
    assert any("do not use" in n for n in m["notes"])


@pytest.fixture
def sdxl_set(tmp_path):
    d = tmp_path / "sdxl"
    d.mkdir()
    files = {
        "sdxl-unet": (write_safetensors(d / "sdxl_unet.safetensors", SDXL_UNET), "diffusion-unet"),
        "sdxl-diffusers": (write_safetensors(d / "sdxl_unet_diffusers.safetensors",
                                             SDXL_UNET_DIFFUSERS), "diffusion-unet"),
        "sdxl-full": (write_safetensors(d / "sdxl_full.safetensors", SDXL_CKPT),
                      "diffusion-unet"),
        "clip_l": (write_safetensors(d / "clip_l.safetensors", CLIP_L), "text-encoder"),
        "clip_g": (write_safetensors(d / "clip_g.safetensors", CLIP_G), "text-encoder"),
        "sdxl_vae": (write_safetensors(d / "sdxl_vae.safetensors", SD_VAE), "vae"),
    }
    for name, (path, mtype) in files.items():
        _register(name, path, mtype)
    return {k: v[0] for k, v in files.items()}


@pytest.mark.parametrize("model", ["sdxl-unet", "sdxl-diffusers"])
def test_an_sdxl_unet_alone_gets_its_parts_and_the_version_hint(sdxl_set, model):
    m = image_native.resolve_models({"native": {"model": model}})
    ctx = m["ctx"]
    assert ctx["diffusion_model_path"] == str(sdxl_set[model])
    assert ctx["clip_l_path"] == str(sdxl_set["clip_l"])
    assert ctx["clip_g_path"] == str(sdxl_set["clip_g"])
    assert ctx["vae_path"] == str(sdxl_set["sdxl_vae"])
    assert ctx["force_sdxl_vae_conv_scale"] is True
    names = families.read_tensors(Path(ctx["model_path"]))[0]
    assert any("conditioner.embedders.1" in n for n in names)
    assert any("middle_block.1" in n for n in names)


def test_an_sdxl_checkpoint_scales_an_external_vae_and_needs_no_hint(sdxl_set):
    alone = image_native.resolve_models({"native": {"model": "sdxl-full"}})
    assert alone["ctx"] == {"model_path": str(sdxl_set["sdxl-full"])}
    with_vae = image_native.resolve_models({"native": {"model": "sdxl-full",
                                                       "vae": "sdxl_vae"}})
    assert with_vae["ctx"] == {"model_path": str(sdxl_set["sdxl-full"]),
                               "vae_path": str(sdxl_set["sdxl_vae"]),
                               "force_sdxl_vae_conv_scale": True}


def test_the_version_hint_file_is_rewritten_when_it_differs(sdxl_set):
    from localm.media.sdcpp import versionhint
    for family in ("sdxl", "sd1", "sd2"):
        path = versionhint.hint_path(family)
        assert path.read_bytes() == versionhint.hint_bytes(family)
        path.write_bytes(b"damaged")
        assert versionhint.hint_path(family) == path
        assert path.read_bytes() == versionhint.hint_bytes(family)


@pytest.fixture
def sd_set(tmp_path):
    d = tmp_path / "sd"
    d.mkdir()
    files = {
        "sd1-unet": (write_safetensors(d / "sd15_unet.safetensors", SD1_UNET_DIFFUSERS),
                     "diffusion-unet"),
        "sd2-unet": (write_safetensors(d / "sd21_unet.safetensors", SD2_UNET_DIFFUSERS),
                     "diffusion-unet"),
        "clip_l": (write_safetensors(d / "clip_l.safetensors", CLIP_L), "text-encoder"),
        "openclip_h": (write_safetensors(d / "model.fp16.safetensors", CLIP_H),
                       "text-encoder"),
        "sd_vae": (write_safetensors(d / "vae-ft-mse.safetensors", SD_VAE), "vae"),
    }
    for name, (path, mtype) in files.items():
        _register(name, path, mtype)
    return {k: v[0] for k, v in files.items()}


@pytest.mark.parametrize("model,encoder,width", [("sd1-unet", "clip_l", 768),
                                                  ("sd2-unet", "openclip_h", 1024)])
def test_an_sd1_or_sd2_unet_alone_gets_its_parts_and_the_version_hint(sd_set, model, encoder,
                                                                      width):
    m = image_native.resolve_models({"native": {"model": model}})
    ctx = m["ctx"]
    assert ctx["diffusion_model_path"] == str(sd_set[model])
    assert ctx["clip_l_path"] == str(sd_set[encoder])
    assert ctx["vae_path"] == str(sd_set["sd_vae"])
    assert set(ctx) == {"diffusion_model_path", "clip_l_path", "vae_path", "model_path"}
    names = families.read_tensors(Path(ctx["model_path"]))[0]
    assert names["text_model.embeddings.token_embedding.weight"] == (1, width)
    assert any("middle_block.1" in n for n in names)
    assert not any("conditioner.embedders" in n for n in names)


def test_an_sd2_unet_is_not_paired_with_a_clip_l_and_offers_openclip_h(tmp_path):
    from localm.media.sdcpp import pairing as pr
    d = tmp_path / "sd"
    d.mkdir()
    _register("sd2-unet", write_safetensors(d / "sd21_unet.safetensors", SD2_UNET_DIFFUSERS),
              "diffusion-unet")
    _register("clip_l", write_safetensors(d / "clip_l.safetensors", CLIP_L), "text-encoder")
    _register("sd_vae", write_safetensors(d / "vae-ft-mse.safetensors", SD_VAE), "vae")
    with pytest.raises(pr.MissingComponents) as err:
        image_native.resolve_models({"native": {"model": "sd2-unet"}})
    (missing,) = err.value.missing
    assert missing["slot"] == "clip_l" and missing["label"] == "OpenCLIP-H text encoder"
    assert missing["download"]["file"] == "text_encoder/model.fp16.safetensors"
    assert "--name openclip_h" in missing["download"]["pull_args"]
    assert "OpenCLIP-H text encoder" in str(err.value)


def test_an_openclip_h_file_is_not_a_model(sd_set):
    with pytest.raises(ModelFileError, match="OpenCLIP-H text encoder, not an image model"):
        image_native.resolve_models({"native": {"model": "openclip_h"}})


def test_a_text_encoder_of_the_wrong_width_is_noted_for_an_sd_family(sd_set):
    sd1 = image_native.resolve_models({"native": {"model": "sd1-unet",
                                                  "clip_l": "openclip_h"}})
    assert any("does not look like" in n for n in sd1["notes"])
    sd2 = image_native.resolve_models({"native": {"model": "sd2-unet", "clip_l": "clip_l"}})
    assert any("OpenCLIP-H text encoder 'clip_l' does not look like" in n for n in sd2["notes"])


def test_the_worker_accepts_the_sdxl_vae_scale_flag():
    from localm.media.sdcpp import _child
    assert "force_sdxl_vae_conv_scale" in _child._CTX_BOOL_FIELDS


def test_sd_family_vaes_follow_their_names(tmp_path):
    xl = write_safetensors(tmp_path / "sdxl_vae.safetensors", SD_VAE)
    sd = write_safetensors(tmp_path / "vae-ft-mse.safetensors", SD_VAE)
    cands = [pairing.Candidate(p.stem, p, families.inspect_file(p), "vae") for p in (xl, sd)]
    assert [c.name for c in pairing.fitting("vae", "sdxl", cands)] == ["sdxl_vae"]
    assert [c.name for c in pairing.fitting("vae", "sd1", cands)] == ["vae-ft-mse"]
    ae = write_safetensors(tmp_path / "ae.safetensors", FLUX_AE)
    assert pairing.fitting("vae", "sd3", [pairing.Candidate("ae", ae,
                                                            families.inspect_file(ae), "vae")]) == []


def test_an_sd3_unet_without_a_vae_offers_the_sd3_vae_under_a_name_the_pairing_picks(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    _register("sd3.5_medium-Q4_K_M", write_gguf(d / "sd3.5_medium-Q4_K_M.gguf", SD3, "sd3"),
              "diffusion-unet")
    _register("clip_l", write_safetensors(d / "clip_l.safetensors", CLIP_L), "text-encoder")
    _register("clip_g", write_safetensors(d / "clip_g.safetensors", CLIP_G), "text-encoder")
    _register("t5xxl", write_safetensors(d / "t5xxl.safetensors", T5XXL), "text-encoder")
    settings = {"native": {"model": "sd3.5_medium-Q4_K_M"}}
    with pytest.raises(pairing.MissingComponents) as ei:
        image_native.resolve_models(settings)
    assert [m["slot"] for m in ei.value.missing] == ["vae"]
    dl = ei.value.missing[0]["download"]
    assert dl["spec"] == "Shio-Koube/SD-3.5-vae:diffusion_pytorch_model.safetensors"
    assert dl["name"] == "sd3.5_vae" and dl["model_type"] == "vae"
    assert f"localm pull {dl['spec']} --type vae --name sd3.5_vae" in str(ei.value)
    vae = write_safetensors(d / dl["filename"], FLUX_AE)
    _register(dl["name"], vae, "vae")
    m = image_native.resolve_models(settings)
    assert m["ctx"]["vae_path"] == str(vae)
    assert m["components"]["vae"] == {"name": "sd3.5_vae", "auto": True}


LLM_LORA = {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": (8, 4096),
            "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight": (11008, 8)}
SD_LORA_WITH_TE = {"lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": (4, 768),
                   "lora_unet_input_blocks_4_1_proj_in.lora_down.weight": (4, 320)}


def test_llm_adapters_are_not_image_loras(tmp_path):
    import localm.model_manager as mm
    info = families.inspect_file(write_safetensors(tmp_path / "a.safetensors", LLM_LORA))
    assert (info.kind, info.family) == ("lora", "llm")
    assert families.inspect_file(
        write_safetensors(tmp_path / "s.safetensors", SD_LORA_WITH_TE)).family == "sd1"
    d = tmp_path / "peft"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    write_safetensors(d / "adapter_model.safetensors", SDXL_LORA)
    assert mm.add_local(str(d / "adapter_model.safetensors"), no_hash=True) is False
    assert mm.add_local(str(tmp_path / "a.safetensors"), no_hash=True) is False
    assert mm.load_registry() == {}


def test_preflight_names_a_missing_part_that_has_no_download(flux_set, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from localm.config import update_config
    from localm.media import backend_choice
    from localm.plugins.engine import attach_engine
    from localm.plugins.gui.web import attach_gui
    monkeypatch.setattr(backend_choice, "managed_comfy_active", lambda cfg=None: False)
    monkeypatch.setattr("localm.media.comfy_client.is_comfy_confirmed", lambda api_url=None: False)
    from localm.config import REGISTRY_FILE
    reg = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    for name in ("t5xxl_fp8_e4m3fn", "ae", "sd-vae"):
        del reg[name]
    REGISTRY_FILE.write_text(json.dumps(reg), encoding="utf-8")
    update_config(lambda cfg: cfg.setdefault("plugins", {}).setdefault("image", {}).update(
        {"backend": "native"}))
    app = FastAPI()
    attach_engine(app)
    attach_gui(app, self_url="http://127.0.0.1:9/v1", switch_model=lambda name: None,
               active_model=lambda: "m")
    with TestClient(app) as c:
        data = c.post("/api/media/image/preflight",
                      json={"native_models": {"model": "fluxunchained"}}).json()
        bad = c.post("/api/media/image/preflight", json={"native_models": {"model": "nope"}})
    assert sorted(m["slot"] for m in data["missing"]) == ["t5xxl", "vae"]
    assert all(m["source"]["sha256"] for m in data["missing"])
    assert bad.status_code == 400


# --------------------------------------------------------------------------- #
#  the page's per-job choice through the real routes                          #
# --------------------------------------------------------------------------- #

@pytest.fixture
def media_app(flux_set, runner, tmp_path, monkeypatch):
    import localm.image_gen.comfy as ic
    import localm.media.comfy_client as cc
    from fastapi import FastAPI

    from localm.config import update_config
    from localm.media import backend_choice
    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.jobs import JobManager
    monkeypatch.setattr(backend_choice, "managed_comfy_active", lambda cfg=None: False)
    monkeypatch.setattr(cc, "is_comfy_confirmed", lambda api_url=None: False)
    monkeypatch.setattr(cc, "_comfy_alive", lambda url, timeout=3.0: False)
    monkeypatch.setattr(ic, "_comfy_alive", lambda url, timeout=3.0: False)
    monkeypatch.setattr("localm.vram.decide_media_swap", lambda s: False)
    monkeypatch.setattr("localm.vram.media_single_device_shortfall", lambda s: None)
    update_config(lambda cfg: cfg.setdefault("plugins", {}).setdefault("image", {}).update(
        {"backend": "native", "native": {"model": "wan2.1_t2v", "vae": "wan_2.1_vae"}}))
    app = FastAPI()
    pm = PluginManager(app, external_root=tmp_path / "noplugins")
    pm.install("image")
    app.state.jobs = JobManager()
    app.state.self_url = "http://127.0.0.1:8642/v1"
    return app


def _finish(app, job_id, timeout=30.0):
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = app.state.jobs.get(job_id)
        if job is not None and job.status != "running" and getattr(job, "finished_at", None):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_imagine_route_runs_the_page_choice_with_its_loras(media_app, runner, flux_set):
    from fastapi.testclient import TestClient
    with TestClient(media_app) as c:
        r = c.post("/api/imagine", json={
            "prompt": "a fox", "size": "512x512",
            "native_models": {"model": "fluxunchained", "clip_l": "ViT-L-14-TEXT"},
            "loras": [{"name": "aidma-flux", "strength": 0.6}]})
        assert r.status_code == 200, r.text
        job = _finish(media_app, r.json()["job_id"])
    lines = [e.get("text", "") for e in job._history if e.get("type") == "line"]
    assert job.status == "done", lines
    ctx = runner.loads[-1]
    assert ctx["diffusion_model_path"] == str(flux_set["fluxunchained"])
    assert ctx["clip_l_path"] == str(flux_set["ViT-L-14-TEXT"])
    assert ctx["vae_path"] == str(flux_set["ae"])
    assert runner.generates[-1]["loras"] == [{"path": str(flux_set["aidma-flux"]),
                                             "multiplier": 0.6}]


@pytest.mark.parametrize("body,needle", [
    ({"native_models": {"model": "D:/x/evil.gguf"}}, "not a registered model"),
    ({"native_models": {"unet": "fluxunchained"}}, "Unknown native image model slot"),
    ({"loras": [{"name": "fluxunchained"}]}, "not a registered LoRA"),
    ({"loras": [{"name": "aidma-flux"}] * 9}, ""),
])
def test_imagine_route_refuses_a_bad_choice_up_front(media_app, runner, body, needle):
    from fastapi.testclient import TestClient
    with TestClient(media_app) as c:
        r = c.post("/api/imagine", json={"prompt": "p", **body})
    assert r.status_code in (400, 422), r.text
    assert needle in r.text
    assert runner.loads == []


def test_native_models_route_lists_and_previews_without_paths(media_app, flux_set):
    from fastapi.testclient import TestClient
    with TestClient(media_app) as c:
        data = c.get("/api/imagine/native-models",
                     params={"model": "fluxunchained"}).json()
        bad = c.get("/api/imagine/native-models", params={"model": "C:/Windows/notepad.exe"})
    assert bad.status_code == 400
    assert {m["name"] for m in data["models"]} >= {"fluxunchained"}
    assert data["current"]["model"] == "wan2.1_t2v"
    prev = data["preview"]
    assert prev["family"] == "flux" and prev["needs"] == ["clip_l", "t5xxl", "vae"]
    assert prev["components"]["vae"] == {"name": "ae", "auto": True}
    shown = json.dumps(data).replace("\\\\", "/")
    assert str(flux_set["ae"].parent).replace("\\", "/") not in shown


def test_native_models_route_lists_fitting_components_first_and_the_family_defaults(
        media_app, flux_set):
    from fastapi.testclient import TestClient
    with TestClient(media_app) as c:
        data = c.get("/api/imagine/native-models", params={"model": "fluxunchained"}).json()
    vae = data["slots"]["vae"]
    assert vae[0]["name"] == "ae" and vae[0]["fits"] is True
    assert {o["name"] for o in vae if o["fits"] is False} == {"sd-vae", "wan_2.1_vae"}
    assert all(o["fits"] is False for o in vae[1:])
    assert data["preview"]["defaults"] == {"steps": 20, "cfg_scale": 1.0, "width": 1024,
                                           "height": 1024}


def test_the_steps_placeholder_follows_settings_then_the_curated_model_then_the_family(
        media_app, flux_set):
    from fastapi.testclient import TestClient

    from localm.config import update_config
    from localm.media.sdcpp import pairing
    rec = {"fluxunchained-dev-q8-0-v2.gguf": {"steps": 2, "cfg_scale": 1.0, "width": 512,
                                              "height": 512}}
    assert pairing.preview("image", "fluxunchained")["defaults"]["steps"] == 20
    assert pairing.preview("image", "fluxunchained", None, None, rec)["defaults"]["steps"] == 2
    update_config(lambda cfg: cfg["plugins"]["image"]["native"].update(
        {"steps": 9, "cfg_scale": 2.5}))
    with TestClient(media_app) as c:
        other = c.get("/api/imagine/native-models", params={"model": "fluxunchained"}).json()
        own = c.get("/api/imagine/native-models").json()
    for data in (other, own):
        assert data["preview"]["defaults"]["steps"] == 9
        assert data["preview"]["defaults"]["cfg_scale"] == 2.5


def test_the_hires_suggestion_follows_the_curated_model_then_the_family(media_app, flux_set):
    from fastapi.testclient import TestClient

    from localm.media.sdcpp import pairing, sampling
    rec = {"fluxunchained-dev-q8-0-v2.gguf": {"steps": 2, "cfg_scale": 1.0, "width": 512,
                                              "height": 512}}
    assert pairing.preview("image", "fluxunchained")["hires"] == sampling.suggested_hires(1024)
    assert pairing.preview("image", "fluxunchained")["hires"]["scale"] == 1.5
    small = pairing.preview("image", "fluxunchained", None, None, rec)["hires"]
    assert small == sampling.suggested_hires(512) and small["scale"] == 2.0
    with TestClient(media_app) as c:
        data = c.get("/api/imagine/native-models", params={"model": "fluxunchained"}).json()
    assert data["preview"]["hires"] == {"upscaler": "lanczos", "scale": 1.5, "steps": 0,
                                        "denoise": 0.35, "latent_denoise": 0.55}


def test_native_models_route_without_a_model_marks_nothing(media_app, flux_set):
    from fastapi.testclient import TestClient

    from localm.config import update_config
    update_config(lambda cfg: cfg["plugins"]["image"].update({"native": {}}))
    with TestClient(media_app) as c:
        data = c.get("/api/imagine/native-models").json()
    assert data["preview"] is None
    assert all("fits" not in o for o in data["slots"]["vae"])


@pytest.fixture(autouse=True)
def _no_live_vram_budget(monkeypatch):
    from localm.media.sdcpp import budget as sd_budget
    monkeypatch.setattr(sd_budget, "vram_budget", lambda backend, **kw: None)


@pytest.mark.parametrize("family", sorted(families.UNET_FAMILIES))
def test_clip_text_matches_clip_lowercasing(family):
    text = "Öl-Stil (ÄPFEL:1.2) BREAK Straße <lora:Ünder:1>"
    out = families.clip_text(text, family)
    assert out == "öl-Stil (äPFEL:1.2) BREAK Straße <lora:ünder:1>"
    assert out.lower() == text.lower()
    assert [c for c in out if c.isascii()] == [c for c in text if c.isascii()]
    assert families.clip_text("İ", family) == "İ".lower()
    assert families.clip_text("ΟΔΟΣ ΚΑΙ ΣΑ", family) == "οδος και σα"


@pytest.mark.parametrize("family", ["flux", "sd3", "wan2.1", "qwen-image", None])
def test_clip_text_leaves_case_sensitive_encoders_alone(family):
    assert families.clip_text("Ein großer ÖLBAUM", family) == "Ein großer ÖLBAUM"
