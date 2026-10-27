# SPDX-License-Identifier: AGPL-3.0-or-later
"""What a model file is to stable-diffusion.cpp, read from its header, and which
companion files a diffusion model needs.

``inspect_file`` reads only the tensor names and shapes of a ``.gguf`` or
``.safetensors`` file (never tensor data) and returns a ``FileInfo``:

* ``kind``: ``checkpoint`` (a full model with its own text encoder and VAE),
  ``diffusion`` (a UNet/DiT alone), ``clip_l``, ``clip_g``, ``clip_h``
  (OpenCLIP-H, the SD 2.x text encoder), ``t5xxl``,
  ``umt5``, ``llm``, ``llm_vision`` (the vision tower of an LLM text encoder,
  a llama.cpp mmproj file), ``vae``, ``clip_vision``, ``lora``, ``controlnet``,
  ``textual_inversion`` (an embedding a prompt names by its trigger word),
  ``upscaler`` (an RGB ESRGAN image upscaler in the RRDBNet layout),
  ``music`` (a component of an ACE-Step music model, never an image or video
  file) or ``unknown``.
* ``family``: the model family a diffusion model, checkpoint, VAE or LoRA
  belongs to (``flux``, ``chroma``, ``sd3``, ``sd1``, ``sd2``, ``sdxl``,
  ``wan2.1``, ``wan2.2``, ``qwen-image``, ``qwen-image-2.1``, ``z-image``, ``flux2``,
  ...), or
  None when the header does not say.

The diffusion-model rules follow ``ModelLoader::get_sd_version`` in
stable-diffusion.cpp, applied to names normalised to its
``model.diffusion_model.`` prefix.

``requirements(info)`` lists the component slots a diffusion model needs
(``clip_l``, ``clip_g``, ``t5xxl``, ``llm``, ``vae``, ``clip_vision``, and the
optional ``llm_vision``) and
``accepts(slot, family, candidate)`` says whether a classified file fits one.
``capabilities(info)`` says which extra inputs (reference images, a
ControlNet, embeddings) a model takes in stable-diffusion.cpp.
"""

from __future__ import annotations

import json
import re
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_ST_MAX_HEADER = 100 * 1024 * 1024
_GGUF_MAX_TENSORS = 1_000_000
_GGUF_MAX_DIMS = 8

DIFFUSION_PREFIX = "model.diffusion_model."

SLOTS = ("clip_l", "clip_g", "t5xxl", "llm", "vae", "clip_vision", "llm_vision")


@dataclass(frozen=True)
class FileInfo:
    """A model file as classified from its header."""
    kind: str
    family: Optional[str] = None
    detail: dict = field(default_factory=dict, compare=False, hash=False)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "family": self.family, **self.detail}


UNKNOWN = FileInfo("unknown")


@dataclass(frozen=True)
class Requirement:
    """One companion slot a diffusion model needs."""
    slot: str
    required: bool = True


# Image families stable-diffusion.cpp generates still images with.
IMAGE_FAMILIES = frozenset({
    "sd1", "sd2", "sdxl", "sd3", "flux", "chroma", "flux2", "qwen-image",
    "qwen-image-2.1", "z-image", "ovis-image", "anima", "ernie-image",
})
# Video families.
VIDEO_FAMILIES = frozenset({"wan2.1", "wan2.2", "hunyuan-video"})

FAMILY_LABELS = {
    "sd1": "Stable Diffusion 1.x", "sd2": "Stable Diffusion 2.x", "sdxl": "SDXL",
    "sd3": "SD3", "flux": "FLUX.1", "chroma": "Chroma", "flux2": "FLUX.2",
    "qwen-image": "Qwen-Image", "qwen-image-2.1": "Qwen-Image 2.1", "z-image": "Z-Image", "ovis-image": "Ovis-Image",
    "anima": "Anima", "ernie-image": "ERNIE-Image", "wan2.1": "Wan 2.1",
    "wan2.2": "Wan 2.2", "hunyuan-video": "HunyuanVideo",
}

# The companion slots each family's diffusion model needs, from the
# stable-diffusion.cpp docs for each family.
_FAMILY_SLOTS: dict[str, tuple[str, ...]] = {
    "sd1": ("clip_l", "vae"),
    "sd2": ("clip_l", "vae"),
    "sdxl": ("clip_l", "clip_g", "vae"),
    "sd3": ("clip_l", "clip_g", "t5xxl", "vae"),
    "flux": ("clip_l", "t5xxl", "vae"),
    "chroma": ("t5xxl", "vae"),
    "flux2": ("llm", "vae"),
    "qwen-image": ("llm", "vae"),
    "qwen-image-2.1": ("llm", "vae"),
    "z-image": ("llm", "vae"),
    "ovis-image": ("llm", "vae"),
    "anima": ("llm", "vae"),
    "ernie-image": ("llm", "vae"),
    "wan2.1": ("t5xxl", "vae"),
    "wan2.2": ("t5xxl", "vae"),
    "hunyuan-video": ("llm", "t5xxl", "vae"),
}

# Optional companion slots: filled when a fitting file is registered, never
# reported missing.
_FAMILY_OPTIONAL_SLOTS: dict[str, tuple[str, ...]] = {"qwen-image": ("llm_vision",),
                                                       "qwen-image-2.1": ("llm_vision",)}

# The VAE latent layout each family decodes: ("2d", channels) or ("3d", z_dim).
_FAMILY_VAE = {
    "sd1": ("2d", 4), "sd2": ("2d", 4), "sdxl": ("2d", 4),
    "sd3": ("2d", 16), "flux": ("2d", 16), "chroma": ("2d", 16),
    "z-image": ("2d", 16), "ovis-image": ("2d", 16),
    "flux2": ("2d", 32), "ernie-image": ("2d", 32),
    "qwen-image": ("3d", 16), "anima": ("3d", 16), "wan2.1": ("3d", 16),
    "qwen-image-2.1": ("3d", 64),
}

# The T5 variant each family's t5xxl slot takes.
_FAMILY_T5 = {"sd3": "t5xxl", "flux": "t5xxl", "chroma": "t5xxl",
              "wan2.1": "umt5", "wan2.2": "umt5"}

# GGUF general.architecture values of LLM text encoders, per family.
_FAMILY_LLM_ARCH = {
    "qwen-image": ("qwen2vl", "qwen25vl", "qwen2"),
    "qwen-image-2.1": ("qwen3vl",),
    "z-image": ("qwen3",),
    "flux2": ("mistral3", "llama", "qwen3"),
    "ovis-image": ("qwen3",),
    "anima": ("qwen3",),
    "ernie-image": ("mistral3", "llama"),
    "hunyuan-video": ("qwen2vl", "qwen25vl", "qwen2"),
}

# Hidden sizes of the LLM text encoders a family's models were trained with
# (Qwen2.5-VL-7B: 3584, Qwen3-VL-8B: 4096, Qwen3-4B: 2560).
_FAMILY_LLM_HIDDEN = {"qwen-image": (3584,), "qwen-image-2.1": (4096,), "z-image": (2560,),
                      "hunyuan-video": (3584,)}

# Families whose text encoder must be Qwen3-VL: a GGUF of a ``_FAMILY_LLM_ARCH``
# architecture or a safetensors file with DeepStack vision tensors
# (``visual.deepstack_merger``), and as vision encoder a llama.cpp mmproj with
# DeepStack tensors (``v.deepstack.``) and a known projection width.
_FAMILY_VISION_DEEPSTACK = frozenset({"qwen-image-2.1"})

# UNet families stable-diffusion.cpp runs a ControlNet and textual-inversion
# embeddings with.
UNET_FAMILIES = frozenset({"sd1", "sd2", "sdxl"})

# Families stable-diffusion.cpp passes reference images to (FLUX.1 Kontext,
# FLUX.2, Qwen-Image-Edit, Qwen-Image 2.1, Z-Image Omni, Anima edit LoRAs).
REFERENCE_FAMILIES = frozenset({"flux", "flux2", "qwen-image", "qwen-image-2.1", "z-image",
                                "anima"})

_GGUF_DIFFUSION_ARCH = {"flux": "flux", "sd3": "sd3", "wan": None, "lumina2": "z-image",
                        "sdxl": "sdxl", "sd1": "sd1", "hyvid": "hunyuan-video",
                        "qwen_image": "qwen-image", "qwen_image21": "qwen-image-2.1"}

_LLM_GGUF_ARCHS = frozenset({
    "qwen2", "qwen2vl", "qwen25vl", "qwen3", "qwen3vl", "llama", "mistral3", "gemma3", "glm4",
})


@dataclass(frozen=True)
class Defaults:
    """Generation settings a family's models work with when none are set."""
    steps: int
    cfg_scale: float
    width: int
    height: int
    flow_shift: Optional[float] = None


_FAMILY_DEFAULTS = {
    "sd1": Defaults(25, 7.0, 512, 512), "sd2": Defaults(25, 7.0, 768, 768),
    "sdxl": Defaults(25, 6.0, 1024, 1024), "sd3": Defaults(28, 4.5, 1024, 1024),
    "flux": Defaults(20, 1.0, 1024, 1024), "chroma": Defaults(26, 4.0, 1024, 1024),
    "flux2": Defaults(28, 1.0, 1024, 1024), "qwen-image": Defaults(20, 2.5, 1024, 1024),
    "qwen-image-2.1": Defaults(40, 6.0, 1024, 1024),
    "z-image": Defaults(9, 1.0, 1024, 1024), "ovis-image": Defaults(20, 5.0, 1024, 1024),
    "anima": Defaults(30, 4.0, 1024, 1024), "ernie-image": Defaults(20, 4.0, 1024, 1024),
    "wan2.1": Defaults(30, 6.0, 832, 480, 3.0), "wan2.2": Defaults(30, 5.0, 1280, 704, 5.0),
    "hunyuan-video": Defaults(30, 6.0, 848, 480, 7.0),
}


def defaults_for(info: FileInfo) -> Optional[Defaults]:
    """Settings *info*'s family works with, or None for an unknown family.
    A FLUX.1 model without a guidance input (schnell) gets 4 steps."""
    d = _FAMILY_DEFAULTS.get(info.family or "")
    if d is not None and info.family == "flux" and info.detail.get("guidance") is False:
        return Defaults(4, 1.0, d.width, d.height)
    return d


def requirements(info: FileInfo) -> list[Requirement]:
    """The companion slots the diffusion model *info* needs, then the optional
    ones (``required`` False). A full checkpoint and a family localm does not
    know need none it can name."""
    if info.kind != "diffusion" or info.family is None:
        return []
    slots = list(_FAMILY_SLOTS.get(info.family, ()))
    if info.family == "wan2.1" and info.detail.get("i2v"):
        slots.append("clip_vision")
    return ([Requirement(s) for s in slots]
            + [Requirement(s, required=False) for s in _FAMILY_OPTIONAL_SLOTS.get(info.family, ())])


def known_family(family: Optional[str]) -> bool:
    return family in _FAMILY_SLOTS


def accepts(slot: str, family: Optional[str], cand: FileInfo,
            variant: Optional[str] = None) -> bool:
    """Whether classified file *cand* fits *slot* of a *family* model.
    *variant* is the wan2.2 size (``"5b"`` needs the Wan2.2 VAE)."""
    if slot == "clip_l":
        return cand.kind == ("clip_h" if family == "sd2" else "clip_l")
    if slot == "clip_g":
        return cand.kind == "clip_g"
    if slot == "clip_vision":
        return cand.kind == "clip_vision"
    if slot == "llm_vision":
        if cand.kind != "llm_vision":
            return False
        hidden = _FAMILY_LLM_HIDDEN.get(family or "")
        got = cand.detail.get("hidden")
        if family in _FAMILY_VISION_DEEPSTACK:
            return bool(cand.detail.get("deepstack")) and got in (hidden or ())
        return hidden is None or got is None or got in hidden
    if slot == "t5xxl":
        if family == "hunyuan-video":
            return cand.kind in ("t5xxl", "byt5")
        want = _FAMILY_T5.get(family or "")
        if want is None:
            return cand.kind in ("t5xxl", "umt5")
        return cand.kind == want
    if slot == "llm":
        if cand.kind != "llm":
            return False
        archs = _FAMILY_LLM_ARCH.get(family or "")
        arch = cand.detail.get("arch")
        if family in _FAMILY_VISION_DEEPSTACK and not (
                arch in (archs or ()) or (arch is None and cand.detail.get("deepstack"))):
            return False
        if archs is not None and arch is not None and arch not in archs:
            return False
        hidden = _FAMILY_LLM_HIDDEN.get(family or "")
        got = cand.detail.get("hidden")
        return hidden is None or got is None or got in hidden
    if slot == "vae":
        if cand.kind != "vae":
            return False
        if family == "wan2.2":
            want = ("3d", 48) if variant == "5b" else ("3d", 16)
        else:
            want = _FAMILY_VAE.get(family or "")
        if want is None:
            return True
        return (cand.detail.get("vae_dims"), cand.detail.get("latent")) == want
    return False


def clip_text(text: str, family: Optional[str]) -> str:
    """*text* as a *family* model's text encoders should receive it: for a
    family that encodes prompts with CLIP alone (``UNET_FAMILIES``), every
    non-ASCII character lowercased, matching the CLIP tokenizers these models
    were trained with; stable-diffusion.cpp lowercases ASCII itself. Other
    families get *text* unchanged. See test_clip_text_matches_clip_lowercasing."""
    if family not in UNET_FAMILIES or text.isascii():
        return text
    low = text.lower()
    out = []
    for i, c in enumerate(text):
        if ord(c) < 128:
            out.append(c)
        elif c == "Σ":
            out.append(low[len(text[:i].lower())])
        else:
            out.append(c.lower())
    return "".join(out)


def capabilities(info: FileInfo) -> dict:
    """The extra inputs the diffusion model or checkpoint *info* takes:
    ``{"references", "controlnet", "embeddings", "inpaint", "edit"}``, each a
    bool. ``references``: reference images (an edit family that is not a
    dedicated inpainting model, a FLUX.1 structural-control model such as
    FLUX.1 Canny/Depth or Flex.2, or Z-Image L2P, or an instruct pix2pix UNet);
    ``controlnet``
    and ``embeddings``: a UNet family;
    ``inpaint``: a dedicated inpainting model (a mask works with every model);
    ``edit``: the header marks an edit model (pix2pix, Qwen-Image-Edit 2511)."""
    family = info.family if info.kind in ("diffusion", "checkpoint") else None
    detail = info.detail or {}
    pix2pix = bool(detail.get("pix2pix"))
    return {
        "references": (family in REFERENCE_FAMILIES and not detail.get("inpaint")
                       and not detail.get("control") and not detail.get("l2p")) or pix2pix,
        "controlnet": family in UNET_FAMILIES,
        "embeddings": family in UNET_FAMILIES,
        "inpaint": bool(detail.get("inpaint")),
        "edit": pix2pix or bool(detail.get("edit")),
    }


def fits_unet_addon(addon: FileInfo, family: Optional[str]) -> bool:
    """Whether the ControlNet or textual-inversion file *addon* works with a
    *family* model: the same UNet family, and for a ControlNet in the
    diffusers layout, SDXL (the runtime converts that layout for SDXL only)."""
    if family not in UNET_FAMILIES or addon.family != family:
        return False
    if addon.kind == "controlnet" and addon.detail.get("layout") == "diffusers":
        return family == "sdxl"
    return addon.kind in ("controlnet", "textual_inversion")


# ---------------------------------------------------------------- headers --

def _read_safetensors(path: Path) -> Optional[dict]:
    size = path.stat().st_size
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            return None
        (n,) = struct.unpack("<Q", raw)
        if n > _ST_MAX_HEADER or 8 + n > size:
            return None
        try:
            header = json.loads(f.read(n))
        except RecursionError:
            return None
    if not isinstance(header, dict):
        return None
    out = {}
    for name, meta in header.items():
        if name == "__metadata__" or not isinstance(meta, dict):
            continue
        shape = meta.get("shape")
        if isinstance(shape, list) and all(isinstance(d, int) and not isinstance(d, bool)
                                           for d in shape):
            out[str(name)] = tuple(shape)
        else:
            out[str(name)] = ()
    return out


def _read_gguf(path: Path) -> Optional[tuple[dict, Optional[str]]]:
    """``({name: shape}, architecture)``; shapes in PyTorch order (ggml's
    ``ne`` reversed)."""
    from localm.model_manager.gguf import (_GGUF_TYPE_STRING, _gguf_read_string_stream,
                                           _gguf_skip_value_stream)
    from localm.model_manager.unsupported import gguf_version_supported
    arch = None
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return None
        (version,) = struct.unpack("<I", f.read(4))
        if not gguf_version_supported(version):
            return None
        tensor_count, kv_count = struct.unpack("<QQ", f.read(16))
        if tensor_count > _GGUF_MAX_TENSORS:
            return None
        for _ in range(kv_count):
            key = _gguf_read_string_stream(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            if key == "general.architecture" and vtype == _GGUF_TYPE_STRING:
                arch = _gguf_read_string_stream(f)
            else:
                _gguf_skip_value_stream(f, vtype)
        tensors = {}
        for _ in range(tensor_count):
            name = _gguf_read_string_stream(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            if n_dims > _GGUF_MAX_DIMS:
                return None
            ne = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)) if n_dims else ()
            f.seek(4 + 8, 1)
            tensors[name] = tuple(reversed(ne))
    return tensors, arch


def read_tensors(path: Path) -> Optional[tuple[dict, Optional[str]]]:
    """``({tensor name: shape}, gguf architecture or None)`` for a ``.gguf``
    or ``.safetensors`` file, or None when the header cannot be read. Never
    raises."""
    try:
        if path.suffix.lower() == ".gguf":
            return _read_gguf(path)
        if path.suffix.lower() in (".safetensors", ".sft"):
            t = _read_safetensors(path)
            return None if t is None else (t, None)
    except (OSError, ValueError, TypeError, OverflowError, MemoryError, struct.error,
            UnicodeDecodeError, RecursionError):
        return None
    return None


# ------------------------------------------------------------- classifying --

def _has(names, *needles) -> bool:
    return any(any(n in name for n in needles) for name in names)


def _shape(tensors: dict, *names) -> Optional[tuple]:
    for n in names:
        if n in tensors:
            return tensors[n]
    return None


def _diffusion_view(tensors: dict) -> dict:
    """The tensors under the ``model.diffusion_model.`` prefix sd.cpp gives a
    diffusion-model file, whatever prefix the file itself uses."""
    out = {}
    for name, shape in tensors.items():
        for pre in (DIFFUSION_PREFIX, "diffusion_model.", "model."):
            if name.startswith(pre):
                name = name[len(pre):]
                break
        out[DIFFUSION_PREFIX + name] = shape
    return out


def _diffusion_family(t: dict) -> Optional[tuple[str, dict]]:
    """``(family, detail)`` of the diffusion model in *t* (names already under
    ``model.diffusion_model.``), or None. Diffusers-layout SD 1.x, SD 2.x and
    SDXL UNets are recognised in safetensors and GGUF files alike."""
    p = DIFFUSION_PREFIX
    names = t.keys()
    if _has(names, p + "joint_blocks."):
        return "sd3", {}
    if _has(names, p + "double_stream_modulation_img.lin.weight"):
        return "flux2", {}
    if _has(names, p + "double_blocks.0.img_mlp.gate_proj.weight"):
        return "ovis-image", {}
    if _has(names, p + "nerf_final_layer_conv."):
        return "chroma", {"radiance": True}
    if _has(names, p + "txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1.weight"):
        return "hunyuan-video", {}
    if _has(names, p + "double_blocks.", p + "single_transformer_blocks."):
        if _has(names, p + "distilled_guidance_layer."):
            return "chroma", {}
        img_in = t.get(p + "img_in.weight") or ()
        detail = {"guidance": _has(names, p + "guidance_in.")}
        if len(img_in) == 2 and img_in[1] == 384:
            detail["inpaint"] = True
        elif len(img_in) == 2 and img_in[1] in (128, 196):
            detail["control"] = True
        return "flux", detail
    if _has(names, p + "txt_in.text_norm.weight"):
        img_in = t.get(p + "img_in.weight") or ()
        return "qwen-image-2.1", {"latent": img_in[1]} if len(img_in) == 2 else {}
    if _has(names, p + "transformer_blocks.0.img_mod.1.weight"):
        return "qwen-image", {"edit": True} if _has(names, "__index_timestep_zero__") else {}
    if _has(names, p + "cap_embedder.0.weight"):
        return "z-image", {"l2p": True} if _has(names, p + "local_decoder.out_conv.weight") else {}
    if _has(names, "llm_adapter.blocks.0.cross_attn.q_proj.weight"):
        return "anima", {}
    if _has(names, p + "layers.0.adaLN_sa_ln.weight"):
        return "ernie-image", {}
    if _has(names, p + "blocks.0.cross_attn.norm_k.weight", p + "blocks.0.cross_attn.k.weight"):
        if not _has(names, p + "patch_embedding."):
            return None
        pe = t.get(p + "patch_embedding.weight") or ()
        in_ch = pe[1] if len(pe) >= 2 else None
        i2v = _has(names, p + "img_emb")
        if in_ch == 48:
            return "wan2.2", {"variant": "5b"}
        if in_ch == 36 and not i2v:
            return "wan2.2", {"variant": "i2v-a14b"}
        return "wan2.1", {"i2v": i2v}
    if _has(names, p + "down_blocks.") and _has(names, p + "add_embedding."):
        return "sdxl", _unet_input_detail(t.get(p + "conv_in.weight"))
    if _has(names, p + "down_blocks."):
        ctx = _diffusers_cross_attention_dim(t, p)
        if ctx in (768, 1024):
            return ("sd2" if ctx == 1024 else "sd1"), _unet_input_detail(t.get(p + "conv_in.weight"))
    if _has(names, p + "input_blocks."):
        detail = _unet_input_detail(t.get(p + "input_blocks.0.0.weight"))
        if _has(names, p + "label_emb."):
            return "sdxl", detail
        k = t.get(p + "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight") or ()
        ctx = k[1] if len(k) >= 2 else None
        return ("sd2" if ctx == 1024 else "sd1"), detail
    return None


def _unet_input_detail(conv_in: Optional[tuple]) -> dict:
    """``{"inpaint": True}`` for a UNet whose first convolution reads 9
    channels, ``{"pix2pix": True}`` for 8, else ``{}``."""
    channels = conv_in[1] if conv_in and len(conv_in) == 4 else None
    if channels == 9:
        return {"inpaint": True}
    if channels == 8:
        return {"pix2pix": True}
    return {}


_DIFFUSERS_CROSS_ATTN_K = re.compile(
    r"down_blocks\.\d+\.attentions\.0\.transformer_blocks\.0\.attn2\.to_k\.weight$")


def _diffusers_cross_attention_dim(t: dict, prefix: str) -> Optional[int]:
    """The text-embedding width a diffusers-layout UNet's first cross-attention
    layer reads (768 for SD 1.x, 1024 for SD 2.x), or None when it has none."""
    for name, shape in t.items():
        if name.startswith(prefix) and _DIFFUSERS_CROSS_ATTN_K.search(name) and len(shape) >= 2:
            return shape[1]
    return None


def _checkpoint_family(t: dict) -> Optional[str]:
    names = t.keys()
    if _has(names, "conditioner.embedders.1", "cond_stage_model.1"):
        return "sdxl"
    if _has(names, "model.diffusion_model.joint_blocks."):
        return "sd3"
    if _has(names, "cond_stage_model.model.", "cond_stage_model.transformer.text_model."):
        tok = _shape(t, "cond_stage_model.model.token_embedding.weight",
                     "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight")
        if tok and len(tok) == 2 and tok[1] == 1024:
            return "sd2"
        if _has(names, "cond_stage_model.model."):
            return "sd2"
        return "sd1"
    return None


def _vae_info(t: dict) -> Optional[FileInfo]:
    names = t.keys()
    for pre in ("", "first_stage_model.", "vae."):
        w = t.get(pre + "decoder.conv_in.weight")
        if w and len(w) == 4 and _has(names, pre + "encoder.conv_in."):
            latent = w[1]
            fam = {4: "sd", 16: "flux", 32: "flux2"}.get(latent)
            return FileInfo("vae", fam, {"vae_dims": "2d", "latent": latent})
    if _has(names, "decoder.head.", "decoder.upsamples.") and _has(names, "encoder.downsamples."):
        w = t.get("conv2.weight") or t.get("decoder.conv1.weight") or ()
        z = w[1] if len(w) == 5 else None
        fam = {48: "wan2.2", 64: "qwen-image-2.1"}.get(z, "wan2.1") if z is not None else "wan2.1"
        return FileInfo("vae", fam, {"vae_dims": "3d", "latent": z})
    return None


_LORA_MARKERS = (".lora_down.", ".lora_up.", ".lora_A.", ".lora_B.", "lora_unet_",
                 "lora_te", ".hada_w1", ".lokr_w1", ".oft_blocks", ".diff_b",
                 "_lora1.down.", "_lora1.up.", "_lora2.down.", "_lora2.up.")


# Qwen-Image 2.1 layer names a Qwen-Image LoRA never has: its single-stream MLP
# (``img_mlp.gate_up``, ``gate_layer``, ``proj``, ``out``) and text input
# (``txt_in.in_layer``, ``txt_in.text_norm``), in the underscore form of
# ``_lora_info``.
_QWEN21_LORA_MARKERS = ("img_mlp_gate_up", "img_mlp_gate_layer", "img_mlp_proj_", "img_mlp_out_",
                        "txt_in_in_layer", "txt_in_text_norm")


def _lora_info(t: dict) -> Optional[FileInfo]:
    names = list(t.keys())
    if not any(any(m in n for m in _LORA_MARKERS) for n in names):
        return None
    flat = " ".join(names[:4000]).replace(".", "_")
    fam = None
    image_markers = ("lora_unet", "lora_te", "text_model", "diffusion_model",
                     "transformer_blocks", "double_blocks", "single_blocks", "joint_blocks")
    if ("model_layers_" in flat and ("self_attn_q_proj" in flat or "mlp_gate_proj" in flat)
            and not any(m in flat for m in image_markers)):
        return FileInfo("lora", "llm")
    if "double_blocks" in flat or "single_blocks" in flat or "single_transformer_blocks" in flat:
        fam = "flux"
    elif "joint_blocks" in flat:
        fam = "sd3"
    elif "transformer_blocks" in flat and any(m in flat for m in _QWEN21_LORA_MARKERS):
        fam = "qwen-image-2.1"
    elif ("img_mod" in flat or "img_mlp" in flat) and "transformer_blocks" in flat:
        fam = "qwen-image"
    elif ("layers_0_attention_to_" in flat and "feed_forward_w1" in flat
          and "adaLN_modulation" in flat):
        fam = "z-image"
    elif "cross_attn" in flat and "blocks_" in flat and "ffn" in flat:
        fam = "wan"
    elif "lora_te2" in flat or "lora_te1" in flat or (
            any(m in flat for m in _UNET_LORA_MARKERS) and _unet_lora_is_sdxl(t)):
        fam = "sdxl"
    elif "input_blocks" in flat or "down_blocks" in flat or "lora_te_" in flat:
        fam = "sd1"
    return FileInfo("lora", fam)


_UNET_LORA_MARKERS = ("input_blocks", "output_blocks", "middle_block", "down_blocks",
                      "up_blocks", "mid_block")
_SDXL_TRANSFORMER_DEPTH = re.compile(r"transformer_blocks[._](?:[2-9]|[1-9]\d)[._]")


def _unet_lora_is_sdxl(t: dict) -> bool:
    """Whether the UNet LoRA tensors *t* are SDXL's: a transformer stack more
    than one block deep (SD 1.x/2.x have exactly one), or cross-attention
    keys that read a 2048-wide text embedding."""
    for name, shape in t.items():
        if _SDXL_TRANSFORMER_DEPTH.search(name):
            return True
        if ("attn2" in name and "to_k" in name and len(shape) >= 2
                and (".lora_down." in name or ".lora_A." in name) and shape[-1] == 2048):
            return True
    return False


def _text_encoder_info(t: dict, arch: Optional[str]) -> Optional[FileInfo]:
    names = t.keys()
    if arch in ("t5encoder", "t5"):
        tok = _shape(t, "token_embd.weight") or ()
        vocab = tok[0] if tok else None
        dim = tok[1] if len(tok) > 1 else None
        if dim and dim < 4096:
            return FileInfo("byt5" if vocab and vocab < 1000 else "t5xxl", detail={"dim": dim})
        return FileInfo("umt5" if vocab and vocab > 200_000 else "t5xxl", detail={"dim": dim})
    if arch in _LLM_GGUF_ARCHS:
        tok = _shape(t, "token_embd.weight") or ()
        return FileInfo("llm", detail={"arch": arch, "hidden": tok[1] if len(tok) > 1 else None})
    if arch == "clip" and _has(names, "v.blk."):
        proj = _shape(t, "mm.2.weight") or ()
        return FileInfo("llm_vision", detail={"hidden": proj[0] if proj else None,
                                              "deepstack": _has(names, "v.deepstack.")})
    for pre in ("", "text_encoders.t5xxl.transformer.", "t5xxl.transformer."):
        shared = _shape(t, pre + "shared.weight", pre + "encoder.embed_tokens.weight")
        if shared and _has(names, pre + "encoder.block."):
            vocab, dim = shared[0], shared[1] if len(shared) > 1 else None
            if dim and dim < 4096:
                return FileInfo("byt5" if vocab < 1000 else "t5xxl", detail={"dim": dim})
            return FileInfo("umt5" if vocab > 200_000 else "t5xxl", detail={"dim": dim})
    for pre in ("", "text_encoders.clip_l.transformer.", "text_encoders.clip_g.transformer.",
                "transformer.", "cond_stage_model.transformer."):
        tok = _shape(t, pre + "text_model.embeddings.token_embedding.weight")
        if tok and len(tok) == 2:
            dim = tok[1]
            if dim == 768:
                return FileInfo("clip_l")
            if dim == 1280:
                return FileInfo("clip_g")
            if dim == 1024:
                return FileInfo("clip_h")
            return FileInfo("unknown", detail={"clip_dim": dim})
    if _has(names, "vision_model.encoder.layers.", "visual.transformer.resblocks."):
        return FileInfo("clip_vision")
    if _has(names, "model.layers.0.self_attn.q_proj.weight", "layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.", "language_model.model.layers.0."):
        emb = _shape(t, "model.embed_tokens.weight", "embed_tokens.weight",
                     "model.language_model.embed_tokens.weight") or ()
        return FileInfo("llm", detail={"arch": None,
                                       "hidden": emb[1] if len(emb) > 1 else None,
                                       "vision": _has(names, "visual."),
                                       "deepstack": _has(names, "visual.deepstack_merger")})
    return None


def _controlnet_info(t: dict) -> Optional[FileInfo]:
    """A ControlNet in the original layout (``input_hint_block`` and
    ``zero_convs``, optionally under ``control_model.``) or the diffusers
    layout (``controlnet_cond_embedding`` and ``controlnet_down_blocks``), with
    its UNet family, or None."""
    names = t.keys()
    for pre in ("control_model.", ""):
        if (any(n.startswith(pre + "input_hint_block.") for n in names)
                and any(n.startswith(pre + "zero_convs.") for n in names)):
            if any(n.startswith(pre + "label_emb.") for n in names):
                family = "sdxl"
            else:
                k = t.get(pre + "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight") or ()
                family = _CROSS_ATTN_FAMILIES.get(k[1] if len(k) >= 2 else None)
            return FileInfo("controlnet", family, {"layout": "original"})
    if (any(n.startswith("controlnet_cond_embedding.") for n in names)
            and any(n.startswith("controlnet_down_blocks.") for n in names)):
        if any(n.startswith("add_embedding.") for n in names):
            family = "sdxl"
        else:
            family = _CROSS_ATTN_FAMILIES.get(_diffusers_cross_attention_dim(t, ""))
        return FileInfo("controlnet", family, {"layout": "diffusers"})
    return None


_TI_MAX_TENSORS = 4
_TI_MAX_ROWS = 512
_TI_WIDTHS = {768: "sd1", 1024: "sd2", 1280: "sdxl"}
_CROSS_ATTN_FAMILIES: dict = {768: "sd1", 1024: "sd2"}


def _textual_inversion_info(t: dict) -> Optional[FileInfo]:
    """A textual-inversion embedding: at most ``_TI_MAX_TENSORS`` vectors or
    small matrices, each named without a module path (no ``.``, as
    ``emb_params`` or ``clip_l``), whose width is a CLIP text encoder's (768
    SD 1.x, 1024 SD 2.x, 768 and 1280 together SDXL), with its family, or
    None."""
    if not t or len(t) > _TI_MAX_TENSORS or any("." in name for name in t):
        return None
    widths = set()
    for shape in t.values():
        if len(shape) not in (1, 2) or (len(shape) == 2 and shape[0] > _TI_MAX_ROWS):
            return None
        if shape[-1] not in _TI_WIDTHS:
            return None
        widths.add(shape[-1])
    if 1280 in widths:
        family = "sdxl" if widths == {768, 1280} else None
    elif widths == {768}:
        family = "sd1"
    elif widths == {1024}:
        family = "sd2"
    else:
        family = None
    return FileInfo("textual_inversion", family)


_ESRGAN_RDB = re.compile(r"body\.\d+\.rdb1\.conv1\.weight")


_ESRGAN_ORIGINAL_RDB = re.compile(r"model\.1\.sub\.\d+\.RDB1\.conv1\.0\.weight")
_ESRGAN_ORIGINAL_CONV = re.compile(r"model\.(\d+)\.weight")
_SRVGG_PARAM = re.compile(r"body\.(\d+)\.(weight|bias)")


def _is_basicsr_rrdbnet(t: dict) -> bool:
    """``conv_first``, ``body.N.rdb1.conv1`` and a ``conv_last`` with three output
    channels."""
    last = t.get("conv_last.weight")
    if not last or len(last) != 4 or last[0] != 3 or len(t.get("conv_first.weight") or ()) != 4:
        return False
    return any(_ESRGAN_RDB.fullmatch(n) for n in t)


def _is_original_esrgan(t: dict) -> bool:
    """The original ESRGAN layout: ``model.0`` first convolution,
    ``model.1.sub.N.RDB1.conv1.0`` dense blocks and the highest numbered
    ``model.N`` convolution with three output channels."""
    if len(t.get("model.0.weight") or ()) != 4:
        return False
    if not any(_ESRGAN_ORIGINAL_RDB.fullmatch(n) for n in t):
        return False
    convs = {int(m.group(1)): shape for n, shape in t.items()
             if (m := _ESRGAN_ORIGINAL_CONV.fullmatch(n)) and len(shape) == 4}
    return len(convs) > 2 and convs[max(convs)][0] == 3


def _is_srvgg_compact(t: dict) -> bool:
    """SRVGGNetCompact: every tensor is ``body.N.weight|bias``; the convolutions
    sit at the even indices; the first takes three channels and the last yields
    3 * scale ** 2 of them."""
    indices = {}
    for name, shape in t.items():
        m = _SRVGG_PARAM.fullmatch(name)
        if m is None:
            return False
        if m.group(2) == "weight" and len(shape) == 4:
            indices[int(m.group(1))] = shape
    if len(indices) < 3 or 0 not in indices or any(i % 2 for i in indices):
        return False
    out = indices[max(indices)][0]
    scale_squared, remainder = divmod(out, 3)
    return indices[0][1] == 3 and remainder == 0 and int(scale_squared ** 0.5) ** 2 == scale_squared


def _upscaler_info(t: dict) -> Optional[FileInfo]:
    """An RGB ESRGAN upscaler (basicsr RRDBNet, original ESRGAN or Real-ESRGAN's
    SRVGGNetCompact layout), or None."""
    if _is_basicsr_rrdbnet(t) or _is_original_esrgan(t) or _is_srvgg_compact(t):
        return FileInfo("upscaler")
    return None


def classify_tensors(tensors: dict, arch: Optional[str] = None) -> FileInfo:
    """Classify a file from its tensor names/shapes and GGUF architecture. A
    file of tensors none of the rules knows is ``unknown`` with ``read`` True
    in its detail, plus its ``arch`` and input ``latent`` channels when
    known; no tensors at all is ``UNKNOWN``."""
    if not tensors:
        return UNKNOWN
    from localm.model_manager.gguf import gguf_is_music_architecture
    if gguf_is_music_architecture(arch):
        return FileInfo("music", detail={"arch": arch})
    names = tensors.keys()
    if _has(names, "model.diffusion_model.") and _has(
            names, "first_stage_model.", "cond_stage_model.", "conditioner.embedders.",
            "text_encoders."):
        fam = _checkpoint_family(tensors)
        dv = {k: v for k, v in tensors.items() if k.startswith(DIFFUSION_PREFIX)}
        found = _diffusion_family(dv)
        return FileInfo("checkpoint", fam or (found[0] if found else None),
                        found[1] if found else {})
    upscaler = _upscaler_info(tensors)
    if upscaler is not None:
        return upscaler
    control = _controlnet_info(tensors)
    if control is not None:
        return control
    lora = _lora_info(tensors)
    if lora is not None:
        return lora
    embedding = _textual_inversion_info(tensors)
    if embedding is not None:
        return embedding
    te = _text_encoder_info(tensors, arch)
    if te is not None:
        return te
    vae = _vae_info(tensors)
    if vae is not None:
        return vae
    found = _diffusion_family(_diffusion_view(tensors))
    if found is not None:
        return FileInfo("diffusion", found[0], found[1])
    if arch in _GGUF_DIFFUSION_ARCH:
        fam = _GGUF_DIFFUSION_ARCH[arch]
        return FileInfo("diffusion", fam, {} if fam else {"arch": arch})
    detail = {k: v for k, v in (("arch", arch), ("latent", _input_channels(tensors)))
              if v is not None}
    return FileInfo("unknown", detail={**detail, "read": True})


_INPUT_LAYERS = ("img_in.weight", "x_embedder.proj.weight", "x_embedder.weight",
                 "patch_embedding.weight", "conv_in.weight", "input_blocks.0.0.weight")


def _input_channels(tensors: dict) -> Optional[int]:
    """The channels the first layer of a diffusion model in *tensors* reads (its
    latent channels times the patch area), or None when no known input layer is
    there."""
    view = _diffusion_view(tensors)
    for name in _INPUT_LAYERS:
        shape = view.get(DIFFUSION_PREFIX + name)
        if shape and len(shape) >= 2:
            return shape[1]
    return None


def unknown_reason(info: FileInfo) -> str:
    """What the header of an unidentified model file *info* shows, as a phrase
    for a user-facing sentence: its GGUF architecture and input channels, or
    that it names neither."""
    detail = info.detail or {}
    parts = []
    if detail.get("arch"):
        parts.append(f"GGUF architecture '{detail['arch']}'")
    if detail.get("latent"):
        parts.append(f"{detail['latent']} input channels")
    if not parts:
        return "its header shows no model layers localm knows"
    return "its header shows " + " and ".join(parts)


_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 512


def inspect_file(path) -> FileInfo:
    """Classify the model file at *path* (see the module docstring). Results
    are cached per path, size and modification time. Never raises; an
    unreadable file is ``UNKNOWN``."""
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return UNKNOWN
    key = (str(p), st.st_size, st.st_mtime_ns)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
    if hit is not None:
        return hit
    read = read_tensors(p)
    info = UNKNOWN if read is None else classify_tensors(read[0], read[1])
    with _CACHE_LOCK:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        _CACHE[key] = info
    return info
