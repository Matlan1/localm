# SPDX-License-Identifier: AGPL-3.0-or-later
"""What a model file is to stable-diffusion.cpp, read from its header, and which
companion files a diffusion model needs.

``inspect_file`` reads only the tensor names and shapes of a ``.gguf`` or
``.safetensors`` file (never tensor data) and returns a ``FileInfo``:

* ``kind``: ``checkpoint`` (a full model with its own text encoder and VAE),
  ``diffusion`` (a UNet/DiT alone), ``clip_l``, ``clip_g``, ``clip_h``
  (OpenCLIP-H, the SD 2.x text encoder), ``t5xxl``,
  ``umt5``, ``llm``, ``vae``, ``clip_vision``, ``lora`` or ``unknown``.
* ``family``: the model family a diffusion model, checkpoint, VAE or LoRA
  belongs to (``flux``, ``chroma``, ``sd3``, ``sd1``, ``sd2``, ``sdxl``,
  ``wan2.1``, ``wan2.2``, ``qwen-image``, ``z-image``, ``flux2``, ...), or
  None when the header does not say.

The diffusion-model rules follow ``ModelLoader::get_sd_version`` in
stable-diffusion.cpp, applied to names normalised to its
``model.diffusion_model.`` prefix.

``requirements(info)`` lists the component slots a diffusion model needs
(``clip_l``, ``clip_g``, ``t5xxl``, ``llm``, ``vae``, ``clip_vision``) and
``accepts(slot, family, candidate)`` says whether a classified file fits one.
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

SLOTS = ("clip_l", "clip_g", "t5xxl", "llm", "vae", "clip_vision")


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
    "z-image", "ovis-image", "anima", "ernie-image",
})
# Video families.
VIDEO_FAMILIES = frozenset({"wan2.1", "wan2.2", "hunyuan-video"})

FAMILY_LABELS = {
    "sd1": "Stable Diffusion 1.x", "sd2": "Stable Diffusion 2.x", "sdxl": "SDXL",
    "sd3": "SD3", "flux": "FLUX.1", "chroma": "Chroma", "flux2": "FLUX.2",
    "qwen-image": "Qwen-Image", "z-image": "Z-Image", "ovis-image": "Ovis-Image",
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
    "z-image": ("llm", "vae"),
    "ovis-image": ("llm", "vae"),
    "anima": ("llm", "vae"),
    "ernie-image": ("llm", "vae"),
    "wan2.1": ("t5xxl", "vae"),
    "wan2.2": ("t5xxl", "vae"),
    "hunyuan-video": ("llm", "t5xxl", "vae"),
}

# The VAE latent layout each family decodes: ("2d", channels) or ("3d", z_dim).
_FAMILY_VAE = {
    "sd1": ("2d", 4), "sd2": ("2d", 4), "sdxl": ("2d", 4),
    "sd3": ("2d", 16), "flux": ("2d", 16), "chroma": ("2d", 16),
    "z-image": ("2d", 16), "ovis-image": ("2d", 16),
    "flux2": ("2d", 32), "ernie-image": ("2d", 32),
    "qwen-image": ("3d", 16), "anima": ("3d", 16), "wan2.1": ("3d", 16),
}

# The T5 variant each family's t5xxl slot takes.
_FAMILY_T5 = {"sd3": "t5xxl", "flux": "t5xxl", "chroma": "t5xxl",
              "wan2.1": "umt5", "wan2.2": "umt5"}

# GGUF general.architecture values of LLM text encoders, per family.
_FAMILY_LLM_ARCH = {
    "qwen-image": ("qwen2vl", "qwen25vl", "qwen2"),
    "z-image": ("qwen3",),
    "flux2": ("mistral3", "llama", "qwen3"),
    "ovis-image": ("qwen3",),
    "anima": ("qwen3",),
    "ernie-image": ("mistral3", "llama"),
    "hunyuan-video": ("qwen2vl", "qwen25vl", "qwen2"),
}

# Hidden sizes of the LLM text encoders a family's models were trained with
# (Qwen2.5-VL-7B: 3584, Qwen3-4B: 2560).
_FAMILY_LLM_HIDDEN = {"qwen-image": (3584,), "z-image": (2560,), "hunyuan-video": (3584,)}

_GGUF_DIFFUSION_ARCH = {"flux": "flux", "sd3": "sd3", "wan": None, "lumina2": "z-image",
                        "sdxl": "sdxl", "sd1": "sd1", "hyvid": "hunyuan-video"}

_LLM_GGUF_ARCHS = frozenset({
    "qwen2", "qwen2vl", "qwen25vl", "qwen3", "llama", "mistral3", "gemma3", "glm4",
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
    """The companion slots the diffusion model *info* needs. A full checkpoint
    and a family localm does not know need none it can name."""
    if info.kind != "diffusion" or info.family is None:
        return []
    slots = list(_FAMILY_SLOTS.get(info.family, ()))
    if info.family == "wan2.1" and info.detail.get("i2v"):
        slots.append("clip_vision")
    return [Requirement(s) for s in slots]


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


def _diffusion_family(t: dict, gguf: bool = False) -> Optional[tuple[str, dict]]:
    """``(family, detail)`` of the diffusion model in *t* (names already under
    ``model.diffusion_model.``), or None. A diffusers-layout SDXL UNet is
    recognised only from a safetensors file (*gguf* False)."""
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
        return "flux", {"guidance": _has(names, p + "guidance_in.")}
    if _has(names, p + "transformer_blocks.0.img_mod.1.weight"):
        return "qwen-image", {}
    if _has(names, p + "cap_embedder.0.weight"):
        return "z-image", {}
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
    if not gguf and _has(names, p + "down_blocks.") and _has(names, p + "add_embedding."):
        return "sdxl", {}
    if not gguf and _has(names, p + "down_blocks."):
        ctx = _diffusers_cross_attention_dim(t, p)
        if ctx in (768, 1024):
            return ("sd2" if ctx == 1024 else "sd1"), {}
    if _has(names, p + "input_blocks."):
        if _has(names, p + "label_emb."):
            return "sdxl", {}
        k = t.get(p + "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight") or ()
        ctx = k[1] if len(k) >= 2 else None
        return ("sd2" if ctx == 1024 else "sd1"), {}
    return None


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
        fam = "wan2.2" if z == 48 else "wan2.1"
        return FileInfo("vae", fam, {"vae_dims": "3d", "latent": z})
    return None


_LORA_MARKERS = (".lora_down.", ".lora_up.", ".lora_A.", ".lora_B.", "lora_unet_",
                 "lora_te", ".hada_w1", ".lokr_w1", ".oft_blocks", ".diff_b")


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
    elif ("img_mod" in flat or "img_mlp" in flat) and "transformer_blocks" in flat:
        fam = "qwen-image"
    elif ("layers_0_attention_to_" in flat and "feed_forward_w1" in flat
          and "adaLN_modulation" in flat):
        fam = "z-image"
    elif "cross_attn" in flat and "blocks_" in flat and "ffn" in flat:
        fam = "wan"
    elif "lora_te2" in flat or "lora_te1" in flat or _unet_lora_is_sdxl(t):
        fam = "sdxl"
    elif "input_blocks" in flat or "down_blocks" in flat or "lora_te_" in flat:
        fam = "sd1"
    return FileInfo("lora", fam)


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
                                       "hidden": emb[1] if len(emb) > 1 else None})
    return None


def classify_tensors(tensors: dict, arch: Optional[str] = None) -> FileInfo:
    """Classify a file from its tensor names/shapes and GGUF architecture."""
    if not tensors:
        return UNKNOWN
    names = tensors.keys()
    if _has(names, "model.diffusion_model.") and _has(
            names, "first_stage_model.", "cond_stage_model.", "conditioner.embedders.",
            "text_encoders."):
        fam = _checkpoint_family(tensors)
        dv = {k: v for k, v in tensors.items() if k.startswith(DIFFUSION_PREFIX)}
        found = _diffusion_family(dv)
        return FileInfo("checkpoint", fam or (found[0] if found else None))
    lora = _lora_info(tensors)
    if lora is not None:
        return lora
    te = _text_encoder_info(tensors, arch)
    if te is not None:
        return te
    vae = _vae_info(tensors)
    if vae is not None:
        return vae
    found = _diffusion_family(_diffusion_view(tensors), gguf=arch is not None)
    if found is not None:
        return FileInfo("diffusion", found[0], found[1])
    if arch in _GGUF_DIFFUSION_ARCH:
        return FileInfo("diffusion", _GGUF_DIFFUSION_ARCH[arch])
    return UNKNOWN


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
