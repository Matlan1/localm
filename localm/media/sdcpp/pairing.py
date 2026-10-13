# SPDX-License-Identifier: AGPL-3.0-or-later
"""Companion files for a native diffusion model, picked from the registry.

A diffusion model file (a UNet/DiT alone) needs text encoders and a VAE before
stable-diffusion.cpp can load it. ``pair_components`` fills each slot the
model's family needs (``families.requirements``): an explicit choice wins,
otherwise the best registered file that fits the slot (``families.accepts``)
is picked, otherwise the slot is reported missing with a public download that
fits it (``suggested_download``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import families
from .families import FileInfo
from .files import ModelFileError

SLOT_LABELS = {
    "clip_l": "CLIP-L text encoder", "clip_g": "CLIP-G text encoder",
    "t5xxl": "T5 text encoder", "llm": "LLM text encoder", "vae": "VAE",
    "clip_vision": "CLIP vision encoder",
}

# Registry types whose entries can fill a native slot or be a native model.
CANDIDATE_TYPES = ("diffusion-unet", "text-encoder", "vae", "lora", "llm", "unknown",
                   "embedding")


@dataclass(frozen=True)
class Candidate:
    """A registered model file and what it is."""
    name: str
    path: Path
    info: FileInfo
    model_type: str


@dataclass(frozen=True)
class Download:
    """A public file that fills a slot, as ``localm pull`` takes it."""
    repo: str
    file: str
    size_bytes: int
    sha256: str
    model_type: str

    @property
    def spec(self) -> str:
        return f"{self.repo}:{self.file}"

    @property
    def filename(self) -> str:
        return self.file.rsplit("/", 1)[-1]

    def as_dict(self) -> dict:
        return {"repo": self.repo, "file": self.file, "spec": self.spec,
                "filename": self.filename, "size_bytes": self.size_bytes,
                "sha256": self.sha256, "model_type": self.model_type}


_CLIP_L = Download("comfyanonymous/flux_text_encoders", "clip_l.safetensors", 246_144_152,
                   "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
                   "text-encoder")
_CLIP_G = Download("Comfy-Org/stable-diffusion-3.5-fp8", "text_encoders/clip_g.safetensors",
                   1_389_382_176,
                   "ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4",
                   "text-encoder")
_T5XXL = Download("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn.safetensors",
                  4_893_934_904,
                  "7d330da4816157540d6bb7838bf63a0f02f573fc48ca4d8de34bb0cbfd514f09",
                  "text-encoder")
_UMT5 = Download("city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q4_K_M.gguf", 3_655_145_312,
                 "17cf97a5bbbc60a646d6105b832b6f657ce904a8a1ad970e4b59df0c67584a40",
                 "text-encoder")
_FLUX_AE = Download("Comfy-Org/Lumina_Image_2.0_Repackaged", "split_files/vae/ae.safetensors",
                    335_304_388,
                    "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38", "vae")
_FLUX2_VAE = Download("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors",
                      336_213_556,
                      "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5", "vae")
_WAN21_VAE = Download("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                      "split_files/vae/wan_2.1_vae.safetensors", 253_815_318,
                      "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b", "vae")
_WAN22_VAE = Download("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                      "split_files/vae/wan2.2_vae.safetensors", 1_409_400_960,
                      "e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156", "vae")
_QWEN_VAE = Download("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors",
                     253_806_246,
                     "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f", "vae")
_CLIP_VISION_H = Download("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                          "split_files/clip_vision/clip_vision_h.safetensors", 1_264_219_396,
                          "64a7ef761bfccbadbaa3da77366aac4185a6c58fa5de5f589b42a65bcc21f161",
                          "text-encoder")
_QWEN25_VL = Download("unsloth/Qwen2.5-VL-7B-Instruct-GGUF", "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                      4_683_072_384,
                      "d16776dcd9a28d42758c2958ed3a752aabf20a305252cd64ff2be72b4a78c503",
                      "text-encoder")
_QWEN3_4B = Download("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors",
                     8_044_982_048,
                     "6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a",
                     "text-encoder")
_SDXL_VAE = Download("stabilityai/sdxl-vae", "sdxl_vae.safetensors", 334_641_164,
                     "63aeecb90ff7bc1c115395962d3e803571385b61938377bc7089b36e81e92e2e", "vae")
_SD_VAE = Download("stabilityai/sd-vae-ft-mse-original",
                   "vae-ft-mse-840000-ema-pruned.safetensors", 334_641_190,
                   "735e4c3a447a3255760d7f86845f09f937809baa529c17370d83e4c3758f3c75", "vae")

# (slot, family) -> the download offered when no registered file fits.
_DOWNLOADS: dict[tuple[str, str], Download] = {
    ("clip_l", "flux"): _CLIP_L, ("clip_l", "sd3"): _CLIP_L, ("clip_l", "sdxl"): _CLIP_L,
    ("clip_l", "sd1"): _CLIP_L,
    ("clip_g", "sd3"): _CLIP_G, ("clip_g", "sdxl"): _CLIP_G,
    ("t5xxl", "flux"): _T5XXL, ("t5xxl", "sd3"): _T5XXL, ("t5xxl", "chroma"): _T5XXL,
    ("t5xxl", "wan2.1"): _UMT5, ("t5xxl", "wan2.2"): _UMT5,
    ("vae", "flux"): _FLUX_AE, ("vae", "chroma"): _FLUX_AE, ("vae", "sd3"): _FLUX_AE,
    ("vae", "z-image"): _FLUX_AE, ("vae", "ovis-image"): _FLUX_AE,
    ("vae", "flux2"): _FLUX2_VAE, ("vae", "wan2.1"): _WAN21_VAE,
    ("vae", "qwen-image"): _QWEN_VAE, ("vae", "sdxl"): _SDXL_VAE, ("vae", "sd1"): _SD_VAE,
    ("vae", "sd2"): _SD_VAE,
    ("clip_vision", "wan2.1"): _CLIP_VISION_H,
    ("llm", "qwen-image"): _QWEN25_VL, ("llm", "z-image"): _QWEN3_4B,
}

# Substrings that make a registered file the preferred fit, per (slot, family).
_NAME_HINTS: dict[tuple[str, Optional[str]], tuple[str, ...]] = {
    ("clip_l", None): ("clip_l", "clip-l", "clipl"),
    ("clip_g", None): ("clip_g", "clip-g", "clipg"),
    ("t5xxl", None): ("t5xxl", "t5-xxl", "t5"),
    ("t5xxl", "wan2.1"): ("umt5",), ("t5xxl", "wan2.2"): ("umt5",),
    ("vae", "flux"): ("ae.", "flux", "ae"), ("vae", "chroma"): ("ae.", "flux", "ae"),
    ("vae", "z-image"): ("ae.", "ae"), ("vae", "sd3"): ("sd3",),
    ("vae", "wan2.1"): ("wan_2.1", "wan2.1", "wan21", "wan"),
    ("vae", "wan2.2"): ("wan2.2", "wan_2.2", "wan22", "wan"),
    ("vae", "qwen-image"): ("qwen",), ("vae", "flux2"): ("flux2",),
    ("llm", "qwen-image"): ("qwen2.5", "qwen_2.5", "qwen25"),
    ("llm", "z-image"): ("qwen3", "qwen_3"), ("llm", "flux2"): ("mistral", "qwen"),
    ("clip_vision", None): ("clip_vision_h", "clip_vision", "vision"),
}


def suggested_download(slot: str, family: Optional[str]) -> Optional[Download]:
    """A public file that fills *slot* for a *family* model, or None."""
    return _DOWNLOADS.get((slot, family or ""))


def registry_candidates(reg: Optional[dict] = None) -> list[Candidate]:
    """Every registered single-file model of a type a native slot can use,
    classified from its header. Entries whose file is missing, unreadable, a
    directory, or a network/device path are left out."""
    from localm.model_manager import is_gguf_adapter_entry, load_registry
    from localm.model_manager.registry import _entry_path
    from localm.pathsafe import is_unc_or_device_path
    reg = load_registry() if reg is None else reg
    out = []
    for name, entry in sorted(reg.items()):
        if not isinstance(entry, dict):
            continue
        mtype = entry.get("model_type") or "llm"
        if mtype not in CANDIDATE_TYPES or is_gguf_adapter_entry(entry):
            continue
        raw = _entry_path(entry)
        if not raw or is_unc_or_device_path(raw):
            continue
        p = Path(raw)
        if p.suffix.lower() not in (".gguf", ".safetensors", ".sft"):
            continue
        try:
            if not p.is_file():
                continue
        except OSError:
            continue
        if mtype == "embedding" and entry.get("architecture") not in ("t5encoder", "t5"):
            continue
        if mtype == "llm":
            arch = entry.get("architecture")
            if arch not in families._LLM_GGUF_ARCHS:
                continue
            out.append(Candidate(str(name), p, FileInfo("llm", detail={"arch": arch}), mtype))
            continue
        info = families.inspect_file(p)
        if info.kind == "unknown":
            continue
        out.append(Candidate(str(name), p, info, mtype))
    return out


def _hint_rank(slot: str, family: Optional[str], cand: Candidate) -> int:
    hints = _NAME_HINTS.get((slot, family)) or _NAME_HINTS.get((slot, None)) or ()
    text = f"{cand.name} {cand.path.name}".lower()
    for i, h in enumerate(hints):
        if h in text:
            return i
    return len(hints)


def fitting(slot: str, family: Optional[str], cands: list[Candidate],
            variant: Optional[str] = None) -> list[Candidate]:
    """The candidates that fit *slot* of a *family* model, best first: name
    hints first, then text-encoder registry entries before chat models, then
    by name."""
    fit = [c for c in cands if families.accepts(slot, family, c.info, variant)]
    return sorted(fit, key=lambda c: (_hint_rank(slot, family, c),
                                      c.model_type == "llm", c.name.lower()))


@dataclass
class Pairing:
    """The companion files picked for a diffusion model.

    ``chosen``: slot -> ``{"name", "path", "auto"}``; ``missing``: slots no
    file fills, each with its ``download`` (or None); ``mismatched``: explicit
    choices that do not look right for the family, as user-facing notes."""
    chosen: dict
    missing: list
    mismatched: list


def pair_components(info: FileInfo, explicit: dict, cands: list[Candidate]) -> Pairing:
    """Fill each slot *info* needs. *explicit* maps a slot to the
    ``(name, path)`` the user chose; those are kept even when they do not look
    right for the family (noted in ``mismatched``). Slots *info* does not
    need but *explicit* names are passed through as chosen."""
    chosen: dict = {}
    missing: list = []
    mismatched: list = []
    family = info.family
    variant = info.detail.get("variant")
    for slot, (name, path) in explicit.items():
        chosen[slot] = {"name": name, "path": str(path), "auto": False}
        need = {r.slot for r in families.requirements(info)}
        if slot in need:
            got = families.inspect_file(path)
            if got.kind != "unknown" and not families.accepts(slot, family, got, variant):
                mismatched.append(
                    f"The {SLOT_LABELS.get(slot, slot)} '{name}' does not look like one "
                    f"{families.FAMILY_LABELS.get(family or '', family)} models use "
                    f"(it reads as {got.kind}"
                    + (f", {got.family}" if got.family else "") + ").")
    for req in families.requirements(info):
        if req.slot in chosen:
            continue
        best = fitting(req.slot, family, cands, variant)
        if best:
            c = best[0]
            chosen[req.slot] = {"name": c.name, "path": str(c.path), "auto": True}
        elif req.required:
            dl = suggested_download(req.slot, family)
            if family == "wan2.2" and req.slot == "vae" and variant == "5b":
                dl = _WAN22_VAE
            missing.append({"slot": req.slot, "label": SLOT_LABELS.get(req.slot, req.slot),
                            "download": dl.as_dict() if dl else None})
    return Pairing(chosen, missing, mismatched)


CTX_FIELDS = {"clip_l": "clip_l_path", "clip_g": "clip_g_path", "t5xxl": "t5xxl_path",
              "llm": "llm_path", "vae": "vae_path", "clip_vision": "clip_vision_path"}

KIND_WORDS = {"clip_l": "a CLIP-L text encoder", "clip_g": "a CLIP-G text encoder",
              "t5xxl": "a T5 text encoder", "umt5": "a UMT5 text encoder",
              "byt5": "a ByT5 text encoder", "llm": "an LLM text encoder", "vae": "a VAE",
              "clip_vision": "a CLIP vision encoder", "lora": "a LoRA"}


class MissingComponents(ModelFileError):
    """A diffusion model cannot load because companion slots have no file.
    ``missing`` is ``Pairing.missing``; ``family`` the model's family."""

    def __init__(self, message: str, missing: list, family: Optional[str]):
        super().__init__(message)
        self.missing = missing
        self.family = family


def build_context(main: Path, label: str, explicit: dict, *, where: str, media: str,
                  cands: Optional[list] = None) -> dict:
    """The sd.cpp load context for model file *main* plus its companions.

    *explicit* maps a slot to the ``(name, path)`` the settings name. Returns
    ``{"ctx", "info", "components", "notes"}``: ``components`` maps each slot
    to ``{"name", "auto"}``; ``notes`` are user-facing remarks about explicit
    choices that do not look right. A diffusion model gets its missing slots
    from the registry (``pair_components``); a full checkpoint loads as
    ``model_path`` with only the explicit slots; a file the header does not
    identify keeps the earlier rule (``diffusion_model_path`` when any text
    encoder is set). Raises ``MissingComponents`` when a needed slot has no
    file, and ``ModelFileError`` when *main* is a component, not a model.
    *media* (``"image"``/``"video"``) and *where* (the settings page) word the
    messages."""
    info = families.inspect_file(main)
    ctx: dict = {}
    components: dict = {}
    notes: list = []
    if info.kind in KIND_WORDS:
        art = "an" if media[:1] in "aeiou" else "a"
        raise ModelFileError(f"'{label}' is {KIND_WORDS[info.kind]}, not {art} {media} model. "
                             f"Choose it as a component in {where}, and {art} {media} model "
                             "as the model.")
    if info.kind == "diffusion":
        pairing = pair_components(info, explicit,
                                  registry_candidates() if cands is None else cands)
        if pairing.missing:
            raise MissingComponents(missing_message(label, info.family, pairing.missing, where),
                                    pairing.missing, info.family)
        ctx["diffusion_model_path"] = str(main)
        for slot, c in pairing.chosen.items():
            if slot in CTX_FIELDS:
                ctx[CTX_FIELDS[slot]] = c["path"]
                components[slot] = {"name": c["name"], "auto": c["auto"]}
        notes = pairing.mismatched
    else:
        for slot, (name, path) in explicit.items():
            if slot in CTX_FIELDS:
                ctx[CTX_FIELDS[slot]] = str(path)
                components[slot] = {"name": name, "auto": False}
        encoders = {"clip_l", "clip_g", "t5xxl", "llm"} & set(explicit)
        if info.kind == "unknown" and encoders:
            ctx["diffusion_model_path"] = str(main)
        else:
            ctx["model_path"] = str(main)
    return {"ctx": ctx, "info": info, "components": components, "notes": notes}


def _lora_fits(lora_family: Optional[str], family: Optional[str]) -> bool:
    if not lora_family or not family:
        return True
    if lora_family == "wan":
        return family.startswith("wan")
    if lora_family == "sd1":
        return family in ("sd1", "sd2")
    return lora_family == family


def resolve_loras(entries: list, family: Optional[str],
                  reg: Optional[dict] = None) -> tuple[list, list]:
    """``(loras, notes)`` for *entries* (``[{"name", "strength"}]``): each
    LoRA as ``{"name", "path", "multiplier"}`` for the sd.cpp worker, and
    user-facing notes for a LoRA made for another family than *family*. A name
    matches a registered ``lora`` entry by registry name, else by file name.
    Raises ``ModelFileError`` for a name that matches none, or whose file is
    missing or a network/device path."""
    from localm.model_manager import is_gguf_adapter_entry, load_registry
    from localm.model_manager.registry import _entry_path
    from localm.pathsafe import is_unc_or_device_path
    reg = load_registry() if reg is None else reg
    loras = {n: e for n, e in reg.items() if isinstance(e, dict)
             and e.get("model_type") == "lora" and not is_gguf_adapter_entry(e)}
    out, notes = [], []
    for item in entries:
        want = str(item.get("name") or "").strip()
        if not want:
            continue
        entry = loras.get(want)
        if entry is None:
            low = want.replace("\\", "/").rsplit("/", 1)[-1].lower()
            entry = next((e for e in loras.values()
                          if str(_entry_path(e) or "").replace("\\", "/").rsplit("/", 1)[-1]
                          .lower() in (low, low + ".safetensors")), None)
        raw = _entry_path(entry) if entry else None
        if not raw:
            raise ModelFileError(f"The LoRA '{want}' is not a registered LoRA. Add it on the "
                                 "Models page (type LoRA) first.")
        if is_unc_or_device_path(raw):
            raise ModelFileError(f"The LoRA '{want}' is on a network or device path, which "
                                 "is not allowed.")
        path = Path(raw)
        try:
            ok = path.is_file()
        except OSError:
            ok = False
        if not ok:
            raise ModelFileError(f"The LoRA '{want}' is registered, but its file is missing.")
        info = families.inspect_file(path)
        if not _lora_fits(info.family, family):
            notes.append(f"The LoRA '{want}' looks made for "
                         f"{families.FAMILY_LABELS.get(info.family or '', info.family)}, not "
                         f"{families.FAMILY_LABELS.get(family or '', family)}; it may have no "
                         "effect.")
        strength = item.get("strength")
        out.append({"name": want, "path": str(path),
                    "multiplier": 1.0 if strength is None else float(strength)})
    return out, notes


MEDIA_SLOTS = {"image": ("clip_l", "clip_g", "t5xxl", "llm", "vae"),
               "video": ("t5xxl", "llm", "clip_vision", "vae")}

_SLOT_KINDS = {"clip_l": ("clip_l",), "clip_g": ("clip_g",), "t5xxl": ("t5xxl", "umt5", "byt5"),
               "llm": ("llm",), "vae": ("vae",), "clip_vision": ("clip_vision",)}


def _option(c: Candidate) -> dict:
    return {"name": c.name, "file": c.path.name, "kind": c.info.kind,
            "family": c.info.family,
            "family_label": families.FAMILY_LABELS.get(c.info.family or "", c.info.family)}


def is_media_model(c: Candidate, media: str) -> bool:
    """Whether candidate *c* can be the main model of a *media* job: a full
    checkpoint or a diffusion model of a family for that media, or a
    ``diffusion-unet`` entry whose header does not say."""
    fams = families.IMAGE_FAMILIES if media == "image" else families.VIDEO_FAMILIES
    if c.info.kind == "checkpoint":
        return media == "image"
    if c.info.kind == "diffusion":
        return c.info.family in fams or not families.known_family(c.info.family)
    return False


def native_options(media: str, cands: Optional[list] = None,
                   reg: Optional[dict] = None) -> dict:
    """What a *media* (``"image"``/``"video"``) model picker offers:
    ``{"models", "slots", "loras"}``. ``models`` are the registered files that
    can be the main model; ``slots`` maps each component slot of that media to
    the registered files of a fitting kind (any family); ``loras`` the
    registered LoRAs. Each entry is ``{"name", "file", "kind", "family",
    "family_label"}``."""
    if cands is None:
        cands = registry_candidates(reg)
    models = [_option(c) for c in cands if is_media_model(c, media)]
    from localm.model_manager import load_registry
    reg = load_registry() if reg is None else reg
    for name, entry in sorted(reg.items()):
        if (isinstance(entry, dict) and entry.get("model_type") == "diffusion-unet"
                and not any(m["name"] == name for m in models)
                and not any(c.name == name for c in cands)):
            raw = str(entry.get("path") or "")
            models.append({"name": name, "file": raw.replace("\\", "/").rsplit("/", 1)[-1],
                           "kind": "unknown", "family": None, "family_label": None})
    slots = {slot: [_option(c) for c in cands if c.info.kind in _SLOT_KINDS[slot]]
             for slot in MEDIA_SLOTS[media]}
    loras = [_option(c) for c in cands if c.info.kind == "lora"]
    return {"models": models, "slots": slots, "loras": loras}


def preview(media: str, model_name: str, explicit: Optional[dict] = None,
            cands: Optional[list] = None) -> dict:
    """What a job would load for registered model *model_name* with the
    *explicit* ``{slot: registered name}`` choices, without loading anything:
    ``{"model", "family", "family_label", "kind", "needs", "components",
    "missing", "notes", "error"}``. ``needs`` lists the slots the family
    needs; ``error`` is a user-facing reason it cannot run, or None."""
    from .files import display_name, resolve_model_file
    cands = registry_candidates() if cands is None else cands
    out = {"model": display_name(model_name), "family": None, "family_label": None, "kind": None,
           "needs": [], "components": {}, "missing": [], "notes": [], "error": None}
    try:
        main = resolve_model_file(model_name, f"{media} model")
        info = families.inspect_file(main)
        out.update(kind=info.kind, family=info.family,
                   family_label=families.FAMILY_LABELS.get(info.family or "", info.family),
                   needs=[r.slot for r in families.requirements(info)])
        chosen = {}
        for slot, name in (explicit or {}).items():
            if name:
                chosen[slot] = (display_name(name), resolve_model_file(name, f"{media} {slot}"))
        built = build_context(main, display_name(model_name), chosen, where=f"the {media} page",
                              media=media, cands=cands)
        out.update(components=built["components"], notes=built["notes"])
    except MissingComponents as e:
        out.update(missing=e.missing, error=str(e))
    except ModelFileError as e:
        out["error"] = str(e)
    return out


def describe_components(components: dict) -> str:
    """``components`` as one user-facing phrase, e.g.
    ``"CLIP-L text encoder clip_l (auto), VAE ae (auto)"``."""
    return ", ".join(f"{SLOT_LABELS.get(slot, slot)} {c['name']}"
                     + (" (auto)" if c.get("auto") else "")
                     for slot, c in components.items())


def missing_message(model_label: str, family: Optional[str], missing: list,
                    where: str) -> str:
    """The user-facing reason a model cannot load because *missing* slots
    have no file, with how to get each."""
    fam = families.FAMILY_LABELS.get(family or "", family or "this")
    parts = []
    for m in missing:
        dl = m.get("download")
        how = (f"download it with 'localm pull {dl['spec']} --type {dl['model_type']}'"
               if dl else f"set it in {where}")
        parts.append(f"a {m['label']} ({how})")
    return (f"{model_label} is a {fam} model and needs " + "; ".join(parts)
            + f". None is registered on this machine. Or choose one in {where}.")
