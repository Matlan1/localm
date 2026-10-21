# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native image backend: stable-diffusion.cpp in an isolated worker process.

Exposes the media backend seam (``ensure_available`` / ``free_vram`` /
``generate``) over ``localm.media.sdcpp``. The runtime is installed on first
use (``runtime.provision``); the model is the one named in the plugin's
``native`` settings, or the first recommended model found in the registry.

The worker is the process-wide one in ``localm.media.sdcpp.shared``: one job at
a time, the model kept loaded between jobs, stopped by ``free_vram``, after an
idle period, or when the server exits.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from localm.media.image_params import png_text_entries
from localm.media.sdcpp import budget as sd_budget
from localm.media.sdcpp import families, pairing, sampling
from localm.media.sdcpp import runtime as sd_runtime
from localm.media.sdcpp import shared
from localm.media.sdcpp.logrelay import RecoveredWarnings
from localm.media.sdcpp.files import ModelFileError as _ModelError
from localm.media.sdcpp.files import display_name, registered_file, resolve_model_file
from localm.media.sdcpp.runner import SdCancelled, SdWorkerError

COMPUTE_ALLOWANCE_BYTES = 2 * 1024 ** 3

MIN_SIDE = 64
MAX_SIDE = 2048
MAX_BATCH = 16


@dataclass(frozen=True)
class RecommendedModel:
    """A curated public model the native backend can download and use as is."""
    name: str
    repo: str
    file: str
    sha256: str
    size_bytes: int
    steps: int
    cfg_scale: float
    width: int
    height: int

    @property
    def spec(self) -> str:
        return f"{self.repo}:{self.file}"


RECOMMENDED_MODELS: tuple[RecommendedModel, ...] = (
    RecommendedModel(
        name="sd-turbo", repo="Green-Sky/SD-Turbo-GGUF", file="sd_turbo-f16-q8_0.gguf",
        sha256="d50be7655f0a554cf8041c145d88b210bd5f3c545423119dee62ae08cae51580",
        size_bytes=2_023_745_376, steps=2, cfg_scale=1.0, width=512, height=512),
)

NEGATIVE_AT_CFG_1 = ("The negative prompt has no effect at CFG 1. Raise CFG (for example "
                     "to 3) to apply it; generation then takes about twice as long per step.")

_COMPONENT_FIELDS = (("clip_l", "clip_l_path"), ("clip_g", "clip_g_path"),
                     ("t5xxl", "t5xxl_path"), ("llm", "llm_path"), ("vae", "vae_path"),
                     ("llm_vision", "llm_vision_path"))

def _native_block(s: dict) -> dict:
    blk = s.get("native")
    return blk if isinstance(blk, dict) else {}


def _say(on_progress) -> Callable[[str], None]:
    def say(text: str) -> None:
        if on_progress is not None:
            try:
                on_progress(text)
            except Exception:
                pass
    return say


def _resolve_file(value: str, what: str) -> Path:
    """The file a native image model setting names (``resolve_model_file``)."""
    return resolve_model_file(value, f"image {what}")


def _recommended_in_registry() -> Optional[tuple[str, Path, RecommendedModel]]:
    for rec in RECOMMENDED_MODELS:
        hit = registered_file(rec.file)
        if hit is not None:
            return hit[0], hit[1], rec
    return None


def _recommended_for(path: Path) -> Optional[RecommendedModel]:
    for rec in RECOMMENDED_MODELS:
        if path.name == rec.file:
            return rec
    return None


def missing_model_message() -> str:
    rec = RECOMMENDED_MODELS[0]
    return (f"No native image model is set up. Download the recommended one "
            f"({rec.name}, {rec.size_bytes / 1024 ** 3:.1f} GB) with "
            f"'localm pull {rec.spec} --type diffusion-unet --sha256 {rec.sha256}', or set "
            "'Native image model' in Settings > Images to a model you have.")


def resolve_models(s: dict) -> dict:
    """The model files the native backend would load for settings *s*:
    ``{"key", "ctx", "label", "recommended", "family", "components",
    "notes", "defaults", "capabilities", "control_net", "embeddings"}`` (see
    ``pairing.build_context``; ``defaults`` is the family's
    ``families.Defaults`` or None; ``capabilities`` is
    ``families.capabilities``). A diffusion model's missing text encoders and
    VAE are picked from the registry. The ``native`` block's ``control_net``
    (a registered ControlNet name) and ``embeddings`` (registered
    textual-inversion names) are loaded with the model
    (``pairing.resolve_addon``); ``control_net`` and ``embeddings`` in the
    result are ``{"name", "path", ...}`` / a list of them, or None / ``[]``.
    Raises ``_ModelError`` (``pairing.MissingComponents`` when a companion file
    is missing) with a user-facing reason when they cannot be resolved."""
    blk = _native_block(s)
    model = (blk.get("model") or "").strip()
    rec = None
    if model:
        main = _resolve_file(model, "model")
        label = display_name(model)
        rec = _recommended_for(main)
    else:
        found = _recommended_in_registry()
        if found is None:
            raise _ModelError(missing_model_message())
        label, main, rec = found
    explicit = {}
    for key, _field in _COMPONENT_FIELDS:
        value = (blk.get(key) or "").strip()
        if value:
            explicit[key] = (display_name(value), _resolve_file(value, key.replace("_", "-")))
    built = pairing.build_context(main, label, explicit, where="Settings > Images",
                                  media="image")
    ctx = built["ctx"]
    info = built["info"]
    control = None
    if (blk.get("control_net") or "").strip():
        control = pairing.resolve_addon(blk["control_net"].strip(), "controlnet", info.family)
        ctx["control_net_path"] = control["path"]
    embeddings = pairing.resolve_embeddings(
        [str(n) for n in (blk.get("embeddings") or []) if str(n).strip()], info.family)
    if embeddings:
        ctx["embeddings"] = tuple((e["trigger"], e["path"]) for e in embeddings)
    key = tuple(sorted(ctx.items()))
    return {"key": key, "ctx": ctx, "label": label, "recommended": rec,
            "family": info.family, "components": built["components"],
            "notes": built["notes"], "defaults": families.defaults_for(info),
            "capabilities": families.capabilities(info), "control_net": control,
            "embeddings": embeddings}


FLASH_ATTENTION_BACKENDS = ("rocm",)


def load_ctx(ctx: dict, backend: str) -> dict:
    """The context fields a load of model set *ctx* uses on a *backend*
    runtime: *ctx* plus diffusion flash attention on ``FLASH_ATTENTION_BACKENDS``."""
    if backend in FLASH_ATTENTION_BACKENDS:
        return dict(ctx, diffusion_flash_attn=True)
    return dict(ctx)


def _runtime_choice(s: dict) -> str:
    return (_native_block(s).get("runtime") or "auto").strip().lower() or "auto"


def _ensure_runtime(s: dict, say) -> sd_runtime.Runtime:
    choice = _runtime_choice(s)
    rt = sd_runtime.resolve(choice)
    if rt is not None:
        return rt
    return sd_runtime.provision(choice, on_progress=say)


def ensure_available(s: dict, on_progress=None) -> tuple[bool, str]:
    """Check the model resolves, then install the runtime when needed. Never
    raises."""
    say = _say(on_progress)
    try:
        models = resolve_models(s)
    except _ModelError as e:
        return False, str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"The native image model could not be checked: {e}"
    try:
        rt = _ensure_runtime(s, say)
    except sd_runtime.ProvisionError as e:
        return False, f"The native image runtime is not available: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"The native image runtime could not be prepared: {e}"
    return True, (f"Native image backend ready (stable-diffusion.cpp, {rt.backend}): "
                  f"{models['label']}.")


def status(s: dict) -> dict:
    """What the native backend would use, without installing or loading
    anything: ``runtime`` (installed backend or None), ``model`` (label or
    None), ``missing`` (the reason no model resolves, or None) and
    ``recommended`` (the curated download offered when no model is set up)."""
    rt = sd_runtime.resolve(_runtime_choice(s))
    family, components, parts, notes = None, {}, [], []
    try:
        models = resolve_models(s)
        model, missing = models["label"], None
        family, components, notes = models["family"], models["components"], models["notes"]
    except pairing.MissingComponents as e:
        model, missing, family, parts = None, str(e), e.family, e.missing
    except Exception as e:  # noqa: BLE001
        model, missing = None, str(e)
    rec = RECOMMENDED_MODELS[0]
    return {
        "runtime": rt.backend if rt else None,
        "runtime_choice": _runtime_choice(s),
        "model": model,
        "missing": missing,
        "family": family,
        "components": components,
        "missing_components": parts,
        "notes": notes,
        "loaded": shared.worker_pid() is not None,
        "recommended": {"name": rec.name, "repo": rec.repo, "file": rec.file,
                        "spec": rec.spec, "sha256": rec.sha256,
                        "size_bytes": rec.size_bytes, "model_type": "diffusion-unet"},
    }


def free_vram(s: dict) -> bool:
    """Stop the worker so every byte of its VRAM is released. True once no
    worker is running."""
    return shared.free()


def worker_pid() -> Optional[int]:
    """The live worker's process id, or None."""
    return shared.worker_pid()


def vram_estimate_bytes(s: dict) -> Optional[int]:
    """Peak VRAM estimate for the configured model files (their size plus a
    fixed compute allowance), or None when they cannot be resolved."""
    try:
        models = resolve_models(s)
    except Exception:  # noqa: BLE001
        return None
    total = 0
    for key, value in models["ctx"].items():
        if key.endswith("_path"):
            try:
                total += Path(value).stat().st_size
            except OSError:
                return None
    return int(total * 1.2) + COMPUTE_ALLOWANCE_BYTES


def _default_size(version: str, rec: Optional[RecommendedModel], blk: dict,
                  defaults: Optional[families.Defaults] = None) -> tuple[int, int]:
    w, h = blk.get("width"), blk.get("height")
    if isinstance(w, int) and isinstance(h, int):
        return w, h
    if rec is not None:
        return rec.width, rec.height
    if defaults is not None:
        return defaults.width, defaults.height
    v = (version or "").lower()
    if v.startswith("sd 1") or v.startswith("sd 2"):
        return 512, 512
    return 1024, 1024


def _check_size(w: int, h: int) -> Optional[str]:
    if not (MIN_SIDE <= w <= MAX_SIDE and MIN_SIDE <= h <= MAX_SIDE):
        return f"Image size {w}x{h} is outside {MIN_SIDE}..{MAX_SIDE} pixels per side."
    if w % 8 or h % 8:
        return f"Image size {w}x{h} is not a multiple of 8 on each side."
    return None


def fit_size(width: int, height: int) -> tuple[int, int]:
    """*width* x *height* scaled down, keeping the aspect ratio, to fit
    ``MAX_SIDE`` on each side, then rounded down to multiples of 8 and raised
    to at least ``MIN_SIDE``."""
    big = max(width, height)
    if big > MAX_SIDE:
        width, height = width * MAX_SIDE // big, height * MAX_SIDE // big
    return max(MIN_SIDE, width // 8 * 8), max(MIN_SIDE, height // 8 * 8)


def _open_image(src):
    """A PIL image of *src*: a path, or encoded image bytes."""
    import io

    from PIL import Image
    return Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)


def _upright(im):
    """*im* turned the way its EXIF orientation says it is shown."""
    from PIL import ImageOps
    return ImageOps.exif_transpose(im)


def _load_init_image(src, width: Optional[int], height: Optional[int]) -> dict:
    """The base image *src* (a path or encoded bytes) as RGB pixels at
    *width* x *height*, or at its own size fitted by ``fit_size`` when either
    is None."""
    from PIL import Image
    with _open_image(src) as im:
        im = _upright(im).convert("RGB")
        if width is None or height is None:
            width, height = fit_size(im.width, im.height)
        if (im.width, im.height) != (width, height):
            im = im.resize((width, height), Image.Resampling.LANCZOS)
        return {"width": width, "height": height, "channel": 3, "data": im.tobytes()}


EMPTY_MASK = ("The mask is empty (all black), so nothing would change. Paint the area to "
              "repaint in white.")


def _load_mask(data: bytes, width: int, height: int) -> dict:
    """The mask image *data* as 8-bit gray pixels at *width* x *height*
    (white repaints, black keeps). Raises ``ValueError`` (``EMPTY_MASK``) when
    no pixel is lighter than black."""
    from PIL import Image
    with _open_image(data) as im:
        im = _upright(im).convert("L")
        if (im.width, im.height) != (width, height):
            im = im.resize((width, height), Image.Resampling.NEAREST)
        if im.getextrema()[1] == 0:
            raise ValueError(EMPTY_MASK)
        return {"width": width, "height": height, "channel": 1, "data": im.tobytes()}


def _load_rgb(data: bytes, size: Optional[tuple] = None) -> dict:
    """The image *data* as RGB pixels, resized to *size* (width, height), or
    fitted by ``fit_size`` when *size* is None."""
    from PIL import Image
    with _open_image(data) as im:
        im = _upright(im).convert("RGB")
        width, height = size if size is not None else fit_size(im.width, im.height)
        if (im.width, im.height) != (width, height):
            im = im.resize((width, height), Image.Resampling.LANCZOS)
        return {"width": width, "height": height, "channel": 3, "data": im.tobytes()}


def image_data_refusal(*, init_image_data=None, mask_image=None, ref_images=None,
                       control_image=None) -> Optional[str]:
    """Why the encoded images of a request cannot be used (one does not decode
    in full, or the mask is all black: ``EMPTY_MASK``), or None. Decodes every
    pixel; nothing is loaded onto the GPU."""
    items = [("base image", init_image_data), ("mask", mask_image),
             ("control image", control_image)]
    items += [(f"reference image {i + 1}", r) for i, r in enumerate(ref_images or [])]
    for what, data in items:
        if data is None:
            continue
        try:
            with _open_image(data) as im:
                im.load()
                if what == "mask" and im.convert("L").getextrema()[1] == 0:
                    return EMPTY_MASK
        except (OSError, ValueError, SyntaxError) as e:
            return f"The {what} could not be read: {e}"
    return None


NO_REFERENCES = ("{model} ({family}) does not take reference images. They work with edit "
                 "models: FLUX.1 Kontext, FLUX.2, Qwen-Image-Edit, and other Qwen-Image, "
                 "Z-Image and Anima models, or an instruct-pix2pix model.")


CONTROL_STRENGTH = 0.9

NO_LLM_VISION = ("The Qwen text encoder has no vision encoder, so the reference images reach the "
                 "image model only and the edit follows them less closely. Register the "
                 "matching Qwen2.5-VL mmproj file (type mmproj) and pick it as the LLM vision "
                 "encoder.")


def _lacks_llm_vision(models: dict) -> bool:
    """Whether the Qwen-Image model set *models* has no vision encoder for its
    LLM text encoder: no ``llm_vision`` file and an LLM file without vision
    weights."""
    ctx = models.get("ctx") or {}
    if models.get("family") != "qwen-image" or ctx.get("llm_vision_path"):
        return False
    llm = ctx.get("llm_path")
    return bool(llm) and not families.inspect_file(llm).detail.get("vision")


def input_refusal(s: dict, *, input_image=None, ref_images=None, control_image=None,
                  mask_image=None, init_image_data=None, **_ignored) -> Optional[str]:
    """Why the native backend cannot run the image inputs of a request with
    settings *s* (a model that does not take reference images, a ControlNet
    without a control image or for another family, a mask without a base
    image, an image that does not decode or an empty mask:
    ``image_data_refusal``), or None. A model set that does not resolve is left
    to ``ensure_available``, which reports it. Checked before any VRAM
    handover."""
    if mask_image is not None and init_image_data is None and input_image is None:
        return "A mask needs a base image: set the input image first."
    bad = image_data_refusal(init_image_data=init_image_data, mask_image=mask_image,
                             ref_images=ref_images, control_image=control_image)
    if bad:
        return bad
    if not (ref_images or control_image is not None or _native_block(s).get("control_net")):
        return None
    try:
        models = resolve_models(s)
    except _ModelError as e:
        return str(e)
    return _input_refusal(models, ref_images=ref_images, control_image=control_image)


def _input_refusal(models: dict, *, ref_images, control_image) -> Optional[str]:
    """Why the resolved model set *models* cannot take these inputs, or None."""
    caps = models.get("capabilities") or {}
    family = families.FAMILY_LABELS.get(models.get("family") or "", models.get("family")
                                        or "unknown family")
    if ref_images and not caps.get("references"):
        return NO_REFERENCES.format(model=models["label"], family=family)
    if control_image is not None and not models.get("control_net"):
        return "A control image needs a ControlNet: pick one."
    if models.get("control_net") and control_image is None:
        return "A ControlNet needs a control image."
    return None


BLANK_IMAGE = ("The model returned a pure white image: its numbers overflowed, usually because "
               "the VAE does not suit the model. Choose a different VAE, or none for a "
               "checkpoint that has its own.")


def _is_blank_white(result: dict) -> bool:
    """Whether the generated image *result* is one flat white (every byte 255)."""
    data = result["data"]
    return len(data) > 0 and data.count(b"\xff") == len(data)


def _write_png(out_path: Path, result: dict,
               text_entries: Optional[list] = None) -> None:
    """Save *result* as a PNG at *out_path*. The file carries no metadata
    except *text_entries* (``[(keyword, text)]``) when given."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    mode = {3: "RGB", 4: "RGBA", 1: "L"}.get(result["channel"])
    if mode is None:
        raise ValueError(f"the runtime returned an image with {result['channel']} channels")
    img = Image.frombytes(mode, (result["width"], result["height"]), result["data"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    try:
        info = None
        if text_entries:
            info = PngInfo()
            for key, text in text_entries:
                info.add_text(key, text)
        img.save(tmp, format="PNG", pnginfo=info)
        os.replace(tmp, out_path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _embed_params(blk: dict, write_sidecar: bool) -> bool:
    """Whether this image carries its parameters as PNG text: the opt-in
    ``embed_params`` setting is on, the caller allows a sidecar, and the
    session is not in privacy mode."""
    if not blk.get("embed_params") or not write_sidecar:
        return False
    from localm.audit import SessionMode, effective_mode
    return effective_mode("server") != SessionMode.PRIVACY


def _write_sidecar(out_path: Path, record: dict) -> str:
    try:
        out_path.with_suffix(out_path.suffix + ".json").write_text(
            json.dumps({k: v for k, v in record.items() if v is not None},
                       indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        return (f"WARNING: the reproducibility sidecar could not be saved ({e}); "
                "the image itself was saved.")
    return ""


class _EventRelay:
    """The worker-event handler for one job: status lines to *say*, preview
    frames to *on_preview*. ``flush`` shows any held warning lines and must be
    called when the job ends."""

    def __init__(self, say, on_preview=None):
        self._say = say
        self._on_preview = on_preview
        self._phase = None
        self._warnings = RecoveredWarnings(say)

    def flush(self) -> None:
        self._warnings.flush()

    def __call__(self, event) -> None:
        kind = event[0]
        if kind == "log":
            self._warnings.line(event[2])
            return
        if kind == "preview":
            if self._on_preview is not None:
                self._on_preview(event[1])
            return
        info = event[1]
        phase, step, steps = info.get("phase"), info.get("step", 0), info.get("steps", 0)
        if phase == "sampling":
            self._warnings.sampling()
            secs = info.get("secs") or 0.0
            rate = f" ({secs:.2f} s/step)" if secs else ""
            self._say(f"Step {step}/{steps}{rate}")
        elif phase == "upscaling":
            self._say(f"Upscaling {step}/{steps}")
        elif phase != self._phase or (steps and step == steps):
            label = "Decoding" if phase == "decoding" else "Loading weights"
            if steps and step == steps:
                self._say(f"{label}: done")
            elif phase != self._phase:
                self._say(f"{label}...")
        self._phase = phase


def _event_relay(say, on_preview=None) -> _EventRelay:
    return _EventRelay(say, on_preview)


def _batch_path(out_path: Path, index: int) -> Path:
    """The file image number *index* (from 0) of a batch is saved to: *out_path*
    for the first, ``<stem>_<n><suffix>`` beside it for the rest."""
    if index == 0:
        return out_path
    return out_path.with_name(f"{out_path.stem}_{index + 1}{out_path.suffix}")


def _hires_refusal(hires: dict, width, height) -> Optional[str]:
    """Why hires settings *hires* (``native_models.validate_hires``) cannot run,
    or None. The size check applies when *width* and *height* are known."""
    if hires.get("upscaler") == sampling.HIRES_MODEL_UPSCALER:
        try:
            _resolve_file(hires.get("model") or "", "hires upscaler")
        except _ModelError as e:
            return str(e)
    if width is not None and height is not None:
        return sampling.check_hires_target(int(width), int(height), float(hires["scale"]))
    return None


def refusal(*, model_overrides=None, lora_name=None,
            width=None, height=None, sample_method=None, scheduler=None,
            hires=None, batch=1, **_ignored) -> Optional[str]:
    """Why the native backend cannot honour a request with these inputs, or
    None. Checked before any download, VRAM handover or load."""
    if model_overrides:
        return ("Workflow model choices apply to ComfyUI. The native backend uses "
                "the model set in Settings > Images.")
    if lora_name:
        try:
            pairing.resolve_loras([{"name": lora_name}], None)
        except _ModelError as e:
            return str(e)
    try:
        sampling.check_sample_method(sample_method)
        sampling.check_scheduler(scheduler)
    except ValueError as e:
        return str(e)
    if not 1 <= int(batch or 1) <= MAX_BATCH:
        return f"Batch count must be between 1 and {MAX_BATCH}."
    if (width is None) != (height is None):
        return "Give both width and height, or neither."
    if width is not None and height is not None:
        reason = _check_size(int(width), int(height))
        if reason:
            return reason
    if hires:
        return _hires_refusal(hires, width, height)
    return None


def generate(s: dict, prompt: str, out_path: Path, **kwargs) -> tuple[bool, str]:
    """Generate one image of *prompt* into *out_path*; see ``_generate``.
    Never raises: an unexpected error becomes ``(False, message)``."""
    try:
        return _generate(s, prompt, out_path, **kwargs)
    except Exception as e:  # noqa: BLE001
        from localm.debuglog import logger
        logger.warning("native image generation failed: %s: %s", type(e).__name__, e)
        return False, f"Native image generation failed: {type(e).__name__}: {e}"


def _generate(s: dict, prompt: str, out_path: Path, *,
             self_url: Optional[str] = None,
             write_sidecar: bool = True,
             instance_token: Optional[str] = None,
             guidance: Optional[float] = None,
             cfg: Optional[float] = None,
             negative_prompt: Optional[str] = None,
             seed: Optional[int] = None,
             input_image: Optional[Path] = None,
             denoise: Optional[float] = None,
             model_overrides: Optional[dict] = None,
             lora_name: Optional[str] = None,
             lora_strength_model: Optional[float] = None,
             lora_strength_clip: Optional[float] = None,
             swap: bool = False,
             delete_outputs: Optional[bool] = None,
             cancel_check=None,
             placement: Optional[dict] = None,
             on_progress=None,
             width: Optional[int] = None,
             height: Optional[int] = None,
             loras: Optional[list] = None,
             sample_method: Optional[str] = None,
             scheduler: Optional[str] = None,
             hires: Optional[dict] = None,
             batch: int = 1,
             on_saved: Optional[Callable[[Path], None]] = None,
             steps: Optional[int] = None,
             clip_skip: Optional[int] = None,
             on_preview=None,
             on_queue_state: Optional[Callable[[str], None]] = None,
             init_image_data: Optional[bytes] = None,
             mask_image: Optional[bytes] = None,
             ref_images: Optional[list] = None,
             control_image: Optional[bytes] = None,
             control_strength: Optional[float] = None,
             control_preprocess: Optional[str] = None) -> tuple[bool, str]:
    """Generate *batch* images of *prompt* (each a PNG without embedded metadata
    unless the ``embed_params`` setting is on and the session is not in privacy
    mode). Returns ``(ok, message)``. Image ``i`` (from 0) is made with seed
    ``seed + i`` and saved to ``_batch_path(out_path, i)`` with, when
    *write_sidecar*, a ``.json`` sidecar beside it; *on_saved* is called with
    each saved path as soon as it is written, so a failure or cancel after
    the first image keeps the images already made (the message says how many).

    *sample_method* and *scheduler* (``sampling`` names) override the settings'
    and the model's own defaults. *hires* (``native_models.validate_hires``)
    adds a hires-fix pass.

    *loras* (``[{"name", "strength"}]``) and *lora_name* (with
    *lora_strength_model*) name registered LoRAs (``pairing.resolve_loras``);
    *lora_strength_clip* has no native counterpart and is ignored.
    ``model_overrides`` (ComfyUI workflow model slots) is refused with a
    reason rather than ignored.
    ``delete_outputs`` has nothing to act on (there is no second copy).
    *swap* asks this backend to unload the chat model itself, used when the
    caller's own unload did not succeed. *steps* and *clip_skip* override the
    settings and the family default for this image. *on_preview*, when given,
    receives ``{"step", "steps", "width", "height", "png"}`` for each
    in-progress frame the worker makes. *on_queue_state* is called while the
    job waits for another image or video job (``shared.held``).

    *init_image_data* (encoded image bytes) is the base image instead of the
    file *input_image*. *mask_image* (encoded bytes) repaints its white area
    of the base image and keeps the black (inpainting; an outpaint is a base
    image extended with white mask over the new area). *ref_images* (encoded
    bytes each) are reference images for an edit model. *control_image*
    (encoded bytes) drives the ControlNet named in the settings' ``native``
    block at *control_strength*; *control_preprocess* ``canny`` turns it into
    an edge map first. These are refused with a reason for a model that cannot
    take them (``_input_refusal``)."""
    say = _say(on_progress)
    refused = refusal(model_overrides=model_overrides, lora_name=lora_name,
                      width=width, height=height, sample_method=sample_method,
                      scheduler=scheduler, hires=hires, batch=batch)
    if refused:
        return False, refused
    try:
        models = resolve_models(s)
        rt = _ensure_runtime(s, say)
    except sd_runtime.ProvisionError as e:
        return False, f"The native image runtime is not available: {e}"
    except _ModelError as e:
        return False, str(e)
    blk = _native_block(s)
    rec = models["recommended"]
    defaults = models.get("defaults")
    if mask_image is not None and init_image_data is None and input_image is None:
        return False, "A mask needs a base image: set the input image first."
    refused = (image_data_refusal(init_image_data=init_image_data, mask_image=mask_image,
                                  ref_images=ref_images, control_image=control_image)
               or _input_refusal(models, ref_images=ref_images, control_image=control_image))
    if refused:
        return False, refused
    wanted = list(loras or [])
    if lora_name:
        wanted.append({"name": lora_name, "strength": lora_strength_model})
    try:
        lora_files, lora_notes = pairing.resolve_loras(wanted, models.get("family"))
    except _ModelError as e:
        return False, str(e)
    for note in list(models.get("notes") or ()) + lora_notes:
        say(note)
    if lora_files:
        say("LoRA: " + ", ".join(f"{x['name']} ({x['multiplier']:g})" for x in lora_files) + ".")
    control = models.get("control_net")
    embeddings = models.get("embeddings") or []
    if control:
        strength = CONTROL_STRENGTH if control_strength is None else control_strength
        say(f"ControlNet: {control['name']} ({strength:g}"
            + (", canny" if control_preprocess == "canny" else "") + ").")
    if embeddings:
        say("Embeddings: " + ", ".join(e["trigger"] for e in embeddings) + ".")
        text = f"{prompt} {negative_prompt or ''}".lower()
        unused = [e["trigger"] for e in embeddings if e["trigger"].lower() not in text]
        if unused:
            say("Not in the prompt or negative prompt, so they have no effect: "
                + ", ".join(unused) + ".")
    if ref_images and _lacks_llm_vision(models):
        say(NO_LLM_VISION)
    if mask_image is not None and (models.get("capabilities") or {}).get("inpaint"):
        say("Inpainting with a dedicated inpainting model.")
    if any(c.get("auto") for c in (models.get("components") or {}).values()):
        say("Using " + pairing.describe_components(models["components"]) + ".")
    if swap and self_url:
        from localm.media.comfy_client import _localm_unload
        say("Freeing VRAM: unloading the chat model...")
        if _localm_unload(self_url, instance_token) is None:
            say("Could not unload the chat model - the image model may run low on VRAM.")
    relay = _event_relay(say, on_preview)
    runner = shared.runner
    hires_path = None
    if hires and hires.get("upscaler") == sampling.HIRES_MODEL_UPSCALER:
        try:
            hires_path = _resolve_file(hires.get("model") or "", "hires upscaler")
        except _ModelError as e:
            return False, str(e)
    saved: list[Path] = []
    result: dict = {}

    def partial(text: str) -> str:
        if not saved:
            return text
        return f"{text} {len(saved)} of {batch} images were saved."

    with shared.held(say, cancel_check, on_queue_state) as acquired:
        if not acquired:
            return False, "Cancelled."
        shared.cancel_idle_timer()
        try:
            ctx = load_ctx(models["ctx"], rt.backend)
            key = (models["key"], rt.backend)
            if runner.loaded_key != key or not runner.is_alive():
                say(f"Loading {models['label']} (stable-diffusion.cpp, {rt.backend})...")
                ctx = sd_budget.reload_ctx(runner, ctx, rt.backend, say,
                                           cancel_check, devices=rt.devices)
            info = runner.ensure_loaded(key, rt.path, ctx,
                                         extra_dirs=rt.extra_dirs, on_event=relay,
                                         cancel_check=cancel_check)
            if not info.get("image"):
                runner.shutdown()
                return False, (f"{models['label']} ({info.get('version') or 'unknown'}) is not "
                               "an image generation model.")
            if ref_images and info.get("ref_images") is False:
                return False, NO_REFERENCES.format(model=models["label"],
                                                   family=info.get("version") or "unknown")
            init = None
            if init_image_data is not None or input_image is not None:
                init = _load_init_image(init_image_data if init_image_data is not None
                                        else Path(input_image), width, height)
                width, height = init["width"], init["height"]
            if width is None or height is None:
                width, height = _default_size(info.get("version", ""), rec, blk, defaults)
            mask = _load_mask(mask_image, int(width), int(height)) if mask_image else None
            refs = [_load_rgb(r) for r in (ref_images or [])]
            control_px = (_load_rgb(control_image, (int(width), int(height)))
                          if control_image is not None else None)
            if hires:
                reason = sampling.check_hires_target(int(width), int(height),
                                                     float(hires["scale"]))
                if reason:
                    return False, reason
            if seed is None or seed < 0:
                seed = secrets.randbelow(2 ** 31)
            steps = steps or blk.get("steps") or (rec.steps if rec else None) or (
                defaults.steps if defaults else None)
            cfg_scale = cfg if cfg is not None else (
                blk.get("cfg_scale") if blk.get("cfg_scale") is not None
                else (rec.cfg_scale if rec else (defaults.cfg_scale if defaults else None)))
            method = sample_method or blk.get("sample_method") or None
            sched = scheduler or blk.get("scheduler") or None
            hires_payload = None
            if hires:
                hires_payload = {
                    "upscaler": sampling.HIRES_UPSCALER_LIBRARY_NAMES[hires["upscaler"]],
                    "model_path": str(hires_path) if hires_path else None,
                    "scale": float(hires["scale"]), "steps": int(hires["steps"]),
                    "denoising_strength": float(hires["denoise"]),
                }
            components = {k: v["name"] for k, v in (models.get("components") or {}).items()} or None
            if negative_prompt and cfg_scale is not None and abs(float(cfg_scale) - 1.0) < 1e-5:
                say(NEGATIVE_AT_CFG_1)
            batch_started = time.monotonic()
            for index in range(batch):
                if cancel_check is not None and cancel_check():
                    raise SdCancelled("Generation cancelled.")
                image_seed = int(seed) + index
                params = {
                    "prompt": prompt, "negative_prompt": negative_prompt or "",
                    "width": int(width), "height": int(height), "seed": image_seed,
                    "steps": steps, "cfg_scale": cfg_scale, "guidance": guidance,
                    "clip_skip": clip_skip, "preview": on_preview is not None,
                    "sample_method": method, "scheduler": sched,
                    "init_image": init,
                    "strength": (denoise if denoise is not None else 0.75) if init else None,
                    "loras": [{"path": x["path"], "multiplier": x["multiplier"]}
                              for x in lora_files],
                    "hires": hires_payload,
                    "mask_image": mask,
                    "ref_images": refs,
                    "control_image": control_px,
                    "control_strength": (None if control_px is None else
                                         (CONTROL_STRENGTH if control_strength is None
                                          else float(control_strength))),
                    "control_canny": control_preprocess == "canny",
                }
                label = f"image {index + 1}/{batch}, " if batch > 1 else ""
                say(f"Generating {label}{width}x{height}"
                    + (f", {steps} steps" if steps else "")
                    + (f", hires x{hires['scale']:g}" if hires else "")
                    + f", seed {image_seed}...")
                image_started = time.monotonic()
                result = runner.generate_image(params, on_event=relay, cancel_check=cancel_check)
                if _is_blank_white(result):
                    raise SdWorkerError(BLANK_IMAGE)
                path = _batch_path(Path(out_path), index)
                record = {
                    "prompt": prompt, "negative_prompt": negative_prompt or None,
                    "seed": image_seed, "cfg": cfg_scale, "guidance": guidance,
                    "steps": steps, "sample_method": method, "scheduler": sched,
                    "clip_skip": clip_skip,
                    "width": result["width"], "height": result["height"],
                    "size": f"{int(width)}x{int(height)}",
                    "input_image": str(input_image) if input_image else None,
                    "input_image_data": True if init_image_data is not None else None,
                    "mask": True if mask is not None else None,
                    "reference_images": len(refs) or None,
                    "control": ({"model": control["name"],
                                 "strength": params["control_strength"],
                                 "preprocess": control_preprocess or "none"}
                                if control else None),
                    "embeddings": [e["name"] for e in embeddings] or None,
                    "denoise": params["strength"], "hires": hires,
                    "batch": {"index": index + 1, "count": batch} if batch > 1 else None,
                    "backend": "native", "model": models["label"],
                    "components": components,
                    "loras": [{"name": x["name"], "strength": x["multiplier"]}
                              for x in lora_files] or None,
                    "runtime": f"stable-diffusion.cpp {sd_runtime.pins.TAG} ({rt.backend})",
                }
                entries = png_text_entries(record) if _embed_params(blk, write_sidecar) else None
                try:
                    _write_png(path, result, entries)
                except (OSError, ValueError) as e:
                    return False, partial(f"The generated image could not be saved: {e}")
                saved.append(path)
                warning = ""
                if write_sidecar:
                    warning = _write_sidecar(path, {
                        **record,
                        "elapsed_seconds": round(time.monotonic() - image_started, 1),
                        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    })
                if warning:
                    say(warning)
                if on_saved is not None:
                    on_saved(path)
        except SdCancelled:
            return False, partial("Cancelled.")
        except SdWorkerError as e:
            return False, partial(f"Native image generation failed: {e}")
        except (OSError, ValueError) as e:
            return False, partial(f"Native image generation failed: {e}")
        finally:
            relay.flush()
            if runner.is_alive():
                shared.arm_idle_timer()
    elapsed = time.monotonic() - batch_started
    if batch == 1:
        return True, (f"Saved {saved[0].name} ({result['width']}x{result['height']}, "
                      f"seed {seed}, {elapsed:.1f} s).")
    return True, (f"Saved {batch} images ({result['width']}x{result['height']}, seeds "
                  f"{seed}-{int(seed) + batch - 1}, {elapsed:.1f} s): "
                  + ", ".join(p.name for p in saved) + ".")


MAX_UPSCALE_INPUT_SIDE = 4096
MAX_UPSCALE_OUTPUT_SIDE = 8192


def ensure_upscaler_available(s: dict, on_progress=None) -> tuple[bool, str]:
    """Install the runtime when needed; an upscale needs no diffusion model.
    Never raises."""
    say = _say(on_progress)
    try:
        rt = _ensure_runtime(s, say)
    except sd_runtime.ProvisionError as e:
        return False, f"The native image runtime is not available: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"The native image runtime could not be prepared: {e}"
    return True, f"Native upscaling ready (stable-diffusion.cpp, {rt.backend})."


def upscaler_vram_estimate_bytes(upscaler: str) -> Optional[int]:
    """Peak VRAM estimate for upscaling with registered model *upscaler*, or
    None when its file cannot be resolved."""
    try:
        size = _resolve_file(upscaler, "upscaler").stat().st_size
    except (OSError, _ModelError):
        return None
    return int(size * 1.2) + COMPUTE_ALLOWANCE_BYTES


def _load_pixels(path: Path) -> dict:
    from PIL import Image
    with Image.open(path) as im:
        if im.width > MAX_UPSCALE_INPUT_SIDE or im.height > MAX_UPSCALE_INPUT_SIDE:
            raise ValueError(f"The image is {im.width}x{im.height}; at most "
                             f"{MAX_UPSCALE_INPUT_SIDE} pixels per side can be upscaled.")
        im.load()
        mode = "RGBA" if "A" in im.getbands() else "RGB"
        im = im.convert(mode)
        return {"width": im.width, "height": im.height,
                "channel": 4 if mode == "RGBA" else 3, "data": im.tobytes()}


def upscale(s: dict, source: Path, upscaler: str, out_path: Path, *,
            extra_meta: Optional[dict] = None, write_sidecar: bool = True,
            cancel_check=None, on_progress=None,
            on_saved: Optional[Callable[[Path], None]] = None,
            on_queue_state: Optional[Callable[[str], None]] = None) -> tuple[bool, str]:
    """Upscale the image *source* with the registered ESRGAN model *upscaler*
    into the PNG *out_path*. Returns ``(ok, message)``. The sidecar is
    *extra_meta* (the source's own) with the new size and an ``upscaled``
    record. Never raises: an unexpected error becomes ``(False, message)``."""
    try:
        return _upscale(s, Path(source), upscaler, Path(out_path), extra_meta or {},
                        write_sidecar, cancel_check, on_progress, on_saved, on_queue_state)
    except Exception as e:  # noqa: BLE001
        from localm.debuglog import logger
        logger.warning("native upscale failed: %s: %s", type(e).__name__, e)
        return False, f"Native upscaling failed: {type(e).__name__}: {e}"


def _upscale(s, source, upscaler, out_path, extra_meta, write_sidecar, cancel_check,
             on_progress, on_saved, on_queue_state) -> tuple[bool, str]:
    say = _say(on_progress)
    try:
        model_file = _resolve_file(upscaler, "upscaler")
        rt = _ensure_runtime(s, say)
    except sd_runtime.ProvisionError as e:
        return False, f"The native image runtime is not available: {e}"
    except _ModelError as e:
        return False, str(e)
    try:
        pixels = _load_pixels(source)
    except (OSError, ValueError) as e:
        return False, f"The image could not be read: {e}"
    relay = _event_relay(say)
    runner = shared.runner
    started = time.monotonic()
    with shared.held(say, cancel_check, on_queue_state) as acquired:
        if not acquired:
            return False, "Cancelled."
        shared.cancel_idle_timer()
        try:
            say(f"Upscaling {pixels['width']}x{pixels['height']} with {display_name(upscaler)} "
                f"(stable-diffusion.cpp, {rt.backend})...")
            result = runner.upscale_image(
                {"model_path": str(model_file), "image": pixels,
                 "max_output_side": MAX_UPSCALE_OUTPUT_SIDE}, rt.path,
                extra_dirs=rt.extra_dirs, on_event=relay, cancel_check=cancel_check)
        except SdCancelled:
            return False, "Cancelled."
        except SdWorkerError as e:
            return False, f"Native upscaling failed: {e}"
        finally:
            relay.flush()
            if runner.is_alive():
                shared.arm_idle_timer()
    record = dict(extra_meta)
    record.update({
        "width": result["width"], "height": result["height"],
        "upscaled": {"from": source.name, "model": display_name(upscaler),
                     "scale": result["scale"]},
        "runtime": f"stable-diffusion.cpp {sd_runtime.pins.TAG} ({rt.backend})",
    })
    entries = (png_text_entries(record)
               if _embed_params(_native_block(s), write_sidecar) else None)
    try:
        _write_png(out_path, result, entries)
    except (OSError, ValueError) as e:
        return False, f"The upscaled image could not be saved: {e}"
    elapsed = time.monotonic() - started
    if write_sidecar:
        warning = _write_sidecar(out_path, {
            **record, "elapsed_seconds": round(elapsed, 1),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        if warning:
            say(warning)
    if on_saved is not None:
        on_saved(out_path)
    return True, (f"Saved {out_path.name} ({result['width']}x{result['height']}, "
                  f"{result['scale']}x, {elapsed:.1f} s).")
