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
from localm.media.sdcpp import families, pairing
from localm.media.sdcpp import runtime as sd_runtime
from localm.media.sdcpp import shared
from localm.media.sdcpp.logrelay import RecoveredWarnings
from localm.media.sdcpp.files import ModelFileError as _ModelError
from localm.media.sdcpp.files import display_name, registered_file, resolve_model_file
from localm.media.sdcpp.runner import SdCancelled, SdWorkerError

COMPUTE_ALLOWANCE_BYTES = 2 * 1024 ** 3

MIN_SIDE = 64
MAX_SIDE = 2048


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
                     ("t5xxl", "t5xxl_path"), ("llm", "llm_path"), ("vae", "vae_path"))

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
    "notes", "defaults"}`` (see ``pairing.build_context``; ``defaults`` is the
    family's ``families.Defaults`` or None). A diffusion model's missing text
    encoders and VAE are picked from the registry. Raises ``_ModelError``
    (``pairing.MissingComponents`` when a companion file is missing) with a
    user-facing reason when they cannot be resolved."""
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
    key = tuple(sorted(ctx.items()))
    info = built["info"]
    return {"key": key, "ctx": ctx, "label": label, "recommended": rec,
            "family": info.family, "components": built["components"],
            "notes": built["notes"], "defaults": families.defaults_for(info)}


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


def _load_init_image(path: Path, width: Optional[int], height: Optional[int]) -> dict:
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        if width is None or height is None:
            width, height = fit_size(im.width, im.height)
        if (im.width, im.height) != (width, height):
            im = im.resize((width, height), Image.Resampling.LANCZOS)
        return {"width": width, "height": height, "channel": 3, "data": im.tobytes()}


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
        elif phase != self._phase or (steps and step == steps):
            label = "Decoding" if phase == "decoding" else "Loading weights"
            if steps and step == steps:
                self._say(f"{label}: done")
            elif phase != self._phase:
                self._say(f"{label}...")
        self._phase = phase


def _event_relay(say, on_preview=None) -> _EventRelay:
    return _EventRelay(say, on_preview)


def refusal(*, model_overrides=None, lora_name=None,
            width=None, height=None, **_ignored) -> Optional[str]:
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
    if (width is None) != (height is None):
        return "Give both width and height, or neither."
    if width is not None and height is not None:
        return _check_size(int(width), int(height))
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
             steps: Optional[int] = None,
             clip_skip: Optional[int] = None,
             on_preview=None) -> tuple[bool, str]:
    """Generate one image of *prompt* into *out_path* (a PNG without embedded
    metadata unless the ``embed_params`` setting is on and the session is not in
    privacy mode). Returns ``(ok, message)``. Writes only
    *out_path* and, when *write_sidecar*, ``<out_path>.json``.

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
    in-progress frame the worker makes."""
    say = _say(on_progress)
    refused = refusal(model_overrides=model_overrides, lora_name=lora_name,
                      width=width, height=height)
    if refused:
        return False, refused
    started = time.monotonic()
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
    if any(c.get("auto") for c in (models.get("components") or {}).values()):
        say("Using " + pairing.describe_components(models["components"]) + ".")
    if swap and self_url:
        from localm.media.comfy_client import _localm_unload
        say("Freeing VRAM: unloading the chat model...")
        if _localm_unload(self_url, instance_token) is None:
            say("Could not unload the chat model - the image model may run low on VRAM.")
    relay = _event_relay(say, on_preview)
    runner = shared.runner
    with shared.lock:
        shared.cancel_idle_timer()
        try:
            if runner.loaded_key != models["key"] or not runner.is_alive():
                say(f"Loading {models['label']} (stable-diffusion.cpp, {rt.backend})...")
            info = runner.ensure_loaded(models["key"], rt.path, models["ctx"],
                                         extra_dirs=rt.extra_dirs, on_event=relay,
                                         cancel_check=cancel_check)
            if not info.get("image"):
                runner.shutdown()
                return False, (f"{models['label']} ({info.get('version') or 'unknown'}) is not "
                               "an image generation model.")
            init = None
            if input_image is not None:
                init = _load_init_image(Path(input_image), width, height)
                width, height = init["width"], init["height"]
            if width is None or height is None:
                width, height = _default_size(info.get("version", ""), rec, blk, defaults)
            if seed is None or seed < 0:
                seed = secrets.randbelow(2 ** 31)
            steps = steps or blk.get("steps") or (rec.steps if rec else None) or (
                defaults.steps if defaults else None)
            cfg_scale = cfg if cfg is not None else (
                blk.get("cfg_scale") if blk.get("cfg_scale") is not None
                else (rec.cfg_scale if rec else (defaults.cfg_scale if defaults else None)))
            params = {
                "prompt": prompt, "negative_prompt": negative_prompt or "",
                "width": int(width), "height": int(height), "seed": int(seed),
                "steps": steps, "cfg_scale": cfg_scale, "guidance": guidance,
                "clip_skip": clip_skip, "preview": on_preview is not None,
                "sample_method": blk.get("sample_method") or None,
                "scheduler": blk.get("scheduler") or None,
                "init_image": init,
                "strength": (denoise if denoise is not None else 0.75) if init else None,
                "loras": [{"path": x["path"], "multiplier": x["multiplier"]}
                          for x in lora_files],
            }
            if negative_prompt and cfg_scale is not None and abs(float(cfg_scale) - 1.0) < 1e-5:
                say(NEGATIVE_AT_CFG_1)
            say(f"Generating {width}x{height}"
                + (f", {steps} steps" if steps else "") + f", seed {seed}...")
            result = runner.generate_image(params, on_event=relay, cancel_check=cancel_check)
        except SdCancelled:
            return False, "Cancelled."
        except SdWorkerError as e:
            return False, f"Native image generation failed: {e}"
        except (OSError, ValueError) as e:
            return False, f"Native image generation failed: {e}"
        finally:
            relay.flush()
            if runner.is_alive():
                shared.arm_idle_timer()
    record = {
        "prompt": prompt, "negative_prompt": negative_prompt or None,
        "seed": int(seed), "cfg": cfg_scale, "guidance": guidance, "steps": steps,
        "clip_skip": clip_skip, "width": result["width"], "height": result["height"],
        "input_image": str(input_image) if input_image else None,
        "denoise": params["strength"],
        "backend": "native", "model": models["label"],
        "components": {k: v["name"] for k, v in (models.get("components") or {}).items()}
        or None,
        "loras": [{"name": x["name"], "strength": x["multiplier"]} for x in lora_files]
        or None,
        "runtime": f"stable-diffusion.cpp {sd_runtime.pins.TAG} ({rt.backend})",
    }
    entries = png_text_entries(record) if _embed_params(blk, write_sidecar) else None
    try:
        _write_png(Path(out_path), result, entries)
    except (OSError, ValueError) as e:
        return False, f"The generated image could not be saved: {e}"
    elapsed = time.monotonic() - started
    warning = ""
    if write_sidecar:
        warning = _write_sidecar(Path(out_path), {
            **record, "elapsed_seconds": round(elapsed, 1),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
    message = (f"Saved {Path(out_path).name} ({result['width']}x{result['height']}, "
               f"seed {seed}, {elapsed:.1f} s).")
    return True, f"{message} {warning}".strip()
