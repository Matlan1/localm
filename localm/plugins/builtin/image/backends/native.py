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

from localm.media.sdcpp import runtime as sd_runtime
from localm.media.sdcpp import shared
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

_COMPONENT_FIELDS = (("clip_l", "clip_l_path"), ("clip_g", "clip_g_path"),
                     ("t5xxl", "t5xxl_path"), ("llm", "llm_path"), ("vae", "vae_path"))

class _ModelError(Exception):
    pass


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


def display_name(value: str) -> str:
    """*value* as shown to users: a registered model name unchanged, a file path
    reduced to its file name."""
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def _resolve_file(value: str, what: str) -> Path:
    """The file a native model setting names: a registered model, else a local
    path. UNC and device paths are refused without touching the filesystem.
    Raises ``_ModelError``; messages name the file, never its directory."""
    from localm.model_manager.registry import get_model_info
    from localm.pathsafe import is_unc_or_device_path
    shown = display_name(value)
    info = get_model_info(str(value))
    raw = str(info[0]) if info is not None else str(value)
    if is_unc_or_device_path(raw):
        raise _ModelError(f"The native image {what} '{shown}' is a network or device "
                          "path, which is not allowed.")
    path = Path(raw).expanduser()
    try:
        exists, is_file = path.exists(), path.is_file()
    except OSError as e:
        raise _ModelError(f"The native image {what} '{shown}' cannot be read "
                          f"({type(e).__name__}).") from e
    if not exists:
        raise _ModelError(f"The native image {what} '{shown}' is neither a registered "
                          "model nor a file on this machine.")
    if not is_file:
        raise _ModelError(f"The native image {what} '{shown}' is not a single model file.")
    return path


def _recommended_in_registry() -> Optional[tuple[str, Path, RecommendedModel]]:
    from localm.model_manager import load_registry
    reg = load_registry()
    for rec in RECOMMENDED_MODELS:
        for name, entry in reg.items():
            path = entry.get("path") if isinstance(entry, dict) else None
            if path and Path(path).name == rec.file and Path(path).is_file():
                return name, Path(path), rec
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
            f"'localm pull {rec.spec} --type diffusion-unet', or set "
            "'Native image model' in Settings > Images to a model you have.")


def resolve_models(s: dict) -> dict:
    """The model files the native backend would load for settings *s*:
    ``{"key", "ctx", "label", "recommended"}``. Raises ``_ModelError`` with a
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
    ctx: dict = {}
    components = {}
    for key, field in _COMPONENT_FIELDS:
        value = (blk.get(key) or "").strip()
        if value:
            components[field] = str(_resolve_file(value, key.replace("_", "-")))
    has_encoders = any(f in components for f in ("clip_l_path", "clip_g_path",
                                                 "t5xxl_path", "llm_path"))
    if has_encoders:
        ctx["diffusion_model_path"] = str(main)
    else:
        ctx["model_path"] = str(main)
    ctx.update(components)
    key = tuple(sorted(ctx.items()))
    return {"key": key, "ctx": ctx, "label": label, "recommended": rec}


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
    try:
        models = resolve_models(s)
        model, missing = models["label"], None
    except Exception as e:  # noqa: BLE001
        model, missing = None, str(e)
    rec = RECOMMENDED_MODELS[0]
    return {
        "runtime": rt.backend if rt else None,
        "runtime_choice": _runtime_choice(s),
        "model": model,
        "missing": missing,
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


def _default_size(version: str, rec: Optional[RecommendedModel], blk: dict) -> tuple[int, int]:
    w, h = blk.get("width"), blk.get("height")
    if isinstance(w, int) and isinstance(h, int):
        return w, h
    if rec is not None:
        return rec.width, rec.height
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


def _write_png(out_path: Path, result: dict) -> None:
    from PIL import Image
    mode = {3: "RGB", 4: "RGBA", 1: "L"}.get(result["channel"])
    if mode is None:
        raise ValueError(f"the runtime returned an image with {result['channel']} channels")
    img = Image.frombytes(mode, (result["width"], result["height"]), result["data"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    try:
        img.save(tmp, format="PNG")
        os.replace(tmp, out_path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _write_sidecar(out_path: Path, record: dict) -> str:
    try:
        out_path.with_suffix(out_path.suffix + ".json").write_text(
            json.dumps({k: v for k, v in record.items() if v is not None},
                       indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        return (f"WARNING: the reproducibility sidecar could not be saved ({e}); "
                "the image itself was saved.")
    return ""


def _event_relay(say):
    state: dict = {"phase": None}

    def on_event(event) -> None:
        if event[0] == "log":
            say(f"stable-diffusion.cpp: {event[2]}")
            return
        info = event[1]
        phase, step, steps = info.get("phase"), info.get("step", 0), info.get("steps", 0)
        if phase == "sampling":
            secs = info.get("secs") or 0.0
            rate = f" ({secs:.2f} s/step)" if secs else ""
            say(f"Step {step}/{steps}{rate}")
        elif phase != state["phase"] or (steps and step == steps):
            label = "Decoding" if phase == "decoding" else "Loading weights"
            if steps and step == steps:
                say(f"{label}: done")
            elif phase != state["phase"]:
                say(f"{label}...")
        state["phase"] = phase
    return on_event


def refusal(*, model_overrides=None, lora_name=None, placement=None,
            width=None, height=None, **_ignored) -> Optional[str]:
    """Why the native backend cannot honour a request with these inputs, or
    None. Checked before any download, VRAM handover or load."""
    if model_overrides:
        return ("Workflow model choices apply to ComfyUI. The native backend uses "
                "the model set in Settings > Images.")
    if lora_name:
        return ("LoRAs are not supported by the native image backend yet; "
                "use the ComfyUI backend for LoRA generation.")
    if placement:
        return ("Per-component GPU placement applies to ComfyUI only; turn it off or "
                "use the ComfyUI backend.")
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
             height: Optional[int] = None) -> tuple[bool, str]:
    """Generate one image of *prompt* into *out_path* (a PNG without embedded
    metadata). Returns ``(ok, message)``. Writes only
    *out_path* and, when *write_sidecar*, ``<out_path>.json``.

    ComfyUI-only inputs are refused with a reason rather than ignored:
    ``model_overrides`` (workflow model slots), ``lora_name`` and ``placement``.
    ``delete_outputs`` has nothing to act on (there is no second copy).
    *swap* asks this backend to unload the chat model itself, used when the
    caller's own unload did not succeed."""
    say = _say(on_progress)
    refused = refusal(model_overrides=model_overrides, lora_name=lora_name,
                      placement=placement, width=width, height=height)
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
    if swap and self_url:
        from localm.media.comfy_client import _localm_unload
        say("Freeing VRAM: unloading the chat model...")
        if _localm_unload(self_url, instance_token) is None:
            say("Could not unload the chat model - the image model may run low on VRAM.")
    relay = _event_relay(say)
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
                width, height = _default_size(info.get("version", ""), rec, blk)
            if seed is None or seed < 0:
                seed = secrets.randbelow(2 ** 31)
            steps = blk.get("steps") or (rec.steps if rec else None)
            cfg_scale = cfg if cfg is not None else (
                blk.get("cfg_scale") if blk.get("cfg_scale") is not None
                else (rec.cfg_scale if rec else None))
            params = {
                "prompt": prompt, "negative_prompt": negative_prompt or "",
                "width": int(width), "height": int(height), "seed": int(seed),
                "steps": steps, "cfg_scale": cfg_scale, "guidance": guidance,
                "sample_method": blk.get("sample_method") or None,
                "scheduler": blk.get("scheduler") or None,
                "init_image": init,
                "strength": (denoise if denoise is not None else 0.75) if init else None,
            }
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
            if runner.is_alive():
                shared.arm_idle_timer()
    try:
        _write_png(Path(out_path), result)
    except (OSError, ValueError) as e:
        return False, f"The generated image could not be saved: {e}"
    elapsed = time.monotonic() - started
    warning = ""
    if write_sidecar:
        warning = _write_sidecar(Path(out_path), {
            "prompt": prompt, "negative_prompt": negative_prompt or None,
            "seed": int(seed), "cfg": cfg_scale, "guidance": guidance, "steps": steps,
            "width": result["width"], "height": result["height"],
            "input_image": str(input_image) if input_image else None,
            "denoise": params["strength"],
            "backend": "native", "model": models["label"],
            "runtime": f"stable-diffusion.cpp {sd_runtime.pins.TAG} ({rt.backend})",
            "elapsed_seconds": round(elapsed, 1),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
    message = (f"Saved {Path(out_path).name} ({result['width']}x{result['height']}, "
               f"seed {seed}, {elapsed:.1f} s).")
    return True, f"{message} {warning}".strip()
