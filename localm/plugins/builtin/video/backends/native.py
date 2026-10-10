# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native video backend: stable-diffusion.cpp video models (Wan2.1/2.2 and
others it supports) in the shared sd.cpp worker, written to MP4 with PyAV.

Exposes the media backend seam (``ensure_available`` / ``free_vram`` /
``generate``). The worker is the process-wide one in
``localm.media.sdcpp.shared``, also used by the native image backend; a job
that needs a different model set replaces the loaded one.
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
from localm.media.sdcpp.files import ModelFileError as _ModelError
from localm.media.sdcpp.files import display_name, registered_file, resolve_model_file
from localm.media.sdcpp.runner import SdCancelled, SdWorkerError

COMPUTE_ALLOWANCE_BYTES = 4 * 1024 ** 3

MIN_SIDE = 64
MAX_SIDE = 1920
MAX_FRAMES = 241
DEFAULT_WIDTH, DEFAULT_HEIGHT = 832, 480


@dataclass(frozen=True)
class RecommendedPart:
    """One file of a curated model set."""
    role: str
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


@dataclass(frozen=True)
class RecommendedVideoModel:
    """A curated public video model set the native backend can use as is."""
    name: str
    parts: tuple[RecommendedPart, ...]
    steps: int
    cfg_scale: float
    flow_shift: float
    width: int
    height: int
    negative_prompt: str


RECOMMENDED_VIDEO_MODELS: tuple[RecommendedVideoModel, ...] = (
    RecommendedVideoModel(
        name="wan2.1-t2v-1.3b",
        parts=(
            RecommendedPart("model", "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                            "split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors",
                            2_838_303_560,
                            "be531024cd9018cb5b48c40cfbb6a6191645b1c792eb8bf4f8c1c6e10f924dc5",
                            "diffusion-unet"),
            RecommendedPart("vae", "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                            "split_files/vae/wan_2.1_vae.safetensors", 253_815_318,
                            "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
                            "vae"),
            RecommendedPart("t5xxl", "city96/umt5-xxl-encoder-gguf",
                            "umt5-xxl-encoder-Q4_K_M.gguf", 3_655_145_312,
                            "17cf97a5bbbc60a646d6105b832b6f657ce904a8a1ad970e4b59df0c67584a40",
                            "text-encoder"),
        ),
        steps=30, cfg_scale=6.0, flow_shift=3.0, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
        negative_prompt="static, blurry, low quality, watermark, text"),
)

_COMPONENT_FIELDS = (("t5xxl", "t5xxl_path"), ("llm", "llm_path"),
                     ("clip_vision", "clip_vision_path"), ("vae", "vae_path"))
_PART_FIELDS = {"model": "diffusion_model_path", "vae": "vae_path", "t5xxl": "t5xxl_path"}


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
    """The file a native video model setting names (``resolve_model_file``)."""
    return resolve_model_file(value, f"video {what}")


def missing_parts(rec: RecommendedVideoModel) -> list[RecommendedPart]:
    """The parts of *rec* not yet in the registry."""
    return [p for p in rec.parts if registered_file(p.filename) is None]


def missing_model_message() -> str:
    rec = RECOMMENDED_VIDEO_MODELS[0]
    total = sum(p.size_bytes for p in rec.parts) / 1024 ** 3
    pulls = "; ".join(f"localm pull {p.spec} --type {p.model_type}"
                      for p in missing_parts(rec) or rec.parts)
    return (f"No native video model is set up. The recommended one is {rec.name} "
            f"({total:.1f} GB in {len(rec.parts)} files): {pulls}. Or set 'Native model' "
            "and its text encoder and VAE in Settings > Video.")


def resolve_models(s: dict) -> dict:
    """The model files the native backend would load for settings *s*:
    ``{"key", "ctx", "label", "recommended"}``. ``ctx`` also turns on
    diffusion flash attention. Raises ``_ModelError`` with a user-facing
    reason when they cannot be resolved."""
    blk = _native_block(s)
    model = (blk.get("model") or "").strip()
    ctx: dict = {}
    rec = None
    if model:
        ctx["diffusion_model_path"] = str(_resolve_file(model, "model"))
        label = display_name(model)
        for key, field in _COMPONENT_FIELDS:
            value = (blk.get(key) or "").strip()
            if value:
                ctx[field] = str(_resolve_file(value, key.replace("_", "-")))
    else:
        rec = RECOMMENDED_VIDEO_MODELS[0]
        found = {p.role: registered_file(p.filename) for p in rec.parts}
        if any(v is None for v in found.values()):
            raise _ModelError(missing_model_message())
        for role, hit in found.items():
            if hit is not None:
                ctx[_PART_FIELDS[role]] = str(hit[1])
        label = rec.name
    ctx["diffusion_flash_attn"] = True
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


def _pyav_problem() -> Optional[str]:
    try:
        import av
        av.codec.Codec("libx264", "w")
    except ImportError:
        return ("Writing the video needs PyAV, which is not installed. Install it with "
                "'uv pip install \"localm[voice]\"' (the installers include it).")
    except Exception as e:  # noqa: BLE001
        return f"PyAV cannot encode H.264 on this machine ({e})."
    return None


def ensure_available(s: dict, on_progress=None) -> tuple[bool, str]:
    """Check the model resolves and PyAV can write MP4, then install the
    runtime when needed. Never raises."""
    say = _say(on_progress)
    try:
        models = resolve_models(s)
    except _ModelError as e:
        return False, str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"The native video model could not be checked: {e}"
    problem = _pyav_problem()
    if problem:
        return False, problem
    try:
        rt = _ensure_runtime(s, say)
    except sd_runtime.ProvisionError as e:
        return False, f"The native video runtime is not available: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"The native video runtime could not be prepared: {e}"
    return True, (f"Native video backend ready (stable-diffusion.cpp, {rt.backend}): "
                  f"{models['label']}.")


def status(s: dict) -> dict:
    """What the native backend would use, without installing or loading
    anything: ``runtime``, ``runtime_choice``, ``model`` (a name or file name),
    ``missing`` (the reason it cannot run, or None), ``loaded`` and
    ``recommended`` (its name, size and the parts not yet downloaded)."""
    rt = sd_runtime.resolve(_runtime_choice(s))
    try:
        models = resolve_models(s)
        model, missing = models["label"], None
    except Exception as e:  # noqa: BLE001
        model, missing = None, str(e)
    rec = RECOMMENDED_VIDEO_MODELS[0]
    return {
        "runtime": rt.backend if rt else None,
        "runtime_choice": _runtime_choice(s),
        "model": model,
        "missing": missing,
        "loaded": shared.worker_pid() is not None,
        "recommended": {
            "name": rec.name, "width": rec.width, "height": rec.height,
            "steps": rec.steps, "cfg_scale": rec.cfg_scale,
            "size_bytes": sum(p.size_bytes for p in rec.parts),
            "parts": [{"role": p.role, "repo": p.repo, "file": p.file, "spec": p.spec,
                       "filename": p.filename, "size_bytes": p.size_bytes,
                       "sha256": p.sha256, "model_type": p.model_type}
                      for p in missing_parts(rec)]},
    }


def free_vram(s: dict) -> bool:
    """Stop the worker so every byte of its VRAM is released."""
    return shared.free()


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


def frame_count(seconds: float, fps: int) -> int:
    """Frames for *seconds* at *fps*, rounded to the 4k+1 counts video models
    take, at least 5 and at most ``MAX_FRAMES``."""
    raw = max(1, round(float(seconds) * int(fps)))
    k = max(1, round((raw - 1) / 4))
    return min(4 * k + 1, MAX_FRAMES)


def _check_size(width: int, height: int) -> Optional[str]:
    if not (MIN_SIDE <= width <= MAX_SIDE and MIN_SIDE <= height <= MAX_SIDE) \
            or width % 16 or height % 16:
        return (f"Video size {width}x{height} must be {MIN_SIDE}..{MAX_SIDE} pixels per side "
                "and a multiple of 16 for the native backend.")
    return None


def refusal(*, model_overrides=None, placement=None, width=None, height=None,
            unsupported=()) -> Optional[str]:
    """Why the native backend cannot honour a request with these inputs, or
    None. Checked before any download, VRAM handover or load. *unsupported*
    names keyword arguments the backend does not take."""
    if unsupported:
        return ("The native video backend does not support: "
                + ", ".join(sorted(unsupported)) + ".")
    if model_overrides:
        return ("Workflow model choices apply to ComfyUI. The native backend uses "
                "the model set in Settings > Video.")
    if placement:
        return ("Per-component GPU placement applies to ComfyUI only; turn it off or "
                "use the ComfyUI backend.")
    if (width is None) != (height is None):
        return "Give both width and height, or neither."
    if width is not None and height is not None:
        return _check_size(int(width), int(height))
    return None


def _write_mp4(out_path: Path, result: dict, fps: int) -> None:
    import av
    from PIL import Image
    w, h, c = result["width"], result["height"], result["channel"]
    mode = {3: "RGB", 4: "RGBA"}.get(c)
    if mode is None:
        raise ValueError(f"the runtime returned frames with {c} channels")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp.mp4")
    try:
        with av.open(str(tmp), mode="w", format="mp4") as container:
            vs = container.add_stream("libx264", rate=int(fps))
            vs.width = w - w % 2
            vs.height = h - h % 2
            vs.pix_fmt = "yuv420p"
            vs.options = {"crf": "18"}
            audio = result.get("audio")
            astream = None
            if audio:
                astream = container.add_stream("aac", rate=int(audio["sample_rate"]))
                astream.layout = "mono" if audio["channels"] == 1 else "stereo"
            for raw in result["frames"]:
                img = Image.frombytes(mode, (w, h), raw).convert("RGB")
                if (vs.width, vs.height) != (w, h):
                    img = img.crop((0, 0, vs.width, vs.height))
                frame = av.VideoFrame.from_image(img)
                for packet in vs.encode(frame):
                    container.mux(packet)
            for packet in vs.encode():
                container.mux(packet)
            if astream is not None and audio is not None:
                layout = "mono" if audio["channels"] == 1 else "stereo"
                af = av.AudioFrame(format="flt", layout=layout,
                                   samples=int(audio["sample_count"]))
                af.planes[0].update(audio["data"])
                af.sample_rate = int(audio["sample_rate"])
                for packet in astream.encode(af):
                    container.mux(packet)
                for packet in astream.encode():
                    container.mux(packet)
        os.replace(tmp, out_path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _load_init_image(path: Path, width: int, height: int) -> dict:
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        if (im.width, im.height) != (width, height):
            im = im.resize((width, height), Image.Resampling.LANCZOS)
        return {"width": width, "height": height, "channel": 3, "data": im.tobytes()}


def _event_relay(say):
    state = {"phase": None}

    def on_event(event) -> None:
        if event[0] == "log":
            say(f"stable-diffusion.cpp: {event[2]}")
            return
        info = event[1]
        phase, step, steps = info.get("phase"), info.get("step", 0), info.get("steps", 0)
        if phase == "sampling":
            secs = info.get("secs") or 0.0
            rate = f" ({secs:.1f} s/step)" if secs else ""
            say(f"Step {step}/{steps}{rate}")
        elif phase != state["phase"]:
            say("Decoding frames..." if phase == "decoding" else "Loading weights...")
        state["phase"] = phase
    return on_event


def generate(s: dict, prompt: str, out_path: Path, **kwargs) -> tuple[bool, str]:
    """Generate one clip of *prompt* into *out_path*; see ``_generate``.
    Never raises: an unexpected error becomes ``(False, message)``."""
    try:
        return _generate(s, prompt, out_path, **kwargs)
    except Exception as e:  # noqa: BLE001
        from localm.debuglog import logger
        logger.warning("native video generation failed: %s: %s", type(e).__name__, e)
        return False, f"Native video generation failed: {type(e).__name__}: {e}"


def _generate(s: dict, prompt: str, out_path: Path, *,
              self_url: Optional[str] = None,
              write_sidecar: bool = True,
              instance_token: Optional[str] = None,
              negative_prompt: Optional[str] = None,
              seconds: float = 5.0,
              fps: int = 16,
              width: Optional[int] = None,
              height: Optional[int] = None,
              steps: Optional[int] = None,
              cfg: Optional[float] = None,
              seed: Optional[int] = None,
              input_image: Optional[Path] = None,
              model_overrides: Optional[dict] = None,
              swap: bool = False,
              delete_outputs: Optional[bool] = None,
              cancel_check=None,
              placement: Optional[dict] = None,
              on_progress=None,
              **unsupported) -> tuple[bool, str]:
    """Generate one clip of *prompt* into *out_path* (MP4, H.264, with an AAC
    track when the model returns audio). Returns ``(ok, message)``. Writes only
    *out_path* and, when *write_sidecar*, ``<out_path>.json``.

    ComfyUI-only inputs (``model_overrides``, ``placement``) and keyword
    arguments this backend does not take are refused with a reason.
    ``delete_outputs`` has nothing to act on (there is no second copy). *swap*
    asks this backend to unload the chat model itself, used when the caller's
    own unload did not succeed."""
    say = _say(on_progress)
    refused = refusal(model_overrides=model_overrides, placement=placement,
                      width=width, height=height, unsupported=tuple(unsupported))
    if refused:
        return False, refused
    started = time.monotonic()
    try:
        models = resolve_models(s)
        rt = _ensure_runtime(s, say)
    except sd_runtime.ProvisionError as e:
        return False, f"The native video runtime is not available: {e}"
    except _ModelError as e:
        return False, str(e)
    problem = _pyav_problem()
    if problem:
        return False, problem
    blk = _native_block(s)
    rec = models["recommended"]
    if width is None or height is None:
        width, height = (rec.width, rec.height) if rec else (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    width, height = int(width), int(height)
    frames = frame_count(seconds, fps)
    if swap and self_url:
        from localm.media.comfy_client import _localm_unload
        say("Freeing VRAM: unloading the chat model...")
        if _localm_unload(self_url, instance_token) is None:
            say("Could not unload the chat model - the video model may run low on VRAM.")
    relay = _event_relay(say)
    runner = shared.runner
    if seed is None or seed < 0:
        seed = secrets.randbelow(2 ** 31)
    steps = steps or blk.get("steps") or (rec.steps if rec else None)
    cfg_scale = cfg if cfg is not None else (
        blk.get("cfg_scale") if blk.get("cfg_scale") is not None
        else (rec.cfg_scale if rec else None))
    flow_shift = blk.get("flow_shift") if blk.get("flow_shift") is not None else (
        rec.flow_shift if rec else None)
    negative = negative_prompt if negative_prompt is not None else (
        rec.negative_prompt if rec else "")
    with shared.lock:
        shared.cancel_idle_timer()
        try:
            if runner.loaded_key != models["key"] or not runner.is_alive():
                say(f"Loading {models['label']} (stable-diffusion.cpp, {rt.backend})...")
            info = runner.ensure_loaded(models["key"], rt.path, models["ctx"],
                                        extra_dirs=rt.extra_dirs, on_event=relay,
                                        cancel_check=cancel_check)
            if not info.get("video"):
                runner.shutdown()
                return False, (f"{models['label']} ({info.get('version') or 'unknown'}) is not "
                               "a video generation model.")
            init = _load_init_image(Path(input_image), width, height) if input_image else None
            params = {
                "prompt": prompt, "negative_prompt": negative,
                "width": width, "height": height, "seed": int(seed),
                "video_frames": frames, "fps": int(fps),
                "steps": steps, "cfg_scale": cfg_scale, "flow_shift": flow_shift,
                "sample_method": blk.get("sample_method") or None,
                "init_image": init,
            }
            say(f"Generating {frames} frames at {width}x{height}"
                + (f", {steps} steps" if steps else "") + f", seed {seed}...")
            result = runner.generate_video(params, on_event=relay, cancel_check=cancel_check)
        except SdCancelled:
            return False, "Cancelled."
        except SdWorkerError as e:
            return False, f"Native video generation failed: {e}"
        except (OSError, ValueError) as e:
            return False, f"Native video generation failed: {e}"
        finally:
            if runner.is_alive():
                shared.arm_idle_timer()
    out_fps = int(result.get("fps") or fps)
    say(f"Encoding {len(result['frames'])} frames to MP4...")
    try:
        _write_mp4(Path(out_path), result, out_fps)
    except Exception as e:  # noqa: BLE001
        return False, f"The generated video could not be written: {e}"
    elapsed = time.monotonic() - started
    warning = ""
    if write_sidecar:
        try:
            Path(out_path).with_suffix(Path(out_path).suffix + ".json").write_text(json.dumps(
                {k: v for k, v in {
                    "prompt": prompt, "negative_prompt": negative or None, "seed": int(seed),
                    "cfg": cfg_scale, "steps": steps, "flow_shift": flow_shift,
                    "width": result["width"], "height": result["height"],
                    "frames": len(result["frames"]), "fps": out_fps,
                    "seconds": round(len(result["frames"]) / out_fps, 2),
                    "input_image": str(input_image) if input_image else None,
                    "backend": "native", "model": models["label"],
                    "runtime": f"stable-diffusion.cpp {sd_runtime.pins.TAG} ({rt.backend})",
                    "elapsed_seconds": round(elapsed, 1),
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }.items() if v is not None}, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            warning = (f"WARNING: the reproducibility sidecar could not be saved ({e}); "
                       "the video itself was saved.")
    message = (f"Saved {Path(out_path).name} ({len(result['frames'])} frames, "
               f"{result['width']}x{result['height']} at {out_fps} fps, seed {seed}, "
               f"{elapsed:.0f} s).")
    return True, f"{message} {warning}".strip()
