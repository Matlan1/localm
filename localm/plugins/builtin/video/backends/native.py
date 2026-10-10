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
from localm.media.sdcpp.runner import SdCancelled, SdWorkerError

COMPUTE_ALLOWANCE_BYTES = 4 * 1024 ** 3

MIN_SIDE = 64
MAX_SIDE = 1920
MAX_FRAMES = 241


@dataclass(frozen=True)
class RecommendedPart:
    """One file of a curated model set."""
    role: str
    repo: str
    file: str
    size_bytes: int

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
                            2_838_303_560),
            RecommendedPart("vae", "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                            "split_files/vae/wan_2.1_vae.safetensors", 253_815_318),
            RecommendedPart("t5xxl", "city96/umt5-xxl-encoder-gguf",
                            "umt5-xxl-encoder-Q4_K_M.gguf", 3_655_145_312),
        ),
        steps=30, cfg_scale=6.0, flow_shift=3.0, width=832, height=480,
        negative_prompt="static, blurry, low quality, watermark, text"),
)

_COMPONENT_FIELDS = (("t5xxl", "t5xxl_path"), ("llm", "llm_path"),
                     ("clip_vision", "clip_vision_path"), ("vae", "vae_path"),
                     ("high_noise_model", "high_noise_diffusion_model_path"))


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


def _resolve_file(value: str, what: str) -> Path:
    from localm.model_manager.registry import get_operator_model_info
    info = get_operator_model_info(str(value))
    if info is None:
        raise _ModelError(f"The native video {what} '{value}' is neither a registered "
                          "model nor a file on this machine.")
    path = Path(info[0])
    if not path.is_file():
        raise _ModelError(f"The native video {what} '{value}' is not a single model file "
                          f"({path}).")
    return path


def _registered_file(filename: str) -> Optional[tuple[str, Path]]:
    from localm.model_manager import load_registry
    for name, entry in load_registry().items():
        path = entry.get("path") if isinstance(entry, dict) else None
        if path and Path(path).name == filename and Path(path).is_file():
            return name, Path(path)
    return None


def missing_parts(rec: RecommendedVideoModel) -> list[RecommendedPart]:
    """The parts of *rec* not yet in the registry."""
    return [p for p in rec.parts if _registered_file(p.filename) is None]


def missing_model_message() -> str:
    rec = RECOMMENDED_VIDEO_MODELS[0]
    total = sum(p.size_bytes for p in rec.parts) / 1024 ** 3
    pulls = "; ".join(f"localm pull {p.spec}" for p in rec.parts)
    return (f"No native video model is set up. The recommended one is {rec.name} "
            f"({total:.1f} GB in {len(rec.parts)} files): {pulls}. Or set 'Native video "
            "model' and its text encoder and VAE in Settings > Video.")


def resolve_models(s: dict) -> dict:
    """The model files the native backend would load for settings *s*:
    ``{"key", "ctx", "label", "recommended"}``. Raises ``_ModelError`` with a
    user-facing reason when they cannot be resolved."""
    blk = _native_block(s)
    model = (blk.get("model") or "").strip()
    ctx: dict = {}
    rec = None
    if model:
        ctx["diffusion_model_path"] = str(_resolve_file(model, "model"))
        label = model
        for key, field in _COMPONENT_FIELDS:
            value = (blk.get(key) or "").strip()
            if value:
                ctx[field] = str(_resolve_file(value, key.replace("_", "-")))
    else:
        rec = RECOMMENDED_VIDEO_MODELS[0]
        found = {p.role: _registered_file(p.filename) for p in rec.parts}
        if any(v is None for v in found.values()):
            raise _ModelError(missing_model_message())
        fields = {"model": "diffusion_model_path", "vae": "vae_path", "t5xxl": "t5xxl_path"}
        for role, hit in found.items():
            ctx[fields[role]] = str(hit[1])
        label = rec.name
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
    anything (the same shape as the image backend's ``status``)."""
    rt = sd_runtime.resolve(_runtime_choice(s))
    try:
        models = resolve_models(s)
        model, missing = models["label"], None
    except _ModelError as e:
        model, missing = None, str(e)
    rec = RECOMMENDED_VIDEO_MODELS[0]
    return {
        "runtime": rt.backend if rt else None,
        "runtime_choice": _runtime_choice(s),
        "model": model,
        "missing": missing,
        "loaded": shared.worker_pid() is not None,
        "recommended": {"name": rec.name, "parts": [
            {"role": p.role, "repo": p.repo, "file": p.file, "spec": p.spec,
             "size_bytes": p.size_bytes} for p in missing_parts(rec)]},
    }


def free_vram(s: dict) -> bool:
    """Stop the worker so every byte of its VRAM is released."""
    return shared.free()


def vram_estimate_bytes(s: dict) -> Optional[int]:
    """Peak VRAM estimate for the configured model files (their size plus a
    fixed compute allowance), or None when they cannot be resolved."""
    try:
        models = resolve_models(s)
    except _ModelError:
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


def generate(s: dict, prompt: str, out_path: Path, *,
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
             **_unused) -> tuple[bool, str]:
    """Generate one clip of *prompt* into *out_path* (MP4, H.264). Returns
    ``(ok, message)``; never raises. Writes only *out_path* and, when
    *write_sidecar*, ``<out_path>.json``.

    ``model_overrides`` and ``placement`` are ComfyUI-only and refused with a
    reason; ``delete_outputs`` has nothing to act on."""
    say = _say(on_progress)
    if _unused:
        return False, ("The native video backend does not support: "
                       + ", ".join(sorted(_unused)) + ".")
    if model_overrides:
        return False, ("Workflow model choices apply to ComfyUI. The native backend uses "
                       "the model set in Settings > Video.")
    if placement:
        return False, ("Per-component GPU placement applies to ComfyUI only; turn it off or "
                       "use the ComfyUI backend.")
    if (width is None) != (height is None):
        return False, "Give both width and height, or neither."
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
    if width is None:
        width = blk.get("width") or (rec.width if rec else 832)
        height = blk.get("height") or (rec.height if rec else 480)
    width, height = int(width), int(height)
    if not (MIN_SIDE <= width <= MAX_SIDE and MIN_SIDE <= height <= MAX_SIDE) \
            or width % 16 or height % 16:
        return False, (f"Video size {width}x{height} must be {MIN_SIDE}..{MAX_SIDE} pixels "
                       "per side and a multiple of 16.")
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
