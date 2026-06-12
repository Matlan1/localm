"""
ComfyUI Wan 2.2 short-video generation.

Mirrors localm.music_gen.comfy: standalone module, reachable from the GUI,
the CLI, or any other caller.  Uses the same ComfyUI server as image and
music generation — the model files for the committed template (the public
Wan 2.2 TI2V 5B stack, ComfyUI v0.3.46+) live in ComfyUI's model dirs:

    models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors
    models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
    models/vae/wan2.2_vae.safetensors

Unlike ACE-Step audio, video length is NOT arbitrary: the model attends over
all frames at once, so VRAM and time grow with the frame count and quality
degrades well past the native ~5 s.  Wan requires 4k+1 frames; the requested
duration is snapped to the nearest valid count (121 frames = 5 s at 24 fps).

The same prompt is text-to-video by default; pass *input_image* to animate an
existing picture (image-to-video) instead.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

# Shared ComfyUI plumbing lives in image_gen — one server, one set of helpers
from localm.image_gen.comfy import (
    _localm_unload,
    _upload_image,
    comfy_http_error_detail,
    default_api_url,
    ensure_comfy,
)

# wan_workflow.json is the committed generic template (public Wan 2.2 5B
# stack).  Drop a wan_workflow_local.json next to it (gitignored) to use
# your own checkpoint/graph without publishing which models you run.  A
# local graph must keep the template's node ids (4 = positive prompt,
# 5 = negative, 6 = video latent, 8 = sampler, 10 = CreateVideo).
_WORKFLOW_PATH = Path(__file__).parent / "wan_workflow.json"
_WORKFLOW_LOCAL_PATH = Path(__file__).parent / "wan_workflow_local.json"


def _workflow_path() -> Path:
    return _WORKFLOW_LOCAL_PATH if _WORKFLOW_LOCAL_PATH.is_file() else _WORKFLOW_PATH


def _snap_frames(seconds: float, fps: int) -> int:
    """Wan generates 4k+1 frames; snap the requested duration to the
    nearest valid count (minimum 5 frames)."""
    frames = round(seconds * fps)
    return max(5, 4 * round((frames - 1) / 4) + 1)


# Output keys that ComfyUI save nodes use in /history, by node family:
# SaveVideo/SaveWEBM report under "images" (with an animated flag),
# VHS_VideoCombine under "gifs"; newer builds may use "videos".
_OUTPUT_KEYS = ("videos", "gifs", "images")


def generate_video(
    prompt: str,
    output_path: Path,
    *,
    negative_prompt: Optional[str] = None,
    seconds: float = 5.0,
    fps: int = 24,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    seed: Optional[int] = None,
    input_image: Optional[Path] = None,
    api_url: Optional[str] = None,
    localm_url: Optional[str] = None,
    max_poll_seconds: int = 3600,
    on_progress=None,
    write_sidecar: bool = True,
) -> tuple[bool, str]:
    """
    Generate a short video clip and save it to *output_path* (MP4).

    Parameters
    ----------
    prompt
        Scene description — subject, motion, camera, lighting.  Motion verbs
        matter; a static description tends to produce a static clip.
    output_path
        Destination file (.mp4).  Parent directories are created if needed.
    negative_prompt
        Things to steer away from.  None keeps the template's default
        (blur/distortion/watermark suppression).
    seconds
        Clip length.  Snapped to Wan's 4k+1 frame rule.  ~5 s is the native
        sweet spot; longer single-pass clips cost VRAM and time linearly and
        degrade in coherence.
    fps
        Output frame rate (default 24, the Wan 2.2 native rate).
    width / height
        Output resolution.  None keeps the template default (1280x704 —
        the model's NATIVE resolution; Wan 2.2 5B was trained at 720p and
        output collapses into washed-out smears well below it, so iterate
        by shortening the clip, not by shrinking the frame).  Must be
        multiples of 16.
    steps / cfg
        Sampler settings.  None keeps the template defaults (30 / 5.0).
    seed
        Noise seed for reproducible output.  Randomised if not given.
    input_image
        Animate this picture instead of starting from noise (image-to-video).
        The image is uploaded to ComfyUI and fed as the latent's start frame.
    api_url
        ComfyUI base URL; defaults to the shared resolution
        (FLUX_API_URL env var, else http://127.0.0.1:8188).
    localm_url
        localm server /v1 URL to unload before generation (VRAM handoff).
    max_poll_seconds
        Timeout waiting for ComfyUI (default 60 minutes — video is slow,
        especially without flash attention).
    on_progress
        Optional ``Callable[[str], None]`` for status lines.
    write_sidecar
        Write a ``<output>.json`` sidecar with the prompt and settings so
        the clip can be reproduced.  Pass False in privacy mode — the
        prompt then never touches disk.

    Returns
    -------
    (ok, message)
    """
    def _say(text: str) -> None:
        if on_progress:
            try:
                on_progress(text)
            except Exception:
                pass

    api_url = (api_url or default_api_url()).rstrip("/")
    if seconds <= 0:
        return False, "Duration must be positive."
    if fps <= 0:
        return False, "FPS must be positive."

    # Make sure ComfyUI is up (auto-launching when configured) — before
    # costing the user an LLM unload
    ok, msg = ensure_comfy(api_url, on_progress=_say)
    if not ok:
        return False, msg

    _localm_unload(localm_url)

    try:
        workflow = json.loads(_workflow_path().read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Failed to load Wan workflow template: {e}"

    seed = seed if seed is not None else random.randint(1, 10 ** 12)
    frames = _snap_frames(seconds, fps)

    workflow["4"]["inputs"]["text"] = prompt
    if negative_prompt is not None:
        workflow["5"]["inputs"]["text"] = negative_prompt
    workflow["6"]["inputs"]["length"] = frames
    if width is not None:
        workflow["6"]["inputs"]["width"] = width
    if height is not None:
        workflow["6"]["inputs"]["height"] = height
    workflow["8"]["inputs"]["seed"] = seed
    if steps is not None:
        workflow["8"]["inputs"]["steps"] = steps
    if cfg is not None:
        workflow["8"]["inputs"]["cfg"] = cfg
    if "10" in workflow and "fps" in workflow["10"].get("inputs", {}):
        workflow["10"]["inputs"]["fps"] = fps

    # Image-to-video: upload the picture and feed it as the start frame
    if input_image is not None:
        if not input_image.is_file():
            return False, f"Input image not found: {input_image}"
        try:
            uploaded_name = _upload_image(input_image, api_url)
        except Exception as e:
            return False, f"Failed to upload input image to ComfyUI: {e}"
        workflow["20"] = {
            "inputs": {"image": uploaded_name, "upload": "image"},
            "class_type": "LoadImage",
        }
        workflow["6"]["inputs"]["start_image"] = ["20", 0]

    # Queue
    try:
        req = urllib.request.Request(
            f"{api_url}/prompt",
            data=json.dumps({"prompt": workflow}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            prompt_id = json.loads(response.read().decode("utf-8")).get("prompt_id")
        if not prompt_id:
            return False, (
                "ComfyUI accepted the request but returned no prompt_id.\n"
                "Check the ComfyUI console for workflow validation errors — "
                "missing Wan 2.2 model files are the usual cause (see "
                "docs/video.md for the download list), and the Wan nodes "
                "need ComfyUI v0.3.46+."
            )
    except urllib.error.HTTPError as e:
        return False, (
            f"ComfyUI rejected the Wan 2.2 workflow (HTTP {e.code}):\n"
            f"{comfy_http_error_detail(e)}\n"
            "Missing Wan 2.2 model files are the usual cause (see "
            "docs/video.md for the download list); the Wan nodes need "
            "ComfyUI v0.3.46+."
        )
    except urllib.error.URLError as e:
        return False, f"Could not connect to ComfyUI at {api_url}: {e}"
    except Exception as e:
        return False, f"Error queuing prompt in ComfyUI: {e}"

    _say(f"Rendering {frames} frames ({frames / fps:.1f}s at {fps} fps)…")

    # Poll history until the clip is rendered
    start_time = time.time()
    video_info = None
    last_said = 0.0
    while time.time() - start_time < max_poll_seconds:
        elapsed = time.time() - start_time
        if elapsed - last_said >= 15:
            _say(f"Rendering… ({int(elapsed)}s elapsed)")
            last_said = elapsed
        try:
            hist_req = urllib.request.Request(f"{api_url}/history/{prompt_id}")
            with urllib.request.urlopen(hist_req, timeout=5) as response:
                history = json.loads(response.read().decode("utf-8"))
            if prompt_id in history:
                for node_output in history[prompt_id].get("outputs", {}).values():
                    for key in _OUTPUT_KEYS:
                        if node_output.get(key):
                            video_info = node_output[key][0]
                            break
                    if video_info:
                        break
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        return False, (
            f"Video generation timed out after {max_poll_seconds // 60} minutes."
        )

    if not video_info:
        return False, (
            "Generation finished but no video output was found in ComfyUI "
            "history. Check the ComfyUI console — a SaveVideo node error or "
            "an outdated ComfyUI (need v0.3.46+ for Wan 2.2) is likely."
        )

    # Fetch the file from ComfyUI and save locally
    try:
        params = urllib.parse.urlencode({
            "filename": video_info.get("filename"),
            "subfolder": video_info.get("subfolder", ""),
            "type": video_info.get("type", "output"),
        })
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{api_url}/view?{params}", timeout=120) as response:
            output_path.write_bytes(response.read())
    except Exception as e:
        return False, f"Failed to download generated clip from ComfyUI: {e}"

    # Delete the original from ComfyUI's output directory so no second copy
    # lingers there. Only possible when the user tells us where it is
    # (COMFY_OUTPUT_DIR env var or "comfy_output_dir" config key) — there is
    # no portable default. Same behaviour as image generation.
    comfy_out = os.environ.get("COMFY_OUTPUT_DIR")
    if not comfy_out:
        try:
            from localm.config import load_config
            comfy_out = load_config().get("comfy_output_dir")
        except Exception:
            comfy_out = None
    if comfy_out:
        try:
            orig = (Path(comfy_out) / video_info.get("subfolder", "")
                    / video_info.get("filename"))
            if orig.exists():
                orig.unlink()
        except Exception:
            pass

    # Sidecar JSON — everything needed to reproduce or tweak the clip.
    # Skipped entirely in privacy mode (write_sidecar=False) so the prompt
    # never touches disk.
    if not write_sidecar:
        return True, (f"Clip saved to {output_path} "
                      f"(seed {seed} — reuse it to reproduce)")
    try:
        sidecar = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seconds": round(frames / fps, 2),
            "frames": frames,
            "fps": fps,
            "width": width,
            "height": height,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "input_image": str(input_image) if input_image else None,
            "elapsed_seconds": round(time.time() - start_time, 1),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        output_path.with_suffix(output_path.suffix + ".json").write_text(
            json.dumps({k: v for k, v in sidecar.items() if v is not None},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass

    return True, (
        f"Clip saved to {output_path} "
        f"(seed {seed} — reuse it to reproduce)"
    )
