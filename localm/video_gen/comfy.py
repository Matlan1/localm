# SPDX-License-Identifier: AGPL-3.0-or-later
"""
ComfyUI Wan 2.2 short-video generation.

Mirrors localm.music_gen.comfy: standalone module, reachable from the GUI,
the CLI, or any other caller.  Uses the same ComfyUI server as image and
music generation - the model files for the committed template (the public
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
import random
import time
from pathlib import Path
from typing import Optional

# urllib.request is no longer used directly here (the transport moved to
# localm.media.comfy_client), but tests patch it as a module attribute -
# patch.object(comfy.urllib.request, "urlopen", ...) - and resolving
# comfy.urllib.request needs the name bound in this namespace. The shared client
# calls the SAME (global) module object, so the patch still bites.
import urllib.request  # noqa: F401

# Shared ComfyUI plumbing lives in localm.media.comfy_client - one server, one
# set of helpers. Imported as bare module globals so a test patching
# localm.video_gen.comfy.<name> still rebinds what generate_video calls.
from localm.media.comfy_client import (
    _localm_unload,
    _upload_image,
    _with_warning,
    apply_model_overrides,
    comfy_console_tail_start,
    comfy_console_warnings_since,
    comfy_exec_error_message,
    comfy_fetch_output,
    comfy_http_error_detail,
    comfy_poll_until_done,
    comfy_submit_prompt,
    contain_comfy_artifacts,
    default_api_url,
    ensure_comfy,
    find_node_by_class,
    inject_device_placement,
    next_node_id,
    POLL_CANCELLED,
    POLL_EXEC_ERROR,
    POLL_TIMEOUT,
    preflight_models,
    resolve_sampler_roles,
    select_output_info,
    set_seed_on_all,
    SUBMIT_ERROR,
    SUBMIT_HTTP_ERROR,
    SUBMIT_NO_ID,
    SUBMIT_URL_ERROR,
)

# wan_workflow.json is the committed generic template (public Wan 2.2 5B
# stack).  Drop a wan_workflow_local.json next to it (gitignored) to use
# your own checkpoint/graph without publishing which models you run.  The
# parameters are injected by ROLE (the sampler is found by class_type and the
# positive / negative / latent / CreateVideo nodes by following its graph
# edges), so a local graph does not have to preserve any particular node ids -
# it only has to wire a KSampler with positive/negative/latent_image inputs.
_WORKFLOW_PATH = Path(__file__).parent / "wan_workflow.json"
_WORKFLOW_LOCAL_PATH = Path(__file__).parent / "wan_workflow_local.json"


def workflow_path() -> Path:
    # 1. a workflow the user selected for the video plugin, 2. the legacy
    # wan_workflow_local.json, 3. the committed template. Selection is additive.
    try:
        from localm.media_workflows import active_workflow_path
        selected = active_workflow_path("video")
        if selected is not None:
            return selected
    except Exception:
        pass
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


def _build_video_workflow(
    workflow: dict,
    *,
    prompt: str,
    negative_prompt: Optional[str],
    frames: int,
    fps: int,
    width: Optional[int],
    height: Optional[int],
    steps: Optional[int],
    cfg: Optional[float],
    seed: int,
    float_type: Optional[str],
    input_image: Optional[Path],
    api_url: str,
) -> tuple[bool, str, Optional[str]]:
    """Shape the Wan workflow in place from the call's parameters.

    Returns ``(ok, message, uploaded_name)``: ``ok=False`` with an error message
    when the graph has no sampler / prompt / latent node localm can drive, or the
    input image is missing / fails to upload. ``uploaded_name`` is the ComfyUI-side
    filename of an uploaded img2video source (for later containment) or None."""
    # Resolve the nodes we drive by ROLE (sampler by class_type, then positive /
    # negative / latent by following its graph edges) instead of hardcoded ids, so
    # a user's own exported Wan graph works without renumbering (I3).
    roles = resolve_sampler_roles(workflow)
    _, sampler = roles["sampler"]
    _, positive = roles["positive"]
    _, negative = roles["negative"]
    _, latent = roles["latent"]
    sampler_inputs = sampler.get("inputs") if sampler else None
    latent_inputs = latent.get("inputs") if latent else None
    if (sampler is None or positive is None or latent is None
            or not isinstance(sampler_inputs, dict)
            or not isinstance(latent_inputs, dict)):
        return False, (
            "The video workflow has no sampler / prompt / latent node (with inputs) "
            "localm can drive. Export a Wan 2.2 workflow from ComfyUI (Save -> API "
            "format) and select it on the Workflow panel (Settings -> Media -> Video)."), None

    # Positive prompt (the conditioning node feeding the sampler's positive input).
    if "text" in positive.get("inputs", {}):
        positive["inputs"]["text"] = prompt
    # Negative prompt - only when the negative branch is actually a text-encode
    # node (Wan wires a CLIPTextEncode; a graph that zeroes the negative has no text).
    if (negative_prompt is not None and negative is not None
            and "text" in negative.get("inputs", {})):
        negative["inputs"]["text"] = negative_prompt
    # Video latent: frame count, then optional resolution.
    latent_inputs["length"] = frames
    if width is not None:
        latent_inputs["width"] = width
    if height is not None:
        latent_inputs["height"] = height
    # Sampler knobs: the seed goes on EVERY sampler (a two-stage Wan high/low-noise
    # graph has two), steps/cfg on the primary sampler driving this latent.
    set_seed_on_all(workflow, seed)
    if steps is not None:
        sampler_inputs["steps"] = steps
    if cfg is not None:
        sampler_inputs["cfg"] = cfg
    # Output frame rate on the CreateVideo node, wherever it sits in the graph.
    _, create_video = find_node_by_class(workflow, "CreateVideo")
    if create_video is not None and "fps" in create_video.get("inputs", {}):
        create_video["inputs"]["fps"] = fps

    if float_type and float_type != "default":
        for node in workflow.values():
            if node.get("class_type") in (
                "UNETLoader", "UnetLoaderGGUF", "UnetLoaderGGUFAdvanced",
                "CheckpointLoaderSimple", "CheckpointLoader"
            ):
                if "inputs" not in node:
                    node["inputs"] = {}
                node["inputs"]["weight_dtype"] = float_type

    # Image-to-video: upload the picture and feed it as the latent's start frame.
    # A fresh node id (not a hardcoded "20") avoids clobbering a user's own node.
    uploaded_name: Optional[str] = None
    if input_image is not None:
        if not input_image.is_file():
            return False, f"Input image not found: {input_image}", None
        try:
            uploaded_name = _upload_image(input_image, api_url)
        except Exception as e:
            return False, f"Failed to upload input image to ComfyUI: {e}", None
        load_id = next_node_id(workflow)
        workflow[load_id] = {
            "inputs": {"image": uploaded_name, "upload": "image"},
            "class_type": "LoadImage",
        }
        latent_inputs["start_image"] = [load_id, 0]

    return True, "", uploaded_name


def _write_video_sidecar(
    output_path: Path,
    *,
    prompt: str,
    negative_prompt: Optional[str],
    frames: int,
    fps: int,
    width: Optional[int],
    height: Optional[int],
    seed: int,
    steps: Optional[int],
    cfg: Optional[float],
    input_image: Optional[Path],
    start_time: float,
    comfy_console_warning: Optional[str] = None,
    comfy_console_checked: bool = False,
) -> Optional[str]:
    """Write the ``<output>.json`` reproducibility sidecar next to the clip.

    The clip is already saved; a failed sidecar must not fail the whole
    generation - returns a note string on failure (surfaced, not swallowed),
    else None.

    ``comfy_console_warning``/``comfy_console_checked`` mirror
    image_gen.comfy._write_image_sidecar's fields: ComfyUI can silently
    under-apply a mismatched checkpoint's weights and still report success, and
    the pair keeps "checked, found nothing" distinct from "could not check at
    all" (a remote or already-running ComfyUI localm did not launch)."""
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
            "comfy_console_warning": comfy_console_warning,
            "comfy_console_checked": comfy_console_checked,
            "input_image": str(input_image) if input_image else None,
            "elapsed_seconds": round(time.time() - start_time, 1),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        output_path.with_suffix(output_path.suffix + ".json").write_text(
            json.dumps({k: v for k, v in sidecar.items() if v is not None},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        # The clip is already saved; a failed sidecar must not fail the whole
        # generation - surface it as a note instead of silently swallowing it.
        return (
            f"the reproducibility sidecar could not be saved ({e}); "
            "the clip itself was saved."
        )
    return None


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
    model_overrides: Optional[dict] = None,
    api_url: Optional[str] = None,
    localm_url: Optional[str] = None,
    instance_token: Optional[str] = None,
    max_poll_seconds: int = 3600,
    on_progress=None,
    write_sidecar: bool = True,
    launch_cmd: Optional[str] = None,
    workdir: Optional[str] = None,
    float_type: Optional[str] = None,
    swap: bool = True,
    cancel_check: Optional[callable] = None,
    delete_outputs: bool = False,
    placement: Optional[dict] = None,
) -> tuple[bool, str]:
    """
    Generate a short video clip and save it to *output_path* (MP4).

    Parameters
    ----------
    prompt
        Scene description - subject, motion, camera, lighting.  Motion verbs
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
        Output resolution.  None keeps the template default (1280x704 -
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
    model_overrides
        Generic per-node model-slot overrides (see image_gen.comfy.generate_image's
        docstring for the full explanation) - ``{node_id: {input_name: value}}``,
        applied before any other workflow shaping.
    api_url
        ComfyUI base URL; defaults to the shared resolution
        (FLUX_API_URL env var, else http://127.0.0.1:8188).
    localm_url
        localm server /v1 URL to unload before generation (VRAM handoff).
    instance_token
        This server's own attach token, forwarded to the ``localm_url``
        unload call so it authenticates on a keyless (open-mode) server too -
        see ``selfclient.self_request``'s docstring. Only used as a fallback
        when no owner API key is configured.
    max_poll_seconds
        Timeout waiting for ComfyUI (default 60 minutes - video is slow,
        especially without flash attention).
    on_progress
        Optional ``Callable[[str], None]`` for status lines.
    write_sidecar
        Write a ``<output>.json`` sidecar with the prompt and settings so
        the clip can be reproduced.  Pass False in privacy mode - the
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

    # Make sure ComfyUI is up (auto-launching when configured) - before
    # costing the user an LLM unload
    ok, msg = ensure_comfy(api_url, on_progress=_say,
                           launch_cmd=launch_cmd, workdir=workdir)
    if not ok:
        return False, msg

    # The LLM unload (the expensive VRAM handoff) is deferred to AFTER the workflow
    # is built and the model preflight passes, so a missing model file fails before
    # it costs the user a pointless unload + reload.

    try:
        workflow = json.loads(workflow_path().read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Failed to load Wan workflow template: {e}"
    if model_overrides:
        apply_model_overrides(workflow, model_overrides)

    seed = seed if seed is not None else random.randint(1, 10 ** 12)
    frames = _snap_frames(seconds, fps)

    ok, msg, uploaded_name = _build_video_workflow(
        workflow,
        prompt=prompt,
        negative_prompt=negative_prompt,
        frames=frames,
        fps=fps,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        seed=seed,
        float_type=float_type,
        input_image=input_image,
        api_url=api_url,
    )
    if not ok:
        return False, msg

    # Pre-submit model validation: confirm the Wan model files exist (substituting
    # an unambiguous precision variant), failing EARLY with the exact missing file
    # BEFORE the LLM unload. Best-effort when /object_info is unreachable.
    pf_ok, pf_msg = preflight_models(workflow, api_url, on_progress=_say)
    if not pf_ok:
        return False, pf_msg

    # Now free VRAM (the workflow is valid). swap=False keeps the chat model hot.
    if swap:
        _localm_unload(localm_url, instance_token)

    # Per-component GPU placement (opt-in, multi-GPU only): inject the core Select*Device
    # nodes per the plan resolve_media_placement() decided. A component whose loader is
    # absent from this graph is surfaced to the user, never silently dropped; the
    # happy-path summary already went out via the placement notice.
    if placement:
        for _note in inject_device_placement(workflow, placement):
            if "could not place" in _note:
                _say(_note)

    # Queue. Mark 'now' in ComfyUI's own console log FIRST (comfy_console_tail_start),
    # so any silent partial-apply warning it prints while running THIS prompt (a
    # mismatched checkpoint's UNet/CLIP/VAE keys, ...) can be attributed to this
    # generation and not an earlier one. None when localm did not launch this
    # ComfyUI itself; comfy_console_warnings_since() then always reports
    # checked=False.
    console_tail_start = comfy_console_tail_start(api_url)
    kind, value = comfy_submit_prompt(api_url, workflow)
    if kind == SUBMIT_NO_ID:
        return False, (
            "ComfyUI accepted the request but returned no prompt_id.\n"
            "Check the ComfyUI console for workflow validation errors - "
            "missing Wan 2.2 model files are the usual cause (see "
            "docs/video.md for the download list), and the Wan nodes "
            "need ComfyUI v0.3.46+."
        )
    elif kind == SUBMIT_HTTP_ERROR:
        return False, (
            f"ComfyUI rejected the Wan 2.2 workflow (HTTP {value.code}):\n"
            f"{comfy_http_error_detail(value)}\n"
            "Missing Wan 2.2 model files are the usual cause (see "
            "docs/video.md for the download list); the Wan nodes need "
            "ComfyUI v0.3.46+."
        )
    elif kind == SUBMIT_URL_ERROR:
        return False, f"Could not connect to ComfyUI at {api_url}: {value}"
    elif kind == SUBMIT_ERROR:
        return False, f"Error queuing prompt in ComfyUI: {value}"
    prompt_id = value

    _say(f"Rendering {frames} frames ({frames / fps:.1f}s at {fps} fps)…")

    # Poll history until the clip is rendered. Throttle the "Rendering…" line to
    # once every 15s (the same cadence as before). start_time is taken here (as
    # the original did) so the sidecar's elapsed_seconds covers poll + download.
    start_time = time.time()
    last_said = [0.0]

    def _tick(elapsed: float) -> None:
        if elapsed - last_said[0] >= 15:
            _say(f"Rendering… ({int(elapsed)}s elapsed)")
            last_said[0] = elapsed

    status, payload = comfy_poll_until_done(
        api_url, prompt_id,
        max_poll_seconds=max_poll_seconds,
        cancel_check=cancel_check,
        on_tick=_tick,
    )
    if status == POLL_CANCELLED:
        return False, "Generation cancelled."
    if status == POLL_EXEC_ERROR:
        return False, comfy_exec_error_message(payload, api_url)
    if status == POLL_TIMEOUT:
        msg = (
            f"Video generation timed out after {max_poll_seconds // 60} minutes."
        )
        if payload is not None:
            # A persistent poll error means ComfyUI likely crashed or went
            # unreachable, not just a slow render - say so instead of hiding it.
            msg += f" Last poll error: {payload}"
        return False, msg

    # status == POLL_FINISHED means ComfyUI reported no execution_error - but a
    # node whose weights only partly matched (a mismatched checkpoint's UNet/CLIP
    # keys, ...) is not an execution_error to ComfyUI, only a console warning, and
    # the run still "succeeds" with that component silently under-applied. Check
    # for any KNOWN warning of that shape printed while THIS prompt ran (see
    # NEW-COMFY-SILENT-PARTIAL-APPLY in image_gen/comfy.py). console_checked
    # reflects whether a real read actually happened just now, not whether
    # console_tail_start found a process before the prompt was even submitted.
    console_checked, comfy_console_warnings = comfy_console_warnings_since(
        api_url, console_tail_start)
    comfy_console_warning_text = ("; ".join(comfy_console_warnings)
                                  if comfy_console_warnings else None)
    comfy_console_msg = (
        "WARNING: ComfyUI's own console reported: "
        + comfy_console_warning_text
        + ". The generation still completed, but the requested model weights "
          "may not have fully applied - see comfy-launch.log."
    ) if comfy_console_warning_text else ""

    video_info = select_output_info(payload, _OUTPUT_KEYS)

    if not video_info:
        return False, (
            "Generation finished but no video output was found in ComfyUI "
            "history. Check the ComfyUI console - a SaveVideo node error or "
            "an outdated ComfyUI (need v0.3.46+ for Wan 2.2) is likely."
        )

    # Fetch the file from ComfyUI and save locally
    try:
        comfy_fetch_output(api_url, video_info, output_path, timeout=120)
    except Exception as e:
        return False, f"Failed to download generated clip from ComfyUI: {e}"

    # Enforce output containment: clear ComfyUI's history entry (the
    # Queue/History + gallery view) and delete ComfyUI's own on-disk copy of
    # the clip plus any uploaded img2video source. Returns a warning when the
    # file copy could not be removed (e.g. comfy_output_dir unset).
    # delete_outputs controls whether ComfyUI's own on-disk copy is removed:
    # opt-in containment (default False keeps the copy), forced True by the
    # caller in privacy mode so no traces remain. The history entry and any
    # uploaded img2video source are always cleared regardless.
    contain_warning = contain_comfy_artifacts(
        api_url, prompt_id,
        {"filename": video_info.get("filename"),
         "subfolder": video_info.get("subfolder", ""),
         "type": video_info.get("type", "output")},
        uploaded_input=uploaded_name,
        delete_outputs=delete_outputs,
    )

    # Sidecar JSON - everything needed to reproduce or tweak the clip.
    # Skipped entirely in privacy mode (write_sidecar=False) so the prompt
    # never touches disk. The console warning (a real quality issue with THIS
    # clip) is still reported in the message either way - only the sidecar's
    # record of it is what privacy mode suppresses.
    if not write_sidecar:
        return True, _with_warning(
            _with_warning(
                f"Clip saved to {output_path} "
                f"(seed {seed} - reuse it to reproduce)", contain_warning),
            comfy_console_msg)

    sidecar_warning = _write_video_sidecar(
        output_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        frames=frames,
        fps=fps,
        width=width,
        height=height,
        seed=seed,
        steps=steps,
        cfg=cfg,
        input_image=input_image,
        start_time=start_time,
        comfy_console_warning=comfy_console_warning_text,
        comfy_console_checked=console_checked,
    )

    return True, _with_warning(
        _with_warning(
            _with_warning(
                f"Clip saved to {output_path} "
                f"(seed {seed} - reuse it to reproduce)", contain_warning),
            comfy_console_msg),
        sidecar_warning)
