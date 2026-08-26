# SPDX-License-Identifier: AGPL-3.0-or-later
"""
ComfyUI ACE-Step music generation.

Mirrors localm.image_gen.comfy: standalone module, reachable from the GUI,
the CLI, or any other caller.  Uses the same ComfyUI server as image
generation - the checkpoint (``ace_step_v1_3.5b.safetensors``) must be in
ComfyUI's ``models/checkpoints`` directory (ComfyUI ships native ACE-Step
support since v0.3.34).

Track length is arbitrary: the ACE-Step latent is sized directly from the
requested duration in seconds, so 30-second jingles and 10-minute ambient
tracks go through the same path.  Output is lossless FLAC.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Optional

# Bound in this namespace so patch.object(comfy.urllib.request, "urlopen", ...)
# resolves; the shared client calls the SAME (global) module object.
import urllib.request  # noqa: F401

# Shared ComfyUI plumbing from localm.media.comfy_client, imported as bare
# module globals so a test patching localm.music_gen.comfy.<name> still rebinds
# what generate_music calls.
from localm.media.comfy_client import (
    _localm_unload,
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
    POLL_CANCELLED,
    POLL_EXEC_ERROR,
    POLL_TIMEOUT,
    preflight_models,
    resolve_sampler_roles,
    set_seed_on_all,
    SUBMIT_ERROR,
    SUBMIT_HTTP_ERROR,
    SUBMIT_NO_ID,
    SUBMIT_URL_ERROR,
)

# ace_workflow.json is the committed generic template (public ACE-Step
# checkpoint).  An ace_workflow_local.json next to it (gitignored) overrides it
# with your own checkpoint/graph.
_WORKFLOW_PATH = Path(__file__).parent / "ace_workflow.json"
_WORKFLOW_LOCAL_PATH = Path(__file__).parent / "ace_workflow_local.json"


def workflow_path() -> Path:
    # 1. a workflow the user selected for the music plugin, 2. the legacy
    # ace_workflow_local.json, 3. the committed template. Selection is additive.
    try:
        from localm.media_workflows import active_workflow_path
        selected = active_workflow_path("music")
        if selected is not None:
            return selected
    except Exception:
        pass
    return _WORKFLOW_LOCAL_PATH if _WORKFLOW_LOCAL_PATH.is_file() else _WORKFLOW_PATH

# Instrumental marker ACE-Step understands when no lyrics are given
_INSTRUMENTAL = "[inst]"


def _build_music_workflow(
    workflow: dict,
    *,
    tags: str,
    lyrics_text: str,
    duration_seconds: float,
    seed: int,
    steps: int,
    cfg: float,
    lyrics_strength: float,
    ckpt_name: Optional[str],
    float_type: Optional[str],
    sampler_name: Optional[str] = None,
    scheduler: Optional[str] = None,
    shift: Optional[float] = None,
) -> tuple[bool, str]:
    """Shape the ACE-Step workflow in place from the call's parameters.

    Returns ``(ok, message)``: ``ok=False`` with an error message when the graph
    has no sampler / prompt / latent node localm can drive."""
    # Resolve the nodes we drive by ROLE (sampler by class_type, then the ACE-Step
    # prompt and latent by following its graph edges) instead of hardcoded ids, so
    # a user's own exported ACE-Step graph works without renumbering.
    roles = resolve_sampler_roles(workflow)
    _, sampler = roles["sampler"]
    _, positive = roles["positive"]
    _, latent = roles["latent"]
    if sampler is None or positive is None or latent is None:
        return False, (
            "The music workflow has no sampler / prompt / latent node localm can "
            "drive. Export an ACE-Step workflow from ComfyUI (Save -> API format) "
            "and select it on the Workflow panel (Settings -> Media -> Music).")

    # ACE-Step latent length (seconds), then the prompt node's tags / lyrics.
    if "seconds" in latent.get("inputs", {}):
        latent["inputs"]["seconds"] = float(duration_seconds)
    pin = positive.get("inputs", {})
    if "tags" in pin:
        pin["tags"] = tags
    if "lyrics" in pin:
        pin["lyrics"] = lyrics_text
    if "lyrics_strength" in pin:
        pin["lyrics_strength"] = lyrics_strength
    # Sampler knobs: the seed goes on EVERY sampler (set_seed_on_all), steps/cfg on
    # the primary sampler driving this latent.
    set_seed_on_all(workflow, seed)
    if "steps" in sampler.get("inputs", {}):
        sampler["inputs"]["steps"] = steps
    if "cfg" in sampler.get("inputs", {}):
        sampler["inputs"]["cfg"] = cfg
    if sampler_name and "sampler_name" in sampler.get("inputs", {}):
        sampler["inputs"]["sampler_name"] = sampler_name
    if scheduler and "scheduler" in sampler.get("inputs", {}):
        sampler["inputs"]["scheduler"] = scheduler
    # Shift lives on the ModelSamplingSD3 node feeding the sampler, not on the
    # sampler node itself - found by class_type (same idiom as ckpt_loader below)
    # so a user's own exported graph works without renumbering.
    if shift is not None:
        _, model_sampling = find_node_by_class(workflow, "ModelSamplingSD3")
        if model_sampling is not None and "shift" in model_sampling.get("inputs", {}):
            model_sampling["inputs"]["shift"] = shift
    # Optional checkpoint override, found by class_type.
    if ckpt_name:
        _, ckpt_loader = find_node_by_class(
            workflow, "CheckpointLoaderSimple", "CheckpointLoader")
        if ckpt_loader is not None and "ckpt_name" in ckpt_loader.get("inputs", {}):
            ckpt_loader["inputs"]["ckpt_name"] = ckpt_name

    if float_type and float_type != "default":
        for node in workflow.values():
            if node.get("class_type") in (
                "UNETLoader", "UnetLoaderGGUF", "UnetLoaderGGUFAdvanced",
                "CheckpointLoaderSimple", "CheckpointLoader"
            ):
                if "inputs" not in node:
                    node["inputs"] = {}
                node["inputs"]["weight_dtype"] = float_type

    return True, ""


def generate_music(
    tags: str,
    output_path: Path,
    *,
    lyrics: Optional[str] = None,
    duration_seconds: float = 120.0,
    api_url: Optional[str] = None,
    seed: Optional[int] = None,
    steps: int = 50,
    cfg: float = 5.0,
    lyrics_strength: float = 0.99,
    sampler_name: Optional[str] = None,
    scheduler: Optional[str] = None,
    shift: Optional[float] = None,
    ckpt_name: Optional[str] = None,
    model_overrides: Optional[dict] = None,
    localm_url: Optional[str] = None,
    instance_token: Optional[str] = None,
    max_poll_seconds: int = 1800,
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
    Generate a music track and save it to *output_path* (FLAC).

    Parameters
    ----------
    tags
        Comma-separated style description - genre, mood, instruments, BPM,
        vocal type (e.g. ``"synthwave, 80s, female vocals, 120 bpm, dreamy"``).
    output_path
        Destination file (.flac).  Parent directories are created if needed.
    lyrics
        Song lyrics, optionally with section markers like ``[verse]`` /
        ``[chorus]``.  None or empty generates an instrumental track.
    duration_seconds
        Track length in seconds - arbitrary; the latent is sized from it.
        Long tracks take proportionally longer and use more VRAM.
    api_url
        ComfyUI base URL; defaults to the shared image-gen URL resolution
        (FLUX_API_URL env var, else http://127.0.0.1:8188).
    seed
        Noise seed for reproducible output.  Randomised if not given.
    steps / cfg
        Sampler settings.  The defaults (50 / 5.0) match the official
        ComfyUI ACE-Step template - raise steps for more polish.
    lyrics_strength
        How strongly the lyrics steer generation (0..1).
    sampler_name / scheduler
        Override the KSampler's sampler/scheduler (defaults from the workflow
        template: "euler" / "simple"). Any value ComfyUI's own KSampler accepts;
        an unsupported one is rejected by ComfyUI at submit time with a clear
        error, not validated here.
    shift
        Override the ModelSamplingSD3 node's shift value (default from the
        workflow template: 3.0).
    ckpt_name
        Override the checkpoint filename inside ComfyUI's models/checkpoints.
    model_overrides
        Generic per-node model-slot overrides (see image_gen.comfy.generate_image's
        docstring for the full explanation) - ``{node_id: {input_name: value}}``,
        applied before ckpt_name/any other shaping, so an explicit ckpt_name for
        the same field still wins if both are given.
    localm_url
        localm server /v1 URL to unload before generation (VRAM handoff).
    instance_token
        This server's own attach token, forwarded to the ``localm_url``
        unload call so it authenticates on a keyless (open-mode) server too -
        see ``selfclient.self_request``'s docstring. Only used as a fallback
        when no owner API key is configured.
    max_poll_seconds
        Timeout waiting for ComfyUI (default 30 minutes - long tracks are slow).
    on_progress
        Optional ``Callable[[str], None]`` for status lines.

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
    if duration_seconds <= 0:
        return False, "Duration must be positive."

    # Make sure ComfyUI is up (auto-launching when configured), before the
    # LLM unload
    ok, msg = ensure_comfy(api_url, on_progress=_say,
                           launch_cmd=launch_cmd, workdir=workdir)
    if not ok:
        return False, msg

    # The LLM unload (the VRAM handoff) is deferred to AFTER the workflow is
    # built and the model preflight passes, so a missing checkpoint fails
    # before the unload.

    try:
        workflow = json.loads(workflow_path().read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Failed to load ACE-Step workflow template: {e}"
    if model_overrides:
        apply_model_overrides(workflow, model_overrides)

    seed = seed if seed is not None else random.randint(1, 10 ** 12)
    lyrics_text = (lyrics or "").strip() or _INSTRUMENTAL

    ok, msg = _build_music_workflow(
        workflow,
        tags=tags,
        lyrics_text=lyrics_text,
        duration_seconds=duration_seconds,
        seed=seed,
        steps=steps,
        cfg=cfg,
        lyrics_strength=lyrics_strength,
        sampler_name=sampler_name,
        scheduler=scheduler,
        shift=shift,
        ckpt_name=ckpt_name,
        float_type=float_type,
    )
    if not ok:
        return False, msg

    # Pre-submit model validation: confirm the ACE-Step checkpoint exists
    # (substituting an unambiguous precision variant), failing EARLY with the exact
    # missing file BEFORE the LLM unload. Best-effort when /object_info is unreachable.
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
    # mismatched checkpoint's UNet/CLIP/VAE keys, ...) is attributed to this
    # generation and not an earlier one. None when localm did not launch this
    # ComfyUI itself; comfy_console_warnings_since() then always reports
    # checked=False.
    console_tail_start = comfy_console_tail_start(api_url)
    kind, value = comfy_submit_prompt(api_url, workflow)
    if kind == SUBMIT_NO_ID:
        return False, (
            "ComfyUI accepted the request but returned no prompt_id.\n"
            "Check the ComfyUI console for workflow validation errors - a "
            "missing ace_step_v1_3.5b.safetensors checkpoint is the usual "
            "cause (download it into ComfyUI/models/checkpoints)."
        )
    elif kind == SUBMIT_HTTP_ERROR:
        return False, (
            f"ComfyUI rejected the ACE-Step workflow (HTTP {value.code}):\n"
            f"{comfy_http_error_detail(value)}\n"
            "The usual cause is a missing checkpoint - ACE-Step needs "
            "ace_step_v1_3.5b.safetensors in ComfyUI/models/checkpoints "
            "(or your own checkpoint via ace_workflow_local.json), and "
            "ComfyUI v0.3.34+ for the ACE-Step nodes."
        )
    elif kind == SUBMIT_URL_ERROR:
        return False, f"Could not connect to ComfyUI at {api_url}: {value}"
    elif kind == SUBMIT_ERROR:
        return False, f"Error queuing prompt in ComfyUI: {value}"
    prompt_id = value

    # Poll history until the track is rendered. Throttle the "Rendering…" line to
    # once every 15s. start_time is taken here so the sidecar's elapsed_seconds
    # covers poll + download.
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
        timeout_msg = (
            f"Music generation timed out after {max_poll_seconds // 60} minutes."
        )
        if payload is not None:
            # Surface the last poll error: an unreachable ComfyUI looks like a timeout otherwise.
            timeout_msg += (
                f" The last attempt to reach ComfyUI failed: {payload}"
            )
        return False, timeout_msg

    # status == POLL_FINISHED means ComfyUI reported no execution_error - but a
    # node whose weights only partly matched (a mismatched checkpoint's UNet/CLIP
    # keys, ...) is not an execution_error to ComfyUI, only a console warning, and
    # the run still "succeeds" with that component silently under-applied. Check
    # for any KNOWN warning of that shape printed while THIS prompt ran.
    # console_checked reflects whether a real read actually happened just now,
    # not whether console_tail_start found a process before the prompt was even
    # submitted.
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

    # status == POLL_FINISHED: find the rendered track (presence-based - the
    # first node carrying an "audio" entry wins).
    audio_info = None
    for node_output in payload.get("outputs", {}).values():
        if "audio" in node_output:
            audio_info = node_output["audio"][0]
            break

    if not audio_info:
        return False, (
            "Generation finished but no audio output was found in ComfyUI "
            "history. Check the ComfyUI console - a SaveAudio node error or "
            "an outdated ComfyUI (need v0.3.34+ for ACE-Step) is likely."
        )

    # Fetch the file from ComfyUI and save locally
    try:
        comfy_fetch_output(api_url, audio_info, output_path, timeout=60)
    except Exception as e:
        return False, f"Failed to download generated track from ComfyUI: {e}"

    # Output containment (opt-in): clear ComfyUI's history entry and delete its
    # own copy of the track ONLY when delete_outputs is set (user opted in, or
    # privacy mode forces no-trace). ACE-Step's SaveAudio node writes into
    # ComfyUI's output dir and records the job in /history, both of which its
    # output browser surfaces. The default keeps ComfyUI's own copies.
    contain_warning = contain_comfy_artifacts(
        api_url, prompt_id,
        {"filename": audio_info.get("filename"),
         "subfolder": audio_info.get("subfolder", ""),
         "type": audio_info.get("type", "output")},
        delete_outputs=delete_outputs,
    )

    # Sidecar JSON - everything needed to reproduce or tweak the track.
    # Skipped entirely in privacy mode (write_sidecar=False), so the prompt and
    # lyrics never touch disk. The console warning is still reported in the
    # message either way; only the sidecar's record of it is suppressed.
    if not write_sidecar:
        return True, _with_warning(
            _with_warning(
                f"Track saved to {output_path} "
                f"(seed {seed} - reuse it to reproduce)", contain_warning),
            comfy_console_msg)
    try:
        sidecar = {
            "tags": tags,
            "lyrics": None if lyrics_text == _INSTRUMENTAL else lyrics_text,
            "duration_seconds": duration_seconds,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "lyrics_strength": lyrics_strength,
            "comfy_console_warning": comfy_console_warning_text,
            "comfy_console_checked": console_checked,
            "elapsed_seconds": round(time.time() - start_time, 1),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        output_path.with_suffix(output_path.suffix + ".json").write_text(
            json.dumps({k: v for k, v in sidecar.items() if v is not None},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        # Note the sidecar miss in the message instead of failing; the track
        # itself is already saved.
        contain_warning = _with_warning(
            "the reproducibility sidecar could not be saved "
            f"({e}); the track itself was saved.", contain_warning)

    return True, _with_warning(
        _with_warning(
            f"Track saved to {output_path} "
            f"(seed {seed} - reuse it to reproduce)", contain_warning),
        comfy_console_msg)
