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
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

# Shared ComfyUI plumbing lives in image_gen - one server, one set of helpers
from localm.image_gen.comfy import (
    _localm_unload,
    _with_warning,
    comfy_http_error_detail,
    contain_comfy_artifacts,
    default_api_url,
    ensure_comfy,
)

# ace_workflow.json is the committed generic template (public ACE-Step
# checkpoint).  Drop an ace_workflow_local.json next to it (gitignored) to
# use your own checkpoint/graph without publishing which models you run.
_WORKFLOW_PATH = Path(__file__).parent / "ace_workflow.json"
_WORKFLOW_LOCAL_PATH = Path(__file__).parent / "ace_workflow_local.json"


def _workflow_path() -> Path:
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
    ckpt_name: Optional[str] = None,
    localm_url: Optional[str] = None,
    max_poll_seconds: int = 1800,
    on_progress=None,
    write_sidecar: bool = True,
    launch_cmd: Optional[str] = None,
    workdir: Optional[str] = None,
    swap: bool = True,
    cancel_check: Optional[callable] = None,
    delete_outputs: bool = False,
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
    ckpt_name
        Override the checkpoint filename inside ComfyUI's models/checkpoints.
    localm_url
        localm server /v1 URL to unload before generation (VRAM handoff).
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

    # Make sure ComfyUI is up (auto-launching when configured) - before
    # costing the user an LLM unload
    ok, msg = ensure_comfy(api_url, on_progress=_say,
                           launch_cmd=launch_cmd, workdir=workdir)
    if not ok:
        return False, msg

    # Unload the chat LLM to free VRAM, unless the caller decided the media model
    # fits alongside it (swap=False) so the chat model stays hot.
    if swap:
        _localm_unload(localm_url)

    try:
        workflow = json.loads(_workflow_path().read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Failed to load ACE-Step workflow template: {e}"

    seed = seed if seed is not None else random.randint(1, 10 ** 12)
    lyrics_text = (lyrics or "").strip() or _INSTRUMENTAL

    workflow["2"]["inputs"]["seconds"] = float(duration_seconds)
    workflow["3"]["inputs"]["tags"] = tags
    workflow["3"]["inputs"]["lyrics"] = lyrics_text
    workflow["3"]["inputs"]["lyrics_strength"] = lyrics_strength
    workflow["6"]["inputs"]["seed"] = seed
    workflow["6"]["inputs"]["steps"] = steps
    workflow["6"]["inputs"]["cfg"] = cfg
    if ckpt_name:
        workflow["1"]["inputs"]["ckpt_name"] = ckpt_name

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
                "Check the ComfyUI console for workflow validation errors - a "
                "missing ace_step_v1_3.5b.safetensors checkpoint is the usual "
                "cause (download it into ComfyUI/models/checkpoints)."
            )
    except urllib.error.HTTPError as e:
        return False, (
            f"ComfyUI rejected the ACE-Step workflow (HTTP {e.code}):\n"
            f"{comfy_http_error_detail(e)}\n"
            "The usual cause is a missing checkpoint - ACE-Step needs "
            "ace_step_v1_3.5b.safetensors in ComfyUI/models/checkpoints "
            "(or your own checkpoint via ace_workflow_local.json), and "
            "ComfyUI v0.3.34+ for the ACE-Step nodes."
        )
    except urllib.error.URLError as e:
        return False, f"Could not connect to ComfyUI at {api_url}: {e}"
    except Exception as e:
        return False, f"Error queuing prompt in ComfyUI: {e}"

    # Poll history until the track is rendered
    start_time = time.time()
    audio_info = None
    last_said = 0.0
    last_poll_error = None  # remember the last /history failure so a timeout can say WHY
    while time.time() - start_time < max_poll_seconds:
        if cancel_check and cancel_check():
            from localm.image_gen.comfy import clear_comfy_history, interrupt_comfy
            interrupt_comfy(api_url)
            clear_comfy_history(api_url, prompt_id)
            return False, "Generation cancelled."
        elapsed = time.time() - start_time
        if elapsed - last_said >= 15:
            _say(f"Rendering… ({int(elapsed)}s elapsed)")
            last_said = elapsed
        try:
            hist_req = urllib.request.Request(f"{api_url}/history/{prompt_id}")
            with urllib.request.urlopen(hist_req, timeout=5) as response:
                history = json.loads(response.read().decode("utf-8"))
            if prompt_id in history:
                from localm.image_gen.comfy import history_execution_error
                err = history_execution_error(history[prompt_id])
                if err:
                    return False, f"ComfyUI execution failed: {err}"
                for node_output in history[prompt_id].get("outputs", {}).values():
                    if "audio" in node_output:
                        audio_info = node_output["audio"][0]
                        break
                break
        except Exception as e:
            # Keep retrying (ComfyUI may be mid-render), but record the failure
            # so a timeout can distinguish a crashed/unreachable server from a slow one.
            last_poll_error = e
        time.sleep(2)
    else:
        timeout_msg = (
            f"Music generation timed out after {max_poll_seconds // 60} minutes."
        )
        if last_poll_error is not None:
            # Surface the last poll error: an unreachable ComfyUI looks like a timeout otherwise.
            timeout_msg += (
                f" The last attempt to reach ComfyUI failed: {last_poll_error}"
            )
        return False, timeout_msg

    if not audio_info:
        return False, (
            "Generation finished but no audio output was found in ComfyUI "
            "history. Check the ComfyUI console - a SaveAudio node error or "
            "an outdated ComfyUI (need v0.3.34+ for ACE-Step) is likely."
        )

    # Fetch the file from ComfyUI and save locally
    try:
        params = urllib.parse.urlencode({
            "filename": audio_info.get("filename"),
            "subfolder": audio_info.get("subfolder", ""),
            "type": audio_info.get("type", "output"),
        })
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{api_url}/view?{params}", timeout=60) as response:
            output_path.write_bytes(response.read())
    except Exception as e:
        return False, f"Failed to download generated track from ComfyUI: {e}"

    # Output containment (opt-in): clear ComfyUI's history entry and delete its
    # own copy of the track ONLY when delete_outputs is set (user opted in, or
    # privacy mode forces no-trace). ACE-Step's SaveAudio node writes into
    # ComfyUI's output dir and records the job in /history, both of which its
    # output browser surfaces - so when the user wants no second copy, that copy
    # must be the one localm saved. Default keeps ComfyUI's own copies.
    contain_warning = contain_comfy_artifacts(
        api_url, prompt_id,
        {"filename": audio_info.get("filename"),
         "subfolder": audio_info.get("subfolder", ""),
         "type": audio_info.get("type", "output")},
        delete_outputs=delete_outputs,
    )

    # Sidecar JSON - everything needed to reproduce or tweak the track.
    # Skipped entirely in privacy mode (write_sidecar=False) so the prompt
    # and lyrics never touch disk.
    if not write_sidecar:
        return True, _with_warning(
            f"Track saved to {output_path} "
            f"(seed {seed} - reuse it to reproduce)", contain_warning)
    try:
        sidecar = {
            "tags": tags,
            "lyrics": None if lyrics_text == _INSTRUMENTAL else lyrics_text,
            "duration_seconds": duration_seconds,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "lyrics_strength": lyrics_strength,
            "elapsed_seconds": round(time.time() - start_time, 1),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        output_path.with_suffix(output_path.suffix + ".json").write_text(
            json.dumps({k: v for k, v in sidecar.items() if v is not None},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        # Surface, don't silence: the track is the deliverable and is already
        # saved, so note the sidecar miss in the message instead of failing.
        contain_warning = _with_warning(
            "the reproducibility sidecar could not be saved "
            f"({e}); the track itself was saved.", contain_warning)

    return True, _with_warning(
        f"Track saved to {output_path} "
        f"(seed {seed} - reuse it to reproduce)", contain_warning)
