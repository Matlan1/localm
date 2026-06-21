# SPDX-License-Identifier: AGPL-3.0-or-later
"""
ComfyUI FLUX image generation.

Standalone module - usable from the localcoder agent tool, the CLI,
or any other caller.  No coder-plugin dependencies.
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


# Your personal workflow (untracked - which models/encoders you use stays
# private). Falls back to the committed example template, which uses the
# vanilla public FLUX stack; export your own from ComfyUI (Save → API format)
# as flux_workflow.json to customise.
_WORKFLOW_PATH = Path(__file__).parent / "flux_workflow.json"
_WORKFLOW_EXAMPLE_PATH = Path(__file__).parent / "flux_workflow.example.json"


def _workflow_path() -> Path:
    return _WORKFLOW_PATH if _WORKFLOW_PATH.is_file() else _WORKFLOW_EXAMPLE_PATH


# ---------------------------------------------------------------------------
#  VRAM management
# ---------------------------------------------------------------------------

def _localm_unload(localm_url: Optional[str] = None) -> Optional[dict]:
    """
    Ask a localm server to release its model from GPU memory.

    Reads LOCALM_URL from the environment if *localm_url* is not given, and
    authenticates with the LOCALM_API_KEY bearer token when one is set. The
    ``/v1/models/unload`` endpoint requires the models-write scope, so an
    UNAUTHENTICATED POST is rejected with 401 and the chat model stays resident
    in VRAM - the media model then loads on top of it, exceeds total VRAM and
    hangs the GPU driver (the AMD TDR the user hit). For the same reason the
    built-in TLS cert of a loopback ``https`` self-call must be trusted, exactly
    as the media-job model reload does (``localm.tls.requests_verify``); plain
    ``urllib`` would reject the self-signed cert and silently skip the unload.

    Silent no-op when the URL is unset. Returns the server's JSON result
    (``status`` / ``vram_freed`` / ``vram_before_bytes`` / ``vram_after_bytes``)
    on success, or None on any failure - never blocks generation if localm is
    not in the picture.
    """
    url = (localm_url or os.environ.get("LOCALM_URL", "")).rstrip("/")
    if not url:
        return None
    try:
        import requests as _rq

        from localm import tls as _tls
        headers = {}
        key = os.environ.get("LOCALM_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        # Unload waits for any in-flight generation to finish AND for VRAM to be
        # actually reclaimed before it returns (it must not free the context
        # mid-decode), so give it time.
        resp = _rq.post(f"{url}/models/unload", headers=headers, timeout=180,
                        verify=_tls.requests_verify(url))
        if not resp.ok:
            return None
        try:
            return resp.json()
        except Exception:
            return {"status": "unloaded"}
    except Exception:
        return None


def default_api_url() -> str:
    """ComfyUI base URL: FLUX_API_URL env override, else the default port."""
    return os.environ.get("FLUX_API_URL", "http://127.0.0.1:8188").rstrip("/")


def free_comfy_vram(api_url: Optional[str] = None) -> bool:
    """
    Ask ComfyUI to unload its models and free VRAM (POST /free).

    Returns True when ComfyUI accepted the request. Used after generation
    so the chat model can be reloaded immediately instead of spilling into
    system RAM next to a resident FLUX. Older ComfyUI builds without /free
    return False; callers should then leave the LLM reload lazy.
    """
    url = (api_url or default_api_url()).rstrip("/")
    try:
        body = json.dumps({"unload_models": True, "free_memory": True}).encode()
        req = urllib.request.Request(
            f"{url}/free", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=30):
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _comfy_alive(api_url: str, timeout: float = 3.0) -> bool:
    """Quick reachability probe so callers can fail fast with a clear error."""
    try:
        with urllib.request.urlopen(f"{api_url}/system_stats", timeout=timeout):
            return True
    except Exception:
        return False


def _derive_workdir_from_cmd(launch_cmd: str) -> Optional[str]:
    """The folder of the launcher script, so a .bat / .sh that references paths
    relative to its own location (the ComfyUI + ZLUDA convention, e.g. a copied
    ``launch-comfyui.bat`` next to ``python_embeded`` / ``venv``) runs from the
    right place even when ``comfy_workdir`` was not set. Best-effort: returns the
    parent of the first token that is an existing file, else None."""
    import shlex
    try:
        tokens = shlex.split(launch_cmd, posix=(os.name != "nt"))
    except ValueError:
        tokens = launch_cmd.split()
    for tok in tokens:
        tok = tok.strip().strip('"').strip("'")
        if not tok:
            continue
        try:
            p = Path(tok)
        except Exception:
            break
        if p.is_file():
            return str(p.parent)
        break  # only the first token is the program/script being launched
    return None


def ensure_comfy(api_url: Optional[str] = None, on_progress=None,
                 wait_seconds: Optional[int] = None,
                 launch_cmd: Optional[str] = None,
                 workdir: Optional[str] = None) -> tuple[bool, str]:
    """
    Make sure ComfyUI is reachable, launching it when configured.

    Used by every generator (image, music, video) from any caller - GUI,
    CLI, or the coder's generate_image tool. When ComfyUI is down and a launch
    command is configured, the command is started (optionally in *workdir*) and
    polled until the API answers.

    *launch_cmd* / *workdir* let a caller pass per-plugin config; when not given
    they fall back to the global ``comfy_launch_cmd`` / ``comfy_workdir`` config
    keys (kept for callers not yet migrated to per-plugin config).

    Returns (ok, message); the message explains what to configure when
    nothing could be launched.
    """
    import shlex
    import subprocess
    import sys as _sys
    import time as _t
    from localm.config import load_config

    def _say(text: str) -> None:
        if on_progress:
            try:
                on_progress(text)
            except Exception:
                pass

    api_url = (api_url or default_api_url()).rstrip("/")
    if _comfy_alive(api_url):
        return True, "ComfyUI is running."

    cfg = load_config()
    if not launch_cmd:
        launch_cmd = cfg.get("comfy_launch_cmd")
    if not launch_cmd:
        return False, (
            f"ComfyUI is not reachable at {api_url}.\n"
            "Start ComfyUI first (default: http://127.0.0.1:8188), set the "
            "FLUX_API_URL environment variable if it runs elsewhere, or let "
            "localm start it for you:\n"
            '  localm config comfy_launch_cmd "<path>\\launch-comfyui.bat"\n'
            '  localm config comfy_workdir "<path>\\ComfyUI"   (optional cwd)'
        )

    # A ZLUDA / ROCm cold start compiles GPU kernels and can take minutes, so
    # honour the configurable timeout when the caller did not pin one.
    if wait_seconds is None:
        try:
            wait_seconds = int(cfg.get("comfy_launch_timeout") or 300)
        except (TypeError, ValueError):
            wait_seconds = 300
    wait_seconds = max(30, wait_seconds)

    _say(f"ComfyUI not running - launching: {launch_cmd}")
    # The command is the user's own config value (their launcher script).
    # On Windows pass `cmd /S /c "<line>"` as a single string: /S strips the
    # outer quotes and runs the line verbatim, so quoted executable paths
    # survive (a `["cmd", "/c", line]` list gets re-quoted by subprocess and
    # mangles them). POSIX uses shlex.
    if workdir is None:
        workdir = cfg.get("comfy_workdir")
    if not workdir:
        # No explicit cwd: run the launcher from its own folder so a .bat/.sh
        # that uses paths relative to itself (ComfyUI + ZLUDA) still works.
        workdir = _derive_workdir_from_cmd(launch_cmd)
        if workdir:
            _say(f"Running the launcher from {workdir}")
    workdir = workdir or None
    if _sys.platform == "win32":
        argv: "str | list" = 'cmd /S /c "' + launch_cmd + '"'
    else:
        argv = shlex.split(launch_cmd)
    try:
        subprocess.Popen(argv, cwd=workdir,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except Exception as e:
        return False, f"Could not launch ComfyUI ({launch_cmd}): {e}"

    deadline = _t.monotonic() + wait_seconds
    last_said = 0.0
    while _t.monotonic() < deadline:
        if _comfy_alive(api_url):
            return True, "ComfyUI is up."
        elapsed = wait_seconds - (deadline - _t.monotonic())
        if elapsed - last_said >= 15:
            _say(f"Waiting for ComfyUI… ({int(elapsed)}s)")
            last_said = elapsed
        _t.sleep(2)
    return False, (
        f"ComfyUI did not come up within {wait_seconds // 60} minutes - "
        "check the launcher window for errors. If it is just a slow first "
        "(ZLUDA / ROCm) start, raise the limit: "
        "localm config comfy_launch_timeout 600"
    )


def comfy_http_error_detail(e: "urllib.error.HTTPError") -> str:
    """
    Human-readable detail from a ComfyUI /prompt error response.

    A 400 from /prompt means the workflow failed validation - not a
    connectivity problem. The response body is JSON naming the failing
    node and why (a model file missing from ComfyUI's models directory
    is the usual cause). Shared by image, music, and video generation.
    """
    try:
        body = json.loads(e.read().decode("utf-8", "replace"))
    except Exception:
        return f"HTTP {e.code}: {e.reason}"
    lines = []
    err = body.get("error") or {}
    if err.get("message"):
        msg = err["message"]
        if err.get("details"):
            msg += f" - {err['details']}"
        lines.append(msg)
    for node_id, info in (body.get("node_errors") or {}).items():
        cls = info.get("class_type") or f"node {node_id}"
        for ne in info.get("errors", []):
            msg = ne.get("message", "")
            if ne.get("details"):
                msg += f" ({ne['details']})"
            lines.append(f"{cls}: {msg}")
    return "\n".join(lines) or f"HTTP {e.code}: {e.reason}"


def _image_dimensions(path: Path) -> tuple[int, int]:
    """Return (width, height) from a PNG or JPEG without any external libs."""
    try:
        with open(path, "rb") as f:
            data = f.read(32)
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
        if data[:2] == b"\xff\xd8":
            # JPEG - scan for SOF0/SOF2 markers
            full = path.read_bytes()
            i = 2
            while i < len(full) - 8:
                if full[i] != 0xFF:
                    break
                marker = full[i + 1]
                length = int.from_bytes(full[i + 2:i + 4], "big")
                if marker in (0xC0, 0xC2):
                    h = int.from_bytes(full[i + 5:i + 7], "big")
                    w = int.from_bytes(full[i + 7:i + 9], "big")
                    return w, h
                i += 2 + length
    except Exception:
        pass
    return 1024, 1024


def _upload_image(image_path: Path, api_url: str) -> str:
    """
    Upload a local image to ComfyUI via POST /upload/image.

    Returns the filename ComfyUI assigned (used in the LoadImage node).
    Raises on failure.
    """
    boundary = "LocalcoderUploadBoundary"
    img_bytes = image_path.read_bytes()
    content_type = "image/jpeg" if image_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'
        f"Content-Type: {content_type}\r\n"
        f"\r\n"
    ).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{api_url}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode())
    name = result.get("name")
    if not name:
        raise RuntimeError(f"ComfyUI upload returned no filename: {result}")
    return name


# ---------------------------------------------------------------------------
#  Output containment
# ---------------------------------------------------------------------------
#
#  localm's rule: NOTHING a generation produces may remain visible inside
#  ComfyUI - the only copy is the one localm saved. Left alone, ComfyUI
#  (a) records every job in /history, which the Queue/History panel and gallery
#  extensions read, and (b) keeps the rendered file in its own output/ dir, plus
#  any img2img source we uploaded into input/. After the artifact has been
#  fetched into localm's own location we therefore clear the history entry AND
#  delete ComfyUI's on-disk copies. Deleting files needs ComfyUI's output/ dir;
#  when it cannot be resolved we still clear history and return a loud warning
#  rather than leaking silently. Shared by image, music, and video generation.

def _comfy_output_root(comfy_output_dir: Optional[str] = None) -> Optional[Path]:
    """ComfyUI's output/ directory, or None when it cannot be resolved.

    Order: explicit arg, COMFY_OUTPUT_DIR env, the ``comfy_output_dir`` config
    key, then a derived ``<comfy_workdir>/output`` when that exists."""
    cand = comfy_output_dir or os.environ.get("COMFY_OUTPUT_DIR")
    if not cand:
        try:
            from localm.config import load_config
            cfg = load_config()
            cand = cfg.get("comfy_output_dir")
            if not cand:
                wd = cfg.get("comfy_workdir")
                if wd:
                    derived = Path(wd) / "output"
                    if derived.is_dir():
                        return derived
        except Exception:
            return None
    return Path(cand) if cand else None


def clear_comfy_history(api_url: str, prompt_id: str) -> bool:
    """Remove this job from ComfyUI's /history (the Queue/History panel and
    gallery views read it). POST /history {"delete": [prompt_id]}. Best-effort;
    returns True when ComfyUI accepted the request."""
    if not prompt_id:
        return False
    try:
        body = json.dumps({"delete": [prompt_id]}).encode()
        req = urllib.request.Request(
            f"{api_url}/history", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10):
            return True
    except Exception:
        return False


def contain_comfy_artifacts(
    api_url: str,
    prompt_id: str,
    info: dict,
    *,
    comfy_output_dir: Optional[str] = None,
    uploaded_input: Optional[str] = None,
) -> str:
    """Enforce output containment after an artifact has been fetched locally.

    Clears the ComfyUI /history entry, deletes ComfyUI's own copy of the output
    (when its output dir is known), and removes any img2img source we uploaded
    into ComfyUI's input/ dir. Returns a warning string to surface to the user
    when full containment could NOT be guaranteed (the on-disk copy could not be
    deleted), or "" when fully contained."""
    clear_comfy_history(api_url, prompt_id)

    root = _comfy_output_root(comfy_output_dir)

    # Remove the uploaded img2img source from ComfyUI's input/ dir (sibling of
    # output/). Best-effort: the source image is not in the output browser, but
    # it is still a stray copy of the user's input.
    if uploaded_input and root is not None:
        try:
            inp = root.parent / "input" / uploaded_input
            if inp.exists():
                inp.unlink()
        except Exception:
            pass

    # Delete ComfyUI's on-disk copy of the output. type "temp" is auto-purged
    # by ComfyUI, so only a real "output" artifact needs removing.
    if (info.get("type") or "output") != "output":
        return ""
    if root is None:
        return (
            "WARNING: a copy of this file remains in ComfyUI's output folder. "
            "localm cleared the ComfyUI history entry, but cannot delete the "
            "file until you set the ComfyUI output dir (Settings -> ComfyUI "
            "output dir, or: localm config comfy_output_dir <path>)."
        )
    try:
        copy = root / info.get("subfolder", "") / info.get("filename", "")
        if copy.exists():
            copy.unlink()
        return ""
    except Exception as e:
        return f"WARNING: could not delete ComfyUI's copy of the output ({e})."


def interrupt_comfy(api_url: str) -> bool:
    """Abort ComfyUI's currently running prompt and clear its queue (POST
    ``/interrupt`` + POST ``/queue {"clear": true}``).

    Best-effort, used to honour a user's Stop so a long media gen actually stops
    instead of running to completion. Shared by image, music, and video. Returns
    True when the interrupt was accepted."""
    ok = False
    try:
        req = urllib.request.Request(f"{api_url}/interrupt", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=10):
            ok = True
    except Exception:
        pass
    try:
        body = json.dumps({"clear": True}).encode()
        req = urllib.request.Request(
            f"{api_url}/queue", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception:
        pass
    return ok


def _with_warning(message: str, warning: str) -> str:
    """Append a containment warning to a success message when present."""
    return f"{message}\n{warning}" if warning else message


# ---------------------------------------------------------------------------
#  Image generation
# ---------------------------------------------------------------------------

def generate_image(
    prompt: str,
    output_path: Path,
    *,
    api_url: str = "http://127.0.0.1:8188",
    guidance: Optional[float] = None,
    negative_prompt: Optional[str] = None,
    cfg: Optional[float] = None,
    seed: Optional[int] = None,
    clip_name1: Optional[str] = None,
    clip_name2: Optional[str] = None,
    lora_name: Optional[str] = None,
    lora_strength_model: float = 1.0,
    lora_strength_clip: float = 0.5,
    input_image: Optional[Path] = None,
    denoise: Optional[float] = None,
    localm_url: Optional[str] = None,
    max_poll_seconds: int = 600,
    write_sidecar: bool = True,
    launch_cmd: Optional[str] = None,
    workdir: Optional[str] = None,
    comfy_output_dir: Optional[str] = None,
    swap: bool = True,
    cancel_check: Optional[callable] = None,
) -> tuple[bool, str]:
    """
    Generate an image from *prompt* and save it to *output_path*.

    Parameters
    ----------
    prompt
        Descriptive text prompt.  For img2img, describe what to *change*
        rather than the full scene - the base image already provides structure.
    output_path
        Destination file (PNG).  Parent directories are created if needed.
    api_url
        ComfyUI base URL.  Defaults to ``http://127.0.0.1:8188``.
        Override with the ``FLUX_API_URL`` environment variable before calling.
    guidance
        FluxGuidance scale.  None keeps the workflow's own default (~3.5).
    negative_prompt
        Things to steer away from (e.g. ``"old, mature, middle-aged"``).
        A real negative requires classifier-free guidance, so when this is
        set the workflow's single-pass ``BasicGuider`` is swapped for a
        ``CFGGuider`` with a dedicated negative branch and ``cfg`` > 1 (see
        below).  This roughly doubles inference time (two forward passes per
        step).  Leave it None to keep the fast single-pass path.
    cfg
        Classifier-free guidance scale for the negative branch.  Only used
        when *negative_prompt* is set; ``None`` defaults to 3.5.  A value of
        1.0 disables the negative entirely (the negative branch is ignored),
        higher values push harder away from it.  Note: guidance-*distilled*
        FLUX (the vanilla dev checkpoint) tends to over-saturate at cfg > 1;
        de-distilled checkpoints (e.g. the "unchained" variants) handle it
        cleanly.  Distinct from *guidance*, which is FLUX's own distilled
        guidance embedding and applies to both branches.
    seed
        Noise seed for reproducible outputs.  Randomised if not given.
    clip_name1
        Override the CLIP-L encoder filename in the workflow.
        Useful for comparing encoder variants without editing the workflow JSON.
    clip_name2
        Override the T5 encoder filename.  If the name ends in ``.gguf``,
        the node is automatically switched to ``DualCLIPLoaderGGUF``.
    lora_name
        LoRA filename to inject (optional).
    lora_strength_model
        How strongly the LoRA patches the UNet weights (default 1.0).
        This is the main lever for unlock/style LoRAs.
    lora_strength_clip
        How strongly the LoRA patches the text encoder (default 0.5).
        Lower than model strength is usually correct for unlock LoRAs -
        the base CLIP already understands the vocabulary.
    input_image
        Path to an existing image to use as the starting point (img2img mode).
        When provided, FLUX refines this image guided by *prompt* instead of
        generating from noise.  Output dimensions match the input image.
    denoise
        How much to change the input image (img2img only).
        0.0 = no change, 1.0 = completely new image.
        Defaults to 0.75 when *input_image* is set and not explicitly given.
        Ignored in txt2img mode.
    localm_url
        localm server URL (e.g. ``http://127.0.0.1:8642/v1``) to unload
        before generation so FLUX gets the full VRAM budget.
        Reads ``LOCALM_URL`` env var if None.  Skipped silently when unset.
    max_poll_seconds
        Timeout waiting for ComfyUI to finish (default 10 minutes).
    write_sidecar
        Write a ``<output>.json`` sidecar with the prompt and settings so
        the image can be reproduced.  Pass False in privacy mode - the
        prompt then never touches disk.

    Returns
    -------
    (ok, message)
        ``ok=True`` and a success description, or ``ok=False`` and an error.
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    _con = Console()

    # 0. Make sure ComfyUI is up (auto-launching when configured) - BEFORE
    # unloading the LLM, so a dead image server doesn't cost the user a
    # pointless model unload + reload
    ok, msg = ensure_comfy(api_url, on_progress=lambda t: _con.print(f"[dim]{t}[/dim]"),
                           launch_cmd=launch_cmd, workdir=workdir)
    if not ok:
        return False, msg

    # 1. Unload LLM to free VRAM before FLUX loads. Skipped when the caller
    # decided the media model fits alongside the chat model (swap=False), so the
    # chat model stays hot.
    if swap:
        _localm_unload(localm_url)

    # 2. Load workflow template (personal flux_workflow.json if present,
    # else the committed example)
    try:
        workflow = json.loads(_workflow_path().read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Failed to load FLUX workflow template: {e}"

    # 3. Override text encoder models if requested
    if clip_name1 is not None or clip_name2 is not None:
        loader_node = None
        if "31" in workflow:
            loader_node = workflow["31"]
        else:
            for node in workflow.values():
                if node.get("class_type") in ("DualCLIPLoader", "DualCLIPLoaderGGUF"):
                    loader_node = node
                    break
        if loader_node is not None:
            if clip_name1 is not None:
                loader_node["inputs"]["clip_name1"] = clip_name1
            if clip_name2 is not None:
                loader_node["inputs"]["clip_name2"] = clip_name2
                # GGUF T5 needs a different loader node
                if clip_name2.lower().endswith(".gguf"):
                    loader_node["class_type"] = "DualCLIPLoaderGGUF"
                else:
                    loader_node["class_type"] = "DualCLIPLoader"

    # 4. img2img: upload input image, add LoadImage + VAEEncode, redirect latent
    uploaded_name: Optional[str] = None
    if input_image is not None:
        if not input_image.is_file():
            return False, f"Input image not found: {input_image}"
        try:
            uploaded_name = _upload_image(input_image, api_url)
        except Exception as e:
            return False, f"Failed to upload input image to ComfyUI: {e}"

        w, h = _image_dimensions(input_image)

        # LoadImage node - ComfyUI loads from its own input/ dir by filename
        workflow["40"] = {
            "inputs": {"image": uploaded_name, "upload": "image"},
            "class_type": "LoadImage",
        }
        # VAEEncode - encode the loaded image into latent space
        workflow["41"] = {
            "inputs": {"pixels": ["40", 0], "vae": ["10", 0]},
            "class_type": "VAEEncode",
        }
        # Redirect SamplerCustomAdvanced latent input from EmptyLatentImage to encoded image
        workflow["13"]["inputs"]["latent_image"] = ["41", 0]

        # Update ModelSamplingFlux dimensions so RoPE embeddings match the image
        workflow["28"]["inputs"]["width"]  = w
        workflow["28"]["inputs"]["height"] = h

        # Set denoise on the scheduler
        workflow["17"]["inputs"]["denoise"] = denoise if denoise is not None else 0.75

    # 5. Inject prompt - node "6" first (default template), then scan
    injected = False
    if "6" in workflow and workflow["6"].get("inputs", {}).get("text") is not None:
        workflow["6"]["inputs"]["text"] = prompt
        injected = True
    if not injected:
        for node in workflow.values():
            if node.get("class_type") == "CLIPTextEncode":
                node["inputs"]["text"] = prompt
                injected = True
                break
    if not injected:
        return False, (
            "Could not find a text-prompt node in the workflow template.\n"
            "Export a fresh workflow from ComfyUI (Save → API format) and replace\n"
            f"{_WORKFLOW_PATH}"
        )

    # 6. Inject guidance
    if guidance is not None:
        if "26" in workflow and workflow["26"].get("class_type") == "FluxGuidance":
            workflow["26"]["inputs"]["guidance"] = guidance
        else:
            for node in workflow.values():
                if node.get("class_type") == "FluxGuidance":
                    node["inputs"]["guidance"] = guidance
                    break

    # 7. Inject LoRA
    if lora_name:
        workflow["100"] = {
            "inputs": {
                "model": ["30", 0],
                "clip": ["31", 0],
                "lora_name": lora_name,
                "strength_model": lora_strength_model,
                "strength_clip": lora_strength_clip,
            },
            "class_type": "LoraLoader",
        }
        if "28" in workflow:
            workflow["28"]["inputs"]["model"] = ["100", 0]
        if "6" in workflow:
            workflow["6"]["inputs"]["clip"] = ["100", 1]

    # 8. Inject negative prompt via real classifier-free guidance.
    #    A negative prompt only works if the model sees a SEPARATE negative
    #    conditioning and subtracts it (cfg > 1). The default workflow uses a
    #    BasicGuider, which has only a positive `conditioning` input and runs
    #    at an implicit cfg of 1 - it has no way to express a negative. We swap
    #    it for a CFGGuider (model, positive, negative, cfg) and build a
    #    dedicated negative branch.
    #
    #    Do NOT use ConditioningConcat here: it APPENDS the negative tokens to
    #    the positive prompt, which makes the model draw those things *more* -
    #    the exact opposite of a negative prompt.
    if negative_prompt:
        neg_cfg = cfg if cfg is not None else 3.5
        guide_scale = guidance if guidance is not None else 3.5
        # Use the LoRA-patched CLIP if a LoRA was injected, otherwise raw DualCLIPLoader
        clip_source = ["100", 1] if lora_name else ["31", 0]

        # Encode the negative prompt on its own branch and give it the same
        # FLUX guidance embedding as the positive side, so both live in the
        # same conditioned space when the sampler compares them.
        workflow["50"] = {
            "inputs": {"text": negative_prompt, "clip": clip_source},
            "class_type": "CLIPTextEncode",
        }
        workflow["52"] = {
            "inputs": {"guidance": guide_scale, "conditioning": ["50", 0]},
            "class_type": "FluxGuidance",
        }

        # Convert the guider into a CFGGuider wired to both branches.
        guider_id = None
        for nid, node in workflow.items():
            if node.get("class_type") in ("BasicGuider", "CFGGuider"):
                guider_id = nid
                break
        if guider_id is not None:
            g = workflow[guider_id]
            # BasicGuider's positive lives under "conditioning"; a CFGGuider
            # we built on a previous override carries it under "positive".
            positive = g["inputs"].get("positive") or g["inputs"].get("conditioning", ["26", 0])
            g["class_type"] = "CFGGuider"
            g["inputs"] = {
                "model": g["inputs"]["model"],
                "positive": positive,
                "negative": ["52", 0],
                "cfg": neg_cfg,
            }

    # 9. Set seed (use provided value or randomise) - on every noise/sampler
    # node, so workflows with more than one of them stay reproducible
    seed = seed if seed is not None else random.randint(1, 10 ** 12)
    for node in workflow.values():
        cls = node.get("class_type", "")
        if cls in ("KSampler", "KSamplerAdvanced"):
            node["inputs"]["seed"] = seed
        elif cls == "RandomNoise":
            node["inputs"]["noise_seed"] = seed

    # 9. Queue the prompt in ComfyUI
    try:
        req_data = json.dumps({"prompt": workflow}).encode("utf-8")
        req = urllib.request.Request(
            f"{api_url}/prompt",
            data=req_data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            res = json.loads(response.read().decode("utf-8"))
            prompt_id = res.get("prompt_id")

        if not prompt_id:
            return False, (
                "ComfyUI accepted the request but returned no prompt_id.\n"
                "Check the ComfyUI console for workflow validation errors."
            )

    except urllib.error.HTTPError as e:
        return False, (
            f"ComfyUI rejected the workflow (HTTP {e.code}):\n"
            f"{comfy_http_error_detail(e)}\n"
            "A model file missing from ComfyUI's models directory is the "
            "usual cause - check the names in your workflow template."
        )
    except urllib.error.URLError as e:
        return False, (
            f"Could not connect to ComfyUI at {api_url}.\n"
            f"Error: {e}\n"
            "Make sure ComfyUI is running."
        )
    except Exception as e:
        return False, f"Error queuing prompt in ComfyUI: {e}"

    # 10. Poll /history with a visible progress spinner
    start_time = time.time()
    finished = False
    filename = None
    subfolder = ""
    img_type = "output"

    with Progress(
        SpinnerColumn(),
        TextColumn("[dim]{task.description}[/dim]"),
        TimeElapsedColumn(),
        transient=True,
        console=_con,
    ) as progress:
        task_id = progress.add_task("Generating image…", total=None)

        while time.time() - start_time < max_poll_seconds:
            if cancel_check and cancel_check():
                interrupt_comfy(api_url)
                clear_comfy_history(api_url, prompt_id)
                return False, "Generation cancelled."
            elapsed = int(time.time() - start_time)
            progress.update(task_id, description=f"Generating image… ({elapsed}s)")

            try:
                hist_req = urllib.request.Request(f"{api_url}/history/{prompt_id}")
                with urllib.request.urlopen(hist_req, timeout=5) as response:
                    history = json.loads(response.read().decode("utf-8"))

                if prompt_id in history:
                    finished = True
                    for node_output in history[prompt_id].get("outputs", {}).values():
                        if "images" in node_output:
                            img_info = node_output["images"][0]
                            filename = img_info.get("filename")
                            subfolder = img_info.get("subfolder", "")
                            img_type = img_info.get("type", "output")
                            break
                    break

            except Exception:
                pass

            time.sleep(2)

    if not finished:
        return False, f"Image generation timed out after {max_poll_seconds // 60} minutes."

    if not filename:
        return False, (
            "Generation finished but no output image was found in ComfyUI history.\n"
            "Check the ComfyUI console - a SaveImage node error is likely."
        )

    # 11. Fetch image from ComfyUI /view, save locally, strip metadata
    try:
        params = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": img_type}
        )
        img_url = f"{api_url}/view?{params}"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with urllib.request.urlopen(img_url, timeout=10) as response:
            output_path.write_bytes(response.read())

        # Strip PNG metadata/EXIF for privacy (pure-Python, zero deps)
        try:
            png_bytes = output_path.read_bytes()
            if png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                clean = bytearray(b"\x89PNG\r\n\x1a\n")
                pos, limit = 8, len(png_bytes)
                while pos + 8 <= limit:
                    length = int.from_bytes(png_bytes[pos:pos + 4], "big")
                    chunk_type = png_bytes[pos + 4:pos + 8]
                    total_len = 12 + length
                    if chunk_type not in (b"tEXt", b"zTXt", b"iTXt", b"eXIf"):
                        clean.extend(png_bytes[pos:pos + total_len])
                    pos += total_len
                output_path.write_bytes(bytes(clean))
        except Exception:
            pass

        # Enforce output containment: clear ComfyUI's history entry (the
        # Queue/History + gallery view) and delete ComfyUI's own on-disk copy
        # of the image plus any uploaded img2img source. Returns a warning when
        # the file copy could not be removed (e.g. comfy_output_dir unset).
        contain_warning = contain_comfy_artifacts(
            api_url, prompt_id,
            {"filename": filename, "subfolder": subfolder, "type": img_type},
            comfy_output_dir=comfy_output_dir,
            uploaded_input=uploaded_name,
        )

        # Sidecar JSON: everything needed to reproduce or tweak this image.
        # Saved as <output>.json next to the image; failure is non-fatal.
        # Skipped entirely in privacy mode (write_sidecar=False).
        if not write_sidecar:
            return True, _with_warning(
                f"Image saved to {output_path} "
                f"(seed {seed} - reuse it to reproduce)", contain_warning)
        try:
            sidecar = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "cfg": (cfg if cfg is not None else 3.5) if negative_prompt else None,
                "seed": seed,
                "guidance": guidance,
                "lora_name": lora_name,
                "lora_strength_model": lora_strength_model if lora_name else None,
                "lora_strength_clip": lora_strength_clip if lora_name else None,
                "input_image": str(input_image) if input_image else None,
                "denoise": (denoise if denoise is not None else 0.75)
                           if input_image else None,
                "clip_name1": clip_name1,
                "clip_name2": clip_name2,
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

        return True, _with_warning(
            f"Image saved to {output_path} (seed {seed} - reuse it to reproduce)",
            contain_warning)

    except Exception as e:
        return False, f"Failed to download generated image from ComfyUI: {e}"
