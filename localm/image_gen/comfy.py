# SPDX-License-Identifier: AGPL-3.0-or-later
"""
ComfyUI FLUX image generation.

Standalone module - usable from the localcoder agent tool, the CLI,
or any other caller.  No coder-plugin dependencies.

The generic ComfyUI plumbing (role resolution, model preflight, the VRAM
handoff, server launch, the queue / poll / download transport, output
containment) lives in ``localm.media.comfy_client`` and is shared with the
video and music generators. This module keeps only the FLUX-specific workflow
shaping. The shared helpers are re-imported into this module's namespace and
called as bare globals so monkeypatching ``localm.image_gen.comfy.<name>``
still takes effect.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Optional

# Bound in this namespace so localm.image_gen.comfy.os and
# localm.image_gen.comfy.urllib.request resolve; the shared client in
# localm.media.comfy_client calls the SAME module objects.
import os  # noqa: F401
import urllib.request  # noqa: F401

# Shared ComfyUI plumbing. Imported as bare module globals (not via the package)
# so a test patching localm.image_gen.comfy.<name> still rebinds what
# generate_image / apply_fast_dequant call.
from localm.media.comfy_client import (
    _amd_rocm_launch_env,
    _combo_options,
    _comfy_alive,
    _comfy_output_root,
    _derive_workdir_from_cmd,
    _format_missing,
    _image_dimensions,
    _is_link,
    _link_source_id,
    _localm_unload,
    _looks_like_model_files,
    _normalize_model_base,
    _pick_variant,
    _upload_image,
    _venv_python,
    _with_warning,
    apply_model_overrides,
    clear_comfy_history,
    comfy_console_tail_start,
    comfy_console_warnings_since,
    comfy_exec_error_message,
    comfy_fetch_output,
    comfy_http_error_detail,
    comfy_object_info,
    comfy_poll_until_done,
    comfy_submit_prompt,
    contain_comfy_artifacts,
    default_api_url,
    discover_launch_cmd,
    ensure_comfy,
    find_node_by_class,
    find_nodes_by_class,
    free_comfy_vram,
    history_execution_error,
    inject_device_placement,
    interrupt_comfy,
    next_node_id,
    POLL_CANCELLED,
    POLL_EXEC_ERROR,
    POLL_FINISHED,
    POLL_TIMEOUT,
    preflight_models,
    resolve_sampler_roles,
    sanitize_comfy_url,
    sanitize_comfy_url_checked,
    set_seed_on,
    set_seed_on_all,
    SUBMIT_ERROR,
    SUBMIT_HTTP_ERROR,
    SUBMIT_NO_ID,
    SUBMIT_OK,
    SUBMIT_URL_ERROR,
    workflow_model_slots,
)

# load_config is referenced by some tests as a module attribute (a soft,
# raising=False patch); the real resolution still happens via localm.config
# inside ensure_comfy. Re-export it here so that attribute exists.
from localm.config import load_config  # noqa: F401

# Re-export the shared ComfyUI helpers as part of this module's surface, so they
# are reachable and patchable as localm.image_gen.comfy.<name>.
__all__ = [
    "generate_image", "apply_fast_dequant", "is_safe_lora_name",
    "_amd_rocm_launch_env", "_combo_options", "_comfy_alive",
    "_comfy_output_root", "_derive_workdir_from_cmd", "_format_missing",
    "_image_dimensions", "_is_link", "_link_source_id", "_localm_unload",
    "_looks_like_model_files", "_normalize_model_base", "_pick_variant",
    "_upload_image", "_venv_python", "_with_warning",
    "apply_model_overrides", "clear_comfy_history",
    "comfy_console_tail_start", "comfy_console_warnings_since",
    "comfy_exec_error_message",
    "comfy_fetch_output", "comfy_http_error_detail", "comfy_object_info",
    "comfy_poll_until_done", "comfy_submit_prompt", "contain_comfy_artifacts",
    "default_api_url", "discover_launch_cmd", "ensure_comfy",
    "find_node_by_class", "find_nodes_by_class", "free_comfy_vram",
    "history_execution_error", "interrupt_comfy", "next_node_id",
    "POLL_CANCELLED", "POLL_EXEC_ERROR", "POLL_FINISHED", "POLL_TIMEOUT",
    "preflight_models", "resolve_sampler_roles", "sanitize_comfy_url",
    "sanitize_comfy_url_checked",
    "set_seed_on", "set_seed_on_all",
    "SUBMIT_ERROR", "SUBMIT_HTTP_ERROR", "SUBMIT_NO_ID", "SUBMIT_OK",
    "SUBMIT_URL_ERROR", "load_config", "workflow_model_slots",
]


# Your personal workflow (untracked - which models/encoders you use stays
# private). Falls back to the committed example template, which uses the
# vanilla public FLUX stack; export your own from ComfyUI (Save → API format)
# as flux_workflow.json to customise.
_WORKFLOW_PATH = Path(__file__).parent / "flux_workflow.json"
_WORKFLOW_EXAMPLE_PATH = Path(__file__).parent / "flux_workflow.example.json"


def workflow_path() -> Path:
    # Resolution order: 1. a workflow the user selected for the image plugin
    # (uploaded + picked on the Image page), 2. the personal
    # flux_workflow.json, 3. the committed example.
    try:
        from localm.media_workflows import active_workflow_path
        selected = active_workflow_path("image")
        if selected is not None:
            return selected
    except Exception:
        pass
    return _WORKFLOW_PATH if _WORKFLOW_PATH.is_file() else _WORKFLOW_EXAMPLE_PATH


# GGUF UNet loader node classes whose `dequant_dtype` controls how the quantized
# weights are unpacked for compute.
_GGUF_UNET_LOADERS = ("UnetLoaderGGUF", "UnetLoaderGGUFAdvanced")

# Common NTFS/ext4 filename-component limit.
_MAX_LORA_NAME_LEN = 255


def is_safe_lora_name(name: str) -> bool:
    """True when *name* is safe to embed as a ComfyUI ``LoraLoader.lora_name``
    value.

    A LoRA name is never a local filesystem path - ComfyUI resolves it against
    its OWN models directory, not localm's - so this is a pure lexical
    predicate with no filesystem call: reject empty, a NUL byte, a value longer
    than _MAX_LORA_NAME_LEN, any path separator, a bare "." / ".." component,
    or a ':' anywhere in the name (a colon anywhere opens an NTFS Alternate
    Data Stream on a Windows-hosted ComfyUI).

    Called from EVERY entry point that can supply ``lora_name``, including the
    coder agent's ``generate_image`` tool, not only the HTTP image route."""
    if not name or len(name) > _MAX_LORA_NAME_LEN or "\x00" in name:
        return False
    if "/" in name or "\\" in name or name in (".", ".."):
        return False
    return ":" not in name


def apply_fast_dequant(workflow: dict) -> int:
    """Rewrite a slow ``dequant_dtype: "float32"`` to the loader's fast default.

    "default" lets ComfyUI-GGUF dequant to the model's own compute dtype
    (fp16/bf16). Only the "float32" value is touched; an explicit
    "float16"/"bfloat16"/"target" choice is left alone. Mutates *workflow* in
    place and returns how many loader nodes were changed."""
    changed = 0
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") in _GGUF_UNET_LOADERS:
            inputs = node.get("inputs")
            if isinstance(inputs, dict) and inputs.get("dequant_dtype") == "float32":
                inputs["dequant_dtype"] = "default"
                changed += 1
    return changed


# ---------------------------------------------------------------------------
#  Image generation
# ---------------------------------------------------------------------------


def _build_image_workflow(
    workflow: dict,
    *,
    prompt: str,
    api_url: str,
    guidance: Optional[float],
    negative_prompt: Optional[str],
    cfg: Optional[float],
    seed: int,
    clip_name1: Optional[str],
    clip_name2: Optional[str],
    lora_name: Optional[str],
    lora_strength_model: float,
    lora_strength_clip: float,
    input_image: Optional[Path],
    denoise: Optional[float],
    fast_dequant: bool,
    con,
) -> tuple[bool, str, Optional[str]]:
    """Shape the FLUX workflow in place from the call's parameters.

    Returns ``(ok, message, uploaded_name)``: ``ok=False`` with an error message
    when the workflow cannot be driven (bad input image, no text-prompt node);
    ``uploaded_name`` is the ComfyUI-side filename of an uploaded img2img source
    (for later containment) or None. Pure workflow shaping; the only network I/O
    is the img2img upload."""
    # 2a. Rewrite a float32 GGUF dequant to the loader's fast default unless the
    # caller opted out.
    if fast_dequant:
        if apply_fast_dequant(workflow):
            con.print("[dim]Using fast fp16 GGUF dequant (was float32) for speed; "
                      "set comfy_fast_dequant=false to keep your workflow's value.[/dim]")

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
            return False, f"Input image not found: {input_image}", None
        try:
            uploaded_name = _upload_image(input_image, api_url)
        except Exception as e:
            return False, f"Failed to upload input image to ComfyUI: {e}", None

        w, h = _image_dimensions(input_image)

        # LoadImage node - ComfyUI loads from its own input/ dir by filename.
        # Fresh ids, so an injected node cannot clobber one a user's own
        # exported graph already uses.
        load_id = next_node_id(workflow)
        workflow[load_id] = {
            "inputs": {"image": uploaded_name, "upload": "image"},
            "class_type": "LoadImage",
        }
        # VAEEncode - encode the loaded image into latent space
        enc_id = next_node_id(workflow)
        workflow[enc_id] = {
            "inputs": {"pixels": [load_id, 0], "vae": ["10", 0]},
            "class_type": "VAEEncode",
        }
        # Redirect SamplerCustomAdvanced latent input from EmptyLatentImage to encoded image
        workflow["13"]["inputs"]["latent_image"] = [enc_id, 0]

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
        ), None

    # 6. Inject guidance
    if guidance is not None:
        if "26" in workflow and workflow["26"].get("class_type") == "FluxGuidance":
            workflow["26"]["inputs"]["guidance"] = guidance
        else:
            for node in workflow.values():
                if node.get("class_type") == "FluxGuidance":
                    node["inputs"]["guidance"] = guidance
                    break

    # 7. Inject LoRA (fresh id so it cannot collide with a user's own graph)
    lora_id: Optional[str] = None
    if lora_name:
        if not is_safe_lora_name(lora_name):
            return False, f"Invalid LoRA name: {lora_name!r}", None
        lora_id = next_node_id(workflow)
        workflow[lora_id] = {
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
            workflow["28"]["inputs"]["model"] = [lora_id, 0]
        if "6" in workflow:
            workflow["6"]["inputs"]["clip"] = [lora_id, 1]

    # 8. Inject negative prompt via real classifier-free guidance: the default
    #    workflow's BasicGuider has only a positive `conditioning` input and
    #    runs at an implicit cfg of 1, so it is swapped for a CFGGuider
    #    (model, positive, negative, cfg) with a dedicated negative branch.
    if negative_prompt:
        neg_cfg = cfg if cfg is not None else 3.5
        guide_scale = guidance if guidance is not None else 3.5
        # Use the LoRA-patched CLIP if a LoRA was injected, otherwise raw DualCLIPLoader
        clip_source = [lora_id, 1] if lora_id else ["31", 0]

        # Encode the negative prompt on its own branch and give it the same
        # FLUX guidance embedding as the positive side, so both live in the
        # same conditioned space when the sampler compares them. Fresh ids again.
        neg_text_id = next_node_id(workflow)
        workflow[neg_text_id] = {
            "inputs": {"text": negative_prompt, "clip": clip_source},
            "class_type": "CLIPTextEncode",
        }
        neg_guid_id = next_node_id(workflow)
        workflow[neg_guid_id] = {
            "inputs": {"guidance": guide_scale, "conditioning": [neg_text_id, 0]},
            "class_type": "FluxGuidance",
        }

        # Convert the guider into a CFGGuider wired to both branches.
        guider_id, guider = find_node_by_class(workflow, "BasicGuider", "CFGGuider")
        if guider is not None:
            g = guider
            # BasicGuider's positive lives under "conditioning"; a CFGGuider
            # we built on a previous override carries it under "positive".
            positive = g["inputs"].get("positive") or g["inputs"].get("conditioning", ["26", 0])
            g["class_type"] = "CFGGuider"
            g["inputs"] = {
                "model": g["inputs"]["model"],
                "positive": positive,
                "negative": [neg_guid_id, 0],
                "cfg": neg_cfg,
            }

    # 9. Set seed (use provided value or randomise) - on every noise/sampler
    # node, so workflows with more than one of them stay reproducible
    set_seed_on_all(workflow, seed)

    return True, "", uploaded_name


def _strip_png_metadata(output_path: Path) -> str:
    """Strip PNG metadata/EXIF for privacy (pure-Python, zero deps), in place.

    Returns a warning string when the strip did not run (the file is not a PNG,
    or the rewrite failed), else ""."""
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
        else:
            # Not a PNG (a custom workflow save node can emit WebP/JPEG to the
            # hardcoded .png path). The bytes are left untouched and the caller
            # gets a warning instead of a clean-strip result.
            return ("WARNING: generated file is not a PNG; image metadata could "
                    "not be stripped and may still contain prompt/EXIF data.")
    except (OSError, ValueError) as e:
        return ("WARNING: could not strip image metadata "
                f"({e}); the file may still contain prompt/EXIF data.")
    return ""


def _write_image_sidecar(
    output_path: Path,
    *,
    prompt: str,
    negative_prompt: Optional[str],
    cfg: Optional[float],
    seed: int,
    guidance: Optional[float],
    lora_name: Optional[str],
    lora_strength_model: float,
    lora_strength_clip: float,
    input_image: Optional[Path],
    denoise: Optional[float],
    clip_name1: Optional[str],
    clip_name2: Optional[str],
    start_time: float,
    comfy_console_warning: Optional[str] = None,
    comfy_console_checked: bool = False,
) -> str:
    """Write the ``<output>.json`` reproducibility sidecar next to the image.

    Returns a warning string on a write failure, else "".

    ``lora_name``/``lora_strength_model``/``lora_strength_clip`` record what
    was REQUESTED, not confirmed applied: ComfyUI can silently skip a
    mismatched LoRA (or checkpoint/VAE/CLIP weights) and still report success.
    ``comfy_console_warning`` and ``comfy_console_checked`` together
    distinguish three states:
      - checked=True,  warning present  -> localm SAW ComfyUI report a
        mismatch during this generation.
      - checked=True,  warning absent   -> localm read ComfyUI's console and
        found no KNOWN silent-partial-apply pattern (not a guarantee nothing
        was skipped - only that nothing in _COMFY_SILENT_PARTIAL_APPLY_PATTERNS
        matched).
      - checked=False, warning absent   -> localm could not read the console
        at all (a remote or already-running ComfyUI it did not launch
        itself) - this says NOTHING about whether anything was skipped."""
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
            "comfy_console_warning": comfy_console_warning,
            "comfy_console_checked": comfy_console_checked,
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
    except OSError as e:
        return ("WARNING: the reproducibility sidecar could not "
                f"be saved ({e}); the image itself was saved.")
    return ""


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
    model_overrides: Optional[dict] = None,
    lora_name: Optional[str] = None,
    lora_strength_model: float = 1.0,
    lora_strength_clip: float = 0.5,
    input_image: Optional[Path] = None,
    denoise: Optional[float] = None,
    localm_url: Optional[str] = None,
    instance_token: Optional[str] = None,
    max_poll_seconds: int = 600,
    write_sidecar: bool = True,
    launch_cmd: Optional[str] = None,
    workdir: Optional[str] = None,
    comfy_output_dir: Optional[str] = None,
    delete_outputs: bool = False,
    swap: bool = True,
    fast_dequant: bool = True,
    cancel_check: Optional[callable] = None,
    placement: Optional[dict] = None,
    on_progress: Optional[callable] = None,
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
    model_overrides
        Generic per-node model-slot overrides: ``{node_id: {input_name: value}}``,
        the shape ``localm.media.comfy_client.workflow_model_slots()`` returns
        slots in. Lets a caller override ANY model-file combo in the active
        workflow (the base UNET/checkpoint, VAE, or anything else a custom
        workflow adds) - not just clip_name1/clip_name2/lora_name, which only
        cover the shipped template's own encoder/LoRA slots. Applied before any
        other workflow shaping, so a later explicit clip_name1/clip_name2/
        lora_name still wins if both are given for the same field.
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
    instance_token
        This server's own attach token, forwarded to the ``localm_url``
        unload call so it authenticates on a keyless (open-mode) server too -
        see ``selfclient.self_request``'s docstring. Only used as a fallback
        when no owner API key is configured.
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

    # 0. Make sure ComfyUI is up (auto-launching when configured), before the
    # LLM unload below.
    ok, msg = ensure_comfy(api_url, on_progress=lambda t: _con.print(f"[dim]{t}[/dim]"),
                           launch_cmd=launch_cmd, workdir=workdir)
    if not ok:
        return False, msg

    # The LLM unload (the VRAM handoff) runs only after the workflow is built
    # and the model preflight passes - see step 9b.

    # 2. Load workflow template (personal flux_workflow.json if present,
    # else the committed example)
    try:
        workflow = json.loads(workflow_path().read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Failed to load FLUX workflow template: {e}"
    if model_overrides:
        apply_model_overrides(workflow, model_overrides)

    # 2a-9. Shape the FLUX workflow from the call's parameters (perf dequant,
    # encoder overrides, img2img, prompt / guidance / LoRA / negative injection,
    # seed). Seed is resolved (or randomised) up front so it lands in the sidecar.
    seed = seed if seed is not None else random.randint(1, 10 ** 12)
    ok, msg, uploaded_name = _build_image_workflow(
        workflow,
        prompt=prompt,
        api_url=api_url,
        guidance=guidance,
        negative_prompt=negative_prompt,
        cfg=cfg,
        seed=seed,
        clip_name1=clip_name1,
        clip_name2=clip_name2,
        lora_name=lora_name,
        lora_strength_model=lora_strength_model,
        lora_strength_clip=lora_strength_clip,
        input_image=input_image,
        denoise=denoise,
        fast_dequant=fast_dequant,
        con=_con,
    )
    if not ok:
        return False, msg

    # 9a. Pre-submit model validation against ComfyUI /object_info: confirm each
    # loader's model file exists (auto-substituting an unambiguous precision
    # variant) and fail with the exact missing filename, before the LLM unload
    # below. Best-effort: a no-op when /object_info is unreachable.
    pf_ok, pf_msg = preflight_models(
        workflow, api_url, on_progress=lambda t: _con.print(f"[dim]{t}[/dim]"))
    if not pf_ok:
        return False, pf_msg

    # 8c. Per-component GPU placement (opt-in, multi-GPU only): apply the plan
    # from comfy_client.resolve_media_placement() to the built graph by
    # injecting the core Select*Device nodes, after preflight and just before
    # submit. A component whose loader is absent from this graph is reported to
    # the caller rather than dropped.
    if placement:
        for _note in inject_device_placement(workflow, placement):
            if on_progress and "could not place" in _note:
                on_progress(_note)

    # 9b. Unload the chat LLM to free VRAM for FLUX, now that the workflow is valid.
    # Skipped when the caller decided the media model fits alongside the chat model
    # (swap=False), so the chat model stays hot.
    if swap:
        _localm_unload(localm_url, instance_token)

    # 9. Queue the prompt in ComfyUI. Mark 'now' in ComfyUI's own console log
    # FIRST (comfy_console_tail_start), so a partial-apply warning printed while
    # THIS prompt runs is attributable to this generation. None when localm did
    # not launch this ComfyUI itself, in which case
    # comfy_console_warnings_since() always reports [].
    console_tail_start = comfy_console_tail_start(api_url)
    kind, value = comfy_submit_prompt(api_url, workflow)
    if kind == SUBMIT_NO_ID:
        return False, (
            "ComfyUI accepted the request but returned no prompt_id.\n"
            "Check the ComfyUI console for workflow validation errors."
        )
    elif kind == SUBMIT_HTTP_ERROR:
        return False, (
            f"ComfyUI rejected the workflow (HTTP {value.code}):\n"
            f"{comfy_http_error_detail(value)}\n"
            "A model file missing from ComfyUI's models directory is the "
            "usual cause - check the names in your workflow template."
        )
    elif kind == SUBMIT_URL_ERROR:
        return False, (
            f"Could not connect to ComfyUI at {api_url}.\n"
            f"Error: {value}\n"
            "Make sure ComfyUI is running."
        )
    elif kind == SUBMIT_ERROR:
        return False, f"Error queuing prompt in ComfyUI: {value}"
    prompt_id = value

    # 10. Poll /history with a visible progress spinner (CLI console only) and a
    # heartbeat on the job stream throttled to once every 15s.
    start_time = time.time()
    filename = None
    subfolder = ""
    img_type = "output"
    last_said = [0.0]

    with Progress(
        SpinnerColumn(),
        TextColumn("[dim]{task.description}[/dim]"),
        TimeElapsedColumn(),
        transient=True,
        console=_con,
    ) as progress:
        task_id = progress.add_task("Generating image…", total=None)

        def _tick(elapsed: float) -> None:
            progress.update(task_id,
                            description=f"Generating image… ({int(elapsed)}s)")
            if on_progress and elapsed - last_said[0] >= 15:
                last_said[0] = elapsed
                try:
                    on_progress(f"Rendering… ({int(elapsed)}s elapsed)")
                except Exception:
                    pass

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
        # Surface the last poll error (if any) so an unreachable ComfyUI reads
        # differently from one that was simply still working when time ran out.
        err_note = (f"; last error contacting ComfyUI: {payload}"
                    if payload is not None else "")
        return False, (f"Image generation timed out after "
                       f"{max_poll_seconds // 60} minutes{err_note}.")

    # status == POLL_FINISHED means ComfyUI reported no execution_error; a node
    # whose weights only partly matched is only a console warning there, and the
    # run still succeeds. Check for a KNOWN warning of that shape printed while
    # THIS prompt ran. console_checked reflects whether a read actually happened
    # just now, and is what drives the sidecar's comfy_console_checked below.
    console_checked, comfy_console_warnings = comfy_console_warnings_since(
        api_url, console_tail_start)

    # status == POLL_FINISHED: find the first SaveImage output (presence-based,
    # matching the original loop - the first node carrying an "images" entry wins).
    for node_output in payload.get("outputs", {}).values():
        if "images" in node_output:
            img_info = node_output["images"][0]
            filename = img_info.get("filename")
            subfolder = img_info.get("subfolder", "")
            img_type = img_info.get("type", "output")
            break

    if not filename:
        return False, (
            "Generation finished but no output image was found in ComfyUI history.\n"
            "Check the ComfyUI console - a SaveImage node error is likely."
        )

    # 11. Fetch image from ComfyUI /view, save locally, strip metadata
    try:
        info = {"filename": filename, "subfolder": subfolder, "type": img_type}
        comfy_fetch_output(api_url, info, output_path, timeout=10)

        # Strip PNG metadata/EXIF for privacy; a strip that did not run returns
        # a warning rather than an empty string.
        strip_warning = _strip_png_metadata(output_path)

        # Output containment (opt-in): clear ComfyUI's history entry and delete its
        # own on-disk copy + any uploaded img2img source ONLY when delete_outputs
        # is set (user opted in, or privacy mode forces no-trace). Default keeps
        # ComfyUI's copies. Returns a warning when a copy could not be removed.
        contain_warning = contain_comfy_artifacts(
            api_url, prompt_id, info,
            comfy_output_dir=comfy_output_dir,
            uploaded_input=uploaded_name,
            delete_outputs=delete_outputs,
        )

        # Sidecar JSON: everything needed to reproduce or tweak this image,
        # saved as <output>.json next to it. Skipped when write_sidecar is
        # False; a write failure returns a warning.
        comfy_console_warning_text = ("; ".join(comfy_console_warnings)
                                      if comfy_console_warnings else None)

        sidecar_warning = ""
        if write_sidecar:
            sidecar_warning = _write_image_sidecar(
                output_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                cfg=cfg,
                seed=seed,
                guidance=guidance,
                lora_name=lora_name,
                lora_strength_model=lora_strength_model,
                lora_strength_clip=lora_strength_clip,
                input_image=input_image,
                denoise=denoise,
                clip_name1=clip_name1,
                clip_name2=clip_name2,
                start_time=start_time,
                comfy_console_warning=comfy_console_warning_text,
                comfy_console_checked=console_checked,
            )

        comfy_warning = (
            "WARNING: ComfyUI's own console reported: "
            + comfy_console_warning_text
            + ". The generation still completed, but the requested LoRA/model "
              "weights may not have fully applied - see comfy-launch.log."
        ) if comfy_console_warning_text else ""

        combined = "\n".join(w for w in (strip_warning, contain_warning, comfy_warning,
                                         sidecar_warning) if w)
        return True, _with_warning(
            f"Image saved to {output_path} (seed {seed} - reuse it to reproduce)",
            combined)

    except Exception as e:
        return False, f"Failed to download generated image from ComfyUI: {e}"
