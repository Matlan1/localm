"""
ComfyUI FLUX image generation.

Standalone module — usable from the localcoder agent tool, the CLI,
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


_WORKFLOW_PATH = Path(__file__).parent / "flux_workflow.json"


# ---------------------------------------------------------------------------
#  VRAM management
# ---------------------------------------------------------------------------

def _localm_unload(localm_url: Optional[str] = None) -> None:
    """
    Ask a localm server to release its model from GPU memory.

    Reads LOCALM_URL from the environment if *localm_url* is not given.
    Silent no-op when the variable is unset or the request fails — never
    blocks image generation if localm is not in the picture.
    """
    url = (localm_url or os.environ.get("LOCALM_URL", "")).rstrip("/")
    if not url:
        return
    try:
        req = urllib.request.Request(f"{url}/models/unload", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
#  Image generation
# ---------------------------------------------------------------------------

def generate_image(
    prompt: str,
    output_path: Path,
    *,
    api_url: str = "http://127.0.0.1:8188",
    guidance: Optional[float] = None,
    lora_name: Optional[str] = None,
    lora_strength: float = 1.0,
    localm_url: Optional[str] = None,
    max_poll_seconds: int = 600,
) -> tuple[bool, str]:
    """
    Generate an image from *prompt* and save it to *output_path*.

    Parameters
    ----------
    prompt
        Descriptive text prompt.
    output_path
        Destination file (PNG).  Parent directories are created if needed.
    api_url
        ComfyUI base URL.  Defaults to ``http://127.0.0.1:8188``.
        Override with the ``FLUX_API_URL`` environment variable before calling.
    guidance
        FluxGuidance scale.  None keeps the workflow's own default (~3.5).
    lora_name
        LoRA filename to inject (optional).
    lora_strength
        Strength applied to both model and clip (default 1.0).
    localm_url
        localm server URL (e.g. ``http://127.0.0.1:8080/v1``) to unload
        before generation so FLUX gets the full VRAM budget.
        Reads ``LOCALM_URL`` env var if None.  Skipped silently when unset.
    max_poll_seconds
        Timeout waiting for ComfyUI to finish (default 10 minutes).

    Returns
    -------
    (ok, message)
        ``ok=True`` and a success description, or ``ok=False`` and an error.
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    _con = Console()

    # 1. Unload LLM to free VRAM before FLUX loads
    _localm_unload(localm_url)

    # 2. Load workflow template
    try:
        workflow = json.loads(_WORKFLOW_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Failed to load FLUX workflow template: {e}"

    # 3. Inject prompt — node "6" first (default template), then scan
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

    # 4. Inject guidance
    if guidance is not None:
        if "26" in workflow and workflow["26"].get("class_type") == "FluxGuidance":
            workflow["26"]["inputs"]["guidance"] = guidance
        else:
            for node in workflow.values():
                if node.get("class_type") == "FluxGuidance":
                    node["inputs"]["guidance"] = guidance
                    break

    # 5. Inject LoRA
    if lora_name:
        workflow["100"] = {
            "inputs": {
                "model": ["30", 0],
                "clip": ["31", 0],
                "lora_name": lora_name,
                "strength_model": lora_strength,
                "strength_clip": lora_strength,
            },
            "class_type": "LoraLoader",
        }
        if "28" in workflow:
            workflow["28"]["inputs"]["model"] = ["100", 0]
        if "6" in workflow:
            workflow["6"]["inputs"]["clip"] = ["100", 1]

    # 6. Randomise seed
    seed = random.randint(1, 10 ** 12)
    for node in workflow.values():
        cls = node.get("class_type", "")
        if cls in ("KSampler", "KSamplerAdvanced"):
            node["inputs"]["seed"] = seed
            break
        if cls == "RandomNoise":
            node["inputs"]["noise_seed"] = seed
            break

    # 7. Queue the prompt in ComfyUI
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

    except urllib.error.URLError as e:
        return False, (
            f"Could not connect to ComfyUI at {api_url}.\n"
            f"Error: {e}\n"
            "Make sure ComfyUI is running."
        )
    except Exception as e:
        return False, f"Error queuing prompt in ComfyUI: {e}"

    # 8. Poll /history with a visible progress spinner
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
            "Check the ComfyUI console — a SaveImage node error is likely."
        )

    # 9. Fetch image from ComfyUI /view, save locally, strip metadata
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

        # Delete original from ComfyUI output directory
        comfy_out = Path(os.environ.get(
            "COMFY_OUTPUT_DIR", "D:/stablematrix/Data/Images/Text2Img"
        ))
        try:
            orig = comfy_out / subfolder / filename
            if orig.exists():
                orig.unlink()
        except Exception:
            pass

        return True, f"Image saved to {output_path}"

    except Exception as e:
        return False, f"Failed to download generated image from ComfyUI: {e}"
