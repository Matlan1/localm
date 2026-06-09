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
#  Helpers
# ---------------------------------------------------------------------------

def _image_dimensions(path: Path) -> tuple[int, int]:
    """Return (width, height) from a PNG or JPEG without any external libs."""
    try:
        data = path.read_bytes(32)
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
        if data[:2] == b"\xff\xd8":
            # JPEG — scan for SOF0/SOF2 markers
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
#  Image generation
# ---------------------------------------------------------------------------

def generate_image(
    prompt: str,
    output_path: Path,
    *,
    api_url: str = "http://127.0.0.1:8188",
    guidance: Optional[float] = None,
    lora_name: Optional[str] = None,
    lora_strength_model: float = 1.0,
    lora_strength_clip: float = 0.5,
    input_image: Optional[Path] = None,
    denoise: Optional[float] = None,
    localm_url: Optional[str] = None,
    max_poll_seconds: int = 600,
) -> tuple[bool, str]:
    """
    Generate an image from *prompt* and save it to *output_path*.

    Parameters
    ----------
    prompt
        Descriptive text prompt.  For img2img, describe what to *change*
        rather than the full scene — the base image already provides structure.
    output_path
        Destination file (PNG).  Parent directories are created if needed.
    api_url
        ComfyUI base URL.  Defaults to ``http://127.0.0.1:8188``.
        Override with the ``FLUX_API_URL`` environment variable before calling.
    guidance
        FluxGuidance scale.  None keeps the workflow's own default (~3.5).
    lora_name
        LoRA filename to inject (optional).
    lora_strength_model
        How strongly the LoRA patches the UNet weights (default 1.0).
        This is the main lever for unlock/style LoRAs.
    lora_strength_clip
        How strongly the LoRA patches the text encoder (default 0.5).
        Lower than model strength is usually correct for unlock LoRAs —
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

    # 3. img2img: upload input image, add LoadImage + VAEEncode, redirect latent
    if input_image is not None:
        if not input_image.is_file():
            return False, f"Input image not found: {input_image}"
        try:
            uploaded_name = _upload_image(input_image, api_url)
        except Exception as e:
            return False, f"Failed to upload input image to ComfyUI: {e}"

        w, h = _image_dimensions(input_image)

        # LoadImage node — ComfyUI loads from its own input/ dir by filename
        workflow["40"] = {
            "inputs": {"image": uploaded_name, "upload": "image"},
            "class_type": "LoadImage",
        }
        # VAEEncode — encode the loaded image into latent space
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

    # 4. Inject prompt — node "6" first (default template), then scan
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

    # 5. Inject guidance
    if guidance is not None:
        if "26" in workflow and workflow["26"].get("class_type") == "FluxGuidance":
            workflow["26"]["inputs"]["guidance"] = guidance
        else:
            for node in workflow.values():
                if node.get("class_type") == "FluxGuidance":
                    node["inputs"]["guidance"] = guidance
                    break

    # 6. Inject LoRA
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

    # 7. Randomise seed
    seed = random.randint(1, 10 ** 12)
    for node in workflow.values():
        cls = node.get("class_type", "")
        if cls in ("KSampler", "KSamplerAdvanced"):
            node["inputs"]["seed"] = seed
            break
        if cls == "RandomNoise":
            node["inputs"]["noise_seed"] = seed
            break

    # 8. Queue the prompt in ComfyUI
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

    # 9. Poll /history with a visible progress spinner
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

    # 10. Fetch image from ComfyUI /view, save locally, strip metadata
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
