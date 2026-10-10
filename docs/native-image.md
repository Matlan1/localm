# Native image generation

localm generates images without ComfyUI through its built-in native backend, which
runs [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (MIT) in a
separate worker process. It needs no Python ML stack and works on CPU and on Vulkan,
CUDA, ROCm and Metal GPUs.

## Which backend runs

Settings > Images > **Image backend** (config `plugins.image.backend`):

| Value | Behaviour |
|---|---|
| `auto` (default) | ComfyUI when it is set up (localm's managed ComfyUI is installed, `comfy_target` is `user`, a ComfyUI address, launcher or folder is configured, or a ComfyUI answers at the configured address), otherwise native. The job log says which one ran and why. |
| `native` | Always the native backend. |
| `comfy` | Always ComfyUI. |

An explicit choice is never swapped for the other backend: if it cannot run, the
generation fails and says why. The choice applies everywhere an image is generated:
the Images page, `POST /api/imagine`, `POST /v1/images/generations`, `localm image`,
the chat `/generate-image` command, the coder agent's image tool and the MCP
`generate_image` tool.

## First use

1. **Runtime.** The first native generation installs the runtime into
   `<data dir>/runtimes/sdcpp/` (about 30 MB for Vulkan or CPU, about 250 MB for ROCm,
   about 900 MB for CUDA including its CUDA runtime). The download goes through the
   network policy like a model pull and is checked against a pinned sha256. To install
   it ahead of time, or offline-prepare a machine:

   ```bash
   localm setup-sdcpp                  # auto: the best build for this machine
   localm setup-sdcpp --backend vulkan # or cpu, cuda, rocm, metal
   localm setup-sdcpp --status
   ```

   `auto` picks Metal on Apple Silicon, CUDA for NVIDIA on Windows, ROCm for AMD when
   hipBLAS is available (the ROCm PyTorch wheels in localm's venv, or a system ROCm),
   CPU when no GPU is found, and Vulkan otherwise. A build that does not load on this
   machine falls back to Vulkan, then CPU, and says so.

2. **Model.** With no model set, the Images page offers to download the recommended
   one (SD-Turbo, about 2 GB). From the command line:

   ```bash
   localm pull Green-Sky/SD-Turbo-GGUF:sd_turbo-f16-q8_0.gguf --type diffusion-unet
   ```

   Any model stable-diffusion.cpp supports works: SD 1.x/2.x and SDXL checkpoints,
   SD3, FLUX, Z-Image, Qwen Image and more, as GGUF, safetensors or ckpt. Set
   **Native image model** to a registered model name or a file path. Models that come
   as separate parts (FLUX, SD3, Z-Image) also need their text encoders and VAE in the
   matching fields (CLIP-L, CLIP-G, T5-XXL, LLM, VAE).

## Settings

| Setting | Meaning |
|---|---|
| Native image model | Checkpoint, or diffusion model when encoders are set. Blank: the recommended model once downloaded. |
| Native runtime | `auto`, `cpu`, `vulkan`, `cuda`, `rocm`, `metal`. |
| Native sampling steps, CFG scale, sampler | Blank uses the model's recommended values (SD-Turbo: 2 steps, CFG 1), else stable-diffusion.cpp's defaults (20 steps, CFG 7). |
| Native VAE, CLIP-L, CLIP-G, T5-XXL, LLM | Optional model parts. |

Paths and model names are owner-only settings.

## What it does and does not do

- txt2img and img2img (the input image is resized to the requested size; without a
  size, its own size rounded down to a multiple of 8). Size: 64 to 2048 pixels per
  side, multiples of 8.
- Progress (loading, each sampling step, decoding) streams to the job log, the CLI
  and the MCP progress notifications. Stop cancels the running generation.
- The chat model is unloaded first when the image model needs the VRAM (Media VRAM
  swap setting) and reloaded afterwards. The worker keeps its model loaded between
  generations and exits after 10 minutes without one, when VRAM is handed back, or
  when the server stops.
- The PNG carries no embedded metadata. A `.json` sidecar with the prompt and
  settings is written next to it except in privacy mode.
- ComfyUI-only features are refused with a reason rather than ignored: workflow model
  picks, ComfyUI LoRA files and per-component GPU placement.

## Tested hardware

Tested on Windows with an AMD Radeon RX 6900 XT (Vulkan and ROCm) and on CPU. The
NVIDIA CUDA, macOS Metal and Linux builds come from the same upstream release but
have not been tested on real hardware by the localm project.
