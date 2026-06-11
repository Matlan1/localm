# FLUX Image Generation with localm

localm generates images through a local ComfyUI instance running a FLUX
GGUF workflow. Everything runs on your own GPU; nothing leaves the machine.
This guide covers the setup for a 16 GB VRAM card and how localm drives it.

## ComfyUI setup

1. Install ComfyUI (directly or via Stability Matrix) and add the
   [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) extension.
2. Download the model files:

| Component | File | ComfyUI folder |
| :--- | :--- | :--- |
| UNET | `flux1-dev-Q8_0.gguf` (or Q6_K) | `models/unet/` |
| Text encoder | `t5-v1_1-xxl-encoder-Q8_0.gguf` + CLIP-L | `models/clip/` |
| VAE | `ae.safetensors` | `models/vae/` |

3. Start ComfyUI on its default port 8188. localm reads `FLUX_API_URL` if
   yours runs elsewhere.

Quantized FLUX dev (Q8_0) generates in roughly 30 to 60 seconds on a 16 GB
card. On AMD, a native Windows ROCm environment works well; the decisions
behind that setup are recorded in
[archive/rocm-migration.md](archive/rocm-migration.md).

## How localm drives it

Three frontends share the same pipeline (`localm/image_gen/comfy.py`):

- The coder agent's `generate_image` tool, invoked when you ask the agent
  for an image
- The GUI's Images page (`localm gui`)
- The MCP server's `generate_image` tool (`localm mcp`)

Features handled for you:

- **VRAM handover**: before generation, localm unloads the LLM
  (`POST /v1/models/unload`) so FLUX gets the full VRAM budget; the LLM
  reloads automatically on the next chat request (a few seconds overhead).
- **Fail-fast probe**: ComfyUI is probed before the LLM is unloaded, so an
  unreachable server costs nothing.
- **Reproducibility**: the seed is applied to every sampler node and
  reported back; a JSON sidecar with prompt, seed, guidance, and encoder
  settings is written next to every output image.
- **img2img**: pass an input image and a denoise strength; output
  dimensions match the input.
- Optional negative prompts (via conditioning concat), LoRA injection, and
  encoder overrides are supported as tool parameters.

## Safety filtering

The base FLUX weights ship without a safety filter and localm does not add
one. If you need filtering, implement it at the ComfyUI pipeline level.
