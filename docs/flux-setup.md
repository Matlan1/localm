# FLUX Image Generation with localm

localm generates images through a local ComfyUI instance running a FLUX
GGUF workflow. Everything runs on your own GPU; nothing leaves the machine.
This guide covers the setup for a 16 GB VRAM card and how localm drives it.

## ComfyUI setup

1. Install ComfyUI (directly or via Stability Matrix) and add the
   [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) extension.
2. Download the model files:

These are the exact files the committed example workflow loads, so a fresh
download matches it on the first run (roughly 20 to 25 GB in total):

| Component | File | ComfyUI folder |
| :--- | :--- | :--- |
| UNET | `flux1-dev-Q8_0.gguf` (or Q6_K) | `models/unet/` |
| Text encoders | `clip_l.safetensors` + `t5xxl_fp8_e4m3fn.safetensors` | `models/clip/` |
| VAE | `ae.safetensors` | `models/vae/` |

3. Start ComfyUI on its default port 8188. localm reads `FLUX_API_URL` if
   yours runs elsewhere. Optionally tell localm how to start it so image
   requests can launch it on demand:

```bash
localm config comfy_launch_cmd "D:\path\to\comfyui.bat"
```

Quantized FLUX dev (Q8_0) generates in roughly 30 to 60 seconds on a 16 GB
card. On AMD, a native Windows ROCm environment works well; the decisions
behind that setup are recorded in
[archive/rocm-migration.md](archive/rocm-migration.md).

## How localm drives it

Four frontends share the same pipeline (`localm/image_gen/comfy.py`):

- `/imagine <prompt>` in chat (GUI and `localm run`)
- The GUI's Images page (`localm gui`)
- The coder agent's `generate_image` tool, invoked when you ask the agent
  for an image
- The MCP server's `generate_image` tool (`localm mcp`)

Features handled for you:

- **VRAM handover**: before generation, localm unloads the LLM
  (`POST /v1/models/unload`, waiting for any in-flight reply to finish) so
  FLUX gets the full VRAM budget. After generation, ComfyUI is asked to
  release its models (`POST /free`) and the LLM reloads immediately; on
  older ComfyUI builds without `/free`, the reload stays lazy and happens
  on the next chat request instead.
- **Fail-fast probe**: ComfyUI is probed before the LLM is unloaded, so an
  unreachable server costs nothing. With `comfy_launch_cmd` set, localm
  starts ComfyUI itself and waits for it to come up.
- **Reproducibility**: the seed is applied to every sampler node and
  reported back; a JSON sidecar with prompt, seed, guidance, and encoder
  settings is written next to every output image.
- **img2img**: pass an input image and a denoise strength; output
  dimensions match the input.
- Optional negative prompts (a real negative branch via classifier-free
  guidance / `CFGGuider`, not conditioning concat), LoRA injection, and
  encoder overrides are supported as tool parameters.

## Safety filtering

The base FLUX weights ship without a safety filter and localm does not add
one. If you need filtering, implement it at the ComfyUI pipeline level.

## Custom workflows stay private

The committed template, `localm/image_gen/flux_workflow.example.json`, uses
the vanilla public FLUX stack (`flux1-dev-Q8_0.gguf`, `clip_l`,
`t5xxl_fp8_e4m3fn`, `ae.safetensors`). To use your own models, encoders, or
node graph, export your workflow from ComfyUI (Save → API format) as
`localm/image_gen/flux_workflow.json` - it takes precedence automatically
and is **gitignored**, so which models you actually run never leaves your
machine. The same applies to ComfyUI's own output folder: set the
`comfy_output_dir` config key (or `COMFY_OUTPUT_DIR` env var) if you want
localm to clean up ComfyUI's duplicate copy after each generation.

Suggested host stack: [StabilityMatrix](https://lykos.ai/) managing ComfyUI -
on RDNA2 (RX 6xxx) combine it with the ROCm/HIP fixes described in the GPU
setup section of the README.
