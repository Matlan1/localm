# FLUX Image Generation with localm

localm generates images through a local ComfyUI instance running a FLUX
GGUF workflow. Everything runs on your own GPU; nothing leaves the machine.
localm drives ComfyUI over HTTP (VRAM handoff, model loading, and output
retrieval are automatic); you do not need to learn the ComfyUI UI, just ask
for images through localm's chat, CLI, or MCP interface. This guide covers the
setup for a 16 GB VRAM card and how localm drives it.

## ComfyUI setup

### 1. Install ComfyUI

Clone the ComfyUI repository and install it:

```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
pip install -r requirements.txt
```

Or use a launcher like Stability Matrix (https://lykos.ai/) which automates
this and manages plugins.

### 2. Install the ComfyUI-GGUF extension

Add support for GGUF quantized models (FLUX with lower VRAM overhead) by
installing the ComfyUI-GGUF extension.

```bash
git clone https://github.com/city96/ComfyUI-GGUF ComfyUI/custom_nodes/ComfyUI-GGUF
```

### 3. Download model files

These are the exact files the committed example workflow loads, so a fresh
download matches it on the first run (roughly 20 to 25 GB in total):

| Component | File | ComfyUI folder |
| :--- | :--- | :--- |
| UNET | `flux1-dev-Q8_0.gguf` (or Q6_K) | `models/unet/` |
| Text encoders | `clip_l.safetensors` + `t5xxl_fp8_e4m3fn.safetensors` | `models/clip/` |
| VAE | `ae.safetensors` | `models/vae/` |

### 4. Start ComfyUI

Launch ComfyUI on its default port 8188:

#### Windows (batch launcher)

If you installed via Stability Matrix or a batch launcher, run:

```bash
launch-comfyui.bat
```

Or from the ComfyUI directory:

```bash
python main.py
```

It prints `Starting server at http://127.0.0.1:8188` - localm reads this URL
automatically.

#### Linux / macOS

From the ComfyUI directory:

```bash
python main.py
```

### 5. Configure localm (optional: for auto-launch)

If you want localm to automatically start ComfyUI when you request an image,
configure the launch command:

```bash
localm config comfy_launch_cmd "D:\path\to\launch-comfyui.bat"
```

(On Windows, use your launcher's full path; on Linux/macOS, use the path to
your `main.py`.)

If your launcher expects to run from its own folder (common for batch scripts):

```bash
localm config comfy_workdir "D:\path\to\ComfyUI"
```

localm respects the `FLUX_API_URL` environment variable if you need to override
the default `http://127.0.0.1:8188`. Otherwise it auto-detects ComfyUI's URL.

To have localm clean up ComfyUI's duplicate output copy after each generation,
set the `comfy_output_dir` config key (or `COMFY_OUTPUT_DIR` env var) to
ComfyUI's own output folder.

## Performance notes

Quantized FLUX dev (Q8_0) generates in roughly 30 to 60 seconds on a 16 GB
NVIDIA card with CUDA. Performance varies by hardware:

- **NVIDIA (CUDA):** 30-60 seconds (Q8_0)
- **AMD (ROCm):** 60-120 seconds (cold start compiles GPU kernels; subsequent runs faster)
- **Vulkan (universal):** varies by GPU model

These numbers are typical, not guaranteed. For AMD hardware, see the ROCm/HIP
setup notes in [gpu-setup.md](gpu-setup.md).

## How localm drives it

Each frontend is provided by a plugin and shares the same pipeline; a frontend
appears only when its plugin is installed and enabled:

- The `image` plugin: its Images page in the GUI (`localm gui`) and the
  image-generation chat slash command
- The `coder` plugin's `generate_image` tool, invoked when you ask the agent
  for an image
- The `mcp` plugin's `generate_image` tool (`localm mcp`)

Features handled for you:

- **VRAM handover**: before generation, localm unloads the LLM so FLUX gets
  the full VRAM budget, then reloads it after ComfyUI releases its models.
- **Fail-fast probe**: ComfyUI is probed before the LLM is unloaded, so an
  unreachable server costs nothing. With `comfy_launch_cmd` set, localm
  starts ComfyUI itself and waits for it to come up.
- **Reproducibility**: the seed is applied to every sampler node and
  reported back; a JSON sidecar with prompt, seed, guidance, and encoder
  settings is written next to every output image.
- **img2img**: pass an input image and a denoise strength; output
  dimensions match the input.
- Optional negative prompts, LoRA injection, and encoder overrides are
  supported as tool parameters.

## Safety filtering

The base FLUX weights ship without a safety filter and localm does not add
one. If you need filtering, implement it at the ComfyUI pipeline level.

## Custom workflows stay private

The committed template, `localm/image_gen/flux_workflow.example.json`, uses
the vanilla public FLUX stack (`flux1-dev-Q8_0.gguf`, `clip_l`,
`t5xxl_fp8_e4m3fn`, `ae.safetensors`). To use your own models, encoders, or
node graph, export your workflow from ComfyUI (Save -> API format) as
`localm/image_gen/flux_workflow.json` - it takes precedence automatically
and is **gitignored**, so which models you actually run never leaves your
machine.

Suggested host stack: Stability Matrix managing ComfyUI - on RDNA2 (RX 6xxx)
combine it with the ROCm/HIP fixes described in the GPU setup section of the
README.
