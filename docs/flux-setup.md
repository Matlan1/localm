# FLUX Image Generation with localm

localm generates images through a local ComfyUI instance running a FLUX
GGUF workflow. Everything runs on your own GPU; nothing leaves the machine.
localm drives ComfyUI over HTTP (VRAM handoff, model loading, and output
retrieval are automatic); you do not need to learn the ComfyUI UI, just ask
for images through localm's chat, CLI, or MCP interface. This guide covers the
setup for a 16 GB VRAM card and how localm drives it.

> This guide is for pointing localm at **your own** ComfyUI. localm can also
> install and run its **own** managed ComfyUI for you (opt-in), which sets the
> models up automatically and pins a known-good version; see
> [managed-comfyui.md](managed-comfyui.md). The model and workflow notes below
> apply to both.

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
download matches it on the first run (roughly 18 GB in total):

| Component | File | ComfyUI folder |
| :--- | :--- | :--- |
| UNET | `flux1-dev-Q8_0.gguf` (or Q6_K) | `models/unet/` |
| Text encoders | `clip_l.safetensors` + `t5xxl_fp8_e4m3fn.safetensors` | `models/clip/` |
| VAE | `ae.safetensors` | `models/vae/` |

You do not have to fetch these by hand: if any of them are missing when you
generate an image, localm's GUI offers to download it for you (showing the
exact source and size first) and places it in the right folder automatically.

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

It prints `Starting server at http://127.0.0.1:8188`. localm does not read this
printed URL; make sure it matches what localm expects (`FLUX_API_URL`, or the
`comfy_api_url` config key, below).

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

localm resolves ComfyUI's URL in this order: the `FLUX_API_URL` environment
variable, then a localm-managed ComfyUI instance if one is active, then the
`comfy_api_url` config key, then the default `http://127.0.0.1:8188`. If
ComfyUI is not on the default port, set one of these to match.

To have localm clean up ComfyUI's duplicate output copy after each
generation, set `comfy_delete_outputs` to `true`. If ComfyUI's output folder
cannot be derived automatically from your launch command, also set the
`comfy_output_dir` config key (or `COMFY_OUTPUT_DIR` env var) to point at it.

## Performance notes

Quantized FLUX dev (Q8_0) generates in roughly 30 to 60 seconds on a 16 GB
NVIDIA card with CUDA. Performance varies by hardware:

- **NVIDIA (CUDA):** 30-60 seconds (Q8_0)
- **AMD (ROCm):** 60-120 seconds (cold start compiles GPU kernels; subsequent runs faster)
- **Vulkan (universal):** varies by GPU model

These numbers are typical, not guaranteed. For AMD hardware, see the ROCm/HIP
setup notes in [gpu-setup.md](gpu-setup.md).

## How localm drives it

All frontends share the same pipeline:

- `localm image "<prompt>"` - a core CLI command, always available (see
  `localm image --help` for img2img, negative prompt, LoRA, and seed flags).
- The `image` plugin: its Images page in the GUI (`localm gui`) and the
  image-generation chat slash command - appears once the plugin is installed
  and enabled.
- The `coder` plugin's `generate_image` tool, invoked when you ask the agent
  for an image (plugin-gated).
- The `mcp` plugin's `generate_image` tool (`localm mcp`) (plugin-gated).

Features handled for you:

- **VRAM handover**: before generation, localm unloads the LLM so FLUX gets
  the full VRAM budget, then reloads it after ComfyUI releases its models.
- **Fail-fast probe**: ComfyUI is probed before the LLM is unloaded, so an
  unreachable server costs nothing. With `comfy_launch_cmd` set, localm
  starts ComfyUI itself and waits for it to come up.
- **Reproducibility**: the seed is applied to every sampler node and
  reported back; a JSON sidecar with prompt, seed, guidance, and encoder
  settings is written next to each output image (skipped in privacy mode,
  so the prompt never touches disk).
- **img2img**: pass an input image and a denoise strength; output
  dimensions match the input.
- Optional negative prompts, LoRA injection, and encoder overrides are
  supported as tool parameters.
- **Guidance controls**: `--guidance` is FLUX's own distilled guidance
  embedding (applies whether or not you set a negative prompt); `--cfg` is the
  classifier-free guidance scale, which only takes effect together with
  `--negative` (the GUI's Images page has both, under Advanced). CLI:
  `localm image "..." --negative "blurry" --cfg 3.5`.
- **Abort on Ctrl-C**: interrupting a CLI generation tells ComfyUI to stop the
  render and free its VRAM instead of leaving it running.

## Safety filtering

The base FLUX weights ship without a safety filter and localm does not add
one. If you need filtering, implement it at the ComfyUI pipeline level.

## Custom workflows stay private

The committed template, `localm/image_gen/flux_workflow.example.json`, uses
the vanilla public FLUX stack (`flux1-dev-Q8_0.gguf`, `clip_l`,
`t5xxl_fp8_e4m3fn`, `ae.safetensors`). To use your own models, encoders, or
node graph, export your workflow from ComfyUI (Save -> API format) and
upload it from the GUI's Images page (its Workflows panel - "Upload + use"),
or from the CLI:

```bash
localm comfy workflow add image my_flux.json --use   # upload and select it
localm comfy workflow list image                     # see what is uploaded and active
localm comfy workflow use image --clear              # back to the built-in default
```

Either way the file is stored under the localm data folder, never in the repo,
so which models you actually run never leaves your machine. This selected
workflow takes precedence over everything below it, and governs `localm image`
too, not just the GUI.

The older method still works: drop the export at
`localm/image_gen/flux_workflow.json` (it is **gitignored**). localm migrates
it into the same private, update-surviving location the first time it starts
and selects it automatically, as long as nothing else is already selected.

## Picking models per slot

The Images page shows a **Models** panel for the active workflow: one dropdown
per model file the workflow uses, labeled by the role it fills alongside the
raw ComfyUI field name (e.g. "Diffusion model (UNet) (unet_name)"). A slot
whose file is not installed in ComfyUI is called out, and if you already have
a model of that kind registered in localm but ComfyUI is not offering it
(wrong folder, for example), the panel says so and where to move it. With
ComfyUI unreachable the panel falls back to listing what the workflow needs
and which of your registered models could fill each slot, instead of going
blank.

Suggested host stack: Stability Matrix managing ComfyUI - on RDNA2 (RX 6xxx)
combine it with the ROCm/HIP setup notes in [gpu-setup.md](gpu-setup.md).
