# Native image generation

localm generates images without ComfyUI through its built-in native backend, which
runs [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (MIT) in a
separate worker process. It needs no Python ML stack and works on CPU and on Vulkan,
CUDA, ROCm and Metal GPUs.

## Which backend runs

Settings > Images > **Generation backend** (config `plugins.image.backend`):

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

   **Linux with an NVIDIA GPU runs on Vulkan.** The pinned upstream release
   publishes a CUDA build for Windows only; its Linux archives are
   CPU, Vulkan and ROCm. So on Linux `auto` picks Vulkan for NVIDIA, and
   `localm setup-sdcpp --backend cuda` there fails with "stable-diffusion.cpp publishes
   no 'cuda' build for this platform (available: cpu, vulkan, rocm)".

2. **Model.** With no model set, the Images page offers to download the recommended
   one (SD-Turbo, about 2 GB). From the command line:

   ```bash
   localm pull Green-Sky/SD-Turbo-GGUF:sd_turbo-f16-q8_0.gguf --type diffusion-unet
   ```

   Any model stable-diffusion.cpp supports works: SD 1.x/2.x and SDXL checkpoints,
   SD3, FLUX, Z-Image, Qwen Image and more, as GGUF, safetensors or ckpt. Pick it in
   the **Model** dropdown on the Images page (for the next generation) or set
   **Native model** in Settings (a registered model or a file path). A model that comes
   as separate parts (a FLUX, Chroma, SD3, SDXL, Qwen-Image or Z-Image diffusion model)
   is recognised from the file, and its text encoders and VAE are taken from your
   registered models: the CLIP-L, T5 or LLM encoder and the VAE that fit that family.
   Choose a component yourself (on the page or in Settings) to override the pick; a
   missing one is offered as a download before generating. A full checkpoint (SD 1.x,
   2.x, SDXL) loads on its own.

## Settings

| Setting | Meaning |
|---|---|
| Native model | A checkpoint or a diffusion model, recognised from the file. Blank: the recommended model once downloaded. |
| Native runtime | `auto`, `cpu`, `vulkan`, `cuda`, `rocm`, `metal`. |
| Embed generation parameters in the PNG | Off by default. See [Parameters in the PNG](#parameters-in-the-png). |
| Native sampling steps, CFG scale, sampler | Blank uses the model's recommended values (SD-Turbo: 2 steps, CFG 1), else the model family's (FLUX.1 dev: 20 steps, CFG 1; schnell: 4 steps), else stable-diffusion.cpp's defaults (20 steps, CFG 7). |
| Native VAE, CLIP-L, CLIP-G, T5-XXL, LLM | Model parts. Blank: picked from the registered models that fit the model's family. |

Paths and model names are owner-only settings: a key without admin rights can neither
set nor read them, and the Images page and job log show only a model's file name. Network
(UNC) and device paths are refused.

## What it does and does not do

- txt2img and img2img (the input image is resized to the requested size; without a
  size, its own size, scaled down to fit 2048 pixels per side with its aspect ratio
  kept, rounded down to a multiple of 8). Size: 64 to 2048 pixels per
  side, multiples of 8.
- Progress (loading, each sampling step, decoding) streams to the job log, the CLI
  and the MCP progress notifications. Stop cancels a running model load or
  generation; the first-use runtime download runs to completion.
- **Live preview.** The Images and Video pages show a small picture of the image taking
  shape, refreshed about eight times per run (every few sampling steps), under the job
  log. It is a quick projection of the model's latent, so it looks rough and washed out
  before the final decode. The preview lives in memory only: it is never saved, never
  written to the job history, and exists in privacy mode too.
- **VRAM shared with other programs.** On Windows the ROCm and Vulkan runtimes cannot see
  the VRAM other programs hold (the desktop, a browser, a game, a loaded chat model). Before
  each model load localm measures the VRAM free on the whole card, shows it in the job log
  ("Free VRAM: 13.9 of 16.0 GB.") and gives the runtime that budget, so a model that does
  not fit next to the other programs keeps part of its weights in RAM and streams them in,
  instead of Windows paging it to system memory, which made each step many times slower on
  ROCm and made Vulkan fail. Close other GPU programs for the fastest runs.
- **Flash attention on ROCm.** Images on the ROCm runtime use the runtime's flash attention,
  which is faster and needs less VRAM there; on Vulkan it is much slower, so it stays off.
- **Warnings that were recovered from.** A decode that does not fit in VRAM is retried in
  tiles (half the image size, then half again) and succeeds; the job log then shows one line
  ("Not enough VRAM to decode in one piece; decoding in tiles.") instead of the runtime's
  failure messages from before the retry. Warnings that are not followed by a recovery are shown unchanged, and every raw
  line is in the debug log.
- **Steps and clip skip per image.** The Images Advanced fold takes Steps and Clip skip
  for the next image; the placeholder shows the step count the picked model's family
  uses. Blank keeps the Settings value, then the model's recommended value, then the family default.
- **Negative prompt and CFG 1.** Models that default to CFG 1 (FLUX.1 dev and schnell, Z-Image,
  FLUX.2) skip the second, negative-prompt pass, so a negative prompt changes nothing there. The
  job log says so when you give one at CFG 1; set CFG to about 3 to apply it, at roughly twice
  the time per step.
- The chat model is unloaded first when the image model needs the VRAM (Media VRAM
  swap setting) and reloaded afterwards. The worker keeps its model loaded between
  generations and exits after 10 minutes without one, when VRAM is handed back, or
  when the server stops.
- The PNG carries no embedded metadata. A `.json` sidecar with the prompt and
  settings is written next to it except in privacy mode.
- LoRAs: the Images page takes one or more registered LoRAs with a strength each (a
  LoRA made for another family is applied with a note). ComfyUI workflow model picks
  are refused with a reason rather than ignored. Per-component GPU placement applies to ComfyUI only; with
  it turned on, a native job says so and loads each model on one GPU.

## Batch, sampler, hires fix and upscaling

These controls sit behind the Advanced fold of the Images page (the sampler and
scheduler also on the Video page); an empty field uses the model's own default.

- **Batch count** makes up to 16 images in one job. Image `i` uses seed `seed + i`
  and is saved with its own sidecar as soon as it is made, so a Stop part-way keeps
  the images already finished. `POST /api/imagine` takes `batch`.
- **Sampler and scheduler** are chosen by the names stable-diffusion.cpp reports
  (`euler`, `euler_a`, `dpm++2m`, ... and `discrete`, `karras`, `simple`, ...); the
  request fields are `sample_method` and `scheduler`. A name outside the list is a 400.
- **Hires fix** generates the image, scales it up, and samples it again. The request
  field `hires` takes `upscaler` (`latent`, `latent-nearest`, `latent-nearest-exact`,
  `latent-antialiased`, `latent-bicubic`, `latent-bicubic-antialiased`, `lanczos`,
  `nearest`, or `model` with `model` naming a registered upscaler), `scale` (1 to 4),
  `steps` (0 keeps the job's steps) and `denoise` (0.05 to 1, default 0.7). Each side of
  the final image is limited to 4096 pixels.
- **Upscale a gallery image** with a registered upscaler model: the detail view of an
  image has an Upscale action (`POST /api/imagine/file/{name}/upscale` with
  `{"upscaler": "<registered name>"}`), which saves a new gallery image at the model's
  scale (4x for RealESRGAN x4) and keeps the original's settings in its sidecar.
- **Upscaler models** are ESRGAN / RealESRGAN files registered with the type `upscaler`
  (`localm pull <repo>:<file> --type upscaler`, or `localm set-type <name> upscaler`).
  Files in a ComfyUI `upscale_models` folder and CivitAI upscalers register with that
  type.
- **Reuse settings** on a gallery image restores the model and its components, LoRAs,
  size, sampler, scheduler and hires settings. **Import parameters from image** does the
  same from a localm image or sidecar; a sampler or scheduler name from another tool is
  applied only when it is one of the names above, and is reported otherwise.

## Parameters in the PNG

By default a generated PNG carries no metadata, so a shared or moved image reveals nothing.
Settings > Images > **Embed generation parameters in the PNG** (`plugins.image.native.embed_params`,
off by default) writes the prompt, negative prompt, seed, steps, CFG, guidance, size, model,
model parts and LoRAs into each new native image, as an A1111-style `parameters` text chunk (other
tools can read it) and a `localm` chunk with the full record. The init-image path, timings and
runtime are never embedded. Privacy mode never embeds, whatever the setting says, and the
setting does not apply to images made through ComfyUI.

**Import parameters from image** (Images page, under the prompt) reads a PNG or a localm `.json`
sidecar and fills the prompt, negative prompt, seed, guidance, CFG, denoise, size and, on the
native backend, the model and LoRAs. It understands localm images and sidecars, A1111/Forge
`parameters` and ComfyUI `prompt` chunks. A model or LoRA is matched against your registered
ones by name or file name; anything that cannot be applied (an unregistered model, or steps and
sampler, which are Settings values) is named in the message instead of being dropped silently.
The file is read in memory and not stored.

## Enhance prompt

The **Enhance prompt** button under the prompt on the Images, Video and Music pages asks the
loaded chat model to expand a short idea. The instruction depends on the model that will
generate: tag lists for Stable Diffusion 1.x, 2.x and SDXL, descriptive sentences for the other
image families, motion and camera wording for video, and style tags for music. The result
replaces the text in the field, still editable, and **Undo** restores your original. With no chat
model loaded the button says so and changes nothing; it never loads a model itself.

## Tested hardware

Tested on Windows with an AMD Radeon RX 6900 XT (Vulkan and ROCm) and on CPU. The
NVIDIA CUDA, macOS Metal and Linux builds come from the same upstream release but
have not been tested on real hardware by the localm project.
