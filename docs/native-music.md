# Native music generation

localm generates music without ComfyUI through its built-in native backend, which runs
[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) on
[KoboldCpp](https://github.com/LostRuins/koboldcpp) (AGPL-3.0), whose music engine is
built on [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) (MIT). It runs in a
separate process, needs no Python ML stack, and works on CPU and on CUDA, Vulkan and
Metal GPUs.

## Which backend runs

Settings > Music > **Music backend** (config `plugins.music.backend`):

| Value | Behaviour |
|---|---|
| `auto` (default) | ComfyUI when it is set up (localm's managed ComfyUI is installed, `comfy_target` is `user`, a ComfyUI address, launcher or folder is configured, or a ComfyUI answers at the configured address), otherwise native. The job log says which one ran and why. |
| `native` | Always the native backend. |
| `comfy` | Always ComfyUI. |

An explicit choice is never swapped for the other backend: if it cannot run, the
generation fails and says why. The choice applies everywhere music is generated: the
Music page, the chat `/music` and `/generate-music` commands, `POST /api/music` and
`localm music`.

## First use

```bash
localm setup-music                  # runtime + default models + a 5 second test track
localm setup-music --backend vulkan # or cpu, cuda, metal
localm setup-music --status
```

1. **Runtime.** A pinned KoboldCpp release goes into `<data dir>/runtimes/koboldcpp/`:
   about 115 MB for the Vulkan/CPU build, about 610 MB for the CUDA build, about 65 MB
   on Apple Silicon. The download goes through the network policy like a model pull, and
   its size and sha256 are checked before it is ever run. `auto` picks CUDA for NVIDIA,
   Metal on Apple Silicon, CPU when no GPU is found, and Vulkan otherwise (AMD and Intel
   GPUs included); a backend that does not start falls back to Vulkan, then CPU, and
   says so. A generation installs the runtime itself when it is missing.

2. **Models.** The default set (about 4.1 GB) comes from
   `Serveurperso/ACE-Step-1.5-GGUF`: the planner (`acestep-5Hz-lm-0.6B`), the text encoder
   (`Qwen3-Embedding-0.6B`), the turbo diffusion model and the VAE. `localm setup-music`
   downloads it; on the Music page, Generate offers each missing file. A generation never
   downloads models by itself.

## Settings

| Setting | Meaning |
|---|---|
| Native runtime | `auto`, `cuda`, `vulkan`, `cpu`, `metal`. |
| Native diffusion model, text encoder, VAE, planner | Registered model name or file path; each must be that ACE-Step component. Blank: the default. |

In `config.json`, `plugins.music.native.plan` (default `true`) turns the planner off and
`plugins.music.native.lowvram` (default `false`) keeps the models out of VRAM between the
stages of a generation.

Paths and model names are owner-only settings. Network (UNC) and device paths are refused.

## What it does and does not do

- Style tags, optional lyrics, duration, seed, steps, CFG and shift. ComfyUI-only inputs
  (workflow model picks, sampler, scheduler, lyrics strength, per-component GPU placement)
  are refused with a reason rather than ignored.
- The planner picks tempo, key and structure first and ends the song itself, so a track
  usually comes out up to a few seconds shorter than requested; the job says by how much.
  Without the planner the length is exact, but the music often ends early and the rest is
  near-silence.
- Progress (loading, planning, generating, with a heartbeat while a stage runs) streams to
  the job log and the CLI. Stop ends a generation by stopping the runtime.
- The chat model is unloaded first when the music models need the VRAM (Media VRAM swap
  setting) and reloaded afterwards. The runtime keeps its models loaded between
  generations and exits after 10 minutes without one, when VRAM is handed back, or with
  localm. It listens only on 127.0.0.1, with a fresh key each start.
- The track is a 48 kHz stereo WAV with no embedded metadata. A `.json` sidecar with the
  tags, lyrics and settings is written next to it except in privacy mode.

## Tested hardware

Tested on Windows with an AMD Radeon RX 6900 XT (Vulkan) and on CPU, and the Linux runtime
install under WSL. The NVIDIA CUDA and macOS builds come from the same release but have not
been tested on real hardware by the localm project.
