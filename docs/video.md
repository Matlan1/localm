# Video generation (ComfyUI Wan 2.2)

localm generates short video clips through the same local ComfyUI server it
uses for images and music. The committed workflow template runs the public
**Wan 2.2 TI2V 5B** stack - text-to-video by default, image-to-video when you
provide a start picture. Output is MP4 (h264).

Nothing leaves your machine: ComfyUI runs locally, and the one-time model
download is the only network access.

## What to expect

Video is the **slowest and most VRAM-hungry generator** in localm. Unlike
ACE-Step music (arbitrary length), a video model attends over all frames at
once:

- **~5 seconds is the native clip length** (121 frames at 24 fps). Quality is
  best there.
- Longer clips (the API accepts up to 20 s) cost VRAM and time linearly and
  lose coherence - treat anything past ~8 s as experimental.
- Wan requires a 4k+1 frame count; the requested duration is snapped to the
  nearest valid count automatically.
- **Render at the native resolution (1280x704) - resolution is not a speed
  dial.** The 5B was trained at 720p; well below that, output collapses into
  washed-out smears rather than a "faster preview". Verified on real
  hardware: the same prompt and seed that produce a crisp, on-prompt clip at
  1280x704 produce unrecognisable mush at 640x368. Iterate by shortening the
  clip and lowering steps instead, then re-render the keeper at full length
  with the same `--seed`.
- Measured on a 16 GB RDNA2 card (RX 6900 XT, native ROCm, no flash
  attention): a **1 s 1280x704 clip at 20 steps takes ~7.5 minutes** end to
  end (~13.5 s per sampler step + ~3 minutes of model loading). Sampling
  cost grows super-linearly with frame count, so a full 5 s clip at 30 steps
  is an **hours-scale job** on this class of hardware - queue it when you
  are away from the GPU.

## Model files

The template expects the public Comfy-Org repackaged files (ComfyUI v0.3.46+
has the Wan 2.2 nodes built in). Download from
`https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged` into ComfyUI's
model directories:

| File | ComfyUI directory |
|---|---|
| `wan2.2_ti2v_5B_fp16.safetensors` (~10 GB) | `models/diffusion_models/` |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (~6 GB) | `models/text_encoders/` |
| `wan2.2_vae.safetensors` (~1.4 GB) | `models/vae/` |

The fp16 encoder (`umt5_xxl_fp16.safetensors`, ~11 GB) works too - it just
occupies ~11 GB of VRAM during text encoding before being offloaded, adding
load time on a 16 GB card. Prefer the fp8_scaled file; with a different
encoder filename you need a `wan_workflow_local.json` override (below).

## Usage

GUI - the **Video** page has the full form (prompt, negative, duration, fps,
resolution, seed, steps, CFG, optional start image), a streamed job log, an
inline player, and a history with play / move-to-folder / delete.

Chat - `/video <prompt>` generates a default ~5 s clip inline and attaches a
player to the conversation.

CLI:

```bash
localm video "a red fox running through fresh snow, low tracking shot"
localm video "waves rolling in at dusk" --image beach.png        # image-to-video
localm video "city timelapse" -d 1 --steps 20 --seed 7   # quick iteration (~7 min)
```

API: `POST /api/video` returns a job id; stream progress via
`GET /api/jobs/{id}/events`. Files are served from
`GET /api/video/file/{name}`, listed by `GET /api/video/history`.

Prompt tip: **motion verbs matter**. "a fox" tends to produce a near-static
shot; "a fox running, camera tracking low" produces motion.

## VRAM handover

Same lifecycle as image and music generation: the chat model is unloaded
before the workflow is queued, and after a successful render ComfyUI is asked
to release its models (`/free`) and the chat model reloads. If ComfyUI is not
running, the job tells you how to start it - or starts it automatically when
`comfy_launch_cmd` is set in the config.

## Using your own workflow

Drop a `wan_workflow_local.json` next to
`localm/video_gen/wan_workflow.json` (it is gitignored - which models you run
stays private). The local graph must keep the template's node ids so
parameter injection still works:

| Node id | Role |
|---|---|
| `4` | positive prompt (`CLIPTextEncode`) |
| `5` | negative prompt (`CLIPTextEncode`) |
| `6` | video latent - `width` / `height` / `length` (+ `start_image` for i2v) |
| `8` | sampler - `seed` / `steps` / `cfg` (`KSampler`) |
| `10` | `CreateVideo` - `fps` |

## Privacy

In privacy mode (the default) no `<clip>.mp4.json` sidecar is written - the
prompt never touches disk. The clip itself is an explicit artifact and is
always saved; the copy in ComfyUI's own output directory is deleted when
`comfy_output_dir` is configured. In `log`/`full` modes the sidecar records
prompt, seed, and settings so a clip can be reproduced (`seed` is also shown
in the success message either way).

## Troubleshooting

- **"returned no prompt_id"** - almost always missing model files (check the
  ComfyUI console) or a ComfyUI older than v0.3.46 (no Wan 2.2 nodes).
- **Washed-out, smeared, unrecognisable output** - resolution below the
  model's native 1280x704, and/or too few steps. Render at native resolution
  with 20+ steps; shorten the clip to save time instead.
- **Out of VRAM during sampling** - shorten the clip (fewer frames) or close
  other GPU users. Don't drop resolution below native to save memory -
  quality collapses (see above).
- **"Ran out of memory when regular VAE decoding"** in the ComfyUI console -
  normal on 16 GB at 720p; ComfyUI automatically retries with tiled decoding
  and the clip comes out fine.
- **Static output** - add motion language to the prompt; raise CFG slightly.
- **Timeout** - the default poll timeout is 60 minutes; very long/large clips
  on slow cards can exceed it. Generate shorter clips or pass a larger
  `max_poll_seconds` via the Python API.
