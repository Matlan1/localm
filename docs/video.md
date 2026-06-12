# Video generation (ComfyUI Wan 2.2)

localm generates short video clips through the same local ComfyUI server it
uses for images and music. The committed workflow template runs the public
**Wan 2.2 TI2V 5B** stack — text-to-video by default, image-to-video when you
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
  lose coherence — treat anything past ~8 s as experimental.
- Wan requires a 4k+1 frame count; the requested duration is snapped to the
  nearest valid count automatically.
- Measured on a 16 GB RDNA2 card (RX 6900 XT, native ROCm, no flash
  attention): a **2 s 640x368 clip at 10 steps took ~9 minutes** end to end
  (including the ~20 GB model load). Attention cost grows super-linearly with
  frames x pixels, so a full 5 s 832x480 clip at 30 steps lands **well over an
  hour** on this class of hardware. The practical workflow: iterate short and
  small (`-d 2 --width 640 --height 368 --steps 10`), then re-render the
  keeper at full settings overnight with the same `--seed`.

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

## Usage

GUI — the **Video** page has the full form (prompt, negative, duration, fps,
resolution, seed, steps, CFG, optional start image), a streamed job log, an
inline player, and a history with play / move-to-folder / delete.

Chat — `/video <prompt>` generates a default ~5 s clip inline and attaches a
player to the conversation.

CLI:

```bash
localm video "a red fox running through fresh snow, low tracking shot"
localm video "waves rolling in at dusk" --image beach.png        # image-to-video
localm video "city timelapse" -d 5 --width 640 --height 368 --steps 20  # faster
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
running, the job tells you how to start it — or starts it automatically when
`comfy_launch_cmd` is set in the config.

## Using your own workflow

Drop a `wan_workflow_local.json` next to
`localm/video_gen/wan_workflow.json` (it is gitignored — which models you run
stays private). The local graph must keep the template's node ids so
parameter injection still works:

| Node id | Role |
|---|---|
| `4` | positive prompt (`CLIPTextEncode`) |
| `5` | negative prompt (`CLIPTextEncode`) |
| `6` | video latent — `width` / `height` / `length` (+ `start_image` for i2v) |
| `8` | sampler — `seed` / `steps` / `cfg` (`KSampler`) |
| `10` | `CreateVideo` — `fps` |

## Privacy

In privacy mode (the default) no `<clip>.mp4.json` sidecar is written — the
prompt never touches disk. The clip itself is an explicit artifact and is
always saved; the copy in ComfyUI's own output directory is deleted when
`comfy_output_dir` is configured. In `log`/`full` modes the sidecar records
prompt, seed, and settings so a clip can be reproduced (`seed` is also shown
in the success message either way).

## Troubleshooting

- **"returned no prompt_id"** — almost always missing model files (check the
  ComfyUI console) or a ComfyUI older than v0.3.46 (no Wan 2.2 nodes).
- **Out of VRAM** — lower `--width/--height` (640x368 works well), shorten
  the clip, or close other GPU users. The 5B model plus text encoder is a
  tight fit on 16 GB at 832x480x121.
- **Static output** — add motion language to the prompt; raise CFG slightly.
- **Timeout** — the default poll timeout is 60 minutes; very long/large clips
  on slow cards can exceed it. Generate shorter clips or pass a larger
  `max_poll_seconds` via the Python API.
