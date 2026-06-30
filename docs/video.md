# Video generation (ComfyUI Wan 2.2)

localm generates short video clips (MP4, h264) through the same local ComfyUI server it uses for images and music. The committed workflow template runs the public **Wan 2.2 TI2V 5B** stack - text-to-video by default, image-to-video when you provide a start picture. Nothing leaves your machine: ComfyUI runs locally, and the one-time model download is the only network access.

## What you get

**The Video page** (when the video plugin is enabled) has the full generation form - prompt, negative, duration, fps, resolution, seed, steps, CFG, optional start image - a streamed job log, an inline player, and a history with play, move-to-folder, and delete actions. Generated clips are stored in the localm data directory at `gui_video/` and are always saved; the prompt is never written to disk in privacy mode (the default).

**CLI**: `localm video "a red fox running through snow, tracking shot"` generates a default ~5 s clip and saves it as `video_<timestamp>.mp4` in the current directory; see [CLI Reference](cli.md) for the full command syntax and options.

**Chat**: `/generate-video <prompt>` generates a default ~5 s clip inline and attaches a player to the conversation.

**API**: `POST /api/video` (with fields like prompt, seconds, fps, width, height, seed, steps, cfg, input_image) returns a job id; stream progress via `GET /api/jobs/{id}/events`. Use `GET /api/video/history` to list all generated clips, and `GET /api/video/file/{name}` to serve a clip.

## Expect slow render times - timing examples

Video is the **slowest and most VRAM-hungry generator** in localm. Unlike ACE-Step music (arbitrary length), a video model attends over all frames at once, so VRAM and time grow with frame count.

**Measured on a 16 GB RDNA2 card (RX 6900 XT, native ROCm, no flash attention):**

- **1 second at 1280x704, 20 steps: ~7.5 minutes end to end** (~13.5 s per sampler step + ~3 minutes of model loading). Queue this when you can step away.
- A full 5 s clip at 30 steps is an **hours-scale job** on this class of hardware - consider batch-generating multiple variations overnight or on a faster GPU.
- Sampling cost grows super-linearly with frame count; 8+ second clips lose coherence and should be treated as experimental.

The default poll timeout is 60 minutes; very long/large clips on slow cards can exceed it. In the Python API, pass a larger `max_poll_seconds` to adjust. Shorter clips always finish faster.

## Specifications

**~5 seconds is the native clip length** (121 frames at 24 fps). Quality is best there. Longer clips (the API accepts up to 20 s) cost VRAM and time linearly and lose coherence.

**Wan requires a 4k+1 frame count.** The requested duration is snapped to the nearest valid count automatically. At 24 fps:
- 1 s = 25 frames
- 3 s = 73 frames
- 5 s = 121 frames (native)
- 10 s = 241 frames

**Render at the native resolution (1280x704).** The 5B was trained at 720p. Resolution is not a speed dial - well below native, output collapses into washed-out smears rather than a "faster preview". Verified on real hardware: the same prompt and seed that produce a crisp, on-prompt clip at 1280x704 produce unrecognisable mush at 640x368. Iterate by shortening the clip and lowering steps instead, then re-render the keeper at full length with the same `--seed`. Supported preset sizes (multiples of 16):

- **1280x704** (recommended - native)
- 1024x576 (not recommended - quality drops sharply)
- 768x432 (mush)

## Model files

The template expects the public Comfy-Org repackaged files (ComfyUI v0.3.46+ has the Wan 2.2 nodes built in). Download from `https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged` into ComfyUI's model directories:

| File | ComfyUI directory |
|---|---|
| `wan2.2_ti2v_5B_fp16.safetensors` (~10 GB) | `models/diffusion_models/` |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (~6 GB) | `models/text_encoders/` |
| `wan2.2_vae.safetensors` (~1.4 GB) | `models/vae/` |

The fp16 encoder (`umt5_xxl_fp16.safetensors`, ~11 GB) works too - it just occupies ~11 GB of VRAM during text encoding before being offloaded, adding load time on a 16 GB card. Prefer the fp8_scaled file; with a different encoder filename you need a `wan_workflow_local.json` override (below).

## Usage

### GUI

The **Video** page has the full form (prompt, negative, duration, fps, resolution, seed, steps, CFG, optional start image), a streamed job log, an inline player, and a history with play, move-to-folder, and delete actions.

### Chat

`/generate-video <prompt>` generates a default ~5 s clip inline and attaches a player to the conversation.

### CLI

```bash
localm video "a red fox running through fresh snow, low tracking shot"
localm video "waves rolling in at dusk" --image beach.png        # image-to-video
localm video "city timelapse" -d 1 --steps 20 --seed 7   # quick iteration (~7 min)
localm video "gentle ocean waves" --width 1280 --height 704      # explicit resolution
```

### API

```bash
# Start a generation (returns job_id)
curl -X POST http://127.0.0.1:8642/api/video \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red fox running", "seconds": 5, "steps": 30}'

# Stream job progress
curl http://127.0.0.1:8642/api/jobs/{job_id}/events

# List all generated clips
curl http://127.0.0.1:8642/api/video/history

# Download a clip
curl -O http://127.0.0.1:8642/api/video/file/{name}

# Delete a clip (and sidecar metadata)
curl -X DELETE http://127.0.0.1:8642/api/video/file/{name}

# Move a clip to a folder
curl -X POST http://127.0.0.1:8642/api/video/file/{name}/move \
  -H "Content-Type: application/json" \
  -d '{"dest": "/path/to/folder"}'
```

Prompt tip: **motion verbs matter**. "a fox" tends to produce a near-static shot; "a fox running, camera tracking low" produces motion.

## Retrieving results

Generated clips are stored at `~/.localm/gui_video/` (or the configured data directory). Each clip is an MP4 file with an optional `.json` sidecar in non-privacy modes containing the prompt, seed, and generation settings for reproducibility.

**Via the GUI Video page:** the history shows all clips with their metadata, and you can play, download, move to a folder, or delete them inline.

**Via the API:** `GET /api/video/history` lists clips as JSON with filenames, metadata, file sizes, and modification times; `GET /api/video/file/{name}` serves the clip. The newest 100 clips are kept in the history.

**Via the CLI:** generated clips are saved to your specified output path (or `video_<timestamp>.mp4` by default in the current directory) and are not moved to `gui_video/`.

## VRAM handover

Same lifecycle as image and music generation: the chat model is unloaded before the workflow is queued, and after a successful render ComfyUI is asked to release its models (`/free`) and the chat model reloads. If ComfyUI is not running, the job tells you how to start it - or starts it automatically when `comfy_launch_cmd` is set in the config.

## Using your own workflow

Drop a `wan_workflow_local.json` next to `localm/video_gen/wan_workflow.json` (it is gitignored - which models you run stays private). The local graph must keep the template's node ids so parameter injection still works:

| Node id | Role |
|---|---|
| `4` | positive prompt (`CLIPTextEncode`) |
| `5` | negative prompt (`CLIPTextEncode`) |
| `6` | video latent - `width` / `height` / `length` (+ `start_image` for i2v) |
| `8` | sampler - `seed` / `steps` / `cfg` (`KSampler`) |
| `10` | `CreateVideo` - `fps` |

## Privacy

In privacy mode (the default) no `<clip>.mp4.json` sidecar is written - the prompt never touches disk. The clip itself is an explicit artifact and is always saved; the copy in ComfyUI's own output directory is deleted when `comfy_output_dir` is configured. In `log`/`full` modes the sidecar records prompt, seed, and settings so a clip can be reproduced (`seed` is also shown in the success message either way).

## Troubleshooting

**"returned no prompt_id"** - almost always missing model files (check the ComfyUI console) or a ComfyUI older than v0.3.46 (no Wan 2.2 nodes). Make sure the three model files are present in ComfyUI's model directories and readable.

**Washed-out, smeared, unrecognisable output** - the resolution is below the model's native 1280x704 and/or you have too few steps. Render at native resolution with 20+ steps; shorten the clip to save time instead. The 5B was trained exclusively at 720p - there is no "lower resolution = faster preview" option.

**Out of VRAM during sampling** - shorten the clip (fewer frames) or close other GPU users. Don't drop resolution below native to save memory - quality collapses (see above). If you have a card with 8-12 GB VRAM, consider starting with 3 s clips and 20 steps.

**"Ran out of memory when regular VAE decoding"** in the ComfyUI console - normal on 16 GB at 720p; ComfyUI automatically retries with tiled decoding and the clip comes out fine. If it keeps failing, the clip is too long for your VRAM; shorten it or reduce steps.

**Static output** - the model defaults to near-static scenes when motion language is absent. Add explicit motion verbs to the prompt ("running", "flying", "rolling", "spinning"). Raise CFG slightly (5.0 to 6.0) to make the model follow the prompt harder; at very high CFG (8+) output becomes erratic.

**Job times out** - the default poll timeout is 60 minutes. Very long or large clips on slow cards can exceed it. Generate shorter clips, or in the Python API pass a larger `max_poll_seconds` (e.g. `max_poll_seconds=7200` for 2 hours).

**ComfyUI does not start** - check that `comfy_launch_cmd` is set in your config (e.g. `comfy_launch_cmd: python -m ComfyUI.main` with `comfy_workdir` set to your ComfyUI repo). The GUI and CLI both auto-launch when configured. You can also start ComfyUI manually via `python main.py` in the ComfyUI directory; it prints its URL (default http://127.0.0.1:8188). Make sure the URL matches what localm expects in the config.
