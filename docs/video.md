# Video generation (ComfyUI Wan 2.2)

localm generates short video clips (MP4, h264) through the same local ComfyUI server it uses for images and music. The committed workflow template runs the public **Wan 2.2 TI2V 5B** stack - text-to-video by default, image-to-video when you provide a start picture. Nothing leaves your machine: ComfyUI runs locally, and the one-time model download is the only network access.

## What you get

Four surfaces: the **Video page** in the GUI, the `localm video` **CLI**, the `/generate-video` **chat** command, and the `POST /api/video` HTTP **API**. See [Usage](#usage) below for each. Generated clips are stored in the localm data directory at `gui_video/` and are always saved (see [Privacy](#privacy) for what metadata is written).

## Expect slow render times - timing examples

Video is the **slowest and most VRAM-hungry generator** in localm. Unlike ACE-Step music (arbitrary length), a video model attends over all frames at once, so VRAM and time grow with frame count.

**Measured on a 16 GB RDNA2 card (RX 6900 XT, native ROCm, no flash attention), not independently re-verified since:**

- **1 second at 1280x704, 20 steps: ~7.5 minutes end to end** (~13.5 s per sampler step + ~3 minutes of model loading). Queue this when you can step away. Note: the current template default is **30** steps (`localm/video_gen/comfy.py`'s `generate_video`), not 20 - at the same per-step cost this scales to roughly ~10 minutes for 1 second at the default step count, but that number is arithmetic from the measurement above, not a fresh benchmark.
- A full 5 s clip at 30 steps (the default) is an **hours-scale job** on this class of hardware - consider batch-generating multiple variations overnight or on a faster GPU.
- Sampling cost grows super-linearly with frame count; 8+ second clips lose coherence and should be treated as experimental.

The default poll timeout is 60 minutes; very long/large clips on slow cards can exceed it. In the Python API, pass a larger `max_poll_seconds` to adjust. Shorter clips always finish faster.

## Specifications

**~5 seconds is the native clip length** (121 frames at 24 fps). Quality is best there. Longer clips (the API accepts up to 20 s) cost VRAM and time linearly and lose coherence.

**Wan requires a 4k+1 frame count.** The requested duration is snapped to the nearest valid count automatically. At 24 fps:
- 1 s = 25 frames
- 3 s = 73 frames
- 5 s = 121 frames (native)
- 10 s = 241 frames

**Render at the native resolution (1280x704).** The 5B was trained at 720p. Resolution is not a speed dial - well below native, output collapses into washed-out smears rather than a "faster preview". Iterate by shortening the clip and lowering steps instead, then re-render the keeper at full length with the same `--seed`. Width and height are free integers (any multiple of 16 works and nothing is enforced); these are illustrative, not a fixed preset list:

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

The fp16 encoder (`umt5_xxl_fp16.safetensors`, ~11 GB) also works but uses more VRAM; prefer the fp8_scaled file. A different encoder filename needs a custom workflow (see [Using your own workflow](#using-your-own-workflow), below).

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
localm video "a fox running, camera tracking low" --cfg 6.0      # follow the prompt harder
```

See the [CLI Reference](cli.md) for the full command syntax and options.

### API

`POST /api/video` accepts prompt, seconds, fps, width, height, seed, steps, cfg, and input_image, and returns a job id.

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

Generated clips are stored at `<data dir>/gui_video/` as MP4 files, each with an optional `.json` sidecar (see [Privacy](#privacy)).

**Via the GUI Video page:** the history shows all clips with their metadata, and you can play, download, move to a folder, or delete them inline.

**Via the API:** `GET /api/video/history` lists clips as JSON with filenames, metadata, file sizes, and modification times; `GET /api/video/file/{name}` serves the clip. The newest 100 clips are kept in the history.

**Via the CLI:** generated clips are saved to your specified output path (or `video_<timestamp>.mp4` by default in the current directory) and are not moved to `gui_video/`.

## VRAM handover

**On the GUI Video page, the `/generate-video` chat command, and a direct `POST /api/video` call** (all three go through the same background job): the chat model is unloaded before the workflow is queued, and once the job finishes - whether it succeeded, failed, or was cancelled - ComfyUI is asked to release its models (`/free`) and the chat model reloads.

**The `localm video` CLI does not do this on its own.** It talks to ComfyUI directly and has no server to hand the chat model back to unless you set the `LOCALM_URL` environment variable to a running localm server (and, for a keyless server, also pass a valid instance token); without it, `localm video` never unloads or reloads anything, and a co-running chat model stays resident in VRAM alongside ComfyUI's.

If ComfyUI is not running, the job tells you how to start it - or starts it automatically when `comfy_launch_cmd` is set in the config. Interrupting a CLI generation with Ctrl-C tells ComfyUI to abort the render and free its VRAM instead of leaving it running.

## Using your own workflow

The **Workflow** card on the Video page is the current way to do this: export your graph from ComfyUI (Save -> API format), upload it there, and select it - or keep the built-in default. The same is available from the CLI, and works fully offline:

```bash
localm comfy workflow add video my_wan.json --use   # upload and select it
localm comfy workflow list video                    # see what is uploaded and active
localm comfy workflow use video --clear             # back to the built-in default
```

Either way, uploaded workflows are stored per-plugin under the localm data directory (`workflows/video/`), which models you run stays private, and the choice survives a self-update (the `localm/` package directory is whole-tree-replaced on update; the data directory is not). Whatever is selected governs `localm video` too, not just the GUI.

Parameters are injected by role, so node ids do not matter: the graph just needs a `KSampler` wired with `positive` / `negative` / `latent_image` inputs and a `CreateVideo` node for fps.

The older `wan_workflow_local.json` file dropped next to `localm/video_gen/wan_workflow.json` still works (it is gitignored) but is superseded by the Workflow card above: on first load, any existing override there is migrated into the new per-plugin store automatically, keeping it selected and preserving your current setup.

## Picking models per slot

The Video page shows a **Models** panel for the active workflow: one dropdown per
model file the workflow uses, labeled by the role it fills (diffusion model,
text encoder, VAE) rather than the raw ComfyUI field name. A slot whose file is
not installed in ComfyUI is called out, and a model of that kind you already have
registered in localm but ComfyUI is not offering gets a note saying where to move
it. With ComfyUI unreachable the panel falls back to listing what the workflow
needs and which of your registered models could fill each slot, instead of going
blank.

## Privacy

In privacy mode (the default) no `<clip>.mp4.json` sidecar is written - the prompt never touches disk. The clip itself is an explicit artifact and is always saved. Privacy mode also forces removal of ComfyUI's own on-disk copy of the clip (and any uploaded image-to-video source) - but that removal needs localm to be able to locate ComfyUI's output folder (the `comfy_output_dir` setting, or a derived `<comfy_workdir>/output`); if neither resolves, a warning tells you a copy remains there instead of silently claiming it was removed. Outside privacy mode, the same cleanup is opt-in via the `comfy_delete_outputs` setting (default off, so ComfyUI's own gallery keeps its copy). In `log`/`full` modes the sidecar records prompt, seed, and settings so a clip can be reproduced (`seed` is also shown in the success message either way).

## Troubleshooting

**"returned no prompt_id"** - almost always missing model files (check the ComfyUI console) or a ComfyUI older than v0.3.46 (no Wan 2.2 nodes). Make sure the three model files are present in ComfyUI's model directories and readable.

**Washed-out, smeared, unrecognisable output** - the resolution is below native 1280x704 and/or you have too few steps. Render at native with 20+ steps and shorten the clip to save time instead (see [Specifications](#specifications)).

**Out of VRAM during sampling** - shorten the clip (fewer frames) or close other GPU users. Don't drop resolution below native to save memory - quality collapses (see above). If you have a card with 8-12 GB VRAM, consider starting with 3 s clips and 20 steps.

**"Ran out of memory when regular VAE decoding"** in the ComfyUI console - normal on 16 GB at 720p; ComfyUI automatically retries with tiled decoding and the clip comes out fine. If it keeps failing, the clip is too long for your VRAM; shorten it or reduce steps.

**Static output** - add explicit motion verbs to the prompt (see the prompt tip above). Raise CFG slightly (5.0 to 6.0) to make the model follow the prompt harder; at very high CFG (8+) output becomes erratic.

**Job times out** - the default poll timeout is 60 minutes (see the timing section). Generate shorter clips, or in the Python API pass a larger `max_poll_seconds` (e.g. `max_poll_seconds=7200` for 2 hours).

**ComfyUI does not start** - check that `comfy_launch_cmd` is set in your config (e.g. `comfy_launch_cmd: python -m ComfyUI.main` with `comfy_workdir` set to your ComfyUI repo). The GUI and CLI both auto-launch when configured. You can also start ComfyUI manually via `python main.py` in the ComfyUI directory; it prints its URL (default http://127.0.0.1:8188). Make sure the URL matches what localm expects in the config. To avoid managing ComfyUI yourself, you can let localm run its own managed instance instead (see [managed-comfyui.md](managed-comfyui.md)).
