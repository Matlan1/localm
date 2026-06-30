# SPDX-License-Identifier: AGPL-3.0-or-later
import sys
from pathlib import Path

import click

from ._core import console, main


def _open_file(path: Path) -> None:
    """Open *path* with the OS default application (best-effort, never fatal)."""
    import os as _os
    import subprocess as _sp
    try:
        if sys.platform == "win32":
            _os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            _sp.Popen(["open", str(path)])
        else:
            _sp.Popen(["xdg-open", str(path)])
    except Exception as e:
        console.print(f"[dim]Could not open it automatically: {e}[/dim]")




def _offer_open(path: Path) -> None:
    """In an interactive terminal, offer to open/play the just-generated media.

    The CLI cannot display an image or play audio/video itself, but the file is
    on disk - so we offer to open it in the OS default app. Skipped silently in
    a non-interactive shell (piped/scripted), where we only print the path.
    """
    try:
        if not (sys.stdin and sys.stdin.isatty()):
            return
    except Exception:
        return
    try:
        ans = console.input("  Open it now? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("")
        return
    if ans in ("", "y", "yes"):
        _open_file(path)




@main.command("image")
@click.argument("prompt")
@click.option("--negative", default=None,
              help="Negative prompt (things to steer away from).")
@click.option("--guidance", type=float, default=None,
              help="FluxGuidance scale (default ~3.5; higher = follows prompt harder).")
@click.option("--cfg", type=float, default=None,
              help="CFG scale for the negative-prompt path (default keeps the workflow's).")
@click.option("--seed", type=int, default=None, help="Reproducible seed.")
@click.option("--image", "input_image", type=click.Path(exists=True), default=None,
              help="Use this picture as a base (image-to-image) instead of noise.")
@click.option("--denoise", type=float, default=None,
              help="img2img strength 0-1 (lower keeps more of the base image).")
@click.option("--lora", "lora_name", default=None,
              help="Optional LoRA file name (in ComfyUI's loras dir) to apply.")
@click.option("-o", "--out", default=None,
              help="Output .png path [default: ./image_<timestamp>.png]")
def image_cmd(prompt, negative, guidance, cfg, seed, input_image, denoise,
              lora_name, out):
    """Generate an image with the local ComfyUI FLUX workflow.

    \b
    Examples:
      localm image "a red fox in snow, photographic"
      localm image "make it look like sunset" --image photo.png --denoise 0.6

    ComfyUI must be running (or start it via the GUI, which can auto-launch it
    when comfy_launch_cmd is configured). The CLI cannot show the image, so it
    is saved to --out (or ./image_<timestamp>.png) and, in an interactive
    terminal, you are offered to open it.
    """
    import time as _time

    from ..audit import SessionMode, effective_mode
    from ..image_gen.comfy import (default_api_url, free_comfy_vram,
                                  generate_image)

    api_url = default_api_url()
    # generate_image() calls ensure_comfy() internally, which auto-launches ComfyUI
    # from comfy_launch_cmd/comfy_workdir (or returns a clear error when they are
    # unset), so the CLI honours the same config the GUI uses (H1).

    out_path = Path(out) if out \
        else Path(f"image_{_time.strftime('%Y%m%d_%H%M%S')}.png")
    kwargs = {k: v for k, v in (
        ("negative_prompt", negative), ("guidance", guidance), ("cfg", cfg),
        ("seed", seed), ("denoise", denoise), ("lora_name", lora_name),
    ) if v is not None}
    if input_image:
        kwargs["input_image"] = Path(input_image)

    console.print("[dim]Generating image via ComfyUI (this can take a minute)...[/dim]")
    ok, message = generate_image(
        prompt, out_path,
        api_url=api_url,
        write_sidecar=effective_mode("server") != SessionMode.PRIVACY,
        **kwargs,
    )
    console.print(f"[{'green' if ok else 'red'}]{message}[/{'green' if ok else 'red'}]")
    if not ok:
        sys.exit(1)
    free_comfy_vram(api_url)
    _offer_open(out_path)




@main.command("music")
@click.argument("tags")
@click.option("--lyrics", type=click.Path(exists=True), default=None,
              help="Lyrics file ([verse]/[chorus] markers supported); "
                   "omit for an instrumental.")
@click.option("-d", "--duration", default=120.0, show_default=True,
              help="Track length in seconds - arbitrary.")
@click.option("-o", "--out", default=None,
              help="Output .flac path [default: ./music_<timestamp>.flac]")
@click.option("--seed", type=int, default=None, help="Reproducible seed.")
@click.option("--steps", type=int, default=None, help="Sampler steps (default 50).")
@click.option("--cfg", type=float, default=None, help="Guidance (default 5.0).")
def music_cmd(tags, lyrics, duration, out, seed, steps, cfg):
    """Generate a music track with the local ComfyUI ACE-Step workflow.

    \b
    Examples:
      localm music "synthwave, 80s, 120 bpm, dreamy"
      localm music "folk ballad, acoustic guitar" --lyrics song.txt -d 180

    ComfyUI must be running (or start it via the GUI, which can auto-launch
    it when comfy_launch_cmd is configured).
    """
    import time as _time
    from rich.console import Console
    from ..audit import SessionMode, effective_mode
    from ..music_gen import generate_music
    console = Console()

    # generate_music() calls ensure_comfy() internally (auto-launch from
    # comfy_launch_cmd/comfy_workdir, or a clear error when unset), so the CLI
    # honours the same config the GUI uses (H1).

    out_path = Path(out) if out \
        else Path(f"music_{_time.strftime('%Y%m%d_%H%M%S')}.flac")
    lyr = Path(lyrics).read_text(encoding="utf-8") if lyrics else None
    kwargs = {k: v for k, v in
              (("seed", seed), ("steps", steps), ("cfg", cfg)) if v is not None}
    ok, message = generate_music(
        tags, out_path,
        lyrics=lyr,
        duration_seconds=duration,
        on_progress=lambda t: console.print(f"  [dim]{t}[/dim]"),
        write_sidecar=effective_mode("server") != SessionMode.PRIVACY,
        **kwargs,
    )
    console.print(f"[{'green' if ok else 'red'}]{message}[/{'green' if ok else 'red'}]")
    if not ok:
        sys.exit(1)
    _offer_open(out_path)




@main.command("video")
@click.argument("prompt")
@click.option("--negative", default=None,
              help="Negative prompt (default suppresses blur/watermarks).")
@click.option("-d", "--duration", default=5.0, show_default=True,
              help="Clip length in seconds (snapped to Wan's 4k+1 frame "
                   "rule; ~5s is the model's native length).")
@click.option("--fps", default=24, show_default=True, help="Frame rate.")
@click.option("--width", type=int, default=None,
              help="Width (multiple of 16; default 1280 - the model's native "
                   "resolution; quality collapses well below it).")
@click.option("--height", type=int, default=None,
              help="Height (multiple of 16; default 704 - see --width).")
@click.option("--image", "input_image", type=click.Path(exists=True),
              default=None,
              help="Animate this picture instead of starting from noise "
                   "(image-to-video).")
@click.option("-o", "--out", default=None,
              help="Output .mp4 path [default: ./video_<timestamp>.mp4]")
@click.option("--seed", type=int, default=None, help="Reproducible seed.")
@click.option("--steps", type=int, default=None, help="Sampler steps (default 30).")
@click.option("--cfg", type=float, default=None, help="Guidance (default 5.0).")
def video_cmd(prompt, negative, duration, fps, width, height, input_image,
              out, seed, steps, cfg):
    """Generate a short video clip with the local ComfyUI Wan 2.2 workflow.

    \b
    Examples:
      localm video "a red fox running through snow, tracking shot"
      localm video "waves rolling in at dusk" --image beach.png -d 5

    ComfyUI must be running (or start it via the GUI, which can auto-launch
    it when comfy_launch_cmd is configured). Video is the slowest generator -
    expect many minutes per clip; see docs/video.md for model setup.
    """
    import time as _time
    from rich.console import Console
    from ..audit import SessionMode, effective_mode
    from ..video_gen import generate_video
    console = Console()

    # generate_video() calls ensure_comfy() internally (auto-launch from
    # comfy_launch_cmd/comfy_workdir, or a clear error when unset), so the CLI
    # honours the same config the GUI uses (H1).

    out_path = Path(out) if out \
        else Path(f"video_{_time.strftime('%Y%m%d_%H%M%S')}.mp4")
    kwargs = {k: v for k, v in
              (("negative_prompt", negative), ("width", width),
               ("height", height), ("seed", seed), ("steps", steps),
               ("cfg", cfg)) if v is not None}
    ok, message = generate_video(
        prompt, out_path,
        seconds=duration,
        fps=fps,
        input_image=Path(input_image) if input_image else None,
        on_progress=lambda t: console.print(f"  [dim]{t}[/dim]"),
        write_sidecar=effective_mode("server") != SessionMode.PRIVACY,
        **kwargs,
    )
    console.print(f"[{'green' if ok else 'red'}]{message}[/{'green' if ok else 'red'}]")
    if not ok:
        sys.exit(1)
    _offer_open(out_path)
