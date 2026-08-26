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




def _is_interactive() -> bool:
    """True when we have an interactive terminal to prompt on (not piped/scripted)."""
    try:
        return bool(sys.stdin and sys.stdin.isatty())
    except Exception:
        return False


def _offer_open(path: Path) -> None:
    """In an interactive terminal, offer to open the just-generated media in
    the OS default app. Skipped silently in a non-interactive shell.
    """
    if not _is_interactive():
        return
    try:
        ans = console.input("  Open it now? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("")
        return
    if ans in ("", "y", "yes"):
        _open_file(path)


def _remember_func_shim() -> None:
    """Persist comfy_func_shim=True so every future localm-spawned ComfyUI gets
    the in-memory __func__ shim automatically."""
    from ..config import update_config
    update_config(lambda cfg: cfg.__setitem__("comfy_func_shim", True))


def _maybe_apply_func_shim_and_retry(message: str, api_url: str, retry):
    """React to a media generation that failed with the known ComfyUI __func__
    regression. On an interactive terminal, offer localm's in-memory,
    localm-side shim and, on consent, apply it to a ComfyUI localm SPAWNS and retry
    ONCE. Returns the (possibly retried) ``(ok, message)``; a decline, an unrelated
    error, or a non-interactive shell returns ``(False, message)`` unchanged. Never
    touches a ComfyUI localm did not start.

    *retry* is a zero-arg callable that re-runs the exact same generation and returns
    its own ``(ok, message)``."""
    from ..media.comfy_client import (enable_func_shim_once,
                                       is_known_comfy_func_regression,
                                       restart_comfy, spawned_pid)
    if not is_known_comfy_func_regression(message):
        return False, message
    if not _is_interactive():
        return False, message
    console.print(
        "[yellow]This looks like the known ComfyUI __func__ regression "
        "(Comfy-Org/ComfyUI #12116).[/yellow] localm can apply an in-memory, "
        "localm-side fix to a ComfyUI it starts: it writes nothing into your "
        "ComfyUI install and self-expires once ComfyUI ships its own fix.")
    # Alongside the fix-this-run shim, offer localm's own managed, patched
    # ComfyUI - once only, and only while the user has no managed instance.
    from ..media.comfy_client import (managed_comfy_setup_offer_message,
                                      mark_managed_comfy_setup_offered,
                                      should_offer_managed_comfy_setup)
    if should_offer_managed_comfy_setup(message):
        console.print("[cyan]" + managed_comfy_setup_offer_message() + "[/cyan]")
        mark_managed_comfy_setup_offered()
    try:
        ans = console.input(
            "  Apply localm's fix? [o]nce / [r]emember (stop asking) / [N]o: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("")
        return False, message
    if ans in ("o", "once"):
        enable_func_shim_once()
    elif ans in ("r", "remember"):
        enable_func_shim_once()
        _remember_func_shim()
    else:
        return False, message
    # Applies to a ComfyUI localm spawns only. If localm launched the live one,
    # restart it with the fix; otherwise the user's own instance is left
    # untouched and they are asked to close it, so the retry starts a fixed one.
    if spawned_pid(api_url) is not None:
        console.print("[dim]Restarting the ComfyUI localm launched, with the fix...[/dim]")
        restart_comfy(api_url)
    else:
        console.print(
            "[yellow]localm did not start this ComfyUI, so it will not touch it. "
            "Close your ComfyUI, then press Enter and localm will start a fixed one "
            "(needs comfy_workdir set).[/yellow]")
        try:
            console.input("  Press Enter when ComfyUI is closed (Ctrl-C to skip): ")
        except (EOFError, KeyboardInterrupt):
            console.print("")
            return False, message
    console.print("[dim]Retrying generation with the fix...[/dim]")
    return retry()




def _generate_or_abort(api_url: str, run):
    """Run a media generation, and on Ctrl-C tell ComfyUI to stop the render.

    On KeyboardInterrupt it calls ``interrupt_comfy`` (abort the running prompt
    and clear the queue) and ``free_comfy_vram`` - plain HTTP calls that work
    from any process - reports whether the abort actually landed, and re-raises
    the KeyboardInterrupt.
    """
    from ..media.comfy_client import free_comfy_vram, interrupt_comfy
    try:
        return run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted - telling ComfyUI to stop...[/yellow]")
        aborted = False
        try:
            aborted = interrupt_comfy(api_url)
            free_comfy_vram(api_url)
        except Exception as e:
            console.print(f"[yellow]![/yellow]  Could not reach ComfyUI to stop it "
                          f"({e}). It may still be rendering and holding VRAM.")
            raise
        if aborted:
            console.print("[dim]Render aborted, queue cleared, VRAM freed.[/dim]")
        else:
            # interrupt_comfy swallows its own transport errors and returns
            # False, which means "could not tell it to stop".
            console.print("[yellow]![/yellow]  ComfyUI did not accept the abort "
                          "(it may already have stopped, or be unreachable). "
                          "Check with [dim]localm comfy status[/dim]")
        raise


@main.command("image")
@click.argument("prompt")
@click.option("--negative", default=None,
              help="Negative prompt (things to steer away from).")
@click.option("--guidance", type=float, default=None,
              help="FluxGuidance scale (default ~3.5; higher = follows prompt harder).")
@click.option("--cfg", type=float, default=None,
              help="CFG scale for the negative-prompt path (only applies with "
                   "--negative; default 3.5).")
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
    # generate_image() calls ensure_comfy() internally, which auto-launches
    # ComfyUI from comfy_launch_cmd/comfy_workdir, or returns a clear error
    # when they are unset.

    out_path = Path(out) if out \
        else Path(f"image_{_time.strftime('%Y%m%d_%H%M%S')}.png")
    kwargs = {k: v for k, v in (
        ("negative_prompt", negative), ("guidance", guidance), ("cfg", cfg),
        ("seed", seed), ("denoise", denoise), ("lora_name", lora_name),
    ) if v is not None}
    if input_image:
        kwargs["input_image"] = Path(input_image)

    console.print("[dim]Generating image via ComfyUI (this can take a minute)...[/dim]")
    _is_privacy = effective_mode("server") == SessionMode.PRIVACY
    _write_sidecar = not _is_privacy

    def _gen_image():
        return generate_image(
            prompt, out_path,
            api_url=api_url,
            write_sidecar=_write_sidecar,
            delete_outputs=_is_privacy,
            **kwargs,
        )

    ok, message = _generate_or_abort(api_url, _gen_image)
    if not ok:
        ok, message = _maybe_apply_func_shim_and_retry(
            message, api_url,
            lambda: _generate_or_abort(api_url, _gen_image))
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
    from ..media.comfy_client import default_api_url
    from ..music_gen import generate_music
    console = Console()

    # generate_music() calls ensure_comfy() internally: auto-launch from
    # comfy_launch_cmd/comfy_workdir, or a clear error when unset.

    api_url = default_api_url()
    out_path = Path(out) if out \
        else Path(f"music_{_time.strftime('%Y%m%d_%H%M%S')}.flac")
    lyr = Path(lyrics).read_text(encoding="utf-8") if lyrics else None
    kwargs = {k: v for k, v in
              (("seed", seed), ("steps", steps), ("cfg", cfg)) if v is not None}
    _is_privacy = effective_mode("server") == SessionMode.PRIVACY
    _write_sidecar = not _is_privacy

    def _gen_music():
        return generate_music(
            tags, out_path,
            lyrics=lyr,
            duration_seconds=duration,
            api_url=api_url,
            on_progress=lambda t: console.print(f"  [dim]{t}[/dim]"),
            write_sidecar=_write_sidecar,
            delete_outputs=_is_privacy,
            **kwargs,
        )

    ok, message = _generate_or_abort(api_url, _gen_music)
    if not ok:
        ok, message = _maybe_apply_func_shim_and_retry(
            message, api_url, lambda: _generate_or_abort(api_url, _gen_music))
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
    from ..media.comfy_client import default_api_url
    from ..video_gen import generate_video
    console = Console()

    # generate_video() calls ensure_comfy() internally: auto-launch from
    # comfy_launch_cmd/comfy_workdir, or a clear error when unset.

    api_url = default_api_url()
    out_path = Path(out) if out \
        else Path(f"video_{_time.strftime('%Y%m%d_%H%M%S')}.mp4")
    kwargs = {k: v for k, v in
              (("negative_prompt", negative), ("width", width),
               ("height", height), ("seed", seed), ("steps", steps),
               ("cfg", cfg)) if v is not None}
    _is_privacy = effective_mode("server") == SessionMode.PRIVACY
    _write_sidecar = not _is_privacy

    def _gen_video():
        return generate_video(
            prompt, out_path,
            seconds=duration,
            fps=fps,
            api_url=api_url,
            input_image=Path(input_image) if input_image else None,
            on_progress=lambda t: console.print(f"  [dim]{t}[/dim]"),
            write_sidecar=_write_sidecar,
            delete_outputs=_is_privacy,
            **kwargs,
        )

    ok, message = _generate_or_abort(api_url, _gen_video)
    if not ok:
        ok, message = _maybe_apply_func_shim_and_retry(
            message, api_url, lambda: _generate_or_abort(api_url, _gen_video))
    console.print(f"[{'green' if ok else 'red'}]{message}[/{'green' if ok else 'red'}]")
    if not ok:
        sys.exit(1)
    _offer_open(out_path)
