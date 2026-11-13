# SPDX-License-Identifier: AGPL-3.0-or-later
import sys
from pathlib import Path

import click

from ._core import console, main


def _media_plugin(plugin: str, submodule: str):
    """The media plugin's *submodule*, or exit 1 with a message when the plugin
    is not installed and enabled."""
    from rich.markup import escape

    from ..plugins.lookup import PluginUnavailable, plugin_api
    try:
        return plugin_api(plugin, submodule)
    except PluginUnavailable as e:
        console.print(f"[yellow]{escape(str(e))}[/yellow]")
        sys.exit(1)


def _open_file(path: Path) -> None:
    """Open *path* with the OS default application (best-effort, never fatal)."""
    import os as _os
    import subprocess as _sp

    from rich.markup import escape
    try:
        if sys.platform == "win32":
            _os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            _sp.Popen(["open", str(path)])
        else:
            _sp.Popen(["xdg-open", str(path)])
    except Exception as e:
        console.print(f"[dim]Could not open it automatically: {escape(str(e))}[/dim]")




def _is_interactive() -> bool:
    """True when stdin is a terminal a person can answer at: not piped, not the
    null device, not scripted."""
    from ..bugreport import stdin_is_interactive
    return stdin_is_interactive()


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
    from ..media.comfy_client import (
        enable_func_shim_once,
        is_known_comfy_func_regression,
        restart_comfy,
        spawned_pid,
    )
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
    from ..media.comfy_client import (
        managed_comfy_setup_offer_message,
        mark_managed_comfy_setup_offered,
        should_offer_managed_comfy_setup,
    )
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
    from rich.markup import escape

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
                          f"({escape(str(e))}). It may still be rendering and holding VRAM.")
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


def _via_backend(media_backend, s: dict, prompt: str, out_path: Path, what: str,
                 **gen_kwargs) -> None:
    """``localm image`` / ``localm music`` / ``localm video`` through the
    plugin's backend facade (``media_config.generate_media``), whichever backend
    is selected: configuration notes, the backend's request refusal,
    availability, generation with progress lines, then the backend's VRAM is
    released. A backend that needs a ComfyUI server also gets the Ctrl-C abort
    of the running render and the offer to apply the ``__func__`` fix. *what*
    names the worker in the interrupt message; *gen_kwargs* go to ``generate``
    together with the privacy, sidecar and VRAM-swap arguments set here. Exits 1
    on failure."""
    import os

    from rich.markup import escape

    from ..audit import SessionMode, effective_mode
    from ..media.comfy_client import localm_v1_base
    from ..plugins import media_config

    def say(text: str) -> None:
        console.print(f"[dim]{escape(str(text))}[/dim]")

    is_privacy = effective_mode("server") == SessionMode.PRIVACY
    localm_url = localm_v1_base(os.environ.get("LOCALM_URL"))

    def run():
        return media_config.generate_media(
            media_backend, s, prompt, out_path, on_progress=say,
            self_url=localm_url, write_sidecar=not is_privacy,
            swap=bool(localm_url), privacy=is_privacy,
            **gen_kwargs)

    try:
        if media_backend.supports(s, "NEEDS_COMFY_SERVER"):
            api_url = s["api_url"]
            ok, message, path = _generate_or_abort(api_url, run)
            if not ok:
                ok, message = _maybe_apply_func_shim_and_retry(
                    message, api_url, lambda: _generate_or_abort(api_url, run)[:2])
        else:
            ok, message, path = run()
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Interrupted - stopping the {what} worker...[/yellow]")
        media_backend.free_vram(s)
        raise
    color = "green" if ok else "red"
    console.print(f"[{color}]{escape(str(message))}[/{color}]")
    if not ok:
        sys.exit(1)
    _offer_open(path)


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
@click.option("--size", default=None, metavar="WIDTHxHEIGHT",
              help="Image size, e.g. 768x512 (multiples of 8, 64-2048 per side).")
@click.option("-o", "--out", default=None,
              help="Output .png path [default: ./image_<timestamp>.png]")
def image_cmd(prompt, negative, guidance, cfg, seed, input_image, denoise,
              lora_name, size, out):
    """Generate an image with the image plugin's backend: the built-in native
    one (stable-diffusion.cpp) or ComfyUI, as set in the image settings
    ('auto' uses ComfyUI when it is set up, else native).

    \b
    Examples:
      localm image "a red fox in snow, photographic"
      localm image "a lighthouse at dusk" --size 768x512
      localm image "make it look like sunset" --image photo.png --denoise 0.6

    The native backend installs its runtime on first use and needs an image
    model (it names the one to download when none is set up). ComfyUI must be
    running, or startable from comfy_launch_cmd. The CLI cannot show the image,
    so it is saved to --out (or ./image_<timestamp>.png) and, in an interactive
    terminal, you are offered to open it.
    """
    import time as _time

    from rich.markup import escape

    from ..config import load_config

    width = height = None
    if size:
        from fastapi import HTTPException

        parse_image_size = _media_plugin("image", "plug").parse_image_size
        try:
            parsed = parse_image_size(size)
        except HTTPException as e:
            console.print(f"[red]{escape(str(e.detail))}[/red]")
            sys.exit(2)
        if parsed:
            width, height = parsed

    out_path = Path(out) if out \
        else Path(f"image_{_time.strftime('%Y%m%d_%H%M%S')}.png")
    _cfg = load_config()
    image_backend = _media_plugin("image", "backend")
    s = image_backend.prepare_for_job(image_backend.settings(_cfg), _cfg)
    kwargs = {k: v for k, v in (
        ("negative_prompt", negative), ("guidance", guidance), ("cfg", cfg),
        ("seed", seed), ("denoise", denoise), ("lora_name", lora_name),
    ) if v is not None}
    if input_image:
        kwargs["input_image"] = Path(input_image)
    if width is not None:
        kwargs["width"], kwargs["height"] = width, height
    _via_backend(image_backend, s, prompt, out_path, "image", **kwargs)


@main.command("music")
@click.argument("tags")
@click.option("--lyrics", type=click.Path(exists=True), default=None,
              help="Lyrics file ([verse]/[chorus] markers supported); "
                   "omit for an instrumental.")
@click.option("-d", "--duration", default=120.0, show_default=True,
              help="Track length in seconds - arbitrary.")
@click.option("-o", "--out", default=None,
              help="Output path [default: ./music_<timestamp>.wav with the native backend, "
                   ".flac with ComfyUI]")
@click.option("--seed", type=int, default=None, help="Reproducible seed.")
@click.option("--steps", type=int, default=None,
              help="Sampler steps (default 50 with ComfyUI, 8 natively).")
@click.option("--cfg", type=float, default=None, help="Guidance (default 5.0 with ComfyUI).")
def music_cmd(tags, lyrics, duration, out, seed, steps, cfg):
    """Generate a music track with ACE-Step, through the music plugin's backend:
    the built-in native one (ACE-Step 1.5 on KoboldCpp) or ComfyUI, as set in
    the music settings ('auto' uses ComfyUI when it is set up, else native).

    \b
    Examples:
      localm music "synthwave, 80s, 120 bpm, dreamy"
      localm music "folk ballad, acoustic guitar" --lyrics song.txt -d 180

    With ComfyUI, ComfyUI must be running (or start it via the GUI, which can
    auto-launch it when comfy_launch_cmd is configured).
    """
    import time as _time

    from rich.markup import escape

    from ..config import load_config
    music_backend = _media_plugin("music", "backend")
    lyr = Path(lyrics).read_text(encoding="utf-8") if lyrics else None
    kwargs = {k: v for k, v in
              (("seed", seed), ("steps", steps), ("cfg", cfg)) if v is not None}
    _cfg = load_config()
    _s = music_backend.prepare_for_job(music_backend.settings(_cfg), _cfg)
    declared = getattr(music_backend._impl(_s), "OUTPUT_SUFFIX", None)
    suffix = declared or ".flac"
    out_path = Path(out) if out \
        else Path(f"music_{_time.strftime('%Y%m%d_%H%M%S')}{suffix}")
    if declared and out_path.suffix.lower() != declared:
        console.print(f"[red]The {escape(str(_s.get('backend')))} music backend writes "
                      f"{declared} files; give an output path ending in {declared}.[/red]")
        sys.exit(2)
    _via_backend(music_backend, _s, tags, out_path, "music", lyrics=lyr,
                 duration_seconds=duration, **kwargs)


@main.command("video")
@click.argument("prompt")
@click.option("--negative", default=None,
              help="Negative prompt (default suppresses blur/watermarks).")
@click.option("-d", "--duration", default=5.0, show_default=True,
              help="Clip length in seconds (snapped to Wan's 4k+1 frame "
                   "rule; ~5s is the model's native length).")
@click.option("--fps", default=24, show_default=True, help="Frame rate.")
@click.option("--width", type=int, default=None,
              help="Width (multiple of 16; default the model's native resolution: "
                   "1280 for ComfyUI Wan 2.2, 832 for the native Wan2.1 1.3B; "
                   "quality collapses well below it).")
@click.option("--height", type=int, default=None,
              help="Height (multiple of 16; default 704 for ComfyUI, 480 native).")
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
    """Generate a short video clip with the video plugin's backend: the built-in
    native one (stable-diffusion.cpp, Wan2.1 1.3B by default) or the ComfyUI
    Wan 2.2 workflow, as set in the video settings ('auto' uses ComfyUI when
    it is set up, else native).

    \b
    Examples:
      localm video "a red fox running through snow, tracking shot"
      localm video "waves rolling in at dusk" --image beach.png -d 5

    On ComfyUI it must be running (or start it via the GUI, which can
    auto-launch it when comfy_launch_cmd is configured). Video is the slowest
    generator - expect many minutes per clip; see docs/video.md for model setup.
    """
    import time as _time

    from ..config import load_config

    out_path = Path(out) if out \
        else Path(f"video_{_time.strftime('%Y%m%d_%H%M%S')}.mp4")
    kwargs = {k: v for k, v in
              (("negative_prompt", negative), ("width", width),
               ("height", height), ("seed", seed), ("steps", steps),
               ("cfg", cfg)) if v is not None}
    _cfg = load_config()
    video_backend = _media_plugin("video", "backend")
    s = video_backend.prepare_for_job(video_backend.settings(_cfg), _cfg)
    _via_backend(video_backend, s, prompt, out_path, "video",
                 seconds=duration, fps=fps,
                 input_image=Path(input_image) if input_image else None, **kwargs)
