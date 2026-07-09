# SPDX-License-Identifier: AGPL-3.0-or-later
import json
import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.panel import Panel

from ..config import load_config
from ..model_manager import get_model_info
from ._core import console, main, _complete_model_name


def _attach_fallback_note(no_server: bool, attach_error: Optional[BaseException]) -> Optional[str]:
    """CLI-1: the note to print when `localm run` is about to load the model in
    THIS process instead of attaching to a background server, so the fallback is
    never silent. None when the user opted out with --no-server (stay quiet)."""
    if no_server:
        return None
    if attach_error is not None:
        return (f"Could not attach to a localm server ({attach_error}); loading "
                "the model in this process.")
    return ("No localm server is serving this directory; loading the model in "
            "this process. Start one with `localm serve` so clients share a "
            "single load.")




# ------------------------------------------------------------------ #
#  run                                                                 #
# ------------------------------------------------------------------ #

@main.command()
@click.argument("model", shell_complete=_complete_model_name)
@click.option("-p", "--prompt",       default=None,  help="Single prompt (non-interactive).")
@click.option("-s", "--system",       default=None,  help="System prompt.")
@click.option("-m", "--max-tokens",   default=None,  type=int,   help="Max tokens to generate.")
@click.option("-t", "--temperature",  default=None,  type=float, help="Sampling temperature.")
@click.option("-c", "--ctx",          default=None,  type=int,   help="Context window (GGUF only).")
@click.option("-g", "--gpu-layers",   default=None,  type=click.IntRange(0, 1000),   help="GPU layers (GGUF only, 99=all).")
@click.option("--mmproj",             default=None,
              help="Path to a multimodal projector (mmproj) GGUF, enabling vision "
                   "on a vision-capable GGUF model. Pair with --image to attach "
                   "pictures.")
@click.option("--device",             default=None,  help="HF device override (cuda / cpu).")
@click.option("--image", "images",   multiple=True, type=click.Path(exists=True),
              help="Local image file to include (repeat for multiple). Use with -p.")
@click.option("--debug", is_flag=True,
              help="Write a debug log (~/.localm/logs/), capture native llama.cpp "
                   "stderr, and record raw model output (with markers) in the log.")
@click.option("--mode", default=None,
              type=click.Choice(["privacy", "log", "full"], case_sensitive=False),
              help="Session persistence [default: config 'chat_mode'/'mode', else privacy]. "
                   "privacy = nothing saved automatically; "
                   "log = JSONL audit trail to ~/.localm/sessions/; "
                   "full = log + markdown transcript.")
@click.option("--no-server", is_flag=True,
              help="Load the model in THIS process instead of attaching to a localm "
                   "server already serving this directory (default: attach, like the "
                   "GUI, so you do not load a second copy).")
def run(model, prompt, system, max_tokens, temperature, ctx, gpu_layers,
        mmproj, device, images, debug, mode, no_server):
    """Run a model - interactive chat or single prompt.

    \b
    MODEL can be a registered name OR a direct path:
      localm run gemma4-12b
      localm run D:\\models\\llama3.gguf
      localm run D:\\hf-models\\gemma-3-4b-it

    \b
    Image input (multimodal models):
      localm run gemma4-12b -p "What is in this photo?" --image photo.jpg
      localm run gemma4-12b -p "Compare" --image a.png --image b.png

    \b
    In interactive mode, attach images with the /image command:
      /image C:\\photos\\cat.jpg

    \b
    Pipe a prompt from stdin:
      echo "Explain RDNA2" | localm run qwen2.5-7b
    """
    if debug:
        from ..debuglog import enable_debug
        console.print(f"[yellow]debug log:[/yellow] {enable_debug()}")

    from ..audit import MODE_ENV_VAR, SessionMode, effective_mode
    if mode:
        os.environ[MODE_ENV_VAR] = mode.lower()
    session_mode = effective_mode("chat")
    if session_mode == SessionMode.PRIVACY:
        from ..readline_privacy import suppress_readline_history
        suppress_readline_history()
        if debug:
            console.print(
                "[yellow]⚠  privacy mode + --debug:[/yellow] the debug log "
                "records requests and raw model output - delete it after "
                "analysis if that matters.")
    else:
        console.print(f"[dim]session mode: {session_mode.value} "
                      f"(audit trail in ~/.localm/sessions/)[/dim]")

    # H3 thin-client: ATTACH to the localm server already serving this directory
    # (route chat through its /v1 over HTTP) instead of loading a SECOND in-process
    # copy of the model, mirroring `localm gui`. Default is to attach when a verified
    # server exists; --no-server forces an in-process load.
    engine = None
    attach_error: Optional[BaseException] = None
    if not no_server:
        from .. import instances
        from ..config import home_dir
        try:
            target = instances.attach_target(
                home_dir(), instances.resolve_root_dir())
        except Exception as e:
            # CLI-1: do not swallow the reason - log it (debug) and remember it so
            # the in-process fallback below can say WHY it did not attach.
            from localm.debuglog import logger as _dbg
            _dbg.exception("attach_target failed; falling back to in-process load")
            attach_error = e
            target = None
        if target:
            from ..inference.http_engine import HttpEngine, remote_model_status
            state, active = remote_model_status(
                target["base_url"], target.get("token"))
            if active and active != model:
                console.print(
                    f"[yellow]Note:[/yellow] attaching to the localm server already "
                    f"running for this directory; it serves [bold]{active}[/bold], so "
                    f"the requested [bold]{model}[/bold] is not loaded. Use "
                    f"[bold]--no-server[/bold] to run it in a separate process.")
            elif state == "empty":
                console.print(
                    "[yellow]Note:[/yellow] a localm server is running for this "
                    "directory but has no model loaded; load one (its GUI / API) or "
                    "use [bold]--no-server[/bold] to run in a separate process.")
            # state == "unknown": the server answered our attach but we could not
            # read /v1/models (it needs the models scope, so a chat-scoped attach
            # token gets a 403). Do NOT claim it has no model - it very likely does;
            # attach quietly and let the reply come from whatever it serves.
            engine = HttpEngine(
                target["base_url"], token=target.get("token"),
                model=active or model, display_name=active or model)
            console.print(
                f"[dim]connected to the localm server at {target['base_url']} "
                f"(no second model load)[/dim]")

    if engine is None and not no_server:
        console.print("[dim]No server running. Starting one in the background...[/dim]")
        import subprocess
        import time
        
        cmd = [sys.executable, "-m", "localm", "gui", "--no-browser", "--api-mode"]
        # Use no_model if no model was provided, else pass it
        if not model:
            cmd.append("--no-model")
        else:
            cmd.append(model)
        
        if ctx is not None: cmd.extend(["-c", str(ctx)])
        if gpu_layers is not None: cmd.extend(["-g", str(gpu_layers)])
        if mmproj: cmd.extend(["--mmproj", mmproj])
        if device: cmd.extend(["--device", device])
        
        kwargs = {}
        env = os.environ.copy()
        env["LOCALM_OWN_CONSOLE"] = "1"
        kwargs["env"] = env
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
        else:
            kwargs["start_new_session"] = True
            
        try:
            subprocess.Popen(cmd, **kwargs)
            # Poll for the server to come up
            for _ in range(40):
                time.sleep(0.5)
                target = instances.attach_target(home_dir(), instances.resolve_root_dir())
                if target:
                    from ..inference.http_engine import HttpEngine
                    engine = HttpEngine(
                        target["base_url"], token=target.get("token"),
                        model=model, display_name=model)
                    console.print(f"[dim]connected to newly started server at {target['base_url']}[/dim]")
                    break
        except Exception as e:
            attach_error = e

    if engine is None:
        _note = _attach_fallback_note(no_server, attach_error)
        if _note:
            console.print(f"[dim]{_note}[/dim]")
        info = get_model_info(model)
        if info is None:
            console.print(f"[red]Model not found:[/red] {model}")
            console.print("  [dim]localm list[/dim]              - downloaded models")
            console.print("  [dim]localm models[/dim]            - GGUF shortcuts")
            console.print("  [dim]localm pull owner/repo[/dim]   - download HF model")
            console.print("  [dim]localm pull name[/dim]         - download GGUF shortcut")
            console.print("  [dim]localm add <path>[/dim]        - register local file/dir")
            console.print("  [dim]localm run /full/path[/dim]    - use path directly")
            sys.exit(1)

        model_path, _display_hint = info

        from ..inference.engine import Engine
        from ..model_manager import load_registry as _reg

        # Priority: registered alias > Ollama manifest hint > engine auto-derive
        if model in _reg():
            display_name = model
        else:
            display_name = _display_hint  # None or Ollama suggested name

        engine = Engine(
            str(model_path),
            mmproj_path=mmproj,
            n_ctx=ctx,
            n_gpu_layers=gpu_layers,
            device=device,
            display_name=display_name,
        )

    cfg = load_config()
    gen_opts = {
        "max_tokens":     max_tokens  if max_tokens  is not None else cfg["max_tokens"],
        "temperature":    temperature if temperature is not None else cfg["temperature"],
        "top_p":          cfg["top_p"],
        "top_k":          cfg["top_k"],
        "repeat_penalty": cfg["repeat_penalty"],
    }

    # Accept piped stdin as the prompt
    if not sys.stdin.isatty() and prompt is None:
        prompt = sys.stdin.read().strip()

    from ..audit import make_audit_log, make_transcript
    audit = make_audit_log(session_mode, label="chat")
    transcript = make_transcript(session_mode, label="chat")

    try:
        with engine:
            if prompt is not None:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append(_build_user_message(prompt, list(images)))
                audit.user(prompt)
                response = _stream_once(engine, messages, **gen_opts)
                audit.llm(response)
                if transcript:
                    transcript.exchange(prompt, response)
            else:
                _interactive(engine, system, gen_opts,
                             audit=audit, transcript=transcript)
    finally:
        audit.close()




def _build_user_message(text: str, image_paths: list) -> dict:
    """Build a user message dict, embedding local images as base64 data-URIs."""
    if not image_paths:
        return {"role": "user", "content": text}

    parts: list = []
    for path in image_paths:
        parts.append({
            "type": "image_url",
            "image_url": {"url": _file_to_data_uri(path)},
        })
    if text:
        parts.append({"type": "text", "text": text})
    return {"role": "user", "content": parts}




def _file_to_data_uri(path: str) -> str:
    """Read a local image file and return a base64 data-URI."""
    import base64
    import mimetypes
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"




class _ThinkPrinter:
    """Stream a reply to stdout, dimming the model's ``<think>`` reasoning so it
    reads as an aside rather than raw ``<think>`` tags inline with the answer
    (H4). The full raw text (tags included) is still returned by the caller for
    the audit/transcript, which separate it themselves.

    R31 (CLI half): streaming here is APPEND-ONLY plain ``print`` plus SGR styling
    (dim/colour) - no alternate screen, no cursor repositioning, no spinner/Live
    region. So the terminal emulator owns scrolling: a user who scrolls up to
    re-read mid-stream is left alone (output keeps appending below), which is the
    CLI's native equivalent of the GUI's ``chat.stick`` autoscroll latch. The GUI
    needed that latch only because its JS re-pinned the viewport to the bottom on
    every token; we deliberately never do the terminal analogue of that here.
    Guarded by tests/test_cli_stream_scroll.py - do not wrap streaming in a Live
    region / alt-screen / ``\\r``-redraw, which would fight the user's scroll."""

    def __init__(self) -> None:
        import sys as _sys
        from localm.inference.textnorm import ThinkSplitter
        tty = _sys.stdout.isatty()
        self._dim, self._reset = ("\033[2m", "\033[22m") if tty else ("", "")
        self._think = ThinkSplitter()

    def _emit(self, content: str, reasoning: str) -> None:
        if reasoning:
            print(f"{self._dim}{reasoning}{self._reset}", end="", flush=True)
        if content:
            print(content, end="", flush=True)

    def feed(self, token: str) -> None:
        self._emit(*self._think.feed(token))

    def flush(self) -> None:
        self._emit(*self._think.flush())




def _stream_once(engine, messages: list, **kwargs) -> str:
    """Stream response to stdout, print tok/s on completion, and return the full text."""
    import time as _time
    from localm.inference.backends.base import UnsupportedInputError
    parts: list[str] = []
    printer = _ThinkPrinter()
    t0 = _time.monotonic()
    try:
        for token in engine.chat_stream(messages, **kwargs):
            parts.append(token)
            printer.feed(token)
        printer.flush()
    except UnsupportedInputError:
        # Capability-aware guidance instead of a flat "can't do that": name a
        # vision model this install has, or how to get one.
        from localm.model_manager import vision_input_guidance
        console.print(f"\n[yellow]{vision_input_guidance()}[/yellow]")
        return ""
    except RuntimeError as e:
        # An attached server returned an error (no model loaded, unreachable, ...).
        # Surface it cleanly instead of a traceback (the interactive path already
        # catches Exception; this is the single-prompt path).
        console.print(f"\n[red]{e}[/red]")
        return ""
    elapsed = _time.monotonic() - t0
    print()
    full = "".join(parts)
    if elapsed > 0.5 and full:
        n_tokens = engine.count_tokens(full)
        console.print(
            f"[dim]{n_tokens} tokens  {n_tokens / elapsed:.1f} tok/s  "
            f"({elapsed:.1f}s)[/dim]"
        )
    return full




def _interactive(engine, system_prompt: Optional[str], gen_opts: dict,
                 audit=None, transcript=None) -> None:  # noqa: C901
    console.print(Panel(
        f"[bold cyan]localm[/bold cyan] - {engine.display_name}\n"
        "[dim]Ctrl+C or [bold]/exit[/bold] to quit  ·  "
        "[bold]/clear[/bold] history  ·  [bold]/image <path>[/bold] attach image  ·  "
        "[bold]/help[/bold][/dim]",
        border_style="dim cyan",
        padding=(0, 1),
    ))

    messages: list = []
    pending_images: list = []   # image paths queued for the next user message

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        console.print(f"[dim]system: {system_prompt}[/dim]\n")

    while True:
        img_hint = f" [dim][{len(pending_images)} image(s) queued][/dim]" if pending_images else ""
        try:
            user_input = console.input(f"\n[bold green]You[/bold green]{img_hint}: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not user_input:
            continue

        # Catch bare-word exits typed without the leading slash
        if user_input.lower() in ("exit", "quit", "q", "bye"):
            console.print("[dim]Bye.[/dim]")
            break

        if user_input.startswith("/"):
            stop = _handle_command(user_input, messages, gen_opts, pending_images,
                                   engine=engine)
            if stop:
                break
            continue

        msg = _build_user_message(user_input, pending_images)
        pending_images.clear()
        messages.append(msg)
        if audit:
            audit.user(user_input)

        # Seamless compaction: summarise older turns before the history
        # collides with the context ceiling. Never fails - falls back to a
        # visible hard trim when summarisation is unavailable.
        from ..inference.compact import maybe_compact
        # Budget against the LOADED model's RESOLVED ceiling (VRAM-derived under
        # ctx_auto), not the static config n_ctx_max: the config value both
        # over-compacted a small-window model and under-protected a large one
        # (memory-audit 2026-07-02 F10). Fall back to the config only when the
        # engine cannot report a capacity (not loaded).
        limit = engine.context_capacity() or load_config().get("n_ctx_max", 16384) or 0
        compacted_msgs, did_compact = maybe_compact(
            messages,
            limit_tokens=limit,
            count_tokens=engine.count_tokens,
            generate=lambda m, max_tok: "".join(
                engine.chat_stream(m, max_tokens=max_tok, temperature=0.3)),
        )
        if did_compact:
            messages[:] = compacted_msgs
            console.print("[dim](older conversation summarised to free context)[/dim]")

        console.print("\n[bold blue]Assistant[/bold blue]: ", end="")

        parts: list[str] = []
        printer = _ThinkPrinter()
        import time as _time
        t0 = _time.monotonic()
        try:
            for token in engine.chat_stream(messages, **gen_opts):
                parts.append(token)
                printer.feed(token)
            printer.flush()
        except KeyboardInterrupt:
            printer.flush()
            console.print("\n[dim](interrupted)[/dim]")
        except Exception as e:
            console.print(f"\n[red]Inference error: {e}[/red]")
            continue

        response = "".join(parts) or "(interrupted)"
        elapsed = _time.monotonic() - t0
        print()
        if elapsed > 0.5 and parts:
            n_tokens = engine.count_tokens(response)
            console.print(
                f"[dim]{n_tokens} tokens  {n_tokens / elapsed:.1f} tok/s  "
                f"({elapsed:.1f}s)[/dim]"
            )
        if response:
            messages.append({"role": "assistant", "content": response})
            if audit:
                audit.llm(response)
            if transcript:
                transcript.exchange(user_input, response)




# The REPL media-generation commands (/generate-image, /generate-music,
# /generate-video) all share one shape: ensure ComfyUI is reachable (auto-
# launching it from the configured comfy_launch_cmd/comfy_workdir if needed),
# unload the chat model to free VRAM, generate one file into <home>/<subdir>,
# then free ComfyUI's VRAM. Only the generator, output subdir/extension and the
# argument label differ, so the three paths are table-driven to stay identical.
# Each helper imports its generator from the `.comfy` submodule at call time so a
# test that patches e.g. localm.image_gen.comfy.generate_image is honoured.
def _media_generate_image():
    from ..image_gen.comfy import generate_image
    return generate_image


def _media_generate_music():
    from ..music_gen.comfy import generate_music
    return generate_music


def _media_generate_video():
    from ..video_gen.comfy import generate_video
    return generate_video


_MEDIA_REPL = {
    "generate-image": {
        "subdir": "gui_images", "ext": ".png", "arg": "prompt",
        "get_generate": _media_generate_image,
    },
    "generate-music": {
        "subdir": "gui_music", "ext": ".flac", "arg": "tags",
        "get_generate": _media_generate_music,
    },
    "generate-video": {
        "subdir": "gui_video", "ext": ".mp4", "arg": "prompt",
        "get_generate": _media_generate_video,
    },
}


def _cmd_generate_media(label: str, arg: str, engine, console, home_dir) -> None:
    """REPL /generate-image|/generate-music|/generate-video: generate one media
    file via the configured ComfyUI backend, unloading the chat model first to
    free VRAM. console + home_dir are passed in so a caller that resolved the
    (monkeypatchable) localm.cli.console / localm.cli.HOME_DIR is honoured here."""
    spec = _MEDIA_REPL[label]
    if engine is None:
        console.print(f"[dim]/{label} not available in this mode[/dim]")
        return
    if not arg:
        console.print(f"[dim]Usage: /{label} <{spec['arg']}>[/dim]")
        return
    from ..image_gen.comfy import default_api_url, ensure_comfy, free_comfy_vram
    api = default_api_url()
    # Auto-launch ComfyUI from the configured comfy_launch_cmd/comfy_workdir (the
    # GUI does this; the CLI used to just bail - H1). Only unload the chat model
    # once ComfyUI is actually available.
    ok, msg = ensure_comfy(
        api, on_progress=lambda t: console.print(f"[dim]{t}[/dim]"))
    if not ok:
        console.print(f"[yellow]{msg}[/yellow]")
        return
    import time as _t
    out_dir = home_dir / spec["subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{_t.strftime('%Y%m%d_%H%M%S')}_cli{spec['ext']}"
    console.print("[dim]Freeing VRAM (chat model unloads, "
                  "reloads on your next message)...[/dim]")
    engine.unload()
    from ..audit import SessionMode, effective_mode
    generate = spec["get_generate"]()
    is_privacy = effective_mode("chat") == SessionMode.PRIVACY
    ok, message = generate(
        arg, out,
        api_url=api,
        write_sidecar=not is_privacy,
        delete_outputs=is_privacy,
    )
    console.print(message)
    if ok:
        free_comfy_vram(api)


def _cmd_generate_image(cmd: str, arg: str, engine, console, home_dir) -> None:
    """REPL /generate-image (and the /imagine alias): generate one image via the
    configured ComfyUI backend. Thin wrapper over the shared media path that
    keeps the /imagine rename notice."""
    if cmd == "imagine":
        console.print("[dim]/imagine was renamed to /generate-image[/dim]")
    _cmd_generate_media("generate-image", arg, engine, console, home_dir)


def _cmd_save(arg: str, messages: list, console) -> None:
    """REPL /save: write the conversation to JSON, confined to the cwd."""
    target = arg or "chat.json"
    # Confine writes to the current working directory: a path like "../x.json" or
    # an absolute path elsewhere must not let the REPL write outside cwd.
    cwd = Path.cwd().resolve()
    try:
        resolved = (cwd / target).resolve()
    except (OSError, ValueError) as e:
        console.print(f"[red]Invalid save path: {e}[/red]")
        return
    if resolved == cwd or cwd not in resolved.parents:
        console.print(
            f"[red]Refusing to save outside the current directory:[/red] "
            f"{target}\n[dim]Use a path inside {cwd}[/dim]"
        )
        return
    _save_chat(messages, str(resolved))


def _handle_command(
    raw: str,
    messages: list,
    gen_opts: dict,
    pending_images: Optional[list] = None,
    engine=None,
) -> bool:
    """Handle a /command. Returns True if the session should exit."""
    # Resolve console + HOME_DIR from the package at call time so tests that
    # monkeypatch localm.cli.console / localm.cli.HOME_DIR affect this call site.
    from localm import cli as _cli
    console = _cli.console
    HOME_DIR = _cli.HOME_DIR
    parts = raw[1:].split(" ", 1)
    cmd  = parts[0].lower()
    arg  = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("exit", "quit", "q", "bye"):
        console.print("[dim]Bye.[/dim]")
        return True
    elif cmd in ("generate-image", "imagine"):
        _cmd_generate_image(cmd, arg, engine, console, HOME_DIR)
    elif cmd == "generate-music":
        _cmd_generate_media("generate-music", arg, engine, console, HOME_DIR)
    elif cmd == "generate-video":
        _cmd_generate_media("generate-video", arg, engine, console, HOME_DIR)
    elif cmd == "compact":
        if engine is None:
            console.print("[dim]/compact not available in this mode[/dim]")
        else:
            from ..inference.compact import compact_messages
            new_messages, changed = compact_messages(
                messages,
                generate=lambda m, max_tok: "".join(
                    engine.chat_stream(m, max_tokens=max_tok, temperature=0.3)),
            )
            if changed:
                messages[:] = new_messages
                console.print("[dim]Older conversation summarised.[/dim]")
            else:
                console.print("[dim]Nothing to compact yet.[/dim]")
    elif cmd == "clear":
        messages[:] = [m for m in messages if m["role"] == "system"]
        if pending_images is not None:
            pending_images.clear()
        console.print("[dim]Cleared.[/dim]")
    elif cmd == "image":
        if pending_images is None:
            console.print("[dim]/image not available in this mode[/dim]")
        elif not arg:
            console.print("[dim]Usage: /image <file path>[/dim]")
        else:
            p = Path(arg)
            if not p.exists():
                console.print(f"[red]File not found:[/red] {arg}")
            else:
                pending_images.append(str(p.resolve()))
                console.print(f"[dim]Queued {p.name} - will attach to your next message[/dim]")
    elif cmd == "images":
        if pending_images:
            for f in pending_images:
                console.print(f"[dim]  {f}[/dim]")
        else:
            console.print("[dim]No images queued.[/dim]")
    elif cmd == "system":
        if arg:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "system":
                    messages.pop(i)
            messages.insert(0, {"role": "system", "content": arg})
            console.print("[dim]System prompt updated.[/dim]")
        else:
            console.print("[dim]Usage: /system <text>[/dim]")
    elif cmd == "save":
        _cmd_save(arg, messages, console)
    elif cmd == "temp":
        try:
            requested = float(arg)
        except ValueError:
            console.print("[dim]Usage: /temp 0.7[/dim]")
        else:
            # Clamp to a sane sampling range; negative or huge temps are not
            # meaningful and would otherwise be stored verbatim.
            clamped = max(0.0, min(2.0, requested))
            gen_opts["temperature"] = clamped
            if clamped != requested:
                console.print(
                    f"[dim]temperature clamped to {clamped} (valid range 0..2)[/dim]"
                )
            else:
                console.print(f"[dim]temperature = {clamped}[/dim]")
    elif cmd == "tokens":
        try:
            requested = int(arg)
        except ValueError:
            console.print("[dim]Usage: /tokens 2048[/dim]")
        else:
            # max_tokens must be at least 1; cap absurd values so a typo cannot
            # request an effectively unbounded generation.
            MAX_TOKENS_CAP = 1_000_000
            clamped = max(1, min(MAX_TOKENS_CAP, requested))
            gen_opts["max_tokens"] = clamped
            if clamped != requested:
                console.print(
                    f"[dim]max_tokens clamped to {clamped} "
                    f"(valid range 1..{MAX_TOKENS_CAP})[/dim]"
                )
            else:
                console.print(f"[dim]max_tokens = {clamped}[/dim]")
    elif cmd == "help":
        console.print(
            "[dim]"
            "/exit                   quit\n"
            "/clear                  clear chat history\n"
            "/image <path>           queue a local image for the next message\n"
            "/images                 list queued images\n"
            "/system <text>          set system prompt\n"
            "/save [file]            save conversation to JSON\n"
            "/compact                summarise older turns to free context\n"
            "/generate-image <prompt> generate an image via ComfyUI FLUX\n"
            "/generate-music <tags>  generate music via ComfyUI ACE-Step\n"
            "/generate-video <prompt> generate a clip via ComfyUI Wan\n"
            "/temp <float>           sampling temperature\n"
            "/tokens <int>           max response tokens"
            "[/dim]"
        )
    else:
        hint = None
        from ..config import load_config
        if load_config().get("suggest_plugins", True):
            from ..plugins import catalog
            hint = catalog.suggestion(cmd)
        if hint:
            console.print(f"[yellow]{hint}[/yellow]")
        else:
            console.print(f"[dim]Unknown: /{cmd} -- try /help[/dim]")
    return False




def _save_chat(messages: list, filepath: str) -> None:
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        console.print(f"[green]✓[/green] Saved: {filepath}")
    except Exception as e:
        console.print(f"[red]Save failed: {e}[/red]")
