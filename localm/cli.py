# SPDX-License-Identifier: AGPL-3.0-or-later
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Force UTF-8 output on Windows so Rich's Unicode markup doesn't crash
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import click
from rich.console import Console
from rich.panel import Panel

from .config import HOME_DIR, find_binary_dir, load_config, save_config
from .model_manager import (
    add_local, get_model_info, list_models, pull_model,
    remove_model, show_shortcuts, sync_models_dir,
)

console = Console()


_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _exposed_bind_warning(host: str) -> Optional[str]:
    """
    Warning text when binding beyond loopback without an API key set -
    that combination serves an unauthenticated LLM API to the whole network.
    Returns None when the configuration is safe.
    """
    from localm.auth import any_key_configured
    if host in _LOOPBACK_HOSTS or any_key_configured():
        return None
    return (
        f"⚠ Binding to {host} WITHOUT authentication - anyone on the network "
        f"can use this server, unload your model, and read every response.\n"
        f"  Set an API key first:  $env:LOCALM_API_KEY = \"<secret>\"  "
        f"(clients send it as a Bearer token)"
    )


def _resolve_tls(host, *, no_tls, tls_cert, tls_key):
    """Decide TLS for a bind and return ``(ssl_certfile, ssl_keyfile)`` - or
    ``(None, None)`` for plain HTTP.

    Built-in TLS (NET-1): on any non-loopback bind localm mints its own local-CA
    certificate so HTTPS works out of the box (the API key and all traffic are
    encrypted). ``--no-tls`` forces plain HTTP (the key would cross the network
    in cleartext); ``--tls-cert``/``--tls-key`` override with a user-supplied
    pair on any bind. Raises on a half-specified override; the cert-generation
    path may raise on a broken crypto stack and the caller refuses to fall back
    to cleartext.
    """
    if tls_cert or tls_key:
        if not (tls_cert and tls_key):
            raise click.UsageError(
                "--tls-cert and --tls-key must be provided together.")
        return str(tls_cert), str(tls_key)
    if no_tls or host in _LOOPBACK_HOSTS:
        return None, None
    from localm import tls
    from localm.config import home_dir
    return tls.ensure_cert(home_dir(), hostnames=[host])


def _setup_tls_or_exit(host, *, no_tls, tls_cert, tls_key):
    """Resolve TLS for *host*, or exit(2) rather than silently serve a network
    bind in cleartext. Returns ``(ssl_certfile, ssl_keyfile)`` (cert is None for
    plain HTTP)."""
    try:
        return _resolve_tls(host, no_tls=no_tls, tls_cert=tls_cert, tls_key=tls_key)
    except click.UsageError:
        raise
    except Exception as e:
        console.print(f"[bold red]Could not set up built-in TLS: {e}[/bold red]")
        console.print(
            "[bold red]Refusing to serve a network bind in cleartext. Re-run with "
            "--no-tls to force HTTP (the API key would cross the network "
            "unencrypted), or pass --tls-cert/--tls-key.[/bold red]")
        sys.exit(2)


def _complete_model_name(ctx, param, incomplete):
    """Shell-completion callback: registered model names matching the prefix."""
    try:
        from .config import load_registry as _lr
        return sorted(n for n in _lr() if n.startswith(incomplete))
    except Exception:
        return []


class _GracefulGroup(click.Group):
    """Single, cross-cutting failure handler for the whole CLI.

    Every subcommand (run, gui, serve, coder, setup-llama, ...) is invoked
    through this group, so an unexpected crash anywhere is caught in ONE place:
    we say "sorry, X went wrong because Y" and offer a prefilled, editable bug
    report. A command that hits a known problem just raises bugreport.LocalmError
    with a good summary/reason; it does not know about reporting itself.

    User errors (ClickException / bad usage), Ctrl+C, and clean exits pass
    through untouched - those are not bugs."""

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except (SystemExit, KeyboardInterrupt, click.exceptions.Abort,
                click.exceptions.Exit, click.ClickException):
            raise
        except Exception as e:  # an actual, unexpected failure
            # Reporting must never itself crash the handler (that would turn a
            # caught bug into the hard crash we are trying to avoid), so guard it.
            try:
                from localm import bugreport
                interactive = bool(getattr(sys.stdin, "isatty", lambda: False)())
                if isinstance(e, bugreport.LocalmError):
                    bugreport.report_failure(
                        summary=e.summary, reason=e.reason,
                        error=e.__cause__ or e, context=e.context,
                        interactive=interactive)
                else:
                    bugreport.report_failure(
                        summary="localm hit an unexpected error",
                        reason=str(e), error=e,
                        context={"command": getattr(ctx, "invoked_subcommand", None)},
                        interactive=interactive)
            except Exception:
                console.print(f"[red]localm failed:[/red] {e}")
            raise SystemExit(1)


@click.group(cls=_GracefulGroup, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option("0.1.0", prog_name="localm")
def main() -> None:
    """Run local LLMs offline - HuggingFace and GGUF models, AMD/NVIDIA/CPU."""
    # Install the process-wide graceful-failure net so a crash anywhere - a
    # background thread (preload, jobs, coder), or any uncaught main-thread
    # error - becomes a "sorry X for Y" + bug-report offer, not a hard crash.
    # The _GracefulGroup above still handles per-command errors with nicer
    # context; this covers everything the group does not wrap.
    from localm import bugreport
    bugreport.install_global_handlers()


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
              help="Multimodal projection GGUF path (reserved; GGUF vision is "
                   "not yet implemented - the backend is text-only).")
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
def run(model, prompt, system, max_tokens, temperature, ctx, gpu_layers,
        mmproj, device, images, debug, mode):
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
        from .debuglog import enable_debug
        console.print(f"[yellow]debug log:[/yellow] {enable_debug()}")

    from .audit import MODE_ENV_VAR, SessionMode, effective_mode
    if mode:
        os.environ[MODE_ENV_VAR] = mode.lower()
    session_mode = effective_mode("chat")
    if session_mode == SessionMode.PRIVACY:
        from .readline_privacy import suppress_readline_history
        suppress_readline_history()
        if debug:
            console.print(
                "[yellow]⚠  privacy mode + --debug:[/yellow] the debug log "
                "records requests and raw model output - delete it after "
                "analysis if that matters.")
    else:
        console.print(f"[dim]session mode: {session_mode.value} "
                      f"(audit trail in ~/.localm/sessions/)[/dim]")

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

    from .inference.engine import Engine
    from .model_manager import load_registry as _reg

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

    from .audit import make_audit_log, make_transcript
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
    the audit/transcript, which separate it themselves."""

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
        from .inference.compact import maybe_compact
        limit = load_config().get("n_ctx_max", 16384) or 0
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


def _handle_command(
    raw: str,
    messages: list,
    gen_opts: dict,
    pending_images: Optional[list] = None,
    engine=None,
) -> bool:
    """Handle a /command. Returns True if the session should exit."""
    parts = raw[1:].split(" ", 1)
    cmd  = parts[0].lower()
    arg  = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("exit", "quit", "q", "bye"):
        console.print("[dim]Bye.[/dim]")
        return True
    elif cmd in ("generate-image", "imagine"):
        if cmd == "imagine":
            console.print("[dim]/imagine was renamed to /generate-image[/dim]")
        if engine is None:
            console.print("[dim]/generate-image not available in this mode[/dim]")
        elif not arg:
            console.print("[dim]Usage: /generate-image <prompt>[/dim]")
        else:
            from .image_gen.comfy import (
                _comfy_alive, default_api_url, free_comfy_vram, generate_image)
            api = default_api_url()
            if not _comfy_alive(api):
                console.print(
                    f"[yellow]ComfyUI is not running at {api}.[/yellow] "
                    "Start it and retry, or set "
                    "[bold]localm config comfy_launch_cmd <path>[/bold]."
                )
            else:
                import time as _t
                out_dir = HOME_DIR / "gui_images"
                out_dir.mkdir(parents=True, exist_ok=True)
                out = out_dir / f"{_t.strftime('%Y%m%d_%H%M%S')}_cli.png"
                console.print("[dim]Freeing VRAM (chat model unloads, "
                              "reloads on your next message)…[/dim]")
                engine.unload()
                from .audit import SessionMode, effective_mode
                ok, message = generate_image(
                    arg, out,
                    api_url=api,
                    write_sidecar=effective_mode("chat") != SessionMode.PRIVACY,
                )
                console.print(message)
                if ok:
                    free_comfy_vram(api)
    elif cmd == "compact":
        if engine is None:
            console.print("[dim]/compact not available in this mode[/dim]")
        else:
            from .inference.compact import compact_messages
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
        target = arg or "chat.json"
        # Confine writes to the current working directory: a path like
        # "../x.json" or an absolute path elsewhere must not let the REPL
        # write outside cwd.
        cwd = Path.cwd().resolve()
        try:
            resolved = (cwd / target).resolve()
        except (OSError, ValueError) as e:
            console.print(f"[red]Invalid save path: {e}[/red]")
        else:
            if resolved == cwd or cwd not in resolved.parents:
                console.print(
                    f"[red]Refusing to save outside the current directory:[/red] "
                    f"{target}\n[dim]Use a path inside {cwd}[/dim]"
                )
            else:
                _save_chat(messages, str(resolved))
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
            "/temp <float>           sampling temperature\n"
            "/tokens <int>           max response tokens"
            "[/dim]"
        )
    else:
        hint = None
        from .config import load_config
        if load_config().get("suggest_plugins", True):
            from .plugins import catalog
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


# ------------------------------------------------------------------ #
#  serve                                                               #
# ------------------------------------------------------------------ #

@main.command()
@click.argument("model", shell_complete=_complete_model_name)
@click.option("-H", "--host",        default="127.0.0.1", help="Bind address (0.0.0.0 for LAN).")
@click.option("-p", "--port",        default=None,        type=click.IntRange(1, 65535),
              help="Port [default: config 'port' (8642); auto-bumps when busy].")
@click.option("-c", "--ctx",         default=None,        type=int)
@click.option("-g", "--gpu-layers",  default=None,        type=click.IntRange(0, 1000))
@click.option("--mmproj",            default=None)
@click.option("--device",            default=None)
@click.option("--no-tls", is_flag=True,
              help="Serve plain HTTP even on a network bind. Built-in TLS is on "
                   "by default past loopback; this disables it (the API key then "
                   "crosses the network in cleartext - only on a trusted LAN).")
@click.option("--tls-cert", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Use this certificate (PEM) instead of localm's built-in "
                   "local-CA cert. Requires --tls-key.")
@click.option("--tls-key", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Private key (PEM) for --tls-cert.")
@click.option("--insecure", is_flag=True,
              help="Allow binding past loopback WITHOUT LOCALM_API_KEY set. This "
                   "serves the API (and any enabled plugin routes) "
                   "unauthenticated to the whole network - only on a trusted, "
                   "isolated network.")
@click.option("--project", default=None, type=click.Path(file_okay=False),
              help="Project root that keys this instance [default: nearest "
                   ".git/.localcoder above the current directory].")
@click.option("--new", "force_new", is_flag=True,
              help="Start a fresh server even if one is already serving this "
                   "project (by default 'serve' reports the running one).")
@click.option("--isolated", is_flag=True,
              help="Start a private server invisible to discovery (test safety). "
                   "Implies --new.")
@click.option("--debug", is_flag=True,
              help="Write a debug log (~/.localm/logs/), capture native llama.cpp "
                   "stderr, and log requests.")
@click.option("--mode", default=None,
              type=click.Choice(["privacy", "log", "full"], case_sensitive=False),
              help="Session persistence [default: config 'mode', else privacy]. "
                   "privacy = nothing saved; log = JSONL audit of chat traffic; "
                   "full = log + markdown transcript.")
def serve(model, host, port, ctx, gpu_layers, mmproj, device, no_tls, tls_cert,
          tls_key, insecure, project, force_new, isolated, debug, mode):
    """Start an OpenAI-compatible inference server.

    \b
    POST http://<host>:<port>/v1/chat/completions
    GET  http://<host>:<port>/v1/models
    GET  http://<host>:<port>/health

    Compatible with any OpenAI client library.
    """
    # A click into this console window must not freeze the server
    # (Windows QuickEdit suspends output, and output blocks inference).
    from .winconsole import disable_quickedit
    disable_quickedit()

    if debug:
        from .debuglog import enable_debug
        console.print(f"[yellow]debug log:[/yellow] {enable_debug()}")

    from .audit import MODE_ENV_VAR, SessionMode, effective_mode
    if mode:
        os.environ[MODE_ENV_VAR] = mode.lower()
    session_mode = effective_mode("server")
    if session_mode != SessionMode.PRIVACY:
        console.print(f"[dim]session mode: {session_mode.value} "
                      f"(audit trail in ~/.localm/sessions/)[/dim]")
    elif debug:
        console.print(
            "[yellow]⚠  privacy mode + --debug:[/yellow] the debug log records "
            "requests and raw model output - delete it after analysis if that "
            "matters.")

    # Refuse a keyless network bind before any other work (fail fast, and match
    # `localm gui`, which already refuses): binding past loopback with no API key
    # serves the OpenAI API - and any enabled plugin routes (e.g. the coder
    # agent's history) - unauthenticated to the whole network. --insecure
    # overrides for a trusted isolated network. (Security review 2026-06-20: this
    # closes a serve-vs-gui fail-open asymmetry.)
    bind_warning = _exposed_bind_warning(host)
    if bind_warning and not insecure:
        console.print(f"[bold red]{bind_warning}[/bold red]")
        console.print(
            "[bold red]Refusing to start: binding past loopback without auth. "
            "Set $env:LOCALM_API_KEY first, or pass --insecure to override.[/bold red]")
        sys.exit(2)
    if bind_warning:
        console.print(f"[bold yellow]{bind_warning}[/bold yellow]")
        console.print("[bold yellow]  Proceeding anyway (--insecure set).[/bold yellow]")

    info = get_model_info(model)
    if info is None:
        console.print(f"[red]Model not found:[/red] {model}")
        sys.exit(1)

    model_path, _display_hint = info

    # Attach-or-spawn (H6 phase 4): if a localm is already serving this project,
    # do not spin a second server that double-loads the model - report it.
    from . import instances
    from .config import home_dir
    _root_dir = instances.resolve_root_dir(override=project)
    if not (force_new or isolated):
        _existing = instances.find_attachable(home_dir(), _root_dir)
        if _existing:
            _url = instances.attach_url(_existing)
            console.print(
                f"[bold]localm is already serving[/bold] [cyan]{_root_dir}[/cyan] "
                f"at [bold]{_url}v1[/bold] (pid {_existing.get('pid')}, "
                f"mode {_existing.get('mode')}).")
            console.print(
                "  [dim]Use that endpoint, or pass [/dim][cyan]--new[/cyan]"
                "[dim] to start a separate server.[/dim]")
            return

    from .inference.engine import Engine
    from .inference.http_server import serve as http_serve
    from .model_manager import load_registry as _reg

    display_name = model if model in _reg() else _display_hint

    from .config import pick_port
    requested = port
    port, was_busy = pick_port(requested, host="127.0.0.1" if host == "0.0.0.0" else host)
    if was_busy:
        console.print(
            f"[yellow]Port {requested if requested is not None else load_config().get('port', 8642)} "
            f"is in use - serving on {port} instead.[/yellow]"
        )

    # Built-in TLS: encrypt the bind out of the box past loopback (NET-1).
    ssl_certfile, ssl_keyfile = _setup_tls_or_exit(
        host, no_tls=no_tls, tls_cert=tls_cert, tls_key=tls_key)
    scheme = "https" if ssl_certfile else "http"

    engine = Engine(
        str(model_path),
        mmproj_path=mmproj,
        n_ctx=ctx,
        n_gpu_layers=gpu_layers,
        device=device,
        display_name=display_name,
    )

    engine.load()
    console.print(
        f"[green]✓[/green] Serving at "
        f"[bold]{scheme}://{host}:{port}/v1/chat/completions[/bold]"
    )
    if scheme == "https":
        console.print("[dim]Built-in TLS (self-signed via localm's local CA). "
                      "First connection from a device shows a one-time trust "
                      "warning; install the CA from /localm-ca.crt to remove it.[/dim]")
    console.print("[dim]Ctrl+C to stop[/dim]\n")

    try:
        http_serve(engine, host=host, port=port,
                   ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile,
                   project=project, isolated=isolated)
    except KeyboardInterrupt:
        pass
    finally:
        engine.unload()


# ------------------------------------------------------------------ #
#  benchmark                                                           #
# ------------------------------------------------------------------ #

@main.command()
@click.argument("model", shell_complete=_complete_model_name)
@click.option("-n", "--gen-tokens", default=128, show_default=True,
              help="Tokens to generate per run.")
@click.option("--prompts", default="64,512,2048", show_default=True,
              help="Comma-separated approximate prompt sizes in tokens.")
@click.option("-c", "--ctx",        default=None, type=int)
@click.option("-g", "--gpu-layers", default=None, type=click.IntRange(0, 1000))
def benchmark(model, gen_tokens, prompts, ctx, gpu_layers):
    """Measure TTFT and generation throughput at increasing prompt lengths.

    Runs a fixed prompt padded to each requested size, streams GEN_TOKENS
    tokens, and reports time to first token, tokens per second, and total
    time. Results depend on your hardware, quantisation, and context size.
    """
    import time as _time

    info = get_model_info(model)
    if info is None:
        console.print(f"[red]Model not found:[/red] {model}")
        sys.exit(1)
    model_path, _hint = info

    try:
        sizes = [int(s) for s in prompts.split(",") if s.strip()]
    except ValueError:
        console.print(f"[red]Invalid --prompts:[/red] {prompts}")
        sys.exit(1)

    from .inference.engine import Engine
    engine = Engine(str(model_path), n_ctx=ctx, n_gpu_layers=gpu_layers,
                    display_name=model)
    console.print(f"Loading [cyan]{model}[/cyan]…")
    engine.load()

    pad_block = (
        "The quick brown fox jumps over the lazy dog while the river keeps "
        "flowing through the quiet valley under a pale morning sky. "
    )
    question = "\n\nReply with a short story continuing the text above."

    rows = []
    try:
        for target in sizes:
            # Build a prompt of roughly `target` tokens using the real tokenizer
            padding = pad_block
            while engine.count_tokens(padding) < target:
                padding += pad_block
            prompt = padding + question
            prompt_tokens = engine.count_tokens(prompt)

            console.print(f"  prompt ≈ {prompt_tokens} tok … ", end="")
            start = _time.perf_counter()
            first_at = None
            generated = 0
            for token in engine.chat_stream(
                [{"role": "user", "content": prompt}],
                max_tokens=gen_tokens, temperature=0.0,
            ):
                if first_at is None:
                    first_at = _time.perf_counter()
                generated += 1
            elapsed = _time.perf_counter() - start

            ttft_ms = (first_at - start) * 1000 if first_at else float("nan")
            gen_time = elapsed - (first_at - start) if first_at else elapsed
            tps = generated / gen_time if gen_time > 0 and generated else 0.0
            console.print("done")
            rows.append((prompt_tokens, generated, ttft_ms, tps, elapsed))
    finally:
        engine.unload()

    from rich.table import Table
    table = Table(title=f"benchmark - {model}")
    table.add_column("prompt tok", justify="right")
    table.add_column("gen tok", justify="right")
    table.add_column("TTFT ms", justify="right")
    table.add_column("tok/s", justify="right")
    table.add_column("total s", justify="right")
    for prompt_tokens, generated, ttft_ms, tps, elapsed in rows:
        table.add_row(
            str(prompt_tokens), str(generated),
            f"{ttft_ms:.0f}", f"{tps:.1f}", f"{elapsed:.1f}",
        )
    console.print(table)
    console.print("[dim]TTFT includes prompt prefill; tok/s measures pure "
                  "generation after the first token.[/dim]")


# ------------------------------------------------------------------ #
#  Model management                                                    #
# ------------------------------------------------------------------ #

@main.command()
@click.argument("model_spec")
@click.option("-n", "--name", default=None, help="Alias for the downloaded model.")
@click.option("--sha256", default=None, metavar="HASH",
              help="Expected SHA256 hex digest (direct-URL and HF single-GGUF pulls; "
                   "refused for full-repo snapshots). Download is deleted on mismatch.")
@click.option("--redownload", is_flag=True,
              help="Download even when an identical model is already registered.")
def pull(model_spec, name, sha256, redownload):
    """Download a model from HuggingFace or a URL.

    \b
    Full HF model (transformers format, for multimodal / HF-native models):
      localm pull google/gemma-3-4b-it
      localm pull microsoft/Phi-4-mini-instruct

    \b
    Single GGUF file (quantized, lighter, works with GGUF backend):
      localm pull llama3.2-3b
      localm pull bartowski/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct-Q4_K_M.gguf

    \b
    Direct URL:
      localm pull https://example.com/model.gguf

    Models are stored in ~/.localm/models/ and registered automatically.
    """
    if not pull_model(model_spec, name, expected_sha256=sha256, redownload=redownload):
        sys.exit(1)


@main.command("search")
@click.argument("query", nargs=-1)
@click.option("-n", "--limit", default=10, show_default=True, help="Max repos.")
@click.option("--files", "list_files", is_flag=True,
              help="Treat QUERY as one repo id and list its GGUF files "
                   "with per-quant VRAM fit.")
def search_cmd(query, limit, list_files):
    """Search HuggingFace for GGUF models (empty query = most downloaded).

    \b
    Examples:
      localm search qwen2.5 7b instruct
      localm search bartowski/Qwen2.5-7B-Instruct-GGUF --files
    """
    from rich.console import Console
    from .discover import (DiscoverError, fit_label, hf_gguf_files,
                           hf_search, vram_info)
    console = Console()
    text = " ".join(query).strip()
    try:
        if list_files:
            if not text:
                console.print("[red]--files needs a repo id, e.g. "
                              "owner/repo[/red]")
                sys.exit(1)
            total = vram_info().get("total")
            if total:
                console.print(f"[dim]fit vs {total / 1024**3:.0f} GB total VRAM "
                              "(weights + ~1.5 GB overhead)[/dim]")
            for f in hf_gguf_files(text):
                fit = fit_label(f["size_bytes"], total)
                badge = {"fits": "[green]fits[/green]",
                         "tight": "[yellow]tight[/yellow]",
                         "too-big": "[red]too big[/red]"}.get(fit, "")
                parts = f" ({f['n_parts']} parts)" if f["n_parts"] > 1 else ""
                console.print(
                    f"  [cyan]{f['quant'] or '?':10}[/cyan] "
                    f"{f['size_bytes'] / 1024**3:6.1f} GB{parts}  {badge}  "
                    f"[dim]{f['file']}[/dim]")
            console.print(f"\n[dim]pull one:  localm pull {text}:<file>[/dim]")
        else:
            results = hf_search(text, limit=limit)
            if not results:
                console.print("[dim](no GGUF repos found)[/dim]")
                return
            for r in results:
                console.print(f"[cyan]{r['id']}[/cyan]  "
                              f"[dim]⬇ {r['downloads']:,}  ♥ {r['likes']:,}[/dim]")
            console.print("\n[dim]list quants:  localm search <repo> "
                          "--files[/dim]")
    except DiscoverError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


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

    from .audit import SessionMode, effective_mode
    from .image_gen.comfy import (_comfy_alive, default_api_url,
                                  free_comfy_vram, generate_image)

    api_url = default_api_url()
    if not _comfy_alive(api_url):
        console.print(
            f"[red]ComfyUI is not running at {api_url}.[/red] Start it and "
            "retry - or use the GUI's Images page, which can launch it "
            "automatically when comfy_launch_cmd is set.")
        sys.exit(1)

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
    from .audit import SessionMode, effective_mode
    from .image_gen.comfy import _comfy_alive, default_api_url
    from .music_gen import generate_music
    console = Console()

    api_url = default_api_url()
    if not _comfy_alive(api_url):
        console.print(
            f"[red]ComfyUI is not running at {api_url}.[/red] Start it and "
            "retry - or use the GUI's Music page, which can launch it "
            "automatically when comfy_launch_cmd is set.")
        sys.exit(1)

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
    from .audit import SessionMode, effective_mode
    from .image_gen.comfy import _comfy_alive, default_api_url
    from .video_gen import generate_video
    console = Console()

    api_url = default_api_url()
    if not _comfy_alive(api_url):
        console.print(
            f"[red]ComfyUI is not running at {api_url}.[/red] Start it and "
            "retry - or use the GUI's Video page, which can launch it "
            "automatically when comfy_launch_cmd is set.")
        sys.exit(1)

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


@main.command("list")
def list_cmd():
    """List registered models (auto-detecting changes in the models folder)."""
    result = sync_models_dir()
    if result.changed:
        bits = []
        if result.added:
            bits.append(f"{result.added} new")
        if result.flagged:
            bits.append(f"{result.flagged} missing")
        if result.restored:
            bits.append(f"{result.restored} restored")
        if result.pruned:
            bits.append(f"{result.pruned} pruned")
        console.print(f"[dim]Models folder synced: {', '.join(bits)}.[/dim]")
    if result.note:
        console.print(f"[yellow]{result.note}[/yellow]")
    list_models()


@main.command()
@click.argument("model", shell_complete=_complete_model_name)
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation.")
def rm(model, yes):
    """Remove a model from the registry (and delete the file if it's in ~/.localm).

    The confirmation prompt describes exactly what will happen (delete vs
    unregister-only). Disable it permanently with:
    localm config confirm_remove false
    """
    from .config import MODELS_DIR, load_registry
    from .model_manager import find_aliases_by_path

    cfg = load_config()
    if not yes and cfg.get("confirm_remove", True):
        reg = load_registry()
        if model in reg:
            path = Path(reg[model]["path"])
            others = [a for a in find_aliases_by_path(path, reg) if a != model]
            if others:
                detail = (f"unregisters the name only - file kept, "
                          f"still registered as: {', '.join(others)}")
            elif str(path).startswith(str(MODELS_DIR)) and path.exists():
                size = path.stat().st_size / 1024**3 if path.is_file() else None
                size_s = f" ({size:.1f} GB)" if size else ""
                detail = f"PERMANENTLY deletes {path}{size_s}"
            else:
                detail = "unregisters the name only (file is outside ~/.localm/models)"
            click.confirm(f"Remove '{model}'? This {detail}. Continue?", abort=True)
        else:
            click.confirm(f"Remove '{model}'?", abort=True)
    remove_model(model)


@main.command()
@click.argument("path")
@click.option("-n", "--name", default=None, help="Name to register the model as.")
@click.option("--no-hash", is_flag=True,
              help="Skip SHA256 computation (disables content-level duplicate detection).")
@click.option("--on-duplicate", default="ask",
              type=click.Choice(["ask", "alias", "copy", "move", "register", "skip"]),
              help="What to do when the model is already registered (default: ask).")
def add(path, name, no_hash, on_duplicate):
    """Register a local model file or HuggingFace directory.

    Duplicate detection is two-tier: the resolved path is checked first,
    then the file's SHA256 against digests stored in the registry.

    \b
    Examples:
      localm add C:\\models\\mymodel.gguf
      localm add D:\\models\\gemma.gguf --name gemma4-12b
      localm add D:\\models\\gemma.gguf -n g2 --on-duplicate alias
    """
    if not add_local(path, name, on_duplicate=on_duplicate, no_hash=no_hash):
        sys.exit(1)


@main.group("rag")
def rag_group():
    """Knowledge collections - chat with your documents (offline RAG).

    Index files or folders into named collections, then ground chat replies
    in them: pick the collection in the GUI's parameters drawer, or query
    from the terminal. Retrieval is BM25 (always available offline);
    embeddings are blended in when indexed via the GUI with a model that
    supports them. PDF needs the [rag] extra:  pip install "localm[rag]"
    """


@rag_group.command("list")
def rag_list():
    """List collections with document and chunk counts."""
    from rich.console import Console
    from .rag import Collection, collection_names
    console = Console()
    names = collection_names()
    if not names:
        console.print("[dim]No collections yet. "
                      "Create one:  localm rag add <name> <path>[/dim]")
        return
    for n in names:
        s = Collection(n).stats()
        retrieval = "hybrid" if s["has_vectors"] else "BM25"
        console.print(f"[cyan]{n}[/cyan]  {s['n_docs']} docs · "
                      f"{s['n_chunks']} chunks · {retrieval}")


@rag_group.command("add")
@click.argument("collection")
@click.argument("paths", nargs=-1, required=True)
def rag_add(collection, paths):
    """Index files/folders into COLLECTION (created if missing).

    Folders are indexed recursively (txt/md/pdf/docx/html/code). Unchanged
    files are skipped; changed ones are re-indexed. CLI indexing is
    lexical-only - index from the GUI to add embeddings when the active
    model supports them.

    \b
    Examples:
      localm rag add manuals D:\\docs\\printer-manual.pdf
      localm rag add project D:\\projects\\myapp
    """
    from rich.console import Console
    from .rag import Collection
    console = Console()
    try:
        coll = Collection(collection)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    coll.create()
    result = coll.add_paths(list(paths),
                            on_progress=lambda t: console.print(f"  [dim]{t}[/dim]"))
    console.print(f"[green]{result['added']} added, {result['updated']} updated, "
                  f"{result['skipped']} unchanged[/green] - "
                  f"{result['chunks']} chunks in '{collection}'")
    for f in result["failed"]:
        console.print(f"  [yellow]failed:[/yellow] {f['path']}: {f['error']}")
    if result["failed"]:
        sys.exit(1)


@rag_group.command("query")
@click.argument("collection")
@click.argument("text")
@click.option("-k", default=4, show_default=True, help="Number of results.")
def rag_query(collection, text, k):
    """Show the top-K chunks COLLECTION returns for TEXT."""
    from rich.console import Console
    from .rag import Collection
    console = Console()
    coll = Collection(collection)
    if not coll.exists():
        console.print(f"[red]No such collection:[/red] {collection}")
        sys.exit(1)
    hits = coll.query(text, k=k)
    if not hits:
        console.print("[dim](no matches)[/dim]")
        return
    for i, h in enumerate(hits, 1):
        console.print(f"[cyan][{i}][/cyan] [bold]{h['source']}[/bold]:{h['pos']} "
                      f"[dim](score {h['score']})[/dim]")
        excerpt = h["text"][:300].replace("\n", " ")
        console.print(f"    {excerpt}\n")


@rag_group.command("rm")
@click.argument("collection")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation.")
def rag_rm(collection, yes):
    """Delete a collection (the index only - original files are untouched)."""
    from rich.console import Console
    from .rag import delete_collection
    console = Console()
    if not yes:
        click.confirm(f"Delete collection '{collection}'? Original files are "
                      "kept; only the index is removed.", abort=True)
    try:
        if delete_collection(collection):
            console.print(f"[green]Deleted '{collection}'.[/green]")
        else:
            console.print(f"[red]No such collection:[/red] {collection}")
            sys.exit(1)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("existing", shell_complete=_complete_model_name)
@click.argument("new_name")
def alias(existing, new_name):
    """Register NEW_NAME as another name for EXISTING (same file, no copy).

    Works like 'ollama cp': both names point at the same model file.
    Removing one name keeps the file as long as another name references it.

    \b
    Example:
      localm alias gemma3-12b daily-driver
    """
    from .model_manager import alias_model

    if not alias_model(existing, new_name):
        sys.exit(1)


_POWERSHELL_COMPLETION = r'''# localm tab completion - add this block to your PowerShell $PROFILE
# (run: notepad $PROFILE)
Register-ArgumentCompleter -Native -CommandName localm -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    $words = @($commandAst.CommandElements | Select-Object -Skip 1 | ForEach-Object { $_.Extent.Text })
    if ($words.Count -eq 0 -or ($wordToComplete -eq '' -and $words[-1] -ne '')) {
        $words += ''
    }
    localm __complete @words 2>$null | Where-Object { $_ } | ForEach-Object {
        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
    }
}
'''


@main.command("__complete", hidden=True)
@click.argument("words", nargs=-1)
def _complete_hidden(words):
    """Internal: print completion candidates for the partial command line."""
    words = list(words)
    partial = words[-1] if words else ""
    prior = words[:-1]

    # Commands whose first positional argument is a registered model name
    model_cmds = {"run", "serve", "rm", "alias"}

    if prior and prior[0] in model_cmds and len(prior) == 1:
        from .config import load_registry
        candidates = sorted(load_registry())
    elif not prior:
        candidates = sorted(
            cmd for cmd, obj in main.commands.items()
            if not getattr(obj, "hidden", False)
        )
    else:
        candidates = []

    for c in candidates:
        if c.startswith(partial):
            click.echo(c)


@main.command("completion")
@click.argument("shell", type=click.Choice(["powershell", "bash", "zsh", "fish"]))
def completion(shell):
    """Print shell tab-completion setup for SHELL.

    \b
    PowerShell:  localm completion powershell >> $PROFILE
    bash:        localm completion bash   (prints the one-liner to add)
    zsh / fish:  same, using Click's built-in completion support
    """
    if shell == "powershell":
        click.echo(_POWERSHELL_COMPLETION)
    elif shell == "bash":
        click.echo('# Add to ~/.bashrc:\neval "$(_LOCALM_COMPLETE=bash_source localm)"')
    elif shell == "zsh":
        click.echo('# Add to ~/.zshrc:\neval "$(_LOCALM_COMPLETE=zsh_source localm)"')
    elif shell == "fish":
        click.echo('# Add to ~/.config/fish/completions/localm.fish:\n'
                   '_LOCALM_COMPLETE=fish_source localm | source')


@main.command()
def models():
    """Show popular GGUF model shortcuts."""
    show_shortcuts()


# ------------------------------------------------------------------ #
#  Config / info                                                       #
# ------------------------------------------------------------------ #

@main.command()
def info():
    """Show paths and current configuration."""
    cfg = load_config()
    binary_dir = find_binary_dir()

    console.print(f"  [bold]models dir[/bold]   {HOME_DIR / 'models'}")
    console.print(f"  [bold]registry   [/bold]   {HOME_DIR / 'registry.json'}")
    console.print(f"  [bold]config     [/bold]   {HOME_DIR / 'config.json'}")
    console.print(f"  [bold]binaries   [/bold]   {binary_dir or '[dim]not found[/dim]'}")
    console.print()
    for k, v in sorted(cfg.items()):
        if v is None:
            v = "[dim](auto)[/dim]"
        console.print(f"  {k:<22} {v}")


@main.command("config")
@click.argument("key")
@click.argument("value")
def config_cmd(key, value):
    """Set a persistent configuration value.

    \b
    Examples:
      localm config n_gpu_layers 99
      localm config n_ctx 8192
      localm config temperature 0.7
    """
    from localm.settings_schema import validate_update
    try:
        validated = validate_update({key: value})
    except ValueError as e:
        raise click.ClickException(str(e))
    cfg = load_config()
    cfg.update(validated)
    save_config(cfg)
    console.print(f"[green]✓[/green] {key} = {validated[key]}")


# ------------------------------------------------------------------ #
#  API key management                                                  #
# ------------------------------------------------------------------ #

def _mask_key(key: str) -> str:
    """Show enough of a key to recognise it without leaking the whole secret to
    a terminal log or screen-share. Short keys are fully masked."""
    key = key.strip()
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


@main.group("key")
def key_group():
    """View and manage the API key that protects localm's HTTP surface.

    localm has no accounts: "auth" is one shared owner key, presented as
    'Authorization: Bearer <key>'. With no key the server runs open (loopback
    dev); set one to require it. You can also mint named, scope-limited keys for
    clients that should only do part of what the owner can.

    \b
    Examples:
      localm key show
      localm key generate
      localm key create dashboard --scope models:read
    """


@key_group.command("show")
@click.option("--reveal", is_flag=True,
              help="Print the full key instead of a masked preview.")
def key_show(reveal):
    """Show the active owner key (masked) or 'open mode'."""
    from localm import auth
    active = auth.get_api_key()
    if not active:
        console.print("[yellow]open mode[/yellow] - no API key set; the server "
                      "accepts unauthenticated requests.")
        if auth.require_auth_enabled():
            console.print("[dim]require_auth is on, so protected routes refuse "
                          "every request until a key is set.[/dim]")
        return
    env_key = os.environ.get(auth.ENV_VAR)
    from_env = bool(env_key and env_key.strip())
    source = f"{auth.ENV_VAR} env" if from_env else "auth.key file"
    shown = active if reveal else _mask_key(active)
    console.print(f"API key [dim](from {source})[/dim]:")
    console.print(f"  [bold]{shown}[/bold]")
    if from_env:
        console.print(f"[dim]{auth.ENV_VAR} overrides the stored auth.key file "
                      "while it is set.[/dim]")


@key_group.command("generate")
def key_generate():
    """Generate a new random owner key, persist it, and print it once."""
    from localm import auth
    key = auth.regenerate_key()
    console.print("[green]New API key (shown once - copy it now):[/green]")
    console.print(f"  [bold]{key}[/bold]")
    console.print("[dim]Clients send it as: Authorization: Bearer <key>[/dim]")
    env_key = os.environ.get(auth.ENV_VAR)
    if env_key and env_key.strip():
        console.print(f"[yellow]Note:[/yellow] {auth.ENV_VAR} is set and "
                      "overrides this stored key until it is unset.")


@key_group.command("set")
@click.argument("key")
def key_set(key):
    """Persist a specific owner KEY (use 'generate' for a random one)."""
    from localm import auth
    if not key or not key.strip():
        raise click.ClickException("Key must be non-empty.")
    auth.set_api_key(key)
    console.print(f"[green]✓[/green] API key set "
                  f"[dim]({_mask_key(key)})[/dim]")


@key_group.command("clear")
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation prompt.")
def key_clear(yes):
    """Remove the owner key, returning the server to open mode."""
    from localm import auth
    if auth.get_api_key() is None:
        console.print("[dim]No API key set (already open mode).[/dim]")
        return
    if not yes and not click.confirm(
            "Clear the API key? The server will then accept unauthenticated "
            "requests (unless require_auth is on)"):
        console.print("[dim]Cancelled.[/dim]")
        return
    auth.clear_api_key()
    console.print("[green]✓[/green] API key cleared - open mode.")
    env_key = os.environ.get(auth.ENV_VAR)
    if env_key and env_key.strip():
        console.print(f"[yellow]Note:[/yellow] {auth.ENV_VAR} is still set, so a "
                      "key remains active from the environment.")


@key_group.command("list")
def key_list():
    """List named, scope-limited keys (metadata only - never the secret)."""
    from localm import auth
    keys = auth.list_keys()
    if not keys:
        console.print("[dim]No named keys. Mint one:  "
                      "localm key create <name> --scope <scope>[/dim]")
        return
    from rich.table import Table
    table = Table(header_style="bold cyan")
    table.add_column("ID", style="bold")
    table.add_column("Name")
    table.add_column("Scopes")
    for k in keys:
        table.add_row(str(k.get("id", "")), str(k.get("name", "")),
                      ", ".join(k.get("scopes", [])))
    console.print(table)


@key_group.command("create")
@click.argument("name")
@click.option("-s", "--scope", "scopes", multiple=True, required=True,
              help="A capability scope to grant (repeatable).")
def key_create(name, scopes):
    """Mint a named key limited to SCOPES; print the secret once.

    Privileged scopes (admin, keys:admin, plugins:admin, config:write) are
    refused here - only the owner key carries those, so a minted key can never
    escalate itself.
    """
    from localm import auth
    try:
        rec = auth.create_key(name, list(scopes), allow_privileged=False)
    except (ValueError, PermissionError) as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print(f"[green]New key '{rec['name']}' "
                  f"({', '.join(rec['scopes'])}) - shown once:[/green]")
    console.print(f"  [bold]{rec['key']}[/bold]")
    console.print(f"[dim]id {rec['id']}; revoke with: "
                  f"localm key rm {rec['id']}[/dim]")


@key_group.command("rm")
@click.argument("key_id")
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation prompt.")
def key_rm(key_id, yes):
    """Revoke a named key by ID (see 'localm key list')."""
    from localm import auth
    if not yes and not click.confirm(f"Revoke named key '{key_id}'?"):
        console.print("[dim]Cancelled.[/dim]")
        return
    if auth.revoke_key(key_id):
        console.print(f"[green]✓[/green] Revoked {key_id}.")
    else:
        console.print(f"[red]No such key:[/red] {key_id}")
        sys.exit(1)


# ------------------------------------------------------------------ #
#  Doctor                                                              #
# ------------------------------------------------------------------ #

@main.command()
def doctor():
    """Check system requirements and report any issues.

    \b
    Verifies:
      - Python version (3.10+ required)
      - llama.dll / llama.so available on PATH or in expected locations
      - CUDA / ROCm GPU driver
      - Available VRAM
      - Required Python packages (huggingface-hub, torch, uvicorn, fastapi)
    """
    import importlib
    import subprocess
    import sys as _sys

    ok_sym    = "[green]✓[/green]"
    warn_sym  = "[yellow]![/yellow]"
    fail_sym  = "[red]✗[/red]"

    # ----- Python version -----
    major, minor = _sys.version_info[:2]
    if (major, minor) >= (3, 10):
        console.print(f"  {ok_sym}  Python {major}.{minor}")
    else:
        console.print(f"  {fail_sym}  Python {major}.{minor} - 3.10+ required")

    # ----- llama.dll / llama.so -----
    binary_dir = find_binary_dir()
    if binary_dir:
        dll_names = ["llama.dll", "llama.so", "libllama.so", "llama"]
        found_dll = next(
            (binary_dir / d for d in dll_names if (binary_dir / d).exists()),
            None,
        )
        if found_dll:
            # Existence alone is not health: a zeroed or truncated llama.dll
            # exists but cannot load. Check the file size, and flag a lib that
            # is present but implausibly small to be a real native library.
            try:
                size = found_dll.stat().st_size
            except OSError as e:
                size = -1
                console.print(
                    f"  {fail_sym}  {found_dll.name} in {binary_dir} cannot be "
                    f"read (corrupt?): {e}"
                )
            else:
                # A genuine llama.dll/.so is multiple MB. 64 KiB is a generous
                # floor that still rejects 0/1-byte stubs and tiny placeholders.
                TINY_LIB_BYTES = 64 * 1024
                if size == 0:
                    console.print(
                        f"  {fail_sym}  {found_dll.name} in {binary_dir} is empty "
                        f"(0 bytes) - corrupt; re-run 'localm setup-llama'"
                    )
                elif size < TINY_LIB_BYTES:
                    console.print(
                        f"  {warn_sym}  {found_dll.name} found in {binary_dir} but "
                        f"is suspiciously small ({size} bytes, expected multiple MB) "
                        f"- it may be truncated/corrupt"
                    )
                else:
                    console.print(f"  {ok_sym}  {found_dll.name} found in {binary_dir}")
        else:
            files = [f.name for f in binary_dir.iterdir() if f.is_file()][:8]
            console.print(
                f"  {warn_sym}  binary dir found ({binary_dir}) but no llama .dll/.so - "
                f"contents: {files}"
            )
    else:
        console.print(f"  {fail_sym}  llama binary dir not found - GGUF backend unavailable")

    # ----- GPU driver (CUDA / ROCm) -----
    gpu_found = False
    for cmd, label in [
        (["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
         "NVIDIA"),
        (["rocm-smi", "--showproductname"],
         "AMD ROCm"),
    ]:
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5
            ).stdout.strip()
            if out:
                first_line = out.splitlines()[0]
                console.print(f"  {ok_sym}  {label} GPU: {first_line}")
                gpu_found = True
                break
        except Exception:
            continue

    # ----- VRAM via torch -----
    # Run the torch GPU probe BEFORE deciding the "CPU mode only" verdict:
    # the smi tools can be absent while torch still sees a usable GPU (common
    # on ROCm installs without rocm-smi on PATH). Printing both "CPU mode only"
    # and a torch GPU/VRAM line in the same run contradicts itself, so let
    # torch's view veto the CPU-only warning.
    torch_gpu_found = False
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                # Driver-level free/total - torch's allocator counters miss
                # everything allocated outside torch (llama.dll, other apps)
                free_b, total_b = torch.cuda.mem_get_info(i)
                console.print(
                    f"  {ok_sym}  GPU {i}: {props.name}  "
                    f"{free_b / 1024**3:.1f} GB free / {total_b / 1024**3:.1f} GB total"
                )
                torch_gpu_found = True
        else:
            console.print(f"  {warn_sym}  torch available but torch.cuda.is_available() = False")
    except ImportError:
        console.print(f"  {warn_sym}  torch not installed - GPU VRAM check skipped")

    if not gpu_found and not torch_gpu_found:
        console.print(f"  {warn_sym}  No GPU driver found (nvidia-smi / rocm-smi) - CPU mode only")

    # ----- Required Python packages -----
    packages = [
        ("fastapi",           "FastAPI (HTTP server)"),
        ("uvicorn",           "uvicorn (ASGI server)"),
        ("huggingface_hub",   "huggingface-hub (model downloads)"),
        ("requests",          "requests (HTTP client)"),
        ("rich",              "rich (terminal output)"),
        ("click",             "click (CLI)"),
    ]
    optional_pkgs = [
        ("torch",             "torch (HF backend / GPU info)"),
        ("transformers",      "transformers (HF backend)"),
    ]
    import importlib.metadata as _ilm
    # Import name -> distribution name where they differ (for version lookup).
    _dist_names = {"huggingface_hub": "huggingface-hub"}
    for mod, label in packages + optional_pkgs:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "")
            if not ver:
                # Some packages (e.g. 'rich') expose no __version__ attribute;
                # read the installed distribution metadata instead of blanking.
                try:
                    ver = _ilm.version(_dist_names.get(mod, mod))
                except _ilm.PackageNotFoundError:
                    ver = ""
            sym = ok_sym
            ver_str = f" {ver}" if ver else ""
        except ImportError:
            sym     = warn_sym if (mod, label) in optional_pkgs else fail_sym
            ver_str = " - not installed"
        console.print(f"  {sym}  {label}{ver_str}")


# ------------------------------------------------------------------ #
#  Plugin: coder (optional extra)                                      #
# ------------------------------------------------------------------ #

# Register ``localm coder`` when the coder plugin is installed.
# The plugin is gated behind ``pip install "localm[coder]"`` so the import
# is wrapped in a try/except - the base localm install keeps working fine
# if the extra was never requested.
# MCP server plugin - expose localm to MCP clients (Claude Desktop, etc.)
try:
    from .plugins.mcpserver.cli import main as _mcp_main
    main.add_command(_mcp_main, name="mcp")
except ImportError:
    pass

# GUI plugin - browser interface for chat and the coder agent
try:
    from .plugins.gui.cli import main as _gui_main
    main.add_command(_gui_main, name="gui")
except ImportError:
    pass

# Jobs plugin - scheduled recurring tasks (localm job add/list/run/...)
try:
    from .plugins.builtin.jobs.cli import main as _jobs_main
    main.add_command(_jobs_main, name="job")
except ImportError:
    pass

try:
    from .plugins.coder.cli import main as _coder_main
    main.add_command(_coder_main, name="coder")
except ImportError:
    @main.command("coder", context_settings={"ignore_unknown_options": True})
    def _coder_stub(**_):
        """Offline AI coding agent (run: pip install "localm[coder]" to enable)."""
        console.print(
            '[yellow]The coder plugin could not be loaded.[/yellow]\n'
            'Install its dependencies with:  [bold]pip install "localm[coder]"[/bold]\n'
            '  or (editable):  [bold]pip install -e ".[coder]"[/bold]\n'
            'then activate it with:  [bold]localm plugin install coder[/bold]'
        )

# Abliterate plugin - decensor a model with Heretic (run as a separate program)
# and register the result. Gated behind ``pip install "localm[abliterate]"``.
try:
    from .plugins.abliterate.cli import main as _abliterate_main
    main.add_command(_abliterate_main, name="abliterate")
except ImportError:
    @main.command("abliterate", context_settings={"ignore_unknown_options": True})
    def _abliterate_stub(**_):
        """Decensor a model with Heretic (run: pip install "localm[abliterate]" to enable)."""
        console.print(
            '[yellow]The abliterate plugin is not installed.[/yellow]\n'
            'Enable it with:  [bold]pip install "localm[abliterate]"[/bold]\n'
            '  or (editable):  [bold]pip install -e ".[abliterate]"[/bold]'
        )

# Provision the native llama.cpp binaries into localm's own venv (self-contained).
from .setup_llama import main as _setup_llama_main
main.add_command(_setup_llama_main, name="setup-llama")


@main.command("bug-report")
@click.option("-m", "--message", default="", help="One-line summary of the problem.")
def bug_report_cmd(message: str) -> None:
    """Generate an editable bug report and offer to send it to the maintainer.

    Collects a safe environment snapshot (OS, GPU, driver, backend - never your
    API key, config, or chat data), saves an editable markdown file, and offers
    to email it, open a GitHub issue, or hand it off yourself."""
    from localm import bugreport
    bugreport.report_failure(
        summary=message or "user-reported issue",
        context={"operation": "bug-report"},
        as_failure=False,
        interactive=bool(getattr(sys.stdin, "isatty", lambda: False)()))


# ------------------------------------------------------------------ #
#  Plugin management (external plugins in ~/.localm/plugins/)          #
# ------------------------------------------------------------------ #

@main.group()
def plugin() -> None:
    """Manage external plugins (installed in ~/.localm/plugins/)."""


@plugin.command("list")
def plugin_list():
    """List installed external plugins."""
    from .plugins.loader import discover_errors, discover_plugins, plugins_dir

    manifests = discover_plugins()
    errors = discover_errors()
    if not manifests and not errors:
        console.print(f"[dim]No external plugins installed ({plugins_dir()})[/dim]")
        return
    for m in manifests:
        desc = f" - {m.description}" if m.description else ""
        console.print(f"  [bold]{m.name}[/bold] v{m.version}{desc}")
        if m.tool_exports:
            console.print(f"    [dim]tools: {', '.join(m.tool_exports)}[/dim]")
    for err in errors:
        console.print(f"  [yellow]invalid:[/yellow] {err}")


@plugin.command("remove")
@click.argument("name")
def plugin_remove(name):
    """Remove an installed plugin by name."""
    from .plugins.loader import PluginError, remove_plugin

    try:
        existed = remove_plugin(name)
    except PluginError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    if existed:
        console.print(f"[green]Removed[/green] plugin [bold]{name}[/bold]")
    else:
        console.print(f"[yellow]Plugin {name!r} is not installed[/yellow]")


# Engine plugins (builtin + external) - install/uninstall + enable/disable +
# status. Two axes: install/uninstall moves a plugin between the available catalog
# (builtin/ + external dirs) and the user's installed set; enable/disable toggles
# an installed plugin active/inactive. A PluginManager with no app is enough to
# flip config (no routes are mounted from the CLI - a running GUI server picks
# HTTP routes up on its next start, while stdio plugins like mcp take effect
# immediately). Nothing is installed by default: only what the user selects.

def _engine_manager():
    from .plugins.engine import PluginManager
    return PluginManager(None)


def _warn_missing_requires(mgr, name):
    missing = mgr.missing_requires(name)
    if missing:
        cmds = "  ".join(f"localm plugin install {m}" for m in missing)
        console.print(
            f"[yellow]Note:[/yellow] {name!r} declares it needs "
            f"{', '.join(missing)}, which {'is' if len(missing) == 1 else 'are'} "
            f"not installed. Install with:  {cmds}")


@plugin.command("install")
@click.argument("target")
@click.option("--force", is_flag=True,
              help="When installing from a directory, overwrite an existing install.")
def plugin_install_engine(target, force):
    """Install a plugin and enable it.

    TARGET is either a first-party plugin NAME from the bundled store
    (e.g. ``localm plugin install coder``) or a path to a DIRECTORY containing a
    plugin.toml (a third-party plugin). For plugins with extra dependencies, also
    run the matching pip extra, e.g. pip install "localm[coder]".
    """
    mgr = _engine_manager()
    src = Path(target)
    if src.is_dir() and (src / "plugin.toml").is_file():
        try:
            spec = mgr.set_installed_from_dir(src, force=force)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
        console.print(f"[green]Installed[/green] plugin [bold]{spec.name}[/bold] "
                      f"v{spec.version}")
        _warn_missing_requires(mgr, spec.name)
        return
    try:
        mgr.set_installed_state(target, True)
    except KeyError:
        console.print(f"[red]No such plugin: {target}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print(f"[green]Installed[/green] plugin [bold]{target}[/bold]")
    _warn_missing_requires(mgr, target)


# Recommended everyday set for a non-interactive install: the surfaces that work
# without a separate external service. image/music/video need a ComfyUI server,
# voice needs the [voice] extra + a model, mcp is for external clients - all opt-in.
_SETUP_DEFAULTS = ("coder", "rag", "web", "tts")


def _installed_plugin_names(mgr) -> set:
    return {p["name"] for p in mgr.api_state()["plugins"] if p.get("installed")}


def _parse_plugin_selection(raw, available):
    """Parse an interactive selection (comma list of 1-based numbers or names,
    or 'all') into a list of plugin names."""
    if not raw.strip():
        return []
    if raw.strip().lower() == "all":
        return [e.name for e in available]
    by_idx = {str(i): e.name for i, e in enumerate(available, 1)}
    names = {e.name for e in available}
    out = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        if t in by_idx:
            out.append(by_idx[t])
        elif t in names:
            out.append(t)
        else:
            console.print(f"[yellow]Ignoring unknown selection: {t}[/yellow]")
    return out


@plugin.command("setup")
@click.option("--plugins", "plugins_csv", default=None,
              help="Comma-list to install non-interactively (skips the prompt).")
@click.option("--all", "install_all", is_flag=True,
              help="Install every first-party plugin.")
@click.option("--defaults", "install_defaults", is_flag=True,
              help="Install the recommended set non-interactively (coder, rag, web, tts).")
def plugin_setup(plugins_csv, install_all, install_defaults):
    """Choose which first-party plugins to install.

    Out of the box only chat is active; this turns on the features you want
    (coder, image/music/video, rag, web, voice, tts, mcp). Run by the installer
    after dependencies are in place, and any time afterwards. Some plugins also
    need a pip extra (e.g. pip install "localm[coder]").
    """
    from .plugins import catalog
    mgr = _engine_manager()
    installed = _installed_plugin_names(mgr)
    available = [e for e in catalog.CATALOG if not e.preinstalled]

    if install_all:
        chosen = [e.name for e in available]
    elif plugins_csv is not None:
        chosen = _parse_plugin_selection(plugins_csv, available)
    elif install_defaults:
        chosen = list(_SETUP_DEFAULTS)
    elif not sys.stdin.isatty():
        console.print(
            "[dim]Non-interactive shell - skipping plugin selection. Run "
            "[bold]localm plugin setup[/bold] to choose, or pass "
            "--plugins/--all/--defaults.[/dim]")
        return
    else:
        console.print("Choose first-party plugins to install (chat is always on):\n")
        for i, e in enumerate(available, 1):
            mark = " [green](installed)[/green]" if e.name in installed else ""
            extra = f" [dim](pip extra: localm[{e.extra}])[/dim]" if e.extra else ""
            console.print(f"  [bold]{i:>2}.[/bold] {e.name:<7} {e.description}{mark}{extra}")
        console.print(
            "\nEnter numbers or names (comma-separated), 'all', or blank to skip.")
        raw = click.prompt("Install", default="", show_default=False)
        chosen = _parse_plugin_selection(raw, available)

    if not chosen:
        console.print("[dim]No plugins selected.[/dim]")
        return
    for name in chosen:
        if name in installed:
            console.print(f"[dim]{name} already installed[/dim]")
            continue
        try:
            mgr.set_installed_state(name, True)
        except (KeyError, ValueError) as e:
            console.print(f"[yellow]Skipped {name}: {e}[/yellow]")
            continue
        console.print(f"[green]Installed[/green] [bold]{name}[/bold]")
        _warn_missing_requires(mgr, name)


@plugin.command("uninstall")
@click.argument("name")
@click.option("--delete-data", is_flag=True,
              help="Also delete this plugin's stored data (default: keep it).")
def plugin_uninstall_engine(name, delete_data):
    """Uninstall (deselect) an engine plugin. Keeps its data unless --delete-data."""
    mgr = _engine_manager()
    try:
        was = mgr.uninstall(name, delete_data=delete_data)
    except KeyError:
        console.print(f"[red]No such plugin: {name}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    if was:
        console.print(f"[yellow]Uninstalled[/yellow] plugin [bold]{name}[/bold]")
    else:
        console.print(f"[dim]Plugin {name!r} was not installed.[/dim]")


@plugin.command("enable")
@click.argument("name")
def plugin_enable(name):
    """Enable an installed engine plugin (must be installed first)."""
    mgr = _engine_manager()
    try:
        mgr.set_enabled_state(name, True)
    except KeyError:
        console.print(f"[red]No such plugin: {name}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print(f"[green]Enabled[/green] plugin [bold]{name}[/bold]")
    _warn_missing_requires(mgr, name)


@plugin.command("disable")
@click.argument("name")
def plugin_disable(name):
    """Disable an installed engine plugin (keeps it installed)."""
    mgr = _engine_manager()
    try:
        mgr.set_enabled_state(name, False)
    except KeyError:
        console.print(f"[red]No such plugin: {name}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print(f"[yellow]Disabled[/yellow] plugin [bold]{name}[/bold]")


@plugin.command("status")
def plugin_status():
    """Show engine plugins: installed/available and their enabled state."""
    state = _engine_manager().api_state()
    plugins = state.get("plugins", [])
    if not plugins:
        console.print("[dim]No engine plugins discovered.[/dim]")
        return
    console.print("[bold]Installed[/bold]")
    any_installed = False
    for p in plugins:
        if not p.get("installed"):
            continue
        any_installed = True
        mark = "[green]on [/green]" if p.get("active") else "[yellow]off[/yellow]"
        desc = f" - {p['description']}" if p.get("description") else ""
        console.print(f"  {mark} [bold]{p['name']}[/bold]{desc}")
    if not any_installed:
        console.print("  [dim](none - only chat is active by default)[/dim]")
    console.print("[bold]Available[/bold] (not installed)")
    any_available = False
    for p in plugins:
        if p.get("installed"):
            continue
        any_available = True
        desc = f" - {p['description']}" if p.get("description") else ""
        console.print(f"  [dim]+[/dim]  [bold]{p['name']}[/bold]{desc}")
    if not any_available:
        console.print("  [dim](none)[/dim]")


# Register external plugin commands at import time so they show in --help.
# A broken plugin must never take down the CLI - warnings only.
try:
    from .plugins.loader import register_external_plugins as _register_ext

    for _warning in _register_ext(main):
        console.print(f"[yellow]plugin warning:[/yellow] {_warning}")
except Exception as _e:  # pragma: no cover - absolute last resort
    console.print(f"[yellow]plugin discovery failed:[/yellow] {_e}")
