import json
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
    add_local, get_model_info, get_model_path, list_models, pull_model,
    remove_model, show_shortcuts,
)

console = Console()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option("0.1.0", prog_name="localm")
def main() -> None:
    """Run local LLMs offline — HuggingFace and GGUF models, AMD/NVIDIA/CPU."""


# ------------------------------------------------------------------ #
#  run                                                                 #
# ------------------------------------------------------------------ #

@main.command()
@click.argument("model")
@click.option("-p", "--prompt",       default=None,  help="Single prompt (non-interactive).")
@click.option("-s", "--system",       default=None,  help="System prompt.")
@click.option("-m", "--max-tokens",   default=None,  type=int,   help="Max tokens to generate.")
@click.option("-t", "--temperature",  default=None,  type=float, help="Sampling temperature.")
@click.option("-c", "--ctx",          default=None,  type=int,   help="Context window (GGUF only).")
@click.option("-g", "--gpu-layers",   default=None,  type=int,   help="GPU layers (GGUF only, 99=all).")
@click.option("--mmproj",             default=None,  help="Multimodal projection GGUF path.")
@click.option("--device",             default=None,  help="HF device override (cuda / cpu).")
@click.option("--image", "images",   multiple=True, type=click.Path(exists=True),
              help="Local image file to include (repeat for multiple). Use with -p.")
@click.option("--output-dir",         default=None,  type=click.Path(),
              help="Directory to save any images the model produces.")
def run(model, prompt, system, max_tokens, temperature, ctx, gpu_layers,
        mmproj, device, images, output_dir):
    """Run a model — interactive chat or single prompt.

    \b
    MODEL can be a registered name OR a direct path:
      localm run gemma4-12b
      localm run D:\\models\\mymodel.gguf
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
      echo "Explain RDNA2" | localm run gemma4-12b
    """
    info = get_model_info(model)
    if info is None:
        console.print(f"[red]Model not found:[/red] {model}")
        console.print("  [dim]localm list[/dim]              — downloaded models")
        console.print("  [dim]localm models[/dim]            — GGUF shortcuts")
        console.print("  [dim]localm pull owner/repo[/dim]   — download HF model")
        console.print("  [dim]localm pull name[/dim]         — download GGUF shortcut")
        console.print("  [dim]localm add <path>[/dim]        — register local file/dir")
        console.print("  [dim]localm run /full/path[/dim]    — use path directly")
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

    # Resolve output directory for any image output
    out_dir = Path(output_dir) if output_dir else Path.cwd()

    # Accept piped stdin as the prompt
    if not sys.stdin.isatty() and prompt is None:
        prompt = sys.stdin.read().strip()

    with engine:
        if prompt is not None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append(_build_user_message(prompt, list(images)))
            response = _stream_once(engine, messages, **gen_opts)
            _handle_image_output(response, out_dir)
        else:
            _interactive(engine, system, gen_opts, out_dir)


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


def _stream_once(engine, messages: list, **kwargs) -> str:
    """Stream response to stdout and return the full text."""
    full = ""
    for token in engine.chat_stream(messages, **kwargs):
        print(token, end="", flush=True)
        full += token
    print()
    return full


def _handle_image_output(response: str, out_dir: Path) -> None:
    """Extract any base64-encoded images from the model's response and save them."""
    import base64
    import re
    pattern = re.compile(
        r"!\[.*?\]\(data:(image/\w+);base64,([A-Za-z0-9+/=]+)\)"
        r"|data:(image/\w+);base64,([A-Za-z0-9+/=]{100,})"
    )
    saved = 0
    for match in pattern.finditer(response):
        mime = match.group(1) or match.group(3)
        b64  = match.group(2) or match.group(4)
        ext  = mime.split("/")[-1].replace("jpeg", "jpg")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"output_{saved + 1}.{ext}"
        out_path.write_bytes(base64.b64decode(b64))
        console.print(f"[green]Image saved:[/green] {out_path}")
        saved += 1


def _interactive(engine, system_prompt: Optional[str], gen_opts: dict,
                 out_dir: Optional[Path] = None) -> None:
    console.print(Panel(
        f"[bold cyan]localm[/bold cyan] — {engine.display_name}\n"
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
            stop = _handle_command(user_input, messages, gen_opts, pending_images)
            if stop:
                break
            continue

        msg = _build_user_message(user_input, pending_images)
        pending_images.clear()
        messages.append(msg)
        console.print("\n[bold blue]Assistant[/bold blue]: ", end="")

        response = ""
        try:
            for token in engine.chat_stream(messages, **gen_opts):
                print(token, end="", flush=True)
                response += token
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/dim]")
            response = response or "(interrupted)"
        except Exception as e:
            console.print(f"\n[red]Inference error: {e}[/red]")
            continue

        print()
        if response:
            messages.append({"role": "assistant", "content": response})
            if out_dir:
                _handle_image_output(response, out_dir)


def _handle_command(
    raw: str,
    messages: list,
    gen_opts: dict,
    pending_images: Optional[list] = None,
) -> bool:
    """Handle a /command. Returns True if the session should exit."""
    parts = raw[1:].split(" ", 1)
    cmd  = parts[0].lower()
    arg  = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("exit", "quit", "q", "bye"):
        console.print("[dim]Bye.[/dim]")
        return True
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
                console.print(f"[dim]Queued {p.name} — will attach to your next message[/dim]")
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
        _save_chat(messages, arg or "chat.json")
    elif cmd == "temp":
        try:
            gen_opts["temperature"] = float(arg)
            console.print(f"[dim]temperature = {gen_opts['temperature']}[/dim]")
        except ValueError:
            console.print("[dim]Usage: /temp 0.7[/dim]")
    elif cmd == "tokens":
        try:
            gen_opts["max_tokens"] = int(arg)
            console.print(f"[dim]max_tokens = {gen_opts['max_tokens']}[/dim]")
        except ValueError:
            console.print("[dim]Usage: /tokens 2048[/dim]")
    elif cmd == "help":
        console.print(
            "[dim]"
            "/exit                   quit\n"
            "/clear                  clear chat history\n"
            "/image <path>           queue a local image for the next message\n"
            "/images                 list queued images\n"
            "/system <text>          set system prompt\n"
            "/save [file]            save conversation to JSON\n"
            "/temp <float>           sampling temperature\n"
            "/tokens <int>           max response tokens"
            "[/dim]"
        )
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
@click.argument("model")
@click.option("-H", "--host",        default="127.0.0.1", help="Bind address (0.0.0.0 for LAN).")
@click.option("-p", "--port",        default=8080,        type=int)
@click.option("-c", "--ctx",         default=None,        type=int)
@click.option("-g", "--gpu-layers",  default=None,        type=int)
@click.option("--mmproj",            default=None)
@click.option("--device",            default=None)
def serve(model, host, port, ctx, gpu_layers, mmproj, device):
    """Start an OpenAI-compatible inference server.

    \b
    POST http://<host>:<port>/v1/chat/completions
    GET  http://<host>:<port>/v1/models
    GET  http://<host>:<port>/health

    Compatible with any OpenAI client library.
    """
    info = get_model_info(model)
    if info is None:
        console.print(f"[red]Model not found:[/red] {model}")
        sys.exit(1)

    model_path, _display_hint = info

    from .inference.engine import Engine
    from .inference.http_server import serve as http_serve
    from .model_manager import load_registry as _reg

    display_name = model if model in _reg() else _display_hint

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
        f"[bold]http://{host}:{port}/v1/chat/completions[/bold]"
    )
    console.print("[dim]Ctrl+C to stop[/dim]\n")

    try:
        http_serve(engine, host=host, port=port)
    except KeyboardInterrupt:
        pass
    finally:
        engine.unload()


# ------------------------------------------------------------------ #
#  Model management                                                    #
# ------------------------------------------------------------------ #

@main.command()
@click.argument("model_spec")
@click.option("-n", "--name", default=None, help="Alias for the downloaded model.")
def pull(model_spec, name):
    """Download a model from HuggingFace or a URL.

    \b
    Full HF model (transformers format, for multimodal / HF-native models):
      localm pull google/gemma-3-4b-it
      localm pull microsoft/Phi-4-mini-instruct

    \b
    Single GGUF file (quantized, lighter, works with GGUF backend):
      localm pull gemma3-4b
      localm pull bartowski/gemma-3-4b-it-GGUF:gemma-3-4b-it-Q4_K_M.gguf

    \b
    Direct URL:
      localm pull https://example.com/model.gguf

    Models are stored in ~/.localm/models/ and registered automatically.
    """
    pull_model(model_spec, name)


@main.command("list")
def list_cmd():
    """List registered models."""
    list_models()


@main.command()
@click.argument("model")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation.")
def rm(model, yes):
    """Remove a model from the registry (and delete the file if it's in ~/.localm)."""
    if not yes:
        click.confirm(f"Remove '{model}'?", abort=True)
    remove_model(model)


@main.command()
@click.argument("path")
@click.option("-n", "--name", default=None, help="Name to register the model as.")
def add(path, name):
    """Register a local model file or HuggingFace directory.

    \b
    Examples:
      localm add C:\\models\\mymodel.gguf
      localm add D:\\projects\\heresy\\gemma-4-12B-it-qat-q4_0-unquantized-bin --name gemma4-12b
    """
    add_local(path, name)


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
    cfg = load_config()
    coerced: object = value
    if value.lower() in ("true", "false"):
        coerced = value.lower() == "true"
    else:
        for cast in (int, float):
            try:
                coerced = cast(value)
                break
            except ValueError:
                pass
    cfg[key] = coerced
    save_config(cfg)
    console.print(f"[green]✓[/green] {key} = {coerced}")


# ------------------------------------------------------------------ #
#  Plugin: coder (optional extra)                                      #
# ------------------------------------------------------------------ #

# Register ``localm coder`` when the coder plugin is installed.
# The plugin is gated behind ``pip install "localm[coder]"`` so the import
# is wrapped in a try/except — the base localm install keeps working fine
# if the extra was never requested.
try:
    from .plugins.coder.cli import main as _coder_main
    main.add_command(_coder_main, name="coder")
except ImportError:
    @main.command("coder", context_settings={"ignore_unknown_options": True})
    def _coder_stub(**_):
        """Offline AI coding agent (run: pip install "localm[coder]" to enable)."""
        console.print(
            '[yellow]The coder plugin is not installed.[/yellow]\n'
            'Enable it with:  [bold]pip install "localm[coder]"[/bold]\n'
            '  or (editable):  [bold]pip install -e ".[coder]"[/bold]'
        )
