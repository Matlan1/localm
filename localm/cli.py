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


_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _exposed_bind_warning(host: str) -> Optional[str]:
    """
    Warning text when binding beyond loopback without an API key set —
    that combination serves an unauthenticated LLM API to the whole network.
    Returns None when the configuration is safe.
    """
    import os
    if host in _LOOPBACK_HOSTS or os.environ.get("LOCALM_API_KEY"):
        return None
    return (
        f"⚠ Binding to {host} WITHOUT authentication — anyone on the network "
        f"can use this server, unload your model, and read every response.\n"
        f"  Set an API key first:  $env:LOCALM_API_KEY = \"<secret>\"  "
        f"(clients send it as a Bearer token)"
    )


def _complete_model_name(ctx, param, incomplete):
    """Shell-completion callback: registered model names matching the prefix."""
    try:
        from .config import load_registry as _lr
        return sorted(n for n in _lr() if n.startswith(incomplete))
    except Exception:
        return []


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option("0.1.0", prog_name="localm")
def main() -> None:
    """Run local LLMs offline — HuggingFace and GGUF models, AMD/NVIDIA/CPU."""


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
    """Stream response to stdout, print tok/s on completion, and return the full text."""
    import time as _time
    parts: list[str] = []
    t0 = _time.monotonic()
    for token in engine.chat_stream(messages, **kwargs):
        print(token, end="", flush=True)
        parts.append(token)
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
                 out_dir: Optional[Path] = None) -> None:  # noqa: C901
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

        parts: list[str] = []
        import time as _time
        t0 = _time.monotonic()
        try:
            for token in engine.chat_stream(messages, **gen_opts):
                print(token, end="", flush=True)
                parts.append(token)
        except KeyboardInterrupt:
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
@click.argument("model", shell_complete=_complete_model_name)
@click.option("-H", "--host",        default="127.0.0.1", help="Bind address (0.0.0.0 for LAN).")
@click.option("-p", "--port",        default=None,        type=int,
              help="Port [default: config 'port' (8642); auto-bumps when busy].")
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

    warning = _exposed_bind_warning(host)
    if warning:
        console.print(f"[bold yellow]{warning}[/bold yellow]")

    from .config import pick_port
    requested = port
    port, was_busy = pick_port(requested, host="127.0.0.1" if host == "0.0.0.0" else host)
    if was_busy:
        console.print(
            f"[yellow]Port {requested if requested is not None else load_config().get('port', 8642)} "
            f"is in use — serving on {port} instead.[/yellow]"
        )

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
#  benchmark                                                           #
# ------------------------------------------------------------------ #

@main.command()
@click.argument("model", shell_complete=_complete_model_name)
@click.option("-n", "--gen-tokens", default=128, show_default=True,
              help="Tokens to generate per run.")
@click.option("--prompts", default="64,512,2048", show_default=True,
              help="Comma-separated approximate prompt sizes in tokens.")
@click.option("-c", "--ctx",        default=None, type=int)
@click.option("-g", "--gpu-layers", default=None, type=int)
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
    table = Table(title=f"benchmark — {model}")
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
              help="Expected SHA256 hex digest (URL downloads only). Download is deleted on mismatch.")
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
    pull_model(model_spec, name, expected_sha256=sha256, redownload=redownload)


@main.command("list")
def list_cmd():
    """List registered models."""
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
                detail = (f"unregisters the name only — file kept, "
                          f"still registered as: {', '.join(others)}")
            elif str(path).startswith(str(MODELS_DIR)) and path.exists():
                size = path.stat().st_size / 1e9 if path.is_file() else None
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
    add_local(path, name, on_duplicate=on_duplicate, no_hash=no_hash)


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


_POWERSHELL_COMPLETION = r'''# localm tab completion — add this block to your PowerShell $PROFILE
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
        console.print(f"  {fail_sym}  Python {major}.{minor} — 3.10+ required")

    # ----- llama.dll / llama.so -----
    binary_dir = find_binary_dir()
    if binary_dir:
        dll_names = ["llama.dll", "llama.so", "libllama.so", "llama"]
        found_dll = next(
            (binary_dir / d for d in dll_names if (binary_dir / d).exists()),
            None,
        )
        if found_dll:
            console.print(f"  {ok_sym}  {found_dll.name} found in {binary_dir}")
        else:
            files = [f.name for f in binary_dir.iterdir() if f.is_file()][:8]
            console.print(
                f"  {warn_sym}  binary dir found ({binary_dir}) but no llama .dll/.so — "
                f"contents: {files}"
            )
    else:
        console.print(f"  {fail_sym}  llama binary dir not found — GGUF backend unavailable")

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

    if not gpu_found:
        console.print(f"  {warn_sym}  No GPU driver found (nvidia-smi / rocm-smi) — CPU mode only")

    # ----- VRAM via torch -----
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props   = torch.cuda.get_device_properties(i)
                total   = props.total_memory / 1e9
                free    = (props.total_memory - torch.cuda.memory_allocated(i)) / 1e9
                console.print(
                    f"  {ok_sym}  GPU {i}: {props.name}  "
                    f"{free:.1f} GB free / {total:.1f} GB total"
                )
        else:
            console.print(f"  {warn_sym}  torch available but torch.cuda.is_available() = False")
    except ImportError:
        console.print(f"  {warn_sym}  torch not installed — GPU VRAM check skipped")

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
    for mod, label in packages + optional_pkgs:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "")
            sym = ok_sym
            ver_str = f" {ver}" if ver else ""
        except ImportError:
            sym     = warn_sym if (mod, label) in optional_pkgs else fail_sym
            ver_str = " — not installed"
        console.print(f"  {sym}  {label}{ver_str}")


# ------------------------------------------------------------------ #
#  Plugin: coder (optional extra)                                      #
# ------------------------------------------------------------------ #

# Register ``localm coder`` when the coder plugin is installed.
# The plugin is gated behind ``pip install "localm[coder]"`` so the import
# is wrapped in a try/except — the base localm install keeps working fine
# if the extra was never requested.
# MCP server plugin — expose localm to MCP clients (Claude Desktop, etc.)
try:
    from .plugins.mcpserver.cli import main as _mcp_main
    main.add_command(_mcp_main, name="mcp")
except ImportError:
    pass

# GUI plugin — browser interface for chat and the coder agent
try:
    from .plugins.gui.cli import main as _gui_main
    main.add_command(_gui_main, name="gui")
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
            '[yellow]The coder plugin is not installed.[/yellow]\n'
            'Enable it with:  [bold]pip install "localm[coder]"[/bold]\n'
            '  or (editable):  [bold]pip install -e ".[coder]"[/bold]'
        )


# ------------------------------------------------------------------ #
#  Plugin management (external plugins in ~/.localm/plugins/)          #
# ------------------------------------------------------------------ #

@main.group()
def plugin() -> None:
    """Manage external plugins (installed in ~/.localm/plugins/)."""


@plugin.command("install")
@click.argument("source", type=click.Path(exists=True, file_okay=False))
@click.option("--force", is_flag=True, help="Overwrite an existing install of the same plugin.")
def plugin_install(source, force):
    """Install a plugin from a local directory containing plugin.toml."""
    from .plugins.loader import PluginError, install_plugin

    try:
        manifest = install_plugin(Path(source), force=force)
    except PluginError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print(
        f"[green]Installed[/green] [bold]{manifest.name}[/bold] "
        f"v{manifest.version} → {manifest.path}"
    )
    if manifest.tool_exports:
        console.print(f"  Tool exports: {', '.join(manifest.tool_exports)}")


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
        desc = f" — {m.description}" if m.description else ""
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


# Register external plugin commands at import time so they show in --help.
# A broken plugin must never take down the CLI — warnings only.
try:
    from .plugins.loader import register_external_plugins as _register_ext

    for _warning in _register_ext(main):
        console.print(f"[yellow]plugin warning:[/yellow] {_warning}")
except Exception as _e:  # pragma: no cover — absolute last resort
    console.print(f"[yellow]plugin discovery failed:[/yellow] {_e}")
