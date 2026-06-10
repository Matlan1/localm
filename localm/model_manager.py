import shutil
import sys
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.progress import (
    BarColumn, DownloadColumn, Progress,
    TextColumn, TimeRemainingColumn, TransferSpeedColumn,
)
from rich.table import Table

from .config import MODELS_DIR, ensure_dirs, load_registry, save_registry

console = Console()

# name -> "owner/repo:filename.gguf"
MODEL_SHORTCUTS: dict[str, str] = {
    "llama3.2-1b":    "bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "llama3.2-3b":    "bartowski/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "phi4-mini":      "bartowski/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q4_K_M.gguf",
    "mistral-7b":     "bartowski/Mistral-7B-Instruct-v0.3-GGUF:Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    "qwen2.5-7b":     "bartowski/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "qwen2.5-14b":    "bartowski/Qwen2.5-14B-Instruct-GGUF:Qwen2.5-14B-Instruct-Q4_K_M.gguf",
    "gemma3-4b":      "bartowski/gemma-3-4b-it-GGUF:gemma-3-4b-it-Q4_K_M.gguf",
    "gemma3-12b":     "bartowski/gemma-3-12b-it-GGUF:gemma-3-12b-it-Q4_K_M.gguf",
    "deepseek-r1-7b": "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
    "deepseek-r1-14b":"bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF:DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
    "smollm2-1.7b":   "bartowski/SmolLM2-1.7B-Instruct-GGUF:SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
}

_SHORTCUT_SIZES: dict[str, str] = {
    "llama3.2-1b": "~0.7 GB", "llama3.2-3b": "~2 GB",
    "phi4-mini": "~2.5 GB",   "mistral-7b": "~4.1 GB",
    "qwen2.5-7b": "~4.7 GB",  "qwen2.5-14b": "~8.5 GB",
    "gemma3-4b": "~2.5 GB",   "gemma3-12b": "~7.5 GB",
    "deepseek-r1-7b": "~4.7 GB", "deepseek-r1-14b": "~8.5 GB",
    "smollm2-1.7b": "~1 GB",
}


def resolve_spec(spec: str) -> str:
    return MODEL_SHORTCUTS.get(spec, spec)


def get_model_path(name: str) -> Optional[Path]:
    """Resolve a model name/alias/path to the model file or directory.

    Returns the resolved Path, or None if not found.
    To also get a display name hint, use get_model_info().
    """
    result = get_model_info(name)
    return result[0] if result else None


def get_model_info(name: str):
    """Like get_model_path(), but returns (path, display_hint) or None.

    display_hint is a human-readable name when the original arg was an Ollama
    manifest path; otherwise it's None (the engine derives its own name).
    """
    reg = load_registry()
    if name in reg:
        p = Path(reg[name]["path"])
        if p.exists():
            return p, None

    direct = Path(name)
    if not direct.exists():
        return None

    # HF model directory
    if direct.is_dir() and (direct / "config.json").exists():
        return direct, None
    # GGUF file
    if direct.is_file() and direct.suffix == ".gguf":
        return direct, None
    # Ollama blob (no extension, sha256- prefix)
    if direct.is_file() and direct.name.startswith("sha256-"):
        return direct, None
    # Ollama manifest directory -> resolve on the fly
    ollama = _resolve_ollama_manifest(direct)
    if ollama is not None:
        blob_path, suggested = ollama
        return blob_path, suggested

    return None


def list_models() -> None:
    reg = load_registry()
    if not reg:
        console.print("[dim]No models yet. Use [bold]localm pull <name>[/bold] to download one.[/dim]")
        console.print("[dim]Run [bold]localm models[/bold] to see what's available.[/dim]")
        return

    table = Table(header_style="bold cyan", show_lines=False, expand=False)
    table.add_column("Name", style="bold white")
    table.add_column("Type", style="cyan")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Source", style="dim")
    table.add_column("Path", style="dim")

    for name, info in sorted(reg.items()):
        path = Path(info["path"])
        source = info.get("source", "local")

        if path.is_dir():
            kind = "hf"
            size = "[dim]dir[/dim]"
        elif path.exists():
            kind = "gguf"
            b = path.stat().st_size
            size = f"{b/1e9:.2f} GB" if b >= 1e9 else f"{b/1e6:.0f} MB"
        else:
            kind = "?"
            size = "[red]missing[/red]"

        table.add_row(name, kind, size, source, str(path))

    console.print(table)


def pull_model(model_spec: str, name: Optional[str] = None) -> None:
    spec = resolve_spec(model_spec)
    if spec.startswith("http://") or spec.startswith("https://"):
        _pull_url(spec, name or _stem_from_url(spec))
    elif "/" in spec:
        if ":" in spec or spec.rsplit("/", 1)[-1].endswith(".gguf"):
            # owner/repo:file.gguf  or  owner/repo/file.gguf  -> single GGUF file
            _pull_gguf_file(spec, name)
        else:
            # owner/repo  (no filename) -> full HuggingFace snapshot
            _pull_hf_snapshot(spec, name)
    else:
        console.print(f"[red]Unknown spec:[/red] {model_spec}")
        console.print("Formats:")
        console.print("  [bold]owner/repo[/bold]              full HF model directory")
        console.print("  [bold]owner/repo:file.gguf[/bold]   single GGUF file")
        console.print("  [bold]https://...[/bold]             direct URL")
        console.print("Run [bold]localm models[/bold] for GGUF shortcuts.")


def _stem_from_url(url: str) -> str:
    return url.split("/")[-1].split("?")[0].removesuffix(".gguf")


def _check_disk_space(dest_dir: Path, required_bytes: int) -> bool:
    """
    Verify there is at least *required_bytes* of free space on the volume that
    holds *dest_dir*.  Prints a warning and returns False when space is
    insufficient; returns True when fine or when the check is skipped
    (e.g. ``required_bytes == 0``).
    """
    if not required_bytes:
        return True
    try:
        usage = shutil.disk_usage(dest_dir)
        if usage.free < required_bytes:
            need_gb  = required_bytes / 1e9
            free_gb  = usage.free / 1e9
            console.print(
                f"[red]Not enough disk space.[/red] "
                f"Need {need_gb:.1f} GB, have {free_gb:.1f} GB free on {dest_dir}"
            )
            return False
    except Exception:
        pass   # disk_usage failure is non-fatal; proceed with the download
    return True


def _pull_gguf_file(spec: str, name: Optional[str]) -> None:
    """Download a single .gguf file from a HuggingFace repo."""
    try:
        from huggingface_hub import hf_hub_download, hf_hub_url
    except ImportError:
        console.print("[red]Missing:[/red] huggingface-hub  (run: uv pip install huggingface-hub)")
        return

    if ":" in spec:
        repo_id, filename = spec.rsplit(":", 1)
    else:
        parts = spec.rsplit("/", 1)
        repo_id, filename = parts[0], parts[1]

    model_name = name or filename.removesuffix(".gguf")
    dest = MODELS_DIR / filename

    if dest.exists():
        console.print(f"[yellow]Already downloaded:[/yellow] {model_name}")
        _register(model_name, dest, f"hf:{repo_id}")
        return

    ensure_dirs()

    # Disk space pre-flight — HEAD the CDN URL to get Content-Length
    try:
        import requests as _req
        cdn_url  = hf_hub_url(repo_id, filename)
        head     = _req.head(cdn_url, allow_redirects=True, timeout=10)
        file_size = int(head.headers.get("content-length", 0))
    except Exception:
        file_size = 0

    if not _check_disk_space(MODELS_DIR, file_size):
        return

    console.print(f"Pulling [bold cyan]{repo_id}[/bold cyan] / [bold]{filename}[/bold]")

    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )
        final = MODELS_DIR / filename
        if Path(local) != final:
            shutil.move(local, final)
    except Exception as e:
        console.print(f"[red]Download failed:[/red] {e}")
        return

    _register(model_name, MODELS_DIR / filename, f"hf:{repo_id}")
    console.print(f"[green]✓[/green] [bold]{model_name}[/bold] is ready")


def _pull_hf_snapshot(repo_id: str, name: Optional[str]) -> None:
    """Download a complete HuggingFace model repo (for transformers/HF format models)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        console.print("[red]Missing:[/red] huggingface-hub  (run: uv pip install huggingface-hub)")
        return

    model_name = name or repo_id.split("/")[-1]
    dest = MODELS_DIR / model_name

    if dest.exists() and (dest / "config.json").exists():
        console.print(f"[yellow]Already downloaded:[/yellow] {model_name}")
        _register(model_name, dest, f"hf:{repo_id}")
        return

    ensure_dirs()
    console.print(
        f"Downloading full model [bold cyan]{repo_id}[/bold cyan] "
        f"-> [bold]{dest}[/bold]"
    )
    console.print("[dim]This may take a while for large models...[/dim]")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        console.print(f"[red]Download failed:[/red] {e}")
        return

    _register(model_name, dest, f"hf:{repo_id}")
    console.print(f"[green]✓[/green] [bold]{model_name}[/bold] downloaded to {dest}")


def _pull_url(url: str, name: str) -> None:
    """Download a model from a direct URL with resumable .part file support."""
    import requests

    filename = _stem_from_url(url) + ".gguf"
    dest      = MODELS_DIR / filename
    part_file = MODELS_DIR / (filename + ".part")

    if dest.exists():
        console.print(f"[yellow]Already downloaded:[/yellow] {name}")
        _register(name, dest, url)
        return

    ensure_dirs()

    # Determine how much we already have (from a prior interrupted download)
    already_have = part_file.stat().st_size if part_file.exists() else 0

    # HEAD request to get total file size for the disk space check
    try:
        head  = requests.head(url, allow_redirects=True, timeout=10)
        total = int(head.headers.get("content-length", 0))
    except Exception:
        total = 0

    remaining = max(0, total - already_have)
    if not _check_disk_space(MODELS_DIR, remaining):
        return

    # Build request — try to resume from where we left off
    headers: dict = {}
    if already_have:
        headers["Range"] = f"bytes={already_have}-"
        console.print(
            f"Resuming [bold cyan]{url}[/bold cyan] "
            f"[dim](skipping first {already_have / 1e6:.1f} MB)[/dim]"
        )
    else:
        console.print(f"Downloading [bold cyan]{url}[/bold cyan]")

    r = requests.get(url, headers=headers, stream=True, timeout=30)
    r.raise_for_status()

    # Server may ignore the Range header — detect and reset if needed
    if already_have and r.status_code == 200:
        # Server returned the full file despite Range request
        already_have = 0

    content_length = int(r.headers.get("content-length", 0))
    total_display  = (already_have + content_length) or None

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as prog:
        task = prog.add_task(filename, total=total_display, completed=already_have)
        mode = "ab" if already_have else "wb"
        with open(part_file, mode) as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
                prog.update(task, advance=len(chunk))

    # Atomically rename on successful completion
    part_file.rename(dest)
    _register(name, dest, url)
    console.print(f"[green]✓[/green] [bold]{name}[/bold] is ready")


def _register(name: str, path: Path, source: str = "local") -> None:
    reg = load_registry()
    reg[name] = {"path": str(path.resolve()), "source": source}
    save_registry(reg)


def remove_model(name: str) -> None:
    reg = load_registry()
    if name not in reg:
        console.print(f"[red]Not found:[/red] {name}")
        return
    path = Path(reg[name]["path"])
    # Only delete files that live inside ~/.localm/models/ — never touch
    # externally registered paths (Ollama blobs, user model dirs, etc.)
    owned = path.is_relative_to(MODELS_DIR) if hasattr(path, "is_relative_to") else \
            str(path).startswith(str(MODELS_DIR))
    if owned and path.exists():
        if path.is_dir():
            import shutil
            shutil.rmtree(path)
        else:
            path.unlink()
        console.print(f"[dim]Deleted {path}[/dim]")
    elif path.exists():
        console.print(f"[dim]Unregistered (file not deleted — lives outside ~/.localm/models)[/dim]")
    del reg[name]
    save_registry(reg)
    console.print(f"[green]✓[/green] Removed [bold]{name}[/bold]")


def _resolve_ollama_manifest(p: Path):
    """
    If p is an Ollama manifest directory, return (blob_path, suggested_name).
    Ollama layout: <root>/manifests/<registry>/<owner>/<model>/<tag>
                   <root>/blobs/sha256-<digest>
    Returns None if p doesn't look like an Ollama manifest.
    """
    import json as _json

    if not p.is_dir():
        return None

    tag_files = [f for f in p.iterdir() if f.is_file()]
    if not tag_files:
        return None

    for tag_file in tag_files:
        try:
            manifest = _json.loads(tag_file.read_text())
        except Exception:
            continue
        if "layers" not in manifest:
            continue

        for layer in manifest["layers"]:
            if layer.get("mediaType") != "application/vnd.ollama.image.model":
                continue

            blob_name = layer["digest"].replace(":", "-")  # sha256:abc -> sha256-abc

            # Walk up from manifest dir to find <root>/blobs/
            candidate = p
            blob_path = None
            for _ in range(6):
                candidate = candidate.parent
                maybe = candidate / "blobs" / blob_name
                if maybe.exists():
                    blob_path = maybe
                    break

            if blob_path is None:
                console.print(
                    f"[yellow]Ollama manifest found but blob missing:[/yellow] {blob_name}\n"
                    f"Expected a 'blobs' sibling directory near {p}"
                )
                return None

            tag        = tag_file.name               # e.g. "Q8_0"
            model_dir  = p.name                      # e.g. "Gemma-4-12B-it-AEON-..."
            suggested  = f"{model_dir}-{tag}".lower().replace(" ", "-")
            return blob_path, suggested

    return None


def add_local(path_str: str, name: Optional[str] = None) -> None:
    p = Path(path_str).resolve()
    if not p.exists():
        console.print(f"[red]Not found:[/red] {path_str}")
        return

    # Ollama manifest directory -> resolve to actual GGUF blob
    ollama = _resolve_ollama_manifest(p)
    if ollama is not None:
        blob_path, suggested = ollama
        model_name = name or suggested
        b = blob_path.stat().st_size
        size = f"{b/1e9:.2f} GB" if b >= 1e9 else f"{b/1e6:.0f} MB"
        _register(model_name, blob_path, "ollama")
        console.print(
            f"[green]✓[/green] Registered [bold]{model_name}[/bold] "
            f"[dim](Ollama GGUF blob, {size})[/dim]"
        )
        return

    is_gguf = p.suffix == ".gguf"
    is_hf   = p.is_dir() and (p / "config.json").exists()
    is_blob = p.is_file() and p.name.startswith("sha256-")  # raw Ollama blob by path

    if not (is_gguf or is_hf or is_blob):
        console.print(
            "[yellow]Warning:[/yellow] path doesn't look like a GGUF file, "
            "HuggingFace model directory, or Ollama blob."
        )

    model_name = name or p.stem
    kind = "hf" if is_hf else "local"
    _register(model_name, p, kind)
    tag = " [dim](HF format)[/dim]" if is_hf else ""
    console.print(f"[green]✓[/green] Registered [bold]{model_name}[/bold]{tag}")


def show_shortcuts() -> None:
    table = Table(title="Popular Models", header_style="bold cyan")
    table.add_column("Shortcut", style="bold")
    table.add_column("HuggingFace repo")
    table.add_column("Approx. size", justify="right", style="dim")

    for shortcut, spec in MODEL_SHORTCUTS.items():
        repo = spec.split(":")[0]
        table.add_row(shortcut, repo, _SHORTCUT_SIZES.get(shortcut, "?"))

    console.print(table)
    console.print("\n[dim]Download with:[/dim] [bold]localm pull <shortcut>[/bold]")
