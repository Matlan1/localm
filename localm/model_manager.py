import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional

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


# ------------------------------------------------------------------ #
#  Split GGUF (multi-part *-00001-of-00003.gguf files)                 #
# ------------------------------------------------------------------ #

# llama.cpp split naming convention: <stem>-00001-of-00003.gguf
_SPLIT_GGUF_RE = re.compile(
    r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE
)


def split_gguf_parts(filename: str) -> Optional[List[str]]:
    """
    If *filename* follows the llama.cpp split convention
    (``model-00001-of-00003.gguf``), return the full ordered list of part
    filenames. Returns None for regular single-file GGUFs.
    """
    m = _SPLIT_GGUF_RE.match(Path(filename).name)
    if not m:
        return None
    total = int(m.group("total"))
    if total < 2:
        return None
    stem = m.group("stem")
    return [f"{stem}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]


def first_split_part(filename: str) -> str:
    """Return the first-part filename for a split GGUF (llama.cpp wants this one)."""
    parts = split_gguf_parts(filename)
    return parts[0] if parts else filename


def missing_split_parts(first_part: Path) -> List[Path]:
    """
    Given the path of any part of a split GGUF, return sibling part paths
    that are missing on disk. Empty list means all parts present (or the
    file is not a split GGUF at all).
    """
    parts = split_gguf_parts(first_part.name)
    if not parts:
        return []
    return [first_part.parent / p for p in parts
            if not (first_part.parent / p).is_file()]


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
    # GGUF file — for split GGUFs, normalise to the first part (llama.cpp
    # needs the *-00001-of-N part to load the whole set)
    if direct.is_file() and direct.suffix == ".gguf":
        first = first_split_part(direct.name)
        if first != direct.name:
            first_path = direct.parent / first
            if first_path.is_file():
                return first_path, None
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


def pull_model(
    model_spec: str,
    name: Optional[str] = None,
    expected_sha256: Optional[str] = None,
    redownload: bool = False,
) -> None:
    spec = resolve_spec(model_spec)
    if spec.startswith("http://") or spec.startswith("https://"):
        _pull_url(spec, name or _stem_from_url(spec),
                  expected_sha256=expected_sha256, redownload=redownload)
    elif "/" in spec:
        if ":" in spec or spec.rsplit("/", 1)[-1].endswith(".gguf"):
            # owner/repo:file.gguf  or  owner/repo/file.gguf  -> single GGUF file
            _pull_gguf_file(spec, name, redownload=redownload)
        else:
            # owner/repo  (no filename) -> full HuggingFace snapshot
            _pull_hf_snapshot(spec, name, redownload=redownload)
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


def _hf_file_sha256(repo_id: str, filename: str) -> Optional[str]:
    """
    Ask the HuggingFace API for a file's LFS sha256 without downloading it.
    Returns None when offline, on any API error, or for non-LFS files.
    """
    try:
        from huggingface_hub import HfApi
        info = HfApi().get_paths_info(repo_id, [filename])
        if info:
            lfs = getattr(info[0], "lfs", None)
            digest = getattr(lfs, "sha256", None) if lfs else None
            return digest.lower() if digest else None
    except Exception:
        pass
    return None


def _prompt_predownload_dup(dup_names: List[str], model_name: str) -> str:
    """
    The exact file about to be downloaded already exists locally.
    Returns "alias", "download", or "skip". No TTY → "skip".
    """
    import click

    names = ", ".join(f"'{n}'" for n in dup_names)
    console.print(
        f"[yellow]You already have this exact file — registered as "
        f"{names}[/yellow] [dim](sha256 match via HF metadata)[/dim]"
    )
    if not sys.stdin.isatty():
        console.print("[dim]Non-interactive session — skipping download. "
                      "Use --redownload to force.[/dim]")
        return "skip"
    choice = click.prompt(
        f"  [a]lias as '{model_name}'  [d]ownload anyway  [s]kip",
        type=click.Choice(["a", "d", "s"], case_sensitive=False),
        default="a",
        show_choices=False,
    )
    return {"a": "alias", "d": "download", "s": "skip"}[choice.lower()]


def _pull_gguf_file(spec: str, name: Optional[str], redownload: bool = False) -> None:
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

    # Split GGUF: normalise to the full ordered part list. llama.cpp loads
    # the model from the first part, so that's what gets registered.
    all_parts = split_gguf_parts(filename) or [filename]
    filename  = all_parts[0]

    model_name = name or filename.removesuffix(".gguf")
    dest = MODELS_DIR / filename

    # Expected digest from HF metadata — free, no download needed.
    # (Only identifies the first part of a split GGUF, which is enough.)
    expected = _hf_file_sha256(repo_id, filename)

    missing = [p for p in all_parts if not (MODELS_DIR / p).exists()]
    if not missing:
        console.print(f"[yellow]Already downloaded:[/yellow] {filename}")
        _register_with_dedup(model_name, dest, f"hf:{repo_id}", digest=expected)
        return

    # Pre-download duplicate check: same bytes already on disk elsewhere?
    if expected and not redownload:
        dups = find_by_sha256(expected)
        if dups:
            action = _prompt_predownload_dup(dups, model_name)
            if action == "skip":
                return
            if action == "alias":
                alias_model(dups[0], model_name)
                return
            # "download" falls through

    ensure_dirs()

    # Disk space pre-flight — HEAD each missing part's CDN URL for Content-Length
    try:
        import requests as _req
        total_size = 0
        for part in missing:
            cdn_url = hf_hub_url(repo_id, part)
            head    = _req.head(cdn_url, allow_redirects=True, timeout=10)
            total_size += int(head.headers.get("content-length", 0))
    except Exception:
        total_size = 0

    if not _check_disk_space(MODELS_DIR, total_size):
        return

    if len(all_parts) > 1:
        console.print(
            f"Pulling [bold cyan]{repo_id}[/bold cyan] / [bold]{filename}[/bold] "
            f"[dim](split GGUF, {len(all_parts)} parts, "
            f"{len(missing)} to download)[/dim]"
        )
    else:
        console.print(f"Pulling [bold cyan]{repo_id}[/bold cyan] / [bold]{filename}[/bold]")

    for part in missing:
        try:
            local = hf_hub_download(
                repo_id=repo_id,
                filename=part,
                local_dir=str(MODELS_DIR),
            )
            final = MODELS_DIR / part
            if Path(local) != final:
                shutil.move(local, final)
        except Exception as e:
            console.print(f"[red]Download failed[/red] ({part}): {e}")
            return

    _register(model_name, MODELS_DIR / filename, f"hf:{repo_id}", sha256=expected)
    console.print(f"[green]✓[/green] [bold]{model_name}[/bold] is ready")


def _pull_hf_snapshot(repo_id: str, name: Optional[str], redownload: bool = False) -> None:
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
        _register_with_dedup(model_name, dest, f"hf:{repo_id}")
        return

    # Same repo already pulled under a different name?
    if not redownload:
        reg = load_registry()
        same_source = sorted(
            n for n, info in reg.items()
            if info.get("source") == f"hf:{repo_id}" and Path(info.get("path", "")).is_dir()
        )
        if same_source:
            console.print(
                f"[yellow]This repo is already downloaded — registered as "
                f"{', '.join(repr(n) for n in same_source)}[/yellow]"
            )
            if not sys.stdin.isatty():
                console.print("[dim]Non-interactive session — skipping. "
                              "Use --redownload to force.[/dim]")
                return
            import click
            choice = click.prompt(
                f"  [a]lias as '{model_name}'  [d]ownload anyway  [s]kip",
                type=click.Choice(["a", "d", "s"], case_sensitive=False),
                default="a", show_choices=False,
            )
            if choice.lower() == "s":
                return
            if choice.lower() == "a":
                alias_model(same_source[0], model_name)
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
        )
    except Exception as e:
        console.print(f"[red]Download failed:[/red] {e}")
        return

    _register(model_name, dest, f"hf:{repo_id}")
    console.print(f"[green]✓[/green] [bold]{model_name}[/bold] downloaded to {dest}")


def _sha256_file(path: Path) -> str:
    """Return the hex SHA256 digest of a file."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ #
#  Model identity — duplicate detection (two-tier: path, then sha256)  #
# ------------------------------------------------------------------ #

def find_aliases_by_path(path: Path, reg: Optional[dict] = None) -> List[str]:
    """Registered names whose path resolves to the same file/dir as *path*."""
    reg = reg if reg is not None else load_registry()
    target = str(Path(path).resolve())
    return sorted(
        name for name, info in reg.items()
        if str(Path(info.get("path", "")).resolve()) == target
    )


def find_by_sha256(digest: str, reg: Optional[dict] = None) -> List[str]:
    """Registered names whose stored sha256 matches *digest* (case-insensitive)."""
    if not digest:
        return []
    reg = reg if reg is not None else load_registry()
    d = digest.lower()
    return sorted(
        name for name, info in reg.items()
        if info.get("sha256", "").lower() == d
    )


def _hash_with_notice(path: Path) -> Optional[str]:
    """
    SHA256 a model file, telling the user why the wait is happening.
    Returns None for directories (HF models are identified by path only).
    """
    if not path.is_file():
        return None
    size_gb = path.stat().st_size / 1e9
    if size_gb > 0.5:
        console.print(
            f"[dim]Hashing {path.name} ({size_gb:.1f} GB) for duplicate "
            f"detection — one-time cost, stored in the registry…[/dim]"
        )
    return _sha256_file(path)


def alias_model(existing: str, new_name: str) -> bool:
    """
    Register *new_name* as an additional name for *existing* (same file,
    same source, same digest). Returns False when *existing* is unknown
    or *new_name* is already taken.
    """
    reg = load_registry()
    if existing not in reg:
        console.print(f"[red]Not found:[/red] {existing}")
        return False
    if new_name in reg:
        console.print(f"[red]Name already in use:[/red] {new_name}")
        return False
    reg[new_name] = dict(reg[existing])
    save_registry(reg)
    console.print(
        f"[green]✓[/green] [bold]{new_name}[/bold] is now an alias of "
        f"[bold]{existing}[/bold]"
    )
    return True


def _prompt_duplicate_action(existing_names: List[str], reason: str) -> str:
    """
    Ask the user what to do about a duplicate model.

    Returns one of: "alias", "copy", "move", "register", "skip".
    Non-interactive sessions (no TTY) default to "skip" so scripts never
    silently create duplicate entries.
    """
    import click

    names = ", ".join(f"'{n}'" for n in existing_names)
    console.print(
        f"[yellow]This model is already registered as {names}[/yellow] "
        f"[dim]({reason})[/dim]"
    )
    if not sys.stdin.isatty():
        console.print("[dim]Non-interactive session — skipping. "
                      "Use 'localm alias' to add a name for it.[/dim]")
        return "skip"

    choice = click.prompt(
        "  [a]lias (new name, same file)  [c]opy into ~/.localm/models  "
        "[m]ove into ~/.localm/models  [r]egister anyway  [s]kip",
        type=click.Choice(["a", "c", "m", "r", "s"], case_sensitive=False),
        default="a",
        show_choices=False,
    )
    return {"a": "alias", "c": "copy", "m": "move",
            "r": "register", "s": "skip"}[choice.lower()]


def _pull_url(
    url: str,
    name: str,
    expected_sha256: Optional[str] = None,
    redownload: bool = False,
) -> None:
    """Download a model from a direct URL with resumable .part file support."""
    import requests

    filename = _stem_from_url(url) + ".gguf"
    dest      = MODELS_DIR / filename
    part_file = MODELS_DIR / (filename + ".part")

    if dest.exists():
        console.print(f"[yellow]Already downloaded:[/yellow] {filename}")
        _register_with_dedup(name, dest, url)
        return

    # Pre-download check by user-supplied hash (URL servers can't tell us one)
    if expected_sha256 and not redownload:
        dups = find_by_sha256(expected_sha256)
        if dups:
            action = _prompt_predownload_dup(dups, name)
            if action == "skip":
                return
            if action == "alias":
                alias_model(dups[0], name)
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

    # SHA256 verification
    actual = _sha256_file(dest)
    if expected_sha256:
        if actual.lower() == expected_sha256.lower():
            console.print(f"[green]✓[/green] SHA256 verified: {actual[:16]}…")
        else:
            console.print(
                f"[red]SHA256 mismatch![/red] Expected {expected_sha256[:16]}…, "
                f"got {actual[:16]}… — deleting corrupted file"
            )
            dest.unlink()
            return
    else:
        console.print(f"[dim]SHA256: {actual}[/dim]")

    # Post-download identity check: did we just download a byte-identical
    # copy of something already registered? (URL downloads can't know the
    # hash up front, so this is the earliest possible detection point.)
    dups = [n for n in find_by_sha256(actual) if n != name]
    if dups and not redownload:
        names = ", ".join(f"'{n}'" for n in dups)
        console.print(
            f"[yellow]Downloaded file is byte-identical to {names}[/yellow]"
        )
        if sys.stdin.isatty():
            import click
            choice = click.prompt(
                f"  [a]lias as '{name}' and delete the duplicate file  "
                "[k]eep both copies",
                type=click.Choice(["a", "k"], case_sensitive=False),
                default="a", show_choices=False,
            )
            if choice.lower() == "a":
                existing_path = Path(load_registry()[dups[0]]["path"])
                if dest.resolve() != existing_path.resolve():
                    dest.unlink()
                alias_model(dups[0], name)
                return

    _register(name, dest, url, sha256=actual)
    console.print(f"[green]✓[/green] [bold]{name}[/bold] is ready")


def _register(
    name: str,
    path: Path,
    source: str = "local",
    sha256: Optional[str] = None,
) -> None:
    reg = load_registry()
    entry = {"path": str(path.resolve()), "source": source}
    if sha256:
        entry["sha256"] = sha256.lower()
    reg[name] = entry
    save_registry(reg)


def _register_with_dedup(
    model_name: str,
    p: Path,
    source: str,
    *,
    on_duplicate: str = "ask",
    digest: Optional[str] = None,
) -> None:
    """
    Register a model, detecting duplicates first.

    Two-tier identity: resolved path (instant), then stored sha256 when a
    *digest* for the new file is known. ``on_duplicate`` is one of
    "ask" / "alias" / "copy" / "move" / "register" / "skip" — "ask" prompts
    interactively and degrades to "skip" without a TTY.
    """
    import click

    reg = load_registry()
    aliases = find_aliases_by_path(p, reg)

    # Same name, same file — true no-op (but backfill a fresh digest)
    if model_name in aliases:
        console.print(
            f"[yellow]'{model_name}' is already registered for this exact "
            f"file[/yellow] [dim]({p})[/dim]"
        )
        if digest and not reg[model_name].get("sha256"):
            reg[model_name]["sha256"] = digest.lower()
            save_registry(reg)
        others = [a for a in aliases if a != model_name]
        if others:
            console.print(f"[dim]Also registered as: {', '.join(others)}[/dim]")
        return

    # Same name, DIFFERENT file — real conflict, never overwrite silently
    if model_name in reg:
        old_path = reg[model_name].get("path", "?")
        console.print(
            f"[yellow]'{model_name}' already points to a different file:"
            f"[/yellow] {old_path}"
        )
        if sys.stdin.isatty():
            if not click.confirm(f"  Overwrite '{model_name}' with {p}?"):
                console.print("[dim]Skipped.[/dim]")
                return
        else:
            console.print("[dim]Non-interactive session — skipped. "
                          "Pick another name with -n.[/dim]")
            return

    # Duplicate content under other names? (path tier, then hash tier)
    dup_names, reason = aliases, "same file"
    if not dup_names and digest:
        dup_names = find_by_sha256(digest, reg)
        reason = "byte-identical content"

    if dup_names:
        action = (
            _prompt_duplicate_action(dup_names, reason)
            if on_duplicate == "ask" else on_duplicate
        )
        if action == "skip":
            console.print("[dim]Skipped.[/dim]")
            return
        if action == "alias":
            alias_model(dup_names[0], model_name)
            return
        if action in ("copy", "move"):
            ensure_dirs()
            dest = MODELS_DIR / p.name
            if dest.exists() and dest.resolve() != p.resolve():
                console.print(
                    f"[red]Cannot {action}:[/red] {dest} already exists"
                )
                return
            if action == "copy":
                console.print(f"[dim]Copying to {dest}…[/dim]")
                shutil.copy2(p, dest)
            else:
                console.print(f"[dim]Moving to {dest}…[/dim]")
                shutil.move(str(p), str(dest))
                # Keep other aliases of the old path working
                moved_from = str(p.resolve())
                for alias_name in dup_names:
                    if reg.get(alias_name, {}).get("path") == moved_from:
                        reg[alias_name]["path"] = str(dest.resolve())
                save_registry(reg)
            p = dest
        # action == "register" falls through unchanged

    _register(model_name, p, source, sha256=digest)
    console.print(f"[green]✓[/green] Registered [bold]{model_name}[/bold]")


def remove_model(name: str) -> None:
    reg = load_registry()
    if name not in reg:
        console.print(f"[red]Not found:[/red] {name}")
        return
    path = Path(reg[name]["path"])

    # Alias-aware: if other names still point at this file, only unregister
    # this name — never delete a file out from under another alias.
    other_aliases = [a for a in find_aliases_by_path(path, reg) if a != name]
    if other_aliases:
        del reg[name]
        save_registry(reg)
        console.print(
            f"[green]✓[/green] Removed [bold]{name}[/bold] "
            f"[dim](file kept — still registered as: "
            f"{', '.join(other_aliases)})[/dim]"
        )
        return

    # Only delete files that live inside ~/.localm/models/ — never touch
    # externally registered paths (Ollama blobs, user model dirs, etc.)
    owned = path.is_relative_to(MODELS_DIR) if hasattr(path, "is_relative_to") else \
            str(path).startswith(str(MODELS_DIR))
    if owned and path.exists():
        if path.is_dir():
            import shutil
            shutil.rmtree(path)
            console.print(f"[dim]Deleted {path}[/dim]")
        else:
            # Split GGUF: remove every sibling part, not just the registered one
            siblings = split_gguf_parts(path.name) or [path.name]
            for part in siblings:
                part_path = path.parent / part
                if part_path.exists():
                    part_path.unlink()
                    console.print(f"[dim]Deleted {part_path}[/dim]")
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


def add_local(
    path_str: str,
    name: Optional[str] = None,
    on_duplicate: str = "ask",
    no_hash: bool = False,
) -> None:
    p = Path(path_str).resolve()
    if not p.exists():
        console.print(f"[red]Not found:[/red] {path_str}")
        return

    # Ollama manifest directory -> resolve to actual GGUF blob
    ollama = _resolve_ollama_manifest(p)
    if ollama is not None:
        blob_path, suggested = ollama
        model_name = name or suggested
        # Ollama blob filenames already ARE the sha256 digest — store it free
        digest = blob_path.name.removeprefix("sha256-") \
            if blob_path.name.startswith("sha256-") else None
        _register_with_dedup(
            model_name, blob_path, "ollama",
            on_duplicate=on_duplicate, digest=digest,
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

    if is_blob:
        digest = p.name.removeprefix("sha256-")
    elif no_hash or p.is_dir():
        digest = None   # HF dirs are identified by path only
    else:
        # Hash when it can change the outcome: unknown path (content-tier
        # check) or known path missing its digest (lazy backfill)
        reg = load_registry()
        already_known = find_aliases_by_path(p, reg)
        needs_backfill = any(
            not reg[n].get("sha256") for n in already_known
        )
        digest = None
        if not already_known or needs_backfill:
            digest = _hash_with_notice(p)

    _register_with_dedup(
        model_name, p, kind, on_duplicate=on_duplicate, digest=digest,
    )


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
