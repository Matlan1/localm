import contextlib
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import List, NamedTuple, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.progress import (
    BarColumn, DownloadColumn, Progress,
    TextColumn, TimeRemainingColumn, TransferSpeedColumn,
)
from rich.table import Table

from .config import (
    HOME_DIR,
    MODELS_DIR,
    REGISTRY_FILE,
    ensure_dirs,
    load_config,
    load_registry,
    save_registry,
    update_registry,
)

console = Console()

# GUI download progress: the GUI runs ``localm pull`` as a subprocess and
# parses its stdout. When LOCALM_PROGRESS_JSON=1 the downloader streams
# structured progress lines (this sentinel + JSON) that the GUI renders as a
# progress bar; interactive CLI use keeps huggingface_hub's own tqdm bars.
PROGRESS_SENTINEL = "\x1flocalm-progress\x1f"


def _emit_progress(downloaded: int, total: int, *, phase: str = "download") -> None:
    pct = round(downloaded * 100 / total, 1) if total else None
    sys.stdout.write(
        PROGRESS_SENTINEL
        + json.dumps({"phase": phase, "downloaded": downloaded,
                      "total": total, "pct": pct})
        + "\n"
    )
    sys.stdout.flush()


@contextlib.contextmanager
def _download_progress(target_parts: List[Path], total_size: int):
    """Stream JSON download progress while files land under MODELS_DIR.

    Active only in GUI mode (LOCALM_PROGRESS_JSON=1) with a known total - a
    no-op otherwise, so the CLI keeps huggingface_hub's tqdm bars. Progress is
    measured from bytes on disk (completed parts + the growing ``.incomplete``
    temp file), which is robust across huggingface_hub versions.
    """
    if os.environ.get("LOCALM_PROGRESS_JSON") != "1" or not total_size:
        yield
        return

    stop = threading.Event()

    def _downloaded_bytes() -> int:
        done = 0
        for p in target_parts:
            try:
                done += p.stat().st_size
            except OSError:
                pass  # not finished yet
        active = 0
        cache = MODELS_DIR / ".cache"
        if cache.is_dir():
            try:
                for f in cache.rglob("*.incomplete"):
                    try:
                        active += f.stat().st_size
                    except OSError:
                        pass
            except OSError:
                pass
        return min(done + active, total_size)

    def _poll() -> None:
        last = -1
        while not stop.is_set():
            dl = _downloaded_bytes()
            if dl != last:
                last = dl
                _emit_progress(dl, total_size)
            stop.wait(0.7)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    _emit_progress(0, total_size)
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=2)
        _emit_progress(total_size, total_size)


@contextlib.contextmanager
def _snapshot_progress(disk_bytes_fn, total_size: int):
    """Like _download_progress but for snapshot_download (many files): byte
    count comes from a caller-supplied directory-size function. Indeterminate
    (total_size == 0) still streams a 'downloading' phase so the GUI can show
    a busy bar; no-op outside GUI mode."""
    if os.environ.get("LOCALM_PROGRESS_JSON") != "1":
        yield
        return

    stop = threading.Event()

    def _poll() -> None:
        last = -1
        while not stop.is_set():
            dl = disk_bytes_fn()
            if total_size:
                dl = min(dl, total_size)
            if dl != last:
                last = dl
                _emit_progress(dl, total_size)
            stop.wait(0.7)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    _emit_progress(0, total_size)
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=2)
        if total_size:
            _emit_progress(total_size, total_size)


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

    # HF model directory (config.json plus real weights/tokenizer, so the data
    # dir's own config.json is not mistaken for a model)
    from localm.inference.engine import _is_hf_dir
    if _is_hf_dir(str(direct)):
        return direct, None
    # GGUF file - for split GGUFs, normalise to the first part (llama.cpp
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
            name_cell = name
        elif path.exists():
            kind = "gguf"
            b = path.stat().st_size
            size = f"{b/1024**3:.2f} GB" if b >= 1024**3 else f"{b/1024**2:.0f} MB"
            name_cell = name
        else:
            # File is gone: flagged (kept) unless autoprune deleted it earlier.
            kind = "[red]missing[/red]"
            size = "[red]-[/red]"
            name_cell = f"[red]{name}[/red]"

        table.add_row(name_cell, kind, size, source, str(path))

    console.print(table)

    if any(not Path(i["path"]).exists() for i in reg.values()):
        console.print(
            "[dim]Models marked [red]missing[/red] have no file on disk. "
            "Set [bold]autoprune_missing_models true[/bold] to drop them "
            "automatically, or re-add the file.[/dim]"
        )


def pull_model(
    model_spec: str,
    name: Optional[str] = None,
    expected_sha256: Optional[str] = None,
    redownload: bool = False,
) -> bool:
    """Download a model from HuggingFace or a URL.

    Returns True on success or a benign no-op (already present / aliased /
    user-skipped), False on a real error, so callers can set a non-zero exit
    code and the GUI can mark the job failed instead of reporting "finished".
    """
    spec = resolve_spec(model_spec)
    if spec.startswith("http://") or spec.startswith("https://"):
        return _pull_url(spec, name or _stem_from_url(spec),
                         expected_sha256=expected_sha256, redownload=redownload)
    elif "/" in spec:
        if ":" in spec or spec.rsplit("/", 1)[-1].endswith(".gguf"):
            # owner/repo:file.gguf  or  owner/repo/file.gguf  -> single GGUF file
            return _pull_gguf_file(spec, name, redownload=redownload)
        else:
            # owner/repo  (no filename) -> full HuggingFace snapshot
            return _pull_hf_snapshot(spec, name, redownload=redownload)
    else:
        console.print(f"[red]Unknown spec:[/red] {model_spec}")
        console.print("Formats:")
        console.print("  [bold]owner/repo[/bold]              full HF model directory")
        console.print("  [bold]owner/repo:file.gguf[/bold]   single GGUF file")
        console.print("  [bold]https://...[/bold]             direct URL")
        console.print("Run [bold]localm models[/bold] for GGUF shortcuts.")
        return False


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
            need_gb  = required_bytes / 1024**3
            free_gb  = usage.free / 1024**3
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
        f"[yellow]You already have this exact file - registered as "
        f"{names}[/yellow] [dim](sha256 match via HF metadata)[/dim]"
    )
    if not sys.stdin.isatty():
        console.print("[dim]Non-interactive session - skipping download. "
                      "Use --redownload to force.[/dim]")
        return "skip"
    choice = click.prompt(
        f"  [a]lias as '{model_name}'  [d]ownload anyway  [s]kip",
        type=click.Choice(["a", "d", "s"], case_sensitive=False),
        default="a",
        show_choices=False,
    )
    return {"a": "alias", "d": "download", "s": "skip"}[choice.lower()]


def _pull_gguf_file(spec: str, name: Optional[str], redownload: bool = False) -> bool:
    """Download a single .gguf file from a HuggingFace repo."""
    try:
        from huggingface_hub import hf_hub_download, hf_hub_url
    except ImportError:
        console.print("[red]Missing:[/red] huggingface-hub  (run: uv pip install huggingface-hub)")
        return False

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

    # Expected digest from HF metadata - free, no download needed.
    # (Only identifies the first part of a split GGUF, which is enough.)
    expected = _hf_file_sha256(repo_id, filename)

    missing = [p for p in all_parts if not (MODELS_DIR / p).exists()]
    if not missing:
        console.print(f"[yellow]Already downloaded:[/yellow] {filename}")
        _register_with_dedup(model_name, dest, f"hf:{repo_id}", digest=expected)
        return True

    # Pre-download duplicate check: same bytes already on disk elsewhere?
    if expected and not redownload:
        dups = find_by_sha256(expected)
        if dups:
            action = _prompt_predownload_dup(dups, model_name)
            if action == "skip":
                return True
            if action == "alias":
                alias_model(dups[0], model_name)
                return True
            # "download" falls through

    ensure_dirs()

    # Disk space pre-flight - HEAD each missing part's CDN URL for Content-Length
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
        return False

    if len(all_parts) > 1:
        console.print(
            f"Pulling [bold cyan]{repo_id}[/bold cyan] / [bold]{filename}[/bold] "
            f"[dim](split GGUF, {len(all_parts)} parts, "
            f"{len(missing)} to download)[/dim]"
        )
    else:
        console.print(f"Pulling [bold cyan]{repo_id}[/bold cyan] / [bold]{filename}[/bold]")

    with _download_progress([MODELS_DIR / p for p in missing], total_size):
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
                return False

    _register(model_name, MODELS_DIR / filename, f"hf:{repo_id}", sha256=expected)
    console.print(f"[green]✓[/green] [bold]{model_name}[/bold] is ready")
    return True


def _pull_hf_snapshot(repo_id: str, name: Optional[str], redownload: bool = False) -> bool:
    """Download a complete HuggingFace model repo (for transformers/HF format models)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        console.print("[red]Missing:[/red] huggingface-hub  (run: uv pip install huggingface-hub)")
        return False

    model_name = name or repo_id.split("/")[-1]
    dest = MODELS_DIR / model_name

    if dest.exists() and (dest / "config.json").exists():
        console.print(f"[yellow]Already downloaded:[/yellow] {model_name}")
        _register_with_dedup(model_name, dest, f"hf:{repo_id}")
        return True

    # Same repo already pulled under a different name?
    if not redownload:
        reg = load_registry()
        same_source = sorted(
            n for n, info in reg.items()
            if info.get("source") == f"hf:{repo_id}" and Path(info.get("path", "")).is_dir()
        )
        if same_source:
            console.print(
                f"[yellow]This repo is already downloaded - registered as "
                f"{', '.join(repr(n) for n in same_source)}[/yellow]"
            )
            if not sys.stdin.isatty():
                console.print("[dim]Non-interactive session - skipping. "
                              "Use --redownload to force.[/dim]")
                return True
            import click
            choice = click.prompt(
                f"  [a]lias as '{model_name}'  [d]ownload anyway  [s]kip",
                type=click.Choice(["a", "d", "s"], case_sensitive=False),
                default="a", show_choices=False,
            )
            if choice.lower() == "s":
                return True
            if choice.lower() == "a":
                alias_model(same_source[0], model_name)
                return True

    ensure_dirs()
    console.print(
        f"Downloading full model [bold cyan]{repo_id}[/bold cyan] "
        f"-> [bold]{dest}[/bold]"
    )
    console.print("[dim]This may take a while for large models...[/dim]")

    # Sum the repo's file sizes so the GUI gets a real percentage (best effort)
    total_size = 0
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id, files_metadata=True)
        total_size = sum(getattr(s, "size", None) or 0 for s in info.siblings)
    except Exception:
        total_size = 0

    def _disk_bytes() -> int:
        try:
            return sum(f.stat().st_size for f in dest.rglob("*")
                       if f.is_file() and ".cache" not in f.parts)
        except OSError:
            return 0

    try:
        with _snapshot_progress(_disk_bytes, total_size):
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(dest),
            )
    except Exception as e:
        console.print(f"[red]Download failed:[/red] {e}")
        return False

    _register(model_name, dest, f"hf:{repo_id}")
    console.print(f"[green]✓[/green] [bold]{model_name}[/bold] downloaded to {dest}")
    return True


def _sha256_file(path: Path) -> str:
    """Return the hex SHA256 digest of a file."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ #
#  Model identity - duplicate detection (two-tier: path, then sha256)  #
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
    size_gb = path.stat().st_size / 1024**3
    if size_gb > 0.5:
        console.print(
            f"[dim]Hashing {path.name} ({size_gb:.1f} GB) for duplicate "
            f"detection - one-time cost, stored in the registry…[/dim]"
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
        console.print("[dim]Non-interactive session - skipping. "
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
) -> bool:
    """Download a model from a direct URL with resumable .part file support."""
    import requests

    stem = _stem_from_url(url)
    if not stem:
        console.print(
            f"[red]Invalid URL - no file name in the path:[/red] {url}\n"
            "A direct URL must point at a file, e.g. https://host/model.gguf"
        )
        return False

    filename = stem + ".gguf"
    dest      = MODELS_DIR / filename
    part_file = MODELS_DIR / (filename + ".part")

    if dest.exists():
        console.print(f"[yellow]Already downloaded:[/yellow] {filename}")
        _register_with_dedup(name, dest, url)
        return True

    # Pre-download check by user-supplied hash (URL servers can't tell us one)
    if expected_sha256 and not redownload:
        dups = find_by_sha256(expected_sha256)
        if dups:
            action = _prompt_predownload_dup(dups, name)
            if action == "skip":
                return True
            if action == "alias":
                alias_model(dups[0], name)
                return True

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
        return False

    # Build request - try to resume from where we left off
    headers: dict = {}
    if already_have:
        headers["Range"] = f"bytes={already_have}-"
        console.print(
            f"Resuming [bold cyan]{url}[/bold cyan] "
            f"[dim](skipping first {already_have / 1024**2:.1f} MB)[/dim]"
        )
    else:
        console.print(f"Downloading [bold cyan]{url}[/bold cyan]")

    try:
        r = requests.get(url, headers=headers, stream=True, timeout=30)
        r.raise_for_status()
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", "?")
        console.print(f"[red]Download failed[/red] (HTTP {code}): {url}")
        return False
    except requests.RequestException as e:
        console.print(f"[red]Could not reach[/red] {url}: {e}")
        return False

    # Server may ignore the Range header - detect and reset if needed
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
                f"got {actual[:16]}… - deleting corrupted file"
            )
            dest.unlink()
            return False
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
                return True

    _register(name, dest, url, sha256=actual)
    console.print(f"[green]✓[/green] [bold]{name}[/bold] is ready")
    return True


def _register(
    name: str,
    path: Path,
    source: str = "local",
    sha256: Optional[str] = None,
) -> None:
    entry = {"path": str(path.resolve()), "source": source}
    if sha256:
        entry["sha256"] = sha256.lower()
    # Atomic read-modify-write so a concurrent registry writer (GUI thread,
    # a parallel pull, sync_models_dir) can't clobber this entry.
    update_registry(lambda reg: reg.__setitem__(name, entry))


def _unique_registry_name(reg: dict, base: str) -> str:
    """Return a registry-safe name derived from ``base``, avoiding collisions."""
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-") or "model"
    if base not in reg:
        return base
    i = 2
    while f"{base}-{i}" in reg:
        i += 1
    return f"{base}-{i}"


class ModelSyncResult(NamedTuple):
    """Outcome of :func:`sync_models_dir`."""

    added: int = 0          # new models discovered and registered
    flagged: int = 0        # entries newly marked missing (file gone)
    restored: int = 0       # entries whose file reappeared (flag cleared)
    pruned: int = 0         # entries deleted (only when autoprune is enabled)
    note: str = ""          # a warning to surface (e.g. autoprune guardrail tripped)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.flagged or self.restored or self.pruned)


def _backup_registry() -> Optional[Path]:
    """Snapshot the registry to ``registry.json.bak`` (one-step revert).

    Returns the backup path, or None if there was no registry file to copy.
    """
    if not REGISTRY_FILE.exists():
        return None
    backup = REGISTRY_FILE.with_name(REGISTRY_FILE.name + ".bak")
    try:
        shutil.copy2(REGISTRY_FILE, backup)
        return backup
    except OSError:
        return None


def sync_models_dir(prune: Optional[bool] = None) -> ModelSyncResult:
    """Reconcile the registry with the models directory.

    Scans ``MODELS_DIR`` for models that aren't registered yet - loose GGUF
    files (split GGUFs are registered by their first part) and HuggingFace
    directories (any subfolder containing ``config.json``) - and registers them.

    Registry entries whose file has gone missing are, by default, **flagged**
    (``"missing": true``) rather than deleted, so a temporarily-unavailable model
    (moved file, unplugged drive, sync hiccup) is not silently forgotten. When a
    flagged file reappears, the flag is cleared. Deletion happens only when
    pruning is enabled - via ``prune=True`` or the ``autoprune_missing_models``
    config setting - and even then only for files under ``MODELS_DIR`` (external
    models are flagged, never deleted). Runs without prompting; safe to call on
    every launch.
    """
    if prune is None:
        prune = bool(load_config().get("autoprune_missing_models", False))

    ensure_dirs()
    reg = load_registry()

    known = {
        str(Path(entry["path"]).resolve())
        for entry in reg.values()
        if entry.get("path")
    }

    added = 0
    if MODELS_DIR.is_dir():
        for child in sorted(MODELS_DIR.iterdir()):
            try:
                # HuggingFace model directory.
                if child.is_dir() and (child / "config.json").is_file():
                    resolved = str(child.resolve())
                    if resolved in known:
                        continue
                    _register(_unique_registry_name(reg, child.name), child)
                    reg = load_registry()
                    known.add(resolved)
                    added += 1
            except OSError:
                continue

        for child in sorted(MODELS_DIR.glob("*.gguf")):
            try:
                if not child.is_file():
                    continue
                # For split GGUFs, only register the first part.
                parts = split_gguf_parts(child.name)
                if parts and child.name != parts[0]:
                    continue
                resolved = str(child.resolve())
                if resolved in known:
                    continue
                _register(_unique_registry_name(reg, child.stem), child)
                reg = load_registry()
                known.add(resolved)
                added += 1
            except OSError:
                continue

    # Reconcile missing / restored / pruned against the (possibly grown) registry.
    reg = load_registry()
    models_root = MODELS_DIR.resolve()

    def _under_models_dir(p: Path) -> bool:
        return models_root in p.resolve().parents

    # Managed models = those whose file lives under the models folder.
    managed = [
        name
        for name, entry in reg.items()
        if entry.get("path") and _under_models_dir(Path(entry["path"]))
    ]
    managed_missing = [n for n in managed if not Path(reg[n]["path"]).exists()]

    # Guardrail: if pruning would delete *every* managed model at once, the folder
    # is almost certainly unavailable (unmounted drive, wrong path) rather than the
    # user having deleted everything - refuse to prune and flag instead.
    suspicious = prune and len(managed) >= 2 and len(managed_missing) == len(managed)

    flagged = restored = pruned = 0
    note = ""
    backed_up = False
    dirty = False

    for name in list(reg.keys()):
        entry = reg[name]
        path_str = entry.get("path")
        if not path_str:
            continue
        path = Path(path_str)

        if path.exists():
            # A previously-missing model is back - clear the flag.
            if entry.pop("missing", None):
                restored += 1
                dirty = True
            continue

        # File is gone.
        if prune and not suspicious and _under_models_dir(path):
            if not backed_up:
                # Snapshot the registry before the first deletion so the sync can
                # be reverted one step if it goes wrong.
                _backup_registry()
                backed_up = True
            del reg[name]
            pruned += 1
            dirty = True
        elif not entry.get("missing"):
            entry["missing"] = True
            flagged += 1
            dirty = True

    if suspicious:
        note = (
            f"Skipped autoprune: all {len(managed)} models under the models folder "
            "appear missing - is the folder/drive available? Left them flagged "
            "rather than deleting the registry."
        )

    if dirty:
        save_registry(reg)

    return ModelSyncResult(
        added=added, flagged=flagged, restored=restored, pruned=pruned, note=note
    )


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
    "ask" / "alias" / "copy" / "move" / "register" / "skip" - "ask" prompts
    interactively and degrades to "skip" without a TTY.
    """
    import click

    reg = load_registry()
    aliases = find_aliases_by_path(p, reg)

    # Same name, same file - true no-op (but backfill a fresh digest)
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

    # Same name, DIFFERENT file - real conflict, never overwrite silently
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
            console.print("[dim]Non-interactive session - skipped. "
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
    # this name - never delete a file out from under another alias.
    other_aliases = [a for a in find_aliases_by_path(path, reg) if a != name]
    if other_aliases:
        del reg[name]
        save_registry(reg)
        console.print(
            f"[green]✓[/green] Removed [bold]{name}[/bold] "
            f"[dim](file kept - still registered as: "
            f"{', '.join(other_aliases)})[/dim]"
        )
        return

    # Only delete files that live inside ~/.localm/models/ - never touch
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
        console.print(f"[dim]Unregistered (file not deleted - lives outside ~/.localm/models)[/dim]")
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
        # Ollama blob filenames already ARE the sha256 digest - store it free
        digest = blob_path.name.removeprefix("sha256-") \
            if blob_path.name.startswith("sha256-") else None
        _register_with_dedup(
            model_name, blob_path, "ollama",
            on_duplicate=on_duplicate, digest=digest,
        )
        return

    # Refuse the localm data directory (and its models root): its config.json is
    # the app's settings file, not a model config, so registering it would poison
    # the registry and make the loader choke on a non-model directory.
    if p in {HOME_DIR.resolve(), MODELS_DIR.resolve()}:
        console.print(
            f"[red]That is the localm data folder, not a model:[/red] {p}\n"
            "Point at a .gguf file or a specific model directory."
        )
        return

    from localm.inference.engine import _is_hf_dir
    is_gguf = p.is_file() and p.suffix == ".gguf"
    is_hf   = _is_hf_dir(str(p))  # config.json AND real weights/tokenizer
    is_blob = p.is_file() and p.name.startswith("sha256-")  # raw Ollama blob by path

    if not (is_gguf or is_hf or is_blob):
        console.print(
            f"[red]Not a model:[/red] {p}\n"
            "Expected a .gguf file or a HuggingFace model directory "
            "(config.json plus weights or a tokenizer)."
        )
        return

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
