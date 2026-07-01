# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model registry: selection/resolution, info + vision capabilities, aliases,
dedup, disk sync, add-local, and removal. Depends on the gguf helpers."""

import localm.model_manager as _mm  # read package-patchable names at call time

import json
import re
import shutil
import sys
from pathlib import Path
from typing import List
from typing import NamedTuple
from typing import Optional
from rich.table import Table
from ..config import REGISTRY_FILE
from ..config import load_config
from ..debuglog import logger
from ._shared import console
from .gguf import _SPLIT_GGUF_RE
from .gguf import _gguf_first_parts
from .gguf import _has_gguf_magic
from .gguf import first_split_part
from .gguf import split_gguf_parts




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
    reg = _mm.load_registry()
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




def find_sibling_mmproj(model_path) -> Optional[Path]:
    """Auto-detect a vision projector (mmproj) GGUF sitting next to a GGUF model.

    Vision GGUFs ship a separate 'mmproj' projector file in the same folder
    (e.g. 'gemma-3-4b-it-Q8_0.gguf' + 'mmproj-gemma-3-4b-it-f16.gguf'), so picking
    it up lets a GUI/registry load get vision without a CLI --mmproj flag (VIS-1).
    Only a GGUF model qualifies and the model file itself is excluded. Returns the
    single candidate, or None when there are none, or the choice is ambiguous
    (>1 with no clear stem match) - we never silently load the wrong projector."""
    p = Path(model_path)
    if p.suffix.lower() != ".gguf" or not p.parent.is_dir():
        return None
    cands = [f for f in p.parent.glob("*.gguf")
             if f.name != p.name and "mmproj" in f.name.lower()]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    # Ambiguous: prefer a projector whose name shares the model's leading token.
    stem = p.stem.lower().replace("mmproj", "").split("-")[0].split(".")[0]
    matches = [f for f in cands if stem and stem in f.name.lower()]
    return matches[0] if len(matches) == 1 else None




def get_model_mmproj(name: str) -> Optional[str]:
    """The mmproj (vision projector) path for a model, if one is known.

    Priority: an explicit 'mmproj' recorded in the registry entry, else a sibling
    mmproj GGUF auto-detected next to the model file. Returns an absolute path
    string, or None when no projector is associated. This is what lets a GGUF keep
    vision after a GUI/registry model switch (VIS-1), the same way the CLI --mmproj
    flag does on a direct run."""
    reg = _mm.load_registry()
    entry = reg.get(name) if isinstance(reg, dict) else None
    if isinstance(entry, dict) and entry.get("mmproj"):
        mmp = Path(entry["mmproj"])
        if mmp.exists():
            return str(mmp)
        # Recorded but gone: fall through to auto-detect rather than handing the
        # backend a dead path that would just fail the mtmd load.
    info = get_model_info(name)
    if info is None:
        return None
    sib = find_sibling_mmproj(info[0])
    return str(sib) if sib else None




# ------------------------------------------------------------------ #
#  Vision-input capability (kernel-level, setup-agnostic)              #
# ------------------------------------------------------------------ #

def _hf_is_vision(model_dir: Path) -> bool:
    """True if a HuggingFace model directory looks vision-capable, judged from
    its on-disk metadata (offline, no model load, no hardware assumption). The
    HF backend can then accept image input for it."""
    try:
        pre = model_dir / "preprocessor_config.json"
        if pre.is_file() and "image" in pre.read_text(
                encoding="utf-8", errors="replace").lower():
            return True
        cfg = model_dir / "config.json"
        if cfg.is_file():
            data = json.loads(cfg.read_text(encoding="utf-8", errors="replace"))
            if any(k in data for k in
                   ("vision_config", "image_token_index", "image_token_id")):
                return True
            arch = " ".join(data.get("architectures") or []).lower()
            if any(k in arch for k in ("vision", "imagetext", "-vl",
                                       "qwen2vl", "qwen2_5_vl")):
                return True
    except (OSError, ValueError):
        pass
    return False




def vision_capable_models() -> List[str]:
    """Registered model names that can accept image INPUT on THIS install.

    Only HuggingFace-format directories with vision metadata qualify here. GGUF
    entries are not auto-listed: a GGUF gains vision only when a matching mmproj
    (vision projector) is loaded with it at run time (the built-in mtmd path),
    which the on-disk registry metadata cannot confirm. Used to ROUTE an image to
    a model already known to be vision-capable instead of dead-ending."""
    out: List[str] = []
    for name, info in _mm.load_registry().items():
        try:
            p = Path(info.get("path", ""))
        except (TypeError, ValueError):
            continue
        if p.is_dir() and _hf_is_vision(p):
            out.append(name)
    return sorted(out)




def vision_input_guidance(mmproj_failed: bool = False) -> str:
    """Capability-aware, install-specific message for when an image is attached
    to a model that cannot see it. Instead of a flat dead-end, point the user at
    a path that EXISTS on THIS install: a vision model already in their library,
    or how to obtain one. Setup-agnostic - it inspects the registry and the
    installed stack, never assuming a particular GPU or runtime. Begins with the
    legacy 'cannot accept image input' phrase so existing callers stay valid.

    *mmproj_failed* is True when the active GGUF model WAS given a vision projector
    (mmproj) but it did not load (supports_images is still False). GGUF vision IS
    implemented (the built-in mtmd path), so do NOT claim it is unimplemented: the
    honest cause is that this particular projector failed to load - likely
    incompatible with the model, or the mtmd vision runtime is unavailable."""
    import importlib.util
    if mmproj_failed:
        head = ("This model cannot accept image input: a vision projector (mmproj) "
                "was provided but failed to load - it may be incompatible with this "
                "model, or the mtmd vision runtime is unavailable (see the server "
                "log for the mtmd error).")
    else:
        head = ("This model cannot accept image input (it is text-only), so the "
                "attached image would be ignored.")
    vlms = vision_capable_models()
    if vlms:
        return (f"{head} A vision-capable model is already in your library: "
                f"{', '.join(vlms[:3])}. Switch to it (e.g. "
                f"`localm run {vlms[0]}`, or pick it in the GUI) and attach the "
                f"image again.")
    if importlib.util.find_spec("transformers") is not None:
        return (f"{head} No vision model is registered yet - pull a vision-capable "
                f"HuggingFace model such as a Gemma 3 vision or Qwen2.5-VL "
                f"checkpoint (`localm pull <repo>`), then run it with --image.")
    return (f"{head} To read images, install the HuggingFace stack "
            f"(`pip install \"localm[gpu]\"`) and load a vision-capable model "
            f"(Gemma 3 vision / Qwen2.5-VL). The built-in GGUF backend is text-only.")




def list_models() -> None:
    reg = _mm.load_registry()
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
            "Re-point one you MOVED with [bold]localm relocate <name> <new-path>[/bold], "
            "set [bold]autoprune_missing_models true[/bold] to drop them "
            "automatically, or re-add the file.[/dim]"
        )


def is_external_path(path) -> bool:
    """True if *path* points OUTSIDE the managed models dir - an external model the
    user referenced from elsewhere (e.g. a shared drive), vs one localm downloaded
    into MODELS_DIR. External models can go 'missing' when their real location moves,
    and are the case `localm relocate` re-points (REC-EXTPATH-RELOCATE)."""
    try:
        Path(path).resolve().relative_to(_mm.MODELS_DIR.resolve())
        return False
    except (ValueError, OSError):
        return True


def model_is_external(name: str) -> bool:
    """Whether the registered model *name* lives outside the managed models dir."""
    info = _mm.load_registry().get(name)
    return bool(info) and is_external_path(info.get("path", ""))


def relocate_model(name: str, new_path: str) -> bool:
    """Re-point a registered model to *new_path* - for an EXTERNAL model whose file
    was MOVED (it shows 'missing' but is not gone, just relocated). Validates the new
    path is a real GGUF file or HF dir, updates the registry, and clears the missing
    flag. Returns True on success (REC-EXTPATH-RELOCATE)."""
    from localm.model_manager.gguf import _has_gguf_magic
    reg = _mm.load_registry()
    if name not in reg:
        console.print(f"[red]No such registered model:[/red] {name}")
        return False
    p = Path(new_path).expanduser()
    if not p.exists():
        console.print(f"[red]Path does not exist:[/red] {p}")
        return False
    if p.is_dir():
        from localm.inference.engine import _is_hf_dir
        if not _is_hf_dir(str(p)):
            console.print(f"[red]Not a HuggingFace model directory:[/red] {p}")
            return False
    elif p.suffix.lower() != ".gguf" or not _has_gguf_magic(p):
        console.print(f"[red]Not a GGUF model file:[/red] {p}")
        return False
    entry = reg[name]
    if not isinstance(entry, dict):
        console.print(f"[red]Corrupt registry entry for {name}[/red]")
        return False
    entry["path"] = str(p.resolve())
    entry.pop("missing", None)              # it is present again at the new path
    _mm.save_registry(reg)
    console.print(f"[green]Relocated[/green] [bold]{name}[/bold] -> {p}")
    return True




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




# ------------------------------------------------------------------ #
#  Model identity - duplicate detection (two-tier: path, then sha256)  #
# ------------------------------------------------------------------ #

def find_aliases_by_path(path: Path, reg: Optional[dict] = None) -> List[str]:
    """Registered names whose path resolves to the same file/dir as *path*."""
    reg = reg if reg is not None else _mm.load_registry()
    target = str(Path(path).resolve())
    return sorted(
        name for name, info in reg.items()
        if str(Path(info.get("path", "")).resolve()) == target
    )




def find_by_sha256(digest: str, reg: Optional[dict] = None) -> List[str]:
    """Registered names whose stored sha256 matches *digest* (case-insensitive)."""
    if not digest:
        return []
    reg = reg if reg is not None else _mm.load_registry()
    d = digest.lower()
    return sorted(
        name for name, info in reg.items()
        if info.get("sha256", "").lower() == d
    )




def find_by_size(size: int, reg: Optional[dict] = None) -> List[str]:
    """Registered names whose on-disk file is exactly *size* bytes.

    A cheap (stat-only) content heuristic used by ``--fast`` imports: it skips
    the full SHA-256 and dedups on size alone. Weaker than the hash tier (two
    different files can share a length), which is why it is opt-in. Entries whose
    path is missing or is a directory are skipped."""
    if not size or size <= 0:
        return []
    reg = reg if reg is not None else _mm.load_registry()
    out = []
    for name, info in reg.items():
        path = info.get("path")
        if not path:
            continue
        try:
            p = Path(path)
            if p.is_file() and p.stat().st_size == size:
                out.append(name)
        except OSError:
            continue
    return sorted(out)




def alias_model(existing: str, new_name: str) -> bool:
    """
    Register *new_name* as an additional name for *existing* (same file,
    same source, same digest). Returns False when *existing* is unknown
    or *new_name* is already taken.
    """
    reg = _mm.load_registry()
    if existing not in reg:
        console.print(f"[red]Not found:[/red] {existing}")
        return False
    if new_name in reg:
        console.print(f"[red]Name already in use:[/red] {new_name}")
        return False
    reg[new_name] = dict(reg[existing])
    _mm.save_registry(reg)
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
    _mm.update_registry(lambda reg: reg.__setitem__(name, entry))




def _sanitize_name(base: str) -> str:
    """Reduce ``base`` to a registry-safe key.

    Strips anything that could turn a registry name into a filesystem path or a
    traversal sequence: only ``A-Za-z0-9._-`` survive, and any run of other
    characters (path separators, ``..`` sequences, spaces) collapses to a single
    hyphen. Leading/trailing hyphens are trimmed; an empty result falls back to
    ``"model"``. Consecutive dots (``..`` traversal) collapse to one and
    leading/trailing dots are trimmed, so the result is never ``.``/``..`` and
    cannot act as a relative path. This is the single sanitizer used for both
    auto-discovered names (sync_models_dir) and user-supplied ``-n`` names
    (add_local).
    """
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base)
    base = re.sub(r"\.{2,}", ".", base)          # ..  ->  .  (no traversal)
    return base.strip("-.") or "model"




def _unique_registry_name(reg: dict, base: str) -> str:
    """Return a registry-safe name derived from ``base``, avoiding collisions."""
    base = _sanitize_name(base)
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

    _mm.ensure_dirs()
    reg = _mm.load_registry()

    known = {
        str(Path(entry["path"]).resolve())
        for entry in reg.values()
        if entry.get("path")
    }

    added = 0
    if _mm.MODELS_DIR.is_dir():
        for child in sorted(_mm.MODELS_DIR.iterdir()):
            try:
                # HuggingFace model directory.
                if child.is_dir() and (child / "config.json").is_file():
                    resolved = str(child.resolve())
                    if resolved in known:
                        continue
                    _mm._register(_unique_registry_name(reg, child.name), child)
                    reg = _mm.load_registry()
                    known.add(resolved)
                    added += 1
            except OSError:
                continue

        for child in sorted(_mm.MODELS_DIR.glob("*.gguf")):
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
                # Skip a file that is not actually a GGUF (foreign file renamed
                # .gguf, an empty/partial copy): registering it would pollute the
                # model list and could crash a later load (R45). Note it so a
                # genuinely-broken file is not silently invisible.
                if not _has_gguf_magic(child):
                    logger.debug("skipping %s: not a GGUF (bad/missing magic)",
                                 child.name)
                    continue
                _mm._register(_unique_registry_name(reg, child.stem), child)
                reg = _mm.load_registry()
                known.add(resolved)
                added += 1
            except OSError:
                continue

    # Reconcile missing / restored / pruned against the (possibly grown) registry.
    reg = _mm.load_registry()
    models_root = _mm.MODELS_DIR.resolve()

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
                _mm._backup_registry()
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
        _mm.save_registry(reg)

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
    size: Optional[int] = None,
) -> None:
    """
    Register a model, detecting duplicates first.

    Identity tiers: resolved path (instant), then stored sha256 when a *digest*
    for the new file is known, then file *size* alone (the ``--fast`` heuristic,
    used only when no digest was computed). ``on_duplicate`` is one of
    "ask" / "alias" / "copy" / "move" / "register" / "skip" - "ask" prompts
    interactively and degrades to "skip" without a TTY.
    """
    import click

    reg = _mm.load_registry()
    aliases = find_aliases_by_path(p, reg)

    # Same name, same file - true no-op (but backfill a fresh digest)
    if model_name in aliases:
        console.print(
            f"[yellow]'{model_name}' is already registered for this exact "
            f"file[/yellow] [dim]({p})[/dim]"
        )
        if digest and not reg[model_name].get("sha256"):
            reg[model_name]["sha256"] = digest.lower()
            _mm.save_registry(reg)
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

    # Duplicate content under other names? (path tier, then hash tier, then the
    # fast size-only tier when no digest was computed)
    dup_names, reason = aliases, "same file"
    if not dup_names and digest:
        dup_names = _mm.find_by_sha256(digest, reg)
        reason = "byte-identical content"
    elif not dup_names and not digest and size:
        dup_names = find_by_size(size, reg)
        reason = "same size (--fast: content not hashed)"

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
            _mm.ensure_dirs()
            dest = _mm.MODELS_DIR / p.name
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
                _mm.save_registry(reg)
            p = dest
        # action == "register" falls through unchanged

    _mm._register(model_name, p, source, sha256=digest)
    console.print(f"[green]✓[/green] Registered [bold]{model_name}[/bold]")




def remove_model(name: str) -> None:
    reg = _mm.load_registry()
    if name not in reg:
        console.print(f"[red]Not found:[/red] {name}")
        return
    path = Path(reg[name]["path"])

    # Alias-aware: if other names still point at this file, only unregister
    # this name - never delete a file out from under another alias.
    other_aliases = [a for a in find_aliases_by_path(path, reg) if a != name]
    if other_aliases:
        del reg[name]
        _mm.save_registry(reg)
        console.print(
            f"[green]✓[/green] Removed [bold]{name}[/bold] "
            f"[dim](file kept - still registered as: "
            f"{', '.join(other_aliases)})[/dim]"
        )
        return

    # Only delete files that live inside ~/.localm/models/ - never touch
    # externally registered paths (Ollama blobs, user model dirs, etc.)
    owned = path.is_relative_to(_mm.MODELS_DIR) if hasattr(path, "is_relative_to") else \
            str(path).startswith(str(_mm.MODELS_DIR))
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
        console.print("[dim]Unregistered (file not deleted - lives outside ~/.localm/models)[/dim]")
    del reg[name]
    _mm.save_registry(reg)
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




def _add_local_gguf_dir(
    first_parts: List[Path],
    name: Optional[str],
    on_duplicate: str,
    no_hash: bool,
    fast: bool = False,
) -> bool:
    """Register every loose .gguf model in a folder (the *first_parts* list).

    Names each model after its filename stem (split GGUFs strip the
    ``-NNNNN-of-NNNNN`` suffix), de-duplicating collisions via
    ``_unique_registry_name``. A user-supplied ``-n`` name is only honoured for
    a single-model folder, since it cannot apply to many. Returns True - the
    caller has already checked *first_parts* is non-empty.
    """
    use_given_name = bool(name) and len(first_parts) == 1
    for gguf in first_parts:
        split = _SPLIT_GGUF_RE.match(gguf.name)
        base = split.group("stem") if (split and split_gguf_parts(gguf.name)) else gguf.stem
        reg = _mm.load_registry()
        wanted = _sanitize_name(name) if use_given_name else base
        model_name = _unique_registry_name(reg, wanted)

        size = None
        if no_hash:
            digest = None
        elif fast:
            # --fast: skip the SHA, dedup on size alone (cheap stat).
            digest = None
            try:
                size = gguf.stat().st_size
            except OSError:
                size = None
        else:
            # Hash only when it can change the outcome (unknown path / missing
            # digest), exactly like the single-file branch below.
            already_known = find_aliases_by_path(gguf, reg)
            needs_backfill = any(not reg[n].get("sha256") for n in already_known)
            digest = _mm._hash_with_progress(gguf) \
                if (not already_known or needs_backfill) else None

        _mm._register_with_dedup(
            model_name, gguf, "local", on_duplicate=on_duplicate,
            digest=digest, size=size,
        )
    return True




def add_local(
    path_str: str,
    name: Optional[str] = None,
    on_duplicate: str = "ask",
    no_hash: bool = False,
    fast: bool = False,
) -> bool:
    """Register a local .gguf / HF dir / Ollama blob. Returns True on a successful
    registration or a benign no-op (alias / user-skipped duplicate), False when the
    path is not a usable model, so `localm pull <path>` can set its exit code."""
    p = Path(path_str).resolve()
    if not p.exists():
        console.print(f"[red]Not found:[/red] {path_str}")
        return False

    # Ollama manifest directory -> resolve to actual GGUF blob
    ollama = _resolve_ollama_manifest(p)
    if ollama is not None:
        blob_path, suggested = ollama
        # Sanitize the user-supplied -n name through the same filter
        # sync_models_dir uses, so a '../evil' or 'a/b' name can never become a
        # raw registry key (GAP-CLI-1).
        model_name = _sanitize_name(name) if name else suggested
        # Ollama blob filenames already ARE the sha256 digest - store it free
        digest = blob_path.name.removeprefix("sha256-") \
            if blob_path.name.startswith("sha256-") else None
        _mm._register_with_dedup(
            model_name, blob_path, "ollama",
            on_duplicate=on_duplicate, digest=digest,
        )
        return True

    # Refuse the localm data directory (and its models root): its config.json is
    # the app's settings file, not a model config, so registering it would poison
    # the registry and make the loader choke on a non-model directory.
    if p in {_mm.HOME_DIR.resolve(), _mm.MODELS_DIR.resolve()}:
        console.print(
            f"[red]That is the localm data folder, not a model:[/red] {p}\n"
            "Point at a .gguf file or a specific model directory."
        )
        return False

    from localm.inference.engine import _is_hf_dir
    is_gguf = p.is_file() and p.suffix == ".gguf"
    is_hf   = _is_hf_dir(str(p))  # config.json AND real weights/tokenizer
    is_blob = p.is_file() and p.name.startswith("sha256-")  # raw Ollama blob by path

    # A directory of loose .gguf files (not a single model, not an HF model dir,
    # not an Ollama manifest) - register each one, the way sync_models_dir does
    # for the models folder (H2). An HF dir (is_hf) falls through to the
    # dir-as-one-model path below; an empty / non-gguf dir falls through to the
    # "Not a model" message so existing rejection tests stay green.
    if p.is_dir() and not is_hf:
        max_depth = max(1, int(load_config().get("import_max_depth", 3)))
        first_parts = _gguf_first_parts(p, max_depth=max_depth)
        if first_parts:
            return _add_local_gguf_dir(first_parts, name, on_duplicate, no_hash, fast)

    if not (is_gguf or is_hf or is_blob):
        console.print(
            f"[red]Not a model:[/red] {p}\n"
            "Expected a .gguf file or a HuggingFace model directory "
            "(config.json plus weights or a tokenizer)."
        )
        return False

    # A directly-supplied split-GGUF part (e.g. `localm add big-00002-of-00002.gguf`)
    # must be normalised: llama.cpp needs *-00001-of-N to load the whole set, so the
    # registry key drops the -NNNNN-of-NNNNN suffix (-> "big") and the stored path
    # points at the first part. Mirrors the folder branch (_add_local_gguf_dir) and
    # the read-time normalisation in get_model_info.
    split_base: Optional[str] = None
    if is_gguf:
        split = _SPLIT_GGUF_RE.match(p.name)
        if split and split_gguf_parts(p.name):
            split_base = split.group("stem")
            first = first_split_part(p.name)
            if first != p.name:
                first_path = p.parent / first
                if first_path.is_file():
                    p = first_path

    # Sanitize the user-supplied -n name (GAP-CLI-1): never let '../evil' or
    # 'a/b' become a raw registry key. p.stem is already path-component-safe.
    model_name = _sanitize_name(name) if name else (split_base or p.stem)
    kind = "hf" if is_hf else "local"

    size: Optional[int] = None
    if is_blob:
        digest = p.name.removeprefix("sha256-")
    elif no_hash or p.is_dir():
        digest = None   # HF dirs are identified by path only
    elif fast:
        # --fast: skip the full-file SHA, dedup on size alone (cheap stat).
        digest = None
        try:
            size = p.stat().st_size
        except OSError:
            size = None
    else:
        # Hash when it can change the outcome: unknown path (content-tier
        # check) or known path missing its digest (lazy backfill)
        reg = _mm.load_registry()
        already_known = find_aliases_by_path(p, reg)
        needs_backfill = any(
            not reg[n].get("sha256") for n in already_known
        )
        digest = None
        if not already_known or needs_backfill:
            digest = _mm._hash_with_progress(p)

    _mm._register_with_dedup(
        model_name, p, kind, on_duplicate=on_duplicate, digest=digest, size=size,
    )
    return True




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

