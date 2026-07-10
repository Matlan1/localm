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

MODEL_TYPES = frozenset({'llm', 'mmproj', 'diffusion-unet', 'text-encoder', 'vae', 'lora', 'unknown'})

# HuggingFace architecture class-name suffixes that deterministically mark a text
# generation (chat) model: LlamaForCausalLM, T5ForConditionalGeneration,
# GPT2LMHeadModel, etc. This follows HF's own stable class-naming convention (a
# hard signal), NOT fuzzy substring tag matching. An architecture not matched here
# is left 'unknown' rather than silently assumed to be an LLM.
_HF_LLM_ARCH_SUFFIXES = ("ForCausalLM", "ForConditionalGeneration", "LMHeadModel")


def is_auto_chat_eligible(entry: dict) -> bool:
    """True when a registry *entry* may be auto-selected as the default chat model.

    A type='unknown' model (its type could not be determined) is never auto-loaded
    as chat, though it stays runnable when named explicitly (``localm run NAME``, an
    API request naming it) and its type can be corrected with ``localm set-type``. A
    legacy entry with no ``model_type`` key is treated as 'llm' (eligible), preserving
    pre-Branch-A behaviour.
    """
    return isinstance(entry, dict) and entry.get("model_type", "llm") != "unknown"


def is_llm(entry: dict) -> bool:
    """True when a registry *entry* is a text-generation (chat) LLM.

    'llm' (or a legacy entry with no ``model_type`` key, treated as 'llm') is an
    LLM; every other type - 'unknown' (unclassified) and the non-text component
    types 'mmproj' / 'diffusion-unet' / 'text-encoder' / 'vae' / 'lora' - is NOT.
    This is the same rule the GUI Models page uses (``isLlm`` in models.js); use it
    wherever a UI must offer ONLY chat-launchable models, e.g. the desktop
    launcher's model selector. Distinct from :func:`is_auto_chat_eligible`, which
    only screens out 'unknown' (for auto-picking a default) and would still admit a
    LoRA / VAE / other component model.
    """
    return isinstance(entry, dict) and entry.get("model_type", "llm") == "llm"


def _detect_local_model_type(path: Path, *, is_gguf: bool, is_hf: bool,
                             is_blob: bool = False) -> str:
    """Deterministically classify a LOCAL model's type from HARD metadata only.

    A .gguf file or Ollama blob is a llama.cpp text model (the format itself is the
    hard signal) -> 'llm'. An HF directory is classified from config.json: a
    LoRA/adapter dir -> 'lora'; an ``architectures`` class ending in ForCausalLM /
    LMHeadModel / ForConditionalGeneration -> 'llm'; anything we cannot resolve ->
    'unknown' (never a silent 'llm'). Embedding models are provisioned via
    ``setup-embeddings``, not the chat registry, so there is no separate 'embedding'
    registry type here.
    """
    try:
        if is_gguf or is_blob:
            return "llm"
        if is_hf:
            if (path / "adapter_config.json").exists():
                return "lora"
            cfg_path = path / "config.json"
            if cfg_path.exists():
                conf = json.loads(cfg_path.read_text(encoding="utf-8"))
                archs = conf.get("architectures") or []
                if isinstance(archs, list) and any(
                    isinstance(a, str) and a.endswith(_HF_LLM_ARCH_SUFFIXES)
                    for a in archs
                ):
                    return "llm"
    except Exception as e:
        # Detection is best-effort metadata reading; an unreadable/odd config.json
        # means "no hard signal", which is exactly 'unknown' - surfaced, not muted.
        logger.debug("local model-type detection failed for %s: %s", path, e)
    return "unknown"





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




def list_models(type_filter: Optional[str] = None) -> None:
    reg = _mm.load_registry()
    if type_filter:
        reg = {k: v for k, v in reg.items() if v.get("model_type", "llm") == type_filter}
    if not reg:
        console.print("[dim]No models yet. Use [bold]localm pull <name>[/bold] to download one.[/dim]")
        console.print("[dim]Run [bold]localm models[/bold] to see what's available.[/dim]")
        return

    table = Table(header_style="bold cyan", show_lines=False, expand=False)
    table.add_column("Name", style="bold white")
    table.add_column("Format", style="cyan")
    table.add_column("Role", style="magenta")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Source", style="dim")
    table.add_column("Path", style="dim")

    for name, info in sorted(reg.items()):
        path = Path(info["path"])
        source = info.get("source", "local")
        role = info.get("model_type", "llm")

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

        table.add_row(name_cell, kind, role, size, source, str(path))

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
    new_path = str(p.resolve())
    # Atomic read-modify-write so a concurrent registry writer (GUI thread, a
    # parallel pull, sync_models_dir) is not clobbered by a stale save (same
    # lost-update fix as the move path in #318).
    def _apply(r: dict) -> None:
        e = r.get(name)
        if isinstance(e, dict):
            e["path"] = new_path
            e.pop("missing", None)          # it is present again at the new path
    _mm.update_registry(_apply)
    console.print(f"[green]Relocated[/green] [bold]{name}[/bold] -> {p}")
    return True


def set_model_type(name: str, new_type: str) -> bool:
    """Change a registered model's type (llm / mmproj / diffusion-unet / text-encoder
    / vae / lora / unknown). Type is MUTABLE at any time: a bulk import or a forgotten
    ``--type`` is corrected here, not frozen at registration. Returns True on success,
    False if the model is not registered or *new_type* is not a MODEL_TYPES value."""
    reg = _mm.load_registry()
    if name not in reg:
        console.print(f"[red]No such registered model:[/red] {name}")
        return False
    if new_type not in MODEL_TYPES:
        console.print(f"[red]Invalid type {new_type!r}.[/red] "
                      f"Choose one of: {', '.join(sorted(MODEL_TYPES))}")
        return False

    def _apply(r: dict) -> None:
        e = r.get(name)
        if isinstance(e, dict):
            e["model_type"] = new_type

    # Atomic read-modify-write so a concurrent registry writer is not clobbered.
    _mm.update_registry(_apply)
    console.print(f"[green]Set[/green] [bold]{name}[/bold] type -> {new_type}")
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
    # Atomic RMW (re-read inside the lock) so a concurrent writer is not lost.
    def _apply(r: dict) -> None:
        if existing in r and new_name not in r:
            r[new_name] = dict(r[existing])
    _mm.update_registry(_apply)
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
    model_type: str = "llm",
) -> None:
    entry = {"path": str(path.resolve()), "source": source, "model_type": model_type}
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




def _store_into_models_dir(path: Path, action: str) -> Path:
    """Copy or move an external model (file or directory) INTO ``MODELS_DIR``, so
    it can be registered from there and treated exactly like a pulled model
    afterward (BRING-IN-1). ``action`` is ``"copy"`` or ``"move"``.

    Handles every on-disk shape a registered model can take:
      - a single-file GGUF
      - a split GGUF - every ``-NNNNN-of-NNNNN.gguf`` part (split_gguf_parts),
        not just the part *path* happens to point at
      - a GGUF's sibling mmproj vision-projector file, if one exists next to it
        (find_sibling_mmproj) - it MUST travel with the model, or vision
        capability silently breaks with no error at registration time
      - an HF-style model directory - the whole tree (shutil.copytree/move)

    Refuses (raises RuntimeError, does not touch anything) when a name inside
    MODELS_DIR is already occupied by a genuinely different file - the same
    path-identity guard the duplicate-content prompt already applied inline.
    Preflights free disk space before copying (a copy needs room for both the
    original and the new copy at once); a same-name/no-op destination (already
    in place) contributes nothing to that check. After each copy the destination
    is re-hashed against a pre-copy digest of the source and the whole operation
    fails loudly on any mismatch (we do not hide problems: a copy that silently
    landed corrupted must never be registered as if it worked). A move is not
    re-verified: on the same volume it is an atomic rename with no data ever
    written twice, and doubling the read of a many-GB model file just to
    reconfirm what the OS's move already guarantees is not worth the cost.

    Returns the new path of the primary file/directory (the first part, for a
    split GGUF) - the caller registers THIS path, not the original.
    """
    _mm.ensure_dirs()
    path = path.resolve()
    verb = "Copying" if action == "copy" else "Moving"

    if path.is_dir():
        dest = _mm.MODELS_DIR / path.name
        if dest.resolve() == path:
            return path                                    # already in place
        if dest.exists():
            raise RuntimeError(f"Cannot {action}: {dest} already exists")
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        if not _mm._check_disk_space(_mm.MODELS_DIR, total):
            raise RuntimeError(
                f"Not enough disk space to {action} {path} into {_mm.MODELS_DIR}"
            )
        console.print(f"[dim]{verb} {path} to {dest}…[/dim]")
        if action == "copy":
            shutil.copytree(path, dest)
        else:
            shutil.move(str(path), str(dest))
        return dest

    # Single GGUF: gather every split part plus a sibling mmproj (if any) -
    # all of it must travel together or the model breaks (multi-part loading)
    # or silently loses a capability (vision) with no error at registration time.
    parts = split_gguf_parts(path.name) or [path.name]
    sources = [path.parent / part for part in parts]
    if path.suffix.lower() == ".gguf":
        mmproj = find_sibling_mmproj(path)
        if mmproj is not None and mmproj not in sources:
            sources.append(mmproj)

    dests: List[Path] = []
    for src in sources:
        dest = _mm.MODELS_DIR / src.name
        if dest.exists() and dest.resolve() != src.resolve():
            raise RuntimeError(f"Cannot {action}: {dest} already exists")
        dests.append(dest)

    to_transfer = [(s, d) for s, d in zip(sources, dests) if s.resolve() != d.resolve()]
    total = sum(s.stat().st_size for s, _ in to_transfer if s.exists())
    if not _mm._check_disk_space(_mm.MODELS_DIR, total):
        raise RuntimeError(
            f"Not enough disk space to {action} {path.name} into {_mm.MODELS_DIR}"
        )

    for src, dest in to_transfer:
        if not src.exists():
            # A split part or mmproj sibling that vanished between discovery and
            # transfer (pre-existing incomplete/broken model on disk) - not this
            # operation's problem to fix; note it and continue with the rest.
            logger.debug("_store_into_models_dir: %s no longer exists, skipping", src)
            continue
        console.print(f"[dim]{verb} {src.name} to {dest}…[/dim]")
        if action == "copy":
            pre_digest = _mm._sha256_file(src)
            shutil.copy2(src, dest)
            post_digest = _mm._sha256_file(dest)
            if post_digest != pre_digest:
                # Don't leave a known-bad copy sitting in MODELS_DIR - it would
                # otherwise be a ticking time bomb for a future sync_models_dir
                # to auto-register as its own (corrupt) model.
                try:
                    dest.unlink()
                except OSError:
                    pass  # best-effort cleanup; the mismatch itself still raises
                raise RuntimeError(
                    f"Copy verification failed for {src.name}: sha256 mismatch "
                    f"after copy to {dest} (source left untouched, bad copy removed)"
                )
        else:
            shutil.move(str(src), str(dest))

    return _mm.MODELS_DIR / path.name




def _register_with_dedup(
    model_name: str,
    p: Path,
    source: str,
    *,
    on_duplicate: str = "ask",
    digest: Optional[str] = None,
    size: Optional[int] = None,
    model_type: str = "llm",
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
            def _backfill(r: dict) -> None:      # atomic RMW
                e = r.get(model_name)
                if isinstance(e, dict) and not e.get("sha256"):
                    e["sha256"] = digest.lower()
            _mm.update_registry(_backfill)
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
            # Capture the pre-move resolved path BEFORE _store_into_models_dir
            # touches anything - it's the key the alias-relink below matches on.
            moved_from = str(p.resolve())
            try:
                dest = _mm._store_into_models_dir(p, action)
            except RuntimeError as e:
                console.print(f"[red]{e}[/red]")
                return
            if action == "move":
                dest_str = str(dest.resolve())
                entry = {"path": dest_str, "source": source, "model_type": model_type}
                if digest:
                    entry["sha256"] = digest.lower()

                # Move first so the file lands under MODELS_DIR (where a launch-time
                # sync_models_dir can always recover it), THEN commit the registry in
                # ONE atomic write: repoint the moved file's other aliases AND register
                # the new name together, so the registry never persists a half-updated
                # state (L7 - previously a save_registry for the aliases and a separate
                # _register for the new name left a window that a crash could split).
                # The move-then-write step is still not one transaction, but a crash
                # between them leaves the file in MODELS_DIR and the registry fully
                # pre-move, which sync reconciles on the next launch.
                def _relink_and_register(r: dict) -> None:
                    for alias_name in dup_names:
                        if r.get(alias_name, {}).get("path") == moved_from:
                            r[alias_name]["path"] = dest_str
                    r[model_name] = entry

                _mm.update_registry(_relink_and_register)
                console.print(f"[green]✓[/green] Registered [bold]{model_name}[/bold]")
                return
            p = dest
        # action == "register" falls through unchanged

    _mm._register(model_name, p, source, sha256=digest, model_type=model_type)
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
        _mm.update_registry(lambda r: r.pop(name, None))   # atomic RMW
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
    _mm.update_registry(lambda r: r.pop(name, None))       # atomic RMW
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




def _store_loose_gguf_dir(first_parts: List[Path], store: str) -> Optional[List[Path]]:
    """Bring every model in a directory-of-loose-ggufs import into MODELS_DIR
    before ``_add_local_gguf_dir`` registers them (mirrors its per-file loop).

    ``first_parts`` is one entry per independent model in the folder - but that
    list also includes any mmproj vision-projector file sitting in the same
    folder (``_gguf_first_parts`` does not filter those out, since they are
    registered as their own model too, same as today without --store). A
    projector is auto-attached to its model by _store_into_models_dir already
    (find_sibling_mmproj), so if we called the helper again on the projector's
    OWN entry we'd either re-move a file that is already gone (crash) or hit a
    false "already exists" collision against the copy that just landed next to
    its model (same name, different source directory). So: precompute, from the
    ORIGINAL on-disk layout, which entries are an unambiguous sibling of some
    OTHER entry in this same folder, and only "claim" a final path for those
    (they ride along with their model) instead of transferring them a second time.

    Returns the new first_parts list (paths now under MODELS_DIR), or None if
    any transfer failed (name collision, disk space, or a copy that verified
    corrupt) - the caller reports the printed error and aborts the whole import.
    """
    claimed_sibling_of: dict = {}   # resolved mmproj path -> its owning model path
    for owner in first_parts:
        sib = find_sibling_mmproj(owner)
        if sib is None:
            continue
        sib_r = sib.resolve()
        if any(sib_r == g.resolve() for g in first_parts):
            claimed_sibling_of[sib_r] = owner

    new_parts: List[Path] = []
    for gguf in first_parts:
        if gguf.resolve() in claimed_sibling_of:
            # Transferred as a side effect of its owning model's own call below
            # (whichever order that happens in) - just point at its new home.
            new_parts.append(_mm.MODELS_DIR / gguf.name if _mm.is_external_path(gguf) else gguf)
            continue
        if not _mm.is_external_path(gguf):
            new_parts.append(gguf)
            continue
        try:
            new_parts.append(_mm._store_into_models_dir(gguf, store))
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            return None
    return new_parts




def _add_local_gguf_dir(
    first_parts: List[Path],
    name: Optional[str],
    on_duplicate: str,
    no_hash: bool,
    fast: bool = False,
    model_type: str = "llm",
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
            digest=digest, size=size, model_type=model_type,
        )
    return True




def add_local(
    path_str: str,
    name: Optional[str] = None,
    on_duplicate: str = "ask",
    no_hash: bool = False,
    fast: bool = False,
    model_type: Optional[str] = None,
    store: Optional[str] = None,
) -> bool:
    """Register a local .gguf / HF dir / Ollama blob. Returns True on a successful
    registration or a benign no-op (alias / user-skipped duplicate), False when the
    path is not a usable model, so `localm pull <path>` can set its exit code.

    *model_type* None (the default) means "detect it": the type is resolved
    deterministically from hard metadata (GGUF -> llm, HF config.json architectures)
    and falls back to the 'unknown' sentinel rather than a silent 'llm'. Pass an
    explicit MODEL_TYPES value to force it. A lone .safetensors file is not a model on
    its own: if it sits inside an HF model dir that directory is registered instead,
    otherwise it is rejected with a precise, actionable reason (A3).

    *store* is ``"copy"``, ``"move"``, or ``None`` (default: register in place,
    today's behavior). When set and the path is OUTSIDE ~/.localm/models, the
    file/dir is brought into managed storage (via _store_into_models_dir) BEFORE
    registration, so the model is treated exactly like a pulled model afterward
    (BRING-IN-1). A failure here (name collision, no disk space, copy corrupted)
    is a hard failure of the whole call - unlike the softer dedup-prompt copy/move
    (where skipping just means "keep the old registration"), a requested --store
    that silently fell back to registering the ORIGINAL external path would be a
    hidden problem, not a benign no-op.
    """
    p = Path(path_str).resolve()
    if not p.exists():
        console.print(f"[red]Not found:[/red] {path_str}")
        return False

    # Ollama manifest directory -> resolve to actual GGUF blob
    ollama = _resolve_ollama_manifest(p)
    if ollama is not None:
        blob_path, suggested = ollama
        if store and _mm.is_external_path(blob_path):
            try:
                blob_path = _mm._store_into_models_dir(blob_path, store)
            except RuntimeError as e:
                console.print(f"[red]{e}[/red]")
                return False
        # Sanitize the user-supplied -n name through the same filter
        # sync_models_dir uses, so a '../evil' or 'a/b' name can never become a
        # raw registry key (GAP-CLI-1).
        model_name = _sanitize_name(name) if name else suggested
        # Ollama blob filenames already ARE the sha256 digest - store it free
        digest = blob_path.name.removeprefix("sha256-") \
            if blob_path.name.startswith("sha256-") else None
        # An Ollama blob is a GGUF text model, so an unspecified type is 'llm'.
        _mm._register_with_dedup(
            model_name, blob_path, "ollama",
            on_duplicate=on_duplicate, digest=digest,
            model_type=(model_type if model_type is not None else "llm"),
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

    # A lone .safetensors file is not loadable on its own: llama.cpp loads .gguf, and
    # the HF backend loads a DIRECTORY (config.json + weights/tokenizer). If the file
    # sits inside a real HF model dir, register that DIRECTORY (deterministic type
    # detection below then classifies it); otherwise reject with a precise, actionable
    # reason instead of the bare "Not a model" (A3).
    if p.is_file() and p.suffix.lower() == ".safetensors" and not (is_gguf or is_hf or is_blob):
        parent = p.parent
        if _is_hf_dir(str(parent)):
            p = parent
            is_hf = True
        else:
            console.print(
                f"[red]Incomplete model:[/red] {p}\n"
                "A .safetensors weight file loads only as part of a HuggingFace model "
                "directory. Point at the model's folder (the one holding config.json "
                "plus a tokenizer), or use a single-file .gguf."
            )
            return False

    # A directory of loose .gguf files (not a single model, not an HF model dir,
    # not an Ollama manifest) - register each one, the way sync_models_dir does
    # for the models folder (H2). An HF dir (is_hf) falls through to the
    # dir-as-one-model path below; an empty / non-gguf dir falls through to the
    # "Not a model" message so existing rejection tests stay green.
    if p.is_dir() and not is_hf:
        max_depth = max(1, int(load_config().get("import_max_depth", 3)))
        first_parts = _gguf_first_parts(p, max_depth=max_depth)
        if first_parts:
            if store:
                stored = _mm._store_loose_gguf_dir(first_parts, store)
                if stored is None:
                    return False
                first_parts = stored
            # Loose .gguf files are llama.cpp text models, so an unspecified type
            # is 'llm' (detection per-file would only ever return 'llm' anyway).
            return _add_local_gguf_dir(
                first_parts, name, on_duplicate, no_hash, fast,
                model_type=(model_type if model_type is not None else "llm"),
            )

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

    # Bring an external file/dir into managed storage BEFORE registering, so the
    # registry ends up pointing at the copy/move destination, not the original.
    # Handles the split-GGUF parts and any sibling mmproj on its own
    # (_store_into_models_dir); works for is_hf's whole directory too.
    if store and _mm.is_external_path(p):
        try:
            p = _mm._store_into_models_dir(p, store)
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            return False

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

    # Resolve an unspecified type deterministically (GGUF -> llm, HF config.json ->
    # architectures, else 'unknown') instead of silently defaulting to 'llm'.
    if model_type is None:
        model_type = _detect_local_model_type(p, is_gguf=is_gguf, is_hf=is_hf, is_blob=is_blob)

    _mm._register_with_dedup(
        model_name, p, kind, on_duplicate=on_duplicate, digest=digest, size=size, model_type=model_type,
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

