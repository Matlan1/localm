# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model registry: selection/resolution, info + vision capabilities, aliases,
dedup, disk sync, add-local, and removal. Depends on the gguf helpers."""

import localm.model_manager as _mm  # read package-patchable names at call time

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from pathlib import PureWindowsPath
from typing import List
from typing import NamedTuple
from typing import Optional
from rich.table import Table
from ..config import REGISTRY_FILE
from ..config import load_config
from ..config import update_config
from ..debuglog import logger
from ._shared import _verify_digest
from ._shared import console
from .gguf import _SPLIT_GGUF_RE
from .gguf import _gguf_first_parts
from .gguf import _has_gguf_magic
from .gguf import first_split_part
from .gguf import split_gguf_parts
from .gguf import gguf_embedding_signal
from .gguf import gguf_is_mmproj
from .gguf import gguf_registry_metadata
from .gguf import _gguf_metadata_probe

MODEL_TYPES = frozenset({'llm', 'mmproj', 'diffusion-unet', 'text-encoder', 'vae', 'lora', 'embedding', 'unknown'})

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
    pre-Branch-A behaviour. type='embedding' is also excluded: it is loaded via a
    dedicated embeddings-mode context (see ``inference/embedder.py``), not the causal
    chat path, so it must never be auto-picked as the default chat model - a real risk
    now that ``setup-embeddings`` can register one into the main registry, making an
    embedding-only registry (a plausible first-run state) a genuine scenario.
    """
    return isinstance(entry, dict) and entry.get("model_type", "llm") not in ("unknown", "embedding")


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


def _entry_path(entry, field: str = "path") -> Optional[str]:
    """The stored file path of a registry *entry*, or None when the entry is
    malformed (not a dict; the named *field* missing / null / not a non-empty
    string; or that path carrying a ``..`` component).

    *field* selects WHICH stored path to validate. It defaults to ``path``, the
    model's own file. ``mmproj`` (the multimodal projector recorded beside a
    vision GGUF) is the other one, and it is a path in exactly the same sense:
    it is read from the same hand-editable file and handed to the same native
    loader. It was previously read raw, so a projector entry bypassed both the
    type check and the ``..`` rejection that this choke point exists to apply.

    registry.json is normally written only by localm and is always well-formed,
    but it is a real, user-visible file (``localm info`` names it) that can be
    hand-edited, half-written, or left behind by an older/newer version. A single
    JSON-valid-but-wrong-shape entry must never crash a read / list / remove /
    dedup / sync operation and take the whole registry down with it: that is the
    same "a damaged file must never take the app down" guarantee load_registry
    already gives for whole-file corruption, extended from the file to the entry.
    Callers that only need the path skip an entry when this returns None; the
    user-facing lister marks it visibly corrupt so it can be dropped with
    ``localm rm <name>`` instead of forcing a hand-edit of the JSON.

    INVARIANT (grep ``_entry_path`` before adding a new registry consumer): EVERY
    site that iterates ``load_registry()`` (or looks an entry up by name) and then
    reads the entry's ``path`` / ``mmproj`` / ``source`` / ``model_type`` MUST route
    that access through this helper (or an explicit ``isinstance(entry, dict)``
    guard). A raw ``entry["path"]`` / ``entry.get("path")`` crashes on a non-dict
    or null/int path. The known consumers, all guarded, are: this module (list/info/vision/
    external/alias/dedup/sync), the MCP ``list_models``
    (plugins/mcpserver/server.py), the GUI ``/api/models`` + ``/api/vram-estimate``
    (plugins/gui/routes/models.py), the API ``model_detail``
    (inference/routes/models.py), the pull dedup scan (model_manager/pull.py), and
    the ComfyUI scan (model_manager/scan.py). There is deliberately no blanket
    syntactic linter for this - ``["path"]`` / ``.get("source")`` occur all over
    unrelated code (coder sessions, RAG cells, HF listings), so a scan would be
    false-positive noise; the guard is enforced by the malformed-entry test matrix
    (tests/test_model_dedup.py::TestMalformedRegistryResilience and
    tests/test_registry_corruption_consumers.py), which drives every consumer with
    the six BAD_ENTRIES shapes.
    """
    if not isinstance(entry, dict):
        return None
    p = entry.get(field)
    if not (isinstance(p, str) and p):
        return None
    # A '..' segment is malformed too, and this is the choke point where every
    # consumer can be made to agree on that. localm never WRITES one - _register
    # stores str(path.resolve()) - so a stored '..' means the file was hand-edited,
    # merged from another machine, or planted, and it is precisely the shape that
    # defeats a lexical containment test downstream (remove_model's gate sits in
    # front of shutil.rmtree). Rejecting it HERE means a foreign registry cannot
    # smuggle one into any consumer at all, rather than each consumer having to
    # remember to normalise. Checked under both path flavours because a registry
    # travels between machines and its entries can carry either separator - the
    # same reason inference/routes/models.py normalises them before taking a
    # basename.
    if any(part == ".." for part in PureWindowsPath(p).parts) or \
       any(part == ".." for part in PurePosixPath(p).parts):
        return None
    return p


def _detect_local_model_type(path: Path, *, is_gguf: bool, is_hf: bool,
                             is_blob: bool = False) -> tuple[str, dict]:
    """Deterministically classify a LOCAL model's type from HARD metadata only.
    Returns ``(model_type, gguf_metadata)``.

    A .gguf file or Ollama blob (the same GGUF byte format under a renamed file)
    is first checked for a vision-projector signal in its OWN GGUF metadata
    (``gguf_is_mmproj`` - ``general.architecture == "clip"``; see gguf.py) ->
    'mmproj'; then an embedding/pooling signal (``gguf_embedding_signal`` -
    architecture or a ``*.pooling_type`` key) -> 'embedding'; otherwise it is a
    llama.cpp text model -> 'llm'. An HF directory is classified from
    config.json: a LoRA/adapter dir -> 'lora'; an ``architectures`` class ending
    in ForCausalLM / LMHeadModel / ForConditionalGeneration -> 'llm'; anything we
    cannot resolve -> 'unknown' (never a silent 'llm').

    *gguf_metadata* (F8-PERSIST-ARCH-AND-EXPERT-COUNT) is ``gguf_registry_metadata``'s
    ``{"architecture", "expert_count"}`` dict for a GGUF/blob path, ``{}`` for an
    HF dir or an undetected type (nothing GGUF-shaped to read). The mmproj/
    embedding checks below already read this file's header once (``meta``,
    shared via the *meta* param both accept - MEASURED: this halves the cost
    of also wanting architecture, from three probe reads down to two); this
    only adds ``gguf_expert_count``'s own separate, equally-bounded read on
    top, so every caller that persists a registry entry gets architecture and
    expert count for close to the cost it already paid for the type alone.
    """
    try:
        if is_gguf or is_blob:
            meta = _gguf_metadata_probe(path)
            gguf_metadata = gguf_registry_metadata(path, meta=meta)
            if gguf_is_mmproj(path, meta=meta):
                return "mmproj", gguf_metadata
            if gguf_embedding_signal(path, meta=meta):
                return "embedding", gguf_metadata
            return "llm", gguf_metadata
        if is_hf:
            if (path / "adapter_config.json").exists():
                return "lora", {}
            cfg_path = path / "config.json"
            if cfg_path.exists():
                conf = json.loads(cfg_path.read_text(encoding="utf-8"))
                archs = conf.get("architectures") or []
                if isinstance(archs, list) and any(
                    isinstance(a, str) and a.endswith(_HF_LLM_ARCH_SUFFIXES)
                    for a in archs
                ):
                    return "llm", {}
    except Exception as e:
        # Detection is best-effort metadata reading; an unreadable/odd config.json
        # means "no hard signal", which is exactly 'unknown' - surfaced, not muted.
        logger.debug("local model-type detection failed for %s: %s", path, e)
    return "unknown", {}





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


@dataclass(frozen=True)
class ComfySource:
    """A known-good HuggingFace download source for one exact ComfyUI workflow
    model filename."""
    spec: str            # "owner/repo:filename" - fed straight to pull_model()
    model_type: str       # "diffusion-unet" | "text-encoder" | "vae" | "lora"
    comfy_subfolder: str  # ComfyUI models/<subfolder> this file belongs in
    size_bytes: int


# filename (exact, as ComfyUI's /object_info reports it) -> curated download source
# for a ComfyUI WORKFLOW model slot. This is a DIFFERENT keyspace from
# MODEL_SHORTCUTS above: MODEL_SHORTCUTS is keyed by a short alias a user TYPES
# for `localm pull <alias>`; this dict is keyed by an exact installed filename and
# is looked up automatically when ComfyUI preflight detects that filename missing
# from a workflow - never typed by a user directly. Each entry's HuggingFace
# source was verified by fetching the repo's live file tree, not assumed. Curated
# only (exact-filename lookup, no fuzzy/heuristic matching) - see
# resolve_comfy_model_source().
#
# ae.safetensors is sourced from the ungated Apache-2.0 FLUX.1-schnell repo rather
# than the gated (license-click-through) FLUX.1-dev repo: the file is
# byte-identical in both (confirmed by matching size), and sourcing it from the
# ungated repo avoids a HuggingFace license-gate failure for users who haven't
# accepted the FLUX.1-dev non-commercial license.
COMFY_MODEL_SOURCES: dict[str, ComfySource] = {
    "flux1-dev-Q8_0.gguf": ComfySource(
        "city96/FLUX.1-dev-gguf:flux1-dev-Q8_0.gguf",
        "diffusion-unet", "unet", 12_708_281_504),
    "clip_l.safetensors": ComfySource(
        "comfyanonymous/flux_text_encoders:clip_l.safetensors",
        "text-encoder", "clip", 246_144_152),
    "t5xxl_fp8_e4m3fn.safetensors": ComfySource(
        "comfyanonymous/flux_text_encoders:t5xxl_fp8_e4m3fn.safetensors",
        "text-encoder", "clip", 4_893_934_904),
    "ae.safetensors": ComfySource(
        "black-forest-labs/FLUX.1-schnell:ae.safetensors",
        "vae", "vae", 335_304_388),
}


def resolve_comfy_model_source(filename: str) -> Optional[ComfySource]:
    """The curated download source for *filename*, or None when it isn't one of
    the known-good exact-filename matches above."""
    return COMFY_MODEL_SOURCES.get(filename)


def resolve_spec(spec: str) -> str:
    return MODEL_SHORTCUTS.get(spec, spec)




def get_model_path(name: str, *, allow_direct_path: bool = False) -> Optional[Path]:
    """Resolve a model name/alias/path to the model file or directory.

    Returns the resolved Path, or None if not found.
    To also get a display name hint, use get_model_info().

    See get_model_info() for what ``allow_direct_path`` opts into, and why it
    defaults to off.
    """
    result = get_model_info(name, allow_direct_path=allow_direct_path)
    return result[0] if result else None




def unregistered_model_error(name) -> Optional[str]:
    """None when an UNTRUSTED caller may use *name* as a model name, else a
    ready-to-show message explaining the refusal.

    This is layer 2 of the model-name gate, and it exists for the error message.
    Layer 1 is get_model_info's ``allow_direct_path``, which is unconditional and
    is what actually stops a path from resolving. Without layer 2 a caller naming
    a path would get a bare "model not found", which is true but useless; with it
    they are told the name must be registered and which names exist.

    Returns None (allowed) in two non-name cases:

    * an empty/None *name*, which means "use the server default" and is not a
      name at all; and
    * an EMPTY registry, mirroring the same convention already used by the /v1
      registration check in http_server ("only enforce ... if the registry is not
      empty"), so a fresh install or a test harness with no registry is not made
      unusable. That is safe precisely BECAUSE layer 1 is unconditional: with an
      empty registry a path still fails to resolve, so skipping this check cannot
      re-open the direct-path hole - it only changes which message is shown.

    Callers raise their own exception type from the returned string (HTTP 400 in
    the jobs routes, ValueError in the MCP server, RuntimeError in the runner) so
    this module stays free of any transport dependency.
    """
    if not name:
        return None
    reg = _mm.load_registry()
    if not reg or name in reg:
        return None
    # Deliberately does NOT list the registered models. An earlier version did,
    # which was friendlier and wrong: this message is returned to callers holding
    # only `jobs` or an MCP scope, while ENUMERATING models requires `models:read`
    # (see inference/routes/models.py). That handed the full model inventory to any
    # scoped key, and cut against the opacity discipline a few lines away in
    # jobs/plug.py, where a foreign job id returns the same 404 as a missing one so
    # a caller cannot even confirm it exists. Name only what the caller already
    # told us, and point at the command that lists models for someone entitled to.
    return (f"Model {name!r} is not registered. Run 'localm list' to see the "
            "registered models, add one with 'localm pull <spec>', or register a "
            "local file with 'localm add <path>'. A filesystem path is not "
            "accepted here: only a command-line run may name a model by path.")


def get_model_info(name: str, *, allow_direct_path: bool = False):
    """Like get_model_path(), but returns (path, display_hint) or None.

    display_hint is a human-readable name when the original arg was an Ollama
    manifest path; otherwise it's None (the engine derives its own name).

    ``allow_direct_path`` opts into resolving *name* as a PATH ON DISK when it is
    not a registry entry (`localm run D:/models/foo.gguf`, an Ollama store, a
    HuggingFace directory). That is a documented CLI feature, so it is kept - but
    it is OFF by default, because the caller decides whether the name it holds is
    operator-typed or came off the wire, and only the caller can know that.

    Why the default is refuse (CodeQL 11-17, 48, 49, 65-71, 73-76, 108-111): the
    fallthrough accepted any path on disk, and the jobs plugin and the MCP server
    both reached it with an attacker-supplied name under a NON-privileged scope.
    Every downstream sink then ran on an attacker-named path - stat, an unbounded
    rglob, read_text of a whole directory - and for a directory, create_backend
    picks HFBackend, which used to enable transformers' remote-code flag and so
    imported and EXECUTED the directory's own .py. Refusing by default turns that
    from "any caller who forgot to check" into "only a caller that opted in".

    Pass True ONLY where *name* is operator-typed on the command line. The audited
    set is localm/cli/*, the `localm gui <model>` startup resolution, and the MCP
    server's own --model default. Anything reachable over HTTP or MCP keeps the
    default, and enforces registry membership on top (see http_server's
    registration check, jobs' _check_model_name, and MCPEngines.resolve_model).
    """
    reg = _mm.load_registry()
    if name in reg:
        epath = _entry_path(reg[name])   # None for a malformed entry -> fall through
        if epath is not None:
            p = Path(epath)
            if p.exists():
                return p, None

    if not allow_direct_path:
        return None

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




def _pick_mmproj_candidate(model_stem: str, names: List[str]) -> Optional[str]:
    """Disambiguate a single mmproj (vision projector) filename out of *names*
    (already filtered to mmproj-looking GGUF filenames) for a model named
    *model_stem*. A lone candidate wins outright; with more than one, prefer a
    name that shares the model's leading token, and give up (None) rather than
    guess when that still doesn't narrow it to one - never silently attach the
    wrong projector. Shared by find_sibling_mmproj (a directory glob) and
    pull.py's HF-repo-listing lookup (same disambiguation, a different
    candidate source)."""
    if not names:
        return None
    if len(names) == 1:
        return names[0]
    stem = model_stem.lower().replace("mmproj", "").split("-")[0].split(".")[0]
    matches = [n for n in names if stem and stem in n.lower()]
    return matches[0] if len(matches) == 1 else None


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
    by_name = {f.name: f for f in cands}
    picked = _pick_mmproj_candidate(p.stem, list(by_name.keys()))
    return by_name[picked] if picked else None




def get_model_mmproj(name: str, *, allow_direct_path: bool = False) -> Optional[str]:
    """The mmproj (vision projector) path for a model, if one is known.

    Priority: an explicit 'mmproj' recorded in the registry entry, else a sibling
    mmproj GGUF auto-detected next to the model file. Returns an absolute path
    string, or None when no projector is associated. This is what lets a GGUF keep
    vision after a GUI/registry model switch (VIS-1), the same way the CLI --mmproj
    flag does on a direct run.

    ``allow_direct_path`` is threaded through to get_model_info for the same reason
    it exists there: without it, the sibling-projector scan (CodeQL 69/70) would
    glob a directory named by an unregistered, caller-supplied path. Those two
    alerts share this root cause; unlike the jobs/MCP sinks they were not shown to
    have their own remote entry point, and are fixed here by identity, not because
    a remote chain was demonstrated for them."""
    reg = _mm.load_registry()
    entry = reg.get(name) if isinstance(reg, dict) else None
    # Through the choke point, not a raw read: the recorded projector is a stored
    # path from the same hand-editable file as ``path``, and it goes to the same
    # native mtmd loader, so it gets the same type check and ``..`` rejection. The
    # raw ``entry["mmproj"]`` this replaces raised TypeError on a null/int value
    # and passed a traversal component straight through. A malformed projector
    # falls through to the auto-detect below, which is the same recovery a
    # recorded-but-missing one already gets.
    recorded = _entry_path(entry, "mmproj")
    if recorded:
        mmp = Path(recorded)
        if mmp.exists():
            return str(mmp)
        # Recorded but gone: fall through to auto-detect rather than handing the
        # backend a dead path that would just fail the mtmd load.
    info = get_model_info(name, allow_direct_path=allow_direct_path)
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




# Filename/repo-id tokens that reliably signal a vision-language GGUF release,
# curated from real HuggingFace GGUF repo naming conventions (mradermacher,
# bartowski, ggml-org, unsloth). A NAME heuristic only - unlike gguf_is_mmproj
# (hard GGUF metadata baked into the file), the main LLM's own GGUF header
# carries no vision marker, so this is the only signal available before a
# companion projector is even known to exist. It is used SOLELY to decide
# whether pull.py prints an informational note when no mmproj sibling was
# found; it never changes what gets downloaded, verified, or how a model is
# typed - a false positive here costs one extra console line, never a wrong
# classification.
_VISION_NAME_TOKENS = frozenset({
    "vl", "vlm", "vision", "llava", "idefics", "idefics2", "idefics3",
    "internvl", "moondream", "pixtral", "paligemma", "smolvlm", "cogvlm",
    "fuyu", "kosmos", "kosmos2",
})


def _looks_like_vision_gguf_name(repo_id: str, filename: str) -> bool:
    """Best-effort NAME-based signal that a GGUF pull spec names a
    vision-language model - see ``_VISION_NAME_TOKENS`` for what this checks
    and why it is name-only. Used by pull.py to decide whether a missing
    mmproj projector is worth flagging at pull time; never used to type or
    refuse a pull."""
    ident = f"{repo_id}/{filename}".lower()
    tokens = re.split(r"[^a-z0-9]+", ident)
    if any(t in _VISION_NAME_TOKENS for t in tokens):
        return True
    return "minicpm-v" in ident or "minicpmv" in ident


def vision_capable_models() -> List[str]:
    """Registered model names that can accept image INPUT on THIS install.

    Two independent ways a model qualifies: a HuggingFace-format directory with
    vision metadata, or a GGUF chat model carrying a mmproj (vision projector)
    that get_model_mmproj() resolves - the SAME lookup the actual load path
    uses (an explicitly recorded mmproj, or one auto-detected sitting next to
    the model file). A name listed here is one the loader can genuinely put an
    image through, not a guess from static metadata alone; a standalone mmproj
    or embedding entry is not itself a model to switch to and is excluded via
    the model_type gate. Used to ROUTE an image to a model already known to be
    vision-capable instead of dead-ending."""
    out: List[str] = []
    for name, info in _mm.load_registry().items():
        epath = _entry_path(info)   # skip malformed entries (a str entry's .get
        if epath is None:           # would raise AttributeError, not caught below)
            continue
        try:
            p = Path(epath)
        except (TypeError, ValueError):
            continue
        if p.is_dir():
            if _hf_is_vision(p):
                out.append(name)
        elif _entry_path(info, "model_type") == "llm" and get_model_mmproj(name):
            out.append(name)
    return sorted(out)




def _active_model_missing_mmproj(active_model_path: str) -> Optional[tuple]:
    """(name, repo_id) when *active_model_path* resolves to a registered LLM
    GGUF pulled from an HF repo with no mmproj recorded yet - #957's
    already-pulled-before-the-fix state. None when the path is not
    registered, not a candidate, or already has one (including a resolved-
    but-failed one, which is ``mmproj_failed``'s case, not this one)."""
    reg = _mm.load_registry()
    try:
        names = find_aliases_by_path(Path(active_model_path), reg=reg)
    except OSError:
        return None
    if not names:
        return None
    entry = reg.get(names[0])
    if not isinstance(entry, dict):
        return None
    source = str(entry.get("source", ""))
    if (entry.get("model_type") == "llm" and source.startswith("hf:")
            and not entry.get("mmproj")):
        return names[0], source[len("hf:"):]
    return None


def persist_cli_mmproj(name: str, mmproj_path: str) -> Optional[str]:
    """Record a CLI ``--mmproj`` override onto *name*'s registry entry, once the
    caller has already confirmed it genuinely loaded for this run (the backend
    reported ``supports_images=True``) - this function never checks that itself
    and must never be called on an unconfirmed load, or a broken projector could
    get recorded as working, making vision_capable_models() list a model whose
    vision does not actually load: a NEW false-positive surface in the exact
    area #1073 just fixed.

    Every other mmproj-discovery path persists (pull.py's auto-attach, the
    registry's own recorded field) - the CLI override was the one path that
    never wired up to the same persistence, which is what this closes. Returns
    a one-line status to print to the user, or None when there is nothing worth
    saying (already recorded with this exact path, or the entry is not a GGUF
    chat model this field means anything for).

    Never silently overwrites a DIFFERENT already-recorded mmproj (hard-won
    rule: never override a user's prior explicit choice without informing
    them) - a differing recorded value returns an explanatory note instead of
    writing, since a one-off --mmproj may be a deliberate experiment the user
    does not want to make permanent."""
    reg = _mm.load_registry()
    entry = reg.get(name)
    if not isinstance(entry, dict):
        return None
    epath = _entry_path(entry)
    if epath is None:
        return None
    try:
        p = Path(epath)
    except (TypeError, ValueError):
        return None
    if p.is_dir() or entry.get("model_type") != "llm":
        return None   # mmproj means nothing on an HF dir / non-chat entry
    try:
        resolved = str(Path(mmproj_path).resolve())
    except (TypeError, ValueError, OSError):
        return None
    recorded = _entry_path(entry, "mmproj")
    if recorded == resolved:
        return None
    if recorded:
        return (f"Note: this run used a different vision projector than the "
                f"one already recorded for '{name}' ({recorded}) - not "
                f"overwriting it. Pass the same --mmproj again next time, or "
                f"`localm rm {name}` and re-add it to replace the recorded "
                f"projector.")

    def _mutator(r):
        e = r.get(name)
        if isinstance(e, dict):
            e["mmproj"] = resolved
    _mm.update_registry(_mutator)
    return (f"Recorded this vision projector for '{name}' - future runs of "
            f"{name} will use it automatically, no --mmproj needed.")


def vision_input_guidance(mmproj_failed: bool = False,
                          active_model_path: Optional[str] = None) -> str:
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
    incompatible with the model, or the mtmd vision runtime is unavailable.

    *active_model_path* (the loaded backend's own resolved model path, when the
    caller has one) is #957's third, previously-uncovered case: the ACTIVE model
    IS a vision-capable GGUF pulled from an HF repo, but no mmproj was ever
    recorded for it (an install predating the auto-attach fix, or interrupted
    before sync_models_dir's backfill has run). Without this check the generic
    "no vision model is registered yet" branch below fires for exactly this
    user - telling them to pull a model they already have."""
    import importlib.util
    if mmproj_failed:
        head = ("This model cannot accept image input: a vision projector (mmproj) "
                "was provided but failed to load - it may be incompatible with this "
                "model, or the mtmd vision runtime is unavailable (see the server "
                "log for the mtmd error).")
    elif active_model_path and (missing := _active_model_missing_mmproj(active_model_path)):
        name, repo_id = missing
        return (
            "This model cannot accept image input yet: "
            f"'{name}' is a vision-capable model, but its vision projector "
            "(mmproj) has not been downloaded. localm checks for it "
            "automatically the next time it starts (subject to your network "
            "setting) - restart localm, or reload the Models page. If network "
            "access is off (net_mode=off), turn it on, or pull the projector "
            f"explicitly: `localm pull {repo_id} --mmproj <repo>:<file>`."
        )
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
        return (f"{head} No vision-capable model could be confirmed in your "
                f"library - pull a vision-capable HuggingFace model such as "
                f"a Gemma 3 vision or Qwen2.5-VL checkpoint "
                f"(`localm pull <repo>`), then run it with --image.")
    return (f"{head} No vision-capable model could be confirmed in your "
            f"library. GGUF vision works too (the built-in mtmd path) - pull "
            f"a vision-capable GGUF model with its projector, e.g. "
            f"`localm pull <repo> --mmproj <repo>:<file>`. Or install the "
            f"HuggingFace stack (`pip install \"localm[gpu]\"`) and load a "
            f"vision-capable model (Gemma 3 vision / Qwen2.5-VL) instead.")




def list_models(type_filter: Optional[str] = None) -> None:
    reg = _mm.load_registry()
    if type_filter:
        reg = {k: v for k, v in reg.items()
               if isinstance(v, dict) and v.get("model_type", "llm") == type_filter}
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
        epath = _entry_path(info)
        if epath is None:
            # A hand-corrupted / wrong-shape entry (not a dict, or no usable
            # path). Show it VISIBLY so the user sees which entry is broken and
            # can drop it with `localm rm <name>` - never crash the whole listing
            # on one bad entry (extends load_registry's damaged-file guarantee).
            table.add_row(f"[red]{name}[/red]", "[red]corrupt[/red]", "-",
                          "[red]-[/red]", "-",
                          "[red](malformed registry entry)[/red]")
            continue
        path = Path(epath)
        source = str(info.get("source", "local"))
        role = str(info.get("model_type", "llm"))

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

    def _is_missing(i) -> bool:
        # A well-formed entry whose file is simply gone (the relocate/autoprune
        # case). Malformed entries are already shown as 'corrupt' above, so they
        # do not trigger the missing-file hint.
        p = _entry_path(i)
        return p is not None and not Path(p).exists()

    if any(_is_missing(i) for i in reg.values()):
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
    epath = _entry_path(_mm.load_registry().get(name))   # None -> malformed/absent
    return epath is not None and is_external_path(epath)


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
    out: List[str] = []
    for name, info in reg.items():
        epath = _entry_path(info)   # skip a malformed sibling entry, never crash
        if epath is None:
            continue
        try:
            if str(Path(epath).resolve()) == target:
                out.append(name)
        except OSError:
            continue
    return sorted(out)




def find_by_sha256(digest: str, reg: Optional[dict] = None) -> List[str]:
    """Registered names whose stored sha256 matches *digest* (case-insensitive)."""
    if not digest:
        return []
    reg = reg if reg is not None else _mm.load_registry()
    d = digest.lower()
    return sorted(
        name for name, info in reg.items()
        # Guard a malformed sibling entry (not a dict, or a non-string sha256):
        # a corrupt entry must not crash the dedup scan run by add / pull.
        if isinstance(info, dict) and str(info.get("sha256", "") or "").lower() == d
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
        path = _entry_path(info)   # skip a malformed entry, never crash the scan
        if path is None:
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
    # Sanitize the user-supplied new name through the SAME filter add_local /
    # pull / sync all use (GAP-CLI-1), so `localm alias real ../../evil`, an empty
    # name, or `a/b/c` can never become a raw registry key. alias was the one
    # registry-key-creating path that skipped this guard. Internal callers already
    # pass a sanitized name, so re-sanitizing is a harmless no-op for them.
    safe_name = _sanitize_name(new_name)
    if safe_name in reg:
        console.print(f"[red]Name already in use:[/red] {safe_name}")
        return False
    # Atomic RMW (re-read inside the lock) so a concurrent writer is not lost.
    def _apply(r: dict) -> None:
        if existing in r and safe_name not in r:
            r[safe_name] = dict(r[existing])
    _mm.update_registry(_apply)
    console.print(
        f"[green]✓[/green] [bold]{safe_name}[/bold] is now an alias of "
        f"[bold]{existing}[/bold]"
    )
    return True




def rename_model(old_name: str, new_name: str) -> bool:
    """
    Rename a registered model from *old_name* to *new_name*. Thin bool-only
    wrapper over :func:`rename_model_with_notes`, for CLI simplicity and
    backward compatibility (a caller that also needs the migration notes -
    e.g. the GUI route, which must show them to the user rather than let
    them sit in the server log - should call that instead). See its
    docstring for the full behavior.
    """
    renamed, _notes = rename_model_with_notes(old_name, new_name)
    return renamed


def rename_model_with_notes(old_name: str, new_name: str) -> "tuple[bool, List[str]]":
    """
    Rename a registered model from *old_name* to *new_name*: MOVES the
    registry entry to a new key (unlike alias_model, which copies - *old_name*
    stops working here), and best-effort migrates every OTHER place inside
    <data dir> that stores the plain name string (config.json's pinned_models
    / embedding_model / coder_reviewer_model, scheduled jobs' `model` field,
    RAG collection metadata). A per-project ``.localcoder/config.toml``
    ``model`` setting lives OUTSIDE <data dir> (in the user's own project
    repo, discoverable only relative to a `cwd` a coder session supplies) and
    cannot be enumerated or migrated from here - always reported in the
    returned notes, never silently dropped (AGENTS.md rule 5).

    Sibling aliases - other registry entries whose file happens to be the same
    one *old_name* pointed at - are left untouched: they are independent
    names for that file, exactly like any alias `localm alias` creates, and
    renaming one name must not delete or repoint the others.

    Returns ``(False, [])`` when *old_name* is not registered, or *new_name*
    (after the same sanitizing alias_model applies) is already taken by a
    DIFFERENT entry. Renaming a name to itself (post-sanitize) is a no-op
    success with no notes (nothing needed migrating). On a genuine rename,
    returns ``(True, notes)`` where *notes* is what
    :func:`_migrate_model_references` reports.
    """
    reg = _mm.load_registry()
    if old_name not in reg:
        console.print(f"[red]Not found:[/red] {old_name}")
        return False, []
    safe_name = _sanitize_name(new_name)
    if safe_name == old_name:
        console.print(f"[dim]'{old_name}' is already named '{safe_name}'[/dim]")
        return True, []
    if safe_name in reg:
        console.print(f"[red]Name already in use:[/red] {safe_name}")
        return False, []

    # Atomic RMW (re-read inside the lock), and - unlike alias_model above -
    # actually notice whether the move happened: a concurrent writer could
    # still take old_name or safe_name between the precheck and this call, and
    # claiming success when nothing moved would violate AGENTS.md rule 5 (never
    # report success on a step that silently did nothing).
    moved = False

    def _apply(r: dict) -> None:
        nonlocal moved
        if old_name in r and safe_name not in r:
            r[safe_name] = r.pop(old_name)
            moved = True

    _mm.update_registry(_apply)
    if not moved:
        reg_now = _mm.load_registry()
        if old_name not in reg_now:
            console.print(f"[red]Not found:[/red] {old_name}")
        else:
            console.print(f"[red]Name already in use:[/red] {safe_name}")
        return False, []

    console.print(
        f"[green]✓[/green] Renamed [bold]{old_name}[/bold] -> [bold]{safe_name}[/bold]"
    )
    notes = _migrate_model_references(old_name, safe_name)
    for note in notes:
        console.print(f"[dim]{note}[/dim]")
    return True, notes


def _migrate_model_references(old_name: str, new_name: str) -> List[str]:
    """Best-effort: rewrite every *old_name* reference this process can reach
    inside <data dir>, after rename_model has already moved the registry
    entry. Never raises - the registry rename has already succeeded and must
    not be undone because a secondary reference could not be updated; a site
    that fails to migrate is reported as a note (AGENTS.md rule 5: surfaced,
    not swallowed), not a crash. Returns human-readable notes: what changed,
    and what could not be reached at all.
    """
    notes: List[str] = []

    try:
        def _apply_cfg(cfg: dict) -> None:
            pinned = cfg.get("pinned_models")
            if isinstance(pinned, list) and old_name in pinned:
                cfg["pinned_models"] = [new_name if n == old_name else n for n in pinned]
            if cfg.get("embedding_model") == old_name:
                cfg["embedding_model"] = new_name
            if cfg.get("coder_reviewer_model") == old_name:
                cfg["coder_reviewer_model"] = new_name
        update_config(_apply_cfg)
    except Exception as e:
        logger.debug("rename_model: config migration failed for %s -> %s: %s",
                     old_name, new_name, e)
        notes.append(f"Could not update config.json references: {e}")

    try:
        from localm.plugins.builtin.jobs.store import JobStore
        store = JobStore()
        migrated = 0
        for job in store.list():
            if job.model == old_name:
                store.update(job.id, model=new_name)
                migrated += 1
        if migrated:
            notes.append(f"Updated {migrated} scheduled job(s) to the new name")
    except Exception as e:
        logger.debug("rename_model: jobs migration failed for %s -> %s: %s",
                     old_name, new_name, e)
        notes.append(f"Could not update scheduled jobs: {e}")

    try:
        from localm.rag.store import Collection, collection_names
        migrated = 0
        for cname in collection_names():
            try:
                coll = Collection(cname)
            except Exception:
                continue
            # No public label-only setter exists (reembed() would trigger a
            # full re-index just to relabel a display string), so this goes
            # through the same private _meta + _save_meta() reembed() itself
            # uses to persist that one field.
            if coll._meta.get("embedding_model") == old_name:
                coll._meta["embedding_model"] = new_name
                coll._save_meta()
                migrated += 1
        if migrated:
            notes.append(f"Updated {migrated} RAG collection metadata record(s)")
    except Exception as e:
        logger.debug("rename_model: RAG collection migration failed for %s -> %s: %s",
                     old_name, new_name, e)
        notes.append(f"Could not update RAG collection metadata: {e}")

    notes.append(
        "A per-project .localcoder/config.toml 'model' setting (if any) lives "
        "outside <data dir> and was NOT updated - fix it by hand in any "
        "project that pinned this model."
    )
    return notes




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
        "  [a]lias (new name, same file)  [c]opy into <data dir>/models  "
        "[m]ove into <data dir>/models  [r]egister anyway  [s]kip",
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
    mmproj: Optional[Path] = None,
    architecture: Optional[str] = None,
    expert_count: Optional[int] = None,
) -> None:
    """*mmproj*, when given, is a vision projector already verified and placed
    on disk (pull.py's job) and is recorded on the entry so get_model_mmproj
    finds it without depending on the directory-sibling fallback (#957).

    *architecture* / *expert_count* (F8-PERSIST-ARCH-AND-EXPERT-COUNT): the
    GGUF header's own ``general.architecture`` and ``<arch>.expert_count``,
    read once at registration (gguf_registry_metadata) so the local model
    list can show the same real architecture/MoE badge HuggingFace search
    results already do, instead of only ever having a name-guess. Both use
    ``is not None`` throughout this module, NEVER a truthiness check:
    ``expert_count=0`` is a real, confirmed fact (a dense model's header WAS
    read and genuinely has no experts) and must stay written and distinct
    from an entry that was never checked at all (the key absent entirely) -
    collapsing the two would show a real MoE model as confirmed-dense the
    moment a caller defaults a missing field to 0."""
    entry = {"path": str(path.resolve()), "source": source, "model_type": model_type}
    if sha256:
        entry["sha256"] = sha256.lower()
    if mmproj:
        entry["mmproj"] = str(Path(mmproj).resolve())
    if architecture is not None:
        entry["architecture"] = architecture
    if expert_count is not None:
        entry["expert_count"] = expert_count
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
    # entries backfilled with architecture/expert_count (F8-PERSIST-ARCH-AND-
    # EXPERT-COUNT), bounded per call - see sync_models_dir's _BACKFILL_CAP
    backfilled: int = 0
    # entries backfilled with a vision projector (#957), bounded per call
    # separately from `backfilled` - see sync_models_dir's _MMPROJ_BACKFILL_CAP
    mmproj_backfilled: int = 0
    note: str = ""          # a warning to surface (e.g. autoprune guardrail tripped)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.flagged or self.restored or self.pruned
                    or self.backfilled or self.mmproj_backfilled)




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

    known = set()
    for entry in reg.values():
        p = _entry_path(entry)   # skip malformed entries; never crash the sync
        if p is None:
            continue
        try:
            known.add(str(Path(p).resolve()))
        except OSError:
            continue

    added = 0
    if _mm.MODELS_DIR.is_dir():
        for child in sorted(_mm.MODELS_DIR.iterdir()):
            try:
                # HuggingFace model directory.
                if child.is_dir() and (child / "config.json").is_file():
                    resolved = str(child.resolve())
                    if resolved in known:
                        continue
                    mtype, _gmeta = _detect_local_model_type(child, is_gguf=False, is_hf=True)
                    _mm._register(_unique_registry_name(reg, child.name), child,
                                  model_type=mtype)
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
                mtype, gmeta = _detect_local_model_type(child, is_gguf=True, is_hf=False)
                _mm._register(_unique_registry_name(reg, child.stem), child,
                              model_type=mtype, architecture=gmeta.get("architecture"),
                              expert_count=gmeta.get("expert_count"))
                reg = _mm.load_registry()
                known.add(resolved)
                added += 1
            except OSError:
                continue

    # Reconcile missing / restored / pruned against the (possibly grown) registry.
    # The whole load-reconcile-save cycle runs inside a single update_registry()
    # call so it is atomic and cross-process-locked, like every other write path
    # in this file (_register() above, add_local, remove_model, alias, ...):
    # this loop does real per-entry filesystem stat calls (path.exists()), which
    # take actual wall-clock time, and a concurrent update_registry() write (a
    # `pull`/`rm`/`alias` in another thread or process) landing during that scan
    # must not be silently discarded by a blind save_registry() overwrite at the
    # end - which is exactly what a load-once/mutate/save-once shape allows.
    models_root = _mm.MODELS_DIR.resolve()

    def _under_models_dir(p: Path) -> bool:
        # Same predicate as everywhere else, with the root hoisted out of the
        # per-entry loop below (see is_owned_model_path). This one was already
        # correct; it is routed through the shared helper so a future edit cannot
        # leave the repo with two definitions that disagree again.
        return is_owned_model_path(p, models_root)

    flagged = restored = pruned = 0
    backfilled = 0
    mmproj_backfilled = 0
    mmproj_attempts = 0    # network round-trips spent this call (cap input);
                            # mmproj_backfilled counts only real SUCCESSES -
                            # see the increment site below for why they differ
    note = ""
    backed_up = False
    # F8-PERSIST-ARCH-AND-EXPERT-COUNT backfill: an entry registered before this
    # feature existed has no architecture/expert_count at all. Opportunistic,
    # bounded, self-correcting - same shape as #1030's RAG stats-cache backfill
    # (measured there: a real one-time cost under the write lock, cheap again on
    # every later read). Capped per sync call (not "every missing entry, every
    # launch") because unlike a per-collection backfill this scan could otherwise
    # be unbounded - a user with 50 pre-existing models would pay 50x the ~200ms
    # measured single-file cost (see gguf_registry_metadata's docstring) on the
    # very first post-upgrade launch. A handful per call makes steady, unnoticed
    # progress across a few ordinary restarts instead.
    _BACKFILL_CAP = 5
    # #957 mmproj backfill: SAME shape (bounded, opportunistic, self-correcting
    # across ordinary restarts) but a much more expensive unit of work - an HF
    # repo listing plus, when found, a real file download, not a local read -
    # so it gets its OWN, smaller budget rather than sharing _BACKFILL_CAP.
    # Sharing one counter would let a run of architecture-only entries starve
    # the (rarer, more urgent) vision-projector backfill's budget, or vice
    # versa: a handful of slow network fetches would stall the cheap local
    # metadata reads for the rest of the call.
    _MMPROJ_BACKFILL_CAP = 3
    net_blocked_mmproj = []   # entry names skipped this call by net_mode=off

    def _reconcile(reg: dict) -> None:
        nonlocal flagged, restored, pruned, note, backed_up, backfilled
        nonlocal mmproj_backfilled, mmproj_attempts

        # Managed models = those whose file lives under the models folder.
        managed = [
            name
            for name, entry in reg.items()
            if _entry_path(entry) is not None
            and _under_models_dir(Path(_entry_path(entry)))
        ]
        managed_missing = [n for n in managed if not Path(_entry_path(reg[n])).exists()]

        # Guardrail: if pruning would delete *every* managed model at once, the
        # folder is almost certainly unavailable (unmounted drive, wrong path)
        # rather than the user having deleted everything - refuse to prune and
        # flag instead.
        suspicious = prune and len(managed) >= 2 and len(managed_missing) == len(managed)

        for name in list(reg.keys()):
            entry = reg[name]
            # Skip a malformed / non-dict entry: it has no valid path to
            # reconcile, and (crucially) `entry` may not be a dict, so
            # entry.get/pop below would raise. It stays in the registry and is
            # shown 'corrupt' by list_models.
            path_str = _entry_path(entry)
            if path_str is None:
                continue
            path = Path(path_str)

            if path.exists():
                # A previously-missing model is back - clear the flag.
                if entry.pop("missing", None):
                    restored += 1
                # Opportunistic architecture/expert_count backfill (see
                # _BACKFILL_CAP above) - only a GGUF file (the format this
                # metadata comes from), only when BOTH keys are genuinely
                # absent (an entry with either key, even a stored 0/false, was
                # already resolved and must never be re-read or overwritten),
                # and only up to the per-call cap.
                if (backfilled < _BACKFILL_CAP and path.suffix.lower() == ".gguf"
                        and "architecture" not in entry and "expert_count" not in entry):
                    gmeta = gguf_registry_metadata(path)
                    if gmeta.get("architecture") is not None:
                        entry["architecture"] = gmeta["architecture"]
                    if gmeta.get("expert_count") is not None:
                        entry["expert_count"] = gmeta["expert_count"]
                    backfilled += 1
                # #957 mmproj backfill (see _MMPROJ_BACKFILL_CAP above): an
                # already-registered LLM pulled from an hf: source, with no
                # mmproj recorded, gets the exact same same-repo auto-attach
                # a fresh pull already does - retroactively, using the source
                # this entry already carries. Cheap candidacy check first
                # (mmproj_backfill_candidate, no I/O) so the cap is spent only
                # on entries that could possibly qualify.
                if mmproj_attempts < _MMPROJ_BACKFILL_CAP:
                    from localm.model_manager.pull import (
                        backfill_mmproj_for_entry, mmproj_backfill_candidate)
                    if mmproj_backfill_candidate(entry, path):
                        from localm.netpolicy import network_mode
                        if network_mode() == "off":
                            # Loud-at-the-right-place, not here: sync_models_dir
                            # runs silently on every launch (see its own
                            # docstring), so this is recorded for the caller to
                            # surface via `note`, not printed - the actual
                            # chat-time surface is vision_input_guidance, which
                            # names net_mode explicitly when the user tries to
                            # use the feature and finds it still missing. Not
                            # counted against mmproj_attempts: it costs no
                            # network round-trip, so it should not starve the
                            # budget the cap exists to protect.
                            net_blocked_mmproj.append(name)
                        else:
                            mmproj_attempts += 1
                            found = backfill_mmproj_for_entry(entry, path)
                            if found is not None:
                                # found's own path already passed
                                # _safe_models_filename inside
                                # _maybe_fetch_repo_mmproj several call-frames
                                # away (pull.py) - re-verify locally, at the
                                # point this HF-repo-derived path is actually
                                # written into the registry, rather than
                                # trusting a distant caller's guarantee. Same
                                # resolve-then-compare idiom as
                                # is_owned_model_path above (a raw string
                                # prefix check would wrongly accept a sibling
                                # directory like <dir>-evil).
                                resolved = found.resolve()
                                if resolved.parent == path.resolve().parent:
                                    # Only a genuine attach counts as
                                    # "backfilled" - a repo that was checked
                                    # and genuinely has no projector (found is
                                    # None) must NOT count: nothing was
                                    # written, and ModelSyncResult.changed
                                    # must not fire for a no-op reconciliation
                                    # pass.
                                    entry["mmproj"] = str(resolved)
                                    mmproj_backfilled += 1
                continue

            # File is gone.
            if prune and not suspicious and _under_models_dir(path):
                if not backed_up:
                    # Snapshot the registry before the first deletion so the
                    # sync can be reverted one step if it goes wrong.
                    _mm._backup_registry()
                    backed_up = True
                del reg[name]
                pruned += 1
            elif not entry.get("missing"):
                entry["missing"] = True
                flagged += 1

        if suspicious:
            note = (
                f"Skipped autoprune: all {len(managed)} models under the models "
                "folder appear missing - is the folder/drive available? Left "
                "them flagged rather than deleting the registry."
            )

    _mm.update_registry(_reconcile)

    if net_blocked_mmproj:
        # Precise per constraint: which model(s), that the projector is
        # missing, and that net_mode is why - never collapsed with "no
        # projector found" (that is a different, silent-by-design outcome;
        # see _hf_repo_files/_maybe_fetch_repo_mmproj's own distinction
        # between "could not look" and "looked and found nothing").
        blocked_note = (
            f"{len(net_blocked_mmproj)} model(s) may be missing a vision "
            f"projector ({', '.join(net_blocked_mmproj)}) but network access "
            "is off (net_mode=off), so localm did not check. Enable network "
            "access to let localm look, or pull the projector explicitly with "
            "--mmproj."
        )
        note = f"{note} {blocked_note}".strip() if note else blocked_note

    return ModelSyncResult(
        added=added, flagged=flagged, restored=restored, pruned=pruned,
        backfilled=backfilled, mmproj_backfilled=mmproj_backfilled, note=note
    )




def _same_volume(a: Path, b: Path) -> bool:
    """True when *a* and *b* live on the same volume, so a move between them is a
    rename rather than a copy+delete. ``st_dev`` is the device id on POSIX and the
    volume serial on Windows, which is why this beats comparing drive letters (it
    is right for junctions, mount points and UNC paths too).

    Fails SAFE: if the volume cannot be read, return False so the caller keeps its
    full free-space check. A wrong "same volume" would skip a check that is real
    and let a cross-volume move fill the disk; a wrong "different volume" only
    asks for space we may not need.
    """
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def _space_needed(sources_on: Path, action: str, total: int) -> int:
    """Bytes that must be free in MODELS_DIR to perform *action*.

    A same-volume move is an os.rename (shutil.move's fast path): it needs ~0
    extra bytes, so demanding the whole model size falsely refused
    `localm add --on-duplicate move` whenever free < model size - which is exactly
    when a user picks move over copy (REG-450). A copy, or a cross-volume move
    (copy+delete under the hood), really does need the bytes.
    """
    if action == "move" and _same_volume(sources_on, _mm.MODELS_DIR):
        return 0
    return total


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
        if not _mm._check_disk_space(_mm.MODELS_DIR, _space_needed(path, action, total)):
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
    # Every source is a sibling of *path*, so one volume check covers them all.
    if not _mm._check_disk_space(_mm.MODELS_DIR,
                                 _space_needed(path.parent, action, total)):
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
            # Two full hashes of a multi-GB model, either side of the copy. This
            # was the longest silence in the whole store path: the "Copying ..."
            # line above is printed once and then nothing moves for minutes,
            # twice, with the copy itself in between.
            pre_digest = _verify_digest(src, purpose="to check the source before copying")
            shutil.copy2(src, dest)
            post_digest = _verify_digest(dest, purpose="to confirm the copy")
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


def _name_collision(model_name: str, p: Path, reg: dict) -> Optional[str]:
    """The conflicting path, if *model_name* is already registered pointing at
    a genuinely different file than *p* - the one case a non-interactive
    caller can never resolve (no terminal to confirm an overwrite) and must
    refuse up front, BEFORE moving or copying anything into place
    (NEW-STORE-MOVE-REGISTRY-DESYNC: add_local used to move the file first and
    only discover this after, when it was too late to avoid displacing it).
    None when there is no conflict: the name is free, or already correctly
    points at *p* itself (same path-identity aliasing _register_with_dedup
    uses for its own no-op case)."""
    if model_name not in reg:
        return None
    if model_name in find_aliases_by_path(p, reg):
        return None
    return _entry_path(reg[model_name]) or "?"


def _register_with_dedup(
    model_name: str,
    p: Path,
    source: str,
    *,
    on_duplicate: str = "ask",
    digest: Optional[str] = None,
    size: Optional[int] = None,
    model_type: str = "llm",
    mmproj: Optional[Path] = None,
    architecture: Optional[str] = None,
    expert_count: Optional[int] = None,
) -> bool:
    """
    Register a model, detecting duplicates first.

    Identity tiers: resolved path (instant), then stored sha256 when a *digest*
    for the new file is known, then file *size* alone (the ``--fast`` heuristic,
    used only when no digest was computed). ``on_duplicate`` is one of
    "ask" / "alias" / "copy" / "move" / "register" / "skip" - "ask" prompts
    interactively and degrades to "skip" without a TTY.

    *mmproj*, when given, is a verified vision projector to record on the
    entry (see ``_register``); backfilled onto an already-registered same-file
    entry the same way a fresh ``digest`` is. *architecture* / *expert_count*
    (F8-PERSIST-ARCH-AND-EXPERT-COUNT) backfill the same way - see ``_register``'s
    docstring for why ``expert_count=0`` must still backfill (a confirmed fact,
    not "nothing to write") while an entry that already HAS either key, even a
    falsy one, is never overwritten.

    Returns True when *model_name* ends up correctly registered for *p*
    (freshly registered, aliased, deduped, or already correct) - False for
    every path where nothing was written: a real name/content conflict
    declined interactively, or skipped because there was no terminal to ask
    (NEW-STORE-MOVE-REGISTRY-DESYNC). A caller that already moved or copied
    *p* into place before calling this MUST check the result rather than
    assume success - see add_local's callers for why.
    """
    import click

    reg = _mm.load_registry()
    aliases = find_aliases_by_path(p, reg)

    # Same name, same file - true no-op (but backfill a fresh digest / mmproj /
    # architecture / expert_count)
    if model_name in aliases:
        console.print(
            f"[yellow]'{model_name}' is already registered for this exact "
            f"file[/yellow] [dim]({p})[/dim]"
        )
        need_sha_backfill = bool(digest) and not reg[model_name].get("sha256")
        need_mmproj_backfill = bool(mmproj) and not reg[model_name].get("mmproj")
        # "key" not in e (not a truthiness check): expert_count=0 is a real,
        # confirmed value that must still overwrite a genuinely absent key, and
        # an entry that already has EITHER key (even 0/falsy) must never be
        # clobbered by a later call that could not itself determine them.
        need_arch_backfill = architecture is not None and "architecture" not in reg[model_name]
        need_expert_backfill = expert_count is not None and "expert_count" not in reg[model_name]
        if need_sha_backfill or need_mmproj_backfill or need_arch_backfill or need_expert_backfill:
            def _backfill(r: dict) -> None:      # atomic RMW
                e = r.get(model_name)
                if not isinstance(e, dict):
                    return
                if need_sha_backfill and not e.get("sha256"):
                    e["sha256"] = digest.lower()
                if need_mmproj_backfill and not e.get("mmproj"):
                    e["mmproj"] = str(Path(mmproj).resolve())
                if need_arch_backfill and "architecture" not in e:
                    e["architecture"] = architecture
                if need_expert_backfill and "expert_count" not in e:
                    e["expert_count"] = expert_count
            _mm.update_registry(_backfill)
        others = [a for a in aliases if a != model_name]
        if others:
            console.print(f"[dim]Also registered as: {', '.join(others)}[/dim]")
        return True

    # Same name, DIFFERENT file - real conflict, never overwrite silently
    old_path = _name_collision(model_name, p, reg)
    if old_path is not None:
        console.print(
            f"[yellow]'{model_name}' already points to a different file:"
            f"[/yellow] {old_path}"
        )
        if sys.stdin.isatty():
            if not click.confirm(f"  Overwrite '{model_name}' with {p}?"):
                console.print("[dim]Skipped.[/dim]")
                return False
        else:
            console.print("[dim]Non-interactive session - skipped. "
                          "Pick another name with -n.[/dim]")
            return False

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
            return False
        if action == "alias":
            alias_model(dup_names[0], model_name)
            return True
        if action in ("copy", "move"):
            # Capture the pre-move resolved path BEFORE _store_into_models_dir
            # touches anything - it's the key the alias-relink below matches on.
            moved_from = str(p.resolve())
            try:
                dest = _mm._store_into_models_dir(p, action)
            except RuntimeError as e:
                console.print(f"[red]{e}[/red]")
                return False
            if action == "move":
                dest_str = str(dest.resolve())
                entry = {"path": dest_str, "source": source, "model_type": model_type}
                if digest:
                    entry["sha256"] = digest.lower()
                if mmproj:
                    entry["mmproj"] = str(Path(mmproj).resolve())
                if architecture is not None:
                    entry["architecture"] = architecture
                if expert_count is not None:
                    entry["expert_count"] = expert_count

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
                return True
            p = dest
        # action == "register" falls through unchanged

    _mm._register(model_name, p, source, sha256=digest, model_type=model_type,
                  mmproj=mmproj, architecture=architecture, expert_count=expert_count)
    console.print(f"[green]✓[/green] Registered [bold]{model_name}[/bold]")
    return True




def is_owned_model_path(path, models_root: Optional[Path] = None) -> bool:
    """THE definition of "this file is localm's to delete": it lives strictly
    inside ``<data dir>/models``.

    Every caller that decides whether a registered file is managed MUST route
    through this. Today that is the deletion in :func:`remove_model`, the
    managed/external split in :func:`sync_models_dir`, and the "PERMANENTLY
    deletes" wording in the ``localm rm`` confirmation prompt
    (``cli/models.py``).

    Including the prompt is what stops the text a user confirms from describing
    something the deletion will not do. It used to carry the weakest of the
    three variants below and could DISAGREE with this gate,
    announcing a permanent delete for a sibling like ``<data dir>/models-old``
    that the gate then correctly declined to delete. That mismatch only ever
    erred toward OVER-warning and could not cause an unexpected deletion, but
    wrong text on a destructive confirmation teaches people to distrust the
    prompt, so the two now share this one predicate by construction.

    Three hand-rolled variants had drifted apart, and two of them were wrong in a
    way that reaches ``shutil.rmtree``:

    * ``path.is_relative_to(MODELS_DIR)`` is purely LEXICAL. Measured:
      ``<models>/../models-old/x.gguf`` tests True while resolving to a file
      OUTSIDE the models folder entirely.
    * ``str(path).startswith(str(MODELS_DIR))`` is a raw string prefix, so the
      sibling directory ``<data dir>/models-old`` matches ``<data dir>/models``.

    Both are answered by resolving first, which is what
    ``sync_models_dir._under_models_dir`` already did correctly. Using
    ``.parents`` rather than an ``is_relative_to`` on the resolved path also
    excludes the models root ITSELF, so a registry row pointing at the models
    folder can never become an rmtree of the whole library.

    *models_root* lets a caller hoist an already-resolved root out of a loop
    (sync_models_dir walks the whole registry); it must be a RESOLVED path.
    An unresolvable path is not one we are willing to delete under, so it is
    reported as not owned rather than raising at the call site."""
    root = models_root if models_root is not None else _mm.MODELS_DIR.resolve()
    try:
        return root in Path(path).resolve().parents
    except (OSError, ValueError):
        # ValueError, not just OSError: a stored path containing an embedded NUL
        # makes resolve() raise ValueError, and this gate is reached from the
        # CORRUPT-REGISTRY RECOVERY path (`localm rm <name>` on a hand-edited or
        # planted registry). An unhandled exception here would break the one
        # command that can clear the bad entry, which is the opposite of what a
        # recovery path is for. Either way an unresolvable path is not one we are
        # willing to delete under, so it is reported as not owned.
        return False


def remove_model(name: str) -> None:
    reg = _mm.load_registry()
    if name not in reg:
        console.print(f"[red]Not found:[/red] {name}")
        return

    epath = _entry_path(reg[name])
    if epath is None:
        # A malformed / corrupt entry (hand-edited registry, half-written state,
        # a shape from another version): there is no valid file to consider
        # deleting, so just drop the NAME. This is the CLI recovery path for a
        # registry that `localm list` / `add` would otherwise choke on - without
        # it, a single bad entry could only be cleared by hand-editing the JSON.
        _mm.update_registry(lambda r: r.pop(name, None))
        console.print(f"[green]✓[/green] Removed corrupt entry [bold]{name}[/bold]")
        return

    path = Path(epath)

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

    # Only delete files that live inside <data dir>/models/ - never touch
    # externally registered paths (Ollama blobs, user model dirs, etc.). This gate
    # stands immediately in front of shutil.rmtree / unlink, and it has to hold on
    # a path that was NOT written by this process, so it goes through the single
    # resolving definition rather than a local test (see is_owned_model_path for
    # the two hand-rolled variants it replaces and why both were wrong).
    #
    # Every operation below runs on the RESOLVED path, because that is the path
    # the ownership decision was made about. Deciding on one and deleting via the
    # other is an escape: a registry entry naming a symlink OUTSIDE the models
    # folder that points INTO it resolves as owned, and the split-GGUF branch then
    # unlinks `path.parent / part` - siblings of the LINK, outside the folder the
    # gate just authorised. Resolving once and using it throughout makes the
    # decision and the action refer to the same file by construction, rather than
    # by a reader noticing they agree.
    try:
        target = path.resolve()
    except (OSError, ValueError):
        target = path
    owned = is_owned_model_path(target)
    if owned and target.exists():
        if target.is_dir():
            import shutil
            shutil.rmtree(target)
            console.print(f"[dim]Deleted {target}[/dim]")
        else:
            # Split GGUF: remove every sibling part, not just the registered one
            siblings = split_gguf_parts(target.name) or [target.name]
            for part in siblings:
                part_path = target.parent / part
                if part_path.exists():
                    part_path.unlink()
                    console.print(f"[dim]Deleted {part_path}[/dim]")
    elif path.exists():
        console.print("[dim]Unregistered (file not deleted - lives outside <data dir>/models)[/dim]")
    _mm.update_registry(lambda r: r.pop(name, None))       # atomic RMW
    console.print(f"[green]✓[/green] Removed [bold]{name}[/bold]")




# The only legal shape of an Ollama blob filename, after ':' -> '-' normalisation.
# Used to confine a remote-authored manifest digest before it becomes a path
# component (see _resolve_ollama_manifest).
_OLLAMA_BLOB_RE = re.compile(r"sha256-[0-9a-f]{64}")


def _resolve_ollama_manifest(p: Path):
    """
    If p is an Ollama manifest directory, return (blob_path, suggested_name).
    Ollama layout: <root>/manifests/<registry>/<owner>/<model>/<tag>
                   <root>/blobs/sha256-<digest>
    Returns None if p doesn't look like an Ollama manifest.
    """
    import json as _json

    try:
        if not p.is_dir():
            return None
        tag_files = [f for f in p.iterdir() if f.is_file()]
    except OSError:
        # A pathological name must resolve to "not an Ollama manifest", not crash
        # the caller. On Windows, Path('....').is_dir() succeeds (trailing dots are
        # stripped when the path is stat'd) but iterdir() then does os.scandir on
        # the RAW '....' and raises FileNotFoundError - which used to escape as an
        # uncaught traceback out of get_model_info, so `localm run ....` mis-reported
        # a nonexistent-model typo as an internal localm crash + bug report.
        return None
    if not tag_files:
        return None

    for tag_file in tag_files:
        try:
            manifest = _json.loads(tag_file.read_text())
        except Exception:
            continue
        # A manifest is REMOTE-AUTHORED content that Ollama downloaded and wrote to
        # disk, so nothing about its shape may be assumed: `layers` can be absent,
        # a string (iterating one yields characters, and `"c".get` is an
        # AttributeError), or a list of non-dicts, and `digest` can be missing or a
        # non-string. Validate each step instead of indexing straight into it - a
        # malformed manifest must read as "not an Ollama manifest", never as a
        # traceback out of get_model_info.
        if not isinstance(manifest, dict):
            continue
        layers = manifest.get("layers")
        if layers is None:
            # The ORDINARY "this JSON file is not an Ollama manifest" case - any
            # directory can hold a config.json. Silent by design: warning here
            # would fire on every `localm add <some dir>`. Distinguished from the
            # ANOMALOUS shapes below, which mean the file IS manifest-shaped but
            # malformed, and which are worth surfacing (rule 5: do not collapse
            # "absent" and "corrupt" into one silent path).
            continue
        if not isinstance(layers, list):
            logger.debug("ollama manifest %s has a non-list 'layers' (%s); "
                         "treating as not-a-manifest", tag_file, type(layers).__name__)
            continue

        for layer in layers:
            if not isinstance(layer, dict):
                logger.debug("ollama manifest %s has a non-dict layer (%s); skipping",
                             tag_file, type(layer).__name__)
                continue
            if layer.get("mediaType") != "application/vnd.ollama.image.model":
                continue

            digest = layer.get("digest")
            if not isinstance(digest, str):
                logger.debug("ollama manifest %s has a model layer whose digest is "
                             "%s, not a string; skipping", tag_file, type(digest).__name__)
                continue
            blob_name = digest.replace(":", "-")  # sha256:abc -> sha256-abc

            # The digest becomes a PATH COMPONENT below, so it has to be confined
            # before the join, not after: a value like "../../../../etc/passwd"
            # walks straight out of <root>/blobs, and because the loop below
            # reports its find only when maybe.exists(), an unconfined name is
            # also a file-existence oracle - and any file it does hit is handed to
            # the GGUF parser as a model. An Ollama blob name has exactly one legal
            # shape, so require it rather than blocklisting bad ones.
            if not _OLLAMA_BLOB_RE.fullmatch(blob_name):
                # Rule 5: surface it. A silent `continue` here would report "not an
                # Ollama manifest" for a manifest that IS one but carries a
                # malformed or hostile digest, hiding the real reason.
                console.print(
                    f"[yellow]Ignoring an Ollama manifest layer with a malformed "
                    f"digest:[/yellow] {digest!r} (expected sha256:<64 hex>)")
                logger.debug("rejected non-conforming ollama digest %r in %s",
                             digest, tag_file)
                continue

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
    registration or a benign no-op (alias / interactively-skipped duplicate), False
    when the path is not a usable model OR when nothing was actually registered -
    a real name/content conflict declined interactively, or refused because there
    was no terminal to ask - so `localm pull <path>`/`localm add` can set their
    exit code accurately (NEW-STORE-MOVE-REGISTRY-DESYNC: this used to always
    return True past this point regardless of whether _register_with_dedup did
    anything, so a refused registration still reported success).

    *model_type* None (the default) means "detect it": the type is resolved
    deterministically from hard metadata (GGUF -> llm, HF config.json architectures)
    and falls back to the 'unknown' sentinel rather than a silent 'llm'. Pass an
    explicit MODEL_TYPES value to force it. A lone .safetensors file is not a model on
    its own: if it sits inside an HF model dir that directory is registered instead,
    otherwise it is rejected with a precise, actionable reason (A3).

    *store* is ``"copy"``, ``"move"``, or ``None`` (default: register in place,
    today's behavior). When set and the path is OUTSIDE <data dir>/models, the
    file/dir is brought into managed storage (via _store_into_models_dir) BEFORE
    registration, so the model is treated exactly like a pulled model afterward
    (BRING-IN-1). A failure here (name collision, no disk space, copy corrupted)
    is a hard failure of the whole call - unlike the softer dedup-prompt copy/move
    (where skipping just means "keep the old registration"), a requested --store
    that silently fell back to registering the ORIGINAL external path would be a
    hidden problem, not a benign no-op. A non-interactive name collision is now
    refused BEFORE the move/copy happens (there is no terminal to confirm an
    overwrite, so the outcome is already known); an interactive decline, or the
    rarer post-move content-dedup skip, can still leave the file sitting in
    MODELS_DIR unregistered under any requested name - reported to the caller
    rather than claimed as success, and recoverable via `localm list` / the next
    server start (sync_models_dir), just under an automatic name.
    """
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
        # raw registry key (GAP-CLI-1). Computed before any move so the
        # collision check below and the eventual registration agree on the
        # exact same name.
        model_name = _sanitize_name(name) if name else suggested
        if store and _mm.is_external_path(blob_path):
            # NEW-STORE-MOVE-REGISTRY-DESYNC: refuse before touching the
            # filesystem when we can already tell registration will be
            # refused - a name collision with no terminal to confirm an
            # overwrite. Moving first and discovering this after used to
            # silently displace the file while reporting success.
            if not sys.stdin.isatty():
                reg = _mm.load_registry()
                conflict = _name_collision(model_name, blob_path, reg)
                if conflict is not None:
                    console.print(
                        f"[red]'{model_name}' already points to a different "
                        f"file:[/red] {conflict}\nRefusing to move {blob_path} "
                        "into place non-interactively - pick another name "
                        "with -n."
                    )
                    return False
            try:
                blob_path = _mm._store_into_models_dir(blob_path, store)
            except RuntimeError as e:
                console.print(f"[red]{e}[/red]")
                return False
        # Ollama blob filenames already ARE the sha256 digest - store it free
        digest = blob_path.name.removeprefix("sha256-") \
            if blob_path.name.startswith("sha256-") else None
        # An Ollama blob is a GGUF text model, so an unspecified type is 'llm'.
        # The interactive path can still decline (or a rarer post-move content
        # dedup can) - report that honestly rather than claiming success for a
        # file that is now sitting in MODELS_DIR unregistered under any name
        # (sync_models_dir/`localm list` will pick it up under an auto name).
        registered = _mm._register_with_dedup(
            model_name, blob_path, "ollama",
            on_duplicate=on_duplicate, digest=digest,
            model_type=(model_type if model_type is not None else "llm"),
        )
        if not registered and store:
            verb = "moved" if store == "move" else "copied"
            console.print(
                f"[yellow]{blob_path} was {verb} into the models folder but "
                f"not registered as '{model_name}'.[/yellow] It will be picked "
                "up under an automatic name the next time models are scanned "
                "(`localm list`, or the next server start)."
            )
        return registered

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

    # Sanitize the user-supplied -n name (GAP-CLI-1): never let '../evil' or
    # 'a/b' become a raw registry key. p.stem is already path-component-safe.
    # Computed BEFORE any store/move (the move preserves the filename, so the
    # value is identical either way) so the collision check below and the
    # eventual registration agree on the exact same name.
    model_name = _sanitize_name(name) if name else (split_base or p.stem)
    kind = "hf" if is_hf else "local"

    # Bring an external file/dir into managed storage BEFORE registering, so the
    # registry ends up pointing at the copy/move destination, not the original.
    # Handles the split-GGUF parts and any sibling mmproj on its own
    # (_store_into_models_dir); works for is_hf's whole directory too.
    if store and _mm.is_external_path(p):
        # NEW-STORE-MOVE-REGISTRY-DESYNC: refuse before touching the filesystem
        # when we can already tell registration will be refused - a name
        # collision with no terminal to confirm an overwrite. This used to
        # move the file first and only discover the refusal after, silently
        # displacing it while _register_with_dedup's result went unchecked
        # and add_local reported success anyway. Reproduced and measured: the
        # file really does move, the registry is untouched, and it becomes
        # unregistered under ANY name until the next sync_models_dir scan
        # picks it up under an auto-generated name - never the one requested.
        if not sys.stdin.isatty():
            reg = _mm.load_registry()
            conflict = _name_collision(model_name, p, reg)
            if conflict is not None:
                console.print(
                    f"[red]'{model_name}' already points to a different "
                    f"file:[/red] {conflict}\nRefusing to {store} {p} into "
                    "place non-interactively - pick another name with -n."
                )
                return False
        try:
            p = _mm._store_into_models_dir(p, store)
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            return False

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
    # gguf_metadata (F8-PERSIST-ARCH-AND-EXPERT-COUNT) is captured regardless of
    # whether model_type needed detecting - an explicit --type override changes
    # what label the model gets, not whether its own header is worth reading.
    detected_type, gguf_metadata = _detect_local_model_type(
        p, is_gguf=is_gguf, is_hf=is_hf, is_blob=is_blob)
    if model_type is None:
        model_type = detected_type

    # The interactive path can still decline (or a rarer post-move content
    # dedup can) - report that honestly rather than claiming success for a
    # file that is now sitting in MODELS_DIR unregistered under any name
    # (sync_models_dir/`localm list` will pick it up under an auto name).
    registered = _mm._register_with_dedup(
        model_name, p, kind, on_duplicate=on_duplicate, digest=digest, size=size, model_type=model_type,
        architecture=gguf_metadata.get("architecture"), expert_count=gguf_metadata.get("expert_count"),
    )
    if not registered and store:
        verb = "moved" if store == "move" else "copied"
        console.print(
            f"[yellow]{p} was {verb} into the models folder but not "
            f"registered as '{model_name}'.[/yellow] It will be picked up "
            "under an automatic name the next time models are scanned "
            "(`localm list`, or the next server start)."
        )
    return registered




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

