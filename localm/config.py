# SPDX-License-Identifier: AGPL-3.0-or-later
import contextlib
import copy
import json
import os
import socket
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional


_warned_unconfigured_home = False

# config files already warned about (present but not a JSON object), so a
# non-dict config.json is surfaced once per process, not on every load_config
# call (see _merge_stored_config).
_warned_bad_config: set = set()


def _warn_unconfigured_home(path: Path) -> None:
    """Surface (once) that no data dir was configured, so a missing / lost config
    is VISIBLE instead of silently masked (do-not-hide-problems). stderr, not the
    logger: this runs at import time before logging is wired."""
    # setup.bat / setup.sh run setup-llama BEFORE the data-dir is chosen, so they
    # set LOCALM_SETUP=1 to suppress this one warning during that phase only (it
    # would otherwise fire spuriously mid-setup). This is the ONLY suppressor; if
    # LOCALM_SETUP lingers in a shell environment past setup it would mask a real
    # lost-config warning at runtime, so it must never be set outside the setup
    # scripts (do-not-hide-problems).
    if os.environ.get("LOCALM_SETUP") == "1":
        return
    global _warned_unconfigured_home
    if _warned_unconfigured_home:
        return
    _warned_unconfigured_home = True
    print("[localm] WARNING: no data directory is configured (no LOCALM_HOME, no "
          "localm-home.cfg, no ./home). Setup may not have run, or the config was "
          f"lost. Using a contained default inside this install: {path}. Run setup "
          "or set LOCALM_HOME to choose where data lives.", file=sys.stderr)


def _detect_home() -> Path:
    """
    Resolve the localm data directory (config, registry, models, logs,
    sessions, generated images/music).

    Priority - each install is self-contained and agnostic of others:
      1. LOCALM_HOME env var (explicit override / custom install location)
      2. Portable mode: a checkout carrying its own data - when this package
         sits in a source tree (pyproject.toml next to it) that contains a
         ``localm-home.cfg`` (one line: the data path) or a ``home``
         directory, that wins.  Created by setup.bat's "keep data in this
         folder" option.
      3. Contained fallback: ``./home`` INSIDE the install, with a stderr
         warning, when NOTHING is configured. Reaching here is a special case
         (setup never ran, or the config/marker was lost), not a normal default -
         so it is surfaced, and it NEVER silently falls back to a shared
         ~/.localm outside the install.
    """
    env = os.environ.get("LOCALM_HOME", "").strip()
    if env:
        return Path(env).expanduser()

    repo_root = Path(__file__).resolve().parents[1]
    if (repo_root / "pyproject.toml").is_file():       # source checkout only
        cfg_marker = repo_root / "localm-home.cfg"
        if cfg_marker.is_file():
            try:
                line = cfg_marker.read_text(encoding="utf-8").strip()
                if line:
                    return Path(line).expanduser()
            except OSError as e:
                # The marker EXISTS but could not be read. Do NOT silently fall
                # through to a different data dir - the user's configured data
                # would appear to vanish with no clue why. Surface it. Use
                # stderr, not the logger: this runs at import time before
                # logging is wired and importing debuglog here would be circular
                # (AUD-DETECTHOME).
                print(f"[localm] WARNING: cannot read {cfg_marker} ({e}); "
                      "falling back to the default data dir.", file=sys.stderr)
        portable = repo_root / "home"
        if portable.is_dir():
            return portable

    # No data dir configured: no LOCALM_HOME, no localm-home.cfg, no ./home. In a
    # real install this never happens (setup records a choice), so reaching here
    # means setup was never run OR the config/marker was lost - a special case, not
    # a normal default. Per rule 1 (never guess an absolute path outside the
    # install) and do-not-hide-problems: default to the instance's OWN root (a
    # contained ./home) and SAY SO on stderr, never a silent shared ~/.localm
    # outside the install.
    fallback = repo_root / "home"
    _warn_unconfigured_home(fallback)
    return fallback


def home_dir() -> Path:
    """Lazy variant of HOME_DIR - resolves the data dir at call time."""
    return _detect_home()


def cache_dir() -> Path:
    """Root for caches localm's OWN subprocesses write (rule 4: self-contained).

    Anything localm downloads on the way to installing something - pip's wheel/http
    cache while provisioning the managed ComfyUI venv, the Whisper STT model - belongs
    INSIDE the data dir, not in the user's home profile. Left to their defaults those
    tools cache to a per-user location (under ``%LOCALAPPDATA%`` on Windows,
    ``~/.cache`` on POSIX) that has nothing to do with localm: measured live, the
    managed-ComfyUI provisioning path alone had put ~11 GB of pip cache there, outside
    the data dir, without ever asking or telling the user.

    Derived from ``home_dir()`` (never a hardcoded path, rule 1), so the cache follows
    LOCALM_HOME: point the data dir somewhere and localm's caches go with it. This is
    deliberately NOT conditional on an ambient ``PIP_CACHE_DIR`` / ``HF_HUB_CACHE``:
    containment is a guarantee of where localm's own bytes land, and a guarantee that
    any unrelated environment variable can silently switch off is not one. LOCALM_HOME
    is the knob."""
    return home_dir() / "cache"


def pip_cache_dir() -> Path:
    """localm's OWN pip cache, inside the data dir (rule 4: self-contained).

    Shared by every pip subprocess localm drives itself - plugin-extra installs
    (plugins/deps.py), the native runtime wheel (setup_llama.py), and managed-ComfyUI
    provisioning (media/managed_comfy_provision.py delegates here) - so wheels are
    cached once, contained, and removed with the data dir. Left unset, pip caches to a
    per-user location OUTSIDE the data dir (``%LOCALAPPDATA%\\pip\\cache`` on Windows,
    ``~/.cache/pip`` on POSIX); measured live, provisioning alone had put multi-GB
    there. A cache (not ``--no-cache-dir``) because those wheels are multi-GB and a
    re-run would otherwise re-download all of it; the cost is disk INSIDE the data dir,
    where it is visible and reclaimed with it."""
    return cache_dir() / "pip"


def uv_cache_dir() -> Path:
    """localm's OWN uv cache, inside the data dir (rule 4: self-contained).

    uv keeps a SEPARATE cache from pip (its own ``UV_CACHE_DIR``, default
    ``%LOCALAPPDATA%\\uv\\cache`` on Windows / ``~/.cache/uv`` on POSIX). localm's
    installers try ``uv pip install`` BEFORE falling back to ``python -m pip``, so uv's
    cache has to be pinned too or the uv path silently leaks GBs outside the data dir
    exactly as pip would. Same location root as pip's (``cache_dir()``), a sibling
    subdir since the two tools' cache formats differ."""
    return cache_dir() / "uv"


def contained_pip_env(base: Optional[dict] = None) -> dict:
    """A subprocess environment with pip's AND uv's caches pinned inside the data dir.

    *base* defaults to a copy of the current process environment. localm's package
    installers (plugins/deps.py, setup_llama.py) shell out to ``uv pip install`` first
    and ``python -m pip install`` second; BOTH tools cache to a per-user location
    outside the data dir when left to their defaults, so BOTH ``PIP_CACHE_DIR`` and
    ``UV_CACHE_DIR`` are set here - pinning only one still leaks via the other. This
    OVERRIDES any ambient value on purpose (see ``cache_dir()``): containment a stray
    environment variable can silently switch off is not a guarantee. Callers pass the
    returned dict as ``subprocess``'s ``env=``; a subprocess that runs neither tool
    simply ignores the two extra vars, so it is safe to use everywhere."""
    env = dict(os.environ if base is None else base)
    env["PIP_CACHE_DIR"] = str(pip_cache_dir())
    env["UV_CACHE_DIR"] = str(uv_cache_dir())
    return env


HOME_DIR = _detect_home()
MODELS_DIR = HOME_DIR / "models"
REGISTRY_FILE = HOME_DIR / "registry.json"
CONFIG_FILE = HOME_DIR / "config.json"


# Only the keys the user actually changed are persisted to config.json; a key
# absent on disk follows this table at load time, so editing a default value
# here reaches every existing install (see save_config / _user_delta).
DEFAULT_CONFIG: dict = {
    "binary_dir": None,    # None = auto-detect (runtime wheel, then legacy dirs)
    "n_ctx": 4096,         # initial context window (grows on demand)
    "n_ctx_max": 16384,    # ceiling the window may grow to (0 = unlimited)
    "n_ctx_grow": 4096,    # growth step - window expands in multiples of this
    # Size ctx ceiling from free VRAM at load (clamped 4k-64k); window still
    # starts at n_ctx and grows on demand. False = use fixed n_ctx_max.
    "ctx_auto": True,
    "n_gpu_layers": 99,    # 99 = offload everything to GPU
    # When n_gpu_layers is left at its "everything" default, auto-size how many
    # layers actually go on the GPU from free VRAM at load: a model too big for
    # full GPU offload runs some layers on CPU (slower) and LOADS, instead of
    # being refused. An explicit n_gpu_layers (any value other than 99, e.g.
    # `-g 24`) is always honoured verbatim. Off => request full offload as-is.
    "n_gpu_layers_auto": True,
    # Ceiling (seconds) for a GGUF model load in its isolated worker process
    # (see llamacpp/_runner.py) before it is treated as hung and killed. A
    # stalled load has no safe "unmeasurable" fallback (unlike a VRAM probe),
    # so raise this only if a genuinely huge model on slow storage needs
    # longer than the generous default.
    "gguf_load_timeout_s": 900.0,
    # How long a reply may take to produce its FIRST token before the worker is
    # treated as hung. This covers prompt PREFILL, not one token's decode, so it
    # is sized like the load timeout rather than the per-token ceiling: on CPU,
    # under heavy partial offload, or with a very long prompt, prefill can
    # legitimately take minutes. Raise this only if a genuinely slow box needs
    # longer than the generous default.
    "gguf_first_token_timeout_s": 900.0,
    # GPU device to load onto / read VRAM from on a multi-GPU box. None = no
    # explicit selection (device 0). A stale index falls back to device 0 with
    # a logged warning, not a wrong/out-of-range GPU (see
    # discover.resolve_main_gpu_index).
    "main_gpu_index": None,
    # Split a model too big for one card across 2+ GPUs (GGUF: llama.cpp
    # layer-split; HF: accelerate device_map restricted to these devices).
    # None/empty/1 entry = off (today's single-GPU behavior via
    # main_gpu_index, unchanged). A device no longer detected at load time is
    # dropped with a logged warning, not trusted blindly (see
    # discover.resolve_gpu_split).
    "gpu_split_indices": None,
    # Optional relative weight per entry in gpu_split_indices (same length,
    # any positive numbers - llama.cpp treats them as proportions, not values
    # that must sum to 1). None, or a length mismatch, means an equal split.
    "gpu_split_ratios": None,
    # Default system prompt for chat. The GUI's per-chat System prompt field
    # OVERRIDES this when set; a blank field inherits this. Empty by default.
    "chat_system_prompt": "",
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.1,
    # Generation budget per reply. Thinking models (qwen3, deepseek-r1) spend
    # most of it reasoning, so 1024 cut answers mid-thought; this is a runaway
    # guard, not a cost control.
    "max_tokens": 4096,
    "confirm_remove": True,   # ask before localm rm deletes files
    # Sidebar wordmark, shared by web GUI and desktop launcher (picker writes,
    # launcher reads). One of: local-m (LocaL white + M blue, default),
    # loca-lm (Loca white + LM blue), localm (lowercase, m blue). Console
    # command, app icon, and shortcut are fixed regardless.
    "logo_style": "local-m",
    "import_max_depth": 3,    # `localm add <dir>` recurses up to this many levels
    "port": 8642,             # default inference server port (auto-bumps if busy; an explicit --port does not)
    "cors_origins": None,     # None = localhost only; list of origins; or "*"
    # Require a configured API key on protected endpoints: true refuses requests
    # until a key is set (see localm/auth.py); env override LOCALM_REQUIRE_AUTH.
    # False (default) = open in local/dev mode on loopback.
    "require_auth": False,
    # Quick-select scope bundles for the "Keys & devices" manager (Settings),
    # each {name, scopes}, offered as one-tap presets when minting a key.
    # Re-seeded only when ABSENT (an emptied list stays empty). Privileged
    # scopes (coder:full / admin) apply only when an OWNER mints the key.
    "key_presets": [
        {"name": "Minimal", "scopes": ["chat"]},
        {"name": "Companion", "scopes": ["chat", "image", "music", "video",
                                         "voice", "rag", "web", "models:read"]},
        {"name": "Full", "scopes": ["chat", "coder", "image", "music", "video",
                                    "rag", "web", "voice", "mcp",
                                    "models:read", "models:write", "config:read"]},
        {"name": "Admin", "scopes": ["admin"]},
    ],
    # Command that starts your ComfyUI install (e.g. a launch .bat). When set,
    # image/music/video generators auto-start ComfyUI if not running (GUI, CLI,
    # or the coder's generate_image tool).
    "comfy_launch_cmd": None,
    # Working directory for comfy_launch_cmd (launchers assuming their own folder,
    # e.g. "python main.py" in a ComfyUI checkout). Blank + a launcher-file cmd =
    # run from that file's folder (the ComfyUI / ZLUDA .bat convention).
    "comfy_workdir": None,
    # Seconds to wait for ComfyUI to answer after launch. A ZLUDA / ROCm cold
    # start compiles GPU kernels and can take minutes, so the default is generous.
    "comfy_launch_timeout": 300,
    # ComfyUI's own output directory (e.g. StabilityMatrix's Images folder).
    # Only needed with comfy_delete_outputs, to find and delete ComfyUI's
    # duplicate copy. Blank = derived from the ComfyUI folder on demand.
    "comfy_output_dir": None,
    # Delete ComfyUI's OWN copy (and /history entry) after localm saves its own.
    # False (default) KEEPs them (a user may want ComfyUI's gallery). Privacy
    # mode forces deletion (no traces). Per-plugin config can override.
    "comfy_delete_outputs": False,
    # ComfyUI base URL localm talks to. None/blank uses the FLUX_API_URL env
    # override when set, else http://127.0.0.1:8188 (the ComfyUI default).
    "comfy_api_url": None,
    # Rewrite a slow `dequant_dtype: "float32"` in a Flux GGUF UNet loader to the
    # fast default (fp16/bf16) on submit: float32 doubles unpacked model size and
    # forces CPU offload on a VRAM-limited card (~36 s/it vs ~6-7 s/it). True
    # (default) auto-corrects; False submits your dequant choice verbatim.
    "comfy_fast_dequant": True,
    # Suppress ComfyUI opening its own web page when localm auto-launches it.
    # False (default) keeps ComfyUI's tab. True appends --disable-auto-launch so
    # it starts headless; stock run_*.bat / comfyui.* / bare "python main.py"
    # forward the flag, a launcher that drops extra args just ignores it
    # (non-breaking). Applies to image/music/video (shared ensure_comfy()).
    "comfy_disable_auto_launch": False,
    # MEDIA-1: reactive, opt-in in-memory shim for the upstream ComfyUI __func__
    # regression (Comfy-Org/ComfyUI #12116). Off by default (touches nothing).
    # When on, a ComfyUI localm SPAWNS gets a localm-owned shim dir on its child
    # PYTHONPATH to patch the regression in memory; localm never writes into the
    # user's install nor shims a ComfyUI it did not start. Set by the reactive
    # offer or `localm config comfy_func_shim on`; self-expires once Comfy is fixed.
    "comfy_func_shim": False,
    # Which ComfyUI localm targets (coexistence). "own" (default) = use a
    # localm-managed instance once one is installed under <LOCALM_HOME>/comfyui
    # (`localm comfy setup`); inert until then, so a fresh install behaves
    # identically to today. "user" = always use the user's own ComfyUI
    # (comfy_workdir / comfy_api_url), even if a managed instance exists. With
    # no managed instance, both behave identically. This key only routes
    # (provisioning is stages S2/S3) and never modifies the user's own ComfyUI.
    # See localm/media/managed_comfy.py.
    "comfy_target": "own",
    # EXPERIMENTAL, default OFF: per-component GPU placement for media generation.
    # When on, and the running ComfyUI offers the multigpu Select*Device nodes
    # (upstream 2026-05-25 or newer) AND a 2+ card split is configured, localm
    # injects those nodes so the text encoder + VAE load on a second card while
    # the heavy diffusion model stays on the preferred one. OFF by default even
    # on a multi-GPU box: the second-card behaviour is UNPROVEN on real multi-GPU
    # hardware (the dev box has one card), and ComfyUI's gpu:N is a POSITION in a
    # reordered visible list, so an off-by-one would land a component on the wrong
    # card and STILL RENDER (a silent wrong result, not a crash). Users opt in
    # knowingly; the default flips to on only after a 2-GPU hardware proof lands,
    # as a separate deliberate change. On one card, or an older ComfyUI, or with
    # the toggle off, media generation is byte-identical to today (single card).
    "comfy_gpu_placement": False,
    # Session persistence mode for ALL surfaces (chat, server, GUI, coder):
    #   privacy = no traces written automatically (default)
    #   log     = JSONL audit trail in ~/.localm/sessions/
    #   full    = log + markdown transcript
    # chat_mode / coder_mode override per surface (None = inherit); CLI --mode
    # overrides everything.
    "mode": "privacy",
    "chat_mode": None,
    "coder_mode": None,
    # Long-term chat memory: recall durable facts/preferences and inject them
    # (server-side) into the system prompt each turn. Recall is free (BM25 over a
    # small store). False stops injecting (existing memories kept, just unused).
    "memory_enabled": True,
    # Grow memory automatically: after a chat turn (log/full mode, never privacy)
    # distil durable facts into the store in the background, debounced to once per
    # MEMORY_AUTO_MIN_INTERVAL (see the chat plugin). Skipped in privacy (no new
    # traces). False requires the manual "Synthesize now" button / jobs "memory"
    # task / POST /api/memory/consolidate.
    "memory_auto_consolidate": True,
    # Privacy mode normally disables memory ENTIRELY (no recall + no writes). On =
    # allow READING existing memories into the prompt in privacy mode (writing
    # stays off - privacy never creates a trace). Off by default. Per-surface:
    "memory_recall_in_privacy": False,
    "memory_recall_in_privacy_chat": True,      # applies only when the master is on
    "memory_recall_in_privacy_coder": True,     # applies only when the master is on
    # Keep diagnostics for bug reports even in privacy mode. Off by default:
    # privacy mode writes NO automatic trace (the hang watchdog trace, the
    # crash-restart breadcrumbs, and the debug log are all suppressed). On =
    # keep those diagnostics regardless of mode, so an intermittent freeze/crash
    # leaves something to attach to a report. Never chat content - code stacks
    # and operational logs only.
    "keep_diagnostics": False,
    # On-device embedding model for semantic search (RAG hybrid retrieval + agent
    # memory): a small dedicated GGUF, loaded separately from the chat model.
    # Value = a known key (bge-small-en-v1.5, nomic-embed-text-v1.5), a registered
    # model name, or a GGUF path. A known model is fetched into
    # <home>/models/embeddings on first use (auto only under net_mode=allow, else
    # run 'localm setup-embeddings'). Until present, memory/RAG fall back to BM25.
    "embedding_model": "bge-small-en-v1.5",
    # How the embedding model's token states are pooled into one vector.
    # "mean" (default) suits the bundled bge/nomic choices and is what every
    # existing index was built with. A DECODER-based embedder (Qwen3-Embedding,
    # gte-Qwen2) is trained for last-token pooling and is degraded by mean: set
    # "last", or "auto" to follow whatever the GGUF declares. Changing this
    # invalidates already-embedded RAG collections and memory vectors (same
    # dimensions, different meaning), so re-index after changing it.
    "embedding_pooling": "mean",
    # Which host folders the document-indexing (RAG) API may READ. All three keys
    # are OWNER-ONLY: a non-owner config:write key can neither see nor set them
    # (enforced at PATCH /v1/config; see settings_schema.admin_only). The localm
    # data dir and credential folders (.ssh, .aws, ...) stay denied in EVERY mode
    # (hard floor; rag/store.py confine_index_path). Read by indexing_policy().
    #   whitelist (default) = only home, the working dir, and rag_allowed_roots.
    #   blacklist           = anywhere EXCEPT rag_denied_roots.
    "rag_indexing_mode": "whitelist",
    "rag_allowed_roots": [],   # extra folders allowed in whitelist mode
    "rag_denied_roots": [],    # folders refused in blacklist mode
    # A document's format label (json/yaml/python/...) is derived heuristic-FIRST:
    # known extension wins, else a structural sniff (rag/extract.classify_format).
    # This toggle only governs the LLM TIE-BREAK: when both are inconclusive AND a
    # chat model is loaded, prompt-guess from a snippet (cached per extension for
    # the process). Off -> labeled "text"; never fired during embedding-only index.
    "rag_classify_unknown_files": True,
    # Seconds a GUI coder approval card may sit unanswered before it is
    # auto-rejected and the agent moves on.
    "coder_confirm_timeout": 600,
    # Wall-clock cap on the coder's startup project-map scan (CODER-1). <= 0
    # disables the deadline (scan to completion however long it takes).
    "coder_index_timeout": 20,
    # Episodic memory: the coder recalls lessons from past sessions on a project
    # (BM25, free) and at session close distils the session into a new lesson (one
    # model call per session that changed files). Writes skipped in privacy mode
    # and for restricted (shareable-key) sessions, stored under the home dir not
    # the project tree. False disables both halves.
    "coder_episodic_memory": True,
    # Provenance tagging: re-frame coder tool results from untrusted (network /
    # MCP) tools as data-not-instructions, so a fetched page or external server
    # cannot inject into the model loop (indirect prompt injection). Defense in
    # depth (blocks nothing, only labels). Leave ON absent a specific reason.
    "coder_untrusted_provenance": True,
    # Pre-done self-review: before the coder declares done, a reviewer model reads
    # the diff and feeds blocking issues back for one more fix pass. Off by default
    # (adds a model round-trip per task that changed files).
    "coder_review": False,
    # Reviewer target: "" = the agent's own model (local); "local" = a different
    # small model on CPU in the coder's process (set coder_reviewer_model; adds CPU
    # latency); "openai"/"anthropic" = cloud; an http(s) URL = a 2nd
    # OpenAI-compatible endpoint. A NETWORK reviewer (cloud / non-loopback URL) is
    # skipped in privacy mode and for shared keys (would send the diff off-machine)
    # and falls back to the local model; the "local" CPU reviewer stays on-machine.
    "coder_reviewer": "",
    # Model name (or path) for a heterogeneous reviewer ("local"/cloud/URL); blank
    # uses a sensible provider default or the agent's own model name.
    "coder_reviewer_model": "",
    # Constrain coder tool calls with a LAZY GBNF grammar: thinking/prose flow
    # free, but a started <tool_call> is forced to valid JSON (no malformed calls
    # to repair). ON by default for grammar-capable local backends since 2026-07-02
    # (REC-CODER-GRAMMAR; the old "runtime sampler faults" blocker was our own
    # double-accept bug). External API / grammar-less builds unaffected
    # (supports_grammar gate + runtime soft-degrade). NOTE: a config.json written
    # before the flip keeps the old dumped False (saved wins) - flip it in Settings.
    "coder_tool_grammar": True,
    # After an image is generated, ask ComfyUI to release VRAM and reload the chat
    # model so the next reply is instant. Off = the chat model reloads lazily on
    # the next message instead (better for many images in a row).
    "reload_llm_after_imagine": True,
    # VRAM-aware media model swap: before an image/music/video gen the chat LLM is
    # unloaded so the media model gets the GPU (on a big card both fit).
    #   auto   = keep chat loaded when the media model fits alongside it (free VRAM
    #            >= estimate + headroom), else swap (default)
    #   always = always unload the chat model (historical behaviour)
    #   never  = never unload; keep chat hot (media may OOM on a small card)
    # reload_llm_after_imagine is a SEPARATE axis (eager-vs-lazy reload AFTER a
    # gen, not this unload-before decision).
    "model_swap_policy": "auto",
    # Free the loaded model from VRAM after this many idle seconds, so a running
    # server stops holding the GPU; the next request reloads it lazily. 0 =
    # disabled (default): resident until an explicit unload or swap. Measured from
    # the last request, like Ollama's keep_alive.
    "idle_unload_seconds": 0,
    # Network policy for model-initiated requests (coder fetch_url/web_search,
    # chat web access). See localm/netpolicy.py and docs/network.md.
    #   off   = all policy-routed network access fails fast
    #   ask   = allowed; the coder asks for approval per request (default)
    #   allow = no confirmation
    "net_mode": "ask",
    "net_allow": [],            # domains; empty = any. "x.com" covers *.x.com
    "net_deny": [],             # domains always refused (wins over allow)
    "net_allow_private": False, # True = permit loopback/private targets (SSRF guard off)
    "net_search_url": None,     # SearXNG base URL; None = DuckDuckGo (no key)
    # Reach localm by NAME, not just IP, on a network bind (see localm/netname.py
    # and docs/naming.md). mDNS/Bonjour advertises "<mdns_name>.local" so a phone
    # opens https://localm.local:PORT with no IP; the name is folded into the TLS
    # cert and the Tailscale MagicDNS name is detected + certified automatically.
    # Loopback binds never advertise.
    "mdns_name": "localm",      # the .local name; sanitized to a DNS label on use
    "mdns_enabled": True,       # advertise the name over mDNS on network binds
    # Speech-to-text (GUI mic button; needs the [voice] extra).
    # Model sizes: tiny / base / small / medium - bigger = better + slower.
    "voice_stt_model": "base",
    "voice_stt_language": None,  # None = auto-detect; or "en", "de", ...
    # When a registered model's file has gone missing, False (default) flags the
    # entry as "missing" (kept, shown in the list); True deletes the entry.
    # Only files under the models folder are ever auto-deleted.
    "autoprune_missing_models": False,
    # When a command belongs to a known first-party plugin that is not
    # installed/enabled (e.g. /generate-image with image off), suggest installing
    # it instead of "unknown command". False silences the hint; a truly unknown
    # command always errors.
    "suggest_plugins": True,
    # When a plugin declaring pip extras (requires_extras) is installed/enabled by
    # the local operator, auto-install those extras on the HOST. A remote client
    # never triggers a server-side pip (only CLI or a loopback GUI request does).
    # `localm plugin setup` records the choice here. Default True (read via
    # .get(..., True) for configs saved before this key existed).
    "auto_install_plugin_deps": True,
    # Bug reports, the read-only Issues view, and the self-updater all talk to ONE
    # small Cloudflare Worker (the localm proxy; see tools/bugreport-proxy/) that
    # holds the GitHub tokens SERVER-SIDE. Shipped as DEFAULTS so a fresh download
    # works with ZERO setup (update_url/token below fall back to these). No GitHub
    # token is in the app - only the public Worker URL and a low-value client
    # token that is intentionally PUBLIC (like a Sentry DSN), NOT a secret: it only
    # gates against drive-by spam (Cloudflare rate limiting is the real control),
    # can ONLY file an issue (never read the repo), and is rotatable at the Worker.
    # Set either to "" (or null) to opt a build out of the hosted channel.
    "bugreport_upload_url": "https://localm-bugreport-proxy.localm.workers.dev",
    "bugreport_upload_token": "3x_HA2UXbwNDnNfdDmpFBvvfcl2S-I-9t7XLQRAShM4",
    # Update channel + read-only issues tracker. One Worker hosts report + issues +
    # update, so these default to the proxy above; set update_url/token ONLY to
    # point updates at a different Worker (needs a Contents:read token, separate
    # from Issues, plus the shared secret). None = no update channel (banner +
    # `localm update` hidden). See tools/bugreport-proxy/ and dev-notes/self-updater-design.
    "update_url": None,
    "update_token": None,
    # Names of enabled engine plugins (WordPress-style). Managed by the plugin
    # engine (plugin enable/disable, GUI Plugins page) via update_config, NOT the
    # settings form. Declared here for a documented home + default (else the
    # settings-save endpoint rejects it as unknown). A plugin is active only when
    # installed on disk AND in this list; see docs/plugins.md.
    "plugins_enabled": [],
    # Per-plugin config namespace (e.g. plugins["image"]["comfy"]["output_dir"]).
    # Written by the plugin engine and media backends via update_config, NOT the
    # flat settings form. Declared here so settings-save accepts it and the
    # per-plugin media-containment knob survives a full-config round-trip.
    "plugins": {},
}

# localm claims 8642-8741 - far from ComfyUI (8188), A1111 (7860),
# Ollama (11434), and the 8000/8080/8888 dev-server crowd.
PORT_RANGE = (8642, 8741)


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """True when something is already listening on host:port."""
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) == 0


class PortInUseError(RuntimeError):
    """An explicitly requested port is already in use.

    An explicit ``--port`` is honored exactly or not at all: if it is busy,
    ``pick_port`` raises this instead of quietly binding a different port. The old
    fallback scanned from ``PORT_RANGE[0]``, so a deliberately chosen high port
    that was busy landed back on the shared default 8642 - the opposite of what a
    user who picked a specific port to avoid a collision asked for. The default (no
    ``--port``) still auto-bumps through the range; only an explicit request refuses.
    """

    def __init__(self, port: int):
        self.port = port
        super().__init__(f"Port {port} is already in use.")


def pick_port(requested: Optional[int] = None, host: str = "127.0.0.1"):
    """
    Resolve the port to serve on. Returns ``(port, default_port_was_busy)``.

    An explicit ``requested`` port is honored exactly: returned if free, otherwise
    :class:`PortInUseError` is raised. It is never silently relocated to a
    different port (see that class for why).

    With no explicit port, uses the configured default (``config['port']``, 8642)
    and, if that is busy, walks localm's range (8642-8741) for the next free port,
    falling back to an OS-assigned port. The returned flag is True only in that
    auto-bump case; an explicit request never returns True (it raises instead).
    """
    if requested is not None:
        if port_in_use(requested, host):
            raise PortInUseError(requested)
        return requested, False
    start = load_config().get("port", PORT_RANGE[0])
    if not port_in_use(start, host):
        return start, False
    for candidate in range(PORT_RANGE[0], PORT_RANGE[1] + 1):
        if candidate != start and not port_in_use(candidate, host):
            return candidate, True
    return get_free_port(), True


def _mkdir_or_explain(path: Path, *, is_home: bool) -> None:
    """``path.mkdir(parents=True, exist_ok=True)`` with one user-error case turned
    into a clean message instead of a crash.

    ``parents=True`` so an explicit ``LOCALM_HOME`` a level or two below a
    not-yet-existing folder (a fresh ``D:\\localm\\data``) is created like
    ``mkdir -p`` rather than crashing with WinError 3 - LOCALM_HOME is an explicit
    "keep my data here" choice, so localm creates the whole path (AUD-ENSUREDIRS).

    ``exist_ok=True`` already swallows "exists AND is a directory", so a
    ``FileExistsError`` from mkdir means exactly "exists but is NOT a directory"
    (a regular file / symlink; WinError 183 on Windows, EEXIST on POSIX). That is
    user misconfiguration - typically ``LOCALM_HOME`` set to a file - not a localm
    bug, so we surface it as a ``click.ClickException``: the CLI's cross-cutting
    handler passes those straight through (a clean "Error: ..." line, exit 1),
    never routing them to the generic "unexpected error" + bug-report path. This
    SURFACES the real problem with an actionable fix (do-not-hide-problems); it
    does not swallow it. Other OSErrors (permission denied) are left to propagate
    unchanged - they are not this case."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        import click  # lazy: keep the CLI framework out of config's import graph
        env = os.environ.get("LOCALM_HOME", "").strip()
        if is_home and env and Path(env).expanduser() == path:
            msg = (f"LOCALM_HOME points at a file, not a directory: {path}. "
                   "Set it to a directory (or remove/rename that file).")
        else:
            msg = (f"localm's data path is a file, not a directory: {path}. "
                   "localm needs it to be a directory; remove or rename that "
                   "file, or point LOCALM_HOME at a directory.")
        raise click.ClickException(msg) from None


def ensure_dirs() -> None:
    _mkdir_or_explain(HOME_DIR, is_home=True)
    _mkdir_or_explain(MODELS_DIR, is_home=False)


# Registry and config are mutated from several places at once - the GUI server
# threads, the `localm pull` subprocess the GUI spawns, and sync_models_dir on
# every launch. A plain open("w")+json.dump truncates the file before writing,
# so a crash, a job cancel (SIGTERM), or simple interleaving could leave a
# half-written file that the next unguarded json.load() would choke on, hiding
# every registered model app-wide. The helpers below make every write atomic
# (write a temp file in the same dir, fsync, then os.replace - readers see only
# the old or the new complete file, never a torn one) and make every read
# crash-proof (fall back to the .bak snapshot, then to the default).
_io_lock = threading.RLock()

# A concurrent open handle makes a Windows os.replace / open raise a TRANSIENT
# PermissionError (WinError 5); a bounded retry rides it out. The lock is usually
# microseconds, but a loaded box (antivirus scanning the file, an indexer, a slow
# second process) can hold it tens of ms, so the backoff escalates and the total
# budget is ~1 s before a PERSISTENT failure is re-raised / falls back
# (do-not-hide-problems). See _replace_atomic / _read_json (AUD-WINREPLACE).
_REPLACE_RETRIES = 16
_REPLACE_BACKOFF = 0.01      # seconds; escalates up to the cap
_REPLACE_BACKOFF_CAP = 0.1   # seconds


def _transient_backoff(attempt: int) -> None:
    """Sleep before the next retry, escalating linearly to a cap so a lock that
    lingers on a busy machine is ridden out without a hot spin."""
    time.sleep(min(_REPLACE_BACKOFF * (attempt + 1), _REPLACE_BACKOFF_CAP))


def _is_transient_permission_error(e: OSError) -> bool:
    """True when a PermissionError plausibly reflects a TRANSIENT file lock worth
    riding out, rather than a stable denial no retry can change.

    Windows is the only platform with the race this retry exists for: another
    handle (a concurrent atomic replace, antivirus, the indexer, Windows Search)
    holds the file open for microseconds and open()/os.replace raises
    PermissionError - ERROR_SHARING_VIOLATION (32), or ERROR_ACCESS_DENIED (5)
    when a replace collides with an AV lock. A genuine Windows ACL denial also
    surfaces as winerror 5 and is indistinguishable from the AV case at this
    layer, so Windows keeps retrying it: ~1s to be sure is the documented,
    accepted trade there.

    POSIX has no such race. open() raises EACCES/EPERM only when the mode or ACL
    genuinely denies us (a mode-000 or root-owned file left by an earlier sudo
    run), which is a stable state - the retry can never succeed and just burns
    ~1s of time.sleep while holding _io_lock, which serializes config access
    process-wide, including the per-request auth path (auth.require_auth ->
    load_config). So on POSIX we fall back at once (the failure is still
    surfaced by the caller, never hidden - it just is not retried)."""
    if os.name != "nt":
        return False
    # winerror is absent only if this was not raised by the Windows layer at all
    # (e.g. a test double); treat that as transient to preserve the established
    # Windows behaviour rather than silently narrowing it.
    return getattr(e, "winerror", None) in (5, 32, None)


def _replace_atomic(src: Path, dst: Path) -> None:
    """``os.replace(src, dst)`` with a bounded retry on a transient Windows
    sharing violation.

    os.replace IS atomic, but on Windows it raises PermissionError (WinError 5)
    when another handle has *dst* open at that instant: a second localm process
    reading the file, an antivirus / indexer / backup scanner, Windows Search.
    That window is short, so retrying briefly rides it out instead of crashing
    the save. A genuine, persistent permission problem still surfaces (re-raised
    after the last attempt) - we retry the transient race, we never hide a real
    failure. On POSIX os.replace does not hit this, so the loop succeeds first
    try; if it does raise there it is a STABLE denial (see
    _is_transient_permission_error), re-raised at once rather than stalling ~1s
    under _io_lock AND the cross-process lock for a retry that cannot succeed
    (REG-566)."""
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError as e:
            if attempt == _REPLACE_RETRIES - 1 or not _is_transient_permission_error(e):
                raise
            _transient_backoff(attempt)


def _atomic_write_json(path: Path, data) -> None:
    """Write *data* as JSON to *path* atomically (temp file + os.replace).

    Keeps a one-step .bak of the previous good file so a corrupt read can
    recover. os.replace is atomic on Windows and POSIX when src/dst share a
    filesystem, which they do (same directory); _replace_atomic additionally
    rides out the transient Windows sharing violation a concurrent reader causes.

    The temp file gets a UNIQUE per-write name (mkstemp), NOT a fixed ``<name>.tmp``.
    Two localm processes writing the same file concurrently (a CLI ``pull``/``config``
    alongside the running GUI, or two CLI invocations) would otherwise both open the
    SAME ``<name>.tmp`` and collide - one save then crashes when the other's replace
    has already consumed the shared temp (Windows WinError 2/32), or on POSIX their
    writes interleave into a torn temp. ``_io_lock`` only serialises writers WITHIN
    one process; _replace_atomic's retry rides out a locked DESTINATION but not two
    writers sharing one temp SOURCE. A unique temp per writer removes the collision
    at the source, so cross-process contention degrades to the documented
    last-writer-wins on the final file, with no crash or torn write."""
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent),
                                    prefix=path.name + ".", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            try:
                _replace_atomic(path, path.with_name(path.name + ".bak"))
            except OSError as e:
                # The .bak snapshot is best-effort: a failed one must NOT fail the
                # primary write (which still proceeds below). _replace_atomic already
                # rode out the transient sharing violation, so reaching here means a
                # PERSISTENT problem (a lock that outlasted the retries, disk full, a
                # real permission error) - note it so it is discoverable rather than
                # totally silent (do-not-hide-problems), without escalating a
                # best-effort path into a hard fail.
                print(f"[localm] note: could not refresh {path.name}.bak ({e}); "
                      "the main write still succeeded.", file=sys.stderr)
        _replace_atomic(tmp, path)
    except BaseException:
        # _replace_atomic consumes tmp on success; on any failure BEFORE that, remove
        # our unique temp so a failed write never leaves an orphan behind (the old
        # fixed-name temp was reused by the next write; a unique one would pile up).
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _read_json(path: Path, default):
    """Read JSON from *path*, falling back to its .bak then *default* on any
    corruption - a damaged file must never take the whole app down."""
    for candidate in (path, path.with_name(path.name + ".bak")):
        if not candidate.is_file():
            continue
        for attempt in range(_REPLACE_RETRIES):
            try:
                with open(candidate, encoding="utf-8") as f:
                    return json.load(f)
            except PermissionError as e:
                # TRANSIENT on Windows: a concurrent atomic replace (another
                # process, or antivirus/indexer) has the file locked for a
                # microsecond. Retry the SAME file before giving up, so a passing
                # scanner does not make us spuriously fall back to .bak/defaults
                # and momentarily discard live settings. A persistent EACCES
                # surfaces after the retries via the same warning + fall-through.
                # A STABLE denial (any POSIX EACCES - see
                # _is_transient_permission_error) skips the retry entirely: it
                # can never succeed, and this loop runs under _io_lock, so ~1s of
                # backoff would stall every config read process-wide - including
                # the per-request auth path - for nothing (REG-566).
                if attempt < _REPLACE_RETRIES - 1 and _is_transient_permission_error(e):
                    _transient_backoff(attempt)
                    continue
                print(f"[localm] {candidate.name} is unreadable ({e}); "
                      "falling back.", file=sys.stderr)
            except (ValueError, OSError, RecursionError) as e:
                # NOT transient (corrupt/non-UTF-8 JSON -> ValueError incl.
                # JSONDecodeError/UnicodeDecodeError, a huge integer -> ValueError,
                # deep nesting -> RecursionError, or a hard OS error): fall back
                # immediately without wasting the retry budget. Previously these
                # ESCAPED and crashed the app instead of honouring the documented
                # "fall back to .bak then default, never take the app down"
                # guarantee (AUD-CFGFALLBACK).
                print(f"[localm] {candidate.name} is unreadable ({e}); "
                      "falling back.", file=sys.stderr)
            break
    return default() if callable(default) else default


def instance_id() -> str:
    """Stable, unguessable identifier for THIS install's data directory. Minted
    once (uuid4 hex) on first call and reused for the life of the install by
    persisting it to a small file under the data dir; never regenerated on a
    normal restart. References the bare ``HOME_DIR`` global (like
    ``ensure_dirs``), not a path frozen at import, so it always tracks whichever
    data directory is actually configured (and so a test can point HOME_DIR
    elsewhere via monkeypatch and this follows immediately).

    Why this exists (AUD-INSTANCEID): browser localStorage is scoped by ORIGIN
    (protocol+host+port) only - never by which backend DATA DIRECTORY is
    actually running behind it. localm's server binds to a fixed default port
    that only changes when it is already busy, so a fresh install opened after a
    prior instance closed typically reuses the SAME origin, and therefore the
    SAME localStorage bucket, as a totally unrelated data directory. The GUI
    fetches this id from /v1/config and compares it against the id it last saw
    for this browser origin, so it can tell "a normal restart of the same
    install" apart from "a different install that happens to share this
    browser" and never render, merge, or re-upload the other install's cached
    conversation history into this one's own data directory."""
    ensure_dirs()
    path = HOME_DIR / "instance_id.txt"
    with _io_lock:
        if path.is_file():
            try:
                val = path.read_text(encoding="utf-8").strip()
                if val:
                    return val
            except OSError as e:
                # The marker exists but could not be read - do not silently
                # treat this as "no id yet" without saying why (rule 5); fall
                # through and mint a fresh one for this run.
                print(f"[localm] WARNING: cannot read {path} ({e}); minting a "
                      "fresh instance id for this run.", file=sys.stderr)
        val = uuid.uuid4().hex
        try:
            path.write_text(val, encoding="utf-8")
        except OSError as e:
            # Cannot persist: this run's id will not survive a restart, so this
            # install will fail to recognise its OWN cache next launch (the
            # client just treats it as a new pairing and starts empty/re-syncs
            # from the server - not a privacy failure, but a real usability
            # degradation worth surfacing rather than hiding).
            print(f"[localm] WARNING: cannot persist instance id to {path} "
                  f"({e}); using an in-memory-only id for this run (it will "
                  "change on the next start).", file=sys.stderr)
        return val


def _merge_stored_config(cfg: dict, stored) -> None:
    """Overlay the persisted config delta *stored* onto *cfg* (the defaults).

    A present-but-non-dict config.json - valid JSON that is a list / string /
    number / null, or any non-object - is ignored so it cannot corrupt the merge,
    but that discard is SURFACED once per process (do-not-hide-problems): the
    user's saved settings are being dropped, which must never be silent. A MISSING
    file arrives here as the ``{}`` default (a dict) and is a normal no-op, not a
    warning; a genuinely unparseable file already warned in _read_json and also
    arrives as ``{}``. stderr, not the logger (see _detect_home / AUD-DETECTHOME)."""
    if isinstance(stored, dict):
        cfg.update(stored)
        return
    key = str(CONFIG_FILE)
    if key not in _warned_bad_config:
        _warned_bad_config.add(key)
        print(f"[localm] config.json is not a JSON object (got "
              f"{type(stored).__name__}); ignoring it and using defaults. The file "
              "is left untouched so you can recover it by hand.", file=sys.stderr)


def load_config() -> dict:
    ensure_dirs()
    # DEEP copy: a shallow .copy() shares the nested mutable defaults (e.g. the
    # "plugins" dict) with DEFAULT_CONFIG, so a caller mutating cfg["plugins"][x]
    # (per-plugin media config, workflow selection) would silently corrupt the
    # module-level DEFAULT_CONFIG for the rest of the process.
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    with _io_lock:
        stored = _read_json(CONFIG_FILE, {})
    _merge_stored_config(cfg, stored)
    return cfg


def keep_diagnostics_enabled() -> bool:
    """Whether to keep diagnostic traces/logs even in privacy mode. Resolved from
    the ``LOCALM_KEEP_DIAGNOSTICS`` env (set by the launcher checkbox / the
    ``--keep-diagnostics`` flag, a per-run override) OR the persistent
    ``keep_diagnostics`` config key (the WebUI Settings > Privacy toggle). So the
    launcher and the in-app toggle both take effect. Never raises."""
    if os.environ.get("LOCALM_KEEP_DIAGNOSTICS", "").strip().lower() in (
            "1", "true", "on", "yes"):
        return True
    try:
        return bool(load_config().get("keep_diagnostics"))
    except Exception:
        return False


def _user_delta(cfg: dict) -> dict:
    """Reduce *cfg* to the keys that need persisting: values that differ from
    the CURRENT DEFAULT_CONFIG, plus keys DEFAULT_CONFIG does not know (a
    config written by a newer version, or version-scoped state such as
    plugins_first_use_done).

    Persisting only this user-set delta is what lets a later change to a
    DEFAULT_CONFIG value reach existing installs (LM-DA-001): an absent key
    means "follow the default" and load_config() reconstructs it at read
    time. The old scheme wrote the full merged dict, freezing every default
    at its then-current value on the user's first save - commit cfa25d5's
    max_tokens 1024 -> 4096 fix never reached a config saved before it.

    Two documented consequences of delta persistence:
      - A value set EQUAL to the current default is indistinguishable from
        "follow the default" and is dropped, so it tracks future default
        changes (the usual user-settings-file semantics).
      - One-time migration ambiguity for config.json files written under the
        old full-dump scheme: a stored key equal to the CURRENT default is
        safely dropped on the next save, but a stored value that differs
        (for example a frozen OLD default such as max_tokens 1024) cannot be
        told apart from a deliberate user choice - provenance was destroyed
        at the original save - so it is KEPT, preserving the user-choice
        reading rather than silently changing a value the user may have set.
    """
    return {k: v for k, v in cfg.items()
            if k not in DEFAULT_CONFIG or v != DEFAULT_CONFIG[k]}


def save_config(cfg: dict) -> None:
    """Persist *cfg* atomically, writing ONLY the user-set delta.

    Callers keep passing the full defaults-merged dict (usually from
    load_config()); keys whose value equals the current default are not
    written to disk and are reconstructed by load_config(), so a shipped
    default-value fix applies to existing installs. See _user_delta for the
    exact rules and the migration of old full-dump files."""
    ensure_dirs()
    with _io_lock:
        _atomic_write_json(CONFIG_FILE, _user_delta(cfg))


# update_config()/update_registry() are read-modify-write: read the file, let a
# mutator edit the in-memory dict, then write it back. _io_lock only serializes
# that whole cycle within THIS process - a genuinely separate localm OS process
# (the CLI `localm config` racing a running server's PATCH /v1/config, or two CLI
# invocations) has its OWN _io_lock and can interleave its own read-modify-write
# entirely inside this process's window, silently losing whichever change gets
# read-before-written-back last (reproduced directly: see
# tests/test_config_cross_process_lock.py). A lock FILE closes that gap across
# processes. Deliberately not the `filelock` package: huggingface-hub (a CORE,
# always-installed dependency - pyproject.toml's base `dependencies`, not the
# `gpu` extra) pulls filelock in transitively, so it IS present in every install
# today - but it is not a DIRECT dependency of THIS project (no pyproject.toml
# entry of its own), so importing it here would rely on another package's
# transitive dependency choice rather than a contract localm actually owns; a
# future huggingface-hub release dropping or relocating it would silently break
# config.py, the one module every install (including a minimal one with no GPU
# extras) needs to work. os.open(..., O_CREAT | O_EXCL) is an atomic
# create-only-if-absent op on both Windows and POSIX, so exactly one process at
# a time can hold the lock.
_CROSS_LOCK_TIMEOUT = 10.0      # seconds to wait for a lock held by another process
_CROSS_LOCK_STALE_AGE = 30.0    # a lock file older than this is presumed abandoned
                                 # by a crashed holder and is reclaimed, so a dead
                                 # process can never wedge every future write forever
_CROSS_LOCK_POLL = 0.02         # seconds between acquire attempts; escalates up to
_CROSS_LOCK_POLL_CAP = 0.25     # this cap under sustained contention


def _cross_lock_backoff(attempt: int) -> None:
    time.sleep(min(_CROSS_LOCK_POLL * (attempt + 1), _CROSS_LOCK_POLL_CAP))


def _lock_owner_pid(raw: bytes):
    """Extract the PID from a lock file's ``<pid>:<nonce>`` token (or a bare
    ``<pid>``, what a test double may write directly). None if unparseable -
    including an orphaned/empty file left by a write that failed partway
    through (see _cross_process_lock's acquire-failure cleanup)."""
    try:
        return int(raw.split(b":", 1)[0])
    except (ValueError, IndexError):
        return None


# The fencing tokens of locks THIS process currently holds, keyed by lock path.
# This - not the pid recorded in the file - is what identifies a lock as ours.
#
# REG-586: a pid NUMBER is not an identity. The OS reuses pids freely across
# process lifetimes, so a lock LEAKED by a crashed holder can carry the very pid
# the OS later hands to a new localm process. Reading that as "held by me" turned
# the clear nested-call error into a permanent wedge of every config/registry
# write for that process's whole life, and short-circuited the staleness reclaim
# that exists to prevent exactly that. The uuid4 nonce in each token makes this
# exact instead: a leaked file can never match a token we are holding right now,
# whatever pid it records.
_held_lock_tokens: dict = {}
_held_lock_tokens_guard = threading.Lock()


def _lock_is_held_by_us(lockpath: Path, held: bytes) -> bool:
    """True only when *held* is a fencing token THIS process wrote and still
    holds - i.e. a genuine nested acquisition, not a pid collision."""
    if not held:
        return False
    with _held_lock_tokens_guard:
        return _held_lock_tokens.get(str(lockpath)) == held


@contextlib.contextmanager
def _cross_process_lock(target: Path):
    """Hold an exclusive, cross-process lock on *target* (a sibling ``<name>.lock``
    marker file) for the duration of the ``with`` block.

    Bounded retry with an escalating backoff, matching _replace_atomic's
    established pattern, but on a wall-clock budget (_CROSS_LOCK_TIMEOUT) rather
    than a fixed attempt count, since a full read-modify-write held by another
    process can legitimately take longer than a single os.replace. Timing out
    raises TimeoutError rather than silently proceeding unprotected - a lock that
    cannot be acquired must never be treated as "acquired" (do-not-hide-problems).

    FENCING TOKEN: each acquisition writes a unique ``<pid>:<nonce>`` token into
    the lock file, not just a bare marker. This closes a real hole a bare
    create/delete marker has: a holder whose critical section legitimately
    outlasts _CROSS_LOCK_STALE_AGE (not crashed, just slow - heavy antivirus
    scanning, a paused debugger) would otherwise have its lock silently stolen
    by a waiter that assumes staleness means "crashed," and when the ORIGINAL
    holder finally finishes and unconditionally deletes "the lock file," it
    would actually be deleting the NEW holder's still-active lock, letting a
    THIRD writer in - reopening the exact silent-clobber race this whole
    mechanism exists to close, via its own recovery path. With the token,
    release only removes the file if it still holds the token THIS call wrote;
    if it doesn't (this call's lock was reclaimed as stale while still legitimately
    held), the file is left alone - whoever's token is on disk owns cleanup, so a
    stale-reclaim can never cascade into deleting a live holder's lock.

    The same token also turns a same-thread NESTED call (a mutator passed to
    update_config()/update_registry() that calls back into either for the same
    file - not supported, this lock is not reentrant like _io_lock) from a
    confusing _CROSS_LOCK_TIMEOUT-long stall into an immediate, clear error: the
    token on disk matches one in _held_lock_tokens, which is only possible via
    the calling process's own nested acquisition (a sibling thread in this
    process would already be blocked on the outer _io_lock before ever reaching
    here). Ownership is decided by that exact token, NOT by the pid recorded in
    the file: pids are reused across process lifetimes, so a leaked lock carrying
    our own pid must still be treated as foreign, and stay eligible for the
    staleness reclaim below (REG-586 - a self-pid check here wedged every write
    for the process's whole life, defeating that reclaim).

    A lock file older than _CROSS_LOCK_STALE_AGE that we do not hold is assumed
    to belong to a crashed holder (a killed CLI, a hard-killed server) and is
    reclaimed instead of wedging every future config/registry write for the rest
    of the install's life; the reclaim is logged, never silent."""
    lockpath = target.with_name(target.name + ".lock")
    token = f"{os.getpid()}:{uuid.uuid4().hex}".encode("ascii")
    deadline = time.time() + _CROSS_LOCK_TIMEOUT
    attempt = 0
    while True:
        try:
            fd = os.open(str(lockpath), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                held = lockpath.read_bytes()
            except OSError:
                held = b""
            if _lock_is_held_by_us(lockpath, held):
                raise RuntimeError(
                    f"{lockpath.name} is already held by this same process "
                    f"(pid {os.getpid()}) - a mutator passed to update_config()/"
                    "update_registry() called back into update_config()/"
                    "update_registry() on the same file. That is not supported "
                    "(the cross-process lock is not reentrant); make both changes "
                    "in one mutator instead of nesting the calls.")
            try:
                age = time.time() - lockpath.stat().st_mtime
            except OSError:
                # The lock vanished between our failed create and this stat (the
                # holder just released it) - retry the create immediately rather
                # than waiting out a backoff for a lock that is already gone.
                continue
            if age > _CROSS_LOCK_STALE_AGE:
                print(f"[localm] note: reclaiming stale lock {lockpath.name} "
                      f"({age:.0f}s old) - the process that held it appears to "
                      "have crashed without releasing it.", file=sys.stderr)
                try:
                    lockpath.unlink()
                except OSError:
                    pass
                continue
            if time.time() >= deadline:
                raise TimeoutError(
                    f"timed out after {_CROSS_LOCK_TIMEOUT:.0f}s waiting for "
                    f"{lockpath.name}, held by another localm process")
            _cross_lock_backoff(attempt)
            attempt += 1
            continue
        # We created the lock file (os.open succeeded). Everything from here
        # must clean up OUR OWN just-created file on failure, or a transient
        # write error (ENOSPC, a momentary AV lock on the file we just made)
        # would leak an orphaned, unowned lock file that blocks every other
        # config/registry writer install-wide until the next staleness reclaim.
        try:
            try:
                os.write(fd, token)
            finally:
                os.close(fd)
        except BaseException:
            try:
                lockpath.unlink()
            except OSError:
                pass
            raise
        # Record the token only once it is actually ON DISK: a failed write above
        # unlinks the file, so registering earlier would leave us believing we
        # hold a lock that does not exist.
        with _held_lock_tokens_guard:
            _held_lock_tokens[str(lockpath)] = token
        break
    try:
        yield
    finally:
        # Forget our claim FIRST, and unconditionally: once we leave this block we
        # no longer hold the lock, whatever happens to the file below. Leaving a
        # stale entry here would make a LATER acquisition of the same path read a
        # foreign lock as our own nested call (the REG-586 wedge, rebuilt in
        # memory).
        with _held_lock_tokens_guard:
            if _held_lock_tokens.get(str(lockpath)) == token:
                del _held_lock_tokens[str(lockpath)]
        # Fencing-token release (see docstring): only remove the lock file if it
        # still holds the token WE wrote. If it doesn't, another process reclaimed
        # it as stale while we were still legitimately inside our critical section
        # (a write that outlasted _CROSS_LOCK_STALE_AGE) - deleting THEIR live
        # lock here would let a third writer in while they still believe they
        # hold exclusive access, so we leave it alone instead.
        try:
            current = lockpath.read_bytes()
        except OSError:
            current = None
        if current == token:
            try:
                lockpath.unlink()
            except OSError:
                pass
        elif current is not None:
            print(f"[localm] note: {lockpath.name} was reclaimed by another "
                  "localm process while this process still held it (this "
                  f"write took longer than _CROSS_LOCK_STALE_AGE={_CROSS_LOCK_STALE_AGE:.0f}s) "
                  "- not deleting the new holder's lock.", file=sys.stderr)


def update_config(mutator: Callable[[dict], None]) -> dict:
    """Atomically read-modify-write the config.

    Holds _io_lock (serializes threads within this process) AND a cross-process
    lock file (serializes separate localm OS processes - e.g. the CLI
    `localm config` racing a running server's PATCH /v1/config, or two CLI
    invocations) across the WHOLE read-modify-write, so no writer, in this
    process or another, can read a stale copy and silently clobber a concurrent
    change. *mutator* receives the loaded config dict (defaults merged) and edits
    it in place; the result is persisted with a single atomic write. Use this
    instead of a bare load_config()/save_config() pair wherever a lost update
    would matter."""
    ensure_dirs()
    with _io_lock, _cross_process_lock(CONFIG_FILE):
        cfg = copy.deepcopy(DEFAULT_CONFIG)   # deep: see load_config (nested dicts)
        stored = _read_json(CONFIG_FILE, {})
        _merge_stored_config(cfg, stored)
        mutator(cfg)
        # The mutator and the return value see the full merged dict; only the
        # user-set delta hits the disk (see _user_delta / save_config).
        _atomic_write_json(CONFIG_FILE, _user_delta(cfg))
        return cfg


def load_registry() -> dict:
    ensure_dirs()
    with _io_lock:
        reg = _read_json(REGISTRY_FILE, {})
    return reg if isinstance(reg, dict) else {}


def save_registry(reg: dict) -> None:
    ensure_dirs()
    with _io_lock:
        _atomic_write_json(REGISTRY_FILE, reg)


def update_registry(mutator: Callable[[dict], None]) -> dict:
    """Atomically read-modify-write the registry.

    Holds _io_lock (serializes threads within this process) AND a cross-process
    lock file (serializes separate localm OS processes - e.g. a CLI `pull`
    running alongside the GUI, or two CLI invocations) across the WHOLE
    read-modify-write, so no writer, in this process or another, can read a
    stale copy and silently clobber a concurrent change (the same closed gap as
    update_config(), see its docstring / _cross_process_lock). *mutator*
    receives the registry dict and edits it in place; the result is persisted
    with a single atomic write. Use this instead of a bare
    load_registry()/save_registry() pair wherever a lost update would matter.
    (save_registry() itself remains a blind overwrite - last-writer-wins by
    design, not a read-modify-write, so it needs no lock beyond the atomic
    write it already has.)"""
    with _io_lock, _cross_process_lock(REGISTRY_FILE):
        reg = _read_json(REGISTRY_FILE, {})
        if not isinstance(reg, dict):
            reg = {}
        mutator(reg)
        _atomic_write_json(REGISTRY_FILE, reg)
        return reg


def _loadable_lib_names() -> tuple:
    """Loadable native llama library filename(s) for this platform."""
    if sys.platform == "win32":
        return ("llama.dll",)
    if sys.platform == "darwin":
        return ("libllama.dylib",)
    return ("libllama.so",)


def find_binary_dir() -> Optional[Path]:
    """Return the directory holding the native llama.cpp binaries (llama.dll,
    plus optional llama-cli/llama-server exes), used by `localm info` and
    `localm doctor`, or None when unprovisioned.

    Project-local resolution only: the user-configured binary_dir, then the
    localm-llama-runtime wheel bundled in this venv. No absolute path is ever
    assumed as a default; an unprovisioned install resolves to None and the
    user runs `localm setup-llama` (or sets binary_dir)."""
    cfg = load_config()
    candidates = []
    if cfg.get("binary_dir"):
        candidates.append(Path(cfg["binary_dir"]))
    try:
        import localm_llama_runtime
        d = localm_llama_runtime.lib_dir()
        if d:
            candidates.append(Path(d))
    except Exception:
        pass
    names = _loadable_lib_names()
    for p in candidates:
        try:
            if any((p / n).exists() for n in names):
                return p
        except OSError:
            continue
    return None


def get_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
