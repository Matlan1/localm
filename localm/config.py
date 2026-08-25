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
    """Surface (once) that no data dir was configured, so a missing / lost config is VISIBLE instead of silently masked (do-not-hide-problems). stderr, not the logger: this runs at import time before logging is wired."""
    # setup.bat / setup.sh run setup-llama before the data dir is chosen and set
    # LOCALM_SETUP=1 to suppress this warning during that phase. The only
    # suppressor; it must never be set outside the setup scripts.
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
    """Resolve the localm data directory (config, registry, models, logs, sessions, generated images/music)."""
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
                # The marker exists but could not be read, so surface it rather
                # than silently falling through to a different data dir. stderr,
                # not the logger: this runs at import time, before logging is
                # wired, and importing debuglog here would be circular.
                print(f"[localm] WARNING: cannot read {cfg_marker} ({e}); "
                      "falling back to the default data dir.", file=sys.stderr)
        portable = repo_root / "home"
        if portable.is_dir():
            return portable

    # No data dir configured: no LOCALM_HOME, no localm-home.cfg, no ./home.
    # Defaults to a contained ./home inside the install and says so on stderr,
    # never a shared ~/.localm outside the install.
    fallback = repo_root / "home"
    _warn_unconfigured_home(fallback)
    return fallback


def home_dir() -> Path:
    """Lazy variant of HOME_DIR - resolves the data dir at call time."""
    return _detect_home()


def cache_dir() -> Path:
    """Root for caches localm's OWN subprocesses write (rule 4: self-contained)."""
    return home_dir() / "cache"


def pip_cache_dir() -> Path:
    """localm's OWN pip cache, inside the data dir (rule 4: self-contained)."""
    return cache_dir() / "pip"


def uv_cache_dir() -> Path:
    """localm's OWN uv cache, inside the data dir (rule 4: self-contained)."""
    return cache_dir() / "uv"


def contained_pip_env(base: Optional[dict] = None) -> dict:
    """A subprocess environment with pip's AND uv's caches pinned inside the data dir."""
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
    # Which llama.cpp release `setup-llama` provisions. Empty = the build localm
    # ships and has tested (setup_llama._PINNED_TAG); the literal "latest" tracks
    # whatever upstream published most recently; a tag such as "b10355" pins that
    # exact build. Read by setup_llama._tag_for()/tracks_latest(), so `localm
    # update`'s re-provision inherits the choice.
    "llama_runtime_pin": "",
    # Append-only log of the runtime builds provisioned on this box, newest last:
    # [{"backend": ..., "tag": ..., "at": <epoch>}, ...]. Backs `setup-llama
    # --rollback`. Capped at setup_llama._RUNTIME_HISTORY_MAX.
    "llama_runtime_history": [],
    "n_ctx": 4096,         # initial context window (grows on demand)
    "n_ctx_max": 16384,    # ceiling the window may grow to (0 = unlimited)
    "n_ctx_grow": 4096,    # growth step - window expands in multiples of this
    # Size ctx ceiling from free VRAM at load (clamped 4k-64k); window still
    # starts at n_ctx and grows on demand. False = use fixed n_ctx_max.
    "ctx_auto": True,
    "n_gpu_layers": 99,    # 99 = offload everything to GPU
    # Opt-in MoE expert placement: keep the expert weights of the first N layers
    # in system RAM while the rest follows the normal layer assignment
    # (llama.cpp's --n-cpu-moe). 0 = off. A VRAM-footprint dial, not a speed-up:
    # at matched VRAM it is throughput-neutral. Only affects Mixture-of-Experts
    # models; a dense model has no expert tensors to move.
    "n_cpu_moe": 0,
    # With n_gpu_layers at its "everything" default, auto-size how many layers go
    # on the GPU from free VRAM at load, so a model too big for full offload runs
    # some layers on CPU and loads instead of being refused. Any explicit
    # n_gpu_layers is honoured verbatim.
    "n_gpu_layers_auto": True,
    # Enable Multi-Token Prediction (MTP) speculative decoding for models with
    # native MTP/next-n prediction heads (e.g. DeepSeek-V3/R1, Qwen MTP models).
    # True = active when model supports it; False = force standard autoregressive.
    "mtp_enabled": True,
    # VRAM (MB) reserved beyond model weights before deciding how many layers fit.
    # Funds the KV cache's compute buffers and llama.cpp's graph/scratch
    # allocations. Lowering it risks a native crash or a GPU driver hang instead
    # of a slower but working load.
    "vram_overhead_mb": 1500,
    # Ceiling (seconds) for a GGUF model load in its isolated worker process
    # before it is treated as hung and killed.
    "gguf_load_timeout_s": 900.0,
    # How long a reply may take to produce its FIRST token before the worker is
    # treated as hung. Covers prompt prefill, not one token's decode, so it is
    # sized like the load timeout.
    "gguf_first_token_timeout_s": 900.0,
    # Ceiling (seconds) for an HF model load in its isolated worker process before
    # it is treated as hung and killed. HF loads read full-precision safetensors
    # with no quantized mmap fast path.
    "hf_load_timeout_s": 900.0,
    # How long an HF reply may take to produce its first token before the worker
    # is treated as hung. Covers prompt prefill, not one token's decode.
    "hf_first_token_timeout_s": 900.0,
    # Bounded wait for one HF embed() RPC. More generous than the dedicated
    # GGUF-based embedder's timeout: HFBackend.embed() loops over texts one at a
    # time with no batching. hf_embed_max_texts/hf_embed_max_chars reject an
    # oversized request outright; this bounds whatever is allowed through.
    "hf_embed_timeout_s": 600.0,
    # Per-request caps on an HF-backed /v1/embeddings request, enforced by
    # HFBackend.embed() before a batch reaches the isolated worker. Two axes: many
    # texts means many one-at-a-time forward passes, and a huge text is slow to
    # tokenize. Not applicable to GGUF or the dedicated on-device embedder.
    "hf_embed_max_texts": 256,
    "hf_embed_max_chars": 200_000,
    # GPU device to load onto and read VRAM from on a multi-GPU box. None = no
    # explicit selection (device 0). A stale index falls back to device 0 with a
    # logged warning.
    "main_gpu_index": None,
    # Split a model too big for one card across 2+ GPUs (GGUF: llama.cpp
    # layer-split; HF: accelerate device_map restricted to these devices).
    # None/empty/one entry = off. A device no longer detected at load time is
    # dropped with a logged warning.
    "gpu_split_indices": None,
    # Optional relative weight per entry in gpu_split_indices, same length, any
    # positive numbers. None distributes automatically in proportion to each
    # card's free VRAM at load time, falling back to an equal split when
    # per-device free cannot be measured or the lengths mismatch.
    "gpu_split_ratios": None,
    # How many chat models may stay loaded at once. None = no cap: free-VRAM
    # arithmetic alone decides. An integer bounds it regardless of headroom; 1 is
    # strict single-resident. Applies to the HTTP server and the MCP server.
    "max_resident_models": None,
    # Model display names that are never chosen as an eviction victim. Pinning
    # protects an already-loaded model; it never loads one. If pins leave nothing
    # evictable the load proceeds and the miss is logged.
    "pinned_models": None,
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
    # Sidebar wordmark, shared by the web GUI and the desktop launcher. One of:
    # local-m (default), loca-lm, localm. The console command, app icon and
    # shortcut are fixed regardless.
    "logo_style": "local-m",
    # Standalone app window (localm[desktop] extra): its close button hides it to
    # the tray by default. True makes the close button quit the app and stop the
    # server. No effect when localm opens in a browser tab.
    "desktop_window_quit_on_close": False,
    # "auto" (default) uses the standalone app window when localm[desktop] is
    # installed, otherwise the browser tab. "browser" always uses the browser tab.
    # There is no "always window" value: run_native_window's fallback-on-failure
    # is a safety net a setting must not defeat.
    "desktop_window_mode": "auto",
    "import_max_depth": 3,    # `localm add <dir>` recurses up to this many levels
    "port": 8642,             # default inference server port (auto-bumps if busy; an explicit --port does not)
    # Bind address for a fresh server start when -H/--host is not given: "" is
    # loopback only, "0.0.0.0" is every interface, or one specific interface IP.
    # GUI-settable and applied on restart; an explicit -H wins for that process
    # and survives an in-place restart. Binding past loopback requires a strong
    # API key: without one the server ignores this key and stays on loopback. The
    # --insecure override has no config form.
    "bind_host": "",
    # Built-in TLS on network binds. tls_enabled False is the persistent form of
    # --no-tls; tls_cert/tls_key are the persistent form of --tls-cert/--tls-key
    # (blank = localm's own local-CA cert). CLI flags win over all three, and
    # loopback binds never use TLS.
    "tls_enabled": True,
    "tls_cert": "",
    "tls_key": "",
    "cors_origins": None,     # None = localhost only; list of origins; or "*"
    # Require a configured API key on protected endpoints: true refuses requests
    # until a key is set (see localm/auth.py); env override LOCALM_REQUIRE_AUTH.
    # False (default) = open in local/dev mode on loopback.
    "require_auth": False,
    # Let transformers import and run a HuggingFace model directory's own bundled
    # Python (its `auto_map` custom code) when loading it. That is arbitrary code
    # execution as the server user, so it is off by default and owner-only. With
    # it off, a model needing custom code is refused with an explanatory error.
    "hf_trust_remote_code": False,
    # Whether localm may treat a Windows drive letter mapped to a network share as
    # a normal local folder. Checked by the GUI folder picker, create-folder,
    # rename and log-export routes, and by RAG document indexing. True (default)
    # keeps a mapped drive working exactly like a local one.
    "allow_network_drives": True,
    # Quick-select scope bundles for the "Keys & devices" manager, each
    # {name, scopes}. Re-seeded only when absent, so an emptied list stays empty.
    # Privileged scopes apply only when an owner mints the key.
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
    # fast default on submit: float32 doubles unpacked model size and forces CPU
    # offload on a VRAM-limited card. False submits the dequant choice verbatim.
    "comfy_fast_dequant": True,
    # Shared fallback for the per-plugin "Model weight dtype" Media setting; the
    # music/video backends read config["plugins"][<name>]["comfy"]["float_type"]
    # first. None inherits the workflow default.
    "comfy_float_type": None,
    # Suppress ComfyUI opening its own web page when localm auto-launches it.
    # True appends --disable-auto-launch; a launcher that drops extra args ignores
    # it. Applies to image, music and video.
    "comfy_disable_auto_launch": False,
    # Opt-in in-memory shim for the upstream ComfyUI __func__ regression. Off by
    # default. When on, a ComfyUI localm spawns gets a localm-owned shim dir on
    # its child PYTHONPATH; localm never writes into the user's install nor shims
    # a ComfyUI it did not start. Self-expires once Comfy is fixed.
    "comfy_func_shim": False,
    # Which ComfyUI localm targets. "own" (default) uses a localm-managed instance
    # once one is installed under <LOCALM_HOME>/comfyui, and is inert until then.
    # "user" always uses the user's own ComfyUI (comfy_workdir / comfy_api_url).
    # This key only routes and never modifies the user's own ComfyUI.
    "comfy_target": "own",
    # Per-component GPU placement for media generation, default off. When on, and
    # the running ComfyUI offers the multigpu Select*Device nodes, and a 2+ card
    # split is configured, localm injects those nodes so the text encoder and VAE
    # load on a second card while the diffusion model stays on the preferred one.
    # ComfyUI's gpu:N is a position in a reordered visible list, so an off-by-one
    # lands a component on the wrong card and still renders.
    "comfy_gpu_placement": False,
    # Session persistence mode for ALL surfaces (chat, server, GUI, coder):
    #   privacy = no traces written automatically (default)
    #   log     = JSONL audit trail in <data dir>/sessions/
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
    # Grow memory automatically: after a chat turn in log or full mode, distil
    # durable facts into the store in the background, debounced to once per
    # MEMORY_AUTO_MIN_INTERVAL. Skipped in privacy mode.
    "memory_auto_consolidate": True,
    # Privacy mode normally disables memory ENTIRELY (no recall + no writes). On =
    # allow READING existing memories into the prompt in privacy mode (writing
    # stays off - privacy never creates a trace). Off by default. Per-surface:
    "memory_recall_in_privacy": False,
    "memory_recall_in_privacy_chat": True,      # applies only when the master is on
    "memory_recall_in_privacy_coder": True,     # applies only when the master is on
    # Keep diagnostics for bug reports even in privacy mode, which otherwise
    # suppresses the hang watchdog trace, the crash-restart breadcrumbs and the
    # debug log. Never chat content: code stacks and operational logs only.
    "keep_diagnostics": False,
    # On-device embedding model for semantic search (RAG hybrid retrieval + agent
    # memory): a small dedicated GGUF, loaded separately from the chat model.
    # Value = a known key (bge-small-en-v1.5, nomic-embed-text-v1.5), a registered
    # model name, or a GGUF path. A known model is fetched into
    # <home>/models/embeddings on first use (auto only under net_mode=allow, else
    # run 'localm setup-embeddings'). Until present, memory/RAG fall back to BM25.
    "embedding_model": "bge-small-en-v1.5",
    # GPU layers for the embedding model. None (default) places it automatically:
    # full GPU offload when free VRAM holds the model (file size + 20%), else CPU,
    # so a resident chat model is not thrashed by WDDM oversubscription. 0 forces
    # CPU; 99 forces full GPU offload. When unset, an explicit global
    # n_gpu_layers other than 99 is inherited.
    "embedding_gpu_layers": None,
    # How the embedding model's token states are pooled into one vector. None is
    # not the same as "mean": embedder.py applies a per-model default - mean for
    # the bundled bge/nomic choices, last for a model that declares last-token
    # pooling. An explicit choice ("mean", "last", "cls", "none", or "auto" to
    # follow whatever the GGUF declares) overrides that. Changing the effective
    # pooling invalidates already-embedded RAG collections and memory vectors, so
    # re-index after changing it.
    "embedding_pooling": None,
    # Which host folders the document-indexing (RAG) API may read. All three keys
    # are owner-only: a non-owner config:write key can neither see nor set them.
    # The localm data dir and credential folders stay denied in every mode. Read
    # by indexing_policy().
    #   whitelist (default) = only home, the working dir, and rag_allowed_roots.
    #   blacklist           = anywhere except rag_denied_roots.
    "rag_indexing_mode": "whitelist",
    "rag_allowed_roots": [],   # extra folders allowed in whitelist mode
    "rag_denied_roots": [],    # folders refused in blacklist mode
    # A document's format label is derived heuristic-first: a known extension
    # wins, else a structural sniff. This toggle governs only the LLM tie-break,
    # when both are inconclusive and a chat model is loaded. Off labels it "text".
    # Never fires during an embedding-only index.
    "rag_classify_unknown_files": True,
    # Seconds a GUI coder approval card may sit unanswered before it is
    # auto-rejected and the agent moves on.
    "coder_confirm_timeout": 600,
    # Wall-clock cap on the coder's startup project-map scan (CODER-1). <= 0
    # disables the deadline (scan to completion however long it takes).
    "coder_index_timeout": 20,
    # Caps on the coder's grep tool, each overridable per call; 0 = no cap.
    # Matches shown per file (the rest are still counted), output lines before the
    # sweep stops, and the per-file size above which a file is skipped.
    "coder_grep_max_per_file": 20,
    "coder_grep_max_output_lines": 300,
    "coder_grep_max_file_bytes": 4194304,
    # Episodic memory: the coder recalls lessons from past sessions on a project
    # and at session close distils the session into a new lesson. Writes are
    # skipped in privacy mode and for restricted sessions, and are stored under
    # the home dir rather than the project tree. False disables both halves.
    "coder_episodic_memory": True,
    # Provenance tagging: re-frame coder tool results from untrusted network or
    # MCP tools as data rather than instructions. Labels only; blocks nothing.
    "coder_untrusted_provenance": True,
    # Pre-done self-review: before the coder declares done, a reviewer model reads
    # the diff and feeds blocking issues back for one more fix pass. Off by default
    # (adds a model round-trip per task that changed files).
    "coder_review": False,
    # Reviewer target: "" is the agent's own model; "local" is a different small
    # model on CPU in the coder's process (set coder_reviewer_model);
    # "openai"/"anthropic" are cloud; an http(s) URL is a second
    # OpenAI-compatible endpoint. A network reviewer is skipped in privacy mode
    # and for shared keys, falling back to the local model.
    "coder_reviewer": "",
    # Model name (or path) for a heterogeneous reviewer ("local"/cloud/URL); blank
    # uses a sensible provider default or the agent's own model name.
    "coder_reviewer_model": "",
    # Constrain coder tool calls with a lazy GBNF grammar: prose flows free, but a
    # started <tool_call> is forced to valid JSON. On by default for
    # grammar-capable local backends; external API and grammar-less builds are
    # unaffected via the supports_grammar gate and a runtime soft-degrade. A
    # config.json written before the default flipped keeps its saved False.
    "coder_tool_grammar": True,
    # After an image is generated, ask ComfyUI to release VRAM and reload the chat
    # model so the next reply is instant. Off = the chat model reloads lazily on
    # the next message instead (better for many images in a row).
    "reload_llm_after_imagine": True,
    # VRAM-aware media model swap: before a media gen the chat LLM is unloaded so
    # the media model gets the GPU.
    #   auto   = keep chat loaded when the media model fits alongside it, else swap
    #   always = always unload the chat model
    #   never  = never unload; media may OOM on a small card
    # reload_llm_after_imagine is a separate axis, covering reload AFTER a gen.
    "model_swap_policy": "auto",
    # Free the loaded model from VRAM after this many idle seconds; the next
    # request reloads it lazily. 0 = disabled: resident until an explicit unload
    # or swap. Measured from the last request.
    "idle_unload_seconds": 0,
    # Network policy for model-initiated requests (coder fetch_url/web_search,
    # chat web access).
    #   off   = all policy-routed network access fails fast
    #   ask   = allowed; the coder asks for approval per request (default)
    #   allow = no confirmation
    "net_mode": "ask",
    "net_allow": [],            # domains; empty = any. "x.com" covers *.x.com
    "net_deny": [],             # domains always refused (wins over allow)
    "net_allow_private": False, # True = permit loopback/private targets (SSRF guard off)
    "net_search_url": None,     # SearXNG base URL; None = DuckDuckGo (no key)
    # Display a remote image a model links in a reply by fetching it server-side
    # and streaming it back, so the browser never contacts the remote origin. Off
    # by default: a rendered remote image is a model-driven exfiltration channel,
    # and this only moves the request from the browser to this machine. With it
    # on, the remote host never sees the browser's IP, User-Agent or referrer, and
    # the fetch is subject to the same SSRF guard and domain lists as any other.
    "gui_proxy_remote_images": False,
    "coder_rail_side": "right",
    "coder_remember_projects": True,
    "coder_projects_remembered": 20,
    # Reach localm by name on a network bind. mDNS advertises "<mdns_name>.local",
    # the name is folded into the TLS cert, and the Tailscale MagicDNS name is
    # detected and certified automatically. Loopback binds never advertise.
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
    # When a command belongs to a known first-party plugin that is not installed
    # or enabled, suggest installing it instead of "unknown command". False
    # silences the hint; a truly unknown command always errors.
    "suggest_plugins": True,
    # When a plugin declaring pip extras is installed or enabled by the local
    # operator, auto-install those extras on the host. A remote client never
    # triggers a server-side pip. `localm plugin setup` records the choice here.
    "auto_install_plugin_deps": True,
    # Bug reports, the read-only Issues view and the self-updater all talk to one
    # Cloudflare Worker that holds the GitHub tokens server-side. Shipped as
    # defaults so a fresh download works with no setup. No GitHub token is in the
    # app: only the public Worker URL and a low-value client token that is
    # intentionally public, gates against drive-by spam, can only file an issue,
    # and is rotatable at the Worker. Set either to "" to opt a build out.
    "bugreport_upload_url": "https://localm-bugreport-proxy.localm.workers.dev",
    "bugreport_upload_token": "3x_HA2UXbwNDnNfdDmpFBvvfcl2S-I-9t7XLQRAShM4",
    # Update channel and read-only issues tracker. One Worker hosts report, issues
    # and update, so these default to the proxy above. Set update_url/token only
    # to point updates at a different Worker. None disables the update channel.
    "update_url": None,
    "update_token": None,
    # Opt-in: the update check stays stable-only unless a local admin turns this
    # on. A prerelease build is signed and anti-rollback checked exactly like a
    # stable one; this only widens which candidate the proxy may offer.
    "update_allow_prerelease": False,
    # Net-policy carve-out for the update channel only. Defaults to False, so the
    # update check obeys net_mode like everything else unless an admin opts this
    # one channel out.
    "update_ignore_net_policy": False,
    # Names of enabled engine plugins. Managed by the plugin engine via
    # update_config, not the settings form; declared here so the settings-save
    # endpoint does not reject it as unknown. A plugin is active only when
    # installed on disk AND in this list.
    "plugins_enabled": [],
    # Per-plugin config namespace, e.g. plugins["image"]["comfy"]["output_dir"].
    # Written by the plugin engine and media backends via update_config, not the
    # flat settings form. Declared here so settings-save accepts it.
    "plugins": {},
}

# localm claims 8642-8741 - far from ComfyUI (8188), A1111 (7860),
# Ollama (11434), and the 8000/8080/8888 dev-server crowd.
PORT_RANGE = (8642, 8741)


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """True when something is already listening on host:port."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        from localm.debuglog import logger
        logger.debug("port_in_use: cannot resolve %r (%s); treating the port as "
                     "free and leaving the real error to the bind", host, e)
        return False
    for family, socktype, proto, _canon, sockaddr in infos:
        try:
            with socket.socket(family, socktype, proto) as s:
                s.settimeout(0.2)
                if s.connect_ex(sockaddr) == 0:
                    return True
        except OSError:
            # This family is unusable on this box, e.g. an IPv6 address with the
            # stack disabled. Keep probing the remaining addresses rather than
            # claiming a conflict that was not observed.
            continue
    return False


class PortInUseError(RuntimeError):
    """An explicitly requested port is already in use."""

    def __init__(self, port: int):
        self.port = port
        super().__init__(f"Port {port} is already in use.")


def pick_port(requested: Optional[int] = None, host: str = "127.0.0.1"):
    """Resolve the port to serve on."""
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
    """``path.mkdir(parents=True, exist_ok=True)`` with one user-error case turned into a clean message instead of a crash."""
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


def _perm_warn(path: Path, why: str) -> None:
    """Record a failed permission tightening at debug level."""
    try:
        from localm.debuglog import logger
        logger.debug("could not restrict permissions on %s (%s); the data "
                     "directory's own scoping still applies", path.name, why)
    except Exception:
        pass


def restrict_file_perms(path: Path, *, mode: int = 0o600) -> bool:
    """Best-effort: restrict *path* to the current user (POSIX chmod *mode*, or Windows icacls - which grants sole full control regardless of *mode*, since ACLs do not encode POSIX bits)."""
    try:
        if os.name == "posix":
            os.chmod(path, mode)
        else:
            import subprocess
            user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
            if not user:
                _perm_warn(path, "no USERNAME/USER in the environment")
                return False
            # /inheritance:r drops the inherited ACEs (BUILTIN\Users et al) and
            # /grant:r replaces any existing grant for this user, so the result
            # is an explicit, sole full-control ACE.
            r = subprocess.run(
                ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"],
                capture_output=True, check=False)
            if r.returncode != 0:
                # icacls fails without raising (access denied, an unresolvable
                # principal, a non-NTFS volume), so without this check the function
                # would return as though the file had been locked down. Reports
                # rather than raising: breaking session persistence over a
                # permissions nicety would be worse.
                _perm_warn(path, (r.stderr or r.stdout or b"").decode(
                    "utf-8", "replace").strip() or f"icacls exit {r.returncode}")
                return False
        return True
    except Exception as e:
        # Best-effort: a failure leaves the home-dir scoping in effect, which is
        # the real protection. Reported at debug and returned as False rather than
        # swallowed, so a caller can retry and nothing reads as a success that did
        # not happen.
        _perm_warn(path, repr(e))
        return False


def atomic_write_private(path: Path, text: str) -> bool:
    """Write *text* to *path* atomically, owner-restricted from the moment the bytes first exist on disk."""
    tmp = path.with_name(path.name + ".tmp")
    data = text.encode("utf-8")
    fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        written = 0
        while written < len(data):
            n = os.write(fd, data[written:])
            if not n:
                raise OSError(
                    f"short write persisting {path.name}: {written} of "
                    f"{len(data)} bytes accepted; the previous file is intact")
            written += n
    finally:
        os.close(fd)
    ok = restrict_file_perms(tmp)
    os.replace(tmp, path)          # atomic on Windows + POSIX (same dir)
    if not ok:
        restrict_file_perms(path)
    return ok


# Registry and config are mutated from several places at once: the GUI server
# threads, the `localm pull` subprocess the GUI spawns, and sync_models_dir on
# every launch. The helpers below make every write atomic (a temp file in the
# same dir, fsync, then os.replace, so readers see only the old or the new
# complete file) and every read crash-proof (fall back to the .bak snapshot,
# then to the default).
_io_lock = threading.RLock()

# A concurrent open handle makes a Windows os.replace or open raise a transient
# PermissionError (WinError 5); a bounded retry rides it out. A loaded box can
# hold it tens of ms, so the backoff escalates to a total budget of ~1s before a
# persistent failure is re-raised or falls back.
_REPLACE_RETRIES = 16
_REPLACE_BACKOFF = 0.01      # seconds; escalates up to the cap
_REPLACE_BACKOFF_CAP = 0.1   # seconds


def _transient_backoff(attempt: int) -> None:
    """Sleep before the next retry, escalating linearly to a cap so a lock that lingers on a busy machine is ridden out without a hot spin."""
    time.sleep(min(_REPLACE_BACKOFF * (attempt + 1), _REPLACE_BACKOFF_CAP))


def _is_transient_permission_error(e: OSError) -> bool:
    """True when a PermissionError plausibly reflects a TRANSIENT file lock worth riding out, rather than a stable denial no retry can change."""
    if os.name != "nt":
        return False
    # winerror is absent only if this was not raised by the Windows layer at all
    # (e.g. a test double); treat that as transient to preserve the established
    # Windows behaviour rather than silently narrowing it.
    return getattr(e, "winerror", None) in (5, 32, None)


def _replace_atomic(src: Path, dst: Path) -> None:
    """``os.replace(src, dst)`` with a bounded retry on a transient Windows sharing violation."""
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError as e:
            if attempt == _REPLACE_RETRIES - 1 or not _is_transient_permission_error(e):
                raise
            _transient_backoff(attempt)


def _atomic_write_json(path: Path, data) -> None:
    """Write *data* as JSON to *path* atomically (temp file + os.replace)."""
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
                # The .bak snapshot is best-effort and must not fail the primary
                # write, which still proceeds below. _replace_atomic already rode
                # out the transient sharing violation, so reaching here means a
                # persistent problem; noted so it is discoverable.
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


class ConfigUnreadable(RuntimeError):
    """A config/registry file EXISTS but could not be read, so a read-modify-write must not proceed (it would persist defaults over the user's real settings)."""


def _read_json(path: Path, default):
    """Read JSON from *path*, falling back to its .bak then *default* on any corruption - a damaged file must never take the whole app down."""
    return _read_json_checked(path, default)[0]


def _read_json_checked(path: Path, default):
    """``(value, read_ok)``. ``read_ok`` is False ONLY when a file was PRESENT and no candidate (neither *path* nor its ``.bak``) could be read - the one state in which the returned *default* is indistinguishable from a genuinely absent file."""
    saw_file = False
    for candidate in (path, path.with_name(path.name + ".bak")):
        if not candidate.is_file():
            continue
        saw_file = True
        for attempt in range(_REPLACE_RETRIES):
            try:
                with open(candidate, encoding="utf-8") as f:
                    return json.load(f), True
            except PermissionError as e:
                # Transient on Windows: a concurrent atomic replace, antivirus or
                # the indexer has the file locked briefly. Retry the same file
                # before giving up, so a passing scanner does not make us fall back
                # to .bak/defaults and discard live settings. A stable POSIX EACCES
                # skips the retry: it can never succeed, and this loop runs under
                # _io_lock, so the backoff would stall every config read
                # process-wide, including the per-request auth path.
                if attempt < _REPLACE_RETRIES - 1 and _is_transient_permission_error(e):
                    _transient_backoff(attempt)
                    continue
                print(f"[localm] {candidate.name} is unreadable ({e}); "
                      "falling back.", file=sys.stderr)
            except (ValueError, OSError, RecursionError) as e:
                # Not transient (corrupt or non-UTF-8 JSON, a huge integer, deep
                # nesting, or a hard OS error): fall back immediately without
                # spending the retry budget.
                print(f"[localm] {candidate.name} is unreadable ({e}); "
                      "falling back.", file=sys.stderr)
            break
    return (default() if callable(default) else default), not saw_file


def instance_id() -> str:
    """Stable, unguessable identifier for THIS install's data directory."""
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
            # install will not recognise its own cache next launch. The client
            # treats it as a new pairing and re-syncs from the server.
            print(f"[localm] WARNING: cannot persist instance id to {path} "
                  f"({e}); using an in-memory-only id for this run (it will "
                  "change on the next start).", file=sys.stderr)
        return val


def _merge_stored_config(cfg: dict, stored) -> None:
    """Overlay the persisted config delta *stored* onto *cfg* (the defaults)."""
    if isinstance(stored, dict):
        cfg.update(stored)
        return
    key = str(CONFIG_FILE)
    if key not in _warned_bad_config:
        _warned_bad_config.add(key)
        print(f"[localm] config.json is not a JSON object (got "
              f"{type(stored).__name__}); ignoring it and using defaults. The file "
              "is left untouched so you can recover it by hand.", file=sys.stderr)


def load_config_checked() -> tuple:
    """``(config, read_ok)``. ``read_ok`` is False ONLY when config.json (or its ``.bak``) was PRESENT and could not be read, so *config* is DEFAULT_CONFIG with nothing merged in - the same shape a genuinely absent config.json produces (see _read_json_checked)."""
    ensure_dirs()
    # Deep copy: a shallow .copy() shares the nested mutable defaults (e.g. the
    # "plugins" dict) with DEFAULT_CONFIG, so a caller mutating cfg["plugins"][x]
    # would corrupt the module-level DEFAULT_CONFIG for the rest of the process.
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    with _io_lock:
        stored, read_ok = _read_json_checked(CONFIG_FILE, {})
    _merge_stored_config(cfg, stored)
    return cfg, read_ok


def load_config() -> dict:
    return load_config_checked()[0]


def keep_diagnostics_enabled() -> bool:
    """Whether to keep diagnostic traces/logs even in privacy mode."""
    if os.environ.get("LOCALM_KEEP_DIAGNOSTICS", "").strip().lower() in (
            "1", "true", "on", "yes"):
        return True
    try:
        return bool(load_config().get("keep_diagnostics"))
    except Exception:
        return False


def _user_delta(cfg: dict) -> dict:
    """Reduce *cfg* to the keys that need persisting: values that differ from the CURRENT DEFAULT_CONFIG, plus keys DEFAULT_CONFIG does not know (a config written by a newer version, or version-scoped state such as plugins_first_use_done)."""
    return {k: v for k, v in cfg.items()
            if k not in DEFAULT_CONFIG or v != DEFAULT_CONFIG[k]}


def save_config(cfg: dict) -> None:
    """Persist *cfg* atomically, writing ONLY the user-set delta."""
    ensure_dirs()
    with _io_lock:
        _atomic_write_json(CONFIG_FILE, _user_delta(cfg))


# update_config() and update_registry() are read-modify-write. _io_lock only
# serializes that cycle within this process; a separate localm OS process has its
# own _io_lock and can interleave its whole read-modify-write inside this
# process's window, losing whichever change is read before being written back. A
# lock FILE closes that gap across processes.
#
# Not the `filelock` package: it is only present transitively via
# huggingface-hub, not as a direct dependency of this project, and config.py is
# the one module every install needs. os.open(..., O_CREAT | O_EXCL) is an atomic
# create-only-if-absent operation on both Windows and POSIX.
_CROSS_LOCK_TIMEOUT = 10.0      # seconds to wait for a lock held by another process
_CROSS_LOCK_STALE_AGE = 30.0    # a lock file older than this is presumed abandoned
                                 # by a crashed holder and is reclaimed, so a dead
                                 # process can never wedge every future write forever
_CROSS_LOCK_POLL = 0.02         # seconds between acquire attempts; escalates up to
_CROSS_LOCK_POLL_CAP = 0.25     # this cap under sustained contention


def _cross_lock_backoff(attempt: int) -> None:
    time.sleep(min(_CROSS_LOCK_POLL * (attempt + 1), _CROSS_LOCK_POLL_CAP))


def _lock_owner_pid(raw: bytes):
    """Extract the PID from a lock file's ``<pid>:<nonce>`` token (or a bare ``<pid>``, what a test double may write directly)."""
    try:
        return int(raw.split(b":", 1)[0])
    except (ValueError, IndexError):
        return None


# The fencing tokens of locks THIS process currently holds, keyed by lock path.
# This, not the pid recorded in the file, identifies a lock as ours: the OS
# reuses pids across process lifetimes, so a lock leaked by a crashed holder can
# carry the pid the OS later hands to a new localm process. The uuid4 nonce makes
# ownership exact, so a leaked file can never match a token held right now.
_held_lock_tokens: dict = {}
_held_lock_tokens_guard = threading.Lock()


def _lock_is_held_by_us(lockpath: Path, held: bytes) -> bool:
    """True only when *held* is a fencing token THIS process wrote and still holds - i.e. a genuine nested acquisition, not a pid collision."""
    if not held:
        return False
    with _held_lock_tokens_guard:
        return _held_lock_tokens.get(str(lockpath)) == held


@contextlib.contextmanager
def _cross_process_lock(target: Path):
    """Hold an exclusive, cross-process lock on *target* (a sibling ``<name>.lock`` marker file) for the duration of the ``with`` block."""
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
        # We created the lock file. Everything from here must clean up our own
        # just-created file on failure, or a transient write error would leak an
        # orphaned lock file that blocks every other writer until the next
        # staleness reclaim.
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
        # Forget our claim first and unconditionally: once we leave this block we
        # no longer hold the lock, whatever happens to the file below. A stale
        # entry would make a later acquisition of the same path read a foreign
        # lock as our own nested call.
        with _held_lock_tokens_guard:
            if _held_lock_tokens.get(str(lockpath)) == token:
                del _held_lock_tokens[str(lockpath)]
        # Fencing-token release: remove the lock file only if it still holds the
        # token we wrote. If it does not, another process reclaimed it as stale
        # while we were still inside our critical section, and deleting their live
        # lock would let a third writer in.
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
    """Atomically read-modify-write the config."""
    ensure_dirs()
    with _io_lock, _cross_process_lock(CONFIG_FILE):
        cfg = copy.deepcopy(DEFAULT_CONFIG)   # deep: see load_config (nested dicts)
        stored, read_ok = _read_json_checked(CONFIG_FILE, {})
        if not read_ok:
            # The file exists and could not be read, so `stored` is {} - the same
            # value a genuinely absent config produces. Merging that leaves cfg ==
            # DEFAULT_CONFIG, and _user_delta would then write only the key this
            # mutator set, replacing every setting the user has with defaults while
            # reporting success. Names the FILE, never the path: this can surface
            # in an HTTP error body.
            raise ConfigUnreadable(
                f"{CONFIG_FILE.name} exists but could not be read, so saving "
                f"would replace every setting in it with defaults; refused. "
                f"Fix or remove {CONFIG_FILE.name} (a .bak may hold a good copy).")
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
    """Atomically read-modify-write the registry."""
    with _io_lock, _cross_process_lock(REGISTRY_FILE):
        reg, read_ok = _read_json_checked(REGISTRY_FILE, {})
        if not read_ok:
            # Same refusal as update_config. The loss is worse here: this writes
            # the whole dict rather than a delta, so an unreadable registry plus
            # one registration would leave a registry.json holding that single
            # model. The registry is not reconstructible from anything else.
            raise ConfigUnreadable(
                f"{REGISTRY_FILE.name} exists but could not be read, so saving "
                f"would drop every model registered in it; refused. Fix or "
                f"remove {REGISTRY_FILE.name} (a .bak may hold a good copy).")
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
    """Return the directory holding the native llama.cpp binaries (llama.dll, plus optional llama-cli/llama-server exes), used by `localm info` and `localm doctor`, or None when unprovisioned."""
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
