# SPDX-License-Identifier: AGPL-3.0-or-later
import copy
import json
import os
import socket
import sys
import threading
from pathlib import Path
from typing import Callable, Optional


_warned_unconfigured_home = False


def _warn_unconfigured_home(path: Path) -> None:
    """Surface (once) that no data dir was configured, so a missing / lost config
    is VISIBLE instead of silently masked (do-not-hide-problems). stderr, not the
    logger: this runs at import time before logging is wired."""
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
    # Size the context ceiling from free VRAM at model load (clamped to
    # 4k-64k). The window still starts at n_ctx and grows on demand; set
    # to false to use the fixed n_ctx_max instead.
    "ctx_auto": True,
    "n_gpu_layers": 99,    # 99 = offload everything to GPU
    # Default system prompt for chat. The GUI's per-chat System prompt field
    # OVERRIDES this when set; a blank field inherits this. Empty by default.
    "chat_system_prompt": "",
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.1,
    # Generation budget per reply. Thinking models (qwen3, deepseek-r1, …)
    # spend most of it on reasoning, so 1024 silently cut answers mid-thought;
    # the cap exists only as a runaway guard, not a cost control.
    "max_tokens": 4096,
    "confirm_remove": True,   # ask before localm rm deletes files
    # Sidebar wordmark treatment, shared by the web GUI and the desktop launcher
    # (the web logo picker writes it here; the launcher reads it). One of:
    # local-m (LocaL white + M blue, the default), loca-lm (Loca white + LM blue),
    # localm (lowercase, m blue). The console command, app icon, and desktop
    # shortcut are fixed and unaffected by this.
    "logo_style": "local-m",
    "import_max_depth": 3,    # `localm add <dir>` recurses up to this many levels
    "port": 8642,             # default inference server port (auto-bumps if busy)
    "cors_origins": None,     # None = localhost only; list of origins; or "*"
    # Require a configured API key on protected endpoints. When true the server
    # refuses requests until a key is set (see localm/auth.py); env override:
    # LOCALM_REQUIRE_AUTH. Default false = open in local/dev mode on loopback.
    "require_auth": False,
    # Quick-select scope bundles for the "Keys & devices" manager (Settings).
    # Each is {name, scopes}; the GUI offers them as one-tap presets when minting
    # a key. Re-seeded only when this key is ABSENT (an emptied list stays empty).
    # coder:full / admin in a preset only take effect when an OWNER mints the key
    # (privileged scopes stay owner-only regardless of the preset).
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
    # the image/music/video generators start ComfyUI automatically if it is
    # not running - from the GUI, the CLI, or the coder's generate_image tool.
    "comfy_launch_cmd": None,
    # Working directory for comfy_launch_cmd (launchers that assume their own
    # folder, e.g. plain "python main.py" inside a ComfyUI checkout). When left
    # blank and comfy_launch_cmd points at a launcher file, localm runs it from
    # that file's own folder - the ComfyUI / ZLUDA .bat convention.
    "comfy_workdir": None,
    # How long to wait for ComfyUI to answer after launching it, in seconds. A
    # ZLUDA / ROCm cold start compiles GPU kernels on first run and can take
    # several minutes, so the default is generous.
    "comfy_launch_timeout": 300,
    # ComfyUI's own output directory (e.g. StabilityMatrix's Images folder).
    # Only needed if you enable comfy_delete_outputs: localm uses it to find and
    # delete ComfyUI's duplicate copy. Left blank, localm derives it from the
    # ComfyUI folder when it needs it.
    "comfy_output_dir": None,
    # Whether to delete ComfyUI's OWN copy (and /history entry) of a generation
    # after localm has saved its own. Default False: KEEP them, because a user
    # may run ComfyUI for its own gallery and want the files. Privacy mode forces
    # deletion regardless (no traces). Per-plugin config can override this.
    "comfy_delete_outputs": False,
    # ComfyUI base URL localm talks to. None/blank uses the FLUX_API_URL env
    # override when set, else http://127.0.0.1:8188 (the ComfyUI default).
    "comfy_api_url": None,
    # Rewrite a slow `dequant_dtype: "float32"` in a Flux GGUF UNet loader to the
    # fast default (fp16/bf16) when the workflow is submitted. float32 dequant
    # doubles the unpacked model size and forces CPU offload on a VRAM-limited
    # card - the ~36 s/it vs ~6-7 s/it gap. True (default) auto-corrects it; set
    # False to submit your workflow's dequant choice verbatim.
    "comfy_fast_dequant": True,
    # Suppress ComfyUI opening its own web page when localm auto-launches it.
    # Off by default (keep the current behavior: ComfyUI opens its tab). When
    # True, localm appends ComfyUI's --disable-auto-launch to the launch command
    # so it starts headless (localm has its own GUI, so the ComfyUI tab is just
    # noise for most localm users). The stock run_*.bat / comfyui.* launchers and
    # a bare "python main.py" forward the flag through to main.py; a custom
    # launcher that drops extra args simply ignores it (no error), so this is
    # non-breaking. Applies to the image, music, and video plugins (they share
    # one ensure_comfy()).
    "comfy_disable_auto_launch": False,
    # Session persistence mode for ALL surfaces (chat, server, GUI, coder):
    #   privacy = no traces written automatically (default)
    #   log     = JSONL audit trail in ~/.localm/sessions/
    #   full    = log + markdown transcript
    # chat_mode / coder_mode override the global mode per surface (None =
    # inherit). CLI --mode flags override everything.
    "mode": "privacy",
    "chat_mode": None,
    "coder_mode": None,
    # Long-term chat memory: recall the user's durable facts/preferences and
    # inject them (server-side) into the system prompt each turn. Recall is free
    # (BM25 over a small structured store). Set False to stop injecting remembered
    # facts (existing memories are kept, just not used).
    "memory_enabled": True,
    # Grow the memory automatically: after a chat turn (in log/full mode, never
    # privacy), if enough time has passed since the last run, distil durable
    # facts from the recent conversation into the store IN THE BACKGROUND, so
    # memory accumulates with no manual step. Debounced to at most once per
    # MEMORY_AUTO_MIN_INTERVAL (see the chat plugin). Still skipped entirely in
    # privacy mode (no new traces). Set False to require the manual "Synthesize
    # now" button / the jobs "memory" task / POST /api/memory/consolidate.
    "memory_auto_consolidate": True,
    # Privacy mode normally disables memory ENTIRELY (no recall + no writes). Turn
    # this on to allow READING existing memories into the prompt while in privacy
    # mode (writing new memories stays off - privacy never creates a trace). Off by
    # default so privacy stays fully inert unless the user opts in. Per-surface:
    "memory_recall_in_privacy": False,
    "memory_recall_in_privacy_chat": True,      # applies only when the master is on
    "memory_recall_in_privacy_coder": True,     # applies only when the master is on
    # On-device embedding model for semantic search (RAG hybrid retrieval + agent
    # memory). A small dedicated GGUF (loaded separately from the chat model) so
    # embeddings work on the default runtime. Value is a known key
    # (bge-small-en-v1.5, nomic-embed-text-v1.5), a registered model name, or a
    # path to a GGUF. A known model is fetched into <home>/models/embeddings on
    # first use (auto only under net_mode=allow; else run 'localm setup-embeddings').
    # Until an embedding model is present, memory/RAG fall back to lexical BM25.
    "embedding_model": "bge-small-en-v1.5",
    # Which host folders the document-indexing (RAG) API may READ. A confinement
    # over a filesystem-read boundary, so all three keys are OWNER-ONLY: a
    # non-owner config:write key can neither see nor set them (enforced at PATCH
    # /v1/config; see settings_schema.admin_only). The localm data dir and
    # credential folders (.ssh, .aws, ...) stay denied in EVERY mode - a hard floor
    # no toggle turns off (rag/store.py confine_index_path). Read by
    # rag/store.py indexing_policy().
    #   whitelist (default) = index only your home folder, the working dir, and
    #                         the rag_allowed_roots below.
    #   blacklist           = index anywhere EXCEPT the rag_denied_roots below.
    "rag_indexing_mode": "whitelist",
    "rag_allowed_roots": [],   # extra folders allowed in whitelist mode
    "rag_denied_roots": [],    # folders refused in blacklist mode
    # Classify custom/unknown text file extensions with the LLM (AI). Uses the
    # active model to prompt-guess the file format (e.g. JSON, YAML, Javascript) from
    # a text snippet. Guesses are cached per extension during the index run.
    "rag_classify_unknown_files": True,
    # Seconds a GUI coder approval card may sit unanswered before it is
    # auto-rejected and the agent moves on.
    "coder_confirm_timeout": 600,
    # Episodic memory: the coder recalls lessons from past sessions on a project
    # (what worked / what failed) and, at session close, distils the finished
    # session into a new lesson. Recall is free (BM25); the reflection is one
    # extra model call per session that changed files. Writes are skipped in
    # privacy mode and for restricted (shareable-key) sessions, and stored under
    # the localm home dir, never the project tree. Set False to disable both halves.
    "coder_episodic_memory": True,
    # Provenance tagging: re-frame coder tool results from untrusted (network /
    # MCP) tools as data-not-instructions and harden their boundary, so a fetched
    # page or external server cannot inject instructions into the model loop
    # (indirect prompt injection). Defense in depth - it blocks nothing, only
    # labels. Leave ON unless you have a specific reason to drop the framing.
    "coder_untrusted_provenance": True,
    # Pre-done self-review: before the coder declares a task done, a reviewer model
    # reads the diff and feeds blocking issues back for one more fix pass. Off by
    # default (it adds a model round-trip per task that changed files).
    "coder_review": False,
    # Reviewer target: "" = the agent's own model (local, private); "local" = a
    # different small model loaded ON CPU in the coder's own process (heterogeneous
    # AND private - set coder_reviewer_model to the model name/path; adds CPU
    # load+inference latency); "openai"/"anthropic" = a cloud model; an http(s) URL =
    # a 2nd OpenAI-compatible endpoint (e.g. a second local server). A NETWORK
    # reviewer (cloud / non-loopback URL) is skipped in privacy mode and for shared
    # keys (it would send the diff off-machine) - those use the local model. The
    # "local" CPU reviewer stays on-machine, so it is allowed in privacy mode.
    "coder_reviewer": "",
    # Model name (or path) for a heterogeneous reviewer ("local"/cloud/URL); blank
    # uses a sensible provider default or the agent's own model name.
    "coder_reviewer_model": "",
    # Constrain coder tool calls with a LAZY GBNF grammar: thinking and prose
    # flow unconstrained, but once the model starts a <tool_call> the call is
    # forced to be structurally valid JSON (no more malformed tool calls to
    # repair). ON by default for grammar-capable local backends since
    # 2026-07-02 (REC-CODER-GRAMMAR; the old "runtime sampler faults" blocker
    # was our own double-accept bug). External API backends and grammar-less
    # builds are unaffected (supports_grammar gate + runtime soft-degrade).
    # NOTE: a config.json written before the flip carries the old dumped False
    # and keeps it (saved values win over this default) - flip it in Settings.
    "coder_tool_grammar": True,
    # After an image is generated, ask ComfyUI to release its VRAM and reload
    # the chat model so the next reply is instant. Turn off when generating
    # many images in a row - the chat model then reloads lazily on the next
    # chat message instead.
    "reload_llm_after_imagine": True,
    # VRAM-aware media model swap. Before an image/music/video generation the chat
    # LLM is unloaded so the media model gets the GPU; on a big card both fit, so
    # the swap is pure latency.
    #   auto   = keep chat loaded when the media model demonstrably fits alongside
    #            it (free VRAM >= estimate + headroom), else swap (default)
    #   always = always unload the chat model (the historical behaviour)
    #   never  = never unload; keep chat hot (media may run out of VRAM on a small
    #            card - an explicit choice for a big workstation card)
    # The legacy reload_llm_after_imagine flag is a SEPARATE axis: it controls
    # eager-vs-lazy reload AFTER a gen, not this unload-before decision.
    "model_swap_policy": "auto",
    # Free the loaded model from VRAM after this many seconds with no inference
    # request, so a localm server left running stops holding the GPU. The next
    # chat/completion reloads it lazily (a one-time load latency). 0 = disabled
    # (default): the model stays resident until an explicit unload or a model
    # swap. Measured from the last request, like Ollama's keep_alive.
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
    # Reach localm by NAME, not just IP, when bound to the network (see
    # localm/netname.py and docs/naming.md). On a network bind, mDNS/Bonjour
    # advertises "<mdns_name>.local" so a phone or laptop opens
    # https://localm.local:PORT with no IP to type; the name is also folded into
    # the TLS certificate, and the Tailscale MagicDNS name is detected +
    # certified automatically. Loopback binds never advertise.
    "mdns_name": "localm",      # the .local name; sanitized to a DNS label on use
    "mdns_enabled": True,       # advertise the name over mDNS on network binds
    # Speech-to-text (GUI mic button; needs the [voice] extra).
    # Model sizes: tiny / base / small / medium - bigger = better + slower.
    "voice_stt_model": "base",
    "voice_stt_language": None,  # None = auto-detect; or "en", "de", …
    # When a registered model's file has gone missing, False (default) flags the
    # entry as "missing" (kept, shown in the list); True deletes the entry.
    # Only files under the models folder are ever auto-deleted.
    "autoprune_missing_models": False,
    # When the user invokes a command that belongs to a known first-party plugin
    # that is not installed/enabled (e.g. /generate-image with the image plugin
    # off), suggest installing it ("that needs the image plugin - install it?")
    # instead of "unknown command". False silences the hint; a truly unknown
    # command always errors regardless.
    "suggest_plugins": True,
    # When a plugin that declares pip extras (requires_extras) is installed or
    # enabled by the local operator, auto-install those extras on the HOST. A
    # remote client never triggers a server-side pip regardless of this flag;
    # only the CLI or a loopback GUI request does. `localm plugin setup` asks and
    # records the choice here. Default True. Read via .get(..., True) since a
    # user config saved before this key existed will not contain it.
    "auto_install_plugin_deps": True,
    # Bug reports, the read-only Issues view, and the self-updater all talk to ONE
    # small Cloudflare Worker (the localm proxy; see tools/bugreport-proxy/) that
    # holds the GitHub tokens SERVER-SIDE. These ship as DEFAULTS so a fresh download
    # works with ZERO setup: the "Report a bug" button, the Issues view, and the
    # update check are live out of the box (update_url/token below fall back to
    # these). No GitHub token is in the app - only the public Worker URL and a
    # low-value client token. That token is intentionally PUBLIC (like a Sentry DSN),
    # NOT a secret: it just gates the endpoint against drive-by spam (Cloudflare rate
    # limiting is the real control), can ONLY file an issue through the proxy (never
    # read the repo), and is rotatable at the Worker. Set either to "" (or null) in
    # config.json to opt a build out of the hosted channel (a report then just saves
    # to a file / opens email).
    "bugreport_upload_url": "https://localm-bugreport-proxy.localm.workers.dev",
    "bugreport_upload_token": "3x_HA2UXbwNDnNfdDmpFBvvfcl2S-I-9t7XLQRAShM4",
    # Update channel + read-only issues tracker. One Worker hosts report + issues +
    # update, so these default to the bug-report proxy above; set update_url/token
    # ONLY to point updates at a different Worker. The updater needs a Contents:read
    # token on the proxy (separate from the Issues token) and the shared secret.
    # None = no update channel (the "Update available" banner + `localm update` are
    # simply hidden). See tools/bugreport-proxy/ and dev-notes/self-updater-design.
    "update_url": None,
    "update_token": None,
    # Names of enabled engine plugins (WordPress-style). Managed by the plugin
    # engine (localm plugin enable/disable and the GUI Plugins page) via
    # update_config, NOT the settings form. Declared here so it has a documented
    # home and a default - without it the settings-save endpoint rejected it as
    # an unknown key. A plugin is active only when installed (on disk) AND in
    # this list; see docs/plugins.md.
    "plugins_enabled": [],
    # Per-plugin config namespace (e.g. plugins["image"]["comfy"]["output_dir"]).
    # Written by the plugin engine and media backends via update_config, NOT the
    # flat settings form. Declared here so the settings-save endpoint does not
    # reject it as an unknown key, and so the per-plugin media-containment knob
    # is reachable through a full-config round-trip (mirrors plugins_enabled).
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


def pick_port(requested: Optional[int] = None, host: str = "127.0.0.1"):
    """
    Resolve the port to serve on.

    Returns (port, requested_port_was_busy). Tries the requested port (or the
    configured default), then walks the localm range for a free one, and as a
    last resort lets the OS assign any free port.
    """
    start = requested if requested is not None else load_config().get("port", PORT_RANGE[0])
    if not port_in_use(start, host):
        return start, False
    for candidate in range(PORT_RANGE[0], PORT_RANGE[1] + 1):
        if candidate != start and not port_in_use(candidate, host):
            return candidate, True
    return get_free_port(), True


def ensure_dirs() -> None:
    HOME_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)


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


def _atomic_write_json(path: Path, data) -> None:
    """Write *data* as JSON to *path* atomically (temp file + os.replace).

    Keeps a one-step .bak of the previous good file so a corrupt read can
    recover. os.replace is atomic on Windows and POSIX when src/dst share a
    filesystem, which they do (same directory)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        try:
            path.replace(path.with_name(path.name + ".bak"))
        except OSError:
            pass  # a missing .bak is not worth failing the write over
    os.replace(tmp, path)


def _read_json(path: Path, default):
    """Read JSON from *path*, falling back to its .bak then *default* on any
    corruption - a damaged file must never take the whole app down."""
    for candidate in (path, path.with_name(path.name + ".bak")):
        if not candidate.is_file():
            continue
        try:
            with open(candidate, encoding="utf-8") as f:
                return json.load(f)
        except (ValueError, OSError, RecursionError) as e:
            # Broadened from (JSONDecodeError, OSError): a non-UTF-8 file raises
            # UnicodeDecodeError, a huge integer raises ValueError, and a deeply
            # nested document raises RecursionError - all of which previously
            # ESCAPED and crashed the app instead of honouring the documented
            # "fall back to .bak then default, never take the app down"
            # guarantee (AUD-CFGFALLBACK). JSONDecodeError/UnicodeDecodeError are
            # ValueError subclasses, so this tuple covers them too.
            print(f"[localm] {candidate.name} is unreadable ({e}); "
                  "falling back.", file=sys.stderr)
            continue
    return default() if callable(default) else default


def load_config() -> dict:
    ensure_dirs()
    # DEEP copy: a shallow .copy() shares the nested mutable defaults (e.g. the
    # "plugins" dict) with DEFAULT_CONFIG, so a caller mutating cfg["plugins"][x]
    # (per-plugin media config, workflow selection) would silently corrupt the
    # module-level DEFAULT_CONFIG for the rest of the process.
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    with _io_lock:
        stored = _read_json(CONFIG_FILE, {})
    if isinstance(stored, dict):
        cfg.update(stored)
    return cfg


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


def update_config(mutator: Callable[[dict], None]) -> dict:
    """Atomically read-modify-write the config under the I/O lock.

    *mutator* receives the loaded config dict (defaults merged) and edits it in
    place; the result is persisted with a single atomic write. Use this instead of
    a bare load_config()/save_config() pair wherever a lost update would matter
    (e.g. two in-process writers toggling different plugins concurrently)."""
    ensure_dirs()
    with _io_lock:
        cfg = copy.deepcopy(DEFAULT_CONFIG)   # deep: see load_config (nested dicts)
        stored = _read_json(CONFIG_FILE, {})
        if isinstance(stored, dict):
            cfg.update(stored)
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
    """Atomically read-modify-write the registry under the I/O lock.

    *mutator* receives the registry dict and edits it in place; the result is
    persisted with a single atomic write. Use this instead of a bare
    load_registry()/save_registry() pair wherever a lost update would matter,
    so two in-process writers can't clobber each other. (Cross-process writers
    - e.g. a CLI `pull` running alongside the GUI - are still last-writer-wins,
    but each write stays atomic and non-corrupting.)"""
    with _io_lock:
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
