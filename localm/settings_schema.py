# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Settings schema: typed metadata for every configurable value.

Attaches per-field metadata (widget, label, help, group, allowed values, secret
flag, when-it-applies, owner) so the settings page can render the right control
- a dropdown for a fixed set of choices, free text for URLs, a folder picker for
directories, a masked input for secrets - and so each plugin can contribute its
own section.

`owner` records which surface a setting belongs to ("core" or a plugin scope).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

# Sanity ceiling for a gpu_split_indices entry: an index above it is a config
# error (typo or malicious PATCH), never a legitimate one. It bounds the ctypes
# tensor_split allocation this value eventually drives in
# discover.apply_gpu_split - uncapped, [0, 500000] would attempt a
# 500,001-element allocation before the native loader is ever invoked.
# discover.resolve_gpu_split re-applies the same ceiling at read time, since a
# hand-edited config.json bypasses this write-time check entirely.
#
# Also the ceiling for main_gpu_index: both values land in the same
# ctypes.c_int32 main_gpu field (llamacpp/_structs.py) before
# llama_load_model_from_file, and discover.resolve_main_gpu_index re-applies
# this ceiling at read time too.
MAX_GPU_SPLIT_INDEX = 127

# Sanity ceiling for cors_origins: a longer browser-origin allowlist is a config
# error (garbage input or a malformed import), not a real deployment. Without
# a cap the raw list is handed straight to starlette's CORSMiddleware
# (allow_origins) and to the membership-tested _cors_allowlist in
# localm/inference/http_server.py, both scanned on relevant requests.
MAX_CORS_ORIGINS = 500


class Widget:
    TEXT     = "text"       # free-form single line (urls, names)
    TEXTAREA = "textarea"   # free-form multi line
    NUMBER   = "number"
    TOGGLE   = "toggle"     # boolean
    SELECT   = "select"     # fixed set of choices (see .options)
    PATH     = "path"       # a file path (free text + file picker)
    FOLDER   = "folder"     # a directory (free text + folder picker)
    SECRET   = "secret"     # masked; never returned in plaintext
    LIST     = "list"       # list of strings (e.g. domains)
    PATHLIST = "pathlist"   # list of folder paths (row editor + folder picker)
    HIDDEN   = "hidden"     # a config value managed elsewhere (e.g. the Plugins
                            # page), NOT rendered as a settings control


class Applies:
    LIVE      = "live"        # takes effect immediately
    NEXT_LOAD = "next_load"   # on the next model load
    RESTART   = "restart"     # needs a server restart


_PRIVACY = ["privacy", "log", "full"]
_PRIVACY_INHERIT = ["", "privacy", "log", "full"]   # "" = inherit the global mode
# Embedding pooling choices, default first (see config.py embedding_pooling).
# Spelled out here rather than imported so this module stays free of the
# inference stack. Pinned to embedder.POOLING_CHOICES by
# test_embedding_pooling_options_match_the_embedder.
_EMBEDDING_POOLING = ["mean", "auto", "cls", "last", "none"]
# Sidebar wordmark treatments (see config.py logo_style). Shared by the web GUI
# logo picker and the desktop launcher; kept here so PATCH /v1/config validates.
LOGO_STYLE_IDS = ["local-m", "loca-lm", "localm"]
# Chat avatars (user_avatar / model_avatar_default / model_avatar_overrides, see
# config.py). A value is "", a short emoji/text glyph, or a raster data URI -
# never a URL or path. See test_avatar_value_rejects_url_or_path.
_AVATAR_DATA_URI_RE = re.compile(r"^data:image/(png|jpeg|gif|webp);base64,", re.IGNORECASE)
_AVATAR_MAX_DATA_URI_LEN = 200_000
_AVATAR_MAX_GLYPH_LEN = 16


@dataclass
class SettingField:
    key: str
    widget: str
    label: str
    help: str = ""
    group: str = "General"
    owner: str = "core"                  # "core" or a plugin scope (see scopes.py)
    options: Optional[list] = None       # for SELECT
    # For a SELECT that also accepts a boolean, as ``(false_option,
    # true_option)``. A boolean (or its 1/0 and "true"/"false" spellings) is
    # mapped to that pair, so a config.json holding `true`, a `localm config
    # <key> true`, and an API client sending JSON true are all accepted. An
    # exact option always wins first, so "on"/"off" are read as modes rather
    # than as booleans. Pairs with config.py's own read-side normalisation -
    # see REMOTE_IMAGE_LEGACY_BOOL, the single definition of the mapping.
    legacy_bool: Optional[tuple] = None
    applies: str = Applies.LIVE
    secret: bool = False
    # Owner-only: a non-ADMIN caller may neither SEE this field in the schema nor
    # WRITE it via PATCH /v1/config (it widens a trust boundary). Distinct from
    # `owner` above, which only records the plugin section a setting belongs to.
    admin_only: bool = False
    # Engine/plugin STATE, not a setting: validate_update has no schema for what
    # is INSIDE it and stores what it is given VERBATIM (the container tail of
    # the HIDDEN branch in _validate_one). Its real write surface lives elsewhere
    # and enforces a STRONGER gate, so a non-ADMIN config:write caller must not
    # reach it through the generic PATCH /v1/config. WRITE-gated only: unlike
    # admin_only the value stays readable (see engine_managed_keys).
    engine_managed: bool = False
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None

    def to_json(self) -> dict:
        d = {
            "key": self.key, "widget": self.widget, "label": self.label,
            "help": self.help, "group": self.group, "owner": self.owner,
            "applies": self.applies, "secret": self.secret,
        }
        if self.admin_only:
            d["admin_only"] = True
        if self.engine_managed:
            d["engine_managed"] = True
        if self.options is not None:
            d["options"] = self.options
        for attr in ("min", "max", "step"):
            v = getattr(self, attr)
            if v is not None:
                d[attr] = v
        return d


# Full schema for the current core config (see localm/config.py DEFAULT_CONFIG).
# Order = display order. `owner != "core"` flags keys that migrate to a plugin.
CORE_FIELDS: list = [
    # ---- Engine ----
    # admin_only: this key names WHICH NATIVE CODE THE SERVER LOADS INTO ITSELF.
    # _loader.py reads it and hands the resolved dir to ctypes.CDLL(), ahead of
    # the bundled localm-llama-runtime wheel, so a file placed there WINS. Unlike
    # comfy_launch_cmd (which spawns a subprocess) this is arbitrary native code
    # IN-PROCESS, with no subprocess boundary: compute_devices() /
    # has_max_devices() in discover.py reach it before any worker spawns.
    SettingField("binary_dir", Widget.FOLDER, "llama.cpp binary folder",
                 "Folder holding llama.dll / ggml libraries. Blank uses the "
                 "bundled runtime (auto-detected path shown); set it to point at "
                 "a custom build.",
                 group="Engine", applies=Applies.NEXT_LOAD, admin_only=True),
    # HIDDEN: both are written by `localm setup-llama` (--tag / --rollback) and
    # read back by doctor and the bug reporter. There is no settings-form widget
    # for them because setting one WITHOUT re-provisioning leaves the config and
    # the installed binaries disagreeing; the CLI moves the pin and installs in
    # one step.
    #
    # admin_only: the pin selects WHICH NATIVE BUILD is downloaded and then
    # loaded in-process. It cannot name an arbitrary path or host (the repo is
    # fixed, and the asset is checksum-verified), but it can hold an install on
    # a specific older upstream build.
    SettingField("llama_runtime_pin", Widget.HIDDEN, "Pinned llama.cpp build",
                 "Release tag setup-llama installs (e.g. 'b10355'). Blank uses "
                 "the build localm confirmed; 'latest' tracks upstream's newest, "
                 "untested. Set with `localm setup-llama --tag`, not here.",
                 group="Engine", applies=Applies.NEXT_LOAD, admin_only=True),
    # NOT engine_managed: that flag means PLUGIN STATE specifically (`plugins`,
    # `plugins_enabled`). The two flags are DISJOINT gates - admin_only also
    # hides a value from a non-owner, engine_managed only refuses the write - so
    # carrying both would narrow reads. admin_only is the gate for this key.
    SettingField("llama_runtime_history", Widget.HIDDEN, "Runtime build history",
                 "Builds provisioned on this machine, newest last. Recorded by "
                 "setup-llama so `--rollback` knows what to return to.",
                 group="Engine", admin_only=True),
    # Help text in this schema names another setting rather than placing it
    # ("below", "above"): .settings-fields is a TWO-COLUMN grid (style.css), so
    # the next field renders to the RIGHT, not below.
    # max is a sanity bound, not a capability claim: large enough for any real
    # long-context model, small enough to keep an absurd value from reaching the
    # native ctypes layer unbounded.
    SettingField("n_ctx", Widget.NUMBER, "Context window (initial)",
                 "Tokens of history the model starts with; it grows on demand up "
                 "to Context window (max).",
                 group="Engine", applies=Applies.NEXT_LOAD, min=512, max=1048576,
                 step=512),
    SettingField("n_ctx_max", Widget.NUMBER, "Context window (max)",
                 "Largest the context window may grow to (0 = unlimited); bigger "
                 "needs more VRAM.",
                 group="Engine", applies=Applies.NEXT_LOAD, min=0, max=1048576,
                 step=512),
    SettingField("n_ctx_grow", Widget.NUMBER, "Context growth step",
                 "When the window fills, it expands by this many tokens at a time.",
                 group="Engine", applies=Applies.NEXT_LOAD, min=256, max=1048576,
                 step=256),
    SettingField("ctx_auto", Widget.TOGGLE, "Auto-size context from VRAM",
                 "Pick the context ceiling from free GPU memory at load time "
                 "instead of the fixed Context window (max).",
                 group="Engine", applies=Applies.NEXT_LOAD),
    # Label carries "(next load)" because the Live tuning card in the SAME nav
    # group has its own "GPU layers (running model)" (index.html) - two controls
    # of the same name, one persisted and one immediate, and the settings search
    # box indexes both under the same words.
    SettingField("n_gpu_layers", Widget.NUMBER, "GPU layers (next load)",
                 "Model layers to run on the GPU. 99 puts the whole model on the "
                 "GPU; lower it if you run out of VRAM.",
                 group="Engine", applies=Applies.NEXT_LOAD, min=0, max=999),
    SettingField("n_cpu_moe", Widget.NUMBER, "MoE expert layers on CPU",
                 "Mixture-of-Experts models only. Keeps this many layers' expert "
                 "weights in system RAM instead of VRAM, so the model fits in far "
                 "less VRAM at about the same speed. 0 turns it off.",
                 group="Engine", applies=Applies.NEXT_LOAD, min=0, max=999),
    SettingField("n_gpu_layers_auto", Widget.TOGGLE, "Auto-size GPU layers from VRAM",
                 "When GPU layers is left at 99 (all), fit as many as free VRAM "
                 "allows at load: an oversized model runs some layers on CPU and "
                 "still loads instead of being refused. An explicit value is kept.",
                 group="Engine", applies=Applies.NEXT_LOAD),
    SettingField("mtp_enabled", Widget.TOGGLE, "Multi-Token Prediction (MTP)",
                 "Speculative drafting on models with MTP/next-n heads. Off by "
                 "default: it pays on large models and costs a little on small "
                 "ones, so try it and keep it if your model gets faster.",
                 group="Engine", applies=Applies.NEXT_LOAD),
    # VRAM reserved beyond model weights for the KV cache's compute buffers and
    # llama.cpp's graph/scratch allocations, deducted before GPU layers or
    # context are auto-sized.
    SettingField("vram_overhead_mb", Widget.NUMBER, "Reserved VRAM overhead (MB)",
                 "Lowering this risks a native crash or GPU driver hang, not just "
                 "a slower load: it is VRAM the loader genuinely needs for the KV "
                 "cache and llama.cpp's scratch buffers.",
                 group="Engine", applies=Applies.NEXT_LOAD, min=256, step=128),
    # HIDDEN: rendered by a dedicated Main GPU selector in the Live Tuning card
    # (populated from GET /api/gpus), not the generic settings form - a plain
    # number box would not show device names/VRAM. Still accepted by PATCH
    # /v1/config (and `localm config main_gpu_index N`) like any other field.
    SettingField("main_gpu_index", Widget.HIDDEN, "Main GPU",
                 "Which GPU device to load models onto, for multi-GPU systems. "
                 "Blank uses device 0.",
                 group="Engine", applies=Applies.NEXT_LOAD, min=0,
                 max=MAX_GPU_SPLIT_INDEX),
    # HIDDEN: rendered by a "Split across GPUs" checkbox row next to the Main
    # GPU selector (populated from GET /api/gpus), not the generic settings
    # form - same reasoning as main_gpu_index above. Still accepted by PATCH
    # /v1/config and `localm config gpu_split_indices 0,1` like any other field.
    # The help-length cap is enforced over the WHOLE schema, HIDDEN fields
    # included. Each card's share follows its free VRAM at load time unless GPU
    # split ratios pins exact weights.
    SettingField("gpu_split_indices", Widget.HIDDEN, "Split across GPUs",
                 "Device indices to split a model across when it is too large for "
                 "one card (2+ needed to take effect). Blank still spreads a model "
                 "over every GPU by free VRAM; set this to choose which cards.",
                 group="Engine", applies=Applies.NEXT_LOAD),
    # HIDDEN: rendered by a ratio-weight input beside each checked device in
    # the "Split across GPUs" row (settings-perf.js renderGpuSplitRatioRow),
    # not the generic settings form - same reasoning as gpu_split_indices
    # above. Still accepted by PATCH /v1/config and `localm config
    # gpu_split_ratios 3,1` like any other field.
    SettingField("gpu_split_ratios", Widget.HIDDEN, "GPU split ratios",
                 "Optional relative weight per device in Split across GPUs "
                 "(same order/length). Blank distributes by each card's free "
                 "VRAM at load time (evenly when that cannot be measured).",
                 group="Engine", applies=Applies.NEXT_LOAD),
    # HIDDEN, like the GPU knobs above: rendered by a dedicated "Max resident
    # models" number input in the Live Tuning card (settings-perf.js
    # setupResidencyControls), not the generic settings form. Still accepted
    # by PATCH /v1/config and `localm config max_resident_models 2` like any
    # other field. HIDDEN also routes it to the explicit branch in
    # _validate_one - a Widget.NUMBER field whose default is None derives
    # want_int=False and would store a cap of 2 as 2.0.
    SettingField("max_resident_models", Widget.HIDDEN, "Max resident models",
                 "How many models may stay loaded at once. Blank lets free-VRAM "
                 "arithmetic decide (a model loads alongside the others only "
                 "when it provably fits); 1 forces strict single-resident.",
                 group="Engine", applies=Applies.NEXT_LOAD, min=1),
    # HIDDEN: rendered by a "Pinned models" text input beside the one above
    # (same Live Tuning card, same setupResidencyControls), not the generic
    # settings form - a plain comma-separated field, no picker, since a pin
    # protects a model by name whether or not it happens to be resident right
    # now. Still accepted by PATCH /v1/config and `localm config pinned_models
    # a,b` like any other field.
    SettingField("pinned_models", Widget.HIDDEN, "Pinned models",
                 "Model names that are never evicted to make room for another. "
                 "Pinning only protects an already-loaded model; it never loads "
                 "one. Blank pins nothing.",
                 group="Engine", applies=Applies.NEXT_LOAD),
    SettingField("idle_unload_seconds", Widget.NUMBER, "Idle model unload (s)",
                 "Free the model's VRAM after this many seconds with no request "
                 "(0 = never; the model stays resident). The next message reloads "
                 "it automatically.",
                 group="Engine", min=0, step=30),
    # The five timeout fields below share one sentence shape, so the panel reads
    # as one family and a difference between two of them stays visible.
    SettingField("gguf_load_timeout_s", Widget.NUMBER, "GGUF load timeout (s)",
                 "How long a GGUF model load may run in its isolated worker "
                 "before it is treated as hung and cancelled. Raise it only for a "
                 "huge model on slow storage.",
                 group="Timeouts", min=10, step=60),
    # The label names the format, like the other load/first-token timeouts: a
    # bare "First-token timeout" reads on screen as the global one. The timer
    # covers reading the whole prompt rather than emitting one token, so on CPU,
    # with most layers off the GPU, or with a very long prompt, a legitimate
    # first token can take minutes.
    SettingField("gguf_first_token_timeout_s", Widget.NUMBER,
                 "GGUF first-token timeout (s)",
                 "How long a reply may take to produce its first token before the "
                 "model is treated as hung. It covers reading your whole prompt, "
                 "so on CPU or with a long prompt this can legitimately take "
                 "minutes.",
                 group="Timeouts", min=10, step=60),
    SettingField("hf_load_timeout_s", Widget.NUMBER,
                 "HuggingFace load timeout (s)",
                 "How long a HuggingFace-format model load may run in its isolated "
                 "worker before it is treated as hung and cancelled. Raise it only "
                 "for a huge model on slow storage.",
                 group="Timeouts", min=10, step=60),
    SettingField("hf_first_token_timeout_s", Widget.NUMBER,
                 "HuggingFace first-token timeout (s)",
                 "How long a HuggingFace-format model's reply may take to produce "
                 "its first token before it is treated as hung. It covers reading "
                 "your whole prompt, so it is generous by default.",
                 group="Timeouts", min=10, step=60),
    SettingField("hf_embed_timeout_s", Widget.NUMBER,
                 "HuggingFace embed timeout (s)",
                 "How long a HuggingFace-format model's embedding request may run "
                 "in its isolated worker before it is treated as hung. Raise it if "
                 "you regularly embed large batches.",
                 group="Timeouts", min=10, step=60),
    # The cap is low because a HuggingFace embed runs one text at a time with no
    # batching, so request cost is linear in the batch size.
    SettingField("hf_embed_max_texts", Widget.NUMBER,
                 "HuggingFace embed batch cap (texts)",
                 "Most texts accepted in one /v1/embeddings request against a "
                 "HuggingFace-format model. These embed one text at a time with no "
                 "batching, so a large request is slow.",
                 group="Timeouts", min=1, step=1),
    SettingField("hf_embed_max_chars", Widget.NUMBER,
                 "HuggingFace embed batch cap (characters)",
                 "Maximum total characters, summed across every text, "
                 "accepted in one /v1/embeddings request against a "
                 "HuggingFace-format model.",
                 group="Timeouts", min=1, step=1000),
    SettingField("import_max_depth", Widget.NUMBER, "Folder import depth",
                 "Subfolder levels `localm add <dir>` scans for models.",
                 group="Models", min=1, max=10, step=1),
    # ---- Chat (the DEFAULTS every chat starts from) ----
    # The GUI's per-chat "parameters" drawer OVERRIDES any of these for a single
    # conversation; a blank field there inherits the value here. So this section is
    # "set your chat defaults", the drawer is "fine-tune this one chat".
    SettingField("chat_system_prompt", Widget.TEXTAREA, "Default system prompt",
                 "The system prompt every new chat starts with. A chat's own System "
                 "prompt field overrides this per conversation; leave that blank to "
                 "use this. Empty = no default system prompt.",
                 group="Chat", owner="chat"),
    SettingField("temperature", Widget.NUMBER, "Temperature",
                 "Randomness of replies. Lower is more focused; higher is more "
                 "varied and creative.",
                 group="Chat", owner="chat", min=0, max=2, step=0.05),
    SettingField("top_p", Widget.NUMBER, "Top-p (nucleus sampling)",
                 "Consider only the top tokens whose probabilities sum to this "
                 "fraction. 1.0 turns it off.",
                 group="Chat", owner="chat", min=0, max=1, step=0.05),
    SettingField("top_k", Widget.NUMBER, "Top-k",
                 "Consider only the k most likely tokens at each step. 0 turns it off.",
                 group="Chat", owner="chat", min=0, step=1),
    SettingField("repeat_penalty", Widget.NUMBER, "Repeat penalty",
                 "How strongly to discourage reusing tokens. 1.0 is no penalty; "
                 "higher reduces loops.",
                 group="Chat", owner="chat", min=0, step=0.05),
    SettingField("max_tokens", Widget.NUMBER, "Max tokens per reply",
                 "Upper limit on tokens per reply, a runaway guard not a target. "
                 "Thinking models need plenty of room.",
                 group="Chat", owner="chat", min=1, step=1),
    SettingField("chat_tool_grammar", Widget.TOGGLE,
                 "Grammar-constrain chat tool calls",
                 "Once the model starts a <tool_call> (the web search/fetch tool), "
                 "force it to be valid tool-call JSON (lazy GBNF grammar; local "
                 "grammar-capable backends only). Free text and thinking are "
                 "unaffected.",
                 group="Chat", owner="chat", applies=Applies.LIVE),
    # HIDDEN: the Settings > Chat "Avatars" panel renders these with its own
    # upload/emoji picker, not a generic text box - see settings.js.
    SettingField("user_avatar", Widget.HIDDEN, "Your icon",
                 "An emoji or a small uploaded image shown next to your own "
                 "messages. Blank shows nothing.",
                 group="Chat", owner="chat", applies=Applies.LIVE),
    SettingField("model_avatar_default", Widget.HIDDEN, "Model icon",
                 "An emoji or a small uploaded image shown next to every "
                 "model's replies. Blank falls back to a generated monogram.",
                 group="Chat", owner="chat", applies=Applies.LIVE),
    SettingField("model_avatar_overrides", Widget.HIDDEN, "Per-model icons",
                 "Override the model icon for one specific model. Falls back "
                 "to the Model icon default, then the monogram, when unset.",
                 group="Chat", owner="chat", applies=Applies.LIVE),
    # ---- Server ----
    SettingField("port", Widget.NUMBER, "Server port",
                 "Port the API/GUI server binds to (default 8642); auto-bumps to "
                 "the next free port if busy.",
                 group="Server", applies=Applies.RESTART, min=1, max=65535, step=1),
    # admin_only: this decides WHICH NETWORK can reach the server at all - the
    # widest reach-widening key in the Server group, same trust boundary class
    # as cors_origins (browser origins) and net_allow_private (outbound reach).
    # A non-owner config:write key must not be able to expose the server to the
    # LAN. Note what this key CANNOT do: bind past loopback without a strong API
    # key. The startup guard (plugins/gui/cli.py) ignores a config-driven
    # network bind when no strong key is set and stays on loopback, loudly -
    # only the CLI's --insecure flag, which has NO config form, can override
    # that, so an unauthenticated network bind always requires a terminal.
    # Applies.RESTART: an in-place restart re-execs the same argv,
    # and the fresh process re-reads this key (cli._resolve_bind_host) when no
    # explicit -H was typed - that read is what makes the Settings > Restart
    # server flow work for a browser-only user.
    SettingField("bind_host", Widget.TEXT, "Bind address",
                 "Which interface the server binds to. Blank = this computer "
                 "only (127.0.0.1). 0.0.0.0 = every interface, so phones on "
                 "your network reach it; set an API key first or it stays on "
                 "loopback.",
                 group="Server", applies=Applies.RESTART, admin_only=True),
    # admin_only, all three: turning TLS off sends the API key over the network
    # in cleartext, and the cert/key pair decides what the server presents to
    # every client - transport-trust decisions are the owner's, never a
    # delegated config:write key's. CLI flags (--no-tls / --tls-cert/--tls-key)
    # win over all three for that process; see cli._resolve_tls.
    SettingField("tls_enabled", Widget.TOGGLE, "Encrypt network traffic (TLS)",
                 "Serve HTTPS on a network bind (built-in certificate, or a "
                 "custom pair). Off = plain HTTP, so the API key crosses the "
                 "network readable by anyone on it. Loopback is always plain "
                 "HTTP.",
                 group="Server", applies=Applies.RESTART, admin_only=True),
    SettingField("tls_cert", Widget.PATH, "Custom TLS certificate (PEM)",
                 "Use this certificate instead of localm's built-in one; blank "
                 "= built-in. Needs the matching private key. If the pair "
                 "cannot be loaded, localm warns and falls back to its "
                 "built-in certificate.",
                 group="Server", applies=Applies.RESTART, admin_only=True),
    SettingField("tls_key", Widget.PATH, "Custom TLS private key (PEM)",
                 "Private key for the custom TLS certificate.",
                 group="Server", applies=Applies.RESTART, admin_only=True),
    # admin_only: this names WHICH BROWSER ORIGINS may call the authenticated
    # API - it widens a trust boundary exactly like net_allow_private (network
    # reach) and the rag_* keys (filesystem reach) do. "*" additionally opts the
    # sensitive GETs in _CROSS_ORIGIN_GET_REFUSED (/whoami, /debug/stacks - see
    # http_server.py) out of their cross-origin refusal, so a non-owner
    # config:write key setting this could disclose root_dir (the OS username) to
    # any website. /whoami is unauthenticated; /debug/stacks additionally
    # requires the shell token in open mode, so for it this waives only the
    # cross-origin half. See routes/config.py admin_only gate.
    SettingField("cors_origins", Widget.TEXT, "CORS origins",
                 "Browser origins allowed to call the API. Blank = localhost "
                 'only; comma-separated list; or "*" for any.',
                 group="Server", applies=Applies.RESTART, admin_only=True),
    # A rename needs a restart: the name is also written into the HTTPS
    # certificate's SANs, so it is not display-only.
    SettingField("mdns_name", Widget.TEXT, "Network name (mDNS)",
                 "The name other devices use to reach this server on your LAN: "
                 "<name>.local, so there is no IP to type. Letters, digits and "
                 "hyphens only.",
                 group="Server", applies=Applies.RESTART),
    SettingField("mdns_enabled", Widget.TOGGLE, "Advertise on the network (mDNS)",
                 "Broadcast <name>.local over mDNS/Bonjour when bound past loopback "
                 "so devices can reach localm by name. Off = reachable by IP "
                 "address only. Loopback binds never advertise.",
                 group="Server", applies=Applies.RESTART),
    # ---- Security ----
    # admin_only: defense in depth on a LATENT fail-closed control, not an active
    # bypass. Both readers sit behind an any_key_configured() check in
    # http_server.py, so while any key exists the flag is INERT and clearing it
    # cannot disable auth. What the gate stops is deferred: an owner who set it
    # true, whose non-owner key cleared it, and who later removes every key,
    # would otherwise get an OPEN server instead of the fail-closed one they
    # asked for. The effective state stays readable without config:read via GET
    # /api/session, and the CLI/launcher write config directly rather than
    # through this route.
    SettingField("require_auth", Widget.TOGGLE, "Require an API key",
                 "Refuse all requests until an API key is set (fail closed). "
                 "Required before exposing localm on a network.",
                 group="Security", applies=Applies.RESTART, admin_only=True),
    # admin_only: turning this ON lets a downloaded model directory run its own
    # bundled Python inside the localm process, which is arbitrary code execution
    # as the server user. Only an owner may make that call, never a config:write
    # key. See routes/config.py admin_only gate and hf.py's refusal path. The
    # mechanism is the model directory's 'auto_map' custom code; a model needing
    # it is refused WITH AN EXPLANATION rather than failing opaquely.
    SettingField("hf_trust_remote_code", Widget.TOGGLE,
                 "Allow model-bundled custom code",
                 "Arbitrary code execution on this machine: lets a HuggingFace "
                 "model directory run its own Python when loading. Off by "
                 "default; a model needing it is refused. Turn on only for a "
                 "source you trust.",
                 group="Engine", applies=Applies.NEXT_LOAD, admin_only=True),
    # admin_only: matches rag_allowed_roots/rag_denied_roots (same "which host
    # locations may localm touch" trust boundary). ON by default, so a mapped
    # network drive keeps working exactly like a local folder; turning it off is
    # opt-in extra confinement. See config.py's DEFAULT_CONFIG comment for
    # allow_network_drives: pathsafe.is_unc_or_device_path does not flag a mapped
    # drive letter.
    SettingField("allow_network_drives", Widget.TOGGLE,
                 "Allow network drives as filesystem locations",
                 "Let the folder picker, RAG indexing, and related routes "
                 "treat a mapped network drive (e.g. Z:\\) as a normal local "
                 "folder. Off refuses them, for keeping localm confined to "
                 "local disks.",
                 group="Security", admin_only=True),
    # ---- Interface ----
    # HIDDEN: chosen with the logo picker in the GUI (Settings -> GUI), not a
    # form control. Accepted by PATCH /v1/config so the launcher stays in sync.
    SettingField("logo_style", Widget.HIDDEN, "Logo style",
                 "Sidebar wordmark, chosen with the logo picker and shared with "
                 "the desktop launcher.",
                 group="General"),
    # OFF means the project list is NOT WRITTEN, never "written but hidden" -
    # see plugins/coder/projects.py, whose privacy-mode refusal is NOT covered
    # by this setting and is not configurable at all.
    SettingField("coder_remember_projects", Widget.TOGGLE,
                 "Remember projects you have coded in",
                 "Keeps a list of project folders so past sessions are easy to "
                 "reach. Privacy-mode sessions are never listed.",
                 group="Coder"),
    # Capped: an unbounded list is both a scrolling wall and a growing
    # disclosure surface.
    SettingField("coder_projects_remembered", Widget.NUMBER,
                 "Projects to remember",
                 "How many recent project folders to keep in that list.",
                 group="Coder", min=0, step=5),
    SettingField("coder_rail_side", Widget.SELECT, "Coder session list side",
                 "Which side the coder's session list sits on. right "
                 "(default) or left.",
                 group="Coder", options=["right", "left"]),
    SettingField("desktop_window_mode", Widget.SELECT, "Default window mode",
                 "How `localm gui` opens when the standalone-window extra "
                 "(localm[desktop]) is installed. auto (default): the app "
                 "window when available, else a browser tab. browser: always "
                 "a browser tab.",
                 group="Desktop", options=["auto", "browser"]),
    SettingField("desktop_window_quit_on_close", Widget.TOGGLE,
                 "Quit when the app window is closed",
                 "Only for the standalone app window (localm[desktop]). Off "
                 "(default): closing hides the window and the server keeps "
                 "running; use Stop to quit. On: closing quits the app and "
                 "stops the server.",
                 group="Desktop"),
    # ---- Privacy ----
    # admin_only, all three: they decide WHETHER LOCALM WRITES THE USER'S
    # CONTENT TO DISK, so only an owner may turn privacy off and start
    # persisting transcripts and an audit trail. Gating the READ costs the GUI
    # nothing: it reads the injected cfg["effective_mode"] /
    # cfg["effective_coder_mode"], which routes/config.py adds AFTER the
    # admin_only strip, never cfg["mode"].
    SettingField("mode", Widget.SELECT, "Session persistence",
                 "What localm saves: privacy = nothing written automatically; "
                 "log = a JSONL audit trail; full = log plus a chat transcript.",
                 group="Privacy", options=_PRIVACY, admin_only=True),
    # Both children name the parent setting rather than pointing at a position:
    # in the two-column grid the parent sits to the LEFT, not above.
    SettingField("chat_mode", Widget.SELECT, "Chat persistence override",
                 "Overrides Session persistence for chat only. Blank inherits "
                 "whatever Session persistence is set to.",
                 group="Privacy", owner="chat", options=_PRIVACY_INHERIT,
                 admin_only=True),
    SettingField("coder_mode", Widget.SELECT, "Coder persistence override",
                 "Overrides Session persistence for the coder only. Blank inherits "
                 "whatever Session persistence is set to.",
                 group="Privacy", owner="coder", options=_PRIVACY_INHERIT,
                 admin_only=True),
    SettingField("memory_enabled", Widget.TOGGLE, "Memory recall",
                 "Recall the durable facts localm has learned about you and add "
                 "them to the system prompt each chat turn. Off = keep the facts "
                 "but stop recalling them.",
                 group="Memory", owner="memory"),
    # No privacy exception here: consolidation WRITES, and privacy mode's
    # contract is that nothing is written automatically, so unlike recall it can
    # never be opted back in.
    SettingField("memory_auto_consolidate", Widget.TOGGLE, "Grow memory automatically",
                 "After a chat turn, quietly distil durable facts from the "
                 "conversation into memory, so it grows with no manual step. Always "
                 "blocked in privacy mode. Off, memory grows only on demand.",
                 group="Memory", owner="memory"),
    SettingField("memory_recall_in_privacy", Widget.TOGGLE,
                 "Allow memory recall in privacy mode",
                 "Privacy mode normally turns memory off entirely. Turn this on to "
                 "still READ existing memories in privacy mode; writing new ones "
                 "stays off, so no new trace is created.",
                 group="Memory", owner="memory"),
    # These two labels must read on their own: a settings SEARCH match renders
    # outside its section, with no parent label to complete the sentence.
    SettingField("memory_recall_in_privacy_chat", Widget.TOGGLE,
                 "Privacy-mode recall: chat",
                 "With Allow memory recall in privacy mode on, recall chat memory "
                 "during privacy-mode chats (read-only).",
                 group="Memory", owner="memory"),
    SettingField("memory_recall_in_privacy_coder", Widget.TOGGLE,
                 "Privacy-mode recall: coder",
                 "With Allow memory recall in privacy mode on, recall the coder's "
                 "past-session lessons during privacy-mode coder sessions "
                 "(read-only).",
                 group="Memory", owner="memory"),
    # Privacy mode's contract is that nothing is written automatically, and the
    # crash/hang diagnostics a bug report needs are part of what it suppresses.
    # This toggle is the opt-in that writes them anyway.
    SettingField("keep_diagnostics", Widget.TOGGLE,
                 "Keep diagnostics for bug reports",
                 "Writes crash and hang diagnostics (stack traces, restart "
                 "breadcrumbs, a debug log) even in privacy mode, so a freeze "
                 "leaves something to attach. Code and operational logs only, "
                 "never chat content.",
                 group="Diagnostics", admin_only=True),
    # ---- Models ----
    # A small on-device model, loaded separately from the chat model. The known
    # keys are bge-small-en-v1.5 and nomic-embed-text-v1.5.
    SettingField("embedding_model", Widget.TEXT, "Embedding model",
                 "Until this is fetched, memory and RAG fall back to lexical "
                 "search. Run 'localm setup-embeddings' to get one. Accepts a "
                 "known key (bge-small-en-v1.5), a registered name, or a GGUF "
                 "path.",
                 # Owner-only because branch 1 of resolve_embedding_model_path()
                 # returns a CALLER-CHOSEN filesystem path with no confined_*
                 # check, and that path is then handed to llama.cpp's native GGUF
                 # parser; on Windows a UNC path also makes the is_file() probe an
                 # outbound SMB/NTLM authentication. The download branch resolves
                 # repo+filename from the fixed KNOWN_EMBEDDING_MODELS dict and
                 # returns None otherwise.
                 # Gating PATCH does NOT break ordinary embedder selection: the CLI
                 # (setup-embeddings) and the RAG picker (POST /api/rag/embedding,
                 # gated by the rag plugin scope, not config:write) both bypass this
                 # route entirely.
                 # This flag closes the config:write path ONLY. The plugin route
                 # POST /api/rag/embedding writes the SAME key under the
                 # non-privileged `rag` scope (rag/plug.py, rag_embedding_set, via
                 # update_config) and carries its own owner check; without that
                 # check, this flag alone does not close the rag path. The re-embed
                 # endpoint only READS this value to label a job.
                 group="Embeddings", applies=Applies.NEXT_LOAD, admin_only=True),
    # Changing this makes existing collections and memory vectors stop matching.
    # mean suits bge/nomic and matches everything already indexed; a
    # decoder-based embedder (Qwen3-Embedding, gte-Qwen2) is trained for
    # last-token pooling and is DEGRADED by mean - last for those, or auto to
    # follow whatever the model declares.
    SettingField("embedding_pooling", Widget.SELECT, "Embedding pooling",
                 "Changing this invalidates existing document collections and "
                 "memory vectors: re-index afterwards. mean suits bge/nomic; a "
                 "decoder-based embedder needs last, or auto to follow the model.",
                 group="Embeddings", applies=Applies.NEXT_LOAD,
                 options=_EMBEDDING_POOLING),
    # Blank is not simply "use the GPU": a large embedder sharing one card with a
    # loaded chat model oversubscribes VRAM and slows the chat model down, so
    # automatic falls back to CPU rather than competing.
    SettingField("embedding_gpu_layers", Widget.NUMBER, "Embedder GPU layers",
                 "GPU layers for the embedding model. Empty = automatic: full GPU "
                 "offload when free VRAM holds it, otherwise CPU. 0 forces CPU; 99 "
                 "forces full offload regardless of free VRAM.",
                 group="Embeddings", applies=Applies.NEXT_LOAD,
                 min=0, max=999, step=1),
    SettingField("confirm_remove", Widget.TOGGLE,
                 "Confirm before deleting models",
                 "Ask for confirmation before `localm rm` deletes a model's "
                 "files on disk.",
                 group="Models"),
    SettingField("autoprune_missing_models", Widget.TOGGLE,
                 "Auto-remove entries for missing files",
                 "When a registered model's file has gone, delete its registry "
                 "entry automatically instead of flagging it as missing.",
                 group="Models"),
    # ---- Plugins ----
    SettingField("suggest_plugins", Widget.TOGGLE,
                 "Suggest installing a plugin for its command",
                 "When a command belongs to a known but inactive plugin, suggest "
                 "installing it instead of reporting an unknown command.",
                 group="Plugins"),
    # Only the LOCAL operator can trigger this - a remote client never causes a
    # server-side install (the install path is gated on the local principal).
    SettingField("auto_install_plugin_deps", Widget.TOGGLE,
                 "Auto-install plugin dependencies",
                 "When you install or enable a plugin that needs extra Python "
                 "packages, install them automatically on this machine.",
                 group="Plugins"),
    SettingField("plugins_enabled", Widget.HIDDEN, "Enabled plugins",
                 "Names of enabled engine plugins. Managed by the Plugins page "
                 "and `localm plugin enable/disable`, not edited here.",
                 group="Plugins", engine_managed=True),
    # ---- Bug-report upload (deployment config) ----
    # HIDDEN: the maintainer sets these in config.json when preparing a tester
    # build (see tools/bugreport-proxy/). Not rendered in the form, so a tester on
    # a shared build does not see or change the proxy URL / shared secret.
    #
    # admin_only, all four: these name an OUTBOUND network target, so they widen
    # trust reach exactly like net_allow_private does. bugreport_upload_url is
    # where "Send to maintainer" POSTs the report (diagnostics plus whatever the
    # user typed) and it ships with a real default, so re-pointing it redirects a
    # live channel, not a dormant one; update_url is the updater's base and falls
    # back to it. HIDDEN is not itself a gate - PATCH /v1/config stores these
    # verbatim (no coercion branch, so _validate_one's tail returns them as given),
    # which without this flag let a non-owner config:write key set them.
    SettingField("bugreport_upload_url", Widget.HIDDEN, "Bug-report upload URL",
                 "Endpoint the in-app 'Send to maintainer' button POSTs the report "
                 "to (the bug-report proxy). Blank = no upload channel.",
                 group="Bug reports", admin_only=True),
    SettingField("bugreport_upload_token", Widget.HIDDEN, "Bug-report upload token",
                 "Optional shared secret the proxy may require, sent as a header. "
                 "Set in config.json, not here.",
                 group="Bug reports", admin_only=True),
    SettingField("update_url", Widget.HIDDEN, "Update endpoint URL",
                 "Override for the updater's Worker base (defaults to the bug-report "
                 "URL when blank). Set in config.json, not here.",
                 group="Updates", admin_only=True),
    SettingField("update_token", Widget.HIDDEN, "Update endpoint token",
                 "Override shared secret for the updater (defaults to the bug-report "
                 "token when blank). Set in config.json, not here.",
                 group="Updates", admin_only=True),
    # admin_only: this decides WHICH BUILDS the updater will suggest installing on
    # this machine - the same "widens trust reach" reasoning as update_url/
    # update_token/bugreport_upload_url right above (all admin_only in this same
    # group). A prerelease is signed and anti-rollback-checked exactly like a
    # stable release, so this is not about authenticity - it is about a non-admin
    # config:write caller being able to make the GUI start suggesting
    # release-candidate builds to whoever actually clicks "Update now". The
    # default MUST be False: this is an OPT-IN channel.
    SettingField("update_allow_prerelease", Widget.TOGGLE,
                 "Offer prerelease updates",
                 "Also offer release-candidate builds when checking for updates. "
                 "A prerelease is signed and verified like a stable release but is "
                 "less field-tested. Turn on only to help test rc builds.",
                 group="Updates", admin_only=True),
    # admin_only: this EXEMPTS the update channel from net_mode - the same
    # "widens network reach" reasoning as update_url/update_token/
    # bugreport_upload_url and update_allow_prerelease right above (all
    # admin_only in this same group). Default MUST be False: net_mode=off is
    # meant to be a real kill switch (netpolicy.py's module docstring - explicit
    # user actions "still respect net_mode = off"), so the update channel obeys
    # it like every other network capability unless an admin opts it out here.
    SettingField("update_ignore_net_policy", Widget.TOGGLE,
                 "Check for updates even when network access is off",
                 # Network access sits in a different NAV GROUP (Server &
                 # network), so the help names it rather than placing it.
                 "The update check normally obeys Network access, so setting that "
                 "to Off also turns off update checks. Turn this on to let the "
                 "update channel through regardless.",
                 group="Updates", admin_only=True),
    SettingField("plugins", Widget.HIDDEN, "Per-plugin config",
                 "Per-plugin settings (e.g. media output dirs). Managed by the "
                 "Plugins/Settings pages and plugin backends, not edited here.",
                 group="Plugins", engine_managed=True),
    SettingField("key_presets", Widget.HIDDEN, "Key presets",
                 "Quick-select scope bundles for the Keys & devices manager. "
                 "Edited there, not in this form.",
                 group="Security"),
    # ---- Coder (plugin) ----
    SettingField("coder_confirm_timeout", Widget.NUMBER,
                 "Coder approval timeout (s)",
                 "Seconds a coder approval card waits for an answer before it is "
                 "auto-rejected and the agent moves on (0 = wait forever).",
                 group="Coder", owner="coder", min=0, step=10),
    SettingField("coder_index_timeout", Widget.NUMBER,
                 "Coder project-scan timeout (s)",
                 "Wall-clock cap on the coder's startup project-map scan (0 = no "
                 "limit; scan to completion however long it takes). Raise this on "
                 "a very large repo if the map is being cut off.",
                 group="Coder", owner="coder", min=0, step=5),
    SettingField("coder_grep_max_per_file", Widget.NUMBER,
                 "Coder grep matches per file",
                 "How many matches the coder's grep shows per file before it "
                 "summarises the rest (0 = show all). Matches beyond the cap are "
                 "still counted and reported, never hidden.",
                 group="Coder", owner="coder", min=0, step=5),
    SettingField("coder_grep_max_output_lines", Widget.NUMBER,
                 "Coder grep output lines",
                 "How many output lines the coder's grep produces before it stops "
                 "and reports how many files it did not reach (0 = no cap). Raise "
                 "it for wide sweeps, lower it to spend less context on search.",
                 group="Coder", owner="coder", min=0, step=50),
    SettingField("coder_grep_max_file_bytes", Widget.NUMBER,
                 "Coder grep file size cap (bytes)",
                 "Files larger than this are skipped by the coder's grep rather "
                 "than read (0 = no cap); the skip is always reported. Keeps a "
                 "multi-MB log or data dump from swamping a code search.",
                 group="Coder", owner="coder", min=0, step=1048576),
    SettingField("coder_tool_grammar", Widget.TOGGLE,
                 "Grammar-constrain coder tool calls",
                 "Once the model starts a <tool_call>, force it to be valid "
                 "tool-call JSON (lazy GBNF grammar; local grammar-capable "
                 "backends only). Free text and thinking are unaffected.",
                 group="Coder", owner="coder", applies=Applies.NEXT_LOAD),
    # Recall is free; the write half is skipped in privacy mode and for shared
    # keys. Episodes are stored under the localm data dir, never inside the
    # project, and are managed with `localcoder --episodes` /
    # `--forget-episodes`.
    SettingField("coder_episodic_memory", Widget.TOGGLE,
                 "Coder episodic memory",
                 "Recall lessons from past sessions on a project and, at session "
                 "close, distil the finished session into a new one. Costs one "
                 "extra model call per session that changed files.",
                 group="Coder", owner="coder", applies=Applies.NEXT_LOAD),
    # admin_only: an INJECTION-HARDENING control, so a non-owner turning it off
    # would re-open the indirect-prompt-injection boundary for the OWNER's coder.
    # The mechanism is marking tool results as DATA rather than instructions; it
    # is defense in depth and blocks no legitimate use.
    SettingField("coder_untrusted_provenance", Widget.TOGGLE,
                 "Tag untrusted external content",
                 "Marks coder tool results from the web, web search and external "
                 "MCP servers as untrusted data, so a fetched page cannot inject "
                 "commands into the agent. Leave on unless you have a specific "
                 "reason.",
                 group="Coder", owner="coder", applies=Applies.NEXT_LOAD,
                 admin_only=True),
    SettingField("coder_review", Widget.TOGGLE,
                 "Review changes before finishing",
                 "Before the coder declares a task done, a reviewer model reads the "
                 "diff and feeds blocking issues back for one more fix pass. Adds a "
                 "model round-trip per task that changed files. Off by default.",
                 # NOT admin_only, unlike the coder_reviewer* fields: this is the
                 # on/off switch for an extra review pass and crosses no trust
                 # boundary. coder_reviewer / coder_reviewer_model ARE owner-only,
                 # because they choose WHICH backend or model file reviews (a URL
                 # or a path).
                 group="Coder", owner="coder", applies=Applies.NEXT_LOAD),
    # 'local' is heterogeneous and private but adds CPU latency (the model is
    # set via coder_reviewer_model); an http(s) URL is a second
    # OpenAI-compatible endpoint, e.g. a second local server. A network reviewer
    # is skipped in privacy mode and for shared keys, which review with the
    # local model instead.
    SettingField("coder_reviewer", Widget.TEXT,
                 "Reviewer model target",
                 "Sends your diff off-machine unless left blank or set to "
                 "'local'. Values: blank = the agent's own model; local = a "
                 "small CPU model; openai; anthropic; or an http(s) URL.",
                 group="Coder", owner="coder", applies=Applies.NEXT_LOAD,
                 admin_only=True),
    # Gated together with coder_reviewer: gating the SELECTOR while leaving its
    # ARGUMENT writable would be a half-fix. registry.py falls through to
    # Path(name), accepting any existing GGUF / HF dir, so this is not
    # registry-only - it names a file the coder process loads into a llama.cpp
    # backend.
    SettingField("coder_reviewer_model", Widget.TEXT,
                 "Reviewer model name",
                 "Model name for a cloud/URL reviewer. Blank uses a sensible provider "
                 "default or the agent's own model name.",
                 group="Coder", owner="coder", applies=Applies.NEXT_LOAD,
                 admin_only=True),
    # ---- Knowledge (RAG plugin) ----
    # All three are OWNER-ONLY: they define which host folders document indexing
    # may read (a filesystem-read boundary). The localm data folder and credential
    # folders (.ssh, .aws, ...) are refused in BOTH modes regardless - a hard floor.
    SettingField("rag_indexing_mode", Widget.SELECT, "Indexing folder rule",
                 "Which folders document indexing may read. whitelist = home, "
                 "the working directory and Allowed folders only. blacklist = "
                 "anything except Denied folders. Data and credential folders "
                 "are always refused.",
                 group="Knowledge", owner="rag", admin_only=True,
                 options=["whitelist", "blacklist"]),
    SettingField("rag_allowed_roots", Widget.PATHLIST, "Allowed folders",
                 "Folders that may be indexed in whitelist mode, in addition to "
                 "your home folder and the working directory. When you pick a "
                 "folder outside this list, localm offers to add it here and "
                 "continue.",
                 group="Knowledge", owner="rag", admin_only=True),
    SettingField("rag_denied_roots", Widget.PATHLIST, "Denied folders",
                 "Folders that are never indexed in blacklist mode (everything "
                 "else is allowed). Ignored in whitelist mode.",
                 group="Knowledge", owner="rag", admin_only=True),
    # Narrow by construction: format tagging from the extension and content is
    # already free, so the LLM is asked only about an unknown extension it cannot
    # tell structurally, and only while a chat model happens to be loaded.
    SettingField("rag_classify_unknown_files", Widget.TOGGLE, "Classify unknown files with AI",
                 "Let the local LLM guess the format of a file whose extension it "
                 "does not recognise, when a chat model is loaded. Off, those files "
                 "are tagged plain text.",
                 group="Knowledge", owner="rag"),
    # ---- Media (ComfyUI: image / music / video plugins) ----
    # These three are the CORE twins of the admin_only MediaFields (see
    # MEDIA_PLUGIN_FIELDS below). The write gate reads CORE_FIELDS only -
    # admin_only_keys() is built from this list - so they must carry admin_only
    # here; gating the mirror alone leaves PATCH /v1/config as a back door around
    # the stronger per-plugin gate in routes/config.py.
    #
    # workdir is not merely a folder: it selects where the launcher is
    # AUTO-DISCOVERED (comfy_client.discover_launch_cmd -> shlex.split -> Popen)
    # AND what the model scanner walks into registry.json, and POST
    # /api/comfy/setup (CONFIG_WRITE-only) executes
    # <comfy_workdir>/venv/Scripts/python.exe. So a blank launch_cmd plus an
    # attacker-chosen workdir still reaches execution.
    SettingField("comfy_workdir", Widget.FOLDER, "ComfyUI folder",
                 "Your ComfyUI install folder. localm runs it from here and "
                 "auto-detects a launcher inside. The one setting most setups need.",
                 group="Media", owner="image", admin_only=True),
    SettingField("comfy_launch_cmd", Widget.TEXT, "ComfyUI launch command",
                 "Launcher script (.bat/.sh) that starts ComfyUI. Blank "
                 "auto-detects one inside the ComfyUI folder.",
                 group="Media", owner="image", admin_only=True),
    SettingField("comfy_api_url", Widget.TEXT, "ComfyUI API URL",
                 "Where ComfyUI listens. Blank uses FLUX_API_URL, else "
                 "http://127.0.0.1:8188.",
                 group="Media", owner="image", admin_only=True),
    SettingField("comfy_launch_timeout", Widget.NUMBER,
                 "ComfyUI launch timeout (s)",
                 "Seconds to wait for ComfyUI after launching. A ZLUDA/ROCm cold "
                 "start can take minutes.",
                 group="Media", owner="image", min=30, step=30),
    SettingField("comfy_output_dir", Widget.FOLDER, "ComfyUI output folder",
                 "ComfyUI's own output folder. Only needed with 'Remove ComfyUI's "
                 "copy after generating' on; blank derives it.",
                 # NOT admin_only, though it names a folder this process DELETES
                 # from when comfy_delete_outputs is on (comfy_client.py, the
                 # unlink() calls under _comfy_output_root). That delete-escape is
                 # held by path confinement at the unlink site, not by an owner
                 # flag here.
                 group="Media", owner="image"),
    SettingField("comfy_delete_outputs", Widget.TOGGLE,
                 "Remove ComfyUI's copy after generating",
                 "Delete ComfyUI's own copy and history entry once localm has "
                 "saved its own. Off by default (keep them); privacy mode forces "
                 "it on.",
                 group="Media", owner="image"),
    SettingField("comfy_fast_dequant", Widget.TOGGLE,
                 "Fast GGUF dequant (fp16)",
                 "Rewrite a slow float32 Flux GGUF dequant to fp16/bf16 on submit. "
                 "float32 is the usual cause of very slow gen on smaller cards.",
                 group="Media", owner="image"),
    # The flag is passed by appending --disable-auto-launch to the launcher, so
    # it needs a launcher that FORWARDS args to main.py. The stock run_*.bat and
    # a plain `python main.py` do; a launcher that drops its args ignores the
    # setting.
    SettingField("comfy_disable_auto_launch", Widget.TOGGLE,
                 "Suppress ComfyUI's web page on launch",
                 "When localm starts ComfyUI for you, keep it headless instead of "
                 "letting it open its own browser tab. Off by default.",
                 group="Media", owner="image"),
    # Reactive fix for an upstream ComfyUI core regression that crashes native
    # ACE-Step audio with "'function' object has no attribute
    # '__func__'". When on, a ComfyUI that localm STARTS gets an in-memory
    # compatibility patch via a PYTHONPATH env var - localm writes NOTHING into
    # your ComfyUI install and never patches a ComfyUI it did not start. The patch
    # self-expires once ComfyUI ships its own fix.
    SettingField("comfy_func_shim", Widget.TOGGLE,
                 "Fix ComfyUI ACE-Step __func__ crash (in-memory)",
                 "Work around a known ComfyUI crash in native ACE-Step audio. Off "
                 "by default. Patches only a ComfyUI that localm starts, in memory "
                 "- nothing is written into your install.",
                 group="Media", owner="image"),
    # Per-component placement needs a ComfyUI that offers the multigpu
    # Select*Device nodes - localm's own managed ComfyUI has them, and an older
    # ComfyUI of your own declines cleanly and says why. It does NOT split one
    # model across cards (no ComfyUI feature does). Off by default; with it off,
    # or on a single-GPU box, media generation is unchanged.
    SettingField("comfy_gpu_placement", Widget.TOGGLE,
                 "Split media across GPUs (experimental)",
                 # The help must keep BOTH halves: that placement is not promised
                 # everywhere, and that single-GPU setups are unaffected. See
                 # test_placement_help_still_states_the_two_gpu_precondition.
                 "EXPERIMENTAL, off by default. With two or more GPUs, put the text "
                 "encoder and VAE on a second card, freeing room for the diffusion "
                 "model. Single-GPU setups are unaffected. Needs a recent ComfyUI.",
                 group="Media", owner="image", applies=Applies.RESTART),
    SettingField("reload_llm_after_imagine", Widget.TOGGLE,
                 "Reload chat model after generating",
                 "Free the media model's VRAM and reload the chat model after a "
                 "gen. Turn off when making many in a row.",
                 group="Media", owner="image"),
    SettingField("model_swap_policy", Widget.SELECT, "Media VRAM swap",
                 "auto = keep chat loaded if the media model fits alongside; "
                 "always = always unload chat; never = keep chat hot.",
                 group="Media", owner="image",
                 options=["auto", "always", "never"]),
    # 'own' is INERT until `localm comfy setup` has been run. The user's own
    # ComfyUI is never modified either way, and neither install's settings are
    # lost by switching back and forth.
    SettingField("comfy_target", Widget.SELECT, "ComfyUI to use",
                 "own = localm's own managed ComfyUI, installed under the localm "
                 "data folder by 'localm comfy setup'. user = always your own "
                 "ComfyUI install, even when a managed one exists.",
                 group="Media", owner="image", options=["own", "user"],
                 applies=Applies.RESTART),
    # The GLOBAL fallback behind the per-plugin "Model weight dtype" Media
    # field (MEDIA_PLUGIN_FIELDS "float_type"): the music/video backends read
    # the plugin block's value, else this key. Options must stay identical to
    # the per-plugin field's (pinned by a test). Not rendered anywhere itself:
    # like every
    # per-plugin-mapped comfy_* global, the GUI edits the per-plugin values
    # (schema_json's media_per_plugin annotation routes it away from the
    # Media section's shared box).
    SettingField("comfy_float_type", Widget.SELECT, "Media weight dtype (shared)",
                 "Shared fallback for the per-plugin 'Model weight dtype' "
                 "Media setting. Blank inherits the workflow default.",
                 group="Media", owner="image",
                 options=["default", "fp16", "bf16", "fp32", "fp8_e4m3fn",
                          "fp8_e5m2"]),
    # ---- Network (web plugin) ----
    # "ask" prompts only a BROWSER-initiated request. A non-browser API or MCP
    # client holding the web scope is NOT prompted and its requests proceed. The
    # private-address SSRF guard and the domain lists apply in every mode, "off"
    # included.
    SettingField("net_mode", Widget.SELECT, "Network access",
                 "Model-initiated web access: off = every request fails; ask = the "
                 "GUI prompts first; allow = no prompt. The SSRF guard and the "
                 "domain lists apply in all three.",
                 group="Network", owner="web", options=["off", "ask", "allow"]),
    SettingField("net_allow", Widget.LIST, "Allowed domains",
                 "Domains the model may reach. Empty = any. e.g. example.com "
                 "(also covers *.example.com).",
                 group="Network", owner="web"),
    # admin_only: net_deny is the SUPPRESSIVE half of the domain policy and is
    # stored verbatim, so without the gate a non-owner could CLEAR it
    # (validate_update accepts {"net_deny": []}) and un-block every host the
    # owner blocked. NOT symmetric with net_allow, which stays non-admin (pinned
    # by test_net_allow_private_is_admin_only): net_allow only names PUBLIC
    # hosts and _check_public_address() still constrains it.
    SettingField("net_deny", Widget.LIST, "Denied domains",
                 "Domains always refused, even when listed in Allowed domains "
                 "(deny wins). Empty = none.",
                 group="Network", owner="web", admin_only=True),
    SettingField("net_allow_private", Widget.TOGGLE,
                 "Allow private/loopback targets (disables the SSRF guard)",
                 "Permit requests to localhost and private IP ranges. Off by "
                 "default (a common SSRF vector); only enable for a trusted setup.",
                 # admin_only: flipping this DISABLES the server-wide SSRF guard, so
                 # it widens a network trust boundary exactly like the rag_* folder
                 # keys widen a filesystem one - a non-owner config:write key must not
                 # be able to set it (else it could reach loopback/metadata via any
                 # model-initiated fetch). See routes/config.py admin_only gate.
                 group="Network", owner="web", admin_only=True),
    SettingField("net_allow_model_downloads", Widget.TOGGLE,
                 "Allow model downloads while network access is off",
                 "Lets a model pull, HuggingFace search, or vision-projector/voice/"
                 "embedding fetch proceed while net_mode is off. Model-initiated "
                 "network access is still blocked.",
                 # admin_only: same reasoning as net_allow_private - this loosens a
                 # network-policy floor, so a non-owner config:write key must not be
                 # able to set it.
                 group="Network", owner="web", admin_only=True),
    # admin_only: this names WHERE every web search is sent, so without the gate
    # a non-owner could re-point the search channel at a host it controls and
    # receive the owner's queries - the same "where does data go" boundary that
    # makes bugreport_upload_url and update_url admin_only.
    SettingField("net_search_url", Widget.TEXT, "Search backend URL",
                 "A SearXNG JSON search endpoint for web search. Blank uses "
                 "DuckDuckGo (no key needed).",
                 group="Network", owner="web", admin_only=True),
    # owner="core", not "web": this is the GUI's own renderer, so it must stay
    # visible on an install with no web plugin. admin_only for the same reason
    # net_search_url is - it decides whether rendering a reply causes an outbound
    # request at all, which is a "where does data go" boundary, and a non-owner
    # config:write key must not be able to switch it on.
    # A remote image a model links is an exfiltration channel: the ADDRESS
    # ITSELF carries the data, and the fetch happens the moment the reply
    # renders.
    #
    # "on" does not close that channel. It moves the request from the user's
    # browser to this server, so the remote site never learns their IP, their
    # browser, or which page they were on, and the fetch obeys the same SSRF
    # guard and allow/deny domain lists as every other outbound request localm
    # makes. It ships OFF.
    #
    # "ask" closes the channel for an ARBITRARY host: the route refuses with 428
    # until the request carries the reader's consent for that ORIGIN, so no
    # fetch happens for a host they have not seen. Consent is per-ORIGIN, not
    # per-image, since the payload is in the URL.
    SettingField("gui_proxy_remote_images", Widget.SELECT,
                 "Show remote images in replies (fetched by this machine)",
                 "Off by default. 'on' loads them; 'ask' checks with you once "
                 "per site per conversation first. Either way this machine "
                 "fetches the image, so the site never learns your IP or "
                 "browser.",
                 # Spelled out rather than imported from config: every
                 # localm.config import in this module is LAZY (inside a
                 # function), because importing config runs its data-directory
                 # detection, and a module-level import here would drag that into
                 # merely importing the schema.
                 # test_the_modes_match_config_s_own_constants pins these two
                 # to config.REMOTE_IMAGE_MODES / REMOTE_IMAGE_LEGACY_BOOL so
                 # the copy cannot drift.
                 options=["off", "ask", "on"],
                 legacy_bool=("off", "on"),
                 # Reachable in the default keyless install regardless: the schema
                 # route treats open mode as owner (is_owner = held is None or
                 # ADMIN in held), so admin_only hides this from a SCOPED key, not
                 # from the ordinary single-user GUI.
                 group="Network", admin_only=True),
    # ---- Voice (plugin) ----
    SettingField("voice_stt_model", Widget.SELECT, "Speech-to-text model",
                 "Whisper model size for the microphone button. Larger is more "
                 "accurate but slower and uses more memory.",
                 group="Voice", owner="voice",
                 options=["tiny", "base", "small", "medium"]),
    SettingField("voice_stt_language", Widget.TEXT, "Speech-to-text language",
                 "Force a language for transcription, e.g. en or de. Blank "
                 "auto-detects from the audio.",
                 group="Voice", owner="voice"),
]


# --------------------------------------------------------------------------- #
#  Validation: coerce + check a dict of config updates against the schema.    #
#  Single source of truth for PATCH /v1/config and `localm config`, so        #
#  neither can persist an unknown key, a wrong-typed value, a SELECT value    #
#  outside its options, or an out-of-range number. The GUI form submits       #
#  strings; raw API clients submit native JSON types - both are coerced here. #
# --------------------------------------------------------------------------- #

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _field_map() -> dict:
    return {f.key: f for f in CORE_FIELDS}


def _to_bool(key: str, val):
    if isinstance(val, bool):
        return val
    if isinstance(val, int) and val in (0, 1):   # JSON clients that send 1/0
        return bool(val)
    if isinstance(val, str):
        low = val.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
    raise ValueError(f"{key}: expected a boolean, got {val!r}")


def _to_number(key: str, val, *, want_int: bool, lo, hi):
    if isinstance(val, bool):       # bool is an int subclass - reject as a number
        raise ValueError(f"{key}: expected a number, not a boolean")
    try:
        # OverflowError: int(float("inf")) raises it, and it is neither a
        # TypeError nor a ValueError - without this an inf into an int field
        # would leak an uncaught OverflowError out of validate_update.
        num = int(val) if want_int else float(val)
    except (TypeError, ValueError, OverflowError):
        kind = "an integer" if want_int else "a number"
        raise ValueError(f"{key}: expected {kind}, got {val!r}")
    # NaN/inf pass every < / > bounds check below (NaN compares False to all, inf
    # only trips a finite upper bound), so a non-finite float would otherwise be
    # persisted and then 500 every GET/PATCH /v1/config (FastAPI renders with
    # allow_nan=False). int coercion can never produce a non-finite value, so this
    # only guards the float path. Mirrors the gpu_split_ratios guard below.
    if not want_int and not math.isfinite(num):
        raise ValueError(f"{key}: expected a finite number, got {val!r}")
    if lo is not None and num < lo:
        raise ValueError(f"{key}: {num} is below the minimum {lo}")
    if hi is not None and num > hi:
        raise ValueError(f"{key}: {num} is above the maximum {hi}")
    return num


def _to_str_list(key: str, val):
    if isinstance(val, str):
        return [s.strip() for s in val.split(",") if s.strip()]
    if isinstance(val, (list, tuple)):
        return [str(s).strip() for s in val if str(s).strip()]
    raise ValueError(f"{key}: expected a list of strings, got {val!r}")


def _validate_key_presets(val):
    """Validate the key_presets config: a list of {name, scopes} bundles. Each
    name is a non-empty string and each scopes is a list of KNOWN scope strings
    (an unknown/typo'd scope is rejected so a preset can never carry a capability
    that does not exist). Returns the normalized list."""
    from localm import scopes as S
    if not isinstance(val, list):
        raise ValueError("key_presets: expected a list of {name, scopes} objects")
    out = []
    for i, item in enumerate(val):
        if not isinstance(item, dict):
            raise ValueError(f"key_presets[{i}]: expected an object")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"key_presets[{i}]: a name is required")
        raw = item.get("scopes", [])
        if not isinstance(raw, list):
            raise ValueError(f"key_presets[{i}].scopes: expected a list")
        clean = S.normalize(raw)
        bad = [s for s in clean if not S.is_valid_scope(s)]
        if bad:
            raise ValueError(f"key_presets[{i}]: unknown scope(s): {', '.join(bad)}")
        out.append({"name": name, "scopes": clean})
    return out


def _validate_avatar_value(key: str, val) -> str:
    """Validate one avatar value: "", a short emoji/text glyph, or a raster
    data URI. See _AVATAR_DATA_URI_RE / test_avatar_value_rejects_url_or_path."""
    if val is None:
        return ""
    s = str(val)
    if not s:
        return ""
    if _AVATAR_DATA_URI_RE.match(s):
        if len(s) > _AVATAR_MAX_DATA_URI_LEN:
            raise ValueError(
                f"{key}: image is too large ({len(s)} bytes encoded, max "
                f"{_AVATAR_MAX_DATA_URI_LEN})")
        return s
    if "://" in s or s.startswith("data:") or "\\" in s or "/" in s:
        raise ValueError(
            f"{key}: must be a short glyph or an uploaded png/jpeg/gif/webp "
            f"image, never a URL or a path, got {s!r}")
    if len(s) > _AVATAR_MAX_GLYPH_LEN:
        raise ValueError(
            f"{key}: glyph is too long ({len(s)} characters, max "
            f"{_AVATAR_MAX_GLYPH_LEN})")
    return s


def _validate_avatar_overrides(key: str, val) -> dict:
    """Validate model_avatar_overrides: {model_id: avatar_value}, each value
    checked with _validate_avatar_value. An empty value clears that entry."""
    if val is None:
        return {}
    if not isinstance(val, dict):
        raise ValueError(f"{key}: expected an object of {{model_id: icon}}")
    out: dict = {}
    for model_id, icon in val.items():
        mid = str(model_id).strip()
        if not mid:
            raise ValueError(f"{key}: a model id is required")
        checked = _validate_avatar_value(f"{key}[{mid}]", icon)
        if checked:
            out[mid] = checked
    return out


def _validate_one(key: str, val, field: "SettingField", default):
    nullable = default is None
    widget = field.widget

    if val is None:
        if nullable or widget in (Widget.TEXT, Widget.FOLDER, Widget.PATH, Widget.SECRET):
            return None
        raise ValueError(f"{key}: a value is required (got null)")

    if widget == Widget.TOGGLE:
        return _to_bool(key, val)

    if widget == Widget.NUMBER:
        want_int = isinstance(default, int) and not isinstance(default, bool)
        return _to_number(key, val, want_int=want_int, lo=field.min, hi=field.max)

    if widget == Widget.SELECT:
        s = "" if val == "" else str(val)
        if s == "" and nullable:
            return None
        if field.options and s in field.options:
            return s
        if field.legacy_bool:
            try:
                b = _to_bool(key, val)
            except ValueError:
                pass
            else:
                return field.legacy_bool[1] if b else field.legacy_bool[0]
        raise ValueError(f"{key}: {val!r} is not one of {field.options}")

    if widget == Widget.LIST:
        return _to_str_list(key, val)

    if widget == Widget.PATHLIST:
        # A list of folder paths. Run each through the RAG indexer's hard floor
        # (confine_index_path with policy=None = only the always-denied checks: the
        # localm data dir and credential folders), so a root that could never be
        # indexed is rejected at SAVE time with a clear error instead of being
        # silently useless. Store the resolved absolute path (what indexing_policy
        # compares against) and drop duplicates, preserving order.
        from localm.rag.store import confine_index_path
        out: list = []
        seen: set = set()
        for item in _to_str_list(key, val):
            try:
                rp = str(confine_index_path(item))
            except ValueError as e:
                raise ValueError(f"{key}: {e}")
            if rp not in seen:
                seen.add(rp)
                out.append(rp)
        return out

    if widget == Widget.HIDDEN:
        if key == "logo_style":
            s = str(val)
            if s in LOGO_STYLE_IDS:
                return s
            raise ValueError(f"{key}: {val!r} is not one of {LOGO_STYLE_IDS}")
        if key == "key_presets":
            return _validate_key_presets(val)
        if key == "main_gpu_index":
            # A HIDDEN NUMBER-shaped field (the dedicated GPU selector renders it,
            # not the generic number box - see its schema comment). val may arrive
            # as a JSON int (GUI) or a CLI string (`localm config main_gpu_index
            # 1`); route it through the shared _to_number helper so config.json
            # stores a real int and this field inherits the SAME coercion guards
            # as every other number (bool/NaN/inf/overflow rejection). A separate
            # hand-rolled int(val) here would leak an uncaught OverflowError on
            # inf (int(float("inf"))).
            return _to_number(key, val, want_int=True, lo=field.min, hi=field.max)
        if key == "gpu_split_indices":
            # A HIDDEN list-of-ints field (the checkbox row renders it, not a
            # generic list box). Accepts a CLI CSV string ("0,1") or a GUI JSON
            # list of ints - _to_str_list normalizes either shape to string
            # tokens first (same convention Widget.LIST fields use), then each
            # token is coerced/bounded via the shared _to_number helper (also
            # rejects booleans). Empty/null clears the split (single-GPU).
            if val is None:
                return None
            tokens = _to_str_list(key, val)
            if not tokens:
                return None
            return [_to_number(key, t, want_int=True, lo=0, hi=MAX_GPU_SPLIT_INDEX)
                    for t in tokens]
        if key == "gpu_split_ratios":
            # Same shape as gpu_split_indices, but positive floats (a ratio of
            # 0 or below is meaningless - llama.cpp would give that device no
            # share at all, silently dropping it from the split). _to_number now
            # rejects non-finite floats itself; the explicit isfinite check below
            # stays as defense-in-depth right at the native tensor_split ctypes
            # boundary, alongside the >0 check that _to_number does not do.
            if val is None:
                return None
            tokens = _to_str_list(key, val)
            if not tokens:
                return None
            out: list = []
            for t in tokens:
                r = _to_number(key, t, want_int=False, lo=None, hi=None)
                if not math.isfinite(r):
                    raise ValueError(f"{key}: {r} must be a finite number")
                if r <= 0:
                    raise ValueError(f"{key}: {r} must be greater than 0")
                out.append(r)
            return out
        if key == "max_resident_models":
            # A HIDDEN nullable INT (see its schema comment). Blank clears the
            # cap back to "no cap", so "" is accepted here rather than falling
            # into _to_number, where int("") is a hard error.
            if isinstance(val, str) and not val.strip():
                return None
            return _to_number(key, val, want_int=True, lo=field.min, hi=field.max)
        if key == "pinned_models":
            # A HIDDEN list-of-strings field: HIDDEN skips the Widget.LIST
            # branch above, so normalize here. Accepts a CLI CSV string
            # ("a,b") or a JSON list; empty clears every pin.
            tokens = _to_str_list(key, val)
            return tokens or None
        if key in ("user_avatar", "model_avatar_default"):
            return _validate_avatar_value(key, val)
        if key == "model_avatar_overrides":
            return _validate_avatar_overrides(key, val)
        # plugins_enabled (list) / plugins (dict): managed by the engine, not the
        # settings form, but accepted with the right container type for the
        # GET->PATCH round-trip the GUI does.
        if isinstance(default, list):
            if not isinstance(val, list):
                raise ValueError(f"{key}: expected a list")
            return [str(s) for s in val]
        if isinstance(default, dict):
            if not isinstance(val, dict):
                raise ValueError(f"{key}: expected an object")
            return val
        return val

    # TEXT / FOLDER / PATH / SECRET
    if key == "bind_host":
        # Store "" (loopback default) or a value the server can actually bind:
        # 'localhost' or an IP literal. Rejecting everything else AT WRITE TIME
        # is what keeps the Settings > Restart server flow safe: an unbindable
        # value would only surface as a startup failure, and the user who set
        # it from the GUI may have no terminal to recover from one. The read
        # site (cli._resolve_bind_host) re-checks with the same predicate as
        # defense in depth against a hand-edited config.json.
        from localm.bindhost import is_valid_bind_host
        s = "" if val is None else str(val).strip()
        if not s:
            return ""
        if not is_valid_bind_host(s):
            raise ValueError(
                f"bind_host: {val!r} is not a bindable address - use an IP "
                f"literal (0.0.0.0 or :: for every interface, or one "
                f"interface's IP like 192.168.1.20 or 2001:db8::5) or "
                f"localhost; blank = this computer only. Hostnames, host:port "
                f"values and zone-scoped addresses like fe80::1%eth0 are not "
                f"accepted (the port has its own setting, and a zone index "
                f"only means anything on the machine that wrote it).")
        return s
    if key in ("tls_cert", "tls_key"):
        # A non-empty value must point at an existing file NOW, so a typo is a
        # clear save-time error instead of a silent fallback at the next
        # restart (the startup read falls back to the built-in cert when the
        # pair is unusable - see cli._resolve_tls - which keeps the server
        # alive but is not what the user asked for; catch it here first).
        from pathlib import Path
        s = "" if val is None else str(val).strip()
        if not s:
            return ""
        if not Path(s).is_file():
            raise ValueError(f"{key}: file not found: {s}")
        return s
    if key == "mdns_name":
        # Store the sanitized DNS label, not the raw input, so config.json always
        # holds a valid mDNS name (ASCII letters/digits/hyphens) whatever was typed.
        # Gate on what actually SANITIZES to a usable label - not str.isalnum(),
        # which is Unicode-aware and would let a non-ASCII name (e.g. Greek/CJK)
        # pass the guard and then silently strip to the "localm" default. A name
        # that reduces to nothing raises here rather than falling back to the
        # default.
        from localm.netname import normalize_label
        s = "" if val is None else str(val).strip()
        if not s:
            raise ValueError("mdns_name: a name is required (letters, digits, hyphens)")
        label = normalize_label(s)
        if not label:
            raise ValueError(
                f"mdns_name: {val!r} has no usable characters "
                f"(need ASCII letters, digits, or hyphens)")
        return label
    if key == "cors_origins":
        # None | "*" | list of origins; a comma string becomes a list so the
        # server's CORS handling (which only honours "*"/list) actually applies.
        # A list is validated here rather than via _to_str_list (which str()-
        # coerces non-strings and has no length cap) because this allowlist is
        # security-relevant and _to_str_list's coercion is depended on by other
        # callers (gpu_split_indices/gpu_split_ratios feed it numeric tokens).
        if isinstance(val, (list, tuple)):
            if len(val) > MAX_CORS_ORIGINS:
                raise ValueError(
                    f"{key}: too many origins ({len(val)}), max {MAX_CORS_ORIGINS}")
            if not all(isinstance(s, str) for s in val):
                raise ValueError(f"{key}: expected a list of strings, got {val!r}")
            return [s.strip() for s in val if s.strip()]
        s = str(val).strip()
        if not s:
            return None
        if s == "*":
            return "*"
        return _to_str_list(key, s)
    if isinstance(val, str):
        s = val.strip()
        return s or (None if nullable else "")
    raise ValueError(f"{key}: expected a string, got {val!r}")


def validate_update(updates: dict) -> dict:
    """Coerce + validate a dict of config updates against CORE_FIELDS.

    Returns a new dict of normalized, correctly-typed values. Raises ValueError
    on an unknown key, a value that cannot be coerced to the field's type, a
    SELECT value outside its options, or a number outside its min/max."""
    from localm.config import DEFAULT_CONFIG
    fields = _field_map()
    out: dict = {}
    for key, val in updates.items():
        if key not in DEFAULT_CONFIG:
            raise ValueError(f"unknown config key: {key!r}")
        field = fields.get(key)
        if field is None:                  # schema/config drift (a test guards this)
            out[key] = val
            continue
        out[key] = _validate_one(key, val, field, DEFAULT_CONFIG[key])
    return out


def all_widgets() -> set:
    """Every defined widget string."""
    return {v for k, v in vars(Widget).items()
            if not k.startswith("_") and isinstance(v, str)}


def fields_by_owner(owner: str) -> list:
    return [f for f in CORE_FIELDS if f.owner == owner]


def admin_only_keys() -> set:
    """Config keys flagged owner-only (``admin_only``). A non-ADMIN caller may
    neither see them in the schema nor write them via PATCH /v1/config, because
    they widen a trust boundary (e.g. the rag_* indexing settings define which
    host folders the indexer may read). The single source of truth for both gates."""
    return {f.key for f in CORE_FIELDS if f.admin_only}


def engine_managed_keys() -> set:
    """Config keys that hold engine/plugin STATE rather than a setting, and that
    PATCH /v1/config must therefore refuse to a non-ADMIN caller.

    validate_update has no schema for what lives INSIDE these keys, so the HIDDEN
    branch stores them verbatim (``plugins``) or with only a per-element str()
    (``plugins_enabled``). Their real write surfaces DO validate, and each guards
    a boundary the generic settings route knows nothing about:

      - ``plugins``         -> POST /v1/tts/config requires an owner for
                               library/wasm_paths (a script every browser
                               imports), and POST /v1/media/config/<name>
                               requires one for launch_cmd/api_url (a shell
                               command / a render target).
      - ``plugins_enabled`` -> POST /api/plugins/<name>/enable requires the
                               PLUGINS_ADMIN scope, which a config:write key
                               need not hold.

    Without this gate the generic route outranks the specific one: a non-owner
    config:write key could write the same state and skip both the value check
    and the stronger scope.

    Unlike admin_only_keys these are WRITE-gated only - the values stay
    readable. Keep this derived from the flag, not a literal set."""
    return {f.key for f in CORE_FIELDS if f.engine_managed}


def schema_json(values: Optional[dict] = None, *, is_owner: bool = True) -> list:
    """Serialize the core schema, injecting each non-secret field's current
    default from DEFAULT_CONFIG (or *values* if given). The GUI renders this.

    When *is_owner* is False, owner-only fields (``admin_only``) are OMITTED, so a
    non-owner never receives the control (the write is also refused server-side;
    hiding it here just avoids rendering a field they cannot use). Callers that
    are not request-scoped (the CLI, tests) default to owner (see everything).

    Auto-detect fields also carry an ``auto`` value: the path localm would
    resolve when the field is left blank, so the GUI can SHOW it (filled, greyed)
    instead of an empty box that hides what is actually in use. Today only
    ``binary_dir`` resolves one (the bundled llama.cpp runtime)."""
    from localm.config import DEFAULT_CONFIG
    base = DEFAULT_CONFIG if values is None else values
    # The GUI's Media section skips group="Media" fields in the flat form and
    # shows per-plugin-mapped globals ONLY inside the per-plugin boxes, so it
    # must be able to tell which those are FROM THE SCHEMA rather than from a
    # client-side name allowlist, which renders any field missing from it
    # nowhere. MEDIA_PLUGIN_FIELDS is the single source of truth; every OTHER
    # Media field renders in the section's shared box by default, so a future
    # field cannot silently vanish.
    media_mapped = {m.global_key for m in MEDIA_PLUGIN_FIELDS}
    out = []
    for f in CORE_FIELDS:
        if f.admin_only and not is_owner:
            continue
        d = f.to_json()
        if f.group == "Media":
            d["media_per_plugin"] = f.key in media_mapped
        if not f.secret and f.key in base:
            d["default"] = base[f.key]
        # The SHIPPED default, independent of `base` above: `base` is the CURRENT
        # value (load_config(), which after a save is the user's own override), so
        # `default` alone cannot tell the GUI "is this still factory-fresh" from
        # "the user set it to this exact number". Always sourced from
        # DEFAULT_CONFIG regardless of *values*, so the GUI can grey a field that
        # still matches what shipped rather than rendering every value - default
        # or override alike - as solid, indistinguishable text.
        if not f.secret and f.key in DEFAULT_CONFIG:
            d["shipped_default"] = DEFAULT_CONFIG[f.key]
        if f.key == "binary_dir":
            try:
                from localm.config import find_binary_dir
                resolved = find_binary_dir()
                d["auto"] = str(resolved) if resolved else ""
            except Exception:
                d["auto"] = ""
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
#  Per-plugin media config (image / music / video).                           #
#                                                                              #
#  Each media plugin keeps its OWN settings block under config["plugins"][name]#
#  so the three are configured INDEPENDENTLY. The backends already read it     #
#  (block value, else the global comfy_* fallback - see media_config.py and    #
#  the media backends). This is what the GUI "Media" section edits, one        #
#  subsection per plugin. The global comfy_* keys remain the shared fallback   #
#  (and the CLI / PATCH /v1/config path) for back-compat.                      #
# --------------------------------------------------------------------------- #

MEDIA_PLUGINS = ("image", "music", "video")


@dataclass
class MediaField:
    key: str                       # API field name, e.g. "workdir"
    block_path: tuple              # where it lives in the plugin block
    global_key: str                # global DEFAULT_CONFIG fallback key
    widget: str
    label: str
    help: str = ""
    options: Optional[list] = None
    image_only: bool = False       # fast_dequant only applies to the Flux image backend
    plugins: Optional[list] = None # restrict to these plugins only (e.g. ["music", "video"])
    admin_only: bool = False       # requires an owner (ADMIN) principal to see or set


# Order = display order within each plugin subsection.
MEDIA_PLUGIN_FIELDS: list = [
    # The per-plugin workdir WINS over the global comfy_workdir (scan.py returns
    # the per-plugin value first; image/backend.py and its music/video twins
    # pass it into ensure_comfy()), so gating only the CORE field would leave
    # this as the live surface. It reaches execution the same way: a blank
    # launch_cmd makes the launcher AUTO-DISCOVERED inside this folder
    # (comfy_client.discover_launch_cmd) and run via Popen. admin_only here
    # hides the resolved value; the WRITE gate is set_media_config in
    # routes/config.py, which covers workdir alongside launch_cmd/api_url.
    MediaField("workdir", ("comfy", "workdir"), "comfy_workdir", Widget.FOLDER,
               "ComfyUI folder",
               "This plugin's ComfyUI install folder. Blank uses the shared default.",
               admin_only=True),
    # launch_cmd is a shell command and api_url is a render target, so both
    # widen a trust boundary the same way the tts library/wasm_paths script-URL
    # fields do. admin_only=True hides their RESOLVED value from a non-owner
    # config:read caller (see media_schema_json), on top of the write-side owner
    # gate in set_media_config.
    MediaField("launch_cmd", ("comfy", "launch_cmd"), "comfy_launch_cmd", Widget.TEXT,
               "ComfyUI launch command",
               "Launcher that starts ComfyUI for this plugin. Blank auto-detects one "
               "in the folder.", admin_only=True),
    MediaField("api_url", ("comfy", "api_url"), "comfy_api_url", Widget.TEXT,
               "ComfyUI API URL",
               "Where this plugin's ComfyUI listens. Blank uses the shared default.",
               admin_only=True),
    MediaField("output_dir", ("comfy", "output_dir"), "comfy_output_dir", Widget.FOLDER,
               "ComfyUI output folder",
               "Only needed if 'Remove ComfyUI's copy' is on; blank derives it."),
    # The swap_policy SELECT sits ahead of the toggle fields so all dropdowns
    # render before all checkboxes in each subsection.
    MediaField("swap_policy", ("model_swap_policy",), "model_swap_policy",
               Widget.SELECT, "Media VRAM swap",
               "auto = keep chat if it fits; always = unload chat; never = keep chat hot.",
               options=["auto", "always", "never"]),
    MediaField("delete_outputs", ("comfy", "delete_outputs"), "comfy_delete_outputs",
               Widget.TOGGLE, "Remove ComfyUI's copy after generating",
               "Delete ComfyUI's own copy once localm saved its own. Off = keep; "
               "privacy mode forces it on."),
    MediaField("fast_dequant", ("comfy", "fast_dequant"), "comfy_fast_dequant",
               Widget.TOGGLE, "Fast GGUF dequant (fp16)",
               "Rewrite a slow float32 Flux GGUF dequant to fp16/bf16 on submit.",
               image_only=True),
    MediaField("float_type", ("comfy", "float_type"), "comfy_float_type",
               Widget.SELECT, "Model weight dtype",
               "Force the compute precision (dtype) of the model weights.",
               options=["default", "fp16", "bf16", "fp32", "fp8_e4m3fn", "fp8_e5m2"],
               plugins=["music", "video"]),
    MediaField("reload_after", ("reload_llm_after_generate",), "reload_llm_after_imagine",
               Widget.TOGGLE, "Reload chat model after generating",
               "Free this backend's VRAM and reload the chat model after a gen."),
]


def _block_get(block: dict, path: tuple):
    """Read a (possibly nested) value out of a plugin block, or None."""
    cur = block
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def media_fields_for(name: str) -> list:
    """The MediaFields that apply to plugin *name* (drops image-only ones else)."""
    return [
        f for f in MEDIA_PLUGIN_FIELDS
        if not (f.image_only and name != "image")
        and not (f.plugins is not None and name not in f.plugins)
    ]


def media_admin_only_fields() -> set:
    """Field keys (across all media plugins) flagged owner-only (today:
    launch_cmd, api_url). A non-owner config:write key must not set them
    (set_media_config's owner gate) and must not see their resolved value
    either (media_schema_json). The single source of truth for both."""
    return {f.key for f in MEDIA_PLUGIN_FIELDS if f.admin_only}


def media_schema_json(name: str, block: Optional[dict], full_config: dict, *,
                       is_owner: bool = True) -> list:
    """Serialize one media plugin's editable fields with their RESOLVED values.

    ``value`` is the per-plugin block value when set, else the global comfy_*
    fallback, so the GUI shows what is actually in effect. ``is_override`` flags
    whether this plugin has its own value (vs inheriting the shared default).

    When *is_owner* is False, admin_only fields (launch_cmd, api_url) are
    OMITTED entirely, mirroring schema_json's admin_only handling for the core
    schema: a non-owner config:read caller must not learn a shell command or a
    render target it is not allowed to set either. Callers that are not
    request-scoped (the CLI, tests) default to owner (see everything)."""
    block = block if isinstance(block, dict) else {}
    out = []
    for f in media_fields_for(name):
        if f.admin_only and not is_owner:
            continue
        block_val = _block_get(block, f.block_path)
        has_own = block_val not in (None, "")
        value = block_val if has_own else full_config.get(f.global_key)
        d = {"key": f.key, "widget": f.widget, "label": f.label, "help": f.help,
             "value": value, "is_override": has_own, "global": full_config.get(f.global_key)}
        if f.options:
            d["options"] = f.options
            
        if f.key == "launch_cmd":
            try:
                workdir = _block_get(block, ("comfy", "workdir")) or full_config.get("comfy_workdir")
                if workdir:
                    from localm.image_gen.comfy import discover_launch_cmd
                    from pathlib import Path
                    found = discover_launch_cmd(Path(workdir))
                    d["auto"] = found if found else ""
                    if not found and Path(workdir).is_dir():
                        import urllib.parse
                        d["action"] = {
                            "label": "Create launch script",
                            "endpoint": f"/api/comfyui/create-launcher?workdir={urllib.parse.quote(str(workdir))}",
                            "success_msg": "Launcher script created!"
                        }
            except Exception:
                pass
                
        out.append(d)
    return out


def _coerce_media_value(f: "MediaField", val):
    """Coerce one media-field value to its widget type (mirrors validate_update)."""
    if f.widget == Widget.TOGGLE:
        return _to_bool(f.key, val)
    if f.widget == Widget.SELECT:
        s = "" if val is None else str(val)
        if f.options and s in f.options:
            return s
        raise ValueError(f"{f.key}: {val!r} is not one of {f.options}")
    # TEXT / FOLDER: empty string clears the override (back to the shared default)
    if val is None:
        return None
    s = str(val).strip()
    return s or None


def _is_http_url(value: str) -> bool:
    """True if *value* is a well-formed absolute http(s) URL with a host.

    Shape-only: scheme in http/https plus a non-empty host. A scheme-less or
    hostless value never works as a ComfyUI endpoint (urllib.request.urlopen
    needs a scheme). SSRF / link-local screening is NOT done here - that stays
    with sanitize_comfy_url at request time."""
    import urllib.parse
    try:
        parsed = urllib.parse.urlparse(value)
    except (ValueError, TypeError):
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.hostname)


def validate_media_block(name: str, updates: dict) -> dict:
    """Coerce + validate a per-plugin media update into a block-merge dict.

    Returns a nested dict shaped like the stored plugin block (e.g.
    ``{"comfy": {"workdir": ...}, "model_swap_policy": ...}``) ready to DEEP-MERGE
    into config["plugins"][name]. Raises ValueError on an unknown plugin/field or
    a bad value. A field set to "" (blank) is written as None, clearing the
    per-plugin override so the plugin falls back to the shared default."""
    if name not in MEDIA_PLUGINS:
        raise ValueError(f"unknown media plugin: {name!r}")
    by_key = {f.key: f for f in media_fields_for(name)}
    merge: dict = {}
    for key, val in updates.items():
        f = by_key.get(key)
        if f is None:
            raise ValueError(f"unknown media field for {name!r}: {key!r}")
        coerced = _coerce_media_value(f, val)
        # api_url shape check: a malformed URL is rejected at SET time with a
        # clear error, rather than stored and dropped back to the loopback
        # default by sanitize_comfy_url at READ time. launch_cmd is NOT
        # shape-checked: it is a free-form shell command, so there is no "valid
        # path" invariant to assert without rejecting real commands.
        if f.key == "api_url" and coerced is not None and not _is_http_url(coerced):
            raise ValueError(
                f"api_url must be a valid http(s) URL "
                f"(e.g. http://127.0.0.1:8188), got {coerced!r}")
        cur = merge
        for p in f.block_path[:-1]:
            cur = cur.setdefault(p, {})
        cur[f.block_path[-1]] = coerced
    return merge


# --------------------------------------------------------------------------- #
#  The tts plugin's own config block (config["plugins"]["tts"]).               #
#                                                                              #
#  Same idea as the media blocks above, one plugin instead of three: the       #
#  shipped defaults live in the plugin's tracked tts.example.json template and #
#  the user's overrides win over them (see the plugin's plug.py). GET/POST     #
#  /v1/tts/config (localm/inference/routes/config.py) is the write surface for #
#  these keys; the GUI voice picker's browser localStorage is a separate store #
#  and does not reach them.                                                    #
#                                                                              #
#  The block is FLAT (no nesting), so a validated update is merged key by key. #
# --------------------------------------------------------------------------- #

TTS_PLUGIN = "tts"

# The engines the plugin actually implements. tts.js speaks Kokoro only, so a
# one-option dropdown would be dead UI (hence gui=False below) - but the key is
# stored, so it still gets a validated write path instead of silently accepting
# an engine that does not exist. Adding an engine is a one-line change here.
TTS_ENGINES = ("kokoro",)
TTS_DEVICES = ("auto", "webgpu", "wasm")
# Only the dtypes the template documents as tried: fp32 is the clean default,
# q8/fp16 are smaller/faster but lower quality (q8 produces audible cracks on
# the WASM path - see the template's _dtype_note).
TTS_DTYPES = ("auto", "fp32", "fp16", "q8")
TTS_SPEED_MIN, TTS_SPEED_MAX = 0.5, 2.0

# A Hugging Face repo id ("owner/name"), which is what the BROWSER downloads the
# voice model from. Anything else (a URL, a path, a bare name) is not a repo id
# and would fail opaquely inside transformers.js at load time.
_HF_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

# Every entry must ship every voice id in voices.json, and be a repo the
# vendored runtime can actually phonemize (English only).
TTS_MODEL_OPTIONS = [
    ("onnx-community/Kokoro-82M-v1.0-ONNX",
     "Kokoro 82M v1.0 (default)"),
    ("onnx-community/Kokoro-82M-v1.0-ONNX-timestamped",
     "Kokoro 82M v1.0, with word timestamps"),
]


@dataclass
class TtsField:
    key: str                        # API field name == the config block key
    widget: str
    label: str
    help: str = ""
    options: Optional[list] = None  # static choices (voice's are loaded live)
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    gui: bool = True                # rendered in the GUI settings section
    advanced: bool = False          # secondary ("Advanced") box in the GUI
    admin_only: bool = False        # requires an owner (ADMIN) principal to set


# Order = display order in the GUI section.
TTS_FIELDS: list = [
    TtsField("voice", Widget.SELECT, "Default voice",
             "The voice new browsers read replies in. Each browser can pick its "
             "own in chat, which overrides this for that browser. The A-F "
             "grade rates its training data; higher is better."),
    TtsField("speed", Widget.NUMBER, "Speaking speed",
             "Playback rate for the generated voice. 1.0 is normal.",
             min=TTS_SPEED_MIN, max=TTS_SPEED_MAX, step=0.05),
    TtsField("model", Widget.SELECT, "Voice model",
             "Hugging Face repo id of the Kokoro model the browser downloads "
             "once and then caches. Blank uses the shipped default. Pick "
             "Custom to enter any other Kokoro-compatible repo id.",
             options=[repo for repo, _ in TTS_MODEL_OPTIONS]),
    TtsField("device", Widget.SELECT, "Compute device",
             "auto uses the GPU (WebGPU) when the browser has one and falls "
             "back to WASM. Force wasm if the GPU path misbehaves.",
             options=list(TTS_DEVICES), advanced=True),
    TtsField("dtype", Widget.SELECT, "Model precision",
             "auto picks fp32 (clean audio). q8/fp16 download less but sound "
             "worse; q8 produced audible cracks on the WASM path.",
             options=list(TTS_DTYPES), advanced=True),
    # Not in the GUI: exactly one engine is implemented (see TTS_ENGINES).
    TtsField("engine", Widget.SELECT, "Engine",
             "Speech engine the plugin uses.",
             options=list(TTS_ENGINES), gui=False),
    # These two become a SCRIPT url and a WASM base url that every browser
    # client loads from, so setting them is code injection into every user's
    # page. admin_only, confined to the plugin's own asset folder by the
    # validator, and not rendered in the GUI - an install-level escape hatch,
    # not a user setting.
    TtsField("library", Widget.TEXT, "Kokoro library path",
             "Path to the vendored kokoro-js bundle, relative to the tts "
             "plugin's static folder.",
             gui=False, admin_only=True),
    TtsField("wasm_paths", Widget.TEXT, "ONNX runtime WASM path",
             "Folder holding the onnxruntime WASM runtime, relative to the tts "
             "plugin's static folder; keep the trailing slash. Blank falls "
             "back to the vendored copy that ships with the plugin.",
             gui=False, admin_only=True),
]


def tts_defaults() -> dict:
    """The tts plugin's shipped template defaults (documentation keys stripped).

    Read through the plugin's own settings module so the template has exactly
    one reader, shared with the plugin's /api/tts/config.
    """
    from localm.plugins.builtin.tts.settings import defaults
    return defaults()


def known_tts_voices() -> list:
    """The shipped voice ids, or [] when the vendored list cannot be read."""
    from localm.plugins.builtin.tts.settings import voice_ids
    return voice_ids()


def tts_admin_only_fields() -> set:
    """Block keys a non-owner ``config:write`` key must not set, and must not
    see the resolved value of either (see tts_schema_json's *is_owner*)."""
    return {f.key for f in TTS_FIELDS if f.admin_only}


def _tts_options(f: "TtsField") -> Optional[list]:
    """The choices for *f*: static, except voice's, which come from the shipped
    voice list (empty list -> None, so the validator falls back to a shape
    check rather than rejecting every voice)."""
    if f.key == "voice":
        return known_tts_voices() or None
    return f.options


def tts_schema_json(block: Optional[dict], *, is_owner: bool = True) -> list:
    """Serialize the tts block's editable fields with their RESOLVED values.

    ``value`` is the block value when the user set one, else the shipped
    template default, so the GUI shows what is actually in effect;
    ``is_override`` says which. ``gui``/``advanced``/``admin_only`` tell the GUI
    what to render and where.

    When *is_owner* is False, admin_only fields (library, wasm_paths) are
    OMITTED entirely (mirrors schema_json's core-schema handling): a non-owner
    config:read caller must not learn the script/wasm path it is not allowed
    to set either. Callers that are not request-scoped (the CLI, tests)
    default to owner (see everything)."""
    block = block if isinstance(block, dict) else {}
    defaults = tts_defaults()
    out = []
    for f in TTS_FIELDS:
        if f.admin_only and not is_owner:
            continue
        own = block.get(f.key)
        has_own = own not in (None, "")
        d = {"key": f.key, "widget": f.widget, "label": f.label, "help": f.help,
             "value": own if has_own else defaults.get(f.key),
             "is_override": has_own, "default": defaults.get(f.key),
             "gui": f.gui, "advanced": f.advanced, "admin_only": f.admin_only}
        options = _tts_options(f)
        if options:
            d["options"] = options
            if f.key == "voice":
                from localm.plugins.builtin.tts.settings import voices
                d["option_labels"] = [v["label"] for v in voices()]
            elif f.key == "model":
                by_repo = dict(TTS_MODEL_OPTIONS)
                d["option_labels"] = [by_repo[o] for o in options]
        for attr in ("min", "max", "step"):
            if getattr(f, attr) is not None:
                d[attr] = getattr(f, attr)
        out.append(d)
    return out


def _tts_relative_asset(key: str, value: str) -> str:
    """Validate a library/wasm_paths value: a real path INSIDE the tts plugin's
    static folder, expressed relatively.

    The browser resolves these against /plugins/tts/, so an absolute, remote, or
    traversing value would make every client load code from somewhere else. The
    target must also EXIST, so a typo raises here instead of silently breaking
    text-to-speech for every client later.
    """
    if ":" in value:
        raise ValueError(
            f"{key}: must be a path relative to the tts plugin's static folder "
            f"(no URL, scheme, or drive letter), got {value!r}")
    if value.startswith("/") or value.startswith("\\") or "\\" in value:
        raise ValueError(
            f"{key}: must be a relative path using forward slashes, got {value!r}")
    parts = value.split("/")
    if ".." in parts:
        raise ValueError(f"{key}: must not step outside the plugin folder, got {value!r}")
    if any(p == "" for p in parts[:-1]):        # a trailing "/" is fine, "a//b" is not
        raise ValueError(f"{key}: has an empty path segment, got {value!r}")
    from localm.plugins.builtin.tts.settings import asset_root
    root = asset_root().resolve()
    target = (root / value).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"{key}: resolves outside the tts plugin's static folder")
    if not target.exists():
        raise ValueError(
            f"{key}: no such file or folder under the tts plugin's static folder "
            f"({value!r}); text-to-speech would fail to load for every browser")
    if key == "library" and not target.is_file():
        raise ValueError(f"{key}: must point at a file, not a folder ({value!r})")
    # The browser requests this path over HTTP, where it is case-SENSITIVE, but
    # Windows and macOS filesystems are not - so an existing-but-differently-spelled
    # path passes the check above and then 404s for every client. Path.resolve()
    # yields the on-disk spelling on those platforms, so comparing tells the user
    # the exact string to use instead of letting text-to-speech quietly break.
    canonical = "." if target == root else target.relative_to(root).as_posix()
    if canonical != value.rstrip("/"):
        raise ValueError(
            f"{key}: write it exactly as it is on disk, {canonical!r} (the browser "
            f"loads this over HTTP, where the path is case-sensitive)")
    return value


def _coerce_tts_value(f: "TtsField", val):
    """Coerce + check one tts field value. Blank/None clears the override (the
    plugin falls back to the shipped template default)."""
    if val is None:
        return None
    if f.widget == Widget.NUMBER:
        if isinstance(val, str) and not val.strip():
            return None
        try:
            num = float(val)
        except (TypeError, ValueError):
            raise ValueError(f"{f.key}: {val!r} is not a number")
        if f.min is not None and num < f.min:
            raise ValueError(f"{f.key}: must be at least {f.min}, got {num}")
        if f.max is not None and num > f.max:
            raise ValueError(f"{f.key}: must be at most {f.max}, got {num}")
        return num
    s = str(val).strip()
    if not s:
        return None
    options = _tts_options(f)
    if f.widget == Widget.SELECT:
        if options and s not in options:
            if f.key == "model":
                if _HF_REPO_ID.match(s):
                    return s
                raise ValueError(
                    f"model: must be a Hugging Face repo id like "
                    f"'onnx-community/Kokoro-82M-v1.0-ONNX', got {val!r}")
            shown = ", ".join(options[:12]) + ("..." if len(options) > 12 else "")
            raise ValueError(f"{f.key}: {val!r} is not one of: {shown}")
        if not options:
            # Only reachable for `voice` when the vendored list is unreadable
            # (already logged): fall back to the id SHAPE so the setting stays
            # usable instead of silently accepting anything.
            if not re.fullmatch(r"[a-z]{2}_[a-z0-9]+", s):
                raise ValueError(f"{f.key}: {val!r} is not a Kokoro voice id")
        return s
    if f.key in ("library", "wasm_paths"):
        return _tts_relative_asset(f.key, s)
    return s


def validate_tts_block(updates: dict) -> dict:
    """Coerce + validate a tts settings update into a flat block-merge dict.

    Raises ValueError on an unknown field or a bad value. A field set to ""
    (blank) is written as None, clearing the override so the plugin falls back
    to the shipped template default.
    """
    by_key = {f.key: f for f in TTS_FIELDS}
    merge: dict = {}
    for key, val in (updates or {}).items():
        f = by_key.get(key)
        if f is None:
            raise ValueError(f"unknown tts setting: {key!r}")
        merge[key] = _coerce_tts_value(f, val)
    return merge


# --------------------------------------------------------------------------- #
#  Generic plugin-contributed settings (host.add_settings()).                 #
#                                                                              #
#  Unlike the media/tts blocks above, this field LIST is not a static module  #
#  constant here - it is supplied at runtime by whichever plugin (built-in or #
#  third-party) called host.add_settings() at register() time (see            #
#  localm/plugins/contract.py's PluginSettingField and                        #
#  PluginManager.get_all_plugin_settings). Every field still lives at         #
#  config["plugins"][<plugin>][key], exactly like the tts block - this is     #
#  just the generic version of tts_schema_json/validate_tts_block, taking the #
#  field list as a parameter instead of a hardcoded TTS_FIELDS.               #
# --------------------------------------------------------------------------- #

def plugin_settings_admin_only_fields(fields) -> set:
    """Field keys (within ONE plugin's add_settings() fields) flagged
    owner-only. Mirrors tts_admin_only_fields/media_admin_only_fields - the
    single source of truth for both the read-side hide (plugin_settings_
    schema_json) and the write-side gate (POST /v1/plugins/<name>/settings)."""
    return {f.key for f in fields if f.admin_only}


def plugin_settings_schema_json(fields, block: Optional[dict], *,
                                is_owner: bool = True) -> list:
    """Serialize one plugin's add_settings() fields with their RESOLVED values
    (the block's own value when set, else the field's own declared default) -
    the generic counterpart to tts_schema_json/media_schema_json.

    When *is_owner* is False, admin_only fields are OMITTED entirely (never
    merely masked), mirroring the tts/media/core schema's owner-only handling:
    a non-owner config:read caller must not learn a value it is not allowed to
    set either. A widget=SECRET field's value/default are never included (the
    widget itself is masked client-side; the value must not round-trip in
    plaintext at all) - derived from the widget alone, so there is no separate
    flag a field could forget to set consistently with it."""
    block = block if isinstance(block, dict) else {}
    out = []
    for f in fields:
        if f.admin_only and not is_owner:
            continue
        own = block.get(f.key)
        has_own = own not in (None, "")
        d = {"key": f.key, "widget": f.widget, "label": f.label, "help": f.help,
             "is_override": has_own, "admin_only": f.admin_only}
        if f.widget != Widget.SECRET:
            d["value"] = own if has_own else f.default
            d["default"] = f.default
        if f.options:
            d["options"] = f.options
        for attr in ("min", "max", "step"):
            v = getattr(f, attr)
            if v is not None:
                d[attr] = v
        out.append(d)
    return out


def _coerce_plugin_field_value(f, val):
    """Coerce + check one plugin-contributed field value against its widget.
    Mirrors _coerce_tts_value/_coerce_media_value's shape, generic over the
    field's `widget` instead of a fixed key list - a third-party plugin's
    keys are not known ahead of time, only the widget vocabulary is."""
    if f.widget == Widget.TOGGLE:
        return _to_bool(f.key, val)
    if val is None:
        return None
    if f.widget == Widget.NUMBER:
        if isinstance(val, str) and not val.strip():
            return None
        return _to_number(f.key, val, want_int=False, lo=f.min, hi=f.max)
    if f.widget == Widget.SELECT:
        s = "" if val is None else str(val).strip()
        if not s:
            return None
        if f.options and s not in f.options:
            shown = ", ".join(f.options[:12]) + ("..." if len(f.options) > 12 else "")
            raise ValueError(f"{f.key}: {val!r} is not one of: {shown}")
        return s
    if f.widget == Widget.LIST:
        return _to_str_list(f.key, val) or None
    # TEXT / TEXTAREA / PATH / FOLDER / SECRET / HIDDEN / PATHLIST: free text.
    # A plugin that needs stronger validation (a path confined to its own
    # asset folder, a shape check) does that itself before calling
    # save_plugin_config with the RESOLVED value, or the plugin author reads
    # this block back through plugin_config() and validates there - this
    # generic layer only guarantees the value round-trips as the widget's
    # basic type, the same floor tts/media give TEXT/FOLDER fields.
    s = str(val).strip()
    return s or None


def validate_plugin_settings_update(fields, updates: dict) -> dict:
    """Coerce + validate a settings update against ONE plugin's add_settings()
    fields. Raises ValueError on an unknown key or a bad value. Mirrors
    validate_tts_block: a field set to "" (blank) is written as None, clearing
    the override so the plugin falls back to its own declared default."""
    by_key = {f.key: f for f in fields}
    merge: dict = {}
    for key, val in (updates or {}).items():
        f = by_key.get(key)
        if f is None:
            raise ValueError(f"unknown setting: {key!r}")
        merge[key] = _coerce_plugin_field_value(f, val)
    return merge


# --------------------------------------------------------------------------- #
#  CLI-facing view of ONE plugin's own settings block                          #
#  (`localm plugin config <name> [<key> [<value>]]`)                           #
#                                                                              #
#  The GUI reaches these blocks through three routes (POST /v1/media/config/   #
#  <name>, POST /v1/tts/config, POST /v1/plugins/<name>/settings). A CLI needs #
#  the same reach, and the awkward part is that those three do NOT share a     #
#  source for their field list:                                                #
#                                                                              #
#    image/music/video  MEDIA_PLUGIN_FIELDS   a module constant, always known  #
#    tts                TTS_FIELDS            a module constant, always known  #
#    anything else      host.add_settings()   supplied at register() time, so  #
#                                             it exists ONLY inside a process  #
#                                             that has LOADED that plugin      #
#                                                                              #
#  A CLI process never loads a plugin: the app-free set_enabled_state /        #
#  set_installed_state / set_installed_from_dir exist so CLI plugin management #
#  never runs a plugin's register() (see their docstrings in                   #
#  plugins/engine.py, and every command in cli/plugins.py uses them). Loading  #
#  one to read its field list is not a free probe either: PluginManager._load  #
#  calls _maybe_fire_first_use, which invokes the on_first_use lifecycle hook  #
#  and writes config for it, so a READ would have a side effect. A no-app load #
#  also fails outright for any plugin that mounts routes (app is None).        #
#                                                                              #
#  So the CLI answers the static two itself, offline, the way `localm config`  #
#  answers the core schema - and for a runtime-declared block it asks a        #
#  RUNNING localm over the existing routes. plugin_config_kind() is the        #
#  discriminator, and its "runtime" answer is a real answer the caller must    #
#  report as such: "this plugin has to be running for its settings to be       #
#  listed" is a different state from "this plugin has no settings", and the    #
#  two are never collapsed into an empty result.                               #
# --------------------------------------------------------------------------- #

#: The two per-plugin media keys that are NOT MediaFields. Both live in the same
#: config["plugins"][<media>] block and both are per-plugin (no global comfy_*
#: fallback), so neither is reachable through `localm config`. Each already has
#: its own rule owned elsewhere, so they are DESCRIBED here for the field
#: listing and DISPATCHED to those owners on write rather than folded into
#: validate_media_block.
#:
#: NOT in MEDIA_PLUGIN_FIELDS: that list is what the GUI's media settings
#: section renders, and `workflow` already has its own GUI affordance (the
#: workflow manager).
MEDIA_EXTRA_KEYS = ("workflow", "use_config_from")


def plugin_config_kind(name: str) -> str:
    """Which source can describe plugin *name*'s settable settings block.

    ``"media"`` / ``"tts"`` - a static schema in this module, readable offline.
    ``"runtime"`` - a host.add_settings() block, known only to a process that
    has actually loaded the plugin, so the caller must ask a running server.
    Note "runtime" is also the answer for a name that is not a plugin at all;
    telling those two apart needs the installed-plugin list, which the caller
    has and this function does not.
    """
    if name in MEDIA_PLUGINS:
        return "media"
    if name == TTS_PLUGIN:
        return "tts"
    return "runtime"


def _media_extra_fields(name: str, block: dict) -> list:
    """Descriptors for MEDIA_EXTRA_KEYS, in media_schema_json's shape."""
    from localm import media_workflows
    others = [p for p in MEDIA_PLUGINS if p != name]
    try:
        choices = [w["name"] for w in media_workflows.list_workflows(name, active=None)]
    except Exception:
        # A listing that could not be MADE is not an empty listing, so omit the
        # options key entirely rather than render "no workflows available". The
        # key stays settable; select_workflow does the real existence check.
        choices = None
    selected = block.get("workflow")
    share = block.get("use_config_from")
    fields = [
        {"key": "workflow", "widget": Widget.SELECT,
         "label": "ComfyUI workflow",
         "help": "Uploaded workflow this plugin generates with. Blank falls back "
                 "to the shipped example template.",
         "value": selected if isinstance(selected, str) and selected else None,
         "is_override": bool(isinstance(selected, str) and selected),
         "global": None},
        {"key": "use_config_from", "widget": Widget.SELECT,
         "label": "Use config from",
         "help": "Reuse another media plugin's backend settings live instead of "
                 "this plugin's own. Blank uses its own.",
         "value": share if isinstance(share, str) and share else None,
         "is_override": bool(isinstance(share, str) and share),
         "global": None, "options": others},
    ]
    if choices is not None:
        fields[0]["options"] = choices
    return fields


def local_plugin_config_fields(name: str, cfg: dict) -> list:
    """Every field ``localm plugin config <name>`` can address, with its
    RESOLVED value, for a plugin whose schema is static.

    Owner view throughout: a CLI caller on this machine IS the owner, which is
    the default media_schema_json / tts_schema_json already document for
    callers that are not request-scoped. Raises ValueError for a
    runtime-declared plugin, which this function structurally cannot describe.
    """
    kind = plugin_config_kind(name)
    plugins = cfg.get("plugins") if isinstance(cfg.get("plugins"), dict) else {}
    block = plugins.get(name) if isinstance(plugins.get(name), dict) else {}
    if kind == "media":
        return media_schema_json(name, block, cfg) + _media_extra_fields(name, block)
    if kind == "tts":
        return tts_schema_json(block)
    raise ValueError(f"{name!r} has no static settings schema")


def local_plugin_config_keys(name: str) -> list:
    """The settable key names for a static-schema plugin, in display order."""
    kind = plugin_config_kind(name)
    if kind == "media":
        return [f.key for f in media_fields_for(name)] + list(MEDIA_EXTRA_KEYS)
    if kind == "tts":
        return [f.key for f in TTS_FIELDS]
    raise ValueError(f"{name!r} has no static settings schema")


def _validate_use_config_from(name: str, value, cfg: dict):
    """Coerce + check a media plugin's share-config pointer. Blank clears it."""
    from localm.plugins import media_config
    if value is None:
        return None
    src = str(value).strip()
    if not src:
        return None
    others = ", ".join(p for p in MEDIA_PLUGINS if p != name)
    if src not in MEDIA_PLUGINS or src == name:
        raise ValueError(f"use_config_from: {value!r} is not one of: {others}")
    # Cycle prevention is the whole reason this key needs a validator rather
    # than being free text: image<-video while video<-image makes resolve_config
    # fall back with a warning on EVERY read instead of failing here, once.
    if media_config.would_cycle(name, src, cfg):
        raise ValueError(
            f"use_config_from: {src!r} already takes its config from {name!r} "
            f"(directly or through another plugin), which would be a cycle")
    return src


def _deep_merge_block(dst: dict, src: dict) -> None:
    """Merge a validated media block into the stored one, nested keys included -
    the same shape POST /v1/media/config/<name> uses, so a CLI write leaves this
    plugin's other fields (and the other plugins) exactly as the GUI would."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge_block(dst[k], v)
        else:
            dst[k] = v


def _own_plugin_block(cfg: dict, name: str) -> dict:
    """This plugin's own block inside *cfg*, created if it is not there yet."""
    plugins = cfg.get("plugins")
    if not isinstance(plugins, dict):
        plugins = cfg["plugins"] = {}
    block = plugins.get(name)
    if not isinstance(block, dict):
        block = plugins[name] = {}
    return block


def _own_block_mutator(name: str, apply):
    """update_config callback that hands *apply* this plugin's own block."""
    def _mutate(cfg: dict) -> None:
        apply(_own_plugin_block(cfg, name))
    return _mutate


def apply_local_plugin_config(name: str, key: str, value) -> tuple:
    """Validate and PERSIST one field of a static-schema plugin's block.

    Returns ``(key, stored_value)``, where a stored None means the override was
    CLEARED - back to the shared global comfy_* default for a media field, or to
    the shipped template default for a tts one. Raises ValueError with a usable
    message on an unknown key or a bad value; the caller reports it and exits
    non-zero.

    Each key is dispatched to whichever writer already owns its rule, so there
    stays exactly one definition of each: validate_media_block /
    validate_tts_block for the schema fields (the same validators the HTTP
    routes call), and media_workflows.select_workflow for `workflow` (which owns
    both the does-this-file-exist check and the pop-on-clear that the GUI's own
    selection route relies on).
    """
    from localm.config import update_config
    kind = plugin_config_kind(name)
    if kind == "runtime":
        raise ValueError(f"{name!r} has no static settings schema")
    known = local_plugin_config_keys(name)
    if key not in known:
        raise ValueError(f"unknown setting for {name!r}: {key!r}. "
                         f"Settable keys: {', '.join(known)}")

    if kind == "media" and key == "workflow":
        from localm import media_workflows
        chosen = media_workflows.select_workflow(
            name, str(value).strip() if value is not None else None)
        return key, chosen

    if kind == "media" and key == "use_config_from":
        # The cycle check runs INSIDE the mutator, not against a load_config()
        # read taken before it. update_config holds a cross-process lock across
        # the whole read-modify-write and hands the mutator the config it is
        # about to persist, so checking there is atomic: a concurrent writer
        # cannot slip the other half of a cycle into the window between the
        # look and the write. Raising here also aborts the write, since
        # update_config only reaches its atomic write once the mutator returns.
        seen = {}

        def _share(cfg: dict) -> None:
            block = _own_plugin_block(cfg, name)
            src = _validate_use_config_from(name, value, cfg)
            seen["src"] = src
            if src is None:
                block.pop("use_config_from", None)
            else:
                block["use_config_from"] = src

        update_config(_share)
        return key, seen.get("src")

    if kind == "media":
        merge = validate_media_block(name, {key: value})
        field = {f.key: f for f in media_fields_for(name)}[key]
        stored = _block_get(merge, field.block_path)
        update_config(_own_block_mutator(
            name, lambda block: _deep_merge_block(block, merge)))
        return key, stored

    merge = validate_tts_block({key: value})
    update_config(_own_block_mutator(
        TTS_PLUGIN, lambda block: block.update(merge)))
    return key, merge[key]
