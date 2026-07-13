# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Settings schema: typed metadata for every configurable value.

The legacy settings page renders the raw config dict with no metadata, which is
why it is a flat, unfriendly grid. This module attaches per-field metadata
(widget, label, help, group, allowed values, secret flag, when-it-applies,
owner) so the redesigned page can render the right control - a dropdown for a
fixed set of choices, free text for URLs, a folder picker for directories, a
masked input for secrets - and so each plugin can contribute its own section.

`owner` records which surface a setting belongs to ("core" or a plugin scope).
Plugin-owned core keys (comfy_*, net_*, voice_*, coder_*) migrate
to those plugins in Phase 3 - this field is the migration map.

Phase 0 ships the schema + the core fields. The renderer (GUI) lands in Phase 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# Sanity ceiling for a gpu_split_indices entry: no real machine has anywhere
# near this many GPU devices, so an index above it is a config error (typo or
# malicious PATCH), never a legitimate one. Bounds the ctypes tensor_split
# allocation this value eventually drives in discover.apply_gpu_split - without
# a cap, [0, 500000] would attempt a 500,001-element allocation before the
# native loader is ever invoked. discover.resolve_gpu_split re-applies the same
# ceiling at read time (defense in depth: a hand-edited config.json bypasses
# this write-time check entirely).
#
# Also the ceiling for main_gpu_index: the same "no machine has this many GPU
# devices" reasoning bounds a single device index exactly as much as a list of
# split indices, and both values ultimately land in the same ctypes.c_int32
# main_gpu field (llamacpp/_structs.py) before llama_load_model_from_file.
# discover.resolve_main_gpu_index re-applies this ceiling at read time, same
# defense-in-depth reasoning as gpu_split_indices above.
MAX_GPU_SPLIT_INDEX = 127


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
# Sidebar wordmark treatments (see config.py logo_style). Shared by the web GUI
# logo picker and the desktop launcher; kept here so PATCH /v1/config validates.
LOGO_STYLE_IDS = ["local-m", "loca-lm", "localm"]


@dataclass
class SettingField:
    key: str
    widget: str
    label: str
    help: str = ""
    group: str = "General"
    owner: str = "core"                  # "core" or a plugin scope (see scopes.py)
    options: Optional[list] = None       # for SELECT
    applies: str = Applies.LIVE
    secret: bool = False
    # Owner-only: a non-ADMIN caller may neither SEE this field in the schema nor
    # WRITE it via PATCH /v1/config (it widens a trust boundary). Distinct from
    # `owner` above, which only records the plugin section a setting belongs to.
    admin_only: bool = False
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
    SettingField("binary_dir", Widget.FOLDER, "llama.cpp binary folder",
                 "Folder holding llama.dll / ggml libraries. Blank uses the "
                 "bundled runtime (auto-detected path shown); set it to point at "
                 "a custom build.",
                 group="Engine", applies=Applies.NEXT_LOAD),
    SettingField("n_ctx", Widget.NUMBER, "Context window (initial)",
                 "Tokens of history the model starts with; grows on demand up to "
                 "the maximum below.",
                 group="Engine", applies=Applies.NEXT_LOAD, min=512, step=512),
    SettingField("n_ctx_max", Widget.NUMBER, "Context window (max)",
                 "Largest the context window may grow to (0 = unlimited); bigger "
                 "needs more VRAM.",
                 group="Engine", applies=Applies.NEXT_LOAD, min=0, step=512),
    SettingField("n_ctx_grow", Widget.NUMBER, "Context growth step",
                 "When the window fills, it expands by this many tokens at a time.",
                 group="Engine", applies=Applies.NEXT_LOAD, min=256, step=256),
    SettingField("ctx_auto", Widget.TOGGLE, "Auto-size context from VRAM",
                 "Pick the context ceiling from free GPU memory at load time "
                 "instead of the fixed maximum above.",
                 group="Engine", applies=Applies.NEXT_LOAD),
    SettingField("n_gpu_layers", Widget.NUMBER, "GPU layers",
                 "Model layers to run on the GPU. 99 puts the whole model on the "
                 "GPU; lower it if you run out of VRAM.",
                 group="Engine", applies=Applies.NEXT_LOAD, min=0, max=999),
    SettingField("n_gpu_layers_auto", Widget.TOGGLE, "Auto-size GPU layers from VRAM",
                 "When GPU layers is left at 99 (all), pick how many layers "
                 "actually fit from free VRAM at load: a model too big for the "
                 "GPU runs some layers on CPU (slower) and still loads, instead "
                 "of being refused. An explicit GPU-layers value is left as-is.",
                 group="Engine", applies=Applies.NEXT_LOAD),
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
    SettingField("gpu_split_indices", Widget.HIDDEN, "Split across GPUs",
                 "Device indices to split a model across when it is too large "
                 "for one card (2+ needed to take effect). Blank uses a single "
                 "GPU (Main GPU above).",
                 group="Engine", applies=Applies.NEXT_LOAD),
    SettingField("gpu_split_ratios", Widget.HIDDEN, "GPU split ratios",
                 "Optional relative weight per device in Split across GPUs "
                 "(same order/length). Blank splits evenly.",
                 group="Engine", applies=Applies.NEXT_LOAD),
    SettingField("idle_unload_seconds", Widget.NUMBER, "Idle model unload (s)",
                 "Free the model's VRAM after this many seconds with no request "
                 "(0 = never; the model stays resident). The next message reloads "
                 "it automatically.",
                 group="Engine", min=0, step=30),
    SettingField("gguf_load_timeout_s", Widget.NUMBER, "GGUF load timeout (s)",
                 "How long a GGUF model load may run in its isolated worker "
                 "process before it is treated as hung and cancelled. Raise this "
                 "only if a genuinely huge model on slow storage needs longer "
                 "than the default.",
                 group="Engine", min=10, step=60),
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
    # ---- Server ----
    SettingField("port", Widget.NUMBER, "Server port",
                 "Port the API/GUI server binds to (default 8642); auto-bumps to "
                 "the next free port if busy.",
                 group="Server", applies=Applies.RESTART, min=1, max=65535, step=1),
    SettingField("cors_origins", Widget.TEXT, "CORS origins",
                 "Browser origins allowed to call the API. Blank = localhost "
                 'only; comma-separated list; or "*" for any.',
                 group="Server", applies=Applies.RESTART),
    SettingField("mdns_name", Widget.TEXT, "Network name (mDNS)",
                 "The name other devices use to reach this server on your LAN. "
                 "Sets <name>.local (e.g. localm -> https://localm.local:PORT), so "
                 "there is no IP to type. Letters, digits and hyphens only; also "
                 "added to the HTTPS certificate.",
                 group="Server", applies=Applies.RESTART),
    SettingField("mdns_enabled", Widget.TOGGLE, "Advertise on the network (mDNS)",
                 "Broadcast <name>.local over mDNS/Bonjour when bound past loopback "
                 "so devices can reach localm by name. Off = reachable by IP "
                 "address only. Loopback binds never advertise.",
                 group="Server", applies=Applies.RESTART),
    # ---- Security ----
    SettingField("require_auth", Widget.TOGGLE, "Require an API key",
                 "Refuse all requests until an API key is set (fail closed). "
                 "Required before exposing localm on a network.",
                 group="Security", applies=Applies.RESTART),
    # ---- Interface ----
    # HIDDEN: chosen with the logo picker in the GUI (Settings -> GUI), not a
    # form control. Accepted by PATCH /v1/config so the launcher stays in sync.
    SettingField("logo_style", Widget.HIDDEN, "Logo style",
                 "Sidebar wordmark, chosen with the logo picker and shared with "
                 "the desktop launcher.",
                 group="General"),
    # ---- Privacy ----
    SettingField("mode", Widget.SELECT, "Session persistence",
                 "What localm saves: privacy = nothing written automatically; "
                 "log = a JSONL audit trail; full = log plus a chat transcript.",
                 group="Privacy", options=_PRIVACY),
    SettingField("chat_mode", Widget.SELECT, "Chat persistence override",
                 "Overrides the global persistence for chat only. Blank inherits "
                 "the global mode above.",
                 group="Privacy", owner="chat", options=_PRIVACY_INHERIT),
    SettingField("coder_mode", Widget.SELECT, "Coder persistence override",
                 "Overrides the global persistence for the coder only. Blank "
                 "inherits the global mode.",
                 group="Privacy", owner="coder", options=_PRIVACY_INHERIT),
    SettingField("memory_enabled", Widget.TOGGLE, "Memory recall",
                 "While the memory plugin is enabled, recall the durable facts "
                 "localm has learned about you and add them to the system prompt "
                 "each chat turn. Off = keep the facts but stop recalling them. "
                 "Privacy mode disables memory entirely unless the privacy-recall "
                 "option below is turned on.",
                 group="Privacy", owner="memory"),
    SettingField("memory_auto_consolidate", Widget.TOGGLE, "Grow memory automatically",
                 "After a chat turn, quietly distil durable facts from the "
                 "conversation into memory in the background, so it accumulates "
                 "with no manual step. Always blocked in privacy mode (no new "
                 "traces). Turn this off to grow memory only via the 'Synthesize "
                 "now' button or a scheduled memory job.",
                 group="Privacy", owner="memory"),
    SettingField("memory_recall_in_privacy", Widget.TOGGLE,
                 "Allow memory recall in privacy mode",
                 "Privacy mode normally turns memory off entirely. Turn this on to "
                 "still READ your existing memories into the prompt while in privacy "
                 "mode - writing new memories stays off, so no new trace is ever "
                 "created. Use the per-surface toggles below to choose where.",
                 group="Privacy", owner="memory"),
    SettingField("memory_recall_in_privacy_chat", Widget.TOGGLE,
                 "...in privacy mode: chat",
                 "When the option above is on, recall chat memory during "
                 "privacy-mode chats (read-only).",
                 group="Privacy", owner="memory"),
    SettingField("memory_recall_in_privacy_coder", Widget.TOGGLE,
                 "...in privacy mode: coder",
                 "When the option above is on, recall the coder's past-session "
                 "lessons during privacy-mode coder sessions (read-only).",
                 group="Privacy", owner="memory"),
    SettingField("keep_diagnostics", Widget.TOGGLE,
                 "Keep diagnostics for bug reports",
                 "Privacy mode writes nothing automatically - including the "
                 "crash/hang diagnostics a bug report needs. Turn this on to keep "
                 "those (a hang stack trace, the restart breadcrumb log, and a "
                 "debug log) even in privacy mode, so an intermittent freeze or "
                 "crash leaves something to attach. Code stacks and operational "
                 "logs only, never your chat content.",
                 group="Privacy"),
    # ---- Models ----
    SettingField("embedding_model", Widget.TEXT, "Embedding model",
                 "Small on-device model for semantic search in memory and RAG "
                 "(loaded separately from the chat model). A known key "
                 "(bge-small-en-v1.5, nomic-embed-text-v1.5), a registered model "
                 "name, or a path to a GGUF. Run 'localm setup-embeddings' to fetch "
                 "it; until then, memory/RAG use lexical search.",
                 group="Models", applies=Applies.NEXT_LOAD),
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
    SettingField("auto_install_plugin_deps", Widget.TOGGLE,
                 "Auto-install plugin dependencies",
                 "When you install or enable a plugin that needs extra Python "
                 "packages, install them automatically on this machine. Only the "
                 "local operator triggers this; a remote client never causes a "
                 "server-side install.",
                 group="Plugins"),
    SettingField("plugins_enabled", Widget.HIDDEN, "Enabled plugins",
                 "Names of enabled engine plugins. Managed by the Plugins page "
                 "and `localm plugin enable/disable`, not edited here.",
                 group="Plugins"),
    # ---- Bug-report upload (deployment config) ----
    # HIDDEN: the maintainer sets these in config.json when preparing a tester
    # build (see tools/bugreport-proxy/). Not rendered in the form, so a tester on
    # a shared build does not see or change the proxy URL / shared secret.
    SettingField("bugreport_upload_url", Widget.HIDDEN, "Bug-report upload URL",
                 "Endpoint the in-app 'Send to maintainer' button POSTs the report "
                 "to (the bug-report proxy). Blank = no upload channel.",
                 group="Bug reports"),
    SettingField("bugreport_upload_token", Widget.HIDDEN, "Bug-report upload token",
                 "Optional shared secret the proxy may require, sent as a header. "
                 "Set in config.json, not here.",
                 group="Bug reports"),
    SettingField("update_url", Widget.HIDDEN, "Update endpoint URL",
                 "Override for the updater's Worker base (defaults to the bug-report "
                 "URL when blank). Set in config.json, not here.",
                 group="Bug reports"),
    SettingField("update_token", Widget.HIDDEN, "Update endpoint token",
                 "Override shared secret for the updater (defaults to the bug-report "
                 "token when blank). Set in config.json, not here.",
                 group="Bug reports"),
    SettingField("plugins", Widget.HIDDEN, "Per-plugin config",
                 "Per-plugin settings (e.g. media output dirs). Managed by the "
                 "Plugins/Settings pages and plugin backends, not edited here.",
                 group="Plugins"),
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
    SettingField("coder_tool_grammar", Widget.TOGGLE,
                 "Grammar-constrain coder tool calls",
                 "Once the model starts a <tool_call>, force it to be valid "
                 "tool-call JSON (lazy GBNF grammar; local grammar-capable "
                 "backends only). Free text and thinking are unaffected.",
                 group="Coder", owner="coder", applies=Applies.NEXT_LOAD),
    SettingField("coder_episodic_memory", Widget.TOGGLE,
                 "Coder episodic memory",
                 "Recall lessons from past sessions on a project (what worked / "
                 "what failed) and, at session close, distil the finished session "
                 "into a new lesson. Recall is free; the reflection is one extra "
                 "model call per session that changed files. Skipped in privacy "
                 "mode and for shared keys; stored under the localm data dir, not "
                 "your project. Manage with: localcoder --episodes / --forget-episodes.",
                 group="Coder", owner="coder", applies=Applies.NEXT_LOAD),
    SettingField("coder_untrusted_provenance", Widget.TOGGLE,
                 "Tag untrusted external content",
                 "Mark coder tool results from the web, web search, and external "
                 "MCP servers as untrusted DATA (not instructions) and harden the "
                 "result boundary, so a fetched page cannot inject commands into "
                 "the agent (indirect prompt injection). Defense in depth - it "
                 "blocks nothing. Leave ON unless you have a specific reason.",
                 group="Coder", owner="coder", applies=Applies.NEXT_LOAD),
    SettingField("coder_review", Widget.TOGGLE,
                 "Review changes before finishing",
                 "Before the coder declares a task done, a reviewer model reads the "
                 "diff and feeds blocking issues back for one more fix pass. Adds a "
                 "model round-trip per task that changed files. Off by default.",
                 group="Coder", owner="coder", applies=Applies.NEXT_LOAD),
    SettingField("coder_reviewer", Widget.TEXT,
                 "Reviewer model target",
                 "Who reviews: blank = the agent's own model (local, private); "
                 "'local' = a different small model on CPU in the coder's own process "
                 "(heterogeneous + private; set the model below; adds CPU latency); "
                 "'openai'/'anthropic' = a cloud model; an http(s) URL = a second "
                 "OpenAI-compatible endpoint (e.g. a 2nd local server). A network "
                 "reviewer is skipped in privacy mode and for shared keys (it would "
                 "send the diff off-machine) - those review with the local model.",
                 group="Coder", owner="coder", applies=Applies.NEXT_LOAD),
    SettingField("coder_reviewer_model", Widget.TEXT,
                 "Reviewer model name",
                 "Model name for a cloud/URL reviewer. Blank uses a sensible provider "
                 "default or the agent's own model name.",
                 group="Coder", owner="coder", applies=Applies.NEXT_LOAD),
    # ---- Knowledge (RAG plugin) ----
    # All three are OWNER-ONLY: they define which host folders document indexing
    # may read (a filesystem-read boundary). The localm data folder and credential
    # folders (.ssh, .aws, ...) are refused in BOTH modes regardless - a hard floor.
    SettingField("rag_indexing_mode", Widget.SELECT, "Indexing folder rule",
                 "How localm decides which folders document (RAG) indexing may "
                 "read. whitelist = only your home folder, the working directory, "
                 "and the allowed folders below. blacklist = any folder except the "
                 "denied folders below. Your data folder and credential folders "
                 "(.ssh, .aws, ...) are always refused either way.",
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
    SettingField("rag_classify_unknown_files", Widget.TOGGLE, "Classify unknown files with AI",
                 "Files are tagged with a format (JSON, YAML, ...) for free from their "
                 "extension and content. Turn this on to let the local LLM guess the "
                 "format of an unknown extension it can't tell structurally - only when "
                 "a chat model is loaded. Off, such files are tagged plain text.",
                 group="Knowledge", owner="rag"),
    # ---- Media (ComfyUI: image / music / video plugins) ----
    SettingField("comfy_workdir", Widget.FOLDER, "ComfyUI folder",
                 "Your ComfyUI install folder. localm runs it from here and "
                 "auto-detects a launcher inside. The one setting most setups need.",
                 group="Media", owner="image"),
    SettingField("comfy_launch_cmd", Widget.TEXT, "ComfyUI launch command",
                 "Launcher script (.bat/.sh) that starts ComfyUI. Blank "
                 "auto-detects one in the ComfyUI folder above.",
                 group="Media", owner="image"),
    SettingField("comfy_api_url", Widget.TEXT, "ComfyUI API URL",
                 "Where ComfyUI listens. Blank uses FLUX_API_URL, else "
                 "http://127.0.0.1:8188.",
                 group="Media", owner="image"),
    SettingField("comfy_launch_timeout", Widget.NUMBER,
                 "ComfyUI launch timeout (s)",
                 "Seconds to wait for ComfyUI after launching. A ZLUDA/ROCm cold "
                 "start can take minutes.",
                 group="Media", owner="image", min=30, step=30),
    SettingField("comfy_output_dir", Widget.FOLDER, "ComfyUI output folder",
                 "ComfyUI's own output folder. Only needed if you turn on "
                 "'Remove ComfyUI's copy' below; blank derives it.",
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
    SettingField("comfy_disable_auto_launch", Widget.TOGGLE,
                 "Suppress ComfyUI's web page on launch",
                 "When localm starts ComfyUI for you, keep it headless instead of "
                 "letting it open its own browser tab (appends "
                 "--disable-auto-launch). Off by default. Needs a launcher that "
                 "forwards args to main.py (the stock run_*.bat and 'python "
                 "main.py' do); a launcher that drops args just ignores it.",
                 group="Media", owner="image"),
    SettingField("comfy_func_shim", Widget.TOGGLE,
                 "Fix ComfyUI ACE-Step __func__ crash (in-memory)",
                 "Reactive fix for a known ComfyUI core regression "
                 "(Comfy-Org/ComfyUI #12116) that crashes native ACE-Step audio "
                 "with \"'function' object has no attribute '__func__'\". Off by "
                 "default; localm normally touches nothing. When on, a ComfyUI that "
                 "localm STARTS gets an in-memory compatibility patch via a "
                 "PYTHONPATH env var - localm writes nothing into your ComfyUI "
                 "install and never patches a ComfyUI it did not start. The patch "
                 "self-expires once ComfyUI ships its own fix.",
                 group="Media", owner="image"),
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
    SettingField("managed_comfy_enabled", Widget.TOGGLE,
                 "Use localm's own managed ComfyUI",
                 "Let localm run its OWN ComfyUI (installed under the localm data "
                 "folder) instead of your install. Off by default and inert until "
                 "one is set up - your own ComfyUI is never modified. Set one up "
                 "with 'localm comfy setup', then turn this on to route media to it "
                 "(comfy_target must be 'own').",
                 group="Media", owner="image", applies=Applies.RESTART),
    SettingField("comfy_target", Widget.SELECT, "ComfyUI to use",
                 "When a managed ComfyUI is installed: own = use localm's managed "
                 "instance; user = always use your own ComfyUI. With no managed "
                 "instance, both use your own ComfyUI.",
                 group="Media", owner="image", options=["own", "user"],
                 applies=Applies.RESTART),
    # ---- Network (web plugin) ----
    SettingField("net_mode", Widget.SELECT, "Network access",
                 "Model-initiated web access: off = blocked (all requests fail); "
                 "ask = the GUI prompts before a browser-initiated request (a "
                 "non-browser API/MCP client with web scope is NOT prompted - its "
                 "requests proceed); allow = no prompt. The private-address SSRF "
                 "guard and domain rules apply in every mode.",
                 group="Network", owner="web", options=["off", "ask", "allow"]),
    SettingField("net_allow", Widget.LIST, "Allowed domains",
                 "Domains the model may reach. Empty = any. e.g. example.com "
                 "(also covers *.example.com).",
                 group="Network", owner="web"),
    SettingField("net_deny", Widget.LIST, "Denied domains",
                 "Domains always refused, even if allowed above (deny wins). "
                 "Empty = none.",
                 group="Network", owner="web"),
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
    SettingField("net_search_url", Widget.TEXT, "Search backend URL",
                 "A SearXNG JSON search endpoint for web search. Blank uses "
                 "DuckDuckGo (no key needed).",
                 group="Network", owner="web"),
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
            # as every other number (bool/NaN/inf/overflow rejection). A parallel
            # hand-rolled int(val) here previously drifted from _to_number and
            # leaked an uncaught OverflowError on inf (int(float("inf"))).
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
    if key == "mdns_name":
        # Store the sanitized DNS label, not the raw input, so config.json always
        # holds a valid mDNS name (ASCII letters/digits/hyphens) whatever was typed.
        # Gate on what actually SANITIZES to a usable label - not str.isalnum(),
        # which is Unicode-aware and would let a non-ASCII name (e.g. Greek/CJK)
        # pass the guard and then silently strip to the "localm" default. A name
        # that reduces to nothing is a clear error, not a silent fall back to the
        # default (which would hide the typo).
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
        if isinstance(val, (list, tuple)):
            return _to_str_list(key, val)
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
    out = []
    for f in CORE_FIELDS:
        if f.admin_only and not is_owner:
            continue
        d = f.to_json()
        if not f.secret and f.key in base:
            d["default"] = base[f.key]
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


# Order = display order within each plugin subsection.
MEDIA_PLUGIN_FIELDS: list = [
    MediaField("workdir", ("comfy", "workdir"), "comfy_workdir", Widget.FOLDER,
               "ComfyUI folder",
               "This plugin's ComfyUI install folder. Blank uses the shared default."),
    MediaField("launch_cmd", ("comfy", "launch_cmd"), "comfy_launch_cmd", Widget.TEXT,
               "ComfyUI launch command",
               "Launcher that starts ComfyUI for this plugin. Blank auto-detects one "
               "in the folder."),
    MediaField("api_url", ("comfy", "api_url"), "comfy_api_url", Widget.TEXT,
               "ComfyUI API URL",
               "Where this plugin's ComfyUI listens. Blank uses the shared default."),
    MediaField("output_dir", ("comfy", "output_dir"), "comfy_output_dir", Widget.FOLDER,
               "ComfyUI output folder",
               "Only needed if 'Remove ComfyUI's copy' is on; blank derives it."),
    # R14: dropdowns before checkboxes - the swap_policy SELECT sits ahead of the
    # toggle fields so all dropdowns render before all checkboxes in each subsection.
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


def media_schema_json(name: str, block: Optional[dict], full_config: dict) -> list:
    """Serialize one media plugin's editable fields with their RESOLVED values.

    ``value`` is the per-plugin block value when set, else the global comfy_*
    fallback, so the GUI shows what is actually in effect. ``is_override`` flags
    whether this plugin has its own value (vs inheriting the shared default)."""
    block = block if isinstance(block, dict) else {}
    out = []
    for f in media_fields_for(name):
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
        cur = merge
        for p in f.block_path[:-1]:
            cur = cur.setdefault(p, {})
        cur[f.block_path[-1]] = coerced
    return merge
