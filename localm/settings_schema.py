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
Plugin-owned core keys (comfy_*, net_*, voice_*, coder_*, heretic_path) migrate
to those plugins in Phase 3 - this field is the migration map.

Phase 0 ships the schema + the core fields. The renderer (GUI) lands in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None

    def to_json(self) -> dict:
        d = {
            "key": self.key, "widget": self.widget, "label": self.label,
            "help": self.help, "group": self.group, "owner": self.owner,
            "applies": self.applies, "secret": self.secret,
        }
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
                 group="Engine", applies=Applies.NEXT_LOAD, min=0, max=1000),
    SettingField("import_max_depth", Widget.NUMBER, "Folder import depth",
                 "Subfolder levels `localm add <dir>` scans for models.",
                 group="Models", min=1, max=10, step=1),
    # ---- Sampling ----
    SettingField("temperature", Widget.NUMBER, "Temperature",
                 "Randomness of replies. Lower is more focused; higher is more "
                 "varied and creative.",
                 group="Sampling", min=0, max=2, step=0.05),
    SettingField("top_p", Widget.NUMBER, "Top-p (nucleus sampling)",
                 "Consider only the top tokens whose probabilities sum to this "
                 "fraction. 1.0 turns it off.",
                 group="Sampling", min=0, max=1, step=0.05),
    SettingField("top_k", Widget.NUMBER, "Top-k",
                 "Consider only the k most likely tokens at each step. 0 turns it off.",
                 group="Sampling", min=0, step=1),
    SettingField("repeat_penalty", Widget.NUMBER, "Repeat penalty",
                 "How strongly to discourage reusing tokens. 1.0 is no penalty; "
                 "higher reduces loops.",
                 group="Sampling", min=0, step=0.05),
    SettingField("max_tokens", Widget.NUMBER, "Max tokens per reply",
                 "Upper limit on tokens per reply, a runaway guard not a target. "
                 "Thinking models need plenty of room.",
                 group="Sampling", min=1, step=1),
    # ---- Server ----
    SettingField("port", Widget.NUMBER, "Server port",
                 "Port the API/GUI server binds to (default 8642); auto-bumps to "
                 "the next free port if busy.",
                 group="Server", applies=Applies.RESTART, min=1, max=65535, step=1),
    SettingField("cors_origins", Widget.TEXT, "CORS origins",
                 "Browser origins allowed to call the API. Blank = localhost "
                 'only; comma-separated list; or "*" for any.',
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
                 group="Privacy", options=_PRIVACY_INHERIT),
    SettingField("coder_mode", Widget.SELECT, "Coder persistence override",
                 "Overrides the global persistence for the coder only. Blank "
                 "inherits the global mode.",
                 group="Privacy", owner="coder", options=_PRIVACY_INHERIT),
    # ---- Models ----
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
    SettingField("plugins_enabled", Widget.HIDDEN, "Enabled plugins",
                 "Names of enabled engine plugins. Managed by the Plugins page "
                 "and `localm plugin enable/disable`, not edited here.",
                 group="Plugins"),
    SettingField("plugins", Widget.HIDDEN, "Per-plugin config",
                 "Per-plugin settings (e.g. media output dirs). Managed by the "
                 "Plugins/Settings pages and plugin backends, not edited here.",
                 group="Plugins"),
    # ---- Coder (plugin) ----
    SettingField("coder_confirm_timeout", Widget.NUMBER,
                 "Coder approval timeout (s)",
                 "Seconds a coder approval card waits for an answer before it is "
                 "auto-rejected and the agent moves on (0 = wait forever).",
                 group="Coder", owner="coder", min=0, step=10),
    SettingField("coder_tool_grammar", Widget.TOGGLE,
                 "Grammar-constrain coder tool calls (experimental)",
                 "Force valid tool-call JSON via a GBNF grammar (grammar-capable "
                 "backend only). Experimental: forces tool-only output, so leave "
                 "off unless you want that.",
                 group="Coder", owner="coder", applies=Applies.NEXT_LOAD),
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
    # ---- Network (web plugin) ----
    SettingField("net_mode", Widget.SELECT, "Network access",
                 "Model-initiated web access: off = blocked; ask = approve each "
                 "request; allow = no prompt.",
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
                 group="Network", owner="web"),
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
    # ---- Abliterate (plugin) ----
    SettingField("heretic_path", Widget.FOLDER, "Heretic checkout path",
                 "Folder of a Heretic checkout for the abliterate plugin (a "
                 "separate program run via subprocess). Blank auto-detects, or "
                 "localm offers to clone it.",
                 group="Advanced", owner="abliterate"),
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
        num = int(val) if want_int else float(val)
    except (TypeError, ValueError):
        kind = "an integer" if want_int else "a number"
        raise ValueError(f"{key}: expected {kind}, got {val!r}")
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

    if widget == Widget.HIDDEN:
        if key == "logo_style":
            s = str(val)
            if s in LOGO_STYLE_IDS:
                return s
            raise ValueError(f"{key}: {val!r} is not one of {LOGO_STYLE_IDS}")
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


def schema_json(values: Optional[dict] = None) -> list:
    """Serialize the core schema, injecting each non-secret field's current
    default from DEFAULT_CONFIG (or *values* if given). The GUI renders this.

    Auto-detect fields also carry an ``auto`` value: the path localm would
    resolve when the field is left blank, so the GUI can SHOW it (filled, greyed)
    instead of an empty box that hides what is actually in use. Today only
    ``binary_dir`` resolves one (the bundled llama.cpp runtime)."""
    from localm.config import DEFAULT_CONFIG
    base = DEFAULT_CONFIG if values is None else values
    out = []
    for f in CORE_FIELDS:
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
