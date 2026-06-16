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


class Applies:
    LIVE      = "live"        # takes effect immediately
    NEXT_LOAD = "next_load"   # on the next model load
    RESTART   = "restart"     # needs a server restart


_PRIVACY = ["privacy", "log", "full"]
_PRIVACY_INHERIT = ["", "privacy", "log", "full"]   # "" = inherit the global mode


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
    SettingField("binary_dir", Widget.FOLDER, "llama.cpp binary dir",
                 "Folder containing llama.dll / ggml. Blank = auto-detect.",
                 group="Engine", applies=Applies.NEXT_LOAD),
    SettingField("n_ctx", Widget.NUMBER, "Context window (initial)",
                 "Starting context size; grows on demand.",
                 group="Engine", applies=Applies.NEXT_LOAD, min=512, step=512),
    SettingField("n_ctx_max", Widget.NUMBER, "Context window (max)",
                 "Ceiling the window may grow to (0 = unlimited).",
                 group="Engine", applies=Applies.NEXT_LOAD, min=0, step=512),
    SettingField("n_ctx_grow", Widget.NUMBER, "Context growth step",
                 group="Engine", applies=Applies.NEXT_LOAD, min=256, step=256),
    SettingField("ctx_auto", Widget.TOGGLE, "Auto-size context from VRAM",
                 group="Engine", applies=Applies.NEXT_LOAD),
    SettingField("n_gpu_layers", Widget.NUMBER, "GPU layers",
                 "99 = offload everything to GPU.",
                 group="Engine", applies=Applies.NEXT_LOAD, min=0),
    # ---- Sampling ----
    SettingField("temperature", Widget.NUMBER, "Temperature",
                 group="Sampling", min=0, max=2, step=0.05),
    SettingField("top_p", Widget.NUMBER, "top_p",
                 group="Sampling", min=0, max=1, step=0.05),
    SettingField("top_k", Widget.NUMBER, "top_k", group="Sampling", min=0, step=1),
    SettingField("repeat_penalty", Widget.NUMBER, "Repeat penalty",
                 group="Sampling", min=0, step=0.05),
    SettingField("max_tokens", Widget.NUMBER, "Max tokens per reply",
                 group="Sampling", min=1, step=1),
    # ---- Server ----
    SettingField("port", Widget.NUMBER, "Server port",
                 "Default 8642; auto-bumps if busy.",
                 group="Server", applies=Applies.RESTART, min=1, max=65535, step=1),
    SettingField("cors_origins", Widget.TEXT, "CORS origins",
                 'Blank = localhost only; a comma list of origins; or "*".',
                 group="Server", applies=Applies.RESTART),
    # ---- Security ----
    SettingField("require_auth", Widget.TOGGLE, "Require an API key",
                 "Refuse requests until a key is configured (fail closed).",
                 group="Security", applies=Applies.RESTART),
    # ---- Privacy ----
    SettingField("mode", Widget.SELECT, "Session persistence",
                 "privacy = nothing saved; log = JSONL audit; full = log + transcript.",
                 group="Privacy", options=_PRIVACY),
    SettingField("chat_mode", Widget.SELECT, "Chat persistence",
                 "Overrides the global mode for chat. Blank = inherit.",
                 group="Privacy", options=_PRIVACY_INHERIT),
    SettingField("coder_mode", Widget.SELECT, "Coder persistence",
                 "Overrides the global mode for the coder. Blank = inherit.",
                 group="Privacy", owner="coder", options=_PRIVACY_INHERIT),
    # ---- Models ----
    SettingField("confirm_remove", Widget.TOGGLE,
                 "Confirm before deleting models", group="Models"),
    SettingField("autoprune_missing_models", Widget.TOGGLE,
                 "Auto-remove registry entries for missing files", group="Models"),
    # ---- Plugins ----
    SettingField("suggest_plugins", Widget.TOGGLE,
                 "Suggest installing a plugin for its command",
                 "When a command belongs to a known but inactive plugin, suggest "
                 "installing it instead of reporting an unknown command.",
                 group="Plugins"),
    # ---- Coder (plugin) ----
    SettingField("coder_confirm_timeout", Widget.NUMBER,
                 "Coder approval timeout (s)", group="Coder", owner="coder",
                 min=0, step=10),
    # ---- ComfyUI (image / music / video plugins) ----
    SettingField("comfy_launch_cmd", Widget.TEXT, "ComfyUI launch command",
                 group="ComfyUI", owner="image"),
    SettingField("comfy_workdir", Widget.FOLDER, "ComfyUI working dir",
                 group="ComfyUI", owner="image"),
    SettingField("comfy_output_dir", Widget.FOLDER, "ComfyUI output dir",
                 group="ComfyUI", owner="image"),
    SettingField("reload_llm_after_imagine", Widget.TOGGLE,
                 "Reload chat model after generating",
                 group="ComfyUI", owner="image"),
    # ---- Network (web plugin) ----
    SettingField("net_mode", Widget.SELECT, "Network access",
                 "off = blocked; ask = per-request approval; allow = no prompt.",
                 group="Network", owner="web", options=["off", "ask", "allow"]),
    SettingField("net_allow", Widget.LIST, "Allowed domains",
                 "Empty = any. e.g. example.com (covers *.example.com).",
                 group="Network", owner="web"),
    SettingField("net_deny", Widget.LIST, "Denied domains",
                 group="Network", owner="web"),
    SettingField("net_allow_private", Widget.TOGGLE,
                 "Allow private/loopback targets (disables the SSRF guard)",
                 group="Network", owner="web"),
    SettingField("net_search_url", Widget.TEXT, "Search backend URL",
                 "SearXNG JSON URL; blank = DuckDuckGo.",
                 group="Network", owner="web"),
    # ---- Voice (plugin) ----
    SettingField("voice_stt_model", Widget.SELECT, "Speech-to-text model",
                 group="Voice", owner="voice",
                 options=["tiny", "base", "small", "medium"]),
    SettingField("voice_stt_language", Widget.TEXT, "STT language",
                 "Blank = auto-detect; or en, de, ...",
                 group="Voice", owner="voice"),
    # ---- Abliterate (plugin) ----
    SettingField("heretic_path", Widget.FOLDER, "Heretic checkout path",
                 group="Advanced", owner="abliterate"),
]


def all_widgets() -> set:
    """Every defined widget string."""
    return {v for k, v in vars(Widget).items()
            if not k.startswith("_") and isinstance(v, str)}


def fields_by_owner(owner: str) -> list:
    return [f for f in CORE_FIELDS if f.owner == owner]


def schema_json(values: Optional[dict] = None) -> list:
    """Serialize the core schema, injecting each non-secret field's current
    default from DEFAULT_CONFIG (or *values* if given). The GUI renders this."""
    from localm.config import DEFAULT_CONFIG
    base = DEFAULT_CONFIG if values is None else values
    out = []
    for f in CORE_FIELDS:
        d = f.to_json()
        if not f.secret and f.key in base:
            d["default"] = base[f.key]
        out.append(d)
    return out
