# SPDX-License-Identifier: AGPL-3.0-or-later
"""The tts plugin's own settings DATA: the shipped defaults and the voice ids.

Both live in files that belong to the plugin (the tracked ``tts.example.json``
template and the vendored ``static/vendor/voices.json`` the picker is built
from), and both are needed in two places: the plugin's own ``/api/tts/config``
(what the browser loads) and the settings write surface in
``localm.settings_schema`` (what the GUI edits). This module is the single
reader for both, so the template is never parsed by two different code paths
that could drift.

Dependency-light (stdlib + the debug logger): ``settings_schema`` imports it
lazily from inside its helpers, so importing the core settings schema never pulls
in the plugin package.
"""

from __future__ import annotations

import json
from pathlib import Path

from localm.debuglog import logger

_PLUGIN_DIR = Path(__file__).resolve().parent
_TEMPLATE = _PLUGIN_DIR / "tts.example.json"
_VOICES = _PLUGIN_DIR / "static" / "vendor" / "voices.json"


def defaults() -> dict:
    """Shipped defaults from the tracked template (sans documentation keys).

    Returns {} if the template cannot be read, and logs a warning: it is a TRACKED
    shipped file, so absent is as broken as corrupt, and a silent {} would hide a
    broken install behind the frontend's hardcoded fallbacks.
    """
    try:
        data = json.loads(_TEMPLATE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("tts: could not read the shipped template %s (%s); "
                       "falling back to the frontend's built-in defaults",
                       _TEMPLATE.name, e)
        return {}
    if not isinstance(data, dict):
        logger.warning("tts: the shipped template %s is not a JSON object; "
                       "falling back to the frontend's built-in defaults",
                       _TEMPLATE.name)
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_")}


def voices() -> list:
    """The shipped Kokoro voices as ``[{"id", "label"}, ...]``, in file order.

    The label matches the one the in-chat picker builds client-side (name,
    language, gender, grade), so the same voice reads the same in Settings and
    in chat.

    Returns [] when the vendored list cannot be read; callers must then fall
    back to a SHAPE check instead of an exact-membership one (a missing asset
    must not make every voice unsettable), and this logs why.
    """
    try:
        data = json.loads(_VOICES.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("tts: could not read the vendored voice list %s (%s); "
                       "the voice setting falls back to a shape check",
                       _VOICES.name, e)
        return []
    if not isinstance(data, dict):
        logger.warning("tts: the vendored voice list %s is not a JSON object; "
                       "the voice setting falls back to a shape check",
                       _VOICES.name)
        return []
    out = []
    for vid, meta in data.items():
        meta = meta if isinstance(meta, dict) else {}
        name = str(meta.get("name") or vid)
        bits = [str(meta.get("language") or ""), str(meta.get("gender") or "")]
        if meta.get("grade"):
            bits.append(str(meta["grade"]))
        detail = ", ".join(b for b in bits if b)
        out.append({"id": str(vid), "label": f"{name} ({detail})" if detail else name})
    return out


def voice_ids() -> list:
    """Just the shipped voice ids (see voices())."""
    return [v["id"] for v in voices()]


def asset_root() -> Path:
    """The static asset directory the BROWSER is served from.

    ``library`` / ``wasm_paths`` are resolved by the browser against this
    directory (tts.js uses ``import.meta.url``, which is
    ``/plugins/tts/tts.js``), so the validator has to check a candidate path
    against the tree that is actually mounted. That is the INSTALLED copy under
    the data directory (``PluginHost.mount_static`` roots at the installed
    plugin's path), not this in-tree source: the two are kept hash-identical by
    the builtin refresh, but a user who drops their own onnxruntime WASM folder
    into the installed copy would otherwise be told it does not exist. (The
    plugin VENDORS that runtime under ``static/vendor/onnxruntime/`` and defaults
    ``wasm_paths`` to it, so that is the override case.) Falls back to the shipped
    source when the plugin is not installed, so the setting can still be validated
    before the plugin is added.
    """
    try:
        from localm.config import home_dir
        installed = Path(home_dir()) / "plugins" / "tts" / "static"
        if installed.is_dir():
            return installed
    except Exception as e:
        # Resolving the data directory is best effort here: the in-tree copy is
        # byte-identical in a normal install, so falling back to it keeps the
        # validator working. Logged, not hidden.
        logger.debug("tts: could not resolve the installed asset dir (%s); "
                     "validating against the shipped copy", e)
    return _PLUGIN_DIR / "static"
