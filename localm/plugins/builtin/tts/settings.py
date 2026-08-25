# SPDX-License-Identifier: AGPL-3.0-or-later
"""The tts plugin's own settings DATA: the shipped defaults and the voice ids."""

from __future__ import annotations

import json
from pathlib import Path

from localm.debuglog import logger

_PLUGIN_DIR = Path(__file__).resolve().parent
_TEMPLATE = _PLUGIN_DIR / "tts.example.json"
_VOICES = _PLUGIN_DIR / "static" / "vendor" / "voices.json"


def defaults() -> dict:
    """Shipped defaults from the tracked template (sans documentation keys)."""
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
    """The shipped Kokoro voices as ``[{'id', 'label'}, ...]``, in file order."""
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
    """The static asset directory the BROWSER is served from."""
    try:
        from localm.config import home_dir
        installed = Path(home_dir()) / "plugins" / "tts" / "static"
        if installed.is_dir():
            return installed
    except Exception as e:
        # Resolving the data directory is best effort here: the in-tree copy is
        # byte-identical in a normal install, so falling back to it keeps the
        # validator working rather than failing every write. Logged, not hidden.
        logger.debug("tts: could not resolve the installed asset dir (%s); "
                     "validating against the shipped copy", e)
    return _PLUGIN_DIR / "static"
