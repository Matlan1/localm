"""Per-plugin config resolution for the media plugins (image, music, video).

Backend-agnostic: this only manipulates the config dict. It knows nothing about
ComfyUI or any specific backend - each plugin's backend reads the resolved block
and fills in its own defaults.

Each media plugin stores a self-contained block under ``config["plugins"][name]``,
e.g.::

    config["plugins"]["image"] = {
        "backend": "comfy",
        "use_config_from": None,          # opt-in share-config pointer
        "comfy": {"api_url": ..., "launch_cmd": ..., "workdir": ..., "output_dir": ...},
        "reload_llm_after_generate": True,
    }

Share-config ("use config from"): a media plugin may point ``use_config_from`` at
ANOTHER media plugin to reuse its backend settings LIVE - edit the source once and
the sharer follows. The sharer's own block is never mutated, so toggling sharing
off restores its own values untouched. Sharing is cycle-prevented (no image<-video
while video<-image) and falls back to the sharer's own block (with a warning) when
the source is missing/disabled. Applies only to the three media plugins.
"""

from __future__ import annotations

import copy
from typing import Optional

MEDIA_PLUGINS = ("image", "music", "video")


def _plugins(cfg: dict) -> dict:
    p = cfg.get("plugins")
    return p if isinstance(p, dict) else {}


def _own_block(name: str, cfg: dict) -> dict:
    # Deep copy so callers can read (or even mutate) the returned block, including
    # nested sub-blocks like {"comfy": {...}}, without ever touching the stored
    # config. This makes the "never clears/mutates the sharer's own values"
    # guarantee hold even against nested writes.
    blk = _plugins(cfg).get(name)
    return copy.deepcopy(blk) if isinstance(blk, dict) else {}


def _source_of(name: str, cfg: dict) -> Optional[str]:
    src = _own_block(name, cfg).get("use_config_from")
    if isinstance(src, str) and src in MEDIA_PLUGINS and src != name:
        return src
    return None


def would_cycle(name: str, source: str, cfg: dict) -> bool:
    """Would setting ``name.use_config_from = source`` create a cycle? Follows the
    source's own ``use_config_from`` chain back; True if it returns to *name*."""
    if source == name:
        return True
    seen = {name}
    cur: Optional[str] = source
    while cur and cur in MEDIA_PLUGINS:
        if cur in seen:
            return True
        seen.add(cur)
        cur = _source_of(cur, cfg)
    return False


def resolve_config(name: str, cfg: dict) -> tuple[dict, Optional[str]]:
    """Effective stored block for media plugin *name*.

    Applies ``use_config_from`` (one hop, using the source's OWN block) when set,
    enabled, and non-cyclic; otherwise returns *name*'s own block. Returns
    ``(block, warning_or_None)``. The returned block is a deep copy, so the stored
    config is never mutated even if the caller writes to nested sub-blocks.
    """
    own = _own_block(name, cfg)
    src = own.get("use_config_from")
    if not (isinstance(src, str) and src in MEDIA_PLUGINS and src != name):
        return own, None

    enabled = set(cfg.get("plugins_enabled", []) or [])
    if src not in _plugins(cfg) or src not in enabled:
        return own, (f"'{name}' is set to use '{src}' config, but '{src}' is not "
                     f"available; using its own settings.")
    if would_cycle(name, src, cfg):
        return own, (f"'{name}' is set to use '{src}' config, but that forms a "
                     f"cycle; using '{name}' own settings.")

    resolved = _own_block(src, cfg)
    resolved["use_config_from"] = src      # keep the marker so the UI can show it
    return resolved, None
