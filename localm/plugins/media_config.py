# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-plugin config resolution for the media plugins (image, music, video)."""

from __future__ import annotations

import copy
from types import SimpleNamespace
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


def active_plugins(cfg: dict) -> set:
    """The set of ACTIVE plugin names = installed (physically present in the installed folder) AND enabled (config). 'Installed' is disk presence in the new store->installed model, NOT a config flag, so we scan the installed folder rather than reading a (no-longer-written) config list."""
    from pathlib import Path

    from localm.debuglog import logger
    from localm.plugins.loader import plugins_dir
    installed = set()
    try:
        for child in Path(plugins_dir()).glob("*"):
            if child.is_dir() and (child / "plugin.toml").is_file():
                installed.add(child.name)
    except OSError as exc:
        # Surface a real disk/permission fault on the plugins-dir scan so it is
        # discoverable under --debug-discoverable; an empty result here does NOT
        # prove no plugins exist, it can just mean the scan itself failed.
        logger.debug("active_plugins: failed to scan plugins dir %s: %s",
                     plugins_dir(), exc)
    enabled = set(cfg.get("plugins_enabled", []) or [])
    return installed & enabled


def would_cycle(name: str, source: str, cfg: dict) -> bool:
    """Would setting ``name.use_config_from = source`` create a cycle? Follows the source's own ``use_config_from`` chain back; True if it returns to *name*."""
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


def resolve_config(name: str, cfg: dict,
                   *, active: "Optional[set]" = None) -> tuple[dict, Optional[str]]:
    """Effective stored block for media plugin *name*."""
    own = _own_block(name, cfg)
    src = own.get("use_config_from")
    if not (isinstance(src, str) and src in MEDIA_PLUGINS and src != name):
        return own, None

    if active is None:
        active = active_plugins(cfg)
    if src not in _plugins(cfg) or src not in active:
        return own, (f"'{name}' is set to use '{src}' config, but '{src}' is not "
                     f"active; using its own settings.")
    if would_cycle(name, src, cfg):
        return own, (f"'{name}' is set to use '{src}' config, but that forms a "
                     f"cycle; using '{name}' own settings.")

    resolved = _own_block(src, cfg)
    resolved["use_config_from"] = src      # keep the marker so the UI can show it
    return resolved, None


# --------------------------------------------------------------------------- #
#  Backend seam (I1): select a media backend implementation by name            #
# --------------------------------------------------------------------------- #
#
# A media backend is a MODULE under ``<plugin>/backends/<name>.py`` exposing
# three callables - the template a non-ComfyUI generator (a remote API, a local
# diffusers/A1111 server, ...) adapts to:
#
#     def ensure_available(s: dict, on_progress=None) -> tuple[bool, str]: ...
#     def free_vram(s: dict) -> bool: ...
#     def generate(s: dict, *args, **kwargs) -> tuple[bool, str]: ...
#
# ``s`` is the dict the plugin's ``backend.settings(config)`` resolved; a backend
# reads what it needs from it. Select a backend per plugin via the ``backend``
# config key (default ``"comfy"``, the reference ComfyUI implementation, which
# each plugin keeps inline). To add one, drop ``backends/<name>.py`` next to the
# plugin and set ``config["plugins"][<plugin>]["backend"] = "<name>"``.

def load_backend(package: str, name: Optional[str]):
    """Import the media backend module *name* from ``<package>.backends``."""
    import importlib
    nm = (name or "comfy").strip().lower() or "comfy"
    return importlib.import_module(f"{package}.backends.{nm}")


def backend_unavailable_warning(package: str, name: Optional[str]) -> Optional[str]:
    """A warning when *name* is a media backend that is not the built-in ``comfy`` reference and cannot be imported (a typo, an unimplemented backend, or one whose dependency is not installed), else None for ``comfy``/empty/loadable."""
    nm = (name or "comfy").strip().lower() or "comfy"
    if nm == "comfy":
        return None
    try:
        load_backend(package, nm)
        return None
    except ModuleNotFoundError:
        return (f"Configured media backend '{nm}' is not available "
                f"(not implemented or its dependency is not installed); "
                f"using ComfyUI instead. Set the backend to 'comfy' to silence this.")


def combine_warnings(*warnings: Optional[str]) -> Optional[str]:
    """Join the non-empty *warnings* with ``'; '``, or None when all are empty."""
    parts = [w for w in warnings if w]
    return "; ".join(parts) if parts else None


def make_backend_facade(package: str, comfy_ref: SimpleNamespace) -> SimpleNamespace:
    """Build the ``ensure_available``/``free_vram``/``generate`` dispatch facade shared by every media plugin's backend.py: *comfy_ref* (the plugin's own inline ComfyUI reference implementation) for the default ``'comfy'`` backend, else ``<package>.backends.<name>`` loaded by ``load_backend`` - falling bac..."""

    def resolve(s: dict):
        name = (s.get("backend") or "comfy").strip().lower()
        if name in ("", "comfy"):
            return comfy_ref
        try:
            return load_backend(package, name)
        except ModuleNotFoundError:
            return comfy_ref

    def ensure_available(s: dict, *args, **kwargs) -> tuple[bool, str]:
        return resolve(s).ensure_available(s, *args, **kwargs)

    def free_vram(s: dict, *args, **kwargs) -> bool:
        return resolve(s).free_vram(s, *args, **kwargs)

    def generate(s: dict, *args, **kwargs) -> tuple[bool, str]:
        return resolve(s).generate(s, *args, **kwargs)

    return SimpleNamespace(resolve=resolve, ensure_available=ensure_available,
                           free_vram=free_vram, generate=generate)
