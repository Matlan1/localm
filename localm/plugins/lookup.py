# SPDX-License-Identifier: AGPL-3.0-or-later
"""How code outside a first-party plugin's folder reaches that plugin.

``plugin_api(name)`` returns the plugin's package module (or one of its
submodules) only while the plugin is installed and enabled in the data dir;
otherwise it raises ``PluginUnavailable``. First-party plugins load from the
localm package under their real module name (``localm.plugins.builtin.<name>``),
so the module this returns is the same object the plugin engine loaded.

``bundled_dir(name)`` is the plugin's folder inside the localm package, for
reading the files it ships without running its code. ``shipped_data(name,
submodule)`` imports one of the few plugin modules listed in ``DATA_MODULES``
whatever the plugin's state: each only reads files the plugin ships (stdlib
and the debug logger, no plugin behaviour), for core code that must read them
while the plugin is off (the settings schema validates the tts block before
the plugin is enabled).

Dependency-light: importing this module imports no plugin code.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Optional

PACKAGE = "localm.plugins.builtin"
SOURCE_MARKER = ".localm-source.json"
DATA_MODULES = frozenset({("tts", "settings")})


class PluginUnavailable(LookupError):
    """The named first-party plugin is not installed and enabled."""

    def __init__(self, name: str, reason: str = "") -> None:
        self.plugin = name
        super().__init__(reason or (
            f"the {name} plugin is not installed and enabled "
            f"(localm plugin install {name})"))


def package_root() -> Path:
    """The folder holding the first-party plugins inside the localm package."""
    return Path(__file__).resolve().parent / "builtin"


def bundled_dir(name: str) -> Optional[Path]:
    """The first-party plugin *name*'s folder in the localm package, or None
    when *name* is not a first-party plugin shipped with this build."""
    from localm.plugins import catalog
    from localm.plugins.ids import _is_valid_plugin_name
    if not _is_valid_plugin_name(name) or catalog.get(name) is None:
        return None
    d = package_root() / name
    return d if (d / "plugin.toml").is_file() else None


def read_source_marker(plugin_dir: Path) -> Optional[dict]:
    """The provenance marker of an installed plugin folder, or None when it is
    absent or unreadable."""
    f = Path(plugin_dir) / SOURCE_MARKER
    try:
        if f.is_file():
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (OSError, ValueError, RecursionError):
        pass
    return None


# Folders whose present but unreadable marker was already logged.
_UNREADABLE_MARKERS_WARNED: set = set()


def installed_from_store(plugin_dir: Path) -> bool:
    """True when the installed folder *plugin_dir* holds a first-party install:
    it has a manifest, and it has no provenance marker (an install made before
    markers existed) or one that does not record a third-party source. A marker
    file that is present but cannot be read or parsed reads False, and is logged
    at WARNING once per folder."""
    d = Path(plugin_dir)
    if not (d / "plugin.toml").is_file():
        return False
    marker_file = d / SOURCE_MARKER
    try:
        present = marker_file.exists()
    except OSError:
        present = True
    if not present:
        return True
    marker = read_source_marker(d)
    if marker is None:
        key = str(d)
        if key not in _UNREADABLE_MARKERS_WARNED:
            _UNREADABLE_MARKERS_WARNED.add(key)
            from localm.debuglog import logger
            logger.warning("plugin folder %s: its provenance marker %s cannot be "
                           "read; the plugin is treated as third-party", d,
                           SOURCE_MARKER)
        return False
    return marker.get("source") in (None, "store")


def is_available(name: str) -> bool:
    """True when first-party plugin *name* ships with this build, is installed
    from the store in the data dir's plugins folder, and is enabled. A
    protected plugin (chat) is always enabled and is never listed in
    ``plugins_enabled``."""
    if bundled_dir(name) is None:
        return False
    from localm.config import load_config
    from localm.plugins import catalog
    from localm.plugins.loader import plugins_dir
    try:
        if not installed_from_store(plugins_dir() / name):
            return False
        if name in catalog.protected():
            return True
        enabled = load_config().get("plugins_enabled", [])
    except OSError:
        return False
    return isinstance(enabled, list) and name in enabled


def plugin_api(name: str, submodule: str = "") -> ModuleType:
    """The first-party plugin *name*'s package module, or its *submodule*
    (dotted, relative to the plugin package), when the plugin is installed and
    enabled.

    Raises ``PluginUnavailable`` when it is not; an ImportError from the
    module itself (a missing pip extra) propagates unchanged."""
    if not is_available(name):
        raise PluginUnavailable(name)
    target = f"{PACKAGE}.{name}" + (f".{submodule}" if submodule else "")
    return importlib.import_module(target)


def shipped_data(name: str, submodule: str) -> ModuleType:
    """The data-only module *submodule* of first-party plugin *name*, imported
    whether or not the plugin is installed or enabled. Raises
    ``PluginUnavailable`` for a module not listed in ``DATA_MODULES``."""
    if (name, submodule) not in DATA_MODULES or bundled_dir(name) is None:
        raise PluginUnavailable(
            name, f"{name}.{submodule} is not a data-only plugin module")
    return importlib.import_module(f"{PACKAGE}.{name}.{submodule}")
