# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Legacy plugin manifest discovery, plus small constants shared with the plugin
engine (``localm/plugins/engine.py``).

The engine's ``register = "plug"`` contract (``PluginManager``) is the ONE
install/enable/disable/list mechanism for a plugin's server surface - it used
to share this module with a second, independent ``entry = "<module>:<attr>"``
CLI-manifest mechanism, but that half (install/remove, the ``/v1/plugins`` HTTP
API, and the ``plugin list``/``plugin remove`` CLI verbs) was dead for every
shipped plugin and has been removed (see PATHFINDER-2026-07-11).

What remains here is still live: a third-party plugin's ``[tools] exports =
[...]`` manifest section (this module's own ``discover_plugins``/
``import_plugin_module``) is how ``localm/plugins/coder/plugin_tools.py``
discovers and loads externally-exported coder-agent tools - a distinct
capability from the engine's server-surface registration, unrelated to the
CF-1/CF-2 install/enable/disable duplication that motivated the cut above.
``plugins_dir()`` and ``_RESERVED_NAMES`` are also read directly by
``engine.py``/``media_config.py``.

    [plugin]
    name = "myplugin"
    version = "0.1.0"
    description = "What it does"
    entry = "myplugin_cli:main"       # "<module>:<attr>" - only needed for tool exports now

    [tools]                           # tool exports for the coder agent
    exports = ["tool_hello"]

The entry module is imported from the plugin directory itself, so a plugin
is fully self-contained: a folder with a manifest and one or more .py files.
Everything works offline - installation is a local directory copy.
"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from localm.plugins.contract import KNOWN_PLUGIN_KEYS


def plugins_dir() -> Path:
    """Root directory scanned for external plugins."""
    from localm.config import home_dir
    return home_dir() / "plugins"


# ------------------------------------------------------------------ #
#  Manifest                                                            #
# ------------------------------------------------------------------ #

@dataclass
class PluginManifest:
    name: str
    version: str
    description: str
    entry: str                    # "<module>:<attr>"
    path: Path                    # plugin directory
    tool_exports: List[str] = field(default_factory=list)

    @property
    def entry_module(self) -> str:
        return self.entry.split(":", 1)[0]

    @property
    def entry_attr(self) -> str:
        parts = self.entry.split(":", 1)
        return parts[1] if len(parts) == 2 else "main"


class PluginError(Exception):
    """Raised when a plugin manifest is invalid or the plugin fails to load."""


_RESERVED_NAMES = {
    "pull", "run", "serve", "list", "remove", "info", "config",
    "doctor", "coder", "plugin", "imagine", "mcp", "alias", "completion",
    "gui", "benchmark",
}


def parse_manifest(plugin_dir: Path, *,
                   warnings: Optional[List[str]] = None) -> PluginManifest:
    """Parse and validate ``plugin.toml`` in *plugin_dir*. When *warnings* is
    given, non-fatal manifest problems (unknown/misspelled [plugin] keys) are
    appended to it as human-readable strings - surfaced, never escalated: a
    plugin with an unknown key must still load (LM-DA-007)."""
    manifest_path = plugin_dir / "plugin.toml"
    if not manifest_path.is_file():
        raise PluginError(f"No plugin.toml in {plugin_dir}")

    try:
        data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise PluginError(f"Invalid TOML in {manifest_path}: {e}") from e

    plugin = data.get("plugin")
    if not isinstance(plugin, dict):
        raise PluginError(f"{manifest_path}: missing [plugin] table")

    name = plugin.get("name", "")
    entry = plugin.get("entry", "")
    if not name or not isinstance(name, str):
        raise PluginError(f"{manifest_path}: [plugin] name is required")
    if not name.replace("-", "_").isidentifier():
        raise PluginError(f"{manifest_path}: invalid plugin name {name!r}")
    if name in _RESERVED_NAMES:
        raise PluginError(f"{manifest_path}: name {name!r} clashes with a built-in command")
    if not entry or ":" not in entry:
        raise PluginError(f"{manifest_path}: [plugin] entry must be '<module>:<attr>'")

    tools = data.get("tools", {})
    exports = tools.get("exports", []) if isinstance(tools, dict) else []
    if not (isinstance(exports, list) and all(isinstance(t, str) for t in exports)):
        raise PluginError(f"{manifest_path}: [tools] exports must be a list of strings")

    if warnings is not None:
        # KNOWN_PLUGIN_KEYS spans both manifest formats (see contract.py), so a
        # key valid for the engine contract never false-alarms here; only a key
        # known to neither format (a typo) warns.
        unknown = sorted(set(plugin) - KNOWN_PLUGIN_KEYS)
        if unknown:
            warnings.append(
                f"{manifest_path}: unknown [plugin] key(s) ignored: "
                + ", ".join(unknown))

    return PluginManifest(
        name=name,
        version=str(plugin.get("version", "0.0.0")),
        description=str(plugin.get("description", "")),
        entry=entry,
        path=plugin_dir,
        tool_exports=exports,
    )


# ------------------------------------------------------------------ #
#  Discovery and loading                                               #
# ------------------------------------------------------------------ #

def discover_plugins(root: Optional[Path] = None) -> List[PluginManifest]:
    """
    Scan the plugins directory and return manifests for every valid legacy
    (``entry =``) plugin - this is how ``plugin_tools.register_plugin_tools()``
    finds third-party coder-agent tool exports.

    Invalid plugins are skipped silently here - use :func:`discover_errors`
    when you want the reasons.
    """
    manifests, _, _ = _scan(root)
    return manifests


def discover_errors(root: Optional[Path] = None) -> List[str]:
    """Return human-readable errors for plugins that failed validation."""
    _, errors, _ = _scan(root)
    return errors


def discover_warnings(root: Optional[Path] = None) -> List[str]:
    """Non-fatal manifest warnings (unknown/misspelled keys, LM-DA-007) for
    plugins that still parse and load - so a typo does not degrade silently."""
    _, _, warns = _scan(root)
    return warns


def _is_engine_plugin(plugin_dir: Path) -> bool:
    """True for a plugin using the engine contract (``register = ...``) rather
    than the legacy CLI manifest (``entry = "<module>:<attr>"``). Both kinds live
    in the same installed dir, so the legacy loader must IGNORE engine plugins -
    otherwise it reports every engine-installed builtin (coder, image, ...) as a
    broken legacy plugin. Engine plugins are owned by ``engine.PluginManager``."""
    try:
        data = tomllib.loads((plugin_dir / "plugin.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    plugin = data.get("plugin")
    if not isinstance(plugin, dict):
        return False
    return bool(plugin.get("register")) and not plugin.get("entry")


def _scan(root: Optional[Path]) -> tuple[List[PluginManifest], List[str], List[str]]:
    root = root or plugins_dir()
    manifests: List[PluginManifest] = []
    errors: List[str] = []
    warns: List[str] = []
    if not root.is_dir():
        return manifests, errors, warns
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not (child / "plugin.toml").is_file():
            continue
        if _is_engine_plugin(child):
            continue   # owned by the plugin engine, not a legacy CLI plugin
        try:
            manifests.append(parse_manifest(child, warnings=warns))
        except PluginError as e:
            errors.append(str(e))
    return manifests, errors, warns


def import_plugin_module(manifest: PluginManifest):
    """
    Import the plugin's entry module from its directory and return the module
    object. Raises :class:`PluginError` on failure and never leaves a
    half-imported module behind in ``sys.modules``.
    """
    module_name = f"_localm_plugin_{manifest.name.replace('-', '_')}"
    module_file = manifest.path / f"{manifest.entry_module}.py"
    if not module_file.is_file():
        # Allow package-style entries: <module>/__init__.py
        pkg_init = manifest.path / manifest.entry_module / "__init__.py"
        if pkg_init.is_file():
            module_file = pkg_init
        else:
            raise PluginError(
                f"Plugin {manifest.name!r}: entry module "
                f"{manifest.entry_module!r} not found in {manifest.path}"
            )

    spec = importlib.util.spec_from_file_location(
        module_name, module_file,
        submodule_search_locations=[str(module_file.parent)],
    )
    if spec is None or spec.loader is None:
        raise PluginError(f"Plugin {manifest.name!r}: cannot create import spec")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        sys.modules.pop(module_name, None)
        raise PluginError(f"Plugin {manifest.name!r} failed to import: {e}") from e

    return module
