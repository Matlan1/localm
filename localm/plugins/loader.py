"""
External plugin discovery and loading.

Plugins live in ``~/.localm/plugins/<name>/`` and are described by a
``plugin.toml`` manifest:

    [plugin]
    name = "myplugin"                 # CLI name: ``localm myplugin``
    version = "0.1.0"
    description = "What it does"
    entry = "myplugin_cli:main"       # "<module>:<attr>" - attr is a Click command

    [tools]                           # optional - tool exports for the agent
    exports = ["tool_hello"]

The entry module is imported from the plugin directory itself, so a plugin
is fully self-contained: a folder with a manifest and one or more .py files.
Everything works offline - installation is a local directory copy.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def plugins_dir() -> Path:
    """Root directory scanned for external plugins."""
    from localm.config import HOME_DIR
    return HOME_DIR / "plugins"


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


def parse_manifest(plugin_dir: Path) -> PluginManifest:
    """Parse and validate ``plugin.toml`` in *plugin_dir*."""
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
    Scan the plugins directory and return manifests for every valid plugin.

    Invalid plugins are skipped silently here - use :func:`discover_errors`
    when you want the reasons (e.g. for ``localm plugin list``).
    """
    manifests, _ = _scan(root)
    return manifests


def discover_errors(root: Optional[Path] = None) -> List[str]:
    """Return human-readable errors for plugins that failed validation."""
    _, errors = _scan(root)
    return errors


def _scan(root: Optional[Path]) -> tuple[List[PluginManifest], List[str]]:
    root = root or plugins_dir()
    manifests: List[PluginManifest] = []
    errors: List[str] = []
    if not root.is_dir():
        return manifests, errors
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not (child / "plugin.toml").is_file():
            continue
        try:
            manifests.append(parse_manifest(child))
        except PluginError as e:
            errors.append(str(e))
    return manifests, errors


def load_entry(manifest: PluginManifest):
    """
    Import the plugin's entry module from its directory and return the
    entry attribute (expected to be a Click command or group).
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

    attr = getattr(module, manifest.entry_attr, None)
    if attr is None:
        raise PluginError(
            f"Plugin {manifest.name!r}: entry attribute "
            f"{manifest.entry_attr!r} not found in {manifest.entry_module}"
        )
    return attr


def register_external_plugins(group) -> List[str]:
    """
    Discover external plugins and add each one's Click command to *group*.

    Returns a list of warning strings for plugins that failed to load.
    Never raises - a broken plugin must not take down the localm CLI.
    """
    warnings: List[str] = []
    existing = set(group.commands) if hasattr(group, "commands") else set()
    for manifest in discover_plugins():
        if manifest.name in existing:
            warnings.append(
                f"Plugin {manifest.name!r} skipped: command name already registered"
            )
            continue
        try:
            cmd = load_entry(manifest)
            group.add_command(cmd, name=manifest.name)
            existing.add(manifest.name)
        except PluginError as e:
            warnings.append(str(e))
        except Exception as e:  # defensive - plugin bugs stay contained
            warnings.append(f"Plugin {manifest.name!r} failed to register: {e}")
    return warnings


# ------------------------------------------------------------------ #
#  Install / remove                                                    #
# ------------------------------------------------------------------ #

def install_plugin(source: Path, *, force: bool = False) -> PluginManifest:
    """
    Install a plugin by copying *source* (a directory containing plugin.toml)
    into the plugins directory. Returns the installed manifest.
    """
    source = source.resolve()
    if not source.is_dir():
        raise PluginError(f"Not a directory: {source}")
    manifest = parse_manifest(source)  # validate before copying

    dest = plugins_dir() / manifest.name
    if dest.exists():
        if not force:
            raise PluginError(
                f"Plugin {manifest.name!r} is already installed "
                f"(use --force to overwrite)"
            )
        shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source, dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"),
    )
    return parse_manifest(dest)


def remove_plugin(name: str) -> bool:
    """Remove an installed plugin by name. Returns True if it existed."""
    target = plugins_dir() / name
    if not target.is_dir():
        return False
    # Refuse to delete anything outside the plugins root
    if target.resolve().parent != plugins_dir().resolve():
        raise PluginError(f"Refusing to remove path outside plugins dir: {target}")
    shutil.rmtree(target)
    return True
