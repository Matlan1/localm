"""
The plugin engine: discovers, loads, enables/disables, and installs plugins at
runtime - without restarting the server or reloading the model.

Everything except chat + the model manager is a plugin (first-party in-tree, or
third-party under ~/.localm/plugins). A plugin ships a ``plugin.toml`` manifest
and a module exposing ``register(host)`` / ``unregister()``; the engine hands it
a `PluginHost` to attach routes, a GUI tab, and settings. The host mounts the
plugin's routes onto the live FastAPI app with the plugin's capability scope
applied, and removes them again on disable - so toggling a plugin is instant.

Phase 2 builds the machinery and the management API; the bundled features become
first-party plugins in Phase 3.
"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path
from typing import Any, Optional

from localm.plugins.contract import API_VERSION, PluginSpec, Surface


# --------------------------------------------------------------------------- #
#  Manifest parsing (the richer contract; superset of loader.PluginManifest)  #
# --------------------------------------------------------------------------- #

def parse_spec(plugin_dir: Path, *, builtin: bool = False) -> PluginSpec:
    """Parse a plugin.toml in *plugin_dir* into a PluginSpec. Raises ValueError
    on an invalid manifest."""
    manifest = plugin_dir / "plugin.toml"
    if not manifest.is_file():
        raise ValueError(f"no plugin.toml in {plugin_dir}")
    try:
        data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"invalid TOML in {manifest}: {e}") from e

    p = data.get("plugin")
    if not isinstance(p, dict):
        raise ValueError(f"{manifest}: missing [plugin] table")
    name = p.get("name", "")
    if not name or not isinstance(name, str) or not name.replace("-", "_").isidentifier():
        raise ValueError(f"{manifest}: invalid or missing plugin name")

    s = data.get("surface", {}) if isinstance(data.get("surface"), dict) else {}
    surface = Surface(
        tab_id=str(s.get("tab_id", "")),
        label=str(s.get("label", "")),
        icon=str(s.get("icon", "")),
        assets_dir=str(s.get("assets_dir", "")),
        settings_group=str(s.get("settings_group", "")),
        group=str(s.get("group", "")),
    )
    return PluginSpec(
        name=name,
        version=str(p.get("version", "0.0.0")),
        api_version=int(p.get("api_version", API_VERSION)),
        description=str(p.get("description", "")),
        scope=str(p.get("scope", "") or name),
        requires_extras=list(p.get("requires_extras", []) or []),
        requires=list(p.get("requires", []) or []),
        capabilities=list(p.get("capabilities", []) or []),
        data_subdir=str(p.get("data_subdir", "")),
        builtin=builtin,
        protected=bool(p.get("protected", False)),
        surface=surface,
        cli_entry=str(p.get("cli", "")),
        register_entry=str(p.get("register", "")),
        path=str(plugin_dir),
    )


def _import_module(spec: PluginSpec):
    """Import the plugin's module fresh from its directory. The module name in
    register_entry is '<module>' or '<module>:<attr>'; we import <module>.py (or
    <module>/__init__.py) and return (module, register_attr_name)."""
    entry = spec.register_entry or "plugin"
    mod_name, _, attr = entry.partition(":")
    attr = attr or "register"
    base = Path(spec.path)
    mod_file = base / f"{mod_name}.py"
    if not mod_file.is_file():
        pkg = base / mod_name / "__init__.py"
        if pkg.is_file():
            mod_file = pkg
        else:
            raise ValueError(f"plugin {spec.name!r}: module {mod_name!r} not found in {base}")
    uniq = f"_localm_plugin_{spec.name.replace('-', '_')}"
    importlib_spec = importlib.util.spec_from_file_location(
        uniq, mod_file, submodule_search_locations=[str(mod_file.parent)])
    if importlib_spec is None or importlib_spec.loader is None:
        raise ValueError(f"plugin {spec.name!r}: cannot create import spec")
    module = importlib.util.module_from_spec(importlib_spec)
    sys.modules[uniq] = module
    try:
        importlib_spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(uniq, None)
        raise
    return module, attr, uniq


# --------------------------------------------------------------------------- #
#  Host: the API a plugin uses to attach itself                               #
# --------------------------------------------------------------------------- #

class PluginHost:
    """Concrete `contract.Host`. One per loaded plugin; tracks what it mounted
    so it can be cleanly removed on unload."""

    def __init__(self, app, manager: "PluginManager", spec: PluginSpec) -> None:
        self.api_version = API_VERSION
        self._app = app
        self._manager = manager
        self._spec = spec
        self._routes: list = []
        self.settings: list = []
        self.surface: Optional[Surface] = spec.surface or None

    def mount_router(self, router) -> None:
        """Mount *router* on the live app, gating every route with the plugin's
        capability scope, and remember the routes added for later removal."""
        from fastapi import Depends
        from localm.inference.http_server import require_scope
        before = {id(r) for r in self._app.router.routes}
        self._app.include_router(
            router, dependencies=[Depends(require_scope(self._spec.scope))])
        new = [r for r in self._app.router.routes if id(r) not in before]
        # include_router appends, but the GUI mounts a catch-all StaticFiles at
        # "/" which would shadow anything added after it (Starlette returns the
        # first matching route, and a "/" Mount matches every path). When a
        # plugin is enabled at runtime - after the GUI mounted "/" - relocate
        # its routes just before that catch-all so they actually match.
        self._relocate_before_catchall(new)
        self._routes += new
        self._app.openapi_schema = None      # force /openapi.json to regenerate

    def _relocate_before_catchall(self, new: list) -> None:
        from starlette.routing import Mount
        routes = self._app.router.routes
        idx = next((i for i, r in enumerate(routes)
                    if isinstance(r, Mount) and r.path in ("", "/")), None)
        if idx is None:
            return                            # no catch-all mount; append is fine
        new_ids = {id(r) for r in new}
        if any(id(r) in new_ids for r in routes[:idx]):
            return                            # already before the catch-all
        for r in new:
            routes.remove(r)
        idx = next(i for i, r in enumerate(routes)
                   if isinstance(r, Mount) and r.path in ("", "/"))
        for j, r in enumerate(new):
            routes.insert(idx + j, r)

    def unmount(self) -> None:
        for r in self._routes:
            try:
                self._app.router.routes.remove(r)
            except ValueError:
                pass
        self._routes = []
        self._app.openapi_schema = None

    def add_settings(self, fields: list) -> None:
        self.settings.extend(fields)

    def register_tab(self, surface: Surface) -> None:
        self.surface = surface

    def plugin_config(self, name: Optional[str] = None) -> dict:
        from localm.config import load_config
        return dict(load_config().get("plugins", {}).get(name or self._spec.name, {}))

    def save_plugin_config(self, name: str, cfg: dict) -> None:
        from localm.config import load_config, save_config
        c = load_config()
        c.setdefault("plugins", {})[name] = cfg
        save_config(c)

    def has_scope(self, scope: str) -> bool:
        # Per-request scope checks happen at the route dependency level
        # (require_scope on the mounted routes). The host has no request context,
        # so this convenience always allows; never use it as a security gate.
        return True

    def require_scope(self, scope: str) -> None:
        return None

    def engine(self) -> Any:
        return self._manager.inference_engine

    def audit(self, event: str, data: dict) -> None:
        try:
            from localm.debuglog import logger as _dbg
            _dbg.debug("plugin %s: %s %s", self._spec.name, event, data)
        except Exception:
            pass

    def browse_dirs(self, path: str) -> dict:
        """Server-side folder-picker helper: immediate subdirectories of *path*
        (blank -> the user's home). Hidden dirs are omitted."""
        base = Path(path).expanduser() if path else Path.home()
        dirs = []
        try:
            for child in sorted(base.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    dirs.append(child.name)
        except OSError:
            pass
        parent = "" if base.parent == base else str(base.parent)
        return {"path": str(base), "parent": parent, "dirs": dirs}


# --------------------------------------------------------------------------- #
#  Manager: discovery + lifecycle                                             #
# --------------------------------------------------------------------------- #

def _builtin_root() -> Optional[Path]:
    d = Path(__file__).resolve().parent / "builtin"
    return d if d.is_dir() else None


class PluginManager:
    """Discovers plugins (first-party in-tree + third-party), and loads /
    unloads / enables / disables / installs them at runtime on *app*."""

    def __init__(self, app, inference_engine=None,
                 builtin_root: Optional[Path] = None,
                 external_root: Optional[Path] = None) -> None:
        self.app = app
        self.inference_engine = inference_engine
        self._builtin_root = builtin_root if builtin_root is not None else _builtin_root()
        if external_root is not None:
            self._external_root = external_root
        else:
            from localm.plugins.loader import plugins_dir
            self._external_root = plugins_dir()
        self._specs: dict[str, PluginSpec] = {}
        self._loaded: dict[str, tuple] = {}     # name -> (spec, module, host, uniq)
        self._errors: dict[str, str] = {}            # load/runtime errors (persist)
        self._discover_errors: dict[str, str] = {}   # bad manifests (reset each discover)

    # ---- discovery ---------------------------------------------------------
    def discover(self) -> dict[str, PluginSpec]:
        self._specs = {}
        self._discover_errors = {}
        for root, builtin in ((self._builtin_root, True), (self._external_root, False)):
            if not root:
                continue
            try:
                children = sorted(Path(root).glob("*"))
            except OSError:
                continue
            for child in children:
                if not child.is_dir() or not (child / "plugin.toml").is_file():
                    continue
                try:
                    spec = parse_spec(child, builtin=builtin)
                    if not spec.compatible():
                        raise ValueError(
                            f"api_version {spec.api_version} != {API_VERSION}")
                    self._specs[spec.name] = spec
                except Exception as e:       # one bad manifest must not break discovery
                    self._discover_errors[child.name] = str(e)
        return self._specs

    # ---- enabled-state (persisted in config) -------------------------------
    def _enabled_set(self) -> set:
        from localm.config import load_config
        return set(load_config().get("plugins_enabled", []))

    def _set_enabled(self, name: str, on: bool) -> None:
        from localm.config import load_config, save_config
        cfg = load_config()
        cur = set(cfg.get("plugins_enabled", []))
        if on:
            cur.add(name)
        else:
            cur.discard(name)
        cfg["plugins_enabled"] = sorted(cur)
        save_config(cfg)

    # ---- load / unload (in-process, isolated) ------------------------------
    def load_enabled(self) -> None:
        """Discover and load every enabled plugin. Never raises - a failing
        plugin is recorded in errors and skipped."""
        self.discover()
        for name in self._enabled_set():
            if name in self._specs and name not in self._loaded:
                self._safe_load(self._specs[name])

    def _safe_load(self, spec: PluginSpec) -> None:
        try:
            self._load(spec)
        except Exception as e:
            self._errors[spec.name] = f"load failed: {e}"

    def _load(self, spec: PluginSpec) -> None:
        module, attr, uniq = _import_module(spec)
        register = getattr(module, attr, None)
        if not callable(register):
            sys.modules.pop(uniq, None)
            raise ValueError(f"plugin {spec.name!r}: no callable {attr!r}")
        host = PluginHost(self.app, self, spec)
        register(host)
        self._loaded[spec.name] = (spec, module, host, uniq)
        self._errors.pop(spec.name, None)       # a successful load clears prior error

    def _unload(self, name: str) -> None:
        entry = self._loaded.pop(name, None)
        if not entry:
            return
        spec, module, host, uniq = entry
        try:
            unreg = getattr(module, "unregister", None)
            if callable(unreg):
                unreg()
        except Exception:
            pass            # teardown errors must not block the unmount
        host.unmount()
        sys.modules.pop(uniq, None)     # drop so a re-enable re-imports fresh

    # ---- public lifecycle --------------------------------------------------
    def enable(self, name: str) -> None:
        if name not in self._specs:
            self.discover()
        if name not in self._specs:
            raise KeyError(f"no such plugin: {name}")
        self._set_enabled(name, True)
        if name not in self._loaded:
            self._load(self._specs[name])     # surface load errors to the caller

    def disable(self, name: str) -> None:
        spec = self._specs.get(name)
        if spec and spec.protected:
            raise ValueError(f"plugin {name!r} is protected and cannot be disabled")
        self._set_enabled(name, False)
        self._unload(name)

    def is_enabled(self, name: str) -> bool:
        return name in self._enabled_set()

    def missing_requires(self, name: str) -> list:
        """Plugins that *name* declares it requires but which are not enabled."""
        if name not in self._specs:
            self.discover()
        spec = self._specs.get(name)
        if not spec:
            return []
        enabled = self._enabled_set()
        return [r for r in spec.requires if r not in enabled]

    def set_enabled_state(self, name: str, on: bool) -> None:
        """Flip a plugin's enabled flag in config WITHOUT loading/unloading routes.
        For CLI/headless use, where there is no live app to mount onto (the GUI
        server picks the state up via load_enabled on its next start). Validates
        the name and honours protection on disable."""
        if name not in self._specs:
            self.discover()
        if name not in self._specs:
            raise KeyError(f"no such plugin: {name}")
        spec = self._specs.get(name)
        if not on and spec and spec.protected:
            raise ValueError(f"plugin {name!r} is protected and cannot be disabled")
        self._set_enabled(name, on)

    def install(self, source: Path, *, force: bool = False):
        """Install a third-party plugin from a directory (admin-gated at the
        route level). Does not enable it."""
        from localm.plugins.loader import install_plugin
        manifest = install_plugin(Path(source), force=force)
        self.discover()
        return manifest

    def uninstall(self, name: str, *, delete_data: bool = False) -> bool:
        """Uninstall a plugin. User content is kept unless *delete_data* is set;
        a plugin's on_uninstall hook (if present) is invoked first."""
        spec = self._specs.get(name)
        if spec and spec.protected:
            raise ValueError(f"plugin {name!r} is protected and cannot be uninstalled")
        # let the plugin clean up its own scaffolding first
        entry = self._loaded.get(name)
        if entry:
            module = entry[1]
            hook = getattr(module, "on_uninstall", None)
            if callable(hook):
                try:
                    hook(delete_data=delete_data)
                except Exception:
                    pass
        self._unload(name)
        self._set_enabled(name, False)
        from localm.plugins.loader import remove_plugin
        existed = remove_plugin(name)
        if delete_data and spec and spec.data_subdir:
            self._delete_plugin_data(spec)
        return existed

    def _delete_plugin_data(self, spec: PluginSpec) -> None:
        import shutil
        from localm.config import home_dir
        d = home_dir() / spec.data_subdir
        try:
            if d.is_dir() and spec.data_subdir:    # never delete the data root itself
                shutil.rmtree(d)
        except OSError:
            pass

    # ---- state for the API / GUI -------------------------------------------
    def api_state(self) -> dict:
        self.discover()
        enabled = self._enabled_set()
        plugins = []
        for name, spec in sorted(self._specs.items()):
            plugins.append({
                "name": spec.name,
                "version": spec.version,
                "description": spec.description,
                "scope": spec.scope,
                "builtin": spec.builtin,
                "protected": spec.protected,
                "tab": spec.surface.tab_id if spec.surface else "",
                "label": spec.surface.label if spec.surface else "",
                "icon": spec.surface.icon if spec.surface else "",
                "group": spec.surface.group if spec.surface else "",
                "requires_extras": spec.requires_extras,
                "requires": spec.requires,
                "enabled": name in enabled,
                "loaded": name in self._loaded,
                "error": self._errors.get(name) or self._discover_errors.get(name),
            })
        return {"plugins": plugins,
                "errors": {**self._discover_errors, **self._errors}}


# --------------------------------------------------------------------------- #
#  Wire the engine + its management API onto a FastAPI app                    #
# --------------------------------------------------------------------------- #

def attach_engine(app, inference_engine=None) -> PluginManager:
    """Instantiate a PluginManager bound to *app*, load enabled plugins, and add
    the management endpoints. Returns the manager."""
    from fastapi import Depends, HTTPException
    from localm import scopes
    from localm.inference.http_server import require_scope

    manager = PluginManager(app, inference_engine=inference_engine)
    manager.load_enabled()
    app.state.plugin_manager = manager

    @app.get("/api/plugins", dependencies=[Depends(require_scope(scopes.PLUGINS_READ))])
    async def list_plugins_engine():
        return manager.api_state()

    @app.post("/api/plugins/{name}/enable",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def enable_plugin(name: str):
        try:
            manager.enable(name)
        except KeyError:
            raise HTTPException(404, f"No such plugin: {name}")
        except Exception as e:
            raise HTTPException(400, f"Enable failed: {e}")
        return {"status": "enabled", "name": name}

    @app.post("/api/plugins/{name}/disable",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def disable_plugin(name: str):
        try:
            manager.disable(name)
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"status": "disabled", "name": name}

    return manager
