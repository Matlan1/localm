# SPDX-License-Identifier: AGPL-3.0-or-later
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
        client_entry=str(s.get("client_entry", "")),
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
        default_enabled=bool(p.get("default_enabled", False)),
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
        self._chat_phases: list = []        # chat-pipeline phases this plugin hooked
        self._static_prefixes: set = set()  # URL prefixes this plugin already serves
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

    def mount_static(self, directory: str, *, url_prefix: str = "") -> str:
        """Serve the plugin's static asset *directory* (relative to the plugin dir)
        read-only at ``/plugins/<name>/`` - or *url_prefix* if given. Returns the
        URL prefix. The mount is tracked like a route so ``unmount`` removes it on
        disable, and relocated before the SPA's "/" catch-all so it matches when
        the plugin is enabled at runtime. Static assets are public (like the SPA
        shell itself) so the browser can ``import()`` the client entry module;
        secrets must live behind scope-gated ``/api`` routes, never here."""
        import mimetypes

        from starlette.staticfiles import StaticFiles
        # ES module import() enforces a JS MIME type; some Windows registries map
        # .js/.mjs to text/plain, which would block a plugin's client module. Pin
        # them (idempotent) so served plugin scripts always load as modules.
        mimetypes.add_type("text/javascript", ".js")
        mimetypes.add_type("text/javascript", ".mjs")
        prefix = "/" + (url_prefix or f"/plugins/{self._spec.name}").strip("/")
        if prefix in self._static_prefixes:
            return prefix                    # already serving this prefix (idempotent)
        base = Path(self._spec.path or ".")
        d = (base / directory).resolve()
        if not d.is_dir():
            raise ValueError(
                f"plugin {self._spec.name!r}: static dir {directory!r} not found")
        before = {id(r) for r in self._app.router.routes}
        self._app.mount(prefix, StaticFiles(directory=str(d)),
                        name=f"plugin-static-{self._spec.name}")
        new = [r for r in self._app.router.routes if id(r) not in before]
        self._relocate_before_catchall(new)
        self._routes += new
        self._static_prefixes.add(prefix)
        return prefix

    def mount_surface_assets(self) -> Optional[str]:
        """Auto-mount the surface's declared ``assets_dir`` at the default
        ``/plugins/<name>`` prefix. The engine calls this right after
        ``register(host)`` so any plugin that declares ``assets_dir`` /
        ``client_entry`` serves its assets without having to call
        ``mount_static`` itself - otherwise the SPA's ``import()`` of the client
        entry 404s silently (api_state already advertises ``assets_base`` for
        such a plugin, so serving it keeps the two in sync). Idempotent: a plugin
        that DID mount the prefix in register() short-circuits here. Best-effort:
        an absent assets_dir is ignored, never fatal."""
        surface = self._spec.surface
        if not surface or not surface.assets_dir:
            return None
        try:
            return self.mount_static(surface.assets_dir)
        except ValueError:
            return None                      # declared but missing on disk

    def unmount(self) -> None:
        for r in self._routes:
            try:
                self._app.router.routes.remove(r)
            except ValueError:
                pass
        self._routes = []
        self._static_prefixes = set()
        self._app.openapi_schema = None
        # Drop any chat-pipeline hooks this plugin registered.
        if self._chat_phases:
            pipeline = getattr(self._app.state, "chat_pipeline", None)
            if pipeline is not None:
                pipeline.remove_plugin(self._spec.name)
            self._chat_phases = []

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

    def register_chat_hook(self, phase: str, fn, *, priority: int = 0) -> None:
        """Register an inlet/stream/outlet transform on the kernel chat pipeline
        (see localm.inference.chat_pipeline). Tracked so unmount drops this
        plugin's hooks when it is disabled or uninstalled."""
        pipeline = getattr(self._app.state, "chat_pipeline", None)
        if pipeline is None:
            # No chat pipeline on this app (e.g. a bare-FastAPI test harness):
            # stay loadable; the hook is simply inert.
            self.audit("chat_hook_skipped", {"phase": phase})
            return
        pipeline.add_hook(phase, fn, priority=priority, plugin=self._spec.name)
        if phase not in self._chat_phases:
            self._chat_phases.append(phase)


# --------------------------------------------------------------------------- #
#  Manager: discovery + lifecycle                                             #
# --------------------------------------------------------------------------- #

_UNSET = object()      # sentinel: distinguish "builtin_root not passed" from "=None"


def _store_root() -> Optional[Path]:
    """The bundled STORE shelf: first-party plugins ship here but are NOT loaded
    from here. Core only reads it to copy a plugin into the installed folder on
    install. (Directory still named 'builtin' on disk.)"""
    d = Path(__file__).resolve().parent / "builtin"
    return d if d.is_dir() else None


class PluginManager:
    """Discovers INSTALLED plugins (in the installed folder) and loads / unloads /
    enables / disables them at runtime on *app*; installs plugins by copying them
    from the bundled store (or their GitHub repo) into the installed folder.

    Two locations: the STORE (bundled shelf, ``store_root``, read only on install)
    and the INSTALLED folder (``installed_root``, the ONLY place discovery/loading
    looks). A plugin not in the installed folder does not exist as far as localm is
    concerned. "Installed" therefore means physically present in installed_root;
    "enabled" is a config toggle within installed; active = installed AND enabled.
    """

    def __init__(self, app, inference_engine=None,
                 store_root: Optional[Path] = None,
                 installed_root: Optional[Path] = None,
                 # back-compat aliases for the old keywords (builtin_root was the
                 # store; external_root was the installed/discovery dir)
                 builtin_root: "Optional[Path] | object" = _UNSET,
                 external_root: Optional[Path] = None) -> None:
        self.app = app
        self.inference_engine = inference_engine
        if store_root is not None:
            self._store_root = store_root
        elif builtin_root is not _UNSET:          # explicit (incl. None = "no store")
            self._store_root = builtin_root
        else:
            self._store_root = _store_root()
        root = installed_root if installed_root is not None else external_root
        if root is not None:
            self._installed_root = root
        else:
            from localm.plugins.loader import plugins_dir
            self._installed_root = plugins_dir()
        self._specs: dict[str, PluginSpec] = {}
        self._loaded: dict[str, tuple] = {}     # name -> (spec, module, host, uniq)
        self._errors: dict[str, str] = {}            # load/runtime errors (persist)
        self._discover_errors: dict[str, str] = {}   # bad manifests (reset each discover)

    # ---- discovery (INSTALLED folder only) ---------------------------------
    def discover(self) -> dict[str, PluginSpec]:
        """Discover INSTALLED plugins only (the installed folder). The store shelf
        is never discovered - it is just the source for install()."""
        self._specs = {}
        self._discover_errors = {}
        root = self._installed_root
        if root:
            try:
                children = sorted(Path(root).glob("*"))
            except OSError:
                children = []
            for child in children:
                if not child.is_dir() or not (child / "plugin.toml").is_file():
                    continue
                try:
                    spec = parse_spec(child, builtin=False)
                    if not spec.compatible():
                        raise ValueError(
                            f"api_version {spec.api_version} != {API_VERSION}")
                    self._specs[spec.name] = spec
                except Exception as e:       # one bad manifest must not break discovery
                    self._discover_errors[child.name] = str(e)
        return self._specs

    def _store_dir(self, name: str) -> Optional[Path]:
        if not self._store_root:
            return None
        d = Path(self._store_root) / name
        return d if (d / "plugin.toml").is_file() else None

    def store_catalog(self) -> dict[str, PluginSpec]:
        """Parse the bundled store shelf (the available first-party plugins). Used
        only to present the catalog / resolve an install source - never loaded."""
        out: dict[str, PluginSpec] = {}
        if not self._store_root:
            return out
        try:
            children = sorted(Path(self._store_root).glob("*"))
        except OSError:
            return out
        for child in children:
            if not child.is_dir() or not (child / "plugin.toml").is_file():
                continue
            try:
                spec = parse_spec(child, builtin=True)
                out[spec.name] = spec
            except Exception:
                pass
        return out

    # ---- installed/enabled state -------------------------------------------
    # "Installed" is PHYSICAL: a plugin is installed iff its directory is present
    # in the installed folder (discoverable). It is NOT a config flag. "Enabled"
    # is a config toggle WITHIN installed (WordPress-style): a plugin is active
    # (loaded) iff installed AND enabled. Load reconciles defensively via the
    # intersection, so a stale 'enabled' entry for a plugin no longer on disk is
    # ignored.
    def _installed_set(self) -> set:
        """Names physically present in the installed folder (have a plugin.toml)."""
        root = self._installed_root
        out = set()
        if root:
            try:
                for child in Path(root).glob("*"):
                    if child.is_dir() and (child / "plugin.toml").is_file():
                        out.add(child.name)
            except OSError:
                pass
        return out

    def _enabled_set(self) -> set:
        from localm.config import load_config
        return set(load_config().get("plugins_enabled", []))

    def _set_enabled(self, name: str, on: bool) -> None:
        """Add/remove *name* in config["plugins_enabled"] atomically (read-modify-
        write under the I/O lock), so concurrent toggles can't lose updates."""
        from localm.config import update_config

        def _mutate(cfg: dict) -> None:
            cur = set(cfg.get("plugins_enabled", []))
            cur.add(name) if on else cur.discard(name)
            cfg["plugins_enabled"] = sorted(cur)
        update_config(_mutate)

    # ---- load / unload (in-process, isolated) ------------------------------
    def load_enabled(self) -> None:
        """Discover INSTALLED plugins and load every active one (installed AND
        enabled). Never raises - a failing plugin is recorded in errors and
        skipped. A 'enabled' config entry for a plugin not on disk is ignored."""
        self._ensure_preinstalled()                # first-run: provision chat etc.
        self.discover()
        enabled = self._enabled_set()
        for name in sorted(self._specs):           # _specs == installed (on disk)
            if name in enabled and name not in self._loaded:
                self._safe_load(self._specs[name])

    def _ensure_preinstalled(self) -> None:
        """First-run provisioning for preinstalled plugins (chat is plugin #0).
        Copy each from the store into the installed folder if absent, and enable
        those marked default_enabled at that first provisioning - so chat ships
        installed + enabled and self-heals if its directory is removed, while a
        user's later disable of a non-protected default_enabled plugin is honoured.
        Best-effort: a missing store source (e.g. a synthetic test store) is
        recorded and skipped, never fatal."""
        from localm.plugins import catalog as _cat
        for name in _cat.preinstalled():
            if (self._installed_dir(name) / "plugin.toml").is_file():
                continue                            # already installed on disk
            try:
                self._provision_from_store(name)
            except Exception as e:                  # no store source / copy failed
                self._discover_errors[name] = f"preinstall: {e}"
                continue
            try:
                spec = parse_spec(self._installed_dir(name))
                if spec.default_enabled and name not in self._enabled_set():
                    self._set_enabled(name, True)
            except Exception:
                pass

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
        # Serve a declared surface assets_dir even if register() did not mount it
        # itself, so a client_entry plugin's module never silently 404s.
        host.mount_surface_assets()
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

    def _invoke_hook(self, name: str, hook_name: str, **kwargs) -> None:
        """Call an optional plugin lifecycle hook if the loaded module defines it.
        Best-effort (mirrors on_uninstall): a hook error never blocks the action."""
        entry = self._loaded.get(name)
        if not entry:
            return
        hook = getattr(entry[1], hook_name, None)
        if callable(hook):
            try:
                hook(**kwargs)
            except Exception:
                pass

    # ---- provisioning helpers ----------------------------------------------
    def _installed_dir(self, name: str) -> Path:
        return Path(self._installed_root) / name

    def _provision_from_store(self, name: str) -> None:
        """Copy the plugin from the bundled store into the installed folder (or,
        if missing from the store, fetch it from its GitHub repo). No-op if it is
        already installed. Raises KeyError when no source exists."""
        import shutil
        dest = self._installed_dir(name)
        if (dest / "plugin.toml").is_file():
            return                                   # already installed on disk
        src = self._store_dir(name)
        if src is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest)
            return
        from localm.plugins import catalog as _cat
        entry = _cat.get(name)
        url = entry.source_url() if entry else ""
        if url:
            raise NotImplementedError(
                f"plugin {name!r} is not in the bundled store; fetching it from "
                f"{url} is not wired up yet")
        raise KeyError(f"no such plugin: {name}")

    def _remove_installed_dir(self, name: str) -> None:
        import shutil
        d = self._installed_dir(name)
        try:
            if d.is_dir():
                shutil.rmtree(d)
        except OSError:
            pass

    def _is_protected(self, name: str) -> bool:
        from localm.plugins import catalog as _cat
        spec = self._specs.get(name)
        return bool(spec and spec.protected) or name in _cat.protected()

    # ---- public lifecycle (install/uninstall = store<->installed) -----------
    def install(self, name: str) -> None:
        """Install a plugin: copy it from the bundled store (or its GitHub repo)
        into the installed folder, then load + enable it on the live app. Rolls
        back the copy if it does not load. KeyError if no such plugin exists."""
        self._provision_from_store(name)             # may raise KeyError
        self.discover()
        if name not in self._specs:
            detail = self._discover_errors.get(name, "bad manifest")
            self._remove_installed_dir(name)
            raise ValueError(f"plugin {name!r} could not be installed: {detail}")
        try:
            if name not in self._loaded:
                self._load(self._specs[name])
        except Exception:
            self._remove_installed_dir(name)         # roll back the copy
            raise
        self._invoke_hook(name, "on_install")        # optional lifecycle hook
        self._set_enabled(name, True)

    def install_external(self, source: Path, *, force: bool = False):
        """Install a THIRD-PARTY plugin from an arbitrary source directory: copy it
        into the installed folder, then load + enable. Rolls back on failure."""
        import shutil
        src = Path(source)
        spec0 = parse_spec(src)                       # validate + name (raises)
        name = spec0.name
        dest = self._installed_dir(name)
        if dest.exists():
            if not force:
                raise ValueError(f"plugin {name!r} is already installed")
            self._remove_installed_dir(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest)
        try:
            self.discover()
            if name not in self._specs:
                detail = self._discover_errors.get(name, "bad manifest")
                raise ValueError(f"plugin {name!r} is not loadable: {detail}")
            if name not in self._loaded:
                self._load(self._specs[name])
        except Exception:
            self._remove_installed_dir(name)
            raise
        self._invoke_hook(name, "on_install")        # optional lifecycle hook
        self._set_enabled(name, True)
        return spec0

    def set_installed_from_dir(self, source: Path, *, force: bool = False,
                               enable: bool = True):
        """CLI/headless install of a THIRD-PARTY plugin from an arbitrary
        directory WITHOUT mounting routes (the app-free sibling of
        ``install_external``, mirroring ``set_installed_state``): validate the
        manifest, copy it into the installed folder, and enable it. Rolls back a
        copy that does not parse. A running GUI server loads it on its next
        start. Returns the parsed PluginSpec."""
        import shutil
        src = Path(source)
        spec0 = parse_spec(src)                       # validate manifest + name (raises)
        name = spec0.name
        dest = self._installed_dir(name)
        if dest.exists():
            if not force:
                raise ValueError(f"plugin {name!r} is already installed")
            self._remove_installed_dir(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest)
        self.discover()
        if name not in self._specs:
            detail = self._discover_errors.get(name, "bad manifest")
            self._remove_installed_dir(name)
            raise ValueError(f"plugin {name!r} could not be installed: {detail}")
        if enable:
            self._set_enabled(name, True)
        return spec0

    def enable(self, name: str) -> None:
        self.discover()
        if name not in self._specs:
            from localm.plugins import catalog as _cat
            if _cat.get(name) or self._store_dir(name):
                raise ValueError(f"plugin {name!r} is not installed; install it first")
            raise KeyError(f"no such plugin: {name}")
        if name not in self._loaded:
            self._load(self._specs[name])             # load first; surface errors
        self._set_enabled(name, True)

    def disable(self, name: str) -> None:
        if self._is_protected(name):
            raise ValueError(f"plugin {name!r} is protected and cannot be disabled")
        self._set_enabled(name, False)
        self._unload(name)

    def is_installed(self, name: str) -> bool:
        return name in self._installed_set()

    def is_enabled(self, name: str) -> bool:
        return name in self._enabled_set()

    def is_active(self, name: str) -> bool:
        """Active (loaded) iff installed (on disk) AND enabled."""
        return name in self._installed_set() and name in self._enabled_set()

    def missing_requires(self, name: str) -> list:
        """Required plugins (declared) that are not currently installed."""
        spec = self._specs.get(name) or self.store_catalog().get(name)
        if not spec:
            return []
        installed = self._installed_set()
        return [r for r in spec.requires if r not in installed]

    def set_installed_state(self, name: str, on: bool, *, enable: bool = True) -> None:
        """CLI/headless install/uninstall WITHOUT loading routes: copy store ->
        installed (or remove the installed dir); the GUI server reconciles via
        load_enabled on its next start. Installing also enables by default;
        uninstalling disables. Honours protection on uninstall."""
        if on:
            self._provision_from_store(name)         # copy store -> installed (raises if unknown)
            self.discover()
            if name not in self._specs:              # copied but unparseable -> roll back
                detail = self._discover_errors.get(name, "bad manifest")
                self._remove_installed_dir(name)
                raise ValueError(f"plugin {name!r} could not be installed: {detail}")
            if enable:
                self._set_enabled(name, True)
        else:
            self.discover()
            if name not in self._installed_set():
                raise KeyError(f"no such plugin: {name}")
            if self._is_protected(name):
                raise ValueError(f"plugin {name!r} is protected and cannot be uninstalled")
            self._set_enabled(name, False)
            self._remove_installed_dir(name)

    def set_enabled_state(self, name: str, on: bool) -> None:
        """CLI/headless enable/disable WITHOUT loading routes. Requires the plugin
        to be installed (on disk); honours protection on disable."""
        self.discover()
        if name not in self._installed_set():
            from localm.plugins import catalog as _cat
            if on and (_cat.get(name) or self._store_dir(name)):
                raise ValueError(f"plugin {name!r} is not installed; install it first")
            raise KeyError(f"no such plugin: {name}")
        if not on and self._is_protected(name):
            raise ValueError(f"plugin {name!r} is protected and cannot be disabled")
        self._set_enabled(name, on)

    def uninstall(self, name: str, *, delete_data: bool = False) -> bool:
        """Uninstall a plugin: unload it, disable it, and DELETE its directory from
        the installed folder (it reverts to being merely available in the store).
        User content is kept unless *delete_data*; the on_uninstall hook runs
        first. Returns True if it was installed; KeyError if wholly unknown."""
        self.discover()
        spec = self._specs.get(name)
        was_installed = name in self._installed_set()
        if spec is None and not was_installed:
            raise KeyError(f"no such plugin: {name}")
        if self._is_protected(name):
            raise ValueError(f"plugin {name!r} is protected and cannot be uninstalled")
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
        if delete_data and spec and spec.data_subdir:
            self._delete_plugin_data(spec)
        self._remove_installed_dir(name)             # delete from the installed folder
        return was_installed

    def _delete_plugin_data(self, spec: PluginSpec) -> None:
        import shutil
        import sys
        from localm.config import home_dir
        if not spec.data_subdir:
            return
        # data_subdir comes verbatim from a (possibly third-party) manifest.
        # Resolve it and confine to home_dir: reject traversal ('../models'),
        # absolute escapes, and the home root itself ('.') before any rmtree.
        home = home_dir().resolve()
        d = (home / spec.data_subdir).resolve()
        if d == home or not d.is_relative_to(home):
            print(
                "[localm] refusing to delete plugin data outside the data dir: "
                f"data_subdir={spec.data_subdir!r} -> {d}",
                file=sys.stderr, flush=True,
            )
            return
        try:
            if d.is_dir():
                shutil.rmtree(d)
        except OSError:
            pass

    # ---- state for the API / GUI -------------------------------------------
    def api_state(self) -> dict:
        """Installed plugins (loaded from the installed folder) plus what is
        AVAILABLE to install (the bundled store + the static catalog, minus what
        is installed). Each entry carries installed/enabled/active/available."""
        from localm.plugins import catalog as _cat
        self.discover()
        installed = self._installed_set()
        enabled = self._enabled_set()
        store = self.store_catalog()
        plugins = []

        def _entry(spec, name, *, available):
            cat = _cat.get(name)
            return {
                "name": name,
                "version": spec.version if spec else "",
                "description": (spec.description if spec and spec.description
                                else (cat.description if cat else "")),
                "scope": spec.scope if spec else name,
                "builtin": (name in store) or bool(cat),
                "protected": self._is_protected(name),
                "tab": spec.surface.tab_id if spec and spec.surface else "",
                "label": spec.surface.label if spec and spec.surface else "",
                "icon": spec.surface.icon if spec and spec.surface else "",
                "group": spec.surface.group if spec and spec.surface else "",
                "client_entry": (spec.surface.client_entry
                                 if spec and spec.surface else ""),
                "assets_base": (f"/plugins/{name}"
                                if spec and spec.surface and spec.surface.assets_dir
                                else ""),
                "requires_extras": spec.requires_extras if spec else [],
                "requires": spec.requires if spec else [],
                "extra": cat.extra if cat else "",
                "commands": list(cat.commands) if cat else [],
                "installed": name in installed,
                "enabled": name in enabled,
                "active": (name in installed) and (name in enabled),
                "available": available,
                "loaded": name in self._loaded,
                "error": self._errors.get(name) or self._discover_errors.get(name),
            }

        for name, spec in sorted(self._specs.items()):       # installed
            plugins.append(_entry(spec, name, available=False))
        seen = set(self._specs)
        for name in sorted(set(store) | set(_cat.names())):  # available, not installed
            if name in seen:
                continue
            plugins.append(_entry(store.get(name), name, available=True))
        from localm.config import load_config
        return {"plugins": plugins,
                "errors": {**self._discover_errors, **self._errors},
                "suggest_plugins": bool(load_config().get("suggest_plugins", True))}


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

    @app.post("/api/plugins/{name}/install",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def install_plugin_engine(name: str):
        try:
            manager.install(name)
        except KeyError:
            raise HTTPException(404, f"No such plugin: {name}")
        except Exception as e:
            raise HTTPException(400, f"Install failed: {e}")
        return {"status": "installed", "name": name}

    @app.post("/api/plugins/{name}/uninstall",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def uninstall_plugin_engine(name: str, delete_data: bool = False):
        try:
            manager.uninstall(name, delete_data=delete_data)
        except KeyError:
            raise HTTPException(404, f"No such plugin: {name}")
        except ValueError as e:
            raise HTTPException(409, str(e))
        except Exception as e:
            raise HTTPException(400, f"Uninstall failed: {e}")
        return {"status": "uninstalled", "name": name}

    @app.post("/api/plugins/{name}/enable",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def enable_plugin(name: str):
        try:
            manager.enable(name)
        except KeyError:
            raise HTTPException(404, f"No such plugin: {name}")
        except ValueError as e:
            raise HTTPException(409, str(e))      # e.g. not installed
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
