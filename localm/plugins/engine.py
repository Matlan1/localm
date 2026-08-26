# SPDX-License-Identifier: AGPL-3.0-or-later
"""
The plugin engine: discovers, loads, enables/disables, and installs plugins at
runtime - without restarting the server or reloading the model.

Everything except chat + the model manager is a plugin (first-party in-tree, or
third-party under <data dir>/plugins). A plugin ships a ``plugin.toml`` manifest
and a module exposing ``register(host)`` / ``unregister()``; the engine hands it
a `PluginHost` to attach routes, a GUI tab, and settings. The host mounts the
plugin's routes onto the live FastAPI app with the plugin's capability scope
applied, and removes them again on disable - so toggling a plugin is instant.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import stat
import sys
import threading
import tomllib
from pathlib import Path
from typing import Any, Callable, Optional

from localm.plugins.contract import (API_VERSION, KNOWN_PLUGIN_KEYS,
                                     KNOWN_SURFACE_KEYS, PluginSettingField,
                                     PluginSpec, Surface)

# Logger for the sequential plugin load, one line per plugin.
_log = logging.getLogger("localm.plugins")

# Provenance sidecar written into an installed plugin dir. Records where the
# copy came from ('store' = the bundled first-party shelf, 'external' = a
# third-party source dir) and a content hash of that source at install time.
# Hidden, so discovery (which only walks directories) ignores it.
_PLUGIN_MARKER = ".localm-source.json"


# --------------------------------------------------------------------------- #
#  Path safety: the plugin id, and an untrusted source tree                    #
# --------------------------------------------------------------------------- #

def _is_valid_plugin_name(name: Any) -> bool:
    """True iff *name* is a legal plugin id: ONE path component, shaped like an
    identifier once hyphens are folded to underscores.

    The SAME rule ``parse_spec`` applies to a manifest's ``[plugin] name``, and
    it is enforced at the two places a name becomes a path (``_installed_dir``
    / ``_store_dir``).
    """
    if not name or not isinstance(name, str):
        return False
    # isidentifier() rejects every separator, dot, space and leading digit, so
    # '.', '..', '../x', 'a/b' and a drive-qualified path are all refused; the
    # Path(name).name comparison re-checks that the id is a single path
    # component. A SHAPE check, not a uniqueness one: 'MyTool' names the same
    # directory as 'mytool' on a case-insensitive filesystem.
    return name == Path(name).name and name.replace("-", "_").isidentifier()


def _check_plugin_name(name: str) -> str:
    """Return *name* if it is a legal plugin id, else raise ValueError."""
    if not _is_valid_plugin_name(name):
        raise ValueError(f"invalid plugin name: {name!r}")
    return name


# Reparse tags that make one path stand in for another. stat exports these
# names on Windows only, so getattr supplies the literal values elsewhere.
_ALIASING_REPARSE_TAGS = frozenset((
    getattr(stat, "IO_REPARSE_TAG_SYMLINK", 0xA000000C),
    getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003),
))


def _reject_source_links(src: Path) -> None:
    """Refuse an untrusted plugin source tree that contains ANY link (symlink or
    Windows directory junction). Raises ValueError.

    ANY link, not merely one that escapes. ``shutil.copytree``'s default
    ``symlinks=False`` DEREFERENCES a link, flattening the target file's
    CONTENTS into the installed plugin dir, and on Windows ``copytree`` demotes
    a directory JUNCTION to a non-symlink and recurses into it, so
    ``symlinks=True`` neither preserves a junction nor bounds a junction cycle.
    An ABSOLUTE link whose target sits inside the source resolves inside it and
    passes an escape check, yet is copied verbatim.

    The result is an installed tree of self-contained plain files.
    """
    root = Path(src).resolve()
    try:
        entries = list(os.scandir(root))
    except OSError as e:
        raise ValueError(f"plugin source is not readable: {root} ({e})") from e
    stack: list[os.DirEntry] = list(entries)
    while stack:
        entry = stack.pop()
        p = Path(entry.path)
        try:
            st = entry.stat(follow_symlinks=False)
            # A Windows junction reports is_symlink() False, so its reparse tag
            # is what detects it. Only the two tags that alias another path are
            # tested, not the broader FILE_ATTRIBUTE_REPARSE_POINT bit.
            tag = getattr(st, "st_reparse_tag", 0)
            is_link = entry.is_symlink() or tag in _ALIASING_REPARSE_TAGS
        except OSError as e:          # unreadable/malformed reparse point
            raise ValueError(
                f"plugin source entry cannot be inspected: {p.name} ({e})") from e
        if is_link:
            # resolve() is non-strict, so a broken link still resolves
            # lexically and is named by where it pointed.
            target = p.resolve()
            escapes = target != root and root not in target.parents
            raise ValueError(
                f"plugin source contains a "
                f"{'link that points outside it' if escapes else 'link'}: "
                f"{p.name} -> {target}. Plugin sources must be plain files "
                f"(no symlinks or directory junctions).")
        try:
            if entry.is_dir(follow_symlinks=False):
                stack.extend(os.scandir(p))
        except OSError as e:
            raise ValueError(f"plugin source is not readable: {p} ({e})") from e


def _dir_content_hash(d: Path) -> str:
    """Deterministic sha256 over a plugin directory's tracked content: every
    file's POSIX relative path plus its bytes. Compiled artefacts (``__pycache__``,
    ``*.pyc``) and the provenance marker itself are excluded, so a bundled store
    source and an installed copy of the same code hash equal."""
    h = hashlib.sha256()
    try:
        files = [p for p in Path(d).rglob("*") if p.is_file()]
    except OSError:
        return ""
    entries = []
    for p in files:
        if p.name == _PLUGIN_MARKER or p.suffix == ".pyc" or "__pycache__" in p.parts:
            continue
        try:
            rel = p.relative_to(d).as_posix()
        except ValueError:
            continue
        entries.append((rel, p))
    for rel, p in sorted(entries):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"\0<unreadable>\0")
        h.update(b"\0")
    return h.hexdigest()


def _read_marker(dest: Path) -> Optional[dict]:
    """Read the provenance marker from an installed plugin dir, or None."""
    f = Path(dest) / _PLUGIN_MARKER
    try:
        if f.is_file():
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (OSError, ValueError):
        pass
    return None


def _write_marker(dest: Path, source: str, src_hash: str) -> None:
    """Record provenance + the source content hash in an installed plugin dir.
    Best-effort: a write failure must never break an install."""
    try:
        (Path(dest) / _PLUGIN_MARKER).write_text(
            json.dumps({"source": source, "src_hash": src_hash}),
            encoding="utf-8")
    except OSError:
        pass


def _purge_plugin_modules(uniq: str) -> None:
    """Remove a plugin's module namespace ("<uniq>" and every "<uniq>.*"
    submodule) from sys.modules so the next import reads fresh from disk."""
    for cached in [k for k in list(sys.modules)
                   if k == uniq or k.startswith(uniq + ".")]:
        sys.modules.pop(cached, None)


# --------------------------------------------------------------------------- #
#  Manifest parsing (the richer contract; superset of loader.PluginManifest)  #
# --------------------------------------------------------------------------- #

def parse_spec(plugin_dir: Path, *, builtin: bool = False,
               warnings: Optional[list] = None) -> PluginSpec:
    """Parse a plugin.toml in *plugin_dir* into a PluginSpec. Raises ValueError
    on an invalid manifest. When *warnings* is given, non-fatal manifest
    problems (unknown/misspelled keys in [plugin] or [surface]) are appended to
    it as human-readable strings. A plugin with an unknown key still parses and
    loads."""
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

    # [tools] exports must be a list of strings.
    _tools = data.get("tools", {})
    exports = _tools.get("exports", []) if isinstance(_tools, dict) else []
    if not (isinstance(exports, list) and all(isinstance(t, str) for t in exports)):
        raise ValueError(f"{manifest}: [tools] exports must be a list of strings")

    s = data.get("surface", {}) if isinstance(data.get("surface"), dict) else {}
    if warnings is not None:
        unknown = [f"[plugin] {k}" for k in sorted(set(p) - KNOWN_PLUGIN_KEYS)]
        unknown += [f"[surface] {k}" for k in sorted(set(s) - KNOWN_SURFACE_KEYS)]
        if unknown:
            warnings.append(
                "unknown manifest key(s) ignored: " + ", ".join(unknown))
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
        tool_exports=list(exports),
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
    # Drop modules cached from a PRIOR load of this plugin: the top-level name
    # and its submodules (e.g. <uniq>.backend), so every module loads from the
    # current directory on disk.
    _purge_plugin_modules(uniq)
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
        self.model_roles: list = []


    def mount_router(self, router) -> None:
        """Mount *router* on the live app, gating every route with the plugin's
        capability scope, and remember the routes added for later removal."""
        from fastapi import Depends
        from localm.inference.http_server import require_scope
        before = {id(r) for r in self._app.router.routes}
        self._app.include_router(
            router, dependencies=[Depends(require_scope(self._spec.scope))])
        new = [r for r in self._app.router.routes if id(r) not in before]
        # include_router appends, and the GUI's catch-all StaticFiles mount at
        # '/' matches every path, so Starlette would return that first.
        # Relocate the plugin's routes to just before the catch-all.
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
        # Pin the JS MIME types (idempotent), overriding any Windows registry
        # mapping, so served plugin scripts load as ES modules.
        mimetypes.add_type("text/javascript", ".js")
        mimetypes.add_type("text/javascript", ".mjs")
        # Pin .wasm likewise; WebAssembly.instantiateStreaming accepts only
        # application/wasm.
        mimetypes.add_type("application/wasm", ".wasm")
        prefix = "/" + (url_prefix or f"/plugins/{self._spec.name}").strip("/")
        if prefix in self._static_prefixes:
            return prefix                    # already serving this prefix (idempotent)
        base = Path(self._spec.path or ".").resolve()
        d = (base / directory).resolve()
        # *directory* comes verbatim from a manifest or a plugin's own
        # mount_static call. Confine it to the plugin dir: reject traversal and
        # absolute escapes (resolve() collapses both, and symlink escapes). The
        # plugin dir itself (d == base) is allowed.
        if not d.is_relative_to(base):
            raise ValueError(
                f"plugin {self._spec.name!r}: static dir {directory!r} escapes "
                f"the plugin dir")
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
        ``mount_static`` itself. Idempotent: a plugin that DID mount the prefix
        in register() short-circuits here. Best-effort: an absent assets_dir is
        ignored, never fatal."""
        surface = self._spec.surface
        if not surface or not surface.assets_dir:
            return None
        try:
            return self.mount_static(surface.assets_dir)
        except ValueError as e:
            # Best-effort: keep serving the rest of the plugin, logging at debug
            # which of mount_static's two failures occurred.
            _log.debug("plugin %s: assets_dir %r not mounted (%s); its client "
                       "entry will 404", self._spec.name, surface.assets_dir, e)
            return None

    def on_startup(self, callback) -> None:
        """Run *callback* once the server's event loop is running.

        On a stock server start, plugins register() before uvicorn creates the
        loop. With no running loop the callback is queued and the app lifespan
        runs it; with a running loop (a plugin enabled at runtime via the
        management API) it runs immediately. Failures are logged, never
        raised."""
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._manager.add_startup_callback(self._spec.name, callback)
            return
        try:
            callback()
        except Exception as e:
            _log.warning("plugin %s: on_startup callback failed: %s",
                         self._spec.name, e)

    def unmount(self) -> None:
        # Drop queued startup work so a disabled plugin's deferred callbacks
        # cannot fire when the lifespan drains them.
        self._manager.discard_startup_callbacks(self._spec.name)
        for r in self._routes:
            try:
                self._app.router.routes.remove(r)
            except ValueError:
                # The route is already gone, which is the desired end state;
                # unmount stays idempotent.
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
        """Contribute PluginSettingField entries for this plugin's own settings
        section, stored under config["plugins"][<name>] and rendered/saved
        generically by GET/POST /v1/plugins/<name>/settings (see
        PluginManager.get_all_plugin_settings and settings_schema.py's
        plugin_settings_* helpers).

        Validates the SHAPE up front - real PluginSettingField instances with a
        widget settings_schema.Widget actually knows - so a malformed field
        raises at register() time instead of silently never rendering."""
        from localm.settings_schema import all_widgets
        widgets = all_widgets()
        for f in fields:
            if not isinstance(f, PluginSettingField):
                raise TypeError(
                    f"plugin {self._spec.name!r}: add_settings() fields must be "
                    f"PluginSettingField instances (see localm.plugins.contract), "
                    f"got {type(f).__name__}")
            if f.widget not in widgets:
                raise ValueError(
                    f"plugin {self._spec.name!r}: add_settings() field {f.key!r} "
                    f"has unknown widget {f.widget!r}; see localm.settings_schema.Widget")
        self.settings.extend(fields)

    def register_tab(self, surface: Surface) -> None:
        self.surface = surface

    def register_model_role(self, descriptor) -> None:
        from localm.model_manager.registry import MODEL_TYPES
        if descriptor.model_type not in MODEL_TYPES:
            raise ValueError(f"Invalid model_type {descriptor.model_type!r}; must be one of {MODEL_TYPES}")
        descriptor.plugin_name = self._spec.name
        self.model_roles.append(descriptor)


    def _own_config_key(self, name: Optional[str]) -> str:
        """Confine plugin config r/w to the plugin's OWN block.

        A plugin cannot read or write ANOTHER plugin's persisted settings through
        the Host API. ``name`` is accepted for backward compatibility (a plugin
        passing its own name is fine); a DIFFERENT name resolves to this plugin's
        own name, and the attempt is logged."""
        if name is not None and name != self._spec.name:
            from localm.debuglog import logger as _dbg
            _dbg.warning(
                "plugin %r tried to access plugin %r config via the Host API; "
                "confining to its own config (cross-plugin config access is denied)",
                self._spec.name, name)
        return self._spec.name

    def plugin_config(self, name: Optional[str] = None) -> dict:
        from localm.config import load_config
        return dict(load_config().get("plugins", {}).get(self._own_config_key(name), {}))

    def save_plugin_config(self, name: Optional[str] = None, cfg: Optional[dict] = None) -> None:
        from localm.config import update_config
        key = self._own_config_key(name)
        value = cfg if cfg is not None else {}

        def _mutate(c: dict) -> None:
            c.setdefault("plugins", {})[key] = value
        update_config(_mutate)

    def has_scope(self, scope: str) -> bool:
        # Not implemented: the host has no request context. Scope checks happen
        # per-request at the route dependency level (require_scope on the
        # routes mount_router adds).
        raise NotImplementedError(
            "host-side scope checks are not implemented; scopes are enforced "
            "per-request on the routes mounted via mount_router()")

    def require_scope(self, scope: str) -> None:
        # See has_scope.
        raise NotImplementedError(
            "host-side scope checks are not implemented; scopes are enforced "
            "per-request on the routes mounted via mount_router()")

    def engine(self) -> Any:
        return self._manager.inference_engine

    def driving_engine(self, engine: Any = None):
        """Context manager: pin *engine* (or engine(), if not passed) as busy and
        reset its idle-unload clock for the DURATION of a real generation call.

        Wrap this around the actual chat_stream/complete call - never around a
        bare engine()/inference_engine access used only to check .loaded or read
        a name. See localm.inference.http_server.driving_engine for the
        mechanism."""
        from localm.inference.http_server import driving_engine as _driving_engine
        return _driving_engine(engine if engine is not None else self.engine())

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
            # the hook is inert.
            self.audit("chat_hook_skipped", {"phase": phase})
            return
        pipeline.add_hook(phase, fn, priority=priority, plugin=self._spec.name)
        # Record which plugin hooked which phase, at debug altitude.
        self.audit("chat_hook_registered", {"phase": phase, "priority": priority})
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
                 # back-compat aliases: builtin_root was the store,
                 # external_root the installed/discovery dir
                 builtin_root: "Optional[Path] | object" = _UNSET,
                 external_root: Optional[Path] = None) -> None:
        self.app = app
        self._inference_engine_static = inference_engine
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
        self._dep_tasks: dict = {}                   # name -> DepInstallTask (GUI installs)
        self._dep_tasks_lock = threading.Lock()
        # (plugin name, callback) queued by Host.on_startup before the event
        # loop exists; drained once by the app lifespan.
        self._startup_callbacks: list = []

    @property
    def inference_engine(self) -> Any:
        try:
            from localm.inference import http_server as _hs
            name = getattr(_hs, "_active_model_name", None)
            if name and name in getattr(_hs, "_engines", {}):
                return _hs._engines[name]
        except Exception as e:
            # Fall back to the statically-injected engine and log why the live
            # one was skipped.
            from localm.debuglog import logger
            logger.debug("plugin host: live inference-engine lookup failed, using "
                         "static engine: %s", e)
        return self._inference_engine_static

    # ---- discovery (INSTALLED folder only) ---------------------------------
    # ---- deferred startup work (loop-dependent plugin init) ---------------- #
    # Plugins register() before uvicorn creates the event loop, so work that
    # needs a running loop is queued here by Host.on_startup and drained by the
    # server lifespan. Runtime enable (loop already running) runs the callback
    # directly in Host.on_startup and never lands here.

    def add_startup_callback(self, name: str, callback) -> None:
        self._startup_callbacks.append((name, callback))

    def discard_startup_callbacks(self, name: str) -> None:
        """Drop queued callbacks of *name* (plugin unloaded before the lifespan
        ran) so a disabled plugin's work can never fire later."""
        self._startup_callbacks = [
            (n, cb) for n, cb in self._startup_callbacks if n != name]

    def get_all_model_roles(self) -> list[dict]:
        """All registered ModelRoleDescriptors across active/loaded plugins."""
        roles = []
        for name, entry in self._loaded.items():
            spec, module, host, uniq = entry
            if hasattr(host, "model_roles"):
                for r in host.model_roles:
                    roles.append({
                        "role_id": r.role_id,
                        "label": r.label,
                        "model_type": r.model_type,
                        "plugin_name": r.plugin_name,
                        "required": r.required,
                        "description": r.description,
                    })
        return roles

    def get_all_plugin_settings(self) -> list[dict]:
        """Settings sections contributed by currently ACTIVE (loaded) plugins
        via host.add_settings(), one entry per plugin that registered at least
        one field.

        Fields are returned as-is (PluginSettingField objects, not yet
        resolved against config or filtered for a caller's ownership);
        GET/POST /v1/plugins/<name>/settings do that per-request.

        Disabling a plugin drops its whole loaded entry (_unload pops it from
        _loaded and unmounts the host), so a disabled plugin's fields vanish
        from this list automatically."""
        out = []
        for name, entry in self._loaded.items():
            spec, module, host, uniq = entry
            fields = getattr(host, "settings", None)
            if not fields:
                continue
            label = (spec.surface.settings_group
                     if spec and spec.surface and spec.surface.settings_group
                     else name)
            out.append({"plugin": name, "label": label, "fields": list(fields)})
        return out

    def run_startup_callbacks(self) -> None:
        """Run every queued startup callback once. Called by the app lifespan
        with the event loop running. Best-effort per callback: one plugin's
        failure is logged and does not stop the others or the server."""
        callbacks, self._startup_callbacks = self._startup_callbacks, []
        for name, cb in callbacks:
            try:
                cb()
            except Exception as e:
                _log.warning(
                    "plugin %s: deferred startup callback failed: %s", name, e)

    def discover(self) -> dict[str, PluginSpec]:
        """Discover INSTALLED plugins only (the installed folder). The store shelf
        is never discovered - it is just the source for install().

        Self-heals preinstalled/protected plugins (chat) onto disk first, so a
        data dir where only headless CLI commands have run still has chat
        physically provisioned before `missing_requires` / `_installed_set`
        read the installed set."""
        self._ensure_preinstalled()
        self._specs = {}
        prior = self._discover_errors        # for change-only warning logs below
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
                    warns: list = []
                    spec = parse_spec(child, builtin=False, warnings=warns)
                    if not spec.compatible():
                        raise ValueError(
                            f"api_version {spec.api_version} != {API_VERSION}")
                    self._specs[spec.name] = spec
                    if warns:
                        # Record unknown manifest keys per plugin (surfaced via
                        # api_state) while the plugin still parses and loads.
                        # Logged only when new or changed, since api_state
                        # re-discovers on every fetch.
                        msg = "warning: " + "; ".join(warns)
                        self._discover_errors[spec.name] = msg
                        if prior.get(spec.name) != msg:
                            _log.warning("plugin %s: %s", spec.name,
                                         "; ".join(warns))
                except Exception as e:       # one bad manifest must not break discovery
                    self._discover_errors[child.name] = str(e)
        return self._specs

    def _store_dir(self, name: str) -> Optional[Path]:
        # Returns None for a non-conforming id, before the join, so no
        # traversing id probes for a plugin.toml outside the store root.
        # Callers turn None into a no-such-builtin-plugin KeyError.
        if not self._store_root or not _is_valid_plugin_name(name):
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

    def cli_entries(self) -> list[tuple[str, str]]:
        """(name, "module.path:attr") for every first-party plugin that
        declares a CLI entry point in its manifest's ``cli`` key, in catalog
        order. Read from the bundled store (NOT the installed set), so a
        first-party CLI command like ``localm coder`` stays reachable
        regardless of plugin install/enable state; only its pip extras
        (ImportError) gate it."""
        from localm.plugins import catalog as _cat
        order = {e.name: i for i, e in enumerate(_cat.CATALOG)}
        out = []
        for name, spec in sorted(self.store_catalog().items(),
                                 key=lambda kv: order.get(kv[0], len(order))):
            if spec.cli_entry:
                out.append((name, spec.cli_entry))
        return out

    # ---- installed/enabled state -------------------------------------------
    # 'Installed' is PHYSICAL: a plugin is installed iff its directory is
    # present in the installed folder (discoverable), not via a config flag.
    # 'Enabled' is a config toggle within installed: a plugin is active
    # (loaded) iff installed AND enabled. Load takes the intersection, so a
    # stale 'enabled' entry for a plugin no longer on disk is ignored.
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
    def load_enabled(
        self,
        on_event: Optional[Callable[[str, str, Optional[str]], None]] = None,
    ) -> None:
        """Discover INSTALLED plugins and load every active one (installed AND
        enabled). Never raises - a failing plugin is recorded in errors and
        skipped. A 'enabled' config entry for a plugin not on disk is ignored.

        Each plugin is loaded sequentially and reported as it goes: a line is
        logged (``localm.plugins`` logger) and, if given, *on_event* is called
        ``on_event(name, status, error)`` with status "loaded" or "failed". A
        raising callback is ignored so it can never break startup."""
        self._ensure_preinstalled()                # first-run: provision chat etc.
        self._refresh_installed_builtins()         # upgrade: re-copy stale builtins
        self.discover()
        enabled = self._enabled_set()

        def _emit(name: str, status: str, error: Optional[str]) -> None:
            if status == "loaded":
                _log.info("plugin %s loaded", name)
            else:
                _log.warning("plugin %s failed to load: %s", name, error)
            if on_event is not None:
                try:
                    on_event(name, status, error)
                except Exception:              # a bad callback must not break load
                    pass

        for name in sorted(self._specs):           # _specs == installed (on disk)
            if name in enabled and name not in self._loaded:
                self._safe_load(self._specs[name])
                if name in self._loaded:
                    _emit(name, "loaded", None)
                else:
                    _emit(name, "failed", self._errors.get(name))

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
            except Exception as e:                  # record (don't swallow) a corrupt just-provisioned manifest
                self._discover_errors[name] = f"preinstall-parse: {e}"

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
        try:
            register(host)
            # Serve a declared surface assets_dir even if register() did not
            # mount it itself.
            host.mount_surface_assets()
        except Exception:
            # register() can raise after it has already mounted some routes or
            # chat hooks. host.unmount() undoes exactly what THIS host tracked
            # (routes, static mounts, chat hooks, deferred on_startup
            # callbacks), so a failed load leaves nothing mounted.
            host.unmount()
            _purge_plugin_modules(uniq)
            raise
        self._loaded[spec.name] = (spec, module, host, uniq)
        self._errors.pop(spec.name, None)       # a successful load clears prior error
        self._maybe_fire_first_use(spec.name)

    def _maybe_fire_first_use(self, name: str) -> None:
        """Invoke the ``on_first_use`` lifecycle hook exactly once per plugin - the
        first time it is loaded (its first activation). Persisted in config so it
        does not re-fire on every server start."""
        # Only plugins that define on_first_use are tracked here, so the rest
        # never trigger a config write on their first load.
        entry = self._loaded.get(name)
        if not entry or not callable(getattr(entry[1], "on_first_use", None)):
            return
        from localm.config import load_config, update_config
        try:
            done = set(load_config().get("plugins_first_use_done", []))
        except Exception:
            return
        if name in done:
            return
        self._invoke_hook(name, "on_first_use")

        def _mutate(cfg: dict) -> None:
            cur = set(cfg.get("plugins_first_use_done", []))
            cur.add(name)
            cfg["plugins_first_use_done"] = sorted(cur)
        try:
            update_config(_mutate)
        except Exception:
            pass  # best-effort: worst case on_first_use fires again next start

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
        _purge_plugin_modules(uniq)     # drop the whole namespace so a re-enable
        #                                 re-imports every module fresh from disk

    def _invoke_hook(self, name: str, hook_name: str, **kwargs) -> None:
        """Call an optional plugin lifecycle hook if the loaded module defines it.
        Best-effort: a hook error is logged at WARNING and never blocks the
        action."""
        entry = self._loaded.get(name)
        if not entry:
            return
        hook = getattr(entry[1], hook_name, None)
        if callable(hook):
            try:
                hook(**kwargs)
            except Exception as e:
                # Log a failing plugin hook at WARNING without failing the
                # install/first-use path that called it.
                _log.warning("plugin %s: %s hook failed: %s", name, hook_name, e)

    # ---- provisioning helpers ----------------------------------------------
    def _installed_dir(self, name: str) -> Path:
        # install, refresh, uninstall and the third-party copy all resolve
        # their directory here, so the id is validated once at this join.
        # Raises ValueError on a bad id.
        return Path(self._installed_root) / _check_plugin_name(name)

    def _provision_from_store(self, name: str) -> bool:
        """Copy the plugin from the bundled store into the installed folder (or,
        if missing from the store, fetch it from its GitHub repo). No-op if it is
        already installed. Raises KeyError when no source exists.

        Returns True only if THIS call created the directory. The caller must
        pass that through to _provision_and_verify's rollback: rolling back a
        directory this call did not create is destructive."""
        import shutil
        dest = self._installed_dir(name)
        if (dest / "plugin.toml").is_file():
            return False                             # already installed on disk
        src = self._store_dir(name)
        if src is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest)
            _write_marker(dest, "store", _dir_content_hash(src))
            return True
        from localm.plugins import catalog as _cat
        entry = _cat.get(name)
        url = entry.source_url() if entry else ""
        if url:
            raise NotImplementedError(
                f"plugin {name!r} is not in the bundled store; fetching it from "
                f"{url} is not wired up yet")
        raise KeyError(f"no such plugin: {name}")

    def _remove_installed_dir(self, name: str) -> bool:
        """Delete the plugin's directory from the installed folder. Returns True
        if the directory is gone afterwards (or never existed), False if it is
        still present because ``shutil.rmtree`` failed - the caller (uninstall(),
        set_installed_state()) must not report a bare success in that case."""
        import shutil
        # Confined by RESOLVED parent rather than identifier shape: uninstall()
        # admits any basename present on disk (_installed_set returns raw
        # directory names, which need not equal the manifest name), so a
        # hand-extracted directory like 'coolplugin-1.0' can still be removed.
        # A traversing name, and a symlinked plugin dir whose target lives
        # outside the root, are refused.
        root = Path(self._installed_root)
        d = root / name
        try:
            contained = d.resolve().parent == root.resolve()
        except OSError:
            contained = False
        if not contained:
            raise ValueError(
                f"refusing to delete {name!r}: it does not resolve to a direct "
                f"child of the installed-plugins root")
        try:
            if d.is_dir():
                shutil.rmtree(d)
        except OSError as e:
            # A locked file, an AV hold or a permission denial leaves the
            # directory - and any code/data in it - on disk. Log it and return
            # the real outcome rather than a bare success.
            _log.warning(
                "plugin %s: could not remove installed directory %s: %s; the "
                "directory (and any residual code/data) remains on disk",
                name, d, e)
            return not d.is_dir()
        return not d.is_dir()

    # ---- refresh stale builtin copies (upgrade safety) ---------------------
    # An installed builtin is a COPY of the bundled store source taken at
    # install time, and install/provision no-ops while that directory exists.
    # Staleness is detected by a CONTENT HASH of the store source recorded in
    # the provenance marker, not by the plugin version.
    def _maybe_refresh_builtin(self, name: str) -> bool:
        """Re-copy an installed builtin from the bundled store when the shipped
        source changed since it was installed. Returns True if a refresh ran.

        Safe by construction:
          - Only plugins WITH a bundled-store source are considered; a third-party
            plugin (no store dir, or a marker recording ``source != "store"``) is
            never touched.
          - User config is NOT in the plugin dir (it lives in config.json under
            ``plugins.<name>``); plugin data lives under ``data_subdir`` in the
            data dir. Re-copying the plugin dir therefore preserves both.
          - The swap goes through temp/backup siblings so a failure leaves the
            existing copy intact.
        """
        store_src = self._store_dir(name)
        if store_src is None:
            return False                              # not a bundled/builtin plugin
        dest = self._installed_dir(name)
        if not (dest / "plugin.toml").is_file():
            return False                              # not installed on disk
        marker = _read_marker(dest)
        if marker and marker.get("source") not in (None, "store"):
            return False                              # third-party: never clobber
        cur = _dir_content_hash(store_src)
        # What we last synced FROM: the recorded source hash, or - for a legacy
        # copy with no marker - the installed copy's own current content.
        known = marker.get("src_hash") if marker else _dir_content_hash(dest)
        if known == cur:
            if marker is None:                        # adopt provenance, no re-copy
                _write_marker(dest, "store", cur)
            return False
        if not self._swap_in_store_copy(name, store_src, dest, cur):
            return False
        # If the plugin is live on an app, reload it so the new code takes effect.
        if self.app is not None and name in self._loaded:
            self._unload(name)
            self.discover()
            if name in self._specs and name in self._enabled_set():
                self._safe_load(self._specs[name])
        _log.info("plugin %s refreshed from the bundled store", name)
        return True

    def _swap_in_store_copy(self, name: str, store_src: Path, dest: Path,
                            cur_hash: str) -> bool:
        """Replace *dest* with a fresh copy of *store_src* (+ marker) as safely as
        the platform allows. The original install is never deleted until the new
        copy is in place, so any failure leaves a usable plugin dir. Returns True
        only on a completed swap."""
        import shutil
        tmp = dest.parent / f".{name}.refresh.tmp"
        backup = dest.parent / f".{name}.refresh.bak"
        # The staging paths interpolate the name into a basename, so re-check
        # the name and that tmp, backup and dest all sit directly in the
        # installed root before the copytree, renames and rmtrees below.
        if (not _is_valid_plugin_name(name)
                or dest.parent != Path(self._installed_root)
                or tmp.parent != dest.parent or backup.parent != dest.parent):
            raise ValueError(
                f"refusing to stage a refresh for {name!r}: the staging paths "
                f"do not sit in the installed root")
        # Stage 1 - build the fresh copy in a temp sibling. A failure here leaves
        # the existing install completely untouched.
        try:
            for stale in (tmp, backup):
                if stale.exists():
                    shutil.rmtree(stale)
            shutil.copytree(store_src, tmp)
            _write_marker(tmp, "store", cur_hash)
        except OSError as e:
            shutil.rmtree(tmp, ignore_errors=True)
            _log.warning("plugin %s: could not stage a refresh: %s", name, e)
            return False
        # Stage 2 - swap it into place. Move the stale copy aside first; if the
        # move-in fails, put it back. The original is kept at dest, or at the
        # backup if even the restore fails.
        try:
            dest.rename(backup)
        except OSError as e:
            shutil.rmtree(tmp, ignore_errors=True)
            _log.warning("plugin %s: could not refresh stale copy: %s", name, e)
            return False
        try:
            tmp.rename(dest)
        except OSError as e:
            try:
                backup.rename(dest)                   # restore the original
            except OSError as restore_err:
                _log.error("plugin %s: refresh failed and the original could not "
                           "be restored; the previous copy is kept at %s (%s)",
                           name, backup, restore_err)
                shutil.rmtree(tmp, ignore_errors=True)
                return False
            shutil.rmtree(tmp, ignore_errors=True)
            _log.warning("plugin %s: could not refresh stale copy: %s", name, e)
            return False
        shutil.rmtree(backup, ignore_errors=True)
        return True

    def _refresh_installed_builtins(self) -> None:
        """Refresh every installed builtin whose bundled-store source changed
        since install (an upgrade left a stale copy). Best-effort; never raises -
        a failing refresh must not break startup."""
        if not self._store_root:
            return
        for name in sorted(self._installed_set()):
            try:
                self._maybe_refresh_builtin(name)
            except Exception as e:
                self._discover_errors[name] = f"refresh: {e}"

    def refresh(self, name: Optional[str] = None) -> list:
        """Re-sync installed builtin plugins with the bundled store, re-copying
        any whose shipped source changed since install. With *name*, refresh just
        that plugin (KeyError if it is not an installed builtin). Returns the
        names actually refreshed. A live, active plugin is reloaded in place."""
        self.discover()
        if name is not None:
            if self._store_dir(name) is None:
                raise KeyError(f"no such builtin plugin: {name}")
            if not (self._installed_dir(name) / "plugin.toml").is_file():
                raise KeyError(f"plugin {name!r} is not installed")
            return [name] if self._maybe_refresh_builtin(name) else []
        refreshed = []
        for n in sorted(self._installed_set()):
            try:
                if self._maybe_refresh_builtin(n):
                    refreshed.append(n)
            except Exception as e:
                self._errors[n] = f"refresh: {e}"
        return refreshed

    def _is_protected(self, name: str) -> bool:
        from localm.plugins import catalog as _cat
        spec = self._specs.get(name)
        return bool(spec and spec.protected) or name in _cat.protected()

    # ---- shared install-sequence helpers -----------------------------------
    def _reject_scope_collision(self, spec0: PluginSpec) -> None:
        """Refuse a third-party manifest whose scope collides with a kernel
        capability, a first-party plugin's scope, a privileged scope, or
        another already-installed plugin's scope. A manifest that omits
        ``scope`` defaults to the plugin's own NAME
        (``PluginSpec.__post_init__``), so this triggers via a plausible
        plugin name like "rag"/"web"/"voice", not only a deliberate
        ``scope = "chat"`` line: ``mount_router`` gates every route the
        plugin registers on this raw string."""
        from localm import scopes as S
        scope = spec0.scope
        # all_known_scopes() is KERNEL_SCOPES | BUILTIN_PLUGIN_SCOPES |
        # EXTRA_SCOPES, and PRIVILEGED_SCOPES is a subset of that union, so
        # this one check covers all three reserved categories.
        if scope in S.all_known_scopes():
            raise ValueError(
                f"plugin {spec0.name!r} declares scope {scope!r}, which is a "
                f"reserved localm scope and cannot be claimed by a "
                f"third-party plugin")
        self.discover()
        collision = next((n for n, sp in self._specs.items()
                           if n != spec0.name and sp.scope == scope), None)
        if collision:
            raise ValueError(
                f"plugin {spec0.name!r} declares scope {scope!r}, which is "
                f"already used by installed plugin {collision!r}")

    def _copy_third_party_source(self, source: Path, *, force: bool):
        """Validate + copy a third-party plugin source dir into the installed
        folder. Returns (parsed spec, dest dir)."""
        import shutil
        src = Path(source)
        # Reject links before anything reads a file out of the untrusted source
        # tree.
        _reject_source_links(src)
        spec0 = parse_spec(src)                       # validate + name (raises)
        name = spec0.name
        # A third-party plugin must not shadow a built-in command name
        # (run/serve/config/coder/...). Only arbitrary-source third-party
        # installs are gated; builtins install via install() from the store.
        from localm.plugins.loader import _RESERVED_NAMES
        if name in _RESERVED_NAMES:
            raise ValueError(
                f"plugin name {name!r} clashes with a built-in command")
        self._reject_scope_collision(spec0)
        # name came out of the MANIFEST; parse_spec has already rejected any
        # name that is not a single identifier-shaped component, and
        # _installed_dir re-checks it. This feeds _remove_installed_dir ->
        # shutil.rmtree just below, and copytree after.
        dest = self._installed_dir(name)
        if dest.exists():
            if not force:
                raise ValueError(f"plugin {name!r} is already installed")
            self._remove_installed_dir(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # symlinks=True keeps a link that appeared after _reject_source_links
        # walked the tree from being DEREFERENCED. It does not cover a Windows
        # junction, which copytree demotes and recurses into.
        shutil.copytree(src, dest, symlinks=True)
        _write_marker(dest, "external", _dir_content_hash(src))
        return spec0, dest

    def _provision_and_verify(self, name: str, *, rollback_on_fail: bool = True,
                              fail_verb: str = "could not be installed") -> None:
        """Re-discover after a copy landed in the installed folder and confirm
        the manifest actually parses, rolling back the copy on failure."""
        self.discover()
        if name not in self._specs:
            detail = self._discover_errors.get(name, "bad manifest")
            if rollback_on_fail:
                self._remove_installed_dir(name)
            raise ValueError(f"plugin {name!r} {fail_verb}: {detail}")

    def _resolve_missing_plugin_error(self, name: str, *, hint_ok: bool = True) -> Exception:
        """Resolve 'not installed' into a ValueError with an install hint (the
        plugin is at least known) or a KeyError (truly unknown). *hint_ok*
        suppresses the install hint for disable
        (``set_enabled_state(on=False)``), where 'install it first' makes no
        sense."""
        from localm.plugins import catalog as _cat
        if hint_ok and (_cat.get(name) or self._store_dir(name)):
            return ValueError(f"plugin {name!r} is not installed; install it first")
        return KeyError(f"no such plugin: {name}")

    # ---- public lifecycle (install/uninstall = store<->installed) -----------
    def install(self, name: str) -> None:
        """Install a plugin: copy it from the bundled store (or its GitHub repo)
        into the installed folder, then load + enable it on the live app. Rolls
        back the copy if it does not load. KeyError if no such plugin exists."""
        copied = self._provision_from_store(name)    # may raise KeyError
        # Roll back only the copy this call created; see _provision_from_store.
        self._provision_and_verify(name, rollback_on_fail=copied)
        try:
            if name not in self._loaded:
                self._load(self._specs[name])
        except Exception:
            if copied:
                self._remove_installed_dir(name)     # roll back OUR copy only
            raise
        self._invoke_hook(name, "on_install")        # optional lifecycle hook
        self._set_enabled(name, True)

    def install_external(self, source: Path, *, force: bool = False):
        """Install a THIRD-PARTY plugin from an arbitrary source directory: copy it
        into the installed folder, then load + enable. Rolls back on failure."""
        spec0, dest = self._copy_third_party_source(source, force=force)
        name = spec0.name
        self._provision_and_verify(name, fail_verb="is not loadable")
        try:
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
        spec0, dest = self._copy_third_party_source(source, force=force)
        name = spec0.name
        self._provision_and_verify(name)
        if enable:
            self._set_enabled(name, True)
        return spec0

    def enable(self, name: str) -> None:
        self.discover()
        if name not in self._specs:
            raise self._resolve_missing_plugin_error(name)
        self._require_deps_installed(name)
        self._maybe_refresh_builtin(name)             # pick up an upgraded builtin
        self.discover()                               # re-read the refreshed spec
        if name not in self._loaded:
            self._load(self._specs[name])             # load first; surface errors
        self._set_enabled(name, True)

    def _require_deps_installed(self, name: str) -> None:
        """Refuse to enable a plugin whose declared ``requires`` are not
        installed."""
        missing = self.missing_requires(name)
        if missing:
            plural = "s" if len(missing) > 1 else ""
            raise ValueError(
                f"plugin {name!r} requires plugin{plural} "
                f"{', '.join(sorted(missing))} which {'are' if plural else 'is'} "
                f"not installed; install {'them' if plural else 'it'} first")

    def _dependents(self, name: str) -> set:
        """Installed plugins that (transitively) declare *name* in their requires."""
        self.discover()
        installed = self._installed_set()
        result: set = set()
        frontier = {name}
        while frontier:
            target = frontier.pop()
            for other in installed:
                if other == name or other in result:
                    continue
                spec = self._specs.get(other)
                if spec and target in spec.requires:
                    result.add(other)
                    frontier.add(other)
        return result

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

    # ---- pip-extra dependencies (host-side install) -------------------------
    def _spec_for(self, name: str):
        return self._specs.get(name) or self.store_catalog().get(name)

    def plugin_requirements(self, name: str) -> list:
        """Concrete requirement strings a plugin's ``requires_extras`` map to,
        resolved from localm's installed metadata."""
        spec = self._spec_for(name)
        if not spec or not spec.requires_extras:
            return []
        from localm.plugins import deps
        return deps.plugin_requirements(spec.requires_extras)

    def plugin_missing_deps(self, name: str) -> list:
        """The subset of a plugin's declared pip-extra requirements that are NOT
        installed on this host. Empty when it declares none or all are present."""
        from localm.plugins import deps
        return deps.missing_requirements(self.plugin_requirements(name))

    def all_missing_deps(self, *, enabled_only: bool = True) -> dict:
        """``{plugin: [missing requirement strings]}`` across installed plugins
        (enabled ones by default) that are missing a declared pip extra."""
        self.discover()
        names = self._enabled_set() if enabled_only else self._installed_set()
        out = {}
        for name in sorted(names):
            miss = self.plugin_missing_deps(name)
            if miss:
                out[name] = miss
        return out

    def scope_deps_warnings(self, granted_scopes) -> list:
        """Warnings for minting a key: a granted capability scope that maps to a
        first-party plugin which is not installed, or is installed but missing a
        declared pip extra. Empty when every granted plugin scope is ready. This
        is the 'catch at grant' check - a key that unlocks a feature the host
        cannot actually serve yet."""
        from localm.plugins import catalog as _cat
        self.discover()
        installed = self._installed_set()
        by_scope = {}
        for name, spec in self.store_catalog().items():
            by_scope.setdefault(spec.scope or name, name)
        for name, spec in self._specs.items():          # installed specs win
            by_scope[spec.scope or name] = name
        for e in _cat.CATALOG:                          # catalog scope == name
            by_scope.setdefault(e.name, e.name)
        warnings = []
        for sc in dict.fromkeys(granted_scopes or ()):  # de-dup, keep order
            pname = by_scope.get(sc)
            if not pname or self._is_protected(pname):   # not a plugin, or chat
                continue
            if pname not in installed:
                warnings.append(
                    f"key grants '{sc}' but the {pname} plugin is not installed")
                continue
            miss = self.plugin_missing_deps(pname)
            if miss:
                warnings.append(
                    f"key grants '{sc}' but {pname} is missing: {', '.join(miss)}")
        return warnings

    def install_plugin_deps(self, name: str, *, on_progress=None):
        """Install a plugin's declared pip extras on THIS host. HOST-ONLY: an
        HTTP route must confirm the server is loopback-bound
        (``deps_task.host_pip_allowed``) before calling this; the CLI is always
        host-side. Returns a ``deps.InstallResult`` (a no-op success when the
        plugin declares none)."""
        from localm.plugins import deps
        spec = self._spec_for(name)
        extras = spec.requires_extras if spec else []
        return deps.install_plugin_extras(extras, on_progress=on_progress)

    def get_dep_task(self, name: str):
        """The in-flight or finished dependency-install task for *name*, if any."""
        with self._dep_tasks_lock:
            return self._dep_tasks.get(name)

    def start_dep_install(self, name: str):
        """Start (or return the still-running) background dep install for *name*.
        HOST-ONLY: the route confirms the request is local first. Idempotent while
        a task is running so a double-click does not launch two pip runs."""
        from localm.plugins.deps_task import DepInstallTask, run_dep_install
        with self._dep_tasks_lock:
            existing = self._dep_tasks.get(name)
            if existing is not None and existing.status == "running":
                return existing
            task = DepInstallTask(name)
            self._dep_tasks[name] = task
        t = threading.Thread(target=run_dep_install, args=(self, name, task),
                             name=f"dep-install-{name}", daemon=True)
        t.start()
        return task

    def set_installed_state(self, name: str, on: bool, *, enable: bool = True) -> None:
        """CLI/headless install/uninstall WITHOUT loading routes: copy store ->
        installed (or remove the installed dir); the GUI server reconciles via
        load_enabled on its next start. Installing also enables by default;
        uninstalling disables. Honours protection on uninstall."""
        if on:
            copied = self._provision_from_store(name)  # copy store -> installed (raises if unknown)
            # Roll back ONLY what this call created - see _provision_from_store.
            self._provision_and_verify(name, rollback_on_fail=copied)
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
            raise self._resolve_missing_plugin_error(name, hint_ok=on)
        if not on and self._is_protected(name):
            raise ValueError(f"plugin {name!r} is protected and cannot be disabled")
        if on:
            self._require_deps_installed(name)
        self._set_enabled(name, on)

    def uninstall(self, name: str, *, delete_data: bool = False) -> bool:
        """Uninstall a plugin: unload it, disable it, and DELETE its directory from
        the installed folder (it reverts to being merely available in the store).
        User content is kept unless *delete_data*, in which case its data
        directory is deleted too; the on_uninstall hook runs first. Returns True
        only if it was installed AND its installed directory was actually removed
        from disk AND - when delete_data was requested - its data directory was
        actually removed too. False if it was not installed to begin with, or if
        either removal could not complete (a locked file, an AV hold, or a
        permission denial - all reachable on Windows - or a data_subdir that
        refused to resolve inside the data dir). In the degraded case the plugin
        is still disabled and unloaded, but some of its files remain on disk; see
        the WARNING logged by _remove_installed_dir / _delete_plugin_data for the
        concrete cause. A caller must never report bare success without checking
        this return value. KeyError if wholly unknown."""
        self.discover()
        spec = self._specs.get(name)
        was_installed = name in self._installed_set()
        if spec is None and not was_installed:
            raise KeyError(f"no such plugin: {name}")
        if self._is_protected(name):
            raise ValueError(f"plugin {name!r} is protected and cannot be uninstalled")
        # Cascade-unload dependents: disable and unload (transitively) every
        # plugin that declares this one in requires, and log each.
        for dep in self._dependents(name):
            if dep in self._enabled_set() or dep in self._loaded:
                _log.warning("disabling plugin %s: it requires %s, which is being "
                             "uninstalled", dep, name)
                try:
                    self._set_enabled(dep, False)
                    self._unload(dep)
                except Exception as e:
                    # Best-effort teardown; the primary uninstall proceeds. A
                    # failed disable leaves dep enabled with a missing
                    # dependency, reported by the enable guard and api_state.
                    _log.debug("could not fully disable dependent plugin %s during "
                               "cascade uninstall of %s: %s", dep, name, e)
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
        data_deleted = True         # vacuously true: nothing was asked to go
        if delete_data and spec and spec.data_subdir:
            data_deleted = self._delete_plugin_data(spec)
        removed = self._remove_installed_dir(name)    # delete from the installed folder
        if was_installed and not removed:
            _log.warning(
                "plugin %s: uninstall disabled and unloaded it, but its "
                "installed directory could not be removed; reporting a "
                "degraded result rather than a bare success", name)
        if delete_data and not data_deleted:
            _log.warning(
                "plugin %s: uninstall disabled and unloaded it, but its data "
                "could not be fully deleted; reporting a degraded result "
                "rather than a bare success", name)
        return was_installed and removed and data_deleted

    def _delete_plugin_data(self, spec: PluginSpec) -> bool:
        """Delete the plugin's data_subdir. Returns True iff it is confirmed gone
        afterwards (or there was nothing to delete); False if a security refusal
        (data_subdir escapes the data dir) or a removal failure (locked file, AV
        hold, permission denial) leaves it on disk. The caller (uninstall())
        must fold this into its reported result rather than treating a removed
        installed-dir as the whole story."""
        import shutil
        from localm.config import home_dir
        if not spec.data_subdir:
            return True
        # data_subdir comes verbatim from a manifest. Resolve it and confine to
        # home_dir: reject traversal, absolute escapes, and the home root
        # itself before any rmtree.
        home = home_dir().resolve()
        d = (home / spec.data_subdir).resolve()
        if d == home or not d.is_relative_to(home):
            _log.warning(
                "plugin %s: refusing to delete data_subdir %r -> %s (it does "
                "not resolve inside the data dir); the data is NOT deleted",
                spec.name, spec.data_subdir, d)
            return False
        try:
            if d.is_dir():
                shutil.rmtree(d)
        except OSError as e:
            _log.warning(
                "plugin %s: could not delete data directory %s: %s; the data "
                "remains on disk", spec.name, d, e)
            return not d.is_dir()
        return not d.is_dir()

    # ---- state for the API / GUI -------------------------------------------
    def api_state(self) -> dict:
        """Installed plugins (loaded from the installed folder) plus what is
        AVAILABLE to install (the bundled store + the static catalog, minus what
        is installed). Each entry carries installed/enabled/active/available."""
        from localm.plugins import catalog as _cat
        from localm.plugins import deps as _deps
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
                # Coder-agent tools a third-party plugin exports ([tools]
                # exports), shown in the GUI's External plugins card.
                "tool_exports": list(spec.tool_exports) if spec else [],
                "requires_extras": spec.requires_extras if spec else [],
                # Declared pip-extra requirements not installed on this host.
                "missing_deps": (
                    _deps.missing_requirements(
                        _deps.plugin_requirements(spec.requires_extras))
                    if spec and spec.requires_extras else []),
                "requires": spec.requires if spec else [],
                # Declared requirements that are not currently installed.
                "missing_requires": [r for r in (spec.requires if spec else [])
                                     if r not in installed],
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
        _cfg = load_config()
        return {"plugins": plugins,
                "errors": {**self._discover_errors, **self._errors},
                "suggest_plugins": bool(_cfg.get("suggest_plugins", True)),
                "auto_install_plugin_deps": bool(
                    _cfg.get("auto_install_plugin_deps", True))}


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

    def _valid_name_or_404(name: str) -> None:
        """A ``{name}`` that is not a legal plugin id can never match a plugin,
        so it is a 404. Called BEFORE each handler's try block, which catches
        broad ``Exception`` and would otherwise turn this into a 400. The engine
        rejects such an id again at the path join (_installed_dir)."""
        if not _is_valid_plugin_name(name):
            raise HTTPException(404, f"No such plugin: {name}")

    @app.get("/api/plugins", dependencies=[Depends(require_scope(scopes.PLUGINS_READ))])
    async def list_plugins_engine():
        return manager.api_state()

    @app.post("/api/plugins/{name}/install",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def install_plugin_engine(name: str):
        _valid_name_or_404(name)
        try:
            manager.install(name)
        except KeyError:
            raise HTTPException(404, f"No such plugin: {name}")
        except Exception as e:
            raise HTTPException(400, f"Install failed: {e}")
        return {"status": "installed", "name": name}

    @app.post("/api/plugins/install-external",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def install_external_plugin_engine(body: dict):
        """Install a THIRD-PARTY plugin from a local directory (the GUI's
        External plugins card). The HTTP sibling of `localm plugin install
        <dir>`, using the same manager call. Copy + verify only, no live mount:
        the plugin loads on the next server start."""
        from pathlib import Path as _P
        source = (body or {}).get("source", "")
        if not source:
            raise HTTPException(400, "Missing 'source' (local directory path)")
        src = _P(source)
        if not (src.is_dir() and (src / "plugin.toml").is_file()):
            raise HTTPException(400, f"No plugin.toml in {source!r}")
        try:
            spec = manager.set_installed_from_dir(src, force=bool((body or {}).get("force")))
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(400, f"Install failed: {e}")
        return {"status": "installed", "name": spec.name, "version": spec.version}

    @app.post("/api/plugins/{name}/uninstall",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def uninstall_plugin_engine(name: str, delete_data: bool = False):
        _valid_name_or_404(name)
        try:
            complete = manager.uninstall(name, delete_data=delete_data)
        except KeyError:
            raise HTTPException(404, f"No such plugin: {name}")
        except ValueError as e:
            raise HTTPException(409, str(e))
        except Exception as e:
            raise HTTPException(400, f"Uninstall failed: {e}")
        if not complete:
            # uninstall() disabled and unloaded the plugin; what did not
            # complete is deleting its files from disk (a locked file, an AV
            # hold, a permission denial, or - with delete_data - its data
            # directory).
            raise HTTPException(
                500,
                f"Plugin {name!r} was disabled and unloaded, but its files "
                f"could not be fully removed from disk; see the server log "
                f"for the cause.")
        return {"status": "uninstalled", "name": name}

    @app.post("/api/plugins/refresh",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def refresh_all_plugins_engine():
        return {"status": "ok", "refreshed": manager.refresh()}

    @app.post("/api/plugins/{name}/refresh",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def refresh_plugin_engine(name: str):
        _valid_name_or_404(name)
        try:
            refreshed = manager.refresh(name)
        except KeyError:
            raise HTTPException(404, f"No such installed builtin plugin: {name}")
        except Exception as e:
            raise HTTPException(400, f"Refresh failed: {e}")
        return {"status": "refreshed" if refreshed else "up-to-date", "name": name}

    @app.post("/api/plugins/{name}/enable",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def enable_plugin(name: str):
        _valid_name_or_404(name)
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
        _valid_name_or_404(name)
        try:
            manager.disable(name)
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"status": "disabled", "name": name}

    # Host-side dependency install (pip extras). In its own module so its
    # fastapi Request annotation resolves against module globals; this file
    # uses PEP 563 string annotations and imports fastapi lazily.
    from localm.plugins.deps_routes import register_dep_routes
    register_dep_routes(app, manager)

    # Background-job registry, registered at KERNEL level so a headless
    # ``localm serve`` has one too. Every consumer reads app.state.jobs PER
    # REQUEST, so registration order inside attach_engine does not matter.
    from localm.plugins.gui.jobs import JobManager
    from localm.plugins.gui.routes import jobs as _job_routes
    app.state.jobs = JobManager()
    _job_routes.register(app, app.state.jobs)

    return manager
