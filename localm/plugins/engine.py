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

import hashlib
import importlib.util
import json
import logging
import sys
import threading
import tomllib
from pathlib import Path
from typing import Any, Callable, Optional

from localm.plugins.contract import (API_VERSION, KNOWN_PLUGIN_KEYS,
                                     KNOWN_SURFACE_KEYS, PluginSpec, Surface)

# User-facing log of the sequential plugin load (one line per plugin). Distinct
# from the debug-only debuglog so it shows in normal server logs (U2).
_log = logging.getLogger("localm.plugins")

# Provenance sidecar written into an installed plugin dir. Records WHERE the copy
# came from ("store" = bundled first-party shelf, "external" = a third-party
# source dir) and a content hash of that source at install time. The refresh
# pass uses it to re-copy a builtin whose shipped source changed on upgrade,
# while never clobbering a third-party plugin. A hidden file, so discovery
# (which only walks directories) ignores it.
_PLUGIN_MARKER = ".localm-source.json"


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
    it as human-readable strings - surfaced, never escalated: a plugin with an
    unknown key must still parse and load (LM-DA-007)."""
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
    # Drop any modules cached from a PRIOR load of this plugin - the top-level
    # name AND its submodules (e.g. "<uniq>.backend"). _unload historically
    # popped only the top-level name, so a stale submodule could survive in
    # sys.modules and shadow a freshly installed/refreshed copy: re-running
    # `from . import backend` returns the cached stale module instead of reading
    # the new file. Purge the whole namespace so every module loads from the
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
        base = Path(self._spec.path or ".").resolve()
        d = (base / directory).resolve()
        # *directory* comes verbatim from a (possibly third-party) manifest or a
        # plugin's own mount_static call. Confine it to the plugin dir: reject
        # traversal ('../x') and absolute escapes (resolve() collapses both, and
        # symlink escapes) before mounting it as public static assets. The plugin
        # dir itself (d == base) is allowed.
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

    def on_startup(self, callback) -> None:
        """Run *callback* once the server's event loop is running.

        On a stock server start, plugins register() before uvicorn creates the
        loop, so loop-dependent work started here (e.g. the jobs scheduler)
        used to no-op silently and never run (memory-audit 2026-07-02 C2).
        With no running loop the callback is queued and the app lifespan runs
        it; with a running loop (plugin enabled at runtime via the management
        API) it runs immediately, so runtime toggling behaves as before.
        Failures are logged, never raised (a bad plugin must not take the
        server down)."""
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
        # A plugin disabled before the lifespan ran must not have its deferred
        # startup work fire later.
        self._manager.discard_startup_callbacks(self._spec.name)
        for r in self._routes:
            try:
                self._app.router.routes.remove(r)
            except ValueError:
                # ValueError = the route is already gone, which is the desired
                # end state; swallowing it keeps unmount idempotent (LM-DA-012).
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

    def _own_config_key(self, name: Optional[str]) -> str:
        """Confine plugin config r/w to the plugin's OWN block.

        A plugin must not read or tamper with ANOTHER plugin's persisted settings
        through the Host API, even under the install=owner-trust model: cross-plugin
        config access is a compartmentalisation / least-privilege break (a buggy or
        supply-chain-compromised plugin could corrupt or exfiltrate a sibling's
        config). ``name`` is accepted for backward compatibility (a plugin passing
        its own name is fine) but a DIFFERENT name is refused - it resolves to this
        plugin's own name, and the attempt is surfaced (we do not hide it)."""
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
        from localm.config import load_config, save_config
        key = self._own_config_key(name)
        c = load_config()
        c.setdefault("plugins", {})[key] = cfg if cfg is not None else {}
        save_config(c)

    def has_scope(self, scope: str) -> bool:
        # NOT implemented: the host has no request context - per-request scope
        # checks happen at the route dependency level (require_scope on the
        # routes mount_router adds). This used to return True unconditionally
        # while the contract said "raises if missing", so an author's guard
        # could never fire; raising makes that misuse fail loudly (LM-DA-008).
        raise NotImplementedError(
            "host-side scope checks are not implemented; scopes are enforced "
            "per-request on the routes mounted via mount_router()")

    def require_scope(self, scope: str) -> None:
        # See has_scope: same contract-honesty rule (LM-DA-008).
        raise NotImplementedError(
            "host-side scope checks are not implemented; scopes are enforced "
            "per-request on the routes mounted via mount_router()")

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
        # A chat hook is a powerful, otherwise-INVISIBLE capability: it sees and can
        # transform EVERY chat turn (message content, the model's reply), so a
        # compromised or unexpected plugin hook is a real injection / exfiltration
        # vector. Surface each registration (which plugin hooked which phase) so it
        # is discoverable, not silent - LM-DA-SEC-02, defense-in-depth traceability.
        # Debug altitude: legitimate first-party hooks (thinking / memory inlets)
        # register too, so this records rather than alarms.
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
        self._dep_tasks: dict = {}                   # name -> DepInstallTask (GUI installs)
        self._dep_tasks_lock = threading.Lock()
        # (plugin name, callback) queued by Host.on_startup before the server's
        # event loop exists; drained once by the app lifespan. See on_startup.
        self._startup_callbacks: list = []

    # ---- discovery (INSTALLED folder only) ---------------------------------
    # ---- deferred startup work (loop-dependent plugin init) ---------------- #
    # On a stock `localm serve`/`localm gui`, plugins register() BEFORE uvicorn
    # creates the event loop, so anything a plugin starts that needs a running
    # loop (the jobs scheduler) silently never ran (memory-audit 2026-07-02,
    # critical C2: no scheduled job ever fired on a normally-started server).
    # Host.on_startup queues such work here; the server lifespan drains it once
    # the loop exists. Runtime enable (loop already running) executes directly
    # in Host.on_startup and never lands here.

    def add_startup_callback(self, name: str, callback) -> None:
        self._startup_callbacks.append((name, callback))

    def discard_startup_callbacks(self, name: str) -> None:
        """Drop queued callbacks of *name* (plugin unloaded before the lifespan
        ran) so a disabled plugin's work can never fire later."""
        self._startup_callbacks = [
            (n, cb) for n, cb in self._startup_callbacks if n != name]

    def run_startup_callbacks(self) -> None:
        """Run every queued startup callback once. Called by the app lifespan
        with the event loop running. Best-effort per callback: one plugin's
        failure is logged (rule 5: discoverable, not silent) and must not stop
        the others or the server."""
        callbacks, self._startup_callbacks = self._startup_callbacks, []
        for name, cb in callbacks:
            try:
                cb()
            except Exception as e:
                _log.warning(
                    "plugin %s: deferred startup callback failed: %s", name, e)

    def discover(self) -> dict[str, PluginSpec]:
        """Discover INSTALLED plugins only (the installed folder). The store shelf
        is never discovered - it is just the source for install()."""
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
                        # LM-DA-007, rule-5 altitude: surface, do not escalate.
                        # A typoed key (client_entyr, default_enabld) means a
                        # tab/module/dependency quietly never materialises, so
                        # it is recorded per-plugin (shown in GUI/CLI via
                        # api_state) while the plugin still parses and loads.
                        # Logged only when new/changed: api_state re-discovers
                        # on every fetch and would otherwise spam the log.
                        msg = "warning: " + "; ".join(warns)
                        self._discover_errors[spec.name] = msg
                        if prior.get(spec.name) != msg:
                            _log.warning("plugin %s: %s", spec.name,
                                         "; ".join(warns))
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
    def load_enabled(
        self,
        on_event: Optional[Callable[[str, str, Optional[str]], None]] = None,
    ) -> None:
        """Discover INSTALLED plugins and load every active one (installed AND
        enabled). Never raises - a failing plugin is recorded in errors and
        skipped. A 'enabled' config entry for a plugin not on disk is ignored.

        Each plugin is loaded sequentially and reported as it goes: a line is
        logged (``localm.plugins`` logger) and, if given, *on_event* is called
        ``on_event(name, status, error)`` with status "loaded" or "failed" - the
        GUI uses this to show load progress (U2). A raising callback is ignored
        so it can never break startup."""
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
        register(host)
        # Serve a declared surface assets_dir even if register() did not mount it
        # itself, so a client_entry plugin's module never silently 404s.
        host.mount_surface_assets()
        self._loaded[spec.name] = (spec, module, host, uniq)
        self._errors.pop(spec.name, None)       # a successful load clears prior error
        self._maybe_fire_first_use(spec.name)   # REC-ONFIRSTUSE

    def _maybe_fire_first_use(self, name: str) -> None:
        """Invoke the ``on_first_use`` lifecycle hook exactly once per plugin - the
        first time it is loaded (its first activation). Persisted in config so it
        does not re-fire on every server start. Previously ``on_first_use`` was
        reserved in the contract but never invoked - a dead promise (REC-ONFIRSTUSE)."""
        # Only plugins that actually DEFINE on_first_use need first-use tracking.
        # Skipping the rest avoids a config write on every plugin's first load -
        # which would create files even in privacy mode (privacy = writes nothing).
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
            _write_marker(dest, "store", _dir_content_hash(src))
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

    # ---- refresh stale builtin copies (upgrade safety) ---------------------
    # An installed builtin is a COPY of the bundled store source taken at install
    # time. A localm upgrade ships newer store code, but the older installed copy
    # keeps shadowing it (install/provision no-op when the dir exists) - so the
    # user silently runs stale plugin code, including missing bug/security fixes.
    # The plugin VERSION is an unreliable signal (a bugfix often does not bump it
    # - e.g. image stayed v1.0.0 across a fix), so staleness is detected by a
    # CONTENT HASH of the store source recorded in the provenance marker.
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
        # move-in fails, put it back. The original is kept (at dest, or at the
        # backup if even the restore fails) so a refresh never destroys a plugin.
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
        # A third-party plugin must not shadow a built-in command name
        # (run/serve/config/coder/...). Builtins install via install() from the
        # trusted store; only arbitrary-source third-party installs are gated.
        # Reuse the legacy loader's set so the two loaders cannot drift.
        from localm.plugins.loader import _RESERVED_NAMES
        if name in _RESERVED_NAMES:
            raise ValueError(
                f"plugin name {name!r} clashes with a built-in command")
        dest = self._installed_dir(name)
        if dest.exists():
            if not force:
                raise ValueError(f"plugin {name!r} is already installed")
            self._remove_installed_dir(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest)
        _write_marker(dest, "external", _dir_content_hash(src))
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
        # A third-party plugin must not shadow a built-in command name
        # (run/serve/config/coder/...). Builtins install via install() from the
        # trusted store; only arbitrary-source third-party installs are gated.
        # Reuse the legacy loader's set so the two loaders cannot drift.
        from localm.plugins.loader import _RESERVED_NAMES
        if name in _RESERVED_NAMES:
            raise ValueError(
                f"plugin name {name!r} clashes with a built-in command")
        dest = self._installed_dir(name)
        if dest.exists():
            if not force:
                raise ValueError(f"plugin {name!r} is already installed")
            self._remove_installed_dir(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest)
        _write_marker(dest, "external", _dir_content_hash(src))
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
        self._require_deps_installed(name)            # REC-PLUGIN-REQUIRES
        self._maybe_refresh_builtin(name)             # pick up an upgraded builtin
        self.discover()                               # re-read the refreshed spec
        if name not in self._loaded:
            self._load(self._specs[name])             # load first; surface errors
        self._set_enabled(name, True)

    def _require_deps_installed(self, name: str) -> None:
        """Refuse to enable a plugin whose declared ``requires`` are not installed.
        Previously 'requires' was surfaced (a print/GUI warning) but never enforced,
        so a plugin could load with an unmet dependency (REC-PLUGIN-REQUIRES)."""
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
        if on:
            self._require_deps_installed(name)        # REC-PLUGIN-REQUIRES
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
        # REC-PLUGIN-REQUIRES: cascade-unload dependents. A plugin that declares
        # this one in `requires` must not keep running with the dependency gone;
        # disable + unload each (transitively) and log it, so the state stays
        # consistent instead of silently broken.
        for dep in self._dependents(name):
            if dep in self._enabled_set() or dep in self._loaded:
                _log.warning("disabling plugin %s: it requires %s, which is being "
                             "uninstalled", dep, name)
                try:
                    self._set_enabled(dep, False)
                    self._unload(dep)
                except Exception:
                    pass
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
                "requires_extras": spec.requires_extras if spec else [],
                # Declared pip-extra requirements NOT installed on this host, so
                # the GUI can offer a host-side "Install dependencies" action.
                "missing_deps": (
                    _deps.missing_requirements(
                        _deps.plugin_requirements(spec.requires_extras))
                    if spec and spec.requires_extras else []),
                "requires": spec.requires if spec else [],
                # Declared requirements that are not currently installed, so the
                # GUI can warn "requires X (missing)" + offer one-click install.
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

    @app.post("/api/plugins/refresh",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def refresh_all_plugins_engine():
        return {"status": "ok", "refreshed": manager.refresh()}

    @app.post("/api/plugins/{name}/refresh",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def refresh_plugin_engine(name: str):
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

    # Host-side dependency install (pip extras). Lives in its own module so its
    # fastapi Request annotation resolves against module globals (this file uses
    # PEP 563 string annotations and imports fastapi lazily).
    from localm.plugins.deps_routes import register_dep_routes
    register_dep_routes(app, manager)

    return manager
