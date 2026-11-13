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
import re
import stat
import sys
import threading
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from fastapi import Request

from localm.plugins.contract import (API_VERSION, KNOWN_PLUGIN_KEYS,
                                     KNOWN_SURFACE_KEYS,
                                     InferenceBackendDescriptor,
                                     PluginSettingField, PluginSpec, Surface)
from localm.plugins.ids import _check_plugin_name, _is_valid_plugin_name
from localm.plugins.lookup import SOURCE_MARKER

# Logger for the sequential plugin load, one line per plugin.
_log = logging.getLogger("localm.plugins")

# Provenance sidecar written into an installed plugin dir. Records where the
# install came from ('store' = the bundled first-party shelf, with
# ``package = true`` for a marker-only install; 'external' = a third-party
# source dir, with a content hash of that source at install time). Hidden, so
# discovery (which only walks directories) ignores it.
_PLUGIN_MARKER = SOURCE_MARKER


# --------------------------------------------------------------------------- #
#  Path safety: the plugin id, and an untrusted source tree                    #
# --------------------------------------------------------------------------- #

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
    except (OSError, ValueError, RecursionError):
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


def _write_store_install(dest: Path, src: Path) -> None:
    """Make *dest* the installed marker of the store plugin in *src*: a copy of
    its manifest and a provenance marker recording ``source = "store"`` and
    ``package = true``. No code is copied; the plugin runs from *src*. Raises
    OSError when either file cannot be written."""
    import shutil
    shutil.copyfile(Path(src) / "plugin.toml", Path(dest) / "plugin.toml")
    (Path(dest) / _PLUGIN_MARKER).write_text(
        json.dumps({"source": "store", "package": True}), encoding="utf-8")


# Eviction tasks started on the server loop's own thread, kept until they finish.
_BACKEND_EVICTIONS: set = set()


def _evict_backend_models(plugin_name: str) -> None:
    """Start, on the HTTP server's event loop, the unload of every loaded model
    that runs on an inference backend *plugin_name* registered
    (``http_server.unload_models_of_backend_plugin``), and return without
    waiting for it. Does nothing in a process with no running server loop,
    which holds no server engines. Each unloaded model is logged at INFO; a
    model that stays loaded, or a failed eviction, at WARNING."""
    import asyncio
    hs = sys.modules.get("localm.inference.http_server")
    loop = getattr(hs, "_server_loop", None)
    if loop is None or not loop.is_running():
        return
    coro = hs.unload_models_of_backend_plugin(plugin_name)
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        fut = loop.create_task(coro)
        _BACKEND_EVICTIONS.add(fut)
        fut.add_done_callback(_BACKEND_EVICTIONS.discard)
    else:
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
    fut.add_done_callback(lambda f: _log_backend_eviction(plugin_name, f))


def _log_backend_eviction(plugin_name: str, fut) -> None:
    """Log the outcome of an ``_evict_backend_models`` run."""
    if fut.cancelled():
        _log.warning("plugin %s: unloading the models on its inference backend "
                     "was cancelled; they may still be loaded", plugin_name)
        return
    exc = fut.exception()
    if exc is not None:
        _log.warning("plugin %s: unloading the models on its inference backend "
                     "failed (%s: %s); they may still be loaded",
                     plugin_name, type(exc).__name__, exc)
        return
    for result in fut.result():
        model = result.get("model")
        if result.get("status") in ("unloaded", "already_unloaded"):
            _log.info("plugin %s: unloaded model %s, which ran on its inference "
                      "backend", plugin_name, model)
        else:
            _log.warning("plugin %s: model %s, which runs on its inference "
                         "backend, is still loaded (%s%s)", plugin_name, model,
                         result.get("status"),
                         f": {result['detail']}" if result.get("detail") else "")


def _purge_plugin_modules(uniq: str) -> None:
    """Remove a plugin's module namespace ("<uniq>" and every "<uniq>.*"
    submodule) from sys.modules so the next import reads fresh from disk."""
    for cached in [k for k in list(sys.modules)
                   if k == uniq or k.startswith(uniq + ".")]:
        sys.modules.pop(cached, None)


# --------------------------------------------------------------------------- #
#  Manifest parsing (the richer contract; superset of loader.PluginManifest)  #
# --------------------------------------------------------------------------- #

_DATA_SUBDIR_RE = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")


def plugin_data_dir(spec: PluginSpec, home: Path) -> Optional[Path]:
    """The data folder *spec* declares with ``data_subdir``, under the data dir
    *home*: ``<home>/<data_subdir>`` for a store-installed plugin and
    ``<home>/plugin-data/<name>/<data_subdir>`` for a third-party one. None when
    it declares none."""
    if not spec.data_subdir:
        return None
    if spec.builtin:
        return Path(home) / spec.data_subdir
    return Path(home) / "plugin-data" / spec.name / spec.data_subdir


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
    except UnicodeDecodeError as e:
        raise ValueError(f"{manifest} is not valid UTF-8: {e}") from e
    except RecursionError as e:
        raise ValueError(f"invalid TOML in {manifest}: nested too deeply") from e
    except OSError as e:
        raise ValueError(f"cannot read {manifest}: {e}") from e

    p = data.get("plugin")
    if not isinstance(p, dict):
        raise ValueError(f"{manifest}: missing [plugin] table")
    name = p.get("name", "")
    if not name or not isinstance(name, str) or not name.replace("-", "_").isidentifier():
        raise ValueError(f"{manifest}: invalid or missing plugin name")

    try:
        api_version = int(p.get("api_version", API_VERSION))
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(f"{manifest}: [plugin] api_version must be an integer") from e
    str_lists = {}
    for key in ("requires_extras", "requires", "capabilities"):
        value = p.get(key, []) or []
        if not (isinstance(value, list) and all(isinstance(t, str) for t in value)):
            raise ValueError(f"{manifest}: [plugin] {key} must be a list of strings")
        str_lists[key] = list(value)
    from localm.plugins.deps import is_valid_extra_name
    bad = [e for e in str_lists["requires_extras"] if not is_valid_extra_name(e)]
    if bad:
        raise ValueError(f"{manifest}: [plugin] requires_extras has an entry that is not "
                         f"a pip extra name: {bad[0]!r}")

    # [tools] exports must be a list of strings.
    _tools = data.get("tools", {})
    exports = _tools.get("exports", []) if isinstance(_tools, dict) else []
    if not (isinstance(exports, list) and all(isinstance(t, str) for t in exports)):
        raise ValueError(f"{manifest}: [tools] exports must be a list of strings")

    data_subdir = p.get("data_subdir", "")
    if not isinstance(data_subdir, str) or (
            data_subdir and not _DATA_SUBDIR_RE.fullmatch(data_subdir)):
        raise ValueError(f"{manifest}: [plugin] data_subdir must be one folder name "
                         f"(letters, digits, '.', '_' or '-', not starting with "
                         f"'.'), got {data_subdir!r}")

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
        api_version=api_version,
        description=str(p.get("description", "")),
        scope=str(p.get("scope", "") or name),
        requires_extras=str_lists["requires_extras"],
        requires=str_lists["requires"],
        capabilities=str_lists["capabilities"],
        data_subdir=data_subdir,
        builtin=builtin,
        protected=builtin and bool(p.get("protected", False)),
        default_enabled=builtin and bool(p.get("default_enabled", False)),
        surface=surface,
        cli_entry=str(p.get("cli", "")),
        register_entry=str(p.get("register", "")),
        entry=str(p.get("entry", "")),
        tool_exports=list(exports),
        path=str(plugin_dir),
    )


def _entry_module_file(base: Path, mod_name: str) -> Optional[Path]:
    """The file of module *mod_name* inside plugin dir *base*: ``<mod>.py`` or
    ``<mod>/__init__.py``; a dotted name is followed into sub-packages only when
    every part is an identifier."""
    candidates = [base / f"{mod_name}.py", base / mod_name / "__init__.py"]
    parts = mod_name.split(".")
    if len(parts) > 1 and all(p.isidentifier() for p in parts):
        nested = base.joinpath(*parts)
        candidates += [nested.with_name(nested.name + ".py"), nested / "__init__.py"]
    return next((c for c in candidates if c.is_file()), None)


def _package_module_name(spec: PluginSpec, mod_name: str) -> Optional[str]:
    """The real module name of *mod_name* when *spec* is a first-party plugin
    whose folder is in the localm package, else None."""
    if not spec.builtin or not spec.path:
        return None
    from localm.plugins.lookup import PACKAGE, package_root
    try:
        d = Path(spec.path).resolve()
        if d.parent != package_root().resolve():
            return None
    except OSError:
        return None
    if not (d.name.isidentifier()
            and all(p.isidentifier() for p in mod_name.split("."))):
        return None
    return f"{PACKAGE}.{d.name}.{mod_name}"


def _is_package_module(uniq: str) -> bool:
    """True when *uniq* is a module of the localm package (a first-party plugin
    loaded under its real name), which is never dropped from sys.modules."""
    from localm.plugins.lookup import PACKAGE
    return uniq == PACKAGE or uniq.startswith(PACKAGE + ".")


def _drop_plugin_modules(uniq: str) -> None:
    """``_purge_plugin_modules`` for a plugin loaded from a copied folder; a
    no-op for a first-party plugin loaded from the localm package, so it keeps
    one module instance per process."""
    if not _is_package_module(uniq):
        _purge_plugin_modules(uniq)


def _import_module(spec: PluginSpec):
    """Import the plugin's module. The module name in register_entry is
    '<module>' or '<module>:<attr>'. A first-party plugin whose folder is in
    the localm package is imported by its real name
    (``localm.plugins.builtin.<name>.<module>``, reusing an already imported
    module); any other plugin is imported fresh from <module>.py (or
    <module>/__init__.py) in its directory under a synthetic name. Returns
    (module, register_attr_name, module_name)."""
    entry = spec.register_entry or "plugin"
    mod_name, _, attr = entry.partition(":")
    attr = attr or "register"
    base = Path(spec.path)
    mod_file = _entry_module_file(base, mod_name)
    if mod_file is None:
        raise ValueError(f"plugin {spec.name!r}: module {mod_name!r} not found in {base}")
    real = _package_module_name(spec, mod_name)
    if real is not None:
        return importlib.import_module(real), attr, real
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


def _scope_problem(spec: PluginSpec, claimed: dict) -> Optional[str]:
    """Why *spec*'s scope cannot be used by a third-party plugin, or None.
    *claimed* maps a scope to the installed plugin that already uses it."""
    from localm import scopes as S
    if spec.scope in S.all_known_scopes():
        return (f"plugin {spec.name!r} declares scope {spec.scope!r}, which is a "
                f"reserved localm scope and cannot be claimed by a third-party "
                f"plugin")
    other = claimed.get(spec.scope)
    if other and other != spec.name:
        return (f"plugin {spec.name!r} declares scope {spec.scope!r}, which is "
                f"already used by installed plugin {other!r}")
    return None


def import_tool_module(spec: PluginSpec):
    """Import the module that holds a plugin's ``[tools] exports``: its legacy
    ``entry`` module, else its ``register`` module. Reuses the module the engine
    already loaded for the plugin when it is the same file."""
    import dataclasses
    entry = spec.entry or spec.register_entry
    if not entry:
        raise ValueError(f"plugin {spec.name!r}: no entry module for tool exports")
    use = dataclasses.replace(spec, register_entry=entry)
    mod_name = entry.partition(":")[0]
    if _package_module_name(spec, mod_name) is not None:
        return _import_module(use)[0]
    uniq = f"_localm_plugin_{spec.name.replace('-', '_')}"
    base = Path(spec.path or ".")
    want = _entry_module_file(base, mod_name)
    cached = sys.modules.get(uniq)
    cached_file = getattr(cached, "__file__", None)
    try:
        if (cached is not None and cached_file and want is not None
                and Path(cached_file).samefile(want)):
            return cached
    except OSError:
        pass
    return _import_module(use)[0]


# --------------------------------------------------------------------------- #
#  Host: the API a plugin uses to attach itself                               #
# --------------------------------------------------------------------------- #

_ROUTE_PREFIXES = ("/api/", "/v1/")


def _mounted_paths(route) -> list:
    """The URL paths *route* serves. An included router (FastAPI keeps it lazy
    and path-less) is expanded into the effective paths of its routes."""
    accessor = getattr(route, "effective_route_contexts", None)
    if callable(accessor):
        out = []
        contexts: Any = accessor()
        for ctx in contexts:
            starlette_route = getattr(ctx, "starlette_route", None)
            target = starlette_route if starlette_route is not None else ctx
            out.append(str(getattr(target, "path", "")))
        return out
    return [str(getattr(route, "path", ""))]


class PluginHost:
    """Concrete `contract.Host`. One per loaded plugin; tracks what it mounted
    so it can be cleanly removed on unload."""

    def __init__(self, app, manager: PluginManager, spec: PluginSpec) -> None:
        self.api_version = API_VERSION
        self._app = app
        self._manager = manager
        self._spec = spec
        self._routes: list = []
        self._chat_phases: list = []        # chat-pipeline phases this plugin hooked
        self._backends: list = []           # inference backends this plugin registered
        self._static_prefixes: set = set()  # URL prefixes this plugin already serves
        self._stoppers: list = []           # stop callables from track()
        self.settings: list = []
        self.surface: Optional[Surface] = spec.surface or None
        self._surface_assets_mounted = False
        self.model_roles: list = []


    def mount_router(self, router) -> None:
        """Mount *router* on the live app, gating every route with the plugin's
        capability scope, and remember the routes added for later removal. With
        no app (a backend-only load, see ``PluginManager.load_backend_plugins``)
        nothing is mounted."""
        if self._app is None:
            self.audit("router_skipped_no_app", {})
            return
        from fastapi import Depends
        from localm.inference.http_server import require_scope
        before = {id(r) for r in self._app.router.routes}
        self._app.include_router(
            router, dependencies=[Depends(require_scope(self._spec.scope))])
        new = [r for r in self._app.router.routes if id(r) not in before]
        outside = [path for r in new for path in _mounted_paths(r)
                   if not path.startswith(_ROUTE_PREFIXES)]
        if outside:
            for r in new:
                self._app.router.routes.remove(r)
            self._app.openapi_schema = None
            raise ValueError(
                f"plugin {self._spec.name!r}: route path(s) {outside!r} are outside "
                f"/api/ and /v1/")
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
        if self._app is None:
            self.audit("static_skipped_no_app", {"prefix": prefix})
            return prefix
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
        ignored, never fatal. Uses the surface :meth:`register_tab` set, else
        the manifest's."""
        self._surface_assets_mounted = True
        surface = self.surface
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

    def track(self, stop_callable) -> None:
        """Register *stop_callable* (no arguments) to run when this plugin is
        unmounted, so a thread or timer the plugin started can be stopped. A
        stopper that raises is logged and does not stop the others."""
        if not callable(stop_callable):
            raise ValueError("track() needs a callable")
        self._stoppers.append(stop_callable)

    def unmount(self) -> None:
        stoppers, self._stoppers = self._stoppers, []
        for stop in reversed(stoppers):
            try:
                stop()
            except Exception as e:
                _log.warning("plugin %s: stop callback failed: %s",
                             self._spec.name, e)
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
        if self._backends:
            from localm.inference import backend_registry
            backend_registry.unregister_plugin(self._spec.name)
            self._backends = []
            _evict_backend_models(self._spec.name)
        if self._app is None:
            return
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
        """Replace the plugin's GUI surface (tab, label, icon, group, settings
        group, assets_dir, client_entry) with *surface* while the plugin is
        loaded; the manifest's ``[surface]`` applies again once it is unloaded.
        Its ``assets_dir`` is served like a manifest one, confined to the
        plugin's folder; a call made after the engine mounted the surface
        assets mounts its assets_dir at once unless ``/plugins/<name>`` is
        already served. Raises TypeError for anything but a
        ``contract.Surface``."""
        if not isinstance(surface, Surface):
            raise TypeError(
                f"plugin {self._spec.name!r}: register_tab() takes a Surface (see "
                f"localm.plugins.contract), got {type(surface).__name__}")
        self.surface = surface
        if self._surface_assets_mounted:
            self.mount_surface_assets()

    def register_model_role(self, descriptor) -> None:
        from localm.model_manager.registry import MODEL_TYPES
        if descriptor.model_type not in MODEL_TYPES:
            raise ValueError(f"Invalid model_type {descriptor.model_type!r}; must be one of {MODEL_TYPES}")
        descriptor.plugin_name = self._spec.name
        self.model_roles.append(descriptor)

    def register_inference_backend(self, descriptor) -> None:
        """Register *descriptor* (a ``contract.InferenceBackendDescriptor``) in
        the process-wide backend registry, stamped with this plugin's name. A
        store-installed plugin's descriptors are asked before a third-party
        plugin's. Removed again by :meth:`unmount`."""
        import dataclasses

        from localm.inference import backend_registry
        if not isinstance(descriptor, InferenceBackendDescriptor):
            raise TypeError(
                f"plugin {self._spec.name!r}: register_inference_backend() takes an "
                f"InferenceBackendDescriptor (see localm.plugins.contract), got "
                f"{type(descriptor).__name__}")
        stamped = dataclasses.replace(descriptor, plugin_name=self._spec.name)
        backend_registry.register(stamped, first_party=self._spec.builtin)
        self._backends.append(stamped)
        self.audit("inference_backend_registered", {"backend": stamped.name})


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

    def set_global_config(self, request: Any, key: str, value: Any) -> None:
        """Write global config *key* with the owner tier of the settings schema
        applied to the caller of *request*. Raises ``PermissionError`` for an
        owner-only key and a non-owner caller, ``ValueError`` for an unknown key
        or a value the schema rejects."""
        from localm.config_write import set_global_config_key
        set_global_config_key(key, value, is_owner=self.caller_is_owner(request))

    def caller_is_owner(self, request: Any) -> bool:
        """True when the caller of *request* is the owner: open mode (no key
        configured anywhere) or an ADMIN/owner key or session. False for a
        scoped key and for a request with no valid key while one is
        configured."""
        from localm.inference.http_server import caller_is_operator
        return caller_is_operator(request)

    def caller_principal(self, request: Any) -> Optional[str]:
        """A stable, opaque identity for the key behind *request* (the same
        SHA-256 the keystore stores, never the key itself), identical for the
        bearer header and the session cookie. None in open mode or when no
        valid key is presented."""
        from localm.inference.http_server import principal_id
        return principal_id(request)

    def caller_owns(self, request: Any, owner: Optional[str]) -> bool:
        """Whether the caller of *request* may act on a resource recorded as
        created by *owner* (a value :meth:`caller_principal` returned when it
        was created). True when *owner* is None, for an ADMIN/owner key, and
        when the caller's principal equals *owner*."""
        from localm.inference.http_server import job_owner_ok
        return job_owner_ok(request, owner)

    def engine(self) -> Any:
        return self._manager.inference_engine

    def driving_engine(self, engine: Any = None):
        """Context manager: pin *engine* (or engine(), if not passed) as busy and
        reset its idle-unload clock for the DURATION of a real generation call.

        Wrap this around the actual chat_stream/complete call - never around a
        bare engine()/inference_engine access used only to check .loaded or read
        a name. Raises HTTPException 503, without running the block, when the
        engine is being unloaded. See localm.inference.http_server.driving_engine
        for the mechanism."""
        from localm.inference.http_server import driving_engine as _driving_engine
        return _driving_engine(engine if engine is not None else self.engine())

    def audit(self, event: str, data: dict) -> None:
        try:
            from localm.debuglog import logger as _dbg
            _dbg.debug("plugin %s: %s %s", self._spec.name, event, data)
        except Exception:
            pass

    def browse_dirs(self, path: str, request: Any) -> dict:
        """Server-side folder-picker helper: immediate subdirectories of *path*
        (blank -> the user's home). Hidden dirs are omitted. Raises
        PermissionError unless the caller of *request* has host filesystem
        access (``http_server.effective_fs_access``)."""
        from localm.inference.http_server import effective_fs_access
        if effective_fs_access(request) != "host":
            raise PermissionError("this caller does not have host filesystem access")
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
        pipeline = getattr(getattr(self._app, "state", None), "chat_pipeline", None)
        if pipeline is None:
            # No app, or no chat pipeline on it (a backend-only load, a
            # bare-FastAPI test harness): the hook is inert.
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
    """The bundled STORE shelf (the ``localm.plugins.builtin`` package). An
    installed first-party plugin is a marker folder in the installed folder;
    its manifest and code are read from here."""
    from localm.plugins.lookup import package_root
    d = package_root()
    return d if d.is_dir() else None


class PluginManager:
    """Discovers INSTALLED plugins (in the installed folder) and loads / unloads /
    enables / disables them at runtime on *app*. Installing a first-party plugin
    writes a marker folder (its manifest and a provenance marker) into the
    installed folder; installing a third-party plugin copies its source there.

    Two locations: the STORE (bundled shelf, ``store_root``: the manifest and code
    of every first-party plugin) and the INSTALLED folder (``installed_root``, the
    ONLY place discovery looks for what is installed). A plugin not in the
    installed folder does not exist as far as localm is concerned. "Installed"
    therefore means physically present in installed_root;
    "enabled" is a config toggle within installed; active = installed AND enabled.
    """

    def __init__(self, app, inference_engine=None,
                 store_root: Optional[Path] = None,
                 installed_root: Optional[Path] = None,
                 # back-compat aliases: builtin_root was the store,
                 # external_root the installed/discovery dir
                 builtin_root: Optional[Path] | object = _UNSET,
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
        self._provision_errors: dict[str, str] = {}  # first-run copy / upgrade refresh
        self._store_errors: dict[str, str] = {}      # unparsable bundled manifests
        self._unload_errors: dict[str, str] = {}     # unregister() failures; cleared by a later load
        self.last_refresh_failures: dict[str, str] = {}
        self._dep_tasks: dict = {}                   # name -> DepInstallTask (GUI installs)
        self._dep_tasks_lock = threading.Lock()
        # Serializes discover(), which may run on a worker thread
        # (load_enabled_pending_off_loop, api_state off the loop) while the
        # event loop calls it too. The config file lock is taken under it;
        # never take this lock while holding the config file lock.
        self._discover_lock = threading.RLock()
        # Bumped when a plugin is disabled or unloaded, so a pending load
        # computed off the loop before that change is dropped.
        self._disable_gen = 0
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
        for entry in self._loaded.values():
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
            surface = self.surface_of(name)
            label = (surface.settings_group
                     if surface and surface.settings_group else name)
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
        with self._discover_lock:
            return self._discover_locked()

    def _discover_locked(self) -> dict[str, PluginSpec]:
        """discover() body. Caller must hold ``_discover_lock``. Builds the new
        spec and error maps aside and swaps them in at the end, so a reader on
        another thread sees either the previous complete map or the new one."""
        self._ensure_preinstalled()
        specs: dict[str, PluginSpec] = {}
        prior = self._discover_errors        # for change-only warning logs below
        errors: dict[str, str] = {}
        root = self._installed_root
        if root:
            try:
                children = sorted(Path(root).glob("*"))
            except OSError:
                children = []
            for child in children:
                if child.name.startswith("."):
                    continue
                if not child.is_dir() or not (child / "plugin.toml").is_file():
                    continue
                try:
                    warns: list = []
                    store_src = self._store_source_for(child)
                    if store_src is not None:
                        spec = parse_spec(store_src, builtin=True, warnings=warns)
                    else:
                        spec = parse_spec(child, builtin=False, warnings=warns)
                    if not spec.compatible():
                        raise ValueError(
                            f"api_version {spec.api_version} != {API_VERSION}")
                    specs[spec.name] = spec
                    if warns:
                        # Record unknown manifest keys per plugin (surfaced via
                        # api_state) while the plugin still parses and loads.
                        # Logged only when new or changed, since api_state
                        # re-discovers on every fetch.
                        msg = "warning: " + "; ".join(warns)
                        errors[spec.name] = msg
                        if prior.get(spec.name) != msg:
                            _log.warning("plugin %s: %s", spec.name,
                                         "; ".join(warns))
                except Exception as e:       # one bad manifest must not break discovery
                    errors[child.name] = str(e)
        self._drop_scope_collisions(specs, errors)
        self._specs = specs
        self._discover_errors = errors
        return specs

    @staticmethod
    def _drop_scope_collisions(specs: dict, errors: dict) -> None:
        """Remove from *specs* every third-party spec whose scope is reserved
        or already claimed by a first-party plugin or an earlier (by name)
        third-party plugin, and record why in *errors*. First-party means
        installed from the store (``PluginSpec.builtin``), never a name."""
        first_party = {n for n, sp in specs.items() if sp.builtin}
        claimed = {sp.scope: n for n, sp in specs.items() if n in first_party}
        for name in sorted(specs):
            if name in first_party:
                continue
            reason = _scope_problem(specs[name], claimed)
            if reason:
                specs.pop(name)
                errors[name] = reason
                _log.warning("plugin %s: %s", name, reason)
            else:
                claimed[specs[name].scope] = name

    def _store_dir(self, name: str) -> Optional[Path]:
        # Returns None for a non-conforming id, before the join, so no
        # traversing id probes for a plugin.toml outside the store root.
        # Callers turn None into a no-such-builtin-plugin KeyError.
        if not self._store_root or not _is_valid_plugin_name(name):
            return None
        d = Path(self._store_root) / name
        return d if (d / "plugin.toml").is_file() else None

    def _store_source_for(self, installed: Path) -> Optional[Path]:
        """The store folder an installed folder was installed from: the store
        has a plugin of the same name and the folder's marker does not record
        a third-party source. None for a third-party install. The spec, and for
        the bundled store the code, of a first-party install come from there."""
        from localm.plugins.lookup import installed_from_store
        src = self._store_dir(Path(installed).name)
        if src is None or not installed_from_store(installed):
            return None
        return src

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
                self._store_errors.pop(child.name, None)
            except Exception as e:
                msg = f"bundled manifest unreadable: {e}"
                if self._store_errors.get(child.name) != msg:
                    _log.warning("plugin %s: %s", child.name, msg)
                self._store_errors[child.name] = msg
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
                    if child.name.startswith("."):
                        continue
                    if child.is_dir() and (child / "plugin.toml").is_file():
                        out.add(child.name)
            except OSError:
                pass
        return out

    def _enabled_set(self) -> set:
        """Names enabled in config["plugins_enabled"], plus every protected
        plugin (chat), which is always enabled and never stored there."""
        from localm.config import load_config
        from localm.plugins import catalog as _cat
        return set(load_config().get("plugins_enabled", [])) | set(_cat.protected())

    def _set_enabled(self, name: str, on: bool) -> None:
        """Add/remove *name* in config["plugins_enabled"] atomically (read-modify-
        write under the I/O lock), so concurrent toggles can't lose updates."""
        from localm.config import update_config

        def _mutate(cfg: dict) -> None:
            cur = set(cfg.get("plugins_enabled", []))
            cur.add(name) if on else cur.discard(name)
            cfg["plugins_enabled"] = sorted(cur)
        if not on:
            self._disable_gen += 1
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
        self._refresh_installed_builtins()         # upgrade: convert old copied installs
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
            if self._specs[name].tools_only:
                continue                           # coder tool plugin: nothing to load
            if name in enabled and name not in self._loaded:
                self._safe_load(self._specs[name])
                if name in self._loaded:
                    _emit(name, "loaded", None)
                else:
                    _emit(name, "failed", self._errors.get(name))

    def load_backend_plugins(self) -> list:
        """Load every installed, enabled plugin whose manifest declares the
        ``inference-backend`` capability and is not loaded yet, for a process
        with no server app (an in-process ``localm run``). Returns the names
        loaded. Never raises: a failing plugin is recorded in errors and logged
        at WARNING. Routes, static mounts and chat hooks a plugin registers are
        skipped here (there is no app); its inference backend is registered."""
        self.discover()
        enabled = self._enabled_set()
        loaded = []
        for name in sorted(self._specs):
            spec = self._specs[name]
            if (name not in enabled or name in self._loaded or spec.tools_only
                    or "inference-backend" not in (spec.capabilities or [])):
                continue
            self._safe_load(spec)
            if name in self._loaded:
                loaded.append(name)
            else:
                _log.warning("plugin %s failed to load: %s", name, self._errors.get(name))
        return loaded

    def load_enabled_pending(self) -> list:
        """Load every installed, enabled plugin that is not loaded yet and has no
        recorded load error, mounting its routes on the live app. Covers a plugin
        installed or enabled by another process (``localm plugin install``) while
        this server runs. Returns the names loaded. Never raises: a failing plugin
        is recorded in errors, as at startup. Runs on the event loop thread, like
        enable(). Returns [] without loading when the manager has no app."""
        if self.app is None:
            return []
        return self._load_pending(self._pending_specs())

    async def load_enabled_pending_off_loop(self) -> list:
        """``load_enabled_pending`` for a request handler: discovery and the
        config read run on a worker thread, and only the import and register()
        of a plugin that is actually pending run on the event loop, as enable()
        does. Returns the names loaded; [] without loading when the manager has
        no app, or when a plugin was disabled or unloaded while discovery ran
        (the next call loads what is still pending). Never raises for a failing
        plugin, which is recorded in errors."""
        if self.app is None:
            return []
        from starlette.concurrency import run_in_threadpool
        gen = self._disable_gen
        specs = await run_in_threadpool(self._pending_specs)
        if gen != self._disable_gen:
            return []
        return self._load_pending(specs)

    def _pending_specs(self) -> list:
        """Discover, then return the specs of installed, enabled plugins that
        are not loaded, have no recorded load error and are not tools-only.
        Safe to call on a worker thread."""
        specs = self.discover()
        enabled = self._enabled_set()
        loaded, errors = set(self._loaded), set(self._errors)
        return [specs[n] for n in sorted(specs)
                if n in enabled and n not in loaded and n not in errors
                and not specs[n].tools_only]

    def _load_pending(self, specs: list) -> list:
        """Load each of *specs* that is still not loaded and has no recorded
        load error. Runs on the event loop thread. Returns the names loaded."""
        loaded = []
        for spec in specs:
            name = spec.name
            if name in self._loaded or name in self._errors:
                continue
            self._safe_load(spec)
            if name in self._loaded:
                loaded.append(name)
        return loaded

    def _ensure_preinstalled(self) -> None:
        """First-run provisioning for preinstalled plugins (chat is plugin #0).
        Copy each from the store into the installed folder if absent, and enable
        those marked default_enabled at that first provisioning - so chat ships
        installed + enabled and self-heals if its directory is removed, while a
        user's later disable of a non-protected default_enabled plugin is honoured.
        Best-effort: a missing store source (e.g. a synthetic test store) is
        recorded and skipped, never fatal."""
        from localm.plugins import catalog as _cat
        protected = set(_cat.protected())
        for name in _cat.preinstalled():
            if (self._installed_dir(name) / "plugin.toml").is_file():
                if name in protected and name not in self._enabled_set():
                    try:
                        self._set_enabled(name, True)
                        self._clear_provision_error(name, "preinstall")
                    except Exception as e:
                        self._note_provision_error(name, f"preinstall-enable: {e}")
                continue                            # already installed on disk
            try:
                self._provision_from_store(name)
            except Exception as e:                  # no store source / copy failed
                self._note_provision_error(name, f"preinstall: {e}")
                continue
            try:
                spec = parse_spec(self._installed_dir(name), builtin=True)
                if spec.default_enabled and name not in self._enabled_set():
                    self._set_enabled(name, True)
                self._clear_provision_error(name, "preinstall")
            except Exception as e:
                self._note_provision_error(name, f"preinstall-parse: {e}")

    def _clear_provision_error(self, name: str, kind: str) -> None:
        """Drop *name*'s recorded provisioning error when it is of *kind*
        ("preinstall" or "refresh")."""
        if (self._provision_errors.get(name) or "").startswith(kind):
            self._provision_errors.pop(name, None)

    def _note_provision_error(self, name: str, msg: str) -> None:
        """Record *msg* for *name* in api_state's errors (kept across discover())
        and log it at WARNING when it is new or changed."""
        if self._provision_errors.get(name) != msg:
            _log.warning("plugin %s: %s", name, msg)
        self._provision_errors[name] = msg

    def _safe_load(self, spec: PluginSpec) -> None:
        try:
            self._load(spec)
        except Exception as e:
            self._errors[spec.name] = f"load failed: {e}"

    def _load(self, spec: PluginSpec) -> None:
        """Import *spec*'s module, call its register(host) and record it as
        loaded. A tools-only manifest (``entry`` and no ``register``) is left
        unloaded with no error. Raises when the module or its register callable
        is missing, or register() raises."""
        if spec.tools_only:
            self._errors.pop(spec.name, None)
            return
        module, attr, uniq = _import_module(spec)
        register = getattr(module, attr, None)
        if not callable(register):
            _drop_plugin_modules(uniq)
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
            _drop_plugin_modules(uniq)
            raise
        self._loaded[spec.name] = (spec, module, host, uniq)
        self._errors.pop(spec.name, None)       # a successful load clears prior error
        self._unload_errors.pop(spec.name, None)
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
        self._disable_gen += 1
        entry = self._loaded.pop(name, None)
        if not entry:
            return
        spec, module, host, uniq = entry
        try:
            unreg = getattr(module, "unregister", None)
            if callable(unreg):
                unreg()
        except Exception as e:
            self._unload_errors[name] = f"unregister failed during unload: {e}"
            _log.warning("plugin %s: unregister failed during unload: %s", name, e,
                         exc_info=True)
        host.unmount()
        _drop_plugin_modules(uniq)      # a copied plugin re-imports fresh on re-enable

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

    def _installed_dir_on_disk(self, name: str) -> bool:
        """True when the installed-plugins root DIRECTLY contains a directory
        literally named *name*, manifest or not - the same raw-name tolerance
        _installed_set() uses (a directory need not equal any manifest name),
        without requiring plugin.toml to be present.

        LEXICAL first, before any syscall: *name* must be exactly one path
        component (``name == Path(name).name``, the same rule
        _is_valid_plugin_name applies, without ITS added identifier-shape
        requirement - an odd but legitimate basename like 'coolplugin-1.0'
        must still pass). Without this, a UNC-shaped name (``\\\\host\\share``)
        joined with ``/`` onto root does not stay confined - an ABSOLUTE
        right-hand operand makes pathlib's ``/`` discard the left side
        entirely - and the is_dir()/resolve() below would dial that host on
        Windows before the resolved-parent check below ever runs.

        Confined AGAIN by RESOLVED PARENT after that, exactly like
        _remove_installed_dir: a traversing name that is lexically one
        component but resolves outside root (e.g. a symlink) is still
        refused."""
        root = self._installed_root
        if not root or not name or name != Path(name).name:
            return False
        d = Path(root) / name
        try:
            return d.is_dir() and d.resolve().parent == Path(root).resolve()
        except OSError:
            return False

    def _provision_from_store(self, name: str) -> bool:
        """Install the plugin from the store: create its folder in the installed
        folder holding only its manifest and the provenance marker
        (``_write_store_install``); the code stays in the store. No-op if it is
        already installed. Raises NotImplementedError when the plugin is
        missing from the store but its catalog entry has a repo URL (not fetched),
        and KeyError when no source exists.

        A directory already present at the destination with no manifest (an
        interrupted or partial previous install) is replaced rather than
        left in place: it is removed first, and ValueError propagates if that
        removal fails, before anything is copied.

        Returns True only if THIS call created the directory. The caller must
        pass that through to _provision_and_verify's rollback: rolling back a
        directory this call did not create is destructive."""
        import shutil
        dest = self._installed_dir(name)
        if (dest / "plugin.toml").is_file():
            return False                             # already installed on disk
        if dest.exists():
            _log.warning(
                "plugin %s: the installed directory exists with no manifest "
                "(an interrupted or partial install); replacing it", name)
            if not self._remove_installed_dir(name):
                raise ValueError(
                    f"plugin {name!r}: a stale installed directory could not "
                    f"be removed; see the log")
        src = self._store_dir(name)
        if src is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.parent / f".{name}.install.tmp"
            try:
                if tmp.exists():
                    shutil.rmtree(tmp)
                tmp.mkdir()
                _write_store_install(tmp, src)
                tmp.rename(dest)
            except BaseException:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
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

    # ---- convert copied first-party installs (upgrade from copy installs) ----
    # An older localm installed a first-party plugin as a full copy of its store
    # folder. The engine no longer imports that copy (discover() takes the spec
    # and the code from the store); this deletes the copied code and keeps every
    # other file found in the folder.
    def _migrate_installed_copy(self, name: str) -> bool:
        """Turn an installed folder that holds a copy of the store plugin
        *name* into the marker-only install (``_write_store_install``). Files at
        a path the store plugin also ships, and compiled ``__pycache__`` /
        ``*.pyc`` files, are deleted; any other file is kept in place and logged
        at WARNING. Returns True when a copy was converted, False when there was
        nothing to convert (no store plugin of that name, not installed, a
        third-party install, or already a marker-only install).

        The marker is written before anything is deleted, so an interrupted
        conversion leaves a folder the engine already treats as a marker-only
        install; the next call deletes what remains. Raises RuntimeError when
        some copied files could not be deleted (the plugin still runs from the
        store; the log names each file)."""
        store_src = self._store_dir(name)
        if store_src is None:
            return False
        dest = self._installed_dir(name)
        if self._store_source_for(dest) is None:
            return False
        leftovers = self._copied_files(dest, store_src)
        marker = _read_marker(dest)
        if marker and marker.get("package") and not any(leftovers.values()):
            return False
        _write_store_install(dest, store_src)
        failed, kept = [], []
        for path, copied in leftovers.items():
            rel = path.relative_to(dest).as_posix()
            if not copied:
                kept.append(rel)
                continue
            try:
                path.unlink()
            except OSError as e:
                failed.append(f"{rel} ({e})")
        self._prune_empty_dirs(dest)
        if kept:
            _log.warning(
                "plugin %s: the old installed copy held files the plugin does not "
                "ship; they are kept in %s but the plugin now runs from the localm "
                "package and does not read them: %s", name, dest, ", ".join(kept))
        if failed:
            _log.warning("plugin %s: could not delete old copied files: %s",
                         name, "; ".join(failed))
            raise RuntimeError(
                f"the old installed copy of {name} could not be fully removed "
                f"({len(failed)} file(s) left; the log names them); the plugin "
                f"runs from the localm package")
        _log.info("plugin %s: the old installed copy was converted; the plugin "
                  "runs from the localm package", name)
        return True

    @staticmethod
    def _is_link(entry) -> bool:
        """True for a symlink or a Windows junction (never followed or removed
        by the conversion)."""
        try:
            return entry.is_symlink() or os.path.isjunction(entry.path)
        except OSError:
            return True

    @classmethod
    def _copied_files(cls, dest: Path, store_src: Path) -> dict:
        """Every entry under *dest* except its manifest and provenance marker,
        mapped to True when it is copied code (a file at a path *store_src* also
        has, or a compiled file) and False when it is kept. A symlink or junction
        is kept and not descended into."""
        out: dict = {}
        stack = [Path(dest)]
        while stack:
            try:
                entries = list(os.scandir(stack.pop()))
            except OSError:
                continue
            for e in entries:
                p = Path(e.path)
                rel = p.relative_to(dest)
                if cls._is_link(e):
                    out[p] = False
                    continue
                try:
                    is_dir = e.is_dir(follow_symlinks=False)
                except OSError:
                    out[p] = False
                    continue
                if is_dir:
                    stack.append(p)
                    continue
                if rel.as_posix() in ("plugin.toml", _PLUGIN_MARKER):
                    continue
                compiled = p.suffix == ".pyc" or "__pycache__" in rel.parts
                out[p] = compiled or (Path(store_src) / rel).is_file()
        return out

    @classmethod
    def _prune_empty_dirs(cls, root: Path) -> None:
        """Remove every empty real directory below *root*, deepest first; a
        symlink or junction is neither descended into nor removed."""
        dirs: list = []
        stack = [Path(root)]
        while stack:
            try:
                entries = list(os.scandir(stack.pop()))
            except OSError:
                continue
            for e in entries:
                try:
                    if cls._is_link(e) or not e.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                dirs.append(Path(e.path))
                stack.append(Path(e.path))
        for d in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass

    def _refresh_installed_builtins(self) -> None:
        """Convert every installed first-party plugin that is still a full copy
        of its store folder (``_migrate_installed_copy``). Best-effort; never
        raises: a failure is recorded as a provisioning error and logged."""
        if not self._store_root:
            return
        for name in sorted(self._installed_set()):
            try:
                self._migrate_installed_copy(name)
                self._clear_provision_error(name, "refresh")
            except Exception as e:
                self._note_provision_error(name, f"refresh: {e}")

    def refresh(self, name: Optional[str] = None) -> list:
        """Convert installed first-party plugins that are still full copies of
        their store folder into marker-only installs
        (``_migrate_installed_copy``). A first-party plugin runs from the store,
        so an upgraded store takes effect without a refresh. With *name*, just
        that plugin (KeyError if it is not an installed first-party plugin).
        Returns the names converted. Without *name*, a plugin whose conversion
        failed is left out of the result and listed in ``last_refresh_failures``
        (name to error); with *name*, the failure raises."""
        self.last_refresh_failures = {}
        self.discover()
        if name is not None:
            if self._store_dir(name) is None:
                raise KeyError(f"no such builtin plugin: {name}")
            if not (self._installed_dir(name) / "plugin.toml").is_file():
                raise KeyError(f"plugin {name!r} is not installed")
            return [name] if self._migrate_installed_copy(name) else []
        refreshed = []
        for n in sorted(self._installed_set()):
            try:
                if self._migrate_installed_copy(n):
                    refreshed.append(n)
            except Exception as e:
                _log.warning("plugin %s: refresh failed: %s", n, e)
                self._errors[n] = f"refresh: {e}"
                self.last_refresh_failures[n] = str(e)
        return refreshed

    def surface_of(self, name: str) -> Optional[Surface]:
        """The GUI surface of plugin *name*: the one its loaded host holds
        (``register_tab``, else the manifest's), else the installed manifest's,
        or None for an unknown name."""
        entry = self._loaded.get(name)
        if entry is not None:
            return getattr(entry[2], "surface", None)
        spec = self._specs.get(name)
        return spec.surface if spec is not None else None

    def _is_protected(self, name: str) -> bool:
        """True when *name* cannot be disabled or uninstalled. An installed
        plugin is protected only through its own spec, which honours
        ``protected`` for a store install alone; a name with no installed spec
        falls back to the catalog."""
        from localm.plugins import catalog as _cat
        spec = self._specs.get(name)
        if spec is not None:
            return bool(spec.protected)
        return name in _cat.protected()

    def _reserved_name_problem(self, name: str) -> Optional[str]:
        """Why a THIRD-PARTY plugin may not be installed under *name*, or None:
        a built-in command verb, a first-party catalog name, or the name of a
        plugin in the bundled store."""
        from localm.plugins import catalog as _cat
        from localm.plugins.loader import _RESERVED_NAMES
        if name in _RESERVED_NAMES:
            return f"plugin name {name!r} clashes with a built-in command"
        if _cat.get(name) is not None or self._store_dir(name) is not None:
            return (f"plugin name {name!r} is reserved for a first-party plugin; "
                    f"a third-party plugin must use another name")
        return None

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
        self.discover()
        claimed = {sp.scope: n for n, sp in self._specs.items() if n != spec0.name}
        reason = _scope_problem(spec0, claimed)
        if reason:
            raise ValueError(reason)

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
        # A third-party plugin must not take a built-in command name or a
        # first-party plugin's name. Only arbitrary-source third-party installs
        # are gated; first-party plugins install via install() from the store.
        reserved = self._reserved_name_problem(name)
        if reserved:
            raise ValueError(reserved)
        self._reject_scope_collision(spec0)
        # name came out of the MANIFEST; parse_spec has already rejected any
        # name that is not a single identifier-shaped component, and
        # _installed_dir re-checks it. This feeds _remove_installed_dir ->
        # shutil.rmtree just below, and copytree after.
        dest = self._installed_dir(name)
        if dest.exists() and (dest / "plugin.toml").is_file() and not force:
            raise ValueError(f"plugin {name!r} is already installed")
        dest.parent.mkdir(parents=True, exist_ok=True)
        # The copy is built beside dest and renamed into place, so an install
        # that dies mid-copy leaves a hidden staging folder, never a
        # half-filled plugin folder, and a failed forced reinstall leaves the
        # existing plugin in place.
        tmp = dest.parent / f".{name}.install.tmp"
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        try:
            # symlinks=True keeps a link that appeared after _reject_source_links
            # walked the tree from being DEREFERENCED. It does not cover a Windows
            # junction, which copytree demotes and recurses into.
            shutil.copytree(src, tmp, symlinks=True)
            _write_marker(tmp, "external", _dir_content_hash(src))
            if dest.exists() and not self._remove_installed_dir(name):
                raise ValueError(
                    f"plugin {name!r}: the existing installed directory could not "
                    f"be removed; see the log")
            tmp.rename(dest)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
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
        """Install a plugin from the bundled store into the installed folder
        (``_provision_from_store``), then load + enable it on the live app. Rolls
        back the install if it does not load. KeyError if no such plugin exists."""
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
        self._record_event("plugin_installed", name)

    def _record_event(self, kind: str, name: str) -> None:
        """Append a plugin install or uninstall to the management log; never raises."""
        from localm import mgmt_events
        from localm.config import home_dir
        spec = self._specs.get(name)
        version = getattr(spec, "version", None)
        field = "version_to" if kind == "plugin_installed" else "version_from"
        try:
            mgmt_events.record(kind, home=home_dir(), plugin=name, **{field: version})
        except Exception as e:
            _log.warning("could not record the %s event for plugin %s: %s", kind, name, e)

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
        self._record_event("plugin_installed", name)
        return spec0

    def set_installed_from_dir(self, source: Path, *, force: bool = False,
                               enable: bool = True):
        """CLI/headless install of a THIRD-PARTY plugin from an arbitrary
        directory WITHOUT mounting routes (the app-free sibling of
        ``install_external``, mirroring ``set_installed_state``): validate the
        manifest, copy it into the installed folder, and enable it. Rolls back a
        copy that does not parse. A running server loads it the next time
        its plugin list is read. Returns the parsed PluginSpec."""
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
        try:
            self._migrate_installed_copy(name)        # convert an old copied install
        except RuntimeError as e:
            # The plugin runs from the store whatever the old copy still holds.
            self._note_provision_error(name, f"refresh: {e}")
        self.discover()
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

    def is_installed_or_on_disk(self, name: str) -> bool:
        """True iff is_installed(name) or an installed directory named
        *name* exists on disk without a manifest. The same union
        uninstall() computes internally as its own "existed" check."""
        return self.is_installed(name) or self._installed_dir_on_disk(name)

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
        """CLI/headless install/uninstall WITHOUT loading routes: install from
        the store (or remove the installed dir); the GUI server reconciles via
        load_enabled on its next start. Installing also enables by default;
        uninstalling disables. Honours protection on uninstall."""
        if on:
            copied = self._provision_from_store(name)  # raises if unknown
            # Roll back ONLY what this call created - see _provision_from_store.
            self._provision_and_verify(name, rollback_on_fail=copied)
            if enable:
                self._set_enabled(name, True)
        else:
            self.discover()
            if name not in self._installed_set() and not self._installed_dir_on_disk(name):
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
        this return value.

        Also removable when the installed directory exists with no manifest
        (an interrupted or corrupted previous install): that directory is
        deleted the same as a fully installed plugin's, and doing so still
        counts as complete. KeyError only when nothing - manifest or
        directory - exists under this name."""
        self.discover()
        spec = self._specs.get(name)
        was_installed = name in self._installed_set()
        on_disk = self._installed_dir_on_disk(name)
        if spec is None and not was_installed and not on_disk:
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
        hook_failed = False
        entry = self._loaded.get(name)
        if entry:
            module = entry[1]
            hook = getattr(module, "on_uninstall", None)
            if callable(hook):
                try:
                    hook(delete_data=delete_data)
                except Exception as e:
                    hook_failed = True
                    _log.warning("plugin %s: on_uninstall(delete_data=%s) failed: %s",
                                 name, delete_data, e, exc_info=True)
        self._unload(name)
        self._set_enabled(name, False)
        data_deleted = not (delete_data and hook_failed)
        if delete_data and spec and spec.data_subdir:
            data_deleted = self._delete_plugin_data(spec) and data_deleted
        removed = self._remove_installed_dir(name)    # delete from the installed folder
        existed = was_installed or on_disk
        if existed and not removed:
            _log.warning(
                "plugin %s: uninstall disabled and unloaded it, but its "
                "installed directory could not be removed; reporting a "
                "degraded result rather than a bare success", name)
        if delete_data and not data_deleted:
            _log.warning(
                "plugin %s: uninstall disabled and unloaded it, but its data "
                "could not be fully deleted; reporting a degraded result "
                "rather than a bare success", name)
        if existed and removed:
            self._record_event("plugin_uninstalled", name)
        return existed and removed and data_deleted

    def _delete_plugin_data(self, spec: PluginSpec) -> bool:
        """Delete the plugin's data_subdir. Returns True iff it is confirmed gone
        afterwards (or there was nothing to delete); False if a security refusal
        (data_subdir escapes the data dir) or a removal failure (locked file, AV
        hold, permission denial) leaves it on disk. The caller (uninstall())
        must fold this into its reported result rather than treating a removed
        installed-dir as the whole story."""
        import shutil
        from localm.config import home_dir
        home = home_dir().resolve()
        target = plugin_data_dir(spec, home)
        if target is None:
            return True
        # Resolve and confine to the plugin's own root (the data dir for a store
        # install, plugin-data/<name> for a third-party one) before any rmtree.
        root = home if spec.builtin else (home / "plugin-data" / spec.name).resolve()
        d = target.resolve()
        if d == home or not root.is_relative_to(home) or not d.is_relative_to(root):
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
            surface = (spec.surface if spec is not None else None) if available \
                else self.surface_of(name)
            return {
                "name": name,
                "version": spec.version if spec else "",
                "description": (spec.description if spec and spec.description
                                else (cat.description if cat else "")),
                "scope": spec.scope if spec else name,
                "builtin": spec.builtin if spec is not None else bool(cat),
                "protected": self._is_protected(name),
                "tab": surface.tab_id if surface else "",
                "label": surface.label if surface else "",
                "icon": surface.icon if surface else "",
                "group": surface.group if surface else "",
                "client_entry": surface.client_entry if surface else "",
                "assets_base": (f"/plugins/{name}"
                                if surface and surface.assets_dir else ""),
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
                "error": (self._errors.get(name) or self._discover_errors.get(name)
                          or self._provision_errors.get(name)
                          or self._store_errors.get(name)
                          or self._unload_errors.get(name)),
            }

        for name, spec in sorted(self._specs.items()):       # installed
            plugins.append(_entry(spec, name, available=False))
        seen = set(self._specs)
        for name in sorted(set(store) | set(_cat.names())):  # available, not installed
            if name in seen or not _cat.offered(name):
                continue
            plugins.append(_entry(store.get(name), name, available=True))
        from localm.config import load_config
        _cfg = load_config()
        return {"plugins": plugins,
                "errors": {**self._store_errors, **self._provision_errors,
                           **self._unload_errors, **self._discover_errors,
                           **self._errors},
                "suggest_plugins": bool(_cfg.get("suggest_plugins", True)),
                "auto_install_plugin_deps": bool(
                    _cfg.get("auto_install_plugin_deps", True))}


# --------------------------------------------------------------------------- #
#  Wire the engine + its management API onto a FastAPI app                    #
# --------------------------------------------------------------------------- #

_backend_manager: Optional[PluginManager] = None
_server_manager_attached = False
# Inner to localm.inference.engine._LOAD_LOCK (taken under it by create_backend's
# lazy plugin load), outer to backend_registry._LOCK; never takes _LOAD_LOCK.
_backend_manager_lock = threading.Lock()


def load_inference_backend_plugins() -> list:
    """For a process without a server app: load the installed, enabled plugins
    that register an inference backend (``PluginManager.load_backend_plugins``),
    once per process. Returns the names loaded by this call; [] on later calls
    and once :func:`attach_engine` has attached a server plugin manager."""
    global _backend_manager
    with _backend_manager_lock:
        if _backend_manager is not None or _server_manager_attached:
            return []
        _backend_manager = PluginManager(app=None)
        return _backend_manager.load_backend_plugins()


def attach_engine(app, inference_engine=None) -> PluginManager:
    """Instantiate a PluginManager bound to *app*, load enabled plugins, and add
    the management endpoints. Returns the manager."""
    from fastapi import Depends, HTTPException
    from fastapi import Request as _Request
    from localm import scopes
    from localm.inference.http_server import require_scope

    global _server_manager_attached
    globals()["Request"] = _Request
    with _backend_manager_lock:
        _server_manager_attached = True
    manager = PluginManager(app, inference_engine=inference_engine)
    manager.load_enabled()
    app.state.plugin_manager = manager

    def _valid_name_or_404(name: str) -> str:
        """*name* when it is a legal plugin id, else HTTPException 404: such a
        ``{name}`` can never match a plugin. Called BEFORE each handler's try
        block, which catches broad ``Exception`` and would otherwise turn this
        into a 400. The engine rejects such an id again at the path join
        (_installed_dir)."""
        if not _is_valid_plugin_name(name):
            raise HTTPException(404, f"No such plugin: {name}")
        return name

    @app.get("/api/plugins", dependencies=[Depends(require_scope(scopes.PLUGINS_READ))])
    async def list_plugins_engine():
        from starlette.concurrency import run_in_threadpool

        from localm.plugins import catalog as _cat
        await manager.load_enabled_pending_off_loop()
        await _cat.check_hardware_off_loop()
        return await run_in_threadpool(manager.api_state)

    @app.post("/api/plugins/{name}/install",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def install_plugin_engine(name: str):
        name = _valid_name_or_404(name)
        try:
            manager.install(name)
        except KeyError as exc:
            raise HTTPException(404, f"No such plugin: {name}") from exc
        except Exception as e:
            raise HTTPException(400, f"Install failed: {e}") from e
        return {"status": "installed", "name": name}

    @app.post("/api/plugins/install-external",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def install_external_plugin_engine(body: dict, request: Request):
        """Install a third-party plugin from a local directory given as
        `source`, which must contain a `plugin.toml` (400 otherwise). Equivalent
        to `localm plugin install <dir>`. The plugin is copied and verified; it
        becomes available the next time the plugin list is read. Third-party
        plugins run unsandboxed in-process, so install only code you trust."""
        from pathlib import Path as _P

        from localm.inference.http_server import effective_fs_access
        from localm.pathsafe import reject_unsafe_path_string
        from localm.plugins.deps_task import host_pip_allowed
        source = (body or {}).get("source", "")
        if not source or not isinstance(source, str):
            raise HTTPException(400, "Missing 'source' (local directory path)")
        try:
            reject_unsafe_path_string(source)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        if effective_fs_access(request) != "host" and not host_pip_allowed(app):
            raise HTTPException(
                403, "Installing a plugin from a server folder needs host "
                     "filesystem access or a server bound to loopback.")
        src = _P(source)
        if not (src.is_dir() and (src / "plugin.toml").is_file()):
            raise HTTPException(400, f"No plugin.toml in {source!r}")
        try:
            spec = manager.set_installed_from_dir(src, force=bool((body or {}).get("force")))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:
            raise HTTPException(400, f"Install failed: {e}") from e
        return {"status": "installed", "name": spec.name, "version": spec.version}

    @app.post("/api/plugins/{name}/uninstall",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def uninstall_plugin_engine(name: str, delete_data: bool = False):
        name = _valid_name_or_404(name)
        # is_installed_or_on_disk(), not is_installed(): a manifest-less
        # directory is something uninstall() below can still act on.
        existed = manager.is_installed_or_on_disk(name)
        try:
            complete = manager.uninstall(name, delete_data=delete_data)
        except KeyError as exc:
            raise HTTPException(404, f"No such plugin: {name}") from exc
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        except Exception as e:
            raise HTTPException(400, f"Uninstall failed: {e}") from e
        if complete:
            return {"status": "uninstalled", "name": name}
        if not existed:
            raise HTTPException(404, f"No such plugin: {name}")
        detail = (
            f"Plugin {name!r} was disabled and unloaded, but it was not "
            f"fully uninstalled: something it owns could not be removed "
            f"from disk; see the server log for the cause.")
        if delete_data:
            detail += (
                " delete_data was requested and this uninstall did not "
                "complete, so do not assume its stored data is gone.")
        raise HTTPException(500, detail)

    @app.post("/api/plugins/refresh",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def refresh_all_plugins_engine():
        refreshed = manager.refresh()
        failed = dict(getattr(manager, "last_refresh_failures", {}) or {})
        return {"status": "partial" if failed else "ok", "refreshed": refreshed,
                "failed": failed}

    @app.post("/api/plugins/{name}/refresh",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def refresh_plugin_engine(name: str):
        name = _valid_name_or_404(name)
        try:
            refreshed = manager.refresh(name)
        except KeyError as exc:
            raise HTTPException(404, f"No such installed builtin plugin: {name}") from exc
        except Exception as e:
            raise HTTPException(500, f"Refresh failed: {e}") from e
        return {"status": "refreshed" if refreshed else "up-to-date", "name": name}

    @app.post("/api/plugins/{name}/enable",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def enable_plugin(name: str):
        name = _valid_name_or_404(name)
        try:
            manager.enable(name)
        except KeyError as exc:
            raise HTTPException(404, f"No such plugin: {name}") from exc
        except ValueError as e:
            raise HTTPException(409, str(e)) from e      # e.g. not installed
        except Exception as e:
            raise HTTPException(400, f"Enable failed: {e}") from e
        return {"status": "enabled", "name": name}

    @app.post("/api/plugins/{name}/disable",
              dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def disable_plugin(name: str):
        name = _valid_name_or_404(name)
        try:
            manager.disable(name)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        return {"status": "disabled", "name": name}

    # Host-side dependency install (pip extras). In its own module so its
    # fastapi Request annotation resolves against module globals; this file
    # uses PEP 563 string annotations and imports fastapi lazily.
    from localm.plugins.deps_routes import register_dep_routes
    register_dep_routes(app, manager)

    # Background-job registry, registered at KERNEL level so a headless
    # ``localm serve`` has one too. Every consumer reads app.state.jobs PER
    # REQUEST, so registration order inside attach_engine does not matter.
    from localm.jobmanager import JobManager
    from localm.plugins.gui.routes import jobs as _job_routes
    app.state.jobs = JobManager()
    _job_routes.register(app, app.state.jobs)

    return manager
