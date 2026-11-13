# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-plugin config resolution for the media plugins (image, music, video).

Only manipulates the config dict: each plugin's backend reads the resolved block
and fills in its own defaults. The one backend name it knows is ``comfy``, the
default backend, which backend resolution falls back to.

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
the sharer follows. The ``backend`` choice and the ``native`` model settings are
never shared: each plugin keeps its own. The sharer's own block is never mutated, so toggling sharing
off restores its own values untouched. Sharing is cycle-prevented (no image<-video
while video<-image) and falls back to the sharer's own block (with a warning) when
the source is missing/disabled. Applies only to the three media plugins.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

MEDIA_PLUGINS = ("image", "music", "video")


def _plugins(cfg: dict) -> dict:
    p = cfg.get("plugins")
    return p if isinstance(p, dict) else {}


def _own_block(name: str, cfg: dict) -> dict:
    # Deep copy so callers can read or mutate the returned block, including
    # nested sub-blocks like {"comfy": {...}}, without touching the stored config.
    blk = _plugins(cfg).get(name)
    return copy.deepcopy(blk) if isinstance(blk, dict) else {}


def _source_of(name: str, cfg: dict) -> Optional[str]:
    src = _own_block(name, cfg).get("use_config_from")
    if isinstance(src, str) and src in MEDIA_PLUGINS and src != name:
        return src
    return None


def _installed_plugins() -> tuple[set, Optional[OSError]]:
    """``(names, error)``: the plugins physically present in the installed
    folder, and the OSError that stopped the scan of that folder, or None. A
    failed scan is logged at debug level; *names* then holds what was found
    before it failed."""
    from pathlib import Path

    from localm.debuglog import logger
    from localm.plugins.loader import plugins_dir
    installed: set = set()
    try:
        for child in Path(plugins_dir()).glob("*"):
            if (not child.name.startswith(".") and child.is_dir()
                    and (child / "plugin.toml").is_file()):
                installed.add(child.name)
    except OSError as exc:
        logger.debug("active_plugins: failed to scan plugins dir %s: %s",
                     plugins_dir(), exc)
        return installed, exc
    return installed, None


def active_plugins(cfg: dict) -> set:
    """The set of ACTIVE plugin names: installed (physically present in the
    installed folder) AND enabled (config). "Installed" is disk presence in the
    store to installed model, NOT a config flag, so this scans the installed
    folder. A scan that fails returns what it found before failing."""
    installed, _error = _installed_plugins()
    return installed & set(cfg.get("plugins_enabled", []) or [])


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


def resolve_config(name: str, cfg: dict,
                   *, active: Optional[set] = None) -> tuple[dict, Optional[str]]:
    """Effective stored block for media plugin *name*.

    Applies ``use_config_from`` (one hop, using the source's OWN block) when the
    source is active and non-cyclic; otherwise returns *name*'s own block.
    *active* is the set of active plugin names (installed AND enabled); when None
    it is derived from disk + config via ``active_plugins`` (the source plugin
    must be installed on disk and enabled, per the store->installed model).
    Returns ``(block, warning_or_None)``. When the plugins folder could not be
    read to tell whether the source is active, the warning says so instead of
    calling the source inactive. The returned block is a deep copy, so the
    stored config is never mutated even if the caller writes to nested blocks.
    """
    own = _own_block(name, cfg)
    src = own.get("use_config_from")
    if not (isinstance(src, str) and src in MEDIA_PLUGINS and src != name):
        return own, None

    derived = active is None
    if active is None:
        active = active_plugins(cfg)
    if src not in _plugins(cfg) or src not in active:
        scan_error = (_installed_plugins()[1]
                      if derived and src in _plugins(cfg) else None)
        if scan_error is not None:
            return own, (f"'{name}' is set to use '{src}' config, but the plugins "
                         f"folder could not be read to check whether '{src}' is "
                         f"active ({scan_error}); using its own settings.")
        return own, (f"'{name}' is set to use '{src}' config, but '{src}' is not "
                     f"active; using its own settings.")
    if would_cycle(name, src, cfg):
        return own, (f"'{name}' is set to use '{src}' config, but that forms a "
                     f"cycle; using '{name}' own settings.")

    resolved = _own_block(src, cfg)
    resolved["use_config_from"] = src      # keep the marker so the UI can show it
    for key in ("backend", "native"):
        if key in own:
            resolved[key] = own[key]
        else:
            resolved.pop(key, None)
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
    """Import the media backend module *name* from ``<package>.backends``.

    e.g. ``load_backend("localm.plugins.builtin.image", "a1111")`` imports
    ``localm.plugins.builtin.image.backends.a1111``. Raises ``ModuleNotFoundError``
    when no such module exists, so the caller falls back to its built-in ``comfy``
    reference rather than hard-crashing on a typo or an uninstalled backend.

    The module declares what it supports as upper-case attributes
    (``CAPABILITIES``); ``backend_capabilities`` reads them."""
    import importlib
    nm = (name or "comfy").strip().lower() or "comfy"
    return importlib.import_module(f"{package}.backends.{nm}")


def backend_unavailable_warning(package: str, name: Optional[str]) -> Optional[str]:
    """A warning when *name* is a media backend that is not the built-in
    ``comfy`` reference and cannot be imported (a typo, an unimplemented
    backend, or one whose dependency is not installed), else None for
    ``comfy``, empty or loadable.

    The caller still falls back to its inline ``comfy`` reference - a typo must
    not hard-crash a generate - but a missing backend is never passed off as the
    active one: the user is told their configured backend was ignored. A plugin
    folds this into its ``settings()['warning']``."""
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
    """Join the non-empty *warnings* with ``'; '``, or None when all are empty.
    Lets a caller merge several independent config notes into one ``warning``
    without one silently dropping another."""
    parts = [w for w in warnings if w]
    return "; ".join(parts) if parts else None


CAPABILITIES = (
    "NEEDS_COMFY_SERVER",
    "SUPPORTS_STEPS",
    "SUPPORTS_CLIP_SKIP",
    "SUPPORTS_JOB_OPTIONS",
    "SUPPORTS_UPSCALE",
    "EMBEDS_OUTPUT_METADATA",
)
"""The capability attributes a backend module may declare. A backend sets the
ones it has to True; an undeclared attribute is False.

- ``NEEDS_COMFY_SERVER``: the backend drives a ComfyUI server (``api_url``,
  GPU placement and the ComfyUI-only settings apply).
- ``SUPPORTS_STEPS`` / ``SUPPORTS_CLIP_SKIP``: the request's ``steps`` /
  ``clip_skip`` reach the generator.
- ``SUPPORTS_JOB_OPTIONS``: per-job model choices, several LoRAs, masks,
  ControlNet, embeddings, sampler, scheduler, hires fix, batch and preview
  frames, with the backend's own request refusals and missing-model offers.
- ``SUPPORTS_UPSCALE``: the backend upscales gallery images.
- ``EMBEDS_OUTPUT_METADATA``: the generated file carries the submitted workflow
  as container metadata, so the facade strips it from the output."""


def backend_capabilities(package: str, name: Optional[str],
                         comfy_ref: Optional[SimpleNamespace] = None) -> dict:
    """``{capability: bool}`` for the media backend *name* of *package*: the
    module ``load_backend`` imports, or *comfy_ref* for ``comfy``, empty or a
    backend that cannot be imported."""
    nm = (name or "comfy").strip().lower() or "comfy"
    impl = comfy_ref
    if nm != "comfy" or comfy_ref is None:
        try:
            impl = load_backend(package, nm)
        except ModuleNotFoundError:
            if comfy_ref is None:
                raise
    return {cap: bool(getattr(impl, cap, False)) for cap in CAPABILITIES}


def make_backend_facade(package: str, comfy_ref: SimpleNamespace,
                        output_kind: str) -> SimpleNamespace:
    """Build the ``ensure_available``/``free_vram``/``generate`` dispatch facade
    shared by every media plugin's backend.py: *comfy_ref* (the plugin's own
    inline ComfyUI reference implementation) for the default ``"comfy"`` backend,
    else ``<package>.backends.<name>`` loaded by ``load_backend`` - falling back
    to *comfy_ref* on an unknown or missing name, so a typo never hard-crashes a
    generate (the settings ``warning`` carries the config note, via
    ``backend_unavailable_warning``).

    A plugin's backend.py assigns the returned callables to its own module-level
    names, including ``resolve`` as ``_impl``, so both ``plug.py`` call sites
    (``_backend.ensure_available``, etc.) and tests that introspect which
    implementation a settings dict resolves to keep working.

    ``supports(s, capability)`` answers whether the backend *s* resolves to
    declares *capability* (one of ``CAPABILITIES``; any other name raises
    KeyError).

    ``generate`` takes the extra keywords ``privacy`` and ``withhold_on_trace``
    (both default False), which are not passed to the backend. It passes the
    backend ``delete_outputs`` as True when ``privacy`` is set and as the
    settings' ``delete_outputs`` otherwise; a caller that passes
    ``delete_outputs`` itself gets a TypeError. The generation is STRICT when
    ``privacy`` or ``withhold_on_trace`` is set (``withhold_on_trace``: the
    file is handed to a caller as it is, for example in an HTTP response).
    A strict generation passes a backend that declares ``NEEDS_COMFY_SERVER``
    ``containment_required=True``, so a ComfyUI copy or history entry it could
    not remove fails the generation.
    When the backend declares
    ``EMBEDS_OUTPUT_METADATA`` and the generation succeeded and left a non-empty
    file at *out_path*, ``generate`` strips
    the metadata from *out_path* as *output_kind* (``image``, ``audio`` or
    ``video``) once, here. A failed strip appends the WARNING to the message;
    a strict generation instead returns ``(False, "Output withheld: <reason>")``
    and deletes the file. A strict generation that fails for any other reason,
    with a backend that declares ``EMBEDS_OUTPUT_METADATA``, deletes *out_path*
    when this generation created or changed it; a file it left as it was is
    kept. The message says when a file could not be removed, so a file that may
    still carry the prompt is never left behind or reported as generated."""

    def resolve(s: dict):
        name = (s.get("backend") or "comfy").strip().lower()
        if name in ("", "comfy"):
            return comfy_ref
        try:
            return load_backend(package, name)
        except ModuleNotFoundError:
            return comfy_ref

    def supports(s: dict, capability: str) -> bool:
        if capability not in CAPABILITIES:
            raise KeyError(capability)
        return bool(getattr(resolve(s), capability, False))

    def ensure_available(s: dict, *args, **kwargs) -> tuple[bool, str]:
        return resolve(s).ensure_available(s, *args, **kwargs)

    def free_vram(s: dict, *args, **kwargs) -> bool:
        return resolve(s).free_vram(s, *args, **kwargs)

    def generate(s: dict, prompt: str, out_path, *args, privacy: bool = False,
                 withhold_on_trace: bool = False, **kwargs) -> tuple[bool, str]:
        if "delete_outputs" in kwargs or "containment_required" in kwargs:
            raise TypeError("generate() sets delete_outputs and containment_required "
                            "from the settings, privacy and withhold_on_trace; do not "
                            "pass them")
        strict = bool(privacy or withhold_on_trace)
        impl = resolve(s)
        embeds = bool(getattr(impl, "EMBEDS_OUTPUT_METADATA", False))
        if strict and getattr(impl, "NEEDS_COMFY_SERVER", False):
            kwargs["containment_required"] = True
        before = _file_state(out_path) if strict and embeds else None
        ok, message = impl.generate(
            s, prompt, out_path, *args,
            delete_outputs=bool(privacy or s.get("delete_outputs")), **kwargs)
        if not ok:
            if strict and embeds and _file_state(out_path) not in (None, before):
                message += _remove_withheld(out_path)
            return False, message
        if (not embeds
                or not Path(out_path).exists() or Path(out_path).stat().st_size == 0):
            return ok, message
        from localm.media.output_metadata import strip_output
        result = strip_output(output_kind, Path(out_path))
        if not result.failed:
            return ok, message
        if strict:
            return False, f"Output withheld: {result.reason}{_remove_withheld(out_path)}"
        return ok, f"{message}\n{result.reason}"

    return SimpleNamespace(resolve=resolve, supports=supports,
                           ensure_available=ensure_available,
                           free_vram=free_vram, generate=generate)


def _file_state(path):
    """``(size, mtime_ns, inode)`` of the file at *path*, or None when there is
    none or it cannot be read."""
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns, st.st_ino)


def _remove_withheld(out_path) -> str:
    """Delete *out_path* when it exists. Returns "" when it is gone, or a
    sentence telling the user to delete it by hand when it could not be
    removed. Never raises."""
    try:
        Path(out_path).unlink()
    except FileNotFoundError:
        return ""   # See test_withheld_file_already_gone_is_not_reported.
    except OSError as e:
        return f" The file could not be removed ({e}); delete {out_path} by hand."
    return ""


def media_session_is_private() -> bool:
    """True when generated media must leave no traces: the server's session
    mode is privacy or cannot be resolved, or the work runs inside an
    ``audit.request_privacy`` block (a caller whose own session mode is
    privacy, carried into a job thread by ``audit.carry_request_privacy``).
    Never raises."""
    from localm.audit import automatic_write_allowed, request_is_privacy
    return request_is_privacy() or not automatic_write_allowed("server")


#: The generation arguments with a fixed range, by argument name:
#: ``(lowest, highest, refusal text)``. ``denoise`` is the image-to-image
#: strength, ``duration_seconds`` a music track's length, ``seconds`` a video
#: clip's length.
REQUEST_RANGES = {
    "denoise": (0.0, 1.0, "Denoise must be between 0 and 1"),
    "duration_seconds": (1.0, 3600.0, "Duration must be between 1 and 3600 seconds"),
    "seconds": (1.0, 20.0, "Duration must be between 1 and 20 seconds"),
}


def range_refusal(**kwargs) -> Optional[str]:
    """The refusal text of the first argument in *kwargs* named in
    :data:`REQUEST_RANGES` whose value is not a finite number inside its range,
    or None when every such argument is None or in range. A number written as
    text is read as that number; a bool or any other value is out of range.
    Other arguments are ignored. Never raises."""
    import math
    for name, (low, high, text) in REQUEST_RANGES.items():
        value = kwargs.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return text
        try:
            number = float(value)
        except ValueError:
            return text
        if not math.isfinite(number) or not low <= number <= high:
            return text
    return None


def generate_media(backend, s: dict, prompt: str, out_path: Path, *,
                   on_progress=None, before_generate=None,
                   **kwargs) -> tuple[bool, str, Path]:
    """Run one generation through a media plugin's backend facade, for any
    caller (REPL, CLI, coder tool, MCP tool): the plugin's configuration notes,
    the backend's request refusal, ``ensure_available``, *before_generate*
    (e.g. unloading a chat model), ``generate`` with *kwargs* (the metadata
    strip runs inside it), and a ``free_vram`` on every exit after
    ``ensure_available``.

    *backend* is the plugin's ``backend`` module and *s* its resolved settings
    (``prepare_for_job``). Returns ``(ok, message, path)`` where *path* is
    *out_path* with the backend's ``OUTPUT_SUFFIX`` when it declares one.
    *on_progress* receives every status line and is handed to the backend as
    given (not passed when None). An argument outside its
    :data:`REQUEST_RANGES` range (:func:`range_refusal`), a backend that cannot
    honour the request, or ``steps`` / ``clip_skip`` given without the matching
    capability, returns ``(False, <reason>, path)`` before anything is loaded."""

    def say(text: str) -> None:
        if on_progress is not None:
            on_progress(text)

    kwargs.setdefault("self_url", None)
    for note in (s.get("warning"), s.get("backend_note")):
        if note:
            say(note)
    impl = backend._impl(s)
    path = out_path.with_suffix(getattr(impl, "OUTPUT_SUFFIX", out_path.suffix))
    out_of_range = range_refusal(**kwargs)
    if out_of_range:
        return False, out_of_range, path
    unsupported = [name for name, capability in (("steps", "SUPPORTS_STEPS"),
                                                 ("clip_skip", "SUPPORTS_CLIP_SKIP"))
                   if kwargs.get(name) is not None
                   and not backend.supports(s, capability)]
    if unsupported:
        return False, (f"The {s.get('backend')} backend does not take "
                       f"{' or '.join(unsupported)}; it uses the values in its "
                       f"workflow."), path
    refusal = getattr(impl, "refusal", None)
    refused = refusal(s=s, **kwargs) if refusal is not None else None
    if refused:
        return False, refused, path
    progress = {} if on_progress is None else {"on_progress": on_progress}
    ok, message = backend.ensure_available(s, **progress)
    if not ok:
        return False, message, path
    say(message)
    try:
        if before_generate is not None:
            before_generate()
        ok, message = backend.generate(s, prompt, path, **progress, **kwargs)
        return ok, message, path
    finally:
        backend.free_vram(s)
