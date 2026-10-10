# SPDX-License-Identifier: AGPL-3.0-or-later
"""Backend selection and the ComfyUI (ACE-Step) backend for the music plugin.

``settings`` resolves the ``backend`` setting (``auto``, ``native`` or
``comfy``) through ``localm.media.backend_choice``. The native backend lives in
``backends/native.py``; the ComfyUI one is inline below.

Mirrors the image plugin's backend: a thin wrapper over the shared Comfy HTTP
plumbing fed with THIS plugin's per-plugin config (resolved through
``media_config``, honouring the "use config from" share-config selector). The
generic transport stays shared; only the config binding lives here, so a future
non-ComfyUI music backend is just another module selected by ``backend`` name.

Legacy global keys (comfy_launch_cmd / comfy_workdir / comfy_output_dir /
reload_llm_after_imagine) seed the defaults until the user saves per-plugin
values, so existing setups keep working with no migration step. api_url /
launch_cmd / workdir specifically - both the legacy global key AND this
plugin's own per-plugin comfy_blk override - are suppressed entirely while
the managed ComfyUI instance is active (comfy_target == "own" and installed;
see managed_comfy.managed_comfy_active). A per-plugin value set before the
user ever touched comfy_target, or before switching it back to "own", reads
identically to an intentional override, so it is suppressed as well. Only
comfy_target == "user" lets any of these three fields win - "own" means own.

Per-plugin output containment: the shared ``generate_music`` has no
``comfy_output_dir`` parameter, so the only way to feed it this plugin's own
output dir is the ``COMFY_OUTPUT_DIR`` env var that ``comfy._comfy_output_root``
resolves from. The backend therefore publishes the per-plugin value on that env
var for the duration of the generation (restoring whatever was there before), so
ComfyUI's on-disk copy AND any uploaded source actually get deleted rather than
the knob being silently ignored.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from localm import music_gen as _music_gen
from localm.image_gen import comfy as _comfy
from localm.media import backend_choice
from localm.media.managed_comfy import legacy_comfy_value, managed_comfy_active
from localm.plugins import media_config
from localm.vram import media_estimate_bytes, resolve_swap_policy


@contextlib.contextmanager
def _comfy_output_dir_env(output_dir: Optional[str]):
    """Publish the per-plugin ComfyUI output dir on ``COMFY_OUTPUT_DIR`` for the
    duration of the block, restoring the prior value afterwards.

    A no-op when no per-plugin output dir is configured, so a value inherited
    from the environment or global config keeps working untouched."""
    if not output_dir:
        yield
        return
    prev = os.environ.get("COMFY_OUTPUT_DIR")
    os.environ["COMFY_OUTPUT_DIR"] = output_dir
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("COMFY_OUTPUT_DIR", None)
        else:
            os.environ["COMFY_OUTPUT_DIR"] = prev


def settings(full_config: dict) -> dict:
    """Resolve the music plugin's effective backend settings.

    ``backend`` is the backend that will run (``auto`` already resolved from
    config alone; ``prepare_for_job`` re-checks it); ``backend_choice`` is what
    is configured and ``backend_note`` says how ``auto`` resolved, or None.
    ``native`` is the native backend's own block, with ``native_runtime``
    (``auto``, ``cuda``, ``vulkan``, ``cpu`` or ``metal``), ``plan`` (use the
    ACE-Step planner) and ``lowvram`` read out of it."""
    block, warning = media_config.resolve_config("music", full_config)
    comfy_blk = block.get("comfy") if isinstance(block.get("comfy"), dict) else {}
    native_blk = block.get("native") if isinstance(block.get("native"), dict) else {}
    backend_choice_value = str(block.get("backend") or "auto").strip().lower() or "auto"
    # When the configured backend cannot be loaded the job still falls back to
    # comfy (best-effort), and the warning says so instead of reporting the
    # chosen backend as active.
    warning = media_config.combine_warnings(
        warning, media_config.backend_unavailable_warning(
            str(__package__),
            "comfy" if backend_choice_value == "auto" else backend_choice_value))
    # When the managed ComfyUI instance is selected ("own"), neither the
    # per-plugin comfy.* fields nor the legacy global launch_cmd/workdir keys
    # may be honoured - any of them defeats ensure_comfy()'s managed-routing
    # branch, which only engages when the caller passes NOTHING.
    # _comfy.default_api_url() itself resolves to the managed instance's URL
    # whenever own_active is True.
    own_active = managed_comfy_active(full_config)
    api_url = ("" if own_active else comfy_blk.get("api_url")) \
        or _comfy.default_api_url()
    # Sanitise the RESOLVED url: a per-plugin comfy.api_url short-circuits
    # before default_api_url()'s guard, so an admin-set link-local/metadata
    # host would otherwise reach the outbound comfy calls. The _checked variant
    # also surfaces a guard fallback through this settings() warning channel
    # instead of only the debug log.
    api_url, url_warning = _comfy.sanitize_comfy_url_checked(api_url.rstrip("/"))
    warning = media_config.combine_warnings(warning, url_warning)
    launch_cmd = "" if own_active else (
        comfy_blk.get("launch_cmd")
        or legacy_comfy_value("comfy_launch_cmd", full_config) or "")
    workdir = "" if own_active else (
        comfy_blk.get("workdir")
        or legacy_comfy_value("comfy_workdir", full_config) or "")
    backend_name, backend_note = backend_choice.resolve(
        backend_choice_value, full_config, comfy_blk or {}, api_url, "Music",
        native_name="ACE-Step")
    return {
        "backend": backend_name,
        "backend_choice": backend_choice_value,
        "backend_note": backend_note,
        "native": native_blk,
        "native_runtime": str(native_blk.get("runtime") or "auto").strip().lower() or "auto",
        "plan": native_blk.get("plan", True) is not False,
        "lowvram": native_blk.get("lowvram", False) is True,
        "api_url": api_url,
        "launch_cmd": launch_cmd,
        "workdir": workdir,
        "output_dir": comfy_blk.get("output_dir")
        or full_config.get("comfy_output_dir", "") or "",
        "reload_after": bool(block.get(
            "reload_llm_after_generate",
            full_config.get("reload_llm_after_imagine", True))),
        # Opt-in output containment: per-plugin "delete_outputs" wins, else the
        # global comfy_delete_outputs (default False = keep ComfyUI's own copy).
        # Privacy mode forces deletion separately, in plug.py.
        "delete_outputs": bool(comfy_blk.get(
            "delete_outputs", full_config.get("comfy_delete_outputs", False))),
        "float_type": comfy_blk.get("float_type")
        or full_config.get("comfy_float_type"),
        "swap_policy": resolve_swap_policy(block, full_config),
        "vram_estimate_bytes": media_estimate_bytes("music", block),
        "warning": warning,
    }


def prepare_for_job(s: dict, full_config: dict) -> dict:
    """Finish resolving *s* (from ``settings``) for one generation, in the job's
    own thread: ``auto`` re-checked against a live ComfyUI
    (``backend_choice.refine_auto``), and a native job's
    ``vram_estimate_bytes`` taken from its model files unless the plugin sets
    ``vram_estimate_gb``. Returns *s*, updated in place."""
    backend_choice.refine_auto(s, "Music")
    if s.get("backend") == "native":
        block, _w = media_config.resolve_config("music", full_config)
        if not isinstance(block.get("vram_estimate_gb"), (int, float)):
            from .backends import native
            s["vram_estimate_bytes"] = native.vram_estimate_bytes(s)
    return s


def generate_unless_comfy(tags: str, out_path: Path, *, on_progress=None,
                          before_generate=None, config: Optional[dict] = None,
                          **kwargs) -> Optional[tuple[bool, str, Path]]:
    """Generate through the configured music backend for a caller that has its
    own ComfyUI path (the chat REPL, ``localm music``).

    Returns None when the backend is ComfyUI, so the caller runs its own
    ComfyUI code unchanged; otherwise ``(ok, message, path)``, where *path* is
    *out_path* with the backend's file suffix. Runs ``ensure_available``, then
    *before_generate* (e.g. unloading a chat model), then ``generate`` with
    *kwargs*, and always releases the backend's VRAM afterwards. *on_progress*
    receives every status line."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    s = prepare_for_job(settings(cfg), cfg)
    if is_comfy(s):
        return None

    def say(text: str) -> None:
        if on_progress is not None:
            on_progress(text)

    for note in (s.get("warning"), s.get("backend_note")):
        if note:
            say(note)
    impl = _impl(s)
    path = out_path.with_suffix(getattr(impl, "OUTPUT_SUFFIX", out_path.suffix))
    refusal = getattr(impl, "refusal", None)
    refused = refusal(**kwargs) if refusal is not None else None
    if refused:
        return False, refused, path
    ok, message = ensure_available(s, on_progress=say)
    if not ok:
        return False, message, path
    say(message)
    if before_generate is not None:
        before_generate()
    try:
        ok, message = generate(s, tags, path, on_progress=say, **kwargs)
        return ok, message, path
    finally:
        free_vram(s)


# --- ComfyUI (ACE-Step) reference implementation (default "comfy" backend) ---

def _comfy_ensure_available(s: dict, on_progress=None) -> tuple[bool, str]:
    return _comfy.ensure_comfy(
        s["api_url"], on_progress=on_progress,
        launch_cmd=s["launch_cmd"] or None, workdir=s["workdir"] or None)


def _comfy_free_vram(s: dict) -> bool:
    return _comfy.free_comfy_vram(s["api_url"])


def _comfy_model_slots(s: dict) -> Optional[list]:
    """Every model-file slot in the ACTIVE music workflow, resolved against the
    currently-reachable ComfyUI. None when ComfyUI is not reachable (the caller
    shows a clear message instead of a silently-empty picker)."""
    import json
    try:
        workflow = json.loads(_music_gen.comfy.workflow_path().read_text(encoding="utf-8"))
    except Exception:
        return None
    return _comfy.workflow_model_slots(workflow, s["api_url"])


def _comfy_model_roles(s: dict, roles: list) -> dict:
    """This plugin's model-picker payload: the live ComfyUI slots joined to the
    localm registry's ``model_type`` slice and to the roles the plugin declared
    through ``host.register_model_role``.

    This seam lives in the backend rather than the route: a future non-ComfyUI
    backend for this media type implements this one function, and the route, the
    GUI and the role contract are unchanged.

    One ComfyUI round trip and one registry read per call - both blocking, so the
    caller runs it off the event loop exactly as it already does for
    ``_comfy_model_slots``."""
    from localm.plugins import media_roles
    return media_roles.resolve_model_roles(_comfy_model_slots(s), roles)


def _comfy_generate(s: dict, tags: str, out_path: Path, *,
                    self_url: str, write_sidecar: bool, on_progress=None,
                    lyrics: Optional[str] = None,
                    duration_seconds: float = 120.0,
                    swap: bool = True,
                    cancel_check=None,
                    delete_outputs: Optional[bool] = None,
                    **kwargs) -> tuple[bool, str]:
    # delete_outputs lets the caller (plug.py) fold privacy mode into the
    # configured value; default to the resolved per-plugin/global setting when
    # the caller does not pass one, so a direct caller still honours the config.
    if delete_outputs is None:
        delete_outputs = s.get("delete_outputs", False)
    # generate_music has no comfy_output_dir param; feed the per-plugin value
    # through the env var its containment step resolves from.
    with _comfy_output_dir_env(s.get("output_dir") or None):
        return _music_gen.generate_music(
            tags, out_path,
            lyrics=lyrics,
            duration_seconds=duration_seconds,
            api_url=s["api_url"],
            localm_url=self_url,
            on_progress=on_progress,
            write_sidecar=write_sidecar,
            launch_cmd=s["launch_cmd"] or None,
            workdir=s["workdir"] or None,
            float_type=s.get("float_type"),
            swap=swap,
            cancel_check=cancel_check,
            delete_outputs=delete_outputs,
            **kwargs,
        )


_COMFY_REF = SimpleNamespace(
    ensure_available=_comfy_ensure_available,
    free_vram=_comfy_free_vram,
    generate=_comfy_generate,
)


# --- backend facade: dispatch to the configured backend ----------------------
# Shared with the image/video plugins - see media_config.make_backend_facade.

_facade = media_config.make_backend_facade(__package__, _COMFY_REF)
_impl = _facade.resolve


def is_comfy(s: dict) -> bool:
    """Whether *s* (from ``settings``) runs the ComfyUI backend, including a
    backend name that cannot be loaded, which falls back to ComfyUI."""
    return _impl(s) is _COMFY_REF
ensure_available = _facade.ensure_available
free_vram = _facade.free_vram
generate = _facade.generate
