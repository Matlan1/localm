# SPDX-License-Identifier: AGPL-3.0-or-later
"""ComfyUI (ACE-Step) backend for the music plugin.

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
identically to a deliberate override and used to silently defeat managed
routing (NEW-COMFY-TARGET-OWN-DEFEATED-BY-STALE-PERPLUGIN-FIELD). Only
comfy_target == "user" lets any of these three fields win - "own" means own.

Per-plugin output containment (FAC-3): the shared ``generate_music`` has no
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
    """Resolve the music plugin's effective backend settings."""
    block, warning = media_config.resolve_config("music", full_config)
    comfy_blk = block.get("comfy") if isinstance(block.get("comfy"), dict) else {}
    backend_name = block.get("backend", "comfy")
    # We do not hide problems: when the configured backend cannot be loaded the
    # job still falls back to comfy (best-effort), but say so instead of silently
    # pretending the chosen backend is active.
    warning = media_config.combine_warnings(
        warning, media_config.backend_unavailable_warning(__package__, backend_name))
    # NEW-COMFY-TARGET-OWN-DEFEATED-BY-STALE-PERPLUGIN-FIELD: when the managed
    # ComfyUI instance is selected ("own"), neither the per-plugin comfy.*
    # fields nor the legacy global launch_cmd/workdir keys may be honoured -
    # any of them defeats ensure_comfy()'s managed-routing branch, which only
    # engages when the caller passes NOTHING. _comfy.default_api_url() itself
    # resolves to the managed instance's URL whenever own_active is True.
    own_active = managed_comfy_active(full_config)
    api_url = ("" if own_active else comfy_blk.get("api_url")) \
        or _comfy.default_api_url()
    # Sanitise the RESOLVED url: a per-plugin comfy.api_url short-circuits
    # before default_api_url()'s guard, so an admin-set link-local/metadata
    # host would otherwise reach the outbound comfy calls (CHK-COMFY-APIURL).
    # The _checked variant also surfaces a guard fallback through this
    # settings() warning channel instead of only the debug log (rule 5).
    api_url, url_warning = _comfy.sanitize_comfy_url_checked(api_url.rstrip("/"))
    warning = media_config.combine_warnings(warning, url_warning)
    launch_cmd = "" if own_active else (
        comfy_blk.get("launch_cmd")
        or legacy_comfy_value("comfy_launch_cmd", full_config) or "")
    workdir = "" if own_active else (
        comfy_blk.get("workdir")
        or legacy_comfy_value("comfy_workdir", full_config) or "")
    return {
        "backend": backend_name,
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

    The backend owns this seam (rather than the route) because resolving which
    models exist IS backend work: a future non-ComfyUI backend for this media
    type implements this one function and the route, the GUI and the role
    contract are unchanged.

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
    # through the env var its containment step resolves from (FAC-3).
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
            swap=s["swap_policy"] != "never",
            cancel_check=cancel_check,
            delete_outputs=delete_outputs,
            **kwargs,
        )


_COMFY_REF = SimpleNamespace(
    ensure_available=_comfy_ensure_available,
    free_vram=_comfy_free_vram,
    generate=_comfy_generate,
)


# --- backend facade: dispatch to the configured backend (the I1 seam) --------
# Shared with the image/video plugins - see media_config.make_backend_facade.

_facade = media_config.make_backend_facade(__package__, _COMFY_REF)
_impl = _facade.resolve
ensure_available = _facade.ensure_available
free_vram = _facade.free_vram
generate = _facade.generate
