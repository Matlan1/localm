# SPDX-License-Identifier: AGPL-3.0-or-later
"""ComfyUI backend for the image plugin.

Thin wrapper over the shared Comfy HTTP plumbing (``localm.image_gen.comfy``)
that feeds it THIS plugin's per-plugin config (resolved through
``media_config``, honouring the "use config from" share-config selector). The
generic transport stays shared; only the config binding lives here, so a future
non-ComfyUI image backend is just another module selected by ``backend`` name.

Legacy global keys (comfy_launch_cmd / comfy_workdir / comfy_output_dir /
reload_llm_after_imagine) seed the defaults until the user saves per-plugin
values. api_url / launch_cmd / workdir specifically - both the legacy global
key AND this plugin's own per-plugin comfy_blk override - are suppressed
entirely while the managed ComfyUI instance is active (comfy_target == "own"
and installed; see managed_comfy.managed_comfy_active). Only comfy_target ==
"user" lets any of these three fields win.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from localm.image_gen import comfy as _comfy
from localm.media.managed_comfy import legacy_comfy_value, managed_comfy_active
from localm.plugins import media_config
from localm.vram import media_estimate_bytes, resolve_swap_policy


def settings(full_config: dict) -> dict:
    """Resolve the image plugin's effective backend settings."""
    block, warning = media_config.resolve_config("image", full_config)
    comfy_blk = block.get("comfy") if isinstance(block.get("comfy"), dict) else {}
    backend_name = block.get("backend", "comfy")
    # When the configured backend cannot be loaded the job still falls back to
    # comfy (best-effort), and says so rather than reporting the chosen backend
    # as active.
    warning = media_config.combine_warnings(
        warning, media_config.backend_unavailable_warning(__package__, backend_name))
    # When the managed ComfyUI instance is selected ("own"), NEITHER the
    # per-plugin comfy.* fields NOR the legacy global
    # comfy_api_url/comfy_launch_cmd/comfy_workdir are honoured here - any of
    # them defeats ensure_comfy()'s managed-routing branch, which only engages
    # when the caller passes NOTHING. Only comfy_target == "user" lets any of
    # these win. _comfy.default_api_url() itself resolves to the managed
    # instance's URL whenever own_active is True, so it is always the correct
    # fallback.
    own_active = managed_comfy_active(full_config)
    api_url = ("" if own_active else comfy_blk.get("api_url")) \
        or (None if own_active else full_config.get("comfy_api_url")) \
        or _comfy.default_api_url()
    launch_cmd = "" if own_active else (
        comfy_blk.get("launch_cmd")
        or legacy_comfy_value("comfy_launch_cmd", full_config) or "")
    workdir = "" if own_active else (
        comfy_blk.get("workdir")
        or legacy_comfy_value("comfy_workdir", full_config) or "")
    return {
        "backend": backend_name,
        # sanitize_comfy_url on the RESOLVED value, not just the
        # default_api_url fallback: a per-plugin comfy.api_url (or the global
        # comfy_api_url) would otherwise short-circuit before
        # default_api_url()'s own guard, letting an admin-set
        # link-local/metadata host reach the outbound comfy calls. Sanitising
        # here is idempotent for the already-guarded default path.
        "api_url": _comfy.sanitize_comfy_url(api_url.rstrip("/")),
        "launch_cmd": launch_cmd,
        "workdir": workdir,
        "output_dir": comfy_blk.get("output_dir")
        or full_config.get("comfy_output_dir", "") or "",
        "reload_after": bool(block.get(
            "reload_llm_after_generate",
            full_config.get("reload_llm_after_imagine", True))),
        "fast_dequant": bool(comfy_blk.get(
            "fast_dequant", full_config.get("comfy_fast_dequant", True))),
        "delete_outputs": bool(comfy_blk.get(
            "delete_outputs", full_config.get("comfy_delete_outputs", False))),
        "swap_policy": resolve_swap_policy(block, full_config),
        "vram_estimate_bytes": media_estimate_bytes("image", block),
        "warning": warning,
    }


# --- ComfyUI reference implementation (the default "comfy" backend, I1) ------

def _comfy_ensure_available(s: dict, on_progress=None) -> tuple[bool, str]:
    return _comfy.ensure_comfy(
        s["api_url"], on_progress=on_progress,
        launch_cmd=s["launch_cmd"] or None, workdir=s["workdir"] or None)


def _comfy_free_vram(s: dict) -> bool:
    return _comfy.free_comfy_vram(s["api_url"])


def _comfy_model_slots(s: dict) -> Optional[list]:
    """Every model-file slot in the ACTIVE image workflow, resolved against the
    currently-reachable ComfyUI. None when ComfyUI is not reachable (the caller
    shows a clear message instead of a silently-empty picker)."""
    import json
    try:
        workflow = json.loads(_comfy.workflow_path().read_text(encoding="utf-8"))
    except Exception:
        return None
    return _comfy.workflow_model_slots(workflow, s["api_url"])


def _comfy_model_roles(s: dict, roles: list) -> dict:
    """This plugin's model-picker payload: the live ComfyUI slots joined to the
    localm registry's ``model_type`` slice and to the roles the plugin declared
    through ``host.register_model_role``.

    The backend owns this seam rather than the route: a non-ComfyUI backend for
    this media type implements this one function, leaving the route, the GUI and
    the role contract unchanged.

    One ComfyUI round trip and one registry read per call - both blocking, so the
    caller runs it off the event loop, as it does for ``_comfy_model_slots``."""
    from localm.plugins import media_roles
    return media_roles.resolve_model_roles(_comfy_model_slots(s), roles)


def _comfy_lora_options(s: dict) -> Optional[list]:
    """LoRA filenames the live ComfyUI currently has installed (``LoraLoader``'s
    ``lora_name`` combo from ``/object_info``). Independent of
    ``_comfy_model_slots`` / ``workflow_model_slots``, which only walks nodes
    already PRESENT in the active workflow JSON - a LoraLoader node is not one
    of them, since the image plugin injects it fresh at generation time only
    when a LoRA is actually requested (see comfy.py's ``_build_image_workflow``).
    None when ComfyUI is not reachable, matching ``_comfy_model_slots``'s
    reachability contract."""
    info = _comfy.comfy_object_info(s["api_url"])
    if info is None:
        return None
    return _comfy._combo_options(info.get("LoraLoader", {}), "lora_name") or []


def _comfy_generate(s: dict, prompt: str, out_path: Path, *,
                    self_url: str, write_sidecar: bool,
                    instance_token: Optional[str] = None,
                    guidance: Optional[float] = None,
                    cfg: Optional[float] = None,
                    negative_prompt: Optional[str] = None,
                    seed: Optional[int] = None,
                    input_image: Optional[Path] = None,
                    denoise: Optional[float] = None,
                    model_overrides: Optional[dict] = None,
                    lora_name: Optional[str] = None,
                    lora_strength_model: Optional[float] = None,
                    lora_strength_clip: Optional[float] = None,
                    swap: bool = True,
                    delete_outputs: Optional[bool] = None,
                    cancel_check=None,
                    placement: Optional[dict] = None,
                    on_progress=None) -> tuple[bool, str]:
    if delete_outputs is None:
        delete_outputs = bool(s.get("delete_outputs", False))
    # Only forward the strength kwargs when the caller actually set them, so an
    # unset (None) strength keeps generate_image()'s own defaults (1.0 / 0.5)
    # instead of this facade silently overriding them with None.
    lora_kwargs = {}
    if lora_name:
        lora_kwargs["lora_name"] = lora_name
        if lora_strength_model is not None:
            lora_kwargs["lora_strength_model"] = lora_strength_model
        if lora_strength_clip is not None:
            lora_kwargs["lora_strength_clip"] = lora_strength_clip
    return _comfy.generate_image(
        prompt, out_path,
        api_url=s["api_url"],
        guidance=guidance,
        cfg=cfg,
        negative_prompt=negative_prompt,
        seed=seed,
        input_image=input_image,
        denoise=denoise,
        model_overrides=model_overrides,
        localm_url=self_url,
        instance_token=instance_token,
        write_sidecar=write_sidecar,
        launch_cmd=s["launch_cmd"] or None,
        workdir=s["workdir"] or None,
        comfy_output_dir=s["output_dir"] or None,
        swap=swap,
        fast_dequant=s.get("fast_dequant", True),
        delete_outputs=delete_outputs,
        cancel_check=cancel_check,
        placement=placement,
        on_progress=on_progress,
        **lora_kwargs,
    )


_COMFY_REF = SimpleNamespace(
    ensure_available=_comfy_ensure_available,
    free_vram=_comfy_free_vram,
    generate=_comfy_generate,
)


# --- backend facade: dispatch to the configured backend (the I1 seam) --------
# Shared with the music/video plugins - see media_config.make_backend_facade.

_facade = media_config.make_backend_facade(__package__, _COMFY_REF)
_impl = _facade.resolve
ensure_available = _facade.ensure_available
free_vram = _facade.free_vram
generate = _facade.generate
