# SPDX-License-Identifier: AGPL-3.0-or-later
"""ComfyUI backend for the image plugin.

Thin wrapper over the shared Comfy HTTP plumbing (``localm.image_gen.comfy``)
that feeds it THIS plugin's per-plugin config (resolved through
``media_config``, honouring the "use config from" share-config selector). The
generic transport stays shared; only the config binding lives here, so a future
non-ComfyUI image backend is just another module selected by ``backend`` name.

Legacy global keys (comfy_launch_cmd / comfy_workdir / comfy_output_dir /
reload_llm_after_imagine) seed the defaults until the user saves per-plugin
values, so existing setups keep working with no migration step - EXCEPT
comfy_launch_cmd/comfy_workdir specifically, which are suppressed while the
managed ComfyUI instance is active (see managed_comfy.legacy_comfy_value):
otherwise a global value left over from before the managed instance existed
would silently defeat its auto-launch routing forever. A genuine per-plugin
override is unaffected and still always wins.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from localm.image_gen import comfy as _comfy
from localm.media.managed_comfy import legacy_comfy_value
from localm.plugins import media_config
from localm.vram import media_estimate_bytes, resolve_swap_policy


def settings(full_config: dict) -> dict:
    """Resolve the image plugin's effective backend settings."""
    block, warning = media_config.resolve_config("image", full_config)
    comfy_blk = block.get("comfy") if isinstance(block.get("comfy"), dict) else {}
    backend_name = block.get("backend", "comfy")
    # We do not hide problems: when the configured backend cannot be loaded the
    # job still falls back to comfy (best-effort), but say so instead of silently
    # pretending the chosen backend is active.
    warning = media_config.combine_warnings(
        warning, media_config.backend_unavailable_warning(__package__, backend_name))
    return {
        "backend": backend_name,
        # sanitize_comfy_url on the RESOLVED value, not just the default_api_url
        # fallback: a per-plugin comfy.api_url (or the global comfy_api_url) would
        # otherwise short-circuit before default_api_url()'s own guard, letting an
        # admin-set link-local/metadata host reach the outbound comfy calls
        # (CHK-COMFY-APIURL residual). Sanitising here is idempotent for the
        # already-guarded default path.
        "api_url": _comfy.sanitize_comfy_url(
            (comfy_blk.get("api_url")
             or full_config.get("comfy_api_url")
             or _comfy.default_api_url()).rstrip("/")),
        # The legacy global fallback is suppressed while the managed instance is
        # active - a genuine per-plugin comfy_blk override still always wins.
        "launch_cmd": comfy_blk.get("launch_cmd")
        or legacy_comfy_value("comfy_launch_cmd", full_config) or "",
        "workdir": comfy_blk.get("workdir")
        or legacy_comfy_value("comfy_workdir", full_config) or "",
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


def _comfy_generate(s: dict, prompt: str, out_path: Path, *,
                    self_url: str, write_sidecar: bool,
                    guidance: Optional[float] = None,
                    negative_prompt: Optional[str] = None,
                    seed: Optional[int] = None,
                    input_image: Optional[Path] = None,
                    denoise: Optional[float] = None,
                    model_overrides: Optional[dict] = None,
                    swap: bool = True,
                    delete_outputs: Optional[bool] = None,
                    cancel_check=None,
                    placement: Optional[dict] = None,
                    on_progress=None) -> tuple[bool, str]:
    if delete_outputs is None:
        delete_outputs = bool(s.get("delete_outputs", False))
    return _comfy.generate_image(
        prompt, out_path,
        api_url=s["api_url"],
        guidance=guidance,
        negative_prompt=negative_prompt,
        seed=seed,
        input_image=input_image,
        denoise=denoise,
        model_overrides=model_overrides,
        localm_url=self_url,
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
