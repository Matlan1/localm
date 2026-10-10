# SPDX-License-Identifier: AGPL-3.0-or-later
"""Resolving a media plugin's ``backend`` setting (``auto``, ``native``,
``comfy``) to the backend that runs.

``auto`` keeps every machine that has a ComfyUI set up on ComfyUI and uses the
native backend everywhere else. An explicit choice is returned unchanged.
"""

from __future__ import annotations

import os
from typing import Optional

from localm.media.managed_comfy import managed_comfy_active

CHOICES = ("auto", "native", "comfy")

AUTO_PROBE_TIMEOUT = 1.5


def comfy_is_set_up(full_config: dict, comfy_blk: dict, api_url: str) -> bool:
    """Whether this machine has a ComfyUI a media plugin is meant to use: the
    managed instance installed and selected, ``comfy_target`` set to the user's
    own ComfyUI, ``FLUX_API_URL`` set, any ComfyUI address/launcher/folder
    configured (globally or in *comfy_blk*), or a ComfyUI that answered at
    *api_url* recently. Reads config and the readiness cache only; no network
    call."""
    if managed_comfy_active(full_config) or full_config.get("comfy_target") == "user":
        return True
    if os.environ.get("FLUX_API_URL"):
        return True
    if any(comfy_blk.get(k) for k in ("api_url", "launch_cmd", "workdir")):
        return True
    if any(full_config.get(k) for k in ("comfy_api_url", "comfy_launch_cmd", "comfy_workdir")):
        return True
    from localm.media.comfy_client import is_comfy_confirmed
    try:
        return is_comfy_confirmed(api_url)
    except Exception:
        return False


def resolve(choice: str, full_config: dict, comfy_blk: dict, api_url: str,
            kind: str) -> tuple[str, Optional[str]]:
    """``(backend, note)`` for the configured *choice*. ``auto`` becomes
    ``comfy`` when ComfyUI is set up and ``native`` otherwise, with a note that
    names *kind* ("Image", "Video") and why; any other choice is returned
    unchanged with no note."""
    if choice != "auto":
        return choice, None
    if comfy_is_set_up(full_config, comfy_blk, api_url):
        return "comfy", f"{kind} backend: ComfyUI (auto: ComfyUI is set up)."
    return "native", (f"{kind} backend: native stable-diffusion.cpp (auto: no ComfyUI is "
                      "set up).")


def refine_auto(s: dict, kind: str) -> dict:
    """In a job's own thread: when ``auto`` chose native, switch *s* to ComfyUI
    if one answers at ``s["api_url"]`` right now (one request, at most
    ``AUTO_PROBE_TIMEOUT`` seconds). Returns *s*, updated in place."""
    if s.get("backend_choice") == "auto" and s.get("backend") == "native":
        from localm.media.comfy_client import _comfy_alive
        if _comfy_alive(s["api_url"], timeout=AUTO_PROBE_TIMEOUT):
            s["backend"] = "comfy"
            s["backend_note"] = (f"{kind} backend: ComfyUI (auto: ComfyUI answered at "
                                 f"{s['api_url']}).")
    return s
