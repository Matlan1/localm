# SPDX-License-Identifier: AGPL-3.0-or-later
"""VRAM-aware model-swap policy for media generation.

Before a media gen (image/music/video via ComfyUI), the chat LLM is unloaded to
hand VRAM to the media model, then reloaded. On a large card both models fit at
once, so the swap is pure latency (and a failed/forgotten reload leaves the card
empty). This module decides whether the swap is actually needed; the unload
endpoint separately verifies VRAM was reclaimed before the media model loads
(the driver-hang guard - see inference/http_server.py).
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

_VALID_POLICIES = ("auto", "always", "never")
# A free-VRAM rise of at least this much after unload counts as "the model was
# actually freed" (guards against a tiny transient fluctuation reading as freed).
_MIN_RELEASE_RISE = int(256e6)  # 256 MB
# Safety margin added on top of the media estimate before we call a fit "safe".
_DEFAULT_HEADROOM = int(1.0e9)  # 1 GB


def should_swap_for_media(
    free_bytes: Optional[int],
    media_estimate_bytes: Optional[int],
    *,
    headroom_bytes: int = _DEFAULT_HEADROOM,
    policy: str = "auto",
) -> bool:
    """Return True to unload the chat LLM before a media gen, False to keep it.

    policy:
      'always' - always swap (the historical behaviour).
      'never'  - never swap; keep chat loaded (media may spill to RAM / OOM -
                 the user's explicit choice, e.g. a big workstation card).
      'auto'   - keep the chat model loaded only when the media model
                 demonstrably fits alongside it:
                 free_bytes >= media_estimate_bytes + headroom_bytes.
                 When free VRAM or the media estimate is unmeasurable, 'auto'
                 swaps (the safe default - same as today).
    """
    if policy == "always":
        return True
    if policy == "never":
        return False
    # auto
    if free_bytes is None or media_estimate_bytes is None:
        return True
    return free_bytes < media_estimate_bytes + headroom_bytes


def resolve_swap_policy(plugin_block: dict, full_config: dict) -> str:
    """Resolve the effective media swap policy: a per-plugin ``model_swap_policy``
    wins, then a global one, else the default ``auto``. An unknown policy string
    falls back to ``auto``.

    The legacy ``reload_llm_after_imagine`` / per-plugin ``reload_llm_after_generate``
    boolean is a SEPARATE axis: it controls whether the chat model is *reloaded
    after* a generation (eager vs lazy), surfaced as the backend ``reload_after``
    setting, and deliberately does NOT influence the swap (unload-before)
    decision. Keeping the two axes independent means an existing ``false`` config
    keeps its lazy-reload behaviour while still getting the safe, VRAM-aware
    ``auto`` swap default - instead of silently never freeing VRAM (an OOM risk on
    a small card) the way mapping ``false -> never`` would.
    """
    block = plugin_block or {}
    explicit = block.get("model_swap_policy",
                         (full_config or {}).get("model_swap_policy"))
    if isinstance(explicit, str) and explicit.lower() in _VALID_POLICIES:
        return explicit.lower()
    return "auto"


# Conservative per-backend VRAM estimates (GB) for the media model, used by the
# 'auto' swap decision when the user has not set plugins.<name>.vram_estimate_gb.
# Deliberately generous so 'auto' errs toward swapping on a small card; a large
# card still keeps chat hot when free VRAM clearly exceeds the estimate + headroom.
# Self-calibrating measurement from the real ComfyUI model files is a Phase 1.5
# follow-up; a fixed estimate keeps the Phase-1 decision deterministic + testable.
DEFAULT_MEDIA_VRAM_GB = {"image": 14.0, "music": 12.0, "video": 16.0}


def media_estimate_bytes(name: str, plugin_block: Optional[dict] = None) -> int:
    """Estimated peak VRAM (bytes) for media plugin *name*'s model. Honours a
    per-plugin override ``vram_estimate_gb``; otherwise a conservative per-backend
    default (see DEFAULT_MEDIA_VRAM_GB)."""
    gb = (plugin_block or {}).get("vram_estimate_gb")
    if not isinstance(gb, (int, float)) or gb <= 0:
        gb = DEFAULT_MEDIA_VRAM_GB.get(name, 14.0)
    return int(gb * 1024 ** 3)


def decide_media_swap(settings: dict, *,
                      read_free: Optional[Callable[[], Optional[int]]] = None) -> bool:
    """Decide whether to unload the chat LLM before a media gen, reading live free
    VRAM. *settings* is a resolved media-plugin settings dict carrying
    ``swap_policy`` and ``vram_estimate_bytes`` (filled by the media backends).
    Pass *read_free* to inject the free-VRAM reading in tests."""
    if read_free is None:
        def read_free() -> Optional[int]:
            from localm.discover import vram_info
            return vram_info().get("free")
    return should_swap_for_media(
        read_free(), settings.get("vram_estimate_bytes"),
        policy=settings.get("swap_policy", "auto"))


def wait_for_vram_release(
    read_free: Callable[[], Optional[int]],
    *,
    before_bytes: Optional[int],
    min_rise_bytes: int = _MIN_RELEASE_RISE,
    timeout_s: float = 5.0,
    poll_s: float = 0.1,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> tuple[Optional[bool], Optional[int]]:
    """Poll ``read_free()`` until free VRAM has risen by at least ``min_rise_bytes``
    versus ``before_bytes`` (the native unload's deferred free has completed), or
    ``timeout_s`` elapses.

    Returns ``(released, final_free)``: ``True`` once the rise is observed,
    ``False`` on timeout, and ``None`` (released) when ``before_bytes`` is None
    (VRAM unmeasurable - cannot verify, so behave as before and do not block).

    The ``/v1/models/unload`` endpoint uses this so it does NOT return until VRAM
    is actually reclaimed; otherwise a media model can load on top of a
    not-yet-freed LLM and exceed total VRAM, hanging the GPU driver (TDR).
    ``read_free``/``sleep``/``monotonic`` are injectable for tests.
    """
    final = read_free()
    if before_bytes is None:
        return (None, final)
    deadline = monotonic() + max(0.0, timeout_s)
    while True:
        if final is not None and final - before_bytes >= min_rise_bytes:
            return (True, final)
        if monotonic() >= deadline:
            return (False, final)
        sleep(poll_s)
        final = read_free()


# --------------------------------------------------------------------------- #
#  Chat<->media VRAM handoff (shared by the image/music/video plugins)         #
# --------------------------------------------------------------------------- #
#
# The three media plugins swap the chat model out before generating and back in
# after, each via the same self-authenticated HTTP round trip to this server's
# own /v1/models/unload and /v1/models/load. Only the progress-message wording
# (which backend is "the image/music/video backend") differs between them, so it
# lives here once instead of copy-pasted per plugin.

def unload_chat_for_media(job: Any, self_url: str, media_label: str) -> bool:
    """Unload the chat model BEFORE a media (image/music/video) model loads, so
    it gets the VRAM.

    Uses the same bearer-token + TLS handling as the reload path: the
    ``/v1/models/unload`` endpoint needs the models-write scope, so an
    unauthenticated call is rejected and the chat model stays resident - the
    media model then loads on top of it and hangs the GPU driver. Logs the
    outcome (and the VRAM freed) on *job* so a failure is visible instead of
    silent. Returns True when the server confirmed the chat model is unloaded.
    *media_label* names the caller ("image"/"music"/"video") for the messages."""
    job.push({"type": "line", "text": "Freeing VRAM: unloading the chat model..."})
    try:
        import requests as _rq
        headers = {}
        key = os.environ.get("LOCALM_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        from localm import tls as _tls
        resp = _rq.post(f"{self_url}/models/unload", headers=headers, timeout=300,
                        verify=_tls.requests_verify(self_url))
        if not resp.ok:
            job.push({"type": "line", "text":
                      f"Could not unload the chat model (HTTP {resp.status_code}) - "
                      f"the {media_label} backend may run low on VRAM."})
            return False
        data = {}
        try:
            data = resp.json()
        except Exception:
            # resp.ok already confirmed the server accepted the unload, so a body
            # that does not parse is non-fatal: fall through with empty data to
            # the generic "Chat model unloaded." message below.
            pass
        if data.get("status") == "already_unloaded":
            job.push({"type": "line", "text":
                      "No chat model was loaded - VRAM already free."})
            return True
        before, after = data.get("vram_before_bytes"), data.get("vram_after_bytes")
        if data.get("vram_freed") and before is not None and after is not None:
            gb = max(0.0, (after - before) / 1024 ** 3)
            job.push({"type": "line", "text":
                      f"Chat model unloaded - freed {gb:.1f} GB of VRAM."})
        elif data.get("vram_freed") is False:
            job.push({"type": "line", "text":
                      "Chat model unloaded, but VRAM has not dropped yet - continuing."})
        else:
            job.push({"type": "line", "text": "Chat model unloaded."})
        return True
    except Exception as e:
        job.push({"type": "line", "text":
                  f"Could not unload the chat model ({e}) - "
                  f"the {media_label} backend may run low on VRAM."})
        return False


def reload_chat_after_media(job: Any, self_url: str, s: dict, backend: Any,
                            media_label: str) -> None:
    """Hand VRAM back: ask *backend* (the plugin's own backend module, exposing
    ``free_vram(s)``) to drop its models, then reload the chat model so the next
    reply is instant. Skipped when reload-after-generate is off."""
    if not s["reload_after"]:
        job.push({"type": "line", "text":
                  f"Keeping the {media_label} backend loaded (reload is off) - "
                  "the chat model reloads on the next message."})
        return
    if not backend.free_vram(s):
        job.push({"type": "line", "text":
                  f"The {media_label} backend kept its models in VRAM - the chat "
                  "model will reload on the next message instead."})
        return
    job.push({"type": "line", "text": "Reloading the chat model..."})
    try:
        import requests as _rq
        headers = {}
        key = os.environ.get("LOCALM_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        from localm import tls as _tls
        _rq.post(f"{self_url}/models/load", headers=headers, timeout=300,
                 verify=_tls.requests_verify(self_url))
        job.push({"type": "line", "text": "Chat model ready."})
    except Exception as e:
        job.push({"type": "line", "text": f"Reload deferred to the next message ({e})."})
