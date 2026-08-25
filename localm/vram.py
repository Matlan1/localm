# SPDX-License-Identifier: AGPL-3.0-or-later
"""VRAM-aware model-swap policy for media generation."""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

_VALID_POLICIES = ("auto", "always", "never")

# Single source of truth for the fixed VRAM headroom a GGUF load reserves beyond
# model weights: KV cache + compute buffers. The GGUF backend (its _check_vram /
# _auto_ctx_max / _auto_gpu_layers preflights), the GUI VRAM estimate
# (sysstats.estimate_vram) and the discover fit badge (discover.fit_label) all
# reason about "does it fit" and MUST use the same number, or a badge and the
# loader can disagree on the same model. They keep their own module/class-level
# name (so existing monkeypatch targets like GgufBackend._VRAM_OVERHEAD_BYTES
# stay patchable) but derive its value from here.
VRAM_OVERHEAD_BYTES = int(1.5e9)
# Weights rarely load at exactly their on-disk size; a small safety factor the
# fit badge applies (discover.fit_label). Kept here alongside the overhead so the
# two "fit math" knobs live in one place.
VRAM_WEIGHT_FACTOR = 1.10

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
    """Return True to unload the chat LLM before a media gen, False to keep it."""
    if policy == "always":
        return True
    if policy == "never":
        return False
    # auto
    if free_bytes is None or media_estimate_bytes is None:
        return True
    return free_bytes < media_estimate_bytes + headroom_bytes


def resolve_swap_policy(plugin_block: dict, full_config: dict) -> str:
    """Resolve the effective media swap policy: a per-plugin ``model_swap_policy`` wins, then a global one, else the default ``auto``."""
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
    """Estimated peak VRAM (bytes) for media plugin *name*'s model."""
    gb = (plugin_block or {}).get("vram_estimate_gb")
    if not isinstance(gb, (int, float)) or gb <= 0:
        gb = DEFAULT_MEDIA_VRAM_GB.get(name, 14.0)
    return int(gb * 1024 ** 3)


def decide_media_swap(settings: dict, *,
                      read_free: Optional[Callable[[], Optional[int]]] = None) -> bool:
    """Decide whether to unload the chat LLM before a media gen, reading live free VRAM. *settings* is a resolved media-plugin settings dict carrying ``swap_policy`` and ``vram_estimate_bytes`` (filled by the media backends)."""
    if read_free is None:
        def read_free() -> Optional[int]:
            from localm.discover import vram_capacity
            return vram_capacity().get("free")
    return should_swap_for_media(
        read_free(), settings.get("vram_estimate_bytes"),
        policy=settings.get("swap_policy", "auto"))


def media_single_device_shortfall(settings: dict, *,
                                  config: Optional[dict] = None) -> Optional[dict]:
    """``{'index', 'needed', 'free'}`` when the ONE card ComfyUI will actually use cannot hold the WHOLE media model, else ``None`` (fine, or nothing to check)."""
    if settings.get("swap_policy", "auto") != "auto":
        return None
    need = settings.get("vram_estimate_bytes")
    if not isinstance(need, int) or need <= 0:
        return None
    from localm.config import load_config
    from localm.discover import (GPU_PROBE_OK, applied_split_device_count,
                                 list_gpus, resolve_preferred_device)
    cfg = config if config is not None else load_config()
    # applied_split_device_count (loader truth), NOT split_device_count: whether a
    # split is ACTIVE for chat/embeddings is a load-time fact, and on vulkan the
    # detected count would wrongly report < 2 for a live split (GPU-SPLIT-VKINDEX).
    # On a torch-blind box resolve_preferred_device() below still returns None, so
    # this stays a no-op there; on a mixed box it measures torch space, which is
    # where media actually runs.
    if applied_split_device_count(cfg) < 2:
        return None
    gpus, status = list_gpus(return_status=True)
    if status != GPU_PROBE_OK:
        return None
    idx = resolve_preferred_device(cfg, gpus=gpus)
    if idx is None:
        return None
    dev = {g.get("index"): g for g in gpus}.get(idx)
    if dev is None or dev.get("free") is None:
        return None
    needed = need + _DEFAULT_HEADROOM
    free = dev["free"]
    if free >= needed:
        return None
    return {"index": idx, "needed": needed, "free": free}


def _placement_targets(placement: Optional[dict]) -> list:
    """The distinct non-empty ``gpu:N`` targets in a placement plan (``[]`` when the plan is None or every component is left on its default card)."""
    if not isinstance(placement, dict):
        return []
    seen = []
    for key in ("model", "clip", "vae"):
        t = placement.get(key)
        if t and t not in seen:
            seen.append(t)
    return seen


def media_split_notice(config: Optional[dict] = None, *,
                       placement: Optional[dict] = None,
                       capability=None) -> Optional[str]:
    """A user-visible line about how media generation uses the box's GPUs, or ``None`` when there is nothing worth saying."""
    from localm.config import load_config
    from localm.discover import applied_split_device_count, resolve_preferred_device
    cfg = config if config is not None else load_config()
    targets = _placement_targets(placement)
    if targets:
        where = " and ".join(targets)
        return (f"Note: image/music/video generation is placing the text encoder and VAE "
                f"on {where} to free the main card for the model. A single model still "
                "cannot be divided across cards, but its components can be spread. Chat "
                "and embeddings use the full split.")

    # applied_split_device_count (loader truth), NOT split_device_count: the user
    # CONFIGURED this split and it IS active for chat/embeddings, so the notice must
    # fire even on the vulkan build, where the detected count wrongly collapses to
    # < 2 and would suppress the notice entirely (GPU-SPLIT-VKINDEX) - the exact
    # user-visible Q2 miss #704 closes.
    n = applied_split_device_count(cfg)
    if n < 2:
        return None

    if capability is not None and not getattr(capability, "available", True):
        reason = getattr(capability, "reason", "") or "per-component placement is unavailable"
        return (f"Note: your GPU split spans {n} cards, but image/music/video generation "
                f"is running on a single card because {reason}. A single model cannot be "
                "divided across cards the way chat and embeddings are. Chat and embeddings "
                "still use the full split.")

    idx = resolve_preferred_device(cfg)
    where = f"GPU {idx}" if idx is not None else "a single GPU"
    return (f"Note: your GPU split spans {n} cards, but image/music/video generation "
            f"currently runs on ONE card ({where}, the one with the most free VRAM). "
            "A single model cannot be divided across cards the way chat and embeddings "
            "are, so your split ratios do not apply here. Chat and embeddings still use "
            "the full split.")


def decide_embedder_swap(embedder_estimate_bytes: Optional[int], *,
                         policy: str = "auto",
                         read_free: Optional[Callable[[], Optional[int]]] = None) -> bool:
    """Decide whether to unload the chat LLM before the shared embedder loads (``localm.inference.embedder.get_embedder()``), reading live free VRAM."""
    if read_free is None:
        def read_free() -> Optional[int]:
            from localm.discover import vram_capacity
            return vram_capacity().get("free")
    return should_swap_for_media(read_free(), embedder_estimate_bytes, policy=policy)


def evict_chat_for_embedder(*, timeout_s: float = 300.0) -> str:
    """Free every loaded chat engine so the shared embedder's own load has VRAM room, through the SAME guarded path the server's 'Unload all' button uses (``http_server.unload_all_models``)."""
    import asyncio
    from localm.debuglog import logger
    from localm.inference import http_server as _hs

    loop = getattr(_hs, "_server_loop", None)
    if loop is None or not loop.is_running():
        logger.debug("embedder: cannot reach the server loop to evict the chat "
                     "model safely; loading the embedder alongside it (may be "
                     "tight on VRAM)")
        return "skipped"
    # BUG #648: if this call is ALREADY running on the server loop's own thread
    # (an async route resolved get_embedder() synchronously without offloading),
    # the run_coroutine_threadsafe(...).result() below would DEADLOCK - the loop
    # thread would block waiting for a coroutine only it can run, freezing the
    # whole server for the full timeout, then swallowing the TimeoutError. Detect
    # that and skip the guarded eviction: the embedder loads alongside the resident
    # chat model (degraded, logged - never a freeze, and never silent per rule 5).
    # Callers should offload to an executor so the eviction can actually run (the
    # memory routes now do; this is the belt-and-braces guard for any other caller).
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        logger.warning("embedder: get_embedder() ran on the server event loop; "
                       "skipping the guarded chat-model eviction to avoid "
                       "deadlocking the loop (loading the embedder alongside the "
                       "resident chat model - may be tight on VRAM). This call "
                       "should be offloaded to an executor.")
        return "skipped"
    try:
        fut = asyncio.run_coroutine_threadsafe(_hs.unload_all_models(), loop)
        res = fut.result(timeout=timeout_s)
    except Exception as e:
        logger.debug("embedder: guarded chat-model eviction did not complete "
                     "(%s); loading the embedder without evicting", e)
        return "error"
    return res.get("status", "unloaded") if isinstance(res, dict) else "unloaded"


def _vram_free_reading() -> tuple:
    """``(free_bytes, fresh, scope)``: a free-VRAM reading, whether the probe behind it was fresh (list_gpus() completed inside its deadline) rather than a timed-out/busy fallback to a stale last-known-good value, and the reading's ``free_scope``."""
    from localm.discover import GPU_PROBE_OK, vram_capacity
    try:
        result = vram_capacity(return_status=True)
    except TypeError:
        result = vram_capacity()
    if isinstance(result, tuple) and len(result) == 2:
        info, probe_status = result
        fresh = probe_status == GPU_PROBE_OK
    else:
        info, fresh = result, True
    return info.get("free"), fresh, info.get("free_scope")


def _live_free_vram_bytes():
    """Free VRAM in bytes from a FRESH probe, or None when the reading is not live - the reader wait_for_vram_release() polls for the 'after' end."""
    free, fresh, _scope = _vram_free_reading()
    return free if fresh else None


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
    """Poll ``read_free()`` until free VRAM has risen by at least ``min_rise_bytes`` versus ``before_bytes`` (the native unload's deferred free has completed), or ``timeout_s`` elapses."""
    final = read_free()
    if before_bytes is None:
        return (None, final)
    deadline = monotonic() + max(0.0, timeout_s)
    while True:
        if final is not None and final - before_bytes >= min_rise_bytes:
            return (True, final)
        if monotonic() >= deadline:
            # Poll to the deadline first (a slow probe may recover and still show
            # the rise); only the reading we actually END on decides. An
            # unmeasurable one proves nothing either way.
            return ((False, final) if final is not None else (None, final))
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

def unload_chat_for_media(job: Any, self_url: str, media_label: str,
                          instance_token: Optional[str] = None) -> bool:
    """Unload the chat model BEFORE a media (image/music/video) model loads, so it gets the VRAM."""
    job.push({"type": "line", "text": "Freeing VRAM: unloading the chat model..."})
    try:
        from localm.selfclient import self_request
        resp = self_request("POST", "/models/unload", timeout=300, base_url=self_url,
                            instance_token=instance_token)
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
        if data.get("status") == "in_use":
            # unload_all_models() never races the in-flight-request pin
            # (AUDIT-CRIT-1): a chat engine mid-generation is reported "in_use"
            # and left resident rather than freed out from under that request,
            # exactly like unload_one_model()'s own pinned-engine check. Without
            # this branch that status fell through to the generic "Chat model
            # unloaded." below - a false success (rule 5) that told the media
            # backend VRAM was free when the chat model was still fully
            # resident, which is the exact driver-hang hazard this module
            # exists to prevent. Treat it like the resp.ok/exception failures
            # above: report it honestly and let the caller fall back to its own
            # conservative swap handling.
            job.push({"type": "line", "text":
                      "Chat model is busy (still generating a reply) and could "
                      f"not be unloaded - the {media_label} backend may run low "
                      "on VRAM."})
            return False
        skipped = [str(m) for m in (data.get("skipped_in_use") or [])]
        if skipped:
            # "unloaded" means SOMETHING was freed (an idle sibling engine, the
            # embedding model), not that the chat model was: released_anything
            # wins the status in unload_all_models(), so a pinned chat engine
            # still lands in skipped_in_use under status "unloaded". For this
            # caller only the chat model matters - falling through to "Chat
            # model unloaded." here was the same rule-5 false success the
            # "in_use" branch above closes for the nothing-freed shape, just
            # hidden behind a freed bystander. Report what actually happened;
            # the caller falls back to its conservative swap handling, exactly
            # as for "in_use".
            job.push({"type": "line", "text":
                      "Freed what could be freed, but still in use and NOT "
                      f"unloaded: {', '.join(skipped)} - the {media_label} "
                      "backend may run low on VRAM."})
            return False
        before, after = data.get("vram_before_bytes"), data.get("vram_after_bytes")
        uncertain = bool(data.get("vram_reading_uncertain"))
        if uncertain:
            # The server flagged its own reading as possibly stale, so neither a
            # GB figure nor "has not dropped yet" is a claim we may pass on to the
            # user as fact - both would be arithmetic on a number the server just
            # said it could not stand behind. This branch is checked BEFORE them
            # for exactly that reason (rule 5).
            job.push({"type": "line", "text":
                      "Chat model unloaded - could not confirm how much VRAM was "
                      "freed - continuing."})
        elif data.get("vram_freed") and before is not None and after is not None:
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
                            media_label: str,
                            instance_token: Optional[str] = None) -> None:
    """Hand VRAM back: ask *backend* (the plugin's own backend module, exposing ``free_vram(s)``) to drop its models, then reload the chat model so the next reply is instant."""
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
        from localm.selfclient import self_request
        resp = self_request("POST", "/models/load", timeout=300, base_url=self_url,
                            instance_token=instance_token)
        if not resp.ok:
            # A non-2xx (503 "No model specified", 401, or a 500 when the engine
            # load fails because the media backend still holds VRAM - the exact
            # hazard this module manages) means the reload did NOT happen. Report
            # that honestly instead of the false "Chat model ready." - lazy reload
            # on the next message still recovers, so this is a message fix, not an
            # escalation. Mirrors unload_chat_for_media's resp.ok check above.
            job.push({"type": "line", "text":
                      f"Reload deferred to the next message (HTTP {resp.status_code})."})
            return
        job.push({"type": "line", "text": "Chat model ready."})
    except Exception as e:
        job.push({"type": "line", "text": f"Reload deferred to the next message ({e})."})
