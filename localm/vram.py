# SPDX-License-Identifier: AGPL-3.0-or-later
"""VRAM-aware model-swap policy for media generation.

Before a media gen (image, music or video), the chat LLM is unloaded to
hand VRAM to the media model, then reloaded. On a large card both models fit at
once, so the swap is pure latency (and a failed/forgotten reload leaves the card
empty). This module decides whether the swap is actually needed; the unload
endpoint separately verifies VRAM was reclaimed before the media model loads
(the driver-hang guard - see inference/http_server.py).
"""
from __future__ import annotations

import contextlib
import functools
import threading
import time
from typing import Any, Callable, Optional

_VALID_POLICIES = ("auto", "always", "never")

# Fixed VRAM headroom a GGUF load reserves beyond model weights: KV cache plus
# compute buffers. The GGUF backend preflights, sysstats.estimate_vram and
# discover.fit_label all derive their value from here.
VRAM_OVERHEAD_BYTES = int(1.5e9)
# Safety factor the fit badge applies to on-disk weight size.
VRAM_WEIGHT_FACTOR = 1.10

# Minimum free-VRAM rise after unload that counts as the model having been freed.
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

    The legacy ``reload_llm_after_imagine`` / per-plugin
    ``reload_llm_after_generate`` boolean is a SEPARATE axis: it controls whether
    the chat model is *reloaded after* a generation (eager vs lazy), surfaced as
    the backend ``reload_after`` setting, and does NOT influence the swap
    (unload-before) decision. The two axes stay independent, so an existing
    ``false`` config keeps its lazy-reload behaviour while still getting the
    VRAM-aware ``auto`` swap default.
    """
    block = plugin_block or {}
    explicit = block.get("model_swap_policy",
                         (full_config or {}).get("model_swap_policy"))
    if isinstance(explicit, str) and explicit.lower() in _VALID_POLICIES:
        return explicit.lower()
    return "auto"


# Per-backend VRAM estimates (GB) for the media model, used by the 'auto' swap
# decision when plugins.<name>.vram_estimate_gb is unset.
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
    Pass *read_free* to inject the free-VRAM reading in tests.

    The default reading uses ``discover.vram_capacity_joined()`` (combined
    free across a configured multi-GPU split, or across every discrete GPU when
    neither a split nor a main GPU is chosen, else the same single-GPU number
    ``vram_info()`` reports, waiting for a probe already in flight) - the same
    "will it fit" ceiling
    ``http_server.switch_engine``'s eviction gate uses, so a split-configured
    machine does not needlessly swap the chat model out when the combined
    capacity already covers the media job."""
    if read_free is None:
        def read_free() -> Optional[int]:
            from localm.discover import vram_capacity_joined
            return vram_capacity_joined()[0].get("free")
    return should_swap_for_media(
        read_free(), settings.get("vram_estimate_bytes"),
        policy=settings.get("swap_policy", "auto"))


def media_keep_note(settings: dict, *,
                    read_free: Optional[Callable[[], Optional[int]]] = None) -> str:
    """The job-log line for a media job that does not unload the chat model, naming
    the reason ``decide_media_swap`` returned False for *settings*."""
    if settings.get("swap_policy", "auto") == "never":
        return ("Swap policy is 'never' - a loaded chat model stays loaded; the job "
                "may run slower if VRAM runs short.")
    if read_free is not None:
        free = read_free()
    else:
        from localm.discover import vram_capacity
        cap = vram_capacity()
        free = cap.get("free") if isinstance(cap, dict) else None
    need = settings.get("vram_estimate_bytes")
    if free is None or need is None:
        return "No chat model unload needed for this job."
    return (f"{free / 1024 ** 3:.1f} GB of VRAM is free and this job needs about "
            f"{need / 1024 ** 3:.1f} GB - no chat model unload needed.")


def media_single_device_shortfall(settings: dict, *,
                                  config: Optional[dict] = None) -> Optional[dict]:
    """``{"index", "needed", "free"}`` when the ONE card ComfyUI will actually use
    cannot hold the WHOLE media model, else ``None`` (fine, or nothing to check).

    THE OTHER HALF OF ``decide_media_swap``. That gate reads COMBINED free across a
    configured split, which is the right CAPACITY question and MUST stay that way
    (every capacity decision uses the combined number). But no single model is
    DIVIDED across the split the way a tensor_split GGUF is: each component loads
    whole onto one ``get_torch_device()`` (``model_management.py:194``). So the gate
    can be satisfied by 8 GB combined across two 4 GB cards while a 4 GB model lands
    on ONE of them and OOMs, spills to shared RAM, or trips the ROCm TDR. The gate
    is not wrong; it cannot see placement.

    localm DOES emit ComfyUI core's ``Select*Device`` nodes
    (:func:`localm.media.comfy_client.inject_device_placement`), but under the v1
    placement policy the big MODEL stays on the preferred card (only the smaller
    CLIP + VAE move to the second card). So the preferred card still holds the model
    whole, and "does the preferred card hold the WHOLE job" is a SAFE, conservative
    bound here: it over-counts (it still demands room for the CLIP + VAE that
    placement moved off this card), so it errs toward swapping the chat model, never
    toward an OOM. A per-component-per-card refinement needs per-component byte
    sizes, which do not exist yet (only whole-job estimates).

    This answers the placement question instead: does the specific card
    ``discover.resolve_preferred_device()`` chose hold the WHOLE model plus the same
    headroom the aggregate demands. NOT ``discover.gpu_split_shortfall``, which
    checks a per-device RATIO SHARE - right for a tensor_split GGUF, an UNDER-check
    here (a card holding 40% of a split would be asked for 40% of the model and then
    handed 100% of it).

    Uses the SAME headroom as ``should_swap_for_media`` so the per-device check is
    not held to a thinner margin than the aggregate ceiling it composes with.

    Returns None (nothing to check) when the policy is not 'auto' (an explicit
    always/never is the user's choice), when the capacity ``decide_media_swap``
    reads is one card's (no split configured and either a main GPU chosen or
    fewer than two discrete GPUs, ``discover.vram_capacity``), when the estimate
    is unknown ('auto' already swaps on unknown), when the live per-device
    reading did not complete fresh this call, or when per-device free is
    unmeasurable (cannot check; do not block a load that might work).

    Freshness, NOT scope, is what this checks. A non-fresh (GPU_PROBE_TIMEOUT or
    BUSY) reading is a frozen last-known-good value from an earlier probe, and like
    ``gpu_split_shortfall`` this does not compute a shortfall from one. It does NOT
    gate on ``free_scope`` the way a reading PRESENTED to a user as current fact
    must (see ``sysstats._vram_reading_trusted``): a PROCESS-scoped reading
    OVER-states free, and this function's ONLY use of the number is to trigger the
    SAME protective action an under-detection would risk skipping - the extra
    chat-model swap that covers what the scope-blind aggregate ``decide_media_swap``
    check cannot see. Gating on scope would make the check silently no-op on every
    Windows + AMD ROCm/HIP box, which is exactly the platform the per-device gap is
    real on."""
    if settings.get("swap_policy", "auto") != "auto":
        return None
    need = settings.get("vram_estimate_bytes")
    if not isinstance(need, int) or need <= 0:
        return None
    from localm.config import load_config
    from localm.discover import (GPU_PROBE_OK, applied_split_device_count,
                                 list_gpus, resolve_preferred_device)
    cfg = config if config is not None else load_config()
    # applied_split_device_count is loader truth; the detected count reports < 2
    # for a live split on vulkan. With no split and no main GPU, the capacity is
    # summed over every discrete GPU (discover.vram_capacity) and needs the same check.
    split = applied_split_device_count(cfg) >= 2
    if not split and (cfg.get("gpu_split_indices")
                      or cfg.get("main_gpu_index") is not None):
        return None
    gpus, status = list_gpus(return_status=True)
    if status != GPU_PROBE_OK:
        return None
    if not split:
        from localm.discover import _discrete_gpus_sum
        if _discrete_gpus_sum(gpus) is None:
            return None
    idx = resolve_preferred_device(cfg, gpus=gpus)
    if idx is None and not split:
        # Nothing chosen: the media backend loads on the first GPU.
        from localm.discover import _llama_visible_torch_devices
        firsts = sorted(g.get("index") for g in _llama_visible_torch_devices(gpus)
                        if isinstance(g.get("index"), int))
        idx = firsts[0] if firsts else None
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
    """The distinct non-empty ``gpu:N`` targets in a placement plan (``[]`` when the plan
    is None or every component is left on its default card)."""
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
    """A user-visible line about how media generation uses the box's GPUs, or ``None``
    when there is nothing worth saying.

    Three cases, in order:

    1. **Placement ACTIVE** (*placement* has a real ``gpu:N`` target): say what was
       placed where. Per-component placement means the second card DOES carry work
       (the text encoder + VAE), even though a single model still cannot be sharded
       across cards. Fires from the plan alone: placement is capability-driven (two
       visible cards is enough), so it does not require a configured chat split.

    2. **Split configured but placement UNAVAILABLE** (*capability* is present and not
       available): the user configured a split and is not getting per-component
       placement either - give the HONEST reason from the capability (old ComfyUI, one
       card visible), not a flat "runs on one card".

    3. **Split configured, no placement info threaded in** (legacy callers): the
       single-card notice.

    Returns None on a single-GPU or unsplit box with no active placement, so a user
    who configured nothing is never nagged about a shortfall that does not exist. Not
    a ``logger.debug`` line: the always-on ring buffer is INFO+, so a debug line is
    invisible without --debug and would never reach a bug report."""
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

    # applied_split_device_count is loader truth; the detected count collapses to
    # < 2 on vulkan and would suppress the notice.
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
    """Decide whether to unload the chat LLM before the shared embedder loads
    (``localm.inference.embedder.get_embedder()``), reading live free VRAM.

    Generalizes ``decide_media_swap`` for a caller that has no per-plugin
    ``settings`` dict of its own - just the embedder's own estimated footprint
    and the resolved ``model_swap_policy``. Both funnel through the same
    ``should_swap_for_media`` decision, so the embedder gets the identical
    auto/always/never semantics the image/music/video plugins already use."""
    if read_free is None:
        def read_free() -> Optional[int]:
            from localm.discover import vram_capacity_joined
            return vram_capacity_joined()[0].get("free")
    return should_swap_for_media(read_free(), embedder_estimate_bytes, policy=policy)


_EVICT_CANCEL_SETTLE_S = 5.0


def _cancel_abandoned_eviction(fut, task, loop) -> None:
    """Cancel the eviction its caller stopped waiting for, and block until the
    loop has applied the cancellation. A queued executor step of the unload is
    cancelled before the calling worker thread is freed to run it; a step
    already running finishes. Never raises.
    See test_abandoned_eviction_is_cancelled_when_its_caller_gives_up."""
    import asyncio
    from localm.debuglog import logger

    fut.cancel()
    if task is None:
        return

    async def _settle():
        # fut.cancel() above already requested one cancellation of the task.
        if not task.done() and not task.cancelling():
            task.cancel()
        try:
            await task
        except BaseException:
            pass

    try:
        asyncio.run_coroutine_threadsafe(_settle(), loop).result(
            timeout=_EVICT_CANCEL_SETTLE_S)
    except Exception as e:
        logger.debug("embedder: could not confirm the abandoned eviction was "
                     "cancelled (%s)", e)


_EMBEDDER_EVICTORS: list = []
_EMBEDDER_EVICTORS_LOCK = threading.Lock()
_NO_EVICTOR_WARNED = False

# Worst first: the status evict_chat_for_embedder reports when owners disagree.
_EVICT_STATUS_ORDER = ("error", "in_use", "pinned", "skipped", "unloaded",
                       "already_unloaded")


def register_embedder_evictor(evictor: Callable[[float], str]) -> Callable[[], None]:
    """Register *evictor* as an in-process owner of chat engines that
    ``evict_chat_for_embedder`` asks to free VRAM before the shared embedder
    loads. Returns a callable that unregisters it (idempotent).

    *evictor* is called as ``evictor(timeout_s)`` and returns one of
    "unloaded", "in_use", "pinned", "already_unloaded" or "error"; a raise or a
    non-string return counts as "error". It is called with neither
    ``embedder._LOCK`` nor ``engine._LOAD_LOCK`` held, and must not wait
    unboundedly."""
    with _EMBEDDER_EVICTORS_LOCK:
        _EMBEDDER_EVICTORS.append(evictor)

    def _unregister() -> None:
        with _EMBEDDER_EVICTORS_LOCK:
            try:
                _EMBEDDER_EVICTORS.remove(evictor)
            except ValueError:
                pass
    return _unregister


def _worst_evict_status(statuses: list) -> str:
    """The first of *statuses* in ``_EVICT_STATUS_ORDER``; an unknown status
    counts as "error"."""
    known = [s if s in _EVICT_STATUS_ORDER else "error" for s in statuses]
    return min(known, key=_EVICT_STATUS_ORDER.index)


def evict_chat_for_embedder(*, timeout_s: float = 300.0) -> str:
    """Free every loaded chat engine in this process so the shared embedder's
    own load has VRAM room.

    Two kinds of owner are asked, within one *timeout_s* budget:

    - the HTTP server, when its event loop (``http_server._server_loop``) is
      running: ``http_server.unload_all_models`` is submitted onto that loop via
      ``asyncio.run_coroutine_threadsafe`` and THIS thread blocks on the result.
      It runs ON the loop, where it honors the in-flight-request pin and
      serializes with ``get_engine``;
    - every evictor registered with ``register_embedder_evictor`` (the MCP
      server's ``EngineCache``).

    Returns the worst status any owner reported ("error", "in_use", "pinned",
    "skipped", "unloaded", "already_unloaded", in that order). With no owner
    reachable it logs a WARNING (once per process) and returns "skipped". On any
    status but "unloaded"/"already_unloaded" the embedder loads alongside
    whatever chat model is resident; this function never raises."""
    from localm.debuglog import logger
    from localm.inference import http_server as _hs

    global _NO_EVICTOR_WARNED
    deadline = time.monotonic() + timeout_s
    with _EMBEDDER_EVICTORS_LOCK:
        evictors = list(_EMBEDDER_EVICTORS)
    statuses = []
    loop = getattr(_hs, "_server_loop", None)
    if loop is not None and loop.is_running():
        statuses.append(_evict_via_server_loop(_hs, loop, timeout_s))
    for evictor in evictors:
        remaining = min(timeout_s, max(0.0, deadline - time.monotonic()))
        try:
            status = evictor(remaining)
        except Exception as e:
            logger.warning("embedder: freeing VRAM for the embedding model "
                           "failed (%s: %s)", type(e).__name__, e)
            status = "error"
        statuses.append(status if isinstance(status, str) else "error")
    if not statuses:
        if not _NO_EVICTOR_WARNED:
            _NO_EVICTOR_WARNED = True
            logger.warning("embedder: no chat model owner in this process can "
                           "free VRAM for the embedding model; it loads "
                           "alongside whatever else holds the GPU")
        return "skipped"
    return _worst_evict_status(statuses)


def _evict_via_server_loop(_hs, loop, timeout_s: float) -> str:
    """``evict_chat_for_embedder``'s HTTP-server half: run
    ``_hs.unload_all_models()`` on the running server *loop* and return its
    status, "skipped" when called on that loop's own thread, or "error" when it
    did not complete within *timeout_s*."""
    import asyncio
    from localm.debuglog import logger

    # Running on the server loop's own thread would deadlock on the
    # run_coroutine_threadsafe(...).result() below, so skip the guarded eviction
    # and load the embedder alongside the resident chat model.
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
    # INVARIANT: unload_all_models runs its own steps on the loop's default
    # executor, so every caller that blocks here ON that executor must be
    # concurrency-bounded on the loop side (as routes/chat.py's _embedder_sem
    # bounds the dedicated /v1/embeddings path).
    # See test_dedicated_embed_path_holds_one_pool_worker_at_a_time.
    holder: dict = {}

    async def _tracked_unload():
        holder["task"] = asyncio.current_task()
        return await _hs.unload_all_models()

    fut = None
    try:
        fut = asyncio.run_coroutine_threadsafe(_tracked_unload(), loop)
        res = fut.result(timeout=timeout_s)
    except Exception as e:
        if fut is not None:
            _cancel_abandoned_eviction(fut, holder.get("task"), loop)
        logger.debug("embedder: guarded chat-model eviction did not complete "
                     "(%s); loading the embedder without evicting", e)
        return "error"
    return res.get("status", "unloaded") if isinstance(res, dict) else "unloaded"


def _vram_free_reading() -> tuple:
    """``(free_bytes, fresh, scope)``: a free-VRAM reading, whether the probe behind
    it was fresh (list_gpus() completed inside its deadline) rather than a
    timed-out or busy fallback to a stale last-known-good value, and the reading's
    ``free_scope``.

    Used for BOTH ends of the before/after delta.

    A caller that reports vram_before_bytes/vram_after_bytes/vram_freed to the
    user must know this: presenting a stale fallback as the CURRENT state makes
    /v1/models/unload report byte-identical before/after readings, and
    vram_freed: false, across calls made minutes apart with genuinely different
    real GPU states.

    ``scope`` is the reading's ``free_scope`` (see
    discover._apply_device_global_free): FREE_SCOPE_PROCESS means the number
    counts only THIS process's allocations, so on that platform it cannot see a
    model the isolated GGUF worker loaded - a SECOND, independent way this same
    before/after report can be wrong while the probe is perfectly fresh (not the
    staleness the 'fresh' flag guards). None when the double or fallback did not
    say.

    Defensive against a test double that patches vram_capacity()/vram_info()/
    list_gpus() with a plain no-kwarg callable or a fixed dict return_value
    (neither accepts nor implements return_status): such a double is treated as
    "status unknown, assume fresh" rather than raising on a rejected kwarg or an
    unexpected shape - only a REAL probe chain needs to prove itself stale."""
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
    """Free VRAM in bytes from a FRESH probe, or None when the reading is not
    live - the reader wait_for_vram_release() polls for the 'after' end.

    Checking freshness on the 'before' end alone is not enough: the before-probe
    is itself what WARMS the driver, and the native unload running between the
    two reads is exactly the kind of work that wedges it. Once any probe
    overruns, discover._gpu_probe_inflight stays True until the abandoned thread
    returns, so every subsequent poll is served the frozen last-known-good value,
    which the before-read just refreshed. That makes after == before with no
    rise, and the endpoint would conclude "vram_freed": false with
    before_fresh=True and therefore no uncertainty flag at all. Returning None
    here makes that unmeasurable rather than falsely equal.

    None means "could not measure now", never a fabricated 0.

    NOT every free-VRAM caller uses this. Ones weighing CAPACITY rather than
    reporting a completed action (the fit badges, the swap gates, the eviction
    ceiling) still decide on a reading that is not fresh: the fit badges call
    vram_capacity() directly, the swap gates discover.vram_capacity_joined()
    (which waits for a probe already in flight, then takes whatever it got). A
    stale ceiling costs an over- or under-swap, while refusing to decide at all
    would break a working box every time the probe merely ran slow.

    ``sysstats.vram_capacity()`` behind GET /api/stats and
    ``gui/routes/models/inventory.py``'s /api/vram-estimate gate on
    ``sysstats._vram_reading_trusted()`` (fresh AND device-global) before showing
    `used`/`free` at all. ``http_server.switch_engine``'s 503 refusals (via
    ``discover.gpu_split_shortfall``) require a fresh GPU_PROBE_OK reading before
    emitting a shortfall but do not gate on free_scope, which is sound for a
    refuse-only decision: an over-stated free only makes the refusal more
    conservative. Any NEW caller that quotes a free figure as current fact needs
    the same freshness check, and, unless it is refuse-only, the scope check
    too."""
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
    """Poll ``read_free()`` until free VRAM has risen by at least ``min_rise_bytes``
    versus ``before_bytes`` (the native unload's deferred free has completed), or
    ``timeout_s`` elapses.

    Returns ``(released, final_free)``: ``True`` once the rise is observed,
    ``False`` on timeout, and ``None`` (released) when the outcome CANNOT BE
    VERIFIED - either because ``before_bytes`` is None, or because the reading was
    still unmeasurable when the timeout expired.

    That second None case is the honesty guard. ``False`` is a positive claim -
    "VRAM did not drop" - and a poll that could not take a reading does not
    support it. ``read_free`` is normally backed by the deadline-bounded GPU probe
    (``discover.list_gpus``), whose status-aware callers pass None here the moment
    a probe stops answering and starts serving a FROZEN last-known-good value.
    Without this branch that frozen value equals ``before_bytes``, shows no rise,
    and reports a confident "no VRAM was freed" while VRAM was in fact freed. None
    reuses this function's already-documented "cannot verify" meaning rather than
    inventing a fourth outcome.

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
            # Poll to the deadline first; only the reading we end on decides.
            return ((False, final) if final is not None else (None, final))
        sleep(poll_s)
        final = read_free()


# --------------------------------------------------------------------------- #
#  Chat<->media VRAM handoff (shared by the image/music/video plugins)         #
# --------------------------------------------------------------------------- #
#
# Each media plugin swaps the chat model out before generating and back in after,
# via a self-authenticated round trip to /v1/models/unload and /v1/models/load.
# Only the progress-message wording differs between them.

def unload_chat_for_media(job: Any, self_url: str, media_label: str,
                          instance_token: Optional[str] = None) -> bool:
    """Unload the chat model BEFORE a media (image/music/video) model loads, so
    it gets the VRAM.

    Uses the same bearer-token + TLS handling as the reload path: the
    ``/v1/models/unload`` endpoint needs the models-write scope, so an
    unauthenticated call is rejected and the chat model stays resident - the
    media model then loads on top of it and hangs the GPU driver. Logs the
    outcome (and the VRAM freed) on *job* so a failure is visible instead of
    silent. Returns True only when the server reports NOTHING left resident
    and in use; any pinned model (the chat engine or a sibling) yields False,
    because what matters to this caller is resident VRAM, and the caller then
    falls back to its own conservative swap handling.
    *media_label* names the caller ("image"/"music"/"video") for the messages.
    *instance_token*: this instance's attach token (``request.app.state.
    instance_token``), forwarded to ``self_request`` so the call authenticates
    in OPEN mode too - see ``selfclient.self_request``'s docstring. Without it
    this round trip 403s on every keyless (default) server, which is a
    permanent no-op of the exact VRAM collision this module exists to prevent."""
    job.push({"type": "line", "text": "Freeing VRAM: unloading the chat model..."})
    try:
        from localm.selfclient import self_request
        resp = self_request("POST", "/models/unload?media=false", timeout=300,
                            base_url=self_url, instance_token=instance_token)
        if not resp.ok:
            job.push({"type": "line", "text":
                      f"Could not unload the chat model (HTTP {resp.status_code}) - "
                      f"the {media_label} backend may run low on VRAM."})
            return False
        data = {}
        try:
            data = resp.json()
        except Exception:
            # resp.ok already confirmed the unload was accepted, so an unparsable
            # body falls through to the generic message below.
            pass
        if data.get("status") == "already_unloaded":
            job.push({"type": "line", "text":
                      "No chat model was loaded - VRAM already free."})
            return True
        if data.get("status") == "in_use":
            # "in_use" means a chat engine mid-generation was left resident, so
            # VRAM was not freed. Reported as a failure so the caller falls back
            # to its own conservative swap handling.
            job.push({"type": "line", "text":
                      "Chat model is busy (still generating a reply) and could "
                      f"not be unloaded - the {media_label} backend may run low "
                      "on VRAM."})
            return False
        skipped = [str(m) for m in (data.get("skipped_in_use") or [])]
        if skipped:
            # "unloaded" means something was freed, not necessarily the chat
            # model: a pinned chat engine lands in skipped_in_use under that same
            # status. Reported as a failure for this caller.
            job.push({"type": "line", "text":
                      "Freed what could be freed, but still in use and NOT "
                      f"unloaded: {', '.join(skipped)} - the {media_label} "
                      "backend may run low on VRAM."})
            return False
        before, after = data.get("vram_before_bytes"), data.get("vram_after_bytes")
        uncertain = bool(data.get("vram_reading_uncertain"))
        if uncertain:
            # The server flagged its own reading as possibly stale, so neither a
            # GB figure nor "has not dropped yet" is reported. Checked first.
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


# --------------------------------------------------------------------------- #
#  Other native media runtimes and other localm instances                      #
# --------------------------------------------------------------------------- #
#
# The native image and video backends share one stable-diffusion.cpp worker; the
# native music backend runs a KoboldCpp server. Each stays loaded after its job
# (idle timeout, 10 minutes), so a job of the other kind starts with that VRAM
# still taken.

_NATIVE_RUNTIME_OF = {"image": "sdcpp", "video": "sdcpp", "music": "koboldcpp"}
_NATIVE_RUNTIME_NAME = {"sdcpp": "native image/video worker",
                        "koboldcpp": "native music runtime"}


def _native_runtime_loaded(key: str) -> bool:
    """Whether this process currently holds the native runtime *key* loaded."""
    if key == "sdcpp":
        from localm.media.sdcpp import shared
        return shared.worker_pid() is not None
    from localm.media.koboldcpp import server
    return server.running_pid() is not None


def _native_runtime_free(key: str,
                         cancel_check: Optional[Callable[[], bool]] = None) -> bool:
    """Stop the native runtime *key* (the same call its backend's ``free_vram``
    makes), waiting for the job it is running unless *cancel_check* turns true
    first. True once it is no longer running; False when the wait was cancelled
    or the runtime kept running."""
    if key == "sdcpp":
        from localm.media.sdcpp import shared
        return shared.free(cancel_check)
    from localm.media.koboldcpp import server
    server.stop(cancel_check)
    return server.running_pid() is None


def loaded_native_media_runtimes() -> list:
    """Display names of the native media runtimes this process holds loaded.
    Never raises."""
    names = []
    for key, name in _NATIVE_RUNTIME_NAME.items():
        try:
            if _native_runtime_loaded(key):
                names.append(name)
        except Exception:
            continue
    return names


def _native_runtime_busy(key: str) -> bool:
    """Whether a job holds the native runtime *key*'s job lock. Takes no lock."""
    if key == "sdcpp":
        from localm.media.sdcpp import shared
        return shared.is_busy()
    from localm.media.koboldcpp import server
    return server.is_busy()


def native_media_runtime_states() -> dict:
    """``{display name: "busy" | "idle" | "unknown"}`` for each native media
    runtime this process holds loaded: "busy" while a job holds it, "unknown"
    when reading whether it is loaded or busy raised. Stops and waits for
    nothing. Never raises."""
    states = {}
    for key, name in _NATIVE_RUNTIME_NAME.items():
        try:
            if _native_runtime_loaded(key):
                states[name] = "busy" if _native_runtime_busy(key) else "idle"
        except Exception:
            states[name] = "unknown"
    return states


def _native_runtime_stop_if_idle(key: str) -> Optional[bool]:
    """Stop the native runtime *key* when no job holds it, without waiting. None
    when a job holds it; otherwise True once it is no longer running."""
    if key == "sdcpp":
        from localm.media.sdcpp import shared
        return shared.free_if_idle()
    from localm.media.koboldcpp import server
    return server.stop_if_idle()


MEDIA_STOPPED = "stopped"
MEDIA_BUSY = "busy"
MEDIA_STOP_FAILED = "failed"


def stop_idle_native_media() -> dict:
    """Stop every native media runtime this process holds loaded that no job is
    using, without waiting for a job to end. Returns ``{display name: outcome}``
    for each runtime that was loaded, the outcome being ``MEDIA_STOPPED``,
    ``MEDIA_BUSY`` (a job holds it; it is left running) or ``MEDIA_STOP_FAILED``
    (the stop raised, or the process kept running; logged at WARNING). Empty
    when none is loaded. Never raises. Takes each runtime's job lock without
    blocking; a stop itself can take several seconds (the worker is asked to
    exit, then killed)."""
    from localm.debuglog import logger
    outcomes = {}
    for key, name in _NATIVE_RUNTIME_NAME.items():
        try:
            if not _native_runtime_loaded(key):
                continue
            stopped = _native_runtime_stop_if_idle(key)
        except Exception as e:
            logger.warning("media: could not stop the %s (%s: %s)",
                           name, type(e).__name__, e)
            outcomes[name] = MEDIA_STOP_FAILED
            continue
        if stopped is None:
            outcomes[name] = MEDIA_BUSY
        elif stopped:
            outcomes[name] = MEDIA_STOPPED
        else:
            logger.warning("media: the %s kept running after it was stopped", name)
            outcomes[name] = MEDIA_STOP_FAILED
    return outcomes


def _short_of(free: Optional[int], need: int) -> Optional[bool]:
    """True when *free* cannot hold *need* plus the standard headroom, False
    when it can, None when there is no reading."""
    if free is None:
        return None
    return free < need + _DEFAULT_HEADROOM


class ReleaseResult(list):
    """The runtimes ``release_other_native_media`` stopped. ``cancelled`` is True
    when the job was cancelled while the helper was stopping runtimes; the caller
    must end the job without unloading the chat model."""
    cancelled = False


def release_other_native_media(
    job: Any, settings: dict, media_label: str, *,
    read_reading: Optional[Callable[[], tuple]] = None,
    read_free: Optional[Callable[[], Optional[int]]] = None,
) -> ReleaseResult:
    """Before a native *media_label* job ("image" / "music" / "video"): stop this
    process's OTHER loaded native media runtimes when the job's
    ``vram_estimate_bytes`` plus headroom exceeds live free VRAM, one job-log line
    per runtime stopped. Returns the runtimes stopped (empty when none) as a
    :class:`ReleaseResult`.

    Does nothing for a ComfyUI job, a job with no estimate, or when free VRAM
    covers the job. A reading that is stale, unmeasurable or process-scoped cannot
    see another process's memory, so it counts as short whenever such a runtime is
    loaded - the same "unknown means act" rule the chat swap follows.

    Stopping a runtime waits for the job it is running to finish; it never kills
    one. When *job* is cancelled during that wait (or before it) the remaining
    runtimes are left alone and the call returns at once; the caller must check
    ``job.cancel_requested``. After a stop, waits for free VRAM to rise so the next load does not race
    the driver's release. *read_reading* / *read_free* are injectable for tests."""
    if settings.get("backend") != "native":
        return ReleaseResult()
    need = settings.get("vram_estimate_bytes")
    if not isinstance(need, int) or isinstance(need, bool) or need <= 0:
        return ReleaseResult()
    own = _NATIVE_RUNTIME_OF.get(media_label)
    others = []
    for key in _NATIVE_RUNTIME_NAME:
        if key == own:
            continue
        try:
            if _native_runtime_loaded(key):
                others.append(key)
        except Exception as e:
            from localm.debuglog import logger
            logger.debug("media: could not tell whether the %s is loaded (%s)",
                         _NATIVE_RUNTIME_NAME[key], e)
    if not others:
        return ReleaseResult()
    from localm.discover import FREE_SCOPE_DEVICE
    try:
        free, fresh, scope = (read_reading or _vram_free_reading)()
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("media: free VRAM reading failed (%s)", e)
        free, fresh, scope = None, False, None
    measured = free if (fresh and scope == FREE_SCOPE_DEVICE) else None
    if _short_of(measured, need) is False:
        return ReleaseResult()
    if read_free is None:
        read_free = _live_free_vram_bytes
    released = ReleaseResult()
    for key in others:
        name = _NATIVE_RUNTIME_NAME[key]
        if job.cancel_requested:
            break
        if measured is None:
            why = "free VRAM could not be read"
        else:
            why = (f"this job needs about {need / 1024 ** 3:.1f} GB and "
                   f"{measured / 1024 ** 3:.1f} GB is free")
        job.push({"type": "line", "text":
                  f"Freeing VRAM: stopping the {name} ({why})."})
        try:
            stopped = _native_runtime_free(key, lambda: job.cancel_requested)
        except Exception as e:
            job.push({"type": "line", "text":
                      f"Could not stop the {name} ({e}) - the {media_label} "
                      "backend may run low on VRAM."})
            continue
        if job.cancel_requested:
            if stopped:
                released.append(key)
            break
        if not stopped:
            job.push({"type": "line", "text":
                      f"The {name} kept its models in VRAM - the {media_label} "
                      "backend may run low on VRAM."})
            continue
        released.append(key)
        wait_for_vram_release(read_free, before_bytes=measured)
    if job.cancel_requested:
        job.push({"type": "line", "outcome": True, "text": "Cancelled."})
        released.cancelled = True
    return released


def note_other_vram_holders(
    job: Any, settings: dict, media_label: str, *,
    read_reading: Optional[Callable[[], tuple]] = None,
    list_peers: Optional[Callable[[], list]] = None,
    gpu_index: Optional[int] = None,
) -> list:
    """After the chat handover, before a native *media_label* job loads: when free
    VRAM is still short of the job's estimate, name in the job log each OTHER
    localm instance on this machine that holds VRAM on the same GPU (its port, its
    chat model, its loaded native media runtimes). Returns the peers named.

    Only informs; it never asks a peer to unload. Silent when the reading is
    unusable, free VRAM covers the job, or no peer holds anything - the
    peer scan probes loopback ports, so it runs only on a shortfall. Never
    raises. *gpu_index* is the device this job loads on (resolved from the
    config when None; a peer on another device is not named)."""
    try:
        if settings.get("backend") != "native":
            return []
        need = settings.get("vram_estimate_bytes")
        if not isinstance(need, int) or isinstance(need, bool) or need <= 0:
            return []
        from localm.discover import FREE_SCOPE_DEVICE
        free, fresh, scope = (read_reading or _vram_free_reading)()
        if not fresh or scope != FREE_SCOPE_DEVICE or _short_of(free, need) is not True:
            return []
        if list_peers is None:
            from localm import gpu_registry
            list_peers = gpu_registry.list_gpu_peers
        if gpu_index is None:
            try:
                from localm.config import load_config
                from localm.discover import last_gpu_reading, resolve_load_gpu_index
                gpu_index = resolve_load_gpu_index(
                    load_config(), gpus=last_gpu_reading() or [], quiet=True)
            except Exception:
                gpu_index = None
        named = []
        for p in list_peers():
            what = []
            if p.get("model"):
                what.append(f"chat model '{p['model']}'")
            media = [str(m) for m in (p.get("media") or [])]
            what.extend(f"the {m}" for m in media)
            if not what:
                continue
            if gpu_index is not None and int(p.get("gpu_index") or 0) != gpu_index:
                continue
            named.append(p)
            job.push({"type": "line", "text":
                      f"Another localm instance (port {p.get('port')}) is holding "
                      f"VRAM on this GPU ({', '.join(what)}); this {media_label} job "
                      f"needs about {need / 1024 ** 3:.1f} GB and "
                      f"{free / 1024 ** 3:.1f} GB is free. Unload it there to make room."})
        return named
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("media: could not look for other VRAM holders (%s)", e)
        return []


# --------------------------------------------------------------------------- #
#  One chat reload at a time, and no media model load under it                 #
# --------------------------------------------------------------------------- #
#
# Every media job unloads the chat model, generates, then reloads the chat model.
# ``media_handover_job`` counts the jobs in flight so those steps never
# interleave: the job that finishes while another is still in flight leaves the
# reload to the last one, and a job that starts while a reload runs waits for it.
# A job that leaves the reload to a later job also leaves its backend to be freed
# there, so the chat model never reloads beside a media runtime still in VRAM.
# ``_handover_cv`` guards the fields below and is never held across a backend
# call, a lock of ``localm.media.sdcpp.shared`` or a request.

HANDOVER_POLL_SECONDS = 0.25
_handover_cv = threading.Condition()
_handover_jobs = 0
_handover_reloading = False
_handover_deferred: Optional[tuple] = None
_handover_deferred_backends: list = []


def _handover_enter(job: Any) -> bool:
    """Count *job* as in flight once no chat reload is running. Waits for a running
    reload, polling ``job.cancel_requested``; returns False (job not counted) when
    cancelled while waiting."""
    global _handover_jobs
    with _handover_cv:
        if _handover_reloading:
            job.push({"type": "line", "text":
                      "Waiting for the chat model to finish reloading..."})
            while _handover_reloading:
                _handover_cv.wait(HANDOVER_POLL_SECONDS)
                if job.cancel_requested:
                    job.push({"type": "line", "outcome": True, "text": "Cancelled."})
                    return False
        _handover_jobs += 1
        return True


def _handover_exit(job: Any) -> None:
    """Uncount *job*. When it is the last one in flight and an earlier job left its
    chat reload to it, run that reload first (while still counted)."""
    global _handover_jobs, _handover_deferred
    with _handover_cv:
        pending = _handover_deferred if _handover_jobs == 1 else None
        if pending is not None:
            _handover_deferred = None
    try:
        if pending is not None:
            reload_chat_after_media(job, *pending)
    except Exception as e:
        job.push({"type": "line", "text": f"Reload deferred to the next message ({e})."})
    finally:
        with _handover_cv:
            _handover_jobs -= 1
            _handover_cv.notify_all()


def media_handover_job(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Wrap a media job body ``fn(job)`` (image, video, music, upscale) in the
    handover accounting described above. Returns False without running *fn* when
    the job is cancelled while waiting for a chat reload; otherwise returns what
    *fn* returns."""
    @functools.wraps(fn)
    def wrapper(job: Any) -> Any:
        if not _handover_enter(job):
            return False
        try:
            return fn(job)
        finally:
            _handover_exit(job)
    return wrapper


def _drop_deferred_reload_if_last() -> None:
    """Forget a chat reload (and the backends to free before it) that an earlier
    job left behind, when no other job is in flight."""
    global _handover_deferred
    with _handover_cv:
        if _handover_jobs <= 1:
            _handover_deferred = None
            _handover_deferred_backends.clear()


@contextlib.contextmanager
def restore_chat_after_media(job: Any, self_url: str, s: dict, backend: Any,
                             media_label: str, instance_token: Optional[str] = None,
                             *, swapped: bool = True):
    """Wrap the part of a media job body that runs after the chat model may have
    been unloaded. On leaving the block, whether it finished, failed or was
    cancelled, run ``reload_chat_after_media`` once when *swapped*.

    When the block raised, the reload is best effort: its own failure is logged to
    *job* and the block's exception propagates. After a clean exit a failing reload
    raises, so the job's ``mark_outcome`` decides the reported status. When not
    *swapped* and the job's reload-after setting is off, a reload an earlier job
    left to it is dropped (see ``reload_chat_after_media``)."""
    try:
        yield
    except BaseException:
        if swapped:
            try:
                reload_chat_after_media(job, self_url, s, backend, media_label,
                                        instance_token)
            except Exception as e:
                job.push({"type": "line", "text":
                          f"Reload deferred to the next message ({e})."})
        raise
    else:
        if swapped:
            reload_chat_after_media(job, self_url, s, backend, media_label,
                                    instance_token)
        elif not s["reload_after"]:
            _drop_deferred_reload_if_last()


def reload_chat_after_media(job: Any, self_url: str, s: dict, backend: Any,
                            media_label: str,
                            instance_token: Optional[str] = None) -> None:
    """Hand VRAM back: ask *backend* (the plugin's own backend module, exposing
    ``free_vram(s)``) to drop its models, then reload the chat model so the next
    reply is instant. Skipped when reload-after-generate is off: the backend
    stays loaded, and the next chat model load stops it first
    (``http_server._switch_free_own_media``).
    *instance_token*: forwarded to ``self_request`` - see
    ``unload_chat_for_media``'s docstring.

    Inside a ``media_handover_job`` with another media job still in flight, nothing
    is freed or reloaded here: the reload, and the freeing of this job's backend, is
    left to the last job to finish. While the reload runs, jobs that start wait for
    it. The last job's reload-after setting governs: when it is off, a reload an
    earlier job left to it is dropped and those backends stay loaded."""
    if not s["reload_after"]:
        _drop_deferred_reload_if_last()
        job.push({"type": "line", "text":
                  f"Keeping the {media_label} backend loaded (reload is off) - "
                  "the chat model reloads on the next message and stops it "
                  "first."})
        return
    global _handover_reloading, _handover_deferred
    with _handover_cv:
        if _handover_jobs > 1:
            _handover_deferred = (self_url, s, backend, media_label, instance_token)
            if all(b is not backend for b, _s, _label in _handover_deferred_backends):
                _handover_deferred_backends.append((backend, s, media_label))
            job.push({"type": "line", "text":
                      "Another media job is waiting - the chat model reloads after "
                      "the last one."})
            return
        _handover_reloading = True
        others = [d for d in _handover_deferred_backends if d[0] is not backend]
        _handover_deferred_backends.clear()
    try:
        for other_backend, other_s, other_label in others:
            if not other_backend.free_vram(other_s):
                job.push({"type": "line", "text":
                          f"The {other_label} backend kept its models in VRAM - the chat "
                          "model will reload on the next message instead."})
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
                # hazard this module manages) means the reload did NOT happen, and is
                # reported as such instead of "Chat model ready.". Lazy reload on the
                # next message still recovers.
                job.push({"type": "line", "text":
                          f"Reload deferred to the next message (HTTP {resp.status_code})."})
                return
            job.push({"type": "line", "text": "Chat model ready."})
        except Exception as e:
            job.push({"type": "line", "text": f"Reload deferred to the next message ({e})."})
    finally:
        with _handover_cv:
            _handover_reloading = False
            _handover_deferred = None
            _handover_deferred_backends.clear()
            _handover_cv.notify_all()
