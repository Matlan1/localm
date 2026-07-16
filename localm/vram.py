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
    Pass *read_free* to inject the free-VRAM reading in tests.

    The default reading uses ``discover.vram_capacity()`` (combined free
    across a configured multi-GPU split, else the same single-GPU number
    ``vram_info()`` reports) - the same "will it fit" ceiling
    ``http_server.switch_engine``'s eviction gate uses, so a split-configured
    machine does not needlessly swap the chat model out when the combined
    capacity already covers the media job."""
    if read_free is None:
        def read_free() -> Optional[int]:
            from localm.discover import vram_capacity
            return vram_capacity().get("free")
    return should_swap_for_media(
        read_free(), settings.get("vram_estimate_bytes"),
        policy=settings.get("swap_policy", "auto"))


def media_single_device_shortfall(settings: dict, *,
                                  config: Optional[dict] = None) -> Optional[dict]:
    """``{"index", "needed", "free"}`` when the ONE card ComfyUI will actually use
    cannot hold the WHOLE media model, else ``None`` (fine, or nothing to check).

    THE OTHER HALF OF ``decide_media_swap`` (REG-532). That gate reads COMBINED free
    across a configured split, which is the right CAPACITY question and MUST stay that
    way (every capacity decision uses the combined number). But no single model is
    DIVIDED across the split the way a tensor_split GGUF is: each component loads whole
    onto one ``get_torch_device()`` (``model_management.py:194``). So the gate can be
    satisfied by 2x4 GB = 8 GB combined while a 4 GB model lands on ONE 4 GB card and
    OOMs, spills to shared RAM, or trips the ROCm TDR. The gate is not wrong; it simply
    cannot see placement.

    NOTE, and do not let this rot: ComfyUI core CAN place DIFFERENT components (model /
    CLIP / VAE) on different cards via ``comfy_extras/nodes_multigpu.py``'s
    ``Select*Device`` nodes. localm does not emit those yet. When it does, this check
    must become per-component against each component's chosen card, not one card for
    the whole job - the estimate here would then be over-strict rather than wrong, but
    it would still be measuring a placement that no longer happens.

    This answers the placement question instead: does the specific card
    ``discover.resolve_preferred_device()`` chose hold the WHOLE model plus the same
    headroom the aggregate demands. Deliberately NOT ``discover.gpu_split_shortfall``,
    which checks a per-device RATIO SHARE - right for a tensor_split GGUF, an
    UNDER-check here (a card holding 40% of a split would be asked for 40% of the model
    and then handed 100% of it).

    Uses the SAME headroom as ``should_swap_for_media`` so the per-device check is not
    held to a thinner margin than the aggregate ceiling it composes with (the
    convention ``gpu_split_shortfall`` documents).

    Returns None (nothing to check) when the policy is not 'auto' (an explicit
    always/never is the user's choice and is not second-guessed), when no split is
    configured (the combined reading already IS the single card's), when the estimate
    is unknown ('auto' already swaps on unknown), or when per-device free is
    unmeasurable (cannot check; do not block a load that might work).
    """
    if settings.get("swap_policy", "auto") != "auto":
        return None
    need = settings.get("vram_estimate_bytes")
    if not isinstance(need, int) or need <= 0:
        return None
    from localm.config import load_config
    from localm.discover import (applied_split_device_count, list_gpus,
                                 resolve_preferred_device)
    cfg = config if config is not None else load_config()
    # applied_split_device_count (loader truth), NOT split_device_count: whether a
    # split is ACTIVE for chat/embeddings is a load-time fact, and on vulkan the
    # detected count would wrongly report < 2 for a live split (GPU-SPLIT-VKINDEX).
    # On a torch-blind box resolve_preferred_device() below still returns None, so
    # this stays a no-op there; on a mixed box it measures torch space, which is
    # where media actually runs.
    if applied_split_device_count(cfg) < 2:
        return None
    gpus = list_gpus()
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


def media_split_notice(config: Optional[dict] = None) -> Optional[str]:
    """A user-visible line for a box with a configured GPU split, saying media runs on
    ONE card and why, or ``None`` when there is nothing to say.

    The user explicitly configured a split and is NOT getting it for media, so this is
    a capability shortfall they must be told about, not a debug detail. The house
    precedent for exactly this shape is the partial-GPU-offload console notice in
    ``inference/backends/llamacpp/_sizing.py``. Deliberately not ``logger.debug``: the
    always-on ring buffer is INFO+, so a debug line is invisible without --debug and
    would never reach a bug report.

    Returns None on a single-GPU / unsplit box so a user who configured nothing is
    never nagged about a shortfall that does not exist.
    """
    from localm.config import load_config
    from localm.discover import applied_split_device_count, resolve_preferred_device
    cfg = config if config is not None else load_config()
    # applied_split_device_count (loader truth), NOT split_device_count: the user
    # CONFIGURED this split and it IS active for chat/embeddings, so the notice must
    # fire even on the vulkan build, where the detected count wrongly collapses to
    # < 2 and would suppress the notice entirely (GPU-SPLIT-VKINDEX) - the exact
    # user-visible Q2 miss this fix closes.
    n = applied_split_device_count(cfg)
    if n < 2:
        return None
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
            from localm.discover import vram_capacity
            return vram_capacity().get("free")
    return should_swap_for_media(read_free(), embedder_estimate_bytes, policy=policy)


def evict_chat_for_embedder(*, timeout_s: float = 300.0) -> str:
    """Free every loaded chat engine so the shared embedder's own load has VRAM
    room, through the SAME guarded path the server's "Unload all" button uses
    (``http_server.unload_all_models``).

    The embedder loads in-process (not as a background job with a self_url), so
    unlike ``unload_chat_for_media`` this does not round-trip over HTTP: it
    mirrors ``jobs/runner.py``'s ``_evict_shared_engine_for_media`` instead,
    submitting the coroutine onto the server's captured event loop
    (``http_server._server_loop``) via ``asyncio.run_coroutine_threadsafe`` and
    blocking THIS (caller's) thread on the result. That is required, not
    optional: ``unload_all_models`` honors the in-flight-request pin and
    serializes with ``get_engine`` only when it runs ON the loop - a raw
    off-loop unload would race a concurrent request the way the pin-during-
    unload fix (BUG-9b) already had to close for the media path.

    Returns the unload status ("unloaded" / "in_use" / "already_unloaded"), or
    "skipped" when the server loop is unreachable (no live server - e.g. a bare
    CLI embed) / "error" on a failure to complete - in which case the embedder
    loads alongside whatever chat model is resident (degraded, logged, never a
    crash - AGENTS.md rule 5: every degrade is surfaced, not silent)."""
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
    """``(free_bytes, fresh)``: a free-VRAM reading, plus whether the probe behind
    it was fresh (list_gpus() completed inside its deadline) rather than a
    timed-out/busy fallback to a stale last-known-good value.

    Used for BOTH ends of the before/after delta (hence the name: it was
    _vram_before_reading when only the 'before' end checked freshness). The
    'after' end needs it just as much - see _live_free_vram_bytes below.

    A caller that reports vram_before_bytes/vram_after_bytes/vram_freed to
    the user must know this: presenting a stale fallback as the CURRENT state
    is exactly the rule-5 gap a release-verify pass caught - /v1/models/unload
    reported byte-identical before/after readings, and vram_freed: false,
    across two calls made minutes apart with genuinely different real GPU
    states (confirmed stale via an OS-level VRAM counter showing the actual
    free/use cycle worked correctly the whole time).

    Defensive against a test double that patches vram_capacity()/vram_info()/
    list_gpus() with a plain no-kwarg callable or a fixed dict return_value
    (neither accepts/implements return_status): such a double is treated as
    "status unknown, assume fresh" (permissive, matching this reading's
    behavior before return_status existed) rather than raising on a rejected
    kwarg or an unexpected shape - only a REAL probe chain needs to prove
    itself stale."""
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
    return info.get("free"), fresh


def _live_free_vram_bytes():
    """Free VRAM in bytes from a FRESH probe, or None when the reading is not
    live - the reader wait_for_vram_release() polls for the 'after' end.

    Checking freshness on the 'before' end alone is not enough, and the gap is
    not theoretical: the before-probe is itself what WARMS the driver, and the
    native unload running between the two reads is exactly the kind of work that
    wedges it. Once any probe overruns, discover._gpu_probe_inflight stays True
    until the abandoned thread returns, so every subsequent poll is served the
    frozen last-known-good value - which the before-read just refreshed. So
    after == before, no rise, and the endpoint concluded "vram_freed": false with
    before_fresh=True and therefore NO uncertainty flag at all: the original lie,
    reproduced over real HTTP against the previous fix. Returning None here makes
    that unmeasurable rather than falsely equal (AGENTS.md rule 5).

    None means "could not measure now", never a fabricated 0.

    NOT every free-VRAM caller is migrated to this, deliberately. Ones weighing
    CAPACITY rather than reporting a completed action (the fit badges, the swap
    gates, the eviction ceiling) still call vram_capacity() directly: a stale
    ceiling costs an over- or under-swap, while refusing to decide at all would
    break a working box every time the probe merely ran slow. What "unknown"
    should MEAN for a decision gate is a real design question, and it is not
    answered here.

    Known gaps that ARE this same defect and are recorded rather than quietly
    half-fixed - each one QUOTES a free figure that can come from a timed-out
    probe, so each needs that design answer first:
      - http_server.switch_engine's 503 refusals ("Not enough VRAM to load 'x'
        (need ~N MB, M MB free)", and the split-shortfall variant beside it);
      - sysstats.vram_capacity() -> GET /api/stats -> the GUI status bar;
      - gui/routes/models.py's /api/vram-estimate -> the Settings free-VRAM line.
    None of these is self-correcting: once a probe overruns,
    discover._gpu_probe_inflight stays True until the abandoned thread returns,
    so every later call keeps serving the same frozen value."""
    free, fresh = _vram_free_reading()
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
    "VRAM did not drop" - and a poll that could not take a reading does not support
    it. It matters because ``read_free`` is normally backed by the deadline-bounded
    GPU probe (``discover.list_gpus``), whose status-aware callers pass None here
    the moment a probe stops answering and starts serving a FROZEN last-known-good
    value. Without this branch that frozen value equals ``before_bytes``, shows no
    rise, and reports a confident "no VRAM was freed" while VRAM was in fact freed
    (AGENTS.md rule 5). None reuses this function's already-documented "cannot
    verify" meaning rather than inventing a fourth outcome.

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
        from localm.selfclient import self_request
        resp = self_request("POST", "/models/unload", timeout=300, base_url=self_url)
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
        uncertain = bool(data.get("vram_reading_uncertain"))
        if uncertain:
            # The server flagged its own reading as possibly stale, so neither a
            # GB figure nor "has not dropped yet" is a claim we may pass on to the
            # user as fact - both would be arithmetic on a number the server just
            # said it could not stand behind. This branch is checked BEFORE them
            # for exactly that reason (rule 5).
            job.push({"type": "line", "text":
                      "Chat model unloaded - could not confirm how much VRAM was "
                      "freed (no live GPU reading) - continuing."})
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
        from localm.selfclient import self_request
        resp = self_request("POST", "/models/load", timeout=300, base_url=self_url)
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
