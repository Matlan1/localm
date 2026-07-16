# SPDX-License-Identifier: AGPL-3.0-or-later
"""
In-app model discovery: search HuggingFace for GGUF models and judge,
per quantization, whether a file fits this machine's VRAM.

Discovery is a user-initiated prelude to ``localm pull`` and sits in the same
policy category (explicit user action - see docs/network.md): it is not
routed through the net_allow/net_deny domain rules, but ``net_mode = off``
still blocks it, so the one kill switch keeps its promise.

"Fits your VRAM" badges compare against TOTAL VRAM, not currently-free VRAM:
the active chat model occupies the GPU while you browse, and it will be
unloaded before the new one loads. The estimate mirrors the GGUF backend's
preflight: weights + ~1.5 GB overhead for KV cache and compute buffers.
"""

from __future__ import annotations

import ctypes
import inspect
import re
import threading
import time
from collections.abc import Sequence
from typing import Optional

from localm.debuglog import logger

HF_API = "https://huggingface.co"
_TIMEOUT = 20

# HuggingFace library-tag filter for each discoverable model format. GGUF repos
# carry the "gguf" tag; transformers-native (safetensors / pytorch) repos carry
# the "transformers" library tag, which is exactly the set localm's HF backend
# loads. Both are real HF /api/models filter values, so classification comes from
# WHICH query a repo answered, not from parsing per-result tag fields the list
# response may omit.
_FORMAT_FILTER = {"gguf": "gguf", "hf": "transformers"}

# Single-sourced from localm.vram, the same overhead (KV cache + compute buffers)
# and weight safety factor GgufBackend._check_vram / sysstats.estimate_vram use,
# so the fit badge and the loader agree on "does it fit". Kept as module-level
# names so the value has one home while fit_label still reads a local constant.
from localm.vram import VRAM_OVERHEAD_BYTES as _OVERHEAD_BYTES
from localm.vram import VRAM_WEIGHT_FACTOR as _WEIGHT_FACTOR

# Quantization label inside a GGUF filename, e.g. Q4_K_M, Q8_0, IQ4_XS,
# Q6_K, F16, BF16. Matched case-insensitively on word-ish boundaries.
_QUANT_RE = re.compile(
    r"(?i)(?<![A-Z0-9])(IQ\d+_[A-Z0-9]+|Q\d+_K(?:_[SML])?|Q\d+_\d+"
    r"|BF16|F16|F32|FP16|FP32)(?![A-Z0-9])")

# Split GGUF naming: model-00001-of-00003.gguf
_SPLIT_RE = re.compile(r"^(?P<stem>.+)-(?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$",
                       re.IGNORECASE)


class DiscoverError(Exception):
    """Discovery failed - network off, HF unreachable, or repo unusable.
    Messages are safe to show in the GUI."""


def _ensure_online() -> None:
    from localm.netpolicy import network_mode
    if network_mode() == "off":
        raise DiscoverError(
            "Network access is disabled (net_mode=off). Enable it with: "
            "localm config net_mode ask")


def _get(url: str, params: Optional[dict] = None) -> object:
    """Policy-checked GET returning parsed JSON.

    Routes through ``netpolicy.safe_fetch_bytes`` so the request is pinned to the
    validated IP and EVERY redirect hop is re-checked against the network policy
    (SSRF-REBIND): a DNS-rebind of the HF host, or a redirect from the HF API,
    cannot bounce discovery into a loopback / link-local / private address. This
    is the same protection the model-pull path already uses; a raw ``requests.get``
    here previously bypassed it (an owner-initiated fetch, so low severity, but the
    inconsistency is closed)."""
    import json as _json
    import urllib.parse

    from localm import netpolicy
    # doseq=True so a list-valued param (expand[]=safetensors&expand[]=downloads
    # ...) encodes as repeated keys, which is how the HF models API takes expand.
    full = url + ("?" + urllib.parse.urlencode(params, doseq=True) if params else "")
    try:
        _final, _ctype, body = netpolicy.safe_fetch_bytes(
            full, max_bytes=32 * 1024 * 1024, timeout=int(_TIMEOUT))
        return _json.loads(body.decode("utf-8"))
    except Exception as e:
        raise DiscoverError(f"HuggingFace request failed: {e}")


def hf_param_bytes(safetensors: Optional[dict]) -> Optional[int]:
    """Estimated GPU weight footprint in bytes for an HF model, from its
    safetensors param metadata (the ``safetensors`` expand field of the HF models
    API: ``{"total": <param count>, "parameters": {...}}``).

    localm's HF backend loads in bf16 on GPU with no on-load quantization (see
    inference/backends/hf.py), so the footprint is ``total_params * 2`` regardless
    of the STORED dtype - the loader casts to bf16. This is the weight size only;
    fit_label() adds KV-cache / compute overhead, the same way it does for a GGUF
    file size. Returns None when the repo has no usable param count so the GUI can
    show "size unknown" rather than a guessed badge (do-not-hide-problems: an
    unknown is surfaced as unknown, not silently treated as zero/fits)."""
    if not isinstance(safetensors, dict):
        return None
    total = safetensors.get("total")
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        return None
    return total * 2


def _search_one(fmt: str, query: str, limit: int) -> list[dict]:
    """One HF /api/models query for a single format, each item tagged with it."""
    params: dict = {
        "filter": _FORMAT_FILTER[fmt],
        "sort": "downloads",
        "direction": "-1",
        "limit": str(limit),
    }
    if query.strip():
        params["search"] = query.strip()
    if fmt == "hf":
        # Expand the safetensors param metadata so each result carries a param
        # count we can turn into a VRAM fit estimate inline (no per-repo tree
        # fetch). `expand` is restrictive - it drops the default stat fields - so
        # re-request downloads/likes/lastModified alongside it.
        params["expand[]"] = ["safetensors", "downloads", "likes", "lastModified"]
    data = _get(f"{HF_API}/api/models", params)
    if not isinstance(data, list):
        raise DiscoverError("Unexpected response from HuggingFace search")
    out = []
    for item in data[:limit]:
        repo = item.get("id") or item.get("modelId")
        if not repo:
            continue
        row = {
            "id": repo,
            "downloads": item.get("downloads", 0),
            "likes": item.get("likes", 0),
            "updated": item.get("lastModified", ""),
            "formats": [fmt],
        }
        if fmt == "hf":
            # bf16 weight footprint from the param count, or None when HF has no
            # safetensors metadata (the row then shows "size unknown").
            row["size_bytes"] = hf_param_bytes(item.get("safetensors"))
        out.append(row)
    return out


def hf_search(query: str = "", limit: int = 20,
              formats: Sequence[str] = ("gguf",)) -> list[dict]:
    """Search HF for model repos in the requested *formats* (a subset of
    {"gguf", "hf"}). Empty query = most downloaded.

    One HF query runs per requested format (gguf -> the bundled GGUF backend,
    hf -> the transformers backend); the results are merged de-duped by repo id
    (a repo that surfaces under both formats keeps a merged ``formats`` list),
    sorted by downloads, and trimmed to *limit*.

    Returns [{id, downloads, likes, updated, formats}]. Defaults to gguf-only so
    the CLI ``localm search`` is unchanged; the GUI passes both formats
    explicitly from its toggles."""
    _ensure_online()
    limit = max(1, min(int(limit), 50))
    wanted = [f for f in formats if f in _FORMAT_FILTER]
    if not wanted:
        raise DiscoverError(
            "No valid model format requested (choose gguf and/or hf).")
    # One list per format (each already download-sorted by the API), de-duped by
    # repo id: a repo that surfaces under both formats stays in the FIRST list it
    # appeared in and its tags are unioned there.
    seen: dict[str, dict] = {}
    per_format: list[list[dict]] = []
    for fmt in wanted:
        lst: list[dict] = []
        for item in _search_one(fmt, query, limit):
            existing = seen.get(item["id"])
            if existing:
                for f in item["formats"]:
                    if f not in existing["formats"]:
                        existing["formats"].append(f)
                # A repo in both formats enters via the FIRST (gguf) list, which
                # carried no size; keep its HF size estimate if the hf pass has one.
                if existing.get("size_bytes") is None and item.get("size_bytes") is not None:
                    existing["size_bytes"] = item["size_bytes"]
            else:
                seen[item["id"]] = item
                lst.append(item)
        per_format.append(lst)
    # Round-robin interleave by per-format rank, then trim to `limit`. A plain
    # merge-then-sort-by-downloads would let the higher-download format (HF repos
    # routinely dwarf GGUF repacks) crowd the other out of the top `limit`
    # entirely, so a "show GGUF" toggle could return zero GGUF. Interleaving keeps
    # every enabled format visible while still leading each with its most popular.
    out: list[dict] = []
    rank = 0
    while len(out) < limit and any(rank < len(lst) for lst in per_format):
        for lst in per_format:
            if rank < len(lst):
                out.append(lst[rank])
                if len(out) >= limit:
                    break
        rank += 1
    return out


def hf_backend_available() -> bool:
    """True when the HF/transformers runtime can actually RUN a model here: both
    torch and transformers are importable. Uses importlib.util.find_spec, a cheap
    capability probe with no heavy import side effect.

    When False, an HF (transformers-format) model can still be DOWNLOADED via
    pull - it simply cannot be loaded until the ``.[gpu]`` extra (torch +
    transformers) is installed. The GUI surfaces exactly that, and does NOT block
    the download (a user may only want the files)."""
    import importlib.util
    try:
        return bool(importlib.util.find_spec("torch")
                    and importlib.util.find_spec("transformers"))
    except (ImportError, ValueError):
        # find_spec can raise on a half-installed namespace package; treat an
        # unresolvable probe as "not available" rather than crash discovery.
        return False


def hf_gguf_files(repo: str) -> list[dict]:
    """
    List the GGUF files of *repo* with size and quant label. Split files
    (``-00001-of-0000N``) are grouped into one logical entry whose ``file``
    is the first part (what ``localm pull repo:file`` expects) and whose
    size is the sum of all parts. Sorted smallest-first.
    """
    _ensure_online()
    repo = repo.strip().strip("/")
    if not re.match(r"^[\w.-]+/[\w.-]+$", repo):
        raise DiscoverError(f"Not a HuggingFace repo id: {repo}")
    tree = _get(f"{HF_API}/api/models/{repo}/tree/main")
    if not isinstance(tree, list):
        raise DiscoverError(f"Unexpected tree response for {repo}")

    singles: list[dict] = []
    groups: dict[tuple, dict] = {}
    for entry in tree:
        path = entry.get("path", "")
        if not path.lower().endswith(".gguf"):
            continue
        size = entry.get("size") or (entry.get("lfs") or {}).get("size") or 0
        m = _SPLIT_RE.match(path)
        if m:
            key = (m.group("stem").lower(), m.group("total"))
            g = groups.setdefault(key, {
                "file": None, "size_bytes": 0, "n_parts": 0,
                "quant": _quant_of(m.group("stem")),
            })
            g["size_bytes"] += size
            g["n_parts"] += 1
            if m.group("part") == "00001":
                g["file"] = path
        else:
            singles.append({
                "file": path,
                "quant": _quant_of(path),
                "size_bytes": size,
                "n_parts": 1,
            })

    files = singles + [g for g in groups.values() if g["file"]]
    if not files:
        raise DiscoverError(
            f"{repo} has no GGUF files. It may be a transformers-format "
            f"repo - pull it whole with:  localm pull {repo}")
    files.sort(key=lambda f: f["size_bytes"])
    return files


def _quant_of(name: str) -> str:
    m = _QUANT_RE.search(name)
    return m.group(1).upper() if m else ""


# ---- GPU probe safety: a hardware probe must never block its caller -------- #
# _list_gpus_probe() calls the GPU driver: torch.cuda.mem_get_info (which, on a
# torch ROCm build, calls into HIP) has NO timeout, and nvidia-smi is a
# subprocess. A busy or wedged driver call would block the CALLER for as long as
# the driver takes. Several callers run on the server's single asyncio event loop
# (GET /api/gpus, GET /api/vram-estimate, the GPU-registry heartbeat), so a stuck
# probe there freezes the WHOLE WebUI while the machine sits idle (diagnosed
# 2026-07). The public list_gpus() below makes the probe safe by construction, so
# NO caller can be frozen regardless of whether the call site remembered to
# offload it: the probe runs on a helper thread with a hard DEADLINE; if it
# overruns, the caller gets the last-known-good reading (or []) and moves on. A
# wedged NATIVE call cannot be interrupted from Python, so that one helper thread
# is abandoned; the in-flight guard means at most ONE such thread ever exists,
# and the overrun is surfaced at debug level (AGENTS.md rule 5), never silently
# eaten.
#
# NOTE - deliberately NO freshness/TTL cache: every call re-probes. A TTL cache
# would hand a STALE "free" reading to callers that need a live one, most
# critically switch_engine's eviction loop, whose wait_for_vram_release polls
# free-VRAM to confirm a native free has landed before re-checking (AUDIT-MED-11);
# a stale value there would defeat that guard and over-evict. The last-known-good
# value is kept ONLY as the wedge fallback, never to short-circuit a live probe.
_GPU_PROBE_DEADLINE = 4.0     # seconds: hard cap a single probe may block a caller
# A BLOCKING, non-event-loop caller (the one-shot `localm gpus` CLI) is not on the
# server's single asyncio loop, so it can afford to wait out a slow COLD driver
# init instead of misreporting it as "no GPU": the FIRST torch.cuda / HIP call
# initializes the ROCm/CUDA driver and was measured at ~6.5s on a cold box,
# overrunning the 4s server cap. This longer deadline is ONLY for such blocking
# callers; the server-loop path keeps _GPU_PROBE_DEADLINE so a wedged driver can
# never freeze the WebUI (PR #541).
_GPU_PROBE_CLI_DEADLINE = 15.0

# Outcome of a probe, surfaced by list_gpus(..., return_status=True) so a caller
# can tell a slow / timed-out probe apart from a genuine "nothing here" reading
# and not misattribute the former (AGENTS.md rule 5). A user-facing "no GPU"
# message MUST branch on this.
GPU_PROBE_OK = "ok"            # a fresh probe completed - an empty list means genuinely none
GPU_PROBE_TIMEOUT = "timeout"  # probe exceeded the deadline (cold driver init / wedge); INCONCLUSIVE
GPU_PROBE_BUSY = "busy"        # another probe is inflight, or the probe thread could not start

_gpu_probe_lock = threading.Lock()
_gpu_last_good: Optional[list] = None    # last SUCCESSFUL probe; served on a wedge
_gpu_probe_inflight = False
# Published TOGETHER with _gpu_probe_inflight (under _gpu_probe_lock) when a probe
# thread is started, and cleared when it lands, so a PATIENT off-loop caller can
# JOIN the running probe instead of being handed an instant GPU_PROBE_BUSY. The
# default guard behaviour is still to refuse to pile a second probe on a driver
# already being probed (BUSY) - that instant answer is what keeps the WebUI
# responsive on a permanent wedge. Joining is strictly opt-in (list_gpus'
# wait_for_inflight), for a caller already OFF the event loop that can afford to
# wait out a cold driver init: it is the ONLY thing that lets a long deadline
# actually help on a cold box, because there the first probe (often a 4s
# event-loop heartbeat) holds the in-flight slot for the whole ~4.6s cold init,
# so every other caller in that window would otherwise short-circuit on BUSY
# without ever probing - the exact 0.0000s no-op a deadline-15 RETRY hits.
_gpu_probe_done: Optional[threading.Event] = None
_gpu_probe_result: Optional[dict] = None
# Bumped by _reset_gpu_probe_cache() to ORPHAN any probe thread still in flight.
# An abandoned probe (see the DEADLINE note above) is by definition still running
# and will write its reading whenever the native call finally returns - which can
# be long after the reset. Clearing the globals alone cannot prevent that write,
# so a stale thread's result is fenced out by epoch instead of raced against.
_gpu_probe_epoch = 0


def _reset_gpu_probe_cache() -> None:
    """Test hook: drop the last-known-good GPU reading + in-flight flag, and
    INVALIDATE any probe still in flight so it cannot bleed into the next test.

    Clearing the globals is not enough on its own: an overrunning probe is
    abandoned, not cancelled (a wedged native call cannot be interrupted from
    Python), so that thread outlives this reset and would otherwise write its
    reading into _gpu_last_good AFTERWARDS. Measured: a cold ROCm/CUDA init takes
    ~6.5s and overruns _GPU_PROBE_DEADLINE, so the abandoned thread lands its
    write several seconds later, inside whichever test is running by then - which
    made the GPU tests fail intermittently with THIS machine's real card where
    they assert a fake or empty reading. Bumping the epoch makes that late write
    a no-op (see _run), which the clears alone provably could not do."""
    global _gpu_last_good, _gpu_probe_inflight, _gpu_probe_epoch
    global _gpu_probe_done, _gpu_probe_result
    with _gpu_probe_lock:
        _gpu_last_good = None
        _gpu_probe_inflight = False
        # Unpublish the join handles too: after a reset the slot reads free, so no
        # caller should join a probe from the epoch just retired. An abandoned
        # thread still holding its own local done/result is unaffected (it sets its
        # local event and, epoch-mismatched, will not touch these globals again).
        _gpu_probe_done = None
        _gpu_probe_result = None
        _gpu_probe_epoch += 1


def list_gpus(*, deadline: float = _GPU_PROBE_DEADLINE, return_status: bool = False,
              wait_for_inflight: bool = False):
    """Every GPU device visible right now: ``[{"index", "name", "total",
    "free"}, ...]``, or ``[]`` when nothing is measurable.

    Safe by construction: the real driver probe (:func:`_list_gpus_probe`) runs on
    a helper thread with a hard ``deadline``-second timeout, so this call NEVER
    blocks its caller for longer than ``deadline`` even if the GPU driver wedges -
    critical because several callers run on the server's single event loop. Every
    call re-probes (see the module note above: no TTL cache, so a live "free"
    reading is never stale); on an overrun the last-known-good value (or ``[]``) is
    returned and the stuck probe thread is abandoned. ``deadline`` is overridable
    for tests and for blocking (non-loop) callers that can wait out a slow cold
    driver init (:data:`_GPU_PROBE_CLI_DEADLINE`).

    When ``return_status`` is True, returns ``(gpus, status)`` where ``status`` is
    :data:`GPU_PROBE_OK` (a fresh probe completed - an empty ``gpus`` then means
    genuinely no measurable GPU), :data:`GPU_PROBE_TIMEOUT` (the probe exceeded
    ``deadline`` - typically a cold ROCm/CUDA driver init that has not finished, so
    an empty ``gpus`` is INCONCLUSIVE and a retry with a longer deadline may
    succeed), or :data:`GPU_PROBE_BUSY` (another probe is already inflight or the
    probe thread could not start; no fresh reading was taken). A caller that
    renders a user-facing "no GPU" message MUST branch on this so a slow cold probe
    is not misreported as "no torch / no GPU" (AGENTS.md rule 5). ``return_status``
    defaults to False, preserving the bare-list contract every existing caller and
    the ~28 test modules that patch this function rely on.

    Tries torch first (CUDA/ROCm - torch's ROCm build aliases torch.cuda.* to
    HIP under the hood, so an AMD card enumerates through the exact same API,
    no special-casing needed) since it also gives a device name; falls back to
    a name-aware ``nvidia-smi`` listing (ALL devices, not just the first) for
    the GGUF-only install that has no torch.

    ``wait_for_inflight`` (opt-in, default False) changes ONLY what happens when a
    probe is already in flight: instead of returning :data:`GPU_PROBE_BUSY` at once
    with the last-known-good reading, this call JOINS the running probe and waits on
    its completion, bounded by its own ``deadline``. This is what makes a longer
    ``deadline`` actually help on a cold box: there the FIRST probe (typically a 4s
    event-loop heartbeat via /api/stats) holds the in-flight slot for the entire
    ~4.6s cold ROCm/CUDA init, so a model-load probe arriving in that window would
    otherwise short-circuit on BUSY without ever probing - the identical 0.0000s
    no-op a deadline-15 RETRY hits on the same guard. Set it ONLY together with a
    long ``deadline`` and ONLY off the event loop: like the long deadline itself, a
    joining wait can block the caller up to ``deadline`` seconds, which must never
    land on the server's single loop (PR #541). It never spawns a second probe, so
    it cannot pile onto a wedged driver; a permanent wedge still just times the
    joiner out at its own ``deadline``.

    Deliberately does NOT fall back to the Windows display-adapter registry:
    that tier (see vram_info()) can only report one aggregate "largest
    adapter" number with no per-device identity, so it cannot support GPU
    *selection* - only vram_info()'s single-number "total VRAM for fit
    badges" use case. That is a scope boundary, not an oversight."""
    gpus, status = _list_gpus_with_status(deadline, wait_for_inflight)
    return (gpus, status) if return_status else gpus


def _list_gpus_with_status(deadline: float, wait_for_inflight: bool = False) -> tuple:
    """The real probe driver behind :func:`list_gpus`, returning ``(gpus, status)``
    where status is one of :data:`GPU_PROBE_OK` / :data:`GPU_PROBE_TIMEOUT` /
    :data:`GPU_PROBE_BUSY`. Split out so ``list_gpus`` can expose the status opt-in
    without duplicating the thread + deadline machinery. ``wait_for_inflight``: see
    :func:`list_gpus` - a patient off-loop caller JOINS a probe already in flight
    (bounded by ``deadline``) rather than short-circuiting on BUSY."""
    global _gpu_last_good, _gpu_probe_inflight, _gpu_probe_done, _gpu_probe_result
    global _probe_deadline_at   # published with the slot for #697's cold-budget check
    join_done = None
    join_result = None
    with _gpu_probe_lock:
        if _gpu_probe_inflight:
            if wait_for_inflight and _gpu_probe_done is not None:
                # A patient off-loop caller. Rather than pile a second probe on a
                # driver already being probed (or be handed an instant BUSY for a
                # reading that is on its way), JOIN the in-flight probe: wait on ITS
                # completion event, bounded by our own deadline. Both handles are
                # captured HERE, under the same lock that observed the in-flight
                # slot, so a concurrent completion cannot null them between this
                # check and the wait below.
                join_done = _gpu_probe_done
                join_result = _gpu_probe_result
            else:
                # Default: never pile on. Hand back the last-known-good reading so
                # this caller stays free. No fresh reading was taken, so the status
                # is BUSY (not a clean OK); [] is the safe "unknown" answer when
                # nothing has succeeded yet. This instant answer is what keeps the
                # event loop responsive on a permanent wedge.
                served = list(_gpu_last_good) if _gpu_last_good is not None else []
                return served, GPU_PROBE_BUSY
        else:
            _gpu_probe_inflight = True
            # Published under the SAME lock that claims the in-flight slot (only one
            # probe is ever in flight, so there is only one deadline to describe):
            # #697's probe body reads it to decide whether it can afford a cold
            # device-global VRAM source without overrunning this deadline. See
            # _apply_device_global_free.
            _probe_deadline_at = time.monotonic() + deadline
            # Captured under the SAME lock that claims the in-flight slot: an
            # unlocked read here could pair this probe with an epoch a concurrent
            # reset has already retired, which is the exact race the epoch exists to
            # close.
            my_epoch = _gpu_probe_epoch
            # Created and PUBLISHED under the lock, atomically with the in-flight
            # slot, so any joiner that sees inflight=True also sees these handles
            # (never a half-published state). Cleared by _run when the probe lands.
            result: dict = {}
            done = threading.Event()
            _gpu_probe_done = done
            _gpu_probe_result = result

    # JOIN path: we did not start a probe; wait on the one already running.
    if join_done is not None:
        if join_done.wait(deadline):
            # The joined probe landed. Its result carries a "value" key iff its
            # thread actually ran to completion; the key is ABSENT only when the
            # starting caller could not spawn the thread and woke joiners via
            # done.set() so they would not hang - that is a BUSY, not a fresh OK.
            if "value" in join_result:
                v = join_result["value"]
                return (list(v) if v is not None else []), GPU_PROBE_OK
            with _gpu_probe_lock:
                served = list(_gpu_last_good) if _gpu_last_good is not None else []
            return served, GPU_PROBE_BUSY
        # Our own deadline expired while waiting on the in-flight probe: same
        # outcome as starting one that overran - the driver is stuck, serve
        # last-known-good and report TIMEOUT (never mistaken for "no GPU"). We
        # spawned nothing, so this never piled onto the wedge.
        logger.debug("list_gpus: waited %.1fs on an in-flight GPU probe that did "
                     "not complete; returning last-known GPU info", deadline)
        with _gpu_probe_lock:
            served = list(_gpu_last_good) if _gpu_last_good is not None else []
            return served, GPU_PROBE_TIMEOUT

    # START path: we own the in-flight slot.
    def _run() -> None:
        global _gpu_last_good, _gpu_probe_inflight, _gpu_probe_done, _gpu_probe_result
        global _probe_deadline_at
        value = None
        try:
            value = _list_gpus_probe()
        except Exception as e:   # the probe swallows its own errors; belt-and-braces
            logger.debug("list_gpus: probe raised unexpectedly: %s", e)
        with _gpu_probe_lock:
            if _gpu_probe_epoch != my_epoch:
                # A reset retired this probe while it ran: its reading describes a
                # state the owner has explicitly dropped, and the in-flight slot is
                # no longer ours to clear (a later probe may already own it). Drop
                # BOTH writes rather than corrupt the current epoch's state.
                # Surfaced, not silenced (AGENTS.md rule 5): debug is the right
                # altitude because this is the deliberate consequence of a reset,
                # not a fault.
                logger.debug("list_gpus: discarding probe result from retired "
                             "epoch %s (current %s)", my_epoch, _gpu_probe_epoch)
            else:
                if value is not None:
                    _gpu_last_good = value
                _gpu_probe_inflight = False
                # Unpublish alongside the in-flight slot: a NEW caller must start a
                # fresh probe, not join one that has already landed. A joiner that
                # already captured its local handle is unaffected - it waits on that
                # same `done`, which is set unconditionally just below.
                _gpu_probe_done = None
                _gpu_probe_result = None
                # Cleared with the in-flight slot too (#697): the budget describes
                # THIS probe and nothing else. Leaving it set would hand a later
                # reader an expired deadline, which reads as "no budget left" and
                # would skip a cold source that in fact had all the time in the world.
                _probe_deadline_at = None
        # Deliberately OUTSIDE the epoch gate and unconditional: a caller still
        # inside its deadline - the starter OR any joiner - is waiting on `done`,
        # and withholding it would make it wait out the full deadline and report a
        # COMPLETED probe as a TIMEOUT, manufacturing the very "no GPU"/inconclusive
        # lie the status contract above exists to prevent.
        result["value"] = value
        done.set()

    try:
        threading.Thread(target=_run, name="localm-gpu-probe", daemon=True).start()
    except Exception as e:
        # Could not spawn the probe thread (e.g. OS thread exhaustion). Reset the
        # in-flight guard so a LATER call can retry (never leave it stuck True with
        # no thread to clear it), surface it at debug (rule 5), and degrade to the
        # last-known-good reading rather than propagating a 500 to the caller. No
        # fresh reading was taken -> BUSY. Epoch-gated for the same reason as the
        # clear in _run: if a reset retired us, the slot is no longer ours and may
        # already belong to a newer probe we must not clear.
        with _gpu_probe_lock:
            if _gpu_probe_epoch == my_epoch:
                _gpu_probe_inflight = False
                _gpu_probe_done = None
                _gpu_probe_result = None
        # Wake any caller that joined between our publish above and this failure, so
        # it does not wait out its full deadline on a probe that will never run.
        # `result` has no "value" key (the thread never set it), which the join
        # path reads as BUSY - the honest status here.
        done.set()
        logger.debug("list_gpus: could not start probe thread: %s", e)
        served = list(_gpu_last_good) if _gpu_last_good is not None else []
        return served, GPU_PROBE_BUSY
    if done.wait(deadline):
        # Fresh probe finished in time: return ITS result (never a cached one).
        v = result.get("value")
        return (list(v) if v is not None else []), GPU_PROBE_OK
    # Deadline exceeded: the driver call is stuck in native code and cannot be
    # cancelled. Serve the last-known-good value and let the abandoned thread
    # finish (or never); _gpu_probe_inflight stays True until it does, so a wedge
    # spawns no further threads. Surfaced, not silenced (rule 5). The status is
    # TIMEOUT so a caller does not mistake an inconclusive probe for "no GPU".
    logger.debug("list_gpus: GPU probe exceeded %.1fs deadline (driver call stuck); "
                 "returning last-known GPU info so the caller does not block", deadline)
    with _gpu_probe_lock:
        served = list(_gpu_last_good) if _gpu_last_good is not None else []
        return served, GPU_PROBE_TIMEOUT


def _list_gpus_probe() -> list:
    """The actual (blocking) GPU driver probe. Call :func:`list_gpus`, not this -
    this one has no timeout and can wedge on a busy/broken driver."""
    try:
        import torch
        if torch.cuda.is_available():
            out = []
            for i in range(torch.cuda.device_count()):
                try:
                    free, total = torch.cuda.mem_get_info(i)
                except Exception:
                    continue   # one device failing to report never hides the rest
                try:
                    name = torch.cuda.get_device_name(i)
                except Exception:
                    name = f"GPU {i}"
                out.append({"index": i, "name": name,
                            "total": int(total), "free": int(free)})
            if out:
                _apply_device_global_free(out)
                return out
    except Exception:
        pass

    try:
        import subprocess
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if proc.returncode == 0 and proc.stdout.strip():
            out = []
            for line in proc.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                idx_s, name, total_mb, free_mb = parts[0], parts[1], parts[2], parts[3]
                try:
                    out.append({
                        "index": int(idx_s), "name": name,
                        "total": int(total_mb) * 1024 ** 2,
                        "free": int(free_mb) * 1024 ** 2,
                        # nvidia-smi's memory.free is the whole board's, across
                        # every process (that is what it exists to report), so
                        # unlike the torch path above it needs no correction.
                        "free_scope": FREE_SCOPE_DEVICE,
                    })
                except ValueError:
                    continue   # a malformed line never hides the rest
            if out:
                return out
    except Exception:
        pass
    return []


# How much of the world a GPU entry's "free" actually accounts for. A caller that
# presents free VRAM as CURRENT FACT (a "will it fit" refusal, a freed-bytes report)
# must know the difference; a caller that only wants a fit CEILING ("total") does not.
FREE_SCOPE_DEVICE = "device"    # every process's VRAM is counted - the number is the board's
FREE_SCOPE_PROCESS = "process"  # ONLY this process's own allocations are counted (see below)

# Probe budget below which a COLD (not-yet-opened) device-global source is skipped
# rather than risk overrunning the probe deadline. The cold open is MEASURED at
# ~750ms; this is that with margin, since overrunning costs the caller its free
# reading entirely. See _apply_device_global_free.
_CORRECTION_COLD_BUDGET_S = 1.5

# When the in-flight probe's deadline expires (monotonic), or None outside a probe.
# Set by _list_gpus_with_status under the same lock that claims the in-flight slot,
# so the probe body can tell how much of its budget is left before spending ~750ms
# on a cold source. Safe as a module global precisely because _gpu_probe_inflight
# serialises probes: only ever one in flight to describe.
_probe_deadline_at = None


def _apply_device_global_free(gpus: list) -> None:
    """Correct each entry's ``free`` to a DEVICE-GLOBAL figure where this platform's
    driver query is not one already, and tag every entry with ``free_scope`` so a
    caller can tell a whole-board number from a process-local one. Mutates *gpus*.

    WHY (measured, see dev-notes/vram-cross-process-blindness.md): on Windows with an
    AMD ROCm/HIP torch build, ``torch.cuda.mem_get_info`` reports
    ``total - the calling process's own allocations`` and is blind to every other
    process. Measured live: 0.14 GB reported "in use" while 10.53 GB genuinely was.
    That is not a staleness bug (PR #693's domain - the probe here is FRESH and still
    wrong), and it is not llama.cpp-specific: a plain torch tensor in a child process
    is equally invisible. It bites localm hard because every GGUF load is
    out-of-process (backends/gguf.py, since #606), so the model's own VRAM is ALWAYS
    in another process from the server measuring it - as is a game or a ComfyUI.

    On Linux, and on NVIDIA, the driver query is device-global BY DOCUMENTATION (CUDA
    specifies *free as "free according to the OS" and warns that another process can
    move it), so nothing is corrected there and the reading is tagged
    :data:`FREE_SCOPE_DEVICE` unchanged.

    When no better source can answer on Windows, the entry keeps the driver's number
    but is tagged :data:`FREE_SCOPE_PROCESS` rather than silently passing a
    known-process-local figure off as the board's (AGENTS.md rule 5). That tag is
    what makes /v1/models/unload say its reading is uncertain instead of asserting a
    wrong one as fact."""
    import sys
    if sys.platform != "win32":
        for g in gpus:
            g["free_scope"] = FREE_SCOPE_DEVICE
        return

    # The scope to use when a device-global correction is NOT available for an entry
    # (source cold-skipped, unmappable, or failed). Tag PROCESS only where the raw
    # reading is KNOWN blind (Windows + an AMD ROCm/HIP torch build); elsewhere on
    # Windows the raw cudaMemGetInfo is device-global by documentation (NVIDIA), so
    # tagging it PROCESS would assert a blindness never measured and raise a spurious
    # uncertainty flag on a number that is actually fine. Computed defensively up
    # front so it is defined on every path below, including the import-failure except.
    try:
        from localm.gpu_usage import raw_reading_is_process_scoped
        uncorrected_scope = (FREE_SCOPE_PROCESS if raw_reading_is_process_scoped()
                             else FREE_SCOPE_DEVICE)
    except Exception:
        # gpu_usage unimportable is a real bug, not an environment condition, but it
        # must not crash a probe. Conservative default: DEVICE - never assert a
        # blindness we cannot confirm.
        uncorrected_scope = FREE_SCOPE_DEVICE

    try:
        from localm.gpu_usage import device_global_used_bytes, source_is_warm
        # This runs INSIDE the deadline-bounded probe, so it spends the SAME budget
        # the driver call already spent. Opening the source costs ~750ms ONCE per
        # process (a driver init); a warm read costs ~0.02ms. Measured, that cold
        # 750ms was enough to push cold probes from a comfortable 2.9-3.5s to
        # 3.6-4.0s against the 4.0s cap and start timing them out - and a timeout
        # costs the caller its free reading ENTIRELY (list_gpus serves [] and
        # vram_info falls to the registry tier, which has no "free" at all). A
        # correct number is not worth trading for no number, so a COLD source is
        # skipped when the remaining budget is too thin to absorb it; the reading is
        # then tagged with the uncorrected scope instead of silently uncorrected. A
        # warm source is free and always runs, so a long-lived server pays this at
        # most once, and a CLI caller passing the longer _GPU_PROBE_CLI_DEADLINE has
        # room for it on the first go.
        if not source_is_warm():
            remaining = None
            if _probe_deadline_at is not None:
                remaining = _probe_deadline_at - time.monotonic()
            if remaining is not None and remaining < _CORRECTION_COLD_BUDGET_S:
                logger.debug(
                    "list_gpus: %.2fs left of the probe budget is too thin for a "
                    "cold device-global source (~%.1fs); leaving this reading "
                    "%s rather than risking a timeout that would return no free VRAM "
                    "at all", remaining, _CORRECTION_COLD_BUDGET_S, uncorrected_scope)
                for g in gpus:
                    g["free_scope"] = uncorrected_scope
                return
        used = device_global_used_bytes(gpus)
    except Exception as e:
        # Surfaced, not silenced: the entries below are then tagged with the
        # uncorrected scope (PROCESS only where the raw reading is known blind), so a
        # real blindness is reported without over-claiming one where it is not.
        logger.debug("list_gpus: device-global VRAM source failed: %s", e)
        used = {}
    for g in gpus:
        u = used.get(g.get("index"))
        if u is None:
            g["free_scope"] = uncorrected_scope
            continue
        total = int(g["total"])
        # Clamp: the used figure and `total` come from different sources (the driver's
        # total vs the adapter's dedicated usage), so their difference can land just
        # outside [0, total] without either being wrong enough to matter.
        g["free"] = max(0, min(total, total - int(u)))
        g["free_scope"] = FREE_SCOPE_DEVICE


def _native_backend_has_vulkan() -> bool:
    """True when the currently-resolved native runtime directory ships the
    Vulkan ggml backend (a ``ggml-vulkan.*`` file) - i.e. the active install
    is the ``vulkan`` build.

    Exists because ``list_gpus()`` (above) enumerates ONLY via torch.cuda
    (CUDA, or HIP under a ROCm-build torch) or nvidia-smi - it never calls the
    Vulkan loader, so it is structurally blind to any device only visible
    through Vulkan. On the vulkan build, the REAL device selection at load
    time happens entirely inside ggml-vulkan/llama.dll's own enumeration, a
    different index space list_gpus() cannot see or validate against (see
    GPU-SPLIT-VKINDEX: confirmed live to silently drop a valid configured
    split device and a valid configured main_gpu_index alike, because
    list_gpus() reported a non-empty but VULKAN-INCOMPLETE device list rather
    than an empty one - the two callers below already handle "empty" as
    "unmeasurable, pass through unchecked"; this handles "non-empty but for
    the wrong backend" the same way).

    Checks the actual shipped DLL/SO set, NOT the ``.localm-backend``
    provisioning marker (setup_llama.py): the marker can be absent (a
    ``--from`` build, an install predating the marker) or generic (e.g.
    ``"custom"`` for a ``--url``/``--sha256`` provision) - the real file set
    is always authoritative for which backend will actually be loaded."""
    try:
        from localm.inference.backends.llamacpp._loader import (
            runtime_binary_dir, _ggml_glob,
        )
        d = runtime_binary_dir()
        if d is None:
            return False
        return any("vulkan" in p.name.lower() for p in d.glob(_ggml_glob()))
    except Exception:
        return False


def resolve_main_gpu_index(configured, *, gpus: Optional[list] = None) -> int:
    """The GPU device index to actually use, given the user's ``main_gpu_index``
    config value.

    None (not configured) resolves to device 0 - today's behaviour - with no
    detection work done at all. An explicitly configured index is validated
    against the devices ``list_gpus()`` (or the injected *gpus*, for tests)
    currently sees: an index that does not match any of them is a real
    problem (silently substituting the wrong GPU, or handing llama.cpp's
    native loader an index past the end of its device array, is worse than
    device 0), so it is surfaced as a WARNING and swapped for device 0 rather
    than trusted blindly (rule 5, do-not-hide-problems).

    An index above ``_MAX_GPU_SPLIT_INDEX`` is rejected unconditionally, the
    same sanity ceiling :func:`resolve_gpu_split` applies to its indices -
    checked BEFORE any device-membership branching below, so it still applies
    when detection is unmeasurable or skipped (see next paragraph).

    When detection itself is unmeasurable (``list_gpus()`` returns nothing -
    no torch, no nvidia-smi) OR the active native backend is ``vulkan``
    (whose real device enumeration list_gpus() cannot see at all - see
    :func:`_native_backend_has_vulkan`), the configured index cannot be
    cross-checked against a reliable, backend-matching device list either
    way; it is passed through unchecked (aside from the ceiling above) rather
    than discarding an explicit user choice we have no way to disprove (the
    same documented boundary as the Windows-registry VRAM fallback)."""
    if configured is None:
        return 0
    try:
        idx = int(configured)
    except (TypeError, ValueError):
        logger.warning("main_gpu_index=%r is not a valid integer; using device 0",
                       configured)
        return 0
    if idx < 0:
        logger.warning("main_gpu_index=%d is negative; using device 0", idx)
        return 0
    if idx > _MAX_GPU_SPLIT_INDEX:
        logger.warning(
            "main_gpu_index=%d is above the sanity ceiling (%d); using device 0",
            idx, _MAX_GPU_SPLIT_INDEX)
        return 0
    if idx == 0:
        return 0   # the native default anyway - no need to enumerate devices
    if gpus is None:
        gpus = list_gpus()
    # Check membership by the "index" field, NOT list position: a device that
    # fails to report (list_gpus() skips it rather than hide the rest) leaves a
    # gap, so "idx < len(gpus)" alone could wrongly wave through an idx that
    # does not actually correspond to any detected device. Skipped entirely
    # when the active backend is vulkan (GPU-SPLIT-VKINDEX): list_gpus() is
    # blind to Vulkan-only devices, so a non-empty result here does not mean
    # it is authoritative for THIS backend's index space.
    if gpus and not _native_backend_has_vulkan() and not any(
            g.get("index") == idx for g in gpus):
        logger.warning(
            "main_gpu_index=%d does not match any of the %d GPU(s) detected "
            "right now; falling back to device 0", idx, len(gpus))
        return 0
    return idx


def apply_main_gpu(mp, *, config: Optional[dict] = None) -> None:
    """Set ``mp.main_gpu`` from the configured ``main_gpu_index``, validated via
    :func:`resolve_main_gpu_index`. Leaves the native default (0, set by
    ``llama_model_default_params()``) untouched when unset. Shared by the
    llama.cpp chat backend and the embedder so both native-load call sites
    honour the same selection with the same fallback/warning behaviour."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    configured = cfg.get("main_gpu_index")
    if configured is None:
        return
    mp.main_gpu = resolve_main_gpu_index(configured)


# llama.cpp's LLAMA_SPLIT_MODE_LAYER (see llamacpp/_structs.py / _abi.py's
# split_mode notes: 0=NONE/single-GPU, 1=LAYER, 2=ROW, 3=TENSOR). LAYER splits
# whole layers across devices proportional to tensor_split - the right default
# for "spread a too-big model over N cards" (as opposed to ROW/TENSOR, which
# split individual tensors and generally need fast inter-GPU interconnect to
# be worthwhile).
_LLAMA_SPLIT_MODE_LAYER = 1

# Fallback tensor_split array capacity when llama_max_devices() cannot be
# probed (an older build without the symbol). Matches LLAMA_MAX_DEVICES from
# the pre-dynamic-backend-registry era this build's own _structs.py docstring
# says it predates. tensor_split is a raw `const float*` with no length of its
# own, so under-allocating would be a genuine out-of-bounds read - this is a
# best-effort safety net, not a verified value (no multi-GPU hardware or
# provisioned native runtime was available to confirm it against the actual
# bundled build; see apply_gpu_split).
_TENSOR_SPLIT_FALLBACK_CAPACITY = 16

# Sanity ceiling for a gpu_split_indices entry - no real machine has anywhere
# near this many GPU devices, so an index above it is a config error, never a
# legitimate one. Bounds the ctypes tensor_split allocation apply_gpu_split
# eventually drives: without this, [0, 500000] would attempt a 500,001-element
# allocation before the native loader is ever invoked. settings_schema.py's
# MAX_GPU_SPLIT_INDEX applies the same value at config WRITE time; this is the
# independent check at READ time, so a hand-edited config.json that bypasses
# schema validation entirely is still bounded here.
#
# Also used by resolve_main_gpu_index below to bound a single main_gpu_index:
# the same sanity reasoning applies to one device index as to a list of them,
# and both values reach the identical ctypes.c_int32 main_gpu field.
_MAX_GPU_SPLIT_INDEX = 127


def resolve_gpu_split(configured_indices, configured_ratios=None, *,
                       gpus: Optional[list] = None) -> list:
    """Validate a configured multi-GPU split (``gpu_split_indices`` /
    ``gpu_split_ratios``) against the devices ``list_gpus()`` (or the injected
    *gpus*, for tests) currently sees, returning ``[(index, ratio), ...]``
    ready to write into ``tensor_split``.

    Mirrors :func:`resolve_main_gpu_index`'s posture: an index that does not
    match a currently-detected device is dropped with a WARNING rather than
    trusted blindly (rule 5, do-not-hide-problems) - a stale config
    referencing a since-removed GPU degrades to single-GPU instead of
    mis-targeting VRAM or crashing a load. Duplicate indices keep their first
    occurrence. Fewer than 2 valid indices after validation means "no split"
    (returns ``[]``) - the single-GPU path driven by ``apply_main_gpu`` is
    unaffected. This validation is SKIPPED (indices pass through unchecked)
    when the active native backend is ``vulkan`` - see
    :func:`_native_backend_has_vulkan` and GPU-SPLIT-VKINDEX: ``list_gpus()``
    cannot see Vulkan-only devices, so on that backend a non-empty result here
    is not authoritative and previously caused a live, confirmed bug (a
    configured split silently collapsed to single-device, with the user's
    ``gpu_split_ratios`` replaced by llama.cpp's own unrelated auto-split).

    ``configured_ratios``, when given, must be the SAME LENGTH as
    ``configured_indices`` (before validation) to be honoured - a length
    mismatch is a real misconfiguration (WARNED), not something to silently
    truncate/pad, so it falls back to an equal split across the surviving
    indices. ``None`` (or any non-positive entry) also means an equal split;
    llama.cpp treats tensor_split entries as relative proportions, not values
    that must sum to 1, so "equal" here is simply the same weight per device.
    """
    if not configured_indices:
        return []
    try:
        raw_indices = [int(i) for i in configured_indices]
    except (TypeError, ValueError):
        logger.warning(
            "gpu_split_indices=%r is not a list of integers; ignoring the "
            "split (single-GPU behavior)", configured_indices)
        return []
    if any(i < 0 for i in raw_indices):
        logger.warning(
            "gpu_split_indices=%r contains a negative index; ignoring the "
            "split (single-GPU behavior)", configured_indices)
        return []
    if any(i > _MAX_GPU_SPLIT_INDEX for i in raw_indices):
        logger.warning(
            "gpu_split_indices=%r contains an index above the sanity ceiling "
            "(%d); ignoring the split (single-GPU behavior)",
            configured_indices, _MAX_GPU_SPLIT_INDEX)
        return []

    seen: set = set()
    deduped: list = []
    for i in raw_indices:
        if i not in seen:
            seen.add(i)
            deduped.append(i)

    if gpus is None:
        gpus = list_gpus()
    if gpus and not _native_backend_has_vulkan():
        known = {g.get("index") for g in gpus}
        valid = [i for i in deduped if i in known]
        dropped = [i for i in deduped if i not in known]
        if dropped:
            logger.warning(
                "gpu_split_indices contains %d device(s) not currently "
                "detected (%s); dropping them from the split",
                len(dropped), dropped)
    else:
        # Detection unmeasurable (no torch, no nvidia-smi) OR the active
        # native backend is vulkan (GPU-SPLIT-VKINDEX - list_gpus() cannot see
        # Vulkan-only devices, so a non-empty result here would not be
        # authoritative for this backend's index space): same documented
        # boundary as resolve_main_gpu_index - cannot cross-check either way,
        # so the configured indices pass through rather than discarding an
        # explicit user choice we have no way to disprove.
        valid = deduped

    if len(valid) < 2:
        return []

    ratios: Optional[list] = None
    if configured_ratios:
        try:
            raw_ratios = [float(r) for r in configured_ratios]
        except (TypeError, ValueError):
            raw_ratios = None
        if raw_ratios is not None and any(r <= 0 for r in raw_ratios):
            raw_ratios = None
        if raw_ratios is not None and len(raw_ratios) == len(raw_indices):
            # Re-pair by ORIGINAL position so a ratio still lines up with its
            # index even when another index was dropped/de-duped above.
            by_index = dict(zip(raw_indices, raw_ratios))
            ratios = [by_index[i] for i in valid]
        else:
            logger.warning(
                "gpu_split_ratios (%d entries) does not match gpu_split_indices "
                "(%d entries); falling back to an equal split",
                len(configured_ratios), len(raw_indices))

    if ratios is None:
        ratios = [1.0] * len(valid)

    return list(zip(valid, ratios))


def _tensor_split_capacity(min_len: int) -> int:
    """Float-slot count to allocate for ``tensor_split``: the native loader's
    own answer when available (authoritative - see the capacity comment
    above), else the documented fallback. Never smaller than *min_len* (the
    caller's highest configured device index + 1)."""
    try:
        from localm.inference.backends.llamacpp import _api
        if _api.has_max_devices():
            return max(_api.llama_max_devices(), min_len)
    except Exception as e:
        logger.debug(
            "llama_max_devices() probe failed (%s); using the fallback "
            "tensor_split capacity", type(e).__name__)
    return max(_TENSOR_SPLIT_FALLBACK_CAPACITY, min_len)


def apply_gpu_split(mp, *, config: Optional[dict] = None):
    """Set ``mp.split_mode``/``mp.tensor_split`` from the configured
    ``gpu_split_indices``/``gpu_split_ratios``, validated via
    :func:`resolve_gpu_split`. Leaves native defaults (a single active GPU -
    whatever :func:`apply_main_gpu` already set) untouched when fewer than 2
    valid devices are configured. Shared by the llama.cpp chat backend and the
    embedder, same as ``apply_main_gpu``.

    Returns the ctypes float array backing ``mp.tensor_split`` (or ``None``
    when no split was applied) - the CALLER MUST keep this referenced until
    after the ``llama_load_model_from_file()`` call that consumes *mp*:
    llama.cpp copies ``tensor_split``'s contents at load time (it is not held
    as a live pointer afterward), so the buffer only needs to survive that one
    call, not the loaded model's lifetime."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    pairs = resolve_gpu_split(cfg.get("gpu_split_indices"), cfg.get("gpu_split_ratios"))
    if len(pairs) < 2:
        return None

    capacity = _tensor_split_capacity(max(idx for idx, _ in pairs) + 1)
    arr = (ctypes.c_float * capacity)()
    for idx, ratio in pairs:
        arr[idx] = ratio

    mp.tensor_split = ctypes.cast(arr, ctypes.c_void_p)
    mp.split_mode = _LLAMA_SPLIT_MODE_LAYER

    split_indices = {idx for idx, _ in pairs}
    if mp.main_gpu not in split_indices:
        new_main = pairs[0][0]
        logger.warning(
            "main_gpu_index=%d is not one of the configured gpu_split_indices "
            "%s; using device %d (the first split device) as the primary "
            "instead", mp.main_gpu, sorted(split_indices), new_main)
        mp.main_gpu = new_main

    return arr


def vram_info(*, return_status: bool = False):
    """{"total": bytes, "free"?: bytes} for the CONFIGURED main GPU device (see
    main_gpu_index / resolve_main_gpu_index), or the largest GPU when none is
    configured, or {} when not measurable. Tries torch (CUDA/ROCm) then
    nvidia-smi (both via list_gpus()), then the Windows display-adapter
    registry - the GGUF-only install has no torch, and the fit badges must
    still work there (total is all fit_label needs).

    When ``return_status`` is True, returns ``(info, status)`` where ``status``
    is list_gpus()'s own GPU_PROBE_OK/GPU_PROBE_TIMEOUT/GPU_PROBE_BUSY - a
    caller that will present a specific number as CURRENT FACT (not just a fit
    ceiling) must check this rather than trust a timed-out probe's stale
    last-known-good fallback (AGENTS.md rule 5; see the vram_before/after
    bytes this fed into /v1/models/unload, which is exactly that case).
    ``return_status`` defaults to False, preserving the plain-dict contract
    (AND the plain, no-kwarg list_gpus() call) every existing caller and test
    double relies on - the status-aware call is made ONLY when a caller opts
    in, never unconditionally."""
    from localm.config import load_config
    if return_status:
        gpus, status = list_gpus(return_status=True)
    else:
        gpus = list_gpus()
        status = None   # unused: _ret() never reads it when return_status=False

    def _ret(info: dict):
        return (info, status) if return_status else info

    if gpus:
        configured = load_config().get("main_gpu_index")
        idx = resolve_main_gpu_index(configured, gpus=gpus)
        # Look up by the "index" field, not list position (list_gpus() can
        # have a gap when one device fails to report - see
        # resolve_main_gpu_index); gpus[0] is a defensive fallback that should
        # not be reachable since resolve_main_gpu_index already validated idx.
        g = next((x for x in gpus if x.get("index") == idx), gpus[0])
        out = {"total": g["total"]}
        if g.get("free") is not None:
            out["free"] = g["free"]
            # Travels WITH the number it describes: a caller presenting free VRAM as
            # current fact must be able to tell a whole-board figure from a
            # process-local one (see _apply_device_global_free). Absent when free is.
            if g.get("free_scope") is not None:
                out["free_scope"] = g["free_scope"]
        return _ret(out)

    import sys
    if sys.platform == "win32":
        try:
            import winreg
            best = 0
            base = (r"SYSTEM\CurrentControlSet\Control\Class"
                    r"\{4d36e968-e325-11ce-bfc1-08002be10318}")
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
                i = 0
                while True:
                    try:
                        sub = winreg.EnumKey(root, i)
                    except OSError:
                        break
                    i += 1
                    if not sub.isdigit():
                        continue
                    try:
                        with winreg.OpenKey(root, sub) as key:
                            val, _typ = winreg.QueryValueEx(
                                key, "HardwareInformation.qwMemorySize")
                            if isinstance(val, int) and val > best:
                                best = val   # largest adapter wins (skip iGPU)
                    except OSError as e:
                        # Unexpected (vs the EnumKey end-of-list break above):
                        # access denied or a removed key. Surface under --debug
                        # so incomplete VRAM detection is diagnosable; the silent
                        # fallback is deliberate (a note beats crashing fit badges).
                        logger.debug("vram_info: registry subkey %s unreadable: %s",
                                     sub, e)
                        continue
            if best:
                return _ret({"total": int(best)})
        except Exception:
            pass
    return _ret({})


def vram_capacity(config: Optional[dict] = None, *, return_status: bool = False):
    """{"total": bytes, "free"?: bytes} to weigh a model's fit against - the
    right ceiling for any "will this model fit" decision (a pre-load refusal
    gate, a fit badge, a VRAM-estimate readout).

    ``vram_info()`` alone is single-GPU by design (see its docstring) and is
    the wrong ceiling once a multi-GPU ``gpu_split_indices`` is configured: a
    model too big for the single main GPU but that fits COMBINED across the
    configured split devices must not be refused or badged "too-big" just
    because the capacity check only ever looked at one device (the bug this
    function fixes - a model refused/mis-badged despite a working split).

    Sums ``total``/``free`` across every device in :func:`resolve_gpu_split`'s
    validated split (via :func:`list_gpus`) when 2+ valid devices are
    configured; ``free`` is included only when EVERY split device reports a
    measurable free value (mirrors vram_info()'s own all-or-nothing "free" key
    - a partially-measurable split must not silently under-count by treating a
    missing device's free as 0). Falls back to :func:`vram_info` untouched
    (single main-GPU number) whenever fewer than 2 valid split devices are
    configured or GPU detection is unmeasurable (registry-fallback tier) -
    resolve_gpu_split already warns and degrades a stale/invalid split to
    single-GPU (rule 5, do-not-hide-problems); this reuses that same
    validation rather than duplicating it.

    ``return_status``: see :func:`vram_info` - propagated through both the
    single-GPU short-circuit and the split-summed path, so a caller weighing
    whether to trust a specific number as CURRENT fact (not just a fit
    ceiling) can tell a fresh reading from a timed-out/stale one. Made ONLY
    when a caller opts in (never unconditionally), so every existing caller
    and test double that patches vram_info()/list_gpus() with a plain, no-kwarg
    stand-in keeps working exactly as before.
    """
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    # Cheap short-circuit for the common (no split configured) case, mirroring
    # resolve_gpu_split's own early return - skips a real hardware probe
    # (list_gpus() -> torch/nvidia-smi) on every request for the vast majority
    # of single-GPU installs that never configured a split.
    if not cfg.get("gpu_split_indices"):
        return vram_info(return_status=True) if return_status else vram_info()

    if return_status:
        gpus, status = list_gpus(return_status=True)
    else:
        gpus = list_gpus()
        status = None

    def _ret(info: dict):
        return (info, status) if return_status else info

    pairs = resolve_gpu_split(
        cfg.get("gpu_split_indices"), cfg.get("gpu_split_ratios"), gpus=gpus)
    by_index = {g.get("index"): g for g in gpus}
    split_gpus = [by_index[idx] for idx, _ in pairs if idx in by_index]
    if len(split_gpus) < 2:
        return vram_info(return_status=True) if return_status else vram_info()

    out = {"total": sum(g["total"] for g in split_gpus)}
    frees = [g.get("free") for g in split_gpus]
    if all(f is not None for f in frees):
        out["free"] = sum(frees)
        # All-or-nothing, mirroring the "free" key above: a sum is only a whole-board
        # figure if EVERY device in it is. One process-scoped device makes the whole
        # sum process-scoped, because that device's other-process VRAM is missing
        # from it. Absent entirely when NO device reported a scope: that means
        # UNKNOWN, and labelling it "process" would assert a blindness we have not
        # measured (as wrong as asserting the number is fact) while also breaking the
        # plain-dict contract every existing caller and test double relies on.
        scopes = [g.get("free_scope") for g in split_gpus if g.get("free_scope")]
        if scopes:
            out["free_scope"] = (FREE_SCOPE_DEVICE
                                 if all(s == FREE_SCOPE_DEVICE for s in scopes)
                                 else FREE_SCOPE_PROCESS)
    return _ret(out)


def split_device_count(config: Optional[dict] = None) -> int:
    """How many DETECTED devices the configured gpu_split resolves to - the
    DETECTED/labelling signal, NOT a load-safety gate.

    This is the exact signal ``vram_capacity()`` uses to decide whether its total
    is COMBINED across a split (>= 2) or the single main GPU (< 2): the same
    ``resolve_gpu_split`` + detected-device re-filter. Callers that LABEL a VRAM
    number ("combined across N GPUs" vs "your main GPU's") must gate on this, not
    on the raw ``gpu_split_indices`` length - a stale/typo'd index or a GGUF-only
    box (no ``list_gpus``) leaves a 2-entry split resolving to one device, where
    the number is single-GPU and calling it "combined" would mislabel it.

    Do NOT use this to decide "will the loader ACTUALLY apply a multi-device
    split" (a VRAM preflight, a swap decision, a "your split spans N cards"
    notice): on the ``vulkan`` build the real split devices live in ggml-vulkan's
    own index space, which ``list_gpus()`` (torch.cuda / nvidia-smi) is
    structurally blind to (GPU-SPLIT-VKINDEX), so the detected re-filter here
    COLLAPSES a live, working 2-way vulkan split to < 2. That is the honest answer
    for a LABEL (``vram_capacity()`` itself cannot sum a split it cannot measure,
    so it too falls back to the single-GPU number, and calling that "combined"
    would lie), but the WRONG answer for a load-safety gate. Use
    :func:`applied_split_device_count` for the "will a split be applied at load
    time" question - it mirrors :func:`apply_gpu_split`'s own gate and does not
    apply the detected re-filter.

    Returns 0 when no split is configured (the common single-GPU path, with no
    hardware probe); otherwise the count of valid split devices (0/1 = effectively
    single, 2+ = combined)."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    split = cfg.get("gpu_split_indices")
    if not split:
        return 0
    gpus = list_gpus()
    pairs = resolve_gpu_split(split, cfg.get("gpu_split_ratios"), gpus=gpus)
    by_index = {g.get("index") for g in gpus}
    return len([idx for idx, _ in pairs if idx in by_index])


def applied_split_device_count(config: Optional[dict] = None) -> int:
    """How many devices the loader will ACTUALLY tensor_split across for a
    GGUF/llama.cpp load - the loader-truth counterpart to
    :func:`split_device_count`'s DETECTED/labelling count.

    Mirrors :func:`apply_gpu_split`'s own gate (``len(resolve_gpu_split(...)) < 2``
    -> no split), so it answers "will a multi-device split be applied at load
    time", NOT "can we MEASURE that split's combined VRAM". The two counts differ
    on exactly one axis: the detected-device re-filter that
    :func:`split_device_count` / :func:`vram_capacity` apply against
    :func:`list_gpus` AFTER ``resolve_gpu_split``. That filter is CORRECT for a
    VRAM LABEL (you cannot honestly call a number "combined across N GPUs" when
    ``list_gpus()`` only measured one device), but WRONG for a load-safety gate on
    the ``vulkan`` build, where ``resolve_gpu_split`` passes the configured indices
    through UNVALIDATED in ggml-vulkan's own index space (GPU-SPLIT-VKINDEX) - a
    real 2-way split ``list_gpus()`` (torch.cuda / nvidia-smi) is structurally
    blind to. There this returns 2 while :func:`split_device_count` collapses to
    < 2. On a NON-vulkan box with a detected device list the two are IDENTICAL
    (``resolve_gpu_split`` already dropped unknown indices, so that later re-filter
    is a proven no-op).

    Deliberately does NOT pass ``gpus=`` (so ``resolve_gpu_split`` calls
    ``list_gpus()`` itself) and does NOT re-filter the result - exactly what
    :func:`apply_gpu_split` does, which is what makes this the loader truth rather
    than a measurability check.

    Returns 0 when no split is configured (the common path, no hardware probe);
    otherwise the count ``resolve_gpu_split`` yields. Domain is {0} U {2, 3, ...}:
    a single surviving index collapses to 0, same as ``apply_gpu_split`` leaving
    the native single-GPU default untouched."""
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    if not cfg.get("gpu_split_indices"):
        return 0
    return len(resolve_gpu_split(
        cfg.get("gpu_split_indices"), cfg.get("gpu_split_ratios")))


def _list_gpus_reading(deadline: Optional[float] = None) -> tuple:
    """``(gpus, status)`` from :func:`list_gpus`, tolerant of a test double patched
    in as a plain no-kwarg callable - the historical bare-list contract that the
    ~28 test modules stubbing ``list_gpus`` rely on. A double whose signature does
    not accept ``return_status`` is called bare and its reading treated as
    :data:`GPU_PROBE_OK`: it models a completed probe, exactly as a bare stub did
    before the status channel existed, so only a REAL status-capable probe can ever
    report itself stale/busy here. Signature-inspected rather than a blanket
    ``except TypeError`` so a genuine ``TypeError`` raised INSIDE ``list_gpus`` is
    never mistaken for a rejected kwarg and swallowed (the refinement over
    ``vram._vram_free_reading``'s try/except that its own author flagged). In
    production ``list_gpus`` always accepts ``return_status``, so the bare branch is
    a test-only affordance, never taken by the real probe.

    *deadline* is forwarded to ``list_gpus`` only when given (None leaves its
    default cap untouched), so an OFF-event-loop caller can spend a longer budget on
    a cold driver init that overruns the short server cap - the only way to get a
    FRESH first-load reading (a timed-out probe cannot be retried: it is abandoned,
    not cancelled, and a retry short-circuits to the frozen last-known-good)."""
    try:
        accepts = "return_status" in inspect.signature(list_gpus).parameters
    except (TypeError, ValueError):
        accepts = False
    if accepts:
        kw = {"return_status": True}
        if deadline is not None:
            kw["deadline"] = deadline
        return list_gpus(**kw)
    return list_gpus(), GPU_PROBE_OK


def gpu_split_shortfall(vram_required: int, config: Optional[dict] = None,
                        *, return_status: bool = False,
                        deadline: Optional[float] = None):
    """``[{"index", "needed", "free"}, ...]`` for every configured split device
    whose free VRAM, read from a FRESH probe this call (:data:`GPU_PROBE_OK`),
    cannot cover its proportional share of *vram_required*. Empty when no split is
    configured, fewer than two split devices resolve, every device has live
    headroom, OR the live per-device check could not run this call (see the probe
    freshness contract below). A shortfall entry is emitted ONLY under a fresh
    ``GPU_PROBE_OK`` reading, so every ``free`` in the result is a current
    measurement a caller may quote to the user as fact.

    ``vram_capacity()`` is an AGGREGATE check: it proves total combined free VRAM
    across the split is enough, but ``apply_gpu_split()`` (the GGUF/llama.cpp
    backend's tensor_split writer) divides a model by a STATIC per-config ratio with
    NO live per-device capacity awareness of its own - unlike the HF/transformers
    backend, whose ``device_map="auto"`` is built from live per-device
    ``torch.cuda.mem_get_info()`` free VRAM instead (see ``backends/hf.py``'s
    ``_cuda_device_map``), so it already self-corrects. Without this check, a model
    too big for one device's actual share could still pass the aggregate check (e.g.
    another already-loaded model sits asymmetrically on one split device more than
    another) and reach llama.cpp's native loader with too little room on that device
    - not always a catchable Python exception, since the native loader can hard-abort
    the WORKER process rather than return NULL (that abort is contained to the
    isolated load worker, never the server - PR #606, see
    ``backends/llamacpp/_runner.py``). Callers should treat any non-empty result as a
    hard refusal for a GGUF-backend load (see ``http_server.switch_engine``), not
    merely a warning.

    Probe freshness (AGENTS.md rule 5). ``list_gpus()`` is deadline-bounded: on a
    TIMEOUT/BUSY it serves a FROZEN last-known-good reading, and a cold driver init
    overruns that cap on a healthy box (~6.5s vs the 4s cap, see
    ``_GPU_PROBE_DEADLINE``), so an overrun is normal, not a fault. This gate
    therefore does NOT compute a shortfall from a stale reading and does NOT refuse
    on one (refusing would break every working box's first load); on a non-OK probe
    it returns ``[]`` (best-effort admit, logged at debug), relying on the isolated
    worker's contained abort above as the backstop. An empty bare-list result thus
    cannot, on its own, be told apart from "verified all-clear": a caller that must
    distinguish "checked, clear" from "could not check" MUST pass
    ``return_status=True`` to receive ``(shortfall, status)`` carrying the underlying
    :data:`GPU_PROBE_OK` / :data:`GPU_PROBE_TIMEOUT` / :data:`GPU_PROBE_BUSY`.

    This proves the reading is LIVE, not COMPLETE. On a backend/OS where ``free``
    counts only this process's own allocations (blind to other processes' VRAM), a
    fresh reading can still under-state usage; that completeness axis is orthogonal
    and is carried per-device elsewhere, not by this probe status.

    Only meaningful for the GGUF/llama.cpp load path - callers should gate on that
    themselves (e.g. via ``inference.engine._is_gguf``); this function has no way to
    know which backend a given load will use.

    Deliberately takes no headroom margin of its own (a device with EXACTLY enough
    free for its proportional share passes) - if a caller wants the same safety
    margin the aggregate ``vram_capacity()`` check demands, add it to *vram_required*
    before calling (e.g. ``vram_required + headroom``), so a per-device share is not
    held to a thinner margin than the aggregate ceiling it composes with.

    With ``return_status=True`` returns ``(shortfall, status)``; otherwise the bare
    ``shortfall`` list (the historical shape every existing caller relies on).

    *deadline* is forwarded to the underlying ``list_gpus`` probe (None leaves its
    default cap). It is a KNOB an OFF-event-loop caller that runs this via
    ``run_in_executor`` (e.g. ``switch_engine``) can opt into by passing
    :data:`_GPU_PROBE_CLI_DEADLINE`, so a cold driver init that overruns the short
    server cap still completes and yields a FRESH per-device reading instead of
    timing out into the best-effort admit above. Passing nothing keeps the short cap
    (the current caller behaviour): such a caller simply falls into that admit on a
    cold first load. An on-loop caller must NOT pass a long deadline (a long block
    would freeze the loop, PR #541).
    """
    from localm.config import load_config
    cfg = config if config is not None else load_config()

    def _ret(shortfall, status):
        return (shortfall, status) if return_status else shortfall

    if not cfg.get("gpu_split_indices"):
        # No split configured: a conclusive answer that needs no hardware probe.
        return _ret([], GPU_PROBE_OK)
    if _native_backend_has_vulkan():
        # GPU-SPLIT-VKINDEX honest-unknown: on the vulkan build the configured
        # split indices live in ggml-vulkan's own index space at load time, which
        # list_gpus() (torch.cuda / nvidia-smi) cannot see or order - torch index
        # N is NOT ggml-vulkan index N (resolve_preferred_device documents exactly
        # this hazard). A per-device share check here would measure the WRONG
        # cards: a silent no-op when torch sees nothing, a wrong refusal/pass on a
        # mixed box. We cannot honestly check per-device fit on this backend, so we
        # do not - but we SURFACE the skip rather than present a check that never
        # ran as "passed" (rule 5, do-not-hide-problems). INFO not debug: the
        # always-on ring buffer is INFO+, so a debug line would never reach a bug
        # report - the same reason vram.py's media_split_notice gives for not
        # burying a user-configured-split shortfall at debug. Not WARNING: the skip
        # is benign whenever the model fits (the common case), so WARNING would cry
        # wolf every load. The GGUF load is subprocess-isolated, so an oversized
        # model still fails as a catchable error, not a lost check - that isolation
        # is the real backstop this defers to.
        logger.info(
            "gpu_split_shortfall: skipping the per-device split VRAM preflight on "
            "the vulkan backend - the configured split indices are in ggml-vulkan's "
            "index space, which list_gpus() cannot map to a card, so no per-device "
            "check can name the right device (GPU-SPLIT-VKINDEX); relying on the "
            "subprocess-isolated loader to catch an oversized load instead.")
        # Conclusive skip with no probe, so it mirrors the no-split return above and
        # reports GPU_PROBE_OK - NOT a non-OK "stale probe" status: nothing was
        # probed, and (like the no-split branch) this is a deterministic routing
        # decision, not an inconclusive reading. A future return_status consumer that
        # needs to tell "checked-clear" from "vulkan-skip" apart would want a distinct
        # status; flagged for the probe-status owner rather than overloaded here.
        return _ret([], GPU_PROBE_OK)

    gpus, status = _list_gpus_reading(deadline)
    if status != GPU_PROBE_OK:
        # No FRESH reading this call: list_gpus served a frozen last-known-good value
        # (or []) after a probe TIMEOUT/BUSY. This gate's whole contract is a LIVE
        # per-device check, so it neither quotes that stale "free" as a current figure
        # (AGENTS.md rule 5) NOR refuses on it: a cold ROCm/CUDA driver init overruns
        # the 4s probe cap on a HEALTHY box (~6.5s, see _GPU_PROBE_DEADLINE), so
        # refusing would break working setups on every first load. The check could not
        # run this call -> admit best-effort, surfaced via debug + the returned status,
        # never a silent success. The GGUF/embedder load runs in an isolated worker
        # whose native abort is contained to that child (PR #606) - the backstop a
        # best-effort admit relies on.
        logger.debug("gpu_split_shortfall: probe status=%s (no fresh per-device VRAM "
                     "reading); admitting split load best-effort, per-device fit "
                     "unverified this call", status)
        return _ret([], status)

    pairs = resolve_gpu_split(
        cfg.get("gpu_split_indices"), cfg.get("gpu_split_ratios"), gpus=gpus)
    if len(pairs) < 2:
        return _ret([], status)
    by_index = {g.get("index"): g for g in gpus}
    total_ratio = sum(ratio for _, ratio in pairs)
    if total_ratio <= 0:
        return _ret([], status)
    shortfall = []
    for idx, ratio in pairs:
        g = by_index.get(idx)
        if g is None or g.get("free") is None:
            # Structural guard for a malformed/absent device dict ONLY - NOT the
            # "probe could not run" handler (that is the status != OK branch above).
            # Under GPU_PROBE_OK, list_gpus emits an int "free" for every device and
            # DROPS a non-reporting one (see :493/:525), so this branch is dead in
            # production; it only stops a None-free entry (e.g. test-injected) from
            # crashing the loop.
            continue
        needed = int(vram_required * (ratio / total_ratio))
        free = g["free"]
        if free < needed:
            shortfall.append({"index": idx, "needed": needed, "free": free})
    return _ret(shortfall, status)


def resolve_preferred_device(config: Optional[dict] = None, *,
                            gpus: Optional[list] = None) -> Optional[int]:
    """The device a media workload should DEFAULT to, with every OTHER card left
    VISIBLE. ``None`` when nothing is configured, or when no torch-visible device can
    be named honestly.

    NEVER use this to MASK the other cards away. That was a real, shipped bug (see the
    rename from ``resolve_whole_model_device``): ComfyUI core ships per-component GPU
    PLACEMENT - ``SelectModelDevice``/``SelectCLIPDevice``/``SelectVAEDevice``
    (``comfy_extras/nodes_multigpu.py``, registered at ``nodes.py:2440``), which call
    ``deepclone_multigpu`` to rehome a component onto another card with independent
    weights. Masking to one device (ComfyUI's ``--cuda-device``, or a bare
    ``CUDA_VISIBLE_DEVICES=N``) deletes the other cards from torch's view and turns
    every one of those nodes into a silent no-op. Prefer ComfyUI's ``--default-device``,
    which reorders rather than masks (``main.py:69-76``), or :func:`visible_device_order`
    for an install we cannot pass argv to.

    The predicate here is PREFERENCE, not exclusivity: "which card should lead", not
    "which card is the only one". It is deliberately NOT :func:`resolve_main_gpu_index`,
    which answers IDENTITY ("which device is primary") and resolves an unset value to
    device 0 - using that here would silently pick card 0 and ignore the split, the
    shape of the #661 regression. On a configured split this is a CAPACITY-informed
    choice: the split device with the MOST live free VRAM.

    INDEX SPACE (this is load-bearing): the answer is always a TORCH device index,
    because media runs on torch (ComfyUI), and :func:`list_gpus` enumerates via
    torch.cuda. It must never leak :func:`resolve_gpu_split`'s Vulkan pass-through
    (GPU-SPLIT-VKINDEX): on the ``vulkan`` llama.cpp build that function returns
    indices UNVALIDATED, in ggml-vulkan's own index space, which torch does not share.
    Handing one of those to ComfyUI as a CUDA/HIP id would name the wrong card. So a
    device is returned only when it is genuinely torch-visible; otherwise ``None``, and
    ComfyUI keeps its own default.
    """
    from localm.config import load_config
    cfg = config if config is not None else load_config()
    split = cfg.get("gpu_split_indices")
    main = cfg.get("main_gpu_index")
    if not split and main is None:
        return None                 # nothing configured: do not invent a device
    devices = gpus if gpus is not None else list_gpus()
    by_index = {g.get("index"): g for g in devices}
    if not by_index:
        # No torch-visible device at all, so any index we named would be a guess in an
        # index space we cannot check. Let ComfyUI choose; do not pretend to know.
        return None
    if split:
        pairs = resolve_gpu_split(split, cfg.get("gpu_split_ratios"), gpus=devices)
        if len(pairs) >= 2:
            visible = [idx for idx, _ in pairs if idx in by_index]
            measured = [(idx, by_index[idx]["free"]) for idx in visible
                        if by_index[idx].get("free") is not None]
            if measured:
                return max(measured, key=lambda t: t[1])[0]
            if visible:
                # Split devices ARE torch-visible but none reports free VRAM, so the
                # capacity-informed choice cannot be made. Lead with the first visible
                # one and SAY SO (rule 5): it may not be the emptiest card. Warned, not
                # raised - refusing would break a working setup over a probe that is
                # allowed to be unmeasurable.
                logger.warning(
                    "gpu_split is configured but no split device reports free VRAM, so "
                    "the best card cannot be chosen for media; defaulting to device %d, "
                    "which may have less free VRAM than its peers.", visible[0])
                return visible[0]
            # NOT ONE configured split device is torch-visible. resolve_gpu_split()
            # passes indices through UNVALIDATED on the vulkan llama.cpp build
            # (GPU-SPLIT-VKINDEX), so these are very likely ggml-vulkan indices, which
            # mean something else entirely to torch. Naming one would point ComfyUI at
            # the wrong card. Say so and let ComfyUI default.
            logger.warning(
                "gpu_split %r resolves to no torch-visible device, so media cannot name "
                "one: those indices are not in torch's index space (a Vulkan-only "
                "llama.cpp split does this). Leaving the device to ComfyUI's default.",
                split)
            return None
        # A split was configured but did not resolve to 2+ detected devices.
        # resolve_gpu_split() has already WARNED about the dropped indices; fall
        # through to the main-index answer below rather than guess a device.
    if main is None:
        return None
    idx = resolve_main_gpu_index(main, gpus=devices)
    return idx if idx in by_index else None


def visible_device_order(config: Optional[dict] = None, *,
                         gpus: Optional[list] = None) -> Optional[list]:
    """Every torch-visible device index with the PREFERRED one FIRST, or ``None`` when
    no device should be named.

    For a ComfyUI localm cannot pass argv to: the user's OWN install, started by their
    own launcher (possibly ZLUDA-wrapped), where the child env is the only lever. This
    mirrors exactly what ComfyUI's own ``--default-device`` does at ``main.py:69-76`` -
    it REORDERS ``CUDA_VISIBLE_DEVICES``/``HIP_VISIBLE_DEVICES`` so the chosen device
    leads, leaving the rest visible - rather than what ``--cuda-device`` does at
    ``main.py:78-81``, which masks them away and silently disables core's
    ``Select*Device`` placement nodes.

    Every index is torch-visible by construction (:func:`resolve_preferred_device` and
    :func:`list_gpus` share torch's index space), so this never emits a Vulkan-space id.

    NOTE the consequence, which callers must respect: after this reorder the preferred
    card becomes torch index 0, so a workflow's ``gpu:N`` refers to the REORDERED
    position, not to localm's own ``list_gpus`` index. Anything emitting ``gpu:N`` into
    a workflow has to map through this order, not around it.
    """
    devices = gpus if gpus is not None else list_gpus()
    chosen = resolve_preferred_device(config, gpus=devices)
    if chosen is None:
        return None
    rest = sorted(g.get("index") for g in devices
                  if g.get("index") is not None and g.get("index") != chosen)
    return [chosen] + rest


def fit_label(size_bytes: int, total_vram: Optional[int]) -> str:
    """
    Capacity badge for one file, against a single-GPU (or combined-split) VRAM
    ceiling: "fits" / "tight" / "too-big", or "" when VRAM is unknown. "tight"
    means it should load with little headroom (small context, nothing else on the
    GPU); "too-big" still runs, with some layers offloaded to system RAM (slower).
    That partial offload is delivered automatically: with n_gpu_layers_auto on
    (the default) the loader sizes how many layers fit from free VRAM at load
    (GgufBackend._auto_gpu_layers), so a "too-big" model loads instead of being
    refused, rather than only if the user manually lowers -g.

    The need estimate here (weights * safety factor + fixed overhead) is
    context-agnostic and deliberately a touch more conservative on weights than
    the loader's exact weights + real-KV + overhead math (GgufBackend._check_vram),
    so at a normal/default context a "fits" badge is not optimistic. It carries no
    explicit KV term, so it is a weights-fit signal, not a guarantee for an
    unusually large -c/n_ctx (whose KV can exceed the weight slack); the loader's
    own preflight remains the authority. A "tight"/"too-big" model may still load
    via partial offload.
    """
    if not total_vram or not size_bytes:
        return ""
    need = size_bytes * _WEIGHT_FACTOR + _OVERHEAD_BYTES
    if need <= total_vram * 0.85:
        return "fits"
    if need <= total_vram:
        return "tight"
    return "too-big"
