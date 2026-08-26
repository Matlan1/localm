# SPDX-License-Identifier: AGPL-3.0-or-later
"""Device-global VRAM usage on Windows: how much VRAM is in use ACROSS EVERY
PROCESS, which the GPU driver's own free-memory query does not tell us here.

WHY THIS EXISTS (measured, not assumed - see
dev-notes/vram-cross-process-blindness.md for the full evidence):
``torch.cuda.mem_get_info()`` and llama.cpp's ``ggml_backend_dev_memory`` BOTH
report, on this platform (Windows + an AMD ROCm/HIP torch build),

    free == total - (the CALLING HIP runtime instance's own allocations)

i.e. they are blind to every other process's VRAM. Measured live: with ~10.5 GB
genuinely in use, both reported 0.14 GB used, byte-identically. A plain torch
tensor allocated in a CHILD process moved the parent's reading by exactly 0, while
the OS counter tracked it exactly; a model loaded by localm's own isolated GGUF
worker (see backends/gguf.py - every GGUF load is out-of-process since #606) is
invisible for the same reason, as is a GAME holding 7 GB (observed).

That blindness is what this module fixes. It is NOT a caching/staleness bug (PR
#693's domain): the reading is live and fresh for the caller's OWN allocations, and
a FRESH process is equally blind, so it is not a snapshot-at-init artifact either.

WHY THE READING IS NOT SIMPLY REPLACED EVERYWHERE: on Linux and on NVIDIA, the
driver query is device-global BY DOCUMENTATION - NVIDIA's CUDA docs specify *free as
"the amount of memory on the device that is free according to the OS" and explicitly
warn that "a different process" can move it. So the existing reading is CORRECT
there and must not be touched. This module is Windows-only and additive: it corrects
a reading that is known-wrong on this platform, and returns nothing everywhere else.

NOT ROOT-CAUSED, and deliberately not guessed at: WHY the AMD driver reports a
per-process view on Windows is not established. An initial analysis blamed a
specific branch in AMD's PAL/rocclr, but that branch is dead code in the public
repo, and at least two other candidates explain the observation equally well. The
fix does not depend on knowing which: the blindness is measured and replicated, and
the sources below are measured device-global against it.

SOURCES, in preference order (both verified device-global on real hardware, agreeing
within 0-1 MB across 38 simultaneous samples, and tracking both an allocation and its
release):

1. AMD ADL (``atiadlxx.dll``, installed in System32 by the AMD driver itself - no
   new dependency, no admin). ``ADL2_Adapter_DedicatedVRAMUsage_Get`` is
   device-global and costs ~0.004 ms/call. Preferred because ADL also reports each
   adapter's PCI BUS NUMBER, which maps unambiguously onto torch's own
   ``get_device_properties(i).pci_bus_id`` - so a multi-GPU box gets the RIGHT
   number per device rather than a guess.
2. PDH (``\\GPU Adapter Memory(*)\\Dedicated Usage``, the vendor-neutral WDDM
   counter Task Manager itself uses), via ``win32pdh`` from pywin32, a
   Windows-gated dependency (declared in pyproject.toml; this line previously
   claimed it was "already a dependency" when it was declared nowhere, and was
   true only of a venv that happened to have it). Covers non-AMD Windows GPUs,
   where ADL does not exist - it is the ONLY device-global source there. Its instance
   name is a LUID and carries NO PCI bus, so it CANNOT be mapped to a device index
   on a multi-GPU box - it is therefore used ONLY when the mapping is unambiguous
   (exactly one GPU and one adapter instance). Guessing there would be worse than
   the blind number it replaces, because it would be confidently wrong.

Deliberately NOT used (each measured and rejected, so nobody re-litigates them):
- ``rocm-smi`` / ``amd-smi``: AMD ships NEITHER for Windows. A recursive search of
  the whole ROCm tree finds no ``*smi*`` binary at all. (This was the intuitive fix;
  it does not exist on this platform.)
- DXGI ``QueryVideoMemoryInfo``: ``CurrentUsage`` is PER-PROCESS (verified: flat
  while another process held 3 GB, but rose correctly on the caller's own alloc), and
  ``Budget`` is a per-process soft ALLOWANCE that is not invertible to a global figure.
- ``hipInfo.exe``: per-process (reports "99% free" under heavy real load) and ~1 s.
- WMI ``Win32_VideoController.AdapterRAM``: static capacity, and uint32-overflowed
  (reports 4 GB for a 16 GB card).

COST: the ADL context and the PDH query handle are opened ONCE and kept open,
because reopening is the entire cost - a persistent PDH handle reads in ~0.02 ms
while ``Get-Counter`` costs ~2278 ms and a fresh handle ~887 ms. Every entry point
here is non-raising and returns "unmeasurable" rather than propagating a driver
failure into a caller that only wanted a number.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from typing import Dict, Optional

from localm.debuglog import logger

_ADL_MAX_PATH = 256
_ADL_OK = 0
_ADL_VENDOR_AMD = 1002        # decimal, NOT 0x1002 - ADL reports it in base 10

# ADL's PMLog sensor array is a fixed 256 slots, each a (supported, value) pair.
_ADL_PMLOG_MAX_SENSORS = 256
# ADL_PMLOG_INFO_ACTIVITY_GFX. MEASURED on this driver rather than taken from a
# header, because a wrong index reads a DIFFERENT sensor and still returns a
# plausible 0-100 number - a silent wrong answer, not an error. The whole
# supported-sensor set was dumped and identified by INTERNAL CONSISTENCY against
# physical reality: 1=GFXCLK 2564MHz and 2=MEMCLK 1988MHz (a 6900 XT's real
# clocks), 14=FAN_RPM 692 alongside 15=FAN_PERCENT 20 (a matching pair),
# 8=TEMP_EDGE 67C below 27=TEMP_HOTSPOT 73C (hotspot is always the higher of the
# two), 23=ASIC_POWER 91W. Index 19 moved 0->99 across a load change while power
# tracked it 41W->91W and the clock 0->2570MHz, which no unrelated sensor would.
_ADL_PMLOG_ACTIVITY_GFX = 19

_lock = threading.Lock()
_adl_state: Optional[dict] = None      # None = not tried yet; {} = tried and unusable
_pdh_state: Optional[dict] = None
_pdh_util_state: Optional[dict] = None


class _AdapterInfo(ctypes.Structure):
    """ADL's AdapterInfo. The field order and the ADL_MAX_PATH-sized char arrays are
    the ABI: a wrong layout silently marshals garbage (the same class of hazard the
    llama.cpp ctypes binding documents in llamacpp/_abi.py), so it is transcribed
    from the ADL SDK header verbatim and validated at runtime by checking that the
    reported bus numbers actually match torch's."""
    _fields_ = [
        ("iSize", ctypes.c_int),
        ("iAdapterIndex", ctypes.c_int),
        ("strUDID", ctypes.c_char * _ADL_MAX_PATH),
        ("iBusNumber", ctypes.c_int),
        ("iDeviceNumber", ctypes.c_int),
        ("iFunctionNumber", ctypes.c_int),
        ("iVendorID", ctypes.c_int),
        ("strAdapterName", ctypes.c_char * _ADL_MAX_PATH),
        ("strDisplayName", ctypes.c_char * _ADL_MAX_PATH),
        ("iPresent", ctypes.c_int),
        ("iExist", ctypes.c_int),
        ("strDriverPath", ctypes.c_char * _ADL_MAX_PATH),
        ("strDriverPathExt", ctypes.c_char * _ADL_MAX_PATH),
        ("strPNPString", ctypes.c_char * _ADL_MAX_PATH),
        ("iOSDisplayIndex", ctypes.c_int),
    ]


class _ADLSingleSensorData(ctypes.Structure):
    """One PMLog sensor slot. ``supported`` is a flag, not a status code: a zero
    there means the board does not publish that sensor, and ``value`` is then
    meaningless rather than zero-valued."""
    _fields_ = [("supported", ctypes.c_int), ("value", ctypes.c_int)]


class _ADLPMLogDataOutput(ctypes.Structure):
    """ADL's PMLogDataOutput. Same ABI hazard as :class:`_AdapterInfo`: the sensor
    array is a FIXED 256 entries and a short array silently marshals garbage."""
    _fields_ = [("size", ctypes.c_int),
                ("sensors", _ADLSingleSensorData * _ADL_PMLOG_MAX_SENSORS)]


_ADL_ALLOC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int)


def _adl_open() -> dict:
    """Open ADL once and cache the context. Returns the cached state, or {} when ADL
    is not usable this call. Never raises.

    Distinguishes PERMANENT from TRANSIENT unusability (rule 5, missing-vs-corrupt),
    mirroring the PDH path: a missing ``atiadlxx.dll`` (any non-AMD box) is permanent
    and latched to a sticky {} so we never re-attempt the throwing load; a
    ``ADL2_Main_Control_Create`` failure (the driver momentarily not answering) is
    NOT latched - it returns {} for this call but leaves ``_adl_state`` None so the
    next call retries, rather than disabling ADL for the whole process on a hiccup."""
    global _adl_state
    if _adl_state is not None:
        return _adl_state
    try:
        dll = ctypes.WinDLL("atiadlxx.dll")
    except Exception as e:
        # No atiadlxx.dll (any non-AMD box) is the ordinary case, not a fault, and it
        # is PERMANENT: latch off so we never re-attempt the throwing load. The PDH
        # path still covers this box.
        logger.debug("gpu_usage: ADL unavailable: %s", e)
        _adl_state = {}
        return _adl_state
    try:
        # ADL calls this back to allocate the AdapterInfo array. The buffers must
        # outlive the call, so they are parked on the cached state, not a local.
        keepalive: list = []

        @_ADL_ALLOC
        def _alloc(size: int):
            buf = ctypes.create_string_buffer(size)
            keepalive.append(buf)
            return ctypes.cast(buf, ctypes.c_void_p).value

        ctx = ctypes.c_void_p()
        # 1 = "connect to a running driver only", so this never spins one up.
        if dll.ADL2_Main_Control_Create(_alloc, 1, ctypes.byref(ctx)) != _ADL_OK:
            # TRANSIENT-possible (driver busy): do NOT latch - retry next call.
            logger.debug("gpu_usage: ADL2_Main_Control_Create failed; will retry")
            return {}
        _adl_state = {"dll": dll, "ctx": ctx, "alloc": _alloc, "keepalive": keepalive}
        return _adl_state
    except Exception as e:
        # An unexpected init error (not the DLL absence handled above): also treat as
        # retryable rather than latching ADL off for the process lifetime.
        logger.debug("gpu_usage: ADL init failed; will retry: %s", e)
        return {}


def _adl_used_by_bus() -> Dict[int, int]:
    """``{pci_bus_number: device_global_used_bytes}`` for every present AMD adapter,
    or ``{}`` when ADL cannot answer.

    ADL reports several logical adapters per physical GPU (7 for one card here), so
    entries are deduped on (bus, device, function) - the first that answers wins."""
    state = _adl_open()
    if not state:
        return {}
    dll, ctx = state["dll"], state["ctx"]
    try:
        n = ctypes.c_int(0)
        if dll.ADL2_Adapter_NumberOfAdapters_Get(ctx, ctypes.byref(n)) != _ADL_OK:
            return {}
        if n.value <= 0:
            return {}
        arr = (_AdapterInfo * n.value)()
        size = ctypes.sizeof(_AdapterInfo) * n.value
        if dll.ADL2_Adapter_AdapterInfo_Get(ctx, ctypes.byref(arr), size) != _ADL_OK:
            return {}
        out: Dict[int, int] = {}
        seen = set()
        for info in arr:
            if not info.iPresent or info.iVendorID != _ADL_VENDOR_AMD:
                continue
            key = (info.iBusNumber, info.iDeviceNumber, info.iFunctionNumber)
            if key in seen:
                continue
            used_mb = ctypes.c_int(0)
            rc = dll.ADL2_Adapter_DedicatedVRAMUsage_Get(
                ctx, info.iAdapterIndex, ctypes.byref(used_mb))
            if rc != _ADL_OK:
                continue   # one adapter failing never hides the rest
            seen.add(key)
            out[int(info.iBusNumber)] = int(used_mb.value) * 1024 * 1024
        return out
    except Exception as e:
        logger.debug("gpu_usage: ADL query failed: %s", e)
        return {}


def _adl_activity_by_bus() -> Dict[int, float]:
    """``{pci_bus_number: whole_gpu_busy_percent}`` per present AMD adapter, or ``{}``.

    THE WHOLE-GPU FIGURE, whoever is causing the load - which is the one thing the
    WDDM ``GPU Engine`` counter cannot give us on this vendor. See
    :func:`adapter_utilisation` for that counter's measured blind spot; this is the
    source that fixes it.

    WHY PMLog AND NOT ONE OF THE Overdrive ACTIVITY CALLS, measured on this driver
    (RX 6900 XT, RDNA2) rather than taken from a doc, because the commonly-cited
    names all FAIL here and each fails differently:

        ADL2_Overdrive_Caps                     -> supported=1 enabled=1 VERSION=8
        ADL2_Overdrive5_CurrentActivity_Get     -> rc=-1  (generic error)
        ADL2_Overdrive6_CurrentStatus_Get       -> rc=-8  ADL_ERR_NOT_SUPPORTED
        ADL2_OverdriveN_PerformanceStatus_Get   -> rc=-8  ADL_ERR_NOT_SUPPORTED
        ADL2_New_QueryPMLogData_Get             -> rc=0, 12 supported sensors

    The board is Overdrive **8**, and Overdrive8 exposes activity ONLY through
    PMLog - so an implementation written against Overdrive5/6/N compiles, runs,
    returns rc != 0 forever and reports nothing on exactly the hardware it was
    added for. Do not "fix" this by reaching for one of those names.

    ``ADL2_New_QueryPMLogData_Get`` needs no ``PMLog_Start`` and no shared-memory
    session: called cold it answers rc=0 immediately (measured).

    HONESTY (AGENTS.md rule 5): a sensor the board does not publish, or a value
    outside 0-100, yields NO ENTRY rather than a fabricated 0%. A card reported
    idle when nothing was measured is exactly the failed reading presented as a
    successful one that rule 5 forbids.
    """
    state = _adl_open()
    if not state:
        return {}
    dll, ctx = state["dll"], state["ctx"]
    try:
        n = ctypes.c_int(0)
        if dll.ADL2_Adapter_NumberOfAdapters_Get(ctx, ctypes.byref(n)) != _ADL_OK:
            return {}
        if n.value <= 0:
            return {}
        arr = (_AdapterInfo * n.value)()
        size = ctypes.sizeof(_AdapterInfo) * n.value
        if dll.ADL2_Adapter_AdapterInfo_Get(ctx, ctypes.byref(arr), size) != _ADL_OK:
            return {}
        out: Dict[int, float] = {}
        seen = set()
        for info in arr:
            if not info.iPresent or info.iVendorID != _ADL_VENDOR_AMD:
                continue
            # ADL reports several LOGICAL adapters per physical card (7 for the one
            # here), so dedupe on the PCI triple exactly as _adl_used_by_bus does -
            # otherwise one card is counted repeatedly.
            key = (info.iBusNumber, info.iDeviceNumber, info.iFunctionNumber)
            if key in seen:
                continue
            data = _ADLPMLogDataOutput()
            rc = dll.ADL2_New_QueryPMLogData_Get(
                ctx, info.iAdapterIndex, ctypes.byref(data))
            if rc != _ADL_OK:
                continue   # one adapter failing never hides the rest
            sensor = data.sensors[_ADL_PMLOG_ACTIVITY_GFX]
            if not sensor.supported:
                continue
            pct = float(sensor.value)
            if not (0.0 <= pct <= 100.0):
                # Out of range means we are not reading what we think we are.
                # Say nothing rather than publish a number we cannot stand behind.
                logger.debug("gpu_usage: ADL activity out of range (%s); ignoring", pct)
                continue
            seen.add(key)
            out[int(info.iBusNumber)] = pct
        return out
    except Exception as e:
        logger.debug("gpu_usage: ADL activity query failed: %s", e)
        return {}


def amd_whole_gpu_activity() -> Optional[float]:
    """Whole-GPU busy percent on AMD, or None when this box cannot answer.

    None means "not measured", never "idle" - the caller must omit the field
    rather than render a 0%.

    WHAT THIS NUMBER IS, because it is easy to over-read: the fraction of TIME
    the graphics engine had work, which is what "GPU utilisation" means in AMD's
    control panel, GPU-Z and nvidia-smi alike. It is NOT how hard the card is
    working. MEASURED: it reads 99% at 87 W under a light continuous workload and
    99% again at 295 W under saturating compute, and it correctly falls to 6% when
    the card parks at 39 MHz. So it distinguishes BUSY from IDLE, never BUSY from
    FLAT OUT - a saturating property every time-based utilisation metric shares.
    Anything wanting "how much headroom is left" needs power or clock, not this.

    MULTI-CARD: the busiest AMD adapter wins. localm's stats payload carries ONE
    system-wide ``gpu.percent`` (the sidebar shows a single GPU metric), so some
    card has to be chosen. The busiest is picked because that is the one a user
    asking "is my GPU busy" means, and an average across an idle second card
    would hide a saturated first one. A genuine per-card breakdown belongs with
    the per-card VRAM rows, which key off the device list rather than a bus.
    """
    by_bus = _adl_activity_by_bus()
    if not by_bus:
        return None
    return round(max(by_bus.values()), 1)


def _pdh_adapter_used() -> list:
    """Device-global used bytes per WDDM adapter instance, via the vendor-neutral
    PDH counter, or [] when unavailable.

    The query handle is opened ONCE and reused: a persistent handle reads in ~0.02 ms,
    against ~887 ms to reopen one per call (and ~2278 ms for a `Get-Counter`
    subprocess), which is the difference between a probe that can sit on a request
    path and one that cannot.

    Returns one value per adapter-LUID instance (the `GPU Adapter Memory` instance
    name IS the adapter LUID, so each appears once - there is no cross-instance
    summing). The ADAPTER counter is used rather than summing the per-process
    `\\GPU Process Memory` instances, because shared surfaces are counted against
    several processes and that sum overcounts (measured: 8.70 GB summed vs 7.33 GB
    actually on the adapter)."""
    global _pdh_state
    if _pdh_state is not None and not _pdh_state:
        return []
    try:
        import win32pdh
    except Exception as e:
        logger.debug("gpu_usage: win32pdh unavailable: %s", e)
        _pdh_state = {}
        return []
    try:
        if _pdh_state is None:
            query = win32pdh.OpenQuery()
            _pdh_state = {"query": query, "counters": {}, "pdh": win32pdh}
        win32pdh_mod = _pdh_state["pdh"]
        query = _pdh_state["query"]
        counters = _pdh_state["counters"]

        # Re-enumerate instances each call: an adapter can appear/disappear (an eGPU,
        # a driver reset), and enumeration is cheap next to reopening the query.
        _objs, instances = win32pdh_mod.EnumObjectItems(
            None, None, "GPU Adapter Memory", win32pdh.PERF_DETAIL_WIZARD)
        totals: Dict[str, int] = {}
        for inst in set(instances):
            key = inst
            if key not in counters:
                path = win32pdh_mod.MakeCounterPath(
                    (None, "GPU Adapter Memory", inst, None, -1, "Dedicated Usage"))
                # AddEnglishCounter: the counter name above is English, and a
                # localized Windows would reject AddCounter's English path.
                counters[key] = win32pdh_mod.AddEnglishCounter(query, path)
        win32pdh_mod.CollectQueryData(query)
        for key, handle in list(counters.items()):
            try:
                _typ, val = win32pdh_mod.GetFormattedCounterValue(
                    handle, win32pdh.PDH_FMT_LARGE)
            except Exception:
                continue   # a vanished instance never hides the rest
            totals[key] = int(val)
        return [v for v in totals.values()]
    except Exception as e:
        # A RUNTIME query failure (a momentary CollectQueryData/GetFormattedCounterValue
        # hiccup, a transient driver state) - NOT proof the source is permanently
        # unusable. Leave _pdh_state as the open query so the NEXT call retries, rather
        # than poisoning it to {} and silently losing the correction for the whole
        # process lifetime (the missing-vs-corrupt collapse AGENTS.md rule 5 warns
        # against). Only the win32pdh ImportError above - a genuinely permanent
        # condition - sets the sticky {}.
        logger.debug("gpu_usage: PDH query failed (will retry next call): %s", e)
        return []


def adapter_utilisation() -> Dict[str, float]:
    """GPU busy percentage per WDDM adapter LUID, or ``{}`` when unavailable.

    VENDOR-NEUTRAL, which is the point. localm's only utilisation source was
    ``nvidia-smi``, so every AMD and Intel board reported no GPU load at all - not
    "unknown", simply a missing metric on a status bar whose job is to show load.
    That was a real gap rather than a fair limitation: Windows exposes the figure
    for EVERY vendor through the same WDDM counters Task Manager itself reads, and
    this module already keeps a persistent PDH query open for adapter memory.

    NOT THE RIGHT SOURCE ON AMD - USE :func:`amd_whole_gpu_activity` THERE FIRST.

    MEASURED 2026-08-20 on an RX 6900 XT, in a controlled idle/load/idle A/B:
    THIS NUMBER DOES NOT TRACK THE CARD'S STATE. Across a card PARKED at 46 W and
    39 MHz and the same card BOOSTED to 87 W and 2574 MHz, it reported 7.1-7.2%
    in BOTH - because those 7.2% were a screen-streaming process's ``Video Codec
    1`` engine, not the GPU. Over the same samples its correlation with core
    clock was -0.010, while the card's own activity sensor read 6% parked and 99%
    boosted and correlated +0.971 with clock. That is the reported defect: the
    maximum over engine types is whatever unrelated engine happens to lead, and
    localm displayed it as GPU load.

    IT IS NOT UNIFORMLY BLIND, and the honest version matters because the tidy
    version ("this vendor's compute is invisible here") is FALSE and was measured
    false: under a synthetic 295 W pure-compute load ``Compute 0`` did read
    93-100% and the fold tracked power at +0.956. So it is UNRELIABLE rather than
    dead - it sees work that lands squarely on one countable engine, and misses
    or misattributes the rest. Unreliable is the harder failure to spot, because
    it is right often enough to look sound.

    Counter: ``GPU Engine`` / ``Utilization Percentage``, aggregated the way Task
    Manager aggregates it. Instances are per PROCESS and per ENGINE (3D, Copy,
    Video Codec, Compute, ...), so per-process values are summed WITHIN an engine
    type and the adapter's figure is then the BUSIEST engine type, not the sum
    across them. Summing across types double-counts work that ran concurrently on
    separate engines and routinely exceeds 100% (measured on a real board: 26.1%
    summed against 11.0% for the busiest engine).

    Kept because it is still the only vendor-neutral device-global source on
    Windows, and it is the ONLY one on Intel, where localm has no equivalent of
    ADL. Its limitation is stated here rather than silently inherited.

    Keyed by adapter LUID, read from the instance name, so a multi-GPU board
    reports each card separately rather than one blended figure.

    UTILISATION IS A RATE, so it needs two collections separated in time. The
    query handle is kept open between calls (the same reason
    :func:`_pdh_adapter_used` keeps one), which makes the interval the gap between
    successive calls - on the stats poll, seconds. The FIRST call after the query
    opens has no previous sample to rate against and returns ``{}`` rather than a
    fabricated 0%: a card reported idle when nothing was measured is the "report
    success when the measurement did not hold" that AGENTS.md rule 5 forbids.
    """
    global _pdh_util_state
    if _pdh_util_state is not None and not _pdh_util_state:
        return {}
    try:
        import win32pdh
    except Exception as e:
        logger.debug("gpu_usage: win32pdh unavailable for utilisation: %s", e)
        _pdh_util_state = {}
        return {}
    try:
        first = _pdh_util_state is None
        if first:
            _pdh_util_state = {"query": win32pdh.OpenQuery(), "counters": {},
                               "pdh": win32pdh}
        pdh = _pdh_util_state["pdh"]
        query = _pdh_util_state["query"]
        counters = _pdh_util_state["counters"]

        _objs, instances = pdh.EnumObjectItems(
            None, None, "GPU Engine", win32pdh.PERF_DETAIL_WIZARD)
        for inst in set(instances):
            if inst in counters:
                continue
            try:
                counters[inst] = pdh.AddEnglishCounter(query, pdh.MakeCounterPath(
                    (None, "GPU Engine", inst, None, -1, "Utilization Percentage")))
            except Exception:
                continue   # a vanished instance never hides the rest
        pdh.CollectQueryData(query)
        if first:
            # No previous sample: a rate counter cannot be read yet. Say nothing.
            return {}

        by_engine: Dict[tuple, float] = {}
        for inst, handle in list(counters.items()):
            try:
                _typ, val = pdh.GetFormattedCounterValue(handle, win32pdh.PDH_FMT_DOUBLE)
            except Exception:
                continue
            luid = _luid_of(inst)
            if luid is None:
                continue
            eng = inst.split("engtype_")[-1] if "engtype_" in inst else "?"
            by_engine[(luid, eng)] = by_engine.get((luid, eng), 0.0) + float(val)

        out: Dict[str, float] = {}
        for (luid, _eng), pct in by_engine.items():
            out[luid] = max(out.get(luid, 0.0), pct)
        return {k: min(100.0, round(v, 1)) for k, v in out.items()}
    except Exception as e:
        # Same reasoning as _pdh_adapter_used: a runtime hiccup is not proof the
        # source is permanently gone, so the open query is kept for the next call
        # rather than latched off for the whole process lifetime.
        logger.debug("gpu_usage: PDH utilisation query failed (will retry): %s", e)
        return {}


def _luid_of(instance: str) -> Optional[str]:
    """The adapter LUID inside a ``GPU Engine`` instance name, or None.

    Instances look like
    ``pid_1234_luid_0x00000000_0x0000C3F1_phys_0_eng_0_engtype_3D``. The LUID pair
    identifies the ADAPTER; the pid and engine index do not.
    """
    if "luid_" not in instance:
        return None
    parts = instance.split("luid_", 1)[1].split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else None


def source_is_warm() -> bool:
    """True once the source that would actually answer is open, so a further reading
    is effectively free.

    Exists because the FIRST open of a source is ~1000x the cost of every later read:
    opening ADL is a driver init MEASURED at ~750 ms, a cold PDH query is ~887 ms,
    against a ~0.02 ms warm read. That matters in one place - the caller's probe runs
    under a hard deadline, and a cold open is enough to push an otherwise-comfortable
    probe over a thin one (measured under the since-retired 4.0s default: cold probes
    went from 2.9-3.5s to 3.6-4.0s and started timing out). The default deadline is
    now cold-init-tolerant, so this bites only callers that pass a deliberately short
    deadline - the caller skips a COLD correction when its remaining budget is too
    thin, and never needs to skip a warm one.

    Gates on EITHER usable source being open, not ADL alone: on a non-AMD box ADL is
    proven-unusable ({}, falsy) after the first try but PDH is the source that will
    answer, and it may still be cold - so keying on ADL alone would call the source
    warm while a ~887 ms cold PDH open still lies ahead of the deadline. A truthy
    _adl_state (ADL open + usable) OR a truthy _pdh_state (PDH query open) means the
    next device_global_used_bytes() is cheap; both None/falsy means a cold open is
    still pending. See discover._apply_device_global_free."""
    return bool(_adl_state) or bool(_pdh_state)


def _known_blind_without_torch(reason: str) -> bool:
    """The no-torch answer for :func:`raw_reading_is_process_scoped`: True when
    the bundled HIP llama.cpp runtime is resident in this process
    (``discover.native_hip_runtime_resident()``), because then every raw
    free-VRAM reading this process can take is HIP-sourced - the source whose
    Windows blindness is the MEASURED one this module exists to correct (the
    in-process ggml query and torch's mem_get_info were measured
    byte-identical; the isolated probe daemon loads the same runtime). Where
    no HIP runtime is resident, False: no measured blindness to assert.

    *reason* says why torch could not be consulted; surfaced at debug (rule 5)
    so the decision that used to be an invisible blanket False is diagnosable."""
    from localm import discover as _discover
    resident = _discover.native_hip_runtime_resident()
    if resident:
        logger.debug(
            "gpu_usage: raw VRAM readings in this process are process-scoped: "
            "torch is not consultable (%s) but the bundled HIP llama.cpp "
            "runtime is resident, which is itself the measured-blind source",
            reason)
    return resident


def raw_reading_is_process_scoped() -> bool:
    """True when this platform's RAW driver free-VRAM query (torch.cuda.mem_get_info)
    is KNOWN to count only the calling process's own allocations - blind to every
    other process - so an uncorrected reading here must be tagged FREE_SCOPE_PROCESS
    rather than trusted or silently passed off as device-global.

    Measured only on Windows + an AMD ROCm/HIP runtime (see
    dev-notes/vram-cross-process-blindness.md - torch's ``mem_get_info`` and the
    bundled HIP llama.cpp runtime's ``ggml_backend_dev_memory`` were measured
    byte-identical and equally blind, so the blindness belongs to the HIP
    runtime on this platform, not to torch specifically). On Windows + a CUDA
    (NVIDIA) build, and on every non-Windows platform, cudaMemGetInfo is
    device-global BY DOCUMENTATION, so an uncorrected reading there is NOT
    known-blind: tagging it process-scoped would assert a blindness never
    measured and raise a spurious uncertainty flag on a number that is actually
    fine.

    Detected via ``torch.version.hip`` (set on ROCm builds, None on CUDA
    builds) whenever torch can be consulted. When it CANNOT - torch is not
    resident and a fresh import is unsafe or impossible (the three guarded
    cases below) - the answer comes from
    ``discover.native_hip_runtime_resident()`` instead: a resident bundled HIP
    runtime means every raw reading this process can take (the in-process ggml
    query, or the isolated probe daemon loading the same runtime) is
    HIP-sourced, i.e. exactly the measured-blind source. That torch-less case
    is not an edge: it is the GGUF WORKER, the process that makes the
    mid-generation context-grow sizing decision - a blanket False there kept
    the #706 device-global correction permanently off on exactly the platform
    it was built for (found live 2026-07-22). Where no HIP runtime is resident
    either (a vulkan/cpu build's worker, a torch-less NVIDIA box), False stays
    the answer: those readings' blindness was never measured, and nvidia-smi
    readings are device-global and already tagged as such upstream.

    Guards against a live-crashing race: this can run on whatever thread called
    :func:`discover._apply_device_global_free`, which can collide with
    ``discover._list_gpus_probe``'s own first ``import torch`` running in a
    BACKGROUND probe thread. That probe thread is deadline-bounded and, on a
    timeout, is ABANDONED while still running (documented in
    ``_list_gpus_with_status``: "the abandoned thread finish[es] (or never)") -
    so it can still be mid-import, stuck deep in the platform's native library
    preload, long after the probe call that spawned it has returned. A second
    thread's plain ``import torch`` then blocks on CPython's per-module import
    lock waiting for that abandoned import, which on this platform's ROCm SDK
    native preload has been observed to hard-crash the process (Windows fatal
    exception, STATUS_ENTRYPOINT_NOT_FOUND) rather than merely block.

    If torch is already resident in ``sys.modules`` (the overwhelmingly common
    case - torch is imported by many other code paths well before a real VRAM
    probe runs), this is a plain attribute read, no import, no race. If it is
    NOT yet resident, a normal ``import torch`` is safe UNLESS a GPU probe is
    currently in flight (``discover._gpu_probe_inflight``, including an
    abandoned/timed-out one - the exact hazard above), or the fresh import is
    the known-doomed native-runtime DLL conflict
    (``discover._torch_gpu_probe_known_doomed`` - e.g. inside the GGUF
    worker, where the bundled HIP runtime is always resident and torch never
    is): in those two cases, and when a permitted fresh import itself fails (a
    GGUF-only install), the answer comes from the resident-HIP-runtime signal
    via :func:`_known_blind_without_torch` rather than a blanket False - see
    the docstring above for why that is the truthful answer there. Reuses
    discover's existing probe-tracking lock rather than inventing new
    cross-module coordination."""
    import sys
    if sys.platform != "win32":
        return False
    try:
        torch = sys.modules.get("torch")
        if torch is None:
            from localm import discover as _discover
            with _discover._gpu_probe_lock:
                probe_may_be_mid_import = _discover._gpu_probe_inflight
            if probe_may_be_mid_import:
                return _known_blind_without_torch("a GPU probe may be mid-import")
            if _discover._torch_gpu_probe_known_doomed():
                # The same known-doomed fresh import _list_gpus_probe skips
                # (see that predicate's docstring): attempting it here can
                # only fault - printing the same stderr trace - so the
                # resident-runtime signal answers instead. Reached with the
                # native lib loaded and torch not resident, e.g. the GGUF
                # worker's sizing gate (_sizing._device_global_free_bytes).
                return _known_blind_without_torch(
                    "a fresh torch import here is the known-doomed DLL conflict")
            try:
                import torch
            except Exception as e:
                return _known_blind_without_torch(
                    "torch import failed (%s)" % type(e).__name__)
        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return False


def _gpu_is_amd(gpu: dict) -> bool:
    """Whether a GPU entry is an AMD card, by its human name.

    The name is whatever the caller put on the entry: ``torch.cuda.get_device_name``
    or nvidia-smi for :func:`discover._list_gpus_probe` entries, and the adapter's
    registry ``DriverDesc`` for :func:`discover.vram_info`'s registry tier - all of
    which read like "AMD Radeon RX 6900 XT", "NVIDIA GeForce ...", "Intel(R) Arc(TM)
    ...", so a substring test is reliable across vendors and never false-positives
    NVIDIA/Intel. A missing/blank name answers False - the safe default, since an
    unrecognised GPU must not be paired with ADL's AMD adapter. Used only to
    authorise the single-adapter ADL fallback in :func:`device_global_used_bytes`,
    which is already gated on there being exactly one ADL (AMD) adapter and exactly
    one requested GPU."""
    name = str(gpu.get("name") or "").lower()
    return "amd" in name or "radeon" in name


def device_global_used_bytes(gpus: list) -> Dict[int, int]:
    """``{gpu_index: device_global_used_bytes}`` for as many of *gpus* as can be
    mapped to a real adapter, or ``{}`` when this platform has no better source than
    the driver's own (already-correct, or unmeasurable) reading.

    *gpus* are ``discover.list_gpus()`` entries (each with an ``index``). Only
    Windows is corrected: everywhere else the driver query is device-global by
    documentation and this returns {} so the caller keeps its existing reading.

    Mapping rules, in order:
    - ADL: match each adapter's PCI bus number to torch's own ``pci_bus_id``. Exact,
      so it is safe on a multi-GPU box.
    - ADL, torch-less: when torch cannot supply a bus id for ANY requested
      device (the GGUF worker) but ADL reports EXACTLY one AMD adapter and
      exactly one GPU was asked about, the pairing is unambiguous without one -
      the same only-one-candidate rule the PDH path below already applies.
      Gated on :func:`raw_reading_is_process_scoped` so it can never fire on a
      box whose single detected GPU is NOT the blind-HIP one (e.g. an NVIDIA
      dGPU next to an idle AMD iGPU, where this pairing would "correct" an
      already device-global reading with the WRONG adapter's usage). Strictly
      no-bus-AVAILABLE, never bus-CONTRADICTS: when torch DID answer a bus and
      it matched no ADL adapter, that is an affirmative mismatch (or a
      degraded ADL view - `_adl_used_by_bus` drops adapters whose usage query
      fails, so a 2-GPU box can transiently present only the wrong one), and
      the reading stays uncorrected exactly as before this rule existed.
    - PDH: used ONLY when there is exactly one GPU and exactly one adapter instance
      reporting. Its LUID instance name carries no PCI bus, so on a multi-GPU box the
      mapping would be a GUESS - and a confidently-wrong per-device number is worse
      than the blind one, so this reports nothing rather than guess (the caller then
      surfaces the reading as process-scoped instead of silently trusting it).
      (An iGPU+dGPU box reports TWO adapter instances here - observed live - which
      is exactly why the ADL torch-less rule above exists: ADL lists only AMD
      adapters, so the dGPU stays unambiguous where the LUID list is not.)"""
    if sys.platform != "win32" or not gpus:
        return {}
    with _lock:
        by_bus = _adl_used_by_bus()
        if by_bus:
            mapped = {}
            any_bus_answered = False
            for g in gpus:
                bus = _torch_pci_bus(g.get("index"))
                if bus is not None:
                    any_bus_answered = True
                    if bus in by_bus:
                        mapped[g["index"]] = by_bus[bus]
            if mapped:
                return mapped
            if (not any_bus_answered and len(by_bus) == 1 and len(gpus) == 1
                    and (raw_reading_is_process_scoped() or _gpu_is_amd(gpus[0]))):
                # Fire the single-adapter pairing when EITHER signal says the one
                # detected GPU is the one AMD adapter ADL sees:
                #   - raw_reading_is_process_scoped(): the platform's raw reading
                #     is the measured-blind HIP source (torch ROCm, or a resident
                #     bundled HIP runtime IN THIS process); or
                #   - _gpu_is_amd(gpus[0]): the detected GPU is itself an AMD card.
                # The second authorises the pairing where the first legitimately
                # answers False: a torch-less process (torch absent, HIP runtime
                # not resident because GGUF loads out-of-process, #606) on an AMD
                # box. The concrete caller is discover.vram_info's registry tier,
                # which on a GGUF-only install is the ONLY VRAM source (list_gpus()
                # is empty there) and passes the adapter's registry name so this
                # can recognise the card; that is what lets a torch-less build
                # recover a device-global free instead of showing total-only. This
                # gate alone does not change the meter - it is the authorisation
                # the vram_info wiring depends on. ADL enumerates ONLY AMD adapters,
                # so a single ADL adapter + a single detected AMD GPU is
                # unambiguous; a non-AMD detected GPU (the NVIDIA-dGPU-beside-an-
                # idle-AMD-iGPU hazard) still declines.
                only_bus, only_used = next(iter(by_bus.items()))
                logger.debug(
                    "gpu_usage: pairing the single AMD adapter (bus %d) with the "
                    "single requested GPU without a torch pci_bus_id - "
                    "unambiguous, same only-one-candidate rule as the PDH path",
                    only_bus)
                return {gpus[0]["index"]: only_used}
            logger.debug(
                "gpu_usage: ADL reported buses %s but none matched the detected "
                "GPUs' PCI bus ids; leaving the reading process-scoped rather than "
                "pairing them by position", sorted(by_bus))

        used = _pdh_adapter_used()
        if len(used) == 1 and len(gpus) == 1:
            return {gpus[0]["index"]: used[0]}
        if used:
            logger.debug(
                "gpu_usage: PDH reported %d adapter instances for %d detected GPUs; "
                "its LUID instance names carry no PCI bus, so the device mapping "
                "would be a guess - reporting unmeasurable instead", len(used), len(gpus))
        return {}


def _torch_pci_bus(index) -> Optional[int]:
    """The PCI bus number torch reports for device *index*, or None.

    ``pci_bus_id`` is what pairs a torch device with an ADL adapter; both are the
    physical bus, so the pairing is exact rather than positional.

    Never triggers a fresh ``import torch`` that is unsafe in this process
    (mid-flight probe, or the known-doomed resident-HIP-runtime conflict -
    the same two guards :func:`raw_reading_is_process_scoped` documents):
    before this guard, opening the correction gate in the GGUF worker would
    have made this the next site to fault with the 0xc0000139 stderr trace
    the #771 skip had just eliminated."""
    if index is None:
        return None
    try:
        if "torch" not in sys.modules:
            from localm import discover as _discover
            with _discover._gpu_probe_lock:
                inflight = _discover._gpu_probe_inflight
            if inflight or _discover._torch_gpu_probe_known_doomed():
                logger.debug("gpu_usage: no pci_bus_id for device %s: torch is "
                             "not consultable in this process", index)
                return None
        import torch
        props = torch.cuda.get_device_properties(int(index))
        bus = getattr(props, "pci_bus_id", None)
        return int(bus) if bus is not None else None
    except Exception as e:
        logger.debug("gpu_usage: no pci_bus_id for device %s: %s", index, e)
        return None
