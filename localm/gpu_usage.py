# SPDX-License-Identifier: AGPL-3.0-or-later
"""Device-global VRAM usage on Windows: how much VRAM is in use ACROSS EVERY
PROCESS, which the GPU driver's own free-memory query does not report here.

On Windows with an AMD ROCm/HIP build, ``torch.cuda.mem_get_info()`` and
llama.cpp's ``ggml_backend_dev_memory`` both report

    free == total - (the CALLING HIP runtime instance's own allocations)

i.e. they are blind to every other process's VRAM. On Linux and on NVIDIA the
driver query is device-global, so this module is Windows-only and additive: it
corrects the reading here and returns nothing everywhere else.

SOURCES, in preference order:

1. AMD ADL (``atiadlxx.dll``, installed in System32 by the AMD driver itself -
   no new dependency, no admin). ``ADL2_Adapter_DedicatedVRAMUsage_Get`` is
   device-global. ADL also reports each adapter's PCI BUS NUMBER, which maps
   unambiguously onto torch's own ``get_device_properties(i).pci_bus_id``, so a
   multi-GPU box gets the right number per device.
2. PDH (the ``GPU Adapter Memory(*)`` object's ``Dedicated Usage`` counter, the
   vendor-neutral WDDM counter Task Manager itself uses), via ``win32pdh``, a
   Windows-gated dependency. Covers non-AMD Windows GPUs, where ADL does not
   exist. Its instance name is a LUID and carries NO PCI bus, so it CANNOT be
   mapped to a device index on a multi-GPU box - it is used ONLY when the
   mapping is unambiguous (exactly one GPU and one adapter instance).

The ADL context and the PDH query handle are opened ONCE and kept open. Every
entry point here is non-raising and returns "unmeasurable" rather than
propagating a driver failure into a caller that only wanted a number.
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
# ADL_PMLOG_INFO_ACTIVITY_GFX.
_ADL_PMLOG_ACTIVITY_GFX = 19

_lock = threading.Lock()
_adl_state: Optional[dict] = None      # None = not tried yet; {} = tried and unusable
_pdh_state: Optional[dict] = None
_pdh_util_state: Optional[dict] = None

# Source-selection notices already written this process, as {site: {key, ...}}.
#
# _notice_lock is a LEAF and must not be replaced by _lock: _lock is HELD across
# the pairing notice in device_global_used_bytes and across the process-scoped
# notice reached from it, and _lock is not reentrant.
_notice_lock = threading.Lock()
_notices_said: Dict[str, set] = {}
# Distinct keys a single site may announce before it stops. Reaching it is
# announced rather than going silent.
_NOTICE_KEY_CAP = 8
_notice_capped: set = set()


def _notice_once(site: str, key) -> bool:
    """Whether *site* should write its notice for *key* now: True the first time
    this exact key is seen this process, False on a repeat.

    A notice announces a SOURCE SELECTION, whose inputs do not change while the
    process runs, so a repeat says exactly what the log already says. The key is
    whatever part of the message varies (a reason, a device index, a bus number),
    so a DIFFERENT selection still gets announced instead of being swallowed by
    an earlier one.

    The cap bounds a site whose key varies without limit; crossing it writes one
    line saying so rather than silently going blind."""
    with _notice_lock:
        seen = _notices_said.setdefault(site, set())
        if key in seen:
            return False
        if len(seen) >= _NOTICE_KEY_CAP:
            if site in _notice_capped:
                return False
            _notice_capped.add(site)
            logger.debug(
                "gpu_usage: %s has announced %d distinct source selections this "
                "process; further distinct ones are suppressed", site,
                _NOTICE_KEY_CAP)
            return False
        seen.add(key)
        return True


def _reset_source_selection_notices() -> None:
    """Test hook: forget which source-selection notices have been written, so a
    notice one test provoked does not stay suppressed for every later test in the
    worker. Called by :func:`discover._reset_gpu_probe_cache`, which the test
    suite already runs around every test."""
    with _notice_lock:
        _notices_said.clear()
        _notice_capped.clear()


class _AdapterInfo(ctypes.Structure):
    """ADL's AdapterInfo. The field order and the ADL_MAX_PATH-sized char arrays are
    the ABI: a wrong layout silently marshals garbage."""
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
    """ADL's PMLogDataOutput. The sensor array is a FIXED 256 entries; a short
    array silently marshals garbage."""
    _fields_ = [("size", ctypes.c_int),
                ("sensors", _ADLSingleSensorData * _ADL_PMLOG_MAX_SENSORS)]


class _ADLPMActivity(ctypes.Structure):
    """ADL's ADLPMActivity, filled by ``ADL2_Overdrive5_CurrentActivity_Get``.

    Ten ints; ``iActivityPercent`` is the whole-GPU busy percent. ``iSize`` must
    be set to the struct size before the call. A wrong layout silently marshals
    garbage."""
    _fields_ = [("iSize", ctypes.c_int),
                ("iEngineClock", ctypes.c_int),
                ("iMemoryClock", ctypes.c_int),
                ("iVddc", ctypes.c_int),
                ("iActivityPercent", ctypes.c_int),
                ("iCurrentPerformanceLevel", ctypes.c_int),
                ("iCurrentBusSpeed", ctypes.c_int),
                ("iCurrentBusLanes", ctypes.c_int),
                ("iMaximumBusLanes", ctypes.c_int),
                ("iReserved", ctypes.c_int)]


class _ADLOD6CurrentStatus(ctypes.Structure):
    """ADL's ADLOD6CurrentStatus, filled by ``ADL2_Overdrive6_CurrentStatus_Get``.

    Nine ints; ``iActivityPercent`` is the whole-GPU busy percent. A wrong layout
    silently marshals garbage."""
    _fields_ = [("iEngineClock", ctypes.c_int),
                ("iMemoryClock", ctypes.c_int),
                ("iActivityPercent", ctypes.c_int),
                ("iCurrentPerformanceLevel", ctypes.c_int),
                ("iCurrentBusSpeed", ctypes.c_int),
                ("iCurrentBusLanes", ctypes.c_int),
                ("iMaximumBusLanes", ctypes.c_int),
                ("iExtValue", ctypes.c_int),
                ("iExtMask", ctypes.c_int)]


class _ADLODNPerformanceStatus(ctypes.Structure):
    """ADL's ADLODNPerformanceStatus, filled by
    ``ADL2_OverdriveN_PerformanceStatus_Get``.

    Eighteen ints; ``iGPUActivityPercent`` is the whole-GPU busy percent. A wrong
    layout silently marshals garbage."""
    _fields_ = [("iCoreClock", ctypes.c_int),
                ("iMemoryClock", ctypes.c_int),
                ("iDCEFClock", ctypes.c_int),
                ("iGFXClock", ctypes.c_int),
                ("iUVDClock", ctypes.c_int),
                ("iVCEClock", ctypes.c_int),
                ("iGPUActivityPercent", ctypes.c_int),
                ("iCurrentCorePerformanceLevel", ctypes.c_int),
                ("iCurrentMemoryPerformanceLevel", ctypes.c_int),
                ("iCurrentDCEFPerformanceLevel", ctypes.c_int),
                ("iCurrentGFXPerformanceLevel", ctypes.c_int),
                ("iUVDPerformanceLevel", ctypes.c_int),
                ("iVCEPerformanceLevel", ctypes.c_int),
                ("iCurrentBusSpeed", ctypes.c_int),
                ("iCurrentBusLanes", ctypes.c_int),
                ("iMaximumBusLanes", ctypes.c_int),
                ("iVDDC", ctypes.c_int),
                ("iVDDCI", ctypes.c_int)]


# Whole-GPU activity entry points predating PMLog, newest first, as
# (export, struct, activity field, whether iSize must be set before the call).
_ADL_LEGACY_ACTIVITY_SOURCES = (
    ("ADL2_OverdriveN_PerformanceStatus_Get", _ADLODNPerformanceStatus,
     "iGPUActivityPercent", False),
    ("ADL2_Overdrive6_CurrentStatus_Get", _ADLOD6CurrentStatus,
     "iActivityPercent", False),
    ("ADL2_Overdrive5_CurrentActivity_Get", _ADLPMActivity,
     "iActivityPercent", True),
)

_ADL_PMLOG_SOURCE = "ADL2_New_QueryPMLogData_Get"


_ADL_ALLOC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int)


def _adl_open() -> dict:
    """Open ADL once and cache the context. Returns the cached state, or {} when ADL
    is not usable this call. Never raises.

    Distinguishes PERMANENT from TRANSIENT unusability: a missing ``atiadlxx.dll``
    (any non-AMD box) is permanent and latched to a sticky {}, so the throwing load
    is never re-attempted; an ``ADL2_Main_Control_Create`` failure (the driver
    momentarily not answering) is NOT latched - it returns {} for this call but
    leaves ``_adl_state`` None so the next call retries."""
    global _adl_state
    if _adl_state is not None:
        return _adl_state
    try:
        dll = ctypes.WinDLL("atiadlxx.dll")
    except Exception as e:
        # A missing atiadlxx.dll is permanent: latch off and never retry.
        logger.debug("gpu_usage: ADL unavailable: %s", e)
        _adl_state = {}
        return _adl_state
    try:
        # ADL calls this back to allocate the AdapterInfo array. The buffers must
        # outlive the call, so they are parked on the cached state.
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
        # An unexpected init error is retryable: not latched.
        logger.debug("gpu_usage: ADL init failed; will retry: %s", e)
        return {}


def _adl_used_by_bus() -> Dict[int, int]:
    """``{pci_bus_number: device_global_used_bytes}`` for every present AMD adapter,
    or ``{}`` when ADL cannot answer.

    ADL reports several logical adapters per physical GPU, so entries are deduped
    on (bus, device, function) - the first that answers wins."""
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


def _adl_usable_pct(raw: float, source: str) -> Optional[float]:
    """*raw* when it is a percentage ADL could really have meant, else None.

    A value outside 0-100 means the field being read is not the one intended - a
    wrong sensor index or a struct-layout drift, both of which marshal garbage
    silently instead of raising. None yields no entry, never a fabricated 0%."""
    if 0.0 <= raw <= 100.0:
        return raw
    logger.debug("gpu_usage: %s returned %s, outside 0-100; ignoring", source, raw)
    return None


def _adl_pmlog_activity(dll, ctx, adapter_index: int) -> Optional[float]:
    """Whole-GPU busy percent for one adapter from ADL's PMLog sensor array, or
    None when this adapter cannot answer through it.

    Needs no ``PMLog_Start`` and no shared-memory session: called cold it answers
    immediately. None covers all three ways it declines - the call failing, the
    board not publishing the sensor, and a value outside 0-100."""
    data = _ADLPMLogDataOutput()
    if dll.ADL2_New_QueryPMLogData_Get(ctx, adapter_index, ctypes.byref(data)) != _ADL_OK:
        return None
    sensor = data.sensors[_ADL_PMLOG_ACTIVITY_GFX]
    if not sensor.supported:
        return None
    return _adl_usable_pct(float(sensor.value), _ADL_PMLOG_SOURCE)


def _adl_legacy_activity(dll, ctx, adapter_index: int):
    """``(percent, export_name)`` from the Overdrive activity APIs that predate
    PMLog, or None when none of them answers for this adapter.

    Tried newest first, and the first that returns a usable percentage wins.
    Reached only when :func:`_adl_pmlog_activity` declined, which is what an
    older driver, a pre-Overdrive-8 board, or a refused PMLog query looks like.

    An export the installed ``atiadlxx.dll`` does not have is skipped rather than
    raising, so an older driver missing one of these still gets the others."""
    for export, struct, field, needs_size in _ADL_LEGACY_ACTIVITY_SOURCES:
        fn = getattr(dll, export, None)
        if fn is None:
            continue
        data = struct()
        if needs_size:
            data.iSize = ctypes.sizeof(struct)
        try:
            rc = fn(ctx, adapter_index, ctypes.byref(data))
        except Exception as e:
            logger.debug("gpu_usage: %s raised: %s", export, e)
            continue
        if rc != _ADL_OK:
            continue
        pct = _adl_usable_pct(float(getattr(data, field)), export)
        if pct is not None:
            return pct, export
    return None


def _adl_activity_by_bus() -> Dict[int, float]:
    """``{pci_bus_number: whole_gpu_busy_percent}`` per present AMD adapter, or ``{}``.

    THE WHOLE-GPU FIGURE, whoever is causing the load - which is the one thing the
    WDDM ``GPU Engine`` counter behind :func:`adapter_utilisation` cannot give on
    this vendor.

    Each adapter is read through PMLog first and, when that declines, through the
    pre-PMLog Overdrive activity APIs. Both are whole-GPU sensors, so an adapter
    answering through either is reported the same way; which one answered is
    written to the debug log once per adapter and source.

    An adapter no source can answer for yields NO ENTRY rather than a fabricated
    0%, and one adapter failing never hides the rest.
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
            # ADL reports several logical adapters per physical card, so dedupe on
            # the PCI triple as _adl_used_by_bus does.
            key = (info.iBusNumber, info.iDeviceNumber, info.iFunctionNumber)
            if key in seen:
                continue
            pct = _adl_pmlog_activity(dll, ctx, info.iAdapterIndex)
            source = _ADL_PMLOG_SOURCE
            if pct is None:
                legacy = _adl_legacy_activity(dll, ctx, info.iAdapterIndex)
                if legacy is None:
                    continue
                pct, source = legacy
            seen.add(key)
            out[int(info.iBusNumber)] = pct
            if _notice_once("amd_activity", (int(info.iBusNumber), source)):
                logger.debug("gpu_usage: whole-GPU activity for AMD bus %d reads "
                             "through %s", int(info.iBusNumber), source)
        return out
    except Exception as e:
        logger.debug("gpu_usage: ADL activity query failed: %s", e)
        return {}


def amd_whole_gpu_activity() -> Optional[float]:
    """Whole-GPU busy percent on AMD, or None when this box cannot answer.

    None means "not measured", never "idle" - the caller must omit the field
    rather than render a 0%.

    The number is the fraction of TIME the graphics engine had work, which is what
    "GPU utilisation" means in AMD's control panel, GPU-Z and nvidia-smi alike. It
    is NOT how hard the card is working: it saturates, so it distinguishes BUSY
    from IDLE and never BUSY from FLAT OUT. Anything wanting "how much headroom is
    left" needs power or clock, not this.

    MULTI-CARD: the busiest AMD adapter wins, since localm's stats payload carries
    ONE system-wide ``gpu.percent``. A per-card breakdown belongs with the per-card
    VRAM rows, which key off the device list rather than a bus.
    """
    by_bus = _adl_activity_by_bus()
    if not by_bus:
        return None
    return round(max(by_bus.values()), 1)


def _pdh_adapter_used() -> list:
    """Device-global used bytes per WDDM adapter instance, via the vendor-neutral
    PDH counter, or [] when unavailable.

    The query handle is opened ONCE and reused; reopening one per call costs
    orders of magnitude more, which is the difference between a probe that can sit
    on a request path and one that cannot.

    Returns one value per adapter-LUID instance: the ``GPU Adapter Memory``
    instance name IS the adapter LUID, so each appears once and there is no
    cross-instance summing."""
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

        # Re-enumerate instances each call: an adapter can appear or disappear.
        _objs, instances = win32pdh_mod.EnumObjectItems(
            None, None, "GPU Adapter Memory", win32pdh.PERF_DETAIL_WIZARD)
        totals: Dict[str, int] = {}
        for inst in set(instances):
            key = inst
            if key not in counters:
                path = win32pdh_mod.MakeCounterPath(
                    (None, "GPU Adapter Memory", inst, None, -1, "Dedicated Usage"))
                # AddEnglishCounter: the counter path above is English, which a
                # localized Windows rejects through AddCounter.
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
        # The open query is kept and the next call retries. Only the win32pdh
        # ImportError above sets the sticky {}.
        logger.debug("gpu_usage: PDH query failed (will retry next call): %s", e)
        return []


def adapter_utilisation() -> Dict[str, float]:
    """GPU busy percentage per WDDM adapter LUID, or ``{}`` when unavailable.

    Vendor-neutral: Windows exposes the figure for every vendor through the same
    WDDM counters Task Manager reads.

    NOT THE RIGHT SOURCE ON AMD - USE :func:`amd_whole_gpu_activity` THERE FIRST.
    On AMD this number does not track the card's state: the fold below reports
    whichever engine type leads, which can be an unrelated engine (a video codec)
    while the graphics core is parked, and it equally misses work that does not
    land squarely on one countable engine. It is unreliable there rather than
    uniformly blind - pure compute does read through it.

    Kept because it is still the only vendor-neutral device-global source on
    Windows, and the ONLY one on Intel, where localm has no equivalent of ADL.

    Counter: ``GPU Engine`` / ``Utilization Percentage``, aggregated the way Task
    Manager aggregates it. Instances are per PROCESS and per ENGINE (3D, Copy,
    Video Codec, Compute, ...), so per-process values are summed WITHIN an engine
    type and the adapter's figure is then the BUSIEST engine type, not the sum
    across them. Summing across types double-counts work that ran concurrently on
    separate engines and routinely exceeds 100%.

    Keyed by adapter LUID, read from the instance name, so a multi-GPU board
    reports each card separately rather than one blended figure.

    UTILISATION IS A RATE, so it needs two collections separated in time. The
    query handle is kept open between calls, which makes the interval the gap
    between successive calls - on the stats poll, seconds. The FIRST call after
    the query opens has no previous sample to rate against and returns ``{}``
    rather than a fabricated 0%.
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
        # Keep the open query; the next call retries.
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

    The FIRST open of a source costs orders of magnitude more than every later read
    (opening ADL is a driver init; a cold PDH query is comparable), so a caller
    running under a short deadline uses this to skip a COLD correction while still
    taking a warm one.

    Gates on EITHER usable source being open, not ADL alone: on a non-AMD box ADL is
    proven-unusable ({}, falsy) after the first try while PDH is the source that will
    answer and may still be cold. A truthy _adl_state (ADL open and usable) OR a
    truthy _pdh_state (PDH query open) means the next device_global_used_bytes() is
    cheap; both None or falsy means a cold open is still pending."""
    return bool(_adl_state) or bool(_pdh_state)


def _known_blind_without_torch(reason: str) -> bool:
    """The no-torch answer for :func:`raw_reading_is_process_scoped`: True when
    the bundled HIP llama.cpp runtime is resident in this process
    (``discover.native_hip_runtime_resident()``), since every raw free-VRAM
    reading this process can then take is HIP-sourced, which is the blind source
    this module corrects. Where no HIP runtime is resident, False.

    *reason* says why torch could not be consulted, and is surfaced at debug."""
    from localm import discover as _discover
    resident = _discover.native_hip_runtime_resident()
    if resident and _notice_once("process-scoped", reason):
        logger.debug(
            "gpu_usage: raw VRAM readings in this process are process-scoped: "
            "torch is not consultable (%s) but the bundled HIP llama.cpp "
            "runtime is resident, which is itself the measured-blind source",
            reason)
    return resident


def raw_reading_is_process_scoped() -> bool:
    """True when this platform's RAW driver free-VRAM query (torch.cuda.mem_get_info)
    counts only the calling process's own allocations - blind to every other
    process - so an uncorrected reading here must be tagged FREE_SCOPE_PROCESS
    rather than trusted or passed off as device-global.

    True only on Windows with an AMD ROCm/HIP runtime. On Windows with a CUDA
    (NVIDIA) build, and on every non-Windows platform, cudaMemGetInfo is
    device-global by documentation, so the answer there is False.

    Detected via ``torch.version.hip`` (set on ROCm builds, None on CUDA builds)
    whenever torch can be consulted. When it CANNOT - torch is not resident and a
    fresh import is unsafe or impossible - the answer comes from
    :func:`_known_blind_without_torch` instead, i.e. from
    ``discover.native_hip_runtime_resident()``: a resident bundled HIP runtime
    means every raw reading this process can take (the in-process ggml query, or
    the isolated probe daemon loading the same runtime) is HIP-sourced. That is
    the case inside the GGUF worker, the process that makes the mid-generation
    context-grow sizing decision. Where no HIP runtime is resident either (a
    vulkan or cpu build's worker, a torch-less NVIDIA box), False.

    Never runs a plain ``import torch`` while a GPU probe may be mid-import
    (``discover._gpu_probe_inflight``, including an abandoned timed-out one) or
    while a fresh import is the known-doomed native-runtime DLL conflict
    (``discover._torch_gpu_probe_known_doomed``): a second thread blocking on
    CPython's per-module import lock behind such an import can hard-crash the
    process on this platform's ROCm native preload. In those two cases, and when
    a permitted fresh import itself fails, the resident-HIP-runtime signal
    answers. Reuses discover's probe-tracking lock. Never raises."""
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
                # The resident-runtime signal answers instead of importing torch.
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
    registry ``DriverDesc`` for :func:`discover.vram_info`'s registry tier, all of
    which read like "AMD Radeon RX 6900 XT" or "NVIDIA GeForce ...", so a substring
    test never false-positives NVIDIA or Intel. A missing or blank name answers
    False. Used only to authorise the single-adapter ADL fallback in
    :func:`device_global_used_bytes`."""
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
      exactly one GPU was asked about, the pairing is unambiguous without one.
      Gated on :func:`raw_reading_is_process_scoped` so it never fires on a
      box whose single detected GPU is NOT the blind-HIP one. Strictly
      no-bus-AVAILABLE, never bus-CONTRADICTS: when torch DID answer a bus and
      it matched no ADL adapter, the reading stays uncorrected.
    - PDH: used ONLY when there is exactly one GPU and exactly one adapter instance
      reporting. Its LUID instance name carries no PCI bus, so on a multi-GPU box the
      mapping would be a guess; this reports nothing instead, and the caller then
      surfaces the reading as process-scoped."""
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
                # Pair the single AMD adapter with the single detected GPU when
                # either the raw reading is the process-scoped HIP source or the
                # detected GPU is itself an AMD card.
                only_bus, only_used = next(iter(by_bus.items()))
                if _notice_once("adapter-pairing", only_bus):
                    logger.debug(
                        "gpu_usage: pairing the single AMD adapter (bus %d) with "
                        "the single requested GPU without a torch pci_bus_id - "
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

    ``pci_bus_id`` is the physical bus, the same quantity ADL reports, so the
    pairing with an ADL adapter is exact rather than positional.

    Never triggers a fresh ``import torch`` while a GPU probe is in flight or
    while the resident-HIP-runtime conflict makes the import known-doomed;
    returns None in those cases. Never raises.
    """
    if index is None:
        return None
    try:
        if "torch" not in sys.modules:
            from localm import discover as _discover
            with _discover._gpu_probe_lock:
                inflight = _discover._gpu_probe_inflight
            if inflight or _discover._torch_gpu_probe_known_doomed():
                if _notice_once("no-pci-bus-id", index):
                    logger.debug("gpu_usage: no pci_bus_id for device %s: torch "
                                 "is not consultable in this process", index)
                return None
        import torch
        props = torch.cuda.get_device_properties(int(index))
        bus = getattr(props, "pci_bus_id", None)
        return int(bus) if bus is not None else None
    except Exception as e:
        logger.debug("gpu_usage: no pci_bus_id for device %s: %s", index, e)
        return None
