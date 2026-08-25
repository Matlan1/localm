# SPDX-License-Identifier: AGPL-3.0-or-later
"""Device-global VRAM usage on Windows: how much VRAM is in use ACROSS EVERY PROCESS, which the GPU driver's own free-memory query does not tell us here."""

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
    """ADL's AdapterInfo."""
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
    """One PMLog sensor slot. ``supported`` is a flag, not a status code: a zero there means the board does not publish that sensor, and ``value`` is then meaningless rather than zero-valued."""
    _fields_ = [("supported", ctypes.c_int), ("value", ctypes.c_int)]


class _ADLPMLogDataOutput(ctypes.Structure):
    """ADL's PMLogDataOutput."""
    _fields_ = [("size", ctypes.c_int),
                ("sensors", _ADLSingleSensorData * _ADL_PMLOG_MAX_SENSORS)]


_ADL_ALLOC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int)


def _adl_open() -> dict:
    """Open ADL once and cache the context."""
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
    """``{pci_bus_number: device_global_used_bytes}`` for every present AMD adapter, or ``{}`` when ADL cannot answer."""
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
    """``{pci_bus_number: whole_gpu_busy_percent}`` per present AMD adapter, or ``{}``."""
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
    """Whole-GPU busy percent on AMD, or None when this box cannot answer."""
    by_bus = _adl_activity_by_bus()
    if not by_bus:
        return None
    return round(max(by_bus.values()), 1)


def _pdh_adapter_used() -> list:
    """Device-global used bytes per WDDM adapter instance, via the vendor-neutral PDH counter, or [] when unavailable."""
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
    """GPU busy percentage per WDDM adapter LUID, or ``{}`` when unavailable."""
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
    """The adapter LUID inside a ``GPU Engine`` instance name, or None."""
    if "luid_" not in instance:
        return None
    parts = instance.split("luid_", 1)[1].split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else None


def source_is_warm() -> bool:
    """True once the source that would actually answer is open, so a further reading is effectively free."""
    return bool(_adl_state) or bool(_pdh_state)


def _known_blind_without_torch(reason: str) -> bool:
    """The no-torch answer for :func:`raw_reading_is_process_scoped`: True when the bundled HIP llama.cpp runtime is resident in this process (``discover.native_hip_runtime_resident()``), because then every raw free-VRAM reading this process can take is HIP-sourced - the source whose Windows blindness is t..."""
    from localm import discover as _discover
    resident = _discover.native_hip_runtime_resident()
    if resident:
        logger.debug(
            "gpu_usage: raw VRAM readings in this process are process-scoped: "
            "torch is not consultable (%s) but the bundled HIP llama.cpp "
            "runtime is resident, which is itself the measured-blind source "
            "(dev-notes/vram-cross-process-blindness.md)", reason)
    return resident


def raw_reading_is_process_scoped() -> bool:
    """True when this platform's RAW driver free-VRAM query (torch.cuda.mem_get_info) is KNOWN to count only the calling process's own allocations - blind to every other process - so an uncorrected reading here must be tagged FREE_SCOPE_PROCESS rather than trusted or silently passed off as device-global."""
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
    """Whether a GPU entry is an AMD card, by its human name."""
    name = str(gpu.get("name") or "").lower()
    return "amd" in name or "radeon" in name


def device_global_used_bytes(gpus: list) -> Dict[int, int]:
    """``{gpu_index: device_global_used_bytes}`` for as many of *gpus* as can be mapped to a real adapter, or ``{}`` when this platform has no better source than the driver's own (already-correct, or unmeasurable) reading."""
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
    """The PCI bus number torch reports for device *index*, or None."""
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
