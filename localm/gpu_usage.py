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
   counter Task Manager itself uses), via ``win32pdh`` from pywin32, already a
   dependency. Covers non-AMD Windows GPUs, where ADL does not exist. Its instance
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

_lock = threading.Lock()
_adl_state: Optional[dict] = None      # None = not tried yet; {} = tried and unusable
_pdh_state: Optional[dict] = None


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


_ADL_ALLOC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int)


def _adl_open() -> dict:
    """Open ADL once and cache the context, or cache {} when unusable (not an AMD
    box, no driver, an ADL that refuses to init). Never raises."""
    global _adl_state
    if _adl_state is not None:
        return _adl_state
    state: dict = {}
    try:
        dll = ctypes.WinDLL("atiadlxx.dll")

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
            _adl_state = {}
            return _adl_state
        state = {"dll": dll, "ctx": ctx, "alloc": _alloc, "keepalive": keepalive}
    except Exception as e:
        # No atiadlxx.dll (any non-AMD box) is the ordinary case, not a fault:
        # debug, not a warning, and the PDH path below still covers this box.
        logger.debug("gpu_usage: ADL unavailable: %s", e)
        _adl_state = {}
        return _adl_state
    _adl_state = state
    return state


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


def _pdh_adapter_used() -> list:
    """Device-global used bytes per WDDM adapter instance, via the vendor-neutral
    PDH counter, or [] when unavailable.

    The query handle is opened ONCE and reused: a persistent handle reads in ~0.02 ms,
    against ~887 ms to reopen one per call (and ~2278 ms for a `Get-Counter`
    subprocess), which is the difference between a probe that can sit on a request
    path and one that cannot.

    Instances are summed per adapter LUID. The ADAPTER counter is used rather than
    summing the per-process `\\GPU Process Memory` instances, because shared surfaces
    are counted against several processes and the sum overcounts (measured: 8.70 GB
    summed vs 7.33 GB actually on the adapter)."""
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


def source_is_warm() -> bool:
    """True once the source that would actually answer is open, so a further reading
    is effectively free.

    Exists because the FIRST open of a source is ~1000x the cost of every later read:
    opening ADL is a driver init MEASURED at ~750 ms, a cold PDH query is ~887 ms,
    against a ~0.02 ms warm read. That matters in one place - the caller's probe runs
    under a hard deadline, and a cold open is enough to push an otherwise-comfortable
    probe over it (measured: cold probes went from 2.9-3.5s to 3.6-4.0s against a
    4.0s cap, timing out). So the caller skips a COLD correction when its remaining
    budget is too thin, and never needs to skip a warm one.

    Gates on EITHER usable source being open, not ADL alone: on a non-AMD box ADL is
    proven-unusable ({}, falsy) after the first try but PDH is the source that will
    answer, and it may still be cold - so keying on ADL alone would call the source
    warm while a ~887 ms cold PDH open still lies ahead of the deadline. A truthy
    _adl_state (ADL open + usable) OR a truthy _pdh_state (PDH query open) means the
    next device_global_used_bytes() is cheap; both None/falsy means a cold open is
    still pending. See discover._apply_device_global_free."""
    return bool(_adl_state) or bool(_pdh_state)


def raw_reading_is_process_scoped() -> bool:
    """True when this platform's RAW driver free-VRAM query (torch.cuda.mem_get_info)
    is KNOWN to count only the calling process's own allocations - blind to every
    other process - so an uncorrected reading here must be tagged FREE_SCOPE_PROCESS
    rather than trusted or silently passed off as device-global.

    Measured only on Windows + an AMD ROCm/HIP torch build (see
    dev-notes/vram-cross-process-blindness.md). On Windows + a CUDA (NVIDIA) build,
    and on every non-Windows platform, cudaMemGetInfo is device-global BY
    DOCUMENTATION, so an uncorrected reading there is NOT known-blind: tagging it
    process-scoped would assert a blindness never measured and raise a spurious
    uncertainty flag on a number that is actually fine. Detected via
    ``torch.version.hip`` (set on ROCm builds, None on CUDA builds). torch absence
    (a GGUF-only install) reaches here only via nvidia-smi, which is device-global
    and already tagged as such upstream, so False is the safe answer there too."""
    import sys
    if sys.platform != "win32":
        return False
    try:
        import torch
        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return False


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
    - PDH: used ONLY when there is exactly one GPU and exactly one adapter instance
      reporting. Its LUID instance name carries no PCI bus, so on a multi-GPU box the
      mapping would be a GUESS - and a confidently-wrong per-device number is worse
      than the blind one, so this reports nothing rather than guess (the caller then
      surfaces the reading as process-scoped instead of silently trusting it)."""
    if sys.platform != "win32" or not gpus:
        return {}
    with _lock:
        by_bus = _adl_used_by_bus()
        if by_bus:
            mapped = {}
            for g in gpus:
                bus = _torch_pci_bus(g.get("index"))
                if bus is not None and bus in by_bus:
                    mapped[g["index"]] = by_bus[bus]
            if mapped:
                return mapped
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
    physical bus, so the pairing is exact rather than positional."""
    if index is None:
        return None
    try:
        import torch
        props = torch.cuda.get_device_properties(int(index))
        bus = getattr(props, "pci_bus_id", None)
        return int(bus) if bus is not None else None
    except Exception as e:
        logger.debug("gpu_usage: no pci_bus_id for device %s: %s", index, e)
        return None
