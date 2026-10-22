# SPDX-License-Identifier: AGPL-3.0-or-later
"""How long one GPU job may run before the operating system or the GPU's
kernel driver resets the device, for the devices a model computes on.

``reset_limit`` reads it per device: on Windows from the adapter's compute
preemption granularity (``D3DKMTQueryAdapterInfo``, ``KMTQAITYPE_WDDM_1_2_CAPS``)
and the ``TdrLevel`` / ``TdrDelay`` values under
``HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers``; on Linux from the
PCI device's kernel driver and that driver's job timeout (amdgpu
``lockup_timeout`` and its kernel-version default, xe ``job_timeout_ms``, i915
``request_timeout_ms`` and its engines' hang detection). Every read is read
only. A device whose limit cannot be read gets ``ASSUMED_RESET_SECONDS`` with
``known`` False."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

ASSUMED_RESET_SECONDS = 2.0
SAFE_FRACTION = 0.5

WINDOWS_DEFAULT_TDR_DELAY = 2
_TDR_LEVEL_OFF = 0

COMPUTE_PREEMPTION_NONE = 0
COMPUTE_PREEMPTION_DMA_BUFFER = 100
COMPUTE_PREEMPTION_DISPATCH = 200
COMPUTE_PREEMPTION_THREAD_GROUP = 300
COMPUTE_PREEMPTION_THREAD = 400
COMPUTE_PREEMPTION_SHADER = 500

_KMTQAITYPE_ADAPTERADDRESS = 6
_KMTQAITYPE_ADAPTERREGISTRYINFO = 8
_KMTQAITYPE_ADAPTERTYPE = 15
_KMTQAITYPE_WDDM_1_2_CAPS = 17
_ADAPTERTYPE_RENDER = 0x1
_ADAPTERTYPE_SOFTWARE = 0x4

_PCI_ID_RE = re.compile(r"\A(?:[0-9a-fA-F]{4,8}:)?([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])\Z")
_KERNEL_RE = re.compile(r"\A(\d+)\.(\d+)")

_AMDGPU_PARAM = "/sys/module/amdgpu/parameters/lockup_timeout"
_AMDGPU_TAINT = "/sys/module/amdgpu/taint"
_I915_REQUEST_TIMEOUT = "/sys/module/i915/parameters/request_timeout_ms"
_DRM_CLASS = "/sys/class/drm"
_PCI_DEVICES = "/sys/bus/pci/devices"
_INTEL_ENGINE_PREFIXES = ("rcs", "ccs")


@dataclass(frozen=True)
class Adapter:
    """A Windows graphics adapter: its PCI bus, device and function numbers
    (None when it has none), its name, whether it is a software or
    display-only adapter, and its compute preemption granularity
    (``COMPUTE_PREEMPTION_*``, None when the adapter did not report it)."""
    bus: Optional[int]
    device: Optional[int]
    function: Optional[int]
    name: str
    software: bool
    compute_preemption: Optional[int]


@dataclass(frozen=True)
class ResetLimit:
    """*seconds*: one GPU job running longer than this can reset the device;
    None when no job length resets it. *known*: False when localm could not
    read it and *seconds* is ``ASSUMED_RESET_SECONDS``. *reason*: one sentence,
    without a final period, saying what resets the device and when."""
    seconds: Optional[float]
    known: bool
    reason: str

    @property
    def safe_seconds(self) -> Optional[float]:
        """The longest single job localm runs on the device:
        ``SAFE_FRACTION`` of *seconds*, or None without a limit."""
        return None if self.seconds is None else self.seconds * SAFE_FRACTION


class Host:
    """The operating-system reads ``reset_limit`` makes. Tests pass a fake."""

    platform = sys.platform

    def tdr_value(self, name: str) -> Optional[int]:
        """The ``GraphicsDrivers`` DWORD *name*, or None when it is not set
        or cannot be read."""
        if sys.platform != "win32":
            return None
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers") as key:
                value, kind = winreg.QueryValueEx(key, name)
        except (OSError, ImportError):
            return None
        return int(value) if kind == 4 else None

    def adapters(self) -> list[Adapter]:
        """Every Windows graphics adapter. Raises OSError when the query fails
        or this is not Windows."""
        return _windows_adapters()

    def read(self, path: str) -> Optional[str]:
        """The stripped text of file *path*, or None when it cannot be read."""
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return None

    def realpath(self, path: str) -> Optional[str]:
        """The resolved *path*, or None when it does not exist."""
        try:
            return os.path.realpath(path) if os.path.exists(path) else None
        except OSError:
            return None

    def listdir(self, path: str) -> list[str]:
        """The sorted entries of directory *path*, empty when it cannot be listed."""
        try:
            return sorted(os.listdir(path))
        except OSError:
            return []

    def kernel_release(self) -> str:
        """``uname -r``, or an empty string."""
        try:
            return os.uname().release
        except AttributeError:
            return ""


_HOST = Host()


def _windows_adapters() -> list[Adapter]:
    if sys.platform != "win32":
        raise OSError("not Windows")
    import ctypes
    from ctypes import wintypes

    gdi = ctypes.WinDLL("gdi32")

    class _Luid(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    class _AdapterInfo(ctypes.Structure):
        _fields_ = [("hAdapter", ctypes.c_uint), ("AdapterLuid", _Luid),
                    ("NumOfSources", wintypes.ULONG), ("bPrecisePresentRegionsPreferred", wintypes.BOOL)]

    class _EnumAdapters2(ctypes.Structure):
        _fields_ = [("NumAdapters", wintypes.ULONG), ("pAdapters", ctypes.POINTER(_AdapterInfo))]

    class _QueryAdapterInfo(ctypes.Structure):
        _fields_ = [("hAdapter", ctypes.c_uint), ("Type", ctypes.c_int),
                    ("pPrivateDriverData", ctypes.c_void_p), ("PrivateDriverDataSize", ctypes.c_uint)]

    class _Wddm12Caps(ctypes.Structure):
        _fields_ = [("GraphicsPreemptionGranularity", ctypes.c_int),
                    ("ComputePreemptionGranularity", ctypes.c_int), ("Value", ctypes.c_uint)]

    class _AdapterAddress(ctypes.Structure):
        _fields_ = [("BusNumber", ctypes.c_uint), ("DeviceNumber", ctypes.c_uint),
                    ("FunctionNumber", ctypes.c_uint)]

    class _RegistryInfo(ctypes.Structure):
        _fields_ = [("AdapterString", ctypes.c_wchar * 260), ("BiosString", ctypes.c_wchar * 260),
                    ("DacType", ctypes.c_wchar * 260), ("ChipType", ctypes.c_wchar * 260)]

    class _CloseAdapter(ctypes.Structure):
        _fields_ = [("hAdapter", ctypes.c_uint)]

    def query(handle: int, kind: int, out) -> bool:
        q = _QueryAdapterInfo(handle, kind, ctypes.cast(ctypes.byref(out), ctypes.c_void_p),
                              ctypes.sizeof(out))
        return gdi.D3DKMTQueryAdapterInfo(ctypes.byref(q)) == 0

    listing = _EnumAdapters2(0, None)
    status = gdi.D3DKMTEnumAdapters2(ctypes.byref(listing))
    if status != 0:
        raise OSError(f"D3DKMTEnumAdapters2 failed (0x{status & 0xFFFFFFFF:08x})")
    infos = (_AdapterInfo * max(1, listing.NumAdapters))()
    listing.pAdapters = infos
    status = gdi.D3DKMTEnumAdapters2(ctypes.byref(listing))
    if status != 0:
        raise OSError(f"D3DKMTEnumAdapters2 failed (0x{status & 0xFFFFFFFF:08x})")
    out = []
    try:
        for i in range(listing.NumAdapters):
            handle = infos[i].hAdapter
            caps = _Wddm12Caps()
            has_caps = query(handle, _KMTQAITYPE_WDDM_1_2_CAPS, caps)
            addr = _AdapterAddress()
            has_addr = query(handle, _KMTQAITYPE_ADAPTERADDRESS, addr) and addr.BusNumber < 256
            reg = _RegistryInfo()
            name = reg.AdapterString if query(handle, _KMTQAITYPE_ADAPTERREGISTRYINFO, reg) else ""
            kind = ctypes.c_uint(0)
            flags = kind.value if query(handle, _KMTQAITYPE_ADAPTERTYPE, kind) else None
            out.append(Adapter(
                bus=addr.BusNumber if has_addr else None,
                device=addr.DeviceNumber if has_addr else None,
                function=addr.FunctionNumber if has_addr else None,
                name=str(name).strip(),
                software=flags is not None and (bool(flags & _ADAPTERTYPE_SOFTWARE)
                                                or not flags & _ADAPTERTYPE_RENDER),
                compute_preemption=int(caps.ComputePreemptionGranularity) if has_caps else None))
    finally:
        for i in range(listing.NumAdapters):
            gdi.D3DKMTCloseAdapter(ctypes.byref(_CloseAdapter(infos[i].hAdapter)))
    return out


def _assumed(detail: str) -> ResetLimit:
    return ResetLimit(ASSUMED_RESET_SECONDS, False,
                      f"localm could not read how long this GPU may run one job before it is "
                      f"reset ({detail}), so it assumes about {ASSUMED_RESET_SECONDS:g} s")


def _pci(device_id: Optional[str]) -> Optional[tuple[int, int, int]]:
    m = _PCI_ID_RE.match((device_id or "").strip())
    return (int(m.group(1), 16), int(m.group(2), 16), int(m.group(3), 16)) if m else None


_PREEMPTION_WORDS = {
    COMPUTE_PREEMPTION_NONE: "cannot pause this GPU's compute work",
    COMPUTE_PREEMPTION_DMA_BUFFER: "can pause this GPU's compute work only between command buffers",
    COMPUTE_PREEMPTION_DISPATCH: "can pause this GPU's compute work only between compute dispatches",
}


def _windows(device: Optional[dict], host: Host) -> ResetLimit:
    if host.tdr_value("TdrLevel") == _TDR_LEVEL_OFF:
        return ResetLimit(None, True, "Windows GPU timeout detection is off (TdrLevel 0)")
    delay = host.tdr_value("TdrDelay")
    delay = delay if delay and delay > 0 else WINDOWS_DEFAULT_TDR_DELAY
    try:
        adapters = [a for a in host.adapters() if not a.software]
    except OSError as e:
        return ResetLimit(float(delay), False,
                          f"localm could not read whether Windows can pause this GPU in the middle of "
                          f"a compute job ({e}), so it assumes Windows resets it when one job runs "
                          f"longer than {delay:g} s")
    pci = _pci((device or {}).get("device_id"))
    name = str((device or {}).get("description") or "").strip().casefold()
    picked = [a for a in adapters if pci and (a.bus, a.device, a.function) == pci]
    if not picked and name:
        picked = [a for a in adapters if a.name.casefold() == name]
    if not picked:
        picked = adapters
    if not picked:
        return ResetLimit(float(delay), False,
                          f"localm could not find this GPU among Windows' graphics adapters, so it "
                          f"assumes Windows resets it when one job runs longer than {delay:g} s")
    if any(a.compute_preemption is None for a in picked):
        return ResetLimit(float(delay), False,
                          f"Windows did not report whether it can pause this GPU in the middle of a "
                          f"compute job, so localm assumes Windows resets it when one job runs "
                          f"longer than {delay:g} s")
    granularity = min(a.compute_preemption or 0 for a in picked)
    if granularity >= COMPUTE_PREEMPTION_THREAD_GROUP:
        return ResetLimit(None, True, "Windows can pause this GPU in the middle of a compute job")
    words = _PREEMPTION_WORDS.get(granularity, _PREEMPTION_WORDS[COMPUTE_PREEMPTION_NONE])
    return ResetLimit(float(delay), True,
                      f"Windows resets this GPU when one job runs longer than {delay:g} s, "
                      f"because it {words}")


def _kernel_version(host: Host) -> Optional[tuple[int, int]]:
    m = _KERNEL_RE.match(host.kernel_release() or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def amdgpu_timeouts(kernel: tuple[int, int], param: str,
                    out_of_tree: bool = False) -> tuple[Optional[int], Optional[int]]:
    """The amdgpu GFX and compute job timeouts in milliseconds (None for no
    timeout) on Linux *kernel* (major, minor) with ``lockup_timeout`` set to
    *param*: 2000 for every queue from 6.19, 10000 from 6.17, earlier 10000 for
    GFX and 60000 for compute (none before 5.14). A value of 0 keeps the
    default, a negative one disables the timeout; one value sets every queue
    from 6.19 and only the non-compute queues before (on bare metal). With
    *out_of_tree* (a module not built with the kernel) the defaults are the
    strictest of any version, 2000 for every queue, and one value sets only
    the GFX queue. Raises ValueError for a value that is not an integer."""
    if out_of_tree or kernel >= (6, 19):
        gfx, compute = 2000, 2000
    elif kernel >= (6, 17):
        gfx, compute = 10000, 10000
    elif kernel >= (5, 14):
        gfx, compute = 10000, 60000
    else:
        gfx, compute = 10000, None
    values = [int(v, 0) for v in (p.strip() for p in (param or "").split(",")) if v]

    def apply(old: Optional[int], value: int) -> Optional[int]:
        return old if value == 0 else (None if value < 0 else value)

    if len(values) == 1:
        gfx = apply(gfx, values[0])
        if kernel >= (6, 19) and not out_of_tree:
            compute = apply(compute, values[0])
    elif values:
        gfx = apply(gfx, values[0])
        if len(values) > 1:
            compute = apply(compute, values[1])
    return gfx, compute


def _amdgpu(host: Host) -> ResetLimit:
    kernel = _kernel_version(host)
    if kernel is None:
        return _assumed("unknown Linux kernel version")
    param = host.read(_AMDGPU_PARAM)
    if param is None and host.realpath(_AMDGPU_PARAM) is not None:
        return _assumed("unreadable amdgpu lockup_timeout")
    taint = host.read(_AMDGPU_TAINT)
    if taint is None and host.realpath(_AMDGPU_TAINT) is not None:
        return _assumed("unreadable amdgpu module taint")
    out_of_tree = "O" in (taint or "")
    try:
        gfx, compute = amdgpu_timeouts(kernel, param or "", out_of_tree)
    except ValueError:
        return _assumed("unreadable amdgpu lockup_timeout")
    limits = [ms for ms in (gfx, compute) if ms is not None]
    if not limits:
        return ResetLimit(None, True, "the amdgpu driver's GPU job timeout is off")
    seconds = min(limits) / 1000
    if out_of_tree:
        return ResetLimit(seconds, False,
                          "this amdgpu module was not built with the kernel, so localm cannot "
                          f"tell its default job timeout and assumes about {seconds:g} s")
    return ResetLimit(seconds, True,
                      f"the amdgpu driver resets this GPU when one job runs longer than {seconds:g} s")


def _ints(values: Iterable[Optional[str]]) -> list[int]:
    out = []
    for v in values:
        if v is None:
            continue
        try:
            out.append(int(v))
        except ValueError:
            continue
    return out


def _intel(driver: str, card: Optional[str], host: Host) -> ResetLimit:
    hard: list[int] = []
    detection_off = False
    if driver == "xe" and card:
        dev = f"{card}/device"
        for tile in (t for t in host.listdir(dev) if t.startswith("tile")):
            for gt in (g for g in host.listdir(f"{dev}/{tile}") if g.startswith("gt")):
                engines = f"{dev}/{tile}/{gt}/engines"
                for cls in (c for c in host.listdir(engines) if c in _INTEL_ENGINE_PREFIXES):
                    hard += [ms for ms in _ints([host.read(f"{engines}/{cls}/job_timeout_ms")])
                             if ms > 0]
    elif driver == "i915":
        hard += [ms for ms in _ints([host.read(_I915_REQUEST_TIMEOUT)]) if ms > 0]
        if card:
            engines = [e for e in host.listdir(f"{card}/engine")
                       if e.startswith(_INTEL_ENGINE_PREFIXES)]
            knobs = _ints(host.read(f"{card}/engine/{e}/{knob}") for e in engines
                          for knob in ("heartbeat_interval_ms", "preempt_timeout_ms"))
            detection_off = bool(engines) and len(knobs) == 2 * len(engines) and not any(knobs)
    if detection_off and not hard:
        return ResetLimit(None, True, "the i915 driver's GPU hang detection is off")
    seconds = min([ASSUMED_RESET_SECONDS] + [ms / 1000 for ms in hard])
    return ResetLimit(seconds, False,
                      f"localm could not read whether the {driver} driver can pause this GPU in the "
                      f"middle of a compute job, so it assumes the GPU is reset when one job runs "
                      f"longer than {seconds:g} s")


def _linux_driver(pci_path: str, host: Host) -> Optional[str]:
    driver = host.realpath(f"{pci_path}/driver")
    return driver.replace("\\", "/").rsplit("/", 1)[-1] if driver else None


def _drm_card(pci_path: str, host: Host) -> Optional[str]:
    target = host.realpath(pci_path)
    for name in host.listdir(_DRM_CLASS):
        if re.fullmatch(r"card\d+", name) and target and \
                host.realpath(f"{_DRM_CLASS}/{name}/device") == target:
            return f"{_DRM_CLASS}/{name}"
    return None


def _linux_one(pci_path: str, host: Host) -> ResetLimit:
    driver = _linux_driver(pci_path, host)
    if driver == "amdgpu":
        return _amdgpu(host)
    if driver in ("i915", "xe"):
        return _intel(driver, _drm_card(pci_path, host), host)
    if driver == "nvidia":
        return _assumed("the NVIDIA driver on Linux does not report it")
    return _assumed(f"Linux GPU driver {driver}" if driver else "no Linux GPU driver found")


def _linux(device: Optional[dict], host: Host) -> ResetLimit:
    device_id = str((device or {}).get("device_id") or "").strip().lower()
    if _pci(device_id):
        if len(device_id.split(":")) == 2:
            device_id = "0000:" + device_id
        return _linux_one(f"{_PCI_DEVICES}/{device_id}", host)
    cards = [host.realpath(f"{_DRM_CLASS}/{n}/device") for n in host.listdir(_DRM_CLASS)
             if re.fullmatch(r"card\d+", n)]
    limits = [_linux_one(c, host) for c in dict.fromkeys(cards) if c and _linux_driver(c, host)]
    return strictest(limits) if limits else _assumed("the GPU has no PCI address")


def _for_device(device: Optional[dict], host: Host) -> ResetLimit:
    if device is not None and str(device.get("type") or "").lower() == "cpu":
        return ResetLimit(None, True, "the model runs on the CPU")
    if host.platform == "win32":
        return _windows(device, host)
    if host.platform.startswith("linux"):
        return _linux(device, host)
    return _assumed(f"no reader for {host.platform}")


def strictest(limits: list[ResetLimit]) -> ResetLimit:
    """The limit of *limits* with the fewest seconds (no limit counts as
    endless); between equal ones, one localm could not read."""
    return min(limits, key=lambda x: (float("inf") if x.seconds is None else x.seconds, x.known))


def reset_limit(devices: Optional[list], host: Optional[Host] = None) -> ResetLimit:
    """The strictest ``ResetLimit`` over *devices* (dicts with ``name``,
    ``description``, ``type`` and ``device_id``, as the sd.cpp worker's load
    info ``compute_devices``). An empty or missing list means the devices are
    not known: every GPU of the machine is considered. Never raises."""
    host = host or _HOST
    try:
        if not devices:
            return _for_device(None, host)
        return strictest([_for_device(d, host) for d in devices])
    except Exception as e:  # noqa: BLE001
        return _assumed(f"{type(e).__name__}: {e}")
