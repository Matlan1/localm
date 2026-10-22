# SPDX-License-Identifier: AGPL-3.0-or-later
"""How long one GPU job may run before the device is reset, per operating
system and kernel driver: Windows from the adapter's compute preemption
granularity and the TDR registry values, Linux from the PCI device's driver and
its job timeout. Every reader is faked here except one real Windows adapter
query."""

from __future__ import annotations

import sys

import pytest

from localm.media.sdcpp import gpureset
from localm.media.sdcpp.gpureset import Adapter

DMA, DISPATCH, GROUP, THREAD, SHADER = 100, 200, 300, 400, 500


class FakeHost(gpureset.Host):
    """A machine: *platform*, TDR registry values, Windows adapters (or an
    OSError to raise), and a Linux file tree of *files* (path -> text) and
    *links* (path -> resolved path)."""

    def __init__(self, platform="win32", tdr=None, adapters=(), files=None, links=None,
                 release="6.8.0-45-generic"):
        self.platform = platform
        self.tdr = dict(tdr or {})
        self._adapters = adapters
        self.files = dict(files or {})
        self.links = dict(links or {})
        self.release = release

    def tdr_value(self, name):
        return self.tdr.get(name)

    def adapters(self):
        if isinstance(self._adapters, Exception):
            raise self._adapters
        return list(self._adapters)

    def read(self, path):
        return self.files.get(path)

    def realpath(self, path):
        if path in self.links:
            return self.links[path]
        if path in self.files or any(k.startswith(path + "/") for k in (*self.files, *self.links)):
            return path
        return None

    def listdir(self, path):
        names = set()
        for key in (*self.files, *self.links):
            if key.startswith(path + "/"):
                names.add(key[len(path) + 1:].split("/", 1)[0])
        return sorted(names)

    def kernel_release(self):
        return self.release


def dgpu(granularity, bus=0x2D, name="AMD Radeon RX 6900 XT"):
    return Adapter(bus, 0, 0, name, False, granularity)


BASIC_RENDER = Adapter(None, None, None, "", True, SHADER)
RX6900XT = {"name": "Vulkan0", "description": "AMD Radeon RX 6900 XT", "type": "gpu",
            "device_id": "0000:2d:00.0"}


def pin_rx6900xt(monkeypatch):
    """Pin the machine the native media tests run on to the reference box:
    Windows with an RX 6900 XT it can pause only between command buffers, the
    default TDR delay, and ROCm as the other GPU runtime."""
    from localm.media.sdcpp import runtime as sd_runtime
    monkeypatch.setattr(gpureset, "_HOST", FakeHost(adapters=[dgpu(DMA), BASIC_RENDER]))
    monkeypatch.setattr(sd_runtime, "other_gpu_runtimes", lambda det=None: ["ROCm"])


def limit(devices, **host):
    return gpureset.reset_limit(devices, FakeHost(**host))


# --------------------------------------------------------------------------- #
#  Windows                                                                     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("granularity", [0, DMA, DISPATCH])
def test_windows_resets_a_gpu_that_cannot_pause_mid_dispatch_after_the_tdr_delay(granularity):
    got = limit([RX6900XT], adapters=[dgpu(granularity), BASIC_RENDER])
    assert got.seconds == 2 and got.known and got.safe_seconds == 1
    assert got.reason.startswith("Windows resets this GPU when one job runs longer than 2 s")


@pytest.mark.parametrize("granularity", [GROUP, THREAD, SHADER])
def test_windows_does_not_reset_a_gpu_it_can_pause_mid_dispatch(granularity):
    got = limit([RX6900XT], adapters=[dgpu(granularity)])
    assert got.seconds is None and got.safe_seconds is None and got.known
    assert "can pause this GPU in the middle of a compute job" in got.reason


def test_the_dma_wording_names_command_buffers():
    assert "only between command buffers" in limit([RX6900XT], adapters=[dgpu(DMA)]).reason
    assert "only between compute dispatches" in limit([RX6900XT],
                                                      adapters=[dgpu(DISPATCH)]).reason


def test_tdr_level_off_means_no_reset():
    got = limit([RX6900XT], tdr={"TdrLevel": 0}, adapters=[dgpu(DMA)])
    assert got.seconds is None and got.known and "TdrLevel 0" in got.reason


@pytest.mark.parametrize("level", [1, 2, 3])
def test_tdr_levels_that_detect_keep_the_limit(level):
    assert limit([RX6900XT], tdr={"TdrLevel": level}, adapters=[dgpu(DMA)]).seconds == 2


def test_a_longer_tdr_delay_raises_the_limit():
    got = limit([RX6900XT], tdr={"TdrDelay": 8}, adapters=[dgpu(DMA)])
    assert got.seconds == 8 and got.safe_seconds == 4 and "longer than 8 s" in got.reason


def test_a_zero_tdr_delay_is_the_default():
    assert limit([RX6900XT], tdr={"TdrDelay": 0}, adapters=[dgpu(DMA)]).seconds == 2


def test_the_device_is_matched_by_its_pci_address():
    igpu = Adapter(0x00, 2, 0, "Intel(R) UHD Graphics 770", False, THREAD)
    adapters = [igpu, dgpu(DMA, bus=0x03)]
    on_dgpu = {"name": "Vulkan1", "description": "x", "type": "gpu", "device_id": "0000:03:00.0"}
    on_igpu = {"name": "Vulkan0", "description": "x", "type": "igpu", "device_id": "0000:00:02.0"}
    assert limit([on_dgpu], adapters=adapters).seconds == 2
    assert limit([on_igpu], adapters=adapters).seconds is None
    assert limit([on_igpu, on_dgpu], adapters=adapters).seconds == 2


def test_the_device_is_matched_by_name_without_a_pci_address():
    nvidia = Adapter(0x01, 0, 0, "NVIDIA GeForce RTX 4070", False, THREAD)
    amd = dgpu(DMA, bus=0x02)
    device = {"name": "Vulkan0", "description": "NVIDIA GeForce RTX 4070", "type": "gpu",
              "device_id": None}
    assert limit([device], adapters=[nvidia, amd]).seconds is None


def test_an_unmatched_device_gets_the_coarsest_adapter():
    adapters = [Adapter(0x01, 0, 0, "NVIDIA GeForce RTX 4070", False, THREAD), dgpu(DMA, bus=2)]
    unknown = {"name": "Vulkan0", "description": "Something", "type": "gpu", "device_id": None}
    assert limit([unknown], adapters=adapters).seconds == 2
    assert limit([], adapters=adapters).seconds == 2
    assert limit([], adapters=[adapters[0]]).seconds is None


def test_software_adapters_are_ignored():
    assert limit([], adapters=[BASIC_RENDER]).known is False


def test_a_failed_adapter_query_assumes_the_tdr_delay():
    got = limit([RX6900XT], tdr={"TdrDelay": 5}, adapters=OSError("D3DKMTEnumAdapters2 failed"))
    assert got.seconds == 5 and not got.known and "D3DKMTEnumAdapters2 failed" in got.reason


def test_an_adapter_without_preemption_caps_is_not_known():
    got = limit([RX6900XT], adapters=[dgpu(None)])
    assert got.seconds == 2 and not got.known


@pytest.mark.skipif(sys.platform != "win32", reason="Windows adapter query")
def test_the_real_windows_adapter_query_runs():
    adapters = gpureset.Host().adapters()
    assert adapters
    for a in adapters:
        assert a.compute_preemption is None or a.compute_preemption in (0, DMA, DISPATCH, GROUP,
                                                                        THREAD, SHADER)


# --------------------------------------------------------------------------- #
#  Linux                                                                       #
# --------------------------------------------------------------------------- #

PCI = "/sys/bus/pci/devices/0000:03:00.0"


def linux(driver, files=None, release="6.8.0-45-generic", card=True):
    links = {f"{PCI}/driver": f"/sys/bus/pci/drivers/{driver}"} if driver else {}
    tree = {PCI + "/vendor": "0x1002"}
    if card:
        links["/sys/class/drm/card1/device"] = PCI
        links["/sys/class/drm/card1-DP-1/device"] = PCI
    tree.update(files or {})
    return dict(platform="linux", files=tree, links=links, release=release)


ON_PCI = {"name": "Vulkan0", "description": "AMD Radeon RX 7900 XTX (RADV NAVI31)", "type": "gpu",
          "device_id": "0000:03:00.0"}


@pytest.mark.parametrize("kernel,param,expected", [
    ((6, 8), "", (10000, 60000)),
    ((5, 10), "", (10000, None)),
    ((6, 17), "", (10000, 10000)),
    ((6, 19), "", (2000, 2000)),
    ((7, 1), "", (2000, 2000)),
    ((6, 8), "5000", (5000, 60000)),
    ((6, 19), "5000", (5000, 5000)),
    ((6, 17), "5000", (5000, 10000)),
    ((6, 18), "-1", (None, 10000)),
    ((6, 18), "30000", (30000, 10000)),
    ((6, 19), "-1", (None, None)),
    ((6, 8), "0,0,0,0", (10000, 60000)),
    ((6, 8), "20000,30000,10000,10000", (20000, 30000)),
    ((6, 8), "0,-1", (10000, None)),
])
def test_amdgpu_timeouts_by_kernel_and_parameter(kernel, param, expected):
    assert gpureset.amdgpu_timeouts(kernel, param) == expected


@pytest.mark.parametrize("kernel,param,expected", [
    ((6, 8), "", (2000, 2000)),
    ((5, 10), "", (2000, 2000)),
    ((6, 8), "-1", (None, 2000)),
    ((6, 19), "30000", (30000, 2000)),
    ((6, 8), "20000,30000", (20000, 30000)),
])
def test_an_out_of_tree_amdgpu_takes_the_strictest_defaults(kernel, param, expected):
    assert gpureset.amdgpu_timeouts(kernel, param, out_of_tree=True) == expected


def test_amdgpu_timeouts_reject_a_bad_value():
    with pytest.raises(ValueError):
        gpureset.amdgpu_timeouts((6, 8), "abc")


@pytest.mark.parametrize("release,param,seconds", [
    ("6.8.0-45-generic", None, 10),
    ("6.19.2-arch1-1", None, 2),
    ("6.19.2-arch1-1", "8000", 8),
    ("5.4.0-150-generic", None, 10),
])
def test_amdgpu_resets_after_its_job_timeout(release, param, seconds):
    files = {gpureset._AMDGPU_PARAM: param} if param is not None else {}
    got = limit([ON_PCI], **linux("amdgpu", files, release))
    assert got.seconds == seconds and got.known
    assert f"amdgpu driver resets this GPU when one job runs longer than {seconds:g} s" \
        in got.reason


def test_amdgpu_with_the_timeout_off_has_no_limit():
    got = limit([ON_PCI], **linux("amdgpu", {gpureset._AMDGPU_PARAM: "-1"}, "6.19.0"))
    assert got.seconds is None and got.known


@pytest.mark.parametrize("release,param", [("", None), ("6.8.0", "abc")])
def test_amdgpu_without_a_readable_timeout_is_assumed(release, param):
    files = {gpureset._AMDGPU_PARAM: param} if param is not None else {}
    got = limit([ON_PCI], **linux("amdgpu", files, release))
    assert got.seconds == gpureset.ASSUMED_RESET_SECONDS and not got.known


@pytest.mark.parametrize("taint", ["", "E"])
def test_an_in_tree_amdgpu_module_keeps_the_kernel_default(taint):
    got = limit([ON_PCI], **linux("amdgpu", {gpureset._AMDGPU_TAINT: taint}, "6.8.0"))
    assert got.seconds == 10 and got.known


@pytest.mark.parametrize("param", [None, "-1"])
def test_an_out_of_tree_amdgpu_module_is_assumed_strict(param):
    files = {gpureset._AMDGPU_TAINT: "O"}
    if param is not None:
        files[gpureset._AMDGPU_PARAM] = param
    got = limit([ON_PCI], **linux("amdgpu", files, "6.8.0-45-generic"))
    assert got.seconds == 2 and not got.known
    assert "not built with the kernel" in got.reason


@pytest.mark.parametrize("param,known", [("20000,30000", True), ("20000", False),
                                         ("0,30000", False), ("20000,0", False)])
def test_an_out_of_tree_amdgpu_is_known_only_when_both_queues_are_set(param, known):
    files = {gpureset._AMDGPU_TAINT: "O", gpureset._AMDGPU_PARAM: param}
    got = limit([ON_PCI], **linux("amdgpu", files, "6.8.0"))
    assert got.known is known
    if known:
        assert got.seconds == 20 and "not built with the kernel" not in got.reason


class _Unreadable(FakeHost):
    def __init__(self, unreadable, **kw):
        super().__init__(**kw)
        self.unreadable = unreadable

    def read(self, path):
        return None if path == self.unreadable else super().read(path)


@pytest.mark.parametrize("path", [gpureset._AMDGPU_PARAM, gpureset._AMDGPU_TAINT])
def test_an_amdgpu_file_that_exists_but_cannot_be_read_is_assumed(path):
    kw = linux("amdgpu", {path: "x"}, "6.19.0")
    got = gpureset.reset_limit([ON_PCI], _Unreadable(path, **kw))
    assert got.seconds == gpureset.ASSUMED_RESET_SECONDS and not got.known


def test_a_short_pci_id_gets_its_domain():
    got = limit([dict(ON_PCI, device_id="03:00.0")], **linux("amdgpu", release="6.19.0"))
    assert got.seconds == 2 and got.known


I915 = "/sys/class/drm/card1/engine"


def test_i915_is_assumed_with_its_hang_detection_on():
    files = {f"{I915}/rcs0/heartbeat_interval_ms": "2500", f"{I915}/rcs0/preempt_timeout_ms": "640"}
    got = limit([ON_PCI], **linux("i915", files))
    assert got.seconds == 2 and not got.known and "i915 driver" in got.reason


def test_i915_with_hang_detection_off_has_no_limit():
    files = {f"{I915}/{e}/{k}": "0" for e in ("rcs0", "ccs0")
             for k in ("heartbeat_interval_ms", "preempt_timeout_ms")}
    files[f"{I915}/bcs0/heartbeat_interval_ms"] = "2500"
    got = limit([ON_PCI], **linux("i915", files))
    assert got.seconds is None and got.known


def test_i915_with_one_engine_still_detecting_keeps_the_limit():
    files = {f"{I915}/rcs0/heartbeat_interval_ms": "0", f"{I915}/rcs0/preempt_timeout_ms": "0",
             f"{I915}/ccs0/heartbeat_interval_ms": "2500", f"{I915}/ccs0/preempt_timeout_ms": "0"}
    assert limit([ON_PCI], **linux("i915", files)).seconds == 2


def test_an_i915_request_timeout_below_the_assumed_limit_is_used():
    files = {gpureset._I915_REQUEST_TIMEOUT: "1000",
             f"{I915}/rcs0/heartbeat_interval_ms": "0", f"{I915}/rcs0/preempt_timeout_ms": "0"}
    got = limit([ON_PCI], **linux("i915", files))
    assert got.seconds == 1 and not got.known


XE = "/sys/class/drm/card1/device/tile0/gt0/engines"


@pytest.mark.parametrize("job_timeout,seconds", [("5000", 2), ("1500", 1.5)])
def test_xe_is_assumed_and_bounded_by_its_job_timeout(job_timeout, seconds):
    files = {f"{XE}/rcs/job_timeout_ms": job_timeout, f"{XE}/ccs/job_timeout_ms": "5000",
             f"{XE}/bcs/job_timeout_ms": "100"}
    got = limit([ON_PCI], **linux("xe", files))
    assert got.seconds == seconds and not got.known and "xe driver" in got.reason


@pytest.mark.parametrize("driver", ["nvidia", "nouveau", "radeon", None])
def test_other_linux_drivers_are_assumed(driver):
    got = limit([ON_PCI], **linux(driver))
    assert got.seconds == gpureset.ASSUMED_RESET_SECONDS and not got.known


def test_without_a_device_every_linux_gpu_counts():
    host = linux("amdgpu", release="6.8.0")
    host["links"]["/sys/class/drm/card0/device"] = "/sys/bus/pci/devices/0000:00:02.0"
    host["links"]["/sys/bus/pci/devices/0000:00:02.0/driver"] = "/sys/bus/pci/drivers/i915"
    got = limit([], **host)
    assert got.seconds == 2 and not got.known
    only_amd = linux("amdgpu", release="6.8.0")
    assert limit([{"name": "Vulkan0", "type": "gpu", "device_id": None}], **only_amd).seconds == 10


# --------------------------------------------------------------------------- #
#  Everything else                                                             #
# --------------------------------------------------------------------------- #

def test_another_os_is_assumed():
    got = limit([ON_PCI], platform="darwin")
    assert got.seconds == gpureset.ASSUMED_RESET_SECONDS and not got.known
    assert "could not read" in got.reason


def test_a_cpu_device_is_never_reset():
    assert limit([{"name": "CPU", "type": "cpu"}], adapters=[dgpu(DMA)]).seconds is None
    assert limit([{"name": "CPU", "type": "cpu"}, RX6900XT], adapters=[dgpu(DMA)]).seconds == 2


def test_the_strictest_limit_wins_and_unknown_wins_a_tie():
    a = gpureset.ResetLimit(10.0, True, "a")
    b = gpureset.ResetLimit(2.0, True, "b")
    c = gpureset.ResetLimit(2.0, False, "c")
    none = gpureset.ResetLimit(None, True, "none")
    assert gpureset.strictest([a, b, none]) is b
    assert gpureset.strictest([b, c]) is c
    assert gpureset.strictest([none]) is none


def test_reset_limit_never_raises():
    class Broken(FakeHost):
        def tdr_value(self, name):
            raise RuntimeError("registry exploded")
    got = gpureset.reset_limit([RX6900XT], Broken())
    assert got.seconds == gpureset.ASSUMED_RESET_SECONDS and not got.known
    assert "registry exploded" in got.reason


def test_the_assumed_limit_is_the_one_the_rx6900xt_measurements_proved():
    assert gpureset.ASSUMED_RESET_SECONDS == gpureset.WINDOWS_DEFAULT_TDR_DELAY
    assert gpureset.ResetLimit(2.0, True, "").safe_seconds == 1.0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows registry")
def test_an_unreadable_or_mistyped_tdr_value_reads_as_unset(monkeypatch):
    import winreg

    class Key:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def denied(*a, **k):
        raise PermissionError("access denied")
    monkeypatch.setattr(winreg, "OpenKey", denied)
    assert gpureset.Host().tdr_value("TdrDelay") is None
    monkeypatch.setattr(winreg, "OpenKey", lambda *a, **k: Key())
    monkeypatch.setattr(winreg, "QueryValueEx", lambda key, name: ("8", winreg.REG_SZ))
    assert gpureset.Host().tdr_value("TdrDelay") is None
    monkeypatch.setattr(winreg, "QueryValueEx", lambda key, name: (8, winreg.REG_DWORD))
    assert gpureset.Host().tdr_value("TdrDelay") == 8
