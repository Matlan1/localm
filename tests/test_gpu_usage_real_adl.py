# SPDX-License-Identifier: AGPL-3.0-or-later
"""gpu_usage's ADL ctypes layouts, checked against AMD's published headers and,
where one is installed, against the real ``atiadlxx.dll``.

The fake ADL in test_sysstats.py writes production's own ctypes classes by field
name, so it confirms that gpu_usage reads the field the fake wrote and nothing
about whether the offsets, sizes and stride agree with what the driver writes.
This file closes that gap in two layers:

* ``TestAdlLayoutsMatchPublishedHeaders`` runs everywhere and pins each struct's
  field order, ``ctypes.sizeof`` and the sensor constants to the values in AMD's
  public ``adl_structures.h`` / ``adl_defines.h``.
* ``TestRealAdlDriverAgreesWithDeclaredLayouts`` (integration, ``real_amd_adl``)
  opens the installed driver through the production ``_adl_open`` and checks the
  declared layouts against what the driver itself reports, validates, allocates
  and fills, then reads live whole-GPU activity through the production path.
  It skips, never fails, when the DLL does not load, the driver refuses a
  context, or ADL enumerates no adapter. Once adapters are enumerated, a layout
  that disagrees with the driver is a failure.

Known blind spot of both layers: a sensor index that is wrong by one still lands
on a 0-100 percentage (``ADL_PMLOG_INFO_ACTIVITY_MEM`` sits right after
``ADL_PMLOG_INFO_ACTIVITY_GFX``), which only an idle-versus-load differential
could catch. Nothing here loads a model or puts load on the GPU.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from localm import gpu_usage as gu

# Transcribed from AMD's public display-library headers (adl_structures.h,
# adl_defines.h); the ints are 4 bytes and ADL_MAX_PATH is 256.
_HEADER_FIELDS = {
    "_AdapterInfo": [
        "iSize", "iAdapterIndex", "strUDID", "iBusNumber", "iDeviceNumber",
        "iFunctionNumber", "iVendorID", "strAdapterName", "strDisplayName",
        "iPresent", "iExist", "strDriverPath", "strDriverPathExt",
        "strPNPString", "iOSDisplayIndex",
    ],
    "_ADLSingleSensorData": ["supported", "value"],
    "_ADLPMLogDataOutput": ["size", "sensors"],
    "_ADLPMActivity": [
        "iSize", "iEngineClock", "iMemoryClock", "iVddc", "iActivityPercent",
        "iCurrentPerformanceLevel", "iCurrentBusSpeed", "iCurrentBusLanes",
        "iMaximumBusLanes", "iReserved",
    ],
    "_ADLOD6CurrentStatus": [
        "iEngineClock", "iMemoryClock", "iActivityPercent",
        "iCurrentPerformanceLevel", "iCurrentBusSpeed", "iCurrentBusLanes",
        "iMaximumBusLanes", "iExtValue", "iExtMask",
    ],
    "_ADLODNPerformanceStatus": [
        "iCoreClock", "iMemoryClock", "iDCEFClock", "iGFXClock", "iUVDClock",
        "iVCEClock", "iGPUActivityPercent", "iCurrentCorePerformanceLevel",
        "iCurrentMemoryPerformanceLevel", "iCurrentDCEFPerformanceLevel",
        "iCurrentGFXPerformanceLevel", "iUVDPerformanceLevel",
        "iVCEPerformanceLevel", "iCurrentBusSpeed", "iCurrentBusLanes",
        "iMaximumBusLanes", "iVDDC", "iVDDCI",
    ],
}

_HEADER_SIZES = {
    "_AdapterInfo": 9 * 4 + 6 * 256,
    "_ADLSingleSensorData": 2 * 4,
    "_ADLPMLogDataOutput": 4 + 256 * 8,
    "_ADLPMActivity": 10 * 4,
    "_ADLOD6CurrentStatus": 9 * 4,
    "_ADLODNPerformanceStatus": 18 * 4,
}

_POISON = 0xEE


class TestAdlLayoutsMatchPublishedHeaders:
    """Hardware-free pins of every ADL struct and constant gpu_usage declares to
    AMD's published header values."""

    @pytest.mark.parametrize("name", sorted(_HEADER_FIELDS))
    def test_field_order_matches_the_header(self, name):
        struct = getattr(gu, name)
        assert [f for f, _t in struct._fields_] == _HEADER_FIELDS[name]

    @pytest.mark.parametrize("name", sorted(_HEADER_SIZES))
    def test_sizeof_matches_the_header(self, name):
        assert ctypes.sizeof(getattr(gu, name)) == _HEADER_SIZES[name]

    def test_char_arrays_are_adl_max_path_wide(self):
        for field, ftype in gu._AdapterInfo._fields_:
            if field.startswith("str"):
                assert ftype._type_ is ctypes.c_char, field
                assert ftype._length_ == 256, field

    def test_sensor_constants_match_the_header(self):
        assert gu._ADL_MAX_PATH == 256
        assert gu._ADL_PMLOG_MAX_SENSORS == 256
        assert gu._ADL_PMLOG_ACTIVITY_GFX == 19
        assert gu._ADL_VENDOR_AMD == 1002
        assert gu._ADL_OK == 0

    def test_legacy_sources_name_each_structs_activity_field(self):
        for export, struct, field, needs_size in gu._ADL_LEGACY_ACTIVITY_SOURCES:
            assert export.startswith("ADL2_")
            assert field in _HEADER_FIELDS[struct.__name__]
            assert field.endswith("ActivityPercent")
            assert needs_size == ("iSize" in _HEADER_FIELDS[struct.__name__])


def _poison(obj) -> None:
    """Fill *obj*'s bytes with a pattern no real ADL value takes, so a field the
    driver did not write is distinguishable from a written zero."""
    ctypes.memset(ctypes.addressof(obj), _POISON, ctypes.sizeof(obj))


def _decode(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("latin-1")


@pytest.fixture(scope="module")
def adl():
    """A live ADL context opened through the production ``_adl_open``, with the
    adapter table read the way ``_adl_activity_by_bus`` reads it.

    Skips the module when ADL cannot be used at all: not Windows, the DLL does
    not load, the driver refuses a context, or no adapter is enumerated. Restores
    ``gpu_usage._adl_state`` and destroys the driver context on teardown, so no
    real context outlives the module and no latched state leaks into other tests.
    """
    if sys.platform != "win32":
        pytest.skip("real_amd_adl: ADL exists only on Windows")
    prior = gu._adl_state
    gu._adl_state = None
    try:
        state = gu._adl_open()
        if not state:
            gu._adl_state = None
            pytest.skip("real_amd_adl: atiadlxx.dll not loadable or the driver "
                        "refused an ADL context")
        dll, ctx = state["dll"], state["ctx"]
        n = ctypes.c_int(0)
        rc = dll.ADL2_Adapter_NumberOfAdapters_Get(ctx, ctypes.byref(n))
        if rc != gu._ADL_OK or n.value <= 0:
            dll.ADL2_Main_Control_Destroy(ctx)
            gu._adl_state = None
            pytest.skip(f"real_amd_adl: ADL enumerates no adapter (rc {rc}, "
                        f"count {n.value})")
        arr = (gu._AdapterInfo * n.value)()
        _poison(arr)
        rc = dll.ADL2_Adapter_AdapterInfo_Get(ctx, ctypes.byref(arr),
                                              ctypes.sizeof(arr))
        yield SimpleNamespace(dll=dll, ctx=ctx, state=state, count=n.value,
                              adapters=arr, adapter_rc=rc)
        dll.ADL2_Main_Control_Destroy(ctx)
    finally:
        gu._adl_state = prior


def _present_amd(adl) -> list:
    """The adapters production would read, one per distinct PCI triple, in
    enumeration order."""
    out, seen = [], set()
    for info in adl.adapters:
        if not info.iPresent or info.iVendorID != gu._ADL_VENDOR_AMD:
            continue
        key = (info.iBusNumber, info.iDeviceNumber, info.iFunctionNumber)
        if key in seen:
            continue
        seen.add(key)
        out.append(info)
    return out


def _windows_pnp_pci_map() -> dict:
    """``{pnp_instance_id: (bus_number, device_number, function_number)}`` for
    every video controller Windows knows, read through PowerShell's PnP cmdlets.
    Returns {} when the query cannot run or answers nothing."""
    exe = shutil.which("powershell") or shutil.which("pwsh")
    if not exe:
        return {}
    script = (
        "foreach ($id in (Get-CimInstance Win32_VideoController).PNPDeviceID) {"
        " $p = Get-PnpDeviceProperty -InstanceId $id"
        "   -KeyName DEVPKEY_Device_BusNumber, DEVPKEY_Device_Address"
        "   -ErrorAction SilentlyContinue;"
        " $bus = ($p | Where-Object KeyName -eq DEVPKEY_Device_BusNumber).Data;"
        " $addr = ($p | Where-Object KeyName -eq DEVPKEY_Device_Address).Data;"
        " if ($null -ne $bus -and $null -ne $addr) { \"$id|$bus|$addr\" } }"
    )
    try:
        out = subprocess.run([exe, "-NoProfile", "-NonInteractive", "-Command", script],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return {}
    if out.returncode != 0:
        return {}
    result = {}
    for line in out.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        try:
            bus, addr = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        result[parts[0]] = (bus, addr >> 16, addr & 0xFFFF)
    return result


@pytest.mark.integration
@pytest.mark.real_amd_adl
class TestRealAdlDriverAgreesWithDeclaredLayouts:
    """The installed AMD driver, asked through production's own structs and
    calls, must agree with the layouts gpu_usage declares."""

    def test_driver_fills_the_adapter_table_at_the_declared_size(self, adl):
        assert adl.adapter_rc == gu._ADL_OK
        for i, info in enumerate(adl.adapters):
            assert info.iAdapterIndex == i

    def test_driver_rejects_an_adapter_table_smaller_than_its_own_stride(self, adl):
        """The production call passes ``sizeof(_AdapterInfo) * n``; the driver
        refuses a buffer below its own ``sizeof(AdapterInfo) * n``, so the
        production call succeeding is a lower bound on the declared size."""
        arr = (gu._AdapterInfo * adl.count)()
        rc = adl.dll.ADL2_Adapter_AdapterInfo_Get(adl.ctx, ctypes.byref(arr),
                                                  ctypes.sizeof(arr) - 1)
        if rc == gu._ADL_OK:
            pytest.skip("the driver accepted a table one byte short of the declared "
                        "size: either it does not validate the size or the declared "
                        "struct is larger than its own; the allocation test settles "
                        "which")
        assert adl.adapter_rc == gu._ADL_OK

    def test_driver_allocates_adapter_records_at_exactly_the_declared_size(self, adl):
        """``ADL2_Adapter_AdapterInfoX3_Get`` allocates the adapter array through
        the production alloc callback, so the byte count it asks for is the
        driver's own ``sizeof(AdapterInfo)`` times the count it reports."""
        x3 = getattr(adl.dll, "ADL2_Adapter_AdapterInfoX3_Get", None)
        if x3 is None:
            pytest.skip("ADL2_Adapter_AdapterInfoX3_Get is not exported by this driver")
        before = len(adl.state["keepalive"])
        count = ctypes.c_int(-1)
        table = ctypes.POINTER(gu._AdapterInfo)()
        rc = x3(adl.ctx, -1, ctypes.byref(count), ctypes.byref(table))
        assert rc == gu._ADL_OK
        assert count.value == adl.count
        buffers = adl.state["keepalive"][before:]
        assert len(buffers) == 1
        assert ctypes.sizeof(buffers[0]) == count.value * ctypes.sizeof(gu._AdapterInfo)
        for i in range(count.value):
            assert table[i].iAdapterIndex == i
            assert _decode(table[i].strPNPString) == _decode(adl.adapters[i].strPNPString)

    def test_int_fields_agree_with_the_char_arrays_at_other_offsets(self, adl):
        """A PCI adapter's vendor id appears three times in AdapterInfo: as the
        int ``iVendorID``, as the hex after ``VEN_`` in ``strUDID`` before the
        ints, and again in ``strPNPString`` after them. A shifted layout makes
        them disagree."""
        pci = [info for info in adl.adapters
               if _decode(info.strPNPString).startswith("PCI\\VEN_")]
        assert pci, "ADL enumerated adapters but none carries a PCI PNP string"
        for info in adl.adapters:
            assert info.iPresent in (0, 1)
            assert info.iExist in (0, 1)
            assert 0 <= info.iBusNumber < 256
            assert 0 <= info.iDeviceNumber < 32
            assert 0 <= info.iFunctionNumber < 8
        for info in pci:
            pnp = _decode(info.strPNPString)
            ven_hex = pnp[len("PCI\\VEN_"):][:4]
            assert int(ven_hex, 16) == int(str(info.iVendorID), 16)
            assert _decode(info.strUDID).startswith("PCI_VEN_" + ven_hex)
            if info.iPresent:
                name = _decode(info.strAdapterName)
                assert name and name.isprintable() and name.isascii()
        amd = [info for info in pci
               if _decode(info.strPNPString).startswith("PCI\\VEN_1002")]
        for info in amd:
            assert info.iVendorID == gu._ADL_VENDOR_AMD

    def test_pci_triple_matches_what_windows_reports_for_the_same_device(self, adl):
        """``iBusNumber`` / ``iDeviceNumber`` / ``iFunctionNumber`` for an adapter
        must equal the bus number and address Windows itself holds for the
        device whose PnP instance id ``strPNPString`` names."""
        windows = _windows_pnp_pci_map()
        if not windows:
            pytest.skip("Windows PnP properties are not queryable here")
        matched = 0
        for info in adl.adapters:
            if not info.iPresent:
                continue
            triple = windows.get(_decode(info.strPNPString))
            if triple is None:
                continue
            matched += 1
            assert (info.iBusNumber, info.iDeviceNumber, info.iFunctionNumber) == triple
        assert matched, ("no present adapter's strPNPString names a video "
                         "controller Windows knows: " + repr(sorted(windows)))

    def test_pmlog_output_size_is_written_as_the_declared_struct_size(self, adl):
        adapters = _present_amd(adl)
        assert adapters, "no present AMD adapter passed the production filter"
        data = gu._ADLPMLogDataOutput()
        _poison(data)
        rc = adl.dll.ADL2_New_QueryPMLogData_Get(adl.ctx, adapters[0].iAdapterIndex,
                                                 ctypes.byref(data))
        if rc != gu._ADL_OK:
            pytest.skip(f"this board declines PMLog (rc {rc}); the legacy test covers it")
        assert data.size == ctypes.sizeof(gu._ADLPMLogDataOutput)
        for sensor in data.sensors:
            assert sensor.supported in (0, 1)

    def test_live_activity_reads_through_the_production_path_within_bounds(self, adl):
        adapters = _present_amd(adl)
        assert adapters, "no present AMD adapter passed the production filter"
        data = gu._ADLPMLogDataOutput()
        _poison(data)
        rc = adl.dll.ADL2_New_QueryPMLogData_Get(adl.ctx, adapters[0].iAdapterIndex,
                                                 ctypes.byref(data))
        if rc != gu._ADL_OK:
            pytest.skip(f"this board declines PMLog (rc {rc}); the legacy test covers it")
        assert data.size == ctypes.sizeof(gu._ADLPMLogDataOutput)
        if not data.sensors[gu._ADL_PMLOG_ACTIVITY_GFX].supported:
            pytest.skip("this board does not publish the PMLog activity sensor; "
                        "the legacy test covers it")
        pct = gu._adl_pmlog_activity(adl.dll, adl.ctx, adapters[0].iAdapterIndex)
        assert isinstance(pct, float)
        assert 0.0 <= pct <= 100.0
        assert gu._adl_usable_pct(pct, "test") == pct
        by_bus = gu._adl_activity_by_bus()
        assert int(adapters[0].iBusNumber) in by_bus
        assert set(by_bus) <= {int(info.iBusNumber) for info in adapters}
        for value in by_bus.values():
            assert 0.0 <= value <= 100.0

    def test_legacy_overdrive_sources_are_exported_and_answer_in_range_or_decline(self, adl):
        """Each pre-PMLog source must be a real export of the installed driver,
        and when it answers ADL_OK the activity field it names must hold a
        percentage. A source that declines writes nothing usable and is skipped
        by production; a board that answers PMLog declines all of them."""
        adapters = _present_amd(adl)
        assert adapters, "no present AMD adapter passed the production filter"
        idx = adapters[0].iAdapterIndex
        for export, struct, field, needs_size in gu._ADL_LEGACY_ACTIVITY_SOURCES:
            fn = getattr(adl.dll, export, None)
            assert fn is not None, f"{export} is not exported by the installed atiadlxx.dll"
            data = struct()
            _poison(data)
            if needs_size:
                data.iSize = ctypes.sizeof(struct)
            rc = fn(adl.ctx, idx, ctypes.byref(data))
            if rc == gu._ADL_OK:
                assert 0 <= getattr(data, field) <= 100, export
        legacy = gu._adl_legacy_activity(adl.dll, adl.ctx, idx)
        if legacy is not None:
            pct, source = legacy
            assert 0.0 <= pct <= 100.0
            assert source in {s[0] for s in gu._ADL_LEGACY_ACTIVITY_SOURCES}
