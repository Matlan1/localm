# SPDX-License-Identifier: AGPL-3.0-or-later
"""gpu_usage's ADL ctypes layouts, pinned to AMD's published headers and, where
one is installed, cross-checked against the real ``atiadlxx.dll``.

* ``TestAdlLayoutsMatchPublishedHeaders`` runs everywhere and pins each struct's
  field order, ``ctypes.sizeof`` and the sensor constants to the values in AMD's
  public ``adl_structures.h`` / ``adl_defines.h``.
* ``TestRealAdlDriverAgreesWithDeclaredLayouts`` (integration, ``real_amd_adl``)
  opens the installed driver through the production ``_adl_open`` and checks the
  declared layouts against what the driver itself validates, allocates and
  fills, then reads live whole-GPU activity through the production path.
  It skips, never fails, when ADL cannot be used at all: not Windows, the DLL
  does not load, the driver refuses a context, ADL enumerates no adapter, or no
  enumerated adapter's PNP string names an AMD PCI device. Once an AMD PCI
  adapter is enumerated, a layout that disagrees with the driver is a failure.

What neither layer can see: a sensor index wrong by one still lands on a 0-100
percentage (``ADL_PMLOG_INFO_ACTIVITY_MEM`` follows ``ADL_PMLOG_INFO_ACTIVITY_GFX``);
a swap of ``supported`` and ``value`` reads identically for an idle sensor at 0;
and the three pre-PMLog structs are driver-checked only on a board whose driver
answers one of them. The header pins are the only check for those. Nothing here
loads a model or puts load on the GPU.
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
_POISON_TEXT = chr(_POISON) * 2
_AMD_PCI_PREFIX = "PCI\\VEN_1002"
_AMD_PCI_MARKER = b"VEN_1002&"


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


def _pnp(info) -> str:
    return _decode(info.strPNPString)


def _raw(table) -> bytes:
    """Every byte of the adapter table as the driver wrote it, independent of
    the declared field offsets."""
    return ctypes.string_at(ctypes.addressof(table), ctypes.sizeof(table))


@pytest.fixture(scope="module")
def adl():
    """A live ADL context opened through the production ``_adl_open``, with the
    adapter table read the way ``_adl_activity_by_bus`` reads it.

    Skips the module when ADL cannot be used at all: not Windows, the DLL does
    not load, the driver refuses a context, no adapter is enumerated, or no
    enumerated adapter's PNP string names an AMD PCI device. Restores
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
        try:
            n = ctypes.c_int(0)
            rc = dll.ADL2_Adapter_NumberOfAdapters_Get(ctx, ctypes.byref(n))
            if rc != gu._ADL_OK or n.value <= 0:
                pytest.skip(f"real_amd_adl: ADL enumerates no adapter (rc {rc}, "
                            f"count {n.value})")
            arr = (gu._AdapterInfo * n.value)()
            _poison(arr)
            rc = dll.ADL2_Adapter_AdapterInfo_Get(ctx, ctypes.byref(arr),
                                                  ctypes.sizeof(arr))
            if rc == gu._ADL_OK and _AMD_PCI_MARKER not in _raw(arr):
                pytest.skip("real_amd_adl: ADL lists no AMD PCI adapter: "
                            + repr([_pnp(info) for info in arr]))
            yield SimpleNamespace(dll=dll, ctx=ctx, state=state, count=n.value,
                                  adapters=arr, adapter_rc=rc)
        finally:
            dll.ADL2_Main_Control_Destroy(ctx)
            gu._adl_state = None
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


def _production_adapters(adl) -> list:
    """``_present_amd`` when it is non-empty. Otherwise skips when the raw
    adapter table carries no AMD PCI marker at all, or carries it only in
    adapters that are not present, and fails when it carries one the production
    filter cannot see."""
    adapters = _present_amd(adl)
    if adapters:
        return adapters
    if _AMD_PCI_MARKER not in _raw(adl.adapters):
        pytest.skip("real_amd_adl: ADL lists no AMD PCI adapter")
    amd = [info for info in adl.adapters if _pnp(info).startswith(_AMD_PCI_PREFIX)]
    if amd and all(info.iPresent == 0 for info in amd):
        pytest.skip("real_amd_adl: the listed AMD PCI adapters are not present")
    pytest.fail("an AMD PCI adapter is listed but the production filter reads none: "
                "(iPresent, iVendorID, strPNPString) = "
                + repr([(info.iPresent, info.iVendorID, _pnp(info))
                        for info in adl.adapters]))


def _windows_pnp_pci_map() -> dict:
    """``{PNP_INSTANCE_ID: (bus_number, device_number, function_number)}`` for
    every video controller Windows knows, read through PowerShell's PnP cmdlets,
    keyed by the upper-cased instance id. Returns {} when the query cannot run
    or answers nothing."""
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
                             capture_output=True, text=True, errors="replace",
                             timeout=60)
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
        result[parts[0].upper()] = (bus, addr >> 16, addr & 0xFFFF)
    return result


def _pmlog_answers(adl, adapters) -> list:
    """``[(info, data)]`` for every adapter in *adapters* whose
    ``ADL2_New_QueryPMLogData_Get`` returned ADL_OK, each *data* poisoned before
    the call. Skips when the driver lacks the export."""
    query = getattr(adl.dll, "ADL2_New_QueryPMLogData_Get", None)
    if query is None:
        pytest.skip("ADL2_New_QueryPMLogData_Get is not exported by this driver")
    out = []
    for info in adapters:
        data = gu._ADLPMLogDataOutput()
        _poison(data)
        if query(adl.ctx, info.iAdapterIndex, ctypes.byref(data)) == gu._ADL_OK:
            out.append((info, data))
    return out


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
        address = ctypes.cast(table, ctypes.c_void_p).value
        buffers = [b for b in adl.state["keepalive"][before:]
                   if ctypes.addressof(b) == address]
        assert len(buffers) == 1, "the returned table is not a buffer the alloc callback handed out"
        assert ctypes.sizeof(buffers[0]) == count.value * ctypes.sizeof(gu._AdapterInfo)
        for i in range(count.value):
            assert table[i].iAdapterIndex == i
            assert _pnp(table[i]) == _pnp(adl.adapters[i])

    def test_int_fields_agree_with_the_char_arrays_at_other_offsets(self, adl):
        """An AMD adapter's vendor id appears three times in AdapterInfo: as the
        int ``iVendorID``, as ``VEN_1002`` in ``strUDID`` before the ints, and
        again in ``strPNPString`` after them. A shifted layout makes them
        disagree."""
        for info in adl.adapters:
            assert info.iPresent in (0, 1)
            assert info.iExist in (0, 1)
            assert 0 <= info.iBusNumber < 256
            assert 0 <= info.iDeviceNumber < 32
            assert 0 <= info.iFunctionNumber < 8
        amd = [info for info in adl.adapters if _pnp(info).startswith(_AMD_PCI_PREFIX)]
        assert amd
        for info in amd:
            assert info.iVendorID == gu._ADL_VENDOR_AMD
            assert _decode(info.strUDID).startswith("PCI_VEN_1002")
            if info.iPresent:
                name = _decode(info.strAdapterName)
                assert name and _POISON_TEXT not in name

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
            triple = windows.get(_pnp(info).upper())
            if triple is None:
                continue
            matched += 1
            assert (info.iBusNumber, info.iDeviceNumber, info.iFunctionNumber) == triple
        assert matched, ("no present adapter's strPNPString names a video "
                         "controller Windows knows: " + repr(sorted(windows)))

    def test_pmlog_output_size_is_written_as_the_declared_struct_size(self, adl):
        answers = _pmlog_answers(adl, _production_adapters(adl))
        if not answers:
            pytest.skip("no present AMD adapter answers PMLog; the legacy test covers it")
        for _info, data in answers:
            assert data.size == ctypes.sizeof(gu._ADLPMLogDataOutput)
            for sensor in data.sensors:
                assert sensor.supported in (0, 1)

    def test_live_activity_reads_through_the_production_path_within_bounds(self, adl):
        adapters = _production_adapters(adl)
        answers = _pmlog_answers(adl, adapters)
        if not answers:
            pytest.skip("no present AMD adapter answers PMLog; the legacy test covers it")
        for _info, data in answers:
            assert data.size == ctypes.sizeof(gu._ADLPMLogDataOutput)
        publishing = [info for info, data in answers
                      if data.sensors[gu._ADL_PMLOG_ACTIVITY_GFX].supported]
        if not publishing:
            pytest.skip("no present AMD adapter publishes the PMLog activity "
                        "sensor; the legacy test covers it")
        for info in publishing:
            pct = gu._adl_pmlog_activity(adl.dll, adl.ctx, info.iAdapterIndex)
            assert isinstance(pct, float)
            assert 0.0 <= pct <= 100.0
        by_bus = gu._adl_activity_by_bus()
        assert {int(info.iBusNumber) for info in publishing} <= set(by_bus)
        assert set(by_bus) <= {int(info.iBusNumber) for info in adapters}
        for value in by_bus.values():
            assert 0.0 <= value <= 100.0

    def test_legacy_overdrive_sources_answer_in_range_or_decline(self, adl):
        """Each pre-PMLog source the installed driver exports, called through the
        production struct for every present AMD adapter, must hold a percentage
        in the activity field it names whenever it answers ADL_OK. A source that
        declines writes nothing production reads."""
        adapters = _production_adapters(adl)
        exported = [(export, struct, field, needs_size)
                    for export, struct, field, needs_size in gu._ADL_LEGACY_ACTIVITY_SOURCES
                    if getattr(adl.dll, export, None) is not None]
        if not exported:
            pytest.skip("this driver exports none of the pre-PMLog activity entry points")
        for info in adapters:
            for export, struct, field, needs_size in exported:
                data = struct()
                _poison(data)
                if needs_size:
                    data.iSize = ctypes.sizeof(struct)
                rc = getattr(adl.dll, export)(adl.ctx, info.iAdapterIndex, ctypes.byref(data))
                if rc == gu._ADL_OK:
                    assert 0 <= getattr(data, field) <= 100, export
            legacy = gu._adl_legacy_activity(adl.dll, adl.ctx, info.iAdapterIndex)
            if legacy is not None:
                pct, source = legacy
                assert 0.0 <= pct <= 100.0
                assert source in {s[0] for s in exported}
