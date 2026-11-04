# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Windows 260-character path limit, on any Windows box.

``max_path`` clears this process's long-path flag in its process environment
block for the length of a test, so Windows itself refuses a path of 260 or more
characters that lacks the extended-length prefix, as it does on a machine without
the LongPathsEnabled policy. Nothing outside this process changes. The fixture
checks that the limit really applies before the test runs. Elsewhere it does
nothing, and the test runs on the real file system without the limit.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes

import pytest

_PEB_BITFIELD_OFFSET = 3
_LONG_PATH_AWARE_BIT = 0x80


class _ProcessBasicInformation(ctypes.Structure):
    _fields_ = [("reserved1", ctypes.c_void_p), ("peb", ctypes.c_void_p),
                ("reserved2", ctypes.c_void_p * 2), ("pid", ctypes.c_void_p),
                ("reserved3", ctypes.c_void_p)]


def _peb_bitfield() -> ctypes.c_ubyte:
    ntdll = ctypes.WinDLL("ntdll")
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ntdll.NtQueryInformationProcess.argtypes = [wintypes.HANDLE, ctypes.c_ulong, ctypes.c_void_p,
                                                ctypes.c_ulong, ctypes.c_void_p]
    info = _ProcessBasicInformation()
    status = ntdll.NtQueryInformationProcess(kernel32.GetCurrentProcess(), 0, ctypes.byref(info),
                                             ctypes.sizeof(info), None)
    assert status == 0, f"NtQueryInformationProcess returned {status:#x}"
    return ctypes.c_ubyte.from_address(info.peb + _PEB_BITFIELD_OFFSET)


@pytest.fixture
def max_path(tmp_path):
    if sys.platform != "win32":
        yield
        return
    field = _peb_bitfield()
    saved = field.value
    probe = tmp_path / "max-path-probe" / "/".join(["x" * 40] * 7) / "probe.txt"
    prefixed = "\\\\?\\" + os.path.abspath(probe)
    os.makedirs(os.path.dirname(prefixed))
    with open(prefixed, "w", encoding="utf-8") as f:
        f.write("probe")
    field.value = saved & ~_LONG_PATH_AWARE_BIT
    try:
        assert len(str(probe)) >= 260
        assert not os.path.isfile(probe), "the limit did not apply to a plain path"
        assert os.path.isfile(prefixed), "the limit applied to an extended-length path"
        yield
    finally:
        field.value = saved
