# SPDX-License-Identifier: AGPL-3.0-or-later
"""Keep a file open from a real second handle, the way another program holds it
on Windows. Only for tests that are skipped on other platforms."""

from __future__ import annotations

import contextlib
import ctypes
import sys
from pathlib import Path


@contextlib.contextmanager
def held(path, *, exclusive: bool = False, share: int | None = None):
    """Hold the file at *path* open for the duration of the block. By default as
    Python's ``open`` does (others may read and write it, nobody may delete or
    rename it); with *exclusive* with no sharing at all (others cannot read it
    either), the way a scanner or an installer can; with *share* with that Windows
    share mode (1 read, 2 write, 4 delete and rename)."""
    path = Path(path)
    if share is None and not exclusive:
        with open(path, "rb"):
            yield
        return
    assert sys.platform == "win32"
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.HANDLE]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    generic_read, open_existing = 0x80000000, 3
    handle = kernel32.CreateFileW(str(path), generic_read, share or 0, None, open_existing, 0,
                                  None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        kernel32.CloseHandle(handle)
