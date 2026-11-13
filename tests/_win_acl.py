# SPDX-License-Identifier: AGPL-3.0-or-later
"""The account a Windows ACL check expects: the user of this process's token,
from :func:`localm.storekit.current_user_sid`, never the USERNAME variable."""

from __future__ import annotations


def token_user_name() -> str:
    """The account name (without the domain) of the process token's user, as
    ``icacls`` prints it before ``:(F)``. Windows only; raises OSError when the
    SID cannot be read or looked up."""
    import ctypes
    from ctypes import wintypes

    from localm.storekit import current_user_sid

    sid_text = current_user_sid()
    if not sid_text:
        raise OSError("the process token's user SID could not be read")
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32.ConvertStringSidToSidW.argtypes = [
        wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    advapi32.LookupAccountSidW.argtypes = [
        wintypes.LPCWSTR, ctypes.c_void_p, wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(ctypes.c_int)]
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    psid = ctypes.c_void_p()
    if not advapi32.ConvertStringSidToSidW(sid_text, ctypes.byref(psid)):
        raise OSError(ctypes.get_last_error(), "ConvertStringSidToSidW failed")
    try:
        name = ctypes.create_unicode_buffer(256)
        domain = ctypes.create_unicode_buffer(256)
        n, d, use = wintypes.DWORD(256), wintypes.DWORD(256), ctypes.c_int()
        if not advapi32.LookupAccountSidW(None, psid, name, ctypes.byref(n), domain,
                                          ctypes.byref(d), ctypes.byref(use)):
            raise OSError(ctypes.get_last_error(), "LookupAccountSidW failed")
        return name.value
    finally:
        kernel32.LocalFree(psid)
