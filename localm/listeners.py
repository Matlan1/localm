# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which TCP ports on this machine are listening, read from the operating system.

Used to find running localm instances without scanning ports one by one: the OS
already knows every listening socket. Windows reads the extended TCP table through
``iphlpapi``, Linux reads ``/proc/net/tcp`` and ``/proc/net/tcp6``, macOS parses
``netstat``. Every function returns None when the platform's table cannot be read,
so a caller can fall back to probing a port range.
"""

from __future__ import annotations

import os
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Optional

from localm.debuglog import logger

Endpoint = tuple  # (address: str, port: int)

_ERROR_INSUFFICIENT_BUFFER = 122
_TCP_TABLE_OWNER_PID_LISTENER = 3
_AF_INET = 2
_AF_INET6 = 23


def _windows_port(raw: int) -> int:
    """The port in a Windows ``dwLocalPort`` field (network byte order in the low
    16 bits)."""
    return ((raw & 0xFF) << 8) | ((raw >> 8) & 0xFF)


def parse_windows_table_owners(buf: bytes, family: int) -> list:
    """Listening ``(address, port, pid)`` triples from a ``GetExtendedTcpTable``
    buffer of ``MIB_TCPROW_OWNER_PID`` (IPv4, 24-byte rows) or
    ``MIB_TCP6ROW_OWNER_PID`` (IPv6, 56-byte rows) entries; *pid* is the owning
    process id."""
    if len(buf) < 4:
        return []
    (count,) = struct.unpack_from("<I", buf, 0)
    out = []
    if family == _AF_INET:
        row = 24
        for i in range(count):
            off = 4 + i * row
            if off + row > len(buf):
                break
            _state, addr, port, _ra, _rp, pid = struct.unpack_from("<6I", buf, off)
            out.append((socket.inet_ntoa(struct.pack("<I", addr)), _windows_port(port), pid))
    else:
        row = 56
        for i in range(count):
            off = 4 + i * row
            if off + row > len(buf):
                break
            addr = bytes(buf[off:off + 16])
            (port,) = struct.unpack_from("<I", buf, off + 20)
            (pid,) = struct.unpack_from("<I", buf, off + 52)
            out.append((socket.inet_ntop(socket.AF_INET6, addr), _windows_port(port), pid))
    return out


def parse_windows_table(buf: bytes, family: int) -> list:
    """Listening ``(address, port)`` pairs from a ``GetExtendedTcpTable`` buffer
    (see :func:`parse_windows_table_owners`)."""
    return [(a, p) for a, p, _pid in parse_windows_table_owners(buf, family)]


def _windows_listeners() -> Optional[list]:
    rows = _windows_listener_owners()
    return None if rows is None else [(a, p) for a, p, _pid in rows]


def _windows_listener_owners() -> Optional[list]:
    import ctypes
    from ctypes import wintypes
    try:
        iphlpapi = ctypes.WinDLL("iphlpapi")
    except OSError as e:
        logger.debug("listeners: iphlpapi unavailable: %s", e)
        return None
    out = []
    for family in (_AF_INET, _AF_INET6):
        size = wintypes.DWORD(0)
        iphlpapi.GetExtendedTcpTable(None, ctypes.byref(size), False, family,
                                     _TCP_TABLE_OWNER_PID_LISTENER, 0)
        for _ in range(8):
            buf = ctypes.create_string_buffer(max(size.value, 4))
            rc = iphlpapi.GetExtendedTcpTable(
                buf, ctypes.byref(size), False, family,
                _TCP_TABLE_OWNER_PID_LISTENER, 0)
            if rc == 0:
                out.extend(parse_windows_table_owners(buf.raw[:size.value], family))
                break
            if rc != _ERROR_INSUFFICIENT_BUFFER:
                logger.debug("listeners: GetExtendedTcpTable(family=%s) returned %s",
                             family, rc)
                return None
        else:
            return None
    return out


def parse_proc_net_tcp_owners(text: str, family: int) -> list:
    """Listening ``(address, port, uid)`` triples from the text of
    ``/proc/net/tcp`` (*family* ``socket.AF_INET``) or ``/proc/net/tcp6``
    (``socket.AF_INET6``): rows whose state column is ``0A`` (LISTEN). *uid* is
    the owning user id, or None when the row has no readable uid column."""
    out = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4 or parts[3] != "0A":
            continue
        addr_hex, _, port_hex = parts[1].partition(":")
        try:
            port = int(port_hex, 16)
            raw = bytes.fromhex(addr_hex)
            if family == socket.AF_INET:
                address = socket.inet_ntoa(raw[::-1])
            else:
                words = [raw[i:i + 4][::-1] for i in range(0, 16, 4)]
                address = socket.inet_ntop(socket.AF_INET6, b"".join(words))
        except (ValueError, OSError):
            continue
        try:
            uid: Optional[int] = int(parts[7])
        except (IndexError, ValueError):
            uid = None
        out.append((address, port, uid))
    return out


def parse_proc_net_tcp(text: str, family: int) -> list:
    """Listening ``(address, port)`` pairs from ``/proc/net/tcp`` text (see
    :func:`parse_proc_net_tcp_owners`)."""
    return [(a, p) for a, p, _uid in parse_proc_net_tcp_owners(text, family)]


def _linux_listener_owners() -> Optional[list]:
    out = []
    found = False
    for path, family in (("/proc/net/tcp", socket.AF_INET),
                         ("/proc/net/tcp6", socket.AF_INET6)):
        try:
            with open(path, encoding="ascii", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        found = True
        out.extend(parse_proc_net_tcp_owners(text, family))
    return out if found else None


def _linux_listeners() -> Optional[list]:
    rows = _linux_listener_owners()
    return None if rows is None else [(a, p) for a, p, _uid in rows]


def parse_netstat(text: str) -> list:
    """Listening ``(address, port)`` pairs from BSD/macOS ``netstat -an -p tcp``
    output, where the local address is ``host.port`` and ``*`` means any."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 6 or not parts[0].startswith("tcp") or parts[-1] != "LISTEN":
            continue
        host, _, port = parts[3].rpartition(".")
        try:
            out.append(("0.0.0.0" if host == "*" else host, int(port)))
        except ValueError:
            continue
    return out


def _macos_listeners() -> Optional[list]:
    try:
        proc = subprocess.run(["netstat", "-an", "-p", "tcp"], capture_output=True,
                              text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("listeners: netstat unavailable: %s", e)
        return None
    if proc.returncode != 0:
        return None
    return parse_netstat(proc.stdout)


def listening_endpoints() -> Optional[list]:
    """Every listening TCP endpoint on this machine as ``(address, port)`` pairs,
    or None when this platform's socket table cannot be read."""
    try:
        if sys.platform == "win32":
            return _windows_listeners()
        if sys.platform == "darwin":
            return _macos_listeners()
        if sys.platform.startswith("linux"):
            return _linux_listeners()
    except Exception as e:
        logger.debug("listeners: reading the socket table failed: %s", e)
    return None


def listening_ports_in(lo: int, hi: int) -> Optional[list]:
    """The sorted distinct listening ports between *lo* and *hi* inclusive, or None
    when the socket table cannot be read."""
    endpoints = listening_endpoints()
    if endpoints is None:
        return None
    return sorted({port for _addr, port in endpoints if lo <= port <= hi})


def parse_lsof_listen_owners(text: str) -> list:
    """Listening ``(address, port, uid)`` triples from ``lsof -nP -iTCP
    -sTCP:LISTEN -Fun`` output: a ``u<uid>`` line starts a process's set, and
    each ``n<address>:<port>`` line after it is one of its listening sockets
    (``*`` means any address, an IPv6 address is in brackets)."""
    out = []
    uid: Optional[int] = None
    for line in text.splitlines():
        if line.startswith("p"):
            uid = None
        elif line.startswith("u"):
            try:
                uid = int(line[1:])
            except ValueError:
                uid = None
        elif line.startswith("n"):
            host, sep, port = line[1:].rpartition(":")
            if not sep:
                continue
            host = host.strip("[]")
            try:
                out.append(("0.0.0.0" if host == "*" else host, int(port), uid))
            except ValueError:
                continue
    return out


def _macos_listener_owners() -> Optional[list]:
    try:
        proc = subprocess.run(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN", "-Fun"],
                              capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("listeners: lsof unavailable: %s", e)
        return None
    if proc.returncode not in (0, 1):
        return None
    return parse_lsof_listen_owners(proc.stdout)


def _windows_pid_is_current_user(pid: int) -> bool:
    """True when process *pid* runs as the same Windows account as this
    process (its token's user SID). False when it cannot be opened or read."""
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                          ctypes.POINTER(wintypes.HANDLE)]
    advapi32.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                             wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    advapi32.GetLengthSid.argtypes = [ctypes.c_void_p]
    advapi32.GetLengthSid.restype = wintypes.DWORD
    process_query_limited_information, token_query, token_user = 0x1000, 0x0008, 1

    def user_sid(process) -> Optional[bytes]:
        token = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(process, token_query, ctypes.byref(token)):
            return None
        try:
            size = wintypes.DWORD(0)
            advapi32.GetTokenInformation(token, token_user, None, 0, ctypes.byref(size))
            if not size.value:
                return None
            buf = ctypes.create_string_buffer(size.value)
            if not advapi32.GetTokenInformation(token, token_user, buf, size,
                                                ctypes.byref(size)):
                return None
            psid = ctypes.cast(buf, ctypes.POINTER(ctypes.c_void_p))[0]
            if not psid:
                return None
            return ctypes.string_at(psid, advapi32.GetLengthSid(psid))
        finally:
            kernel32.CloseHandle(token)

    mine = user_sid(kernel32.GetCurrentProcess())
    if mine is None:
        return False
    handle = kernel32.OpenProcess(process_query_limited_information, False, int(pid))
    if not handle:
        return False
    try:
        return user_sid(handle) == mine
    finally:
        kernel32.CloseHandle(handle)


_ANY_ADDRESS = ("0.0.0.0", "::", "*", "")


def _reaches(address: str, dial: str) -> bool:
    """Whether a socket listening on *address* can accept a connection to
    *dial*: the same address, its IPv4-mapped IPv6 form, or a wildcard."""
    return (address in _ANY_ADDRESS or address == dial
            or address == "::ffff:" + dial)


#: How long a True answer of port_owned_by_current_user is reused for the
#: same (port, dial), in seconds. A False answer is never reused.
OWNED_CACHE_TTL_S = 5.0
_owned_cache: dict = {}
_owned_cache_lock = threading.Lock()


def port_owned_by_current_user(port: int, dial: str = "127.0.0.1") -> bool:
    """True only when at least one socket is listening on *port* where a
    connection to *dial* can reach it, and every such socket belongs to the OS
    user running this process (Windows: the owning process's account; Linux
    and macOS: the socket's uid). False when the platform's table, or any of
    those owners, cannot be read.

    A True answer is reused for :data:`OWNED_CACHE_TTL_S` seconds for the same
    (port, dial); a False answer is recomputed on every call."""
    key = (int(port), dial)
    now = time.monotonic()
    with _owned_cache_lock:
        expiry = _owned_cache.get(key)
        if expiry is not None and expiry > now:
            return True
        _owned_cache.pop(key, None)
    owned = _port_owned_by_current_user(port, dial)
    if owned:
        with _owned_cache_lock:
            _owned_cache[key] = time.monotonic() + OWNED_CACHE_TTL_S
    return owned


def _port_owned_by_current_user(port: int, dial: str) -> bool:
    """:func:`port_owned_by_current_user` without its cache."""
    try:
        if sys.platform == "win32":
            rows = _windows_listener_owners()
        elif sys.platform == "darwin":
            rows = _macos_listener_owners()
        elif sys.platform.startswith("linux"):
            rows = _linux_listener_owners()
        else:
            rows = None
    except Exception as e:
        logger.debug("listeners: reading the socket owners failed: %s", e)
        rows = None
    if rows is None:
        return False
    owners = [owner for address, p, owner in rows if p == port and _reaches(address, dial)]
    if not owners:
        return False
    if sys.platform == "win32":
        return all(owner is not None and _windows_pid_is_current_user(owner)
                   for owner in set(owners))
    uid = os.getuid()
    return all(owner == uid for owner in owners)
