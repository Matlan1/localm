# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which TCP ports on this machine are listening, read from the operating system.

Used to find running localm instances without scanning ports one by one: the OS
already knows every listening socket. Windows reads the extended TCP table through
``iphlpapi``, Linux reads ``/proc/net/tcp`` and ``/proc/net/tcp6``, macOS parses
``netstat``. Every function returns None when the platform's table cannot be read,
so a caller can fall back to probing a port range.
"""

from __future__ import annotations

import socket
import struct
import subprocess
import sys
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


def parse_windows_table(buf: bytes, family: int) -> list:
    """Listening ``(address, port)`` pairs from a ``GetExtendedTcpTable`` buffer of
    ``MIB_TCPROW_OWNER_PID`` (IPv4, 24-byte rows) or ``MIB_TCP6ROW_OWNER_PID``
    (IPv6, 56-byte rows) entries."""
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
            _state, addr, port, _ra, _rp, _pid = struct.unpack_from("<6I", buf, off)
            out.append((socket.inet_ntoa(struct.pack("<I", addr)), _windows_port(port)))
    else:
        row = 56
        for i in range(count):
            off = 4 + i * row
            if off + row > len(buf):
                break
            addr = bytes(buf[off:off + 16])
            (port,) = struct.unpack_from("<I", buf, off + 20)
            out.append((socket.inet_ntop(socket.AF_INET6, addr), _windows_port(port)))
    return out


def _windows_listeners() -> Optional[list]:
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
                out.extend(parse_windows_table(buf.raw[:size.value], family))
                break
            if rc != _ERROR_INSUFFICIENT_BUFFER:
                logger.debug("listeners: GetExtendedTcpTable(family=%s) returned %s",
                             family, rc)
                return None
        else:
            return None
    return out


def parse_proc_net_tcp(text: str, family: int) -> list:
    """Listening ``(address, port)`` pairs from the text of ``/proc/net/tcp``
    (*family* ``socket.AF_INET``) or ``/proc/net/tcp6`` (``socket.AF_INET6``):
    rows whose state column is ``0A`` (LISTEN)."""
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
        out.append((address, port))
    return out


def _linux_listeners() -> Optional[list]:
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
        out.extend(parse_proc_net_tcp(text, family))
    return out if found else None


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
