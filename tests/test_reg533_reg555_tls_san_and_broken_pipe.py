# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two over-broad matches that would read a REAL thing as a benign one.

localm/tls.py - _VPN_ADAPTER_NAME_MARKERS must not match by UNANCHORED substring,
or a real NIC whose name merely CONTAINS "tun"/"tap"/"ppp" reads as a VPN tunnel.
_primary_lan_ip() then discards its address and san_targets() leaves it out of the
leaf certificate, so a device reaching that machine on its real address gets a TLS
SAN mismatch.

localm/cli/_core.py - `except OSError` must not treat ANY errno EINVAL as a closed
pipe (sys.stdout.close() then SystemExit(0)). Windows raises EINVAL for a broad
set of GENUINE I/O misuse, so a command that hard-failed would exit 0, print
nothing and file no report, telling the user and every script checking the exit
code that it SUCCEEDED.

The EINVAL split cannot be made on the exception type: on Windows a real early
pipe close surfaces as OSError errno=22 with isinstance(e, BrokenPipeError)
FALSE, so "only treat a true BrokenPipeError as a pipe" would break every
`localm ... | head`. os.fstat's S_ISFIFO is what separates the two at the handler.
"""

import errno
import os
import socket
import stat
import sys

import click
import pytest
from click.testing import CliRunner

from localm import bugreport, tls
from localm.cli._core import _GracefulGroup


# --------------------------------------------------------------------------- #
#  A real adapter must not be mistaken for a VPN tunnel                        #
# --------------------------------------------------------------------------- #

class _Addr:
    def __init__(self, address):
        self.address = address
        self.family = socket.AF_INET


@pytest.mark.parametrize("name", [
    "ppp0",                              # a PPPoE/DSL WAN link - the machine's REAL address
    "Fortune 10G Adapter",               # contains "tun"
    "Neptune Ethernet",                  # contains "tun"
    "Metaphor Gigabit NIC",              # contains "tap"
    "Datapath Capture Card",             # contains "tap"
    "Ethernet 2",
    "Wi-Fi",
    "Intel(R) Ethernet Connection I219-V",
])
def test_real_adapter_is_not_classified_as_a_vpn(name):
    """THE REGRESSION. An unanchored substring match reads "tun" out of
    "Fortune", "tap" out of "Datapath", and "ppp" out of "ppp0" (a real PPPoE WAN
    link, frequently the default route and the machine's only real address).

    Each false positive silently drops that adapter's IP from the TLS cert SAN."""
    assert tls._is_vpn_adapter_name(name) is False, (
        f"{name!r} is a real adapter, but its name was read as a VPN tunnel - "
        "its IP gets dropped from the certificate SAN")


@pytest.mark.parametrize("name", [
    "tun0", "tap0", "tap5", "utun3",                 # POSIX tunnel device names
    "TAP-Windows Adapter V9",                        # OpenVPN on Windows
    "OpenVPN TAP-Windows6",
    "WireGuard Tunnel",
    "NordLynx",
    "Zscaler Network Adapter",
    "GlobalProtect",
    "Cisco AnyConnect Secure Mobility Client Virtual Miniport Adapter",
    "wintun",
    "My Corporate VPN",
])
def test_real_vpn_adapter_is_still_classified_as_a_vpn(name):
    """NEGATIVE CASE, the important one: narrowing the match must not blind the
    detection it exists for. Every genuine tunnel adapter shape must still match,
    on both POSIX device names and Windows friendly names."""
    assert tls._is_vpn_adapter_name(name) is True, (
        f"{name!r} is a VPN/tunnel adapter and must still be excluded")


def test_vpn_adapter_ips_only_collects_the_vpn(monkeypatch):
    """End to end through the real _vpn_adapter_ips: a box that dials PPPoE AND
    runs a VPN must yield only the VPN's address."""
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "net_if_addrs", lambda: {
        "ppp0": [_Addr("203.0.113.9")],              # real PPPoE WAN
        "Fortune 10G Adapter": [_Addr("192.168.1.50")],
        "WireGuard Tunnel": [_Addr("10.66.0.7")],
    })
    assert tls._vpn_adapter_ips() == {"10.66.0.7"}


def test_pppoe_address_survives_into_the_cert_san(monkeypatch):
    """End to end through the REAL san_targets -> _primary_lan_ip ->
    _vpn_adapter_ips chain (nothing in that chain is stubbed - only the OS-level
    probes it reads): a Linux box that dials PPPoE has ppp0 as its default route,
    so the outbound probe reports ppp0's address as the primary LAN IP. That
    address must be certified, or the admin reaching https://<that-ip>:port over
    the port-forward hits a TLS SAN mismatch."""
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "net_if_addrs", lambda: {
        "ppp0": [_Addr("203.0.113.9")],
    })

    class _FakeSock:
        def connect(self, addr): pass
        def getsockname(self): return ("203.0.113.9", 0)
        def close(self): pass

    monkeypatch.setattr(socket, "socket", lambda *a, **kw: _FakeSock())
    monkeypatch.setattr(tls, "_host_ips", lambda: [])      # isolate the probe path

    assert tls._primary_lan_ip() == "203.0.113.9", (
        "the PPPoE link's real address was discarded as a VPN tunnel")
    _hostnames, ips = tls.san_targets()
    assert "203.0.113.9" in ips, (
        "the PPPoE link's real address was left out of the certificate SAN")


def test_vpn_tunnel_address_is_still_kept_out_of_the_san(monkeypatch):
    """NEGATIVE CASE: a genuine VPN tunnel address must still be excluded from
    _primary_lan_ip (it is reachable only through the tunnel, not on the LAN)."""
    monkeypatch.setattr(tls, "_vpn_adapter_ips", lambda: {"10.66.0.7"})

    class _FakeSock:
        def connect(self, addr): pass
        def getsockname(self): return ("10.66.0.7", 0)
        def close(self): pass

    monkeypatch.setattr(socket, "socket", lambda *a, **kw: _FakeSock())
    assert tls._primary_lan_ip() == ""


# --------------------------------------------------------------------------- #
#  Only an ACTUAL broken pipe may exit 0 without a report                      #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def cli(tmp_path, monkeypatch):
    """A _GracefulGroup whose bug reports are captured instead of filed."""
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)
    reported = []
    monkeypatch.setattr(bugreport, "report_failure", lambda **k: reported.append(k))
    return reported


def _group(exc):
    @click.group(cls=_GracefulGroup)
    def g():
        pass

    @g.command()
    def boom():
        raise exc

    return g


@pytest.mark.parametrize("exc, label", [
    (OSError(errno.EINVAL, "Invalid argument"), "a bare EINVAL"),
    (IsADirectoryError(errno.EINVAL, "Invalid argument"), "reading a directory"),
    (OSError(errno.EINVAL, "The parameter is incorrect"), "a Windows native call"),
])
def test_genuine_einval_is_reported_not_silently_swallowed(cli, exc, label):
    """THE REGRESSION. stdout here is NOT a pipe (no downstream consumer exists),
    so an EINVAL cannot be a pipe close - it is a real failure. It must exit
    non-zero and file a report, not exit 0 in silence claiming success.

    Pre-fix every one of these exited 0 with reported == []."""
    res = CliRunner().invoke(_group(exc), ["boom"])
    assert res.exit_code == 1, (
        f"{label} exited {res.exit_code} - a real failure reported SUCCESS")
    assert cli, f"{label} filed no bug report - the failure was hidden entirely"


def test_a_real_broken_pipe_is_still_silent(cli):
    """NEGATIVE CASE: an actual early pipe close (`localm ... | head`) must still
    exit 0 with no report and no traceback cascade - reporting would write to the
    same dead stdout and re-crash.

    Uses a REAL os.pipe() as stdout so the code's own S_ISFIFO check sees a
    genuine pipe, not a mock of the check being tested. Driven through click's own
    main(), not CliRunner, which REPLACES sys.stdout with a StringIO for the
    duration of invoke() and would hide the pipe."""
    r, w = os.pipe()
    pipe_stdout = os.fdopen(w, "w")
    g = _group(OSError(errno.EINVAL, "closed pipe"))
    old = sys.stdout
    try:
        assert stat.S_ISFIFO(os.fstat(pipe_stdout.fileno()).st_mode), (
            "the fixture must present a REAL pipe or this proves nothing")
        sys.stdout = pipe_stdout
        with pytest.raises(SystemExit) as ei:
            g.main(["boom"], standalone_mode=False)
    finally:
        sys.stdout = old
        try:
            pipe_stdout.close()
        except OSError:
            pass
        os.close(r)
    assert ei.value.code == 0, (
        "a real early pipe close must exit cleanly, not file a bug report")
    assert cli == [], "a real broken pipe must NOT be reported as a bug"


def test_broken_pipe_error_is_still_silent_whatever_stdout_is(cli):
    """NEGATIVE CASE: a true BrokenPipeError is unambiguous and needs no stdout
    inspection at all - it stays silent even when stdout is not a pipe."""
    res = CliRunner().invoke(
        _group(BrokenPipeError(errno.EPIPE, "broken pipe")), ["boom"])
    assert res.exit_code == 0
    assert cli == []


def test_epipe_oserror_is_still_silent(cli):
    """NEGATIVE CASE: EPIPE means exactly "broken pipe" and needs no
    qualification. Python maps OSError(EPIPE) to BrokenPipeError; this pins
    that."""
    e = OSError(errno.EPIPE, "broken pipe")
    assert isinstance(e, BrokenPipeError), "OSError(EPIPE) must map to BrokenPipeError"
    res = CliRunner().invoke(_group(e), ["boom"])
    assert res.exit_code == 0
    assert cli == []


def test_other_oserrors_are_still_reported(cli):
    """NEGATIVE CASE: the ordinary unexpected-OSError path is unchanged."""
    res = CliRunner().invoke(
        _group(OSError(errno.ENOENT, "No such file or directory")), ["boom"])
    assert res.exit_code == 1
    assert cli


def test_stdout_is_a_pipe_is_false_for_a_replaced_stdout(monkeypatch):
    """The discriminator itself: a stdout with no real fileno (a wrapped or
    captured stream) is not a confirmable pipe, so an EINVAL under it is treated
    as a real error and reported - failing toward surfacing, never swallowing."""
    import io

    from localm.cli._core import _stdout_is_a_pipe

    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert _stdout_is_a_pipe() is False


def test_stdout_is_a_pipe_is_true_for_a_real_pipe(monkeypatch):
    """The discriminator's positive direction, on a genuine OS pipe."""
    from localm.cli._core import _stdout_is_a_pipe

    r, w = os.pipe()
    f = os.fdopen(w, "w")
    try:
        monkeypatch.setattr(sys, "stdout", f)
        assert _stdout_is_a_pipe() is True
    finally:
        monkeypatch.undo()
        f.close()
        os.close(r)
