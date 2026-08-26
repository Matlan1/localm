# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end IPv6 bind support.

``localm gui -H ::`` must start: an AF_INET-only port-availability probe raises
``socket.gaierror`` on any IPv6 host before anything is bound, and the process
exits through the unexpected-error path.

Each test names the control that makes it go red.
"""

from __future__ import annotations

import socket

import pytest

from localm import netlisten
from localm.bindhost import (is_loopback_host, is_valid_bind_host,
                             self_connect_host, url_host)
from localm.config import port_in_use
from localm.console import show_url


# ------------------------------------------------------------------ #
#  The crash itself                                                   #
# ------------------------------------------------------------------ #

class TestPortProbeIsFamilyAware:
    @pytest.mark.parametrize("host", ["::", "::1", "2001:db8::5", "127.0.0.1",
                                      "0.0.0.0", "localhost"])
    def test_probe_never_raises_on_any_bindable_host(self, host):
        """The original defect verbatim: a bare ``socket.socket()`` is AF_INET,
        so ``connect_ex(("::1", port))`` raised gaierror. Control: restore the
        AF_INET socket and this goes red with socket.gaierror on the IPv6
        params."""
        assert port_in_use(9, host) in (True, False)

    def test_probe_reports_a_listening_ipv6_socket(self):
        """The instrument must be able to answer TRUE over IPv6, or every False
        above proves nothing: a probe that could only ever say "free" would pass
        the test above while being useless."""
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        try:
            s.bind(("::1", 0))
            s.listen(1)
            port = s.getsockname()[1]
            assert port_in_use(port, "::1") is True
        finally:
            s.close()

    def test_unresolvable_host_is_not_reported_as_busy(self):
        """A name that cannot resolve is not evidence the port is taken. The
        caller goes on to bind and produces the accurate error; claiming a
        conflict here would send the user hunting a collision that is not
        there."""
        assert port_in_use(9, "no-such-host.invalid") is False


# ------------------------------------------------------------------ #
#  The listening socket: :: must reach IPv4 clients too               #
# ------------------------------------------------------------------ #

class TestListenSocket:
    def test_ipv6_wildcard_is_dual_stack(self):
        """asyncio's create_server sets IPV6_V6ONLY unconditionally, which would
        make ``-H ::`` unreachable from every IPv4 client on the LAN. Control:
        drop the setsockopt in netlisten and this goes red with V6ONLY == 1."""
        if not socket.has_dualstack_ipv6():
            pytest.skip("this platform cannot clear IPV6_V6ONLY")
        sock = netlisten.create_listen_socket("::", 0)
        try:
            assert sock.family == socket.AF_INET6
            assert sock.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 0
        finally:
            sock.close()

    def test_a_specific_ipv6_literal_is_not_widened(self):
        """A single address names one family. Clearing V6ONLY on ``::1`` would
        claim a reach it does not have."""
        sock = netlisten.create_listen_socket("::1", 0)
        try:
            assert sock.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 1
        finally:
            sock.close()

    @pytest.mark.parametrize("host, family", [
        ("127.0.0.1", socket.AF_INET), ("0.0.0.0", socket.AF_INET),
        ("::1", socket.AF_INET6), ("::", socket.AF_INET6),
    ])
    def test_family_follows_the_host(self, host, family):
        sock = netlisten.create_listen_socket(host, 0)
        try:
            assert sock.family == family
        finally:
            sock.close()

    def test_dual_stack_wildcard_actually_accepts_an_ipv4_client(self):
        """The property users care about, measured rather than inferred from a
        socket option: an IPv4 client connects to a ``::`` listener.

        BOTH arms are asserted. Without the IPv4 arm this passes on a v6-only
        socket; without the IPv6 arm a dead listener looks like a passing test.
        Control: revert the dual-stack upgrade and the IPv4 arm goes red while
        the IPv6 arm stays green, which is what distinguishes this regression
        from a listener that is simply broken."""
        if not socket.has_dualstack_ipv6():
            pytest.skip("this platform cannot clear IPV6_V6ONLY")
        sock = netlisten.create_listen_socket("::", 0)
        try:
            sock.setblocking(True)
            sock.listen(8)
            port = sock.getsockname()[1]
            for family, addr in ((socket.AF_INET, ("127.0.0.1", port)),
                                 (socket.AF_INET6, ("::1", port))):
                client = socket.socket(family, socket.SOCK_STREAM)
                client.settimeout(5)
                try:
                    client.connect(addr)
                    conn, _peer = sock.accept()
                    conn.close()
                except OSError as exc:
                    pytest.fail(f"a {family.name} client could not reach the "
                                f":: listener: {exc!r}")
                finally:
                    client.close()
        finally:
            sock.close()

    @pytest.mark.parametrize("host, expected", [
        ("::", True), ("0.0.0.0", True), ("", True),
        ("::1", False), ("127.0.0.1", False), ("192.168.1.5", False),
    ])
    def test_wildcard_classification(self, host, expected):
        assert netlisten.is_wildcard_host(host) is expected


# ------------------------------------------------------------------ #
#  Self-calls follow the bind                                         #
# ------------------------------------------------------------------ #

class TestSelfConnectHost:
    @pytest.mark.parametrize("bind, expected", [
        (None, "127.0.0.1"), ("", "127.0.0.1"), ("0.0.0.0", "127.0.0.1"),
        ("localhost", "127.0.0.1"), ("127.0.0.1", "127.0.0.1"),
        # The IPv6 wildcard maps to the IPv6 loopback, NOT 127.0.0.1: localm
        # binds :: dual-stack so the IPv4 loopback happens to work today, but
        # ::1 is reachable even if that upgrade ever degrades.
        ("::", "::1"), ("::1", "::1"),
        ("2001:db8::5", "2001:db8::5"), ("192.168.1.5", "192.168.1.5"),
    ])
    def test_mapping(self, bind, expected):
        assert self_connect_host(bind) == expected

    def test_never_returns_a_wildcard(self):
        """A wildcard is not a connectable address; returning one would produce
        a self-call to an address nothing answers on."""
        for bind in (None, "", "0.0.0.0", "::"):
            assert not netlisten.is_wildcard_host(self_connect_host(bind))

    def test_the_probe_helpers_agree(self):
        """_watchdog_probe_host, _hang_alarm._probe_host and mount_gui_surface
        must all resolve the self-connect host through self_connect_host rather
        than an inline copy. Control: restore any one of the inline copies and
        this goes red."""
        from localm.inference._hang_alarm import _probe_host
        from localm.inference.routes.admin import _watchdog_probe_host
        for bind in (None, "", "0.0.0.0", "::", "::1", "10.0.0.5"):
            assert _probe_host(bind) == self_connect_host(bind)
            assert _watchdog_probe_host(bind) == self_connect_host(bind)


class TestUrlHost:
    @pytest.mark.parametrize("host, expected", [
        ("::1", "[::1]"), ("::", "[::]"), ("2001:db8::5", "[2001:db8::5]"),
        ("fe80::1", "[fe80::1]"),
        ("127.0.0.1", "127.0.0.1"), ("localhost", "localhost"),
        ("localm.local", "localm.local"), ("", ""),
    ])
    def test_brackets_only_ipv6(self, host, expected):
        assert url_host(host) == expected

    def test_is_idempotent(self):
        """Two layers each reaching for url_host must not produce [[::1]]."""
        assert url_host(url_host("::1")) == "[::1]"

    def test_produces_a_parseable_authority(self):
        """The point of the brackets: without them a URL parser reads the
        address's own colons as the port separator."""
        from urllib.parse import urlsplit
        parts = urlsplit("https://" + url_host("::1") + ":8642/v1/models")
        assert parts.hostname == "::1"
        assert parts.port == 8642


class TestShowUrl:
    @pytest.mark.parametrize("host", ["fd7a:115c:a1e0::e44:2839", "fe80::1",
                                      "abcd::1", "::1", "2001:db8::5"])
    def test_a_bracketed_ipv6_url_survives_rich_markup(self, host):
        """Rich reads ``[...]`` as a style tag, so an unescaped printed address
        loses its HOST for every literal starting with a lowercase hex letter -
        every link-local (fe80::) and every unique-local (fd..), while ``[::1]``
        survives.

        Control: drop the escape in console.show_url and the fd7a/fe80/abcd
        params go red while ::1 and 2001:db8::5 stay green."""
        import io

        from rich.console import Console
        url = "https://" + url_host(host) + ":8642/"
        buf = io.StringIO()
        Console(file=buf, width=300, no_color=True).print(show_url(url))
        assert buf.getvalue().strip() == url, "Rich markup ate part of the URL"


# ------------------------------------------------------------------ #
#  The config key, widened LAST                                       #
# ------------------------------------------------------------------ #

class TestBindHostWidening:
    @pytest.mark.parametrize("host", ["::", "::1", "2001:db8::5", "fe80::1"])
    def test_ipv6_literals_are_accepted(self, host):
        assert is_valid_bind_host(host) is True

    @pytest.mark.parametrize("host", ["fe80::1%eth0", "::1%13", "[::1]", "[::]"])
    def test_zone_ids_and_brackets_stay_rejected(self, host):
        """A zone index names an interface as numbered on ONE machine and does
        not survive the getaddrinfo(AI_PASSIVE) the server binds through."""
        assert is_valid_bind_host(host) is False

    def test_widening_did_not_disturb_the_loopback_predicate(self):
        """is_valid_bind_host and is_loopback_host answer different questions,
        and every trust decision in the server reads the second one. Widening
        the first must not move the second."""
        assert is_loopback_host("::1") is True
        assert is_loopback_host("::") is False
        assert is_loopback_host("2001:db8::5") is False

    @pytest.mark.parametrize("host", ["::", "::1", "2001:db8::5"])
    def test_every_accepted_ipv6_value_is_screened_for_bindability(self, host):
        """The unbrickable-startup invariant. Syntax acceptance is not enough,
        so the read site probes a real bind; this asserts the probe can SPEAK
        for an IPv6 host (a reason string or None) rather than raising, which is
        what that call site depends on."""
        from localm.cli._core import _bind_preflight_error
        result = _bind_preflight_error(host)
        assert result is None or isinstance(result, str)

    def test_the_preflight_refuses_a_wellformed_but_unbindable_ipv6(self):
        """Control for the test above: the preflight must be able to say NO, or
        its None answers prove nothing.

        ``::ffff:127.0.0.1`` is why the read site does not skip the loopback
        class: it is well-formed, it IS loopback, and the OS refuses to bind
        it."""
        from localm.cli._core import _bind_preflight_error
        assert is_valid_bind_host("::ffff:127.0.0.1") is True
        assert is_loopback_host("::ffff:127.0.0.1") is True
        if _bind_preflight_error("::ffff:127.0.0.1") is None:
            pytest.skip("this platform binds the IPv4-mapped form")
        assert _bind_preflight_error("::ffff:127.0.0.1") is not None
