# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pins the address-classification helpers of localm.netpolicy that decide which
resolved addresses a request may be pinned to: the search-backend address
rules, the local-name test, the family interleave, the pinned resolution and
the pinned session."""

import ipaddress
import socket

import pytest

import localm.netpolicy as netpolicy
from localm.netpolicy import NetworkPolicyError
from tests._net_config import set_config


def _addrinfo(*addresses):
    return lambda host, port, *a, **k: [(2, 1, 6, "", (ip, 0)) for ip in addresses]


def _unresolvable(host, port, *a, **k):
    raise socket.gaierror("no such host")


class TestCanonicalHostMapping:
    def test_a_full_width_name_is_mapped_to_its_ascii_form(self):
        name = "ｅｘａｍｐｌｅ.com"
        assert netpolicy._canonical_host(name) == "example.com"


class TestSearchBackendAddressCheck:
    def test_a_dotless_decimal_link_local_literal_is_refused(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _unresolvable)
        with pytest.raises(NetworkPolicyError):
            netpolicy._check_search_backend_address("2852039166")

    def test_a_link_local_ipv6_literal_is_refused(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _unresolvable)
        with pytest.raises(NetworkPolicyError):
            netpolicy._check_search_backend_address("fe80::1")

    def test_a_loopback_ipv6_literal_is_accepted(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _unresolvable)
        netpolicy._check_search_backend_address("::1")

    def test_a_host_name_resolving_to_a_metadata_address_is_refused(self, monkeypatch):
        seen = []

        def resolve(host, port, *a, **k):
            seen.append(host)
            return [(2, 1, 6, "", ("169.254.169.254", 0))]

        monkeypatch.setattr("socket.getaddrinfo", resolve)
        with pytest.raises(NetworkPolicyError, match="169.254.169.254"):
            netpolicy._check_search_backend_address("searx.lan")
        assert seen == ["searx.lan"]

    def test_an_unresolvable_host_is_accepted(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _unresolvable)
        netpolicy._check_search_backend_address("searx.lan")

    def test_an_unparseable_resolver_answer_is_skipped(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [
            (2, 1, 6, "", ("not-an-ip", 0)), (2, 1, 6, "", ("10.0.0.5", 0))])
        netpolicy._check_search_backend_address("searx.lan")


class TestSearchBackendAddressOk:
    @pytest.mark.parametrize("address, expected", [
        ("93.184.216.34", True), ("127.0.0.1", True), ("10.0.0.5", True),
        ("192.168.1.1", True), ("::1", True), ("fd12::1", True),
        ("::ffff:10.0.0.5", True), ("::ffff:127.0.0.1", True),
        ("169.254.169.254", False), ("0.0.0.0", False), ("fe80::1", False),
        ("fd00:ec2::254", False), ("::ffff:169.254.169.254", False),
        ("100.64.0.1", False), ("224.0.0.1", False),
    ])
    def test_which_addresses_the_search_backend_may_use(self, address, expected):
        assert netpolicy._search_backend_address_ok(
            ipaddress.ip_address(address)) is expected


class TestInterleaveFamilies:
    def test_a_longer_second_family_is_appended_in_order(self):
        v4 = ipaddress.ip_address("1.1.1.1")
        v6 = [ipaddress.ip_address(f"2606:4700::{n}") for n in (1, 2, 3)]
        assert netpolicy._interleave_families([v4, *v6]) == [v4, v6[0], v6[1], v6[2]]

    def test_a_longer_first_family_is_appended_in_order(self):
        v4 = [ipaddress.ip_address(f"1.1.1.{n}") for n in (1, 2, 3)]
        v6 = ipaddress.ip_address("2606:4700::1")
        assert netpolicy._interleave_families([*v4, v6]) == [v4[0], v6, v4[1], v4[2]]

    def test_an_empty_list_stays_empty(self):
        assert netpolicy._interleave_families([]) == []


class TestIsLocalNameOrLiteral:
    @pytest.mark.parametrize("host, expected", [
        ("localhost", True), ("box", True), ("searx.lan", True), ("nas.local", True),
        ("a.home.arpa", True), ("10.0.0.5", True), ("2130706433", True),
        ("::1", True), ("::ffff:8.8.8.8", True), ("fe80::1", True),
        ("example.com", False), ("www.example.org", False), ("lan.example.com", False),
    ])
    def test_which_hosts_are_local_by_their_form(self, host, expected):
        assert netpolicy._is_local_name_or_literal(host) is expected


class TestResolvePinnedAll:
    @pytest.fixture(autouse=True)
    def _config(self, monkeypatch):
        set_config(monkeypatch, lambda: {"net_mode": "allow"})

    def test_a_numeric_loopback_literal_is_refused(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _unresolvable)
        with pytest.raises(NetworkPolicyError):
            netpolicy._resolve_pinned_all("2130706433")

    def test_a_link_local_ipv6_literal_is_refused_and_a_public_one_is_returned(
            self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _unresolvable)
        with pytest.raises(NetworkPolicyError):
            netpolicy._resolve_pinned_all("fe80::1")
        assert netpolicy._resolve_pinned_all("2606:4700::1111") == ["2606:4700::1111"]

    def test_the_host_name_is_what_the_resolver_is_asked_for(self, monkeypatch):
        seen = []

        def resolve(host, port, *a, **k):
            seen.append(host)
            return [(2, 1, 6, "", ("93.184.216.34", 0))]

        monkeypatch.setattr("socket.getaddrinfo", resolve)
        assert netpolicy._resolve_pinned_all("example.com") == ["93.184.216.34"]
        assert seen == ["example.com"]

    def test_net_allow_private_lets_a_private_address_through(self, monkeypatch):
        set_config(monkeypatch, lambda: {"net_allow_private": True})
        monkeypatch.setattr("socket.getaddrinfo", _addrinfo("10.0.0.5"))
        assert netpolicy._resolve_pinned_all("box.lan") == ["10.0.0.5"]
        assert netpolicy._resolve_pinned_all("10.0.0.5") == ["10.0.0.5"]

    def test_a_blocked_address_does_not_hide_a_later_public_one(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _addrinfo("10.0.0.5", "93.184.216.34"))
        assert netpolicy._resolve_pinned_all("example.com") == ["93.184.216.34"]

    def test_only_blocked_addresses_are_refused_with_the_exact_message(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _addrinfo("10.0.0.5"))
        with pytest.raises(NetworkPolicyError) as exc:
            netpolicy._resolve_pinned_all("example.com")
        assert str(exc.value) == (
            "'example.com' resolves to the non-public address 10.0.0.5. "
            "Requests to local/private networks are blocked "
            "(set net_allow_private true to permit them).")

    def test_an_answer_with_no_parseable_address_is_unresolvable_not_blocked(
            self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [
            (2, 1, 6, "", ("not-an-ip", 0))])
        assert netpolicy._resolve_pinned_all("example.com") == []

    def test_an_unresolvable_host_gives_no_addresses(self, monkeypatch):
        monkeypatch.setattr("socket.getaddrinfo", _unresolvable)
        assert netpolicy._resolve_pinned_all("example.com") == []


class TestSessionFor:
    @pytest.fixture(autouse=True)
    def _config(self, monkeypatch):
        set_config(monkeypatch, lambda: {"net_mode": "allow"})

    def test_the_session_is_pinned_to_the_address_of_the_urls_host(self, monkeypatch):
        seen = []

        def resolve(host, port, *a, **k):
            seen.append(host)
            return [(2, 1, 6, "", ("93.184.216.34", 0))]

        pinned = []
        monkeypatch.setattr("socket.getaddrinfo", resolve)
        monkeypatch.setattr("localm.netpin.pinned_session",
                            lambda ips: pinned.append(list(ips)) or object())
        netpolicy._session_for("https://Example.com/x")
        assert seen == ["example.com"]
        assert pinned == [["93.184.216.34"]]

    def test_a_url_without_a_host_is_unresolvable(self, monkeypatch):
        def resolve(host, port, *a, **k):
            if host == "":
                raise socket.gaierror("empty host")
            return [(2, 1, 6, "", ("93.184.216.34", 0))]

        monkeypatch.setattr("socket.getaddrinfo", resolve)
        with pytest.raises(netpolicy.UnresolvableHost):
            netpolicy._session_for("http:///x")
