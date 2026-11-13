# SPDX-License-Identifier: AGPL-3.0-or-later
"""With net_allow_private on, loopback and private targets are reachable, and
link-local and cloud-metadata addresses are still refused: by check_url before
a request and by the pinned resolve on the exact addresses to be dialled."""

from __future__ import annotations

import socket

import pytest

from localm import netpolicy
from localm.netpolicy import NetworkPolicyError, check_url
from tests._net_config import set_config

_ON = {"net_mode": "allow", "net_allow_private": True}


def _resolves_to(monkeypatch, *addresses):
    def fake(host, port, *a, **k):
        fam = lambda ip: socket.AF_INET6 if ":" in ip else socket.AF_INET  # noqa: E731
        return [(fam(ip), socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in addresses]
    monkeypatch.setattr("socket.getaddrinfo", fake)


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",
    "http://2852039166/latest/meta-data/",
    "http://169.254.1.1/",
    "http://[fe80::1]/",
    "http://[fd00:ec2::254]/",
    "http://[::ffff:169.254.169.254]/",
])
def test_check_url_refuses_link_local_literals_with_private_allowed(monkeypatch, url):
    set_config(monkeypatch, _ON)
    with pytest.raises(NetworkPolicyError) as exc:
        check_url(url)
    assert "even with net_allow_private on" in str(exc.value)
    assert "set net_allow_private true" not in str(exc.value)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:3000/", "http://192.168.1.5/", "http://10.0.0.2:8080/x",
    "http://[::1]:9000/",
])
def test_check_url_permits_loopback_and_private_with_private_allowed(monkeypatch, url):
    set_config(monkeypatch, _ON)
    check_url(url)


def test_check_url_refuses_a_name_that_resolves_to_metadata(monkeypatch):
    set_config(monkeypatch, _ON)
    _resolves_to(monkeypatch, "169.254.169.254")
    with pytest.raises(NetworkPolicyError, match="even with net_allow_private on"):
        check_url("http://metadata.example/latest/")


def test_check_url_permits_a_name_that_resolves_to_a_lan_address(monkeypatch):
    set_config(monkeypatch, _ON)
    _resolves_to(monkeypatch, "192.168.1.20")
    check_url("http://nas.example/")


def test_pinned_resolve_refuses_a_rebind_to_metadata(monkeypatch):
    set_config(monkeypatch, _ON)
    _resolves_to(monkeypatch, "169.254.169.254")
    with pytest.raises(NetworkPolicyError) as exc:
        netpolicy._resolve_pinned_all("rebind.example")
    assert "169.254.169.254" in str(exc.value)
    assert "even with net_allow_private on" in str(exc.value)


def test_pinned_resolve_drops_only_the_link_local_address(monkeypatch):
    set_config(monkeypatch, _ON)
    _resolves_to(monkeypatch, "169.254.169.254", "127.0.0.1", "192.168.1.20")
    assert netpolicy._resolve_pinned_all("mixed.example") == ["127.0.0.1", "192.168.1.20"]


def test_pinned_resolve_refuses_a_link_local_literal(monkeypatch):
    set_config(monkeypatch, _ON)
    with pytest.raises(NetworkPolicyError, match="even with net_allow_private on"):
        netpolicy._resolve_pinned_all("169.254.169.254")


def test_private_off_still_names_the_setting(monkeypatch):
    set_config(monkeypatch, {"net_mode": "allow", "net_allow_private": False})
    with pytest.raises(NetworkPolicyError, match="set net_allow_private true"):
        check_url("http://127.0.0.1:3000/")
