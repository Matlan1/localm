# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reach-by-name (mDNS ``localm.local`` + Tailscale MagicDNS): localm/netname.py.

Covers name sanitization, Tailscale ``status --json`` parsing, the certificate
SAN name list, the printed reachable-target ordering, config validation, and a
live mDNS round-trip that proves ``<name>.local`` is actually advertised and
browsable. Tailscale is always mocked (no real tailnet needed); the synthetic
node/tailnet names here are fictitious, never a real machine's.
"""
import socket
import time

import pytest

from localm import netname, settings_schema


@pytest.fixture(autouse=True)
def _reset_ts_cache():
    """Each test gets a clean Tailscale-status memo (netname caches it per run)."""
    netname._reset_status_cache()
    yield
    netname._reset_status_cache()


def _ts_status(node="mybox", suffix="tailnet.ts.net", magic=True, running=True):
    """A synthetic ``tailscale status --json`` payload (fictitious names)."""
    return {
        "BackendState": "Running" if running else "Stopped",
        "MagicDNSSuffix": suffix if magic else "",
        "CurrentTailnet": {"MagicDNSEnabled": magic, "MagicDNSSuffix": suffix,
                           "Name": "example.com"},
        "Self": {"DNSName": f"{node}.{suffix}.", "HostName": node.upper(),
                 "TailscaleIPs": ["100.64.0.9", "fd7a:115c::9"]},
    }


# ------------------------------------------------------------------ #
#  Name sanitization                                                 #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("raw,expect", [
    ("localm", "localm"),
    ("My Studio!", "my-studio"),
    ("  UPPER  ", "upper"),
    ("--a__b--", "a-b"),
    ("a.b.c", "a-b-c"),
    ("!!!", "localm"),          # nothing usable -> default
    ("", "localm"),
    ("x" * 80, "x" * 63),       # capped at a 63-char DNS label
])
def test_mdns_label_sanitization(raw, expect):
    assert netname.mdns_label(raw) == expect


@pytest.mark.parametrize("raw,expect", [
    ("localm", "localm"),
    ("My Box", "my-box"),
    ("a.b", "a-b"),
    ("!!!", ""),               # no fallback: unusable input -> ""
    ("日本", ""),               # non-ASCII strips to nothing (no silent default)
    ("ΩΩΩ", ""),
    ("   ", ""),
])
def test_normalize_label_no_fallback(raw, expect):
    assert netname.normalize_label(raw) == expect


def test_mdns_label_from_config(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"mdns_name": "My Box"})
    assert netname.mdns_label() == "my-box"
    assert netname.mdns_fqdn() == "my-box.local"


def test_mdns_label_default_when_config_missing(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    assert netname.mdns_label() == "localm"
    assert netname.mdns_fqdn() == "localm.local"


def test_mdns_enabled_default_true(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    assert netname.mdns_enabled() is True
    monkeypatch.setattr("localm.config.load_config", lambda: {"mdns_enabled": False})
    assert netname.mdns_enabled() is False


def test_mdns_enabled_fails_closed_when_config_unreadable(monkeypatch):
    """An unreadable config must resolve mdns_enabled() to FALSE, never the
    on-by-default True: a transiently unreadable config must not silently
    re-enable mDNS for a user who set mdns_enabled:false as an opt-out (mirrors
    netpolicy.network_mode's fail-safe). Label callers still get the default."""
    def boom():
        raise OSError("config unreadable")
    monkeypatch.setattr("localm.config.load_config", boom)
    assert netname.mdns_enabled() is False               # fail CLOSED, not True
    # but a defaulted read still works (mdns_label falls back to the default):
    assert netname.mdns_label() == "localm"


def test_hostname_fqdn(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "Desk-Top.corp.example")
    assert netname.hostname_fqdn() == "desk-top.local"   # first label, lowercased
    monkeypatch.setattr(socket, "gethostname", lambda: "")
    assert netname.hostname_fqdn() == ""


# ------------------------------------------------------------------ #
#  Tailscale parsing                                                 #
# ------------------------------------------------------------------ #

def test_tailscale_names_running_magicdns(monkeypatch):
    monkeypatch.setattr(netname, "_run_tailscale_status", lambda: _ts_status())
    assert netname.tailscale_names() == ["mybox.tailnet.ts.net", "mybox"]
    assert netname.tailscale_node_label() == "mybox"
    assert netname.tailscale_name_reachable() is True


def test_tailscale_absent(monkeypatch):
    monkeypatch.setattr(netname, "_run_tailscale_status", lambda: None)
    assert netname.tailscale_names() == []
    assert netname.tailscale_node_label() == ""
    assert netname.tailscale_name_reachable() is False
    assert netname.tailscale_rename_hint() is None


def test_tailscale_stopped(monkeypatch):
    monkeypatch.setattr(netname, "_run_tailscale_status",
                        lambda: _ts_status(running=False))
    # A stopped backend is not reachable and offers no names.
    assert netname.tailscale_names() == []
    assert netname.tailscale_name_reachable() is False


def test_tailscale_magicdns_off_still_names_but_not_reachable(monkeypatch):
    monkeypatch.setattr(netname, "_run_tailscale_status",
                        lambda: _ts_status(magic=False))
    # Names still belong in the cert, but peers cannot resolve them.
    assert netname.tailscale_names() == ["mybox.tailnet.ts.net", "mybox"]
    assert netname.tailscale_name_reachable() is False


def test_tailscale_rename_hint(monkeypatch):
    monkeypatch.setattr(netname, "_run_tailscale_status",
                        lambda: _ts_status(node="mybox"))
    monkeypatch.setattr("localm.config.load_config", lambda: {"mdns_name": "localm"})
    hint = netname.tailscale_rename_hint()
    assert hint and "tailscale up --hostname=localm" in hint


def test_tailscale_rename_hint_none_when_already_named(monkeypatch):
    monkeypatch.setattr(netname, "_run_tailscale_status",
                        lambda: _ts_status(node="localm"))
    monkeypatch.setattr("localm.config.load_config", lambda: {"mdns_name": "localm"})
    assert netname.tailscale_rename_hint() is None


def test_tailscale_status_bad_json(monkeypatch):
    """A CLI that returns junk must not raise - it degrades to 'no Tailscale'."""
    class _P:
        returncode = 0
        stdout = "not json {["
    monkeypatch.setattr(netname, "_tailscale_cli", lambda: "tailscale")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: _P())
    assert netname._run_tailscale_status() is None


# ------------------------------------------------------------------ #
#  Certificate SANs                                                  #
# ------------------------------------------------------------------ #

def test_cert_hostnames_includes_localm_local(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"mdns_name": "localm",
                                                              "mdns_enabled": True})
    monkeypatch.setattr(netname, "_run_tailscale_status", lambda: None)
    monkeypatch.setattr(socket, "gethostname", lambda: "thebox")
    names = netname.cert_hostnames()
    assert "localm.local" in names
    assert "thebox.local" in names


def test_cert_hostnames_omits_name_when_mdns_off(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"mdns_enabled": False})
    monkeypatch.setattr(netname, "_run_tailscale_status", lambda: None)
    assert "localm.local" not in netname.cert_hostnames()


def test_cert_hostnames_includes_tailscale_and_dedups(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"mdns_name": "localm",
                                                              "mdns_enabled": True})
    monkeypatch.setattr(netname, "_run_tailscale_status", lambda: _ts_status())
    monkeypatch.setattr(socket, "gethostname", lambda: "mybox")
    names = netname.cert_hostnames()
    assert "mybox.tailnet.ts.net" in names
    assert "mybox" in names
    assert len(names) == len(set(names))          # de-duplicated
    assert names == [n.lower() for n in names]    # all lowercased


# ------------------------------------------------------------------ #
#  Reachable-target ordering                                         #
# ------------------------------------------------------------------ #

def test_network_targets_names_before_ips(monkeypatch):
    monkeypatch.setattr("localm.tls.companion_addresses",
                        lambda: {"lan": "192.168.1.9", "tailscale": "100.64.0.9"})
    monkeypatch.setattr(netname, "_run_tailscale_status", lambda: _ts_status())
    # mdns_name = the LIVE advertised name (the caller passes it only when the
    # advertiser actually started); names come before IPs.
    targets = netname.network_targets(mdns_name="localm.local")
    assert targets == [
        ("on this network", "localm.local"),
        ("on this network (IP)", "192.168.1.9"),
        ("via Tailscale", "mybox.tailnet.ts.net"),
        ("via Tailscale (IP)", "100.64.0.9"),
    ]


def test_network_targets_omits_name_without_advertiser(monkeypatch):
    """The LAN name is recommended ONLY when the advertiser is live (mdns_name
    passed). Without it (loopback/isolated/name-taken/mdns-off) we never point a
    user at a name this host does not own - only the IP."""
    monkeypatch.setattr("localm.tls.companion_addresses",
                        lambda: {"lan": "192.168.1.9", "tailscale": ""})
    monkeypatch.setattr(netname, "_run_tailscale_status", lambda: None)
    targets = netname.network_targets()          # no mdns_name -> no LAN name
    assert all(lbl != "on this network" for lbl, _ in targets)
    assert ("on this network (IP)", "192.168.1.9") in targets
    # When the advertiser IS live, the name leads.
    targets2 = netname.network_targets(mdns_name="studio.local")
    assert targets2[0] == ("on this network", "studio.local")


def test_network_targets_magicdns_off_uses_ip_only(monkeypatch):
    monkeypatch.setattr("localm.tls.companion_addresses",
                        lambda: {"lan": "192.168.1.9", "tailscale": "100.64.0.9"})
    monkeypatch.setattr(netname, "_run_tailscale_status",
                        lambda: _ts_status(magic=False))
    targets = netname.network_targets(mdns_name="localm.local")
    labels = [t[0] for t in targets]
    assert "via Tailscale" not in labels          # name would not resolve
    assert ("via Tailscale (IP)", "100.64.0.9") in targets


def test_ca_trust_host(monkeypatch):
    assert netname.ca_trust_host("localm.local") == "localm.local"
    monkeypatch.setattr("localm.tls.companion_addresses",
                        lambda: {"lan": "192.168.1.9", "tailscale": ""})
    assert netname.ca_trust_host() == "192.168.1.9"     # no advertised name -> LAN IP


# ------------------------------------------------------------------ #
#  Config validation                                                 #
# ------------------------------------------------------------------ #

def test_config_defaults_present():
    from localm.config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["mdns_name"] == "localm"
    assert DEFAULT_CONFIG["mdns_enabled"] is True


def test_validate_mdns_name_sanitized():
    assert settings_schema.validate_update({"mdns_name": "My Box"}) == {"mdns_name": "my-box"}


@pytest.mark.parametrize("bad", ["", "!!!", "日本", "ΩΩΩ", "   "])
def test_validate_mdns_name_rejects_unusable(bad):
    # Non-ASCII names must be REJECTED, not silently rewritten to 'localm'
    # (str.isalnum() is Unicode-aware; normalize_label is the correct gate).
    with pytest.raises(ValueError):
        settings_schema.validate_update({"mdns_name": bad})


def test_validate_mdns_enabled_bool():
    assert settings_schema.validate_update({"mdns_enabled": "off"}) == {"mdns_enabled": False}
    assert settings_schema.validate_update({"mdns_enabled": "yes"}) == {"mdns_enabled": True}


# ------------------------------------------------------------------ #
#  mDNS advertiser                                                   #
# ------------------------------------------------------------------ #

def test_start_advertiser_disabled_returns_none(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"mdns_enabled": False})
    assert netname.start_advertiser(8642, tls=True) is None


def test_start_advertiser_no_lan_returns_none(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"mdns_enabled": True,
                                                              "mdns_name": "localm"})
    # start_advertiser falls back to companion_addresses()'s "lan" pick, not
    # the raw outbound-probe, so patch that: no LAN address -> nothing to
    # advertise.
    monkeypatch.setattr("localm.tls.companion_addresses",
                        lambda: {"lan": "", "tailscale": ""})
    assert netname.start_advertiser(8642, tls=True, addresses=[]) is None


def test_start_advertiser_falls_back_to_companion_lan(monkeypatch):
    """No explicit addresses supplied: uses companion_addresses()'s "lan" pick
    (which already excludes a VPN's virtual tunnel adapter), not the raw
    outbound-route probe directly. With a VPN as the default route the raw
    probe makes the mDNS <name>.local advertisement point at the unreachable
    VPN tunnel IP instead of the real LAN address."""
    monkeypatch.setattr("localm.config.load_config", lambda: {"mdns_enabled": True,
                                                              "mdns_name": "localm"})
    monkeypatch.setattr("localm.tls.companion_addresses",
                        lambda: {"lan": "192.168.1.50", "tailscale": ""})

    captured = {}

    class _FakeServiceInfo:
        def __init__(self, *a, addresses=None, **kw):
            captured["addresses"] = addresses

    class _FakeZeroconf:
        def register_service(self, info):
            pass

        def close(self):
            pass

    monkeypatch.setattr("zeroconf.ServiceInfo", _FakeServiceInfo)
    monkeypatch.setattr("zeroconf.Zeroconf", _FakeZeroconf)

    adv = netname.start_advertiser(8642, tls=True)
    assert adv is not None
    assert captured["addresses"] == [socket.inet_aton("192.168.1.50")]


def test_start_advertiser_live_round_trip(monkeypatch):
    """Start the advertiser and browse it back from a separate zeroconf instance:
    proves ``<name>.local`` is really published with the right host/port/address.
    Uses a unique name so it never collides with a real localm on the network."""
    from zeroconf import ServiceBrowser, Zeroconf

    name = "localmtest%d" % (int(time.time()) % 100000)
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"mdns_enabled": True, "mdns_name": name})

    adv = netname.start_advertiser(8642, tls=True, addresses=["192.0.2.7"])
    if adv is None:
        pytest.skip("mDNS advertiser could not start in this environment")

    found = {}

    class _Listener:
        def add_service(self, zc, type_, sname):
            info = zc.get_service_info(type_, sname, timeout=3000)
            if info and sname.startswith(name + "."):
                found["server"] = info.server
                found["port"] = info.port
                found["addrs"] = [socket.inet_ntoa(a) for a in info.addresses]

        def update_service(self, *a):
            pass

        def remove_service(self, *a):
            pass

    zc = Zeroconf()
    try:
        ServiceBrowser(zc, "_https._tcp.local.", _Listener())
        deadline = time.time() + 10
        while time.time() < deadline and "server" not in found:
            time.sleep(0.2)
    finally:
        zc.close()
        adv.close()

    if "server" not in found:
        pytest.skip("mDNS browse found nothing (no multicast in this environment)")
    assert found["server"] == f"{name}.local."
    assert found["port"] == 8642
    assert "192.0.2.7" in found["addrs"]
