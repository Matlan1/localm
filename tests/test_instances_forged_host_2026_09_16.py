# SPDX-License-Identifier: AGPL-3.0-or-later
"""The /whoami identity handshake behind `instances.default_probe` only ever
dials an address THIS machine holds.

A per-install registry entry (`<LOCALM_HOME>/run/<id>.json`) is untrusted
input. `default_probe` forwards the entry's recorded `host` into
`fetch_whoami` so an instance bound on `::1`, or on one interface's own
address, is probed where it actually listens - and that forward must never
become a request to an address the entry merely names. `bindhost.is_own_address`
is the gate. These tests pin the gate and the probe's use of it, running the
handshake against a REAL loopback HTTP server rather than a patched `requests`,
so the legitimate forward is exercised for real alongside the refused one.

Every consumer of `default_probe` inherits the property: `snapshot` (the
`localm ps` listing and GET /api/instances), `find_attachable` /
`attach_target` (every CLI attach), the MCP `server_activity` tool and
`selfclient.read_model_file_hold`.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import bindhost, gpu_registry, instances

FORGED_ID = "f0f0f0f0f0f0f0f0"


class _WhoamiHandler(BaseHTTPRequestHandler):
    payload: dict = {}
    redirect_to: str = ""
    hits: list = []

    def do_GET(self):
        if self.path != "/whoami":
            self.send_response(404)
            self.end_headers()
            return
        self.hits.append(self.path)
        if self.redirect_to:
            self.send_response(302)
            self.send_header("Location", self.redirect_to)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = json.dumps(self.payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


class _V6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


@contextmanager
def whoami_server(payload: dict, bind: str = "127.0.0.1", redirect_to: str = "",
                  hits: list | None = None):
    """A real HTTP server answering GET /whoami with *payload*, bound on *bind*
    at a throwaway port. Yields the port. With *redirect_to* it answers a 302
    to that URL instead; *hits* collects every /whoami request path it saw.
    Skips the test when *bind* cannot be bound on this box (no IPv6 loopback,
    an interface address that went away)."""
    handler = type("_Bound", (_WhoamiHandler,), {
        "payload": payload, "redirect_to": redirect_to,
        "hits": hits if hits is not None else []})
    cls = _V6Server if ipaddress.ip_address(bind).version == 6 else ThreadingHTTPServer
    try:
        srv = cls((bind, 0), handler)
    except OSError as e:
        pytest.skip(f"cannot bind a listener on {bind}: {e}")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def write_home_entry(home, *, instance_id, port, host, scheme="http",
                     root_dir="/proj/mine"):
    """An entry in this install's OWN registry, matching register_instance's
    schema, with this process's pid so the entry is live and never reaped."""
    entry = dict(instance_id=instance_id, pid=os.getpid(), port=port, host=host,
                 root_dir=root_dir, mode="api", version="test",
                 token=instances.new_token(),
                 started="2026-09-16T00:00:00+00:00")
    if scheme is not None:
        entry["scheme"] = scheme
    path = instances.registry_path(home, instance_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry), encoding="utf-8")
    return path


def this_machines_non_loopback_ipv4():
    """One IPv4 address configured on a non-loopback interface of this box, or
    None. The positive control for the interface-address arm of the gate."""
    psutil = pytest.importorskip("psutil")
    for addrs in psutil.net_if_addrs().values():
        for a in addrs:
            if a.family != socket.AF_INET:
                continue
            ip = ipaddress.ip_address(a.address)
            if not ip.is_loopback and not ip.is_link_local and not ip.is_unspecified:
                return a.address
    return None


@pytest.fixture
def requests_spy(monkeypatch):
    """Records every URL `requests.get` is asked for. A loopback or own-address
    URL goes through to the real `requests.get`; anything else is recorded and
    refused with a ConnectionError, so a regression can never send a packet off
    this machine from inside the suite. The recorded list is the evidence: the
    property under test is that the forged address is never asked for at all."""
    real_get = requests.get
    calls: list = []

    def spy_get(url, *a, **kw):
        calls.append(url)
        host = requests.utils.urlparse(url).hostname or ""
        if not bindhost.is_own_address(host):
            raise requests.ConnectionError(f"refused by the test spy: {url}")
        return real_get(url, *a, **kw)

    monkeypatch.setattr(requests, "get", spy_get)
    return calls


@pytest.fixture
def dns_spy(monkeypatch):
    """Records every name handed to the resolver, so "never dialed" can also be
    shown to mean "never looked up"."""
    looked_up: list = []
    real = socket.getaddrinfo

    def spy(host, *a, **kw):
        looked_up.append(host)
        return real(host, *a, **kw)

    monkeypatch.setattr(socket, "getaddrinfo", spy)
    return looked_up


@pytest.fixture
def gpu_dir(tmp_path, monkeypatch):
    """Point the machine-wide registry at a throwaway dir, so the listing route
    never reads or probes whatever localm is really running on this box."""
    d = tmp_path / "machine-registry"
    d.mkdir()
    monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
    return d


# --------------------------------------------------------------------------- #
#  The gate: bindhost.is_own_address                                          #
# --------------------------------------------------------------------------- #

class TestIsOwnAddress:
    @pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.5", "::1"])
    def test_a_loopback_literal_is_own(self, host):
        assert bindhost.is_own_address(host) is True

    def test_an_interface_address_of_this_machine_is_own(self):
        """The positive control for the interface arm: without it, every
        refusal below could be the enumeration returning nothing."""
        addr = this_machines_non_loopback_ipv4()
        if addr is None:
            pytest.skip("this box has no non-loopback IPv4 address")
        assert bindhost.is_own_address(addr) is True

    def test_a_hostname_is_refused_without_a_lookup(self, dns_spy):
        assert bindhost.is_own_address("attacker.example") is False
        assert bindhost.is_own_address("localhost") is False
        assert dns_spy == [], (
            f"classifying a hostname resolved it: {dns_spy}")

    @pytest.mark.parametrize("host", [
        "203.0.113.5",          # TEST-NET-3: never on an interface
        "198.51.100.7",         # TEST-NET-2
        "2001:db8::1",          # documentation prefix
        "0.0.0.0", "::",        # wildcards are not connectable addresses
        "", None, 123, ["127.0.0.1"],
        "127.0.0.1:80", "[::1]",
    ])
    def test_anything_this_machine_does_not_hold_is_refused(self, host):
        assert bindhost.is_own_address(host) is False

    def test_a_zone_id_is_ignored_in_the_listing_and_refused_in_the_input(
            self, monkeypatch):
        """psutil lists a link-local address with its zone; a bind host never
        carries one (is_valid_bind_host refuses it), so a zoned input is
        refused rather than matched on its prefix and dialed with the suffix."""
        psutil = pytest.importorskip("psutil")
        monkeypatch.setattr(psutil, "net_if_addrs", lambda: {
            "eth0": [SimpleNamespace(family=socket.AF_INET6,
                                     address="fe80::1%eth0"),
                     SimpleNamespace(family=socket.AF_INET, address="10.9.8.7"),
                     SimpleNamespace(family=-1, address="00-11-22-33-44-55")],
        })
        assert bindhost.is_own_address("fe80::1") is True
        assert bindhost.is_own_address("10.9.8.7") is True
        assert bindhost.is_own_address("fe80::2") is False
        assert bindhost.is_own_address("fe80::1%eth0") is False
        assert bindhost.is_own_address("::1%attacker.example") is False

    def test_without_psutil_only_loopback_is_own(self, monkeypatch, caplog):
        addr = this_machines_non_loopback_ipv4()
        if addr is None:
            pytest.skip("this box has no non-loopback IPv4 address")
        assert bindhost.is_own_address(addr) is True   # with psutil: the control
        monkeypatch.setitem(sys.modules, "psutil", None)
        with caplog.at_level("WARNING", logger="localm"):
            assert bindhost.is_own_address(addr) is False, (
                "with no way to enumerate interfaces, a non-loopback literal "
                "must fail closed rather than be taken on trust")
        assert "could not enumerate this machine's interface addresses" in caplog.text, (
            "the refusal must be traceable to its cause, not read as a bad "
            "address")
        assert bindhost.is_own_address("127.0.0.1") is True
        assert bindhost.is_own_address("::1") is True


# --------------------------------------------------------------------------- #
#  The probe: fetch_whoami / default_probe / snapshot / find_attachable        #
# --------------------------------------------------------------------------- #

class TestDefaultProbeDialsOnlyThisMachine:
    def test_a_forged_host_is_never_dialed_and_never_verified(
            self, tmp_path, requests_spy, dns_spy):
        """The registry entry names an outside host; a real loopback listener
        on the same port answers with the forged id, so a probe that fell back
        to loopback (or dialed the named host and got an answer) would mark the
        entry alive. Neither happens: nothing is dialed, nothing is looked up,
        and the entry is reported dead."""
        home = tmp_path / "home"
        with whoami_server({"app": "localm", "instance_id": FORGED_ID,
                            "root_dir": "/proj/mine", "mode": "api",
                            "version": "9.9.9"}) as port:
            write_home_entry(home, instance_id=FORGED_ID, port=port,
                             host="attacker.example")

            rows = instances.snapshot(home)

        assert requests_spy == [], (
            f"a forged registry entry made this process dial: {requests_spy}")
        assert "attacker.example" not in dns_spy, (
            f"a forged registry entry made this process resolve its host: "
            f"{dns_spy}")
        assert [r["instance_id"] for r in rows] == [FORGED_ID]
        assert rows[0]["alive"] is False

    def test_a_forged_literal_is_never_dialed(self, tmp_path, requests_spy):
        """A literal address (TEST-NET-3) that no interface here holds. Two
        entries, one without a recorded scheme so both http and https would be
        tried: still not a single request."""
        home = tmp_path / "home"
        write_home_entry(home, instance_id=FORGED_ID, port=8642,
                         host="203.0.113.5")
        write_home_entry(home, instance_id="f1f1f1f1f1f1f1f1", port=8643,
                         host="203.0.113.5", scheme=None)

        rows = instances.snapshot(home)

        assert requests_spy == [], (
            f"a forged registry entry made this process dial: {requests_spy}")
        assert {r["instance_id"]: r["alive"] for r in rows} == {
            FORGED_ID: False, "f1f1f1f1f1f1f1f1": False}

    def test_default_probe_refuses_a_forged_host_directly(self, requests_spy):
        assert instances.default_probe({"port": 8642, "instance_id": FORGED_ID,
                                        "scheme": "http",
                                        "host": "attacker.example"}) is False
        assert instances.default_probe({"port": 8642, "instance_id": FORGED_ID,
                                        "scheme": "http",
                                        "host": ["127.0.0.1"]}) is False
        assert requests_spy == []

    def test_find_attachable_never_attaches_to_a_forged_host(
            self, tmp_path, requests_spy):
        """The attach path: a CLI in the entry's project dir must not be handed
        the forged endpoint. The entry's pid is live, so it is not reaped either -
        it is simply never verified."""
        home = tmp_path / "home"
        proj = tmp_path / "proj"
        proj.mkdir()
        path = write_home_entry(home, instance_id=FORGED_ID, port=8642,
                                host="203.0.113.5", root_dir=str(proj))

        assert instances.find_attachable(home, str(proj)) is None
        assert instances.attach_target(home, str(proj)) is None
        assert requests_spy == []
        assert path.exists()

    @pytest.mark.parametrize("recorded,bind,dialed", [
        (None, "127.0.0.1", "127.0.0.1"),
        ("0.0.0.0", "127.0.0.1", "127.0.0.1"),
        ("localhost", "127.0.0.1", "127.0.0.1"),
        ("::", "::1", "[::1]"),
    ])
    def test_a_wildcard_or_localhost_or_absent_host_still_dials_loopback(
            self, tmp_path, requests_spy, recorded, bind, dialed):
        """The gate sees the address self_connect_host would dial, not the raw
        recorded value: ``localhost`` and the wildcards are not addresses this
        machine holds, yet an entry recording one is a loopback server and is
        probed there. Moving the gate ahead of that mapping would report every
        default-bound instance dead."""
        home = tmp_path / "home"
        with whoami_server({"app": "localm", "instance_id": FORGED_ID,
                            "root_dir": "/proj/mine", "mode": "api",
                            "version": "9.9.9"}, bind=bind) as port:
            write_home_entry(home, instance_id=FORGED_ID, port=port,
                             host=recorded)

            rows = instances.snapshot(home)

        assert requests_spy == [f"http://{dialed}:{port}/whoami"]
        assert rows[0]["alive"] is True

    def test_the_recorded_ipv6_loopback_bind_host_is_still_forwarded_for_real(
            self, tmp_path, requests_spy):
        """The reason the host is forwarded at all: a server bound on ``::1``
        has nothing listening on 127.0.0.1. Exercised end to end through the
        real requests path against a real IPv6 loopback listener."""
        home = tmp_path / "home"
        with whoami_server({"app": "localm", "instance_id": FORGED_ID,
                            "root_dir": "/proj/mine", "mode": "api",
                            "version": "9.9.9"}, bind="::1") as port:
            write_home_entry(home, instance_id=FORGED_ID, port=port, host="::1")

            rows = instances.snapshot(home)

        assert requests_spy == [f"http://[::1]:{port}/whoami"]
        assert rows[0]["alive"] is True

    def test_a_bind_on_this_machines_own_interface_address_is_still_probed(
            self, tmp_path, requests_spy):
        """A server bound on one interface's address (not the wildcard) has
        nothing listening on loopback at all, so the probe must dial that
        address - the case a loopback-only rule would break."""
        addr = this_machines_non_loopback_ipv4()
        if addr is None:
            pytest.skip("this box has no non-loopback IPv4 address")
        home = tmp_path / "home"
        with whoami_server({"app": "localm", "instance_id": FORGED_ID,
                            "root_dir": "/proj/mine", "mode": "api",
                            "version": "9.9.9"}, bind=addr) as port:
            write_home_entry(home, instance_id=FORGED_ID, port=port, host=addr)

            rows = instances.snapshot(home)

        assert requests_spy == [f"http://{addr}:{port}/whoami"]
        assert rows[0]["alive"] is True

    def test_a_redirect_from_a_local_listener_is_not_followed(
            self, tmp_path, requests_spy):
        """A listener on the entry's own port answers /whoami with a redirect
        to a second listener that serves a matching payload. The handshake is
        one hop: the redirect is not followed, the second listener is never
        asked, and a 302 is not a verified answer."""
        home = tmp_path / "home"
        target_hits: list = []
        with whoami_server({"app": "localm", "instance_id": FORGED_ID,
                            "root_dir": "/proj/mine", "mode": "api",
                            "version": "9.9.9"}, hits=target_hits) as target_port:
            with whoami_server({}, redirect_to=(
                    f"http://127.0.0.1:{target_port}/whoami")) as port:
                write_home_entry(home, instance_id=FORGED_ID, port=port,
                                 host="127.0.0.1")
                rows = instances.snapshot(home)

        assert target_hits == [], (
            f"the probe followed a redirect off the entry's own port: {target_hits}")
        assert requests_spy == [f"http://127.0.0.1:{port}/whoami"]
        assert rows[0]["alive"] is False

    def test_the_forward_is_refused_not_redirected_to_loopback(
            self, tmp_path, requests_spy):
        """An entry naming an outside host describes a server this machine
        cannot reach on loopback; probing loopback instead would verify a
        different listener than the entry describes. So the forged host is
        refused outright, even when a loopback listener on that port would have
        answered with the matching id."""
        home = tmp_path / "home"
        with whoami_server({"app": "localm", "instance_id": FORGED_ID,
                            "root_dir": "/proj/mine", "mode": "api",
                            "version": "9.9.9"}) as port:
            write_home_entry(home, instance_id=FORGED_ID, port=port,
                             host="203.0.113.5")
            rows = instances.snapshot(home)

        assert requests_spy == []
        assert rows[0]["alive"] is False


# --------------------------------------------------------------------------- #
#  The route the entry is reachable through: GET /api/instances (CONFIG_READ)  #
# --------------------------------------------------------------------------- #

@pytest.fixture
def instances_app(tmp_path, monkeypatch):
    """The GUI stack on a throwaway home, standing in for a real advertised
    server (instance_id set, not isolated), same shape as the cross-install
    route tests."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    from localm.plugins.engine import attach_engine
    from localm.plugins.gui.web import attach_gui
    app = FastAPI()
    attach_engine(app)
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None,
               active_model=lambda: "")
    app.state.instance_id = "5e1f000000000000"
    app.state.instance_isolated = False
    return app, home


class TestListingRoute:
    def test_a_listing_reports_a_forged_entry_dead_without_dialing(
            self, instances_app, gpu_dir, requests_spy, dns_spy):
        app, home = instances_app
        with whoami_server({"app": "localm", "instance_id": FORGED_ID,
                            "root_dir": "/proj/mine", "mode": "api",
                            "version": "9.9.9"}) as port:
            write_home_entry(home, instance_id=FORGED_ID, port=port,
                             host="attacker.example")
            with TestClient(app) as c:
                body = c.get("/api/instances").json()

        assert requests_spy == [], (
            f"a read-scope listing made this server dial: {requests_spy}")
        assert "attacker.example" not in dns_spy
        rows = {r["instance_id"]: r for r in body["instances"]}
        assert rows[FORGED_ID]["alive"] is False
        assert rows[FORGED_ID]["same_install"] is True
