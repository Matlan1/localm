# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-instance model ROUTING (the "cheap half" of NEW-CROSS-INSTANCE-MODEL-
SHARING): localm/peer_routing.py + GET/POST/DELETE /v1/models/{id}/peer-route
+ the /v1/chat/completions and /v1/completions forwarding short-circuit.

Mirrors tests/test_gpu_registry.py's isolation pattern: every test redirects
gpu_registry.registry_dir() to a per-test tmp_path so nothing ever touches the
real machine-wide registry.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from fastapi.testclient import TestClient

from localm import gpu_registry, peer_routing
from localm.inference import http_server as hs
from localm.inference.http_server import create_app


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Redirect the gpu-registry location to a throwaway directory, clear
    hs._gpu_coord, and clear peer_routing's in-memory route table - all reset
    before AND after every test so nothing leaks between tests in this file
    or into any other test file sharing the same process."""
    d = tmp_path / "gpu"
    monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
    hs._gpu_coord = None
    peer_routing._ROUTES.clear()
    yield d
    hs._gpu_coord = None
    peer_routing._ROUTES.clear()


def _write_peer(d, iid, port, model=None, pid=None, host="127.0.0.1"):
    if pid is None:
        pid = os.getpid() + 1
    return gpu_registry.write_entry(
        d, instance_id=iid, pid=pid, port=port, host=host,
        scheme="http", model=model, vram_estimate_bytes=None, gpu_index=0,
        coordination_token=f"tok-{iid}")


def _make_live(monkeypatch):
    monkeypatch.setattr(gpu_registry, "pid_alive", lambda pid: True)
    monkeypatch.setattr(gpu_registry, "_try_whoami",
                        lambda scheme, port, iid, timeout: True)


# ------------------------------------------------------------------ #
#  registry_name_and_aliases                                         #
# ------------------------------------------------------------------ #

class TestRegistryNameAndAliases:
    def test_unregistered_name_has_no_aliases(self):
        name, aliases = peer_routing.registry_name_and_aliases({}, "startup-model")
        assert name == "startup-model"
        assert aliases == frozenset()

    def test_siblings_sharing_a_path_are_aliases(self):
        registry = {
            "canonical": {"path": "Z:/models/m.gguf"},
            "fast": {"path": "Z:/models/m.gguf"},
            "other-model": {"path": "Z:/models/other.gguf"},
        }
        _, aliases = peer_routing.registry_name_and_aliases(registry, "canonical")
        assert aliases == frozenset({"fast"})

    def test_resolving_via_an_alias_returns_the_same_alias_set(self):
        registry = {
            "canonical": {"path": "Z:/models/m.gguf"},
            "fast": {"path": "Z:/models/m.gguf"},
        }
        _, aliases = peer_routing.registry_name_and_aliases(registry, "fast")
        assert aliases == frozenset({"canonical"})

    def test_non_dict_entry_is_treated_as_unregistered(self):
        name, aliases = peer_routing.registry_name_and_aliases({"bad": "not-a-dict"}, "bad")
        assert name == "bad" and aliases == frozenset()


# ------------------------------------------------------------------ #
#  find_offer                                                        #
# ------------------------------------------------------------------ #

class TestFindOffer:
    def test_exact_name_match(self, tmp_path, monkeypatch):
        d = tmp_path / "reg"
        monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
        _write_peer(d, "peer1", 9101, model="my-model")
        _make_live(monkeypatch)
        peer = peer_routing.find_offer("my-model", frozenset())
        assert peer is not None and peer["instance_id"] == "peer1"

    def test_alias_match(self, tmp_path, monkeypatch):
        d = tmp_path / "reg"
        monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
        _write_peer(d, "peer2", 9102, model="Peer-Chosen-Name")
        _make_live(monkeypatch)
        peer = peer_routing.find_offer("canonical", frozenset({"Peer-Chosen-Name"}))
        assert peer is not None and peer["instance_id"] == "peer2"

    def test_casefolded_match(self, tmp_path, monkeypatch):
        d = tmp_path / "reg"
        monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
        _write_peer(d, "peer3", 9103, model="MyModel")
        _make_live(monkeypatch)
        peer = peer_routing.find_offer("mymodel", frozenset())
        assert peer is not None and peer["instance_id"] == "peer3"

    def test_no_match_returns_none(self, tmp_path, monkeypatch):
        d = tmp_path / "reg"
        monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
        _write_peer(d, "peer4", 9104, model="unrelated-model")
        _make_live(monkeypatch)
        assert peer_routing.find_offer("my-model", frozenset()) is None

    def test_peer_with_no_model_is_never_offered(self, tmp_path, monkeypatch):
        d = tmp_path / "reg"
        monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
        _write_peer(d, "peer5", 9105, model=None)
        _make_live(monkeypatch)
        assert peer_routing.find_offer("my-model", frozenset()) is None

    def test_empty_name_returns_none_without_a_lookup(self, tmp_path, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("list_gpu_peers must not be called for an empty name")
        monkeypatch.setattr(gpu_registry, "list_gpu_peers", boom)
        assert peer_routing.find_offer("", frozenset()) is None

    def test_registry_lookup_failure_is_swallowed(self, monkeypatch):
        def boom(**k):
            raise RuntimeError("registry unreadable")
        monkeypatch.setattr(gpu_registry, "list_gpu_peers", boom)
        assert peer_routing.find_offer("my-model", frozenset()) is None


# ------------------------------------------------------------------ #
#  Route state: get/set/clear/list, and PeerRoute.safe_dict           #
# ------------------------------------------------------------------ #

class TestRouteState:
    def _route(self, model="m", api_key="peer-real-key"):
        return peer_routing.PeerRoute(
            model=model, instance_id="iid", host="127.0.0.1", port=9999,
            scheme="http", api_key=api_key)

    def test_safe_dict_never_includes_the_api_key(self):
        d = self._route(api_key="super-secret").safe_dict()
        assert "api_key" not in d
        assert "super-secret" not in str(d)

    def test_set_then_get_round_trips(self):
        route = self._route(model="m1")
        peer_routing.set_route(route)
        assert peer_routing.get_route("m1") is route

    def test_get_missing_returns_none(self):
        assert peer_routing.get_route("nope") is None

    def test_get_with_falsy_name_returns_none(self):
        assert peer_routing.get_route(None) is None
        assert peer_routing.get_route("") is None

    def test_clear_removes_and_returns_the_route(self):
        route = self._route(model="m2")
        peer_routing.set_route(route)
        removed = peer_routing.clear_route("m2")
        assert removed is route
        assert peer_routing.get_route("m2") is None

    def test_clear_missing_returns_none(self):
        assert peer_routing.clear_route("never-set") is None

    def test_list_routes_never_includes_api_key(self):
        peer_routing.set_route(self._route(model="m3", api_key="leak-me-not"))
        listing = peer_routing.list_routes()
        assert listing["m3"]["model"] == "m3"
        assert "api_key" not in listing["m3"]
        assert "leak-me-not" not in str(listing)


# ------------------------------------------------------------------ #
#  forward(): the raw reverse proxy                                  #
# ------------------------------------------------------------------ #

class _FakeRequest:
    """Enough of a starlette Request for forward() to consume."""
    def __init__(self, body: bytes, content_type="application/json"):
        self._body = body
        self.headers = {"content-type": content_type} if content_type else {}

    async def body(self):
        return self._body


class _FakePeerResponse:
    def __init__(self, status_code=200, chunks=(b'{"ok": true}',),
                content_type="application/json"):
        self.status_code = status_code
        self._chunks = list(chunks)
        self.headers = {"content-type": content_type}

    def iter_content(self, chunk_size=None):
        return iter(self._chunks)


class TestForward:
    """No pytest-asyncio in this repo (see tests/test_gpu_registry.py's own
    ``asyncio.run(scenario())`` shape) - each test is a plain sync function
    wrapping an inner async scenario, matching the established convention."""

    def test_forwards_body_and_peer_api_key_never_coordination_token(self, monkeypatch):
        captured = {}

        def fake_post(url, data=None, headers=None, stream=None, timeout=None, verify=None):
            captured["url"] = url
            captured["data"] = data
            captured["headers"] = headers
            return _FakePeerResponse()

        monkeypatch.setattr("requests.post", fake_post)
        route = peer_routing.PeerRoute(
            model="m", instance_id="peer-x", host="127.0.0.1", port=8123,
            scheme="http", api_key="the-peers-real-api-key")
        req = _FakeRequest(b'{"model": "m", "messages": []}')

        async def scenario():
            return await peer_routing.forward(route, req, "/v1/chat/completions")

        resp = asyncio.run(scenario())

        assert "127.0.0.1:8123/v1/chat/completions" in captured["url"]
        assert captured["data"] == b'{"model": "m", "messages": []}'
        assert captured["headers"]["Authorization"] == "Bearer the-peers-real-api-key"
        # The auth trap: forwarding must NEVER present a coordination_token,
        # by any name, as the credential for a forwarded inference request.
        assert "X-LocalM-Coordination-Token" not in captured["headers"]
        assert "coordination_token" not in str(captured["headers"]).lower()
        assert resp.status_code == 200

    def test_network_failure_clears_the_route_and_raises_502(self, monkeypatch):
        import requests

        def boom(*a, **k):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr("requests.post", boom)
        route = peer_routing.PeerRoute(
            model="gone-model", instance_id="peer-y", host="127.0.0.1",
            port=8124, scheme="http", api_key="k")
        peer_routing.set_route(route)
        req = _FakeRequest(b"{}")

        from fastapi import HTTPException

        async def scenario():
            return await peer_routing.forward(route, req, "/v1/chat/completions")

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(scenario())
        assert exc_info.value.status_code == 502

        # Fires-control on the "clears the route" claim: the route must
        # actually be gone, not merely that an exception happened to fire.
        assert peer_routing.get_route("gone-model") is None

    def test_peer_error_response_is_passed_through_unchanged(self, monkeypatch):
        """A real answer from a live peer (even a 4xx) is not "unavailable" -
        it must be relayed, not swallowed into a 502."""
        def fake_post(*a, **k):
            return _FakePeerResponse(status_code=400, chunks=(b'{"error": "bad request"}',))

        monkeypatch.setattr("requests.post", fake_post)
        route = peer_routing.PeerRoute(
            model="m", instance_id="peer-z", host="127.0.0.1", port=8125,
            scheme="http", api_key="k")
        req = _FakeRequest(b"{}")

        async def scenario():
            return await peer_routing.forward(route, req, "/v1/chat/completions")

        resp = asyncio.run(scenario())
        assert resp.status_code == 400
        # The route survives a real (non-network-failure) response.
        peer_routing.set_route(route)
        assert peer_routing.get_route("m") is not None


# ------------------------------------------------------------------ #
#  Endpoints: GET/POST/DELETE /v1/models/{id}/peer-route(s)          #
# ------------------------------------------------------------------ #

class TestPeerOfferEndpoint:
    def _client(self):
        return TestClient(create_app(None))

    def test_no_peer_reports_unavailable(self):
        client = self._client()
        r = client.get("/v1/models/no-such-model/peer-offer")
        assert r.status_code == 200
        assert r.json() == {"available": False, "peer": None}

    def test_live_peer_with_matching_model_is_offered(self, tmp_path, monkeypatch):
        d = tmp_path / "reg"
        monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
        _write_peer(d, "peer-off", 9201, model="shared-model")
        _make_live(monkeypatch)
        client = self._client()
        r = client.get("/v1/models/shared-model/peer-offer")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        assert body["peer"]["instance_id"] == "peer-off"
        assert body["peer"]["port"] == 9201
        # Never leaks the coordination_token to a client.
        assert "coordination_token" not in body["peer"]


class TestAcceptPeerRouteEndpoint:
    def _client(self):
        return TestClient(create_app(None))

    def _write_hdr(self, client):
        # MODELS_WRITE (an unsafe method) in open mode requires proof of a
        # local process - the loopback shell token the GUI shell itself
        # presents - same convention as test_bearer_auth.py's
        # test_model_load_with_shell_token.
        return {"Authorization": f"Bearer {client.app.state.shell_token}"}

    def test_accept_stores_the_route(self, tmp_path, monkeypatch):
        d = tmp_path / "reg"
        monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
        _write_peer(d, "peer-acc", 9301, model="shared-model")
        _make_live(monkeypatch)
        client = self._client()
        r = client.post("/v1/models/shared-model/peer-route",
                        headers=self._write_hdr(client),
                        json={"instance_id": "peer-acc", "api_key": "peer-key-123"})
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "routed"
        route = peer_routing.get_route("shared-model")
        assert route is not None
        assert route.api_key == "peer-key-123"
        assert route.instance_id == "peer-acc"

    def test_stale_offer_is_rejected(self, tmp_path, monkeypatch):
        """The client's offer names a peer that is no longer live/matching -
        the endpoint must re-verify, not trust the caller."""
        d = tmp_path / "reg"
        monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
        # No peer registered at all - "peer-acc" never existed on this box.
        client = self._client()
        r = client.post("/v1/models/shared-model/peer-route",
                        headers=self._write_hdr(client),
                        json={"instance_id": "peer-acc", "api_key": "k"})
        assert r.status_code == 409
        assert peer_routing.get_route("shared-model") is None

    def test_missing_fields_are_rejected(self):
        client = self._client()
        r = client.post("/v1/models/shared-model/peer-route",
                        headers=self._write_hdr(client),
                        json={"instance_id": "", "api_key": ""})
        assert r.status_code in (400, 422)
        assert peer_routing.get_route("shared-model") is None

    def test_coordination_token_alone_is_not_a_valid_credential(self, tmp_path, monkeypatch):
        """The auth trap, at the endpoint boundary: presenting a real
        coordination_token as if it were the JSON body's api_key does not
        get special-cased or rejected differently - it is stored as an
        opaque string like any other caller-supplied value, and it is never
        read from/compared against gpu_registry's coordination_token at all.
        This endpoint has no coordination_token concept."""
        d = tmp_path / "reg"
        monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
        _write_peer(d, "peer-tok", 9302, model="shared-model")
        _make_live(monkeypatch)
        client = self._client()
        # Presenting the (unrelated) coordination_token field name in the body
        # is simply an unknown field to this endpoint's schema - it does not
        # authenticate anything and api_key is still required.
        r = client.post("/v1/models/shared-model/peer-route",
                        headers=self._write_hdr(client),
                        json={"instance_id": "peer-tok", "coordination_token": "tok-peer-tok"})
        assert r.status_code == 422


class TestClearPeerRouteEndpoint:
    def _client(self):
        return TestClient(create_app(None))

    def _write_hdr(self, client):
        return {"Authorization": f"Bearer {client.app.state.shell_token}"}

    def test_clear_is_idempotent_and_reports_prior_state(self):
        client = self._client()
        hdr = self._write_hdr(client)
        r = client.delete("/v1/models/never-routed/peer-route", headers=hdr)
        assert r.status_code == 200
        assert r.json() == {"status": "unrouted", "model": "never-routed", "was_routed": False}

        peer_routing.set_route(peer_routing.PeerRoute(
            model="was-routed", instance_id="i", host="127.0.0.1", port=1,
            scheme="http", api_key="k"))
        r = client.delete("/v1/models/was-routed/peer-route", headers=hdr)
        assert r.status_code == 200
        assert r.json()["was_routed"] is True
        assert peer_routing.get_route("was-routed") is None


# ------------------------------------------------------------------ #
#  chat_completions / completions: the forwarding short-circuit      #
# ------------------------------------------------------------------ #

class TestChatCompletionsForwarding:
    def _client(self):
        return TestClient(create_app(None))

    def test_routed_model_is_forwarded_and_never_touches_local_engines(self, monkeypatch):
        def fake_post(url, data=None, headers=None, stream=None, timeout=None, verify=None):
            return _FakePeerResponse(
                chunks=(b'{"id": "x", "choices": [{"message": {"content": "hi"}}]}',))

        monkeypatch.setattr("requests.post", fake_post)
        hs._engines.clear()
        peer_routing.set_route(peer_routing.PeerRoute(
            model="routed-model", instance_id="peer-fwd", host="127.0.0.1",
            port=8199, scheme="http", api_key="peer-key"))

        client = self._client()
        r = client.post("/v1/chat/completions", json={
            "model": "routed-model",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert r.status_code == 200, r.text
        assert r.json()["choices"][0]["message"]["content"] == "hi"
        # Never created/loaded a local engine for the routed name.
        assert "routed-model" not in hs._engines

    def test_completions_route_also_forwards(self, monkeypatch):
        def fake_post(url, data=None, headers=None, stream=None, timeout=None, verify=None):
            assert url.endswith("/v1/completions")
            return _FakePeerResponse(chunks=(b'{"choices": [{"text": "ok"}]}',))

        monkeypatch.setattr("requests.post", fake_post)
        hs._engines.clear()
        peer_routing.set_route(peer_routing.PeerRoute(
            model="routed-model-2", instance_id="peer-fwd2", host="127.0.0.1",
            port=8198, scheme="http", api_key="peer-key-2"))

        client = self._client()
        r = client.post("/v1/completions", json={
            "model": "routed-model-2", "prompt": "hello",
        })
        assert r.status_code == 200, r.text
        assert "routed-model-2" not in hs._engines

    def test_peer_becoming_unavailable_falls_back_to_local_load_on_the_next_request(self, monkeypatch):
        """Fires-control on the documented failure behavior: the FAILED
        request gets a clean 502 naming the peer, and the route is cleared
        so a SUBSEQUENT request for the same name proceeds as an ordinary
        (here: unregistered-model 404) local resolution rather than hanging
        or silently retrying forever."""
        import requests

        def boom(*a, **k):
            raise requests.exceptions.ConnectionError("peer is gone")

        monkeypatch.setattr("requests.post", boom)
        hs._engines.clear()
        hs._active_model_name = None
        hs._default_model_name = None
        peer_routing.set_route(peer_routing.PeerRoute(
            model="flaky-model", instance_id="peer-flaky", host="127.0.0.1",
            port=8197, scheme="http", api_key="k"))

        client = self._client()
        r1 = client.post("/v1/chat/completions", json={
            "model": "flaky-model",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert r1.status_code == 502
        assert peer_routing.get_route("flaky-model") is None

        # Next request for the same name: no route left, falls through to
        # ordinary local resolution (404: never registered on this instance).
        r2 = client.post("/v1/chat/completions", json={
            "model": "flaky-model",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert r2.status_code != 502


# ------------------------------------------------------------------ #
#  The peer host a forwarded credential may reach                    #
# ------------------------------------------------------------------ #

class TestRoutablePeerHost:
    """gpu_registry.list_gpu_peers verifies a peer's identity by calling
    /whoami with no bind_host, i.e. against loopback. _peer_url dials
    self_connect_host(entry["host"]) instead, which returns any other literal
    as itself. is_routable_peer_host is what keeps those two the same
    endpoint."""

    @pytest.mark.parametrize("host", [
        "127.0.0.1", "127.0.0.5", "::1", "localhost", "0.0.0.0", "::", "", None,
    ])
    def test_a_loopback_or_wildcard_bind_is_routable(self, host):
        assert peer_routing.is_routable_peer_host(host) is True

    @pytest.mark.parametrize("host", [
        "192.168.1.5", "10.0.0.7", "8.8.8.8", "attacker.example",
        "peer.local", "2001:db8::1", "169.254.169.254",
    ])
    def test_anything_else_is_not_routable(self, host):
        assert peer_routing.is_routable_peer_host(host) is False

    def test_dial_host_matches_what_peer_url_actually_builds(self):
        """The predicate is only meaningful if it judges the same string
        _peer_url dials. A divergence there is the whole defect class."""
        from localm.bindhost import url_host
        for host in ("0.0.0.0", "::", "localhost", "127.0.0.1", "192.168.1.5"):
            route = peer_routing.PeerRoute(
                model="m", instance_id="i", host=host, port=9000,
                scheme="http", api_key="k")
            url = peer_routing._peer_url(route, "/v1/chat/completions")
            assert url_host(peer_routing.dial_host(host)) + ":9000" in url


class TestNonLoopbackPeerIsNeverOffered:
    def test_a_peer_advertising_a_non_loopback_host_is_never_offered(
            self, _isolated_state, monkeypatch):
        """A matching, live, whoami-passing peer is still refused when the
        address this instance would DIAL was never the address it verified."""
        _write_peer(_isolated_state, "peer-lan", 9401,
                    model="shared-model", host="192.168.1.5")
        _make_live(monkeypatch)
        assert peer_routing.find_offer("shared-model", frozenset()) is None

    def test_the_same_peer_on_loopback_is_offered(self, _isolated_state, monkeypatch):
        """The control for the test above: everything else identical, so the
        host is provably the discriminator rather than some unrelated reason
        the lookup returned nothing."""
        _write_peer(_isolated_state, "peer-lo", 9402,
                    model="shared-model", host="127.0.0.1")
        _make_live(monkeypatch)
        peer = peer_routing.find_offer("shared-model", frozenset())
        assert peer is not None and peer["instance_id"] == "peer-lo"

    def test_the_endpoint_reports_unavailable_for_a_non_loopback_peer(
            self, _isolated_state, monkeypatch):
        _write_peer(_isolated_state, "peer-lan2", 9403,
                    model="shared-model", host="10.0.0.7")
        _make_live(monkeypatch)
        client = TestClient(create_app(None))
        r = client.get("/v1/models/shared-model/peer-offer")
        assert r.status_code == 200
        assert r.json() == {"available": False, "peer": None}

    def test_accepting_a_non_loopback_peer_is_refused_and_stores_no_route(
            self, _isolated_state, monkeypatch):
        _write_peer(_isolated_state, "peer-lan3", 9404,
                    model="shared-model", host="203.0.113.9")
        _make_live(monkeypatch)
        client = TestClient(create_app(None))
        r = client.post(
            "/v1/models/shared-model/peer-route",
            headers={"Authorization": f"Bearer {client.app.state.shell_token}"},
            json={"instance_id": "peer-lan3", "api_key": "the-users-real-key"})
        assert peer_routing.get_route("shared-model") is None
        assert r.status_code == 409


class TestForwardRefusesANonLoopbackRoute:
    def test_forward_refuses_a_non_loopback_route_without_sending_the_key(
            self, monkeypatch):
        """The second of the two checks: it must hold however the PeerRoute
        was constructed, so it is asserted here against a route built directly
        rather than through find_offer."""
        calls = []

        def fake_post(url, data=None, headers=None, stream=None, timeout=None, verify=None):
            calls.append((url, headers))
            return _FakePeerResponse()

        monkeypatch.setattr("requests.post", fake_post)
        route = peer_routing.PeerRoute(
            model="lan-model", instance_id="peer-lan4", host="192.168.1.5",
            port=9405, scheme="http", api_key="the-users-real-key")
        peer_routing.set_route(route)
        req = _FakeRequest(b"{}")

        from fastapi import HTTPException

        async def scenario():
            return await peer_routing.forward(route, req, "/v1/chat/completions")

        raised = None
        try:
            asyncio.run(scenario())
        except HTTPException as e:
            raised = e

        # The exception is captured rather than asserted with pytest.raises so
        # that these run FIRST: a request that was never made is the property,
        # and the status code is only a proxy for it. Under pytest.raises the
        # runner reports "DID NOT RAISE" and never reaches these lines, which
        # reads as an assertion to relax rather than as a credential leaving.
        assert calls == [], "the bearer credential left this process"
        assert peer_routing.get_route("lan-model") is None
        assert raised is not None and raised.status_code == 502

    def test_a_loopback_route_still_forwards(self, monkeypatch):
        """The control: same shape, loopback host, request goes out."""
        calls = []

        def fake_post(url, data=None, headers=None, stream=None, timeout=None, verify=None):
            calls.append((url, headers))
            return _FakePeerResponse()

        monkeypatch.setattr("requests.post", fake_post)
        route = peer_routing.PeerRoute(
            model="lo-model", instance_id="peer-lo2", host="127.0.0.1",
            port=9406, scheme="http", api_key="the-users-real-key")
        req = _FakeRequest(b"{}")

        async def scenario():
            return await peer_routing.forward(route, req, "/v1/chat/completions")

        assert asyncio.run(scenario()).status_code == 200
        assert len(calls) == 1
        assert calls[0][1]["Authorization"] == "Bearer the-users-real-key"


# ------------------------------------------------------------------ #
#  The peer lookup runs off the event loop                           #
# ------------------------------------------------------------------ #

class TestPeerLookupIsOffloaded:
    """A sync handler with no await points is implicitly atomic on the event
    loop; these routes reach requests.get through find_offer, so the lookup
    must not run there."""

    def _record_thread(self, monkeypatch, seen):
        real = peer_routing.find_offer

        def probe(*a, **k):
            try:
                asyncio.get_running_loop()
                seen.append("event-loop")
            except RuntimeError:
                seen.append("worker-thread")
            return real(*a, **k)

        monkeypatch.setattr(peer_routing, "find_offer", probe)

    def test_peer_offer_looks_the_peer_up_off_the_event_loop(
            self, _isolated_state, monkeypatch):
        seen = []
        self._record_thread(monkeypatch, seen)
        _write_peer(_isolated_state, "peer-off2", 9501, model="shared-model")
        _make_live(monkeypatch)
        client = TestClient(create_app(None))
        assert client.get("/v1/models/shared-model/peer-offer").status_code == 200
        assert seen == ["worker-thread"], (
            "find_offer ran on the event loop, so its requests.get stalls "
            "every other request on this server")

    def test_accept_verifies_the_peer_off_the_event_loop(
            self, _isolated_state, monkeypatch):
        seen = []
        self._record_thread(monkeypatch, seen)
        _write_peer(_isolated_state, "peer-acc2", 9502, model="shared-model")
        _make_live(monkeypatch)
        client = TestClient(create_app(None))
        r = client.post(
            "/v1/models/shared-model/peer-route",
            headers={"Authorization": f"Bearer {client.app.state.shell_token}"},
            json={"instance_id": "peer-acc2", "api_key": "k"})
        assert r.status_code == 200, r.text
        assert seen == ["worker-thread"]


# ------------------------------------------------------------------ #
#  _ROUTES is shared with worker threads now                         #
# ------------------------------------------------------------------ #

class TestRoutesLock:
    def _route(self, model):
        return peer_routing.PeerRoute(
            model=model, instance_id="i", host="127.0.0.1", port=1,
            scheme="http", api_key="k")

    def test_listing_survives_concurrent_mutation(self):
        """list_routes() iterates _ROUTES while the offloaded accept handler
        writes to it from a worker thread. Unlocked, that raises
        "dictionary changed size during iteration"."""
        import threading

        for i in range(200):
            peer_routing.set_route(self._route(f"seed-{i}"))
        stop = threading.Event()
        errors = []

        def churn():
            i = 0
            while not stop.is_set():
                peer_routing.set_route(self._route(f"churn-{i}"))
                peer_routing.clear_route(f"churn-{i}")
                i += 1

        t = threading.Thread(target=churn, daemon=True)
        t.start()
        try:
            for _ in range(3000):
                try:
                    peer_routing.list_routes()
                    peer_routing.get_route("seed-0")
                except RuntimeError as e:
                    errors.append(repr(e))
                    break
        finally:
            stop.set()
            t.join(timeout=5)
        assert errors == [], f"concurrent access to _ROUTES raised: {errors}"

    def test_find_offer_does_not_hold_the_routes_lock(
            self, _isolated_state, monkeypatch):
        """Half of the invariant _ROUTES_LOCK's comment states. This pins the
        peer-lookup path only, and it is insensitive to whether the four
        accessors lock at all: it fails when find_offer ACQUIRES the lock, not
        when someone removes a `with` from set_route."""
        import threading

        acquired = []

        def peers_probe(*a, **k):
            def probe():
                got = peer_routing._ROUTES_LOCK.acquire(blocking=False)
                acquired.append(got)
                if got:
                    peer_routing._ROUTES_LOCK.release()

            # A separate thread, because _ROUTES_LOCK is an RLock and would
            # re-enter for the calling one.
            t = threading.Thread(target=probe)
            t.start()
            t.join(timeout=5)
            return []

        monkeypatch.setattr(gpu_registry, "list_gpu_peers", peers_probe)
        peer_routing.find_offer("some-model", frozenset())
        assert acquired == [True], (
            "_ROUTES_LOCK was held across the peer lookup's network call")

    def test_forward_does_not_hold_the_routes_lock_across_the_request(self, monkeypatch):
        """The other half. forward() is the path that genuinely takes the lock
        (clear_route) near a network call, so this is the one find_offer's
        sibling test cannot cover: the lock must be free while the forwarded
        request is in flight."""
        import threading

        acquired = []

        def probing_post(url, data=None, headers=None, stream=None, timeout=None, verify=None):
            def probe():
                got = peer_routing._ROUTES_LOCK.acquire(blocking=False)
                acquired.append(got)
                if got:
                    peer_routing._ROUTES_LOCK.release()

            t = threading.Thread(target=probe)
            t.start()
            t.join(timeout=5)
            return _FakePeerResponse()

        monkeypatch.setattr("requests.post", probing_post)
        route = peer_routing.PeerRoute(
            model="lock-fwd", instance_id="peer-lockfwd", host="127.0.0.1",
            port=9607, scheme="http", api_key="k")
        req = _FakeRequest(b"{}")

        assert asyncio.run(
            peer_routing.forward(route, req, "/v1/chat/completions")).status_code == 200
        assert acquired == [True], (
            "_ROUTES_LOCK was held while the forwarded request was in flight")


# ------------------------------------------------------------------ #
#  scheme is the other half of the same untrusted record             #
# ------------------------------------------------------------------ #

class TestPeerSchemeIsPinnedToo:
    """_peer_url interpolates scheme into "{scheme}://{host}:{port}{path}", so
    a scheme containing "://" moves the URL authority off host entirely and
    the loopback host check never sees it."""

    @pytest.mark.parametrize("scheme", [
        "http://attacker.example/collect?x",
        "https://attacker.example/#",
        "HTTP", "ftp", "file", "", None, "http ",
    ])
    def test_only_http_and_https_are_routable(self, scheme):
        assert peer_routing.is_routable_peer_endpoint("127.0.0.1", scheme) is False

    @pytest.mark.parametrize("scheme", ["http", "https"])
    def test_http_and_https_on_loopback_are_routable(self, scheme):
        assert peer_routing.is_routable_peer_endpoint("127.0.0.1", scheme) is True

    def test_a_smuggled_scheme_really_moves_the_url_authority(self):
        """The reason this is pinned at all, asserted against the real URL
        builder rather than argued: with a loopback host and a crafted scheme,
        the authority is the attacker's."""
        from urllib3.util import parse_url
        route = peer_routing.PeerRoute(
            model="m", instance_id="i", host="127.0.0.1", port=8642,
            scheme="http://attacker.example/collect?x", api_key="k")
        assert parse_url(peer_routing._peer_url(route, "/v1/chat/completions")).host \
            == "attacker.example"

    def test_a_peer_whose_scheme_smuggles_an_authority_is_never_offered(
            self, _isolated_state, monkeypatch):
        gpu_registry.write_entry(
            _isolated_state, instance_id="peer-smuggle", pid=os.getpid() + 1,
            port=9601, host="127.0.0.1",
            scheme="http://attacker.example/collect?x", model="shared-model",
            vram_estimate_bytes=None, gpu_index=0, coordination_token="t")
        _make_live(monkeypatch)
        assert peer_routing.find_offer("shared-model", frozenset()) is None

    def test_forward_refuses_a_smuggled_scheme_without_sending_the_key(self, monkeypatch):
        calls = []

        def fake_post(url, data=None, headers=None, stream=None, timeout=None, verify=None):
            calls.append((url, headers))
            return _FakePeerResponse()

        monkeypatch.setattr("requests.post", fake_post)
        route = peer_routing.PeerRoute(
            model="smuggled", instance_id="peer-smuggle2", host="127.0.0.1",
            port=9602, scheme="http://attacker.example/collect?x",
            api_key="the-users-real-key")
        peer_routing.set_route(route)
        req = _FakeRequest(b"{}")

        from fastapi import HTTPException

        raised = None
        try:
            asyncio.run(peer_routing.forward(route, req, "/v1/chat/completions"))
        except HTTPException as e:
            raised = e

        assert calls == [], "the bearer credential left this process"
        assert peer_routing.get_route("smuggled") is None
        assert raised is not None and raised.status_code == 502


# ------------------------------------------------------------------ #
#  find_offer's never-raises contract                                #
# ------------------------------------------------------------------ #

class TestMalformedEntryDoesNotRaise:
    """find_offer documents that any failure reading the registry yields None
    and is never raised. The host predicate runs outside the try that wraps
    list_gpu_peers, so a non-string host must be rejected rather than raise."""

    @pytest.mark.parametrize("host", [127, True, ["127.0.0.1"], {"a": 1}, 1.5])
    def test_a_non_string_host_is_rejected_not_raised(self, host):
        """None is deliberately absent: an unset host is the wildcard-bind
        case, which self_connect_host maps to 127.0.0.1, and
        test_a_loopback_or_wildcard_bind_is_routable pins it as True."""
        assert peer_routing.is_routable_peer_host(host) is False

    def test_find_offer_yields_none_on_a_malformed_host(self, monkeypatch):
        monkeypatch.setattr(
            gpu_registry, "list_gpu_peers",
            lambda **k: [{"instance_id": "bad", "model": "shared-model",
                          "host": 127, "port": 9603, "scheme": "http"}])
        assert peer_routing.find_offer("shared-model", frozenset()) is None

    def test_the_offer_endpoint_reports_unavailable_rather_than_500(
            self, _isolated_state, monkeypatch):
        monkeypatch.setattr(
            gpu_registry, "list_gpu_peers",
            lambda **k: [{"instance_id": "bad", "model": "shared-model",
                          "host": 127, "port": 9604, "scheme": "http"}])
        client = TestClient(create_app(None))
        r = client.get("/v1/models/shared-model/peer-offer")
        assert r.status_code == 200, r.text
        assert r.json() == {"available": False, "peer": None}


# ------------------------------------------------------------------ #
#  A request the client was told failed must change nothing          #
# ------------------------------------------------------------------ #

class TestTimedOutAcceptInstallsNothing:
    def test_a_timed_out_accept_never_installs_a_route(
            self, _isolated_state, monkeypatch):
        """run_in_threadpool_bounded abandons the AWAITING CALLER at its
        deadline and lets the worker thread run to completion, so any mutation
        inside the offloaded closure lands after the client was told 504. The
        user sees the failure toast while their next chat request would be
        forwarded with the key they typed."""
        import time

        from localm.inference.routes import peer_routing as routes_peer_routing

        _write_peer(_isolated_state, "peer-slow", 9605, model="shared-model")
        _make_live(monkeypatch)

        real = peer_routing.find_offer

        def slow(*a, **k):
            time.sleep(1.2)
            return real(*a, **k)

        monkeypatch.setattr(peer_routing, "find_offer", slow)
        monkeypatch.setattr(routes_peer_routing, "_PEER_LOOKUP_TIMEOUT_S", 0.2)

        client = TestClient(create_app(None))
        r = client.post(
            "/v1/models/shared-model/peer-route",
            headers={"Authorization": f"Bearer {client.app.state.shell_token}"},
            json={"instance_id": "peer-slow", "api_key": "the-users-real-key"})

        assert r.status_code == 504, r.text
        assert peer_routing.get_route("shared-model") is None

        # The abandoned worker is still running here; the assertion that
        # matters is the one AFTER it has had time to finish.
        time.sleep(2.0)
        late = peer_routing.get_route("shared-model")
        assert late is None, (
            "a request the client was told had FAILED installed a route "
            f"carrying the user's key: {late}")
