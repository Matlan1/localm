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


def _write_peer(d, iid, port, model=None, pid=None):
    if pid is None:
        pid = os.getpid() + 1
    return gpu_registry.write_entry(
        d, instance_id=iid, pid=pid, port=port, host="127.0.0.1",
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
