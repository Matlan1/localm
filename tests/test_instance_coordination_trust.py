# SPDX-License-Identifier: AGPL-3.0-or-later
"""Who may make an instance release its VRAM, and who may read its full status.

A cooperation request names its requester's port, and the receiver calls that
port back to confirm the request. The listener on that port must belong to the
OS user running the receiver; otherwise any local process, including another
user's, could answer the callback for itself. A requester bound only on ``::1``
is called back on ``::1``.

``GET /v1/instances/status`` without a credential carries no process id and no
model file identity (path, size, sha256), and, on a key-protected instance, no
model names. The full status needs this instance's attach token, an API key
granting ``models:read``, or a ``POST`` naming a requester that confirms it, the
same callback a cooperation request uses.

The peers are real loopback HTTP servers and the receiver is the real app on a
real socket where the exchange matters."""

from __future__ import annotations

import os
import socket
import threading
import time
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

import pytest
import requests
from fastapi.testclient import TestClient

from localm import gpu_registry, listeners
from localm.inference import http_server as hs
from localm.inference.http_server import create_app
from tests._peer_servers import (FOREIGN_PID, _Handler, default_status,
                                 enable_detection, localm_whoami, peer_server)

_PRIVATE_PATH = os.path.join(os.sep, "home", "alice-s1", "models", "secret-model.gguf")
_DIGEST = "ab" * 32
_IDENTITIES = [{"name": "secret-model", "path": _PRIVATE_PATH, "size": 1234,
                "sha256": _DIGEST}]


@pytest.fixture(autouse=True)
def _clean_coordination_state():
    hs._gpu_coord = None
    gpu_registry.set_local_status_provider(None)
    gpu_registry._pending.clear()
    listeners._owned_cache.clear()
    yield
    hs._gpu_coord = None
    hs._active_model_name = None
    gpu_registry.set_local_status_provider(None)
    gpu_registry._pending.clear()
    listeners._owned_cache.clear()


def _as_another_os_user(monkeypatch):
    """Every listening socket reads as owned by a different OS user."""
    monkeypatch.setattr(listeners, "_port_owned_by_current_user", lambda port, dial: False)


def _deferring_vouch(body):
    ok = gpu_registry.vouch_for(body.get("request_id"), body.get("asker_instance_id"),
                                body.get("asker_port"), body.get("purpose"))
    return (200, {"vouched": True}) if ok else (403, {"detail": "no"})


def _coordinating_app(bind_host="127.0.0.1", port=8700, model=None):
    app = create_app(None)
    app.state.bind_host = bind_host
    app.state.insecure_bind = True
    app.state.instance_id = "self-1"
    hs._active_model_name = model
    hs._gpu_coord = {"instance_id": "self-1", "port": port, "host": bind_host,
                     "scheme": "http"}
    return app


@contextmanager
def _served(app):
    """*app* on a real loopback uvicorn socket; yields the port."""
    import uvicorn
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    app.state.instance_port = port
    app.state.instance_scheme = "http"
    if hs._gpu_coord is not None:
        hs._gpu_coord["port"] = port
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 20
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        assert server.started
        gpu_registry.set_local_status_provider(None)
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=20)


class _V6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


@contextmanager
def _peer_server_v6(whoami, *, post=None):
    """A fake instance listening on ``::1`` only."""
    spec = {"whoami": whoami, "status": default_status(whoami), "post": post or {},
            "requests": []}
    srv = _V6Server(("::1", 0), _Handler)
    srv.spec = spec
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()

    class _Running:
        port = srv.server_address[1]
        requests = spec["requests"]

    try:
        yield _Running
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def _ipv6_loopback_available() -> bool:
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    except OSError:
        return False
    try:
        s.bind(("::1", 0))
        return True
    except OSError:
        return False
    finally:
        s.close()


# ------------------------------------------------------------------ #
#  The requester callback: the listener must be this OS user's       #
# ------------------------------------------------------------------ #

class TestTheRequesterMustBeThisUsersListener:
    def _requester(self, port, instance_id="req-1"):
        return {"instance_id": instance_id, "port": port, "scheme": "http"}

    def test_a_listener_of_this_user_that_vouches_is_verified(self):
        with peer_server(localm_whoami("req-1"),
                         post={"/v1/instances/vouch":
                               lambda b: (200, {"vouched": True})}) as srv:
            assert gpu_registry.verify_requester(
                self._requester(srv.port), "rid", "me") is True

    def test_another_users_listener_that_vouches_for_itself_is_refused(self, monkeypatch):
        _as_another_os_user(monkeypatch)
        with peer_server(localm_whoami("req-1"),
                         post={"/v1/instances/vouch":
                               lambda b: (200, {"vouched": True})}) as srv:
            assert gpu_registry.verify_requester(
                self._requester(srv.port), "rid", "me") is False
            assert srv.requests == []

    def test_a_listener_that_changes_owner_during_the_callback_is_refused(
            self, monkeypatch):
        answers = iter([True, False])
        monkeypatch.setattr(listeners, "_port_owned_by_current_user",
                            lambda port, dial: next(answers, False))
        with peer_server(localm_whoami("req-1"),
                         post={"/v1/instances/vouch":
                               lambda b: (200, {"vouched": True})}) as srv:
            assert gpu_registry.verify_requester(
                self._requester(srv.port), "rid", "me") is False

    @pytest.mark.skipif(not _ipv6_loopback_available(), reason="no IPv6 loopback")
    def test_a_requester_bound_only_on_ipv6_loopback_is_called_back_there(self):
        calls = []

        def vouch(body):
            calls.append(body)
            return 200, {"vouched": True}

        with _peer_server_v6(localm_whoami("req-6"),
                             post={"/v1/instances/vouch": vouch}) as srv:
            assert gpu_registry.verify_requester(
                self._requester(srv.port, "req-6"), "rid-6", "me") is True
        assert calls == [{"request_id": "rid-6", "asker_instance_id": "me"}]


class TestCooperateUnloadEndToEnd:
    def _spy_unload(self, monkeypatch):
        calls = []

        async def unload_all_models():
            calls.append(1)
            return {"status": "unloaded"}

        monkeypatch.setattr(hs, "unload_all_models", unload_all_models)
        return calls

    def test_a_self_vouching_listener_of_another_user_cannot_unload(self, monkeypatch):
        calls = self._spy_unload(monkeypatch)
        _as_another_os_user(monkeypatch)
        with _served(_coordinating_app(model="secret-model")) as port, \
                peer_server(localm_whoami("attacker"),
                            post={"/v1/instances/vouch":
                                  lambda b: (200, {"vouched": True})}) as fake:
            r = requests.post(
                f"http://127.0.0.1:{port}/v1/instances/cooperate-unload",
                json={"requester": {"instance_id": "attacker", "port": fake.port,
                                    "scheme": "http"},
                      "request_id": "self-made"}, timeout=10)
        assert r.status_code == 403
        assert calls == []

    def test_a_sibling_of_this_user_still_gets_the_unload(self, monkeypatch):
        calls = self._spy_unload(monkeypatch)
        with _served(_coordinating_app(model="secret-model")) as port, \
                peer_server(localm_whoami("requester-1"),
                            post={"/v1/instances/vouch": _deferring_vouch}) as req:
            gpu_registry.set_local_status_provider(
                lambda: {"instance_id": "requester-1", "port": req.port,
                         "scheme": "http"})
            peer = {"instance_id": "self-1", "port": port, "host": "127.0.0.1",
                    "scheme": "http"}
            assert gpu_registry.request_cooperative_unload(peer, timeout=10) is True
        assert calls == [1]


# ------------------------------------------------------------------ #
#  GET /v1/instances/status without a credential                     #
# ------------------------------------------------------------------ #

class TestStatusWithoutACredential:
    @pytest.fixture(autouse=True)
    def _loaded(self, monkeypatch):
        monkeypatch.setattr(hs, "_loaded_model_identities", lambda: list(_IDENTITIES))

    def test_open_mode_names_the_models_but_no_file_identity_or_pid(self):
        r = TestClient(_coordinating_app(model="secret-model")).get("/v1/instances/status")
        assert r.status_code == 200
        body = r.json()
        assert "alice-s1" not in r.text and _DIGEST not in r.text
        assert "pid" not in body
        assert body["models"] == [{"name": "secret-model", "file": "secret-model.gguf",
                                   "size": 1234}]
        assert body["model"] == "secret-model"
        assert body["instance_id"] == "self-1"
        assert {"vram_estimate_bytes", "gpu_index", "port", "scheme"} <= set(body)

    def test_a_key_protected_instance_names_no_models_at_all(self, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "k" * 40)
        r = TestClient(_coordinating_app(model="secret-model")).get("/v1/instances/status")
        assert r.status_code == 200
        body = r.json()
        assert "secret-model" not in r.text and "alice-s1" not in r.text
        assert "pid" not in body
        assert body["models"] == [] and body["model"] is None

    def test_require_auth_without_a_key_names_no_models(self, monkeypatch):
        monkeypatch.setenv("LOCALM_REQUIRE_AUTH", "1")
        r = TestClient(_coordinating_app(model="secret-model")).get("/v1/instances/status")
        assert "secret-model" not in r.text and "pid" not in r.json()

    def test_a_session_cookie_is_no_credential(self, monkeypatch):
        key = "k" * 40
        monkeypatch.setenv("LOCALM_API_KEY", key)
        client = TestClient(_coordinating_app(model="secret-model"))
        client.cookies.set(hs.SESSION_COOKIE, key)
        r = client.get("/v1/instances/status")
        assert "secret-model" not in r.text and "pid" not in r.json()

    def test_a_wrong_key_is_no_credential(self, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "k" * 40)
        r = TestClient(_coordinating_app(model="secret-model")).get(
            "/v1/instances/status", headers={"Authorization": "Bearer " + "x" * 40})
        assert r.status_code == 200
        assert "secret-model" not in r.text and "pid" not in r.json()


class TestFullStatus:
    @pytest.fixture(autouse=True)
    def _loaded(self, monkeypatch):
        monkeypatch.setattr(hs, "_loaded_model_identities", lambda: list(_IDENTITIES))

    def _assert_full(self, body):
        assert body["pid"] == os.getpid()
        assert body["models"] == _IDENTITIES

    def test_this_instances_attach_token_reads_it(self):
        app = _coordinating_app(model="secret-model")
        app.state.instance_token = "attach-" + "t" * 30
        client = TestClient(app)
        r = client.get("/v1/instances/status",
                       headers={"Authorization": "Bearer " + app.state.instance_token})
        assert r.status_code == 200
        self._assert_full(r.json())
        r = client.get("/v1/instances/status",
                       headers={"Authorization": "Bearer attach-wrong"})
        assert "pid" not in r.json()

    def test_an_api_key_with_models_read_reads_it(self, monkeypatch):
        key = "k" * 40
        monkeypatch.setenv("LOCALM_API_KEY", key)
        r = TestClient(_coordinating_app(model="secret-model")).get(
            "/v1/instances/status", headers={"Authorization": "Bearer " + key})
        assert r.status_code == 200
        self._assert_full(r.json())

    def test_an_api_key_in_x_api_key_reads_it(self, monkeypatch):
        key = "k" * 40
        monkeypatch.setenv("LOCALM_API_KEY", key)
        r = TestClient(_coordinating_app(model="secret-model")).get(
            "/v1/instances/status", headers={"x-api-key": key})
        self._assert_full(r.json())

    def test_a_browser_page_cannot_read_it_even_with_the_attach_token(self):
        app = _coordinating_app(model="secret-model")
        app.state.instance_token = "attach-" + "t" * 30
        r = TestClient(app).get(
            "/v1/instances/status",
            headers={"Authorization": "Bearer " + app.state.instance_token,
                     "Origin": "http://localhost:3000"})
        assert r.status_code == 404
        assert "alice-s1" not in r.text

    def test_a_verified_requester_reads_it_by_post(self, monkeypatch):
        seen = {}

        def fake_verify(requester, request_id, self_id, **k):
            seen.update(requester=requester, request_id=request_id, self_id=self_id)
            return True

        monkeypatch.setattr(gpu_registry, "verify_requester", fake_verify)
        r = TestClient(_coordinating_app(model="secret-model")).post(
            "/v1/instances/status",
            json={"requester": {"instance_id": "req", "port": 1}, "request_id": "rid"})
        assert r.status_code == 200
        self._assert_full(r.json())
        assert seen == {"requester": {"instance_id": "req", "port": 1},
                        "request_id": "rid", "self_id": "self-1"}

    def test_an_unverified_post_is_refused(self, monkeypatch):
        monkeypatch.setattr(gpu_registry, "verify_requester", lambda *a, **k: False)
        r = TestClient(_coordinating_app(model="secret-model")).post(
            "/v1/instances/status",
            json={"requester": {"instance_id": "req", "port": 1}, "request_id": "rid"})
        assert r.status_code == 403
        assert "alice-s1" not in r.text and "secret-model" not in r.text

    @pytest.mark.parametrize("app_kind", ["plain", "network", "browser"])
    def test_the_post_is_refused_where_the_get_is_not_served(self, app_kind, monkeypatch):
        monkeypatch.setattr(gpu_registry, "verify_requester", lambda *a, **k: True)
        headers = {}
        if app_kind == "plain":
            app = create_app(None)
        elif app_kind == "network":
            app = _coordinating_app(bind_host="0.0.0.0", model="secret-model")
        else:
            app = _coordinating_app(model="secret-model")
            headers = {"Origin": "http://localhost:3000"}
        r = TestClient(app).post(
            "/v1/instances/status", headers=headers,
            json={"requester": {"instance_id": "req", "port": 1}, "request_id": "rid"})
        assert r.status_code == 403
        assert "secret-model" not in r.text


# ------------------------------------------------------------------ #
#  The reading side: list_gpu_peers asks for the full status         #
# ------------------------------------------------------------------ #

class TestPeerDiscoveryReadsTheFullStatus:
    """The served app runs in this process, so its status names another pid;
    discovery always leaves out a status naming this process."""

    @pytest.fixture(autouse=True)
    def _loaded(self, monkeypatch):
        monkeypatch.setattr(hs, "_loaded_model_identities", lambda: list(_IDENTITIES))
        real = hs._gpu_status

        def as_another_process():
            status = real()
            return None if status is None else {**status, "pid": FOREIGN_PID}

        monkeypatch.setattr(hs, "_gpu_status", as_another_process)

    def test_a_coordinating_reader_gets_the_file_identity_over_real_sockets(
            self, monkeypatch):
        with _served(_coordinating_app(model="secret-model")) as port, \
                peer_server(localm_whoami("reader-1"),
                            post={"/v1/instances/vouch": _deferring_vouch}) as me:
            enable_detection(monkeypatch, port)
            gpu_registry.set_local_status_provider(
                lambda: None,
                identity={"instance_id": "reader-1", "port": me.port, "scheme": "http"})
            peers = gpu_registry.list_gpu_peers(timeout=5)
            plain = requests.get(f"http://127.0.0.1:{port}/v1/instances/status",
                                 timeout=10)
        assert [p["instance_id"] for p in peers] == ["self-1"]
        assert peers[0]["models"] == _IDENTITIES
        assert peers[0]["pid"] == FOREIGN_PID
        assert "alice-s1" not in plain.text and "pid" not in plain.json()

    def test_a_reader_without_coordination_gets_the_redacted_status(self, monkeypatch):
        with _served(_coordinating_app(model="secret-model")) as port:
            enable_detection(monkeypatch, port)
            peers = gpu_registry.list_gpu_peers(timeout=5)
        assert [p["instance_id"] for p in peers] == ["self-1"]
        assert peers[0]["models"] == [{"name": "secret-model", "file": "secret-model.gguf",
                                       "size": 1234}]
        assert "pid" not in peers[0]

    def test_a_same_install_reader_presents_the_attach_token(self, monkeypatch):
        from localm import instances
        from localm.config import home_dir
        app = _coordinating_app(model="secret-model")
        app.state.instance_token = "attach-" + "s" * 30
        with _served(app) as port:
            path = instances.register_instance(
                home_dir(), instance_id="self-1", port=port, host="127.0.0.1",
                root_dir=str(home_dir()), mode="api", token=app.state.instance_token)
            try:
                enable_detection(monkeypatch, port)
                peers = gpu_registry.list_gpu_peers(timeout=5)
            finally:
                instances.unregister_instance(path)
        assert peers[0]["models"] == _IDENTITIES

    def test_the_attach_token_is_not_sent_to_another_users_listener(self, monkeypatch):
        from localm import instances
        from localm.config import home_dir
        seen = []
        _as_another_os_user(monkeypatch)
        with peer_server(localm_whoami("self-1")) as fake:
            path = instances.register_instance(
                home_dir(), instance_id="self-1", port=fake.port, host="127.0.0.1",
                root_dir=str(home_dir()), mode="api", token="attach-secret-zz")
            real_get = requests.get

            def spy(url, **kw):
                seen.append(kw.get("headers") or {})
                return real_get(url, **kw)

            monkeypatch.setattr(requests, "get", spy)
            try:
                enable_detection(monkeypatch, fake.port)
                gpu_registry.list_gpu_peers(timeout=5)
            finally:
                instances.unregister_instance(path)
        assert seen and all("attach-secret-zz" not in str(h) for h in seen)

    def test_the_reader_excludes_itself_by_instance_id_without_a_pid(self, monkeypatch):
        with _served(_coordinating_app(model="secret-model")) as port:
            enable_detection(monkeypatch, port)
            gpu_registry.set_local_status_provider(
                lambda: None,
                identity={"instance_id": "self-1", "port": port, "scheme": "http"})
            assert gpu_registry.list_gpu_peers(timeout=5) == []


# ------------------------------------------------------------------ #
#  A request id cannot be relayed                                    #
# ------------------------------------------------------------------ #

class TestARequestIdCannotBeRelayed:
    """Another user's listener claiming a sibling's instance id must not receive
    a request id it could forward to that sibling, and an id confirms only the
    purpose and the port it was sent for."""

    def _owned_except(self, monkeypatch, *foreign_ports):
        monkeypatch.setattr(listeners, "_port_owned_by_current_user",
                            lambda port, dial: port not in foreign_ports)

    def test_an_impostor_claiming_a_siblings_id_gets_no_request_id(self, monkeypatch):
        with peer_server(localm_whoami("self-1")) as impostor:
            self._owned_except(monkeypatch, impostor.port)
            enable_detection(monkeypatch, impostor.port)
            gpu_registry.set_local_status_provider(
                lambda: None,
                identity={"instance_id": "reader-1", "port": 18642, "scheme": "http"})
            gpu_registry.list_gpu_peers(timeout=5)
            posts = [r for r in impostor.requests if r[0] == "POST"]
        assert posts == []
        assert gpu_registry._pending == {}

    def test_no_unload_request_is_sent_to_another_users_listener(self, monkeypatch):
        gpu_registry.set_local_status_provider(
            lambda: {"instance_id": "me-1", "port": 18642, "scheme": "http"})
        with peer_server(localm_whoami("peer-1"),
                         post={"/v1/instances/cooperate-unload":
                               lambda b: (200, {"status": "unloaded"})}) as srv:
            self._owned_except(monkeypatch, srv.port)
            peer = {"instance_id": "peer-1", "port": srv.port, "host": "127.0.0.1",
                    "scheme": "http"}
            assert gpu_registry.request_cooperative_unload(peer) is False
            assert srv.requests == []

    def test_a_status_request_id_does_not_confirm_an_unload(self):
        gpu_registry._remember_request("rid-s", "peer-1", 8700, gpu_registry.PURPOSE_STATUS)
        assert gpu_registry.vouch_for("rid-s", "peer-1", 8700,
                                      gpu_registry.PURPOSE_UNLOAD) is False
        assert gpu_registry.vouch_for("rid-s", "peer-1", 8700,
                                      gpu_registry.PURPOSE_STATUS) is True

    def test_a_request_id_confirms_only_for_the_port_it_was_sent_to(self):
        gpu_registry._remember_request("rid-p", "peer-1", 8700, gpu_registry.PURPOSE_UNLOAD)
        assert gpu_registry.vouch_for("rid-p", "peer-1", 8701,
                                      gpu_registry.PURPOSE_UNLOAD) is False
        assert gpu_registry.vouch_for("rid-p", "peer-1", 8700,
                                      gpu_registry.PURPOSE_UNLOAD) is True

    @pytest.mark.parametrize("port, purpose", [(True, None), ("8700", None),
                                               (None, 5), (None, ["unload"])])
    def test_a_malformed_port_or_purpose_is_refused(self, port, purpose):
        gpu_registry._remember_request("rid-m", "peer-1", 8700, gpu_registry.PURPOSE_UNLOAD)
        assert gpu_registry.vouch_for("rid-m", "peer-1", port, purpose) is False

    def test_an_asker_naming_neither_is_matched_on_the_id(self):
        gpu_registry._remember_request("rid-o", "peer-1", 8700, gpu_registry.PURPOSE_UNLOAD)
        assert gpu_registry.vouch_for("rid-o", "peer-1") is True

    def test_a_relayed_status_id_cannot_unload_a_real_sibling(self, monkeypatch):
        calls = []

        async def unload_all_models():
            calls.append(1)
            return {"status": "unloaded"}

        monkeypatch.setattr(hs, "unload_all_models", unload_all_models)
        with _served(_coordinating_app()) as port, \
                peer_server(localm_whoami("reader-1"),
                            post={"/v1/instances/vouch": _deferring_vouch}) as reader:
            gpu_registry._remember_request("leaked", "self-1", port,
                                           gpu_registry.PURPOSE_STATUS)
            r = requests.post(
                f"http://127.0.0.1:{port}/v1/instances/cooperate-unload",
                json={"requester": {"instance_id": "reader-1", "port": reader.port,
                                    "scheme": "http"},
                      "request_id": "leaked"}, timeout=10)
            vouch_bodies = [b for m, p, b in reader.requests if p == "/v1/instances/vouch"]
        assert r.status_code == 403
        assert calls == []
        assert vouch_bodies == [{"request_id": "leaked", "asker_instance_id": "self-1",
                                 "asker_port": port,
                                 "purpose": gpu_registry.PURPOSE_UNLOAD}]


# ------------------------------------------------------------------ #
#  A redacted identity still tells files apart                       #
# ------------------------------------------------------------------ #

class TestRedactedIdentityStillMatchesByFile:
    def _local(self):
        return {"path": _PRIVATE_PATH, "size": 1234, "sha256": None}

    def test_the_same_file_name_and_size_is_the_same_file(self):
        from localm.peer_routing import _same_file
        assert _same_file(self._local(), {"name": "m", "file": "secret-model.gguf",
                                          "size": 1234}) is True

    def test_a_different_size_is_a_different_file(self):
        from localm.peer_routing import _same_file
        assert _same_file(self._local(), {"name": "m", "file": "secret-model.gguf",
                                          "size": 99}) is False

    def test_a_different_file_name_is_a_different_file(self):
        from localm.peer_routing import _same_file
        assert _same_file(self._local(), {"name": "m", "file": "other.gguf",
                                          "size": 1234}) is False

    def test_a_name_alone_cannot_be_compared(self):
        from localm.peer_routing import _same_file
        assert _same_file(self._local(), {"name": "m"}) is None
