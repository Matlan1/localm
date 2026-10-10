# SPDX-License-Identifier: AGPL-3.0-or-later
"""JSON bodies from a peer, a proxy, a remote site or another localm instance:
an over-nested or over-long-integer body is answered exactly like any other
non-JSON body, not with an uncaught ``RecursionError``. Each test serves the body
from a real loopback HTTP server and calls the real client function."""

from __future__ import annotations

import base64
import json

import pytest

from localm import _proxy, gpu_registry, instances, peer_routing, selfclient
from localm.bugreport import transport
from localm.cli._core import server_call
from localm.bugreport import LocalmError
from localm.peer_routing import PeerCredentialError, PeerRoute
from localm.plugins.gui import cli as gui_cli
from localm.web_retrieval import sites
from tests._hostile_json import HOSTILE, port_of, serve

WHOAMI = json.dumps({"app": "localm", "instance_id": "peer-1"})


# ------------------------------------------------------------- web retrieval

@HOSTILE
def test_github_readme_with_a_hostile_body_is_not_a_readme(doc):
    assert sites._decode_github_readme(doc) is None


@HOSTILE
def test_stackexchange_items_of_a_hostile_body_are_empty(doc):
    assert sites._items(doc) == []


def test_github_readme_still_decodes_a_well_formed_body():
    body = json.dumps({"encoding": "base64",
                       "content": base64.b64encode(b"# hi").decode()})
    assert sites._decode_github_readme(body) == "# hi"


# ------------------------------------------------------------------ peers

@HOSTILE
def test_forward_body_passes_a_hostile_request_body_through(doc):
    route = PeerRoute(model="m", instance_id="i", host="127.0.0.1", port=1,
                      scheme="http", api_key="")
    assert peer_routing.forward_body(route, doc.encode()) == doc.encode()


@HOSTILE
def test_peer_credential_check_falls_back_to_the_models_probe(doc):
    with serve({"/api/session": doc, "/v1/models": "{}"}) as base:
        peer = {"host": "127.0.0.1", "port": port_of(base), "scheme": "http",
                "instance_id": "peer-1"}
        assert peer_routing.verify_peer_credential(peer, "k") is None


@HOSTILE
def test_peer_credential_check_still_refuses_after_a_hostile_session_body(doc):
    with serve({"/api/session": doc}, status=200) as base:
        peer = {"host": "127.0.0.1", "port": port_of(base), "scheme": "http",
                "instance_id": "peer-1"}
        with pytest.raises(PeerCredentialError):
            peer_routing.verify_peer_credential(peer, "k")


# --------------------------------------------------------------- instances

@HOSTILE
def test_whoami_with_a_hostile_body_is_no_instance(doc):
    with serve(doc) as base:
        assert instances.fetch_any_whoami("http", port_of(base), 2.0) is None


@HOSTILE
def test_activity_read_with_a_hostile_body_is_http_not_ok(doc):
    with serve(doc) as base:
        assert selfclient.read_activity("http", port_of(base)) == ("http", 200)


@HOSTILE
def test_model_file_hold_with_a_hostile_body_is_http(doc):
    with serve(doc) as base:
        assert selfclient.read_model_file_hold(
            "http", port_of(base), "m", None) == ("http", 200)


@HOSTILE
def test_model_file_hold_404_with_a_hostile_body_is_unsupported(doc):
    with serve(doc, status=404) as base:
        assert selfclient.read_model_file_hold(
            "http", port_of(base), "m", None) == ("unsupported", 404)


@HOSTILE
def test_cli_server_call_ok_with_a_hostile_body_is_http(doc):
    with serve(doc) as base:
        assert server_call(base, {}, "GET", "/x") == (
            "http", (200, "the reply was not JSON"))


@HOSTILE
def test_cli_server_call_error_with_a_hostile_body_keeps_the_text(doc):
    with serve(doc, status=500) as base:
        state, (code, detail) = server_call(base, {}, "GET", "/x")
    assert (state, code) == ("http", 500)
    assert detail == doc[:200]


@HOSTILE
def test_remote_gui_mount_with_a_hostile_error_body_reports_the_status(doc):
    with serve(doc, status=500) as base:
        entry = {"port": port_of(base), "token": "t"}
        assert gui_cli._mount_remote_gui(entry) == "HTTP 500"


# ------------------------------------------------------------ gpu_registry

@HOSTILE
def test_peer_status_with_a_hostile_body_is_none(doc):
    with serve(doc) as base:
        assert gpu_registry.fetch_status("http", port_of(base), 2.0) is None


@HOSTILE
def test_requester_whose_vouch_body_is_hostile_is_not_verified(doc):
    with serve({"/whoami": WHOAMI, gpu_registry.VOUCH_PATH: doc}) as base:
        requester = {"instance_id": "peer-1", "port": port_of(base), "scheme": "http"}
        assert gpu_registry.verify_requester(requester, "req-1", "me") is False


@HOSTILE
def test_unload_request_answered_with_a_hostile_body_is_false(doc, monkeypatch):
    monkeypatch.setattr(gpu_registry, "own_status",
                        lambda: {"instance_id": "me", "port": 1, "scheme": "http"})
    with serve({gpu_registry.UNLOAD_PATH: doc}) as base:
        peer = {"instance_id": "peer-1", "port": port_of(base), "scheme": "http",
                "host": "127.0.0.1"}
        assert gpu_registry.request_cooperative_unload(peer) is False


# ----------------------------------------------------------- proxy / upload

@HOSTILE
def test_proxy_request_with_a_hostile_reply_is_a_localm_error(doc):
    with pytest.raises(LocalmError, match="non-JSON"):
        _proxy.request("http://proxy.invalid", "/x",
                       opener=lambda *a: (200, doc.encode()))


@HOSTILE
def test_report_upload_with_a_hostile_reply_returns_the_raw_text(doc):
    out = transport.upload_report("t", "b", url="https://upload.invalid/x",
                                  opener=lambda *a: (200, doc))
    assert out == {"raw": doc[:300]}

