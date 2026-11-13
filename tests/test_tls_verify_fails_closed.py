"""A failure while deciding TLS verification sends nothing, and the owner-key
memo handle is not an unsalted hash of the secret."""
from __future__ import annotations

import pytest
import requests

from localm import gpu_registry, peer_routing, tls


class _Boom(Exception):
    pass


@pytest.fixture
def verify_raises(monkeypatch):
    def _raise(url):
        raise _Boom("cannot decide")
    monkeypatch.setattr(tls, "requests_verify", _raise)


@pytest.fixture
def no_network(monkeypatch):
    sent = []

    def _send(*a, **k):
        sent.append((a, k))
        raise AssertionError("a request was sent after verification failed")
    monkeypatch.setattr(requests, "post", _send)
    monkeypatch.setattr(requests, "get", _send)
    monkeypatch.setattr(requests, "request", _send)
    monkeypatch.setattr(requests.Session, "request", _send)
    from localm import loopback_http
    monkeypatch.setattr(loopback_http, "local_request", _send)
    return sent


def test_mount_remote_gui_does_not_dial_when_verification_cannot_be_decided(
        verify_raises, no_network):
    from localm.plugins.gui import cli
    reason = cli._mount_remote_gui(
        {"scheme": "https", "port": 8651, "token": "the-token"})
    assert no_network == []
    assert reason == "could not determine TLS verification (_Boom)"


@pytest.mark.parametrize("exc", [FileNotFoundError, _Boom])
def test_gpu_registry_probes_send_nothing_when_verification_cannot_be_decided(
        monkeypatch, no_network, exc):
    def _raise(url):
        raise exc("x")
    monkeypatch.setattr(tls, "requests_verify", _raise)
    assert gpu_registry._get_status("https", 8651, 1.0, "127.0.0.1") is None
    assert gpu_registry._post_json("https://127.0.0.1:8651/x", {}, 1.0) is None
    assert no_network == []


def test_peer_recording_mode_sends_nothing_when_verification_cannot_be_decided(
        monkeypatch, verify_raises, no_network):
    monkeypatch.setattr(peer_routing, "is_routable_peer_endpoint",
                        lambda host, scheme: True)
    peer = {"scheme": "https", "host": "127.0.0.1", "port": 8651,
            "instance_id": "p"}
    assert peer_routing.peer_recording_mode(peer, "k", "privacy") is None
    assert no_network == []


@pytest.mark.parametrize("exc", [FileNotFoundError, _Boom])
def test_verify_peer_credential_sends_nothing_when_verification_cannot_be_decided(
        monkeypatch, no_network, exc):
    def _raise(url):
        raise exc("x")
    monkeypatch.setattr(tls, "requests_verify", _raise)
    monkeypatch.setattr(peer_routing, "is_routable_peer_endpoint",
                        lambda host, scheme: True)
    monkeypatch.setattr(peer_routing, "peer_listener_is_ours",
                        lambda host, port: True)
    peer = {"scheme": "https", "host": "127.0.0.1", "port": 8651,
            "instance_id": "p"}
    with pytest.raises(exc):
        peer_routing.verify_peer_credential(peer, "k")
    assert no_network == []
