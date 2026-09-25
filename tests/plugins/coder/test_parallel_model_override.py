# SPDX-License-Identifier: AGPL-3.0-or-later
"""dispatch_parallel's per-child model: a child that asks for another model gets
a new local backend, unless the parent's backend declares
``supports_model_override = False``. That backend answers through another
instance, and a new backend would be built for that instance's port with this
install's credential."""

from types import SimpleNamespace

import pytest

from localm.plugins.coder.tools import parallel as par


@pytest.fixture
def built(monkeypatch):
    """Every make_localm_backend call, as (model, port)."""
    import localm.plugins.coder.backends.http as http
    calls = []

    def make(model, port=None, **kw):
        calls.append((model, port))
        return SimpleNamespace(model_id=model)
    monkeypatch.setattr(http, "make_localm_backend", make)
    return calls


def _backend(**kw):
    return SimpleNamespace(model_id="parent-model",
                           _base_url="http://127.0.0.1:9123/v1", **kw)


def test_a_backend_that_refuses_overrides_builds_nothing_and_keeps_the_parent_model(built):
    parent = _backend(supports_model_override=False)
    backend, detail = par._child_backend(parent, "other-model")
    assert built == [], "a backend was built for the peer's port"
    assert backend is parent
    assert "not available" in detail and "parent's model" in detail


def test_an_ordinary_backend_gets_a_new_backend_for_the_requested_model(built):
    backend, detail = par._child_backend(_backend(), "other-model")
    assert built == [("other-model", 9123)]
    assert backend.model_id == "other-model"
    assert detail == ""


def test_no_override_or_the_same_model_keeps_the_parent_backend(built):
    parent = _backend()
    assert par._child_backend(parent, None) == (parent, "")
    assert par._child_backend(parent, "parent-model") == (parent, "")
    assert built == []


def test_a_backend_that_cannot_be_built_falls_back_to_the_parent(monkeypatch):
    import localm.plugins.coder.backends.http as http

    def fail(model, port=None, **kw):
        raise RuntimeError("no server")
    monkeypatch.setattr(http, "make_localm_backend", fail)
    parent = _backend()
    backend, detail = par._child_backend(parent, "other-model")
    assert backend is parent
    assert "unavailable (no server)" in detail


def test_the_mcp_peer_coder_backend_refuses_overrides():
    from localm.plugins.mcpserver.tools.media_coder import PeerCoderBackend
    assert PeerCoderBackend.supports_model_override is False
