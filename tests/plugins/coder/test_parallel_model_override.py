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


def test_a_new_backend_keeps_the_parent_session_mode(built):
    backend, _detail = par._child_backend(_backend(session_mode="privacy"),
                                          "other-model")
    assert backend.session_mode == "privacy"


def test_a_spawned_agent_on_another_model_keeps_the_parent_session_mode(
        tmp_path, monkeypatch):
    from unittest.mock import patch

    from localm.plugins.coder.agent import Agent
    from localm.plugins.coder.audit import SessionMode
    from localm.plugins.coder.backends.http import HTTPBackend
    from localm.plugins.coder.tools.agents import _prepare_child

    parent_backend = HTTPBackend("http://127.0.0.1:9123/v1", "parent-model",
                                 localm_server=True)
    parent_backend.session_mode = "privacy"
    monkeypatch.setattr(
        "localm.plugins.coder.backends.http.make_localm_backend",
        lambda model, port=None, **kw: HTTPBackend(
            f"http://127.0.0.1:{port}/v1", model, localm_server=True))
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        parent = Agent(backend=parent_backend, cwd=tmp_path, mode=SessionMode.LOG,
                       auto_approve=True)
        try:
            child, _task = _prepare_child(
                tmp_path, "do it", "child", None, "other-model", 3, None, parent,
                confirm_handler=None, tool="spawn_agent")
            assert child.backend is not parent_backend
            assert child.backend.model_id == "other-model"
            assert child.backend._headers()["X-Localm-Session-Mode"] == "privacy"
            child.close()
        finally:
            parent.close()


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
