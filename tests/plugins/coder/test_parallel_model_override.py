# SPDX-License-Identifier: AGPL-3.0-or-later
"""A sub-agent's model override (dispatch_parallel and spawn_agent share
``_child_backend``): a child that asks for another model gets
``backend.with_model(model)``, the same server reached the same way, unless the
parent's backend declares ``supports_model_override = False`` (it answers
through another instance) or cannot build one. Every fallback to the parent's
model is said, in the detail line and in the spawn_agent result."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from localm.plugins.coder.backends.http import HTTPBackend
from localm.plugins.coder.tools import parallel as par


def _backend(**kw):
    calls = []

    def with_model(model):
        calls.append(model)
        return SimpleNamespace(model_id=model)
    ns = SimpleNamespace(model_id="parent-model", with_model=with_model, **kw)
    ns.built = calls
    return ns


def test_a_backend_that_refuses_overrides_builds_nothing_and_keeps_the_parent_model():
    parent = _backend(supports_model_override=False)
    backend, detail = par._child_backend(parent, "other-model")
    assert parent.built == [], "a backend was built for the peer"
    assert backend is parent
    assert "not available" in detail and "parent's model" in detail


def test_an_ordinary_backend_gets_a_new_backend_for_the_requested_model():
    parent = _backend()
    backend, detail = par._child_backend(parent, "other-model")
    assert parent.built == ["other-model"]
    assert backend.model_id == "other-model"
    assert detail == ""


def test_no_override_or_the_same_model_keeps_the_parent_backend():
    parent = _backend()
    assert par._child_backend(parent, None) == (parent, "")
    assert par._child_backend(parent, "parent-model") == (parent, "")
    assert parent.built == []


def test_a_backend_that_cannot_be_built_falls_back_to_the_parent_and_says_so():
    def fail(model):
        raise RuntimeError("no server")
    parent = SimpleNamespace(model_id="parent-model", with_model=fail)
    backend, detail = par._child_backend(parent, "other-model")
    assert backend is parent
    assert "unavailable (no server)" in detail


def test_a_backend_without_with_model_keeps_the_parent_model_and_says_so():
    parent = SimpleNamespace(model_id="parent-model")
    backend, detail = par._child_backend(parent, "other-model")
    assert backend is parent
    assert "parent's model" in detail


@pytest.mark.parametrize("url", [
    "https://127.0.0.1:9123/v1",
    "http://[::1]:9123/v1",
    "http://192.168.1.20:8080/v1",
])
def test_with_model_keeps_the_server_the_parent_reaches(url):
    parent = HTTPBackend(url, "parent-model", api_key="typed-key",
                         localm_server=True, verify=False, trusted_peer=True,
                         temperature=0.2)
    parent.session_mode = "privacy"
    child = parent.with_model("other-model")
    assert child is not parent
    assert child._base_url == parent._base_url
    assert child._api_key == "typed-key"
    assert child._verify is False
    assert child._is_local_server is True
    assert child._extra == {"temperature": 0.2}
    assert child.session_mode == "privacy"
    assert child.model_id == "other-model" and child.model_pinned is True
    assert parent.model_id == "parent-model"


def test_with_model_keeps_a_provider_backend_on_its_provider():
    parent = HTTPBackend("https://api.anthropic.com/v1", "claude-a", api_key="k",
                         anthropic=True, native_tools=True, trusted_peer=True)
    child = parent.with_model("claude-b")
    assert child._base_url == "https://api.anthropic.com/v1"
    assert child.anthropic is True and child.native_tools is True
    assert child._is_local_server is False


def _agent(tmp_path, backend):
    from localm.plugins.coder.agent import Agent
    from localm.plugins.coder.audit import SessionMode
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        return Agent(backend=backend, cwd=tmp_path, mode=SessionMode.LOG,
                     auto_approve=True)


def test_a_spawned_agent_on_another_model_keeps_the_parent_server_and_mode(tmp_path):
    from localm.plugins.coder.tools.agents import _prepare_child
    parent_backend = HTTPBackend("https://127.0.0.1:9123/v1", "parent-model",
                                 api_key="typed-key", localm_server=True,
                                 verify=False, trusted_peer=True)
    parent_backend.session_mode = "privacy"
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        parent = _agent(tmp_path, parent_backend)
        try:
            child, _task = _prepare_child(
                tmp_path, "do it", "child", None, "other-model", 3, None, parent,
                confirm_handler=None, tool="spawn_agent")
            assert child.backend is not parent_backend
            assert child.backend.model_id == "other-model"
            assert child.backend._base_url == "https://127.0.0.1:9123/v1"
            assert child.backend._api_key == "typed-key"
            assert child.backend._headers()["X-Localm-Session-Mode"] == "privacy"
            assert child.model_note == ""
            child.close()
        finally:
            parent.close()


def test_spawn_agent_says_when_the_child_ran_on_the_parent_model(tmp_path, monkeypatch):
    from localm.plugins.coder.agent import Agent
    from localm.plugins.coder.tools.agents import tool_spawn_agent
    parent_backend = HTTPBackend("http://127.0.0.1:9123/v1", "parent-model",
                                 localm_server=True, trusted_peer=True)

    def fail(model):
        raise RuntimeError("no server")
    monkeypatch.setattr(parent_backend, "with_model", fail)
    monkeypatch.setattr(Agent, "run_task", lambda self, task, **kw: "done")
    parent = _agent(tmp_path, parent_backend)
    try:
        result = tool_spawn_agent(tmp_path, "do it", name="w", model="other-model",
                                  _parent_agent=parent)
    finally:
        parent.close()
    assert result.ok
    assert "unavailable (no server)" in result.output
    assert "parent's model" in result.summary


def test_the_mcp_peer_coder_backend_refuses_overrides():
    from localm.plugins.mcpserver.tools.media_coder import PeerCoderBackend
    assert PeerCoderBackend.supports_model_override is False
