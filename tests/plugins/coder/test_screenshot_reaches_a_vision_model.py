# SPDX-License-Identifier: AGPL-3.0-or-later
"""A browser screenshot is shown to a model that can read images, as image parts
of the next request only, and a model that cannot read images is told it was
not shown the picture."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from localm.plugins.coder.audit import SessionMode
from localm.plugins.coder.backends.http import HTTPBackend, _anthropic_parts
from localm.plugins.coder.tools import browser as browser_tools

_PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64


@pytest.fixture(autouse=True)
def _trusted(monkeypatch):
    monkeypatch.setattr("localm.plugins.coder.hook_trust.project_trusted", lambda cwd: True)


@pytest.fixture
def page(monkeypatch):
    fake = SimpleNamespace(screenshot=lambda full_page=False: _PNG)
    monkeypatch.setattr(browser_tools, "_existing", lambda session: fake)
    return fake


class _Recording:
    model_id = "m"
    native_tools = False
    supports_grammar = False
    last_usage = {"total_tokens": 0}

    def __init__(self, script, reads_images):
        self.script = list(script)
        self.sent: list = []
        self._reads = reads_images

    def reads_images(self):
        return self._reads

    def set_tools(self, defs):
        pass

    def chat(self, messages, **kw):
        self.sent.append(messages)
        return self.script.pop(0) if self.script else "Done."

    def chat_stream(self, messages, **kw):
        yield self.chat(messages, **kw)


def _agent(cwd, backend):
    from localm.plugins.coder.agent import Agent
    with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        PM.build.return_value.file_count.return_value = 0
        PM.build.return_value.truncated = False
        agent = Agent(backend, cwd=cwd, self_verify=False, mode=SessionMode.LOG,
                      auto_approve=True, browser_enabled=True)
    agent.browser_enabled = True
    return agent


def _shot_call():
    return "<tool_call>" + json.dumps({"name": "browser_screenshot", "args": {}}) + "</tool_call>"


def test_the_tool_returns_the_image_and_marks_it_untrusted(tmp_path, page):
    res = browser_tools.tool_browser_screenshot(
        tmp_path, _session=SimpleNamespace(browser_enabled=True))
    assert res.ok and (tmp_path / "screenshot.png").read_bytes() == _PNG
    assert res.images == ["data:image/png;base64," + base64.b64encode(_PNG).decode()]
    assert res.untrusted_output is True


def test_a_vision_model_gets_the_image_in_the_next_request_only(tmp_path, page):
    backend = _Recording([_shot_call(), "I see it", "next"], reads_images=True)
    agent = _agent(tmp_path, backend)
    agent.run_task("look at the page")
    after_tool = backend.sent[1][-1]
    assert isinstance(after_tool["content"], list)
    assert after_tool["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert all(isinstance(m.get("content"), str) for m in agent._messages)


def test_a_text_only_model_is_told_it_did_not_see_the_image(tmp_path, page):
    backend = _Recording([_shot_call(), "ok"], reads_images=False)
    agent = _agent(tmp_path, backend)
    agent.run_task("look at the page")
    after_tool = backend.sent[1][-1]
    assert isinstance(after_tool["content"], str)
    assert "The image was not shown to you" in after_tool["content"]


class TestReadsImages:
    def test_a_localm_server_is_asked_once_per_model(self, monkeypatch):
        be = HTTPBackend("http://127.0.0.1:9/v1", "vl", localm_server=True,
                         trusted_peer=True)
        asked = []

        def fake_get(url):
            asked.append(url)
            return SimpleNamespace(status_code=200, json=lambda: {"vision": True},
                                   text=json.dumps({"vision": True}),
                                   headers={"Content-Type": "application/json"})

        monkeypatch.setattr(be, "_get", fake_get)
        monkeypatch.setattr("localm.jsonreply.response_json", lambda r: r.json())
        assert be.reads_images() is True and be.reads_images() is True
        assert asked == ["http://127.0.0.1:9/v1/models/vl"]

    def test_a_server_that_cannot_answer_reads_no_images(self, monkeypatch):
        be = HTTPBackend("http://127.0.0.1:9/v1", "m", localm_server=True,
                         trusted_peer=True)

        def boom(url):
            raise OSError("refused")

        monkeypatch.setattr(be, "_get", boom)
        assert be.reads_images() is False

    def test_anthropic_gets_base64_image_blocks(self):
        parts = [{"type": "text", "text": "t"},
                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}]
        assert _anthropic_parts(parts) == [
            {"type": "text", "text": "t"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": "QUJD"}}]
