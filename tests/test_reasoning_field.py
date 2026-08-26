# SPDX-License-Identifier: AGPL-3.0-or-later
"""The model's <think> reasoning is separated from the visible answer.

The API surfaces it in a `reasoning_content` field (message + streaming delta)
with clean `content`; the full-mode transcript writes it to a collapsed
`<details>` block instead of dumping raw <think> tags inline.
"""

import json
import os
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from localm.inference.http_server import create_app


def _engine(pieces):
    """A mock engine whose chat_stream yields *pieces* (already canonical, as the
    real engine emits post-textnorm)."""
    e = MagicMock()
    e.display_name = "test-model"
    e.count_tokens.return_value = 5
    e.supports_images = False
    type(e).loaded = property(lambda self: True)
    e.chat_stream.side_effect = lambda messages, **k: iter(list(pieces))
    return e


def _client(pieces):
    os.environ.pop("LOCALM_API_KEY", None)
    return TestClient(create_app(_engine(pieces)), raise_server_exceptions=True)


_PAYLOAD = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


class TestNonStreaming:
    def test_reasoning_split_into_field(self):
        with _client(["<think>because reasons</think>The answer."]) as c:
            r = c.post("/v1/chat/completions", json=_PAYLOAD)
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        assert msg["content"] == "The answer."
        assert msg["reasoning_content"] == "because reasons"
        # NEGATIVE: the raw tag must never appear in the visible content.
        assert "<think>" not in msg["content"]

    def test_no_think_leaves_reasoning_null(self):
        with _client(["Just a plain answer."]) as c:
            r = c.post("/v1/chat/completions", json=_PAYLOAD)
        msg = r.json()["choices"][0]["message"]
        assert msg["content"] == "Just a plain answer."
        assert msg["reasoning_content"] is None


class TestStreaming:
    def _collect(self, resp):
        content, reasoning = "", ""
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            delta = json.loads(payload)["choices"][0]["delta"]
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
        return content, reasoning

    def test_stream_routes_reasoning_and_content(self):
        # The think tags are fragmented across pieces to exercise the splitter.
        pieces = ["Pre ", "<thi", "nk>secret", " stuff</thin", "k> Post"]
        body = {**_PAYLOAD, "stream": True}
        with _client(pieces) as c:
            with c.stream("POST", "/v1/chat/completions", json=body) as resp:
                assert resp.status_code == 200
                content, reasoning = self._collect(resp)
        assert content == "Pre  Post"
        assert reasoning == "secret stuff"
        # NEGATIVE: no fragment of the tag leaks into either channel.
        assert "think" not in content and "<thi" not in content
        assert "</thin" not in reasoning

    def test_stream_no_think_is_content_only(self):
        body = {**_PAYLOAD, "stream": True}
        with _client(["hello ", "world"]) as c:
            with c.stream("POST", "/v1/chat/completions", json=body) as resp:
                content, reasoning = self._collect(resp)
        assert content == "hello world"
        assert reasoning == ""


class TestTranscript:
    def test_full_transcript_separates_reasoning(self, tmp_path, monkeypatch):
        import localm.audit as audit
        monkeypatch.setattr(audit, "_sessions_dir", lambda: tmp_path)
        t = audit.MarkdownTranscript(label="chat")
        t.exchange("the question", "<think>my private reasoning</think>The answer.")
        text = t.path.read_text(encoding="utf-8")
        assert "The answer." in text
        assert "my private reasoning" in text          # preserved...
        assert "<details>" in text and "reasoning" in text   # ...but in a block
        # NEGATIVE: the raw think tags are not dumped inline.
        assert "<think>" not in text and "</think>" not in text

    def test_transcript_without_reasoning_has_no_details(self, tmp_path, monkeypatch):
        import localm.audit as audit
        monkeypatch.setattr(audit, "_sessions_dir", lambda: tmp_path)
        t = audit.MarkdownTranscript(label="chat")
        t.exchange("q", "a plain answer")
        text = t.path.read_text(encoding="utf-8")
        assert "a plain answer" in text
        assert "<details>" not in text
