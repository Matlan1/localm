# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.plugins.coder.backends.http - usage capture from responses."""

import json
import unittest
from unittest.mock import MagicMock, patch

from localm.plugins.coder.backends.http import HTTPBackend


def _make_backend():
    return HTTPBackend("http://127.0.0.1:8080/v1", "test-model")


def _mock_non_streaming_response(content: str, usage: dict, reasoning: str = None):
    """Build a mock requests.Response for a non-streaming completion."""
    message = {"content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": message}],
        "usage": usage,
    }
    return resp


def _sse_lines(chunks: list, include_done: bool = True) -> list:
    """Convert a list of chunk dicts into SSE line bytes for iter_lines()."""
    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}".encode())
    if include_done:
        lines.append(b"data: [DONE]")
    return lines


class TestHTTPBackendChat(unittest.TestCase):
    @patch("requests.post")
    def test_chat_returns_content(self, mock_post):
        mock_post.return_value = _mock_non_streaming_response(
            "Hello!", {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        )
        backend = _make_backend()
        result = backend.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "Hello!")

    @patch("requests.post")
    def test_chat_captures_usage(self, mock_post):
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        mock_post.return_value = _mock_non_streaming_response("OK", usage)
        backend = _make_backend()
        backend.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(backend.last_usage["total_tokens"], 15)
        self.assertEqual(backend.last_usage["prompt_tokens"], 10)

    @patch("requests.post")
    def test_chat_clears_last_usage_before_call(self, mock_post):
        """last_usage from a previous call should not leak into a failed call."""
        backend = _make_backend()
        backend._last_usage = {"total_tokens": 999}
        mock_post.return_value = _mock_non_streaming_response("hi", {})
        backend.chat([{"role": "user", "content": "x"}])
        self.assertEqual(backend.last_usage, {})

    @patch("requests.post")
    def test_last_usage_is_a_copy(self, mock_post):
        """Mutating the returned dict should not affect the backend's internal state."""
        usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        mock_post.return_value = _mock_non_streaming_response("hi", usage)
        backend = _make_backend()
        backend.chat([{"role": "user", "content": "x"}])
        lu = backend.last_usage
        lu["total_tokens"] = 999
        self.assertEqual(backend.last_usage["total_tokens"], 3)


class TestHTTPBackendChatReasoning(unittest.TestCase):
    """AUD-HIGH-17-3: chat() must capture the server's H4 `reasoning_content`
    field into last_reasoning WITHOUT mixing it into the returned text - the
    coder loop has no downstream splitter, so any inline <think> tag in the
    returned string would leak into the visible answer, audit log, and
    conversation history verbatim."""

    @patch("requests.post")
    def test_chat_captures_reasoning_separately(self, mock_post):
        mock_post.return_value = _mock_non_streaming_response(
            "The answer.", {}, reasoning="because reasons")
        backend = _make_backend()
        result = backend.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "The answer.")
        self.assertEqual(backend.last_reasoning, "because reasons")
        # NEGATIVE: the raw reasoning text never leaks into the returned answer.
        self.assertNotIn("because reasons", result)
        self.assertNotIn("<think>", result)

    @patch("requests.post")
    def test_chat_no_reasoning_field_leaves_last_reasoning_empty(self, mock_post):
        mock_post.return_value = _mock_non_streaming_response("Just an answer.", {})
        backend = _make_backend()
        backend.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(backend.last_reasoning, "")

    @patch("requests.post")
    def test_chat_clears_last_reasoning_before_call(self, mock_post):
        """last_reasoning from a previous call must not leak into the next one."""
        backend = _make_backend()
        backend._last_reasoning = "stale reasoning from a prior turn"
        mock_post.return_value = _mock_non_streaming_response("hi", {})
        backend.chat([{"role": "user", "content": "x"}])
        self.assertEqual(backend.last_reasoning, "")


class TestHTTPBackendChatStream(unittest.TestCase):
    def _make_stream_response(self, tokens: list, usage: dict):
        """Build a mock streaming response that yields SSE chunks."""
        chunks = []
        for token in tokens:
            chunks.append({
                "choices": [{"delta": {"content": token}, "finish_reason": None}],
            })
        # Final done chunk with usage
        chunks.append({
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": usage,
        })

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = _sse_lines(chunks)
        return mock_resp

    @patch("requests.post")
    def test_stream_yields_tokens(self, mock_post):
        mock_post.return_value = self._make_stream_response(["Hi", " there"], {})
        backend = _make_backend()
        tokens = list(backend.chat_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(tokens, ["Hi", " there"])

    @patch("requests.post")
    def test_stream_captures_usage_from_final_chunk(self, mock_post):
        usage = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
        mock_post.return_value = self._make_stream_response(["ok"], usage)
        backend = _make_backend()
        list(backend.chat_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(backend.last_usage["total_tokens"], 6)

    @patch("requests.post")
    def test_stream_clears_usage_before_call(self, mock_post):
        backend = _make_backend()
        backend._last_usage = {"total_tokens": 777}
        mock_post.return_value = self._make_stream_response(["x"], {})
        list(backend.chat_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(backend.last_usage, {})


class TestHTTPBackendChatStreamReasoning(unittest.TestCase):
    """AUD-HIGH-17-3: chat_stream() must route the server's H4
    `reasoning_content` deltas to on_reasoning and last_reasoning, and NEVER
    yield them as part of the content Iterator[str] - the coder loop's
    _stream_hiding_tool_calls, terminal display, event-sink, audit log, and
    conversation history all consume that iterator directly with no splitter,
    so an inlined reasoning chunk would leak through every one of them."""

    @staticmethod
    def _stream_response(delta_dicts: list):
        chunks = [{"choices": [{"delta": d, "finish_reason": None}]} for d in delta_dicts]
        chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = _sse_lines(chunks)
        return mock_resp

    @patch("requests.post")
    def test_reasoning_never_yielded_as_content(self, mock_post):
        mock_post.return_value = self._stream_response([
            {"reasoning_content": "because "},
            {"reasoning_content": "reasons"},
            {"content": "The "},
            {"content": "answer."},
        ])
        backend = _make_backend()
        pieces = list(backend.chat_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual("".join(pieces), "The answer.")
        self.assertEqual(backend.last_reasoning, "because reasons")
        # NEGATIVE: no piece of the yielded content stream carries reasoning or
        # an inline <think> tag - there is no downstream splitter for it here.
        for p in pieces:
            self.assertNotIn("because", p)
            self.assertNotIn("<think>", p)

    @patch("requests.post")
    def test_on_reasoning_callback_fires_per_delta(self, mock_post):
        mock_post.return_value = self._stream_response([
            {"reasoning_content": "step one. "},
            {"reasoning_content": "step two."},
            {"content": "done"},
        ])
        backend = _make_backend()
        seen = []
        pieces = list(backend.chat_stream(
            [{"role": "user", "content": "hi"}], on_reasoning=seen.append))
        self.assertEqual(seen, ["step one. ", "step two."])
        self.assertEqual("".join(pieces), "done")

    @patch("requests.post")
    def test_no_reasoning_field_leaves_last_reasoning_empty(self, mock_post):
        mock_post.return_value = self._stream_response([{"content": "hi"}])
        backend = _make_backend()
        list(backend.chat_stream([{"role": "user", "content": "x"}]))
        self.assertEqual(backend.last_reasoning, "")

    @patch("requests.post")
    def test_broken_on_reasoning_callback_does_not_kill_the_stream(self, mock_post):
        """A raising sink (mirrors agent/core.py's _emit contract: 'a broken
        sink must not kill the agent loop') must not abort the stream."""
        mock_post.return_value = self._stream_response([
            {"reasoning_content": "x"},
            {"content": "The answer."},
        ])
        backend = _make_backend()

        def _boom(_piece):
            raise RuntimeError("sink exploded")

        pieces = list(backend.chat_stream(
            [{"role": "user", "content": "hi"}], on_reasoning=_boom))
        self.assertEqual("".join(pieces), "The answer.")

    @patch("requests.post")
    def test_reasoning_does_not_disturb_tool_call_accumulation(self, mock_post):
        """Reasoning deltas interleaved with native tool_call deltas must not
        perturb the existing tool-call XML accumulation."""
        mock_post.return_value = self._stream_response([
            {"reasoning_content": "planning the call"},
            {"tool_calls": [{"index": 0, "function": {"name": "read_file", "arguments": ""}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": '{"path": "a.py"}'}}]},
        ])
        backend = _make_backend()
        pieces = list(backend.chat_stream([{"role": "user", "content": "hi"}]))
        full = "".join(pieces)
        self.assertIn('"name": "read_file"', full)
        self.assertIn('"path": "a.py"', full)
        self.assertNotIn("planning the call", full)


if __name__ == "__main__":
    unittest.main()
