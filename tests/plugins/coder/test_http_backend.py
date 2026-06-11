"""Tests for localm.plugins.coder.backends.http — usage capture from responses."""

import json
import unittest
from unittest.mock import MagicMock, patch

from localm.plugins.coder.backends.http import HTTPBackend


def _make_backend():
    return HTTPBackend("http://127.0.0.1:8080/v1", "test-model")


def _mock_non_streaming_response(content: str, usage: dict):
    """Build a mock requests.Response for a non-streaming completion."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
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

    @patch("requests.post")
    def test_stream_empty_usage_when_not_provided(self, mock_post):
        """If the server omits usage, last_usage should be empty dict (not crash)."""
        mock_post.return_value = self._make_stream_response(["hello"], {})
        backend = _make_backend()
        list(backend.chat_stream([{"role": "user", "content": "hi"}]))
        # {} means the server didn't send usage — that's fine
        self.assertIsInstance(backend.last_usage, dict)


if __name__ == "__main__":
    unittest.main()
