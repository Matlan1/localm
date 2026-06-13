"""Tests for localm.inference.protocol - UsageInfo and ChatChunk/ChatResponse."""

import unittest

from localm.inference.protocol import (
    UsageInfo, ChatChunk, ChatResponse, FullChoice, Message, make_chunk_id,
)


class TestUsageInfo(unittest.TestCase):
    def test_defaults_are_zero(self):
        u = UsageInfo()
        self.assertEqual(u.prompt_tokens, 0)
        self.assertEqual(u.completion_tokens, 0)
        self.assertEqual(u.total_tokens, 0)

    def test_set_values(self):
        u = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        self.assertEqual(u.prompt_tokens, 10)
        self.assertEqual(u.completion_tokens, 5)
        self.assertEqual(u.total_tokens, 15)

    def test_serialises_to_dict(self):
        u = UsageInfo(prompt_tokens=3, completion_tokens=7, total_tokens=10)
        d = u.model_dump()
        self.assertEqual(d["prompt_tokens"], 3)
        self.assertEqual(d["completion_tokens"], 7)
        self.assertEqual(d["total_tokens"], 10)


class TestChatChunkDone(unittest.TestCase):
    def test_done_without_usage(self):
        chunk = ChatChunk.done("localm", "id123", 0)
        self.assertIsNone(chunk.usage)
        self.assertEqual(chunk.choices[0].finish_reason, "stop")

    def test_done_with_usage(self):
        usage = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunk = ChatChunk.done("localm", "id123", 0, usage=usage)
        self.assertIsNotNone(chunk.usage)
        self.assertEqual(chunk.usage.total_tokens, 15)

    def test_done_serialises_usage(self):
        usage = UsageInfo(prompt_tokens=8, completion_tokens=4, total_tokens=12)
        chunk = ChatChunk.done("localm", "id123", 0, usage=usage)
        payload = chunk.model_dump_json()
        self.assertIn('"total_tokens":12', payload.replace(" ", ""))

    def test_token_chunk_no_usage(self):
        chunk = ChatChunk.token("hello", "localm", "id123", 0)
        self.assertIsNone(chunk.usage)


class TestChatResponse(unittest.TestCase):
    def _make_response(self, usage=None):
        return ChatResponse(
            id=make_chunk_id(),
            created=0,
            model="localm",
            choices=[
                FullChoice(message=Message(role="assistant", content="hi"), finish_reason="stop")
            ],
            usage=usage,
        )

    def test_response_no_usage(self):
        resp = self._make_response()
        self.assertIsNone(resp.usage)

    def test_response_with_usage(self):
        usage = UsageInfo(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        resp = self._make_response(usage=usage)
        self.assertEqual(resp.usage.total_tokens, 30)

    def test_response_serialises(self):
        usage = UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        resp = self._make_response(usage=usage)
        d = resp.model_dump()
        self.assertEqual(d["usage"]["total_tokens"], 3)


if __name__ == "__main__":
    unittest.main()
