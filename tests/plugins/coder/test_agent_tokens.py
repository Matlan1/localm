# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for agent token accumulation (Phase 1.2) and memory integration (Phase 1.3)."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock


def _make_mock_llm_backend(stream_tokens=None, usage=None, model_id="test-model"):
    """
    Return a mock backend that:
    - streams the given tokens on chat_stream()
    - returns concatenation on chat()
    - exposes last_usage; pre-populated with *usage* so _accumulate_usage()
      works even when called without iterating chat_stream() first
    """
    backend = MagicMock()
    backend.model_id = model_id
    _resolved_usage = usage or {}

    def _chat_stream(messages, **kwargs):
        for tok in (stream_tokens or ["Hello"]):
            yield tok
        backend._last_usage = _resolved_usage

    def _chat(messages, **kwargs):
        backend._last_usage = _resolved_usage
        return "".join(stream_tokens or ["Hello"])

    backend.chat_stream.side_effect = _chat_stream
    backend.chat.side_effect = _chat
    # Pre-populate so direct _accumulate_usage() calls work in unit tests
    backend._last_usage = _resolved_usage

    # Simulate the last_usage property (returns a copy, like the real backend)
    type(backend).last_usage = property(lambda self: dict(self._last_usage))
    return backend


def _make_agent(backend, tmp_path):
    from localm.plugins.coder.agent import Agent
    return Agent(backend=backend, cwd=tmp_path, name="test", max_turns=5)


class TestAgentTokenAccumulation(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())

    def test_initial_total_tokens_zero(self):
        backend = _make_mock_llm_backend()
        agent = _make_agent(backend, self.tmp)
        self.assertEqual(agent.total_tokens, 0)

    def test_accumulate_usage_after_streaming_call(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        backend = _make_mock_llm_backend(usage=usage)
        agent = _make_agent(backend, self.tmp)
        agent._accumulate_usage()
        self.assertEqual(agent.total_tokens, 15)

    def test_accumulate_usage_adds_across_calls(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        backend = _make_mock_llm_backend(usage=usage)
        agent = _make_agent(backend, self.tmp)
        agent._accumulate_usage()
        agent._accumulate_usage()
        self.assertEqual(agent.total_tokens, 30)

    def test_accumulate_usage_zero_when_no_usage(self):
        backend = _make_mock_llm_backend(usage={})
        agent = _make_agent(backend, self.tmp)
        agent._accumulate_usage()
        self.assertEqual(agent.total_tokens, 0)

    def test_accumulate_usage_zero_total_tokens_key(self):
        """total_tokens=0 should not be added (guard against dummy responses)."""
        backend = _make_mock_llm_backend(usage={"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 0})
        agent = _make_agent(backend, self.tmp)
        agent._accumulate_usage()
        self.assertEqual(agent.total_tokens, 0)

    def test_reset_clears_total_tokens(self):
        usage = {"total_tokens": 100}
        backend = _make_mock_llm_backend(usage=usage)
        agent = _make_agent(backend, self.tmp)
        agent._accumulate_usage()
        self.assertEqual(agent.total_tokens, 100)
        agent.reset()
        self.assertEqual(agent.total_tokens, 0)

    def test_reset_also_clears_turns(self):
        backend = _make_mock_llm_backend()
        agent = _make_agent(backend, self.tmp)
        agent._turns = 5
        agent._total_tokens = 200
        agent.reset()
        self.assertEqual(agent.turns, 0)
        self.assertEqual(agent.total_tokens, 0)


class TestAgentMemoryIntegration(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())

    def test_memory_loaded_at_init(self):
        mem_file = self.tmp / "LOCALCODER.md"
        mem_file.write_text("# Memory\n\n- use ruff\n", encoding="utf-8")
        backend = _make_mock_llm_backend()
        agent = _make_agent(backend, self.tmp)
        self.assertIn("use ruff", agent._memory)

    def test_memory_in_system_prompt(self):
        mem_file = self.tmp / "LOCALCODER.md"
        mem_file.write_text("# Memory\n\n- always type hint\n", encoding="utf-8")
        backend = _make_mock_llm_backend()
        agent = _make_agent(backend, self.tmp)
        self.assertIn("always type hint", agent._system_prompt)

    def test_no_memory_when_file_absent(self):
        backend = _make_mock_llm_backend()
        agent = _make_agent(backend, self.tmp)
        self.assertEqual(agent._memory, "")
        self.assertNotIn("Project Memory", agent._system_prompt)

    def test_remember_updates_system_prompt(self):
        backend = _make_mock_llm_backend()
        agent = _make_agent(backend, self.tmp)
        agent.remember("prefer pathlib over os.path")
        self.assertIn("prefer pathlib", agent._system_prompt)

    def test_forget_updates_system_prompt(self):
        mem_file = self.tmp / "LOCALCODER.md"
        mem_file.write_text("# Memory\n\n- use ruff\n- use mypy\n", encoding="utf-8")
        backend = _make_mock_llm_backend()
        agent = _make_agent(backend, self.tmp)
        agent.forget("ruff")
        self.assertNotIn("use ruff", agent._system_prompt)
        self.assertIn("use mypy", agent._system_prompt)

    def test_set_cwd_reloads_memory(self):
        import tempfile
        other_tmp = Path(tempfile.mkdtemp())
        (other_tmp / "LOCALCODER.md").write_text("- new project note\n", encoding="utf-8")
        backend = _make_mock_llm_backend()
        agent = _make_agent(backend, self.tmp)
        self.assertEqual(agent._memory, "")
        agent.set_cwd(other_tmp)
        self.assertIn("new project note", agent._memory)


class TestAgentModelFamily(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())

    def test_gemma_model_gets_native_format_hint(self):
        backend = _make_mock_llm_backend(model_id="gemma4-12b")
        agent = _make_agent(backend, self.tmp)
        self.assertIn("<|tool_call>", agent._system_prompt)

    def test_thinking_model_gets_reasoning_section(self):
        backend = _make_mock_llm_backend(model_id="deepseek-r1-7b")
        agent = _make_agent(backend, self.tmp)
        self.assertIn("REASONING", agent._system_prompt)

    def test_small_model_gets_condensed_prompt(self):
        backend_small   = _make_mock_llm_backend(model_id="phi3-mini")
        backend_default = _make_mock_llm_backend(model_id="llama3-8b")
        agent_small   = _make_agent(backend_small, self.tmp)
        agent_default = _make_agent(backend_default, self.tmp)
        self.assertLess(len(agent_small._system_prompt), len(agent_default._system_prompt))


if __name__ == "__main__":
    unittest.main()
