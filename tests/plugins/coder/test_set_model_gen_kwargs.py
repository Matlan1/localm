# SPDX-License-Identifier: AGPL-3.0-or-later
"""Agent.set_model's gen_kwargs and grammar-latch handling across a switch.

A model switch must not let the OLD model's profile-filled gen kwargs (e.g.
temperature) survive as if they had been chosen for the NEW model, must
re-derive a CLI-managed max_tokens for the new model's family, must leave an
explicit caller value alone, and must clear the grammar-unsupported latches,
since those describe the backend that was loaded before the switch.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from localm.plugins.coder.harness_profiles import cli_max_tokens


class _StubBackend:
    native_tools = False
    supports_grammar = False
    last_usage: dict = {}
    last_reasoning = ""

    def __init__(self, model_id: str):
        self.model_id = model_id

    def set_model(self, model: str) -> None:
        self.model_id = model

    def chat_stream(self, messages, on_reasoning=None, **kwargs):
        yield "done"

    def set_tools(self, tool_defs):
        pass

    def context_capacity(self):
        return None


def _make_agent(tmp_path: Path, model: str = "test-model", **kwargs):
    from localm.plugins.coder.agent import Agent
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        return Agent(backend=_StubBackend(model), cwd=tmp_path, **kwargs)


class TestGenKwargsAcrossASwitch:
    def test_switching_to_a_no_profile_model_drops_the_old_models_temperature(
            self, tmp_path):
        agent = _make_agent(tmp_path, model="phi4-mini")
        assert agent.gen_kwargs.get("temperature") == 0.3

        agent.set_model("llama3.1-8b")

        assert "temperature" not in agent.gen_kwargs

    def test_switching_to_a_thinking_model_recomputes_derived_max_tokens(
            self, tmp_path):
        initial = cli_max_tokens("llama3.1-8b")
        agent = _make_agent(tmp_path, model="llama3.1-8b",
                            max_tokens_explicit=False, max_tokens=initial)
        assert agent.gen_kwargs["max_tokens"] == initial

        agent.set_model("qwen3-8b")

        assert agent.gen_kwargs["max_tokens"] == cli_max_tokens("qwen3-8b")

    def test_explicit_temperature_survives_a_model_switch(self, tmp_path):
        agent = _make_agent(tmp_path, model="phi4-mini", temperature=0.9)
        assert agent.gen_kwargs["temperature"] == 0.9

        agent.set_model("qwen3-8b")

        assert agent.gen_kwargs["temperature"] == 0.9

    def test_explicit_max_tokens_survives_a_model_switch(self, tmp_path):
        agent = _make_agent(tmp_path, model="llama3.1-8b",
                            max_tokens_explicit=True, max_tokens=777)

        agent.set_model("qwen3-8b")

        assert agent.gen_kwargs["max_tokens"] == 777

    def test_a_gui_style_agent_never_gains_a_max_tokens_key(self, tmp_path):
        """max_tokens_explicit defaults to None: an Agent built the GUI way
        (no gen kwargs at all) must not have set_model start injecting one,
        since the GUI relies on the backend's own default."""
        agent = _make_agent(tmp_path, model="llama3.1-8b")
        assert "max_tokens" not in agent.gen_kwargs

        agent.set_model("qwen3-8b")

        assert "max_tokens" not in agent.gen_kwargs


class TestGrammarLatchesResetOnSwitch:
    def test_set_model_resets_both_grammar_unsupported_latches(self, tmp_path):
        agent = _make_agent(tmp_path, model="llama3.1-8b")
        agent._grammar_confirmed_unsupported = True
        agent._lazy_grammar_confirmed_unsupported = True

        agent.set_model("qwen3-8b")

        assert agent._grammar_confirmed_unsupported is False
        assert agent._lazy_grammar_confirmed_unsupported is False
