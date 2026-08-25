# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-model harness profiles: a weak/small model and a long-reasoning model run
with gen-kwarg defaults tuned to them, while an explicit caller value always wins.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from localm.plugins.coder.harness_profiles import (
    CLI_DEFAULT_MAX_TOKENS,
    agent_gen_overrides,
    cli_max_tokens,
    family_profile,
)


# --------------------------------------------------------------------------- #
#  Pure profile lookups                                                        #
# --------------------------------------------------------------------------- #

def test_family_profile_small_and_thinking():
    assert family_profile("phi3-mini") == {"temperature": 0.3}
    assert family_profile("deepseek-r1-7b") == {"max_tokens": 4096}


def test_family_profile_baseline_families_have_no_overrides():
    assert family_profile("llama3.1-8b") == {}
    assert family_profile("gemma4-12b") == {}
    assert family_profile("") == {}            # unknown/empty -> default family


def test_agent_gen_overrides_only_universal_safe_keys():
    # temperature is safe to fill anywhere...
    assert agent_gen_overrides("phi3-mini") == {"temperature": 0.3}
    # ...but max_tokens is NOT filled by the Agent path.
    assert "max_tokens" not in agent_gen_overrides("deepseek-r1-7b")
    assert agent_gen_overrides("deepseek-r1-7b") == {}
    assert agent_gen_overrides("llama3-8b") == {}


def test_cli_max_tokens_thinking_gets_more_room():
    assert cli_max_tokens("deepseek-r1-7b") == 4096
    assert cli_max_tokens("qwq-32b") == 4096


def test_cli_max_tokens_baseline_for_other_families():
    assert cli_max_tokens("llama3-8b") == CLI_DEFAULT_MAX_TOKENS
    assert cli_max_tokens("phi3-mini") == CLI_DEFAULT_MAX_TOKENS   # small has no max_tokens
    assert cli_max_tokens("") == CLI_DEFAULT_MAX_TOKENS
    assert cli_max_tokens(None) == CLI_DEFAULT_MAX_TOKENS          # type: ignore[arg-type]


def test_cli_max_tokens_respects_an_explicit_baseline_arg():
    assert cli_max_tokens("llama3-8b", baseline=999) == 999
    # A thinking override still wins over a passed baseline.
    assert cli_max_tokens("qwq-32b", baseline=999) == 4096


# --------------------------------------------------------------------------- #
#  Agent integration: the profile fills gen kwargs, explicit values win        #
# --------------------------------------------------------------------------- #

def _make_agent(tmp_path: Path, model_id: str, **kwargs):
    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = model_id
    backend.native_tools = False
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        return Agent(backend=backend, cwd=tmp_path, **kwargs)


def test_small_model_gets_steadier_temperature(tmp_path):
    agent = _make_agent(tmp_path, "phi3-mini")
    assert agent.gen_kwargs.get("temperature") == 0.3


def test_explicit_temperature_wins_over_profile(tmp_path):
    agent = _make_agent(tmp_path, "phi3-mini", temperature=0.9)
    assert agent.gen_kwargs["temperature"] == 0.9


def test_default_model_gets_no_temperature_injected(tmp_path):
    agent = _make_agent(tmp_path, "llama3.1-8b")
    assert "temperature" not in agent.gen_kwargs


def test_agent_does_not_inject_max_tokens(tmp_path):
    # Even a thinking model gets no max_tokens from the Agent path (CLI owns that).
    agent = _make_agent(tmp_path, "deepseek-r1-7b")
    assert "max_tokens" not in agent.gen_kwargs
