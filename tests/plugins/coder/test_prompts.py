# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.plugins.coder.prompts - family detection and per-family prompt content."""

import pytest
from pathlib import Path

from localm.plugins.coder.prompts import detect_model_family, build_system_prompt


# ---------------------------------------------------------------------------
#  Family detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    # Gemma
    ("gemma4-12b-it",    "gemma"),
    ("gemma-2-9b-it",    "gemma"),
    ("gemma3",           "gemma"),
    ("Gemma4-4B",        "gemma"),
    # Thinking models
    ("deepseek-r1-7b",   "thinking"),
    ("deepseek-r1",      "thinking"),
    ("qwq-32b",          "thinking"),
    ("qwen3-8b",         "thinking"),
    # Descriptively-named reasoning fine-tunes (broadened markers)
    ("Llama-3.3-8B-Instruct-Thinking-High-Reasoning", "thinking"),
    ("mistral-nemo-12b-thinking",                     "thinking"),
    ("some-model-reasoning-v2",                        "thinking"),
    ("magistral-small",                                "thinking"),
    # Small / phi
    ("phi3-mini-4k",     "small"),
    ("phi-4-mini",       "small"),
    ("phi2",             "small"),
    ("llama-tiny",       "small"),
    # Default
    ("llama3.1-8b",      "default"),
    ("mistral-7b",       "default"),
    ("qwen2.5-7b",       "default"),
    ("codellama-13b",    "default"),
    ("",                 "default"),
])
def test_detect_model_family(name, expected):
    assert detect_model_family(name) == expected


# ---------------------------------------------------------------------------
#  Prompt content sanity checks per family
# ---------------------------------------------------------------------------

CWD = Path("/tmp/myproject")


def _prompt(model_name=""):
    return build_system_prompt(CWD, model_name=model_name)


def test_default_prompt_contains_xml_format():
    p = _prompt("llama3-8b")
    assert "<tool_call>" in p
    assert '{"name": "TOOL_NAME"' in p
    assert "AVAILABLE TOOLS" in p
    assert "RULES" in p


def test_default_prompt_full_tool_list():
    p = _prompt("llama3-8b")
    # All tools should be present
    for tool in ("read_file", "write_file", "edit_file", "patch_file",
                 "run_shell", "list_dir", "search_files", "grep",
                 "spawn_agent", "generate_image"):
        assert tool in p, f"Tool '{tool}' missing from default prompt"


def test_gemma_prompt_mentions_native_format():
    p = _prompt("gemma4-12b")
    assert "<|tool_call>" in p or "native" in p.lower() or "gemma" in p.lower()
    # XML should still be mentioned as primary
    assert "<tool_call>" in p


def test_gemma_prompt_has_full_tool_list():
    p = _prompt("gemma4-12b")
    for tool in ("read_file", "write_file", "edit_file", "patch_file"):
        assert tool in p


def test_thinking_prompt_has_think_hint():
    p = _prompt("deepseek-r1-7b")
    assert "<think>" in p or "think" in p.lower()
    # Still has XML tool format
    assert "<tool_call>" in p


def test_thinking_prompt_reasoning_section():
    p = _prompt("qwq-32b")
    assert "REASONING" in p or "reasoning" in p.lower()


def test_small_prompt_is_shorter_than_default():
    p_default = _prompt("llama3-8b")
    p_small   = _prompt("phi3-mini")
    assert len(p_small) < len(p_default), (
        f"Small prompt ({len(p_small)}) should be shorter than default ({len(p_default)})"
    )


def test_small_prompt_has_condensed_rules():
    p = _prompt("phi3-mini")
    # Should have fewer rules (≤5)
    rule_lines = [l for l in p.splitlines() if l.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8."))]
    assert len(rule_lines) <= 5, f"Expected ≤5 rules in small prompt, got {len(rule_lines)}"


def test_small_prompt_lists_every_tool():
    """The small-model list is condensed (one line per tool, no JSON
    examples) but no longer omits tools - the docs are generated from
    TOOL_REGISTRY so models know everything that is callable."""
    from localm.plugins.coder.tools import TOOL_REGISTRY
    p = _prompt("phi3-mini")
    for tool in TOOL_REGISTRY:
        assert tool in p, f"Tool '{tool}' missing from small prompt"


def test_memory_injected():
    p = build_system_prompt(CWD, memory="- always use ruff\n- run tests first")
    assert "Project Memory" in p
    assert "always use ruff" in p


def test_no_memory_when_empty():
    p = _prompt("llama3-8b")
    assert "Project Memory" not in p


def test_model_name_empty_uses_default():
    p_no_name = _prompt("")
    p_default = _prompt("llama3-8b")
    # Both should use the XML format and have the same structural sections
    for section in ("TOOL USE", "AVAILABLE TOOLS", "RULES"):
        assert section in p_no_name
        assert section in p_default
