# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.plugins.coder.prompts - family detection and per-family prompt content."""

import pytest
from pathlib import Path

from localm.plugins.coder.prompts import (
    detect_model_family,
    build_system_prompt,
    build_subagent_system_prompt,
    _prependable_leaf,
)


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
    examples) and still lists every tool - the docs are generated from
    TOOL_REGISTRY."""
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


# ---------------------------------------------------------------------------
#  Untrusted-content rule (provenance tagging)
# ---------------------------------------------------------------------------

def test_untrusted_content_rule_present_by_default():
    p = _prompt("llama3-8b")
    assert "UNTRUSTED CONTENT" in p
    assert "untrusted-external" in p
    assert "not as instructions" in p or "not instructions" in p.lower() \
        or "INFORMATION ONLY" in p


def test_untrusted_content_rule_omitted_when_disabled():
    p = build_system_prompt(CWD, model_name="llama3-8b",
                            untrusted_provenance=False)
    assert "UNTRUSTED CONTENT" not in p
    assert "untrusted-external" not in p
    # The rest of the prompt is intact.
    assert "AVAILABLE TOOLS" in p and "RULES" in p


def test_untrusted_content_rule_in_small_family():
    p = _prompt("phi3-mini")
    assert "UNTRUSTED CONTENT" in p
    assert "untrusted-external" in p


# ---------------------------------------------------------------------------
#  The clause telling the model not to repeat the folder name it was shown
# ---------------------------------------------------------------------------

def _patch_home(monkeypatch, path):
    """Point Path.home() at *path*. USERPROFILE is what expanduser reads on
    Windows, HOME on POSIX; set both so the test is not platform-conditional."""
    monkeypatch.setenv("USERPROFILE", str(path))
    monkeypatch.setenv("HOME", str(path))
    monkeypatch.delenv("HOMEDRIVE", raising=False)
    monkeypatch.delenv("HOMEPATH", raising=False)


def test_clause_names_the_folder_the_model_was_shown(tmp_path):
    """A project outside the user's home is shown as a bare folder name, which
    is the case the clause exists for: it must name that exact folder."""
    proj = tmp_path / "myproj"
    proj.mkdir()
    p = build_system_prompt(proj, model_name="llama3-8b")
    assert 'do not repeat "myproj"' in p
    assert '"file.py" not "myproj/file.py"' in p


def test_clause_present_for_small_family_too(tmp_path):
    proj = tmp_path / "myproj"
    proj.mkdir()
    assert 'do not repeat "myproj"' in build_system_prompt(proj, model_name="phi3-mini")


def test_cwd_at_home_does_not_leak_the_account_name(monkeypatch, tmp_path):
    """cwd == the user's home directory. _display_cwd renders that as "~/.",
    withholding the account name, and the clause must not put it back.

    Re-resolving the cwd independently would return Path(cwd).resolve().name,
    which for the home directory IS the account name."""
    home = tmp_path / "zz_account_name_zz"
    home.mkdir()
    _patch_home(monkeypatch, home)

    p = build_system_prompt(home, model_name="llama3-8b")

    assert "zz_account_name_zz" not in p
    assert "Working directory: ~/." in p


def test_subagent_brief_at_home_does_not_leak_the_account_name(monkeypatch, tmp_path):
    """Same property on the sub-agent brief, which builds the clause the same
    way."""
    home = tmp_path / "zz_account_name_zz"
    home.mkdir()
    _patch_home(monkeypatch, home)

    brief = build_subagent_system_prompt(home, role="explorer")

    assert "zz_account_name_zz" not in brief


def test_subagent_brief_names_the_folder_shown(tmp_path):
    proj = tmp_path / "myproj"
    proj.mkdir()
    brief = build_subagent_system_prompt(proj, role="explorer")
    assert 'do not repeat "myproj"' in brief


@pytest.mark.parametrize("shown,expected", [
    ("~/projects/proj", "proj"),   # under home: the leaf is a real folder
    ("proj",            "proj"),   # outside home: shown as a bare name
    ("~/.",             None),     # home itself: nothing to prepend, say nothing
    ("",                None),
    (".",               None),
    ("..",              None),
    ("D:\\",            None),     # bare drive root
    ("/",               None),
])
def test_prependable_leaf_only_returns_a_plain_folder_name(shown, expected):
    assert _prependable_leaf(shown) == expected


def test_prependable_leaf_guards_a_bare_tilde_even_though_unreachable_today(monkeypatch):
    """_display_cwd never renders a bare "~" today - the home directory comes
    out as "~/." - so this branch is unreachable via any current caller. It is
    guarded anyway, against a future rendering that collapses "~/." to "~".

    Driven by monkeypatching _display_cwd to return "~": without "~" in
    _prependable_leaf's no-clause tuple, THIS test fails and the clause
    appears."""
    assert _prependable_leaf("~") is None

    monkeypatch.setattr(
        "localm.plugins.coder.prompts._display_cwd", lambda cwd: "~")
    p = build_system_prompt(Path("/anything"), model_name="llama3-8b")
    assert "do not repeat" not in p, (
        'a simulated _display_cwd("~") rendering leaked a clause naming "~"')


def test_clause_names_the_leaf_for_a_project_under_home(monkeypatch, tmp_path):
    """An end-to-end prompt for a cwd UNDER the home directory: the clause must
    name the right leaf."""
    home = tmp_path / "zz_account_name_zz"
    home.mkdir()
    _patch_home(monkeypatch, home)
    proj = home / "projects" / "app"
    proj.mkdir(parents=True)

    p = build_system_prompt(proj, model_name="llama3-8b")
    assert 'do not repeat "app"' in p
    assert '"file.py" not "app/file.py"' in p
    assert "zz_account_name_zz" not in p
