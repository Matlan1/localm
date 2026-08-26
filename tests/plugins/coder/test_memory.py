# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.plugins.coder.memory"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.coder.memory import (
    _MAX_CUSTOM_INSTRUCTIONS_CHARS,
    _MAX_MEMORY_CHARS,
    cap_user_instructions,
    custom_instructions_warning,
    find_memory_file,
    load_custom_instructions,
    load_memory,
    memory_warning,
    remember,
    forget,
    default_memory_file,
)


def test_find_memory_file_none(tmp_path):
    assert find_memory_file(tmp_path) is None


def test_find_memory_file_localcoder_md(tmp_path):
    p = tmp_path / "LOCALCODER.md"
    p.write_text("# mem\n- foo\n")
    assert find_memory_file(tmp_path) == p


def test_find_memory_file_dot_localcoder(tmp_path):
    d = tmp_path / ".localcoder"
    d.mkdir()
    p = d / "memory.md"
    p.write_text("- bar\n")
    assert find_memory_file(tmp_path) == p


def test_find_memory_file_prefers_localcoder_md(tmp_path):
    """LOCALCODER.md takes priority over .localcoder/memory.md."""
    top = tmp_path / "LOCALCODER.md"
    top.write_text("# top\n")
    d = tmp_path / ".localcoder"
    d.mkdir()
    nested = d / "memory.md"
    nested.write_text("# nested\n")
    assert find_memory_file(tmp_path) == top


def test_load_memory_empty(tmp_path):
    assert load_memory(tmp_path) == ""


def test_load_memory_returns_content(tmp_path):
    p = tmp_path / "LOCALCODER.md"
    p.write_text("# Project Memory\n\n- Use ruff for linting\n")
    result = load_memory(tmp_path)
    assert "Use ruff for linting" in result


def test_remember_creates_file(tmp_path):
    p = remember(tmp_path, "always write tests")
    assert p.exists()
    content = p.read_text()
    assert "- always write tests" in content
    assert "# Project Memory" in content


def test_remember_appends(tmp_path):
    remember(tmp_path, "first note")
    remember(tmp_path, "second note")
    content = (tmp_path / "LOCALCODER.md").read_text()
    assert "- first note" in content
    assert "- second note" in content


def test_remember_no_duplicate(tmp_path):
    remember(tmp_path, "same note")
    remember(tmp_path, "same note")
    content = (tmp_path / "LOCALCODER.md").read_text()
    assert content.count("- same note") == 1


def test_remember_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        remember(tmp_path, "   ")


def test_forget_removes_matching(tmp_path):
    p = tmp_path / "LOCALCODER.md"
    p.write_text("# Memory\n\n- keep this\n- remove me\n- also keep\n")
    path, n = forget(tmp_path, "remove me")
    assert n == 1
    content = p.read_text()
    assert "remove me" not in content
    assert "keep this" in content
    assert "also keep" in content


def test_forget_case_insensitive(tmp_path):
    p = tmp_path / "LOCALCODER.md"
    p.write_text("# Memory\n\n- Use RUFF for linting\n")
    path, n = forget(tmp_path, "ruff")
    assert n == 1


def test_forget_no_match(tmp_path):
    p = tmp_path / "LOCALCODER.md"
    p.write_text("# Memory\n\n- some note\n")
    path, n = forget(tmp_path, "notpresent")
    assert n == 0
    assert "some note" in p.read_text()


def test_forget_preserves_headers(tmp_path):
    p = tmp_path / "LOCALCODER.md"
    p.write_text("# Project Memory\n\n- remove this\n")
    forget(tmp_path, "remove this")
    content = p.read_text()
    assert "# Project Memory" in content


def test_forget_no_file(tmp_path):
    path, n = forget(tmp_path, "anything")
    assert path is None
    assert n == 0


def test_default_memory_file(tmp_path):
    assert default_memory_file(tmp_path) == tmp_path / "LOCALCODER.md"


# --------------------------------------------------------------------------- #
#  Injection budget: project memory and user instructions are capped, with an
#  in-band notice to the model and a warning naming the file.
# --------------------------------------------------------------------------- #

def _oversized(n: int = _MAX_MEMORY_CHARS * 2) -> str:
    """A memory file comfortably over the budget, as realistic bullets."""
    bullet = "- the build uses ruff, pytest, and npm for the frontend tests\n"
    return "# Project Memory\n\n" + bullet * (n // len(bullet) + 1)


def test_normal_memory_is_injected_verbatim(tmp_path):
    """The common case must be byte-identical to no capping at all."""
    body = "# Project Memory\n\n- Use ruff for linting\n- Prefer pathlib"
    (tmp_path / "LOCALCODER.md").write_text(body, encoding="utf-8")

    assert load_memory(tmp_path) == body.strip()
    assert memory_warning(tmp_path) == ""


def test_normal_memory_right_at_the_limit_is_untouched(tmp_path):
    """Boundary: exactly _MAX_MEMORY_CHARS fits, and is not annotated."""
    body = "x" * _MAX_MEMORY_CHARS
    (tmp_path / "LOCALCODER.md").write_text(body, encoding="utf-8")

    assert load_memory(tmp_path) == body
    assert memory_warning(tmp_path) == ""


def test_oversized_memory_is_capped(tmp_path):
    raw = _oversized()
    (tmp_path / "LOCALCODER.md").write_text(raw, encoding="utf-8")

    out = load_memory(tmp_path)

    # Bounded: the whole file did NOT go into the prompt.
    assert len(out) < len(raw)
    # The budget is respected apart from the appended notice.
    assert len(out) <= _MAX_MEMORY_CHARS + 200
    # It still starts with the real content, so the useful part survives.
    assert out.startswith("# Project Memory")


def test_oversized_memory_says_so_in_band(tmp_path):
    """The MODEL must be able to tell it is reading a partial file."""
    (tmp_path / "LOCALCODER.md").write_text(_oversized(), encoding="utf-8")

    out = load_memory(tmp_path)

    assert "omitted" in out
    assert "project memory" in out
    assert str(_MAX_MEMORY_CHARS) in out


def test_oversized_memory_warns_the_user(tmp_path):
    """The HUMAN must be told, with the numbers and a way to fix it."""
    raw = _oversized()
    (tmp_path / "LOCALCODER.md").write_text(raw, encoding="utf-8")

    warning = memory_warning(tmp_path)
    injected = load_memory(tmp_path)
    kept = injected.split("\n\n[...")[0]

    assert "LOCALCODER.md" in warning
    assert str(len(raw.strip())) in warning                   # the real size
    assert str(len(raw.strip()) - len(kept)) in warning       # the overflow
    assert "/forget" in warning                               # the remedy


def test_notice_and_warning_agree_on_the_number(tmp_path):
    """Two honest numbers that disagree still erode trust, so the in-band notice
    and the user warning must derive the omitted count from the same split."""
    raw = _oversized()
    (tmp_path / "LOCALCODER.md").write_text(raw, encoding="utf-8")

    injected = load_memory(tmp_path)
    kept = injected.split("\n\n[...")[0]
    omitted = len(raw.strip()) - len(kept)

    assert f"{omitted} characters of project memory omitted" in injected
    assert f"({omitted} omitted)" in memory_warning(tmp_path)


def test_capping_keeps_whole_lines(tmp_path):
    """A cap that slices mid-word can invert a bullet's meaning, so cut on a
    line boundary when one is available."""
    (tmp_path / "LOCALCODER.md").write_text(_oversized(), encoding="utf-8")

    body = load_memory(tmp_path).split("\n\n[...")[0]

    assert body.endswith("frontend tests")


def test_unreadable_memory_file_is_reported_not_swallowed(tmp_path):
    """A file that EXISTS but cannot be read must not look like "no memory":
    absent and corrupt must not collapse into one silent path."""
    (tmp_path / "LOCALCODER.md").write_text("- something important\n", encoding="utf-8")

    def boom(*a, **kw):
        raise OSError("permission denied")

    with patch.object(Path, "read_text", boom):
        assert load_memory(tmp_path) == ""        # degrades, does not crash
        assert "could not be read" in memory_warning(tmp_path)


def test_absent_memory_file_is_silent(tmp_path):
    """The negative half of the above: absent is normal and says nothing."""
    assert load_memory(tmp_path) == ""
    assert memory_warning(tmp_path) == ""


# ---- the same contract for user instructions (.localcoder/system.md) ------- #

def _write_instructions(tmp_path, body: str):
    d = tmp_path / ".localcoder"
    d.mkdir(exist_ok=True)
    (d / "system.md").write_text(body, encoding="utf-8")


def test_normal_instructions_are_injected_verbatim(tmp_path):
    body = "Always run pytest before finishing."
    _write_instructions(tmp_path, body)

    assert load_custom_instructions(tmp_path) == body
    assert custom_instructions_warning(tmp_path) == ""


def test_oversized_instructions_are_capped_and_reported(tmp_path):
    raw = "Always run pytest before finishing.\n" * 200
    _write_instructions(tmp_path, raw)

    out = load_custom_instructions(tmp_path)
    warning = custom_instructions_warning(tmp_path)

    assert len(out) < len(raw)
    assert len(out) <= _MAX_CUSTOM_INSTRUCTIONS_CHARS + 200
    assert "omitted" in out and "user instructions" in out
    assert "system.md" in warning
    assert str(len(raw.strip()) - len(out.split("\n\n[...")[0])) in warning


def test_absent_instructions_are_silent(tmp_path):
    assert load_custom_instructions(tmp_path) == ""
    assert custom_instructions_warning(tmp_path) == ""


def test_system_flag_override_is_capped_too(tmp_path):
    """--system bypasses the file loader, so it needs the same bound or the
    documented way to set instructions stays unbounded."""
    raw = "Do the thing carefully.\n" * 300

    out = cap_user_instructions(raw)
    warning = custom_instructions_warning(tmp_path, raw)

    assert len(out) <= _MAX_CUSTOM_INSTRUCTIONS_CHARS + 200
    assert "omitted" in out
    assert "--system" in warning
    assert str(len(raw.strip()) - len(out.split("\n\n[...")[0])) in warning


def test_normal_system_flag_override_is_untouched(tmp_path):
    text = "Always run pytest before finishing."
    assert cap_user_instructions(text) == text
    assert custom_instructions_warning(tmp_path, text) == ""


# --------------------------------------------------------------------------- #
#  End-to-end through a real Agent: the capped text is injected into the
#  prompt it really builds, and the warning reaches the human.
# --------------------------------------------------------------------------- #

def _make_agent(tmp_path):
    """A real Agent over *tmp_path*; returns (agent, print_warning mock)."""
    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = "test-model"
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.print_warning") as warn:
        MockPM.build.return_value.file_count.return_value = 0
        agent = Agent(backend=backend, cwd=tmp_path)
    return agent, warn


def _memory_warnings(warn) -> str:
    """Just the print_warning calls that are about the memory file.

    Agent startup legitimately warns about other things (MCP servers, plugins,
    skills that failed to register), so asserting on "no warnings at all" would
    be testing the wrong thing and would break for unrelated reasons.
    """
    return " ".join(str(c) for c in warn.call_args_list
                    if "LOCALCODER.md" in str(c) or "system.md" in str(c))


def test_agent_prompt_is_bounded_and_user_is_warned(tmp_path):
    raw = _oversized()
    (tmp_path / "LOCALCODER.md").write_text(raw, encoding="utf-8")

    agent, warn = _make_agent(tmp_path)
    prompt = agent._system_prompt

    # The whole file did NOT reach the prompt - only the capped version did.
    assert raw.strip() not in prompt
    injected = load_memory(tmp_path)
    assert injected in prompt
    assert len(injected) <= _MAX_MEMORY_CHARS + 200
    # ...the model is told the memory it is reading is partial...
    assert "characters of project memory omitted" in prompt
    # ...and the human is told which file to fix.
    assert "omitted" in _memory_warnings(warn)


def test_agent_injects_normal_memory_whole_and_stays_quiet(tmp_path):
    """The common case: no truncation, no notice, no warning."""
    body = "# Project Memory\n\n- Use ruff for linting\n"
    (tmp_path / "LOCALCODER.md").write_text(body, encoding="utf-8")

    agent, warn = _make_agent(tmp_path)

    assert body.strip() in agent._system_prompt      # verbatim, not merely present
    assert "characters of project memory omitted" not in agent._system_prompt
    assert _memory_warnings(warn) == ""


def test_agent_warns_once_more_after_remember_pushes_it_over(tmp_path):
    """/remember is how the file grows, so the warning must fire on that path
    too - not only at session start."""
    (tmp_path / "LOCALCODER.md").write_text("- small\n", encoding="utf-8")
    agent, warn = _make_agent(tmp_path)
    assert _memory_warnings(warn) == ""       # quiet while it is small

    with patch("localm.plugins.coder.agent.print_warning") as warn2:
        agent.remember("x" * (_MAX_MEMORY_CHARS + 500))

    assert "omitted" in _memory_warnings(warn2)
