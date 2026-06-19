# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.plugins.coder.memory"""

import pytest
from pathlib import Path

from localm.plugins.coder.memory import (
    find_memory_file,
    load_memory,
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
