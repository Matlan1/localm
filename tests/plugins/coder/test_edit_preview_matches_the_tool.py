# SPDX-License-Identifier: AGPL-3.0-or-later
"""The edit_file preview (approval diff, patch-mode capture) is computed by the
same resolution the tool applies, so what a person approves is what is written."""

from __future__ import annotations

import pytest

from localm.plugins.coder.diffutil import compute_tool_diff, resolve_new_content
from localm.plugins.coder.tools.files import tool_edit_file

_PY = "def f():\n    if x:\n        return 1\n    return 2\n"

CASES = [
    # exact match
    ("a.py", _PY, "        return 1\n", "        return 3\n"),
    # whitespace-tolerant match (old written with other indentation)
    ("a.py", _PY, "if x:\n  return 1", "if x:\n        return 5"),
    # a multi-line new re-indented to keep the file parsing
    ("a.py", _PY, "return 2", "y = 1\nreturn y"),
    # no match
    ("a.py", _PY, "not in the file", "z"),
    # empty old
    ("a.txt", "one\ntwo\n", "", "zero\n"),
    # an edit that breaks the syntax even re-indented
    ("a.py", _PY, "    return 2\n", "    return (\n"),
]


@pytest.mark.parametrize("name, text, old, new", CASES)
def test_preview_is_what_the_tool_writes(tmp_path, name, text, old, new):
    target = tmp_path / name
    target.write_text(text, encoding="utf-8")
    args = {"path": name, "old": old, "new": new}
    planned = resolve_new_content("edit_file", args, text)

    res = tool_edit_file(tmp_path, name, old, new)
    written = target.read_text(encoding="utf-8")

    if res.ok:
        assert written != text
    else:
        assert written == text
    assert planned == written


def test_whitespace_tolerant_edit_has_a_diff_for_the_approval_card(tmp_path):
    args = {"path": "a.py", "old": "if x:\n  return 1", "new": "if x:\n        return 5"}
    diff = compute_tool_diff("edit_file", args, _PY)
    assert diff is not None and "+        return 5" in diff


def test_empty_old_shows_no_change(tmp_path):
    args = {"path": "a.txt", "old": "", "new": "zero\n"}
    assert compute_tool_diff("edit_file", args, "one\n") is None
