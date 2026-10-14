# SPDX-License-Identifier: AGPL-3.0-or-later
"""The coder's edit tools change only the edited line: every other line keeps
its bytes, line ending included, whatever mix of LF and CRLF the file holds."""

from __future__ import annotations

import pytest

from localm.plugins.coder.tools import (
    tool_edit_file, tool_edit_files, tool_patch_file, tool_search_replace,
)

FILES = {
    "lf": b"alpha\nbeta = 1\ngamma\n",
    "crlf": b"alpha\r\nbeta = 1\r\ngamma\r\n",
    "mixed": b"alpha\r\nbeta = 1\ngamma\r\ndelta\n",
}


def _expected(raw: bytes) -> bytes:
    return raw.replace(b"beta = 1", b"beta = 2")


def _edit_file(cwd, name):
    return tool_edit_file(cwd, name, "beta = 1", "beta = 2")


def _edit_files(cwd, name):
    return tool_edit_files(cwd, [{"path": name, "old": "beta = 1", "new": "beta = 2"}])


def _patch_file(cwd, name):
    return tool_patch_file(cwd, name, "@@ -2,1 +2,1 @@\n-beta = 1\n+beta = 2\n")


def _search_replace(cwd, name):
    return tool_search_replace(cwd, r"beta = 1", "beta = 2", glob=name)


@pytest.mark.parametrize("tool", [_edit_file, _edit_files, _patch_file, _search_replace],
                         ids=["edit_file", "edit_files", "patch_file", "search_replace"])
@pytest.mark.parametrize("kind", sorted(FILES))
def test_an_edit_keeps_every_other_byte(tmp_path, tool, kind):
    name = f"{kind}.txt"
    (tmp_path / name).write_bytes(FILES[kind])
    result = tool(tmp_path, name)
    assert result.ok, result.output
    assert (tmp_path / name).read_bytes() == _expected(FILES[kind])


def test_a_changed_line_in_a_crlf_file_gets_crlf(tmp_path):
    (tmp_path / "w.txt").write_bytes(b"one\r\ntwo\r\n")
    tool_edit_file(tmp_path, "w.txt", "two", "two\nthree")
    assert (tmp_path / "w.txt").read_bytes() == b"one\r\ntwo\r\nthree\r\n"
