# SPDX-License-Identifier: AGPL-3.0-or-later
"""A symlinked FILE inside an added folder must still be indexed.

_walk_files replaces rglob('*') because rglob follows NTFS junctions, which
report is_symlink()==False, so pathlib's own loop guard misses them and a
self-referential junction spins until the path overflows. That guard must not
extend to a symlink resolving to a plain FILE: a docs folder that aggregates
links to files elsewhere (docs/spec.pdf -> /projects/x/spec.pdf) is a common
layout, and dropping those documents is invisible to the user.

A cycle needs a DIRECTORY to recurse into, so following a link to a file cannot
loop. Escape is handled a layer up: _expand's confine loop rejects symlinks
escaping an allowed folder when a policy is given, and with no policy (the local
CLI) the operator is unconfined.

These tests need a real symlink on disk; a mock-based test stays green either
way.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from localm.rag.store import _walk_files


def _symlink_or_skip(src: Path, dst: Path, **kw) -> None:
    """Create a real symlink, or skip. On Windows this needs Developer Mode or
    SeCreateSymbolicLinkPrivilege."""
    try:
        os.symlink(src, dst, **kw)
    except (OSError, NotImplementedError, AttributeError) as e:
        pytest.skip(f"cannot create a real symlink on this platform/account: {e}")


class TestSymlinkedFilesAreIndexed:
    def test_symlinked_file_in_an_added_folder_is_walked(self, tmp_path):
        target_dir = tmp_path / "elsewhere"
        target_dir.mkdir()
        real = target_dir / "spec.txt"
        real.write_text("the linked specification body", encoding="utf-8")

        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "plain.txt").write_text("a plain local note", encoding="utf-8")
        _symlink_or_skip(real, docs / "spec.txt")

        found = {p.name for p in _walk_files(docs)}
        assert found == {"plain.txt", "spec.txt"}

    def test_the_linked_file_content_is_readable_through_the_walk(self, tmp_path):
        target_dir = tmp_path / "elsewhere"
        target_dir.mkdir()
        real = target_dir / "spec.txt"
        real.write_text("the linked specification body", encoding="utf-8")
        docs = tmp_path / "docs"
        docs.mkdir()
        _symlink_or_skip(real, docs / "spec.txt")

        walked = list(_walk_files(docs))
        assert len(walked) == 1
        assert walked[0].read_text(encoding="utf-8") == "the linked specification body"

    def test_symlinked_file_nested_in_a_subfolder_is_walked(self, tmp_path):
        real = tmp_path / "target.txt"
        real.write_text("nested link target", encoding="utf-8")
        docs = tmp_path / "docs"
        (docs / "sub").mkdir(parents=True)
        _symlink_or_skip(real, docs / "sub" / "linked.txt")

        assert {p.name for p in _walk_files(docs)} == {"linked.txt"}


class TestLoopSafetyIsPreserved:
    """NEGATIVE CASES: the loop guard still holds. Following every symlink would
    pass the tests above while re-opening the junction-loop hang."""

    def test_symlinked_directory_is_still_not_followed(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("must not be walked", encoding="utf-8")

        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "own.txt").write_text("own note", encoding="utf-8")
        _symlink_or_skip(outside, docs / "linked_dir", target_is_directory=True)

        found = {p.name for p in _walk_files(docs)}
        assert found == {"own.txt"}
        assert "secret.txt" not in found

    def test_self_referential_symlinked_directory_terminates(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "own.txt").write_text("own note", encoding="utf-8")
        _symlink_or_skip(docs, docs / "loop", target_is_directory=True)

        # Must terminate (not hang / not overflow) and not re-walk itself.
        found = [p.name for p in _walk_files(docs)]
        assert found == ["own.txt"]

    def test_broken_symlink_is_skipped_without_raising(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "own.txt").write_text("own note", encoding="utf-8")
        _symlink_or_skip(tmp_path / "does_not_exist.txt", docs / "dangling.txt")

        assert {p.name for p in _walk_files(docs)} == {"own.txt"}

    def test_ordinary_files_and_dirs_unaffected(self, tmp_path):
        docs = tmp_path / "docs"
        (docs / "sub").mkdir(parents=True)
        (docs / "a.txt").write_text("a", encoding="utf-8")
        (docs / "sub" / "b.txt").write_text("b", encoding="utf-8")
        assert {p.name for p in _walk_files(docs)} == {"a.txt", "b.txt"}
