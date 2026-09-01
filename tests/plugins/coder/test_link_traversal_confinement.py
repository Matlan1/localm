# SPDX-License-Identifier: AGPL-3.0-or-later
"""A directory link inside the project root must not carry a walker outside it.

``_confine`` guards the ``path`` ARGUMENT of a tool. It does not guard what a
walker descends into afterwards, so a directory link planted inside the root is
an ordinary entry to ``iterdir`` and to ``os.walk`` and leads out of the root.

THE TWO LINK KINDS ARE NOT INTERCHANGEABLE HERE, which is why every test is
parameterised over both rather than taking whichever the box can make:

  * ``tool_tree`` uses ``iterdir`` + ``is_dir()``, which follow BOTH a symlink
    and a junction, so both kinds reach it.
  * ``ProjectMap`` uses ``os.walk(followlinks=False)``, which skips a directory
    SYMLINK but descends a Windows JUNCTION - a junction is not ``islink``. A
    symlink-only fixture therefore cannot fail for the indexer at all.

Each test asserts BOTH directions: the out-of-root name is absent AND an
in-root name is present, so a walker that returned nothing cannot pass.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from localm.plugins.coder.indexer import ProjectMap
from localm.plugins.coder.tools.files import tool_read_file, tool_tree

OUT_OF_ROOT = "r42_out_of_root_marker.txt"
IN_ROOT = "r42_in_root_marker.txt"

LINK_KINDS = ("symlink", "junction")


def _make_dir_link(kind: str, link: Path, target: Path) -> None:
    """Create *link* as a directory link of exactly *kind*, or skip.

    Never falls back to the other kind: the two are different code paths in
    ``os.walk`` and a silent substitution makes the test unable to fail.
    """
    if kind == "symlink":
        try:
            os.symlink(target, link, target_is_directory=True)
        except (OSError, NotImplementedError) as e:
            pytest.skip(f"no directory-symlink privilege here: {e}")
        return
    if os.name != "nt":
        pytest.skip("junctions are a Windows-only reparse point")
    if subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                      capture_output=True, text=True).returncode != 0:
        pytest.skip("this volume does not support junctions")


@pytest.fixture(params=LINK_KINDS)
def linked_root(request, tmp_path: Path):
    """A project root holding an in-root file and a directory link pointing out.

    Returns (root, kind). The link is asserted to really escape, so a test that
    later finds nothing cannot be passing because the link was never functional.
    """
    kind = request.param
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    (root / "sub").mkdir(parents=True)
    outside.mkdir()
    (root / IN_ROOT).write_text("in root\n", encoding="utf-8")
    (outside / OUT_OF_ROOT).write_text("out of root\n", encoding="utf-8")
    _make_dir_link(kind, root / "linked", outside)

    assert (root / "linked" / OUT_OF_ROOT).read_text(encoding="utf-8") == "out of root\n", \
        f"the {kind} does not actually escape, so this test could not fail"
    return root, kind


def test_tree_does_not_walk_out_of_the_project_root(linked_root):
    root, kind = linked_root
    out = tool_tree(root, ".", max_depth=9).output

    assert OUT_OF_ROOT not in out, (
        f"tool_tree disclosed a filename outside the project root by following a "
        f"{kind}:\n{out}")
    assert IN_ROOT in out, (
        f"the in-root control file is missing, so the assertion above would pass "
        f"even for a tree that returned nothing:\n{out}")


def test_tree_reports_the_links_it_did_not_follow(linked_root):
    """A narrowed walk must not masquerade as a complete one."""
    root, _ = linked_root
    out = tool_tree(root, ".", max_depth=9).output

    assert "link outside the project root" in out, (
        f"tool_tree silently dropped the link instead of reporting it:\n{out}")
    assert "linked" in out, f"the skipped entry is not named:\n{out}"


def test_tree_still_recurses_into_ordinary_subdirectories(linked_root):
    """The containment check must not become a blanket refusal to descend."""
    root, _ = linked_root
    (root / "sub" / "deep.txt").write_text("deep\n", encoding="utf-8")

    out = tool_tree(root, ".", max_depth=9).output

    assert "deep.txt" in out, f"an ordinary nested file stopped being listed:\n{out}"


def test_project_map_does_not_index_through_a_directory_link(linked_root):
    root, kind = linked_root
    pm = ProjectMap.build(root, cache_path=None)
    indexed = {Path(f.path).as_posix() for f in pm.files}

    assert not any(OUT_OF_ROOT in p for p in indexed), (
        f"ProjectMap indexed a file outside the project root through a {kind}: "
        f"{sorted(indexed)}")
    assert any(IN_ROOT in p for p in indexed), (
        f"the in-root control file is missing, so the assertion above would pass "
        f"even for an empty index: {sorted(indexed)}")


def test_project_map_does_not_extract_symbols_from_outside_the_root(linked_root):
    """The map carries symbol NAMES, so an escape leaks file content, not paths."""
    root, kind = linked_root
    (root.parent / "outside" / "hidden_source.py").write_text(
        "def r42_leaked_symbol():\n    pass\n", encoding="utf-8")
    (root / "own.py").write_text("def r42_own_symbol():\n    pass\n", encoding="utf-8")

    pm = ProjectMap.build(root, cache_path=None)
    symbols = {s for f in pm.files for s in f.symbols}

    assert "r42_leaked_symbol" not in symbols, (
        f"a symbol was extracted from a source file outside the root via a {kind}: "
        f"{sorted(symbols)}")
    assert "r42_own_symbol" in symbols, (
        f"the in-root control symbol is missing, so the assertion above would pass "
        f"even for an empty index: {sorted(symbols)}")


def test_project_map_does_not_index_a_symlinked_file(tmp_path: Path):
    """A linked FILE inside the root is an escape too, not just a linked dir."""
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / IN_ROOT).write_text("in root\n", encoding="utf-8")
    target = outside / OUT_OF_ROOT
    target.write_text("out of root\n", encoding="utf-8")
    try:
        os.symlink(target, root / "linked.txt")
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"file symlink creation not permitted here: {e}")

    pm = ProjectMap.build(root, cache_path=None)
    indexed = {Path(f.path).as_posix() for f in pm.files}

    assert "linked.txt" not in indexed, (
        f"ProjectMap indexed a symlink pointing outside the root: {sorted(indexed)}")
    assert any(IN_ROOT in p for p in indexed), (
        f"the in-root control file is missing: {sorted(indexed)}")


def test_read_file_still_refuses_the_linked_path(linked_root):
    """The pre-existing per-call guard is unchanged by the walker fix."""
    root, _ = linked_root

    refused = tool_read_file(root, f"linked/{OUT_OF_ROOT}").output
    assert "resolves outside the working directory" in refused, refused

    allowed = tool_read_file(root, IN_ROOT).output
    assert "in root" in allowed, allowed
