# SPDX-License-Identifier: AGPL-3.0-or-later
"""ProjectMap.build: directory pruning, bounded collection, and a wall-clock
deadline so a coder session pointed at a huge root (C:\\) cannot appear to hang
(CODER-1). The old build did sorted(root.rglob("*")) - it materialised and
sorted the WHOLE tree (descending into node_modules / .git / system dirs) before
the file cap was ever checked."""

from localm.plugins.coder.indexer import ProjectMap


def test_small_repo_indexed_with_symbols(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n\nclass Bar:\n    pass\n",
                                   encoding="utf-8")
    (tmp_path / "readme.md").write_text("# hi\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path)
    assert not pm.truncated
    by = {f.path.name: f for f in pm.files}
    assert "a.py" in by and "readme.md" in by
    assert "foo" in by["a.py"].symbols and "Bar" in by["a.py"].symbols


def test_prunes_skip_hidden_and_gitignored_dirs(tmp_path):
    (tmp_path / "keep.py").write_text("k = 1\n", encoding="utf-8")
    nm = tmp_path / "node_modules" / "pkg"; nm.mkdir(parents=True)
    (nm / "index.js").write_text("a = 1\n", encoding="utf-8")
    git = tmp_path / ".git"; git.mkdir()
    (git / "config").write_text("x\n", encoding="utf-8")
    hidden = tmp_path / ".secret"; hidden.mkdir()
    (hidden / "s.py").write_text("s = 1\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("build/\n", encoding="utf-8")
    bd = tmp_path / "build"; bd.mkdir()
    (bd / "out.py").write_text("o = 1\n", encoding="utf-8")

    pm = ProjectMap.build(tmp_path)
    names = {str(f.path).replace("\\", "/") for f in pm.files}
    assert "keep.py" in names
    assert not any("node_modules" in n for n in names)   # never descended
    assert not any(".git" in n for n in names)
    assert not any(".secret" in n for n in names)         # hidden dir pruned
    assert not any(n.startswith("build/") for n in names) # gitignored dir pruned


def test_large_tree_is_bounded_and_truncated(tmp_path):
    # More files than the candidate cap (for max_files=5 the cap floor is 55) so
    # the walk stops early; the map is bounded and flagged truncated.
    for i in range(120):
        (tmp_path / f"f{i:03d}.py").write_text("x = 1\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path, max_files=5)
    assert pm.truncated is True
    assert pm.file_count() <= 5


def test_deadline_truncates_and_stops_walking(tmp_path, monkeypatch):
    # A subdir under the root; force the clock to jump past the deadline when the
    # walk reaches the second directory, so the subdir is never scanned.
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    sub = tmp_path / "sub"; sub.mkdir()
    (sub / "b.py").write_text("y = 2\n", encoding="utf-8")

    import localm.plugins.coder.indexer as idx
    seq = iter([0.0, 0.0, 5.0])   # start, root-iter, sub-iter (over the 1s deadline)
    monkeypatch.setattr(idx.time, "monotonic", lambda: next(seq, 5.0))

    pm = ProjectMap.build(tmp_path, deadline_s=1.0)
    assert pm.truncated is True
    names = {f.path.name for f in pm.files}
    assert "a.py" in names          # collected before the deadline
    assert "b.py" not in names      # the subdir was cut off


def test_deadline_none_disables_the_cap(tmp_path):
    # A normal repo with the deadline disabled indexes fully (no spurious truncate).
    (tmp_path / "a.py").write_text("z = 1\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path, deadline_s=None)
    assert not pm.truncated
    assert any(f.path.name == "a.py" for f in pm.files)
