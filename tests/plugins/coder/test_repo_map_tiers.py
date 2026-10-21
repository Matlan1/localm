# SPDX-License-Identifier: AGPL-3.0-or-later
"""The project map past max_files: a directory tier that counts every file the
walk found, a file tier of the shallowest indexed files, one character budget
for both, and a cache that carries the per-directory counts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from localm.plugins.coder import indexer
from localm.plugins.coder.indexer import _MAX_MAP_CHARS, ProjectMap


def _make_tree(root: Path, dirs: int, per_dir: int, ext: str = ".py") -> int:
    n = 0
    for d in range(dirs):
        sub = root / f"pkg{d:03d}" / "inner"
        sub.mkdir(parents=True)
        for f in range(per_dir):
            (sub / f"m{f:03d}{ext}").write_text(f"def fn_{d}_{f}():\n    pass\n", encoding="utf-8")
            n += 1
    return n


def _text(pm: ProjectMap) -> str:
    return pm.to_context_string()


def test_a_small_repo_renders_exactly_as_before(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# t\n", encoding="utf-8")
    out = _text(ProjectMap.build(tmp_path))
    lines = out.splitlines()
    assert lines[0].startswith("## Codebase map")
    assert lines[1].startswith("  a.py (2L) [python]  - foo")
    assert "Directories" not in out and "Files:" not in out


def test_every_counted_file_shows_in_the_directory_tier_past_max_files(tmp_path):
    total = _make_tree(tmp_path, dirs=8, per_dir=60)             # 480 files
    pm = ProjectMap.build(tmp_path)
    assert total > indexer._MAX_FILE_COUNT
    assert len(pm.files) == indexer._MAX_FILE_COUNT
    assert pm.files_capped and pm.truncated and not pm.walk_capped
    out = _text(pm)
    assert f"{total} files in" in out
    assert "Directories (recursive file counts)" in out
    for d in range(8):
        assert f"pkg{d:03d}/  60 files" in out, f"pkg{d:03d} missing from:\n{out}"
    assert "not indexed in detail" in out
    assert len(out) <= _MAX_MAP_CHARS


def test_map_far_past_300_files_stays_in_budget_and_covers_the_whole_walk(tmp_path):
    total = _make_tree(tmp_path, dirs=120, per_dir=20)           # 2400 files, 120 dirs
    pm = ProjectMap.build(tmp_path)
    out = _text(pm)
    assert len(out) <= _MAX_MAP_CHARS
    assert f"{total} files in 240 directories" in out
    assert "more top-level directories" in out, "dirs that do not fit are counted, not dropped"
    shown = out.count("files  [python")
    assert 5 < shown < 120


def test_file_tier_lists_the_shallowest_indexed_files_first(tmp_path):
    (tmp_path / "README.md").write_text("# r\n", encoding="utf-8")
    (tmp_path / "setup.py").write_text("def setup():\n    pass\n", encoding="utf-8")
    _make_tree(tmp_path, dirs=10, per_dir=40)
    pm = ProjectMap.build(tmp_path)
    out = _text(pm)
    files_part = out.split("Files:\n", 1)[1]
    assert files_part.splitlines()[0].lstrip().startswith(("README.md", "setup.py"))
    assert "README.md" in files_part and "setup.py" in files_part


def test_detail_cap_prefers_shallow_files_over_alphabetical_order(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    for i in range(5):
        (deep / f"deep{i}.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "z_top.py").write_text("y = 1\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path, max_files=1)
    assert [f.path.name for f in pm.files] == ["z_top.py"]
    assert pm.unindexed_paths == {f"a/b/c/deep{i}.py" for i in range(5)}


def test_detail_slots_are_spread_across_directories(tmp_path):
    for d in ("alpha", "beta", "gamma"):
        (tmp_path / d).mkdir()
        for i in range(5):
            (tmp_path / d / f"f{i}.py").write_text("x = 1\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path, max_files=3)
    assert sorted(f.path.parent.name for f in pm.files) == ["alpha", "beta", "gamma"]
    assert {f.path.name for f in pm.files} == {"f0.py"}


def test_file_tier_listing_is_spread_across_directories(tmp_path):
    _make_tree(tmp_path, dirs=60, per_dir=6)                     # 360 files, 60 dirs
    out = _text(ProjectMap.build(tmp_path))
    listed_dirs = {line.split("\\")[0].split("/")[0].strip()
                   for line in out.split("Files:\n", 1)[1].splitlines() if "inner" in line}
    assert len(listed_dirs) > 15, "the file tier must not be spent on the first few directories"


def test_directory_counts_are_recursive_and_list_languages(tmp_path):
    for sub, ext, n in (("src", ".py", 30), ("web", ".js", 20), ("docs", ".md", 5)):
        (tmp_path / sub / "nested").mkdir(parents=True)
        for i in range(n):
            (tmp_path / sub / "nested" / f"f{i}{ext}").write_text("x\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path, max_files=10)
    out = _text(pm)
    assert "src/  30 files  [python 30]" in out
    assert "web/  20 files  [javascript 20]" in out
    assert "docs/  5 files  [text 5]" in out
    assert "nested/  30 files" in out, "a child directory is listed under its parent"


def test_binary_files_are_not_counted(tmp_path):
    for i in range(10):
        (tmp_path / f"img{i}.png").write_bytes(b"\x89PNG")
    (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path, max_files=1)
    assert not pm.truncated and pm.unindexed_paths == set()


def test_walk_cap_is_reported_and_bounds_the_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(indexer, "_MAX_WALK_FILES", 100)
    _make_tree(tmp_path, dirs=5, per_dir=40)                     # 200 files
    pm = ProjectMap.build(tmp_path, max_files=10)
    assert pm.walk_capped is True
    assert len(pm.files) == 10
    assert len(pm.unindexed_paths) == 90
    assert "lower bound" in _text(pm)


def test_a_uncapped_walk_is_not_flagged_as_a_lower_bound(tmp_path):
    _make_tree(tmp_path, dirs=2, per_dir=10)
    pm = ProjectMap.build(tmp_path, max_files=5)
    assert pm.walk_capped is False
    assert "lower bound" not in _text(pm)


def test_find_references_still_covers_the_indexed_files_of_a_big_repo(tmp_path):
    (tmp_path / "a.py").write_text("def helper():\n    pass\n\nhelper()\n", encoding="utf-8")
    _make_tree(tmp_path, dirs=5, per_dir=80)
    pm = ProjectMap.build(tmp_path)
    assert pm.find_references("helper") == [(Path("a.py"), 4)]


# ---------------------------------------------------------------------------
#  Cache
# ---------------------------------------------------------------------------

def test_cache_round_trip_keeps_the_directory_tier(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_tree(repo, dirs=6, per_dir=60)
    cache = tmp_path / "cache" / "map.json"
    first = ProjectMap.build(repo, cache_path=cache)
    second = ProjectMap.build(repo, cache_path=cache)
    assert second.from_cache is True
    assert second.unindexed_paths == first.unindexed_paths and second.unindexed_paths
    assert second.walk_capped == first.walk_capped
    assert "pkg005/  60 files" in _text(second)
    assert len(_text(second)) <= _MAX_MAP_CHARS


def test_a_capped_cache_without_directory_counts_is_rebuilt_not_trusted(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_tree(repo, dirs=6, per_dir=60)
    cache = tmp_path / "cache" / "map.json"
    ProjectMap.build(repo, cache_path=cache)
    data = json.loads(cache.read_text(encoding="utf-8"))
    assert data["files_capped"] is True
    del data["unindexed"]
    cache.write_text(json.dumps(data), encoding="utf-8")
    assert ProjectMap.load_cached_and_reconcile(repo, cache) is None
    rebuilt = ProjectMap.build(repo, cache_path=cache)
    assert rebuilt.from_cache is False and rebuilt.unindexed_paths


def test_an_old_uncapped_cache_without_the_field_still_loads(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("a = 1\n", encoding="utf-8")
    cache = tmp_path / "cache" / "map.json"
    ProjectMap.build(repo, cache_path=cache)
    data = json.loads(cache.read_text(encoding="utf-8"))
    data.pop("unindexed"), data.pop("walk_capped")
    cache.write_text(json.dumps(data), encoding="utf-8")
    loaded = ProjectMap.load_cached_and_reconcile(repo, cache)
    assert loaded is not None and loaded.unindexed_paths == set()


def test_a_malformed_unindexed_field_rejects_the_cache(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("a = 1\n", encoding="utf-8")
    cache = tmp_path / "cache" / "map.json"
    ProjectMap.build(repo, cache_path=cache)
    data = json.loads(cache.read_text(encoding="utf-8"))
    data["unindexed"] = {"x": "not-a-dict"}
    cache.write_text(json.dumps(data), encoding="utf-8")
    assert ProjectMap.load_cached_and_reconcile(repo, cache) is None


def test_edits_after_a_capped_build_still_refresh_indexed_files(tmp_path):
    _make_tree(tmp_path, dirs=6, per_dir=60)
    pm = ProjectMap.build(tmp_path)
    target = pm.files[0].path
    (tmp_path / target).write_text("def brand_new_symbol():\n    pass\n", encoding="utf-8")
    pm.refresh_file(tmp_path / target)
    assert "brand_new_symbol" in next(f for f in pm.files if f.path == target).symbols
    assert len(_text(pm)) <= _MAX_MAP_CHARS


def test_a_malformed_unindexed_list_rejects_the_cache(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("a = 1\n", encoding="utf-8")
    cache = tmp_path / "cache" / "map.json"
    ProjectMap.build(repo, cache_path=cache)
    data = json.loads(cache.read_text(encoding="utf-8"))
    data["unindexed"] = ["ok.py", 7]
    cache.write_text(json.dumps(data), encoding="utf-8")
    assert ProjectMap.load_cached_and_reconcile(repo, cache) is None


def test_indexing_an_unindexed_file_moves_it_out_of_the_unindexed_count(tmp_path):
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text("x = 1\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path, max_files=4)
    assert len(pm.files) == 4 and len(pm.unindexed_paths) == 6
    victim = sorted(pm.unindexed_paths)[0]
    pm.refresh_file(tmp_path / victim)
    assert len(pm.files) == 5 and len(pm.unindexed_paths) == 5
    assert "10 files in" in _text(pm) and "5 indexed in detail" in _text(pm)
    assert "5 file(s) not indexed in detail" in _text(pm)


def test_a_new_file_is_added_to_the_count_but_not_to_the_unindexed_set(tmp_path):
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text("x = 1\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path, max_files=4)
    (tmp_path / "brand_new.py").write_text("y = 1\n", encoding="utf-8")
    pm.refresh_file(tmp_path / "brand_new.py")
    assert len(pm.unindexed_paths) == 6
    assert "11 files in" in _text(pm)


def test_deleting_an_unindexed_file_drops_it_from_the_count(tmp_path):
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text("x = 1\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path, max_files=4)
    victim = sorted(pm.unindexed_paths)[0]
    (tmp_path / victim).unlink()
    pm.refresh_file(tmp_path / victim)
    assert len(pm.unindexed_paths) == 5 and "9 files in" in _text(pm)


def test_files_deleted_by_a_shell_command_leave_the_unindexed_count(tmp_path):
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text("x = 1\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path, max_files=4)
    gone = sorted(pm.unindexed_paths)[:3]
    for rel in gone:
        (tmp_path / rel).unlink()
    pm.mark_dirty()
    text = _text(pm)
    assert len(pm.unindexed_paths) == 3 and not set(gone) & pm.unindexed_paths
    assert "7 files in" in text and "3 file(s) not indexed in detail" in text


def test_a_cached_capped_map_drops_unindexed_files_deleted_since(tmp_path):
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text("x = 1\n", encoding="utf-8")
    cache = tmp_path.parent / (tmp_path.name + "-cache.json")
    first = ProjectMap.build(tmp_path, max_files=4)
    first.save_cache(cache)
    victim = sorted(first.unindexed_paths)[0]
    (tmp_path / victim).unlink()
    loaded = ProjectMap.load_cached_and_reconcile(tmp_path, cache)
    assert loaded is not None and victim not in loaded.unindexed_paths
    assert len(loaded.unindexed_paths) == 5


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("this account cannot create symbolic links")


def test_links_are_not_counted_as_unindexed_files(tmp_path):
    repo = tmp_path / "repo"
    (repo / "z").mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("o = 1\n", encoding="utf-8")
    for name in ("a.py", "b.py"):
        (repo / name).write_text("x = 1\n", encoding="utf-8")
    (repo / "z" / "c.py").write_text("c = 1\n", encoding="utf-8")
    _symlink_or_skip(repo / "z" / "link_out.py", outside)
    _symlink_or_skip(repo / "z" / "broken.py", tmp_path / "does-not-exist.py")

    capped = ProjectMap.build(repo, max_files=2)
    assert capped.unindexed_paths == {"z/c.py"} and capped.truncated

    whole = ProjectMap.build(repo, max_files=3)
    assert whole.unindexed_paths == set(), "no real file was left out"
    assert whole.truncated is False and whole.files_capped is False
    assert "Directories" not in _text(whole)
