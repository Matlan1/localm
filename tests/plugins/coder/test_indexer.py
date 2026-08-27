# SPDX-License-Identifier: AGPL-3.0-or-later
"""ProjectMap.build: directory pruning, bounded collection, and a wall-clock
deadline so a coder session pointed at a huge root cannot appear to hang. A
``sorted(root.rglob("*"))`` materialises and sorts the WHOLE tree (descending
into node_modules / .git / system dirs) before the file cap is ever checked.

Also mark_dirty() / _rescan_if_dirty(): the stat-diff + listdir reconciliation
that keeps the map from going stale after a run_shell write (which - unlike
write_file/edit_file - has no `path` arg to refresh_file() ahead of time; see
execution.py's _refresh_map_for_tool and context.py's _build_messages)."""

import json
import os
import time
from unittest.mock import patch

from localm.plugins.coder.indexer import FileSummary, ProjectMap


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


# ------------------- _index_deadline() reads the registered setting -------- #

def test_index_deadline_uses_the_default_from_config(monkeypatch):
    # _index_deadline() imports load_config locally, so the patch target is the
    # source (localm.config.load_config), not the checkpoint module's namespace.
    from localm.plugins.coder.agent.checkpoint import _index_deadline
    from localm.config import DEFAULT_CONFIG
    monkeypatch.setattr("localm.config.load_config", lambda: dict(DEFAULT_CONFIG))
    assert _index_deadline() == DEFAULT_CONFIG["coder_index_timeout"]


def test_index_deadline_honours_override(monkeypatch):
    from localm.plugins.coder.agent.checkpoint import _index_deadline
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"coder_index_timeout": 45})
    assert _index_deadline() == 45.0


def test_index_deadline_zero_or_negative_disables_it(monkeypatch):
    from localm.plugins.coder.agent.checkpoint import _index_deadline
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"coder_index_timeout": 0})
    assert _index_deadline() is None
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"coder_index_timeout": -5})
    assert _index_deadline() is None


def test_index_deadline_falls_back_on_malformed_value(monkeypatch):
    # A non-numeric value falls back to the built-in default instead of raising.
    from localm.plugins.coder.agent.checkpoint import _index_deadline
    from localm.plugins.coder.indexer import _BUILD_DEADLINE_S
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"coder_index_timeout": "disabled"})
    assert _index_deadline() == _BUILD_DEADLINE_S


# --------------------------------------------------------------------------- #
#  mark_dirty() / _rescan_if_dirty(): staleness after a run_shell write       #
# --------------------------------------------------------------------------- #

def test_build_records_mtime_and_size(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n", encoding="utf-8")
    st = f.stat()
    pm = ProjectMap.build(tmp_path)
    fs = next(x for x in pm.files if x.path.name == "a.py")
    assert fs.mtime == st.st_mtime
    assert fs.size == st.st_size


def test_file_summary_mtime_size_default_when_omitted(tmp_path):
    # Both new fields are defaulted and ordered after `symbols`, so construction
    # without them still works, positionally or by keyword.
    fs = FileSummary(path=tmp_path / "x.py", lang="python", lines=1)
    assert fs.mtime == 0.0
    assert fs.size == 0


def test_rescan_is_a_noop_until_marked_dirty(tmp_path):
    # Editing a tracked file without mark_dirty() is not picked up: the flag is
    # what gates the rescan.
    f = tmp_path / "a.py"
    f.write_text("x = 1\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path)

    f.write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8")   # size changes too
    assert "3L" not in pm.to_context_string()   # still the stale 1-line read


def test_rescan_picks_up_a_new_file_in_a_known_directory(tmp_path):
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "a.py").write_text("a = 1\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path)
    assert pm.file_count() == 1

    (sub / "b.py").write_text("def new_func():\n    pass\n", encoding="utf-8")
    pm.mark_dirty()
    assert pm.file_count() == 2
    by_name = {f.path.name: f for f in pm.files}
    assert "new_func" in by_name["b.py"].symbols


def test_rescan_picks_up_a_modified_file(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def old_func():\n    pass\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path)

    f.write_text("def new_func():\n    pass\n\ndef second():\n    pass\n",
                  encoding="utf-8")
    pm.mark_dirty()
    pm.file_count()   # trigger the rescan - mark_dirty() alone does not
    fs = next(x for x in pm.files if x.path.name == "a.py")
    assert "new_func" in fs.symbols and "second" in fs.symbols
    assert "old_func" not in fs.symbols


def test_rescan_picks_up_a_deleted_file(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path)
    assert pm.file_count() == 1

    f.unlink()
    pm.mark_dirty()
    assert pm.file_count() == 0


def test_rescan_skips_binary_new_files_like_build_does(tmp_path):
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "a.py").write_text("a = 1\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path)

    (sub / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
    pm.mark_dirty()
    pm.file_count()
    assert "image.png" not in {f.path.name for f in pm.files}


def test_rescan_respects_gitignore_for_new_files(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path)

    (tmp_path / "debug.log").write_text("noisy\n", encoding="utf-8")
    pm.mark_dirty()
    pm.file_count()
    assert "debug.log" not in {f.path.name for f in pm.files}


def test_rescan_does_not_discover_a_brand_new_directory(tmp_path):
    # Only a directory that ALREADY has a tracked file is listdir()-ed, so a file
    # inside a directory created from scratch by run_shell needs a full reindex().
    # Contrasted here against the one-level case that does work.
    (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path)

    newdir = tmp_path / "newpkg"
    newdir.mkdir()
    (newdir / "b.py").write_text("b = 1\n", encoding="utf-8")
    pm.mark_dirty()
    pm.file_count()   # trigger the rescan - mark_dirty() alone does not
    assert "b.py" not in {f.path.name for f in pm.files}   # the known gap

    fresh = ProjectMap.build(tmp_path)                     # a full reindex DOES see it
    assert "b.py" in {f.path.name for f in fresh.files}


def test_rescan_skips_new_file_discovery_on_a_capped_build(tmp_path):
    """build() truncates at max_files, so a directory with one tracked file can
    hold dozens more that were NEVER candidates. A listdir sweep cannot tell
    those apart from a file run_shell genuinely just created, and treating them
    the same grows the map past max_files a little more on every dirty read.
    files_capped gates exactly that pass off; the stat-diff pass for
    already-tracked files is unaffected, since a tracked file's own stat is
    meaningful either way.

    Spies on refresh_file directly, not just the outcome, so the GATE is what is
    under test: without files_capped this is exactly the call the spy would
    record for z.py. (The touched-a.py edit is what makes this a REAL rescan
    rather than a no-op - mark_dirty() alone never triggers one; only the next
    read, here pm.file_count(), does.)"""
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "a.py").write_text("def old():\n    pass\n", encoding="utf-8")
    (sub / "z.py").write_text("z = 1\n", encoding="utf-8")   # excluded by the cap from the start
    pm = ProjectMap.build(tmp_path, max_files=1)   # forces truncation with room to spare
    assert pm.truncated is True and pm.files_capped is True
    assert pm.file_count() == 1
    assert pm.files[0].path.name == "a.py"   # "a.py" sorts first; z.py is the excluded one

    # A genuinely NEW file in the same known directory, plus an edit to the
    # already-tracked one, so the rescan has real work to do.
    (sub / "brand_new.py").write_text("new = 1\n", encoding="utf-8")
    (sub / "a.py").write_text("def old():\n    pass\n\ndef added():\n    pass\n",
                               encoding="utf-8")
    pm.mark_dirty()
    with patch.object(pm, "refresh_file", wraps=pm.refresh_file) as spy:
        pm.file_count()   # trigger the rescan - mark_dirty() alone does not
        called = {str(c.args[0]) for c in spy.call_args_list}
        assert str(sub / "z.py") not in called, (
            "the excluded-by-cap file must never reach refresh_file while files_capped")
        assert str(sub / "brand_new.py") not in called

    fs = next(x for x in pm.files if x.path.name == "a.py")
    assert "added" in fs.symbols   # the tracked file's own edit still lands
    tracked_names = {f.path.name for f in pm.files}
    assert "z.py" not in tracked_names and "brand_new.py" not in tracked_names


def test_many_mark_dirty_calls_before_one_read_cost_one_rescan(tmp_path):
    """The whole point of the flag: any number of run_shell calls before the
    map is next read must reconcile in ONE pass, not one per call - and a
    later read with nothing newly dirty must not re-scan at all."""
    a = tmp_path / "a.py"
    a.write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("b = 1\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path)

    a.write_text("a = 1\nextra = True\n", encoding="utf-8")   # only a.py changed
    for _ in range(10):
        pm.mark_dirty()

    with patch.object(pm, "refresh_file", wraps=pm.refresh_file) as spy:
        pm.to_context_string()
        assert spy.call_count == 1          # only the one file that actually moved
        assert pm.dirty is False

        pm.to_context_string()              # a second, later read
        assert spy.call_count == 1          # nothing new since - no re-scan


def test_file_count_also_triggers_the_rescan(tmp_path):
    # to_context_string and file_count are independent read paths; both self-heal.
    f = tmp_path / "a.py"
    f.write_text("a = 1\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path)

    f.unlink()
    pm.mark_dirty()
    assert pm.file_count() == 0


# --------------------------------------------------------------------------- #
#  Cross-session cache: save_cache() / load_cached_and_reconcile()            #
#                                                                              #
#  The cache file lives under ``tmp_path / ".cache"`` in every test below, a  #
#  HIDDEN directory, so build()'s dot-dir pruning keeps it out of the scanned #
#  tree, matching production (the real cache lives under                      #
#  HOME/checkpoints/<digest>/, never inside the project it describes). A      #
#  cache file in a directory reconciliation already treats as "known" gets    #
#  discovered by the new-file listdir sweep and added to the map as if it     #
#  were project source.                                                       #
# --------------------------------------------------------------------------- #

def test_save_cache_then_load_reconcile_round_trips_unchanged(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    cache = tmp_path / ".cache" / "cache.json"
    pm = ProjectMap.build(tmp_path)
    pm.save_cache(cache)

    loaded = ProjectMap.load_cached_and_reconcile(tmp_path, cache)
    assert loaded is not None
    assert loaded.file_count() == 1
    fs = next(x for x in loaded.files if x.path.name == "a.py")
    assert "foo" in fs.symbols


def test_load_reconcile_picks_up_an_edit_made_after_caching(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def old_func():\n    pass\n", encoding="utf-8")
    cache = tmp_path / ".cache" / "cache.json"
    ProjectMap.build(tmp_path).save_cache(cache)

    f.write_text("def new_func():\n    pass\n", encoding="utf-8")
    # Force a detectably later mtime. old_func/new_func are the same byte
    # length, so _rescan_if_dirty()'s stat-diff (mtime+size, not a content
    # hash) needs the mtime to differ - and two back-to-back writes can
    # otherwise land in the same filesystem timestamp tick.
    st = f.stat()
    os.utime(f, (st.st_atime, st.st_mtime + 2))

    loaded = ProjectMap.load_cached_and_reconcile(tmp_path, cache)
    assert loaded.file_count() == 1   # still just a.py - no self-contamination
    fs = next(x for x in loaded.files if x.path.name == "a.py")
    assert "new_func" in fs.symbols and "old_func" not in fs.symbols


def test_load_reconcile_picks_up_a_deletion_made_after_caching(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n", encoding="utf-8")
    cache = tmp_path / ".cache" / "cache.json"
    ProjectMap.build(tmp_path).save_cache(cache)

    f.unlink()
    loaded = ProjectMap.load_cached_and_reconcile(tmp_path, cache)
    assert loaded.file_count() == 0


def test_load_reconcile_discovers_a_new_file_when_not_capped(tmp_path):
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "a.py").write_text("a = 1\n", encoding="utf-8")
    cache = tmp_path / ".cache" / "cache.json"
    pm = ProjectMap.build(tmp_path)
    assert pm.files_capped is False
    pm.save_cache(cache)

    (sub / "b.py").write_text("def new_func():\n    pass\n", encoding="utf-8")
    loaded = ProjectMap.load_cached_and_reconcile(tmp_path, cache)
    assert loaded.file_count() == 2
    assert {f.path.name for f in loaded.files} == {"a.py", "b.py"}


def test_load_reconcile_does_not_discover_a_new_file_when_capped(tmp_path):
    """Mirrors test_rescan_skips_new_file_discovery_on_a_capped_build: caching
    must not weaken the existing files_capped guard - a project bigger than
    max_files still cannot tell a genuinely-new file from one that was simply
    never a candidate, so pass 2 stays off, same as an intra-session dirty
    rescan."""
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "a.py").write_text("a = 1\n", encoding="utf-8")
    (sub / "z.py").write_text("z = 1\n", encoding="utf-8")
    cache = tmp_path / ".cache" / "cache.json"
    pm = ProjectMap.build(tmp_path, max_files=1)
    assert pm.files_capped is True
    pm.save_cache(cache)

    (sub / "brand_new.py").write_text("new = 1\n", encoding="utf-8")
    loaded = ProjectMap.load_cached_and_reconcile(tmp_path, cache)
    assert "brand_new.py" not in {f.path.name for f in loaded.files}
    assert loaded.files_capped is True   # the flag itself survives the round trip


def test_load_reconcile_returns_none_for_a_missing_cache(tmp_path):
    assert ProjectMap.load_cached_and_reconcile(
        tmp_path, tmp_path / ".cache" / "nope.json") is None


def test_load_reconcile_returns_none_for_a_corrupt_cache(tmp_path):
    cache = tmp_path / ".cache" / "cache.json"
    cache.parent.mkdir(parents=True)
    cache.write_text("{not valid json", encoding="utf-8")
    assert ProjectMap.load_cached_and_reconcile(tmp_path, cache) is None


def test_load_reconcile_returns_none_for_wrong_shape(tmp_path):
    """A future-version or hand-edited cache with the wrong fields must fall
    back to a full build(), never raise and never be trusted."""
    cache = tmp_path / ".cache" / "cache.json"
    cache.parent.mkdir(parents=True)
    cache.write_text('{"version": 2, "files": []}', encoding="utf-8")
    assert ProjectMap.load_cached_and_reconcile(tmp_path, cache) is None


def test_load_reconcile_returns_none_for_a_different_root(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    cache = tmp_path / ".cache" / "cache.json"
    ProjectMap.build(tmp_path).save_cache(cache)

    other = tmp_path.parent / (tmp_path.name + "-other")
    other.mkdir()
    assert ProjectMap.load_cached_and_reconcile(other, cache) is None


def test_load_reconcile_returns_none_when_older_than_max_age(tmp_path):
    """A cache older than max_age_s is rejected. Backdates cached_at directly
    rather than asserting elapsed real time > 0.0, because time.time() on this
    platform resolves via GetSystemTimeAsFileTime() at ~15.6ms granularity: a
    save-then-load round trip can complete inside one tick, making elapsed == 0.0
    and racing a max_age_s=0.0 bound instead of testing the staleness check."""
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    cache = tmp_path / ".cache" / "cache.json"
    ProjectMap.build(tmp_path).save_cache(cache)

    data = json.loads(cache.read_text(encoding="utf-8"))
    data["cached_at"] = time.time() - 1000.0
    cache.write_text(json.dumps(data), encoding="utf-8")

    assert ProjectMap.load_cached_and_reconcile(
        tmp_path, cache, max_age_s=1.0) is None


def test_save_cache_is_atomic_no_tmp_file_left_behind(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    cache = tmp_path / ".cache" / "cache.json"
    ProjectMap.build(tmp_path).save_cache(cache)
    assert cache.is_file()
    assert not cache.with_suffix(cache.suffix + ".tmp").exists()


def test_save_cache_never_raises_on_write_failure(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    pm = ProjectMap.build(tmp_path)
    # A path whose PARENT is a FILE, not a directory: mkdir(parents=True) fails
    # and save_cache swallows that rather than raising.
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    pm.save_cache(blocker / "sub" / "cache.json")   # must not raise
