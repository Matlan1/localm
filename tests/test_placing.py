# SPDX-License-Identifier: AGPL-3.0-or-later
"""A file gets a name only where the name is free or still belongs to the file that is
replaced, and a replacement that was killed is put back."""

from __future__ import annotations

import ast
import errno
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from localm import model_manager as mm
from localm.model_manager import placing
from localm.model_manager import pull as pull_mod
from tests import test_folder_import_progress as _progress

home, printed = _progress.home, _progress.printed
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["links", "rename", "neither"])
def way(request, monkeypatch):
    """Hard links work, only an exclusive rename does, or neither."""
    def _link(src, dst, **kw):
        raise OSError(errno.EPERM, "links are not supported here")
    if request.param != "links":
        monkeypatch.setattr(os, "link", _link)
    if request.param == "neither":
        monkeypatch.setattr(placing, "_rename_exclusive", lambda src, dst: False)
    return request.param


def _file(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class TestANewName:
    def test_a_free_one_is_taken(self, way, tmp_path):
        src, dest = _file(tmp_path / "src", "new"), tmp_path / "dest"
        placing.put_new(src, dest)
        assert dest.read_text() == "new" and not src.exists()

    def test_one_that_exists_is_never_replaced(self, way, tmp_path):
        src, dest = _file(tmp_path / "src", "new"), _file(tmp_path / "dest", "mine")
        with pytest.raises(FileExistsError):
            placing.put_new(src, dest)
        assert dest.read_text() == "mine" and src.read_text() == "new"

    def test_a_folder_never_replaces_one(self, way, tmp_path):
        src, dest = tmp_path / "src", tmp_path / "dest"
        _file(src / "f", "new")
        dest.mkdir()
        with pytest.raises(FileExistsError):
            placing.rename_new(src, dest)
        assert (src / "f").read_text() == "new"

    def test_the_first_free_one_of_several(self, way, tmp_path):
        src = _file(tmp_path / "src", "new")
        taken = [_file(tmp_path / n, n) for n in ("a", "b")]
        free = tmp_path / "c"
        assert placing.put_first_free(src, [*taken, free]) == free
        assert [p.read_text() for p in taken] == ["a", "b"] and free.read_text() == "new"
        with pytest.raises(FileExistsError):
            placing.put_first_free(free, taken)


class TestAReplacement:
    def test_the_file_it_was_decided_for_is_replaced(self, tmp_path):
        src, dest = _file(tmp_path / "src", "new"), _file(tmp_path / "dest", "old")
        placing.put_over(src, dest, placing.stamp(dest))
        assert dest.read_text() == "new"

    @pytest.mark.parametrize("change", ["other file", "other content", "gone"])
    def test_a_file_that_is_another_one_by_then_is_not(self, tmp_path, change):
        src, dest = _file(tmp_path / "src", "new"), _file(tmp_path / "dest", "old")
        old = placing.stamp(dest)
        if change == "other content":
            dest.write_text("longer")
        else:
            dest.unlink()
        if change == "other file":
            _file(dest, "old")
            os.utime(dest, ns=(1, 1))
        with pytest.raises(FileExistsError):
            placing.put_over(src, dest, old)
        assert src.read_text() == "new"
        assert change == "gone" or dest.read_text() in ("old", "longer")


_KILLED = """
import errno, os, sys
from pathlib import Path
from localm.model_manager import placing as p
base, new, links, nth = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "1", int(sys.argv[4])
if not links:
    def _link(*a, **k):
        raise OSError(errno.EPERM, "no links")
    os.link = _link
count = [0]
for name in ("put_new", "rename_new"):
    real = getattr(p, name)
    def _call(*a, _real=real):
        count[0] += 1
        if count[0] > nth:
            os._exit(3)
        return _real(*a)
    setattr(p, name, _call)
real_replace = os.replace
def _replace(*a):
    count[0] += 1
    if count[0] > nth:
        os._exit(3)
    return real_replace(*a)
os.replace = _replace
names = ("config.json", "sub/model.safetensors")
p.place_files({n: new / n for n in names}, base, {n: p.stamp(base / n) for n in names})
"""


class TestAReplacementThatWasKilled:
    NAMES = ("config.json", "sub/model.safetensors")

    def _kill(self, base: Path, new: Path, links: bool, nth: int) -> None:
        for n in self.NAMES:
            _file(base / n, f"old {n}")
            _file(new / n, f"new {n}")
        run = subprocess.run([sys.executable, "-c", _KILLED, str(base), str(new),
                              "1" if links else "0", str(nth)], cwd=ROOT)
        assert run.returncode == 3

    def _check(self, base: Path, new: Path) -> None:
        assert {n: (base / n).read_text() for n in self.NAMES} == {
            n: f"old {n}" for n in self.NAMES}
        assert {n: (new / n).read_text() for n in self.NAMES} == {
            n: f"new {n}" for n in self.NAMES}
        assert placing.unfinished(base) == []

    @pytest.mark.parametrize("links, nth", [(True, 0), (True, 1), (False, 0), (False, 1),
                                            (False, 2), (False, 3)])
    def test_it_is_put_back_whole(self, tmp_path, links, nth):
        base, new = tmp_path / "base", tmp_path / "new"
        self._kill(base, new, links, nth)
        assert placing.put_back(base) is True
        self._check(base, new)

    def test_discarding_the_download_first_puts_the_old_files_back(self, home, printed):
        base = home / "models" / "snap"
        self._kill(base, base / pull_mod._SNAPSHOT_STAGING, False, 1)
        assert not (base / self.NAMES[1]).exists()
        pull_mod.discard_partial_download("o/r", "snap")
        assert {n: (base / n).read_text() for n in self.NAMES} == {
            n: f"old {n}" for n in self.NAMES}
        assert placing.unfinished(base) == []

    def test_the_next_scan_puts_the_old_files_back(self, home, printed):
        base = home / "models" / "snap"
        self._kill(base, base / pull_mod._SNAPSHOT_STAGING, False, 1)
        mm.sync_models_dir(backfill_mmproj=False)
        self._check(base, base / pull_mod._SNAPSHOT_STAGING)

    def test_a_replacement_that_runs_is_left_alone_by_the_scan(self, home, printed):
        base = home / "models" / "snap"
        self._kill(base, base / pull_mod._SNAPSHOT_STAGING, False, 1)
        with pull_mod._snapshot_dest_lock("snap"):
            mm.sync_models_dir(backfill_mmproj=False)
            assert len(placing.unfinished(base)) == 1
        mm.sync_models_dir(backfill_mmproj=False)
        assert placing.unfinished(base) == []

    def test_a_file_of_somebody_else_at_the_name_is_kept_and_so_is_the_old_one(self, tmp_path):
        base, new = tmp_path / "base", tmp_path / "new"
        self._kill(base, new, False, 1)
        _file(base / self.NAMES[1], "somebody's own")
        assert placing.put_back(base) is False
        assert (base / self.NAMES[1]).read_text() == "somebody's own"
        ((root, _names),) = placing.unfinished(base)
        assert (root / "old" / self.NAMES[1]).read_text() == f"old {self.NAMES[1]}"

    def test_an_unreadable_journal_deletes_nothing(self, tmp_path):
        base, new = tmp_path / "base", tmp_path / "new"
        self._kill(base, new, False, 1)
        ((root, _names),) = placing.unfinished(base)
        (root / "journal.json").write_text("{ not json")
        assert placing.put_back(base) is False
        assert (root / "old" / self.NAMES[1]).read_text() == f"old {self.NAMES[1]}"

    def test_only_the_replacements_of_the_given_names_are_touched(self, tmp_path):
        base, new = tmp_path / "base", tmp_path / "new"
        self._kill(base, new, False, 1)
        assert placing.put_back(base, ["another.gguf"]) is True
        assert len(placing.unfinished(base)) == 1


_CALLS = {("os", n) for n in ("rename", "renames", "replace", "link")} | {
    ("shutil", n) for n in ("move", "copy", "copy2", "copyfile", "copytree")}


def _direct_calls(module: str) -> set:
    tree = ast.parse((ROOT / "localm" / "model_manager" / module).read_text(encoding="utf-8"))
    out = set()

    def _walk(node, where):
        for child in ast.iter_child_nodes(node):
            name = where
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not where:
                name = child.name
            elif isinstance(child, ast.ClassDef):
                name = ""
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                attr, owner = child.func.attr, child.func.value
                if isinstance(owner, ast.Name) and (owner.id, attr) in _CALLS:
                    out.add((name, f"{owner.id}.{attr}"))
                elif not isinstance(owner, ast.Name) or owner.id not in ("os", "shutil", "placing"):
                    if (attr in ("rename", "hardlink_to") and child.args) or (
                            attr == "replace" and len(child.args) == 1):
                        out.add((name, f".{attr}"))
            _walk(child, name)
    _walk(tree, "")
    return out


def test_model_files_are_named_only_through_placing():
    """Every other call that renames, links or copies, with what it works on."""
    assert _direct_calls("pull.py") == {
        ("_adopt_partial", ".rename"),             # partial data to a new temp name
        ("hold", "os.replace"),                    # a finished download to a new temp name
        ("_refetch_snapshot_files", "os.replace"),  # huggingface_hub's metadata files
        ("_rename_dir", "os.rename"),              # lock folders
        ("_rename_noreplace", "os.rename"),        # lock folders
    }
    assert _direct_calls("registry.py") == {
        ("_backup_registry", "shutil.copy2"),      # registry.json to its backup
        ("_copy_verified", "shutil.copy2"),        # to a new temp name
        ("_transfer_model_dir", "shutil.copytree"),  # to a new temp name
    }


def test_a_journal_names_every_file(tmp_path):
    base = tmp_path / "base"
    new = {n: _file(tmp_path / "new" / n, n) for n in ("a", "b")}
    _file(base / "a", "old a")
    seen = []
    real = placing._replace_one

    def _look(*a):
        ((root, _names),) = placing.unfinished(base)
        seen.append(json.loads((root / "journal.json").read_text(encoding="utf-8")))
        return real(*a)
    old = {"a": placing.stamp(base / "a"), "b": None}
    placing._replace_one = _look
    try:
        placing.place_files(new, base, old)
    finally:
        placing._replace_one = real
    assert set(seen[0]) == {"a", "b"} and placing.unfinished(base) == []
    assert (base / "a").read_text() == "a" and (base / "b").read_text() == "b"
