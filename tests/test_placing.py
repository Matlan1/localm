# SPDX-License-Identifier: AGPL-3.0-or-later
"""A file gets a name only where the name is free or still belongs to the file that is
replaced; whatever moment a process is killed at, every file is still there and the next
scan or pull puts it right; and nothing a model folder holds is read as an instruction."""

from __future__ import annotations

import ast
import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from localm import model_manager as mm
from localm.model_manager import placing
from localm.model_manager import pull as pull_mod
from localm.model_manager import registry as registry_mod
from tests import test_folder_import_progress as _progress
from tests.test_cached_download_reverify import (
    _snapshot_hub, CONFIG, DAMAGED_WEIGHTS, GOOD_WEIGHTS)

home, printed, _gguf = _progress.home, _progress.printed, _progress._gguf
ROOT = Path(__file__).resolve().parents[1]
WAYS = ("exclusive", "links", "neither")


def _no_link(src, dst, **kw):
    raise OSError(errno.EPERM, "links are not supported here")


def _use(monkeypatch, way: str) -> None:
    """The system renames without replacing, or only links do, or neither."""
    if way != "exclusive":
        monkeypatch.setattr(placing, "_rename_exclusive", lambda src, dst: False)
    if way == "neither":
        monkeypatch.setattr(os, "link", _no_link)


@pytest.fixture(params=WAYS)
def way(request, monkeypatch):
    _use(monkeypatch, request.param)
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
            placing.put_new(src, dest)
        assert (src / "f").read_text() == "new"

    def test_the_first_free_one_of_several(self, way, tmp_path):
        src = _file(tmp_path / "src", "new")
        taken = [_file(tmp_path / n, n) for n in ("a", "b")]
        free = tmp_path / "c"
        assert placing.put_first_free(src, [*taken, free]) == free
        assert [p.read_text() for p in taken] == ["a", "b"] and free.read_text() == "new"
        with pytest.raises(FileExistsError):
            placing.put_first_free(free, taken)

    def test_links_are_not_used_where_a_rename_that_never_replaces_works(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(os, "link", _no_link)
        src = _file(tmp_path / "src", "new")
        placing.put_new(src, tmp_path / "dest")
        assert (tmp_path / "dest").read_text() == "new"

    def test_an_old_name_that_stays_is_a_failure_and_leaves_one_name(self, tmp_path, monkeypatch):
        _use(monkeypatch, "links")
        src, dest = _file(tmp_path / "src", "new"), tmp_path / "dest"
        real = os.unlink

        def _unlink(path, *a, **k):
            if Path(path) == src:
                raise PermissionError(13, "in use")
            return real(path, *a, **k)
        monkeypatch.setattr(os, "unlink", _unlink)
        with pytest.raises(PermissionError):
            placing.put_new(src, dest)
        assert src.read_text() == "new" and not dest.exists()


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


# ------------------------------------------------------------ killed at any moment

class _Killed(BaseException):
    pass


class _Kill:
    """After *nth* system calls that change a name every further one fails, so nothing
    the process would still do in an except or finally block happens."""
    CALLS = ("rename", "replace", "link", "unlink", "rmdir", "mkdir", "utime", "chmod")

    def __init__(self, monkeypatch, nth: int) -> None:
        self.left = nth
        for name in self.CALLS:
            monkeypatch.setattr(os, name, self._counted(getattr(os, name)))
        if sys.platform != "win32":
            monkeypatch.setattr(placing, "_rename_exclusive",
                                self._counted(placing._rename_exclusive))

    def _counted(self, real):
        def _call(*a, **k):
            if self.left == 0:
                raise _Killed
            out = real(*a, **k)
            self.left -= 1
            if self.left == 0:
                raise _Killed
            return out
        return _call


def _killed_after(monkeypatch, nth: int, way: str, step) -> bool:
    """Run *step*, killed after *nth* system calls (0: before the first)."""
    with monkeypatch.context() as m:
        _use(m, way)
        _Kill(m, nth)
        try:
            step()
        except _Killed:
            return True
    return False


HAD = ("config.json", "sub/model.safetensors")
ADDED = "added.bin"
NAMES = (HAD[0], ADDED, HAD[1])


class _Library:
    """A model folder with old files and a staged new version that adds one."""

    def __init__(self, root: Path) -> None:
        self.models, self.records = root / "models", root / "locks" / "replacing"
        self.base = self.models / "snap"
        self.stage = self.base / placing.REPO_STAGING
        for n in HAD:
            _file(self.base / n, f"old {n}")
        for n in NAMES:
            _file(self.stage / n, f"new {n}")

    def place(self) -> None:
        old = {n: placing.stamp(self.base / n) for n in NAMES}
        placing.place_files({n: self.stage / n for n in NAMES}, self.base, old,
                            self.models, self.records)

    def put_back(self) -> bool:
        return placing.put_back(self.models, self.records, "snap")

    def texts(self) -> list:
        return [p.read_text() for p in self.models.rglob("*") if p.is_file()]

    def at(self, names=NAMES) -> dict:
        return {n: (self.base / n).read_text() for n in names if (self.base / n).exists()}

    def check_all_there(self) -> None:
        """Every old file is there, under its name or kept by a record, or the whole new
        version is; every new one is there."""
        texts = self.texts()
        assert {f"new {n}" for n in NAMES} <= set(texts)
        if self.at() == {n: f"new {n}" for n in NAMES}:
            return
        assert {f"old {n}" for n in HAD} <= set(texts)
        for n in HAD:
            if self.at().get(n) != f"old {n}":
                (found,) = placing.unfinished(self.records)
                kept = placing._kept(self.base, found.record.stem) / str(found.names.index(n))
                assert kept.read_text() == f"old {n}"

    def check_settled(self) -> None:
        """The whole old version with the new one staged, or the whole new version."""
        new = {n: f"new {n}" for n in NAMES}
        if self.at() != new:
            assert self.at() == {n: f"old {n}" for n in HAD}
            assert {n: (self.stage / n).read_text() for n in NAMES} == new
        assert sorted(self.texts()) == sorted(
            list(self.at().values()) + ([] if self.at() == new else list(new.values())))
        assert placing.unfinished(self.records) == []
        assert list(self.models.rglob(".localm-old-*")) == []


class TestAReplacementKilledAtAnyMoment:
    @pytest.mark.parametrize("how", WAYS)
    def test_every_file_is_there_and_it_is_put_right(self, tmp_path, monkeypatch, how):
        """Every state, and every state a recovery killed in turn leaves."""
        states = undone = done = 0
        for nth in range(200):
            lib = _Library(tmp_path / f"{nth}")
            if not _killed_after(monkeypatch, nth, how, lib.place):
                break
            states += 1
            lib.check_all_there()
            for again in range(200):
                twice = _Library.__new__(_Library)
                root = tmp_path / f"{nth}-{again}"
                _copy_with_links(tmp_path / f"{nth}", root)
                twice.models, twice.records = root / "models", root / "locks" / "replacing"
                twice.base = twice.models / "snap"
                twice.stage = twice.base / placing.REPO_STAGING
                cut = _killed_after(monkeypatch, again, how, twice.put_back)
                twice.check_all_there()
                assert twice.put_back() is True
                twice.check_settled()
                shutil.rmtree(root)
                if not cut:
                    break
            with monkeypatch.context() as m:
                _use(m, how)
                assert lib.put_back() is True
            lib.check_settled()
            if lib.at() == {n: f"new {n}" for n in NAMES}:
                done += 1
            else:
                undone += 1
            shutil.rmtree(tmp_path / f"{nth}")
        assert lib.at() == {n: f"new {n}" for n in NAMES}
        lib.check_settled()
        assert states > 15 and undone > 8 and done > 2

    def test_a_record_stays_while_the_model_folder_is_elsewhere(
            self, tmp_path, monkeypatch, caplog):
        lib = _Library(tmp_path)
        assert _killed_after(monkeypatch, 10, "exclusive", lib.place)
        (kept,) = lib.base.glob(".localm-old-*")
        lib.base.rename(lib.models / "renamed")
        assert lib.put_back() is True
        assert len(placing.unfinished(lib.records)) == 1 and kept.name in caplog.text
        assert placing.being_replaced(lib.models, lib.records) == set()
        _file(lib.base / HAD[0], "another model")
        assert lib.put_back() is True and lib.at() == {HAD[0]: "another model"}
        assert len(placing.unfinished(lib.records)) == 1
        shutil.rmtree(lib.base)
        (lib.models / "renamed").rename(lib.base)
        assert lib.put_back() is True
        lib.check_settled()

    def test_models_that_cannot_be_reached_end_nothing(self, tmp_path, monkeypatch):
        lib = _Library(tmp_path)
        assert _killed_after(monkeypatch, 10, "exclusive", lib.place)
        lib.models.rename(tmp_path / "unplugged")
        assert lib.put_back() is False
        (tmp_path / "unplugged").rename(lib.models)
        assert lib.put_back() is True
        lib.check_settled()


def _copy_with_links(src: Path, dest: Path) -> None:
    seen: dict = {}
    for p in sorted(src.rglob("*")):
        to = dest / p.relative_to(src)
        if p.is_dir():
            to.mkdir(parents=True, exist_ok=True)
            continue
        to.parent.mkdir(parents=True, exist_ok=True)
        number = p.stat().st_ino
        if number in seen:
            os.link(seen[number], to)
        else:
            seen[number] = shutil.copy2(p, to)


class TestOneFileKilledAtAnyMoment:
    @pytest.mark.parametrize("how", WAYS)
    @pytest.mark.parametrize("over", [False, True])
    def test_the_bytes_have_a_name_and_the_next_pull_goes_on(
            self, tmp_path, monkeypatch, how, over):
        for nth in range(50):
            part = _file(tmp_path / str(nth) / "m.gguf.part", "new")
            dest = part.with_name("m.gguf")
            if over:
                _file(dest, "old")
            cut = _killed_after(monkeypatch, nth, how, partial(
                placing.put, part, dest, placing.stamp(dest)))
            there = [p.read_text() for p in (part, dest) if p.exists()]
            assert "new" in there and "" not in there
            assert not over or dest.read_text() in ("old", "new")
            placing.drop_second_name(part, dest)
            if not part.exists():
                assert dest.read_text() == "new"
            else:
                assert part.read_text() == "new" and dest.exists() is over
                placing.put(part, dest, placing.stamp(dest))
            assert dest.read_text() == "new"
            if not cut:
                break
        assert nth >= 1

    @pytest.mark.parametrize("copy", [False, True])
    def test_a_name_that_is_dropped_is_there_or_gone(self, tmp_path, monkeypatch, copy):
        for nth in range(50):
            keeper = _file(tmp_path / str(nth) / "models" / "m.gguf", "body")
            victim = tmp_path / str(nth) / "mine" / "m.gguf"
            victim.parent.mkdir()
            (shutil.copy2 if copy else os.link)(keeper, victim)
            cut = _killed_after(monkeypatch, nth, "exclusive", partial(
                placing.drop_name, victim, keeper, copy=copy))
            assert keeper.read_text() == "body"
            assert [p for p in victim.parent.iterdir() if p != victim] == []
            if not cut:
                break
        assert nth >= 1 and not victim.exists()


_REAL_KILL = """
import os, sys
from pathlib import Path
import localm.storekit as sk
from localm.model_manager import placing as p
root, nth = Path(sys.argv[1]), int(sys.argv[2])
models, base = root / "models", root / "models" / "snap"
left = [nth]
for module, name in ((p, "put_new"), (sk, "atomic_write")):
    def _call(*a, _real=getattr(module, name), **k):
        _real(*a, **k)
        left[0] -= 1
        if left[0] == 0:
            os._exit(3)
    setattr(module, name, _call)
names = sys.argv[3:]
p.place_files({n: base / p.REPO_STAGING / n for n in names}, base,
              {n: p.stamp(base / n) for n in names}, models, root / "locks" / "replacing")
"""
# The steps of it: the record, the old and new last file, the added one, the old and new
# first file, the record.
_LAST_IS_ASIDE, _LAST_IS_NEW, _FIRST_IS_ASIDE, _ALL_ARE_NEW, _IS_DONE = 2, 3, 5, 6, 7


def _really_killed(lib: _Library, nth: int) -> None:
    run = subprocess.run([sys.executable, "-c", _REAL_KILL, str(lib.models.parent), str(nth),
                          *NAMES], cwd=ROOT, stdin=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert run.returncode == 3


@pytest.fixture
def lib(home):
    return _Library(home)


class TestAReplacementThatWasKilled:
    @pytest.mark.parametrize("nth", [_LAST_IS_ASIDE, _LAST_IS_NEW, _FIRST_IS_ASIDE,
                                     _ALL_ARE_NEW, _IS_DONE])
    def test_the_next_scan_puts_it_right(self, home, printed, lib, nth):
        _really_killed(lib, nth)
        assert len(placing.unfinished(lib.records)) == 1
        mm.sync_models_dir(backfill_mmproj=False)
        lib.check_settled()
        assert (lib.at() == {n: f"new {n}" for n in NAMES}) is (nth == _IS_DONE)

    def test_discarding_the_download_puts_the_old_files_back_first(self, lib, printed):
        _really_killed(lib, _FIRST_IS_ASIDE)
        pull_mod.discard_partial_download("o/r", "snap")
        assert lib.at() == {n: f"old {n}" for n in HAD}
        assert not lib.stage.exists() and placing.unfinished(lib.records) == []

    def test_one_that_runs_is_left_alone_by_the_scan(self, lib, printed):
        _really_killed(lib, _FIRST_IS_ASIDE)
        with pull_mod._snapshot_dest_lock("snap"):
            mm.sync_models_dir(backfill_mmproj=False)
            assert len(placing.unfinished(lib.records)) == 1
        mm.sync_models_dir(backfill_mmproj=False)
        lib.check_settled()

    def test_the_parts_of_a_split_model_are_put_back_under_their_locks(
            self, home, printed, monkeypatch):
        models, records = home / "models", registry_mod.replacement_records()
        names = ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]
        held = {}
        for n in names:
            _file(models / n, f"old {n}")
            held[n] = _file(models / placing.FILE_STAGING / f"{n}.incomplete", f"new {n}")
        old = {n: placing.stamp(models / n) for n in names}
        assert _killed_after(monkeypatch, 7, "exclusive", lambda: placing.place_files(
            held, models, old, models, records))
        with pull_mod._part_lock(names[0]):
            mm.sync_models_dir(backfill_mmproj=False)
            assert len(placing.unfinished(records)) == 1
            assert placing.put_back(models, records, "", ["another.gguf"]) is True
            assert len(placing.unfinished(records)) == 1
        mm.sync_models_dir(backfill_mmproj=False)
        assert [(models / n).read_text() for n in names] == [f"old {n}" for n in names]
        assert placing.unfinished(records) == []

    def test_an_entry_whose_file_is_moved_aside_is_neither_missing_nor_pruned(
            self, home, printed, monkeypatch):
        models, records = home / "models", registry_mod.replacement_records()
        names = ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]
        held = {}
        for n in names:
            _gguf(models / n)
            held[n] = _file(models / placing.FILE_STAGING / f"{n}.incomplete", f"new {n}")
        mm._register("m", models / names[0])
        mm._register("other", _gguf(models / "other.gguf"))
        real = placing.put_new
        seen = []

        def _put(src, dest):
            if Path(dest) == models / names[0]:
                assert not dest.exists()
                mm.sync_models_dir(prune=True, backfill_mmproj=False)
                seen.append(dict(mm.load_registry()["m"]))
            return real(src, dest)
        monkeypatch.setattr(placing, "put_new", _put)
        with pull_mod._part_lock(names[0]), pull_mod._part_lock(names[1]):
            placing.place_files(held, models, {n: placing.stamp(models / n) for n in names},
                                models, records)
        assert len(seen) == 1 and "missing" not in seen[0]

    def test_a_file_of_somebody_else_at_the_name_is_kept_and_so_is_the_old_one(
            self, lib, printed, caplog):
        _really_killed(lib, _LAST_IS_ASIDE)
        (found,) = placing.unfinished(lib.records)
        taken = next(n for n in HAD if not (lib.base / n).exists())
        _file(lib.base / taken, "somebody's own")
        for _ in range(2):
            assert lib.put_back() is False
        assert (lib.base / taken).read_text() == "somebody's own"
        kept = placing._kept(lib.base, found.record.stem) / str(found.names.index(taken))
        assert kept.read_text() == f"old {taken}"
        assert caplog.text.count("could not be put back") == 1 and str(kept) in caplog.text
        (lib.base / taken).rename(lib.base.parent / "moved away")
        assert lib.put_back() is True
        assert lib.at(HAD) == {n: f"old {n}" for n in HAD}

    def test_while_it_cannot_be_put_back_nothing_is_deleted_or_placed(
            self, lib, printed, monkeypatch):
        _really_killed(lib, _LAST_IS_ASIDE)
        taken = next(n for n in HAD if not (lib.base / n).exists())
        _file(lib.base / taken, "somebody's own")
        before = sorted(lib.texts())
        with pytest.raises(pull_mod.PullInFlight, match="could not be put back"):
            pull_mod.discard_partial_download("o/r", "snap")
        _snapshot_hub(monkeypatch, downloaded := [])
        monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
        assert mm._pull_hf_snapshot("o/r", "snap", redownload=True) is False
        assert downloaded == [] and sorted(lib.texts()) == before
        assert any("could not be put back" in line for line in printed)

    def test_a_pull_of_a_split_model_stops_while_it_cannot_be_put_back(
            self, home, printed, monkeypatch):
        monkeypatch.setattr(placing, "put_back", lambda *a: False)
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda *a, **k: None)
        with pytest.raises(pull_mod.PullInFlight, match="could not be put back"):
            pull_mod.discard_partial_download("o/r:m.gguf")
        assert mm._pull_gguf_file("o/r:m.gguf", None) is False
        assert any("could not be put back" in line for line in printed)


class TestAReplacementOfAnotherFile:
    @pytest.mark.parametrize("when", ["before", "while it is moved"])
    def test_a_file_that_is_another_one_by_then_stays(self, lib, monkeypatch, when):
        real = placing.put_new
        target = lib.base / HAD[1]

        def _swap():
            os.unlink(target)
            _file(target, "somebody's own")

        def _put(src, dest):
            if Path(src) == target and when != "before":
                _swap()
            return real(src, dest)
        monkeypatch.setattr(placing, "put_new", _put)
        old = {n: placing.stamp(lib.base / n) for n in NAMES}
        if when == "before":
            _swap()
        with pytest.raises(FileExistsError):
            placing.place_files({n: lib.stage / n for n in NAMES}, lib.base, old,
                                lib.models, lib.records)
        assert lib.at() == {HAD[0]: f"old {HAD[0]}", HAD[1]: "somebody's own"}
        assert {f"new {n}" for n in NAMES} <= set(lib.texts())
        assert placing.unfinished(lib.records) == []

    def test_a_link_to_a_file_elsewhere_is_replaced_and_its_file_stays(self, lib, tmp_path):
        target = _file(tmp_path / "shared" / "weights", "shared")
        (lib.base / HAD[1]).unlink()
        try:
            (lib.base / HAD[1]).symlink_to(os.path.relpath(target, (lib.base / HAD[1]).parent))
        except OSError:
            pytest.skip("symlinks are not available here")
        lib.place()
        assert lib.at() == {n: f"new {n}" for n in NAMES} and target.read_text() == "shared"
        assert placing.unfinished(lib.records) == []


class _Split:
    """A split model in the models folder and its staged new version."""
    NAMES = ("m-00001-of-00002.gguf", "m-00002-of-00002.gguf")

    def __init__(self, home: Path) -> None:
        self.models, self.records = home / "models", registry_mod.replacement_records()
        self.held = {}
        for n in self.NAMES:
            _file(self.models / n, f"old {n}")
            self.held[n] = _file(self.models / placing.FILE_STAGING / f"{n}.incomplete", f"new {n}")

    def place(self) -> None:
        placing.place_files(self.held, self.models, {
            n: placing.stamp(self.models / n) for n in self.NAMES}, self.models, self.records)

    def put_back(self, *names) -> bool:
        return placing.put_back(self.models, self.records, "", names)

    def versions(self) -> list:
        return [(self.models / n).read_text().split()[0] if (self.models / n).exists() else None
                for n in self.NAMES]


@pytest.fixture
def split(home):
    return _Split(home)


def _refuse_reads(monkeypatch, times: int) -> list:
    """The next *times* reads of a record fail as while it is being replaced."""
    real, refused = Path.read_text, []

    def _read(self, *a, **k):
        if self.parent.name == "replacing" and len(refused) < times:
            refused.append(self)
            raise PermissionError(13, "in use")
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", _read)
    return refused


class TestAReplacementThatRuns:
    """No recovery touches it, whatever the moment it looks at the record."""

    @pytest.mark.parametrize("which, times", [
        ("another", 1), ("another", 1000), ("the same", 0), ("the same", 1)])
    def test_every_step_of_it_is_left_alone(self, split, monkeypatch, which, times):
        from localm import storekit
        monkeypatch.setattr(storekit, "_transient_backoff", lambda attempt: None)
        names = ["another.gguf"] if which == "another" else list(split.NAMES)
        answers = []

        def _after(real):
            def _call(*a, **k):
                out = real(*a, **k)
                before = split.versions()
                with monkeypatch.context() as m:
                    refused = _refuse_reads(m, times)
                    answers.append(split.put_back(*names))
                    assert bool(refused) is (times > 0)
                assert split.versions() == before
                return out
            return _call
        monkeypatch.setattr(placing, "put_new", _after(placing.put_new))
        monkeypatch.setattr(storekit, "atomic_write", _after(storekit.atomic_write))
        split.place()
        assert split.versions() == ["new", "new"] and placing.unfinished(split.records) == []
        waited = os.name == "nt" or not times
        if which == "another":
            assert answers == [times == 1 and waited] * 6
        else:
            assert answers == [False] * 5 + [waited]

    def test_two_pulls_of_different_models_never_mix_the_parts(self, split, monkeypatch):
        passed, aside, placed = threading.Event(), threading.Event(), []
        undo, put = placing._undo_one, placing.put_new
        first = split.models / split.NAMES[0]
        answer: list = []

        def _undo(final, *a):
            undo(final, *a)
            if threading.current_thread() is other and final == first:
                passed.set()
                aside.wait(10)

        def _put(src, dest):
            if threading.current_thread() is other:
                return put(src, dest)
            if Path(src) == first:
                other.start()
                while other.is_alive() and not passed.wait(0.02):
                    pass
            out = put(src, dest)
            placed.append(dest)
            if Path(src) == first:
                aside.set()
                other.join(20)
            return out

        def _other():
            with monkeypatch.context() as m:
                _refuse_reads(m, 1)
                answer.append(split.put_back("another.gguf"))
        other = threading.Thread(target=_other)
        monkeypatch.setattr(placing, "_undo_one", _undo)
        monkeypatch.setattr(placing, "put_new", _put)
        split.place()
        assert split.versions() == ["new", "new"] and answer == [os.name == "nt"]
        assert placing.unfinished(split.records) == []

    def test_a_record_that_changed_since_it_was_read_is_not_acted_on(self, split, monkeypatch):
        assert _killed_after(monkeypatch, 9, "exclusive", split.place)
        (found,) = placing.unfinished(split.records)
        text = found.record.read_text(encoding="utf-8")
        found.record.write_text(text.replace('"placing"', '"done"'), encoding="utf-8")
        monkeypatch.setattr(placing, "unfinished", lambda records: [found])
        before = sorted(p.read_text() for p in split.models.rglob("*") if p.is_file())
        assert split.put_back(*split.NAMES) is False
        assert sorted(p.read_text() for p in split.models.rglob("*") if p.is_file()) == before

    @pytest.mark.parametrize("alive", [True, False])
    def test_one_of_a_process_that_lives_is_left_alone(self, split, monkeypatch, alive):
        from localm import instances
        assert _killed_after(monkeypatch, 9, "exclusive", split.place)
        (found,) = placing.unfinished(split.records)
        rec = json.loads(found.record.read_text(encoding="utf-8"))
        pid = os.getppid()
        rec["owner"].update(pid=pid, start=instances.process_start_identity(pid))
        if not alive:
            monkeypatch.setattr(instances, "pid_alive", lambda p: False)
        found.record.write_text(json.dumps(rec), encoding="utf-8")
        before = split.versions()
        assert split.put_back(*split.NAMES) is not alive
        assert split.versions() == (before if alive else ["old", "old"])

    def test_a_process_that_gave_up_on_it_leaves_it_to_others(self, split, monkeypatch):
        real = placing.put_new

        def _put(src, dest):
            if Path(dest) == split.models / split.NAMES[0]:
                raise PermissionError(13, "in use")
            return real(src, dest)
        with monkeypatch.context() as m:
            m.setattr(placing, "put_new", _put)
            with pytest.raises(placing.NotPutBack):
                split.place()
        (found,) = placing.unfinished(split.records)
        assert found.placing and found.owner is None
        assert split.put_back(*split.NAMES) is True and split.versions() == ["old", "old"]


class TestRemovingAModelThatWasBeingReplaced:
    def _killed(self, split, monkeypatch):
        mm._register("m", split.models / split.NAMES[0])
        assert _killed_after(monkeypatch, 9, "exclusive", split.place)
        assert split.versions() == [None, "new"]

    def test_the_replacement_is_settled_first_and_nothing_comes_back(
            self, split, printed, monkeypatch):
        self._killed(split, monkeypatch)
        assert mm.remove_model("m") is True
        assert placing.unfinished(split.records) == []
        mm.sync_models_dir(backfill_mmproj=False)
        assert split.versions() == [None, None] and mm.load_registry() == {}

    def test_a_folder_too(self, lib, printed):
        _really_killed(lib, _FIRST_IS_ASIDE)
        mm._register("snap", lib.base)
        assert mm.remove_model("snap") is True
        assert not lib.base.exists() and placing.unfinished(lib.records) == []

    def test_it_is_refused_when_that_fails_or_a_pull_runs(self, split, printed, monkeypatch):
        self._killed(split, monkeypatch)
        before = split.versions()
        with pull_mod._part_lock(split.NAMES[1]):
            assert mm.remove_model("m") is False
        with monkeypatch.context() as m:
            m.setattr(placing, "_undo_one", _raise_in_use)
            assert mm.remove_model("m") is False
        assert split.versions() == before and "m" in mm.load_registry()
        assert any("could not be put back" in line for line in printed)

    def test_another_model_is_removed_as_ever(self, split, printed, monkeypatch):
        self._killed(split, monkeypatch)
        mm._register("other", _gguf(split.models / "other.gguf"))
        assert mm.remove_model("other") is True
        assert len(placing.unfinished(split.records)) == 1


class TestAReplacementThatIsDone:
    """What is left of it once every new file has its name is never an undo."""

    def test_kept_copies_that_cannot_be_deleted_stay_without_undoing_it(
            self, lib, printed, monkeypatch):
        with monkeypatch.context() as m:
            m.setattr(placing, "_unlink_read_only_too", _raise_in_use)
            lib.place()
            mm.sync_models_dir(backfill_mmproj=False)
            assert lib.put_back() is True
        assert lib.at() == {n: f"new {n}" for n in NAMES}
        (found,) = placing.unfinished(lib.records)
        assert not found.placing
        mm.sync_models_dir(backfill_mmproj=False)
        lib.check_settled()

    def test_a_record_that_stays_when_the_kept_copies_are_gone_undoes_nothing(
            self, lib, printed, monkeypatch):
        real = os.unlink

        def _unlink(path, *a, **k):
            if Path(path).suffix == ".json":
                raise PermissionError(13, "in use")
            return real(path, *a, **k)
        with monkeypatch.context() as m:
            m.setattr(os, "unlink", _unlink)
            lib.place()
            assert list(lib.base.glob(".localm-old-*")) == []
        mm.sync_models_dir(backfill_mmproj=False)
        pull_mod.discard_partial_download("o/r", "snap")
        lib.check_settled()
        assert lib.at() == {n: f"new {n}" for n in NAMES}

    def test_a_missing_kept_copy_never_takes_a_name_away(self, lib, printed, monkeypatch):
        with monkeypatch.context() as m:
            m.setattr(placing, "_recover", lambda *a: True)
            lib.place()
        (record,) = lib.records.glob("*.json")
        rec = json.loads(record.read_text(encoding="utf-8"))
        rec["phase"] = "placing"
        rec["files"] = [r for r in rec["files"] if r["had"]]
        record.write_text(json.dumps(rec), encoding="utf-8")
        shutil.rmtree(next(lib.base.glob(".localm-old-*")))
        assert lib.put_back() is True
        assert lib.at(HAD) == {n: f"new {n}" for n in HAD}


def test_a_repository_that_was_partly_replaced_is_not_called_unchanged(
        lib, printed, monkeypatch):
    real = placing.put_new
    _snapshot_hub(monkeypatch, [])
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    (lib.base / "model.safetensors").write_bytes(DAMAGED_WEIGHTS)
    mm._register("snap", lib.base, "hf:o/r")

    def _partly(src, dest):
        if Path(src) == lib.base / "config.json" or (
                Path(dest) == lib.base / "model.safetensors" and lib.stage not in Path(src).parents):
            raise PermissionError(13, "in use")
        return real(src, dest)
    monkeypatch.setattr(placing, "put_new", _partly)
    assert mm._pull_hf_snapshot("o/r", "snap", redownload=True) is False
    said = " ".join(printed)
    assert "could not all be put back" in said and ".localm-old-" in said
    assert "Nothing was replaced" not in said and "are unchanged" not in said


def _raise_in_use(*a):
    raise PermissionError(13, "in use")


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses to move an open file")
class TestAFileThatIsHeldOpen:
    @pytest.mark.parametrize("which", [*(f"old {n}" for n in HAD), *(f"new {n}" for n in NAMES)])
    def test_everything_goes_back_or_waits_for_it(self, lib, printed, which):
        kind, name = which.split(" ", 1)
        held = (lib.base if kind == "old" else lib.stage) / name
        with open(held, "rb"):
            with pytest.raises(OSError):
                lib.place()
            lib.check_all_there()
            lib.put_back()
            lib.check_all_there()
        assert lib.put_back() is True
        assert lib.at() == {n: f"old {n}" for n in HAD}
        lib.check_settled()

    def test_a_recovery_waits_for_it_in_every_state(self, tmp_path, monkeypatch):
        for nth in range(200):
            lib = _Library(tmp_path / str(nth))
            if not _killed_after(monkeypatch, nth, "exclusive", lib.place):
                break
            for path in [p for p in lib.models.rglob("*") if p.is_file()]:
                if not path.exists():
                    continue
                with open(path, "rb"):
                    lib.put_back()
                    lib.check_all_there()
            assert lib.put_back() is True
            lib.check_settled()
            shutil.rmtree(tmp_path / str(nth))


# ------------------------------------------------------------ records and reserved names

class TestARecord:
    def _write(self, lib, folder="snap", **row) -> Path:
        row = {"name": "config.json", "had": True, "new": [1, 2],
               "from": f"{placing.REPO_STAGING.as_posix()}/config.json", **row}
        lib.records.mkdir(parents=True, exist_ok=True)
        record = lib.records / f"{'ab' * 16}.json"
        record.write_text(json.dumps({"phase": "placing", "folder": folder, "files": [row]}),
                          encoding="utf-8")
        return record

    @pytest.mark.parametrize("field", ["name", "from"])
    @pytest.mark.parametrize("value", [
        "", "/etc/x", "C:/x", "C:x", "x:stream", "../x", "a/../../x", "a//x", "./x", "a\\x",
        None, 5, "a\0b"])
    def test_a_name_that_is_not_below_its_folder_is_refused(self, lib, field, value, caplog):
        if field == "from" and isinstance(value, str) and value:
            value = f"{placing.REPO_STAGING.as_posix()}/{value}"
        kept = _file(placing._kept(lib.base, "ab" * 16) / "0", "kept")
        record = self._write(lib, **{field: value})
        before = sorted(lib.texts())
        assert lib.put_back() is False
        assert record.exists() and kept.exists() and sorted(lib.texts()) == before
        assert "Nothing was changed" in caplog.text

    @pytest.mark.parametrize("value", ["config.json", ".cache/huggingface/x", "sub/x",
                                       ".cache/localm-staging-x/y"])
    def test_a_staged_file_outside_localms_staging_is_refused(self, lib, value):
        self._write(lib, **{"from": value})
        assert lib.put_back() is False

    @pytest.mark.parametrize("folder", ["..", "a/b", "a\\b", "."])
    def test_a_folder_that_is_not_in_the_models_folder_is_refused(self, lib, folder):
        self._write(lib, folder=folder)
        assert placing.put_back(lib.models, lib.records, folder) is False

    def test_a_folder_behind_a_link_out_of_the_model_is_refused(self, lib, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        try:
            (lib.base / "ln").symlink_to(outside, target_is_directory=True)
        except OSError:
            import _winapi
            _winapi.CreateJunction(str(outside), str(lib.base / "ln"))
        self._write(lib, name="ln/evil.pth")
        _file(placing._kept(lib.base, "ab" * 16) / "0", "evil")
        assert lib.put_back() is False and list(outside.iterdir()) == []

    def test_one_that_cannot_be_read_now_stops_the_caller(self, lib, monkeypatch):
        record = self._write(lib)
        real = Path.read_text

        def _read(self, *a, **k):
            if self == record:
                raise PermissionError(13, "in use")
            return real(self, *a, **k)
        monkeypatch.setattr(Path, "read_text", _read)
        assert lib.put_back() is False

    def test_a_damaged_one_changes_nothing(self, lib, printed):
        _really_killed(lib, _FIRST_IS_ASIDE)
        (found,) = placing.unfinished(lib.records)
        found.record.write_text("{ not json")
        before = sorted(lib.texts())
        mm.sync_models_dir(backfill_mmproj=False)
        assert sorted(lib.texts()) == before and found.record.exists()

    def test_it_is_written_before_anything_moves_and_ends_in_one_step(self, lib, monkeypatch):
        phases = []
        real = placing.put_new

        def _put(src, dest):
            (found,) = placing.unfinished(lib.records)
            phases.append((found.folder, sorted(found.names), found.placing))
            return real(src, dest)
        monkeypatch.setattr(placing, "put_new", _put)
        lib.place()
        assert phases == [("snap", sorted(NAMES), True)] * 5
        lib.check_settled()


def _asked_the_hub(*a, **k):
    raise AssertionError("the hub was asked for it")


class TestWhatAModelFolderHolds:
    """A repository can ship any file. None of them is read as an instruction."""

    def _hostile(self, folder: Path) -> Path:
        journal = {"../../../evil.pth": {"new": [0, 0, 0, 0], "from": "x"}}
        _file(folder / ".cache" / "huggingface" / "localm-replaced" / "ab" / "journal.json",
              json.dumps(journal))
        _file(folder / ".cache" / "huggingface" / "evil.pth", "evil")
        _file(folder / f".localm-old-{'ab' * 16}" / "0", "evil")
        _file(folder / ".localm-revision.json", json.dumps({"repo": "evil/evil"}))
        _file(folder / "locks" / "replacing" / f"{'ab' * 16}.json", json.dumps({
            "phase": "placing", "folder": "snap", "files": [{
                "name": "evil.pth", "had": True, "new": [1, 2],
                "from": f"{placing.REPO_STAGING.as_posix()}/evil.pth"}]}))
        return folder

    def test_a_scan_a_pull_and_a_discard_place_nothing_by_it(
            self, home, printed, monkeypatch):
        folder = self._hostile(home / "models" / "snap")
        (folder / "config.json").write_bytes(CONFIG)
        (folder / "model.safetensors").write_bytes(GOOD_WEIGHTS)
        mm._register("snap", folder, "hf:o/r")
        monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
        monkeypatch.setattr(pull_mod, "_resolve_snapshot_type", lambda *a, **k: "llm")
        _snapshot_hub(monkeypatch, [])
        before = {p for p in home.parent.rglob("*") if "models" not in p.relative_to(home.parent).parts}
        mm.sync_models_dir(backfill_mmproj=False)
        assert mm._pull_hf_snapshot("o/r", "snap", redownload=True) is True
        pull_mod.discard_partial_download("o/r", "snap")
        mm.sync_models_dir(backfill_mmproj=False)
        assert list(home.parent.rglob("evil.pth")) == [
            folder / ".cache" / "huggingface" / "evil.pth"]
        after = {p for p in home.parent.rglob("*") if "models" not in p.relative_to(home.parent).parts}
        assert not any("evil" in p.name for p in after - before)

    @pytest.mark.parametrize("first, listed", [(True, True), (True, False), (False, True)])
    def test_a_pull_does_not_download_what_lies_under_localms_names(
            self, home, printed, monkeypatch, first, listed):
        dest = home / "models" / "snap"
        shipped = {"config.json": CONFIG, "model.safetensors": GOOD_WEIGHTS,
                   ".cache/huggingface/localm-replaced/ab/journal.json": b"{}",
                   ".CACHE/x": b"x", ".localm-revision.json": b"{}", "sub/.localm-old-a/0": b"x"}
        siblings = [SimpleNamespace(rfilename=n, size=len(b), blob_id=None, lfs={
            "sha256": hashlib.sha256(b).hexdigest()}) for n, b in shipped.items()]

        class _Api:
            def __init__(self, *a, **kw):
                pass

            def model_info(self, repo_id, files_metadata=True):
                if not listed:
                    raise OSError("offline")
                return SimpleNamespace(siblings=siblings, sha="a" * 40)

            def repo_info(self, repo_id):
                return SimpleNamespace(sha="a" * 40)

            def list_repo_files(self, repo_id, revision=None):
                return list(shipped)

        from huggingface_hub.utils import filter_repo_objects
        got = []

        def _download(repo_id, local_dir, allow_patterns=None, ignore_patterns=None, **kw):
            for n in filter_repo_objects(shipped, allow_patterns=allow_patterns,
                                         ignore_patterns=ignore_patterns):
                got.append(n)
                p = Path(local_dir) / n
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(shipped[n])
            return local_dir
        monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", _download)
        monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
        monkeypatch.setattr(pull_mod, "_resolve_snapshot_type", lambda *a, **k: "llm")
        if not first:
            dest.mkdir()
            (dest / "model.safetensors").write_bytes(DAMAGED_WEIGHTS)
            mm._register("snap", dest, "hf:o/r")
        assert mm._pull_hf_snapshot("o/r", "snap") is True
        assert set(got) == {"config.json", "model.safetensors"}

    @pytest.mark.parametrize("name", [".cache", ".localm-incoming-x.1.0123abcd", "m.gguf.part",
                                      "m.gguf.part.json", "mmproj.gguf.origin.json"])
    def test_a_single_file_of_such_a_name_is_not_downloaded(
            self, home, printed, monkeypatch, name):
        monkeypatch.setattr(mm, "_hf_file_sha256", _asked_the_hub)
        assert mm._pull_gguf_file(f"o/r:{name}", None) is False
        assert any("Unsafe model filename" in line for line in printed)

    def test_nor_one_in_a_folder_of_such_a_name(self, home, printed, monkeypatch):
        monkeypatch.setattr(mm, "_hf_file_sha256", _asked_the_hub)
        assert mm._pull_gguf_file("o/r:.cache/huggingface/download/m.gguf", None) is False
        assert any("Unsafe repository path" in line for line in printed)

    def _model_dir(self, folder: Path) -> Path:
        self._hostile(folder)
        _file(folder / "config.json", '{"model_type": "llama", "architectures": ["L"]}')
        _file(folder / "model.safetensors", "weights")
        _file(folder / "sub" / ".localm-x", "x")
        return folder

    def test_a_copy_leaves_it_out(self, home, tmp_path, printed):
        src = self._model_dir(tmp_path / "mine" / "snap")
        assert mm.add_local(str(src), store="copy") is True
        dest = home / "models" / "snap"
        assert (dest / "model.safetensors").read_text() == "weights"
        assert not any(placing.reserved(p.relative_to(dest).as_posix()) for p in dest.rglob("*"))
        assert (src / ".localm-revision.json").exists()

    @pytest.mark.parametrize("renames", [True, False])
    def test_a_move_loses_none_of_it_and_places_nothing_by_it(
            self, home, tmp_path, printed, monkeypatch, renames):
        src = self._model_dir(tmp_path / "mine" / "snap")
        files = sorted(p.relative_to(src) for p in src.rglob("*") if p.is_file())
        if not renames:
            real = placing.put_new

            def _put(a, b):
                if Path(a) == src:
                    raise OSError(errno.EXDEV, "other volume")
                return real(a, b)
            monkeypatch.setattr(placing, "put_new", _put)
        assert mm.add_local(str(src), store="move") is True
        dest = home / "models" / "snap"
        mm.sync_models_dir(backfill_mmproj=False)
        assert sorted(p.relative_to(dest) for p in dest.rglob("*") if p.is_file()) == files
        assert not src.exists() and len(list(tmp_path.rglob("evil.pth"))) == 1


# ------------------------------------------------------------ partial downloads

class TestAPartialWithASecondName:
    """A file that has another name is never opened for writing."""

    def _two_names(self, home) -> tuple:
        dest = _file(home / "models" / "m.gguf", "a whole model")
        part = dest.with_name("m.gguf.part")
        os.link(dest, part)
        return dest, part

    def test_a_new_download_does_not_write_into_it(self, home):
        dest, part = self._two_names(home)
        with pull_mod._start_part(part, {"source": "url"}) as f:
            f.write(b"new")
        assert dest.read_text() == "a whole model" and part.read_text() == "new"

    def test_it_is_not_continued(self, home):
        dest, part = self._two_names(home)
        pull_mod._part_record_path(part).write_text('{"source": "url"}')
        assert pull_mod._part_is_resumable(part, {"source": "url"}) is False
        part.unlink()
        _file(part, "half")
        assert pull_mod._part_is_resumable(part, {"source": "url"}) is True

    def test_hugging_face_data_is_not_continued_from_it(self, home, tmp_path):
        dest, _part = self._two_names(home)
        inc = tmp_path / "abc.etag.incomplete"
        os.link(dest, inc.with_name("abc.etag.1.incomplete"))
        half = _file(inc.with_name("abc.etag.2.incomplete"), "half")
        assert pull_mod._partial_candidates(inc) == [half]
        out = tmp_path / "abc.etag.3.incomplete"
        assert pull_mod._adopt_partial(inc, out, None) == 4
        assert dest.read_text() == "a whole model" and out.read_text() == "half"


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
        ("_rename_dir", "os.rename"),              # lock folders
        ("_rename_noreplace", "os.rename"),        # lock folders
    }
    assert _direct_calls("registry.py") == {
        ("_backup_registry", "shutil.copy2"),      # registry.json to its backup
        ("_copy_verified", "shutil.copy2"),        # to a new temp name
        ("_transfer_model_dir", "shutil.copytree"),  # to a new temp name
    }


def test_a_partial_is_never_opened_so_that_it_is_emptied():
    """Every open for writing in the model manager, by function and mode."""
    out = set()
    for module in ("pull.py", "registry.py", "placing.py"):
        tree = ast.parse((ROOT / "localm" / "model_manager" / module).read_text(encoding="utf-8"))
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef):
                continue
            for call in ast.walk(func):
                if not (isinstance(call, ast.Call) and (
                        getattr(call.func, "id", None) == "open"
                        or getattr(call.func, "attr", None) == "open")):
                    continue
                modes = [a.value for a in call.args if isinstance(a, ast.Constant)
                         and isinstance(a.value, str)]
                modes += ["(variable)" for a in call.args if isinstance(a, ast.Name)
                          and a.id == "mode"]
                out |= {(func.name, m) for m in modes if m != "rb"}
    assert out == {
        ("_resumable_download_to_tmp_and_move", "(variable)"),   # "r+b" adopted, "xb" new
        ("_reclaim_guard", "a+b"), ("_write_lock_record", "x"),  # lock files
        ("_pull_url_locked", "ab"), ("_pull_civitai_file_locked", "ab"),  # _part_is_resumable
        ("_start_part", "xb"), ("_extract_gguf_member", "xb"),
    }
