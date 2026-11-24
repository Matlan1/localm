# SPDX-License-Identifier: AGPL-3.0-or-later
"""The update mechanism never leaves a mix and never loses a file of the user: held
files, damaged records, lost files, a disk that stays full and a power cut."""

from __future__ import annotations

import builtins
import errno
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from localm.update_mechanism import disk, job, mover
from tests import _update_mechanism as um
from tests._held_file import held

ID = um.UPDATE_ID
OTHER_ID = "f" * 32
windows_only = pytest.mark.skipif(os.name != "nt", reason="Windows share modes")


@pytest.fixture
def root(tmp_path):
    return um.install(tmp_path / "root")


def planned(root: Path, *, old_files="statement", update_id: str = ID, new: dict | None = None):
    files = um.stage(root, new, update_id)
    if old_files == "statement":
        old_files = um.stated(um.OLD)
    with job.Lock(root) as lock:
        job.start(lock, update_id, old="1", new="2")
        job.advance(lock, job.MOVING)
        plan = mover.plan_files(root, update_id, files, old_files, swap=um.SWAP, protect=["mydata"])
        mover.write_plan(root, update_id, plan)
    return plan


def to_starting(root: Path) -> None:
    with job.Lock(root) as lock:
        job.advance(lock, job.STEPS)
        job.advance(lock, job.STARTING)


def finish(root: Path) -> job.Result:
    with job.Lock(root) as lock:
        return job.finish(lock)


def undo(root: Path, **kw) -> job.Result:
    with job.Lock(root) as lock:
        return job.restore(lock, **kw)


def everything(root: Path) -> set:
    return {p.read_bytes() for p in root.rglob("*") if p.is_file() and p.name != "lock"}


def hold(path: Path, seconds: float, *, share: int):
    """Another process that holds *path* with the Windows share mode *share*."""
    code = ("import ctypes, sys, time\n"
            "from ctypes import wintypes\n"
            "k = ctypes.WinDLL('kernel32'); k.CreateFileW.restype = wintypes.HANDLE\n"
            "k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, "
            "wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]\n"
            f"h = k.CreateFileW(sys.argv[1], 0x80000000, {share}, None, 3, 0, None)\n"
            "assert h != wintypes.HANDLE(-1).value\n"
            "print('held', flush=True)\n"
            f"time.sleep({seconds})\n")
    p = subprocess.Popen([um.PYTHON, "-I", "-S", "-c", code, str(path)], stdout=subprocess.PIPE,
                         **um.NO_WINDOW)
    assert p.stdout is not None and p.stdout.readline().strip() == b"held"
    return p


def let_go(p) -> None:
    p.wait(timeout=60)
    p.stdout.close()


# ---------------------------------------------------------- the record and DONE

def killed_after_done(root: Path, monkeypatch) -> None:
    """A finish that stops right after DONE is on disk and one file was handed over."""
    (root / "localm" / "changed.py").write_bytes(b"my version")
    plan = planned(root)
    mover.forward(root, ID, plan)
    to_starting(root)
    with monkeypatch.context() as m:
        def stop(*_a, **_kw):
            raise SystemExit("killed")
        m.setattr(job, "_close", stop)
        with pytest.raises(SystemExit):
            finish(root)
    assert um.phase(root) == job.DONE
    kept = root / ".localm-update" / "kept" / ID / "localm" / "changed.py"
    assert kept.read_bytes() == b"my version"


@windows_only
def test_a_record_held_for_a_moment_does_not_undo_a_finished_update(root, tmp_path, monkeypatch):
    killed_after_done(root, monkeypatch)
    p = hold(root / ".localm-update" / "job.json", 1.0, share=0)
    try:
        result = undo(root)
    finally:
        let_go(p)
    assert (result.state, result.errors) == ("new", [])
    assert um.tree(root) == um.new_tree(tmp_path / "fresh")
    assert b"my version" in everything(root)


@windows_only
def test_a_record_that_stays_held_is_blocked_and_not_damaged(root, monkeypatch):
    plan = planned(root)
    mover.forward(root, ID, plan)
    before = um.tree(root)
    monkeypatch.setattr(disk, "READ_WAIT_S", 0.3)
    with held(root / ".localm-update" / "job.json", exclusive=True):
        with pytest.raises(job.Blocked, match=f"pid {os.getpid()}"):
            undo(root)
    assert um.tree(root) == before and um.phase(root) == job.MOVING


@pytest.mark.parametrize("damage", [b"", b"{", b"\0" * 80, None], ids=["empty", "cut", "zeros", "gone"])
def test_a_damaged_record_of_a_finished_update_is_rebuilt_as_finished(root, tmp_path, monkeypatch,
                                                                      damage):
    killed_after_done(root, monkeypatch)
    record = root / ".localm-update" / "job.json"
    record.unlink() if damage is None else record.write_bytes(damage)
    result = undo(root)
    assert result.state == "new" and um.tree(root) == um.new_tree(tmp_path / "fresh")
    assert b"my version" in everything(root)


def test_a_finished_update_is_never_undone_by_any_phase_in_its_record(root, tmp_path, monkeypatch):
    killed_after_done(root, monkeypatch)
    record = root / ".localm-update" / "job.json"
    raw = json.loads(record.read_text(encoding="utf-8"))
    for phase in (job.MOVING, job.STEPS, job.STARTING, job.RESTORING):
        record.write_text(json.dumps({**raw, "phase": phase}), encoding="utf-8")
        with job.Lock(root) as lock:
            with pytest.raises(ValueError):
                job.advance(lock, job.STARTING)
    assert undo(root).state == "new"
    assert um.tree(root) == um.new_tree(tmp_path / "fresh")


@windows_only
def test_a_damaged_record_never_restores_an_earlier_update_that_ended(root, tmp_path, monkeypatch):
    """The first update ends while an edited file is held, so its folder stays."""
    (root / "localm" / "changed.py").write_bytes(b"my version")
    plan = planned(root)
    mover.forward(root, ID, plan)
    to_starting(root)
    monkeypatch.setattr(disk, "HELD_WAIT_S", 0.2)
    with held(root / ".localm-update" / ID / "old" / "localm" / "changed.py"):
        assert finish(root).state == "new"
    now = um.tree(root)
    um.stage(root, {**um.NEW, "VERSION": "3"}, OTHER_ID)
    with job.Lock(root) as lock:
        job.start(lock, OTHER_ID, old="2", new="3")
    (root / ".localm-update" / "job.json").write_bytes(b"\0" * 40)
    result = undo(root)
    assert result.errors == [] and um.tree(root) == now
    assert (root / "VERSION").read_text(encoding="utf-8") == "2"
    assert b"my version" in everything(root)


def test_an_update_whose_end_cannot_be_written_ends_at_the_next_try(root, tmp_path, monkeypatch):
    plan = planned(root)
    mover.forward(root, ID, plan)
    to_starting(root)
    real = job._write_state

    def full(lock, record):
        if record.phase == "ended":
            raise OSError(errno.ENOSPC, "No space left on device")
        real(lock, record)
    with monkeypatch.context() as m:
        m.setattr(job, "_write_state", full)
        result = finish(root)
        assert result.state == "new" and any("for now" in n for n in result.notes)
        assert um.phase(root) == job.DONE
    assert undo(root).state == "new"
    assert um.tree(root) == um.new_tree(tmp_path / "fresh") and um.left(root) == []


# ------------------------------------------------------------- files of the user

def test_without_an_installed_statement_every_file_that_differs_is_kept(root):
    (root / "localm" / "changed.py").write_bytes(b"my version")
    (root / "localm" / "added.py").write_bytes(b"my own file")
    new = {rel: t for rel, t in um.NEW.items() if not rel.startswith(("localm/o", "localm/t"))}
    plan = planned(root, old_files=None, new=new)
    mover.forward(root, ID, plan)
    to_starting(root)
    result = finish(root)
    assert (root / "localm" / "added.py").read_bytes() == b"added"
    kept = root / ".localm-update" / "kept" / ID
    assert (kept / "localm" / "changed.py").read_bytes() == b"my version"
    assert (kept / "localm" / "added.py").read_bytes() == b"my own file"
    assert (kept / "VERSION").read_bytes() == b"1"
    assert len(result.notes) == 1 and str(ID) in result.notes[0]


def test_a_lost_old_file_is_never_called_restored(root):
    plan = planned(root)
    mover.forward(root, ID, plan)
    (root / ".localm-update" / ID / "old" / "localm" / "changed.py").unlink()
    (root / "localm" / "changed.py").write_bytes(b"damaged")
    for _ in range(3):
        result = undo(root)
        assert result.state == "failed" and any("changed.py" in e and "lost" in e
                                                for e in result.errors)
    assert (root / "localm" / "changed.py").read_bytes() == b"damaged"
    assert (root / "VERSION").read_text(encoding="utf-8") == "1"


def test_a_lost_removed_file_is_never_called_restored(root):
    plan = planned(root)
    mover.forward(root, ID, plan)
    (root / ".localm-update" / ID / "old" / "localm" / "gone.py").unlink()
    (root / "localm" / "gone.py").write_bytes(b"something else")
    for _ in range(2):
        assert undo(root).state == "failed"


def test_a_lost_swapped_file_is_never_called_restored(root):
    plan = planned(root)
    mover.forward(root, ID, plan)
    (root / ".localm-update" / ID / "old" / "localm.bat").unlink()
    (root / "localm.bat").write_bytes(b"damaged")
    for _ in range(2):
        assert undo(root).state == "failed"


def test_a_file_that_became_a_folder_goes_back_past_what_the_new_version_wrote(root):
    before = um.tree(root)
    plan = planned(root)
    mover.forward(root, ID, plan)
    (root / "localm" / "thing" / "cache.txt").write_bytes(b"cache of the new version")
    (root / "localm" / "thing" / "deep").mkdir()
    (root / "localm" / "thing" / "deep" / "more.txt").write_bytes(b"more")
    result = undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert um.tree(root) == before
    kept = root / ".localm-update" / "kept" / ID / "localm" / "thing"
    assert (kept / "cache.txt").read_bytes() == b"cache of the new version"
    assert (kept / "deep" / "more.txt").read_bytes() == b"more"
    assert any("cache.txt" in n for n in result.notes)


def test_what_the_new_version_wrote_into_its_folders_stays_and_is_named(root):
    before = um.tree(root)
    plan = planned(root)
    mover.forward(root, ID, plan)
    extra = root / "localm" / "newdir" / "cache.txt"
    extra.write_bytes(b"cache")
    result = undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert extra.read_bytes() == b"cache"
    assert any("newdir" in n for n in result.notes)
    now = um.tree(root)
    assert {k: v for k, v in now.items() if not k.startswith("localm/newdir")} == before


def test_a_file_that_appears_in_the_place_of_a_new_one_stays(root):
    plan = planned(root)
    mine = root / "localm" / "added.py"
    mine.write_bytes(b"mine")
    before = um.tree(root)
    with pytest.raises(mover.StepFailed, match="already there") as e:
        mover.forward(root, ID, plan)
    assert e.value.errors == [] and any("added.py" in n for n in e.value.notes)
    assert um.tree(root) == before
    assert undo(root).state == "old" and um.tree(root) == before


@windows_only
def test_a_read_only_swapped_file_is_replaced_and_put_back(root, tmp_path):
    launcher = root / "localm.bat"
    launcher.chmod(stat.S_IREAD)
    before = um.tree(root)
    plan = planned(root)
    mover.forward(root, ID, plan, wait_s=0.5)
    assert launcher.read_bytes() == b"start 2"
    result = undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert um.tree(root) == before
    assert not launcher.stat().st_mode & stat.S_IWRITE
    launcher.chmod(stat.S_IWRITE)


# --------------------------------------------------------- held for a moment

@windows_only
@pytest.mark.parametrize("rel", ["localm/changed.py", "localm/newdir/deep/n.py", "localm.bat"])
def test_a_new_file_held_for_a_moment_does_not_fail_the_update(root, tmp_path, rel):
    plan = planned(root)
    p = hold(root / ".localm-update" / ID / "new" / rel, 1.0, share=1)
    try:
        mover.forward(root, ID, plan, wait_s=30)
    finally:
        let_go(p)
    assert (root / rel).read_bytes() == um.NEW[rel].encode()


@windows_only
def test_a_new_file_that_stays_held_is_named_with_its_holder(root):
    before = um.tree(root)
    plan = planned(root)
    with held(root / ".localm-update" / ID / "new" / "localm" / "changed.py", share=1):
        with pytest.raises(mover.StepFailed, match=f"changed.py.*pid {os.getpid()}") as e:
            mover.forward(root, ID, plan, wait_s=0.3)
    assert e.value.errors == [] and um.tree(root) == before


@windows_only
def test_a_file_held_without_sharing_during_the_check_does_not_fail_the_update(root, monkeypatch):
    plan = planned(root)
    real = mover.check
    holders = []

    def check(a, plan, wait=None):
        holders.append(hold(root / "localm" / "changed.py", 1.0, share=0))
        return real(a, plan, wait)
    monkeypatch.setattr(mover, "check", check)
    try:
        mover.forward(root, ID, plan, wait_s=30)
    finally:
        let_go(holders[0])


@windows_only
def test_one_held_file_does_not_stop_the_others_from_going_back(root):
    before = um.tree(root)
    plan = planned(root)
    mover.forward(root, ID, plan)
    with held(root / "localm" / "changed.py", exclusive=True):
        out = mover.back(root, ID, plan, wait_s=0.3)
    assert len(out.errors) == 1 and "changed.py" in out.errors[0]
    now = um.tree(root)
    assert {k for k in before if before[k] != now.get(k)} == {"localm/changed.py"}
    assert mover.back(root, ID, plan).errors == [] and um.tree(root) == before


def test_an_interruption_by_the_user_goes_through_the_way_back(root, monkeypatch):
    before = um.tree(root)
    plan = planned(root)
    real = mover.check

    def interrupted(*a, **kw):
        real(*a, **kw)
        raise KeyboardInterrupt
    monkeypatch.setattr(mover, "check", interrupted)
    with pytest.raises(mover.StepFailed, match="KeyboardInterrupt") as e:
        mover.forward(root, ID, plan)
    assert e.value.errors == [] and um.tree(root) == before


# ------------------------------------------------------------- the disk stays full

class FullDisk:
    """From the write number *start* on, everything that needs space fails, until the
    reserve of the update is freed, which is room for *room* more."""

    def __init__(self, monkeypatch, root: Path, start: int, room: int = 400) -> None:
        self.reserve = root / ".localm-update" / ID / "reserve"
        self.start, self.room, self.seen, self.fired = start, room, 0, False
        for name in ("mkdir", "fsync"):
            monkeypatch.setattr(os, name, self._wrap(getattr(os, name)))

    def _wrap(self, real):
        def call(*args, **kw):
            n = self.seen
            self.seen += 1
            if 0 <= self.start <= n:
                if self.fired and not self.reserve.exists() and self.room > 0:
                    self.room -= 1
                else:
                    self.fired = True
                    raise OSError(errno.ENOSPC, "No space left on device")
            return real(*args, **kw)
        return call


def test_a_disk_that_stays_full_never_stops_the_way_back(tmp_path, monkeypatch):
    def run(root: Path, start: int):
        with monkeypatch.context() as m:
            disk_ = FullDisk(m, root, start)
            try:
                plan = planned(root)
                mover.forward(root, ID, plan)
            except (OSError, mover.StepFailed):
                pass
            if um.phase(root) is not None:
                result = undo(root)
                assert result.state == "old" and result.errors == [], (start, result)
            return disk_
    total = run(um.install(tmp_path / "count" / "root"), -1).seen
    assert total > 20
    fired = 0
    for start in range(total):
        root = um.install(tmp_path / f"f{start}" / "root")
        um.stage(root)
        before = um.tree(root)
        fired += run(root, start).fired
        assert um.tree(root) == before, start
        shutil.rmtree(root.parent)
    assert fired > total // 2


# --------------------------------------------------------------------- power cut

class Journal:
    """Records what a run does to the files below *root*, in order, and which of it
    is on disk: renames, deletes and folders in the order made, the content of a
    file only once it was flushed."""

    def __init__(self, monkeypatch, root: Path) -> None:
        self.root, self.log, self.paths, self.quiet = root, [], {}, False
        for name, kind in (("rename", "rename"), ("replace", "rename"), ("unlink", "unlink"),
                           ("mkdir", "mkdir"), ("rmdir", "rmdir")):
            monkeypatch.setattr(os, name, self._wrap(getattr(os, name), kind))
        monkeypatch.setattr(os, "fsync", self._fsync(os.fsync))
        monkeypatch.setattr(os, "open", self._open(os.open))
        monkeypatch.setattr(builtins, "open", self._builtin_open(builtins.open))
        monkeypatch.setattr(disk, "sync_disk", lambda: self.log.append(("barrier",)))
        monkeypatch.setattr(disk, "sync_folder", lambda folder: None)
        monkeypatch.setattr(disk, "replace_through", self._through)
        real_tree = disk.remove_tree

        def remove_tree(path):
            rel = self._rel(path)
            if self.quiet or rel is None:
                return real_tree(path)
            self.quiet = True
            try:
                return real_tree(path)
            finally:
                self.quiet = False
                self.log.append(("rmtree", rel))
        monkeypatch.setattr(disk, "remove_tree", remove_tree)

    def _rel(self, path):
        text = os.path.abspath(str(path))
        if text.startswith("\\\\?\\"):
            text = text[4:]
        try:
            rel = Path(text).relative_to(os.path.realpath(self.root)).as_posix()
        except ValueError:
            return None
        return None if rel in (".localm-update/lock",) else rel

    def _note(self, kind, *paths) -> None:
        rels = [self._rel(p) for p in paths]
        if not self.quiet and all(r is not None for r in rels):
            self.log.append((kind, *rels))

    def _wrap(self, real, kind):
        def call(*args, **kw):
            result = real(*args, **kw)
            if kind != "rename":
                self._note(kind, args[0])
            else:
                self._note(kind, args[0], args[1])
            return result
        return call

    def _through(self, src, dst) -> None:
        quiet, self.quiet = self.quiet, True
        os.replace(src, dst)
        self.quiet = quiet
        self._note("rename", src, dst)
        self.log.append(("barrier",))

    def _open(self, real):
        def call(path, flags, *args, **kw):
            rel = self._rel(path)
            fresh = rel is not None and flags & os.O_CREAT and not os.path.lexists(path)
            fd = real(path, flags, *args, **kw)
            self.paths[fd] = path
            if fresh:
                self._note("create", path)
            return fd
        return call

    def _builtin_open(self, real):
        def call(file, mode="r", *args, **kw):
            f = real(file, mode, *args, **kw)
            if isinstance(file, (str, Path)) and "w" in mode:
                self.paths[f.fileno()] = file
                self._note("create", file)
            return f
        return call

    def _fsync(self, real):
        def call(fd):
            real(fd)
            path = self.paths.get(fd)
            rel = self._rel(path) if path is not None else None
            if rel is not None and os.path.isfile(path):
                self.log.append(("content", rel, Path(path).read_bytes()))
        return call


def replay(template: Path, dest: Path, log: list, *, lost: int | None = None,
           unflushed: frozenset = frozenset()) -> None:
    """*dest* as it is after a power cut behind *log*, from the state *template*; the
    operation number *lost* never reached the disk. A file made since whose content
    was not flushed, and one of *unflushed*, holds zeros."""
    shutil.copytree(template, dest, symlinks=True)
    dirty = {dest / rel for rel in unflushed}
    missing: set = set()

    def forget(path: Path, new: Path | None = None) -> None:
        for p in [p for p in dirty if p == path or path in p.parents]:
            dirty.discard(p)
            if new is not None:
                dirty.add(new / p.relative_to(path) if p != path else new)

    for n, entry in enumerate(log):
        kind = entry[0]
        if kind == "barrier" or n == lost:
            if n == lost and kind == "rename":
                missing.add(entry[2])
            continue
        path = dest / entry[1]
        if entry[1] in missing:
            if kind in ("unlink", "rename", "content"):
                continue
        if kind == "rename":
            target = dest / entry[2]
            if not os.path.lexists(path):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_dir():
                continue
            os.replace(path, target)
            forget(target)
            forget(path, target)
            missing.discard(entry[2])
        elif kind == "unlink":
            if path.is_file():
                path.chmod(stat.S_IWRITE | stat.S_IREAD)
                path.unlink()
            forget(path)
        elif kind == "mkdir":
            path.mkdir(parents=True, exist_ok=True)
        elif kind == "rmdir":
            if path.is_dir() and not os.listdir(path):
                path.rmdir()
        elif kind == "rmtree":
            shutil.rmtree(path, ignore_errors=True)
            forget(path)
        elif kind == "create":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
            dirty.add(path)
        elif kind == "content":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(entry[2])
            dirty.discard(path)
    for path in dirty:
        if path.is_file():
            size = path.stat().st_size
            path.chmod(stat.S_IWRITE | stat.S_IREAD)
            path.write_bytes(b"\0" * size)


def cuts(log: list):
    """Each prefix of *log*, and the whole of each stretch between two flushes with
    one of its operations lost."""
    for n in range(len(log) + 1):
        yield n, None
    start = 0
    for n, entry in enumerate([*log, ("barrier",)]):
        if entry[0] == "barrier":
            for lost in range(start, n):
                if log[lost][0] != "content":
                    yield n, lost
            start = n + 1


def mine(root: Path) -> None:
    (root / "localm" / "changed.py").write_bytes(b"my version")


def reached_done(log: list) -> bool:
    return any(e[0] == "content" and e[1].endswith("state.json.tmp")
               and b'"done"' in e[2] for e in log)


def test_a_power_cut_at_any_moment_leaves_the_old_or_the_new_version(tmp_path, monkeypatch):
    root = um.install(tmp_path / "root")
    mine(root)
    old = um.tree(root)
    staged = um.stage(root)
    template = tmp_path / "template"
    shutil.copytree(root, template, symlinks=True)
    unflushed = frozenset(p.relative_to(root).as_posix()
                          for p in (root / ".localm-update" / ID / "new").rglob("*") if p.is_file())
    with monkeypatch.context() as m:
        journal = Journal(m, root)
        with job.Lock(root) as lock:
            job.start(lock, ID, old="1", new="2")
            job.advance(lock, job.MOVING)
            plan = mover.plan_files(root, ID, staged, um.stated(um.OLD), swap=um.SWAP,
                                    protect=["mydata"])
            mover.write_plan(root, ID, plan)
            mover.forward(root, ID, plan)
            job.advance(lock, job.STEPS)
            job.advance(lock, job.STARTING)
            job.finish(lock)
    new = um.tree(root)
    log = journal.log
    assert len(log) > 100
    outcomes = set()
    for at, lost in cuts(log):
        dest = tmp_path / f"c{at}-{lost}"
        replay(template, dest, log[:at], lost=lost, unflushed=unflushed)
        result = undo(dest)
        assert result.errors == [], (at, lost, result, log[max(0, at - 3):at])
        now = um.tree(dest)
        assert now in (old, new), (at, lost, log[max(0, at - 3):at])
        if not reached_done(log[:at]):
            assert now == old, (at, lost)
        assert b"my version" in everything(dest), (at, lost, log[max(0, at - 3):at])
        assert undo(dest).state == "none"
        outcomes.add(now == new)
        shutil.rmtree(dest)
    assert outcomes == {True, False}


def test_a_power_cut_during_the_way_back_still_leads_back(tmp_path, monkeypatch):
    root = um.install(tmp_path / "root")
    mine(root)
    old = um.tree(root)
    plan = planned(root)
    mover.forward(root, ID, plan)
    template = tmp_path / "template"
    shutil.copytree(root, template, symlinks=True)
    with monkeypatch.context() as m:
        journal = Journal(m, root)
        assert undo(root).state == "old"
    log = journal.log
    assert len(log) > 40
    for at, lost in cuts(log):
        dest = tmp_path / f"c{at}-{lost}"
        replay(template, dest, log[:at], lost=lost)
        result = undo(dest)
        assert result.errors == [], (at, lost, result, log[max(0, at - 3):at])
        assert um.tree(dest) == old, (at, lost, log[max(0, at - 3):at])
        shutil.rmtree(dest)


def test_the_power_cut_model_sees_an_unflushed_write(tmp_path, monkeypatch):
    """A record written without a flush is zeros after the cut, so the model can fail."""
    root = um.install(tmp_path / "root")
    template = tmp_path / "template"
    shutil.copytree(root, template, symlinks=True)
    with monkeypatch.context() as m:
        journal = Journal(m, root)
        with open(root / "note.tmp", "w", encoding="utf-8") as f:
            f.write("text")
        os.replace(root / "note.tmp", root / "note")
    replay(template, tmp_path / "cut", journal.log)
    assert (tmp_path / "cut" / "note").read_bytes() == b""
