# SPDX-License-Identifier: AGPL-3.0-or-later
"""The update mechanism's file step, record, lock and way back, with real files and
real processes."""

from __future__ import annotations

import ast
import errno
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from localm.update_mechanism import disk, job, mover
from tests import _update_mechanism as um
from tests._held_file import held

ID = um.UPDATE_ID
PACKAGE = um.REPO / "localm" / "update_mechanism"
windows_only = pytest.mark.skipif(os.name != "nt", reason="Windows share modes")
posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX modes")


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return um.applier(tmp_path_factory.mktemp("applier") / "applier")


@pytest.fixture
def root(tmp_path):
    return um.install(tmp_path / "root")


def job_for(root: Path, files: dict | None = None) -> dict:
    return {"do": "update", "root": str(root), "id": ID, "swap": list(um.SWAP),
            "new_files": um.stage(root) if files is None else files,
            "old_files": um.stated(um.OLD)}


def planned(root: Path, **kw):
    """A plan on disk for *root*, with the lock free again."""
    new_files = kw.pop("new_files", None) or um.stage(root)
    kw.setdefault("old_files", um.stated(um.OLD))
    kw.setdefault("swap", um.SWAP)
    with job.Lock(root) as lock:
        job.start(lock, ID, old="1", new="2")
        job.advance(lock, job.MOVING)
        plan = mover.plan_files(root, ID, new_files, protect=["MyData"], **kw)
        mover.write_plan(root, ID, plan)
    return plan


def undo(root: Path, **kw) -> job.Result:
    with job.Lock(root) as lock:
        return job.restore(lock, **kw)


# ---------------------------------------------------------------- the way there

def test_an_update_leads_to_exactly_the_new_release(root, tmp_path):
    plan = planned(root)
    mover.forward(root, ID, plan)
    with job.Lock(root) as lock:
        job.advance(lock, job.STEPS)
        job.advance(lock, job.STARTING)
        result = job.finish(lock)
    assert um.tree(root) == um.new_tree(tmp_path / "fresh")
    assert (result.state, result.errors, result.notes) == ("new", [], [])
    assert um.left(root) == []


def test_the_files_that_are_never_absent_go_first_and_only_changed_files_move(root):
    plan = planned(root)
    assert [op.path for op in plan.ops[:2]] == list(um.SWAP)
    assert all(op.swap for op in plan.ops[:2]) and not any(op.swap for op in plan.ops[2:])
    paths = {op.path for op in plan.ops}
    assert "localm/same.py" not in paths
    assert not paths & set(um.INTRUDING)
    assert {op.kind for op in plan.ops if op.path == "localm/gone.py"} == {"remove"}


def test_a_swapped_file_is_never_absent(root, monkeypatch):
    plan = planned(root)
    seen = []
    for name in ("rename", "replace"):
        real = getattr(os, name)

        def call(*args, _real=real, **kw):
            _real(*args, **kw)
            seen.append(all((root / rel).is_file() for rel in um.SWAP))
        monkeypatch.setattr(os, name, call)
    mover.forward(root, ID, plan)
    monkeypatch.undo()
    assert len(seen) > 10 and all(seen)


def test_without_the_installed_statement_nothing_is_removed(root):
    plan = planned(root, old_files=None)
    assert {op.kind for op in plan.ops} == {"replace", "add"}
    assert not any(op.edited for op in plan.ops)


def test_a_new_file_that_is_not_the_stated_one_is_refused(root):
    files = um.stage(root)
    (root / ".localm-update" / ID / "new" / "localm" / "changed.py").write_bytes(b"cut")
    before = um.tree(root)
    with pytest.raises(mover.Refused, match="localm/changed.py"):
        planned(root, new_files=files)
    assert um.tree(root) == before


def test_a_new_file_damaged_after_the_plan_takes_the_whole_step_back(root):
    before = um.tree(root)
    plan = planned(root)
    damaged = root / ".localm-update" / ID / "new" / "localm" / "newdir" / "deep" / "n.py"
    damaged.write_bytes(b"!")
    with pytest.raises(mover.StepFailed, match="newdir/deep/n.py") as e:
        mover.forward(root, ID, plan)
    assert um.tree(root) == before
    assert e.value.errors == []


def test_a_file_changed_after_the_plan_takes_the_whole_step_back(root):
    plan = planned(root)
    (root / "localm" / "changed.py").write_bytes(b"edited meanwhile")
    before = um.tree(root)
    with pytest.raises(mover.StepFailed, match="localm/changed.py changed while") as e:
        mover.forward(root, ID, plan)
    now = um.tree(root)
    assert now == before
    assert e.value.errors == []
    assert undo(root).state == "old" and um.tree(root) == before


def test_an_edited_file_is_kept_after_the_update(root):
    (root / "localm" / "changed.py").write_bytes(b"my version")
    (root / "localm" / "gone.py").write_bytes(b"my version too")
    plan = planned(root)
    assert {op.path for op in plan.ops if op.edited} == {"localm/changed.py", "localm/gone.py"}
    mover.forward(root, ID, plan)
    with job.Lock(root) as lock:
        job.advance(lock, job.STEPS)
        job.advance(lock, job.STARTING)
        result = job.finish(lock)
    kept = root / ".localm-update" / "kept" / ID / "localm"
    assert (kept / "changed.py").read_bytes() == b"my version"
    assert (kept / "gone.py").read_bytes() == b"my version too"
    assert [p.name for p in kept.iterdir()] == ["changed.py", "gone.py"]
    assert str(ID) in result.notes[0] and um.left(root) == ["kept"]


@windows_only
def test_an_edited_file_that_cannot_be_kept_is_not_deleted(root):
    (root / "localm" / "changed.py").write_bytes(b"my version")
    plan = planned(root)
    mover.forward(root, ID, plan)
    old = root / ".localm-update" / ID / "old" / "localm" / "changed.py"
    with job.Lock(root) as lock, held(old):
        job.advance(lock, job.STEPS)
        job.advance(lock, job.STARTING)
        wait, disk.HELD_WAIT_S = disk.HELD_WAIT_S, 0.2
        try:
            result = job.finish(lock)
        finally:
            disk.HELD_WAIT_S = wait
    assert result.state == "new" and any("stays in" in note for note in result.notes)
    assert old.read_bytes() == b"my version" and um.phase(root) is None
    with job.Lock(root) as lock:
        assert job.restore(lock).state == "none"
    assert old.read_bytes() == b"my version"


def test_a_file_in_the_place_of_a_new_folder_is_refused(root):
    files = um.stage(root, {**um.NEW, "localm/user.txt/x.py": "x"})
    before = um.tree(root)
    with pytest.raises(mover.Refused, match="user.txt is in the place of"):
        planned(root, new_files=files)
    assert um.tree(root) == before


def test_a_folder_with_a_user_file_in_the_place_of_a_new_file_is_refused(root):
    (root / "localm" / "other" / "mine.txt").write_bytes(b"mine")
    with pytest.raises(mover.Refused, match="mine.txt"):
        planned(root)
    assert (root / "localm" / "other" / "mine.txt").read_bytes() == b"mine"


# ------------------------------------------------------------- killed anywhere

def test_a_kill_at_every_operation_leaves_the_old_or_the_new_version(run_dir, tmp_path):
    """Every rename, replace, flush, folder and delete of a whole update, phase
    writes included."""
    count_root = um.install(tmp_path / "count" / "root")
    done = um.drive(run_dir, job_for(count_root), -1)
    assert done.returncode == 0, done.stderr
    points = int(done.stdout.split("POINTS")[1])
    assert points > 60
    new = um.new_tree(tmp_path / "fresh")
    assert um.tree(count_root) == new
    ends = set()
    for n in range(points):
        root = um.install(tmp_path / f"k{n}" / "root")
        old = um.tree(root)
        spec = job_for(root)
        killed = um.drive(run_dir, spec, n)
        assert killed.returncode != 0 and "POINTS" not in killed.stdout, (n, killed.stderr)
        reached = um.phase(root)
        ended = "FINISH" in killed.stdout and reached in (job.DONE, None)
        if ended:
            assert um.tree(root) == new, n
        again = um.restore(run_dir, root)
        assert again.returncode == 0, (n, reached, again.stderr)
        assert um.tree(root) == (new if ended else old), (n, reached)
        assert um.phase(root) is None
        if reached or ended:
            assert um.left(root) == [], (n, reached)
        once_more = um.restore(run_dir, root)
        assert once_more.returncode == 0 and um.tree(root) == (new if ended else old)
        ends.add((reached, ended))
        shutil.rmtree(root.parent)
    assert {e[0] for e in ends} >= {None, job.ENDING, job.MOVING, job.STEPS, job.STARTING, job.DONE}


def test_a_kill_at_every_operation_of_the_way_back_still_leads_back(run_dir, tmp_path):
    first = um.install(tmp_path / "first" / "root")
    old = um.tree(first)
    plan = planned(first)
    mover.forward(first, ID, plan)
    for root, stop in ((tmp_path / "half", 1), (tmp_path / "whole", 0)):
        shutil.copytree(first, root / "root", symlinks=True)
        if stop:
            files = um.new_tree(tmp_path / "unused")
            assert files
            for rel in ("localm/added.py", "localm/changed.py"):
                new = root / "root" / ".localm-update" / ID / "new" / rel
                os.rename(root / "root" / rel, new)
        template = root / "root"
        work = tmp_path / "count" / root.name / "root"
        shutil.copytree(template, work, symlinks=True)
        done = um.drive(run_dir, {"do": "restore", "root": str(work)}, -1)
        assert done.returncode == 0, done.stderr
        points = int(done.stdout.split("POINTS")[1])
        assert points > 20
        assert um.tree(work) == old
        for n in range(points):
            work = tmp_path / f"{root.name}{n}" / "root"
            shutil.copytree(template, work, symlinks=True)
            killed = um.drive(run_dir, {"do": "restore", "root": str(work)}, n)
            assert killed.returncode != 0 and "POINTS" not in killed.stdout, (n, killed.stderr)
            again = um.restore(run_dir, work)
            assert again.returncode == 0, (n, again.stderr)
            assert um.tree(work) == old, n
            assert um.phase(work) is None and um.left(work) == [], n
            shutil.rmtree(work.parent)


def test_the_way_back_decides_from_the_files_and_not_from_a_position(root):
    """Each file of a finished step in a state of its own choosing."""
    before = um.tree(root)
    plan = planned(root)
    work = root / ".localm-update" / ID
    for path in ("localm/changed.py", "localm/gone.py"):
        (work / "old" / path).parent.mkdir(parents=True, exist_ok=True)
        os.rename(root / path, work / "old" / path)
    (root / "localm" / "newdir" / "deep").mkdir(parents=True)
    os.rename(work / "new" / "localm" / "newdir" / "deep" / "n.py",
              root / "localm" / "newdir" / "deep" / "n.py")
    assert mover.back(root, ID, plan) == []
    assert um.tree(root) == before


def test_a_file_that_is_neither_version_is_kept_aside_and_named(root):
    before = um.tree(root)
    plan = planned(root)
    mover.forward(root, ID, plan)
    (root / "localm" / "added.py").write_bytes(b"written by somebody")
    (root / "localm" / "changed.py").write_bytes(b"and this one")
    result = undo(root)
    assert result.state == "old" and um.tree(root) == before
    kept = root / ".localm-update" / "kept" / ID / "localm"
    assert (kept / "added.py").read_bytes() == b"written by somebody"
    assert (kept / "changed.py").read_bytes() == b"and this one"
    assert any("kept" in note for note in result.notes)


def test_a_lost_old_file_is_named_and_the_record_stays(root):
    plan = planned(root)
    mover.forward(root, ID, plan)
    (root / ".localm-update" / ID / "old" / "localm" / "changed.py").unlink()
    result = undo(root)
    assert result.state == "failed"
    assert any("localm/changed.py" in e for e in result.errors)
    assert um.phase(root) == job.RESTORING


def test_a_damaged_old_file_is_not_put_back_as_the_old_version(root):
    plan = planned(root)
    mover.forward(root, ID, plan)
    (root / ".localm-update" / ID / "old" / "localm" / "changed.py").write_bytes(b"???")
    result = undo(root)
    assert result.state == "failed" and um.phase(root) == job.RESTORING
    assert (root / "localm" / "changed.py").read_bytes() == b"new"


# ----------------------------------------------------------------- held files

@windows_only
@pytest.mark.parametrize("rel", ["localm/changed.py", "localm.bat", "localm/gone.py"])
@pytest.mark.parametrize("share", [0, 1, 3], ids=["no-sharing", "read", "read-write"])
def test_a_held_file_takes_everything_back_and_names_the_holder(root, rel, share):
    before = um.tree(root)
    plan = planned(root)
    with held(root / rel, share=share):
        with pytest.raises(mover.StepFailed) as e:
            mover.forward(root, ID, plan, wait_s=0.3)
        assert Path(rel).name in str(e.value)
        assert f"pid {os.getpid()}" in str(e.value)
        assert e.value.errors == [] or share == 0
    assert mover.back(root, ID, plan) == []
    assert um.tree(root) == before


@windows_only
def test_a_file_held_with_delete_sharing_is_replaced_and_put_back(root, tmp_path):
    before = um.tree(root)
    plan = planned(root)
    with held(root / "localm" / "changed.py", share=7):
        mover.forward(root, ID, plan, wait_s=0.3)
        assert (root / "localm" / "changed.py").read_bytes() == b"new"
        assert mover.back(root, ID, plan) == []
    assert um.tree(root) == before


@windows_only
def test_a_file_let_go_within_the_wait_does_not_stop_the_update(root):
    plan = planned(root)
    holder = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c",
         "import sys, time; f = open(sys.argv[1], 'rb'); print('held', flush=True); time.sleep(1.5)",
         str(root / "localm" / "changed.py")], stdout=subprocess.PIPE, **um.NO_WINDOW)
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == b"held"
        mover.forward(root, ID, plan, wait_s=30)
    finally:
        holder.wait(timeout=30)
        assert holder.stdout is not None
        holder.stdout.close()
    assert (root / "localm" / "changed.py").read_bytes() == b"new"


@windows_only
def test_a_file_held_during_the_way_back_is_named_and_the_next_try_finishes(root):
    before = um.tree(root)
    plan = planned(root)
    mover.forward(root, ID, plan)
    with held(root / "localm" / "changed.py"):
        result = undo(root, wait_s=0.3)
        assert result.state == "failed" and len(result.errors) == 1
        assert "changed.py" in result.errors[0] and f"pid {os.getpid()}" in result.errors[0]
        assert um.phase(root) == job.RESTORING
    assert undo(root).state == "old"
    assert um.tree(root) == before and um.left(root) == []


# ------------------------------------------------------------------- disk full

def test_a_full_disk_at_every_write_leaves_the_old_version(tmp_path, monkeypatch):
    """Each flush and each folder fails once with ENOSPC, the way a full disk does."""
    def run(root: Path, fail_at: int) -> tuple:
        seen = [0]
        full = [False]

        def failing(real):
            def call(*args, **kw):
                n = seen[0]
                seen[0] += 1
                if n == fail_at:
                    full[0] = True
                    raise OSError(errno.ENOSPC, "No space left on device")
                return real(*args, **kw)
            return call
        error = None
        with monkeypatch.context() as m:
            for name in ("fsync", "mkdir"):
                m.setattr(os, name, failing(getattr(os, name)))
            try:
                plan = planned(root, new_files=um.stage(root))
                mover.forward(root, ID, plan)
            except (OSError, mover.StepFailed) as e:
                error = e
        return seen[0], full[0], error

    total, _full, error = run(um.install(tmp_path / "count" / "root"), -1)
    assert error is None and total > 15
    failed = 0
    for n in range(total):
        root = um.install(tmp_path / f"f{n}" / "root")
        before = um.tree(root)
        _seen, full, error = run(root, n)
        assert full, n
        if error is not None:
            failed += 1
            assert "No space left" in str(error), (n, error)
        if um.phase(root) is not None:
            undo(root)
        assert um.tree(root) == before, n
        assert um.phase(root) is None, n
        shutil.rmtree(root.parent)
    assert failed > 10


def test_a_full_disk_does_not_stop_the_way_back(root, monkeypatch):
    before = um.tree(root)
    plan = planned(root)
    mover.forward(root, ID, plan)
    reserve = root / ".localm-update" / ID / "reserve"
    assert reserve.stat().st_size >= 64 * 1024
    real = os.fsync

    def full(fd):
        if reserve.exists():
            raise OSError(errno.ENOSPC, "No space left on device")
        real(fd)
    monkeypatch.setattr(os, "fsync", full)
    result = undo(root)
    monkeypatch.undo()
    assert result.errors == [] and um.tree(root) == before


# ------------------------------------------------- paths, links, planted files

@pytest.mark.parametrize("bad", ["../out.txt", "/etc/x", "C:/x", "a/../../out.txt", "a//b", "",
                                 "a\\..\\b", "a/./b", "a/\x00b"])
def test_a_path_outside_the_install_is_refused(root, bad):
    files = um.stage(root)
    outside = root.parent / "out.txt"
    outside.write_bytes(b"not mine")
    digest = um.stated({"x": "x"})["x"]
    with pytest.raises(disk.Unsafe):
        planned(root, new_files={**files, bad: digest})
    with pytest.raises(disk.Unsafe):
        mover.plan_files(root, ID, files, {**um.stated(um.OLD), bad: digest}, protect=())
    with pytest.raises(disk.Unsafe):
        mover.plan_files(root, ID, files, protect=[bad])
    assert outside.read_bytes() == b"not mine"


@windows_only
@pytest.mark.parametrize("bad", ["a:stream", "localm/a.", "localm/a ", "localm/NUL", "home./x",
                                 "con/x"])
def test_a_windows_spelling_of_another_path_is_refused(root, bad):
    digest = um.stated({"x": "x"})["x"]
    with pytest.raises(disk.Unsafe):
        mover.plan_files(root, ID, {**um.stage(root), bad: digest}, protect=())


@windows_only
def test_the_short_name_of_a_folder_an_update_never_changes_is_refused(root):
    import ctypes
    (root / "node_modules").mkdir()
    text = ctypes.create_unicode_buffer(1024)
    assert ctypes.windll.kernel32.GetShortPathNameW(str(root / "node_modules"), text, 1024)
    short = Path(text.value).name
    if short.lower() == "node_modules":
        pytest.skip("this volume makes no short names")
    with pytest.raises(disk.Unsafe, match="not changed by an update"):
        mover.area(root, ID, mover.FILES).find(f"{short}/x.js")


@pytest.mark.parametrize("rel", ["home/models/m.gguf", "HOME/models/m.gguf", ".venv/x", ".git/HEAD",
                                 ".localm-update/lock",
                                 "runtime/localm_llama_runtime/lib/llama.dll",
                                 "Runtime/Localm_llama_runtime/LIB/llama.dll"])
def test_a_planted_plan_cannot_touch_what_an_update_never_changes(root, rel):
    before = um.tree(root)
    plan = planned(root)
    live = {"sha256": "0" * 64, "size": 1}
    plan.ops = [mover.Op(rel, mover.ADD, new=live)]
    mover.write_plan(root, ID, plan)
    result = undo(root)
    assert result.state == "failed" and "is not changed by an update" in result.errors[0]
    assert um.tree(root) == before


@pytest.mark.parametrize("change", [
    lambda raw: raw["ops"][0].update(path="../../out.txt"),
    lambda raw: raw["ops"][0].update(kind="run"),
    lambda raw: raw["ops"][0].update(extra="x"),
    lambda raw: raw["ops"].append(dict(raw["ops"][0])),
    lambda raw: raw["ops"][0].update(old={"sha256": "x", "size": 1}),
    lambda raw: raw.update(step="shell"),
    lambda raw: raw.update(format="localm-update-plan/9"),
    lambda raw: raw.update(program="calc"),
    lambda raw: raw.update(folders=["../.."]),
    lambda raw: raw.update(before={"files": {"a": 1}, "folders": []}),
], ids=["outside", "kind", "field", "twice", "identity", "step", "format", "top-field", "folder",
        "listing"])
def test_a_planted_plan_is_not_read(root, change):
    plan = planned(root)
    mover.forward(root, ID, plan)
    before = um.tree(root)
    path = root / ".localm-update" / ID / "plans" / "01-files.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    change(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((job.Unreadable, disk.Unsafe)):
        undo(root)
    assert um.tree(root) == before and um.phase(root) == job.MOVING


@pytest.mark.parametrize("text", [
    "", "{", "[]", '{"format": "localm-update-job/1"}',
    json.dumps({"format": "localm-update-job/1", "id": "../../x", "phase": "moving",
                "old": "1", "new": "2"}),
    json.dumps({"format": "localm-update-job/1", "id": ID.upper(), "phase": "moving",
                "old": "1", "new": "2"}),
    json.dumps({"format": "localm-update-job/1", "id": ID, "phase": "run", "old": "1",
                "new": "2"}),
    json.dumps({"format": "localm-update-job/1", "id": ID, "phase": "moving", "old": "1",
                "new": "2", "program": "calc"}),
    json.dumps({"format": "localm-update-job/1", "id": ID, "phase": "done", "old": "1",
                "new": "x" * 65}),
    "[" * 200000,
], ids=["empty", "cut", "list", "fields", "id", "upper-id", "phase", "extra", "long", "deep"])
def test_a_planted_record_changes_nothing(root, run_dir, text):
    before = um.tree(root)
    folder = root / ".localm-update"
    folder.mkdir()
    (folder / "job.json").write_text(text, encoding="utf-8")
    with pytest.raises(job.Unreadable):
        undo(root)
    done = um.restore(run_dir, root)
    assert done.returncode == 1 and "Traceback" not in done.stderr
    assert um.tree(root) == before and (folder / "job.json").read_text(encoding="utf-8") == text


def test_a_damaged_record_is_restored_from_the_only_unfinished_update(root):
    before = um.tree(root)
    plan = planned(root)
    mover.forward(root, ID, plan)
    (root / ".localm-update" / "job.json").write_bytes(b"\0" * 40)
    assert undo(root).state == "old"
    assert um.tree(root) == before and um.left(root) == []


def test_a_damaged_record_with_two_candidates_changes_nothing(root):
    plan = planned(root)
    mover.forward(root, ID, plan)
    before = um.tree(root)
    folder = root / ".localm-update"
    shutil.copytree(folder / ID, folder / ("f" * 32))
    (folder / "job.json").write_bytes(b"\0" * 40)
    with pytest.raises(job.Unreadable):
        undo(root)
    assert um.tree(root) == before


def test_a_record_whose_folder_is_missing_is_not_taken_for_a_restored_install(root):
    plan = planned(root)
    mover.forward(root, ID, plan)
    shutil.rmtree(root / ".localm-update" / ID)
    with pytest.raises(job.Unreadable, match="missing"):
        undo(root)
    assert um.phase(root) == job.MOVING


def link_folder(link: Path, target: Path) -> None:
    """A junction on Windows, a symbolic link elsewhere."""
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


def outside_folder(root: Path) -> Path:
    outside = root.parent / "outside"
    um.write(outside, {"changed.py": "old", "n.py": "not mine", "sub/y.py": "not mine"})
    return outside


@pytest.mark.parametrize("rel", ["localm", "localm/olddir", "localm/newdir"])
def test_a_link_to_a_folder_in_the_install_is_refused(root, rel):
    outside = outside_folder(root)
    there = um.tree(outside)
    path = root.joinpath(*rel.split("/"))
    if path.exists():
        shutil.rmtree(path)
    link_folder(path, outside)
    with pytest.raises(disk.Unsafe, match="is a link"):
        planned(root)
    assert um.tree(outside) == there


@pytest.mark.parametrize("rel", ["localm/olddir", "localm/newdir"])
def test_a_link_that_appears_after_the_plan_is_not_followed(root, rel):
    outside = outside_folder(root)
    there = um.tree(outside)
    plan = planned(root)
    path = root.joinpath(*rel.split("/"))
    if path.exists():
        path.rename(root.parent / "moved")
    link_folder(path, outside)
    with pytest.raises(mover.StepFailed, match="is a link"):
        mover.forward(root, ID, plan)
    assert um.tree(outside) == there


def test_a_link_to_a_file_in_the_install_is_refused(root):
    target = root.parent / "target.py"
    target.write_bytes(b"not mine")
    path = root / "localm" / "changed.py"
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError as e:
        pytest.skip(f"this account cannot make a symbolic link: {e}")
    with pytest.raises(disk.Unsafe, match="regular file"):
        planned(root)
    assert target.read_bytes() == b"not mine"


@pytest.mark.parametrize("rel", ["", ID, f"{ID}/old", f"{ID}/plans", f"{ID}/old/localm", "kept"])
def test_a_link_in_the_folders_of_the_update_is_not_followed(root, rel):
    plan = planned(root)
    mover.forward(root, ID, plan)
    outside = outside_folder(root)
    folder = root / ".localm-update"
    path = folder / rel if rel else folder
    if path.exists():
        path.rename(root.parent / "moved-away")
    link_folder(path, outside)
    there = um.tree(outside)
    try:
        result = undo(root)
    except (disk.Unsafe, job.Unreadable):
        pass
    else:
        assert result.state == ("none" if not rel else "failed") and (result.errors or not rel)
    assert um.tree(outside) == there
    assert (root / "VERSION").read_text(encoding="utf-8") in ("1", "2")


def test_deleting_what_an_update_left_does_not_follow_a_link(root):
    plan = planned(root)
    mover.forward(root, ID, plan)
    outside = outside_folder(root)
    there = um.tree(outside)
    link_folder(root / ".localm-update" / ID / "old" / "link", outside)
    link_folder(root / ".localm-update" / ID / "linked", outside)
    with job.Lock(root) as lock:
        job.advance(lock, job.STEPS)
        job.advance(lock, job.STARTING)
        job.finish(lock)
    assert um.tree(outside) == there


# ------------------------------------------------------------ read-only folder

def test_a_folder_that_may_not_be_changed_takes_everything_back(root):
    before = um.tree(root)
    plan = planned(root)
    folder = root / "localm" / "olddir"
    with no_write(folder):
        with pytest.raises(mover.StepFailed, match="x.py") as e:
            mover.forward(root, ID, plan, wait_s=0.2)
    assert e.value.errors == []
    assert um.tree(root) == before


class no_write:
    """Takes the right to change *folder* away for the block."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def _icacls(self, *args) -> None:
        from tests._win_acl import token_user_name
        from localm import sysprog
        subprocess.run([sysprog.find("icacls"), str(self.folder),
                        *[a.format(user=token_user_name()) for a in args]],
                       check=True, capture_output=True, **um.NO_WINDOW)

    def __enter__(self):
        if os.name == "nt":
            self._icacls("/deny", "{user}:(OI)(CI)(DE,DC,WD,AD)")
        elif os.geteuid() == 0:
            pytest.skip("root may change every folder")
        else:
            self.mode = stat.S_IMODE(self.folder.stat().st_mode)
            self.folder.chmod(0o555)

    def __exit__(self, *_exc):
        if os.name == "nt":
            self._icacls("/remove:d", "{user}")
        else:
            self.folder.chmod(self.mode)


@posix_only
def test_the_mode_of_a_file_survives_the_update_and_the_way_back(root):
    plan = planned(root)
    (root / ".localm-update" / ID / "new" / "localm.bat").chmod(0o755)
    before = um.tree(root)
    mover.forward(root, ID, plan)
    assert stat.S_IMODE((root / "localm.bat").stat().st_mode) == 0o755
    assert undo(root).state == "old" and um.tree(root) == before


# --------------------------------------------------------- processes, interface

def test_only_one_process_updates_an_install(root, run_dir, tmp_path):
    plan = planned(root)
    mover.forward(root, ID, plan)
    before = um.tree(root)
    spec = tmp_path / "hold.json"
    spec.write_text(json.dumps({"do": "hold", "root": str(root)}), encoding="utf-8")
    holder = subprocess.Popen([sys.executable, "-I", "-S", str(um.DRIVER), str(run_dir), str(spec),
                               "-1"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, **um.NO_WINDOW)
    assert holder.stdin is not None and holder.stdout is not None
    try:
        assert holder.stdout.readline().strip() == b"LOCKED"
        assert job.locked(root) is True
        busy = um.restore(run_dir, root)
        assert busy.returncode == 3 and "another process" in busy.stderr
        assert um.tree(root) == before and um.phase(root) == job.MOVING
        with pytest.raises(job.Busy):
            job.Lock(root, wait_s=0.1).__enter__()
        holder.kill()
        holder.wait(timeout=30)
        assert job.locked(root) is False
        assert um.restore(run_dir, root).returncode == 0
        assert um.phase(root) is None
    finally:
        holder.kill()
        holder.wait(timeout=30)
        holder.stdin.close()
        holder.stdout.close()


def test_two_restores_at_once_put_back_one_version(root, run_dir, tmp_path):
    before = um.tree(root)
    plan = planned(root)
    mover.forward(root, ID, plan)
    both = [subprocess.Popen([sys.executable, "-I", "-S", str(run_dir), "restore", str(root)],
                             stderr=subprocess.PIPE, **um.NO_WINDOW) for _ in range(2)]
    codes = []
    for p in both:
        p.communicate(timeout=120)
        codes.append(p.returncode)
    assert set(codes) <= {0, 3} and 0 in codes
    assert um.tree(root) == before and um.phase(root) is None


def test_the_waiting_probe_does_not_stop_an_update_that_begins(root):
    """`locked` holds the lock for a moment; taking it waits longer than that."""
    folder = root / ".localm-update"
    folder.mkdir()
    fd = disk.open_lock(folder / "lock")
    disk.lock_file(fd)
    import threading
    threading.Timer(0.4, os.close, [fd]).start()
    start = time.monotonic()
    with job.Lock(root) as lock:
        assert lock.held
    assert time.monotonic() - start >= 0.2


def test_nothing_is_created_by_asking_about_an_install_that_was_never_updated(root):
    before = um.tree(root, skip=())
    assert job.read(root) is None and job.locked(root) is False
    assert um.tree(root, skip=()) == before


def test_the_phases_only_go_forward(root):
    um.stage(root)
    with job.Lock(root) as lock:
        job.start(lock, ID, old="1", new="2")
        with pytest.raises(job.Busy):
            job.start(lock, "f" * 32, old="1", new="3")
        with pytest.raises(ValueError):
            job.advance(lock, job.STEPS)
        with pytest.raises(ValueError):
            job.finish(lock)
        for phase in (job.MOVING, job.STEPS, job.STARTING):
            job.advance(lock, phase)
            assert um.phase(root) == phase
        with pytest.raises(ValueError):
            job.advance(lock, job.MOVING)
        with pytest.raises(ValueError):
            job.advance(lock, job.DONE)
    with pytest.raises(ValueError, match="lock"):
        job.restore(lock)
    with pytest.raises(ValueError, match="lock"):
        job.advance(lock, job.STARTING)


def test_an_update_that_never_changed_anything_is_only_forgotten(root):
    um.stage(root)
    before = um.tree(root)
    with job.Lock(root) as lock:
        job.start(lock, ID, old="1", new="2")
        result = job.restore(lock)
    assert (result.state, result.errors, result.notes) == ("old", [], [])
    assert um.tree(root) == before and um.left(root) == []


# ----------------------------------------------------------------- the applier

def test_a_lazy_import_of_localm_raises_in_the_applier(tmp_path):
    """Run with localm importable (the repository is on the path the interpreter
    adds), so only the applier's own block can refuse it."""
    run_dir = um.applier(tmp_path / "applier")
    job_py = run_dir / "job.py"
    text = job_py.read_text(encoding="utf-8")
    probe = (f"    import sys; sys.path.insert(0, {str(um.REPO)!r})\n"
             "    try:\n        from localm import debuglog\n"
             "    except ImportError:\n        print('REFUSED', file=sys.stderr)\n"
             "    else:\n        print('IMPORTED', file=sys.stderr)\n"
             "    try:\n        import localm\n"
             "    except ImportError:\n        print('REFUSED', file=sys.stderr)\n"
             "    else:\n        print('IMPORTED', file=sys.stderr)\n")
    anchor = '    if not lock.held:\n        raise ValueError("the lock of the install is not held")\n    try:\n'
    assert text.count(anchor) == 1
    job_py.write_text(text.replace(anchor, probe + anchor), encoding="utf-8")
    root = um.install(tmp_path / "root")
    done = um.restore(run_dir, root)
    assert done.returncode == 0, done.stderr
    assert done.stderr.split() == ["REFUSED", "REFUSED"]


def test_the_applier_runs_from_a_copy_with_nothing_else_of_the_build(tmp_path):
    run_dir = um.applier(tmp_path / "x" / "applier")
    assert sorted(p.name for p in run_dir.parent.iterdir()) == sorted(
        ["update_mechanism", "_apply_update.py", "update_package.py", "updater_components.py"])
    root = um.install(tmp_path / "root")
    plan = planned(root)
    mover.forward(root, ID, plan)
    done = um.restore(run_dir, root)
    assert done.returncode == 0, done.stderr
    assert "localm 1 is in place" in done.stderr
    assert um.run([run_dir]).returncode == 2
    with pytest.raises(FileExistsError):
        um.applier(tmp_path / "x" / "applier")


def test_the_mechanism_imports_only_the_standard_library():
    """At module level and inside functions, in the folder and in what it loads at
    module level."""
    names = set(sys.stdlib_module_names)
    for path in sorted(PACKAGE.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                assert {a.name.split(".")[0] for a in node.names} <= names, (path.name, node.lineno)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert (node.module or "").split(".")[0] in names, (path.name, node.lineno)
    for name in ("_apply_update", "update_package", "updater_components"):
        body = ast.parse((um.REPO / "localm" / f"{name}.py").read_text(encoding="utf-8")).body
        for node in body:
            if isinstance(node, ast.Import):
                assert {a.name.split(".")[0] for a in node.names} <= names, (name, node.lineno)
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0 and (node.module or "").split(".")[0] in names, (
                    name, node.lineno)


def test_the_mechanism_is_in_the_wheel_and_the_release():
    import tomllib
    pyproject = tomllib.loads((um.REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert "localm" in pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert (PACKAGE / "__main__.py").is_file() and (PACKAGE / "__init__.py").is_file()


@windows_only
def test_a_path_longer_than_260_characters_is_moved_and_put_back(tmp_path):
    deep = "localm/" + "/".join(["d" * 60] * 4) + "/deep.py"
    root = tmp_path / "root"
    long_root = disk.long_path(root)
    um.write(long_root, {"VERSION": "1", deep: "old"})
    um.write(long_root / ".localm-update" / ID / "new", {"VERSION": "2", deep: "new"})
    plan = planned(root, new_files=um.stated({"VERSION": "2", deep: "new"}),
                   old_files=um.stated({"VERSION": "1", deep: "old"}))
    live = long_root.joinpath(*deep.split("/"))
    assert len(str(root / deep)) > 260
    mover.forward(root, ID, plan)
    assert live.read_bytes() == b"new"
    assert undo(root).state == "old"
    assert live.read_bytes() == b"old"
    shutil.rmtree(long_root)
