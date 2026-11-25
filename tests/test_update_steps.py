# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm.update_mechanism.steps`` and the parts of an update: a component is
replaced whole or not at all (``test_update_steps_kills`` ends the process at each
place)."""

from __future__ import annotations

import errno
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from localm.update_mechanism import disk, job, mover, steps
from tests import _update_mechanism as um

#: A component before and after: changed, unchanged, added, removed, a name that
#: changes only its case, a folder that comes and one that goes, and a file that is
#: none of the component's.
OLD = {"main.py": "old main", "same.py": "same", "gone/a.py": "a", "Case.py": "case",
       "nodes/n.py": "old node", "user/mine.txt": "mine"}
NEW = {"main.py": "new main", "same.py": "same", "deep/er/b.py": "b", "case.py": "case 2",
       "nodes/n.py": "new node"}
LEAVES = ("gone/a.py", "Case.py")


def component(root: Path) -> Path:
    um.write(root, OLD)
    return root


def prepared(work: steps.Work, step: str = mover.TREE, files: dict | None = None) -> None:
    um.write(work.new(step), NEW if files is None else files)


def test_a_change_moves_the_new_files_in(tmp_path):
    root = component(tmp_path / "c")
    with steps.Work(root) as work:
        prepared(work)
        result = steps.change(work, [lambda: steps.move_in(root, work.id, mover.TREE,
                                                           remove=LEAVES)])
    assert (result.state, result.errors) == ("new", [])
    files = {k: v for k, v in um.tree(root).items() if v[0] == "file"}
    assert set(files) == set(NEW) | {"user/mine.txt"}
    assert (root / "main.py").read_text() == "new main"
    assert um.left(root) == []


def test_a_failed_proof_puts_the_old_files_back(tmp_path):
    root = component(tmp_path / "c")
    before = um.tree(root)
    with steps.Work(root) as work:
        prepared(work)
        with pytest.raises(steps.Failed) as failed:
            steps.change(work, [lambda: steps.move_in(root, work.id, mover.TREE, remove=LEAVES)],
                         prove=lambda: "it does not start")
    assert (failed.value.reason, failed.value.errors) == ("it does not start", [])
    assert um.tree(root) == before
    assert um.left(root) == []


def test_an_interrupt_during_a_change_puts_the_old_files_back(tmp_path):
    root = component(tmp_path / "c")
    before = um.tree(root)

    def stop():
        raise KeyboardInterrupt

    with steps.Work(root) as work, pytest.raises(KeyboardInterrupt):
        prepared(work)
        steps.change(work, [lambda: steps.move_in(root, work.id, mover.TREE, remove=LEAVES), stop])
    assert um.tree(root) == before


def test_equal_files_are_not_in_the_plan(tmp_path):
    root = component(tmp_path / "c")
    with steps.Work(root) as work:
        prepared(work)
        plan = steps.plan_staged(root, work.id, mover.TREE, remove=LEAVES)
    assert "same.py" not in [op.path for op in plan.ops]
    assert [op.path for op in plan.ops if op.kind == mover.REMOVE] == sorted(LEAVES)


def test_a_file_that_ends_in_control_z_is_moved_in_whole(tmp_path):
    root = component(tmp_path / "c")
    with steps.Work(root) as work:
        (work.new(mover.TREE) / "main.py").write_bytes(b"binary\x1a")
        steps.change(work, [lambda: steps.move_in(root, work.id, mover.TREE)])
    assert (root / "main.py").read_bytes() == b"binary\x1a"


def test_a_folder_in_the_place_of_a_new_file_is_refused(tmp_path):
    root = component(tmp_path / "c")
    um.write(root, {"deep/er/b.py/inner.txt": "the user's"})
    before = um.tree(root)
    with steps.Work(root) as work:
        prepared(work)
        with pytest.raises(steps.Failed, match="in the place of deep/er/b.py"):
            steps.change(work, [lambda: steps.move_in(root, work.id, mover.TREE,
                                                      remove=LEAVES)])
    assert um.tree(root) == before


@pytest.mark.skipif(os.name == "nt", reason="file modes")
def test_a_changed_execute_permission_is_a_change(tmp_path):
    root = component(tmp_path / "c")
    with steps.Work(root) as work:
        prepared(work, files={"same.py": "same"})
        (work.new(mover.TREE) / "same.py").chmod(0o755)
        steps.change(work, [lambda: steps.move_in(root, work.id, mover.TREE)])
    assert os.access(root / "same.py", os.X_OK)


# ------------------------------------------------------------------- killed

def test_settle_costs_nothing_and_creates_nothing_without_a_record(tmp_path):
    root = component(tmp_path / "c")
    assert steps.settle(root) is None
    assert not (root / ".localm-update").exists()


def test_settle_waits_for_a_change_that_is_moving_files(tmp_path, monkeypatch):
    root = component(tmp_path / "c")
    with steps.Work(root) as work, job.Lock(root) as lock:
        job.start(lock, work.id, old="", new="")
        work.recorded()
        job.advance(lock, job.MOVING)
        assert steps.settle(root, wait_s=0.2).state == "busy"
        job.advance(lock, job.STEPS)
        job.advance(lock, job.STARTING)
        assert steps.settle(root, wait_s=0.2) is None
        job.restore(lock)


def test_what_a_running_process_prepares_is_not_swept(tmp_path):
    root = component(tmp_path / "c")
    with steps.Work(root) as first:
        prepared(first)
        steps.sweep(root)
        assert (first.new(mover.TREE) / "main.py").exists()


def _interrupted(root: Path) -> None:
    """Leave the component *root* as a process does that ended while it changed it."""
    work = steps.Place(root, um.UPDATE_ID)
    with job.Lock(root) as lock:
        prepared(work)
        (work.path / "plans").mkdir()
        job.start(lock, work.id, old="1", new="2")
        job.advance(lock, job.MOVING)
        job.advance(lock, job.STEPS)
        steps.move_in(root, work.id, mover.TREE, remove=LEAVES)
    assert (root / "main.py").read_text() == "new main"


def _lose(root: Path) -> None:
    """Delete an old file of the change :func:`_interrupted` left."""
    (mover.area(root, um.UPDATE_ID, mover.TREE, 1).old / "main.py").unlink()


def _damage(root: Path) -> None:
    plan = next((root / ".localm-update" / um.UPDATE_ID / "plans").iterdir())
    plan.write_text("{", encoding="utf-8")


@pytest.mark.parametrize("harm", [_lose, _damage])
def test_replacing_the_whole_component_ends_a_change_that_cannot_be_undone(tmp_path, harm):
    root = component(tmp_path / "c")
    _interrupted(root)
    harm(root)
    assert steps.mend(root) is True
    with steps.Work(root) as work:
        prepared(work)
        with pytest.raises(steps.Failed, match="cannot be undone"):
            steps.change(work, [lambda: steps.move_in(root, work.id, mover.TREE)])
    assert job.read(root) is not None
    with steps.Work(root) as work:
        prepared(work, files={"main.py": "newest"})
        steps.change(work, [lambda: steps.move_in(root, work.id, mover.TREE)], whole=True)
    assert (root / "main.py").read_text() == "newest"
    assert um.left(root) == [] and steps.mend(root) is False


def test_a_change_that_can_be_undone_is_undone_before_the_component_is_replaced(tmp_path):
    root = component(tmp_path / "c")
    before = um.tree(root)
    _interrupted(root)
    assert steps.mend(root) is False
    assert um.tree(root) == before


@pytest.mark.skipif(os.name != "nt", reason="an open file can be moved")
def test_a_held_file_is_no_reason_to_give_an_interrupted_change_up(tmp_path, monkeypatch):
    monkeypatch.setattr(disk, "HELD_WAIT_S", 0.2)
    root = component(tmp_path / "c")
    before = um.tree(root)
    _interrupted(root)
    with open(root / "main.py", "rb"), pytest.raises(steps.Failed) as failed:
        steps.mend(root)
    assert failed.value.held
    assert steps.mend(root) is False and um.tree(root) == before


def test_a_part_of_an_update_is_not_given_up(tmp_path):
    root, part = _install_with_part(tmp_path)
    um.stage(root)
    with job.Lock(root) as lock:
        job.start(lock, um.UPDATE_ID, old="1", new="2")
        job.advance(lock, job.MOVING)
        job.advance(lock, job.STEPS)
        job.add_part(lock, part)
    with pytest.raises(steps.Failed, match="an update of localm is not finished"):
        steps.mend(part)
    with job.Lock(part) as lock, pytest.raises(job.Busy):
        job.abandon(lock)
    assert job.read(part) is not None


def test_somebody_is_asked_about_a_held_file_until_it_is_free(tmp_path, monkeypatch):
    if os.name != "nt":
        pytest.skip("an open file can be moved")
    monkeypatch.setattr(disk, "HELD_WAIT_S", 0.1)
    root = component(tmp_path / "c")
    asked: list = []
    with open(root / "main.py", "rb") as held:
        def ask(e: disk.Held) -> bool:
            asked.append(e.path)
            held.close()
            return True
        with steps.Work(root) as work:
            prepared(work)
            steps.change(work, [lambda: steps.move_in(root, work.id, mover.TREE, ask=ask)],
                         ask=ask)
    assert len(asked) == 1 and asked[0].endswith("main.py")
    assert (root / "main.py").read_text() == "new main"


def test_the_runtime_is_put_back_before_it_is_loaded(tmp_path, monkeypatch):
    from localm.inference.backends.llamacpp import _loader
    root = component(tmp_path / "localm_llama_runtime")
    _interrupted(root)
    monkeypatch.setitem(sys.modules, "localm_llama_runtime",
                        types.SimpleNamespace(LIB_DIR=str(root / "lib")))
    for found in (root / "lib", None):
        seen: list = []
        monkeypatch.setattr(_loader, "runtime_binary_dir", lambda found=found, seen=seen:
                            seen.append((root / "main.py").read_text()) or found)
        _loader._settled_binary_dir(None)
        assert seen == ["new main", "old main"]
        _interrupted(root)


def test_a_runtime_the_user_chose_is_neither_put_back_nor_blocked(tmp_path, monkeypatch):
    from localm.inference.backends.llamacpp import _loader
    root, chosen = component(tmp_path / "localm_llama_runtime"), tmp_path / "chosen"
    _interrupted(root)
    _lose(root)
    um.write(chosen, {_loader.lib_filename(): "x"})
    monkeypatch.setitem(sys.modules, "localm_llama_runtime",
                        types.SimpleNamespace(LIB_DIR=str(root / "lib"), lib_dir=lambda: None))
    monkeypatch.setattr(_loader, "_loaded_lib", None)
    monkeypatch.setenv("LLAMA_CPP_LIB", str(chosen / _loader.lib_filename()))
    with pytest.raises(Exception) as failed:
        _loader.load_lib()
    assert "could not be undone" not in str(failed.value)
    monkeypatch.delenv("LLAMA_CPP_LIB")
    with pytest.raises(RuntimeError, match="is lost.*localm setup-llama"):
        _loader.load_lib()


def test_a_runtime_whose_files_are_being_replaced_is_not_loaded(tmp_path, monkeypatch):
    from localm.inference.backends.llamacpp import _loader
    root = component(tmp_path / "localm_llama_runtime")
    _interrupted(root)
    monkeypatch.setitem(sys.modules, "localm_llama_runtime",
                        types.SimpleNamespace(LIB_DIR=str(root / "lib")))
    monkeypatch.setattr(_loader, "_loaded_lib", None)
    monkeypatch.setattr(_loader, "runtime_binary_dir", lambda: None)
    with job.Lock(root), pytest.raises(RuntimeError, match="are being replaced"):
        _loader.load_lib()


# -------------------------------------------------------------------- parts

def _install_with_part(tmp_path):
    root = um.install(tmp_path / "root")
    part = component(tmp_path / "data" / "c")
    return root, part


def test_a_part_of_a_running_update_is_not_put_back_by_its_user(tmp_path):
    root, part = _install_with_part(tmp_path)
    um.stage(root)
    with job.Lock(root) as lock:
        job.start(lock, um.UPDATE_ID, old="1", new="2")
        job.advance(lock, job.MOVING)
        job.advance(lock, job.STEPS)
        held = job.add_part(lock, part)
        um.write(mover.new_folder(part, um.UPDATE_ID, mover.TREE), NEW)
        steps.move_in(part, um.UPDATE_ID, mover.TREE, remove=LEAVES)
        job.advance(held, job.STARTING)
    # The updating process is gone, its record is there.
    assert steps.settle(part).state == "busy"
    assert (part / "main.py").read_text() == "new main"
    with job.Lock(root) as lock:
        result = job.restore(lock)
    assert result.errors == [] and result.notes == [
        "The update to 2 was undone. localm 1 is in place."]
    assert (part / "main.py").read_text() == "old main"


def test_an_install_that_moved_puts_its_parts_back(tmp_path):
    root = um.install(tmp_path / "root")
    part = component(root / "mydata" / "c")
    before = um.tree(root)
    um.stage(root)
    with job.Lock(root) as lock:
        job.start(lock, um.UPDATE_ID, old="1", new="2")
        job.advance(lock, job.MOVING)
        job.advance(lock, job.STEPS)
        held = job.add_part(lock, part)
        um.write(mover.new_folder(part, um.UPDATE_ID, mover.TREE), NEW)
        steps.move_in(part, um.UPDATE_ID, mover.TREE, remove=LEAVES)
        job.advance(held, job.STARTING)
    root = root.rename(tmp_path / "moved")
    part = root / "mydata" / "c"
    assert steps.settle(part).state == "busy"
    with job.Lock(root) as lock:
        assert job.restore(lock).errors == []
    skip = (".localm-update", ".localm-run", "mydata/c/.localm-update")
    assert um.tree(root, skip) == before


def test_a_part_that_cannot_be_reached_keeps_the_update_unfinished(tmp_path):
    root, part = _install_with_part(tmp_path)
    um.stage(root)
    with job.Lock(root) as lock:
        job.start(lock, um.UPDATE_ID, old="1", new="2")
        job.advance(lock, job.MOVING)
        job.advance(lock, job.STEPS)
        job.add_part(lock, part)
    part.parent.rename(tmp_path / "unplugged")
    with job.Lock(root) as lock:
        result = job.restore(lock)
    assert result.state == "failed" and "cannot be reached" in result.errors[0]
    (tmp_path / "unplugged").rename(part.parent)
    with job.Lock(root) as lock:
        assert job.restore(lock).state == "old"


def test_a_part_the_user_removed_does_not_keep_the_update_unfinished(tmp_path):
    root, part = _install_with_part(tmp_path)
    um.stage(root)
    with job.Lock(root) as lock:
        job.start(lock, um.UPDATE_ID, old="1", new="2")
        job.advance(lock, job.MOVING)
        job.advance(lock, job.STEPS)
        job.add_part(lock, part)
    shutil.rmtree(part)
    with job.Lock(root) as lock:
        result = job.restore(lock)
    assert (result.state, result.errors) == ("old", [])
    assert any("was removed" in note for note in result.notes)


def test_a_part_that_cannot_be_reached_at_the_end_stays_new(tmp_path):
    root, part = _install_with_part(tmp_path)
    um.stage(root)
    with job.Lock(root) as lock:
        job.start(lock, um.UPDATE_ID, old="1", new="2")
        job.advance(lock, job.MOVING)
        job.advance(lock, job.STEPS)
        held = job.add_part(lock, part)
        um.write(mover.new_folder(part, um.UPDATE_ID, mover.TREE), NEW)
        steps.move_in(part, um.UPDATE_ID, mover.TREE, remove=LEAVES)
        job.advance(held, job.STARTING)
        job.advance(lock, job.STARTING)
    part.rename(part.with_name("unplugged"))
    with job.Lock(root) as lock:
        assert job.finish(lock).errors == []
    assert job.read(root) is None
    part.with_name("unplugged").rename(part)
    assert steps.settle(part).state == "new"
    assert (part / "main.py").read_text() == "new main" and job.read(part) is None


def test_a_record_of_parts_that_is_damaged_changes_nothing(tmp_path):
    root, part = _install_with_part(tmp_path)
    um.stage(root)
    with job.Lock(root) as lock:
        job.start(lock, um.UPDATE_ID, old="1", new="2")
        job.advance(lock, job.MOVING)
        job.advance(lock, job.STEPS)
        job.add_part(lock, part)
        disk.write_json(mover.work_folder(root, um.UPDATE_ID) / job.PARTS, {"no": "list"})
    with job.Lock(root) as lock, pytest.raises(job.Unreadable):
        job.restore(lock)
    assert job.read(root) is not None


def test_a_plan_of_a_part_that_is_gone_after_its_moves_is_an_error(tmp_path):
    root = component(tmp_path / "c")
    _interrupted(root)
    work = mover.work_folder(root, um.UPDATE_ID)
    next((work / "plans").iterdir()).unlink()
    (work / "moved-01").unlink()
    with job.Lock(root) as lock, pytest.raises(job.Unreadable, match="a plan is missing"):
        job.settle(lock)


# ----------------------------------------------------- prepared and never started

def test_a_second_preparation_leaves_the_first_one_alone(tmp_path):
    root = component(tmp_path / "c")
    um.write(root, {".venv/pyvenv.cfg": "home = x\n"})
    with steps.Work(root) as first:
        staged = mover.venv_folder(root, first.id) / "stage" / "01"
        staged.mkdir(parents=True)
        with steps.Work(root) as second:
            assert staged.is_dir() and second.id != first.id
            with job.Lock(root) as lock:
                job.start(lock, second.id, old="", new="")
                assert staged.is_dir()
                job.restore(lock)
        assert staged.is_dir()
    assert not staged.exists()


_KILLED = """
import os, sys
sys.path.insert(0, sys.argv[1])
from localm.update_mechanism import job, mover, steps, steps_update
root, part, update_id = sys.argv[2:5]
work = steps_update.Preparing(root, update_id) if part else steps.Work(root)
work.__enter__()
work.new(mover.TREE).joinpath("big").write_text("x")
work.beside("lib.incoming").mkdir()
if part:
    job.own_part(root, update_id, part)
    steps.Place(part, update_id).new(mover.TREE).joinpath("big").write_text("x")
if len(sys.argv) > 5:
    steps.sweep(part)
    print(os.path.exists(os.path.join(part, ".localm-update", update_id)))
    sys.stdout.flush()
os._exit(9)
"""


def _killed(root, part="", *more):
    done = subprocess.run([sys.executable, "-c", _KILLED, str(um.REPO), str(root), str(part),
                           um.UPDATE_ID, *more], capture_output=True, text=True, timeout=120,
                          **um.NO_WINDOW)
    assert done.returncode == 9, done.stderr
    return done.stdout.strip()


def test_what_a_killed_preparation_left_is_deleted_by_the_next_one(tmp_path):
    root = component(tmp_path / "c")
    before = um.tree(root)
    _killed(root)
    assert um.left(root) != [] and (root / "lib.incoming").exists()
    with steps.Work(root):
        pass
    assert um.left(root) == [] and um.tree(root) == before


def test_what_a_killed_update_prepared_goes_from_the_install_and_its_parts(tmp_path):
    root, part = _install_with_part(tmp_path)
    before = um.tree(part)
    assert _killed(root, part, "look") == "True"
    assert um.left(part) == [um.UPDATE_ID]
    steps.sweep(part)
    assert um.left(part) == [] and um.tree(part) == before
    with job.Lock(root) as lock:
        job.cleanup(lock)
    assert um.left(root) == []


# ------------------------------------------------------------- the way back fails

def test_a_way_back_that_fails_is_said(tmp_path, monkeypatch):
    root = component(tmp_path / "c")
    real = job.restore
    with steps.Work(root) as work:
        prepared(work)

        def held(*_a, **_k):
            raise disk.Held(root / "state.json", "in use")
        monkeypatch.setattr(job, "restore", held)
        with pytest.raises(steps.Failed, match="it does not start") as failed:
            steps.change(work, [lambda: steps.move_in(root, work.id, mover.TREE, remove=LEAVES)],
                         prove=lambda: "it does not start")
        assert failed.value.errors and failed.value.held
    assert (root / "main.py").read_text() == "new main"
    monkeypatch.setattr(job, "restore", real)
    assert steps.settle(root).state == "old"


@pytest.mark.parametrize("where, state", [("_set_over", "old"), ("_set_phase", "new")])
def test_an_end_that_fails_is_an_old_or_a_new_component(tmp_path, monkeypatch, where, state):
    root = component(tmp_path / "c")
    real = getattr(job, where)

    def held(lock, record, what):
        if what == job.DONE:
            monkeypatch.setattr(job, where, real)
            raise disk.Held(root / "state.json", "in use")
        return real(lock, record, what)
    monkeypatch.setattr(job, where, held)
    with steps.Work(root) as work:
        prepared(work)

        def change():
            return steps.change(work, [lambda: steps.move_in(root, work.id, mover.TREE)])
        if state == "new":
            assert change().state == "new"
        else:
            with pytest.raises(steps.Failed) as failed:
                change()
            assert failed.value.errors == []
    assert (root / "main.py").read_text() == f"{state} main" and um.left(root) == []


def test_a_passing_error_gives_no_change_up(tmp_path, monkeypatch):
    monkeypatch.setattr(steps, "_RETRY_S", 0.01)
    root = component(tmp_path / "c")
    before = um.tree(root)
    _interrupted(root)
    real, failing = os.rename, [2]

    def rename(src, dst):
        if failing[0] and str(dst).endswith("main.py"):
            failing[0] -= 1
            raise OSError(errno.EIO, "I/O error")
        return real(src, dst)
    monkeypatch.setattr(os, "rename", rename)
    assert steps.mend(root) is False
    assert um.tree(root) == before


# ---------------------------------------------------------------- another volume

def across(monkeypatch, *inside: Path) -> None:
    """Make every rename fail that crosses the border of one of the folders *inside*."""
    tops = [os.path.realpath(p) for p in inside]

    def crossing(real):
        def call(src, dst, *rest):
            a, b = (os.path.realpath(disk.short_path(Path(p)).rpartition(os.sep)[0])
                    for p in (src, dst))
            if any((a + os.sep).startswith(t + os.sep) != (b + os.sep).startswith(t + os.sep)
                   for t in tops):
                raise OSError(errno.EXDEV, "another volume", str(src))
            return real(src, dst, *rest)
        return call
    monkeypatch.setattr(os, "rename", crossing(os.rename))
    monkeypatch.setattr(os, "replace", crossing(os.replace))
    monkeypatch.setattr(disk, "replace_through", crossing(disk.replace_through))


def test_a_file_and_a_folder_go_to_another_volume(tmp_path, monkeypatch):
    far = tmp_path / "far"
    um.write(tmp_path, {"file": "f", "folder/a": "a", "folder/deep/b": "b"})
    far.mkdir()
    across(monkeypatch, far)
    disk.move(tmp_path / "file", far / "file", disk.Wait())
    disk.move(tmp_path / "folder", far / "folder", disk.Wait())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["far"]
    assert {k for k, v in um.tree(far).items() if v[0] == "file"} == {
        "file", "folder/a", "folder/deep/b"}
    um.write(tmp_path, {"file": "newer"})
    disk.put_over(tmp_path / "file", far / "file", disk.Wait())
    assert (far / "file").read_text() == "newer" and not (tmp_path / "file").exists()


def test_a_copy_to_another_volume_that_differs_is_not_taken_for_the_file(tmp_path, monkeypatch):
    far = tmp_path / "far"
    far.mkdir()
    um.write(tmp_path, {"file": "f"})
    across(monkeypatch, far)
    real = shutil.copyfile

    def copy(src, dst, **kw):
        real(src, dst, **kw)
        Path(dst).write_text("x")
    monkeypatch.setattr(shutil, "copyfile", copy)
    with pytest.raises(OSError, match="not the same"):
        disk.move(tmp_path / "file", far / "file", disk.Wait())
    assert (tmp_path / "file").read_text() == "f"


@pytest.mark.parametrize("works", [True, False])
def test_a_component_with_a_folder_on_another_volume_is_replaced_or_put_back(
        tmp_path, monkeypatch, works):
    root = component(tmp_path / "c")
    before = um.tree(root)
    across(monkeypatch, root / "nodes", root / "gone")
    with steps.Work(root) as work:
        prepared(work)
        try:
            steps.change(work, [lambda: steps.move_in(root, work.id, mover.TREE, remove=LEAVES)],
                         prove=lambda: None if works else "no")
        except steps.Failed as e:
            assert (e.reason, e.errors) == ("no", [])
    assert (root / "nodes/n.py").read_text() == ("new node" if works else "old node")
    if not works:
        assert um.tree(root) == before
    assert um.left(root) == []


_CROSSING = """
import errno, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from localm.update_mechanism import disk, mover, steps
root, left, fail = Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "fail"
def far(p):
    return "nodes" in Path(disk.short_path(Path(p))).relative_to(root).parts[:1]
def crossing(real):
    def call(src, dst, *rest):
        if far(src) != far(dst):
            raise OSError(errno.EXDEV, "another volume")
        return real(src, dst, *rest)
    return call
os.rename, os.replace = crossing(os.rename), crossing(os.replace)
disk.replace_through = crossing(disk.replace_through)
def counted(real):
    def call(*a, **k):
        global left
        left -= 1
        if left < 0:
            os._exit(9)
        done = real(*a, **k)
        return done
    return call
for name in ("copy_whole", "_compare"):
    setattr(disk, name, counted(getattr(disk, name)))
with steps.Work(root) as work:
    new = work.new(mover.TREE)
    for rel, text in (("nodes/n.py", "new node"), ("nodes/added.py", "added")):
        (new / rel).parent.mkdir(exist_ok=True)
        (new / rel).write_text(text)
    try:
        steps.change(work, [lambda: steps.move_in(root, work.id, mover.TREE)],
                     prove=lambda: "no" if fail else None)
    except steps.Failed:
        pass
"""


@pytest.mark.parametrize("end", ["fail", "works"])
def test_a_move_to_another_volume_that_is_killed_is_undone(tmp_path, end):
    killed = 0
    for at in range(40):
        root = component(tmp_path / f"{at}")
        before = um.tree(root)
        done = subprocess.run([sys.executable, "-c", _CROSSING, str(um.REPO), str(root), str(at),
                               end], capture_output=True, text=True, timeout=120, **um.NO_WINDOW)
        if done.returncode == 0:
            break
        assert done.returncode == 9, done.stderr
        killed += 1
        result = steps.settle(root)
        assert result.errors == [] and um.tree(root) == before, at
    assert killed >= 6 and done.returncode == 0
