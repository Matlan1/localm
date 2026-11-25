# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm.update_mechanism.steps`` and the parts of an update: a component is
replaced whole or not at all (``test_update_steps_kills`` ends the process at each
place)."""

from __future__ import annotations

import os
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
    monkeypatch.setattr(_loader, "_loaded_lib", None)
    seen = []

    def binary_dir():
        seen.append((root / "main.py").read_text())
        raise RuntimeError("far enough")
    monkeypatch.setattr(_loader, "runtime_binary_dir", binary_dir)
    with pytest.raises(RuntimeError, match="far enough"):
        _loader.load_lib()
    assert seen == ["old main"]


def test_a_runtime_whose_files_are_being_replaced_is_not_loaded(tmp_path, monkeypatch):
    from localm.inference.backends.llamacpp import _loader
    root = component(tmp_path / "localm_llama_runtime")
    _interrupted(root)
    monkeypatch.setitem(sys.modules, "localm_llama_runtime",
                        types.SimpleNamespace(LIB_DIR=str(root / "lib")))
    monkeypatch.setattr(_loader, "_loaded_lib", None)
    monkeypatch.setattr(steps.settle, "__defaults__", (0.2,))
    with job.Lock(root), pytest.raises(RuntimeError, match="is being replaced"):
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
    part.rename(part.with_name("unplugged"))
    with job.Lock(root) as lock:
        result = job.restore(lock)
    assert result.state == "failed" and "cannot be reached" in result.errors[0]
    part.with_name("unplugged").rename(part)
    with job.Lock(root) as lock:
        assert job.restore(lock).state == "old"


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
