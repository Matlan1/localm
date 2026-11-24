# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the way back of the update mechanism takes for old, for lost and for the
user's, and what is checked before anything of an update is deleted."""

from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from localm.update_mechanism import disk, job, mover, packages
from tests import _update_mechanism as um
from tests import _update_mechanism_venv as uv
from tests._update_mechanism_venv import ALPHA1, ALPHA2, BETA, GAMMA, ID, SITE
from tests._held_file import held
from tests.test_update_mechanism_safety import planned, to_starting, windows_only

OTHER_ID = "f" * 32
#: A version of alpha that is neither the old nor the new one.
THEIRS = ("alpha", "1.5", {"alpha/__init__.py": "V = 15", "alpha/extra.py": "e", "alpha/new.py": "t"})
NEW_FILE = f".venv/{SITE}/alpha/new.py"
insensitive = pytest.mark.skipif(os.name != "nt", reason="a system that keeps case and ignores it")


@pytest.fixture
def root(tmp_path):
    return uv.installed(tmp_path / "root")


def site(root: Path) -> Path:
    return root / ".venv" / SITE


def packages_in(root: Path) -> list:
    return sorted(p.name for p in site(root).glob("*.dist-info"))


def everything(root: Path) -> set:
    return {p.read_bytes() for p in root.rglob("*") if p.is_file() and p.name != "lock"}


def moved(root: Path, **kw) -> mover.Plan:
    """A package step of which every move is made."""
    uv.begun(root)
    plan = uv.package_step(root, **kw)
    uv.forward(root, plan)
    return plan


def failed_once(root: Path, rel: str, monkeypatch) -> None:
    """A restore that fails because the added file *rel* cannot be moved."""
    real = mover.aside

    def aside(a, op, path, wait, out):
        if path.as_posix().endswith(rel):
            raise disk.Held(path, "in use")
        real(a, op, path, wait, out)
    with monkeypatch.context() as m:
        m.setattr(mover, "aside", aside)
        result = uv.undo(root)
    assert result.state == "failed" and len(result.errors) == 1, result


# -------------------------------------------------------- a venv made anew

@pytest.mark.parametrize("step", ["forward", "nothing"])
@pytest.mark.parametrize("anew", [(ALPHA1, BETA), (ALPHA2, GAMMA), (), None],
                         ids=["old", "new", "empty", "removed"])
def test_a_venv_that_was_made_anew_is_the_users(root, step, anew):
    if step == "forward":
        moved(root)
    else:
        uv.begun(root)
        uv.package_step(root)
    shutil.rmtree(root / ".venv")
    if anew is not None:
        for package in anew:
            uv.install(root / ".venv" if (root / ".venv").exists() else uv.venv(root), package)
        if not anew:
            uv.venv(root)
    mine = uv.tree(root)
    result = uv.undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert sum("stays as it is" in n for n in result.notes) == 1
    assert uv.tree(root) == mine and um.left(root) == []
    assert uv.undo(root).state == "none"


def test_a_venv_whose_old_files_alone_are_gone_is_not_one_made_anew(root):
    moved(root)
    shutil.rmtree(root / ".venv" / mover.VENV_FOLDER)
    assert uv.undo(root).state == "failed" and um.phase(root) == job.RESTORING


def test_a_link_to_a_venv_that_is_not_there_now_is_not_a_removed_venv(root, tmp_path):
    from tests.test_update_mechanism import link_folder
    (root / ".venv").rename(tmp_path / "elsewhere")
    link_folder(root / ".venv", tmp_path / "elsewhere")
    before = uv.tree(tmp_path / "elsewhere")
    moved(root)
    (tmp_path / "elsewhere").rename(tmp_path / "unplugged")
    result = uv.undo(root)
    assert result.state == "failed" and "leads nowhere" in result.errors[0]
    (tmp_path / "unplugged").rename(tmp_path / "elsewhere")
    assert uv.undo(root).state == "old"
    now = um.tree(tmp_path / "elsewhere", skip=(".lock", ".localm-old"))
    assert now == {k: v for k, v in before.items() if k != ".lock"}


# ------------------------------------------------------- lost and healed

@pytest.mark.parametrize("gone", ["alpha/data.txt", "alpha-1.0.dist-info/RECORD",
                                  "alpha-1.0.dist-info/METADATA"])
def test_a_lost_package_file_is_named_and_the_same_version_heals_it(root, gone):
    moved(root)
    (root / ".venv" / mover.VENV_FOLDER / ID / "old" / SITE / gone).unlink()
    for _ in range(2):
        result = uv.undo(root)
        assert result.state == "failed" and result.errors == [e for e in result.errors if "lost" in e]
        assert any("Install alpha 1.0 again" in e and gone in e for e in result.errors)
    uv.install(root / ".venv", ALPHA1)
    assert uv.undo(root).state == "old"
    assert packages_in(root) == ["alpha-1.0.dist-info", "beta-1.0.dist-info"]
    assert (site(root) / "alpha" / "data.txt").read_text(encoding="utf-8") == "one"
    assert um.left(root) == []


def test_everything_lost_is_healed_by_the_same_version_with_other_times(root):
    moved(root)
    shutil.rmtree(root / ".venv" / mover.VENV_FOLDER / ID / "old")
    result = uv.undo(root)
    assert result.state == "failed" and len(result.errors) == 1
    uv.install(root / ".venv", ALPHA1)
    (site(root) / "alpha-1.0.dist-info" / "INSTALLER").write_bytes(b"another installer")
    result = uv.undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert packages_in(root) == ["alpha-1.0.dist-info", "beta-1.0.dist-info"]


def test_half_of_the_old_version_does_not_heal_a_lost_package(root):
    moved(root)
    shutil.rmtree(root / ".venv" / mover.VENV_FOLDER / ID / "old")
    uv.install(root / ".venv", ALPHA1)
    (site(root) / "alpha" / "old.py").unlink()
    assert uv.undo(root).state == "failed"
    uv.install(root / ".venv", ALPHA1)
    assert uv.undo(root).state == "old"


def test_an_edited_package_file_that_is_lost_is_healed_by_the_release(root):
    (site(root) / "alpha" / "data.txt").write_bytes(b"MINE")
    moved(root)
    (root / ".venv" / mover.VENV_FOLDER / ID / "old" / SITE / "alpha" / "data.txt").unlink()
    assert uv.undo(root).state == "failed"
    uv.install(root / ".venv", ALPHA1)
    assert uv.undo(root).state == "old"


def test_an_edited_release_file_that_is_lost_is_healed_by_the_release(root):
    (root / "localm" / "changed.py").write_bytes(b"MINE")
    mover.forward(root, ID, planned(root))
    (root / ".localm-update" / ID / "old" / "localm" / "changed.py").unlink()
    for _ in range(2):
        result = uv.undo(root)
        assert result.state == "failed" and "Put that version" in result.errors[0]
    (root / "localm" / "changed.py").write_bytes(um.OLD["localm/changed.py"].encode())
    assert uv.undo(root).state == "old"


def test_a_file_of_the_user_that_is_lost_is_healed_by_any_file_but_the_new_one(root):
    (root / "localm" / "added.py").write_bytes(b"MINE")
    mover.forward(root, ID, planned(root))
    (root / ".localm-update" / ID / "old" / "localm" / "added.py").unlink()
    for _ in range(2):
        assert uv.undo(root).state == "failed"
    (root / "localm" / "added.py").write_bytes(b"MINE, from a backup")
    assert uv.undo(root).state == "old"
    assert (root / "localm" / "added.py").read_bytes() == b"MINE, from a backup"


def test_a_damaged_old_file_is_lost_until_the_old_content_is_back(root):
    mover.forward(root, ID, planned(root))
    (root / ".localm-update" / ID / "old" / "localm" / "changed.py").write_bytes(b"???")
    for _ in range(2):
        assert uv.undo(root).state == "failed"
    (root / "localm" / "changed.py").write_bytes(um.OLD["localm/changed.py"].encode())
    result = uv.undo(root)
    assert result.state == "old" and b"???" in everything(root)


def test_a_healed_state_stays_healed_while_something_else_fails(root, monkeypatch):
    moved(root)
    (root / ".venv" / mover.VENV_FOLDER / ID / "old" / SITE / "alpha" / "data.txt").unlink()
    failed = [disk.Held("x", "y")]

    def aside(a, op, path, wait, out):
        if failed and op.path.endswith("gamma.py"):
            raise failed[0]
        return real(a, op, path, wait, out)
    real = mover.aside
    monkeypatch.setattr(mover, "aside", aside)
    assert len(uv.undo(root).errors) == 2
    uv.install(root / ".venv", ALPHA1)
    assert len(uv.undo(root).errors) == 1
    (site(root) / "alpha" / "data.txt").write_bytes(b"edited after that")
    assert len(uv.undo(root).errors) == 1
    failed.clear()
    result = uv.undo(root)
    assert result.state == "old"
    assert (site(root) / "alpha" / "data.txt").read_bytes() == b"edited after that"


# ---------------------------------------------- the user's after a failed restore

def test_a_reinstall_after_a_failed_restore_is_the_users(root, monkeypatch):
    before = uv.tree(root)
    moved(root)
    failed_once(root, NEW_FILE, monkeypatch)
    uv.install(root / ".venv", ALPHA1)
    for path in site(root).glob("alpha/*.py"):
        os.utime(path, (1, 1))
    result = uv.undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert uv.tree(root) == before and um.left(root) == []


@pytest.mark.parametrize("version", ["1.5", "2.0"])
@pytest.mark.parametrize("new", ["theirs", ALPHA2[2]["alpha/new.py"]], ids=["theirs", "same"])
def test_another_version_after_a_failed_restore_is_the_users(root, monkeypatch, version, new):
    moved(root)
    failed_once(root, NEW_FILE, monkeypatch)
    theirs = ("alpha", version, {"alpha/__init__.py": "V = 'theirs'", "alpha/new.py": new})
    uv.install(root / ".venv", theirs)
    result = uv.undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert packages_in(root) == [f"alpha-{version}.dist-info", "beta-1.0.dist-info"]
    assert (site(root) / "alpha" / "new.py").read_bytes() == new.encode()
    assert (site(root) / "alpha" / "__init__.py").read_bytes() == b"V = 'theirs'"
    assert sum(n.startswith("The package alpha:") for n in result.notes) == 1


def test_a_package_removed_after_a_failed_restore_stays_removed(root, monkeypatch):
    moved(root)
    failed_once(root, NEW_FILE, monkeypatch)
    uv.uninstall(root / ".venv", "alpha")
    result = uv.undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert packages_in(root) == ["beta-1.0.dist-info"]
    assert sum(n.startswith("The package alpha:") for n in result.notes) == 1


@pytest.mark.parametrize("rel", ["localm/changed.py", "localm.bat", "localm/gone.py"])
@pytest.mark.parametrize("what", ["edit", "remove"])
def test_a_file_changed_after_a_failed_restore_is_the_users(root, monkeypatch, rel, what):
    mover.forward(root, ID, planned(root))
    failed_once(root, "localm/added.py", monkeypatch)
    assert (root / rel).read_bytes() == um.OLD[rel].encode()
    (root / rel).write_bytes(b"MINE") if what == "edit" else (root / rel).unlink()
    for _ in range(2):
        failed_once(root, "localm/added.py", monkeypatch)
    result = uv.undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert ((root / rel).read_bytes() == b"MINE") if what == "edit" else not (root / rel).exists()
    assert any(rel in n for n in result.notes)
    assert um.left(root) == []


@pytest.mark.parametrize("change", ["edit", "remove", "reinstall", "other", "uninstall"])
def test_what_the_user_did_before_the_first_move_is_the_users(root, change):
    uv.begun(root)
    uv.package_step(root)
    data = site(root) / "alpha" / "data.txt"
    {"edit": lambda: data.write_bytes(b"MINE"), "remove": data.unlink,
     "reinstall": lambda: uv.install(root / ".venv", ALPHA1),
     "other": lambda: uv.install(root / ".venv", ("alpha", "1.5", {"alpha/__init__.py": "15"})),
     "uninstall": lambda: uv.uninstall(root / ".venv", "alpha")}[change]()
    mine = uv.tree(root)
    result = uv.undo(root)
    assert (result.state, result.errors, uv.tree(root) == mine) == ("old", [], True)
    assert sum("alpha" in n for n in result.notes) == (change != "reinstall")


def killed_before(root: Path, rel: str, monkeypatch) -> None:
    """A package step that ends right before the move of the old file *rel*."""
    uv.begun(root)
    plan = uv.package_step(root)
    real = os.rename

    def stop(src, dst):
        if Path(src).as_posix().endswith(f"{SITE}/{rel}"):
            raise SystemExit("killed")
        real(src, dst)
    with monkeypatch.context() as m:
        m.setattr(os, "rename", stop)
        m.setattr(packages, "back", lambda *a, **kw: mover.Outcome())
        with pytest.raises(mover.StepFailed):
            uv.forward(root, plan)


@pytest.mark.parametrize("stop", ["alpha-1.0.dist-info/METADATA", "alpha-1.0.dist-info/RECORD",
                                  "alpha/old.py"])
def test_another_version_over_half_of_the_old_one_is_the_users_as_a_whole(root, monkeypatch, stop):
    killed_before(root, stop, monkeypatch)
    uv.install(root / ".venv", THEIRS)
    result = uv.undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert packages_in(root) == ["alpha-1.5.dist-info", "beta-1.0.dist-info"]
    assert packages._is_complete(site(root), root / ".venv", "alpha-1.5.dist-info", "alpha")
    assert not (site(root) / "alpha" / "old.py").exists()
    assert {p.read_bytes() for p in (root / ".venv" / mover.VENV_KEPT).rglob("*") if p.is_file()}


def test_a_package_removed_with_a_part_of_it_moved_stays_removed(root, monkeypatch):
    killed_before(root, "alpha-1.0.dist-info/METADATA", monkeypatch)
    uv.uninstall(root / ".venv", "alpha")
    result = uv.undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert packages_in(root) == ["beta-1.0.dist-info"] and not (site(root) / "alpha").exists()


def test_a_move_that_was_marked_and_not_made_is_not_lost(root, monkeypatch):
    killed_before(root, "alpha/data.txt", monkeypatch)
    (site(root) / "alpha" / "data.txt").write_bytes(b"MINE")
    failed = [disk.Held("x", "y")]
    move = disk.move

    def held(src, dst, wait, **kw):
        if failed and src.name == "__init__.py":
            raise failed[0]
        move(src, dst, wait, **kw)
    monkeypatch.setattr(disk, "move", held)
    for _ in range(2):
        result = uv.undo(root)
        assert result.errors == ["x cannot be used (y)"]
    failed.clear()
    result = uv.undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert (site(root) / "alpha" / "data.txt").read_bytes() == b"MINE"


def test_the_marks_of_a_step_are_only_what_the_engine_wrote(root):
    """Lines that were cut or planted say nothing."""
    plan = moved(root)
    path = root / ".localm-update" / ID / f"moved-{plan.number:02d}"
    text = path.read_bytes()
    path.write_bytes(text + b"+99999999\n+x\n-1 1\n" + b"+" + b"9" * 5000 + b"\n=")
    with mover.Marks(root, ID, plan) as marks:
        assert not marks.over and len(marks.changed) == len(plan.ops) + 1
        marks.note(".")
    path.write_bytes(b"\0" * len(text))
    assert uv.undo(root).state == "old"


# ---------------------------------------------------- the work is linear

def counted(real, seen: list):
    def call(*args, **kw):
        seen.append(1)
        return real(*args, **kw)
    return call


@pytest.mark.parametrize("state", ["user", "lost"])
def test_the_way_back_looks_at_each_file_a_fixed_number_of_times(root, monkeypatch, state):
    uv.begun(root)
    identity = {"size": 3, "mtime_ns": 5, "copy": "0" * 64}
    calls = {}
    for count in (500, 5000):
        ops = [mover.Op(f"{SITE}/big/f{n}.py", mover.REMOVE, identity, dist="big")
               for n in range(count)]
        plan = mover.Plan("packages", 1, ops, dists=["big"])
        if state == "lost":
            with mover.Marks(root, ID, plan) as marks:
                for n in range(count):
                    marks.note(f"+{n}")
                marks.note(".")
        seen: list = []
        with monkeypatch.context() as m:
            for owner, name in ((disk, "lstat"), (disk, "same"), (os, "lstat"), (os, "listdir")):
                m.setattr(owner, name, counted(getattr(owner, name), seen))
            out = packages.back(root, ID, plan)
        calls[count] = len(seen)
        assert len(out.notes) + len(out.errors) == 1
    assert calls[5000] < 11 * calls[500] and calls[5000] < 5000 * 60


# ------------------------------------------- nothing is deleted unchecked

def rewrite(root: Path, name: str, **changes) -> None:
    path = root / ".localm-update" / name
    path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), **changes}),
                    encoding="utf-8")


def half_moved(root: Path, monkeypatch) -> None:
    """Files that are out and none that is in."""
    plan = planned(root)
    with monkeypatch.context() as m:
        def stop(*_a, **_kw):
            raise SystemExit("killed")
        real = disk.move
        m.setattr(disk, "move", lambda src, dst, wait, **kw: (
            stop() if f"{os.sep}new{os.sep}" in str(src) else real(src, dst, wait, **kw)))
        m.setattr(mover, "_back", lambda *a, **kw: mover.Outcome())
        with pytest.raises(mover.StepFailed):
            mover.forward(root, ID, plan)
    assert not (root / "VERSION").exists()


@pytest.mark.parametrize("phase", ["ended", "done", "left", "ending"])
@pytest.mark.parametrize("record", [True, False], ids=["record", "no-record"])
@pytest.mark.parametrize("state", ["half", "moved", "nothing"])
def test_a_state_that_is_not_what_is_on_disk_deletes_nothing(root, monkeypatch, phase, record, state):
    before = uv.tree(root)
    if state == "half":
        half_moved(root, monkeypatch)
    else:
        plan = planned(root)
        if state == "moved":
            mover.forward(root, ID, plan)
    fits = {"half": (), "moved": ("done", "left"), "nothing": ("ended", "ending")}[state]
    rewrite(root, f"{ID}/state.json", phase=phase)
    if not record:
        (root / ".localm-update" / "job.json").unlink()
    now = uv.tree(root)
    if phase in fits:
        assert uv.undo(root).errors == [] and uv.tree(root) == now
        return
    if record or phase in ("done", "ending"):
        with pytest.raises(job.Unreadable, match="state.json says"):
            uv.undo(root)
    else:
        assert uv.undo(root).state == "none"
        with job.Lock(root) as lock:
            assert len(job.cleanup(lock)) == 1
    assert uv.tree(root) == now
    rewrite(root, f"{ID}/state.json", phase="moving")
    assert uv.undo(root).state == "old" and uv.tree(root) == before


@pytest.mark.parametrize("phase", ["ended", "done"])
def test_an_edited_state_of_a_package_step_deletes_nothing(root, phase):
    before = uv.tree(root)
    plan = moved(root, coming=[GAMMA], leaving=[])
    assert all(op.old is None for op in plan.ops)
    rewrite(root, f"{ID}/state.json", phase="ended")
    with pytest.raises(job.Unreadable):
        uv.undo(root)
    rewrite(root, f"{ID}/state.json", phase="steps")
    assert uv.undo(root).state == "old" and uv.tree(root) == before


def test_a_record_that_claims_more_than_the_state_does_not_decide(root):
    before = uv.tree(root)
    mover.forward(root, ID, planned(root))
    rewrite(root, "job.json", phase="done")
    assert uv.undo(root).state == "old" and uv.tree(root) == before


@pytest.mark.parametrize("damage", [b"", b"\0" * 60, b"{}"], ids=["empty", "zeros", "other"])
@pytest.mark.parametrize("record", ["fine", "gone", "damaged"])
def test_an_unreadable_state_beside_plans_is_refused(root, damage, record):
    before = uv.tree(root)
    mover.forward(root, ID, planned(root))
    (root / "localm" / "added.py").unlink()
    now = uv.tree(root)
    state = root / ".localm-update" / ID / "state.json"
    good = state.read_bytes()
    state.write_bytes(damage)
    path = root / ".localm-update" / "job.json"
    if record != "fine":
        path.unlink() if record == "gone" else path.write_bytes(b"\0" * 40)
    for _ in range(2):
        with pytest.raises(job.Unreadable):
            uv.undo(root)
    assert uv.tree(root) == now and (root / ".localm-update" / ID / "old").exists()
    state.write_bytes(good)
    assert uv.undo(root).state == "old" and uv.tree(root) == before


def test_plans_without_a_state_and_a_record_are_refused(root):
    mover.forward(root, ID, planned(root))
    (root / ".localm-update" / "job.json").unlink()
    (root / ".localm-update" / ID / "state.json").unlink()
    now = uv.tree(root)
    with pytest.raises(job.Unreadable):
        uv.undo(root)
    with job.Lock(root) as lock:
        assert len(job.cleanup(lock)) == 1
    assert uv.tree(root) == now


def test_a_finish_before_every_step_is_made_is_refused(root):
    plan = planned(root)
    to_starting(root)
    with job.Lock(root) as lock, pytest.raises(job.Unreadable):
        job.finish(lock)
    assert um.phase(root) == job.STARTING
    mover.forward(root, ID, plan)
    with job.Lock(root) as lock:
        assert job.finish(lock).state == "new"


def test_a_step_that_was_taken_back_does_not_stop_the_end(root, tmp_path, monkeypatch):
    """The packages fail and are put back; the update goes on without them."""
    with job.Lock(root) as lock:
        job.start(lock, ID, old="1", new="2")
        job.advance(lock, job.MOVING)
        files = um.stage(root)
        plan = mover.plan_files(root, ID, files, um.stated(um.OLD), swap=um.SWAP, protect=["mydata"])
        mover.write_plan(root, ID, plan)
        mover.forward(root, ID, plan)
        job.advance(lock, job.STEPS)
        made = uv.package_step(root)
        with pytest.raises(mover.StepFailed) as e:
            packages.forward(root, ID, made, ["uv"], env={}, run=lambda *a: (1, "no"))
        assert e.value.errors == []
        job.advance(lock, job.STARTING)
        assert job.finish(lock).state == "new"
    assert packages_in(root) == ["alpha-1.0.dist-info", "beta-1.0.dist-info"]
    assert um.left(root) == []


@pytest.mark.parametrize("rel", ["localm/changed.py", "localm/added.py", "localm/gone.py",
                                 f".venv/{SITE}/alpha/data.txt"])
@pytest.mark.parametrize("what", ["edit", "remove"])
def test_what_the_user_does_after_done_does_not_stop_the_end(root, monkeypatch, rel, what):
    um.stage(root)
    uv.update(root, stop_before="finish")
    with monkeypatch.context() as m, job.Lock(root) as lock:
        m.setattr(job, "_close", lambda *a: None)
        job.finish(lock)
    assert um.phase(root) == job.DONE
    if what == "edit":
        (root / rel).write_bytes(b"MINE")
    elif (root / rel).exists():
        (root / rel).unlink()
    mine = uv.tree(root)
    assert uv.undo(root).state == "new" and uv.tree(root) == mine
    assert um.left(root) == []


# ------------------------------------------------- edited packages are kept

@pytest.mark.parametrize("size", [8, 5 * 1024 * 1024])
@pytest.mark.parametrize("age", [-10**9, 0, 10**9])
def test_an_edited_package_file_of_any_size_and_time_is_kept(root, size, age):
    content = "a" * size
    for path in (root / ".venv" / SITE).glob("alpha*"):
        shutil.rmtree(path)
    uv.install(root / ".venv", ("alpha", "1.0", {"alpha/lib.bin": content}))
    path = site(root) / "alpha" / "lib.bin"
    when = path.stat().st_mtime_ns
    edited = b"b" + content.encode()[1:]
    path.write_bytes(edited)
    os.utime(path, ns=(when + age, when + age))
    moved(root, coming=[ALPHA2])
    to_end(root)
    assert (root / ".venv" / mover.VENV_KEPT / ID / SITE / "alpha" / "lib.bin").read_bytes() == edited


def to_end(root: Path) -> None:
    with job.Lock(root) as lock:
        job.advance(lock, job.STARTING)
        assert job.finish(lock).state == "new"


# ------------------------------------------------- a folder that changes its case

CASEY1 = ("casey", "1.0", {"Casey/__init__.py": "1", "Casey/Sub/x.py": "x", "Casey/same.py": "s"})
CASEY2 = ("casey", "2.0", {"casey/__init__.py": "2", "casey/sub/x.py": "xx", "casey/same.py": "s"})


def names(root: Path) -> set:
    return {p.relative_to(site(root)).as_posix() for p in site(root).rglob("*")
            if ".dist-info" not in str(p)}


@pytest.mark.parametrize(("old", "new"), [(CASEY1, CASEY2), (CASEY2, CASEY1)], ids=["down", "up"])
def test_a_package_folder_that_changes_only_its_case_goes_there_and_back(root, old, new):
    uv.install(root / ".venv", old)
    (site(root) / old[0].capitalize() / "__pycache__").mkdir(parents=True, exist_ok=True)
    before, there = uv.tree(root), names(root)
    moved(root, coming=[new], leaving=["casey"])
    fresh = root.parent / "fresh"
    uv.install(uv.venv(fresh), new)
    assert names(root) - {"beta.py", "alpha"} >= {n for n in names(fresh) if "alpha" not in n}
    assert (uv.undo(root).state, uv.tree(root) == before) == ("old", True)
    assert {n for n in names(root) if "__pycache__" not in n} == {
        n for n in there if "__pycache__" not in n}


@insensitive
def test_a_folder_that_changed_its_case_goes_back_around_a_file_of_the_user(root):
    uv.install(root / ".venv", CASEY1)
    moved(root, coming=[CASEY2], leaving=["casey"])
    (site(root) / "casey" / "mine.txt").write_bytes(b"MINE")
    result = uv.undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert "Casey" in os.listdir(site(root)) and "Sub" in os.listdir(site(root) / "Casey")
    assert (site(root) / "Casey" / "mine.txt").read_bytes() == b"MINE"
    assert (site(root) / "Casey" / "__init__.py").read_bytes() == b"1"


@insensitive
def test_a_release_folder_that_changes_only_its_case_goes_there_and_back(tmp_path):
    old = {**um.OLD, "localm/Plug/a.py": "a1", "localm/Plug/b.py": "b"}
    new = {**um.NEW, "localm/plug/a.py": "a2", "localm/plug/b.py": "b"}
    root = um.install(tmp_path / "root", old)
    before = um.tree(root)
    files = um.stage(root, new)
    with job.Lock(root) as lock:
        job.start(lock, ID, old="1", new="2")
        job.advance(lock, job.MOVING)
        plan = mover.plan_files(root, ID, files, um.stated(old), swap=um.SWAP, protect=["mydata"])
        mover.write_plan(root, ID, plan)
    mover.forward(root, ID, plan)
    assert "plug" in os.listdir(root / "localm")
    (root / "localm" / "plug" / "cache.txt").write_bytes(b"c")
    assert uv.undo(root).state == "old"
    assert "Plug" in os.listdir(root / "localm")
    now = um.tree(root)
    assert now.pop("localm/Plug/cache.txt")[0] == "file" and now == before


# --------------------------------------------------------------- the rest

def test_a_record_with_a_size_of_thousands_of_digits_is_read(root):
    moved(root)
    record = site(root) / "alpha-2.0.dist-info" / "RECORD"
    record.write_bytes(record.read_bytes() + b"alpha/x.py,sha256=AA," + b"9" * 5000 + b"\n")
    assert uv.undo(root).state == "old"


def test_a_package_set_aside_halfway_is_set_aside_whole(root):
    before = uv.tree(root)
    moved(root)
    uv.install(root / ".venv", ("alpha", "2.5", {"alpha/__init__.py": "25", "extra25.py": "e"}))
    template = root.parent / "template"
    shutil.copytree(root, template)
    with uv.Points() as points:
        uv.undo(root)
    assert uv.tree(root) == before
    for at in range(points.count):
        work = root.parent / f"k{at}"
        shutil.copytree(template, work)
        with uv.Points(at):
            uv.undo(work)
        result = uv.undo(work)
        assert result.errors == [] and uv.tree(work) == before, (at, result)
        assert b"e" in everything(work)
        shutil.rmtree(work)


@pytest.mark.parametrize("by", ["cleanup", "start"])
def test_what_a_killed_run_staged_is_deleted(root, by):
    before = uv.tree(root)
    uv.stage(root, ["alpha"], [ALPHA2])
    uv.stage(root, ["alpha"], [ALPHA2], OTHER_ID)
    folder = root / ".venv" / mover.VENV_FOLDER
    with job.Lock(root) as lock:
        if by == "cleanup":
            job.cleanup(lock, keep=ID)
        else:
            (root / ".localm-update" / ID / "new").mkdir(parents=True)
            job.start(lock, ID, old="1", new="2")
        assert sorted(os.listdir(folder)) == [ID]
        assert packages.stage_folder(root, ID).exists()
        if by == "cleanup":
            job.cleanup(lock)
            assert not folder.exists()
    assert uv.tree(root) == before


def test_old_files_of_an_update_nobody_knows_are_not_deleted(root):
    moved(root)
    shutil.rmtree(root / ".localm-update")
    with job.Lock(root) as lock:
        job.cleanup(lock)
    assert (root / ".venv" / mover.VENV_FOLDER / ID / "old" / SITE / "alpha" / "data.txt").exists()


def test_a_finished_update_ends_on_a_disk_that_stays_full(root, monkeypatch):
    um.stage(root)
    uv.update(root, stop_before="finish")
    with monkeypatch.context() as m, job.Lock(root) as lock:
        m.setattr(job, "_close", lambda *a: None)
        job.finish(lock)
    (root / ".localm-update" / ID / "reserve").unlink()
    now = uv.tree(root)

    def full(*_a, **_kw):
        raise OSError(errno.ENOSPC, "No space left on device")
    with monkeypatch.context() as m:
        m.setattr(json, "dump", full)
        m.setattr(os, "fsync", full)
        result = uv.undo(root)
    assert result.state == "new" and result.errors == [] and uv.tree(root) == now
    assert uv.undo(root).state == "new" and um.left(root) == []


def test_uv_waits_while_packages_are_moved(root, monkeypatch):
    seen = []

    def lock_is_free() -> bool:
        fd = disk.open_lock(root / ".venv" / ".lock")
        try:
            disk.lock_file(fd)
        except (PermissionError, BlockingIOError):
            return False
        finally:
            os.close(fd)
        return True
    real = disk.move

    def move(*args, **kw):
        seen.append(lock_is_free())
        real(*args, **kw)
    uv.begun(root)
    plan = uv.package_step(root)
    monkeypatch.setattr(disk, "move", move)
    packages.forward(root, ID, plan, ["uv"], env={},
                     run=lambda *a: (0, "" if lock_is_free() else pytest.fail("uv cannot run")))
    assert seen and not any(seen)
    seen.clear()
    assert uv.undo(root).state == "old"
    assert seen and not any(seen)


def test_a_uv_at_work_in_the_venv_stops_the_step_before_any_move(root):
    uv.begun(root)
    plan = uv.package_step(root)
    before = uv.tree(root)
    fd = disk.open_lock(root / ".venv" / ".lock")
    disk.lock_file(fd)
    try:
        with pytest.raises(mover.StepFailed, match="uv is at work") as e:
            uv.forward(root, plan, wait_s=0.1)
        result = uv.undo(root, wait_s=0.1)
        assert result.state == "failed"
        for said in (e.value.held, result.held):
            assert [Path(h.path).name for h in said] == [".lock"]
            if os.name == "nt" or sys.platform == "linux":
                assert os.getpid() in [p["pid"] for p in said[0].holders]
    finally:
        os.close(fd)
    assert uv.tree(root) == before
    assert uv.undo(root).state == "old" and uv.tree(root) == before


# ------------------------------------------------- held files and who holds them

IN_USE = ["localm/changed.py", "localm.bat", "localm/gone.py", f".venv/{SITE}/alpha/data.txt",
          f".venv/{SITE}/alpha/old.py", f".localm-update/{ID}/new/localm/added.py",
          f".venv/.localm-old/{ID}/stage/{SITE}/alpha/new.py"]


def step(root: Path, rel: str):
    """``(plan, forward)`` of the step that moves *rel*."""
    if ".venv" in rel:
        uv.begun(root)
        return uv.package_step(root), lambda plan, **kw: uv.forward(root, plan, **kw)
    return planned(root), lambda plan, **kw: mover.forward(root, ID, plan, **kw)


def check_held(found: list, rel: str) -> None:
    assert found and all(Path(h.path).name == Path(rel).name for h in found), found
    assert [h["pid"] for h in found[0].holders] == [os.getpid()] and found[0].holders[0]["name"]


@windows_only
@pytest.mark.parametrize("rel", IN_USE)
def test_a_held_file_is_asked_about_and_the_step_goes_on_once_it_is_free(root, rel):
    plan, forward = step(root, rel)
    asked: list = []
    block = held(root / rel, share=1)
    block.__enter__()

    def ask(h) -> bool:
        asked.append(h)
        if len(asked) == 2:
            block.__exit__(None, None, None)
        return True
    forward(plan, wait_s=0.1, ask=ask)
    assert len(asked) == 2
    check_held(asked, rel)
    assert uv.undo(root).state == "old"


@windows_only
@pytest.mark.parametrize("rel", IN_USE)
def test_after_no_everything_is_back_and_the_same_plan_goes_again(root, rel):
    before = uv.tree(root)
    plan, forward = step(root, rel)
    asked: list = []
    with held(root / rel, share=1):
        with pytest.raises(mover.StepFailed) as e:
            forward(plan, wait_s=0.1, ask=lambda h: bool(asked.append(h)))
    assert len(asked) == 1 and e.value.errors == []
    check_held(e.value.held, rel)
    assert uv.tree(root) == before
    forward(plan)
    assert uv.tree(root) != before
    assert uv.undo(root).state == "old" and uv.tree(root) == before


@windows_only
@pytest.mark.parametrize("rel", ["localm/changed.py", "localm.bat", "localm/added.py",
                                 f".venv/{SITE}/alpha/data.txt"])
def test_a_file_held_on_the_way_back_is_asked_about_and_named(root, rel):
    before = uv.tree(root)
    plan, forward = step(root, rel)
    forward(plan)
    asked: list = []
    with held(root / rel, share=1):
        result = uv.undo(root, wait_s=0.1)
        assert result.state == "failed"
        check_held(result.held, rel)
        assert uv.undo(root, wait_s=0.1, ask=lambda h: bool(asked.append(h))).state == "failed"
        assert len(asked) == 1
    block = held(root / rel, share=1)
    block.__enter__()

    def ask(h) -> bool:
        block.__exit__(None, None, None)
        return True
    result = uv.undo(root, wait_s=0.1, ask=ask)
    assert (result.state, result.errors, result.held) == ("old", [], [])
    assert uv.tree(root) == before


@windows_only
def test_an_edited_file_held_at_the_end_is_asked_about_and_named(root, monkeypatch):
    (root / "localm" / "changed.py").write_bytes(b"MINE")
    mover.forward(root, ID, planned(root))
    to_starting(root)
    monkeypatch.setattr(disk, "HELD_WAIT_S", 0.1)
    old = root / ".localm-update" / ID / "old" / "localm" / "changed.py"
    asked: list = []
    with job.Lock(root) as lock, held(old):
        result = job.finish(lock, ask=lambda h: bool(asked.append(h)))
    assert result.state == "new" and len(asked) == 1
    check_held(result.held, "changed.py")


@windows_only
def test_a_record_that_cannot_be_read_names_its_holder(root, monkeypatch):
    mover.forward(root, ID, planned(root))
    monkeypatch.setattr(disk, "READ_WAIT_S", 0.1)
    with held(root / ".localm-update" / "job.json", exclusive=True):
        with pytest.raises(job.Held) as e:
            uv.undo(root)
    check_held([e.value], "job.json")
    assert uv.undo(root).state == "old"


@windows_only
@pytest.mark.parametrize("way", ["there", "back"])
def test_a_program_that_stands_in_a_folder_in_the_way_is_asked_about(root, way):
    """Its working folder becomes a file, or comes back as one."""
    before = uv.tree(root)
    plan = planned(root)
    if way == "back":
        mover.forward(root, ID, plan)
    inside = root / "localm" / ("other" if way == "there" else "thing")
    program = subprocess.Popen([um.PYTHON, "-I", "-S", "-c", "input()"], cwd=inside,
                               stdin=subprocess.PIPE, **um.NO_WINDOW)
    asked: list = []

    def ask(h) -> bool:
        asked.append(h)
        program.kill()
        program.wait(timeout=30)
        return True
    try:
        if way == "there":
            mover.forward(root, ID, plan, wait_s=0.2, ask=ask)
            assert (root / "localm" / "other").read_bytes() == um.NEW["localm/other"].encode()
        result = uv.undo(root, wait_s=0.2, ask=ask)
    finally:
        program.kill()
        program.wait(timeout=30)
    assert [Path(h.path) for h in asked] == [disk.long_path(inside)]
    assert (result.state, result.errors) == ("old", []) and uv.tree(root) == before
    assert not (root / ".localm-update" / "kept").exists()


@windows_only
def test_one_no_is_not_asked_again_for_the_next_file_of_the_same_run(root):
    plan = planned(root)
    mover.forward(root, ID, plan)
    asked: list = []
    with held(root / "localm" / "changed.py"), held(root / "localm" / "added.py"):
        result = uv.undo(root, wait_s=0.1, ask=lambda h: bool(asked.append(h)))
    assert len(asked) == 1 and len(result.held) == 2 and result.state == "failed"
    assert uv.undo(root).state == "old"


@windows_only
def test_a_staged_file_in_use_is_named_when_packages_are_staged_anew(root):
    uv.begun(root)
    uv.stage(root, ["alpha"], [ALPHA2])
    path = packages.stage_folder(root, ID) / SITE / "alpha" / "new.py"
    said = (0, "Would uninstall 1 package\nWould install 1 package\n - alpha==1.0\n + alpha==2.0\n", "")
    with held(path), pytest.raises(disk.Held) as e:
        packages.stage_packages(root, ID, ["uv", "sync"], env={}, run=lambda *a: said[:2])
    check_held([e.value], "new.py")


# ------------------------------------------ killed, then the user does something

def edit(rel: str):
    def act(root: Path) -> None:
        if (root / rel).is_file():
            (root / rel).write_bytes(b"THE USER'S EDIT")
    return act


def rebuild(root: Path) -> None:
    shutil.rmtree(root / ".venv")
    uv.install(uv.venv(root), ALPHA1)
    uv.install(root / ".venv", BETA)


ACTIONS = {
    "edit-package": edit(f".venv/{SITE}/alpha/data.txt"),
    "edit-release": edit("localm/changed.py"),
    "reinstall": lambda root: uv.install(root / ".venv", ALPHA1),
    "reinstall-new": lambda root: uv.install(root / ".venv", ALPHA2),
    "reinstall-other": lambda root: uv.install(root / ".venv", THEIRS),
    "uninstall": lambda root: uv.uninstall(root / ".venv", "alpha"),
    "rebuild": rebuild,
}


@pytest.mark.parametrize("action", ACTIONS)
def test_killed_anywhere_and_changed_by_the_user_there_is_always_a_way_back(tmp_path, action):
    """The actions share the points of an update and of its way back; together they
    take every one."""
    template = uv.installed(tmp_path / "template")
    um.stage(template)
    uv.stage(template, ["alpha"], [ALPHA2, GAMMA])
    old = uv.tree(template)
    full = tmp_path / "full"
    shutil.copytree(template, full)
    uv.update(full, stop_before="finish")
    first = sorted(ACTIONS).index(action)
    for start, run in ((template, uv.update), (full, uv.undo)):
        work = tmp_path / "count"
        shutil.copytree(start, work)
        with uv.Points() as points:
            run(work)
        end = uv.tree(work)
        shutil.rmtree(work)
        assert points.count > 100
        for at in range(first, points.count, len(ACTIONS)):
            shutil.copytree(start, work)
            with uv.Points(at):
                run(work)
            ended = run is uv.update and (
                um.done(work) or (um.phase(work) is None and uv.tree(work) == end))
            want = end if ended or run is uv.undo else old
            ACTIONS[action](work)
            result = uv.undo(work)
            assert result.errors == [], (at, result)
            now = uv.tree(work)
            changed = {k for k in {*now, *want} if now.get(k) != want.get(k)}
            if action.startswith("edit"):
                assert len(changed) <= 1 and all(
                    (work / k).read_bytes() == b"THE USER'S EDIT" for k in changed), (at, changed)
            assert all(k.startswith(".venv") for k in changed) or action == "edit-release", (at, changed)
            assert len([p for p in packages_in(work) if p.startswith("alpha")]) <= 1, (
                at, packages_in(work), result)
            if action in ("reinstall", "rebuild"):
                assert (site(work) / "alpha" / "data.txt").read_bytes() == b"one", at
            assert uv.undo(work).state == "none" and uv.tree(work) == now
            shutil.rmtree(work)
