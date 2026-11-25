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


def again(root: Path, result: job.Result, *names: str) -> job.Result:
    """Install exactly the packages *result* names, which are *names*, again and
    restore."""
    assert result.state == "failed" and len(result.errors) == len(names), result
    assert [(p["step"], p["name"], p["version"]) for p in result.packages] == [
        ("packages", name, "1.0") for name in names], result
    assert not any("stays so" in n and "alpha/" in n.rpartition("stays so")[2] for n in result.notes)
    for p in result.packages:
        uv.install(root / ".venv" if (root / ".venv").exists() else uv.venv(root),
                   {"alpha": ALPHA1, "beta": BETA}[p["name"]])
    return uv.undo(root)


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
def test_a_venv_that_was_made_anew_needs_the_old_packages(root, step, anew):
    (site(root) / "gamma.py").write_bytes(b"MINE")
    if step == "forward":
        moved(root)
    else:
        uv.begun(root)
        uv.package_step(root)
    shutil.rmtree(root / ".venv")
    if anew is not None:
        uv.venv(root)
        for package in anew:
            uv.install(root / ".venv", package)
    result = uv.undo(root)
    if anew != (ALPHA1, BETA):
        result = again(root, result, "alpha")
    assert (result.state, result.errors) == ("old", [])
    assert [p for p in packages_in(root) if "beta" not in p] == ["alpha-1.0.dist-info"]
    assert {p.name for p in (site(root) / "alpha").iterdir()} == {
        "__init__.py", "data.txt", "old.py", "same.py"}
    assert um.left(root) == [] and uv.undo(root).state == "none"


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
    assert result.packages == []
    (tmp_path / "unplugged").rename(tmp_path / "elsewhere")
    assert uv.undo(root).state == "old"
    now = um.tree(tmp_path / "elsewhere", skip=(".lock", ".localm-old"))
    assert now == {k: v for k, v in before.items() if k != ".lock"}


# ------------------------------------------------------- lost and healed

@pytest.mark.parametrize("gone", ["alpha/data.txt", "alpha-1.0.dist-info/RECORD",
                                  "alpha-1.0.dist-info/METADATA"])
def test_a_lost_package_file_is_named_and_the_same_version_heals_it(root, gone):
    moved(root)
    (root / ".venv" / mover.VENV_FOLDER / ID / "old" / "01" / SITE / gone).unlink()
    for _ in range(2):
        errors = uv.undo(root).errors
        assert any("Install alpha 1.0 again" in e and gone in e for e in errors)
    assert again(root, uv.undo(root), "alpha").state == "old"
    assert packages_in(root) == ["alpha-1.0.dist-info", "beta-1.0.dist-info"]
    assert (site(root) / "alpha" / "data.txt").read_text(encoding="utf-8") == "one"
    assert um.left(root) == []


def test_everything_lost_is_healed_by_the_same_version_with_other_times(root):
    moved(root)
    shutil.rmtree(root / ".venv" / mover.VENV_FOLDER / ID / "old")
    result = uv.undo(root)
    assert result.state == "failed" and len(result.errors) == 1
    assert [p.name for p in (site(root) / "alpha").glob("*.*")] == ["same.py"]
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
    (root / ".venv" / mover.VENV_FOLDER / ID / "old" / "01" / SITE / "alpha" / "data.txt").unlink()
    assert uv.undo(root).state == "failed"
    uv.install(root / ".venv", ALPHA1)
    assert uv.undo(root).state == "old"


def test_an_edited_package_file_that_is_lost_is_healed_by_the_same_content(root):
    (site(root) / "alpha" / "data.txt").write_bytes(b"MINE")
    moved(root)
    (root / ".venv" / mover.VENV_FOLDER / ID / "old" / "01" / SITE / "alpha" / "data.txt").unlink()
    (site(root) / "alpha" / "data.txt").write_bytes(b"MINE")
    os.utime(site(root) / "alpha" / "data.txt", (1, 1))
    result = uv.undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert (site(root) / "alpha" / "data.txt").read_bytes() == b"MINE"


def test_an_edited_release_file_that_is_lost_is_healed_by_the_release(root):
    (root / "localm" / "changed.py").write_bytes(b"MINE")
    mover.forward(root, ID, planned(root))
    (root / ".localm-update" / ID / "old" / "01" / "localm" / "changed.py").unlink()
    for _ in range(2):
        result = uv.undo(root)
        assert result.state == "failed" and "Put that version" in result.errors[0]
    (root / "localm" / "changed.py").write_bytes(um.OLD["localm/changed.py"].encode())
    assert uv.undo(root).state == "old"


def test_a_file_of_the_user_that_is_lost_is_healed_by_any_file_but_the_new_one(root):
    (root / "localm" / "added.py").write_bytes(b"MINE")
    mover.forward(root, ID, planned(root))
    (root / ".localm-update" / ID / "old" / "01" / "localm" / "added.py").unlink()
    for _ in range(2):
        assert uv.undo(root).state == "failed"
    (root / "localm" / "added.py").write_bytes(b"MINE, from a backup")
    assert uv.undo(root).state == "old"
    assert (root / "localm" / "added.py").read_bytes() == b"MINE, from a backup"


def test_a_damaged_old_file_is_lost_until_the_old_content_is_back(root):
    mover.forward(root, ID, planned(root))
    (root / ".localm-update" / ID / "old" / "01" / "localm" / "changed.py").write_bytes(b"???")
    for _ in range(2):
        assert uv.undo(root).state == "failed"
    (root / "localm" / "changed.py").write_bytes(um.OLD["localm/changed.py"].encode())
    result = uv.undo(root)
    assert result.state == "old" and b"???" in everything(root)


def test_a_healed_package_stays_healed_while_something_else_fails(root, monkeypatch):
    moved(root)
    (root / ".venv" / mover.VENV_FOLDER / ID / "old" / "01" / SITE / "alpha" / "data.txt").unlink()
    failed = [disk.Held("x", "y")]

    def aside(a, op, path, wait, out):
        if failed and op.path.endswith("gamma.py"):
            raise failed[0]
        return real(a, op, path, wait, out)
    real = mover.aside
    monkeypatch.setattr(mover, "aside", aside)
    for _ in range(2):
        result = uv.undo(root)
        assert result.errors == ["x cannot be used (y)"] and result.packages == []
    failed.clear()
    result = uv.undo(root)
    assert [p["name"] for p in result.packages] == ["alpha"]
    failed.append(disk.Held("x", "y"))
    assert again(root, result, "alpha").state == "old"


def test_a_lost_release_file_that_was_healed_is_the_users(root, monkeypatch):
    mover.forward(root, ID, planned(root))
    path = root / "localm" / "changed.py"
    (root / ".localm-update" / ID / "old" / "01" / "localm" / "changed.py").unlink()
    path.write_bytes(um.OLD["localm/changed.py"].encode())
    failed_once(root, "localm/added.py", monkeypatch)
    path.write_bytes(b"edited after that")
    result = uv.undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert path.read_bytes() == b"edited after that"


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
def test_another_version_after_a_failed_restore_goes_to_kept(root, monkeypatch, version, new):
    moved(root)
    failed_once(root, NEW_FILE, monkeypatch)
    theirs = ("alpha", version, {"alpha/__init__.py": "V = 'theirs'", "alpha/new.py": new})
    uv.install(root / ".venv", theirs)
    result = uv.undo(root)
    assert not list((site(root) / "alpha").glob("*.*")) and b"V = 'theirs'" in everything(root)
    result = again(root, result, "alpha")
    assert (result.state, result.errors) == ("old", [])
    assert packages_in(root) == ["alpha-1.0.dist-info", "beta-1.0.dist-info"]
    assert not (site(root) / "alpha" / "new.py").exists()


def test_a_package_removed_after_a_failed_restore_is_named(root, monkeypatch):
    before = uv.tree(root)
    moved(root)
    failed_once(root, NEW_FILE, monkeypatch)
    assert "alpha-1.0.dist-info" in packages_in(root)
    uv.uninstall(root / ".venv", "alpha")
    assert again(root, uv.undo(root), "alpha").state == "old"
    assert uv.tree(root) == before


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
def test_a_package_changed_before_the_first_move_is_named(root, change):
    before = uv.tree(root)
    uv.begun(root)
    uv.package_step(root)
    data = site(root) / "alpha" / "data.txt"
    {"edit": lambda: data.write_bytes(b"MINE"), "remove": data.unlink,
     "reinstall": lambda: uv.install(root / ".venv", ALPHA1),
     "other": lambda: uv.install(root / ".venv", THEIRS),
     "uninstall": lambda: uv.uninstall(root / ".venv", "alpha")}[change]()
    result = uv.undo(root)
    if change != "reinstall":
        result = again(root, result, "alpha")
    assert (result.state, uv.tree(root) == before) == ("old", True)
    assert change != "edit" or b"MINE" in everything(root)


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
@pytest.mark.parametrize("theirs", [THEIRS, ALPHA2, ("alpha", "2.0", {"alpha/__init__.py": "V = 'x'"})],
                         ids=["1.5", "2.0", "other-2.0"])
@pytest.mark.parametrize("change", ["nothing", "edit", "remove", "no-record"])
def test_another_version_over_half_of_the_old_one_ends_whole(root, monkeypatch, stop, theirs, change):
    before = uv.tree(root)
    killed_before(root, stop, monkeypatch)
    uv.install(root / ".venv", theirs)
    path = site(root) / ("alpha/__init__.py" if change != "no-record"
                         else f"alpha-{theirs[1]}.dist-info/RECORD")
    path.write_bytes(b"MINE") if change == "edit" else path.unlink() if change != "nothing" else None
    result = uv.undo(root)
    if result.state == "failed":
        result = again(root, result, "alpha")
    assert (result.state, result.errors) == ("old", [])
    now = uv.tree(root)
    assert change == "no-record" or now == before
    assert {k: v for k, v in now.items() if "extra.py" not in k} == before
    assert change != "edit" or b"MINE" in everything(root)


@pytest.mark.parametrize("stop", ["alpha-1.0.dist-info/METADATA", "alpha-1.0.dist-info/RECORD",
                                  "alpha/data.txt"])
def test_a_package_removed_with_a_part_of_it_moved_is_named(root, monkeypatch, stop):
    before = uv.tree(root)
    killed_before(root, stop, monkeypatch)
    shutil.rmtree(site(root) / "alpha")
    shutil.rmtree(site(root) / "alpha-1.0.dist-info", ignore_errors=True)
    result = uv.undo(root)
    assert (site(root) / "alpha-1.0.dist-info" / "METADATA").exists() == ("METADATA" not in stop)
    assert again(root, result, "alpha").state == "old"
    assert uv.tree(root) == before


SCRIPT = "../../Scripts/run" if os.name == "nt" else "../../../bin/run"


def scripted(package: tuple) -> tuple:
    return package[0], package[1], {**package[2], SCRIPT: f"run {package[1]}"}


@pytest.mark.parametrize("theirs", [THEIRS, ALPHA2], ids=["1.5", "2.0"])
def test_a_script_of_another_version_is_not_the_old_one(root, monkeypatch, theirs):
    uv.install(root / ".venv", scripted(ALPHA1))
    uv.begun(root)
    uv.package_step(root, [scripted(ALPHA2)], ["alpha"])
    uv.install(root / ".venv", scripted(theirs))
    (site(root) / f"alpha-{theirs[1]}.dist-info" / "RECORD").unlink()
    result = uv.undo(root)
    assert [p["name"] for p in result.packages] == ["alpha"]
    uv.install(root / ".venv", scripted(ALPHA1))
    script = Path(os.path.normpath(site(root) / SCRIPT))
    script.write_bytes(f"run {theirs[1]}".encode())
    assert [p["name"] for p in uv.undo(root).packages] == ["alpha"]
    uv.install(root / ".venv", (*ALPHA1[:2], {**ALPHA1[2], SCRIPT: "run 1.0, written by pip"}))
    assert uv.undo(root).state == "old"
    assert script.read_bytes() == b"run 1.0, written by pip"


WHEEL = "alpha-1.0.dist-info/WHEEL"


def built(generator: str) -> tuple:
    return "alpha", "1.0", {**ALPHA1[2], WHEEL: f"Generator: {generator}"}


@pytest.mark.parametrize("how", ["copies", "killed"])
def test_a_package_built_by_another_backend_is_the_old_one(root, monkeypatch, how):
    """Also with a file nobody lists in its dist-info folder, whose copy is lost."""
    uv.install(root / ".venv", built("1"))
    (site(root) / "alpha-1.0.dist-info" / "desktop.ini").write_bytes(b"MINE")
    if how == "copies":
        moved(root)
        shutil.rmtree(root / ".venv" / mover.VENV_FOLDER / ID / "old")
        assert [p["name"] for p in uv.undo(root).packages] == ["alpha"]
    else:
        killed_before(root, "alpha/old.py", monkeypatch)
        for name in ("WHEEL", "desktop.ini"):
            for path in (root / ".venv" / mover.VENV_FOLDER / ID / "old").rglob(name):
                path.unlink()
    uv.install(root / ".venv", built("2"))
    result = uv.undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert (site(root) / WHEEL).read_bytes() == b"Generator: 2"


@pytest.mark.parametrize("name", ["python", "pythonw", "python3", "python3.12"])
def test_an_interpreter_of_the_venv_is_never_moved(root, name):
    rel = f"Scripts/{name}.exe" if os.name == "nt" else f"bin/{name}"
    python = root / ".venv" / rel
    python.parent.mkdir(exist_ok=True)
    python.write_bytes(b"PYTHON")
    up = "/".join([".."] * len(SITE.split("/")))
    theirs = ("alpha", "1.5", {"alpha/__init__.py": "15", f"{up}/{rel}": "PLANTED"})
    moved(root)
    uv.install(root / ".venv", theirs)
    python.write_bytes(b"PYTHON")
    result = uv.undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert python.read_bytes() == b"PYTHON"
    uv.install(root / ".venv", theirs)
    assert not any(rel in packages._records(site(root), root / ".venv", f, "alpha")
                   for f in packages_in(root) if f.startswith("alpha"))


@pytest.mark.parametrize("what", ["metadata-only", "no-record", "record-of-nothing"])
def test_a_package_without_its_whole_dist_info_is_named(root, what):
    uv.begun(root)
    uv.package_step(root)
    info = site(root) / "alpha-1.0.dist-info"
    if what == "metadata-only":
        uv.uninstall(root / ".venv", "alpha")
        info.mkdir()
        (info / "METADATA").write_bytes(b"Name: alpha\nVersion: 1.0\n")
    elif what == "no-record":
        (info / "RECORD").unlink()
    else:
        (info / "RECORD").write_bytes(b"alpha/nothing.py,,\n")
    result = again(root, uv.undo(root), "alpha")
    assert (result.state, result.errors) == ("old", [])
    assert what != "record-of-nothing" or b"alpha/nothing.py,,\n" in everything(root)


# ------------------------------------------------- a file two packages list

S = "shared/s.py"


def more(package: tuple, text: str) -> tuple:
    return package[0], package[1], {**package[2], S: text}


@pytest.fixture
def sharing(tmp_path):
    """alpha 1.0 and beta 1.0, whose file the new alpha replaces."""
    path = um.install(tmp_path / "sharing")
    for package in (ALPHA1, more(BETA, "of beta")):
        uv.install(path / ".venv" if (path / ".venv").exists() else uv.venv(path), package)
    return path


def test_a_package_upgraded_meanwhile_that_lost_a_file_is_named(sharing):
    before = uv.tree(sharing)
    moved(sharing, coming=[more(ALPHA2, "of alpha"), GAMMA])
    uv.install(sharing / ".venv", ("beta", "2.0", {"beta.py": "V = 2", S: "of beta 2"}))
    result = uv.undo(sharing)
    assert result.state == "failed" and "beta 1.0" in result.errors[0] and S in result.errors[0]
    assert [(p["name"], p["version"]) for p in result.packages] == [("beta", "1.0")]
    uv.install(sharing / ".venv", more(BETA, "of beta"))
    assert uv.undo(sharing).state == "old" and uv.tree(sharing) == before
    assert b"of beta 2" in everything(sharing)


def test_an_edited_file_of_another_package_is_kept_when_that_is_named(sharing):
    uv.begun(sharing)
    uv.package_step(sharing, [more(ALPHA2, "of alpha"), GAMMA])
    (site(sharing) / S).write_bytes(b"MINE")
    result = uv.undo(sharing)
    assert [(p["name"], p["version"]) for p in result.packages] == [("beta", "1.0")]
    assert b"MINE" in everything(sharing) and not (site(sharing) / S).exists()
    uv.install(sharing / ".venv", more(BETA, "of beta"))
    assert uv.undo(sharing).state == "old"


@pytest.mark.parametrize("again_", [(), ("alpha",), ("beta",)], ids=["nothing", "alpha", "beta"])
@pytest.mark.parametrize("last", ["alpha", "beta"])
def test_packages_that_share_a_file_differently_are_as_they_were(tmp_path, last, again_):
    old = {"alpha": more(ALPHA1, "of alpha"), "beta": more(BETA, "of beta")}
    path = um.install(tmp_path / "differently")
    uv.venv(path)
    for name in sorted(old, key=lambda n: n == last):
        uv.install(path / ".venv", old[name])
    before = uv.tree(path)
    moved(path)
    for name in again_:
        uv.install(path / ".venv", old[name])
    result = uv.undo(path)
    assert (result.state, result.errors) == ("old", [])
    assert uv.tree(path) == before


def test_an_edit_in_a_named_package_is_kept_before_it_is_installed_again(sharing):
    moved(sharing, coming=[more(ALPHA2, "of alpha"), GAMMA])
    theirs = ("beta", "2.0", {"beta.py": "V = 2", "beta2.py": "b", S: "of beta 2"})
    uv.install(sharing / ".venv", theirs)
    (site(sharing) / "beta2.py").write_bytes(b"MINE")
    result = uv.undo(sharing)
    assert [p["name"] for p in result.packages] == ["beta"]
    assert not (site(sharing) / "beta2.py").exists() and b"MINE" in everything(sharing)
    assert (site(sharing) / "beta.py").read_bytes() == b"V = 2"
    uv.install(sharing / ".venv", more(BETA, "of beta"))
    assert uv.undo(sharing).state == "old" and b"MINE" in everything(sharing)


def test_a_version_that_took_a_file_of_another_package_names_that_one(root):
    before = uv.tree(root)
    moved(root)
    uv.install(root / ".venv", ("alpha", "1.5", {"alpha/__init__.py": "15", "beta.py": "of alpha"}))
    result = again(root, uv.undo(root), "beta")
    assert (result.state, uv.tree(root) == before) == ("old", True)


def test_a_package_installed_meanwhile_that_lost_a_file_is_noted(root):
    before = uv.tree(root)
    moved(root)
    uv.install(root / ".venv", ("delta", "1.0", {"delta.py": "d", "alpha/data.txt": "of delta"}))
    result = uv.undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert any("delta" in n and "alpha/data.txt" in n for n in result.notes)
    now = uv.tree(root)
    assert {k: v for k, v in now.items() if "delta" not in k} == before


# ----------------------------------------------- two plans of one update

ALPHA3 = ("alpha", "3.0", {"alpha/__init__.py": "V = 3", "alpha/data.txt": "three"})


@pytest.mark.parametrize("second", ["written", "forward"])
@pytest.mark.parametrize("then", ["nothing", "reinstall", "other", "copies"])
def test_a_second_plan_for_the_same_package_leaves_the_copies_of_the_first(root, second, then):
    before = uv.tree(root)
    moved(root)
    uv.stage(root, ["alpha"], [ALPHA3])
    plan = uv.package_step(root, [ALPHA3], ["alpha"])
    assert plan.number == 2 and packages_in(root)[0] == "alpha-2.0.dist-info"
    if second == "forward":
        uv.forward(root, plan)
    if then in ("reinstall", "other"):
        uv.install(root / ".venv", ALPHA1 if then == "reinstall" else THEIRS)
    if then == "copies":
        shutil.rmtree(root / ".venv" / mover.VENV_FOLDER / ID / "old")
    result = uv.undo(root)
    if then == "copies":
        result = again(root, result, "alpha")
    assert (result.state, result.errors) == ("old", [])
    assert uv.tree(root) == before and um.left(root) == []


def test_a_second_plan_that_fails_leaves_the_first_one_made(root):
    moved(root)
    now = uv.tree(root)
    uv.stage(root, ["alpha"], [ALPHA3])
    plan = uv.package_step(root, [ALPHA3], ["alpha"])
    with pytest.raises(mover.StepFailed) as e:
        packages.forward(root, ID, plan, ["uv"], env={}, run=lambda *a: (1, "no"))
    assert (e.value.errors, e.value.packages) == ([], []) and uv.tree(root) == now
    uv.forward(root, plan)
    to_end(root)
    assert packages_in(root)[0] == "alpha-3.0.dist-info" and um.left(root) == []


def test_a_step_that_cannot_be_put_back_names_its_packages(root):
    uv.begun(root)
    plan = uv.package_step(root)

    def run(*_a):
        shutil.rmtree(root / ".venv" / mover.VENV_FOLDER / ID / "old")
        return 1, "no"
    with pytest.raises(mover.StepFailed) as e:
        packages.forward(root, ID, plan, ["uv"], env={}, run=run)
    assert e.value.packages == [{"step": "packages", "name": "alpha", "version": "1.0"}]
    assert again(root, uv.undo(root), "alpha").state == "old"


def test_a_move_that_was_marked_and_not_made_is_not_lost(root, monkeypatch):
    plan = planned(root)
    real = os.rename

    def stop(src, dst):
        if Path(src).name == "changed.py":
            raise SystemExit("killed")
        real(src, dst)
    with monkeypatch.context() as m:
        m.setattr(os, "rename", stop)
        m.setattr(mover, "_back", lambda *a, **kw: mover.Outcome())
        with pytest.raises(mover.StepFailed):
            mover.forward(root, ID, plan)
    (root / "localm" / "changed.py").write_bytes(b"MINE")
    failed = [disk.Held("x", "y")]
    move = disk.move

    def held(src, dst, wait, **kw):
        if failed and src.name == "VERSION":
            raise failed[0]
        move(src, dst, wait, **kw)
    monkeypatch.setattr(disk, "move", held)
    for _ in range(2):
        result = uv.undo(root)
        assert result.errors == ["x cannot be used (y)"]
    failed.clear()
    result = uv.undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert (root / "localm" / "changed.py").read_bytes() == b"MINE"


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


@pytest.mark.parametrize("phase", ["ended", "done", "left", "ending", "starting"])
@pytest.mark.parametrize("record", ["kept", "gone", "edited"])
@pytest.mark.parametrize("state", ["half", "moved", "nothing"])
def test_a_phase_that_was_edited_decides_nothing(root, monkeypatch, phase, record, state):
    before = uv.tree(root)
    if state == "half":
        half_moved(root, monkeypatch)
    else:
        plan = planned(root)
        if state == "moved":
            mover.forward(root, ID, plan)
    rewrite(root, f"{ID}/state.json", phase=phase)
    if record == "gone":
        (root / ".localm-update" / "job.json").unlink()
    elif record == "edited" and phase in job.PHASES:
        rewrite(root, "job.json", phase=phase)
    assert uv.undo(root).state == "old" and uv.tree(root) == before
    assert um.left(root) == []


@pytest.mark.parametrize("phase", ["ended", "done"])
def test_an_edited_state_of_a_package_step_decides_nothing(root, phase):
    before = uv.tree(root)
    plan = moved(root, coming=[GAMMA], leaving=[])
    assert all(op.old is None for op in plan.ops)
    rewrite(root, f"{ID}/state.json", phase=phase)
    assert uv.undo(root).state == "old" and uv.tree(root) == before


def test_a_record_that_claims_more_than_the_state_does_not_decide(root):
    before = uv.tree(root)
    mover.forward(root, ID, planned(root))
    rewrite(root, "job.json", phase="done")
    assert uv.undo(root).state == "old" and uv.tree(root) == before


DAMAGE = {"zeros": lambda p: p.write_bytes(b"\0" * 60), "empty": lambda p: p.write_bytes(b""),
          "other": lambda p: p.write_bytes(b"{}"), "gone": lambda p: p.unlink(), "fine": lambda p: None}


@pytest.mark.parametrize("state", DAMAGE)
@pytest.mark.parametrize("record", DAMAGE)
@pytest.mark.parametrize("step", ["ending", "files", "half", "all", "done", "left"])
def test_what_is_damaged_of_the_state_never_stops_a_start(root, monkeypatch, state, record, step):
    """``files``: the files are moved and the packages not even planned."""
    (root / "localm" / "changed.py").write_bytes(b"MINE")
    before = uv.tree(root)
    if step == "half":
        half_moved(root, monkeypatch)
    elif step == "ending":
        um.stage(root)
        with job.Lock(root) as lock:
            job.start(lock, ID, old="1", new="2")
    elif step == "files":
        mover.forward(root, ID, planned(root))
        uv.stage(root, ["alpha"], [ALPHA2])
    else:
        um.stage(root)
        uv.update(root, stop_before="finish")
    if step in ("done", "left"):
        with monkeypatch.context() as m, job.Lock(root) as lock:
            m.setattr(job, "_hand_over", lambda *a: False)
            if step == "done":
                m.setattr(job, "_close", lambda *a: None)
            job.finish(lock)
        before = uv.tree(root)
    for name, how in ((f"{ID}/state.json", state), ("job.json", record)):
        path = root / ".localm-update" / name
        if path.exists():
            DAMAGE[how](path)
    result = uv.undo(root)
    assert result.errors == [] and uv.tree(root) == before
    assert result.state in {"done": ("new", "none"), "left": ("new", "none"),
                            "ending": ("old", "none")}.get(step, ("old",))
    assert b"MINE" in everything(root)
    assert um.left(root) in ([], ["kept"], *([[ID]] if step == "ending" else []))
    assert uv.undo(root).state == "none"


def test_an_update_that_ended_and_whose_record_stayed_is_not_put_back_again(root, monkeypatch):
    moved(root)

    def held(path):
        raise disk.Held(path, "in use")
    with monkeypatch.context() as m:
        m.setattr(disk, "remove_through", held)
        result = uv.undo(root)
    assert result.state == "old" and any("for now" in n for n in result.notes)
    uv.install(root / ".venv", THEIRS)
    mine = uv.tree(root)
    result = uv.undo(root)
    assert (result.state, result.errors) == ("none", [])
    assert uv.tree(root) == mine and um.left(root) == []


@pytest.mark.parametrize("state", DAMAGE)
def test_a_damaged_plan_is_named_with_what_to_do(root, state):
    plan = planned(root)
    mover.forward(root, ID, plan)
    now = uv.tree(root)
    DAMAGE[state](root / ".localm-update" / ID / "state.json")
    path = root / ".localm-update" / ID / "plans" / "01-files.json"
    good = path.read_bytes()
    path.write_bytes(b"\0" * 60)
    with pytest.raises(job.Unreadable):
        uv.undo(root)
    with pytest.raises(job.Unreadable, match="setup") as e:
        uv.undo(root)
    assert str(root / ".localm-update" / ID) in str(e.value) and "job.json" in str(e.value)
    assert uv.tree(root) == now
    path.write_bytes(good)
    assert uv.undo(root).state == "old"


@pytest.mark.parametrize("lost", ["01-files", "02-packages", "both"])
@pytest.mark.parametrize("marks", ["kept", "gone"])
@pytest.mark.parametrize("record", ["fine", "gone"])
def test_a_plan_that_is_missing_is_like_a_damaged_one(root, lost, marks, record):
    before = uv.tree(root)
    um.stage(root)
    uv.update(root, stop_before="finish")
    now = uv.tree(root)
    work = root / ".localm-update" / ID
    good = {}
    for path in [*work.glob("plans/*.json"), *work.glob("moved-*")]:
        if (path.stem in (lost, "both") or lost == "both") if path.suffix else marks == "gone":
            good[path] = path.read_bytes()
            path.unlink()
    if record == "gone":
        (work.parent / "job.json").unlink()
    for _ in range(2):
        with pytest.raises(job.Unreadable, match="a plan is missing.*setup"):
            uv.undo(root)
    with job.Lock(root) as lock, pytest.raises((job.Unreadable, ValueError, job.Busy)):
        job.finish(lock)
    assert uv.tree(root) == now
    for path, data in good.items():
        path.write_bytes(data)
    assert uv.undo(root).state == "old" and uv.tree(root) == before


def test_a_first_plan_without_any_trace_is_missed(root):
    um.stage(root)
    uv.update(root, stop_before="finish")
    work = root / ".localm-update" / ID
    (work / "plans" / "01-files.json").unlink()
    (work / "moved-01").unlink()
    shutil.rmtree(work / "old")
    with pytest.raises(job.Unreadable, match="a plan is missing"):
        uv.undo(root)


def test_only_marks_of_a_missing_plan_are_enough(root):
    moved(root, coming=[GAMMA], leaving=[])
    assert not (root / ".venv" / mover.VENV_FOLDER / ID / "old").exists()
    (root / ".localm-update" / ID / "plans" / "01-packages.json").unlink()
    with pytest.raises(job.Unreadable, match="a plan is missing"):
        uv.undo(root)


def test_a_finished_update_whose_folders_are_half_deleted_is_deleted(root, monkeypatch):
    um.stage(root)
    uv.update(root, stop_before="finish")
    with monkeypatch.context() as m, job.Lock(root) as lock:
        m.setattr(disk, "remove_tree", lambda path: False)
        assert job.finish(lock).state == "new"
    work = root / ".localm-update" / ID
    (work / "gone" / "01-files.json").unlink()
    assert uv.undo(root).state == "none" and um.left(root) == []


def test_two_unfinished_updates_without_a_record_are_named_with_what_to_do(root):
    mover.forward(root, ID, planned(root))
    folder = root / ".localm-update"
    shutil.copytree(folder / ID, folder / OTHER_ID)
    (folder / "job.json").unlink()
    with pytest.raises(job.Unreadable, match="setup") as e:
        uv.undo(root)
    assert ID in str(e.value) and OTHER_ID in str(e.value)
    shutil.rmtree(folder / OTHER_ID)
    assert uv.undo(root).state == "old"


@pytest.mark.parametrize("rel", ["localm", "localm/newdir", ".venv"])
def test_a_folder_that_became_a_link_after_done_does_not_stop_a_start(root, tmp_path, monkeypatch, rel):
    from tests.test_update_mechanism import link_folder
    (root / "localm" / "changed.py").write_bytes(b"MINE")
    um.stage(root)
    uv.update(root, stop_before="finish")
    with monkeypatch.context() as m, job.Lock(root) as lock:
        m.setattr(job, "_hand_over", lambda *a: False)
        m.setattr(job, "_close", lambda *a: None)
        job.finish(lock)
    (root / rel).rename(tmp_path / "elsewhere")
    link_folder(root / rel, tmp_path / "elsewhere")
    result = uv.undo(root)
    assert (result.state, result.errors) == ("new", [])
    assert b"MINE" in everything(root) and uv.undo(root).state == "none"


def test_a_finish_before_every_step_is_made_is_refused(root):
    plan = planned(root)
    to_starting(root)
    with job.Lock(root) as lock, pytest.raises(job.Unreadable):
        job.finish(lock)
    assert um.phase(root) == job.STARTING
    mover.forward(root, ID, plan)
    with job.Lock(root) as lock:
        assert job.finish(lock).state == "new"


@pytest.mark.parametrize("rel", ["localm/changed.py", "localm/added.py"])
def test_an_edit_before_the_end_does_not_stop_it(root, rel):
    mover.forward(root, ID, planned(root))
    (root / rel).write_bytes(b"MINE")
    to_starting(root)
    with job.Lock(root) as lock:
        assert job.finish(lock).state == "new"
    assert (root / rel).read_bytes() == b"MINE"


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


def test_an_empty_folder_that_was_there_before_the_update_stays(root):
    (site(root) / "alpha" / "sub" / "__pycache__").mkdir(parents=True)
    before = uv.tree(root)
    moved(root, coming=[("alpha", "2.0", {**ALPHA2[2], "alpha/sub/x.py": "x"})])
    assert (uv.undo(root).state, uv.tree(root) == before) == ("old", True)


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
    assert (root / ".venv" / mover.VENV_FOLDER / ID / "old" / "01" / SITE / "alpha" / "data.txt").exists()


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
    assert uv.undo(root).state in ("new", "none") and um.left(root) == []


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
          f".venv/.localm-old/{ID}/stage/01/{SITE}/alpha/new.py"]


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
def test_a_held_file_of_a_foreign_version_is_found_again(root):
    before = uv.tree(root)
    moved(root)
    uv.install(root / ".venv", THEIRS)
    path = site(root) / "alpha" / "extra.py"
    asked: list = []
    with held(path, share=1):
        result = uv.undo(root, wait_s=0.1, ask=lambda h: bool(asked.append(h)))
    assert result.state == "failed" and len(asked) == 1
    check_held(result.held, "extra.py")
    result = uv.undo(root)
    if result.packages:
        result = again(root, result, "alpha")
    assert result.state == "old" and uv.tree(root) == before


@windows_only
def test_an_edited_file_held_at_the_end_is_asked_about_and_named(root, monkeypatch):
    (root / "localm" / "changed.py").write_bytes(b"MINE")
    mover.forward(root, ID, planned(root))
    to_starting(root)
    monkeypatch.setattr(disk, "HELD_WAIT_S", 0.1)
    old = root / ".localm-update" / ID / "old" / "01" / "localm" / "changed.py"
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
    "reinstall-new-edit": lambda root: (uv.install(root / ".venv", ALPHA2),
                                        edit(f".venv/{SITE}/alpha/__init__.py")(root)),
    "reinstall-other": lambda root: uv.install(root / ".venv", THEIRS),
    "reinstall-other-edit": lambda root: (uv.install(root / ".venv", THEIRS),
                                          edit(f".venv/{SITE}/alpha/__init__.py")(root)),
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
            planned = any((work / ".localm-update" / ID).glob("plans/*-packages.json"))
            ACTIONS[action](work)
            mine = b"THE USER'S EDIT" in everything(work)
            result = uv.undo(work)
            if result.packages:
                result = again(work, result, "alpha")
            assert result.errors == [], (at, result)
            assert not mine or b"THE USER'S EDIT" in everything(work), at
            now = uv.tree(work)
            changed = {k for k in {*now, *want} if now.get(k) != want.get(k)}
            if action == "edit-release":
                assert changed <= {"localm/changed.py"}, (at, changed)
            elif want == old and planned and result.state == "old":
                assert changed == set(), (at, changed)
            assert all(k.startswith(".venv") for k in changed) or action == "edit-release", (at, changed)
            assert uv.undo(work).state == "none" and uv.tree(work) == now
            shutil.rmtree(work)
