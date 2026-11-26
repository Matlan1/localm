# SPDX-License-Identifier: AGPL-3.0-or-later
"""An update into a release that ships no update program of its own (an older one,
``localm update --to``): the installed version gives it a copy of its own program and,
when it has none, the list of its files; that program installs its files by that list
with the gate last, its packages from its own lock, checks it with its command line and
leaves the rest to that release's first start."""

from __future__ import annotations

import json
import os
import re
import shutil
import time

import pytest

from localm.update_mechanism import applier, mover, steps_update
from tests import _update_mechanism as um
from tests.test_update_applier import Update, _signed, _with_main
from tests.test_update_gate import HIDDEN, ID, NEW, REPO, _statement

@pytest.fixture
def update(tmp_path):
    return Update(tmp_path / "inst")


#: The ``__main__`` of an older release: no ``_update-prepare``, ``_update-probe`` or
#: ``_update-record``; ``--version`` answers as click's version option does.
_OLDER = """\
import sys as _sys
if _sys.argv[1:2] in (['_update-prepare'], ['_update-probe'], ['_update-record']):
    print("Error: No such command '%s'." % _sys.argv[1], file=_sys.stderr)
    _sys.exit(2)
if _sys.argv[1:] == ['--version']:
    print('localm, version 2')
    _sys.exit(0)
"""


def _older() -> dict:
    """The files of release 2 as an older release ships them."""
    files = {k: v for k, v in _with_main(NEW).items()
             if not k.startswith("localm/update_mechanism/")}
    files["localm/__main__.py"] = _OLDER + files["localm/__main__.py"]
    return files


def _make_older(update) -> dict:
    """Turn the prepared update of *update* into one to an older release 2."""
    new = _older()
    folder = update.root / ".localm-update" / ID / "new"
    shutil.rmtree(folder)
    um.write(folder, {**new, "update-package.json": _statement(new, "2")})
    later = time.time() + 10
    for path in folder.rglob("*"):
        os.utime(path, (later, later))
    return new


def test_an_update_into_an_older_release_installs_exactly_that_release(update):
    """``localm update`` to a release without its own program: this build's program
    replaces the files by that release's list (its own update mechanism is gone after),
    runs none of the commands that release lacks, and checks it with ``--version``."""
    new = _make_older(update)
    assert applier.is_older(update.root, ID)
    proc = update.popen("-m", "localm", "update", ID)
    out, err = proc.communicate(timeout=300)
    assert proc.returncode == 0, err
    assert update.version() == "2" and not update.record.exists()
    tree = update.tree()
    assert not [rel for rel in tree if rel.startswith("localm/update_mechanism/")]
    assert {rel for rel in tree if rel != "home/mine.txt"} >= set(new)
    assert (update.root / "localm" / "__main__.py").read_text(encoding="utf-8") \
        == new["localm/__main__.py"]
    assert update.result()["state"] == "new"


def test_into_a_release_without_the_gate_its_gate_files_go_in_a_plan_of_their_own(update):
    """Into an older release the files go in three plans: every file but the gate files
    first, none of them kept from being absent; after the steps the gate files but the
    gate, each never absent, the launchers first, a gate file the release lacks
    (``localm/procs.py``) taken out after them; and the gate, ``localm/__init__.py``,
    last and alone."""
    new = _older()
    del new["localm/procs.py"]
    folder = update.root / ".localm-update" / ID / "new"
    shutil.rmtree(folder)
    um.write(folder, new)
    old = json.loads((update.root / "update-package.json").read_text(encoding="utf-8"))
    files = mover.plan_files(update.root, ID, um.stated(new), old["files"], protect=[],
                             swap=(), among=mover.is_not_gate)
    assert files.ops and not [op for op in files.ops if op.swap or op.path in mover.GATE_FILES]
    (update.root / ".localm-update" / ID / "plans").mkdir()   # as job.start makes it
    mover.write_plan(update.root, ID, files)
    rest = mover.plan_gate(update.root, ID, 0, um.stated(new), old["files"], protect=[])
    assert rest.number == files.number + 1
    order = [op.path for op in rest.ops]
    assert len(order) > 2 and order[-1] == "localm/procs.py"
    assert order[:-1] == [p for p in mover.GATE_FILES_LAST if p in order[:-1]]
    assert mover.GATE not in order and set(order) <= set(mover.GATE_FILES)
    assert [op.swap for op in rest.ops] == [True] * (len(order) - 1) + [False]
    assert rest.ops[-1].new is None
    mover.write_plan(update.root, ID, rest)
    gate = mover.plan_gate(update.root, ID, 1, um.stated(new), old["files"], protect=[])
    assert gate.number == rest.number + 1
    assert [(op.path, op.swap) for op in gate.ops] == [(mover.GATE, True)]


def test_a_plan_whose_files_that_are_never_absent_do_not_come_first_is_refused(update):
    """The moves go in the order of the plan, the files that are never absent first: a
    plan that puts one of them after another file is never written or read."""
    new = _older()
    folder = update.root / ".localm-update" / ID / "new"
    shutil.rmtree(folder)
    um.write(folder, new)
    old = json.loads((update.root / "update-package.json").read_text(encoding="utf-8"))
    plan = mover.plan_files(update.root, ID, um.stated(new), old["files"], protect=[])
    swaps = [op for op in plan.ops if op.swap]
    assert swaps and len(swaps) < len(plan.ops)
    plan.ops = [*(op for op in plan.ops if not op.swap), *swaps]
    (update.root / ".localm-update" / ID / "plans").mkdir()   # as job.start makes it
    with pytest.raises(mover.Unreadable, match="never absent comes after"):
        mover.write_plan(update.root, ID, plan)
    assert not mover.has_plans(update.root, ID)


def _gateless(files: dict) -> dict:
    """*files* of :data:`tests.test_update_gate.NEW` as a release without the gate ships
    them: a ``localm/__init__.py`` that only names its version, no ``localm/procs.py`` and
    no update mechanism."""
    out = {k: v for k, v in files.items()
           if not k.startswith("localm/update_mechanism/") and k != "localm/procs.py"}
    out["localm/__init__.py"] = '__version__ = "2"\n'
    return out


def test_killed_between_the_moves_of_the_gate_plan_a_start_runs_one_whole_version(tmp_path):
    """An update into a release without the gate killed at points across its plans, at
    every point from the first move of a gate file on: the next start either puts
    the old version back and runs it, or runs the new release over all of its files;
    never the old gate over new files left as they were, nor the new code over a mix."""
    from tests.test_update_gate import NEW, OLD, Install, ran
    new = _gateless(NEW)

    def install(root):
        one = Install(root)
        folder = root / ".localm-update" / ID / "new"
        shutil.rmtree(folder)
        um.write(folder, {**new, "update-package.json": _statement(new, "2")})
        return one

    def spec(root):
        return {"do": "update", "root": str(root), "id": ID, "swap": [], "gate_last": True,
                "new_files": um.stated(new), "old_files": um.stated(OLD)}
    whole = install(tmp_path / "count")
    counted = um.drive(whole.root / ".localm-update" / ID / "applier" / "update_mechanism",
                       spec(whole.root), -1)
    assert counted.returncode == 0, counted.stderr
    points = int(re.search(r"POINTS (\d+)", counted.stdout).group(1))
    gate = int(re.search(r"GATE (\d+)", counted.stdout).group(1))
    new_tree = um.tree(whole.root)
    assert 0 < gate < points - 20, (gate, points)
    # A few points of the other steps, and every point of the gate plans and the end.
    chosen = sorted({*range(0, gate, max(1, gate // 4)), *range(gate, points)})

    def one(n):
        inst = install(tmp_path / f"k{n}")
        run_dir = inst.root / ".localm-update" / ID / "applier" / "update_mechanism"
        assert um.drive(run_dir, spec(inst.root), n).returncode != 0, n
        done = inst.localm()
        assert done.code == 0, (n, done.err)
        tree = um.tree(inst.root)
        if ran(done, "1"):
            assert tree == inst.old and not inst.record.exists(), (n, done.err)
            return "old"
        assert ran(done, "2"), (n, done.out, done.err)
        assert tree == new_tree, n
        return "new"
    found = um.each(len(chosen), lambda i: one(chosen[i]))
    assert "old" in found, found
    pairs = list(zip(chosen, found, strict=True))
    between = [p for p, f in pairs if gate <= p and f == "old"]
    assert len(between) >= 3, pairs


@pytest.mark.parametrize("catches_up", [False, True])
def test_prepare_gives_an_older_build_this_program_and_the_list_of_its_files(
        update, monkeypatch, catches_up):
    """``updater_handoff.prepare`` of a build that ships no update program and no list of
    its files: the list is made from its verified files, this build's program is copied
    in and runs the check, which prepares only the packages. Its note says what becomes
    of the runtime and the components: brought to that build's pins when it starts, for
    a build that does so (``catch_up``, 0.3.2 on), else that they stay as they are."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from localm import _version, config, update_package, updater, updater_handoff
    from tests._net_config import set_config
    from tests._update_signing import keyset
    files = _older()
    if catches_up:
        for name in ("updater_runtime", "updater_components"):
            files[f"localm/{name}.py"] = "\n\ndef catch_up(root=None, *, repair=False):\n" \
                                         "    return None\n"
    opener, signature, key = _signed(files)
    monkeypatch.setattr(updater, "_PINNED_KEYSET", keyset(
        1, [key], [Ed25519PrivateKey.generate() for _ in range(3)]))
    monkeypatch.setattr(_version, "read_version", lambda: "1")
    set_config(monkeypatch, lambda: {"update_url": "https://updates.invalid"})
    monkeypatch.setattr(config, "HOME_DIR", update.root / "home")
    uv = update.root / ".uv" / ("uv.cmd" if os.name == "nt" else "uv.sh")
    uv.parent.mkdir(parents=True)
    uv.write_text("@exit /b 0\r\n" if os.name == "nt" else "#!/bin/sh\nexit 0\n",
                  encoding="utf-8")
    uv.chmod(0o755)
    (update.root / ".localm-install.json").write_text(json.dumps({"uv_exe": str(uv)}),
                                                      encoding="utf-8")
    tree = update.tree()
    prepared = updater_handoff.prepare(3, signature=signature, download_opener=opener,
                                       installed=update.root, say=lambda _t: None)
    work = update.root / ".localm-update" / prepared.id
    try:
        assert not prepared.legacy and prepared.new == "2"
        assert applier.is_older(update.root, prepared.id)
        own = work / "applier" / "update_mechanism"
        assert (own / "applier.py").read_bytes() \
            == (REPO / "localm" / "update_mechanism" / "applier.py").read_bytes()
        listed = update_package.parse(
            (work / "new" / update_package.NAME).read_text(encoding="utf-8"))
        assert listed["version"] == "2" and set(listed["files"]) == set(files)
        assert listed["files"] == um.stated(files)
        assert json.loads((work / "check.json").read_text(encoding="utf-8"))["refused"] is None
        assert prepared.notes == [steps_update.OLDER_NOTE if catches_up
                                  else steps_update.OLDER_NOTE_AS_THEY_ARE]
        assert "setup-llama --force" in steps_update.OLDER_NOTE_AS_THEY_ARE
    finally:
        prepared.discard()
    assert not work.exists() and update.tree() == tree


@pytest.mark.parametrize("needs", ["trust", "uv"])
def test_an_older_release_chosen_by_version_gets_the_bridges_and_a_new_list(
        tmp_path, monkeypatch, needs):
    """A release from before 0.3.2 chosen with ``--to`` gets this build's signature
    check, one from 0.3.1 or earlier the uv of this install, as the old way of updating
    gave them; the list of its files is made after that, so it matches what is moved."""
    from localm import update_package, updater, updater_handoff
    new = tmp_path / "new"
    files = {"VERSION": "0.3.0", "localm/updater.py": "# an old updater\n",
             "localm/_apply_update.py": ('def f():\n    return ["uv", "pip"]\n'
                                         if needs == "uv" else "")}
    if needs == "uv":
        files["localm/updater.py"] += f"{updater._SIG_FORMAT}\n_REVOKED_KEYS = ()\n"
    um.write(new, files)
    monkeypatch.setattr(updater, "_uv_for_deps", lambda target: str(tmp_path / "uv.exe"))
    notes = updater_handoff._ready_older(tmp_path, new, "0.3.0", "0.3.0")
    listed = update_package.parse((new / update_package.NAME).read_text(encoding="utf-8"))
    assert listed["files"] == update_package.files(new)
    text = (new / "localm" / "updater.py").read_text(encoding="utf-8")
    if needs == "trust":
        assert (new / "localm" / "_update_trust_bridge.py").is_file()
        assert "_localm_bridged_verify_signature" in text
        assert any("signature check" in n for n in notes)
    else:
        assert "_localm_bridged_post_swap_command" in text
        assert any("uv" in n for n in notes) and len(notes) == 1
    assert updater_handoff._ready_older(tmp_path, new, "0.3.0", None) == []


@pytest.mark.parametrize("prints, stderr, ok", [
    ("localm, version 2", "a warning after it\n", True),
    ("localm, version 2", "localm, version 3\n", True),
    ("localm, version 3", "localm, version 2\n", False),
    ("something else", "localm, version 2\n", False)])
def test_the_check_of_an_older_release_reads_the_version_line_it_prints(
        tmp_path, prints, stderr, ok):
    """``localm --version`` of a release without ``_update-probe``: the version line on
    its standard output decides, whatever its standard error says after it."""
    import sys

    from localm.update_mechanism import steps_probe
    um.write(tmp_path, {"localm/__init__.py": "", "localm/__main__.py": (
        f"import sys\nprint({prints!r})\nsys.stderr.write({stderr!r})\n")})
    why = steps_probe._version_of(sys.executable, tmp_path, version="2",
                                  env={**os.environ, "PYTHONPATH": str(tmp_path)},
                                  flags=dict(HIDDEN),
                                  limit_s=60)
    assert (why is None) == ok, why
    if not ok:
        assert why.startswith("localm 2 could not be checked: it said ")


def test_the_list_an_older_release_ships_is_checked_against_it(tmp_path):
    """An older release that ships the list of its files: the list is checked against
    the release before anything else, as the old way of updating checked it; one that
    does not match is refused, and one that matches is kept as it is."""
    from localm import update_package, updater_components, updater_handoff
    from localm.bugreport import LocalmError
    new = tmp_path / "new"
    um.write(new, {"VERSION": "0.3.3", "localm/a.py": "a = 1\n"})
    listed = {"format": update_package.FORMAT, "version": "0.3.3",
              "files": update_package.files(new),
              "components": updater_components.stated_components(new)}
    statement = new / update_package.NAME
    statement.write_text(json.dumps(listed), encoding="utf-8")
    assert updater_handoff._ready_older(tmp_path, new, "0.3.3", None) == []
    assert json.loads(statement.read_text(encoding="utf-8")) == listed
    (new / "localm" / "a.py").write_text("a = 2\n", encoding="utf-8")
    with pytest.raises(LocalmError, match="does not match what it states"):
        updater_handoff._ready_older(tmp_path, new, "0.3.3", None)
