# SPDX-License-Identifier: AGPL-3.0-or-later
"""An update into a release that ships no update program of its own (an older one,
``localm update --to``): the installed version gives it a copy of its own program and,
when it has none, the list of its files; that program installs its files by that list
with the gate last, its packages from its own lock, checks it with its command line and
leaves the rest to that release's first start."""

from __future__ import annotations

import json
import os
import shutil
import time

import pytest

from localm.update_mechanism import applier, mover, steps_update
from tests import _update_mechanism as um
from tests.test_update_applier import Update, _signed, _with_main
from tests.test_update_gate import ID, NEW, REPO, _statement

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


def test_into_a_release_without_the_gate_the_gate_files_go_last(update):
    """The plan into an older release: every other file first, then the launchers, then
    ``localm/__init__.py``, and ``localm/procs.py``, which that gate imports, after it."""
    new = _older()
    del new["localm/procs.py"]
    folder = update.root / ".localm-update" / ID / "new"
    shutil.rmtree(folder)
    um.write(folder, new)
    old = json.loads((update.root / "update-package.json").read_text(encoding="utf-8"))
    plan = mover.plan_files(update.root, ID, um.stated(new), old["files"], protect=[],
                            swap=mover.GATE_FILES_LAST, last=True)
    order = [op.path for op in plan.ops]
    gate = [p for p in order if p in mover.GATE_FILES]
    assert order[-len(gate):] == gate, order
    assert gate[-2:] == ["localm/__init__.py", "localm/procs.py"]
    assert next(op for op in plan.ops if op.path == "localm/procs.py").new is None
    assert len(gate) > 2


def test_prepare_gives_an_older_build_this_program_and_the_list_of_its_files(
        update, monkeypatch):
    """``updater_handoff.prepare`` of a build that ships no update program and no list of
    its files: the list is made from its verified files, this build's program is copied
    in and runs the check, which prepares only the packages."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from localm import _version, config, update_package, updater, updater_handoff
    from tests._net_config import set_config
    from tests._update_signing import keyset
    files = _older()
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
        assert prepared.notes == [steps_update.OLDER_NOTE]
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
