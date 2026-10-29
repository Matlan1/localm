# SPDX-License-Identifier: AGPL-3.0-or-later
"""Storing a model folder into the models folder: the folder is built under a
temporary name and renamed into place, so a destination another import created
meanwhile is never replaced or removed, an interrupted copy leaves nothing a scan
would register, and a leftover from a dead process is cleaned on the next import."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from localm import model_manager as mm
from localm.config import load_registry
from localm.model_manager import registry as _registry
from localm.model_manager.gguf import TRANSFER_TEMP_PREFIX, _find_model_units
from tests import test_folder_import_progress as _progress
from tests.test_folder_import_store_conflicts import _cross_device

home, printed, _REPO = _progress.home, _progress.printed, _progress._REPO


def _hf(parent: Path, name: str = "model", payload: bytes = b"x" * 64) -> Path:
    folder = parent / name
    folder.mkdir(parents=True)
    (folder / "config.json").write_text('{"model_type": "llama"}')
    (folder / "tokenizer.json").write_text("{}")
    (folder / "model.safetensors").write_bytes(payload)
    return folder


def _temps(home: Path) -> list[str]:
    return [p.name for p in (home / "models").iterdir()
            if p.name.startswith(TRANSFER_TEMP_PREFIX)]


def _another_import_creates(monkeypatch, dest: Path):
    real = mm._check_disk_space

    def _gate(*a, **k):
        dest.mkdir()
        (dest / "marker.txt").write_text("the other import's model")
        return real(*a, **k)
    monkeypatch.setattr(mm, "_check_disk_space", _gate)


class TestADestinationAnotherImportCreated:
    @pytest.mark.parametrize("action, cross_device", [
        ("copy", False), ("move", False), ("move", True)])
    def test_it_survives_and_the_source_is_untouched(
            self, home, tmp_path, printed, monkeypatch, action, cross_device):
        src = _hf(tmp_path / "pile")
        if cross_device:
            _cross_device(monkeypatch)
        dest = home / "models" / "model"
        _another_import_creates(monkeypatch, dest)
        assert mm.add_local(str(src), store=action) is False
        assert (dest / "marker.txt").read_text() == "the other import's model"
        assert sorted(p.name for p in dest.iterdir()) == ["marker.txt"]
        assert (src / "model.safetensors").read_bytes() == b"x" * 64
        assert _temps(home) == []
        assert load_registry() == {}
        assert any("already exists" in line for line in printed), printed


class TestAnInterruptedCopy:
    @pytest.mark.parametrize("where", ["copying", "verifying"])
    def test_leaves_nothing_behind_and_the_retry_works(
            self, home, tmp_path, printed, monkeypatch, where):
        src = _hf(tmp_path / "pile")
        real_copytree, real_verify = _registry.shutil.copytree, _registry._verify_dir_copy
        if where == "copying":
            def _interrupted(s, d, *a, **k):
                real_copytree(s, d, *a, **k)
                raise KeyboardInterrupt
            monkeypatch.setattr(_registry.shutil, "copytree", _interrupted)
        else:
            def _interrupted(*a, **k):
                raise KeyboardInterrupt
            monkeypatch.setattr(_registry, "_verify_dir_copy", _interrupted)
        with pytest.raises(KeyboardInterrupt):
            mm.add_local(str(src), store="copy")
        assert sorted(p.name for p in (home / "models").iterdir()) == []
        assert (src / "model.safetensors").read_bytes() == b"x" * 64
        monkeypatch.setattr(_registry.shutil, "copytree", real_copytree)
        monkeypatch.setattr(_registry, "_verify_dir_copy", real_verify)
        assert mm.add_local(str(src), store="copy") is True
        assert (home / "models" / "model" / "config.json").is_file()


class TestTemporaryFolders:
    def _dead_pid(self) -> int:
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        return child.pid

    def test_a_scan_never_registers_one(self, home, printed):
        _hf(home / "models", f"{TRANSFER_TEMP_PREFIX}model.{os.getpid()}.0123abcd")
        assert mm.sync_models_dir(backfill_mmproj=False).added == 0
        assert load_registry() == {}
        assert _find_model_units(home / "models") == ([], [])

    def test_a_retry_removes_the_leftover_of_a_dead_process_only(
            self, home, tmp_path, printed):
        models = home / "models"
        dead = f"{TRANSFER_TEMP_PREFIX}old.{self._dead_pid()}.0123abcd"
        live = f"{TRANSFER_TEMP_PREFIX}busy.{os.getpid()}.0123abcd"
        _hf(models, dead)
        _hf(models, live)
        (models / ".cache").mkdir()
        (models / f"{TRANSFER_TEMP_PREFIX}not-ours").mkdir()
        assert mm.add_local(str(_hf(tmp_path / "pile")), store="copy") is True
        assert sorted(p.name for p in models.iterdir()) == sorted(
            [".cache", f"{TRANSFER_TEMP_PREFIX}not-ours", live, "model"])
        assert any("half-copied" in line for line in printed), printed

    def test_a_registered_folder_is_never_removed(self, home, tmp_path, printed):
        models = home / "models"
        name = f"{TRANSFER_TEMP_PREFIX}odd.{self._dead_pid()}.0123abcd"
        kept = _hf(models, name)
        mm._register("odd", kept)
        _registry._remove_stale_transfer_folders()
        assert (kept / "config.json").is_file()


class TestRenameRefusedWhileTheFolderIsFree:
    def _flaky_rename(self, monkeypatch, times):
        real = os.rename
        calls = {"n": 0}

        def _rename(src, dst):
            if Path(src).name.startswith(TRANSFER_TEMP_PREFIX):
                calls["n"] += 1
                if calls["n"] <= times:
                    raise PermissionError(5, "Access is denied")
            return real(src, dst)
        monkeypatch.setattr(_registry.os, "rename", _rename)
        monkeypatch.setattr(_registry, "_RENAME_RETRY_SECONDS", 0)
        return calls

    def test_a_refusal_that_passes_is_retried(self, home, tmp_path, printed, monkeypatch):
        calls = self._flaky_rename(monkeypatch, 2)
        assert mm.add_local(str(_hf(tmp_path / "pile")), store="copy") is True
        assert calls["n"] == 3
        assert (home / "models" / "model" / "model.safetensors").read_bytes() == b"x" * 64

    def test_a_refusal_that_stays_is_reported_and_cleaned_up(
            self, home, tmp_path, printed, monkeypatch):
        self._flaky_rename(monkeypatch, 10 ** 6)
        src = _hf(tmp_path / "pile")
        assert mm.add_local(str(src), store="copy") is False
        assert list((home / "models").iterdir()) == []
        assert (src / "model.safetensors").read_bytes() == b"x" * 64
        assert any("Access is denied" in line for line in printed), printed


class TestTheRegistryUpdateAfterAMove:
    def _lock_timeout(self, monkeypatch):
        real = mm.update_registry

        def _update(fn):
            if fn.__name__ == "_repoint":
                raise TimeoutError("registry.json.lock is held")
            return real(fn)
        monkeypatch.setattr(mm, "update_registry", _update)

    def test_a_failure_is_printed_and_never_raised(self, home, tmp_path, printed, monkeypatch):
        src = _hf(tmp_path / "pile")
        mm._register("model", src)
        self._lock_timeout(monkeypatch)
        _registry._repoint_directory_references(src, home / "models" / "model")
        assert any("could not be updated" in line and "registry.json.lock is held" in line
                   for line in printed), printed

    def test_the_rest_of_a_multi_folder_import_goes_on(
            self, home, tmp_path, printed, monkeypatch):
        pile = tmp_path / "pile"
        _hf(pile, "one")
        _hf(pile, "two", payload=b"y" * 64)
        assert mm.add_local(str(pile)) is True
        self._lock_timeout(monkeypatch)
        mm.add_local(str(pile), store="move")
        assert (home / "models" / "one" / "model.safetensors").read_bytes() == b"x" * 64
        assert (home / "models" / "two" / "model.safetensors").read_bytes() == b"y" * 64
        assert not (pile / "one").exists() and not (pile / "two").exists()
        assert sum("could not be updated" in line for line in printed) == 2, printed


class TestMovingAFolderRegisteredUnderAnotherName:
    def test_it_is_reported_as_moved_and_registered(self, home, tmp_path, printed):
        src = _hf(tmp_path / "ext", "hfm")
        assert mm.add_local(str(src), "mymodel") is True
        printed.clear()
        assert mm.add_local(str(src), store="move", on_duplicate="skip") is True
        reg = load_registry()
        assert set(reg) == {"mymodel"}, set(reg)
        assert Path(reg["mymodel"]["path"]).resolve() == (home / "models" / "hfm").resolve()
        assert not src.exists()
        assert not any("not registered" in line for line in printed), printed
        assert any("Already registered as mymodel" in line for line in printed), printed

    def test_a_folder_registered_nowhere_still_registers_under_its_own_name(
            self, home, tmp_path, printed):
        src = _hf(tmp_path / "ext", "hfm")
        assert mm.add_local(str(src), store="move") is True
        assert set(load_registry()) == {"hfm"}


_CHILD = textwrap.dedent("""
    import sys
    import time
    from pathlib import Path
    from localm import model_manager as mm

    src, mine, theirs, action = sys.argv[1:5]
    real = mm._check_disk_space

    def gate(*a, **k):
        Path(mine).write_text("ready")
        deadline = time.time() + 120
        while not Path(theirs).exists():
            if time.time() > deadline:
                raise SystemExit(3)
            time.sleep(0.02)
        return real(*a, **k)

    mm._check_disk_space = gate
    print("RESULT", mm.add_local(src, store=action))
""")


class TestTwoImportsOfTheSameNameAtOnce:
    @pytest.mark.parametrize("action", ["move", "copy"])
    def test_one_wins_and_the_other_deletes_nothing(self, home, tmp_path, action):
        a = _hf(tmp_path / "a", payload=b"A" * 64)
        b = _hf(tmp_path / "b", payload=b"B" * 64)
        flags = tmp_path / "flags"
        flags.mkdir()
        env = {**os.environ, "LOCALM_HOME": str(home), "PYTHONPATH": str(_REPO)}
        procs = []
        for src, mine, theirs in ((a, "a", "b"), (b, "b", "a")):
            procs.append(subprocess.Popen(
                [sys.executable, "-c", _CHILD, str(src), str(flags / mine),
                 str(flags / theirs), action],
                env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True))
        outs = [p.communicate(timeout=300) for p in procs]
        assert [p.returncode for p in procs] == [0, 0], outs
        results = ["RESULT True" in out for out, _ in outs]
        assert sorted(results) == [False, True], outs
        winner, loser = (a, b) if results[0] else (b, a)
        stored = home / "models" / "model"
        assert (stored / "model.safetensors").read_bytes() == \
            (b"A" if winner is a else b"B") * 64
        assert (loser / "model.safetensors").exists()
        assert (winner / "model.safetensors").exists() == (action == "copy")
        assert _temps(home) == []
        assert [Path(e["path"]).resolve() for e in load_registry().values()] == [
            stored.resolve()]
