# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bringing a model into the models folder never takes a file of the user away
unless the model is then registered: duplicates are decided before anything is
transferred, and whatever fails afterwards puts every file back."""

from __future__ import annotations

import errno
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from localm import model_manager as mm
from localm.config import load_registry
from localm.model_manager import registry as _registry
from localm.model_manager.gguf import TRANSFER_TEMP_PREFIX
from tests import test_folder_import_progress as _progress

home, printed, _gguf = _progress.home, _progress.printed, _progress._gguf

PARTS = ("big-00001-of-00002.gguf", "big-00002-of-00002.gguf")


def _names(home: Path) -> list[str]:
    return sorted(p.name for p in (home / "models").iterdir())


def _transferred(printed: list[str]) -> bool:
    return any(line.startswith(("[dim]Moving", "[dim]Copying")) for line in printed)


def _no_links(monkeypatch) -> None:
    def _link(src, dst, **kw):
        raise OSError(errno.EPERM, "links are not supported here")
    monkeypatch.setattr(_registry.os, "link", _link)


def _other_volume(monkeypatch) -> None:
    _no_links(monkeypatch)
    real = os.rename

    def _rename(src, dst):
        if not Path(src).name.startswith(TRANSFER_TEMP_PREFIX):
            raise OSError(errno.EXDEV, "cross-device")
        return real(src, dst)
    monkeypatch.setattr(_registry.os, "rename", _rename)


class TestAByteIdenticalFile:
    @pytest.mark.parametrize("action", ["move", "copy"])
    @pytest.mark.parametrize("answer, gui, result", [
        ("skip", False, False), ("ask", False, False), ("alias", False, True), ("ask", True, True)])
    def test_it_is_left_where_it_is_and_nothing_is_transferred(
            self, home, tmp_path, printed, monkeypatch, action, answer, gui, result):
        _gguf(home / "models" / "first.gguf")
        mm._register("first", home / "models" / "first.gguf")
        mine = _gguf(tmp_path / "mine" / "second.gguf")
        if gui:
            monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")
        assert mm.add_local(str(mine), "second", on_duplicate=answer, store=action) is result
        assert mine.is_file() and _names(home) == ["first.gguf"]
        assert not _transferred(printed), printed
        assert set(load_registry()) == ({"first", "second"} if result else {"first"})
        assert any("was left where it is" in line for line in printed), printed

    @pytest.mark.parametrize("action", ["move", "copy"])
    def test_register_anyway_brings_it_in(self, home, tmp_path, printed, action):
        _gguf(home / "models" / "first.gguf")
        mm._register("first", home / "models" / "first.gguf")
        mine = _gguf(tmp_path / "mine" / "second.gguf")
        assert mm.add_local(str(mine), "second", on_duplicate="register", store=action) is True
        assert _names(home) == ["first.gguf", "second.gguf"]
        assert mine.exists() is (action == "copy")
        assert Path(load_registry()["second"]["path"]).parent.name == "models"

    @pytest.mark.parametrize("action", ["move", "copy"])
    def test_in_a_folder_import_it_is_left_and_the_rest_is_brought_in(
            self, home, tmp_path, printed, action):
        _gguf(home / "models" / "first.gguf")
        mm._register("first", home / "models" / "first.gguf")
        pile = tmp_path / "pile"
        twin = _gguf(pile / "twin.gguf")
        _gguf(pile / "new.gguf", payload=b"n" * 64)
        assert mm.add_local(str(pile), store=action) is True
        assert twin.is_file() and _names(home) == ["first.gguf", "new.gguf"]
        assert not any("twin.gguf to " in line for line in printed), printed
        assert set(load_registry()) == {"first", "new"}


class TestARegistrationThatFails:
    @pytest.fixture(params=["link", "rename", "copy"])
    def how(self, request, monkeypatch):
        if request.param == "rename":
            _no_links(monkeypatch)
        elif request.param == "copy":
            _other_volume(monkeypatch)
        return request.param

    @pytest.mark.parametrize("error", [OSError("disk full"), KeyboardInterrupt()])
    def test_every_part_is_back_where_it_was(self, home, tmp_path, printed, monkeypatch,
                                             how, error):
        src = tmp_path / "mine"
        parts = [_gguf(src / n, payload=n.encode()) for n in PARTS]
        before = [p.read_bytes() for p in parts]

        def _fail(*a, **k):
            raise error
        monkeypatch.setattr(mm, "_register", _fail)
        with pytest.raises(type(error)):
            mm.add_local(str(parts[0]), store="move")
        assert [p.read_bytes() for p in parts] == before
        assert _names(home) == [] and load_registry() == {}

    def test_in_a_folder_import_only_the_registered_model_is_kept(
            self, home, tmp_path, printed, monkeypatch, how):
        pile = tmp_path / "pile"
        a, b = _gguf(pile / "a.gguf"), _gguf(pile / "b.gguf", payload=b"b" * 64)
        real = mm._register

        def _register(name, *args, **kw):
            if name == "b":
                raise OSError("disk full")
            return real(name, *args, **kw)
        monkeypatch.setattr(mm, "_register", _register)
        with pytest.raises(OSError):
            mm.add_local(str(pile), store="move")
        assert not a.exists() and b.is_file()
        assert _names(home) == ["a.gguf"] and set(load_registry()) == {"a"}

    def test_a_folder_model_is_put_back_under_its_name_with_the_old_entry(
            self, home, tmp_path, printed, monkeypatch):
        src = tmp_path / "hfm"
        src.mkdir()
        (src / "config.json").write_text('{"model_type": "llama"}')
        (src / "tokenizer.json").write_text("{}")
        (src / "model.safetensors").write_bytes(b"x")

        def _fail(*a, **k):
            raise KeyboardInterrupt
        monkeypatch.setattr(mm, "_register", _fail)
        with pytest.raises(KeyboardInterrupt):
            mm.add_local(str(src), store="move")
        assert (src / "model.safetensors").read_bytes() == b"x" and _names(home) == []

    def test_an_unreadable_registry_undoes_nothing(self, home, tmp_path, printed):
        src = _gguf(tmp_path / "mine" / "m.gguf")
        dest = home / "models" / "m.gguf"
        placement = _registry._Placement()
        placement.place_file(src, dest, "move", dest)
        (home / "registry.json").write_text("{ not json", encoding="utf-8")
        placement.settle()
        assert src.is_file() and dest.is_file()
        assert any("could not be read" in line for line in printed), printed


class TestAMoveToAnotherVolume:
    def test_the_source_goes_only_after_the_registration(self, home, tmp_path, printed,
                                                         monkeypatch):
        _other_volume(monkeypatch)
        src = _gguf(tmp_path / "mine" / "m.gguf")
        seen = []
        real = mm._register

        def _register(*a, **k):
            seen.append(src.is_file())
            return real(*a, **k)
        monkeypatch.setattr(mm, "_register", _register)
        assert mm.add_local(str(src), store="move") is True
        assert seen == [True] and not src.exists() and _names(home) == ["m.gguf"]

    def test_a_copy_that_differs_is_refused_and_removed(self, home, tmp_path, printed,
                                                        monkeypatch):
        _other_volume(monkeypatch)
        src = _gguf(tmp_path / "mine" / "m.gguf")
        real = _registry.shutil.copy2

        def _copy(a, b, **kw):
            out = real(a, b, **kw)
            Path(b).write_bytes(Path(b).read_bytes()[:-1] + b"X")
            return out
        monkeypatch.setattr(_registry.shutil, "copy2", _copy)
        assert mm.add_local(str(src), store="move") is False
        assert src.is_file() and _names(home) == [] and load_registry() == {}
        assert any("Copy verification failed" in line for line in printed), printed

    @pytest.mark.parametrize("action", ["move", "copy"])
    def test_an_interrupted_copy_leaves_no_file_a_scan_would_register(
            self, home, tmp_path, printed, monkeypatch, action):
        _other_volume(monkeypatch)
        src = _gguf(tmp_path / "mine" / "m.gguf", payload=b"p" * 4096)

        def _copy(a, b, **kw):
            Path(b).write_bytes(Path(a).read_bytes()[:-100])
            assert mm.sync_models_dir(backfill_mmproj=False).added == 0
            raise KeyboardInterrupt
        monkeypatch.setattr(_registry.shutil, "copy2", _copy)
        monkeypatch.setattr(mm, "_GGUF_SETTLE_SECONDS", 0)
        with pytest.raises(KeyboardInterrupt):
            mm.add_local(str(src), store=action)
        assert src.is_file() and _names(home) == [] and load_registry() == {}

    def test_a_full_disk_is_reported_and_leaves_nothing(self, home, tmp_path, printed,
                                                        monkeypatch):
        src = _gguf(tmp_path / "mine" / "m.gguf")

        def _copy(a, b, **kw):
            Path(b).write_bytes(b"half")
            raise OSError(errno.ENOSPC, "No space left on device")
        monkeypatch.setattr(_registry.shutil, "copy2", _copy)
        assert mm.add_local(str(src), store="copy") is False
        assert src.is_file() and _names(home) == []
        assert any("No space left on device" in line for line in printed), printed

    def test_a_half_copied_file_of_a_dead_process_is_removed_by_the_next_import(
            self, home, tmp_path, printed):
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        left = home / "models" / f"{TRANSFER_TEMP_PREFIX}old.gguf.{child.pid}.0123abcd"
        left.write_bytes(b"half")
        assert mm.add_local(str(_gguf(tmp_path / "m.gguf")), store="copy") is True
        assert _names(home) == ["m.gguf"]


class TestAFileRegisteredWhereItIs:
    def test_moving_it_in_keeps_its_name_and_entry(self, home, tmp_path, printed):
        src = _gguf(tmp_path / "mine" / "m.gguf")
        assert mm.add_local(str(src)) is True
        mm.update_registry(lambda r: r["m"].__setitem__("note", "kept"))
        assert mm.add_local(str(src), store="move") is True
        reg = load_registry()
        assert set(reg) == {"m"} and reg["m"]["note"] == "kept"
        assert Path(reg["m"]["path"]) == (home / "models" / "m.gguf").resolve()
        assert not src.exists() and _names(home) == ["m.gguf"]

    def test_copying_it_in_points_its_entry_at_the_copy(self, home, tmp_path, printed):
        src = _gguf(tmp_path / "mine" / "m.gguf")
        assert mm.add_local(str(src)) is True
        assert mm.add_local(str(src), store="copy") is True
        reg = load_registry()
        assert set(reg) == {"m"}
        assert Path(reg["m"]["path"]) == (home / "models" / "m.gguf").resolve()
        assert src.is_file()

    def test_a_folder_import_moves_it_in_under_its_name(self, home, tmp_path, printed):
        pile = tmp_path / "pile"
        _gguf(pile / "a.gguf")
        assert mm.add_local(str(pile)) is True
        assert mm.rename_model("a", "renamed") is True
        assert mm.add_local(str(pile), store="move") is True
        reg = load_registry()
        assert set(reg) == {"renamed"}
        assert Path(reg["renamed"]["path"]) == (home / "models" / "a.gguf").resolve()

    def test_a_registry_that_cannot_be_updated_moves_nothing(self, home, tmp_path, printed,
                                                             monkeypatch):
        src = _gguf(tmp_path / "mine" / "m.gguf")
        assert mm.add_local(str(src)) is True
        before = load_registry()

        def _update(fn):
            raise TimeoutError("registry.json.lock is held")
        monkeypatch.setattr(mm, "update_registry", _update)
        assert mm.add_local(str(src), store="move") is False
        assert src.is_file() and _names(home) == [] and load_registry() == before


class TestAMoveThatWasKilled:
    def test_the_next_one_finishes_it(self, home, tmp_path, printed):
        src = _gguf(tmp_path / "mine" / "m.gguf")
        os.link(src, home / "models" / "m.gguf")
        assert mm.sync_models_dir(backfill_mmproj=False).added == 1
        assert mm.add_local(str(src), store="move") is True
        assert not src.exists() and _names(home) == ["m.gguf"]
        assert set(load_registry()) == {"m"}

    def test_a_different_file_of_that_name_still_refuses(self, home, tmp_path, printed):
        src = _gguf(tmp_path / "mine" / "m.gguf")
        _gguf(home / "models" / "m.gguf", payload=b"other")
        assert mm.add_local(str(src), "other-name", store="move") is False
        assert src.is_file()


def test_a_read_only_file_is_moved_and_stays_read_only(home, tmp_path, printed):
    src = _gguf(tmp_path / "mine" / "m.gguf")
    os.chmod(src, stat.S_IREAD)
    dest = home / "models" / "m.gguf"
    try:
        assert mm.add_local(str(src), store="move") is True
        assert not src.exists() and not dest.stat().st_mode & stat.S_IWRITE
    finally:
        os.chmod(dest if dest.exists() else src, stat.S_IREAD | stat.S_IWRITE)


@pytest.mark.parametrize("nested", [False, True])
def test_a_scan_leaves_the_folder_of_an_unfinished_pull_alone(home, printed, nested):
    folder = home / "models" / "repo"
    if nested:
        _gguf(folder / "sub" / "m.gguf")
    else:
        folder.mkdir()
        (folder / "config.json").write_text('{"model_type": "llama"}')
        (folder / "tokenizer.json").write_text("{}")
    claim = folder / _registry.SNAPSHOT_CLAIM
    claim.write_text('{"repo": "o/r", "revision": null}')
    assert mm.sync_models_dir(backfill_mmproj=False).added == 0
    claim.unlink()
    assert mm.sync_models_dir(backfill_mmproj=False).added == 1
