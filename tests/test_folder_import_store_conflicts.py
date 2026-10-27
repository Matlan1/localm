# SPDX-License-Identifier: AGPL-3.0-or-later
"""Importing a folder with ``--store`` when the models folder already holds a
DIFFERENT file of the same name: the conflicting model is skipped, every other
model is stored AND registered, and the counts say so. Re-adding an imported
folder reports its models as already registered."""

from __future__ import annotations

from pathlib import Path

from localm import model_manager as mm
from localm.config import load_registry
from tests import test_adopt_hub_cache as _hub
from tests import test_folder_import_progress as _progress

_REV_A, _gguf_bytes, _repo = _hub._REV_A, _hub._gguf_bytes, _hub._repo
_folder, _gguf = _progress._folder, _progress._gguf
isolated_home, layout = _hub.isolated_home, _hub.layout
home, printed = _progress.home, _progress.printed


def _models_names(home: Path) -> set[str]:
    return {p.name for p in (home / "models").iterdir()}


class TestHubCacheStoreWithConflict:
    def _cache(self, tmp_path, layout) -> Path:
        root = tmp_path / "hub"
        root.mkdir()
        _repo(root, layout, "org/pair", _REV_A, {
            "alpha.Q4_K_M.gguf": _gguf_bytes("alpha"),
            "beta.Q4_K_M.gguf": _gguf_bytes("beta"),
        })
        return root

    def test_the_rest_registers_and_nothing_is_copied_unregistered(
            self, tmp_path, isolated_home, layout):
        (isolated_home / "models" / "alpha.Q4_K_M.gguf").write_bytes(_gguf_bytes("other"))
        root = self._cache(tmp_path, layout)
        assert mm.add_local(str(root), store="copy") is True
        reg = load_registry()
        registered = {Path(e["path"]).name for e in reg.values()}
        assert "beta.Q4_K_M.gguf" in registered
        stored_in_models = _models_names(isolated_home) - {"alpha.Q4_K_M.gguf"}
        assert stored_in_models == {"beta.Q4_K_M.gguf"}
        assert (isolated_home / "models" / "alpha.Q4_K_M.gguf").read_bytes() \
            == _gguf_bytes("other")
        assert not any(Path(e["path"]).name == "alpha.Q4_K_M.gguf"
                       and Path(e["path"]).parent != (isolated_home / "models").resolve()
                       for e in reg.values())

    def test_a_single_model_repo_whose_only_file_conflicts_registers_nothing(
            self, tmp_path, isolated_home, layout):
        (isolated_home / "models" / "solo.Q4_K_M.gguf").write_bytes(_gguf_bytes("other"))
        root = tmp_path / "hub"
        root.mkdir()
        _repo(root, layout, "org/solo", _REV_A, {"solo.Q4_K_M.gguf": _gguf_bytes("solo")})
        assert mm.add_local(str(root), store="copy") is False
        assert load_registry() == {}
        assert _models_names(isolated_home) == {"solo.Q4_K_M.gguf"}


class TestFolderStoreWithConflict:
    def _conflicted(self, home, tmp_path) -> Path:
        _gguf(home / "models" / "alpha.gguf", payload=b"\x09" * 2048)
        return _folder(tmp_path / "pile")

    def test_labels_reach_the_total_and_the_conflict_counts_as_skipped(
            self, home, tmp_path, printed):
        folder = self._conflicted(home, tmp_path)
        assert mm.add_local(str(folder), store="copy") is True
        assert any("Hashing beta.gguf for duplicate detection (1 of 2" in line
                   for line in printed), printed
        assert any("Hashing gamma.gguf for duplicate detection (2 of 2" in line
                   for line in printed), printed
        assert not any("of 3" in line for line in printed), printed
        done = printed[-1]
        assert done == (
            "Done: 2 registered, 0 already registered, 1 skipped or refused "
            "(1 because a different file with the same name is already in the "
            "models folder), 0 not usable"), done

    def test_every_other_model_is_stored_and_registered(self, home, tmp_path, printed):
        folder = self._conflicted(home, tmp_path)
        mm.add_local(str(folder), store="move")
        reg = load_registry()
        assert {Path(e["path"]).name for e in reg.values()} == {"beta.gguf", "gamma.gguf"}
        assert (folder / "alpha.gguf").exists()
        assert _models_names(home) == {"alpha.gguf", "beta.gguf", "gamma.gguf"}


class TestReaddingAFolder:
    def test_second_add_reports_every_model_as_already_registered(
            self, home, tmp_path, printed):
        folder = _folder(tmp_path / "pile")
        assert mm.add_local(str(folder)) is True
        first = dict(load_registry())
        printed.clear()
        assert mm.add_local(str(folder)) is True
        assert printed[-1] == (
            "Done: 0 registered, 3 already registered, 0 skipped or refused, 0 not usable"
        ), printed[-1]
        assert load_registry() == first

    def test_a_suffixed_registration_is_found_again(self, home, tmp_path, printed):
        folder = _folder(tmp_path / "pile")
        _gguf(tmp_path / "elsewhere" / "alpha.gguf", payload=b"" * 2048)
        mm.add_local(str(tmp_path / "elsewhere"))
        mm.add_local(str(folder))
        before = dict(load_registry())
        assert "alpha-2" in before
        printed.clear()
        assert mm.add_local(str(folder), on_duplicate="skip") is True
        assert load_registry() == before
        assert "0 registered, 3 already registered, 0 skipped" in printed[-1], printed[-1]

    def test_a_hugging_face_folder_inside_is_also_already_registered(
            self, home, tmp_path, printed):
        folder = tmp_path / "pile"
        hf = folder / "hfm"
        hf.mkdir(parents=True)
        (hf / "config.json").write_text('{"model_type": "llama"}')
        (hf / "tokenizer.json").write_text("{}")
        (hf / "model.safetensors").write_bytes(b"x" * 64)
        _gguf(folder / "alpha.gguf", payload=b"" * 2048)
        assert mm.add_local(str(folder)) is True
        before = dict(load_registry())
        printed.clear()
        assert mm.add_local(str(folder)) is True
        assert load_registry() == before
        assert printed[-1] == (
            "Done: 0 registered, 2 already registered, 0 skipped or refused, 0 not usable"
        ), printed[-1]
