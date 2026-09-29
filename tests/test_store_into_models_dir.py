# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm add/pull PATH --store copy|move` brings an external model file/dir
INTO <data dir>/models before registering it, instead of registering it in
place. Covers the shared helper (_store_into_models_dir /
_store_loose_gguf_dir) directly, add_local()'s and pull_model()'s --store
threading, and the `localm add` / `localm pull` CLI options.

Split GGUFs, a sibling mmproj vision-projector file, an HF-style model
directory, and a folder of several independent loose GGUFs (including a
model+mmproj pair) all have to travel together.
"""

import struct
from pathlib import Path

import pytest

import localm.model_manager as mm
from localm.cli import main
from localm.config import load_registry
from localm.model_manager import add_local, pull_model


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    # load_registry/save_registry/ensure_dirs read config.py's own module-level
    # HOME_DIR/MODELS_DIR (patched below), so registry.json / config.json land in
    # the throwaway dir. _store_into_models_dir and pull.py read MODELS_DIR and
    # HOME_DIR through the localm.model_manager PACKAGE attribute, a plain-value
    # copy taken once at import time, so that has to be patched separately.
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    models_dir = home / "models"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", models_dir)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
    monkeypatch.setattr(mm, "HOME_DIR", home)
    return home


@pytest.fixture
def cli_isolated_home(isolated_home):
    """Same isolation, for invoking the click CLI directly."""
    from click.testing import CliRunner
    return CliRunner()


def _gguf(d: Path, name: str, content: bytes | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(content if content is not None else b"GGUF\x00\x00\x00\x00" + name.encode())
    return p


def _hf_dir(tmp_path: Path, name: str = "myhf") -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text('{"model_type": "llama"}')
    (d / "tokenizer.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"weights")
    return d


def _models_dir() -> Path:
    return mm.MODELS_DIR


# --------------------------------------------------------------------------- #
#  Single-file GGUF
# --------------------------------------------------------------------------- #

class TestStoreSingleFile:
    def test_copy_lands_in_models_dir_and_keeps_original(self, tmp_path, isolated_home):
        src_dir = tmp_path / "external"
        f = _gguf(src_dir, "mymodel.gguf")
        assert add_local(str(f), store="copy") is True

        dest = _models_dir() / "mymodel.gguf"
        assert dest.is_file()
        assert dest.read_bytes() == f.read_bytes()
        assert f.exists()                                   # original untouched
        reg = load_registry()
        assert reg["mymodel"]["path"] == str(dest.resolve())

    def test_move_lands_in_models_dir_and_removes_original(self, tmp_path, isolated_home):
        src_dir = tmp_path / "external"
        f = _gguf(src_dir, "mymodel.gguf")
        original_bytes = f.read_bytes()
        assert add_local(str(f), store="move") is True

        dest = _models_dir() / "mymodel.gguf"
        assert dest.is_file()
        assert dest.read_bytes() == original_bytes
        assert not f.exists()                                # original gone
        reg = load_registry()
        assert reg["mymodel"]["path"] == str(dest.resolve())

    def test_no_store_registers_in_place_unchanged(self, tmp_path, isolated_home):
        """Default behavior (no --store) registers the file in place."""
        src_dir = tmp_path / "external"
        f = _gguf(src_dir, "mymodel.gguf")
        assert add_local(str(f)) is True
        assert f.exists()
        assert not (_models_dir() / "mymodel.gguf").exists()
        reg = load_registry()
        assert reg["mymodel"]["path"] == str(f.resolve())


# --------------------------------------------------------------------------- #
#  Split GGUF - every part must travel together
# --------------------------------------------------------------------------- #

class TestStoreSplitGguf:
    def _split(self, tmp_path):
        src_dir = tmp_path / "external"
        p1 = _gguf(src_dir, "big-00001-of-00003.gguf")
        p2 = _gguf(src_dir, "big-00002-of-00003.gguf")
        p3 = _gguf(src_dir, "big-00003-of-00003.gguf")
        return p1, p2, p3

    def test_copy_moves_every_part(self, tmp_path, isolated_home):
        p1, p2, p3 = self._split(tmp_path)
        assert add_local(str(p1), store="copy") is True
        for p in (p1, p2, p3):
            assert p.exists(), f"original part {p.name} should be untouched"
            assert (_models_dir() / p.name).is_file(), f"part {p.name} not copied"
        reg = load_registry()
        assert reg["big"]["path"] == str((_models_dir() / "big-00001-of-00003.gguf").resolve())

    def test_move_moves_every_part(self, tmp_path, isolated_home):
        p1, p2, p3 = self._split(tmp_path)
        assert add_local(str(p1), store="move") is True
        for p in (p1, p2, p3):
            assert not p.exists(), f"original part {p.name} should be gone"
            assert (_models_dir() / p.name).is_file(), f"part {p.name} not moved"
        reg = load_registry()
        assert reg["big"]["path"] == str((_models_dir() / "big-00001-of-00003.gguf").resolve())

    def test_copy_via_non_first_part_still_moves_all(self, tmp_path, isolated_home):
        """`add PATH` on a non-first split part must normalise to the first part
        (existing behavior) AND still bring every part along under --store."""
        p1, p2, p3 = self._split(tmp_path)
        assert add_local(str(p2), store="copy") is True
        for p in (p1, p2, p3):
            assert (_models_dir() / p.name).is_file()


# --------------------------------------------------------------------------- #
#  GGUF + sibling mmproj vision-projector
# --------------------------------------------------------------------------- #

class TestStoreMmproj:
    def _model_and_mmproj(self, tmp_path):
        src_dir = tmp_path / "external"
        model = _gguf(src_dir, "vision-model.gguf")
        mmproj = _gguf(src_dir, "mmproj-vision-model-f16.gguf")
        return model, mmproj

    def test_copy_brings_mmproj_along(self, tmp_path, isolated_home):
        model, mmproj = self._model_and_mmproj(tmp_path)
        assert add_local(str(model), store="copy") is True
        assert model.exists() and mmproj.exists()            # originals untouched
        assert (_models_dir() / model.name).is_file()
        assert (_models_dir() / mmproj.name).is_file(), \
            "mmproj sibling must travel with the model or vision silently breaks"

    def test_move_brings_mmproj_along(self, tmp_path, isolated_home):
        model, mmproj = self._model_and_mmproj(tmp_path)
        assert add_local(str(model), store="move") is True
        assert not model.exists() and not mmproj.exists()    # both relocated
        assert (_models_dir() / model.name).is_file()
        assert (_models_dir() / mmproj.name).is_file()

    def test_mmproj_findable_after_move(self, tmp_path, isolated_home):
        """get_model_mmproj must still resolve the projector for the model once
        both live under MODELS_DIR."""
        model, mmproj = self._model_and_mmproj(tmp_path)
        add_local(str(model), store="move")
        from localm.model_manager import get_model_mmproj
        found = get_model_mmproj("vision-model")
        assert found is not None
        assert Path(found).name == mmproj.name

    def test_copy_brings_generically_named_mmproj_along(self, tmp_path, isolated_home):
        """A projector named mmproj-F16.gguf, alone with its model and of the
        model's embedding width, travels with it and is attached to it."""
        src_dir = tmp_path / "external"
        model = _text_model(src_dir / "vision-model.gguf", 2560)
        mmproj = _projector(src_dir / "mmproj-F16.gguf", 2560)

        assert add_local(str(model), store="copy") is True

        dest = _models_dir() / mmproj.name
        assert dest.is_file() and dest.read_bytes() == mmproj.read_bytes(), \
            "mmproj sibling must travel with the model or vision silently breaks"
        assert load_registry()["vision-model"].get("mmproj") == str(dest.resolve())
        from localm.model_manager import get_model_mmproj
        assert get_model_mmproj("vision-model") == str(dest.resolve())

    def test_generically_named_mmproj_of_another_width_is_not_attached(
            self, tmp_path, isolated_home):
        src_dir = tmp_path / "external"
        model = _text_model(src_dir / "vision-model.gguf", 2560)
        _projector(src_dir / "mmproj-F16.gguf", 5120)

        assert add_local(str(model), store="copy") is True

        assert "mmproj" not in load_registry()["vision-model"]
        from localm.model_manager import get_model_mmproj
        assert get_model_mmproj("vision-model") is None


# --------------------------------------------------------------------------- #
#  Projector files travel by fit, not only by the auto-attach heuristic
# --------------------------------------------------------------------------- #

_T_UINT32 = 4
_T_STRING = 8


def _gguf_str(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _real_gguf(path: Path, kv) -> Path:
    """A minimal but real GGUF v3 header carrying the metadata *kv*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [b"GGUF", struct.pack("<I", 3), struct.pack("<QQ", 0, len(kv))]
    for key, vtype, val in kv:
        out.append(_gguf_str(key))
        out.append(struct.pack("<I", vtype))
        out.append(_gguf_str(val) if vtype == _T_STRING else struct.pack("<I", val))
    path.write_bytes(b"".join(out))
    return path


def _text_model(path: Path, width: int, arch: str = "gemma3") -> Path:
    """A text model whose ``<arch>.embedding_length`` is *width*; its filename is
    in the metadata so two models never hash alike."""
    return _real_gguf(path, [("general.architecture", _T_STRING, arch),
                             (f"{arch}.embedding_length", _T_UINT32, width),
                             ("general.name", _T_STRING, path.name)])


def _projector(path: Path, width: int, tag: str = "") -> Path:
    """A clip mmproj whose ``clip.vision.projection_dim`` is *width*. *tag*
    (default: the filename) goes into the metadata and decides the bytes."""
    return _real_gguf(path, [("general.architecture", _T_STRING, "clip"),
                             ("clip.vision.projection_dim", _T_UINT32, width),
                             ("general.name", _T_STRING, tag or path.name)])


@pytest.fixture
def wide_console(monkeypatch):
    from tests.conftest import make_console_wide_and_plain
    make_console_wide_and_plain(monkeypatch, width="400")


class TestProjectorTravel:
    def test_move_brings_generically_named_projector_along(self, tmp_path, isolated_home):
        src = tmp_path / "downloads"
        model = _text_model(src / "gemma-3-4b-it-Q4_K_M.gguf", 2560)
        proj = _projector(src / "mmproj-F16.gguf", 2560)
        attached_in_place = mm.find_sibling_mmproj(model)

        assert add_local(str(model), store="move") is True

        moved = _models_dir() / "mmproj-F16.gguf"
        assert moved.is_file(), "the projector must travel with its model"
        assert not proj.exists(), "a move must not leave the projector behind"
        resolved = mm.get_model_mmproj("gemma-3-4b-it-Q4_K_M")
        expected = str(moved.resolve()) if attached_in_place is not None else None
        assert (str(Path(resolved).resolve()) if resolved else None) == expected, (
            "the moved model must resolve exactly the projector it resolved in place")

    def test_copy_brings_generically_named_projector_along(self, tmp_path, isolated_home):
        src = tmp_path / "downloads"
        model = _text_model(src / "gemma-3-4b-it-Q4_K_M.gguf", 2560)
        proj = _projector(src / "mmproj-F16.gguf", 2560)

        mm._store_into_models_dir(model, "copy")

        assert (_models_dir() / "mmproj-F16.gguf").read_bytes() == proj.read_bytes()
        assert model.exists() and proj.exists()

    def test_move_leaves_another_models_projector_in_the_source_folder(
            self, tmp_path, isolated_home):
        src = tmp_path / "downloads"
        model_a = _text_model(src / "gemma-3-4b-it-Q4_K_M.gguf", 2560)
        model_b = _text_model(src / "qwen2.5-vl-7b-instruct-Q4_K_M.gguf", 3584, arch="qwen2")
        proj_a = _projector(src / "mmproj-F16.gguf", 2560)
        proj_b = _projector(src / "mmproj-model-f16.gguf", 3584)

        mm._store_into_models_dir(model_a, "move")

        assert (_models_dir() / proj_a.name).is_file() and not proj_a.exists()
        assert proj_b.is_file(), "the other model's projector must stay beside it"
        assert not (_models_dir() / proj_b.name).exists()
        assert model_b.is_file()

    def test_move_uses_names_when_widths_are_unknown(self, tmp_path, isolated_home):
        d = tmp_path / "drop"
        model_a = _gguf(d, "alpha-7b.gguf")
        _gguf(d, "bravo-7b.gguf")
        proj_a = _gguf(d, "mmproj-alpha-7b-f16.gguf")
        proj_b = _gguf(d, "mmproj-bravo-7b-f16.gguf")

        mm._store_into_models_dir(model_a, "move")

        assert (_models_dir() / proj_a.name).is_file() and not proj_a.exists()
        assert proj_b.is_file()
        assert not (_models_dir() / proj_b.name).exists()

    def test_projector_of_a_different_width_stays_behind(self, tmp_path, isolated_home):
        src = tmp_path / "downloads"
        model = _text_model(src / "gemma-3-4b-it-Q4_K_M.gguf", 2560)
        proj = _projector(src / "mmproj-F16.gguf", 5120)

        mm._store_into_models_dir(model, "move")

        assert proj.is_file()
        assert not (_models_dir() / proj.name).exists()

    def test_move_copies_a_projector_another_model_in_the_folder_can_use(
            self, tmp_path, isolated_home, wide_console, capsys):
        src = tmp_path / "downloads"
        q4 = _text_model(src / "gemma-3-4b-it-Q4_K_M.gguf", 2560)
        _text_model(src / "gemma-3-4b-it-Q8_0.gguf", 2560)
        proj = _projector(src / "mmproj-F16.gguf", 2560)

        mm._store_into_models_dir(q4, "move")

        out = capsys.readouterr().out
        assert (_models_dir() / "mmproj-F16.gguf").read_bytes() == proj.read_bytes()
        assert proj.is_file(), "the Q8 quant in the same folder still needs it"
        assert "Copied mmproj-F16.gguf instead of moving it: gemma-3-4b-it-Q8_0.gguf" in out

    def test_move_keeps_a_shared_attached_projector_for_the_other_quant(
            self, tmp_path, isolated_home):
        src = tmp_path / "downloads"
        q4 = _text_model(src / "google_gemma-3-4b-it-Q4_K_M.gguf", 2560)
        _text_model(src / "google_gemma-3-4b-it-Q8_0.gguf", 2560)
        proj = _projector(src / "mmproj-google_gemma-3-4b-it-f16.gguf", 2560)
        assert mm.find_sibling_mmproj(q4) == proj

        assert add_local(str(q4), store="move") is True

        dest = _models_dir() / proj.name
        assert proj.is_file(), "moving the Q4 quant must not strip the Q8 quant's projector"
        assert dest.read_bytes() == proj.read_bytes()
        assert load_registry()["google_gemma-3-4b-it-Q4_K_M"]["mmproj"] == str(dest.resolve())

    def test_attached_projector_is_recorded_so_a_crowded_models_folder_resolves_it(
            self, tmp_path, isolated_home):
        mm.ensure_dirs()
        _text_model(_models_dir() / "gemma-3-12b-it-Q4_K_M.gguf", 3840)
        _projector(_models_dir() / "mmproj-gemma-3-12b-it-f16.gguf", 3840)
        src = tmp_path / "downloads"
        model = _text_model(src / "gemma-3-4b-it-Q4_K_M.gguf", 2560)
        proj = _projector(src / "mmproj-gemma-3-4b-it-f16.gguf", 2560)
        assert mm.find_sibling_mmproj(model) == proj

        assert add_local(str(model), store="move") is True

        dest = _models_dir() / proj.name
        assert mm.get_model_mmproj("gemma-3-4b-it-Q4_K_M") == str(dest.resolve())

    def test_duplicate_move_records_the_projector_on_every_name(self, tmp_path, isolated_home):
        mm.ensure_dirs()
        _text_model(_models_dir() / "gemma-3-12b-it-Q4_K_M.gguf", 3840)
        _projector(_models_dir() / "mmproj-gemma-3-12b-it-f16.gguf", 3840)
        src = tmp_path / "downloads"
        model = _text_model(src / "gemma-3-4b-it-Q4_K_M.gguf", 2560)
        proj = _projector(src / "mmproj-gemma-3-4b-it-f16.gguf", 2560)
        assert add_local(str(model)) is True                     # registered in place

        assert add_local(str(model), name="g4", on_duplicate="move") is True

        reg = load_registry()
        dest = _models_dir() / proj.name
        assert not proj.exists()
        assert reg["g4"]["mmproj"] == str(dest.resolve())
        assert reg["gemma-3-4b-it-Q4_K_M"]["path"] == str(
            (_models_dir() / model.name).resolve())
        assert reg["gemma-3-4b-it-Q4_K_M"]["mmproj"] == str(dest.resolve())

    def test_identical_projector_already_in_models_folder_is_reused(
            self, tmp_path, isolated_home):
        src = tmp_path / "downloads"
        q8 = _text_model(src / "gemma-3-4b-it-Q8_0.gguf", 2560)
        proj = _projector(src / "mmproj-F16.gguf", 2560)
        mm.ensure_dirs()
        existing = _models_dir() / "mmproj-F16.gguf"
        existing.write_bytes(proj.read_bytes())

        assert add_local(str(q8), store="move") is True

        assert (_models_dir() / q8.name).is_file()
        assert existing.read_bytes() == proj.read_bytes()
        assert proj.is_file(), "the original is left where it is"
        assert not (_models_dir() / "mmproj-F16-2.gguf").exists()

    def test_different_projector_with_the_same_name_lands_under_a_numbered_name(
            self, tmp_path, isolated_home):
        mm.ensure_dirs()
        other = _projector(_models_dir() / "mmproj-F16.gguf", 3584, tag="another model")
        other_bytes = other.read_bytes()
        src = tmp_path / "downloads"
        model = _text_model(src / "gemma-3-4b-it-Q4_K_M.gguf", 2560)
        proj = _projector(src / "mmproj-F16.gguf", 2560, tag="gemma")
        proj_bytes = proj.read_bytes()

        assert add_local(str(model), store="move") is True

        renamed = _models_dir() / "mmproj-F16-2.gguf"
        assert renamed.read_bytes() == proj_bytes
        assert other.read_bytes() == other_bytes
        assert not proj.exists()

    def test_numbering_continues_past_every_taken_name(self, tmp_path, isolated_home):
        mm.ensure_dirs()
        first = _projector(_models_dir() / "mmproj-F16.gguf", 3584, tag="model one")
        second = _projector(_models_dir() / "mmproj-F16-2.gguf", 4096, tag="model two")
        before = (first.read_bytes(), second.read_bytes())
        src = tmp_path / "downloads"
        model = _text_model(src / "gemma-3-4b-it-Q4_K_M.gguf", 2560)
        proj = _projector(src / "mmproj-F16.gguf", 2560, tag="gemma")
        proj_bytes = proj.read_bytes()

        assert add_local(str(model), store="move") is True

        assert (_models_dir() / "mmproj-F16-3.gguf").read_bytes() == proj_bytes
        assert (first.read_bytes(), second.read_bytes()) == before

    def test_renamed_projector_does_not_hide_another_models_projector(
            self, tmp_path, isolated_home):
        mm.ensure_dirs()
        big = _text_model(_models_dir() / "gemma-3-12b-it-Q4_K_M.gguf", 3840)
        big_proj = _projector(_models_dir() / "mmproj-gemma-3-12b-it-f16.gguf", 3840)
        _projector(_models_dir() / "mmproj-F16.gguf", 3584, tag="another model")
        assert add_local(str(big)) is True
        resolved = mm.get_model_mmproj("gemma-3-12b-it-Q4_K_M")
        assert resolved and Path(resolved).resolve() == big_proj.resolve()
        src = tmp_path / "downloads"
        model = _text_model(src / "gemma-3-4b-it-Q4_K_M.gguf", 2560)
        _projector(src / "mmproj-F16.gguf", 2560, tag="gemma 4b")

        assert add_local(str(model), store="move") is True

        resolved = mm.get_model_mmproj("gemma-3-12b-it-Q4_K_M")
        assert resolved and Path(resolved).resolve() == big_proj.resolve(), (
            "a projector renamed into the models folder must not make another "
            "model's own projector ambiguous")

    def test_projector_named_for_a_model_that_cannot_use_it_still_travels(
            self, tmp_path, isolated_home):
        src = tmp_path / "downloads"
        model = _text_model(src / "google_gemma-3-4b-it-Q4_K_M.gguf", 2560)
        _text_model(src / "gemma-2-2b-it-Q4_K_M.gguf", 2304, arch="gemma2")
        proj = _projector(src / "mmproj-gemma-3-4b-it-f16.gguf", 2560)

        mm._store_into_models_dir(model, "move")

        assert (_models_dir() / proj.name).is_file()
        assert not proj.exists()

    def test_duplicate_move_repoints_a_projector_path_recorded_on_the_original_entry(
            self, tmp_path, isolated_home):
        src = tmp_path / "downloads"
        model = _text_model(src / "gemma-3-4b-it-Q4_K_M.gguf", 2560)
        proj = _projector(src / "mmproj-F16.gguf", 2560)
        assert add_local(str(model)) is True
        assert mm.persist_cli_mmproj("gemma-3-4b-it-Q4_K_M", str(proj)) is not None

        assert add_local(str(model), name="g4", on_duplicate="move") is True

        dest = _models_dir() / "mmproj-F16.gguf"
        assert dest.is_file() and not proj.exists()
        assert load_registry()["gemma-3-4b-it-Q4_K_M"]["mmproj"] == str(dest.resolve())
        assert mm.get_model_mmproj("gemma-3-4b-it-Q4_K_M") == str(dest.resolve())

    def test_move_repoints_a_projector_another_models_entry_records(
            self, tmp_path, isolated_home):
        other = _text_model(tmp_path / "elsewhere" / "gemma-3-4b-it-Q8_0.gguf", 2560)
        src = tmp_path / "downloads"
        model = _text_model(src / "gemma-3-4b-it-Q4_K_M.gguf", 2560)
        proj = _projector(src / "mmproj-F16.gguf", 2560)
        assert add_local(str(other)) is True
        assert mm.persist_cli_mmproj("gemma-3-4b-it-Q8_0", str(proj)) is not None

        assert add_local(str(model), store="move") is True

        dest = _models_dir() / "mmproj-F16.gguf"
        assert dest.is_file() and not proj.exists()
        assert mm.get_model_mmproj("gemma-3-4b-it-Q8_0") == str(dest.resolve())

    def test_recorded_projector_is_repointed_even_when_a_later_transfer_fails(
            self, tmp_path, isolated_home, monkeypatch):
        other = _text_model(tmp_path / "elsewhere" / "gemma-3-4b-it-Q8_0.gguf", 2560)
        src = tmp_path / "downloads"
        model = _text_model(src / "gemma-3-4b-it-Q4_K_M.gguf", 2560)
        _text_model(src / "qwen2.5-vl-7b-instruct-Q4_K_M.gguf", 3584, arch="qwen2")
        mine = _projector(src / "mmproj-F16.gguf", 2560)
        _gguf(src, "mmproj-model-f16.gguf")          # unknown width: copied, shared
        assert add_local(str(other)) is True
        assert mm.persist_cli_mmproj("gemma-3-4b-it-Q8_0", str(mine)) is not None
        real_sha = mm._sha256_file

        def _sha(path, progress=None):
            if Path(path).name == "mmproj-model-f16.gguf":
                return "source" if Path(path).parent == src else "corrupted copy"
            return real_sha(path, progress)

        monkeypatch.setattr(mm, "_sha256_file", _sha)
        error = None
        try:
            mm._store_into_models_dir(model, "move")
        except RuntimeError as e:
            error = e

        dest = _models_dir() / "mmproj-F16.gguf"
        assert dest.is_file() and not mine.exists()
        assert load_registry()["gemma-3-4b-it-Q8_0"]["mmproj"] == str(dest.resolve())
        assert error is not None and "Copy verification failed" in str(error)

    def test_repoint_failure_does_not_replace_the_transfer_error(
            self, tmp_path, isolated_home, monkeypatch, wide_console, capsys):
        other = _text_model(tmp_path / "elsewhere" / "gemma-3-4b-it-Q8_0.gguf", 2560)
        src = tmp_path / "downloads"
        model = _text_model(src / "gemma-3-4b-it-Q4_K_M.gguf", 2560)
        _text_model(src / "qwen2.5-vl-7b-instruct-Q4_K_M.gguf", 3584, arch="qwen2")
        mine = _projector(src / "mmproj-F16.gguf", 2560)
        _gguf(src, "mmproj-model-f16.gguf")          # unknown width: copied, shared
        assert add_local(str(other)) is True
        assert mm.persist_cli_mmproj("gemma-3-4b-it-Q8_0", str(mine)) is not None
        real_sha = mm._sha256_file

        def _sha(path, progress=None):
            if Path(path).name == "mmproj-model-f16.gguf":
                return "source" if Path(path).parent == src else "corrupted copy"
            return real_sha(path, progress)

        def _registry_locked(mutator):
            raise OSError("registry is locked")

        monkeypatch.setattr(mm, "_sha256_file", _sha)
        monkeypatch.setattr(mm, "update_registry", _registry_locked)
        error = None
        try:
            mm._store_into_models_dir(model, "move")
        except Exception as e:
            error = e

        out = capsys.readouterr().out
        assert isinstance(error, RuntimeError), f"the transfer error was replaced: {error!r}"
        assert "Copy verification failed" in str(error)
        assert "Could not update the registry entries that record mmproj-F16.gguf" in out

    def test_storing_a_projector_file_repoints_entries_that_record_it(
            self, tmp_path, isolated_home):
        other = _text_model(tmp_path / "elsewhere" / "gemma-3-4b-it-Q8_0.gguf", 2560)
        proj = _projector(tmp_path / "downloads" / "mmproj-F16.gguf", 2560)
        assert add_local(str(other)) is True
        assert mm.persist_cli_mmproj("gemma-3-4b-it-Q8_0", str(proj)) is not None

        assert add_local(str(proj), store="move") is True

        dest = _models_dir() / "mmproj-F16.gguf"
        assert dest.is_file() and not proj.exists()
        assert load_registry()["gemma-3-4b-it-Q8_0"]["mmproj"] == str(dest.resolve())
        assert mm.get_model_mmproj("gemma-3-4b-it-Q8_0") == str(dest.resolve())

    def test_unattached_projectors_are_named_in_a_note(
            self, tmp_path, isolated_home, wide_console, capsys):
        src = tmp_path / "downloads"
        model = _text_model(src / "gemma-3-4b-it-Q4_K_M.gguf", 2560)
        _projector(src / "mmproj-BF16.gguf", 2560)
        _projector(src / "mmproj-F16.gguf", 2560)
        assert mm.find_sibling_mmproj(model) is None

        mm._store_into_models_dir(model, "move")

        out = capsys.readouterr().out
        assert (_models_dir() / "mmproj-BF16.gguf").is_file()
        assert (_models_dir() / "mmproj-F16.gguf").is_file()
        assert "mmproj-BF16.gguf, mmproj-F16.gguf came along but none is attached" in out

    def test_folder_import_with_a_generic_projector_transfers_each_file_once(
            self, tmp_path, isolated_home):
        d = tmp_path / "drop"
        _text_model(d / "gemma-3-4b-it-Q4_K_M.gguf", 2560)
        _projector(d / "mmproj-F16.gguf", 2560)

        assert add_local(str(d), store="move") is True

        assert (_models_dir() / "gemma-3-4b-it-Q4_K_M.gguf").is_file()
        assert (_models_dir() / "mmproj-F16.gguf").is_file()
        assert not any(d.iterdir())

    def test_folder_duplicate_move_registers_a_carried_projector_where_it_landed(
            self, tmp_path, isolated_home):
        d = tmp_path / "downloads"
        model = _text_model(d / "gemma-3-4b-it-Q4_K_M.gguf", 2560)
        _projector(d / "mmproj-gemma-3-4b-it-f16.gguf", 2560)
        assert add_local(str(model)) is True                     # registered in place

        assert add_local(str(d), on_duplicate="move") is True

        entry = load_registry()["mmproj-gemma-3-4b-it-f16"]
        assert Path(entry["path"]).is_file(), f"dangling registry entry: {entry['path']}"
        assert Path(entry["path"]).resolve().parent == _models_dir().resolve()

    def test_folder_duplicate_move_keeps_a_generic_projector_registered_in_place(
            self, tmp_path, isolated_home):
        """Two generic projectors beside one model: neither is attached, so the
        duplicate move carries neither."""
        d = tmp_path / "downloads"
        model = _text_model(d / "gemma-3-4b-it-Q4_K_M.gguf", 2560)
        proj = _projector(d / "mmproj-F16.gguf", 2560)
        _projector(d / "mmproj-BF16.gguf", 2560)
        assert mm.find_sibling_mmproj(model) is None
        assert add_local(str(model)) is True

        assert add_local(str(d), on_duplicate="move") is True

        entry = load_registry()["mmproj-F16"]
        assert Path(entry["path"]).is_file(), f"dangling registry entry: {entry['path']}"
        assert proj.is_file()

    def test_folder_duplicate_move_carries_a_lone_generic_projector(
            self, tmp_path, isolated_home):
        d = tmp_path / "downloads"
        model = _text_model(d / "gemma-3-4b-it-Q4_K_M.gguf", 2560)
        proj = _projector(d / "mmproj-F16.gguf", 2560)
        assert mm.find_sibling_mmproj(model) == proj
        assert add_local(str(model)) is True

        assert add_local(str(d), on_duplicate="move") is True

        entry = load_registry()["mmproj-F16"]
        assert Path(entry["path"]).is_file(), f"dangling registry entry: {entry['path']}"
        assert Path(entry["path"]).resolve().parent == _models_dir().resolve()
        assert not proj.exists()

    def test_folder_duplicate_move_repoints_a_projector_registered_before_its_model(
            self, tmp_path, isolated_home):
        d = tmp_path / "downloads"
        model = _text_model(d / "qwen2.5-vl-7b-instruct-Q4_K_M.gguf", 3584, arch="qwen2")
        _projector(d / "mmproj-qwen2.5-vl-7b-instruct-f16.gguf", 3584)
        assert add_local(str(model)) is True

        assert add_local(str(d), on_duplicate="move") is True

        entry = load_registry()["mmproj-qwen2.5-vl-7b-instruct-f16"]
        assert Path(entry["path"]).is_file(), f"dangling registry entry: {entry['path']}"
        assert Path(entry["path"]).resolve().parent == _models_dir().resolve()

    def test_folder_duplicate_move_repoints_a_models_projector_moved_as_its_own_duplicate(
            self, tmp_path, isolated_home):
        mm.ensure_dirs()
        _projector(_models_dir() / "mmproj-qwen2-vl-2b-instruct-f16.gguf", 1536)
        d = tmp_path / "downloads"
        model = _text_model(d / "qwen2.5-vl-7b-instruct-Q4_K_M.gguf", 3584, arch="qwen2")
        proj = _projector(d / "mmproj-qwen2.5-vl-7b-instruct-f16.gguf", 3584)
        assert add_local(str(model)) is True
        assert add_local(str(proj)) is True
        assert mm.persist_cli_mmproj("qwen2.5-vl-7b-instruct-Q4_K_M", str(proj)) is not None

        assert add_local(str(d), on_duplicate="move") is True

        dest = _models_dir() / proj.name
        assert dest.is_file() and not proj.exists()
        entry = load_registry()["qwen2.5-vl-7b-instruct-Q4_K_M"]
        assert entry["mmproj"] == str(dest.resolve()), f"dangling mmproj: {entry['mmproj']}"
        assert mm.get_model_mmproj("qwen2.5-vl-7b-instruct-Q4_K_M") == str(dest.resolve())


# --------------------------------------------------------------------------- #
#  HF-style model directory (whole tree)
# --------------------------------------------------------------------------- #

class TestStoreHfDir:
    def test_copy_whole_tree(self, tmp_path, isolated_home):
        d = _hf_dir(tmp_path, "myhf")
        assert add_local(str(d), store="copy") is True
        assert d.exists()
        dest = _models_dir() / "myhf"
        assert dest.is_dir()
        assert (dest / "config.json").is_file()
        assert (dest / "model.safetensors").is_file()
        reg = load_registry()
        assert reg["myhf"]["path"] == str(dest.resolve())
        assert reg["myhf"]["source"] == "hf"

    def test_move_whole_tree(self, tmp_path, isolated_home):
        d = _hf_dir(tmp_path, "myhf")
        assert add_local(str(d), store="move") is True
        assert not d.exists()
        dest = _models_dir() / "myhf"
        assert dest.is_dir()
        assert (dest / "tokenizer.json").is_file()


# --------------------------------------------------------------------------- #
#  Directory of several independent loose GGUFs (mirrors _add_local_gguf_dir)
# --------------------------------------------------------------------------- #

class TestStoreLooseGgufDir:
    def test_copy_each_model_individually(self, tmp_path, isolated_home):
        d = tmp_path / "drop"
        _gguf(d, "alpha.gguf")
        _gguf(d, "beta.gguf")
        assert add_local(str(d), store="copy") is True
        reg = load_registry()
        assert {"alpha", "beta"} <= set(reg)
        assert (_models_dir() / "alpha.gguf").is_file()
        assert (_models_dir() / "beta.gguf").is_file()
        assert (d / "alpha.gguf").exists()                   # copy: originals kept
        assert (d / "beta.gguf").exists()

    def test_move_each_model_individually(self, tmp_path, isolated_home):
        d = tmp_path / "drop"
        _gguf(d, "alpha.gguf")
        _gguf(d, "beta.gguf")
        assert add_local(str(d), store="move") is True
        assert not (d / "alpha.gguf").exists()
        assert not (d / "beta.gguf").exists()
        assert (_models_dir() / "alpha.gguf").is_file()
        assert (_models_dir() / "beta.gguf").is_file()

    def test_model_with_mmproj_in_batch_dir_no_double_processing(
            self, tmp_path, isolated_home):
        """A model + its mmproj both appear as their own entries in
        _gguf_first_parts (mmproj isn't filtered out) - naively calling the
        single-file helper on EVERY entry would try to move/copy the mmproj
        TWICE (once as the model's sibling, once as its own top-level entry),
        which either crashes (move: source already gone) or false-positives a
        name collision (copy: dest already exists from the sibling copy).
        _store_loose_gguf_dir's claimed-sibling precompute must avoid this.

        Only two files, and the projector's name carries the model's own name,
        so find_sibling_mmproj pairs them by name alone and the claimed-sibling
        precompute is what this exercises.
        """
        d = tmp_path / "drop"
        _gguf(d, "modelA.gguf")
        _gguf(d, "mmproj-modelA-f16.gguf")
        assert add_local(str(d), store="copy") is True
        reg = load_registry()
        # Both get their own registry entry, same as without --store.
        assert {"modelA", "mmproj-modelA-f16"} <= set(reg)
        assert (_models_dir() / "modelA.gguf").is_file()
        assert (_models_dir() / "mmproj-modelA-f16.gguf").is_file()
        # copy: originals still present
        assert (d / "modelA.gguf").exists()
        assert (d / "mmproj-modelA-f16.gguf").exists()

    def test_model_with_mmproj_in_batch_dir_move_no_double_processing(
            self, tmp_path, isolated_home):
        d = tmp_path / "drop"
        _gguf(d, "modelA.gguf")
        _gguf(d, "mmproj-modelA-f16.gguf")
        assert add_local(str(d), store="move") is True
        reg = load_registry()
        assert {"modelA", "mmproj-modelA-f16"} <= set(reg)
        assert (_models_dir() / "modelA.gguf").is_file()
        assert (_models_dir() / "mmproj-modelA-f16.gguf").is_file()
        assert not (d / "modelA.gguf").exists()
        assert not (d / "mmproj-modelA-f16.gguf").exists()


# --------------------------------------------------------------------------- #
#  Collision refusal - a DIFFERENT file already occupies that name
# --------------------------------------------------------------------------- #

class TestStoreCollisionRefusal:
    def test_copy_refuses_when_name_occupied_by_different_file(
            self, tmp_path, isolated_home):
        mm.ensure_dirs()
        occupied = _models_dir() / "mymodel.gguf"
        occupied.parent.mkdir(parents=True, exist_ok=True)
        occupied.write_bytes(b"GGUF completely different bytes already here")

        src_dir = tmp_path / "external"
        f = _gguf(src_dir, "mymodel.gguf", b"GGUF new incoming content")

        assert add_local(str(f), store="copy") is False
        # Neither side was touched by the refused operation.
        assert occupied.read_bytes() == b"GGUF completely different bytes already here"
        assert f.exists()
        assert f.read_bytes() == b"GGUF new incoming content"
        assert load_registry() == {}

    def test_move_refuses_when_name_occupied_by_different_file(
            self, tmp_path, isolated_home):
        mm.ensure_dirs()
        occupied = _models_dir() / "mymodel.gguf"
        occupied.write_bytes(b"pre-existing")

        src_dir = tmp_path / "external"
        f = _gguf(src_dir, "mymodel.gguf", b"incoming")

        assert add_local(str(f), store="move") is False
        assert occupied.read_bytes() == b"pre-existing"
        assert f.exists()                                    # never moved away
        assert load_registry() == {}


# --------------------------------------------------------------------------- #
#  Disk-space preflight
# --------------------------------------------------------------------------- #

class TestStoreDiskSpacePreflight:
    def test_copy_refuses_when_disk_space_check_fails(
            self, tmp_path, isolated_home, monkeypatch):
        monkeypatch.setattr(mm, "_check_disk_space", lambda dest, need: False)
        src_dir = tmp_path / "external"
        f = _gguf(src_dir, "mymodel.gguf")

        assert add_local(str(f), store="copy") is False
        assert f.exists()                                    # source untouched
        assert not (_models_dir() / "mymodel.gguf").exists()  # nothing landed
        assert load_registry() == {}


# --------------------------------------------------------------------------- #
#  Post-copy checksum verification
# --------------------------------------------------------------------------- #

class TestStoreCopyVerification:
    def test_copy_verification_failure_is_surfaced_and_cleaned_up(
            self, tmp_path, isolated_home, monkeypatch):
        """Simulate a copy that silently corrupted: force the pre/post digest
        to disagree even though the bytes are identical. Must fail loudly
        (return False, nothing registered) and not leave the bad copy behind."""
        calls = {"n": 0}

        def _fake_sha256(path, progress=None):
            calls["n"] += 1
            return "predigest-aaaa" if calls["n"] == 1 else "postdigest-bbbb"

        monkeypatch.setattr(mm, "_sha256_file", _fake_sha256)
        src_dir = tmp_path / "external"
        f = _gguf(src_dir, "mymodel.gguf")

        assert add_local(str(f), store="copy") is False
        assert f.exists()                                    # source left alone
        assert not (_models_dir() / "mymodel.gguf").exists(), \
            "a verified-corrupt copy must not be left behind"
        assert load_registry() == {}


# --------------------------------------------------------------------------- #
#  pull_model() threads --store through its local-path branch
# --------------------------------------------------------------------------- #

class TestPullModelThreadsStore:
    def test_pull_local_path_store_copy(self, tmp_path, isolated_home):
        src_dir = tmp_path / "external"
        f = _gguf(src_dir, "mymodel.gguf")
        assert pull_model(str(f), store="copy") is True
        assert f.exists()
        assert (_models_dir() / "mymodel.gguf").is_file()
        reg = load_registry()
        assert reg["mymodel"]["path"] == str((_models_dir() / "mymodel.gguf").resolve())

    def test_pull_local_path_store_move(self, tmp_path, isolated_home):
        src_dir = tmp_path / "external"
        f = _gguf(src_dir, "mymodel.gguf")
        assert pull_model(str(f), store="move") is True
        assert not f.exists()
        assert (_models_dir() / "mymodel.gguf").is_file()

    def test_pull_remote_spec_ignores_store(self, monkeypatch):
        """store is meaningless for an HF/URL spec - must never reach add_local
        and must never break a normal remote pull."""
        called = {"add_local": False}
        monkeypatch.setattr(mm, "add_local",
                             lambda *a, **k: called.__setitem__("add_local", True) or True)
        monkeypatch.setattr(mm, "_pull_hf_snapshot", lambda *a, **k: True)
        assert pull_model("owner/repo", store="copy") is True
        assert called["add_local"] is False


# --------------------------------------------------------------------------- #
#  CLI: `localm add PATH --store` / `localm pull PATH --store`
# --------------------------------------------------------------------------- #

class TestCliStoreOption:
    def test_add_store_copy(self, tmp_path, cli_isolated_home):
        src_dir = tmp_path / "external"
        f = _gguf(src_dir, "mymodel.gguf")
        result = cli_isolated_home.invoke(main, ["add", str(f), "--store", "copy"])
        assert result.exit_code == 0, result.output
        assert f.exists()
        assert (_models_dir() / "mymodel.gguf").is_file()
        assert load_registry()["mymodel"]["path"] == str(
            (_models_dir() / "mymodel.gguf").resolve())

    def test_add_store_move(self, tmp_path, cli_isolated_home):
        src_dir = tmp_path / "external"
        f = _gguf(src_dir, "mymodel.gguf")
        result = cli_isolated_home.invoke(main, ["add", str(f), "--store", "move"])
        assert result.exit_code == 0, result.output
        assert not f.exists()
        assert (_models_dir() / "mymodel.gguf").is_file()

    def test_add_invalid_store_value_rejected(self, tmp_path, cli_isolated_home):
        src_dir = tmp_path / "external"
        f = _gguf(src_dir, "mymodel.gguf")
        result = cli_isolated_home.invoke(main, ["add", str(f), "--store", "delete"])
        assert result.exit_code != 0
        assert f.exists()
        assert not (_models_dir() / "mymodel.gguf").exists()

    def test_pull_local_path_store_copy(self, tmp_path, cli_isolated_home):
        src_dir = tmp_path / "external"
        f = _gguf(src_dir, "mymodel.gguf")
        result = cli_isolated_home.invoke(main, ["pull", str(f), "--store", "copy"])
        assert result.exit_code == 0, result.output
        assert f.exists()
        assert (_models_dir() / "mymodel.gguf").is_file()

    def test_add_without_store_keeps_default_in_place_behavior(
            self, tmp_path, cli_isolated_home):
        src_dir = tmp_path / "external"
        f = _gguf(src_dir, "mymodel.gguf")
        result = cli_isolated_home.invoke(main, ["add", str(f)])
        assert result.exit_code == 0, result.output
        assert f.exists()
        assert not (_models_dir() / "mymodel.gguf").exists()
        assert load_registry()["mymodel"]["path"] == str(f.resolve())


# --------------------------------------------------------------------------- #
#  A SAME-VOLUME move needs no free space (it is an os.rename)                #
# --------------------------------------------------------------------------- #

def _tiny_free(monkeypatch, free_bytes):
    """Force the free-space reading _check_disk_space takes, without filling a
    real disk. The model files stay real; only the free-space number is staged."""
    import shutil as _sh
    from localm.model_manager import pull as _pull
    real = _sh.disk_usage

    def _fake(p):
        u = real(p)
        return type(u)(u.total, u.total - free_bytes, free_bytes)

    monkeypatch.setattr(_pull.shutil, "disk_usage", _fake)


def test_same_volume_is_detected_for_real_paths(isolated_home, tmp_path):
    """The volume check itself, against REAL paths (no mocks): a dir and its own
    subdir are on one volume."""
    from localm.model_manager.registry import _same_volume
    sub = tmp_path / "sub"
    sub.mkdir()
    assert _same_volume(tmp_path, sub) is True
    # An unreadable/missing path must fail SAFE (unknown -> assume cross-volume,
    # keep the strict space check) rather than claim "same volume, skip it".
    assert _same_volume(tmp_path, tmp_path / "does-not-exist") is False


def test_same_volume_move_does_not_demand_copy_sized_free_space(isolated_home, tmp_path):
    """`localm add <path> --on-duplicate move` on the SAME volume is an
    os.rename needing ~0 extra bytes, so the preflight must not demand the full
    model size: moving a 40 GB model onto a drive with 30 GB free must not fail
    with 'Not enough disk space'."""
    import pytest as _pytest
    src = _gguf(tmp_path / "ext", "big.gguf", b"G" * 4096)
    with _pytest.MonkeyPatch.context() as mp:
        _tiny_free(mp, free_bytes=1)          # 1 byte free, model is 4096 bytes
        dest = mm._store_into_models_dir(src, "move")
    assert dest == _models_dir() / "big.gguf"
    assert dest.exists() and dest.read_bytes() == b"G" * 4096
    assert not src.exists(), "a move must not leave the original behind"


def test_copy_still_refuses_when_the_volume_is_actually_full(isolated_home, tmp_path):
    """Negative case: a COPY really does need the bytes, so the preflight still
    refuses."""
    import pytest as _pytest
    src = _gguf(tmp_path / "ext", "big.gguf", b"G" * 4096)
    with _pytest.MonkeyPatch.context() as mp:
        _tiny_free(mp, free_bytes=1)
        with _pytest.raises(RuntimeError, match="Not enough disk space"):
            mm._store_into_models_dir(src, "copy")
    assert src.exists(), "a refused copy must leave the source alone"
    assert not (_models_dir() / "big.gguf").exists()


def test_cross_volume_move_still_requires_space(isolated_home, tmp_path):
    """Negative case: a CROSS-volume move is a real copy+delete, so it still needs
    the bytes. The skip must be conditioned on the volume, not on the verb."""
    import pytest as _pytest
    from localm.model_manager import registry as _reg
    src = _gguf(tmp_path / "ext", "big.gguf", b"G" * 4096)
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(_reg, "_same_volume", lambda a, b: False)   # pretend another drive
        _tiny_free(mp, free_bytes=1)
        with _pytest.raises(RuntimeError, match="Not enough disk space"):
            mm._store_into_models_dir(src, "move")
    assert src.exists(), "a refused move must leave the source alone"


def test_same_volume_move_of_a_directory_skips_the_space_check(isolated_home, tmp_path):
    """The HF-directory branch takes the same same-volume fast path."""
    import pytest as _pytest
    src = _hf_dir(tmp_path, "myhf")
    with _pytest.MonkeyPatch.context() as mp:
        _tiny_free(mp, free_bytes=1)
        dest = mm._store_into_models_dir(src, "move")
    assert dest == _models_dir() / "myhf"
    assert (dest / "config.json").is_file()
    assert not src.exists()
