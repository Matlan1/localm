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

        Only two files: find_sibling_mmproj auto-resolves to the SOLE
        mmproj-named candidate in a folder regardless of naming correlation, so
        a third, unrelated model in the same folder would hit that ambiguity
        heuristic instead of the logic under test.
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
