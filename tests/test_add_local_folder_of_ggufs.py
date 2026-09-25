# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm add <dir>` / `localm pull <dir>` over a folder of loose GGUFs must
register each model, never reject the directory as "Not a model".

These tests pin the directory branch: walk *.gguf (non-recursive), register
split GGUFs by their first part only.
"""

import pytest

from localm.config import load_registry
from localm.model_manager import add_local, pull_model


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    # load_registry/save_registry read the module-level REGISTRY_FILE frozen at
    # import, so the autouse LOCALM_HOME env alone does not isolate them. Redirect
    # the config paths to a throwaway dir.
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return home


def _gguf(d, name, content=None):
    """Write a .gguf. Default content is unique to its name so two files
    never collide on the sha256 content-dedup path; pass *content* explicitly
    to control that (e.g. two same-named files at different paths that must
    compare byte-identical, or differ, on purpose)."""
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(content if content is not None else b"GGUF\x00\x00\x00\x00" + name.encode())
    return p


def _folder(tmp_path, *names):
    d = tmp_path / "drop"
    d.mkdir()
    for n in names:
        _gguf(d, n)
    return d


def _empty_dir(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    return d


def _junk_dir(tmp_path):
    d = tmp_path / "junk"
    d.mkdir()
    (d / "readme.txt").write_text("not a model")
    (d / "weights.bin").write_bytes(b"\x00")
    return d


# ---------------------------------------------------------------------------
#  The new directory branch
# ---------------------------------------------------------------------------

class TestFolderOfLooseGGUFs:
    def test_single_loose_gguf_registers(self, tmp_path, isolated_home):
        d = _folder(tmp_path, "mymodel.gguf")
        assert add_local(str(d)) is True
        assert "mymodel" in load_registry()

    def test_multiple_loose_ggufs_each_register(self, tmp_path, isolated_home):
        d = _folder(tmp_path, "alpha.gguf", "beta.gguf", "gamma.gguf")
        assert add_local(str(d)) is True
        reg = load_registry()
        assert {"alpha", "beta", "gamma"} <= set(reg)

    def test_split_gguf_registers_first_part_only(self, tmp_path, isolated_home):
        d = _folder(
            tmp_path,
            "big-00001-of-00002.gguf",
            "big-00002-of-00002.gguf",
        )
        assert add_local(str(d)) is True
        reg = load_registry()
        # Exactly one entry, named for the stripped stem, pointing at the FIRST part.
        assert set(reg) == {"big"}
        assert reg["big"]["path"].endswith("big-00001-of-00002.gguf")

    def test_split_plus_loose_yields_two_entries(self, tmp_path, isolated_home):
        d = _folder(
            tmp_path,
            "big-00001-of-00002.gguf",
            "big-00002-of-00002.gguf",
            "solo.gguf",
        )
        assert add_local(str(d)) is True
        reg = load_registry()
        assert set(reg) == {"big", "solo"}

    def test_name_collision_loose_and_split_both_kept(self, tmp_path, isolated_home):
        # model.gguf and model-00001-of-00002.gguf both derive base name "model";
        # the second must get a "-2" suffix, never overwrite the first.
        d = _folder(
            tmp_path,
            "model.gguf",
            "model-00001-of-00002.gguf",
            "model-00002-of-00002.gguf",
        )
        assert add_local(str(d)) is True
        reg = load_registry()
        assert set(reg) == {"model", "model-2"}
        paths = {reg[n]["path"] for n in reg}
        assert any(p.endswith("model.gguf") for p in paths)
        assert any(p.endswith("model-00001-of-00002.gguf") for p in paths)
        # the non-first split part is never its own entry
        assert not any(p.endswith("model-00002-of-00002.gguf") for p in paths)

    @pytest.mark.parametrize(
        "make_dir",
        [_empty_dir, _junk_dir],
        ids=["empty_dir", "dir_of_non_gguf"],
    )
    def test_dir_with_no_usable_gguf_returns_false(self, tmp_path, isolated_home, make_dir):
        d = make_dir(tmp_path)
        assert add_local(str(d)) is False
        assert load_registry() == {}

    def test_hf_dir_with_stray_gguf_registers_the_dir(self, tmp_path, isolated_home):
        # An HF model dir that happens to contain a stray .gguf must register as
        # ONE hf entry pointing at the dir - not be looped over file-by-file.
        d = tmp_path / "hfmodel"
        d.mkdir()
        (d / "config.json").write_text('{"model_type": "llama"}')
        (d / "tokenizer.json").write_text("{}")
        _gguf(d, "extra.gguf")
        assert add_local(str(d), "hfmodel") is True
        reg = load_registry()
        assert set(reg) == {"hfmodel"}
        assert reg["hfmodel"]["source"] == "hf"
        assert reg["hfmodel"]["path"].rstrip("/\\").endswith("hfmodel")

    def test_import_is_recursive_by_default(self, tmp_path, isolated_home):
        # Import recurses by default: a .gguf one subfolder deep is picked up.
        d = tmp_path / "drop"
        d.mkdir()
        _gguf(d, "top.gguf")
        sub = d / "nested"
        sub.mkdir()
        _gguf(sub, "deep.gguf")
        assert add_local(str(d)) is True
        reg = load_registry()
        assert set(reg) == {"top", "deep"}

    def test_import_depth_cap(self, tmp_path, isolated_home):
        # Default import_max_depth=3 counts the filename as level 1, so a file
        # two subfolders down (depth 3) is found and one three down (depth 4) is not.
        d = tmp_path / "drop"
        d.mkdir()
        lvl3 = d / "a" / "b"          # file here = depth 3
        lvl3.mkdir(parents=True)
        _gguf(lvl3, "found.gguf")
        lvl4 = d / "a" / "b" / "c"    # file here = depth 4
        lvl4.mkdir(parents=True)
        _gguf(lvl4, "toodeep.gguf")
        assert add_local(str(d)) is True
        reg = load_registry()
        assert "found" in reg
        assert "toodeep" not in reg

    def test_import_depth_respects_config(self, tmp_path, isolated_home):
        import localm.model_manager as _mm
        from localm.config import load_config, save_config
        cfg = load_config()
        cfg["import_max_depth"] = 1          # top-level only
        save_config(cfg)
        d = tmp_path / "drop"
        d.mkdir()
        _gguf(d, "top.gguf")
        sub = d / "nested"
        sub.mkdir()
        _gguf(sub, "deep.gguf")
        assert _mm.add_local(str(d)) is True
        reg = load_registry()
        assert set(reg) == {"top"}         # depth 1 only -> nested not picked up


# ---------------------------------------------------------------------------
#  pull_model delegates an absolute directory path here too
# ---------------------------------------------------------------------------

class TestPullFolder:
    def test_pull_absolute_dir_registers_all(self, tmp_path, isolated_home):
        d = _folder(tmp_path, "one.gguf", "two.gguf")
        assert pull_model(str(d)) is True
        reg = load_registry()
        assert {"one", "two"} <= set(reg)


# ---------------------------------------------------------------------------
#  --store move/copy over a folder: a projector name collision must not
#  abort the whole import mid-walk.
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_home_for_store(tmp_path, monkeypatch):
    # _store_into_models_dir / _store_loose_gguf_dir read MODELS_DIR/HOME_DIR
    # through the localm.model_manager PACKAGE attribute (a plain-value copy
    # taken at import time), so that has to be patched too, on top of the
    # config.py module-level values isolated_home already redirects.
    import localm.config as cfg
    import localm.model_manager as mm
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
    mm.ensure_dirs()
    return models_dir


class TestFolderStoreProjectorCollision:
    def test_byte_identical_projector_is_reused_not_aborted(
            self, tmp_path, isolated_home_for_store):
        models_dir = isolated_home_for_store
        projector_bytes = b"GGUF identical projector bytes"
        _gguf(models_dir, "mmproj-F16.gguf", projector_bytes)

        d = tmp_path / "downloads"
        _gguf(d, "gemma-3-4b-it-Q4_K_M.gguf", b"GGUF gemma model bytes")
        _gguf(d, "mmproj-F16.gguf", projector_bytes)   # byte-identical to the one already there

        assert add_local(str(d), store="move") is True
        reg = load_registry()
        assert {"gemma-3-4b-it-Q4_K_M", "mmproj-F16"} <= set(reg)
        # Reused in place: the pre-existing file is untouched and the
        # incoming duplicate is never deleted, even under --store move.
        assert (models_dir / "mmproj-F16.gguf").read_bytes() == projector_bytes
        assert not (d / "gemma-3-4b-it-Q4_K_M.gguf").exists()
        assert (d / "mmproj-F16.gguf").exists()

    def test_different_content_projector_lands_under_a_free_name(
            self, tmp_path, isolated_home_for_store):
        models_dir = isolated_home_for_store
        _gguf(models_dir, "mmproj-F16.gguf", b"GGUF pre-existing DIFFERENT projector bytes")

        d = tmp_path / "downloads"
        _gguf(d, "gemma-3-4b-it-Q4_K_M.gguf", b"GGUF gemma model bytes")
        _gguf(d, "mmproj-F16.gguf", b"GGUF incoming NEW projector bytes")

        assert add_local(str(d), store="move") is True
        reg = load_registry()
        assert "gemma-3-4b-it-Q4_K_M" in reg
        # A free <stem>-<n>.gguf name, not a silently dropped/overwritten file.
        renamed = [n for n in reg if n != "gemma-3-4b-it-Q4_K_M"]
        assert len(renamed) == 1
        assert reg[renamed[0]]["path"].endswith("mmproj-F16-2.gguf")
        assert (models_dir / "mmproj-F16.gguf").read_bytes() == \
            b"GGUF pre-existing DIFFERENT projector bytes"
        assert (models_dir / "mmproj-F16-2.gguf").read_bytes() == \
            b"GGUF incoming NEW projector bytes"
        # --store move: both incoming files are gone from the source folder.
        assert not (d / "gemma-3-4b-it-Q4_K_M.gguf").exists()
        assert not (d / "mmproj-F16.gguf").exists()

    def test_genuine_model_collision_still_refuses_and_leaves_folder_untouched(
            self, tmp_path, isolated_home_for_store):
        models_dir = isolated_home_for_store
        _gguf(models_dir, "shared-model.gguf", b"GGUF existing DIFFERENT model content")

        d = tmp_path / "downloads"
        # alpha sorts first and would transfer cleanly on its own - it must
        # NOT be moved before the later, colliding file is discovered.
        _gguf(d, "alpha.gguf", b"GGUF alpha content")
        _gguf(d, "shared-model.gguf", b"GGUF incoming NEW model content")

        assert add_local(str(d), store="move") is False
        assert load_registry() == {}
        # Preflighted: nothing in the folder transferred, not even the file
        # that would otherwise have succeeded.
        assert (d / "alpha.gguf").exists()
        assert (d / "shared-model.gguf").exists()
        assert not (models_dir / "alpha.gguf").exists()
        assert (models_dir / "shared-model.gguf").read_bytes() == \
            b"GGUF existing DIFFERENT model content"
