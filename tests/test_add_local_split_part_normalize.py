# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm add/pull <some-part>.gguf` pointed directly at a non-first split part
normalises to the first part.

llama.cpp loads a split GGUF set from its ``*-00001-of-N`` part. The folder
branch (_gguf_first_parts) and sync_models_dir normalise, and get_model_info
normalises at read time; these tests pin the single-file path.
"""

import pytest

from localm.config import load_registry
from localm.model_manager import add_local


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return home


def _gguf(d, name):
    p = d / name
    p.write_bytes(b"GGUF\x00\x00\x00\x00" + name.encode())
    return p


def test_direct_non_first_part_normalizes_to_first(tmp_path, isolated_home):
    """Registers key 'big' pointing at part 1, not 'big-00002-of-00002'
    pointing at part 2."""
    d = tmp_path / "models"
    d.mkdir()
    _gguf(d, "big-00001-of-00002.gguf")
    part2 = _gguf(d, "big-00002-of-00002.gguf")
    assert add_local(str(part2)) is True
    reg = load_registry()
    assert "big" in reg
    assert "big-00002-of-00002" not in reg
    assert reg["big"]["path"].endswith("big-00001-of-00002.gguf")


def test_direct_first_part_stays_correct(tmp_path, isolated_home):
    d = tmp_path / "models"
    d.mkdir()
    part1 = _gguf(d, "big-00001-of-00002.gguf")
    _gguf(d, "big-00002-of-00002.gguf")
    assert add_local(str(part1)) is True
    reg = load_registry()
    assert "big" in reg
    assert reg["big"]["path"].endswith("big-00001-of-00002.gguf")


def test_non_split_single_file_unaffected(tmp_path, isolated_home):
    d = tmp_path / "models"
    d.mkdir()
    solo = _gguf(d, "solo.gguf")
    assert add_local(str(solo)) is True
    assert "solo" in load_registry()


def test_only_non_first_part_present_does_not_register_missing_first(tmp_path, isolated_home):
    """EDGE: if only part 2 exists on disk, do not rewrite the path to a
    non-existent first part."""
    d = tmp_path / "models"
    d.mkdir()
    part2 = _gguf(d, "lonely-00002-of-00002.gguf")
    assert add_local(str(part2)) is True
    reg = load_registry()
    # Whatever key it registers under, the stored path must exist on disk.
    paths = [entry["path"] for entry in reg.values()]
    assert any(p.endswith("lonely-00002-of-00002.gguf") for p in paths)
    for p in paths:
        from pathlib import Path
        assert Path(p).is_file()
