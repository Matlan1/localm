"""
Tests for "load from folder" validation: add_local must refuse the localm data
directory and non-model paths instead of polluting the registry, and the HF-dir
detector must require real model artifacts (not just a bare config.json, which
the data dir also has).
"""

from pathlib import Path

import pytest

from localm import model_manager as mm
from localm.inference.engine import _is_hf_dir


@pytest.fixture()
def fake_registry(tmp_path, monkeypatch):
    """In-memory registry + temp HOME_DIR/MODELS_DIR wired into model_manager."""
    store: dict = {}
    home_dir = tmp_path / "home"
    models_dir = home_dir / "models"
    models_dir.mkdir(parents=True)
    monkeypatch.setattr(mm, "HOME_DIR", home_dir)
    monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "load_registry", lambda: dict(store))

    def _save(reg):
        store.clear()
        store.update(reg)

    monkeypatch.setattr(mm, "save_registry", _save)

    def _update(mutator):
        reg = dict(store)
        mutator(reg)
        store.clear()
        store.update(reg)
        return dict(store)

    monkeypatch.setattr(mm, "update_registry", _update)
    return store, home_dir, models_dir


def _hf_dir(parent: Path, name: str, *, weights: bool = True) -> Path:
    d = parent / name
    d.mkdir(parents=True)
    (d / "config.json").write_text('{"model_type": "llama"}')
    if weights:
        (d / "tokenizer.json").write_text("{}")
    return d


# ---------------------------------------------------------------------------
#  _is_hf_dir: a bare config.json is not a model
# ---------------------------------------------------------------------------

class TestIsHfDir:
    def test_bare_config_json_is_not_a_model(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "config.json").write_text('{"port": 8642}')   # localm settings file
        assert _is_hf_dir(str(d)) is False

    def test_config_plus_tokenizer_is_a_model(self, tmp_path):
        d = _hf_dir(tmp_path, "model")
        assert _is_hf_dir(str(d)) is True

    def test_config_plus_safetensors_is_a_model(self, tmp_path):
        d = tmp_path / "model"
        d.mkdir()
        (d / "config.json").write_text('{"model_type": "llama"}')
        (d / "model.safetensors").write_bytes(b"\x00")
        assert _is_hf_dir(str(d)) is True

    def test_plain_directory_is_not_a_model(self, tmp_path):
        d = tmp_path / "plain"
        d.mkdir()
        assert _is_hf_dir(str(d)) is False


# ---------------------------------------------------------------------------
#  add_local: refuse the data dir and non-models
# ---------------------------------------------------------------------------

class TestAddLocalValidation:
    def test_refuses_home_dir(self, fake_registry, capsys):
        store, home_dir, _ = fake_registry
        (home_dir / "config.json").write_text('{"port": 8642}')
        mm.add_local(str(home_dir))
        assert store == {}
        assert "data folder" in capsys.readouterr().out.lower()

    def test_refuses_models_dir(self, fake_registry, capsys):
        store, _, models_dir = fake_registry
        mm.add_local(str(models_dir))
        assert store == {}
        assert "data folder" in capsys.readouterr().out.lower()

    def test_refuses_bare_config_dir(self, fake_registry, tmp_path, capsys):
        """A directory that only carries a config.json is not a model."""
        store, _, _ = fake_registry
        d = tmp_path / "notamodel"
        d.mkdir()
        (d / "config.json").write_text('{"whatever": true}')
        mm.add_local(str(d))
        assert store == {}
        assert "not a model" in capsys.readouterr().out.lower()

    def test_refuses_plain_dir(self, fake_registry, tmp_path, capsys):
        store, _, _ = fake_registry
        d = tmp_path / "emptyish"
        d.mkdir()
        mm.add_local(str(d))
        assert store == {}
        assert "not a model" in capsys.readouterr().out.lower()

    def test_accepts_real_hf_dir(self, fake_registry, tmp_path):
        store, _, _ = fake_registry
        d = _hf_dir(tmp_path, "realmodel")
        mm.add_local(str(d), "realmodel")
        assert "realmodel" in store
        assert store["realmodel"]["source"] == "hf"

    def test_accepts_gguf_file(self, fake_registry, tmp_path):
        """Regression: a plain .gguf file still registers as a local model."""
        store, _, _ = fake_registry
        f = tmp_path / "m.gguf"
        f.write_bytes(b"GGUF-bytes")
        mm.add_local(str(f), "m", on_duplicate="register", no_hash=True)
        assert "m" in store
        assert store["m"]["source"] == "local"
