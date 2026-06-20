# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for split GGUF support (multi-part *-00001-of-00003.gguf files)."""

from unittest.mock import patch

import pytest

from localm.model_manager import (
    first_split_part,
    missing_split_parts,
    split_gguf_parts,
)


class TestSplitGgufParts:
    def test_regular_gguf_returns_none(self):
        assert split_gguf_parts("model-Q4_K_M.gguf") is None

    def test_three_part_split(self):
        parts = split_gguf_parts("big-model-Q8_0-00001-of-00003.gguf")
        assert parts == [
            "big-model-Q8_0-00001-of-00003.gguf",
            "big-model-Q8_0-00002-of-00003.gguf",
            "big-model-Q8_0-00003-of-00003.gguf",
        ]

    def test_middle_part_yields_same_list(self):
        parts = split_gguf_parts("m-00002-of-00003.gguf")
        assert parts[0] == "m-00001-of-00003.gguf"
        assert len(parts) == 3

    def test_single_part_total_returns_none(self):
        # "-00001-of-00001" is not really split
        assert split_gguf_parts("m-00001-of-00001.gguf") is None

    def test_case_insensitive_extension(self):
        parts = split_gguf_parts("m-00001-of-00002.GGUF")
        assert parts is not None
        assert len(parts) == 2

    def test_path_with_directory(self):
        parts = split_gguf_parts(r"D:\models\m-00001-of-00002.gguf")
        assert parts == ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]

    def test_non_gguf_extension_returns_none(self):
        assert split_gguf_parts("m-00001-of-00003.bin") is None


class TestFirstSplitPart:
    def test_regular_file_unchanged(self):
        assert first_split_part("model.gguf") == "model.gguf"

    def test_later_part_normalised_to_first(self):
        assert first_split_part("m-00003-of-00003.gguf") == "m-00001-of-00003.gguf"


class TestMissingSplitParts:
    def test_regular_file_no_missing(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"x")
        assert missing_split_parts(f) == []

    def test_all_parts_present(self, tmp_path):
        for i in (1, 2):
            (tmp_path / f"m-0000{i}-of-00002.gguf").write_bytes(b"x")
        assert missing_split_parts(tmp_path / "m-00001-of-00002.gguf") == []

    def test_missing_part_reported(self, tmp_path):
        (tmp_path / "m-00001-of-00003.gguf").write_bytes(b"x")
        (tmp_path / "m-00003-of-00003.gguf").write_bytes(b"x")
        missing = missing_split_parts(tmp_path / "m-00001-of-00003.gguf")
        assert [p.name for p in missing] == ["m-00002-of-00003.gguf"]


class TestGgufBackendSplitCheck:
    def test_load_raises_on_incomplete_split(self, tmp_path):
        from localm.inference.backends.gguf import GgufBackend
        first = tmp_path / "m-00001-of-00002.gguf"
        first.write_bytes(b"GGUF")
        backend = GgufBackend(str(first))
        with pytest.raises(FileNotFoundError, match="m-00002-of-00002.gguf"):
            backend.load()

    def test_load_proceeds_when_all_parts_present(self, tmp_path):
        from localm.inference.backends.gguf import GgufBackend
        for i in (1, 2):
            (tmp_path / f"m-0000{i}-of-00002.gguf").write_bytes(b"GGUF")
        backend = GgufBackend(str(tmp_path / "m-00001-of-00002.gguf"))
        # The split check must pass; the native load itself is mocked out
        with patch.object(backend, "_load_native") as mock_native:
            backend.load()
        mock_native.assert_called_once()


class TestGetModelInfoSplitNormalisation:
    def test_later_part_path_resolves_to_first(self, tmp_path):
        from localm import model_manager
        for i in (1, 2):
            (tmp_path / f"m-0000{i}-of-00002.gguf").write_bytes(b"x")
        with patch.object(model_manager, "load_registry", return_value={}):
            result = model_manager.get_model_info(
                str(tmp_path / "m-00002-of-00002.gguf"))
        assert result is not None
        assert result[0].name == "m-00001-of-00002.gguf"

    def test_regular_gguf_path_unchanged(self, tmp_path):
        from localm import model_manager
        f = tmp_path / "model.gguf"
        f.write_bytes(b"x")
        with patch.object(model_manager, "load_registry", return_value={}):
            result = model_manager.get_model_info(str(f))
        assert result is not None
        assert result[0] == f


class TestRemoveModelSplitParts:
    def test_remove_deletes_all_parts(self, tmp_path, monkeypatch):
        from localm import model_manager
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        for i in (1, 2, 3):
            (models_dir / f"m-0000{i}-of-00003.gguf").write_bytes(b"x")
        first = models_dir / "m-00001-of-00003.gguf"

        monkeypatch.setattr(model_manager, "MODELS_DIR", models_dir)
        reg = {"bigmodel": {"path": str(first), "source": "hf:owner/repo"}}
        monkeypatch.setattr(model_manager, "load_registry", lambda: dict(reg))
        saved = {}
        monkeypatch.setattr(model_manager, "save_registry", saved.update)

        model_manager.remove_model("bigmodel")

        assert not any(models_dir.glob("*.gguf"))
        assert "bigmodel" not in saved
