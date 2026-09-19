# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for resumable HuggingFace model downloads: range-resumption from
.incomplete files, disk space preflight on resume, and adopt-orphan recovery.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from localm import model_manager as mm
from localm.model_manager.pull import (
    _ensure_hf_resumable_download,
    _resumable_download_to_tmp_and_move,
)


@pytest.fixture()
def hf_env(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    monkeypatch.setattr(mm, "find_by_sha256", lambda *a, **k: [])
    reg_spy = MagicMock()
    monkeypatch.setattr(mm, "_register", reg_spy)
    monkeypatch.setattr(mm, "_register_with_dedup", MagicMock(return_value=True))
    _ensure_hf_resumable_download()
    return models, reg_spy


class TestHfResumableDownload:
    def test_resume_uses_existing_incomplete_bytes(self, tmp_path, monkeypatch):
        inc_path = tmp_path / "cache" / "file.etag.incomplete"
        inc_path.parent.mkdir(parents=True)
        inc_path.write_bytes(b"01234")  # 5 bytes already on disk
        dest = tmp_path / "dest.gguf"

        captured = {}
        def fake_http_get(url, f, resume_size=0, headers=None, expected_size=None, **kw):
            captured["resume_size"] = resume_size
            captured["expected_size"] = expected_size
            f.write(b"56789")

        monkeypatch.setattr("huggingface_hub.file_download.http_get", fake_http_get)
        monkeypatch.setattr("huggingface_hub.file_download._check_disk_space", lambda *a: None)

        _resumable_download_to_tmp_and_move(
            incomplete_path=inc_path,
            destination_path=dest,
            url_to_download="https://example.com/file",
            headers={},
            expected_size=10,
            filename="dest.gguf",
            force_download=False,
            etag="etag",
            xet_file_data=None,
        )

        assert captured["resume_size"] == 5
        assert captured["expected_size"] == 10
        assert dest.exists()
        assert dest.read_bytes() == b"0123456789"
        assert not inc_path.exists()

    def test_adopts_uuid_incomplete_file_left_by_unpatched_run(self, tmp_path, monkeypatch):
        inc_path = tmp_path / "cache" / "file.etag.incomplete"
        inc_path.parent.mkdir(parents=True)
        orphan = tmp_path / "cache" / "file.etag.abcd1234.incomplete"
        orphan.write_bytes(b"HELLO")
        dest = tmp_path / "dest.gguf"

        captured = {}
        def fake_http_get(url, f, resume_size=0, **kw):
            captured["resume_size"] = resume_size
            f.write(b" WORLD")

        monkeypatch.setattr("huggingface_hub.file_download.http_get", fake_http_get)
        monkeypatch.setattr("huggingface_hub.file_download._check_disk_space", lambda *a: None)

        _resumable_download_to_tmp_and_move(
            incomplete_path=inc_path,
            destination_path=dest,
            url_to_download="https://example.com/file",
            headers={},
            expected_size=11,
            filename="dest.gguf",
            force_download=False,
            etag="etag",
            xet_file_data=None,
        )

        assert captured["resume_size"] == 5
        assert dest.read_bytes() == b"HELLO WORLD"
        assert not orphan.exists()

    def test_oversized_incomplete_file_is_reset(self, tmp_path, monkeypatch):
        inc_path = tmp_path / "cache" / "file.etag.incomplete"
        inc_path.parent.mkdir(parents=True)
        inc_path.write_bytes(b"WAY_TOO_LONG_DATA_HERE")
        dest = tmp_path / "dest.gguf"

        captured = {}
        def fake_http_get(url, f, resume_size=0, **kw):
            captured["resume_size"] = resume_size
            f.write(b"CORRECT")

        monkeypatch.setattr("huggingface_hub.file_download.http_get", fake_http_get)
        monkeypatch.setattr("huggingface_hub.file_download._check_disk_space", lambda *a: None)

        _resumable_download_to_tmp_and_move(
            incomplete_path=inc_path,
            destination_path=dest,
            url_to_download="https://example.com/file",
            headers={},
            expected_size=7,
            filename="dest.gguf",
            force_download=False,
            etag="etag",
            xet_file_data=None,
        )

        assert captured["resume_size"] == 0
        assert dest.read_bytes() == b"CORRECT"

    def test_force_download_discards_incomplete_file(self, tmp_path, monkeypatch):
        inc_path = tmp_path / "cache" / "file.etag.incomplete"
        inc_path.parent.mkdir(parents=True)
        inc_path.write_bytes(b"OLD_PARTIAL")
        dest = tmp_path / "dest.gguf"

        captured = {}
        def fake_http_get(url, f, resume_size=0, **kw):
            captured["resume_size"] = resume_size
            f.write(b"FRESH")

        monkeypatch.setattr("huggingface_hub.file_download.http_get", fake_http_get)
        monkeypatch.setattr("huggingface_hub.file_download._check_disk_space", lambda *a: None)

        _resumable_download_to_tmp_and_move(
            incomplete_path=inc_path,
            destination_path=dest,
            url_to_download="https://example.com/file",
            headers={},
            expected_size=5,
            filename="dest.gguf",
            force_download=True,
            etag="etag",
            xet_file_data=None,
        )

        assert captured["resume_size"] == 0
        assert dest.read_bytes() == b"FRESH"

    def test_gguf_pull_disk_space_preflight_charges_only_remaining(self, hf_env, monkeypatch):
        models, _ = hf_env
        cache_dir = models / ".cache"
        cache_dir.mkdir(parents=True)

        # Place 6 bytes of partial data in cache
        inc_file = cache_dir / "prefix.etag.incomplete"
        inc_file.write_bytes(b"123456")

        checked = []
        monkeypatch.setattr(mm, "_check_disk_space", lambda path, size: checked.append(size) or True)
        monkeypatch.setattr("localm.model_manager.pull._incomplete_prefixes", lambda *a: {"prefix"})
        monkeypatch.setattr("localm.model_manager.pull._hf_file_sha256", lambda *a: None)
        monkeypatch.setattr("huggingface_hub.hf_hub_url", lambda *a, **kw: "https://example.com/test.gguf")

        fake_head = MagicMock(status_code=200, headers={"content-length": "10"})
        monkeypatch.setattr("requests.head", lambda *a, **kw: fake_head)

        def fake_hf_download(repo_id, filename, local_dir, **kw):
            out = Path(local_dir) / filename
            out.write_bytes(b"0" * 10)
            return str(out)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_download)

        res = mm._pull_gguf_file("owner/repo:test.gguf", name="test")
        assert res is True
        # Total is 10, partial is 6 -> checked size should be 4
        assert checked == [4]
