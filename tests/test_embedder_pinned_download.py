# SPDX-License-Identifier: AGPL-3.0-or-later
"""The known embedding models download from a pinned commit and are refused when the
file's sha256 is not the pinned one."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from localm.inference import embedder


@pytest.fixture
def allow(monkeypatch):
    monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "allow")


def _fake_download(content: bytes, seen: dict):
    def _download(repo, filename, local_dir=None, endpoint=None, **kw):
        seen.update(kw)
        out = Path(local_dir) / filename
        out.write_bytes(content)
        return str(out)
    return _download


class TestPinTable:
    def test_every_known_model_is_pinned_to_a_commit_and_a_sha256(self):
        for name, key in embedder.KNOWN_EMBEDDING_MODELS.items():
            assert key in embedder._EMBEDDING_PINS, f"{name} has no pin"
            revision, digest = embedder._EMBEDDING_PINS[key]
            assert re.fullmatch(r"[0-9a-f]{40}", revision), (name, revision)
            assert re.fullmatch(r"[0-9a-f]{64}", digest), (name, digest)

    def test_no_pin_names_a_model_that_is_not_known(self):
        assert set(embedder._EMBEDDING_PINS) == set(embedder.KNOWN_EMBEDDING_MODELS.values())


class TestPinnedDownload:
    def _pin(self, monkeypatch, content: bytes):
        digest = hashlib.sha256(content).hexdigest()
        monkeypatch.setattr(embedder, "_EMBEDDING_PINS",
                            {("org/repo", "model.gguf"): ("a" * 40, digest)})

    def test_the_download_is_made_from_the_pinned_commit(self, tmp_path, monkeypatch, allow):
        self._pin(monkeypatch, b"the real bytes")
        seen: dict = {}
        monkeypatch.setattr("huggingface_hub.hf_hub_download",
                            _fake_download(b"the real bytes", seen))
        dest = tmp_path / "model.gguf"
        got = embedder._download_known("m", "org/repo", "model.gguf", dest, True)
        assert got == str(dest) and dest.is_file()
        assert dest.read_bytes() == b"the real bytes"
        assert seen["revision"] == "a" * 40
        assert [p.name for p in tmp_path.iterdir()] == ["model.gguf"]

    def test_a_file_with_another_sha256_is_deleted_and_refused(self, tmp_path, monkeypatch, allow):
        self._pin(monkeypatch, b"the real bytes")
        monkeypatch.setattr("huggingface_hub.hf_hub_download",
                            _fake_download(b"swapped bytes", {}))
        dest = tmp_path / "model.gguf"
        assert embedder._download_known("m", "org/repo", "model.gguf", dest, True) is None
        assert list(tmp_path.iterdir()) == []
        assert "sha256" in (embedder.last_error() or "")

    def test_a_model_without_a_pin_downloads_as_before(self, tmp_path, monkeypatch, allow):
        monkeypatch.setattr(embedder, "_EMBEDDING_PINS", {})
        seen: dict = {}
        monkeypatch.setattr("huggingface_hub.hf_hub_download",
                            _fake_download(b"anything", seen))
        dest = tmp_path / "model.gguf"
        assert embedder._download_known("m", "org/repo", "model.gguf", dest, True) == str(dest)
        assert "revision" not in seen


class TestSetupEmbeddingsRegistration:
    def _register_with_file(self, tmp_path, monkeypatch, content):
        from click.testing import CliRunner

        from localm.cli.maintenance import setup_embeddings
        from localm.config import load_registry

        repo_file = embedder.KNOWN_EMBEDDING_MODELS[embedder.DEFAULT_EMBEDDING_MODEL]
        pin = embedder._EMBEDDING_PINS[repo_file]
        monkeypatch.setattr(embedder, "_EMBEDDING_PINS", {
            repo_file: (pin[0], hashlib.sha256(b"the real bytes").hexdigest())})
        model_file = tmp_path / repo_file[1]
        model_file.write_bytes(content)
        monkeypatch.setattr(embedder, "_embeddings_dir", lambda: tmp_path)
        monkeypatch.setattr(embedder, "resolve_embedding_model_path",
                            lambda allow_download=True: str(model_file))
        result = CliRunner().invoke(setup_embeddings, [])
        assert result.exit_code == 0, result.output
        return [e for n, e in load_registry().items() if n.startswith("embedding-")]

    def test_a_file_that_matches_the_pin_is_registered_with_its_digest(
            self, tmp_path, monkeypatch):
        entries = self._register_with_file(tmp_path, monkeypatch, b"the real bytes")
        assert len(entries) == 1
        assert entries[0]["sha256"] == hashlib.sha256(b"the real bytes").hexdigest()

    def test_a_file_that_does_not_match_is_registered_without_a_digest(
            self, tmp_path, monkeypatch):
        entries = self._register_with_file(tmp_path, monkeypatch, b"some other bytes")
        assert len(entries) == 1 and "sha256" not in entries[0]
