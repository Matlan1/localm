# SPDX-License-Identifier: AGPL-3.0-or-later
"""A known embedding model (e.g. bge-small-en-v1.5) is not downloaded when a
registered model already is that file: same file name, and the pinned sha256
when the registry recorded a digest (or when the caller forces verification)."""

from __future__ import annotations

import hashlib
import json

import pytest

import localm.config as config
from localm.inference import embedder

REPO_FILE = embedder.KNOWN_EMBEDDING_MODELS["bge-small-en-v1.5"]
REAL = b"the real bytes"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REGISTRY_FILE", tmp_path / "registry.json")
    monkeypatch.setattr(embedder, "_embeddings_dir", lambda: tmp_path / "emb")
    monkeypatch.setattr(embedder, "_current_spec", lambda: "bge-small-en-v1.5")
    monkeypatch.setattr(embedder, "_EMBEDDING_PINS", {
        REPO_FILE: ("rev", hashlib.sha256(REAL).hexdigest())})
    downloads = []
    monkeypatch.setattr(
        embedder, "_download_known",
        lambda *a, **kw: downloads.append(a) or None)
    folder = tmp_path / "central"
    folder.mkdir()
    return tmp_path, folder, downloads


def _register(root, name, path, **extra):
    (root / "registry.json").write_text(json.dumps(
        {name: {"path": str(path), "source": "local", "model_type": "embedding",
                **extra}}), encoding="utf-8")


def test_a_registered_file_with_the_known_name_is_used_without_a_download(env):
    root, folder, downloads = env
    f = folder / REPO_FILE[1]
    f.write_bytes(REAL)
    _register(root, "bge-small-en-v1.5-q4_k_m", f)

    assert embedder.resolve_embedding_model_path() == str(f)
    assert embedder.configured_model_path() == str(f)
    assert downloads == []


def test_the_name_match_ignores_case(env):
    root, folder, downloads = env
    f = folder / REPO_FILE[1].upper().replace(".GGUF", ".gguf")
    f.write_bytes(REAL)
    _register(root, "mine", f)
    assert embedder.resolve_embedding_model_path() == str(f)


def test_a_recorded_digest_that_differs_from_the_pin_is_not_used(env):
    root, folder, downloads = env
    f = folder / REPO_FILE[1]
    f.write_bytes(b"other bytes")
    _register(root, "mine", f, sha256=hashlib.sha256(b"other bytes").hexdigest())

    assert embedder.resolve_embedding_model_path() is None
    assert len(downloads) == 1
    assert embedder.configured_model_path() is None


def test_a_forced_resolve_hashes_a_file_with_no_recorded_digest(env):
    root, folder, downloads = env
    f = folder / REPO_FILE[1]
    f.write_bytes(b"other bytes")
    _register(root, "mine", f)

    assert embedder.resolve_embedding_model_path() == str(f)
    assert downloads == []
    assert embedder.resolve_embedding_model_path(allow_download=True) is None
    assert len(downloads) == 1


def test_a_registered_file_with_another_name_is_not_taken(env):
    root, folder, downloads = env
    f = folder / "something-else.gguf"
    f.write_bytes(REAL)
    _register(root, "mine", f)
    assert embedder.resolve_embedding_model_path() is None
    assert len(downloads) == 1


def test_a_missing_registered_file_falls_through_to_the_download(env):
    root, folder, downloads = env
    _register(root, "mine", folder / REPO_FILE[1])
    assert embedder.resolve_embedding_model_path() is None
    assert len(downloads) == 1
