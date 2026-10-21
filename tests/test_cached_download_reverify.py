# SPDX-License-Identifier: AGPL-3.0-or-later
"""A file that is already on disk is checked against the digest its source
publishes before a re-pull, setup-embeddings or a model load treats it as the
real thing: a whole-repo snapshot, a vision projector, a pinned Whisper
model.bin and a pinned embedding GGUF. Each damaged file is replaced, not kept."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import huggingface_hub
import pytest

from localm import model_manager as mm
from localm import voice
from localm.inference import embedder
from tests._net_config import set_config
from tests.test_mmproj_discovery import _real_mmproj_gguf, _real_text_model_gguf


@pytest.fixture()
def fake_registry(tmp_path, monkeypatch):
    store: dict = {}
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
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
    monkeypatch.setattr("localm.model_manager.pull._policy_head", lambda *a, **k: MagicMock(
        headers={"content-length": "16"}))
    return store, models_dir


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# MOD-02: a whole-repo snapshot
# --------------------------------------------------------------------------- #

GOOD_WEIGHTS = b"w" * 1000
DAMAGED_WEIGHTS = b"w" * 499 + b"X" + b"w" * 500
CONFIG = b"{}"


def _snapshot_hub(monkeypatch, downloaded: list):
    siblings = [
        SimpleNamespace(rfilename="config.json", size=len(CONFIG), lfs=None),
        SimpleNamespace(rfilename="model.safetensors", size=len(GOOD_WEIGHTS),
                        lfs={"sha256": _sha(GOOD_WEIGHTS)}),
    ]

    class _Api:
        def __init__(self, *a, **kw):
            pass

        def model_info(self, repo_id, files_metadata=True):
            return SimpleNamespace(siblings=siblings, sha="a" * 40)

    def _snapshot_download(repo_id, local_dir, **kw):
        d = Path(local_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_bytes(CONFIG)
        (d / "model.safetensors").write_bytes(GOOD_WEIGHTS)
        downloaded.append(repo_id)
        return str(d)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snapshot_download)


class TestSnapshotRepull:
    def _lay(self, models_dir, weights: bytes) -> Path:
        dest = models_dir / "snap"
        dest.mkdir()
        (dest / "config.json").write_bytes(CONFIG)
        (dest / "model.safetensors").write_bytes(weights)
        return dest

    def test_a_damaged_weight_file_is_replaced_on_a_repull(
            self, fake_registry, monkeypatch, capsys):
        store, models_dir = fake_registry
        dest = self._lay(models_dir, DAMAGED_WEIGHTS)
        store["snap"] = {"path": str(dest), "source": "hf:o/r", "model_type": "llm"}
        downloaded: list = []
        _snapshot_hub(monkeypatch, downloaded)

        ok = mm._pull_hf_snapshot("o/r", "snap")

        assert (dest / "model.safetensors").read_bytes() == GOOD_WEIGHTS
        assert downloaded == ["o/r"]
        assert ok is True
        assert "already downloaded" not in " ".join(capsys.readouterr().out.lower().split())

    def test_an_intact_snapshot_is_left_alone(self, fake_registry, monkeypatch, capsys):
        store, models_dir = fake_registry
        dest = self._lay(models_dir, GOOD_WEIGHTS)
        downloaded: list = []
        _snapshot_hub(monkeypatch, downloaded)

        ok = mm._pull_hf_snapshot("o/r", "snap")

        assert ok is True
        assert downloaded == []
        assert (dest / "model.safetensors").read_bytes() == GOOD_WEIGHTS
        assert "already downloaded" in " ".join(capsys.readouterr().out.lower().split())

    def test_a_truncated_weight_file_of_a_registered_model_is_fetched_again(
            self, fake_registry, monkeypatch, capsys):
        store, models_dir = fake_registry
        dest = self._lay(models_dir, GOOD_WEIGHTS[:400])
        store["snap"] = {"path": str(dest), "source": "hf:o/r", "model_type": "llm"}
        downloaded: list = []
        _snapshot_hub(monkeypatch, downloaded)

        ok = mm._pull_hf_snapshot("o/r", "snap")

        assert (dest / "model.safetensors").read_bytes() == GOOD_WEIGHTS
        assert downloaded == ["o/r"]
        assert ok is True
        assert "skipping" not in capsys.readouterr().out.lower()

    def test_a_damaged_file_that_is_in_use_is_reported_and_nothing_is_fetched(
            self, fake_registry, monkeypatch, capsys):
        store, models_dir = fake_registry
        dest = self._lay(models_dir, DAMAGED_WEIGHTS)
        store["snap"] = {"path": str(dest), "source": "hf:o/r", "model_type": "llm"}
        downloaded: list = []
        _snapshot_hub(monkeypatch, downloaded)
        real_unlink = Path.unlink

        def _locked(self, missing_ok=False):
            if self.name == "model.safetensors":
                raise PermissionError(32, "in use")
            return real_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", _locked)

        ok = mm._pull_hf_snapshot("o/r", "snap")

        assert ok is False
        assert downloaded == []
        assert (dest / "model.safetensors").read_bytes() == DAMAGED_WEIGHTS
        assert "could not delete" in " ".join(capsys.readouterr().out.lower().split())

    def test_nothing_is_deleted_when_the_folder_belongs_to_another_model(
            self, fake_registry, monkeypatch):
        store, models_dir = fake_registry
        dest = self._lay(models_dir, DAMAGED_WEIGHTS)
        store["other"] = {"path": str(dest), "source": "hf:someone/else", "model_type": "llm"}
        downloaded: list = []
        _snapshot_hub(monkeypatch, downloaded)

        assert mm._pull_hf_snapshot("o/r", "snap") is False

        assert (dest / "model.safetensors").read_bytes() == DAMAGED_WEIGHTS
        assert downloaded == []

    def test_a_failed_refetch_says_the_damaged_files_are_gone(
            self, fake_registry, monkeypatch, capsys):
        store, models_dir = fake_registry
        dest = self._lay(models_dir, DAMAGED_WEIGHTS)
        store["snap"] = {"path": str(dest), "source": "hf:o/r", "model_type": "llm"}
        _snapshot_hub(monkeypatch, [])

        def _offline(**kw):
            raise OSError("offline")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", _offline)

        assert mm._pull_hf_snapshot("o/r", "snap") is False

        out = " ".join(capsys.readouterr().out.lower().split())
        assert "damaged files were deleted" in out
        assert not (dest / "model.safetensors").exists()

    def test_a_listing_without_digests_keeps_the_size_only_check(
            self, fake_registry, monkeypatch):
        store, models_dir = fake_registry
        dest = self._lay(models_dir, DAMAGED_WEIGHTS)
        downloaded: list = []
        _snapshot_hub(monkeypatch, downloaded)

        class _NoDigestApi:
            def __init__(self, *a, **kw):
                pass

            def model_info(self, repo_id, files_metadata=True):
                return SimpleNamespace(siblings=[
                    SimpleNamespace(rfilename="config.json", size=len(CONFIG), lfs=None),
                    SimpleNamespace(rfilename="model.safetensors",
                                    size=len(GOOD_WEIGHTS), lfs=None)], sha="a" * 40)

        monkeypatch.setattr(huggingface_hub, "HfApi", _NoDigestApi)

        assert mm._pull_hf_snapshot("o/r", "snap") is True
        assert downloaded == []
        assert (dest / "model.safetensors").read_bytes() == DAMAGED_WEIGHTS


# --------------------------------------------------------------------------- #
# MOD-02: a vision projector
# --------------------------------------------------------------------------- #

def _wire_hub_by_repo(monkeypatch, files: dict):
    """A hub where *files* maps (repo, filename) to bytes: published digests,
    the repo listing and the downloads all come from it."""
    downloaded: list = []

    monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo, fn: (
        _sha(files[(repo, fn)]) if (repo, fn) in files and "mmproj" in fn else None))

    class _Api:
        def __init__(self, *a, **kw):
            pass

        def list_repo_files(self, repo_id):
            return [fn for (r, fn) in files if r == repo_id]

    def _download(repo_id, filename, local_dir, **kw):
        downloaded.append((repo_id, filename))
        out = Path(local_dir) / filename
        out.write_bytes(files[(repo_id, filename)])
        return str(out)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    return downloaded


class TestProjectorRepull:
    @pytest.fixture()
    def blobs(self, tmp_path):
        model = _real_text_model_gguf(tmp_path / "_model.gguf", "qwen2", 3584).read_bytes()
        right = _real_mmproj_gguf(tmp_path / "_right.gguf", 3584).read_bytes()
        return model, right

    def test_the_repos_own_damaged_projector_is_replaced_in_place(
            self, fake_registry, monkeypatch, blobs):
        store, models_dir = fake_registry
        model, right = blobs
        downloaded = _wire_hub_by_repo(monkeypatch, {
            ("o/r", "main.gguf"): model, ("o/r", "mmproj-main-f16.gguf"): right})
        assert mm._pull_gguf_file("o/r:main.gguf", None) is True
        projector = models_dir / "mmproj-main-f16.gguf"
        projector.write_bytes(right + b"damaged")
        downloaded.clear()

        assert mm._pull_gguf_file("o/r:main.gguf", None) is True

        assert projector.read_bytes() == right
        assert not (models_dir / "mmproj-main-f16-2.gguf").exists()
        assert downloaded == [("o/r", "mmproj-main-f16.gguf")]
        assert store["main"]["mmproj"] == str(projector.resolve())

    def test_a_projector_without_a_recorded_origin_is_never_replaced(
            self, fake_registry, monkeypatch, blobs):
        store, models_dir = fake_registry
        model, right = blobs
        _wire_hub_by_repo(monkeypatch, {
            ("o/r", "main.gguf"): model, ("o/r", "mmproj-main-f16.gguf"): right})
        assert mm._pull_gguf_file("o/r:main.gguf", None) is True
        projector = models_dir / "mmproj-main-f16.gguf"
        projector.write_bytes(right + b"damaged")
        (models_dir / "mmproj-main-f16.gguf.origin.json").unlink()

        assert mm._pull_gguf_file("o/r:main.gguf", None) is True

        assert projector.read_bytes() == right + b"damaged"
        assert (models_dir / "mmproj-main-f16-2.gguf").read_bytes() == right

    def test_a_hostile_origin_file_is_read_as_no_origin(
            self, fake_registry, monkeypatch, blobs):
        store, models_dir = fake_registry
        model, right = blobs
        _wire_hub_by_repo(monkeypatch, {
            ("o/r", "main.gguf"): model, ("o/r", "mmproj-main-f16.gguf"): right})
        assert mm._pull_gguf_file("o/r:main.gguf", None) is True
        projector = models_dir / "mmproj-main-f16.gguf"
        projector.write_bytes(right + b"damaged")
        (models_dir / "mmproj-main-f16.gguf.origin.json").write_text("[" * 200000, encoding="utf-8")

        assert mm._pull_gguf_file("o/r:main.gguf", None) is True

        assert projector.read_bytes() == right + b"damaged"

    def test_a_projector_chosen_from_another_repo_is_not_replaced_by_the_model_repos(
            self, fake_registry, monkeypatch, blobs):
        store, models_dir = fake_registry
        model, theirs = blobs
        ours = theirs + b"-from-the-model-repo"
        downloaded = _wire_hub_by_repo(monkeypatch, {
            ("b/r", "main.gguf"): model, ("b/r", "other.gguf"): model,
            ("b/r", "mmproj-model-f16.gguf"): ours,
            ("c/r", "mmproj-model-f16.gguf"): theirs})
        assert mm.pull_model("b/r:main.gguf", mmproj_spec="c/r:mmproj-model-f16.gguf") is True
        assert (models_dir / "mmproj-model-f16.gguf").read_bytes() == theirs
        downloaded.clear()

        assert mm._pull_gguf_file("b/r:other.gguf", None) is True

        assert (models_dir / "mmproj-model-f16.gguf").read_bytes() == theirs
        assert (models_dir / "mmproj-model-f16-2.gguf").read_bytes() == ours
        assert store["main"]["mmproj"] == str((models_dir / "mmproj-model-f16.gguf").resolve())

    def test_a_projector_that_is_in_use_is_left_and_the_pull_still_succeeds(
            self, fake_registry, monkeypatch, blobs):
        store, models_dir = fake_registry
        model, right = blobs
        _wire_hub_by_repo(monkeypatch, {
            ("o/r", "main.gguf"): model, ("o/r", "mmproj-main-f16.gguf"): right})
        assert mm._pull_gguf_file("o/r:main.gguf", None) is True
        projector = models_dir / "mmproj-main-f16.gguf"
        projector.write_bytes(right + b"damaged")
        real_unlink = Path.unlink

        def _locked(self, missing_ok=False):
            if self.name == "mmproj-main-f16.gguf":
                raise PermissionError(32, "in use")
            return real_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", _locked)

        assert mm._pull_gguf_file("o/r:main.gguf", None) is True

        assert projector.read_bytes() == right + b"damaged"
        assert (models_dir / "mmproj-main-f16-2.gguf").read_bytes() == right


# --------------------------------------------------------------------------- #
# VOX-05: a cached Whisper model.bin
# --------------------------------------------------------------------------- #

WHISPER_BODY = b"whisper-model-bytes" * 100
WHISPER_REPO = "Systran/faster-whisper-base"


class _Model:
    instances: list = []

    def __init__(self, source, **kw):
        self.source, self.kw = source, kw
        _Model.instances.append(self)


class TestCachedWhisper:
    @pytest.fixture(autouse=True)
    def _home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "home"))
        monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
        set_config(monkeypatch, lambda: {"voice_stt_model": "base", "net_mode": "off"})
        monkeypatch.setattr(voice, "_STT_PINS", {
            repo: (commit, _sha(WHISPER_BODY)) for repo, (commit, _) in voice._STT_PINS.items()})
        _Model.instances = []

    def _lay(self, body: bytes) -> Path:
        commit = voice._STT_PINS[WHISPER_REPO][0]
        repo_dir = voice.stt_cache_dir() / "models--Systran--faster-whisper-base"
        snap = repo_dir / "snapshots" / commit
        snap.mkdir(parents=True)
        (snap / "model.bin").write_bytes(body)
        return repo_dir

    def test_a_changed_model_bin_in_the_cache_is_deleted_and_never_loaded(self):
        repo_dir = self._lay(b"a changed byte" + WHISPER_BODY)

        with pytest.raises(RuntimeError, match="integrity check"):
            voice._load_whisper(_Model, "base", voice.stt_cache_dir(), True)

        assert _Model.instances == []
        assert not repo_dir.exists()
        assert voice.stt_model_cached() == (False, "base")

    def test_an_intact_older_snapshot_survives_a_bad_pinned_one_and_stays_loadable(self):
        commit = voice._STT_PINS[WHISPER_REPO][0]
        repo_dir = self._lay(b"a changed byte" + WHISPER_BODY)
        old = repo_dir / "snapshots" / ("1" * 40)
        old.mkdir()
        (old / "model.bin").write_bytes(b"an older but intact model")
        (repo_dir / "refs").mkdir()
        (repo_dir / "refs" / "main").write_text(commit, encoding="utf-8")

        with pytest.raises(RuntimeError, match="integrity check"):
            voice._load_whisper(_Model, "base", voice.stt_cache_dir(), True)

        assert not (repo_dir / "snapshots" / commit).exists()
        assert (old / "model.bin").read_bytes() == b"an older but intact model"
        assert (repo_dir / "refs" / "main").read_text(encoding="utf-8") == "1" * 40
        assert voice.stt_model_cached() == (True, "base")

    def test_the_file_the_bad_snapshot_points_at_is_deleted_too(self):
        commit = voice._STT_PINS[WHISPER_REPO][0]
        repo_dir = voice.stt_cache_dir() / "models--Systran--faster-whisper-base"
        (repo_dir / "blobs").mkdir(parents=True)
        blob = repo_dir / "blobs" / "etag"
        blob.write_bytes(b"a changed byte" + WHISPER_BODY)
        snap = repo_dir / "snapshots" / commit
        snap.mkdir(parents=True)
        try:
            (snap / "model.bin").symlink_to(blob)
        except OSError:
            pytest.skip("symlinks are not available here")
        old = repo_dir / "snapshots" / ("1" * 40)
        old.mkdir()
        (old / "model.bin").write_bytes(b"an older but intact model")

        with pytest.raises(RuntimeError, match="integrity check"):
            voice._load_whisper(_Model, "base", voice.stt_cache_dir(), True)

        assert not blob.exists() and not snap.exists()

    def test_the_error_does_not_claim_a_deletion_that_did_not_happen(self, monkeypatch):
        self._lay(b"a changed byte" + WHISPER_BODY)
        monkeypatch.setattr(voice.shutil, "rmtree", lambda *a, **k: None)

        with pytest.raises(RuntimeError) as raised:
            voice._load_whisper(_Model, "base", voice.stt_cache_dir(), True)

        assert "could not all be deleted" in str(raised.value)
        assert "files were deleted" not in str(raised.value)

    def test_an_intact_cached_model_bin_loads(self):
        self._lay(WHISPER_BODY)
        voice._load_whisper(_Model, "base", voice.stt_cache_dir(), True)
        assert len(_Model.instances) == 1


# --------------------------------------------------------------------------- #
# RAG-07: an existing embedding file
# --------------------------------------------------------------------------- #

EMBED_GOOD = b"the real embedding bytes"


class TestExistingEmbeddingFile:
    @pytest.fixture()
    def known(self, tmp_path, monkeypatch):
        monkeypatch.setattr(embedder, "_EMBEDDING_PINS",
                            {("org/repo", "model.gguf"): ("a" * 40, _sha(EMBED_GOOD))})
        monkeypatch.setattr(embedder, "KNOWN_EMBEDDING_MODELS",
                            {"m": ("org/repo", "model.gguf")})
        monkeypatch.setattr(embedder, "_current_spec", lambda: "m")
        monkeypatch.setattr(embedder, "_embeddings_dir", lambda: tmp_path)
        monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "allow")
        return tmp_path / "model.gguf"

    def _hub(self, monkeypatch, content: bytes):
        seen: list = []

        def _download(repo, filename, local_dir=None, **kw):
            seen.append(kw.get("revision"))
            out = Path(local_dir) / filename
            out.write_bytes(content)
            return str(out)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)
        return seen

    def test_a_damaged_file_is_fetched_again_by_an_install(self, known, monkeypatch):
        known.write_bytes(b"a bit-flipped file")
        seen = self._hub(monkeypatch, EMBED_GOOD)

        assert embedder.resolve_embedding_model_path(allow_download=True) == str(known)

        assert known.read_bytes() == EMBED_GOOD
        assert seen == ["a" * 40]

    def test_a_damaged_file_that_cannot_be_fetched_again_is_reported_not_accepted(
            self, known, monkeypatch):
        known.write_bytes(b"a bit-flipped file")
        self._hub(monkeypatch, b"swapped bytes")

        assert embedder.resolve_embedding_model_path(allow_download=True) is None

        assert "pinned" in (embedder.last_error() or "")

    def test_an_unreadable_file_is_fetched_again_not_raised(self, known, monkeypatch):
        known.write_bytes(b"whatever")
        seen = self._hub(monkeypatch, EMBED_GOOD)

        def _unreadable(path):
            raise PermissionError(13, "denied")

        monkeypatch.setattr(embedder, "file_sha256", _unreadable)
        embedder.resolve_embedding_model_path(allow_download=True)

        assert seen == ["a" * 40]

    def test_an_intact_file_is_not_fetched_again(self, known, monkeypatch):
        known.write_bytes(EMBED_GOOD)
        seen = self._hub(monkeypatch, b"never used")

        assert embedder.resolve_embedding_model_path(allow_download=True) == str(known)

        assert known.read_bytes() == EMBED_GOOD
        assert seen == []

    def test_a_per_embed_resolve_does_not_read_the_file(self, known, monkeypatch):
        known.write_bytes(b"a bit-flipped file")
        monkeypatch.setattr(embedder, "file_sha256",
                            lambda p: pytest.fail("a per-embed resolve must not hash"))

        assert embedder.resolve_embedding_model_path(allow_download=False) == str(known)
        assert embedder.resolve_embedding_model_path() == str(known)
