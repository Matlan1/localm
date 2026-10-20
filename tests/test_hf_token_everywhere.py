# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Hugging Face token saved in localm's settings reaches every Hugging Face
call a download makes. The hub answers an unauthenticated request with a
warning ("You are sending unauthenticated requests to the HF Hub"), so a call
that leaves the token out prints that warning even when a token is saved.

Each test asserts the token the real huggingface_hub entry point (or the real
HTTP layer) received, not that a call happened."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from localm import model_manager as mm
from localm.model_manager import pull as pull_mod

TOKEN = "hf_SavedInSettings0123456789"


@pytest.fixture()
def saved_token(monkeypatch):
    """A token saved in settings, with no HF_TOKEN in the environment."""
    from localm.model_source_credentials import set_credentials
    monkeypatch.delenv("HF_TOKEN", raising=False)
    set_credentials({"hf_token": TOKEN})
    return TOKEN


@pytest.fixture()
def no_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)


class _SpyHfApi:
    """Records the token given to the constructor or to a call."""

    seen: dict = {}

    def __init__(self, *a, **kw):
        type(self).seen = {"ctor": kw.get("token")}

    def get_paths_info(self, repo_id, paths, **kw):
        type(self).seen["call"] = kw.get("token")
        return []

    def list_repo_files(self, repo_id, **kw):
        return []

    def model_info(self, repo_id, **kw):
        return SimpleNamespace(siblings=[], sha=None)


@pytest.fixture()
def spy_api(monkeypatch):
    import huggingface_hub
    _SpyHfApi.seen = {}
    monkeypatch.setattr(huggingface_hub, "HfApi", _SpyHfApi)
    return _SpyHfApi


class TestHubMetadataCalls:
    def test_the_digest_lookup_sends_the_saved_token(self, saved_token, spy_api):
        mm._hf_file_sha256("o/r", "f.gguf")
        assert spy_api.seen == {"ctor": TOKEN, "call": TOKEN}

    def test_the_repo_listing_sends_the_saved_token(self, saved_token, spy_api):
        mm._hf_repo_files("o/r")
        assert spy_api.seen["ctor"] == TOKEN

    def test_the_snapshot_listing_and_download_send_the_saved_token(
            self, saved_token, spy_api, tmp_path, monkeypatch):
        import huggingface_hub
        models = tmp_path / "models"
        models.mkdir()
        monkeypatch.setattr(mm, "MODELS_DIR", models)
        monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
        monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
        monkeypatch.setattr(mm, "load_registry", lambda: {})
        got = {}

        def fake_snapshot(**kw):
            got.update(kw)
            Path(kw["local_dir"]).mkdir(parents=True, exist_ok=True)
            return kw["local_dir"]

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)

        mm._pull_hf_snapshot("o/r", "snap")

        assert spy_api.seen["ctor"] == TOKEN
        assert got["token"] == TOKEN

    def test_no_saved_token_sends_none(self, no_token, spy_api, monkeypatch):
        monkeypatch.setattr("localm.model_source_credentials.get_hf_token",
                            lambda: None)
        mm._hf_file_sha256("o/r", "f.gguf")
        assert spy_api.seen == {"ctor": None, "call": None}


class TestFileDownloads:
    def test_a_single_file_pull_sends_the_saved_token(
            self, saved_token, tmp_path, monkeypatch):
        models = tmp_path / "models"
        models.mkdir()
        monkeypatch.setattr(mm, "MODELS_DIR", models)
        monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
        monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
        monkeypatch.setattr(mm, "find_by_sha256", lambda *a, **k: [])
        monkeypatch.setattr(mm, "_register", MagicMock())
        monkeypatch.setattr(mm, "_register_with_dedup", MagicMock(return_value=True))
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda *a, **k: None)
        monkeypatch.setattr("huggingface_hub.hf_hub_url",
                            lambda *a, **kw: "https://example.com/test.gguf")
        monkeypatch.setattr(pull_mod, "_policy_head", lambda *a, **kw: MagicMock(
            status_code=200, headers={"content-length": "4"}, history=[]))
        seen = {}

        def fake_download(repo_id, filename, local_dir, **kw):
            seen.update(kw)
            out = Path(local_dir) / filename
            out.write_bytes(b"0123")
            return str(out)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

        assert mm._pull_gguf_file("owner/repo:test.gguf", name="test",
                                  register=False) is True
        assert seen["token"] == TOKEN

    def test_a_projector_download_sends_the_saved_token(
            self, saved_token, tmp_path, monkeypatch):
        seen = {}

        def fake_download(repo_id, filename, local_dir, **kw):
            seen.update(kw)
            out = Path(local_dir) / filename
            out.write_bytes(b"x")
            return str(out)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda *a, **k: None)
        monkeypatch.setattr(pull_mod, "_ensure_hf_resumable_download", lambda: None)

        pull_mod._download_mmproj_file("o/r", "mmproj.gguf",
                                       tmp_path / "mmproj.gguf", tmp_path)

        assert seen["token"] == TOKEN

    def test_the_embedding_model_download_sends_the_saved_token(
            self, saved_token, tmp_path, monkeypatch):
        from localm.inference import embedder
        seen = {}

        def fake_download(repo, filename, **kw):
            seen.update(kw)
            out = Path(kw["local_dir"]) / filename
            out.write_bytes(b"x")
            return str(out)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        monkeypatch.setattr(embedder, "_EMBEDDING_PINS", {})

        embedder._download_known("m", "o/r", "e.gguf", tmp_path / "e.gguf", True)

        assert seen["token"] == TOKEN

    def test_the_pinned_embedding_model_download_sends_the_saved_token(
            self, saved_token, tmp_path, monkeypatch):
        import hashlib

        from localm.inference import embedder
        seen = {}

        def fake_download(repo, filename, **kw):
            seen.update(kw)
            out = Path(kw["local_dir"]) / filename
            out.write_bytes(b"x")
            return str(out)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        pin = ("a" * 40, hashlib.sha256(b"x").hexdigest())

        embedder._download_pinned("m", "o/r", "e.gguf", tmp_path / "e.gguf", pin, {})

        assert seen["token"] == TOKEN

    def test_the_speech_model_download_sends_the_saved_token(
            self, saved_token, tmp_path, monkeypatch):
        from localm import voice
        seen = {}

        def fake_snapshot(repo, **kw):
            seen.update(kw)

        monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
        monkeypatch.setattr(voice, "stt_cache_dir", lambda: tmp_path / "stt")
        monkeypatch.setattr(voice, "stt_model_cached", lambda: (False, "base"))

        voice.prefetch_stt_model(allow_download=True)

        assert seen["token"] == TOKEN


class TestRawHubRequests:
    @staticmethod
    def _capture(monkeypatch):
        sent = []

        def fake_request(method, url, **kw):
            sent.append((url, kw.get("headers")))
            if url.startswith("https://huggingface.co/"):
                return MagicMock(status_code=302, headers={
                    "Location": "https://cdn.example.net/blob"})
            return MagicMock(status_code=200, headers={"content-length": "4"})

        monkeypatch.setattr("localm.netpolicy.pinned_request", fake_request)
        monkeypatch.setattr("localm.netpolicy.check_url", lambda *a, **k: None)
        return sent

    def test_the_size_probe_sends_the_token_to_the_hub_and_not_to_the_redirect_target(
            self, saved_token, monkeypatch):
        sent = self._capture(monkeypatch)

        pull_mod._policy_head("https://huggingface.co/o/r/resolve/main/f.gguf")

        assert sent[0][1] == {"Authorization": f"Bearer {TOKEN}"}
        assert sent[1][0] == "https://cdn.example.net/blob"
        assert not sent[1][1]

    def test_the_size_probe_sends_no_authorization_without_a_token(
            self, no_token, monkeypatch):
        monkeypatch.setattr("localm.model_source_credentials.get_hf_token",
                            lambda: None)
        sent = self._capture(monkeypatch)

        pull_mod._policy_head("https://huggingface.co/o/r/resolve/main/f.gguf")

        assert not sent[0][1]

    def test_the_model_type_probe_sends_the_saved_token(self, saved_token, monkeypatch):
        from localm import discover
        seen = {}

        def fake_get(url, params=None, *, token=None):
            seen["token"] = token
            return {"pipeline_tag": "text-generation", "tags": []}

        monkeypatch.setattr(discover, "_get", fake_get)

        discover._hf_pipeline_tag_to_type("o/r")

        assert seen["token"] == TOKEN
