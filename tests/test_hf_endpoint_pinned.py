# SPDX-License-Identifier: AGPL-3.0-or-later
"""HF_ENDPOINT/HF_HUB_ENDPOINT are ambient env vars localm never exposes as a
setting anywhere (no settings_schema.py key, no CLI flag, no docs). Every
huggingface_hub call site in pull.py and embedder.py pins
endpoint="https://huggingface.co" explicitly, so a stray HF_ENDPOINT or
HF_HUB_ENDPOINT left over in the user's shell cannot redirect a model pull to a
different host.

One test per call site, each asserting the ACTUAL kwarg the real
huggingface_hub function/constructor received - not merely that the call
happened. A spy that only counted calls would pass identically whether the
endpoint kwarg is pinned or left to the env var.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from localm import model_manager as mm
from localm.inference import embedder as embedder_mod

_HF_ENDPOINT = "https://huggingface.co"


@pytest.fixture()
def fake_registry(tmp_path, monkeypatch):
    """Same shape as test_pull_mmproj_autofetch.py's fixture of the same name."""
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
    monkeypatch.setattr("requests.head", lambda *a, **k: MagicMock(
        headers={"content-length": "16"}))
    return store, models_dir


class TestHfApiConstructorEndpointPinned:
    """HfApi() is constructed fresh at each call site - endpoint must be
    pinned on the constructor every time, not just on some calls."""

    def test_hf_file_sha256_pins_endpoint(self, monkeypatch):
        seen = {}

        class _SpyHfApi:
            def __init__(self, *a, **kw):
                seen.update(kw)

            def get_paths_info(self, repo_id, filenames):
                return []

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", _SpyHfApi)

        mm._hf_file_sha256("o/r", "f.gguf")

        assert seen.get("endpoint") == _HF_ENDPOINT

    def test_hf_repo_files_pins_endpoint(self, monkeypatch):
        seen = {}

        class _SpyHfApi:
            def __init__(self, *a, **kw):
                seen.update(kw)

            def list_repo_files(self, repo_id):
                return []

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", _SpyHfApi)

        mm._hf_repo_files("o/r")

        assert seen.get("endpoint") == _HF_ENDPOINT

    def test_pull_hf_snapshot_model_info_pins_endpoint(self, fake_registry, monkeypatch):
        store, models_dir = fake_registry
        seen = {}

        class _SpyHfApi:
            def __init__(self, *a, **kw):
                seen.update(kw)

            def model_info(self, repo_id, files_metadata=True):
                return SimpleNamespace(siblings=[])

        def _fake_snapshot_download(**kw):
            Path(kw["local_dir"]).mkdir(parents=True, exist_ok=True)
            return kw["local_dir"]

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", _SpyHfApi)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download)

        mm._pull_hf_snapshot("o/r", "snap")

        assert seen.get("endpoint") == _HF_ENDPOINT


class TestHfHubDownloadEndpointPinned:
    """hf_hub_download() is called at four distinct sites in pull.py, plus
    embedder.py's own known-embedding-model fetch - endpoint must be pinned
    on every one."""

    def test_mmproj_autofetch_pins_endpoint(self, monkeypatch, tmp_path):
        """_maybe_fetch_repo_mmproj's own-repo projector download."""
        seen = {}

        class _FakeHfApi:
            def __init__(self, *a, **kw):
                pass

            def list_repo_files(self, repo_id):
                return ["main.gguf", "mmproj-main-f16.gguf"]

        def _fake_download(repo_id, filename, local_dir, **kw):
            seen.update(kw)
            p = Path(local_dir) / filename
            p.write_bytes(b"not-a-real-gguf")
            return str(p)

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)

        mm._maybe_fetch_repo_mmproj("o/r", "main.gguf", tmp_path)

        assert seen.get("endpoint") == _HF_ENDPOINT

    def test_explicit_mmproj_pins_endpoint(self, monkeypatch, tmp_path):
        """_fetch_explicit_mmproj's user-named --mmproj download."""
        seen = {}

        def _fake_download(repo_id, filename, local_dir, **kw):
            seen.update(kw)
            p = Path(local_dir) / filename
            p.write_bytes(b"not-a-real-gguf")
            return str(p)

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)

        mm._fetch_explicit_mmproj("owner/repo:file.gguf", tmp_path)

        assert seen.get("endpoint") == _HF_ENDPOINT

    def test_gguf_parts_loop_pins_endpoint_on_download_and_url(
            self, fake_registry, monkeypatch):
        """_pull_gguf_file's main download loop - covers BOTH hf_hub_download
        (the transfer itself) and hf_hub_url (the disk-space preflight HEAD
        target), which sit side by side in the same function. Leaving either
        one keyed to the ambient env var while the other is pinned would let
        the preflight check a DIFFERENT host than the one actually fetched
        from."""
        store, models_dir = fake_registry
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo, fn: None)
        download_seen = {}
        url_seen = {}

        def _fake_download(repo_id, filename, local_dir, **kw):
            download_seen.update(kw)
            p = Path(local_dir) / filename
            p.write_bytes(b"gguf-bytes")
            return str(p)

        def _fake_hf_hub_url(repo_id, filename, **kw):
            url_seen.update(kw)
            return f"https://example.invalid/{repo_id}/{filename}"

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
        monkeypatch.setattr(huggingface_hub, "hf_hub_url", _fake_hf_hub_url)

        ok = mm._pull_gguf_file("o/r:main.gguf", None)

        assert ok is True
        assert download_seen.get("endpoint") == _HF_ENDPOINT
        assert url_seen.get("endpoint") == _HF_ENDPOINT

    def test_embedder_known_model_download_pins_endpoint(self, monkeypatch, tmp_path):
        """embedder.py's _download_known - the automatic on-demand fetch of a
        known small embedding GGUF (bge-small etc)."""
        seen = {}

        def _fake_download(repo, filename, local_dir, **kw):
            seen.update(kw)
            p = Path(local_dir) / filename
            p.write_bytes(b"gguf-bytes")
            return str(p)

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
        monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "allow")

        dest = tmp_path / "bge-small-en-v1.5-q4_k_m.gguf"
        result = embedder_mod._download_known(
            "bge-small-en-v1.5", "CompendiumLabs/bge-small-en-v1.5-gguf",
            "bge-small-en-v1.5-q4_k_m.gguf", dest, allow_download=True)

        assert result is not None
        assert seen.get("endpoint") == _HF_ENDPOINT


class TestEndpointConstantIsTheRealHfHost:
    """Guard the literal itself: a typo'd or blank endpoint would still pass
    every spy assertion above as long as it's SOME string every call site
    agrees on. Pin the value, not just its uniformity."""

    def test_pull_module_constant_is_the_real_hf_host(self):
        from localm.model_manager import pull as pull_mod
        assert pull_mod._HF_ENDPOINT == "https://huggingface.co"

    def test_embedder_module_constant_is_the_real_hf_host(self):
        assert embedder_mod._HF_ENDPOINT == "https://huggingface.co"
