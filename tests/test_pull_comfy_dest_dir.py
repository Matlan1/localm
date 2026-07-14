# SPDX-License-Identifier: AGPL-3.0-or-later
"""dest_dir/register routing added to _pull_gguf_file() / pull_model() (gap B):
a download routed to a ComfyUI models subfolder must land there (not
MODELS_DIR), skip localm's own registry when asked, still run the traversal
guard against the REAL destination, and refuse rather than silently fall back
to MODELS_DIR when dest_dir is paired with a spec that doesn't dispatch to the
single-file backend."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from localm import model_manager as mm


@pytest.fixture()
def fake_registry(tmp_path, monkeypatch):
    """In-memory registry + temp MODELS_DIR wired into model_manager (mirrors
    test_model_manager_phase3.py's fixture of the same name)."""
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
    return store, models_dir


def _wire_fake_download(monkeypatch, body: bytes = b"fake-model-bytes"):
    def _fake_download(repo_id, filename, local_dir, **kw):
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
        return str(p)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)

    def _fake_head(url, allow_redirects=None, timeout=None):
        h = MagicMock()
        h.headers = {"content-length": str(len(body))}
        return h

    monkeypatch.setattr("requests.head", _fake_head)


class TestPullGgufFileDestDir:
    def test_lands_in_dest_dir_not_models_dir(self, fake_registry, tmp_path, monkeypatch):
        store, models_dir = fake_registry
        _wire_fake_download(monkeypatch)
        dest_dir = tmp_path / "comfyui-models" / "clip"

        ok = mm._pull_gguf_file(
            "comfyanonymous/flux_text_encoders:clip_l.safetensors", None,
            model_type="text-encoder", dest_dir=dest_dir, register=False)

        assert ok is True
        assert (dest_dir / "clip_l.safetensors").is_file()
        assert not (models_dir / "clip_l.safetensors").exists()

    def test_register_false_skips_the_registry(self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        _wire_fake_download(monkeypatch)
        dest_dir = tmp_path / "comfyui-models" / "vae"

        ok = mm._pull_gguf_file(
            "black-forest-labs/FLUX.1-schnell:ae.safetensors", None,
            model_type="vae", dest_dir=dest_dir, register=False)

        assert ok is True
        assert store == {}          # nothing added to localm's own registry

    def test_register_true_still_works_with_dest_dir(self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        _wire_fake_download(monkeypatch)
        dest_dir = tmp_path / "comfyui-models" / "unet"

        ok = mm._pull_gguf_file(
            "city96/FLUX.1-dev-gguf:flux1-dev-Q8_0.gguf", None,
            model_type="diffusion-unet", dest_dir=dest_dir, register=True)

        assert ok is True
        assert "flux1-dev-Q8_0" in store
        # the registered path points at dest_dir, not MODELS_DIR
        assert str(dest_dir) in str(store["flux1-dev-Q8_0"].get("path", ""))

    def test_traversal_guard_checks_the_real_dest_dir(self, fake_registry, tmp_path, monkeypatch):
        _wire_fake_download(monkeypatch)
        dest_dir = tmp_path / "comfyui-models" / "unet"

        ok = mm._pull_gguf_file(
            "owner/repo:../../evil.gguf", None, dest_dir=dest_dir, register=False)

        assert ok is False
        assert not (tmp_path / "evil.gguf").exists()
        assert not (tmp_path.parent / "evil.gguf").exists()

    def test_already_downloaded_short_circuit_uses_dest_dir(
            self, fake_registry, tmp_path, monkeypatch):
        """The 'already downloaded' fast path must check dest_dir, not
        MODELS_DIR, or a file already routed to a ComfyUI folder would look
        'missing' forever and re-download every time."""
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "clip"
        dest_dir.mkdir(parents=True)
        (dest_dir / "clip_l.safetensors").write_bytes(b"already-here")

        downloaded = []
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                            lambda **kw: downloaded.append(1) or "")

        ok = mm._pull_gguf_file(
            "comfyanonymous/flux_text_encoders:clip_l.safetensors", None,
            model_type="text-encoder", dest_dir=dest_dir, register=False)

        assert ok is True
        assert downloaded == []      # no network call - it was already there


class TestPullModelDestDirGuard:
    def test_refuses_dest_dir_with_a_bare_repo_snapshot_spec(self, fake_registry, tmp_path):
        _store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "unet"
        ok = mm.pull_model("some/bare-repo", dest_dir=dest_dir, register=False)
        assert ok is False
        assert not dest_dir.exists()

    def test_refuses_dest_dir_with_a_url_spec(self, fake_registry, tmp_path):
        _store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "unet"
        ok = mm.pull_model("https://example.invalid/model.safetensors",
                           dest_dir=dest_dir, register=False)
        assert ok is False
        assert not dest_dir.exists()

    def test_accepts_dest_dir_with_a_single_file_spec(
            self, fake_registry, tmp_path, monkeypatch):
        store, models_dir = fake_registry
        _wire_fake_download(monkeypatch)
        dest_dir = tmp_path / "comfyui-models" / "clip"

        ok = mm.pull_model(
            "comfyanonymous/flux_text_encoders:clip_l.safetensors",
            model_type="text-encoder", dest_dir=dest_dir, register=False)

        assert ok is True
        assert (dest_dir / "clip_l.safetensors").is_file()
