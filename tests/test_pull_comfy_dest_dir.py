# SPDX-License-Identifier: AGPL-3.0-or-later
"""dest_dir/register routing in _pull_gguf_file() / pull_model(): a download
routed to a ComfyUI models subfolder must land there (not MODELS_DIR), skip
localm's own registry when asked, still run the traversal guard against the REAL
destination, and refuse rather than silently fall back to MODELS_DIR when
dest_dir is paired with a spec that does not dispatch to the single-file
backend."""

import os
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
    monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo, fn: None)
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


class TestRepoFolderSpec:
    """``owner/repo:dir/file`` names a file inside a folder of the repository;
    it is fetched from that path, staged under the destination's hidden
    ``.cache/localm-staging`` and saved under its bare filename."""

    SPEC = "Comfy-Org/ACE-Step_ComfyUI_repackaged:all_in_one/ace_step_v1_3.5b.safetensors"

    def _record_downloads(self, monkeypatch, body=b"ace-bytes"):
        """Fake huggingface_hub.hf_hub_download the way the real one behaves
        for a local_dir download: the file's folder is created when the
        download starts, and the finished file is moved into it without
        creating it again."""
        calls = []

        def _fake_download(repo_id, filename, local_dir, **kw):
            calls.append((repo_id, filename, Path(local_dir)))
            p = Path(local_dir) / filename
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(body)
            return str(p)

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)

        def _fake_head(url, allow_redirects=None, timeout=None):
            # An ETag, as HuggingFace sends, so the resume check runs
            # huggingface_hub's real local-dir path helpers.
            h = MagicMock()
            h.history = []
            h.headers = {"content-length": str(len(body)), "ETag": '"0123abcd"'}
            return h

        monkeypatch.setattr("requests.head", _fake_head)
        return calls

    def test_lands_under_its_bare_name_with_no_repo_folder_in_the_destination(
            self, fake_registry, tmp_path, monkeypatch):
        calls = self._record_downloads(monkeypatch)
        dest_dir = tmp_path / "comfyui-models" / "checkpoints"

        ok = mm._pull_gguf_file(self.SPEC, None, model_type="diffusion-unet",
                                dest_dir=dest_dir, register=False)

        assert (dest_dir / "ace_step_v1_3.5b.safetensors").read_bytes() == b"ace-bytes"
        assert not (dest_dir / "all_in_one").exists(), "a repo folder appeared in the destination"
        assert calls == [("Comfy-Org/ACE-Step_ComfyUI_repackaged",
                          "all_in_one/ace_step_v1_3.5b.safetensors",
                          dest_dir / ".cache" / "localm-staging")]
        assert ok is True

    def test_the_digest_is_looked_up_by_the_path_in_the_repo(
            self, fake_registry, tmp_path, monkeypatch):
        self._record_downloads(monkeypatch)
        asked = []
        monkeypatch.setattr(mm, "_hf_file_sha256",
                            lambda repo, fn: asked.append((repo, fn)) or None)

        mm._pull_gguf_file(self.SPEC, None, model_type="diffusion-unet",
                           dest_dir=tmp_path / "ckpt", register=False)

        assert asked == [("Comfy-Org/ACE-Step_ComfyUI_repackaged",
                          "all_in_one/ace_step_v1_3.5b.safetensors")]

    def test_a_same_named_folder_in_the_destination_is_left_alone(
            self, fake_registry, tmp_path, monkeypatch):
        self._record_downloads(monkeypatch)
        dest_dir = tmp_path / "comfyui-models" / "checkpoints"
        keep = dest_dir / "all_in_one" / "mine.txt"
        keep.parent.mkdir(parents=True)
        keep.write_text("user file")
        empty = dest_dir / "all_in_one" / "empty"
        empty.mkdir()

        assert mm._pull_gguf_file(self.SPEC, None, model_type="diffusion-unet",
                                  dest_dir=dest_dir, register=False) is True

        assert keep.read_text() == "user file"
        assert empty.is_dir()
        assert (dest_dir / "ace_step_v1_3.5b.safetensors").is_file()

    def test_two_pulls_sharing_a_repo_folder_do_not_break_each_other(
            self, fake_registry, tmp_path, monkeypatch):
        import threading

        dest_dir = tmp_path / "comfyui-models" / "vae"
        b_started = threading.Event()
        a_done = threading.Event()

        def _fake_download(repo_id, filename, local_dir, **kw):
            p = Path(local_dir) / filename
            p.parent.mkdir(parents=True, exist_ok=True)
            if filename.endswith("b_vae_v1.safetensors"):
                b_started.set()
                assert a_done.wait(10), "pull A never finished"
            if not p.parent.is_dir():
                raise FileNotFoundError(f"[Errno 2] No such file or directory: {p}")
            p.write_bytes(filename.encode())
            return str(p)

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
        monkeypatch.setattr("requests.head", lambda *a, **k: MagicMock(history=[], headers={}))

        results = {}

        def _pull_b():
            results["b"] = mm._pull_gguf_file(
                "Comfy-Org/x:split_files/vae/b_vae_v1.safetensors", None,
                model_type="vae", dest_dir=dest_dir, register=False)

        t = threading.Thread(target=_pull_b)
        t.start()
        assert b_started.wait(10)
        results["a"] = mm._pull_gguf_file(
            "Comfy-Org/x:split_files/vae/a_vae_v1.safetensors", None,
            model_type="vae", dest_dir=dest_dir, register=False)
        a_done.set()
        t.join(10)

        assert (dest_dir / "a_vae_v1.safetensors").is_file()
        assert (dest_dir / "b_vae_v1.safetensors").is_file(), "pull B lost its folder to pull A"
        assert results == {"a": True, "b": True}

    def test_a_refused_download_creates_nothing_in_the_destination(
            self, fake_registry, tmp_path, monkeypatch):
        calls = self._record_downloads(monkeypatch)
        monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: False)
        dest_dir = tmp_path / "comfyui-models" / "checkpoints"

        ok = mm._pull_gguf_file(self.SPEC, None, model_type="diffusion-unet",
                                dest_dir=dest_dir, register=False)

        assert not (dest_dir / "all_in_one").exists()
        assert calls == []
        assert ok is False

    def test_a_failed_download_creates_nothing_in_the_destination(
            self, fake_registry, tmp_path, monkeypatch):
        def _failing_download(repo_id, filename, local_dir, **kw):
            (Path(local_dir) / filename).parent.mkdir(parents=True, exist_ok=True)
            raise OSError("connection reset")

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _failing_download)
        monkeypatch.setattr("requests.head", lambda *a, **k: MagicMock(history=[], headers={}))
        dest_dir = tmp_path / "comfyui-models" / "checkpoints"

        assert mm._pull_gguf_file(self.SPEC, None, model_type="diffusion-unet",
                                  dest_dir=dest_dir, register=False) is False

        assert not (dest_dir / "all_in_one").exists()
        assert not (dest_dir / "ace_step_v1_3.5b.safetensors").exists()

    @pytest.mark.parametrize("spec", [
        "owner/repo:../evil.safetensors",
        "owner/repo:a/../../evil.safetensors",
        "owner/repo:./evil.safetensors",
        "owner/repo:a//evil.safetensors",
        "owner/repo:/evil.safetensors",
        "owner/repo:a\\b/evil.safetensors",
        "owner/repo:C:/evil.safetensors",
        "owner/repo:.hidden/evil.safetensors",
    ])
    def test_an_unsafe_repo_folder_is_refused_before_any_download(
            self, fake_registry, tmp_path, monkeypatch, spec):
        calls = self._record_downloads(monkeypatch)
        dest_dir = tmp_path / "comfyui-models" / "checkpoints"

        ok = mm._pull_gguf_file(spec, None, model_type="diffusion-unet",
                                dest_dir=dest_dir, register=False)

        assert calls == [], "an unsafe spec reached the downloader"
        assert not any(p.name == "evil.safetensors" for p in tmp_path.rglob("*"))
        assert ok is False

    def test_a_repo_folder_that_resolves_outside_the_staging_folder_is_refused(
            self, fake_registry, tmp_path, monkeypatch):
        calls = self._record_downloads(monkeypatch)
        dest_dir = tmp_path / "comfyui-models" / "checkpoints"
        staging = dest_dir / ".cache" / "localm-staging"
        staging.mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        link = staging / "linked"
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(str(outside), str(link))
        else:
            os.symlink(outside, link, target_is_directory=True)
        assert link.resolve() == outside.resolve(), "precondition: the link points outside"

        ok = mm._pull_gguf_file("owner/repo:linked/evil_model_v1.safetensors", None,
                                model_type="diffusion-unet", dest_dir=dest_dir,
                                register=False)

        assert calls == [], "a folder resolving outside the staging folder was used"
        assert list(outside.iterdir()) == []
        assert ok is False


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


# ---------------------------------------------------------------------------
# A dest_dir download must not report success without writing the file.
#
# The pre-download duplicate check asks localm's own REGISTRY whether the bytes
# are already present. With dest_dir set the file is wanted somewhere ELSE
# entirely (ComfyUI's models folder), so a registry hit says nothing about the
# destination. In a pull JOB (no TTY) _prompt_predownload_dup returns "skip".
# ---------------------------------------------------------------------------

import hashlib

_DIGEST = hashlib.sha256(b"fake-model-bytes").hexdigest()


class TestDestDirDuplicateSkip:
    def test_dest_dir_download_happens_even_when_the_sha256_is_registered(
            self, fake_registry, tmp_path, monkeypatch):
        """The file must physically land in dest_dir. Reporting success without
        writing it is a step reporting success after not doing the work."""
        store, models_dir = fake_registry
        # The very same bytes are already registered in localm's own registry...
        already = models_dir / "flux1-dev-Q8_0.gguf"
        already.write_bytes(b"fake-model-bytes")
        store["flux-local"] = {"path": str(already), "source": "local",
                               "sha256": _DIGEST, "model_type": "diffusion-unet"}
        # ...and HF metadata reports that same digest for the file being pulled.
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo, fn: _DIGEST)
        _wire_fake_download(monkeypatch)
        dest_dir = tmp_path / "comfyui-models" / "unet"

        ok = mm._pull_gguf_file(
            "city96/FLUX.1-dev-gguf:flux1-dev-Q8_0.gguf", None,
            model_type="diffusion-unet", dest_dir=dest_dir, register=False)

        assert ok is True
        assert (dest_dir / "flux1-dev-Q8_0.gguf").is_file(), (
            "the pull reported success, so the file must actually be in the "
            "ComfyUI folder - a registry dup elsewhere does not put it there")

    def test_dest_dir_pull_does_not_alias_into_the_registry(
            self, fake_registry, tmp_path, monkeypatch):
        """A dest_dir pull is explicitly register=False. Aliasing the dup (the
        other half of the dup branch) would write localm's registry for a file
        the caller asked to keep out of it, and STILL not deliver it."""
        store, models_dir = fake_registry
        already = models_dir / "clip_l.safetensors"
        already.write_bytes(b"fake-model-bytes")
        store["clip-local"] = {"path": str(already), "source": "local",
                               "sha256": _DIGEST, "model_type": "text-encoder"}
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo, fn: _DIGEST)
        _wire_fake_download(monkeypatch)
        dest_dir = tmp_path / "comfyui-models" / "clip"

        ok = mm._pull_gguf_file(
            "comfyanonymous/flux_text_encoders:clip_l.safetensors", None,
            model_type="text-encoder", dest_dir=dest_dir, register=False)

        assert ok is True
        assert (dest_dir / "clip_l.safetensors").is_file()
        assert list(store) == ["clip-local"], "no new registry entry/alias"

    def test_a_normal_pull_still_skips_a_known_duplicate(
            self, fake_registry, tmp_path, monkeypatch):
        """Negative case: with NO dest_dir the destination IS localm's models dir,
        so an already-registered identical file is a real duplicate and the
        no-TTY skip must still apply. The fix must be scoped to dest_dir, not
        disable dedup for every pull."""
        store, models_dir = fake_registry
        already = models_dir / "other.gguf"
        already.write_bytes(b"fake-model-bytes")
        store["already-have"] = {"path": str(already), "source": "local",
                                 "sha256": _DIGEST, "model_type": "llm"}
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo, fn: _DIGEST)

        downloaded = []

        def _fake_download(repo_id, filename, local_dir, **kw):
            downloaded.append(filename)
            p = Path(local_dir) / filename
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"fake-model-bytes")
            return str(p)

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
        monkeypatch.setattr("requests.head", lambda *a, **k: MagicMock(
            headers={"content-length": "16"}))

        ok = mm._pull_gguf_file("city96/FLUX.1-dev-gguf:brand-new.gguf", None,
                                model_type="llm")

        assert ok is True
        assert downloaded == [], (
            "a duplicate destined for MODELS_DIR is still skipped without a TTY")
