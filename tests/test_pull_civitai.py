# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for _pull_civitai_file() and pull_model()'s civitai: spec dispatch:
dest_dir routing to the ComfyUI-subfoldered tree, eager registration with the
right source/model_type, the SSRF-guarded redirect treatment, and SHA256
verification. No real network - CivitAISource.resolve_download and the HTTP
layer are mocked."""

import hashlib
from pathlib import Path

import pytest

from localm import model_manager as mm
from localm.model_manager.pull import _pull_civitai_file
from localm.model_manager.sources import ResolvedDownload


@pytest.fixture(autouse=True)
def _online(monkeypatch):
    monkeypatch.setenv("LOCALM_NET_MODE", "ask")


@pytest.fixture()
def fake_registry(tmp_path, monkeypatch):
    """Mirrors test_pull_comfy_dest_dir.py's fixture of the same name."""
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


class _FakeStreamResponse:
    def __init__(self, body: bytes, status_code: int = 200, headers=None):
        self.body = body
        self.status_code = status_code
        self.headers = headers or {"content-length": str(len(body))}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(response=self)

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i:i + chunk_size]


def _resolved(**over) -> ResolvedDownload:
    body = b"fake-lora-bytes"
    base = dict(
        url="https://civitai.com/api/download/models/135867?fileId=99264",
        filename="add-detail-xl.safetensors",
        source_tag="civitai:135867",
        model_type="lora",
        size_bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
        comfy_subfolder="loras",
    )
    base.update(over)
    return ResolvedDownload(**base)


def _wire_happy_path(monkeypatch, resolved: ResolvedDownload, dest_dir: Path,
                      body: bytes = b"fake-lora-bytes"):
    monkeypatch.setattr(
        "localm.model_manager.sources.CivitAISource.resolve_download",
        lambda self, ref, file, **kw: resolved)
    monkeypatch.setattr(
        "localm.media.managed_comfy.comfy_models_dest_dir",
        lambda subfolder, cfg=None, plugin=None: dest_dir)
    monkeypatch.setattr(
        "localm.model_manager.pull._ssrf_resolve_final_url", lambda url: url)
    monkeypatch.setattr(
        "localm.netpolicy.pinned_request",
        lambda method, url, **kw: _FakeStreamResponse(body))


class TestPullCivitaiFile:
    def test_lands_in_the_resolved_comfy_subfolder_and_registers(
            self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "loras"
        resolved = _resolved()
        _wire_happy_path(monkeypatch, resolved, dest_dir)

        ok = _pull_civitai_file("135867", None)

        assert ok is True
        landed = dest_dir / "add-detail-xl.safetensors"
        assert landed.is_file()
        assert landed.read_bytes() == b"fake-lora-bytes"
        assert not (fake_registry[1] / "add-detail-xl.safetensors").exists()
        entry = store["add-detail-xl"]
        assert entry["source"] == "civitai:135867"
        assert entry["model_type"] == "lora"
        assert entry["sha256"] == resolved.sha256

    def test_explicit_type_overrides_the_registry_label_not_the_subfolder(
            self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "loras"
        resolved = _resolved(model_type="lora", comfy_subfolder="loras")
        _wire_happy_path(monkeypatch, resolved, dest_dir)

        ok = _pull_civitai_file("135867", None, model_type="unknown")

        assert ok is True
        assert (dest_dir / "add-detail-xl.safetensors").is_file()
        assert store["add-detail-xl"]["model_type"] == "unknown"

    def test_register_false_skips_the_registry(self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "loras"
        _wire_happy_path(monkeypatch, _resolved(), dest_dir)

        ok = _pull_civitai_file("135867", None, register=False)

        assert ok is True
        assert (dest_dir / "add-detail-xl.safetensors").is_file()
        assert store == {}

    def test_already_downloaded_short_circuits_without_a_fetch(
            self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "loras"
        dest_dir.mkdir(parents=True)
        (dest_dir / "add-detail-xl.safetensors").write_bytes(b"fake-lora-bytes")
        resolved = _resolved()
        monkeypatch.setattr(
            "localm.model_manager.sources.CivitAISource.resolve_download",
            lambda self, ref, file, **kw: resolved)
        monkeypatch.setattr(
            "localm.media.managed_comfy.comfy_models_dest_dir",
            lambda subfolder, cfg=None, plugin=None: dest_dir)

        def _forbidden(*a, **kw):
            raise AssertionError("must not fetch when the file already exists")

        monkeypatch.setattr("localm.model_manager.pull._ssrf_resolve_final_url", _forbidden)
        monkeypatch.setattr("localm.netpolicy.pinned_request", _forbidden)

        ok = _pull_civitai_file("135867", None)

        assert ok is True
        assert store["add-detail-xl"]["source"] == "civitai:135867"

    def test_sha256_mismatch_deletes_the_file_and_refuses(
            self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "loras"
        resolved = _resolved(sha256="0" * 64)   # wrong digest for the fake body
        _wire_happy_path(monkeypatch, resolved, dest_dir)

        ok = _pull_civitai_file("135867", None)

        assert ok is False
        assert not (dest_dir / "add-detail-xl.safetensors").exists()
        assert store == {}

    def test_no_comfy_folder_configured_refuses_cleanly(
            self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        monkeypatch.setattr(
            "localm.model_manager.sources.CivitAISource.resolve_download",
            lambda self, ref, file, **kw: _resolved())
        monkeypatch.setattr(
            "localm.media.managed_comfy.comfy_models_dest_dir",
            lambda subfolder, cfg=None, plugin=None: None)

        ok = _pull_civitai_file("135867", None)

        assert ok is False
        assert store == {}

    def test_resolve_download_error_is_reported_not_raised(
            self, fake_registry, monkeypatch):
        from localm.model_manager.sources import ModelSourceError

        def _refuse(self, ref, file, **kw):
            raise ModelSourceError("excluded")

        monkeypatch.setattr(
            "localm.model_manager.sources.CivitAISource.resolve_download", _refuse)

        assert _pull_civitai_file("135867", None) is False


class TestPullCivitaiFileSSRF:
    """The download redirect must get the SAME per-hop SSRF guard as
    _pull_url, never a trust-the-metadata shortcut."""

    def test_a_refused_redirect_chain_downloads_nothing(
            self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "loras"
        monkeypatch.setattr(
            "localm.model_manager.sources.CivitAISource.resolve_download",
            lambda self, ref, file, **kw: _resolved())
        monkeypatch.setattr(
            "localm.media.managed_comfy.comfy_models_dest_dir",
            lambda subfolder, cfg=None, plugin=None: dest_dir)

        from localm.netpolicy import NetworkPolicyError

        def _refuse(url):
            raise NetworkPolicyError("refused: private address")

        monkeypatch.setattr("localm.model_manager.pull._ssrf_resolve_final_url", _refuse)

        def _forbidden_get(*a, **kw):
            raise AssertionError("must not GET after the redirect resolver refused")

        monkeypatch.setattr("localm.netpolicy.pinned_request", _forbidden_get)

        ok = _pull_civitai_file("135867", None)

        assert ok is False
        assert not dest_dir.exists() or list(dest_dir.iterdir()) == []
        assert store == {}

    def test_real_ssrf_resolver_refuses_a_private_redirect_target(
            self, fake_registry, tmp_path, monkeypatch):
        """A URL that is itself private-IP shaped: even the second,
        immediately-before-connect check_url call alone (the same "revalidate
        right before the GET" pattern _pull_url_locked uses) would catch this
        one, so this proves the OVERALL guarantee - no private address is ever
        connected to - rather than isolating _ssrf_resolve_final_url on its
        own (see the redirect-chain variant below for that)."""
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "loras"
        monkeypatch.setenv("LOCALM_NET_MODE", "allow")   # isolate the IP-class check
        resolved = _resolved(url="http://127.0.0.1:9/whatever")
        monkeypatch.setattr(
            "localm.model_manager.sources.CivitAISource.resolve_download",
            lambda self, ref, file, **kw: resolved)
        monkeypatch.setattr(
            "localm.media.managed_comfy.comfy_models_dest_dir",
            lambda subfolder, cfg=None, plugin=None: dest_dir)

        ok = _pull_civitai_file("135867", None)

        assert ok is False
        assert not dest_dir.exists() or list(dest_dir.iterdir()) == []
        assert store == {}

    def test_real_ssrf_resolver_refuses_a_redirect_TO_a_private_target(
            self, fake_registry, tmp_path, monkeypatch):
        """The representative CivitAI threat model: the STARTING url is the
        legitimate public civitai.com download endpoint -
        exactly what resolve_download() returns for a real pull - and only the
        redirect it answers with points at a private address. This isolates
        _ssrf_resolve_final_url's own per-hop re-validation: the immediately-
        before-connect check_url in _pull_civitai_file never even runs, because
        the resolver itself must refuse before returning."""
        store, _ = fake_registry
        dest_dir = tmp_path / "comfyui-models" / "loras"
        monkeypatch.setenv("LOCALM_NET_MODE", "allow")   # isolate the IP-class check
        resolved = _resolved()   # a real https://civitai.com/... starting URL
        monkeypatch.setattr(
            "localm.model_manager.sources.CivitAISource.resolve_download",
            lambda self, ref, file, **kw: resolved)
        monkeypatch.setattr(
            "localm.media.managed_comfy.comfy_models_dest_dir",
            lambda subfolder, cfg=None, plugin=None: dest_dir)

        def _head_redirects_to_private(method, url, **kw):
            assert method == "HEAD", "only the resolver's own HEAD probe should ever fire here"
            resp = _FakeStreamResponse(b"", status_code=307,
                                       headers={"Location": "http://169.254.169.254/latest/meta-data/"})
            return resp

        monkeypatch.setattr("localm.netpolicy.pinned_request", _head_redirects_to_private)

        ok = _pull_civitai_file("135867", None)

        assert ok is False
        assert not dest_dir.exists() or list(dest_dir.iterdir()) == []
        assert store == {}


class TestPullModelCivitaiDispatch:
    def test_parses_version_and_file_id(self, monkeypatch):
        monkeypatch.setenv("LOCALM_NET_MODE", "ask")
        calls = []
        monkeypatch.setattr(
            mm, "_pull_civitai_file",
            lambda version_id, name, **kw: calls.append((version_id, kw)) or True)
        assert mm.pull_model("civitai:135867:99264") is True
        assert calls[0][0] == "135867"
        assert calls[0][1]["file_id"] == "99264"

    def test_parses_bare_version_with_no_file_id(self, monkeypatch):
        monkeypatch.setenv("LOCALM_NET_MODE", "ask")
        calls = []
        monkeypatch.setattr(
            mm, "_pull_civitai_file",
            lambda version_id, name, **kw: calls.append((version_id, kw)) or True)
        assert mm.pull_model("civitai:135867") is True
        assert calls[0][0] == "135867"
        assert calls[0][1]["file_id"] is None

    def test_refuses_an_explicit_dest_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_NET_MODE", "ask")
        called = []
        monkeypatch.setattr(mm, "_pull_civitai_file", lambda *a, **kw: called.append(1) or True)
        assert mm.pull_model("civitai:135867", dest_dir=tmp_path) is False
        assert called == []

    def test_net_mode_off_refuses_before_dispatch(self, monkeypatch):
        import localm.netpolicy as netpolicy
        monkeypatch.setattr(netpolicy, "network_mode", lambda: "off")
        called = []
        monkeypatch.setattr(mm, "_pull_civitai_file", lambda *a, **kw: called.append(1) or True)
        assert mm.pull_model("civitai:135867") is False
        assert called == []

    def test_mmproj_spec_is_ignored_not_crashed_on(self, monkeypatch):
        monkeypatch.setenv("LOCALM_NET_MODE", "ask")
        monkeypatch.setattr(mm, "_pull_civitai_file", lambda *a, **kw: True)
        assert mm.pull_model("civitai:135867", mmproj_spec="org/repo:mmproj.gguf") is True
