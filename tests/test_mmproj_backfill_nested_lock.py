# SPDX-License-Identifier: AGPL-3.0-or-later
"""The mmproj backfill runs inside ``sync_models_dir``'s registry mutator. When
it finds a differing, origin-less projector recorded by another entry of the
same model file it repoints that entry; doing that through a second
``update_registry`` raised ``registry.json.lock is already held by this same
process`` and aborted the whole sync, again on every later sync (each one
downloading another projector copy). The repoint must apply to the registry
dict the sync is already mutating, and a projector a previous pass fetched must
be reused instead of downloaded again."""

import json

import pytest

from localm import model_manager as mm
from localm.model_manager import pull


@pytest.fixture()
def two_names_one_file(monkeypatch):
    from localm import config as cfg
    models = cfg.MODELS_DIR
    models.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mm, "MODELS_DIR", models)
    model = models / "model-Q4_K_M.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 64)
    plain = models / "mmproj-F16.gguf"
    plain.write_bytes(b"old projector bytes")
    base = {"source": "hf:owner/repo", "model_type": "llm", "architecture": "llama",
            "expert_count": 0, "tool_use": False, "context_length": 4096,
            "path": str(model.resolve())}
    reg = {"foo": dict(base), "bar": dict(base, mmproj=str(plain.resolve()))}
    cfg.REGISTRY_FILE.write_text(json.dumps(reg))

    downloads: list[str] = []

    def fake_download(repo_id, remote_name, dest, base_dir):
        downloads.append(dest.name)
        dest.write_bytes(b"new projector bytes")
        pull._record_mmproj_origin(dest, repo_id, remote_name, "ab" * 32)

    monkeypatch.setattr(pull, "_hf_repo_files",
                        lambda repo_id: ["model-Q4_K_M.gguf", "mmproj-F16.gguf"])
    monkeypatch.setattr(mm, "_hf_file_sha256", lambda *a, **k: "ab" * 32)
    monkeypatch.setattr(pull, "_download_mmproj_file", fake_download)
    monkeypatch.setattr(mm, "gguf_is_mmproj", lambda p: True)
    monkeypatch.setattr(pull, "_hf_endpoint_refusal", lambda: None)
    monkeypatch.setattr("localm.netpolicy.network_mode", lambda *a, **k: "on")
    # The digest the repo publishes is what the new copy hashes to; the old
    # plain copy never matches it.
    monkeypatch.setattr(
        pull, "_verify_digest",
        lambda p, purpose="": "ab" * 32 if p.read_bytes() == b"new projector bytes"
        else "cd" * 32)
    return downloads, cfg.REGISTRY_FILE


def _registry(path):
    return json.loads(path.read_text())


def test_sync_twice_attaches_the_projector_once_and_downloads_it_once(two_names_one_file):
    downloads, registry_file = two_names_one_file
    first = mm.sync_models_dir()
    second = mm.sync_models_dir()
    reg = _registry(registry_file)
    assert first.mmproj_backfilled == 1
    assert second.mmproj_backfilled == 0
    assert downloads == ["mmproj-F16-2.gguf"]
    assert reg["foo"]["mmproj"].endswith("mmproj-F16-2.gguf")
    assert reg["bar"]["mmproj"] == reg["foo"]["mmproj"]


def test_an_earlier_passs_numbered_projector_is_reused_not_downloaded_again(two_names_one_file):
    downloads, registry_file = two_names_one_file
    mm.sync_models_dir()
    reg = _registry(registry_file)
    del reg["foo"]["mmproj"]
    registry_file.write_text(json.dumps(reg))
    downloads.clear()
    result = mm.sync_models_dir()
    assert downloads == []
    assert result.mmproj_backfilled == 1
    assert _registry(registry_file)["foo"]["mmproj"].endswith("mmproj-F16-2.gguf")


def test_repoint_without_a_registry_dict_still_updates_the_registry(two_names_one_file):
    from localm import config as cfg
    models = cfg.MODELS_DIR
    new = models / "mmproj-F16-2.gguf"
    new.write_bytes(b"new projector bytes")
    pull._repoint_unrecorded_projector(
        models / "model-Q4_K_M.gguf", models / "mmproj-F16.gguf", new)
    assert _registry(cfg.REGISTRY_FILE)["bar"]["mmproj"].endswith("mmproj-F16-2.gguf")
