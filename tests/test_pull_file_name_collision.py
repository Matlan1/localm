# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two Hugging Face repos can each hold a file with the same name and different
bytes, while the models folder keeps one slot per file name. Pulling the second
leaves the first untouched and says which registered model holds the name,
instead of calling the file damaged. The hub is replaced by fakes; the pull, the
hashing and the registry are the real code."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from localm import model_manager as mm
from localm.model_manager import pull as pull_mod

FIRST = b"first-repo-bytes-" * 64
SECOND = b"second-repo-bytes-" * 64
NAME = "shared.safetensors"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture()
def hub(tmp_path, monkeypatch):
    """A fake hub: ``files`` maps ``(repo, file)`` to the bytes served and
    published for it."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    state = SimpleNamespace(files={}, models_dir=models_dir)

    def fake_sha(repo, filename, token=None):
        data = state.files.get((repo, filename))
        return pull_mod._PublishedDigest(_sha(data), None) if data else None

    def fake_download(repo_id, filename, local_dir, revision=None, **kw):
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(state.files[(repo_id, filename)])
        return str(p)

    def fake_head(url, allow_redirects=None, timeout=None):
        h = MagicMock()
        h.headers = {"content-length": "1"}
        return h

    import huggingface_hub
    monkeypatch.setattr(mm, "_hf_file_sha256", fake_sha)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr("requests.head", fake_head)
    return state


def _pull(spec, name=None):
    return mm._pull_gguf_file(spec, name, model_type="vae")


def _text(capsys) -> str:
    return " ".join(capsys.readouterr().out.split())


class TestSecondRepoSharingAFileName:
    def _first_pulled(self, hub):
        hub.files[("a/vae", NAME)] = FIRST
        hub.files[("b/vae-fix", NAME)] = SECOND
        assert _pull("a/vae:" + NAME) is True
        assert (hub.models_dir / NAME).read_bytes() == FIRST

    def test_the_first_file_is_kept_and_the_pull_is_refused(self, hub):
        self._first_pulled(hub)
        assert _pull("b/vae-fix:" + NAME, "vae_fix") is False
        assert (hub.models_dir / NAME).read_bytes() == FIRST
        assert "vae_fix" not in mm.load_registry()

    def test_the_message_names_the_owner_and_does_not_call_it_damaged(self, hub, capsys):
        self._first_pulled(hub)
        capsys.readouterr()
        _pull("b/vae-fix:" + NAME, "vae_fix")
        out = _text(capsys)
        assert "File name already in use" in out
        assert f"'{NAME}' (hf:a/vae)" in out
        assert f"localm rm {NAME}" in out
        for blamed in ("damaged", "incomplete", "SHA256 mismatch"):
            assert blamed not in out

    def test_a_file_registered_from_the_same_repo_still_reads_as_damaged(self, hub, capsys):
        self._first_pulled(hub)
        (hub.models_dir / NAME).write_bytes(b"truncated")
        capsys.readouterr()
        assert _pull("a/vae:" + NAME) is False
        out = _text(capsys)
        assert "damaged" in out and "File name already in use" not in out

    def test_the_same_file_pulled_again_still_succeeds(self, hub):
        self._first_pulled(hub)
        assert _pull("a/vae:" + NAME) is True
