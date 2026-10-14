# SPDX-License-Identifier: AGPL-3.0-or-later
"""The starter models (`localm pull llama3.2-1b` and its siblings) are pulled
with the sha256 localm pins for each file: by alias, by the spec the GUI picker
fills in, and never over a digest the user typed."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from localm import model_manager as mm

ALIAS = "llama3.2-1b"
SPEC = mm.MODEL_SHORTCUTS[ALIAS]
FILE = SPEC.split(":", 1)[1]
BODY = b"starter-model-bytes" * 40


@pytest.fixture()
def pulled(monkeypatch):
    """Record what pull_model hands the single-file download."""
    seen = SimpleNamespace(calls=[])

    def fake(spec, name, expected_sha256=None, **kw):
        seen.calls.append((spec, expected_sha256, kw.get("pinned_starter")))
        return True

    monkeypatch.setattr(mm, "_pull_gguf_file", fake)
    return seen


def test_every_starter_model_has_a_pinned_sha256():
    assert set(mm.registry._STARTER_SHA256) == set(mm.MODEL_SHORTCUTS.values())
    for digest in mm.registry._STARTER_SHA256.values():
        assert len(digest) == 64 and digest == digest.lower() and int(digest, 16) >= 0
    assert mm.starter_model_sha256("owner/repo:other.gguf") is None


def test_pulling_a_starter_by_alias_passes_its_pin(pulled):
    assert mm.pull_model(ALIAS) is True
    assert pulled.calls == [(SPEC, mm.starter_model_sha256(SPEC), True)]


def test_the_spec_the_gui_picker_fills_in_is_pinned_too(pulled):
    assert mm.pull_model(SPEC) is True
    assert pulled.calls == [(SPEC, mm.starter_model_sha256(SPEC), True)]


def test_a_digest_the_user_typed_is_never_replaced(pulled):
    assert mm.pull_model(ALIAS, expected_sha256="ab" * 32) is True
    assert pulled.calls == [(SPEC, "ab" * 32, False)]


def test_a_spec_that_is_not_a_starter_is_not_pinned(pulled):
    assert mm.pull_model("owner/repo:other.gguf") is True
    assert pulled.calls == [("owner/repo:other.gguf", None, False)]


class TestAgainstTheHub:
    @pytest.fixture()
    def hub(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
        monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
        monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
        monkeypatch.setitem(mm.registry._STARTER_SHA256, SPEC,
                            hashlib.sha256(BODY).hexdigest())
        state = SimpleNamespace(served=BODY, published=None, downloads=[], models_dir=models_dir)

        def download(repo_id, filename, local_dir, **kw):
            state.downloads.append(filename)
            p = Path(local_dir) / filename
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(state.served)
            return str(p)

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo, fn, **kw: state.published)
        monkeypatch.setattr("localm.model_manager.pull._policy_head", lambda *a, **k: MagicMock(
            headers={"content-length": "1"}))
        return state

    def _pull(self, hub):
        dest = hub.models_dir.parent / "out"
        return mm.pull_model(ALIAS, dest_dir=dest, register=False), dest

    def test_the_pinned_file_is_accepted(self, hub):
        ok, dest = self._pull(hub)
        assert ok is True and (dest / FILE).read_bytes() == BODY

    def test_bytes_that_are_not_the_pinned_file_are_deleted_when_the_hub_publishes_no_digest(
            self, hub, capsys):
        hub.served = b"a different upload" * 40
        ok, dest = self._pull(hub)
        assert ok is False and not (dest / FILE).exists()
        assert "SHA256 mismatch" in capsys.readouterr().out

    def test_a_hub_that_now_serves_another_file_refuses_before_downloading(self, hub, capsys):
        hub.published = "cd" * 32
        ok, _dest = self._pull(hub)
        out = " ".join(capsys.readouterr().out.split())
        assert ok is False and hub.downloads == []
        assert "has changed on Hugging Face" in out and f"--sha256 {'cd' * 32}" in out

    def test_a_file_already_on_disk_that_is_not_the_pinned_one_is_refused(self, hub, capsys):
        dest = hub.models_dir.parent / "out"
        dest.mkdir()
        (dest / FILE).write_bytes(b"older upload")
        ok, _ = self._pull(hub)
        out = " ".join(capsys.readouterr().out.split())
        assert ok is False and hub.downloads == []
        assert "pins for this starter model" in out
