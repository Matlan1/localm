# SPDX-License-Identifier: AGPL-3.0-or-later
"""A CLI `--mmproj` override never persisted to the registry, unlike every
other mmproj-discovery path (pull.py's auto-attach, the registry's own
recorded field). So vision_input_guidance()'s own "switch to it" suggestion
(`localm run <model>`) lost the projector again the moment --mmproj was left
off. persist_cli_mmproj() closes that, but ONLY once the caller has confirmed
the projector genuinely loaded (supports_images=True for this run) - it must
never record a broken association, or vision_capable_models() would list a
model whose vision does not actually work: a new false-positive surface in
the area PR #1073 just fixed.
"""

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from localm import model_manager as mm


@pytest.fixture()
def fake_registry(tmp_path, monkeypatch):
    """In-memory registry + temp MODELS_DIR (mirrors test_pull_mmproj_autofetch.py)."""
    store: dict = {}
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
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


# --------------------------------------------------------------------------- #
#  persist_cli_mmproj() - the registry-level decision, unit tested directly   #
# --------------------------------------------------------------------------- #

class TestPersistCliMmproj:
    def test_writes_when_nothing_recorded_yet(self, fake_registry, tmp_path):
        store, models_dir = fake_registry
        model_f = models_dir / "m.gguf"
        model_f.write_bytes(b"GGUF")
        proj_f = tmp_path / "proj.gguf"
        proj_f.write_bytes(b"GGUF")
        store["m"] = {"path": str(model_f), "source": "local", "model_type": "llm"}

        note = mm.persist_cli_mmproj("m", str(proj_f))

        assert note is not None and "recorded" in note.lower()
        assert store["m"]["mmproj"] == str(proj_f.resolve())

    def test_noop_when_exact_same_path_already_recorded(self, fake_registry, tmp_path):
        store, models_dir = fake_registry
        model_f = models_dir / "m.gguf"
        model_f.write_bytes(b"GGUF")
        proj_f = tmp_path / "proj.gguf"
        proj_f.write_bytes(b"GGUF")
        store["m"] = {"path": str(model_f), "source": "local", "model_type": "llm",
                      "mmproj": str(proj_f.resolve())}

        note = mm.persist_cli_mmproj("m", str(proj_f))

        assert note is None
        assert store["m"]["mmproj"] == str(proj_f.resolve())   # unchanged

    def test_does_not_overwrite_a_different_recorded_mmproj(self, fake_registry, tmp_path):
        # Hard-won rule: never silently override a user's prior explicit
        # choice. A one-off --mmproj may be a deliberate experiment.
        store, models_dir = fake_registry
        model_f = models_dir / "m.gguf"
        model_f.write_bytes(b"GGUF")
        old_proj = tmp_path / "old.gguf"
        old_proj.write_bytes(b"GGUF")
        new_proj = tmp_path / "new.gguf"
        new_proj.write_bytes(b"GGUF")
        store["m"] = {"path": str(model_f), "source": "local", "model_type": "llm",
                      "mmproj": str(old_proj.resolve())}

        note = mm.persist_cli_mmproj("m", str(new_proj))

        assert note is not None
        assert "not overwriting" in note.lower()
        assert str(old_proj.resolve()) in note
        assert store["m"]["mmproj"] == str(old_proj.resolve())   # untouched

    def test_unregistered_name_is_a_noop(self, fake_registry, tmp_path):
        store, _ = fake_registry
        proj_f = tmp_path / "proj.gguf"
        proj_f.write_bytes(b"GGUF")

        assert mm.persist_cli_mmproj("does-not-exist", str(proj_f)) is None
        assert store == {}

    def test_hf_directory_entry_is_a_noop(self, fake_registry, tmp_path):
        # mmproj means nothing for an HF-format directory - must never write it.
        store, models_dir = fake_registry
        hf_dir = models_dir / "hf-model"
        hf_dir.mkdir()
        proj_f = tmp_path / "proj.gguf"
        proj_f.write_bytes(b"GGUF")
        store["hf-model"] = {"path": str(hf_dir), "source": "local", "model_type": "llm"}

        note = mm.persist_cli_mmproj("hf-model", str(proj_f))

        assert note is None
        assert "mmproj" not in store["hf-model"]

    def test_standalone_mmproj_entry_is_a_noop(self, fake_registry, tmp_path):
        # A projector registered as its own entry is not a chat model to
        # attach a (second) projector to.
        store, models_dir = fake_registry
        proj_entry_f = models_dir / "some.gguf"
        proj_entry_f.write_bytes(b"GGUF")
        other_proj = tmp_path / "other.gguf"
        other_proj.write_bytes(b"GGUF")
        store["some-mmproj"] = {"path": str(proj_entry_f), "source": "local",
                                "model_type": "mmproj"}

        note = mm.persist_cli_mmproj("some-mmproj", str(other_proj))

        assert note is None
        assert "mmproj" not in store["some-mmproj"]


# --------------------------------------------------------------------------- #
#  Driven end-to-end through `localm run`'s real CLI entry point              #
# --------------------------------------------------------------------------- #

class TestCliRunPersistsConfirmedMmproj:
    def _invoke(self, model_name, mmproj_path, *, supports_images, monkeypatch):
        from localm.cli import chat as chat_mod
        engine_instance = MagicMock(name="EngineInstance")
        engine_instance.__enter__ = MagicMock(return_value=engine_instance)
        engine_instance.__exit__ = MagicMock(return_value=False)
        engine_instance._backend = MagicMock(supports_images=supports_images)
        engine_cls = MagicMock(name="Engine", return_value=engine_instance)

        monkeypatch.setattr("localm.inference.engine.Engine", engine_cls)
        monkeypatch.setattr("localm.instances.attach_target", lambda *a, **k: None)

        return CliRunner().invoke(
            chat_mod.run,
            [model_name, "--mmproj", str(mmproj_path), "--no-server", "-p", "hi"],
        )

    def test_confirmed_load_persists_the_override(self, fake_registry, tmp_path, monkeypatch):
        store, models_dir = fake_registry
        model_f = models_dir / "vision-model.gguf"
        model_f.write_bytes(b"GGUF")
        proj_f = tmp_path / "proj.gguf"
        proj_f.write_bytes(b"GGUF")
        store["vision-model"] = {"path": str(model_f), "source": "local",
                                 "model_type": "llm"}

        result = self._invoke("vision-model", proj_f, supports_images=True,
                              monkeypatch=monkeypatch)

        assert result.exit_code == 0, result.output
        assert store["vision-model"]["mmproj"] == str(proj_f.resolve())

    def test_failed_load_does_not_persist_a_broken_association(
            self, fake_registry, tmp_path, monkeypatch):
        # The critical negative case: a --mmproj that did NOT load successfully
        # (backend reports supports_images=False) must leave the registry
        # untouched - persisting it would make vision_capable_models() list a
        # model whose vision does not actually work.
        store, models_dir = fake_registry
        model_f = models_dir / "vision-model.gguf"
        model_f.write_bytes(b"GGUF")
        proj_f = tmp_path / "bad-proj.gguf"
        proj_f.write_bytes(b"GGUF")
        store["vision-model"] = {"path": str(model_f), "source": "local",
                                 "model_type": "llm"}

        result = self._invoke("vision-model", proj_f, supports_images=False,
                              monkeypatch=monkeypatch)

        assert result.exit_code == 0, result.output
        assert "mmproj" not in store["vision-model"]

    def test_unregistered_direct_path_run_does_not_persist(
            self, fake_registry, tmp_path, monkeypatch):
        # `localm run D:\models\foo.gguf --mmproj ...` - a direct path, not a
        # registry entry, so there is nothing to persist to.
        store, models_dir = fake_registry
        model_f = models_dir / "unregistered.gguf"
        model_f.write_bytes(b"GGUF")
        proj_f = tmp_path / "proj.gguf"
        proj_f.write_bytes(b"GGUF")

        result = self._invoke(str(model_f), proj_f, supports_images=True,
                              monkeypatch=monkeypatch)

        assert result.exit_code == 0, result.output
        assert store == {}
