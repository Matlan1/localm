# SPDX-License-Identifier: AGPL-3.0-or-later
"""A text-to-speech GGUF is a recognised registry kind, 'tts': typed from its own
header, attached to its mmproj, kept out of every chat path, and listed where the
other kinds are."""

import re
from pathlib import Path

import pytest

from localm.model_manager import gguf as g
from localm.model_manager import registry as reg_mod
from localm.model_manager.registry import (
    MODEL_TYPES, is_auto_chat_eligible, is_llm)
from tests._gguf_builder import build_gguf


@pytest.fixture
def home(tmp_path, monkeypatch):
    import localm.config as cfg
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


class TestTyping:
    def test_qwen3tts_is_typed_tts(self):
        assert g.gguf_non_chat_model_type("qwen3tts") == "tts"
        assert g.gguf_is_tts_architecture("qwen3tts") is True

    def test_pockettts_stays_unknown_and_refused(self):
        assert g.gguf_non_chat_model_type("pockettts") == "unknown"
        assert g.gguf_is_tts_architecture("pockettts") is False
        refusal = g.gguf_chat_refusal("pockettts")
        assert refusal is not None and "cannot chat with it" in refusal

    def test_a_chat_architecture_is_untouched(self):
        assert g.gguf_non_chat_model_type("llama") is None
        assert g.gguf_chat_refusal("llama") is None

    def test_chatting_with_a_tts_model_is_refused_and_points_at_speech(self):
        refusal = g.gguf_chat_refusal("qwen3tts")
        assert refusal is not None
        assert "not a chat model" in refusal
        assert "localm speak" in refusal and "/v1/audio/speech" in refusal

    def test_a_local_file_with_a_qwen3tts_header_is_detected_as_tts(self, tmp_path):
        f = tmp_path / "Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf"
        f.write_bytes(build_gguf("qwen3tts"))
        kind, meta = reg_mod._detect_local_model_type(f, is_gguf=True, is_hf=False)
        assert kind == "tts" and meta["architecture"] == "qwen3tts"

    def test_the_tts_mmproj_is_still_typed_mmproj(self, tmp_path):
        f = tmp_path / "mmproj-Qwen3-TTS.gguf"
        f.write_bytes(build_gguf("clip", [("general.type", 8, "mmproj")]))
        kind, _meta = reg_mod._detect_local_model_type(f, is_gguf=True, is_hf=False)
        assert kind == "mmproj"


class TestKindPredicates:
    def test_tts_is_a_model_type(self):
        assert "tts" in MODEL_TYPES

    def test_a_tts_entry_is_not_an_llm_and_never_auto_chat(self):
        entry = {"path": "x.gguf", "model_type": "tts"}
        assert is_llm(entry) is False
        assert is_auto_chat_eligible(entry) is False

    def test_an_llm_entry_is_still_auto_chat(self):
        assert is_auto_chat_eligible({"path": "x.gguf", "model_type": "llm"}) is True

    def test_capability_routing_never_targets_a_tts_model(self):
        from localm.inference import capability_routing as cr
        assert cr._is_routing_target({"model_type": "tts", "path": "x"}) is False


class TestProjectorAttachment:
    def test_a_tts_pull_fetches_the_same_repo_mmproj(self, tmp_path, monkeypatch):
        from localm.model_manager import pull
        seen = []
        monkeypatch.setattr(pull, "_maybe_fetch_repo_mmproj",
                            lambda repo, fn, base: (seen.append((repo, fn)), tmp_path / "mm.gguf")[1])
        got = pull._mmproj_for_registration("tts", "o/r", "m.gguf", tmp_path, None, None)
        assert got == tmp_path / "mm.gguf" and seen == [("o/r", "m.gguf")]

    @pytest.mark.parametrize("kind", ["unknown", "embedding", "diffusion-unet"])
    def test_other_non_chat_kinds_fetch_no_mmproj(self, tmp_path, monkeypatch, kind):
        from localm.model_manager import pull
        monkeypatch.setattr(pull, "_maybe_fetch_repo_mmproj",
                            lambda *a: pytest.fail("must not fetch"))
        assert pull._mmproj_for_registration(kind, "o/r", "m.gguf", tmp_path, None, None) is None

    def test_the_pull_note_for_a_tts_model_names_localm_speak(self):
        from localm.model_manager.pull import _non_chat_detection_note
        assert "localm speak" in _non_chat_detection_note("qwen3tts")
        assert "cannot chat" in _non_chat_detection_note("pockettts")

    def test_localm_add_records_the_sibling_mmproj_on_a_tts_model(self, home, monkeypatch):
        from localm.model_manager import registry
        folder = home / "src"
        folder.mkdir()
        model = folder / "Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf"
        model.write_bytes(build_gguf("qwen3tts"))
        mmproj = folder / "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"
        mmproj.write_bytes(build_gguf("clip", [("general.type", 8, "mmproj")]))
        assert registry.add_local(str(model), "voice-model", on_duplicate="skip",
                                  no_hash=True) is True
        from localm.config import load_registry
        entry = load_registry()["voice-model"]
        assert entry["model_type"] == "tts"
        assert Path(registry.get_model_mmproj("voice-model")).name == mmproj.name


    def test_a_copied_tts_model_records_its_mmproj_on_the_entry(self, home, monkeypatch):
        import localm.model_manager as mm
        from localm.config import load_registry
        from localm.model_manager import registry
        models = home / "models"
        models.mkdir(exist_ok=True)
        monkeypatch.setattr(mm, "MODELS_DIR", models)
        monkeypatch.setattr(mm, "ensure_dirs", lambda: models.mkdir(parents=True, exist_ok=True))
        folder = home / "outside"
        folder.mkdir()
        model = folder / "Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf"
        model.write_bytes(build_gguf("qwen3tts"))
        mmproj = folder / "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"
        mmproj.write_bytes(build_gguf("clip", [("general.type", 8, "mmproj")]))
        assert registry.add_local(str(model), "copied", on_duplicate="skip",
                                  no_hash=True, store="copy") is True
        entry = load_registry()["copied"]
        assert entry["model_type"] == "tts"
        assert Path(entry["mmproj"]).name == mmproj.name
        assert Path(entry["mmproj"]).parent == Path(entry["path"]).parent


class TestSurfaces:
    def test_the_cli_type_choices_include_tts(self):
        from click.testing import CliRunner

        from localm.cli import main
        out = CliRunner().invoke(main, ["set-type", "--help"]).output
        assert "tts" in out

    def test_the_gui_type_list_mirrors_model_types(self):
        js = (Path(__file__).resolve().parents[1] / "localm" / "plugins" / "gui"
              / "static" / "pages" / "models.js").read_text(encoding="utf-8")
        m = re.search(r"const MODEL_TYPE_OPTIONS =\s*\[([^\]]*)\]", js)
        assert m, "MODEL_TYPE_OPTIONS not found in models.js"
        assert set(re.findall(r'"([^"]+)"', m.group(1))) == set(MODEL_TYPES)

    def test_the_reranker_names_a_tts_model_as_what_it_is(self):
        from localm.inference.reranker import _KIND_WORDS
        assert _KIND_WORDS["tts"] == "a text-to-speech model"
