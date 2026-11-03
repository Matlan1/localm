# SPDX-License-Identifier: AGPL-3.0-or-later
"""Text encoders of image, video and music models, and the parts of a music
pipeline, register with their own types instead of ``embedding`` / ``unknown``.

Headers and configs are built by the tests; one test reads a real encoder GGUF
from the central models folder, read-only, when it exists."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import localm.model_manager as mm
from localm.discover import classify_hf_metadata
from localm.model_manager.gguf import (GGUF_TEXT_ENCODER_ARCHITECTURES,
                                       gguf_embedding_signal, gguf_non_chat_model_type)
from localm.model_manager.media_parts import hf_media_part
from localm.model_manager.registry import (MODEL_TYPES, NON_CHAT_TYPES, UNLISTED_TYPES,
                                           _detect_local_model_type)
from tests.test_gguf_architecture_roles import _gguf, _kv_uint32
from tests.test_sdcpp_families import UMT5_GGUF, write_gguf


@pytest.fixture
def home(tmp_path, monkeypatch):
    import localm.config as cfg
    h = tmp_path / ".localm"
    (h / "models").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    monkeypatch.setattr(mm, "MODELS_DIR", h / "models")
    monkeypatch.setattr(mm, "REGISTRY_FILE", h / "registry.json")
    return h


def _put(reg_path: Path, name: str, **entry) -> None:
    reg = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else {}
    reg[name] = entry
    reg_path.write_text(json.dumps(reg), encoding="utf-8")


def _entry(home: Path, name: str) -> dict:
    return json.loads((home / "registry.json").read_text(encoding="utf-8"))[name]


# ----------------------------------------------------------------------- GGUF

class TestGgufTextEncoder:
    def test_a_t5_encoder_without_pooling_is_a_text_encoder(self, tmp_path):
        f = _gguf(tmp_path / "umt5.gguf", "t5encoder")
        assert gguf_embedding_signal(f) is False
        mtype, meta = _detect_local_model_type(f, is_gguf=True, is_hf=False)
        assert mtype == "text-encoder"
        assert meta["architecture"] == "t5encoder"

    def test_a_t5_encoder_with_a_pooling_type_is_a_retrieval_embedder(self, tmp_path):
        f = _gguf(tmp_path / "sentence-t5.gguf", "t5encoder",
                  [("t5encoder.pooling_type", _kv_uint32(1))])
        assert gguf_embedding_signal(f) is True
        assert _detect_local_model_type(f, is_gguf=True, is_hf=False)[0] == "embedding"

    def test_every_text_encoder_architecture_has_the_text_encoder_role(self):
        for arch in GGUF_TEXT_ENCODER_ARCHITECTURES:
            assert gguf_non_chat_model_type(arch) == "text-encoder"

    def test_real_umt5_encoder_from_the_central_folder(self):
        real = Path("D:/projects/localm/home/models/umt5-xxl-encoder-Q4_K_M.gguf")
        if not real.is_file():
            pytest.skip("the central models folder has no UMT5 encoder")
        assert _detect_local_model_type(real, is_gguf=True, is_hf=False)[0] == "text-encoder"


class TestHfSearchBadge:
    def test_a_t5_encoder_gguf_is_badged_text_encoder(self):
        assert classify_hf_metadata(None, None, ["gguf"], architecture="t5encoder") == "text-encoder"

    def test_a_t5_encoder_tagged_for_retrieval_stays_an_embedding(self):
        assert classify_hf_metadata("feature-extraction", None, ["gguf"],
                                    architecture="t5encoder") == "embedding"
        assert classify_hf_metadata(None, None, ["sentence-similarity"],
                                    architecture="t5encoder") == "embedding"

    def test_a_bert_gguf_is_still_an_embedding(self):
        assert classify_hf_metadata(None, None, [], architecture="bert") == "embedding"


# ---------------------------------------------------------- pipeline parts (HF)

ACE_TRANSFORMER = {"_class_name": "ACEStepTransformer2DModel", "in_channels": 8,
                   "text_embedding_dim": 768}
ACE_DCAE = {"_class_name": "AutoencoderDC", "in_channels": 2, "latent_channels": 8,
            "_name_or_path": "checkpoints/music_dcae_f8c8"}
ACE_VOCODER = {"_class_name": "ADaMoSHiFiGANV1", "sampling_rate": 44100}
UMT5_BASE = {"architectures": ["UMT5EncoderModel"], "model_type": "umt5", "d_model": 768}


def _folder(root: Path, name: str, config: dict) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (d / "diffusion_pytorch_model.safetensors").write_bytes(b"\0" * 16)
    return d


class TestHfMediaParts:
    @pytest.mark.parametrize("config,expected", [
        (ACE_TRANSFORMER, ("diffusion-unet", "ACE-Step")),
        (ACE_DCAE, ("vae", "ACE-Step")),
        (ACE_VOCODER, ("vocoder", "ACE-Step")),
        (UMT5_BASE, ("text-encoder", None)),
        ({"_class_name": "AutoencoderDC", "in_channels": 3}, ("vae", None)),
        ({"_class_name": "WanTransformer3DModel"}, ("diffusion-unet", None)),
        ({"_class_name": "UNet2DConditionModel"}, ("diffusion-unet", None)),
        ({"architectures": ["CLIPTextModel"]}, ("text-encoder", None)),
    ])
    def test_part_roles_come_from_the_config(self, tmp_path, config, expected):
        part = hf_media_part(_folder(tmp_path, "any-name-at-all", config))
        assert part is not None and (part.model_type, part.part_of) == expected

    def test_the_folder_name_decides_nothing(self, tmp_path):
        d = _folder(tmp_path, "music_vocoder", {"architectures": ["LlamaForCausalLM"]})
        assert hf_media_part(d) is None

    def test_an_unreadable_config_is_no_part(self, tmp_path):
        d = tmp_path / "broken"
        d.mkdir()
        (d / "config.json").write_text("{not json", encoding="utf-8")
        assert hf_media_part(d) is None

    def test_detection_types_each_part_and_keeps_a_sentence_transformer_an_embedding(self, tmp_path):
        for name, config in (("t", ACE_TRANSFORMER), ("v", ACE_VOCODER), ("u", UMT5_BASE)):
            d = _folder(tmp_path, name, config)
            mtype, meta = _detect_local_model_type(d, is_gguf=False, is_hf=True)
            assert mtype == hf_media_part(d).model_type
            assert meta["architecture"]
        st = _folder(tmp_path, "sentence-t5", UMT5_BASE)
        (st / "modules.json").write_text(json.dumps([
            {"type": "sentence_transformers.models.Transformer"},
            {"type": "sentence_transformers.models.Pooling"}]), encoding="utf-8")
        assert _detect_local_model_type(st, is_gguf=False, is_hf=True)[0] == "embedding"

    def test_new_types_are_registry_types_the_chat_engine_never_loads(self):
        for t in ("vocoder", "planner"):
            assert t in MODEL_TYPES and t in NON_CHAT_TYPES and t in UNLISTED_TYPES


# ----------------------------------------------------------- the registry sync

class TestSyncMigration:
    def test_an_embedding_entry_with_a_text_encoder_architecture_is_retyped_keeping_its_pairing(
            self, home):
        from localm.media.sdcpp import pairing
        f = write_gguf(home / "models" / "umt5-xxl-encoder-Q4_K_M.gguf", UMT5_GGUF, "t5encoder")
        _put(home / "registry.json", "umt5-xxl-encoder-Q4_K_M", path=str(f), source="local",
             model_type="embedding", architecture="t5encoder", reranker=False)
        result = mm.sync_models_dir(backfill_mmproj=False)
        entry = _entry(home, "umt5-xxl-encoder-Q4_K_M")
        assert entry["model_type"] == "text-encoder"
        assert entry["path"] == str(f) and entry["architecture"] == "t5encoder"
        assert result.retyped == 1 and result.changed is True
        names = {c.name: c.model_type for c in pairing.registry_candidates()}
        assert names == {"umt5-xxl-encoder-Q4_K_M": "text-encoder"}

    def test_a_pooled_t5_encoder_stays_an_embedding(self, home):
        f = _gguf(home / "models" / "st5.gguf", "t5encoder",
                  [("t5encoder.pooling_type", _kv_uint32(1))])
        _put(home / "registry.json", "st5", path=str(f), source="local",
             model_type="embedding", architecture="t5encoder", reranker=False)
        assert mm.sync_models_dir(backfill_mmproj=False).retyped == 0
        assert _entry(home, "st5")["model_type"] == "embedding"

    def test_a_type_the_user_set_by_hand_is_not_overridden(self, home):
        f = _gguf(home / "models" / "umt5.gguf", "t5encoder")
        _put(home / "registry.json", "umt5", path=str(f), source="local",
             model_type="llm", architecture="t5encoder")
        assert mm.set_model_type("umt5", "embedding") is True
        assert mm.sync_models_dir(backfill_mmproj=False).retyped == 0
        assert _entry(home, "umt5")["model_type"] == "embedding"

    def test_an_explicit_add_type_that_differs_from_detection_survives_the_sync(self, home, tmp_path):
        f = _gguf(tmp_path / "te.gguf", "t5encoder")
        assert mm.add_local(str(f), name="te", on_duplicate="register",
                            model_type="embedding") is True
        entry = _entry(home, "te")
        assert entry["model_type"] == "embedding" and entry["type_user_set"] is True
        assert mm.sync_models_dir(backfill_mmproj=False).retyped == 0
        assert _entry(home, "te")["model_type"] == "embedding"

    def test_adding_with_the_detected_type_records_no_user_choice(self, home, tmp_path):
        f = _gguf(tmp_path / "te.gguf", "t5encoder")
        assert mm.add_local(str(f), name="te", on_duplicate="register",
                            model_type="text-encoder") is True
        assert "type_user_set" not in _entry(home, "te")

    def test_the_sync_is_idempotent(self, home):
        f = _gguf(home / "models" / "umt5.gguf", "t5encoder")
        _put(home / "registry.json", "umt5", path=str(f), source="local",
             model_type="embedding", architecture="t5encoder")
        assert mm.sync_models_dir(backfill_mmproj=False).retyped == 1
        assert mm.sync_models_dir(backfill_mmproj=False).retyped == 0

    def test_an_unknown_acestep_planner_gguf_is_retyped(self, home):
        f = _gguf(home / "models" / "lm.gguf", "acestep-lm")
        _put(home / "registry.json", "lm", path=str(f), source="local",
             model_type="unknown", architecture="acestep-lm")
        mm.sync_models_dir(backfill_mmproj=False)
        assert _entry(home, "lm")["model_type"] == "planner"

    def test_the_four_acestep_folders_are_typed_and_tagged_as_one_model(self, home):
        models = home / "models"
        folders = {"ace_step_transformer": ACE_TRANSFORMER, "music_dcae_f8c8": ACE_DCAE,
                   "music_vocoder": ACE_VOCODER, "umt5-base": UMT5_BASE}
        for name, config in folders.items():
            d = _folder(models, name, config)
            _put(home / "registry.json", name, path=str(d), source="hf", model_type="unknown")
        mm.sync_models_dir(backfill_mmproj=False)
        got = {n: (_entry(home, n)["model_type"], _entry(home, n).get("part_of"))
               for n in folders}
        assert got == {"ace_step_transformer": ("diffusion-unet", "ACE-Step"),
                       "music_dcae_f8c8": ("vae", "ACE-Step"),
                       "music_vocoder": ("vocoder", "ACE-Step"),
                       "umt5-base": ("text-encoder", "ACE-Step")}
        assert _entry(home, "umt5-base")["architecture"] == "UMT5EncoderModel"

    def test_a_umt5_folder_with_no_acestep_transformer_is_not_tagged(self, home):
        d = _folder(home / "models", "umt5-base", UMT5_BASE)
        _put(home / "registry.json", "umt5-base", path=str(d), source="hf", model_type="unknown")
        mm.sync_models_dir(backfill_mmproj=False)
        entry = _entry(home, "umt5-base")
        assert entry["model_type"] == "text-encoder" and "part_of" not in entry

    def test_a_new_acestep_folder_registers_with_its_part_of(self, home):
        d = _folder(home / "models", "music_vocoder", ACE_VOCODER)
        mm.sync_models_dir(backfill_mmproj=False)
        entry = _entry(home, "music_vocoder")
        assert (entry["model_type"], entry["part_of"]) == ("vocoder", "ACE-Step")
        assert Path(entry["path"]) == d


# ------------------------------------------------------- the embedder refuses

class TestEmbedderNeverUsesATextEncoder:
    def _setup(self, monkeypatch, tmp_path, model_type):
        from localm.inference import embedder
        f = _gguf(tmp_path / "te.gguf", "t5encoder")
        reg = {"my-encoder": {"path": str(f), "source": "local", "model_type": model_type}}
        monkeypatch.setattr(embedder, "_current_spec", lambda: "my-encoder")
        monkeypatch.setattr("localm.config.load_registry", lambda: reg)
        monkeypatch.setattr("localm.model_manager.registry.get_model_info",
                            lambda name: (f, "local"))
        return embedder, f

    def test_a_registered_text_encoder_is_refused_with_a_reason(self, monkeypatch, tmp_path):
        embedder, _f = self._setup(monkeypatch, tmp_path, "text-encoder")
        assert embedder.resolve_embedding_model_path(allow_download=False) is None
        assert "text encoder" in (embedder.last_error() or "")

    def test_the_same_file_typed_embedding_still_resolves(self, monkeypatch, tmp_path):
        embedder, f = self._setup(monkeypatch, tmp_path, "embedding")
        assert embedder.resolve_embedding_model_path(allow_download=False) == str(f)
