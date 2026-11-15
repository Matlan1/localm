# SPDX-License-Identifier: AGPL-3.0-or-later
"""Text encoders of image, video and music models, and the parts of a music
pipeline, register with their own types instead of ``embedding`` / ``unknown``.

Headers and configs are built by the tests; one test reads a real encoder GGUF
named by LOCALM_TEST_UMT5_GGUF, read-only, when that is set."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

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

    def test_a_real_umt5_encoder(self):
        real = Path(os.environ.get("LOCALM_TEST_UMT5_GGUF") or "")
        if not real.is_file():
            pytest.skip("LOCALM_TEST_UMT5_GGUF names no UMT5 encoder GGUF")
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

    def test_an_unreadable_header_leaves_an_embedding_entry_alone(self, home):
        f = home / "models" / "torn.gguf"
        f.write_bytes(b"not a gguf header")
        _put(home / "registry.json", "torn", path=str(f), source="local",
             model_type="embedding", architecture="t5encoder")
        assert mm.sync_models_dir(backfill_mmproj=False).retyped == 0
        assert _entry(home, "torn")["model_type"] == "embedding"

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


# ------------------------------------------------- a type given to a pull

class _UrlResponse:
    status_code = 200

    def __init__(self, content: bytes):
        self.content = content
        self.headers = {"content-length": str(len(content))}

    def raise_for_status(self):
        pass

    def iter_content(self, n):
        yield self.content


def _serve_url(monkeypatch, content: bytes) -> None:
    monkeypatch.setattr("socket.getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    monkeypatch.setattr("localm.netpolicy.pinned_request",
                        lambda method, url, **k: _UrlResponse(content))
    monkeypatch.setattr(mm, "_check_disk_space", lambda d, b: True)


def _serve_snapshot(monkeypatch, config: dict) -> None:
    import huggingface_hub
    raw = json.dumps(config).encode("utf-8")
    siblings = [SimpleNamespace(rfilename="config.json", size=len(raw), lfs=None)]

    class _Api:
        def __init__(self, *a, **kw):
            pass

        def model_info(self, repo_id, files_metadata=True):
            return SimpleNamespace(siblings=siblings, sha="a" * 40)

    def _download(repo_id, local_dir, **kw):
        d = Path(local_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_bytes(raw)
        return str(d)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _download)
    monkeypatch.setattr("localm.model_manager.pull._hf_endpoint_refusal", lambda: None)


class TestATypeGivenToAPull:
    def _encoder_bytes(self, tmp_path) -> bytes:
        return _gguf(tmp_path / "src.gguf", "t5encoder").read_bytes()

    def test_a_url_pull_keeps_an_explicit_type_through_the_sync(self, home, tmp_path, monkeypatch):
        _serve_url(monkeypatch, self._encoder_bytes(tmp_path))
        assert mm.pull_model("https://x.test/enc.gguf", name="enc", model_type="embedding") is True
        assert _entry(home, "enc")["type_user_set"] is True
        assert mm.sync_models_dir(backfill_mmproj=False).retyped == 0
        assert _entry(home, "enc")["model_type"] == "embedding"

    def test_a_url_pull_with_a_hint_is_corrected_by_the_sync(self, home, tmp_path, monkeypatch):
        _serve_url(monkeypatch, self._encoder_bytes(tmp_path))
        assert mm.pull_model("https://x.test/enc.gguf", name="enc", model_type="embedding",
                             type_is_hint=True) is True
        assert "type_user_set" not in _entry(home, "enc")
        assert _entry(home, "enc")["model_type"] == "embedding"
        mm.sync_models_dir(backfill_mmproj=False)
        assert _entry(home, "enc")["model_type"] == "text-encoder"

    def test_a_url_pull_without_a_type_records_no_choice(self, home, tmp_path, monkeypatch):
        _serve_url(monkeypatch, self._encoder_bytes(tmp_path))
        assert mm.pull_model("https://x.test/enc.gguf", name="enc") is True
        assert "type_user_set" not in _entry(home, "enc")

    def test_a_snapshot_pull_keeps_an_explicit_type_through_the_sync(self, home, monkeypatch):
        _serve_snapshot(monkeypatch, ACE_TRANSFORMER)
        assert mm.pull_model("o/ace", name="ace", model_type="unknown") is True
        assert _entry(home, "ace")["type_user_set"] is True
        assert _entry(home, "ace")["model_type"] == "unknown"
        assert mm.sync_models_dir(backfill_mmproj=False).retyped == 0
        assert _entry(home, "ace")["model_type"] == "unknown"

    def test_a_snapshot_pull_with_a_hint_is_corrected_by_the_sync(self, home, monkeypatch):
        _serve_snapshot(monkeypatch, ACE_TRANSFORMER)
        assert mm.pull_model("o/ace", name="ace", model_type="unknown", type_is_hint=True) is True
        assert "type_user_set" not in _entry(home, "ace")
        mm.sync_models_dir(backfill_mmproj=False)
        assert _entry(home, "ace")["model_type"] == "diffusion-unet"

    def test_a_snapshot_already_on_disk_records_the_explicit_type(self, home, monkeypatch):
        _serve_snapshot(monkeypatch, ACE_TRANSFORMER)
        assert mm.pull_model("o/ace", name="ace", model_type="unknown", type_is_hint=True) is True
        assert "type_user_set" not in _entry(home, "ace")
        assert mm.pull_model("o/ace", name="ace", model_type="unknown") is True
        assert _entry(home, "ace")["type_user_set"] is True

    def test_a_url_pull_of_a_registered_file_takes_the_explicit_type(
            self, home, tmp_path, monkeypatch):
        _serve_url(monkeypatch, self._encoder_bytes(tmp_path))
        assert mm.pull_model("https://x.test/enc.gguf", name="enc") is True
        assert _entry(home, "enc")["model_type"] != "embedding"
        assert mm.pull_model("https://x.test/enc.gguf", name="enc", model_type="embedding") is True
        assert _entry(home, "enc")["model_type"] == "embedding"
        assert _entry(home, "enc")["type_user_set"] is True

    def test_a_hint_on_a_registered_file_changes_nothing(self, home, tmp_path, monkeypatch):
        _serve_url(monkeypatch, self._encoder_bytes(tmp_path))
        assert mm.pull_model("https://x.test/enc.gguf", name="enc") is True
        before = _entry(home, "enc")
        assert before["model_type"] != "embedding"
        assert mm.pull_model("https://x.test/enc.gguf", name="enc", model_type="embedding",
                             type_is_hint=True) is True
        assert _entry(home, "enc")["model_type"] == before["model_type"]
        assert "type_user_set" not in _entry(home, "enc")

    def test_a_moved_duplicate_keeps_the_explicit_type(self, home, tmp_path):
        f = _gguf(tmp_path / "te.gguf", "t5encoder")
        assert mm.add_local(str(f), name="first", on_duplicate="register") is True
        assert mm.add_local(str(f), name="second", on_duplicate="move",
                            model_type="embedding") is True
        entry = _entry(home, "second")
        assert entry["model_type"] == "embedding" and entry["type_user_set"] is True

    def test_a_single_file_pull_with_a_hint_records_no_choice(self, monkeypatch):
        seen = {}

        def fake(spec, name, **kw):
            seen.update(kw)
            return True

        monkeypatch.setattr(mm, "_pull_gguf_file", fake)
        monkeypatch.setattr("localm.model_manager.pull._hf_endpoint_refusal", lambda: None)
        assert mm.pull_model("o/r:f.gguf", model_type="embedding", type_is_hint=True) is True
        assert seen["type_user_set"] is False
        assert mm.pull_model("o/r:f.gguf", model_type="embedding") is True
        assert seen["type_user_set"] is True
        assert mm.pull_model("o/r:f.gguf") is True
        assert seen["type_user_set"] is False

    def test_the_cli_hint_flag_reaches_the_pull(self, monkeypatch):
        from click.testing import CliRunner

        import localm.cli as cli_pkg
        import localm.cli.models as cli_models
        seen = {}

        def fake(spec, name=None, **kw):
            seen.update(kw)
            return True

        monkeypatch.setattr(cli_models, "pull_model", fake)
        runner = CliRunner()
        assert runner.invoke(cli_pkg.main, ["pull", "--type", "embedding", "--type-hint",
                                            "--", "o/r:f.gguf"]).exit_code == 0
        assert seen["type_is_hint"] is True
        assert runner.invoke(cli_pkg.main, ["pull", "--type", "embedding",
                                            "--", "o/r:f.gguf"]).exit_code == 0
        assert seen["type_is_hint"] is False


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


class TestChoosingATextEncoderForEmbeddings:
    OWNER_KEY = "owner-key-text-encoder-choice"

    def test_the_job_says_what_the_model_is(self, home, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        from localm.inference.http_server import create_app
        from localm.plugins.engine import PluginManager
        f = _gguf(tmp_path / "te.gguf", "t5encoder")
        _put(home / "registry.json", "my-encoder", path=str(f), source="local",
             model_type="text-encoder", architecture="t5encoder")
        monkeypatch.setenv("LOCALM_API_KEY", self.OWNER_KEY)
        app = create_app(None)
        PluginManager(app, external_root=tmp_path / "noplugins").install("rag")
        headers = {"Authorization": f"Bearer {self.OWNER_KEY}"}
        with TestClient(app) as c:
            r = c.post("/api/rag/embedding", headers=headers,
                       json={"model": "my-encoder", "confirm": True})
        assert r.status_code == 400, r.text
        assert "registered as a text-encoder model, not an embedding model" in r.text
        assert "not a known embedding key" not in r.text
