# SPDX-License-Identifier: AGPL-3.0-or-later
"""Files and folders localm cannot run are recognised and explained.

Each case builds the real bytes (or the verbatim ``config.json`` of a real
repository) the signature keys on, asserts the exact sentence, and has a control
that must NOT trigger: a normal GGUF v3, a plain AWQ folder, a HuggingFace folder
that also holds ``consolidated*`` files, a ``pytorch_model.bin`` of an HF folder.
"""

from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path

import pytest

from localm import model_manager as mm
from localm.inference.backends.base import UnsupportedModelRoleError
from localm.inference.backends.gguf import GgufBackend
from localm.model_manager import _shared
from localm.model_manager import gguf as gguf_mod
from localm.model_manager.gguf import (
    _find_model_units,
    _gguf_metadata_probe,
    _has_gguf_magic,
    gguf_general_type,
    gguf_unusable_reason,
)
from localm.model_manager.unsupported import (
    explain_unsupported_model,
    gguf_header_refusal,
    hf_folder_refusal,
    mlx_quantized_refusal,
    unsupported_quant_method_refusal,
)

_T_STRING = 8

V1_SENTENCE = ("This is a GGUF version 1 file, an early format llama.cpp no longer "
               "loads. Use a current GGUF (version 3) of the same model.")
BIG_ENDIAN_SENTENCE = ("This GGUF was written for a big-endian machine (for example "
                       "IBM s390x), so its bytes are swapped for this CPU and it "
                       "cannot be loaded here. Use the standard little-endian GGUF "
                       "of the same model.")
V4_SENTENCE = ("This GGUF declares version 4, which this build of llama.cpp does "
               "not understand (it loads versions 2 to 3). Use a GGUF of version 3, "
               "or update localm if the file is newer.")
IMATRIX_SENTENCE = ("This GGUF is an importance matrix (general.type 'imatrix') "
                    "produced by llama-imatrix to help quantise a model, not a model "
                    "itself. Point localm at the model's own .gguf file instead.")
GGJT_SENTENCE = ("This is a legacy llama.cpp model in the old GGJT format, from "
                 "before GGUF. localm loads GGUF only. Use a GGUF of the same model.")
GGMF_SENTENCE = ("This is a legacy llama.cpp model in the old GGMF format, from "
                 "before GGUF. localm loads GGUF only. Use a GGUF of the same model.")
GGML_SENTENCE = ("This is an old-format GGML file (the pre-GGUF format used by early "
                 "llama.cpp and by whisper.cpp). localm loads GGUF, not GGML. For a "
                 "language model, use a GGUF of the same model; a whisper.cpp speech "
                 "model cannot be run from this file.")
MLX_SENTENCE = ("This is an MLX-quantized model (mlx-community format), which runs "
                "only in Apple's MLX and cannot be loaded here. Use the GGUF of the "
                "same model, or the original non-MLX weights.")
EXL2_SENTENCE = ("This model is quantized with EXL2, a format only ExLlamaV2 can run, "
                 "so localm cannot load it. Use a GGUF of the same model, or the "
                 "original Hugging Face weights.")
EXL3_SENTENCE = ("This model is quantized with EXL3, a format only ExLlamaV3 can run, "
                 "so localm cannot load it. Use a GGUF of the same model, or the "
                 "original Hugging Face weights.")
MISTRAL_SENTENCE = ("This is a Mistral-native model folder (params.json plus "
                    "consolidated*.safetensors, no config.json). localm loads GGUF "
                    "files and Hugging Face-format folders (config.json plus "
                    "model-*.safetensors), not this layout. Use the model's GGUF or "
                    "its Hugging Face-format files.")

# Verbatim from the public repositories named in each constant.
MLX_CONFIG = json.loads(
    '{"architectures":["LlamaForCausalLM"],"attention_bias":false,"attention_dropout":0.0,'
    '"bos_token_id":128000,"eos_token_id":[128001,128008,128009],"head_dim":64,"hidden_act":"silu",'
    '"hidden_size":2048,"initializer_range":0.02,"intermediate_size":8192,"max_position_embeddings":131072,'
    '"mlp_bias":false,"model_type":"llama","num_attention_heads":32,"num_hidden_layers":16,'
    '"num_key_value_heads":8,"pretraining_tp":1,"quantization":{"group_size":64,"bits":4},'
    '"quantization_config":{"group_size":64,"bits":4},"rms_norm_eps":1e-05,'
    '"rope_scaling":{"factor":32.0,"high_freq_factor":4.0,"low_freq_factor":1.0,'
    '"original_max_position_embeddings":8192,"rope_type":"llama3"},"rope_theta":500000.0,'
    '"tie_word_embeddings":true,"torch_dtype":"bfloat16","transformers_version":"4.45.0.dev0",'
    '"use_cache":true,"vocab_size":128256}')  # mlx-community/Llama-3.2-1B-Instruct-4bit
EXL2_QUANT = json.loads(
    '{"quant_method":"exl2","version":"0.1.7","bits":4.0,"head_bits":6,'
    '"calibration":{"rows":115,"length":2048,"dataset":"(default)"}}')  # turboderp/...-exl2 @4.0bpw
EXL3_QUANT = json.loads(
    '{"quant_method":"exl3","version":"0.0.0","bits":4.0,'
    '"calibration":{"rows":100,"cols":2048}}')  # turboderp/...-exl3 @4.0bpw
AWQ_QUANT = json.loads(
    '{"bits":4,"group_size":128,"modules_to_not_convert":null,"quant_method":"awq",'
    '"version":"gemm","zero_point":true}')  # Qwen/Qwen2.5-0.5B-Instruct-AWQ
MISTRAL_PARAMS = json.loads(
    '{"dim":5120,"n_layers":40,"head_dim":128,"hidden_dim":32768,"n_heads":32,"n_kv_heads":8,'
    '"rope_theta":1000000000.0,"norm_eps":1e-05,"vocab_size":131072,'
    '"max_position_embeddings":131072}')  # mistralai/Mistral-Small-3.1-24B-Instruct-2503


@pytest.fixture
def home(tmp_path, monkeypatch):
    import localm.config as cfg
    h = tmp_path / ".localm"
    (h / "models").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    monkeypatch.setattr(mm, "MODELS_DIR", h / "models")
    monkeypatch.setattr(mm, "REGISTRY_FILE", h / "registry.json")
    return h


@pytest.fixture
def printed(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(_shared.console, "print",
                        lambda *a, **k: lines.append(" ".join(str(x) for x in a)))
    return lines


def _gguf(path: Path, *, version: int = 3, big_endian: bool = False,
          arch: str | None = "llama", general_type: str | None = None) -> Path:
    """A GGUF header in the byte layout the given version and endianness use,
    padded past the registration size floor and aged past the settle window."""
    e = ">" if big_endian else "<"
    kvs = []
    if general_type is not None:
        kvs.append(("general.type", general_type))
    if arch is not None:
        kvs.append(("general.architecture", arch))
    wide = version >= 2 or big_endian
    qf, lf = (e + "Q", e + "Q") if wide else (e + "I", e + "I")
    body = b""
    for key, value in kvs:
        kb, vb = key.encode(), value.encode()
        body += (struct.pack(lf, len(kb)) + kb + struct.pack(e + "I", _T_STRING)
                 + struct.pack(lf, len(vb)) + vb)
    head = (b"GGUF" + struct.pack(e + "I", version) + struct.pack(qf, 0)
            + struct.pack(qf, len(kvs)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(head + body + b"\0" * 4096)
    old = time.time() - 3600
    os.utime(path, (old, old))
    return path


def _hf_dir(d: Path, config: dict | None = None, *, weights=("model.safetensors",)) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        config if config is not None else {"architectures": ["LlamaForCausalLM"]}))
    for w in weights:
        (d / w).write_bytes(b"\0" * 64)
    (d / "tokenizer.json").write_text("{}")
    return d


# --------------------------------------------------------------------------- #
#  1. GGUF version 1, big-endian, newer than the loader                        #
# --------------------------------------------------------------------------- #

BAD_GGUFS = [
    ("v1", dict(version=1), V1_SENTENCE),
    ("big_endian_v3", dict(version=3, big_endian=True), BIG_ENDIAN_SENTENCE),
    ("big_endian_v2", dict(version=2, big_endian=True), BIG_ENDIAN_SENTENCE),
    ("v4", dict(version=4), V4_SENTENCE),
]


class TestGgufVersionAndEndianness:
    @pytest.mark.parametrize("label,kw,sentence", BAD_GGUFS, ids=[b[0] for b in BAD_GGUFS])
    def test_header_refusal_is_the_exact_sentence(self, tmp_path, label, kw, sentence):
        f = _gguf(tmp_path / "m.gguf", **kw)
        assert gguf_header_refusal(f.read_bytes()[:24]) == sentence
        assert gguf_unusable_reason(f) == sentence

    @pytest.mark.parametrize("version", [2, 3])
    def test_supported_versions_are_not_refused(self, tmp_path, version):
        f = _gguf(tmp_path / "m.gguf", version=version)
        assert gguf_unusable_reason(f) is None
        assert _has_gguf_magic(f) is True

    @pytest.mark.parametrize("version", [0, 256, 1000, 0x6D656720])
    def test_values_that_are_not_a_plausible_version_get_no_verdict(self, tmp_path, version):
        f = _gguf(tmp_path / "m.gguf", version=version)
        assert gguf_header_refusal(f.read_bytes()[:24]) is None
        assert gguf_unusable_reason(f) is None

    @pytest.mark.parametrize("content", [b"GGUF-bytes", b"GGUF gemma model bytes",
                                         b"GGUF\x00\x00\x00\x00name"])
    def test_placeholder_files_with_the_magic_get_no_verdict(self, tmp_path, content):
        f = tmp_path / "placeholder.gguf"
        f.write_bytes(content)
        assert gguf_unusable_reason(f) is None

    def test_non_gguf_and_short_files_get_no_verdict(self, tmp_path):
        assert gguf_header_refusal(b"") is None
        assert gguf_header_refusal(b"GGUF") is None
        assert gguf_header_refusal(b"PK\x03\x04" + b"\0" * 32) is None
        assert gguf_unusable_reason(tmp_path / "missing.gguf") is None

    def test_big_endian_needs_plausible_big_endian_counts(self, tmp_path):
        head = b"GGUF" + struct.pack(">I", 3) + struct.pack(">Q", 2**40) + struct.pack(">Q", 0)
        assert gguf_header_refusal(head) is None
        sane = b"GGUF" + struct.pack(">I", 3) + struct.pack(">Q", 2) + struct.pack(">Q", 1)
        assert gguf_header_refusal(sane) == BIG_ENDIAN_SENTENCE

    @pytest.mark.parametrize("label,kw,sentence", BAD_GGUFS, ids=[b[0] for b in BAD_GGUFS])
    def test_magic_check_rejects_every_case(self, tmp_path, label, kw, sentence):
        assert _has_gguf_magic(_gguf(tmp_path / "m.gguf", **kw)) is False

    def test_probes_report_no_signal_for_a_byte_swapped_file(self, tmp_path):
        f = _gguf(tmp_path / "m.gguf", version=3, big_endian=True)
        assert _gguf_metadata_probe(f) == {}
        assert gguf_mod._gguf_declared_min_size(f) is None
        assert gguf_mod.gguf_architecture(f) is None

    def test_sync_skips_them_and_logs_the_reason(self, home, monkeypatch):
        models = home / "models"
        good = _gguf(models / "good.gguf")
        for label, kw, _s in BAD_GGUFS:
            _gguf(models / f"{label}.gguf", **kw)
        logged: list[str] = []
        monkeypatch.setattr(gguf_mod.logger, "debug",
                            lambda msg, *a, **k: logged.append(msg % a if a else msg))
        mm.sync_models_dir(prune=False)
        assert set(mm.load_registry()) == {"good"}
        assert any(V1_SENTENCE in m for m in logged)
        assert any(BIG_ENDIAN_SENTENCE in m for m in logged)
        assert good.exists()

    @pytest.mark.parametrize("label,kw,sentence", BAD_GGUFS, ids=[b[0] for b in BAD_GGUFS])
    def test_add_local_refuses_with_the_sentence(self, home, tmp_path, printed,
                                                 label, kw, sentence):
        f = _gguf(tmp_path / "m.gguf", **kw)
        assert mm.add_local(str(f)) is False
        assert mm.load_registry() == {}
        assert sentence in "\n".join(printed)

    def test_add_local_still_registers_a_normal_v3(self, home, tmp_path):
        f = _gguf(tmp_path / "ok.gguf")
        assert mm.add_local(str(f)) is True
        assert "ok" in mm.load_registry()

    def test_folder_add_skips_the_unusable_file_and_registers_the_rest(
            self, home, tmp_path, printed):
        folder = tmp_path / "pile"
        _gguf(folder / "ok.gguf")
        _gguf(folder / "old.gguf", version=1)
        _gguf(folder / "swapped.gguf", big_endian=True)
        assert mm.add_local(str(folder)) is True
        assert set(mm.load_registry()) == {"ok"}
        out = "\n".join(printed)
        assert "Skipped old.gguf" in out and V1_SENTENCE in out
        assert "Skipped swapped.gguf" in out and BIG_ENDIAN_SENTENCE in out

    def test_relocate_target_refuses_with_the_sentence(self, tmp_path):
        f = _gguf(tmp_path / "m.gguf", version=1)
        path, reason = mm.relocate_target(str(f))
        assert path is None and V1_SENTENCE in reason

    @pytest.mark.parametrize("label,kw,sentence", BAD_GGUFS, ids=[b[0] for b in BAD_GGUFS])
    def test_load_never_reaches_the_native_loader(self, tmp_path, monkeypatch,
                                                  label, kw, sentence):
        f = _gguf(tmp_path / "m.gguf", **kw)
        backend = GgufBackend(str(f))
        for hook in ("_check_vram", "_load_native", "_effective_gpu_layers"):
            monkeypatch.setattr(backend, hook, lambda *a, hook=hook, **k: pytest.fail(
                f"{hook} ran for an unloadable GGUF"))
        with pytest.raises(UnsupportedModelRoleError) as caught:
            backend.load()
        assert str(caught.value) == sentence


# --------------------------------------------------------------------------- #
#  2. Legacy GGML files                                                        #
# --------------------------------------------------------------------------- #

def _legacy_bin(path: Path, magic: bytes, *, versioned: bool, vocab: int = 32000) -> Path:
    body = magic + (struct.pack("<I", 3) if versioned else b"")
    body += struct.pack("<7i", vocab, 4096, 256, 32, 32, 128, 2) + b"\0" * 256
    path.write_bytes(body)
    return path


class TestLegacyGgml:
    @pytest.mark.parametrize("magic,versioned,sentence", [
        (b"tjgg", True, GGJT_SENTENCE),
        (b"fmgg", True, GGMF_SENTENCE),
        (b"lmgg", False, GGML_SENTENCE),
    ])
    def test_lone_bin_with_a_legacy_magic_is_explained(self, home, tmp_path, printed,
                                                       magic, versioned, sentence):
        f = _legacy_bin(tmp_path / "ggml-model-q4_0.bin", magic, versioned=versioned)
        assert explain_unsupported_model(f) == sentence
        assert mm.add_local(str(f)) is False
        assert sentence in "\n".join(printed)
        assert mm.load_registry() == {}

    def test_a_whisper_model_is_not_called_a_llama_model(self, tmp_path):
        f = _legacy_bin(tmp_path / "ggml-base.bin", b"lmgg", versioned=False, vocab=51865)
        sentence = explain_unsupported_model(f)
        assert sentence == GGML_SENTENCE
        assert "whisper.cpp" in sentence and "legacy llama" not in sentence.lower()

    def test_unknown_magic_gets_no_sentence(self, home, tmp_path, printed):
        f = tmp_path / "weights.bin"
        f.write_bytes(b"PK\x03\x04" + b"\0" * 64)
        assert explain_unsupported_model(f) is None
        assert mm.add_local(str(f)) is False
        assert "Expected a .gguf file or a HuggingFace model directory" in "\n".join(printed)

    def test_the_magic_is_only_read_from_a_bin_file(self, tmp_path):
        f = _legacy_bin(tmp_path / "model.dat", b"tjgg", versioned=True)
        assert explain_unsupported_model(f) is None

    def test_pytorch_model_bin_inside_an_hf_folder_is_untouched(self, home, tmp_path):
        d = _hf_dir(tmp_path / "hf", weights=("pytorch_model.bin",))
        (d / "pytorch_model.bin").write_bytes(b"lmgg" + b"\0" * 64)
        assert explain_unsupported_model(d) is None
        assert explain_unsupported_model(d / "pytorch_model.bin") is None
        assert mm.add_local(str(d)) is True
        assert "hf" in mm.load_registry()


# --------------------------------------------------------------------------- #
#  3. MLX-quantized folders                                                    #
# --------------------------------------------------------------------------- #

def _no_runner(monkeypatch):
    from localm.inference.backends import hf as hf_mod

    class _Runner:
        def __init__(self):
            raise AssertionError("a worker was spawned")

    monkeypatch.setattr(hf_mod, "HFRunner", _Runner)
    return hf_mod


class TestMlxQuantized:
    def test_real_config_is_recognised(self):
        assert config_refusal(MLX_CONFIG) == MLX_SENTENCE

    def test_registered_as_today_and_load_refused_before_any_worker(
            self, home, tmp_path, monkeypatch):
        hf_mod = _no_runner(monkeypatch)
        d = _hf_dir(tmp_path / "Llama-3-2-1B-Instruct-4bit", MLX_CONFIG)
        assert mm.add_local(str(d)) is True
        assert "Llama-3-2-1B-Instruct-4bit" in mm.load_registry()
        with pytest.raises(UnsupportedModelRoleError) as caught:
            hf_mod.HFBackend(str(d)).load()
        assert str(caught.value) == MLX_SENTENCE

    def test_explained_when_add_is_pointed_at_one_without_weights(self, tmp_path):
        d = tmp_path / "mlx"
        d.mkdir()
        (d / "config.json").write_text(json.dumps(MLX_CONFIG))
        assert explain_unsupported_model(d) == MLX_SENTENCE

    def test_plain_hf_awq_and_unquantized_mlx_are_not_refused(self):
        plain = {"architectures": ["LlamaForCausalLM"]}
        awq = {**plain, "quantization_config": AWQ_QUANT}
        mlx_bf16 = {k: v for k, v in MLX_CONFIG.items()
                    if k not in ("quantization", "quantization_config")}
        for cfg in (plain, awq, mlx_bf16):
            assert config_refusal(cfg) is None

    def test_a_declared_quant_method_beside_quantization_is_not_mlx(self):
        cfg = {**MLX_CONFIG, "quantization_config": {**AWQ_QUANT}}
        assert config_refusal(cfg) is None

    @pytest.mark.parametrize("quant", [None, "x", {"bits": 4}, {"group_size": 64},
                                       {"group_size": True, "bits": 4}])
    def test_a_quantization_key_without_integer_group_size_and_bits_is_not_mlx(self, quant):
        assert config_refusal({"architectures": ["LlamaForCausalLM"],
                                      "quantization": quant}) is None


def config_refusal(config: dict):
    return mlx_quantized_refusal(config) or unsupported_quant_method_refusal(config)


# --------------------------------------------------------------------------- #
#  4. Quantizations only another runtime loads                                 #
# --------------------------------------------------------------------------- #

class TestUnsupportedQuantMethod:
    @pytest.mark.parametrize("quant,sentence", [(EXL2_QUANT, EXL2_SENTENCE),
                                                (EXL3_QUANT, EXL3_SENTENCE)])
    def test_exllama_folders_are_refused_by_name_at_load(self, home, tmp_path, monkeypatch,
                                                         quant, sentence):
        hf_mod = _no_runner(monkeypatch)
        d = _hf_dir(tmp_path / "exl", {"architectures": ["LlamaForCausalLM"],
                                       "quantization_config": quant})
        assert mm.add_local(str(d)) is True
        with pytest.raises(UnsupportedModelRoleError) as caught:
            hf_mod.HFBackend(str(d)).load()
        assert str(caught.value) == sentence

    def test_a_plain_awq_folder_passes_the_format_check(self, tmp_path, monkeypatch):
        hf_mod = _no_runner(monkeypatch)
        d = _hf_dir(tmp_path / "awq", {"architectures": ["Qwen2ForCausalLM"],
                                       "quantization_config": AWQ_QUANT})
        with pytest.raises(AssertionError, match="a worker was spawned"):
            hf_mod.HFBackend(str(d)).load()

    @pytest.mark.parametrize("method", ["gptq", "bitsandbytes", "fp8", "compressed-tensors",
                                        "mxfp4", "something-new", None, 3])
    def test_other_methods_are_left_to_the_loader(self, tmp_path, method):
        d = tmp_path / "q"
        d.mkdir()
        (d / "config.json").write_text(json.dumps(
            {"quantization_config": {"quant_method": method, "bits": 4}}))
        assert hf_folder_refusal(d) is None


# --------------------------------------------------------------------------- #
#  5. Mistral-native folders                                                   #
# --------------------------------------------------------------------------- #

def _mistral_native(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "params.json").write_text(json.dumps(MISTRAL_PARAMS))
    (d / "consolidated.safetensors").write_bytes(b"\0" * 64)
    (d / "tekken.json").write_text("{}")
    return d


class TestMistralNative:
    def test_folder_is_explained_and_refused(self, home, tmp_path, printed):
        d = _mistral_native(tmp_path / "Mistral-Small")
        assert explain_unsupported_model(d) == MISTRAL_SENTENCE
        assert mm.add_local(str(d)) is False
        assert MISTRAL_SENTENCE in "\n".join(printed)
        assert mm.load_registry() == {}

    def test_pointing_at_its_consolidated_file_explains_the_layout(self, home, tmp_path, printed):
        d = _mistral_native(tmp_path / "Mistral-Small")
        assert mm.add_local(str(d / "consolidated.safetensors")) is False
        out = "\n".join(printed)
        assert MISTRAL_SENTENCE in out and "Incomplete model" not in out

    def test_a_folder_with_both_layouts_is_one_hf_model(self, home, tmp_path):
        d = _hf_dir(tmp_path / "Mistral-Small-3-1", weights=(
            "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
            "consolidated.safetensors"))
        (d / "params.json").write_text(json.dumps(MISTRAL_PARAMS))
        (d / "tekken.json").write_text("{}")
        assert explain_unsupported_model(d) is None
        assert mm.add_local(str(d)) is True
        reg = mm.load_registry()
        assert list(reg) == ["Mistral-Small-3-1"]
        assert Path(reg["Mistral-Small-3-1"]["path"]).is_dir()

    def test_sync_registers_the_two_layout_folder_once(self, home):
        d = _hf_dir(home / "models" / "Mistral-Small-3.1", weights=(
            "model-00001-of-00002.safetensors", "consolidated.safetensors"))
        (d / "params.json").write_text(json.dumps(MISTRAL_PARAMS))
        mm.sync_models_dir(prune=False)
        assert list(mm.load_registry()) == ["Mistral-Small-3.1"]

    def test_the_walk_reports_the_two_layout_folder_as_one_unit(self, tmp_path):
        root = tmp_path / "root"
        d = _hf_dir(root / "m", weights=("model.safetensors", "consolidated.safetensors"))
        (d / "params.json").write_text(json.dumps(MISTRAL_PARAMS))
        ggufs, hf_dirs = _find_model_units(root)
        assert ggufs == [] and hf_dirs == [d]


# --------------------------------------------------------------------------- #
#  6. Importance-matrix GGUFs                                                  #
# --------------------------------------------------------------------------- #

class TestImatrix:
    def test_reader_returns_the_declared_type(self, tmp_path):
        assert gguf_general_type(_gguf(tmp_path / "a.gguf", arch=None,
                                       general_type="imatrix")) == "imatrix"
        assert gguf_general_type(_gguf(tmp_path / "b.gguf")) is None

    def test_add_local_refuses_it(self, home, tmp_path, printed):
        f = _gguf(tmp_path / "imatrix.gguf", arch=None, general_type="imatrix")
        assert gguf_unusable_reason(f) == IMATRIX_SENTENCE
        assert mm.add_local(str(f)) is False
        assert IMATRIX_SENTENCE in "\n".join(printed)
        assert mm.load_registry() == {}

    def test_sync_keeps_it_out_of_the_model_list(self, home):
        _gguf(home / "models" / "imatrix.gguf", arch=None, general_type="imatrix")
        _gguf(home / "models" / "real.gguf")
        mm.sync_models_dir(prune=False)
        assert set(mm.load_registry()) == {"real"}

    def test_load_is_refused(self, tmp_path, monkeypatch):
        f = _gguf(tmp_path / "imatrix.gguf", arch=None, general_type="imatrix")
        backend = GgufBackend(str(f))
        monkeypatch.setattr(backend, "_load_native", lambda *a, **k: pytest.fail("loaded"))
        with pytest.raises(UnsupportedModelRoleError) as caught:
            backend.load()
        assert str(caught.value) == IMATRIX_SENTENCE

    @pytest.mark.parametrize("general_type", ["model", "adapter", "clip-vision"])
    def test_other_general_types_are_left_alone(self, tmp_path, general_type):
        f = _gguf(tmp_path / "x.gguf", general_type=general_type)
        assert gguf_unusable_reason(f) is None


# --------------------------------------------------------------------------- #
#  7. Other formats named when add is pointed straight at them                 #
# --------------------------------------------------------------------------- #

ALT = "Use a GGUF of the same model, or the original Hugging Face weights."
FILE_SENTENCES = {
    "model.onnx": ("An .onnx file is an ONNX model, which needs an ONNX runtime; localm "
                   "runs GGUF files and Hugging Face-format folders. " + ALT),
    "model.engine": ("An .engine file is a TensorRT engine, built for one GPU model and "
                     "TensorRT version and runnable only by TensorRT. " + ALT),
    "model.plan": ("A .plan file is a TensorRT engine, built for one GPU model and "
                   "TensorRT version and runnable only by TensorRT. " + ALT),
    "gemma.litertlm": ("A .litertlm file is a LiteRT-LM bundle for Google's on-device "
                       "runtime, which localm does not include. " + ALT),
    "gemma.task": ("A .task file is a MediaPipe / LiteRT bundle for Google's on-device "
                   "runtime, which localm does not include. " + ALT),
    "asr.nemo": ("A .nemo file is an NVIDIA NeMo checkpoint archive that loads only in "
                 "NeMo. " + ALT),
}
FOLDER_SENTENCES = {
    "mlc-chat-config.json": ("This folder is an MLC LLM compiled model, which runs only "
                             "in the MLC LLM runtime. " + ALT),
    "openvino_model.xml": ("This folder is an OpenVINO IR model, which runs only in the "
                           "OpenVINO runtime. " + ALT),
}


class TestOtherFormats:
    @pytest.mark.parametrize("name,sentence", FILE_SENTENCES.items())
    def test_file_formats(self, home, tmp_path, printed, name, sentence):
        f = tmp_path / name
        f.write_bytes(b"\0" * 64)
        assert explain_unsupported_model(f) == sentence
        assert mm.add_local(str(f)) is False
        assert "Not a model" in "\n".join(printed) and sentence in "\n".join(printed)

    @pytest.mark.parametrize("marker,sentence", FOLDER_SENTENCES.items())
    def test_folder_formats(self, home, tmp_path, printed, marker, sentence):
        d = tmp_path / "pack"
        d.mkdir()
        (d / marker).write_text("{}")
        assert explain_unsupported_model(d) == sentence
        assert mm.add_local(str(d)) is False
        assert sentence in "\n".join(printed)

    @pytest.mark.parametrize("quant,sentence", [(EXL2_QUANT, EXL2_SENTENCE),
                                                (EXL3_QUANT, EXL3_SENTENCE)])
    def test_exllama_folder(self, tmp_path, quant, sentence):
        d = tmp_path / "exl"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"quantization_config": quant}))
        assert explain_unsupported_model(d) == sentence

    def test_nothing_runnable_or_unknown_gets_a_sentence(self, tmp_path):
        gguf = _gguf(tmp_path / "ok.gguf")
        hf = _hf_dir(tmp_path / "hf")
        awq = _hf_dir(tmp_path / "awq", {"quantization_config": AWQ_QUANT})
        empty = tmp_path / "empty"
        empty.mkdir()
        note = tmp_path / "notes.txt"
        note.write_text("hello")
        for p in (gguf, hf, awq, empty, note, tmp_path / "missing.onnx"):
            assert explain_unsupported_model(p) is None, p
        assert explain_unsupported_model(tmp_path) is None

    def test_unknown_folder_keeps_the_generic_message(self, home, tmp_path, printed):
        d = tmp_path / "stuff"
        d.mkdir()
        (d / "readme.txt").write_text("x")
        assert mm.add_local(str(d)) is False
        assert "Expected a .gguf file or a HuggingFace model directory" in "\n".join(printed)


# --------------------------------------------------------------------------- #
#  A downloaded GGUF that cannot be registered                                 #
# --------------------------------------------------------------------------- #

class TestPulledFile:
    def _fake_hub(self, monkeypatch, tmp_path, make):
        import huggingface_hub
        import requests

        def _download(repo_id, filename, local_dir, **kw):
            return str(make(Path(local_dir) / filename))

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo_id, filename: None)
        monkeypatch.setattr(requests, "head", lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("no network in tests")))

    @pytest.mark.parametrize("label,kw,sentence", [
        ("v1", dict(version=1), V1_SENTENCE),
        ("big_endian", dict(big_endian=True), BIG_ENDIAN_SENTENCE),
        ("imatrix", dict(arch=None, general_type="imatrix"), IMATRIX_SENTENCE),
    ])
    def test_it_is_not_registered_and_the_sentence_is_printed(
            self, home, tmp_path, monkeypatch, printed, label, kw, sentence):
        self._fake_hub(monkeypatch, tmp_path, lambda p: _gguf(p, **kw))
        assert mm.pull_model("owner/repo:pulled.gguf") is False
        assert mm.load_registry() == {}
        assert sentence in "\n".join(printed)

    def test_a_normal_gguf_still_registers(self, home, tmp_path, monkeypatch):
        self._fake_hub(monkeypatch, tmp_path, lambda p: _gguf(p))
        assert mm.pull_model("owner/repo:pulled.gguf") is True
        assert "pulled" in mm.load_registry()

    def test_a_file_already_in_the_models_folder_is_not_registered_either(
            self, home, tmp_path, monkeypatch, printed):
        _gguf(home / "models" / "pulled.gguf", version=1)
        self._fake_hub(monkeypatch, tmp_path, lambda p: pytest.fail("downloaded again"))
        assert mm.pull_model("owner/repo:pulled.gguf") is False
        assert mm.load_registry() == {}
        assert V1_SENTENCE in "\n".join(printed)


# --------------------------------------------------------------------------- #
#  Other registration paths                                                    #
# --------------------------------------------------------------------------- #

def _ollama_manifest(root: Path, blob_writer) -> Path:
    digest = "a" * 64
    blob = root / "blobs" / f"sha256-{digest}"
    blob.parent.mkdir(parents=True)
    blob_writer(blob)
    manifest_dir = root / "manifests" / "registry.ollama.ai" / "library" / "m" / "latest"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "latest").write_text(json.dumps({"layers": [
        {"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{digest}"}]}))
    return manifest_dir


class TestOllamaBlob:
    @pytest.mark.parametrize("label,kw,sentence", [
        ("v1", dict(version=1), V1_SENTENCE),
        ("imatrix", dict(arch=None, general_type="imatrix"), IMATRIX_SENTENCE),
    ])
    def test_an_unusable_blob_is_not_registered(self, home, tmp_path, printed,
                                                label, kw, sentence):
        manifest = _ollama_manifest(tmp_path / "ollama", lambda p: _gguf(p, **kw))
        assert mm.add_local(str(manifest)) is False
        assert mm.load_registry() == {}
        assert sentence in "\n".join(printed)

    def test_a_normal_blob_still_registers(self, home, tmp_path):
        manifest = _ollama_manifest(tmp_path / "ollama", lambda p: _gguf(p))
        assert mm.add_local(str(manifest)) is True
        assert len(mm.load_registry()) == 1


class TestUrlPull:
    URL = "http://host.example/pulled.gguf"

    @pytest.fixture
    def url_env(self, home, monkeypatch):
        monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
        monkeypatch.setattr(mm, "find_by_sha256", lambda *a, **k: [])
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])

    def _serve(self, monkeypatch, body: bytes):
        from unittest.mock import MagicMock

        def fake_pinned_request(method, url, **kwargs):
            if method == "HEAD":
                h = MagicMock()
                h.status_code = 200
                h.headers = {"content-length": str(len(body))}
                return h
            r = MagicMock()
            r.status_code = 200
            r.raise_for_status = MagicMock()
            r.headers = {"content-length": str(len(body))}
            r.iter_content = lambda chunk_size: iter([body])
            return r

        monkeypatch.setattr("localm.netpolicy.pinned_request", fake_pinned_request)

    @pytest.mark.parametrize("label,kw,sentence", [
        ("v1", dict(version=1), V1_SENTENCE),
        ("big_endian", dict(big_endian=True), BIG_ENDIAN_SENTENCE),
        ("imatrix", dict(arch=None, general_type="imatrix"), IMATRIX_SENTENCE),
    ])
    def test_a_downloaded_unusable_gguf_is_not_registered(
            self, url_env, home, tmp_path, monkeypatch, printed, label, kw, sentence):
        body = _gguf(tmp_path / "src.gguf", **kw).read_bytes()
        self._serve(monkeypatch, body)
        assert mm.pull_model(self.URL) is False
        assert mm.load_registry() == {}
        assert sentence in "\n".join(printed)

    def test_a_normal_download_registers(self, url_env, home, tmp_path, monkeypatch):
        self._serve(monkeypatch, _gguf(tmp_path / "src.gguf").read_bytes())
        assert mm.pull_model(self.URL) is True
        assert "pulled" in mm.load_registry()

    def test_a_file_already_in_the_models_folder_is_not_registered(
            self, url_env, home, monkeypatch, printed):
        _gguf(home / "models" / "pulled.gguf", version=1)
        self._serve(monkeypatch, b"unused")
        assert mm.pull_model(self.URL) is False
        assert mm.load_registry() == {}
        assert V1_SENTENCE in "\n".join(printed)


OPENVINO_SENTENCE = FOLDER_SENTENCES["openvino_model.xml"]


class TestFoldersThatRegisterAsHuggingFace:
    def test_an_openvino_export_registers_with_a_note_and_is_refused_at_load(
            self, home, tmp_path, monkeypatch, printed):
        hf_mod = _no_runner(monkeypatch)
        d = _hf_dir(tmp_path / "ov", weights=("openvino_model.bin",))
        (d / "openvino_model.xml").write_text("<net/>")
        assert mm.add_local(str(d)) is True
        assert "Registered, but localm cannot load it" in "\n".join(printed)
        assert OPENVINO_SENTENCE in "\n".join(printed)
        with pytest.raises(UnsupportedModelRoleError) as caught:
            hf_mod.HFBackend(str(d)).load()
        assert str(caught.value) == OPENVINO_SENTENCE

    def test_a_folder_with_openvino_and_real_weights_is_not_refused(self, tmp_path):
        d = _hf_dir(tmp_path / "both")
        (d / "openvino_model.xml").write_text("<net/>")
        assert hf_folder_refusal(d) is None

    def test_an_exllama_folder_registers_with_the_note(self, home, tmp_path, printed):
        d = _hf_dir(tmp_path / "exl", {"architectures": ["LlamaForCausalLM"],
                                       "quantization_config": EXL2_QUANT})
        assert mm.add_local(str(d)) is True
        out = "\n".join(printed)
        assert "Registered, but localm cannot load it" in out and EXL2_SENTENCE in out

    def test_a_plain_hf_folder_gets_no_note(self, home, tmp_path, printed):
        d = _hf_dir(tmp_path / "plain")
        assert mm.add_local(str(d)) is True
        assert "cannot load it" not in "\n".join(printed)


class TestConsolidatedFilesAreNotCounted:
    def test_two_layout_folder_is_sized_by_the_hf_shards_only(self, tmp_path):
        from localm.inference.residency import alternate_layout_files, model_footprint_bytes
        d = tmp_path / "m"
        d.mkdir()
        (d / "config.json").write_text("{}")
        (d / "model-00001-of-00001.safetensors").write_bytes(b"\0" * 1000)
        (d / "consolidated.safetensors").write_bytes(b"\0" * 1000)
        assert alternate_layout_files(d) == frozenset({d / "consolidated.safetensors"})
        assert model_footprint_bytes(d) == 1000 + len("{}")

    def test_a_folder_with_only_consolidated_weights_counts_them(self, tmp_path):
        from localm.inference.residency import alternate_layout_files, model_footprint_bytes
        d = tmp_path / "m"
        d.mkdir()
        (d / "config.json").write_text("{}")
        (d / "consolidated.safetensors").write_bytes(b"\0" * 1000)
        assert alternate_layout_files(d) == frozenset()
        assert model_footprint_bytes(d) == 1000 + len("{}")

    def test_a_folder_without_config_json_counts_everything(self, tmp_path):
        from localm.inference.residency import alternate_layout_files
        d = tmp_path / "m"
        d.mkdir()
        (d / "model.safetensors").write_bytes(b"\0" * 10)
        (d / "consolidated.safetensors").write_bytes(b"\0" * 10)
        assert alternate_layout_files(d) == frozenset()
