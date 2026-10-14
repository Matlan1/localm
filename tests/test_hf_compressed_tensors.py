# SPDX-License-Identifier: AGPL-3.0-or-later
"""HuggingFace checkpoints quantized with compressed-tensors load dequantized.

The HF worker passes ``CompressedTensorsConfig(dequantize=True)`` for a checkpoint
whose config.json declares ``quant_method: compressed-tensors``, and gives a torch
built without ``torch.distributed`` the ``Work`` class the compressed-tensors
package names at import. The real-model tests build a tiny int8 checkpoint with
the compressed-tensors package itself and load it through ``HFWorker``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types

import pytest

from localm.inference.backends._hf_worker import (
    _define_missing_distributed_work,
    _is_compressed_tensors,
)

needs_stack = pytest.mark.skipif(
    importlib.util.find_spec("compressed_tensors") is None
    or importlib.util.find_spec("transformers") is None,
    reason="needs the compressed-tensors and transformers packages")


def _config(tmp_path, **extra):
    folder = tmp_path / "ckpt"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(extra), encoding="utf-8")
    return str(folder)


class TestDetection:
    def test_declared_method_is_detected(self, tmp_path):
        path = _config(tmp_path, quantization_config={"quant_method": "compressed-tensors"})
        assert _is_compressed_tensors(path) is True

    def test_method_name_is_case_insensitive(self, tmp_path):
        path = _config(tmp_path, quantization_config={"quant_method": "Compressed-Tensors"})
        assert _is_compressed_tensors(path) is True

    @pytest.mark.parametrize("extra", [
        {}, {"quantization_config": {"quant_method": "awq"}},
        {"quantization_config": {"quant_method": "fp8"}}, {"quantization_config": "x"}])
    def test_other_checkpoints_are_not(self, tmp_path, extra):
        assert _is_compressed_tensors(_config(tmp_path, **extra)) is False

    def test_missing_config_is_not(self, tmp_path):
        assert _is_compressed_tensors(str(tmp_path)) is False


class TestDistributedWorkShim:
    @staticmethod
    def _torch(*, available, work):
        dist = types.SimpleNamespace(is_available=lambda: available)
        if work is not None:
            dist.Work = work
        return types.SimpleNamespace(distributed=dist)

    def test_defined_when_distributed_is_missing(self):
        torch = self._torch(available=False, work=None)
        _define_missing_distributed_work(torch)
        assert isinstance(torch.distributed.Work, type)

    def test_left_alone_when_distributed_is_available(self):
        torch = self._torch(available=True, work=None)
        _define_missing_distributed_work(torch)
        assert not hasattr(torch.distributed, "Work")

    def test_an_existing_class_is_kept(self):
        sentinel = type("Work", (), {})
        torch = self._torch(available=False, work=sentinel)
        _define_missing_distributed_work(torch)
        assert torch.distributed.Work is sentinel

    def test_torch_without_a_distributed_module_is_tolerated(self):
        _define_missing_distributed_work(types.SimpleNamespace())


_IMPORT_WITHOUT_WORK = """
import torch.distributed as dist
dist.is_available = lambda: False
if hasattr(dist, "Work"):
    del dist.Work
import torch
from localm.inference.backends._hf_worker import _define_missing_distributed_work
try:
    import compressed_tensors
    print("IMPORTED_WITHOUT_SHIM")
except AttributeError:
    print("FAILS_WITHOUT_SHIM")
    for name in [m for m in sys.modules if m.startswith("compressed_tensors")]:
        del sys.modules[name]
    _define_missing_distributed_work(torch)
    import compressed_tensors
    print("IMPORTS_WITH_SHIM")
"""


@needs_stack
def test_compressed_tensors_loads_without_torch_distributed():
    code = "import sys\n" + _IMPORT_WITHOUT_WORK
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stderr
    assert "FAILS_WITHOUT_SHIM" in r.stdout and "IMPORTS_WITH_SHIM" in r.stdout, r.stdout


def _build_int8_checkpoint(folder):
    """A 2-layer llama with int8 channel-quantized Linear layers saved in the
    compressed-tensors format; returns the logits of the quantized model for a
    fixed prompt."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    _define_missing_distributed_work(torch)
    from compressed_tensors.compressors import ModelCompressor
    from compressed_tensors.quantization import QuantizationConfig, apply_quantization_config

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=128, bos_token_id=1, eos_token_id=2,
                      pad_token_id=0)
    model = LlamaForCausalLM(cfg).eval()
    qcfg = QuantizationConfig.model_validate({
        "config_groups": {"group_0": {"targets": ["Linear"], "weights": {
            "num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel",
            "dynamic": False}}},
        "ignore": ["lm_head"], "quant_method": "compressed-tensors",
        "format": "int-quantized"})
    apply_quantization_config(model, qcfg)
    for module in model.modules():
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is None or scheme.weights is None:
            continue
        scale = (module.weight.data.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-8)
        module.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        module.weight_zero_point = torch.nn.Parameter(
            torch.zeros_like(scale, dtype=torch.int8), requires_grad=False)
    with torch.no_grad():
        reference = model(torch.tensor([[1, 5, 6, 7]])).logits
    compressor = ModelCompressor.from_pretrained_model(model, quantization_format="int-quantized")
    compressor.compress_model(model)
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    compressor.update_config(str(folder))
    tok = Tokenizer(models.WordLevel({f"t{i}": i for i in range(64)}, unk_token="t0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="t0", bos_token="t1",
                                   eos_token="t2", pad_token="t0")
    fast.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    fast.save_pretrained(folder)
    return reference


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    from localm.inference.backends._hf_worker import HFWorker

    folder = tmp_path_factory.mktemp("ct") / "tiny-int8"
    reference = _build_int8_checkpoint(folder)
    worker = HFWorker(str(folder), device="cpu")
    worker.load()
    yield worker, reference, folder
    worker.unload()


@needs_stack
class TestRealCheckpoint:
    def test_checkpoint_declares_compressed_tensors(self, loaded):
        _, _, folder = loaded
        assert _is_compressed_tensors(str(folder)) is True

    def test_weights_are_expanded_and_match_the_quantized_model(self, loaded):
        import torch
        worker, reference, _ = loaded
        proj = worker._model.model.layers[0].self_attn.q_proj
        assert proj.weight.dtype == torch.float32
        with torch.no_grad():
            logits = worker._model(torch.tensor([[1, 5, 6, 7]])).logits
        assert torch.allclose(logits.float(), reference.float(), atol=1e-5)

    def test_the_load_reports_the_expansion(self, loaded):
        worker, _, _ = loaded
        assert any("compressed-tensors weights expanded to float32" in n
                   for n in worker.load_notes), worker.load_notes


def _build_quantized(folder, *, num_bits, fmt):
    """A 2-layer llama whose Linear layers are channel-quantized to *num_bits*
    ints and saved in the compressed-tensors *fmt*; also saves the same model
    unquantized beside it and returns that folder."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    _define_missing_distributed_work(torch)
    from compressed_tensors.compressors import ModelCompressor
    from compressed_tensors.quantization import QuantizationConfig, apply_quantization_config

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=128, bos_token_id=1, eos_token_id=2,
                      pad_token_id=0)
    model = LlamaForCausalLM(cfg).eval()
    plain = folder.parent / (folder.name + "-plain")
    model.save_pretrained(plain)
    qcfg = QuantizationConfig.model_validate({
        "config_groups": {"group_0": {"targets": ["Linear"], "weights": {
            "num_bits": num_bits, "type": "int", "symmetric": True, "strategy": "channel",
            "dynamic": False}}},
        "ignore": ["lm_head"], "quant_method": "compressed-tensors", "format": fmt})
    apply_quantization_config(model, qcfg)
    top = float(2 ** (num_bits - 1) - 1)
    for module in model.modules():
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is None or scheme.weights is None:
            continue
        scale = (module.weight.data.abs().amax(dim=1, keepdim=True) / top).clamp_min(1e-8)
        module.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        module.weight_zero_point = torch.nn.Parameter(
            torch.zeros_like(scale, dtype=torch.int8), requires_grad=False)
    compressor = ModelCompressor.from_pretrained_model(model, quantization_format=fmt)
    compressor.compress_model(model)
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    compressor.update_config(str(folder))
    return plain


@needs_stack
class TestExpandedSize:
    @pytest.mark.parametrize("num_bits,fmt", [(4, "pack-quantized"), (8, "int-quantized")])
    @pytest.mark.parametrize("value_bytes", [2, 4])
    def test_the_estimate_is_the_unquantized_model_at_the_load_dtype(
            self, tmp_path, num_bits, fmt, value_bytes):
        from localm.inference.backends import _hf_fp8
        folder = tmp_path / "q"
        plain = _build_quantized(folder, num_bits=num_bits, fmt=fmt)
        on_disk = sum(f.stat().st_size for f in _hf_fp8.weight_files(str(folder)))
        expected = _hf_fp8.expanded_bf16_bytes(str(plain)) // 2 * value_bytes
        assert _hf_fp8.compressed_expanded_bytes(str(folder), value_bytes) == expected
        assert expected > on_disk

    def test_a_packed_checkpoint_that_does_not_fit_is_refused_before_its_weights_load(
            self, tmp_path, monkeypatch):
        from localm.inference.backends import _hf_fp8, _hf_worker
        from localm.inference.backends._hf_worker import HFWorker
        folder = tmp_path / "q"
        _build_quantized(folder, num_bits=4, fmt="pack-quantized")
        _write_tokenizer(folder)
        needed = _hf_fp8.compressed_expanded_bytes(str(folder), 4)
        monkeypatch.setattr(_hf_fp8, "memory_budget", lambda *a: needed - 1)
        monkeypatch.setattr(_hf_worker, "_build_load_kwargs", lambda *a, **k: pytest.fail(
            "the weights started loading"))
        worker = HFWorker(str(folder), device="cpu")
        with pytest.raises(RuntimeError) as ei:
            worker.load()
        msg = str(ei.value)
        assert "compressed-tensors checkpoint" in msg and "float32" in msg, msg

    def test_a_packed_checkpoint_that_fits_loads(self, tmp_path, monkeypatch):
        from localm.inference.backends import _hf_fp8
        from localm.inference.backends._hf_worker import HFWorker
        folder = tmp_path / "q"
        _build_quantized(folder, num_bits=4, fmt="pack-quantized")
        needed = _hf_fp8.compressed_expanded_bytes(str(folder), 4)
        monkeypatch.setattr(_hf_fp8, "memory_budget", lambda *a: needed)
        _write_tokenizer(folder)
        worker = HFWorker(str(folder), device="cpu")
        worker.load()
        try:
            assert worker._model is not None
        finally:
            worker.unload()


def _build_nvfp4(folder):
    """A 1-layer llama whose Linear layers are saved in the compressed-tensors
    nvfp4-pack-quantized format; saves the same model unquantized beside it and
    returns that folder."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    _define_missing_distributed_work(torch)
    from compressed_tensors.compressors import ModelCompressor
    from compressed_tensors.quantization import QuantizationConfig, apply_quantization_config

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=64)
    model = LlamaForCausalLM(cfg).eval()
    plain = folder.parent / (folder.name + "-plain")
    model.save_pretrained(plain)
    qcfg = QuantizationConfig.model_validate({
        "config_groups": {"g": {"targets": ["Linear"], "weights": {
            "num_bits": 4, "type": "float", "symmetric": True,
            "strategy": "tensor_group", "group_size": 16, "dynamic": False,
            "scale_dtype": "float8_e4m3fn"}}},
        "ignore": ["lm_head"], "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized"})
    apply_quantization_config(model, qcfg)
    for module in model.modules():
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is None or scheme.weights is None:
            continue
        weight = module.weight.data
        global_scale = torch.tensor([448.0 * 6.0 / weight.abs().max().item()])
        module.weight_global_scale = torch.nn.Parameter(global_scale, requires_grad=False)
        group = weight.reshape(weight.shape[0], -1, 16).abs().amax(-1) / 6.0
        scale = (group * global_scale).clamp_min(1e-4).to(torch.float8_e4m3fn)
        module.weight_scale = torch.nn.Parameter(scale.to(torch.float32), requires_grad=False)
    compressor = ModelCompressor.from_pretrained_model(
        model, quantization_format="nvfp4-pack-quantized")
    compressor.compress_model(model)
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    compressor.update_config(str(folder))
    _write_tokenizer(folder)
    return plain


@needs_stack
def test_an_nvfp4_checkpoint_runs_at_the_cpu_load_dtype(tmp_path):
    import torch
    from localm.inference.backends import _hf_fp8
    from localm.inference.backends._hf_worker import HFWorker
    folder = tmp_path / "q"
    plain = _build_nvfp4(folder)
    worker = HFWorker(str(folder), device="cpu")
    worker.load()
    try:
        proj = worker._model.model.layers[0].self_attn.q_proj
        assert proj.weight.dtype == torch.float32
        with torch.no_grad():
            logits = worker._model(torch.tensor([[1, 2, 3]])).logits
        assert logits.shape == (1, 3, 64)
    finally:
        worker.unload()
    assert (_hf_fp8.compressed_expanded_bytes(str(folder), 4)
            == _hf_fp8.expanded_bf16_bytes(str(plain)) * 2)


def _write_tokenizer(folder):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    tok = Tokenizer(models.WordLevel({f"t{i}": i for i in range(64)}, unk_token="t0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="t0", bos_token="t1",
                            eos_token="t2", pad_token="t0").save_pretrained(folder)


def _write_safetensors(path, tensors):
    import struct
    header, blob = {}, b""
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [len(blob), len(blob) + len(raw)]}
        blob += raw
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob)


class TestExpandedSizeParsing:
    def _folder(self, tmp_path, tensors, groups=None):
        folder = tmp_path / "m"
        folder.mkdir()
        qc = {"quant_method": "compressed-tensors"}
        if groups is not None:
            qc["config_groups"] = groups
        (folder / "config.json").write_text(json.dumps({"quantization_config": qc}),
                                            encoding="utf-8")
        _write_safetensors(folder / "model.safetensors", tensors)
        return str(folder)

    def test_a_packed_weight_without_its_shape_uses_the_declared_bit_width(self, tmp_path):
        from localm.inference.backends._hf_fp8 import compressed_expanded_bytes
        path = self._folder(tmp_path, {
            "l.weight_packed": ("U8", [8, 4], b"\0" * 32),
            "l.weight_scale": ("BF16", [8, 1], b"\0" * 16),
            "norm.weight": ("BF16", [8], b"\0" * 16)},
            groups={"g": {"weights": {"num_bits": 4}}})
        assert compressed_expanded_bytes(path, 2) == (8 * 8 + 8) * 2
        assert compressed_expanded_bytes(path, 4) == (8 * 8 + 8) * 4

    def test_the_shape_tensor_wins_over_the_packed_size(self, tmp_path):
        import struct
        from localm.inference.backends._hf_fp8 import compressed_expanded_bytes
        path = self._folder(tmp_path, {
            "l.weight_packed": ("I32", [6, 1], b"\0" * 24),
            "l.weight_shape": ("I64", [2], struct.pack("<2q", 6, 7))})
        assert compressed_expanded_bytes(path, 2) == 6 * 7 * 2

    def test_mixed_bit_widths_without_a_shape_cannot_be_sized(self, tmp_path):
        from localm.inference.backends._hf_fp8 import compressed_expanded_bytes
        path = self._folder(tmp_path, {"l.weight_packed": ("U8", [8, 4], b"\0" * 32)},
                            groups={"a": {"weights": {"num_bits": 4}},
                                    "b": {"weights": {"num_bits": 8}}})
        assert compressed_expanded_bytes(path, 2) is None

    def test_no_weight_files_cannot_be_sized(self, tmp_path):
        from localm.inference.backends._hf_fp8 import compressed_expanded_bytes
        folder = tmp_path / "m"
        folder.mkdir()
        assert compressed_expanded_bytes(str(folder), 2) is None
