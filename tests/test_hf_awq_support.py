# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for native HuggingFace AWQ model loading and inference support."""

import json
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from localm.inference.backends.awq import (
    NativeAWQLinear,
    _try_create_native_awq_quantizer_cls,
    register_native_awq_quantizer,
)


def test_native_awq_linear_initialization_and_shapes():
    """Verify NativeAWQLinear allocates proper buffer shapes and properties."""
    in_f, out_f = 256, 512
    group_size = 128
    bits = 4
    layer = NativeAWQLinear(in_features=in_f, out_features=out_f, bias=True, group_size=group_size, bits=bits)

    assert layer.in_features == in_f
    assert layer.out_features == out_f
    assert layer.group_size == group_size
    assert layer.bits == bits

    # Packed buffer shapes
    assert layer.qweight.shape == (in_f, out_f // 8)
    assert layer.qweight.dtype == torch.int32
    assert layer.qzeros.shape == (in_f // group_size, out_f // 8)
    assert layer.qzeros.dtype == torch.int32
    assert layer.scales.shape == (in_f // group_size, out_f)
    assert layer.bias is not None
    assert layer.bias.shape == (out_f,)


def test_native_awq_linear_dequantize_numerical_correctness():
    """Verify dequantize() matches reference AWQ unpacking and scaling math."""
    in_f, out_f = 128, 256
    group_size = 64
    bits = 4

    layer = NativeAWQLinear(in_features=in_f, out_features=out_f, bias=False, group_size=group_size, bits=bits)

    # Populate deterministic packed data
    torch.manual_seed(42)
    layer.qweight.copy_(torch.randint(0, 0x7FFFFFFF, layer.qweight.shape, dtype=torch.int32))
    layer.qzeros.copy_(torch.randint(0, 0x7FFFFFFF, layer.qzeros.shape, dtype=torch.int32))
    layer.scales.copy_(torch.randn(layer.scales.shape, dtype=torch.float16))

    dequant = layer.dequantize()
    assert dequant.shape == (in_f, out_f)
    assert dequant.dtype == torch.float16

    # Reference manual calculation
    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32)
    w_int = ((layer.qweight.unsqueeze(-1) >> shifts) & 0x0F).reshape(in_f, out_f)
    z_int = ((layer.qzeros.unsqueeze(-1) >> shifts) & 0x0F).reshape(in_f // group_size, out_f)
    ref_zeros = z_int.repeat_interleave(group_size, dim=0)[:in_f]
    ref_scales = layer.scales.repeat_interleave(group_size, dim=0)[:in_f]
    expected = (w_int - ref_zeros).to(torch.float16) * ref_scales

    torch.testing.assert_close(dequant, expected)


def test_native_awq_linear_forward_pass():
    """Verify forward() performs correct matrix multiplication and bias addition."""
    in_f, out_f = 64, 128
    group_size = 32
    layer = NativeAWQLinear(in_features=in_f, out_features=out_f, bias=True, group_size=group_size, dtype=torch.float32)

    torch.manual_seed(100)
    layer.qweight.copy_(torch.randint(0, 0x7FFFFFFF, layer.qweight.shape, dtype=torch.int32))
    layer.qzeros.copy_(torch.randint(0, 0x7FFFFFFF, layer.qzeros.shape, dtype=torch.int32))
    layer.scales.copy_(torch.randn(layer.scales.shape, dtype=torch.float32))
    layer.bias.copy_(torch.randn(layer.bias.shape, dtype=torch.float32))

    x = torch.randn((2, 8, in_f), dtype=torch.float32)
    out = layer(x)

    assert out.shape == (2, 8, out_f)
    assert out.dtype == torch.float32

    # Verify equivalent to F.linear with dequantized weights
    expected = F.linear(x, layer.dequantize(dtype=torch.float32).t(), layer.bias)
    torch.testing.assert_close(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm not available")
def test_native_awq_linear_cuda_execution():
    """Verify NativeAWQLinear runs on GPU (CUDA/ROCm) without errors."""
    in_f, out_f = 128, 256
    layer = NativeAWQLinear(in_features=in_f, out_features=out_f, bias=True, device=torch.device("cuda"))

    layer.qweight.copy_(torch.randint(0, 0x7FFFFFFF, layer.qweight.shape, dtype=torch.int32, device="cuda"))
    layer.qzeros.copy_(torch.randint(0, 0x7FFFFFFF, layer.qzeros.shape, dtype=torch.int32, device="cuda"))
    layer.scales.copy_(torch.randn(layer.scales.shape, dtype=torch.float16, device="cuda"))
    layer.bias.copy_(torch.randn(layer.bias.shape, dtype=torch.float16, device="cuda"))

    x = torch.randn((1, 4, in_f), dtype=torch.float16, device="cuda")
    out = layer(x)

    assert out.device.type == "cuda"
    assert out.dtype == torch.float16
    assert out.shape == (1, 4, out_f)


def test_native_awq_quantizer_replaces_layers():
    """Verify NativeAwqQuantizer replaces transformer linear layers on model skeleton."""
    quantizer_cls = _try_create_native_awq_quantizer_cls()
    assert quantizer_cls is not None

    class DummyConfig:
        group_size = 128
        bits = 4
        quant_method = "awq"

    class DummyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64, bias=False)
            self.v_proj = nn.Linear(64, 64, bias=True)

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = DummyBlock()
            self.lm_head = nn.Linear(64, 100, bias=False)
            self.visual_layer = nn.Linear(64, 64, bias=False)
            self.in_proj_a = nn.Linear(64, 64, bias=False)

    model = DummyModel()
    quantizer = quantizer_cls(DummyConfig())
    quantizer._process_model_before_weight_loading(model)

    assert isinstance(model.block.q_proj, NativeAWQLinear)
    assert isinstance(model.block.v_proj, NativeAWQLinear)
    # lm_head, visual, and in_proj_a are preserved in full precision
    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, NativeAWQLinear)
    assert isinstance(model.visual_layer, nn.Linear)
    assert not isinstance(model.visual_layer, NativeAWQLinear)
    assert isinstance(model.in_proj_a, nn.Linear)
    assert not isinstance(model.in_proj_a, NativeAWQLinear)


def test_register_native_awq_quantizer_in_transformers():
    """Verify register_native_awq_quantizer injects into transformers quantizer mapping."""
    registered = register_native_awq_quantizer()
    assert registered is True

    from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING
    assert "awq" in AUTO_QUANTIZER_MAPPING
    quant_cls = AUTO_QUANTIZER_MAPPING["awq"]
    assert quant_cls.__name__ == "NativeAwqQuantizer"


def test_native_awq_automodel_loading_and_inference(tmp_path):
    """Verify AutoModelForCausalLM loads and runs an AWQ checkpoint via NativeAwqQuantizer."""
    from transformers import AutoModelForCausalLM
    from safetensors.torch import save_file

    register_native_awq_quantizer()

    model_dir = tmp_path / "dummy_qwen2_awq_model"
    model_dir.mkdir()

    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "vocab_size": 100,
        "quantization_config": {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 32,
            "zero_point": True,
            "version": "gemm",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    group_size = 32
    pack_factor = 8

    def make_awq_tensors(in_f, out_f):
        qw = torch.randint(0, 0x7FFFFFFF, (in_f, out_f // pack_factor), dtype=torch.int32)
        qz = torch.randint(0, 0x7FFFFFFF, (in_f // group_size, out_f // pack_factor), dtype=torch.int32)
        sc = torch.randn((in_f // group_size, out_f), dtype=torch.float16)
        b = torch.zeros(out_f, dtype=torch.float16)
        return qw, qz, sc, b

    q_qw, q_qz, q_sc, q_b = make_awq_tensors(64, 64)
    k_qw, k_qz, k_sc, k_b = make_awq_tensors(64, 64)
    v_qw, v_qz, v_sc, v_b = make_awq_tensors(64, 64)
    o_qw, o_qz, o_sc, o_b = make_awq_tensors(64, 64)
    gate_qw, gate_qz, gate_sc, _ = make_awq_tensors(64, 128)
    up_qw, up_qz, up_sc, _ = make_awq_tensors(64, 128)
    down_qw, down_qz, down_sc, _ = make_awq_tensors(128, 64)

    tensors = {
        "model.embed_tokens.weight": torch.randn((100, 64), dtype=torch.float16),
        "model.norm.weight": torch.ones(64, dtype=torch.float16),
        "lm_head.weight": torch.randn((100, 64), dtype=torch.float16),
        "model.layers.0.input_layernorm.weight": torch.ones(64, dtype=torch.float16),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(64, dtype=torch.float16),
        "model.layers.0.self_attn.q_proj.qweight": q_qw,
        "model.layers.0.self_attn.q_proj.qzeros": q_qz,
        "model.layers.0.self_attn.q_proj.scales": q_sc,
        "model.layers.0.self_attn.q_proj.bias": q_b,
        "model.layers.0.self_attn.k_proj.qweight": k_qw,
        "model.layers.0.self_attn.k_proj.qzeros": k_qz,
        "model.layers.0.self_attn.k_proj.scales": k_sc,
        "model.layers.0.self_attn.k_proj.bias": k_b,
        "model.layers.0.self_attn.v_proj.qweight": v_qw,
        "model.layers.0.self_attn.v_proj.qzeros": v_qz,
        "model.layers.0.self_attn.v_proj.scales": v_sc,
        "model.layers.0.self_attn.v_proj.bias": v_b,
        "model.layers.0.self_attn.o_proj.qweight": o_qw,
        "model.layers.0.self_attn.o_proj.qzeros": o_qz,
        "model.layers.0.self_attn.o_proj.scales": o_sc,
        "model.layers.0.mlp.gate_proj.qweight": gate_qw,
        "model.layers.0.mlp.gate_proj.qzeros": gate_qz,
        "model.layers.0.mlp.gate_proj.scales": gate_sc,
        "model.layers.0.mlp.up_proj.qweight": up_qw,
        "model.layers.0.mlp.up_proj.qzeros": up_qz,
        "model.layers.0.mlp.up_proj.scales": up_sc,
        "model.layers.0.mlp.down_proj.qweight": down_qw,
        "model.layers.0.mlp.down_proj.qzeros": down_qz,
        "model.layers.0.mlp.down_proj.scales": down_sc,
    }

    save_file(tensors, str(model_dir / "model.safetensors"))

    # Load via AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float16)
    assert model is not None

    # Perform a forward pass
    input_ids = torch.tensor([[1, 5, 10, 20]], dtype=torch.long)
    outputs = model(input_ids)
    assert outputs.logits is not None
    assert outputs.logits.shape == (1, 4, 100)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm not available")
def test_native_awq_automodel_gpu_loading_and_generation(tmp_path):
    """Verify AutoModelForCausalLM loads and performs GPU generation with device_map='cuda'."""
    from transformers import AutoModelForCausalLM
    from safetensors.torch import save_file

    register_native_awq_quantizer()

    model_dir = tmp_path / "dummy_qwen2_awq_gpu_model"
    model_dir.mkdir()

    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "vocab_size": 100,
        "quantization_config": {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 32,
            "zero_point": True,
            "version": "gemm",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    group_size = 32
    pack_factor = 8

    def make_awq_tensors(in_f, out_f):
        qw = torch.randint(0, 0x7FFFFFFF, (in_f, out_f // pack_factor), dtype=torch.int32)
        qz = torch.randint(0, 0x7FFFFFFF, (in_f // group_size, out_f // pack_factor), dtype=torch.int32)
        sc = torch.randn((in_f // group_size, out_f), dtype=torch.float16)
        b = torch.zeros(out_f, dtype=torch.float16)
        return qw, qz, sc, b

    q_qw, q_qz, q_sc, q_b = make_awq_tensors(64, 64)
    k_qw, k_qz, k_sc, k_b = make_awq_tensors(64, 64)
    v_qw, v_qz, v_sc, v_b = make_awq_tensors(64, 64)
    o_qw, o_qz, o_sc, o_b = make_awq_tensors(64, 64)
    gate_qw, gate_qz, gate_sc, _ = make_awq_tensors(64, 128)
    up_qw, up_qz, up_sc, _ = make_awq_tensors(64, 128)
    down_qw, down_qz, down_sc, _ = make_awq_tensors(128, 64)

    tensors = {
        "model.embed_tokens.weight": torch.randn((100, 64), dtype=torch.float16),
        "model.norm.weight": torch.ones(64, dtype=torch.float16),
        "lm_head.weight": torch.randn((100, 64), dtype=torch.float16),
        "model.layers.0.input_layernorm.weight": torch.ones(64, dtype=torch.float16),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(64, dtype=torch.float16),
        "model.layers.0.self_attn.q_proj.qweight": q_qw,
        "model.layers.0.self_attn.q_proj.qzeros": q_qz,
        "model.layers.0.self_attn.q_proj.scales": q_sc,
        "model.layers.0.self_attn.q_proj.bias": q_b,
        "model.layers.0.self_attn.k_proj.qweight": k_qw,
        "model.layers.0.self_attn.k_proj.qzeros": k_qz,
        "model.layers.0.self_attn.k_proj.scales": k_sc,
        "model.layers.0.self_attn.k_proj.bias": k_b,
        "model.layers.0.self_attn.v_proj.qweight": v_qw,
        "model.layers.0.self_attn.v_proj.qzeros": v_qz,
        "model.layers.0.self_attn.v_proj.scales": v_sc,
        "model.layers.0.self_attn.v_proj.bias": v_b,
        "model.layers.0.self_attn.o_proj.qweight": o_qw,
        "model.layers.0.self_attn.o_proj.qzeros": o_qz,
        "model.layers.0.self_attn.o_proj.scales": o_sc,
        "model.layers.0.mlp.gate_proj.qweight": gate_qw,
        "model.layers.0.mlp.gate_proj.qzeros": gate_qz,
        "model.layers.0.mlp.gate_proj.scales": gate_sc,
        "model.layers.0.mlp.up_proj.qweight": up_qw,
        "model.layers.0.mlp.up_proj.qzeros": up_qz,
        "model.layers.0.mlp.up_proj.scales": up_sc,
        "model.layers.0.mlp.down_proj.qweight": down_qw,
        "model.layers.0.mlp.down_proj.qzeros": down_qz,
        "model.layers.0.mlp.down_proj.scales": down_sc,
    }

    save_file(tensors, str(model_dir / "model.safetensors"))

    # Load on GPU with device_map='cuda'
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float16, device_map="cuda")
    input_ids = torch.tensor([[1, 5, 10, 20]], dtype=torch.long, device="cuda")
    outputs = model(input_ids)
    assert outputs.logits.device.type == "cuda"
    assert outputs.logits.shape == (1, 4, 100)

    # Test generation (auto-regressive token sampling)
    generated = model.generate(input_ids, max_new_tokens=5, do_sample=False)
    assert generated.shape == (1, 9)
    assert generated.device.type == "cuda"

