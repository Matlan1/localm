# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native AWQ (Activation-aware Weight Quantization) dequantization and inference module.

Enables loading and inference of HuggingFace AWQ quantized checkpoints on all
platforms (including Windows ROCm, Linux ROCm, NVIDIA CUDA, Intel XPU, and CPU)
without external compiled dependencies like gptqmodel, autoawq, or torchao.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from localm.debuglog import logger


class NativeAWQLinear(nn.Module):
    """Linear layer holding 4-bit packed AWQ weights, dequantizing on demand during forward pass."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 128,
        bits: int = 4,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size if group_size > 0 else in_features
        self.bits = bits
        self.pack_factor = 32 // bits

        # Register packed weight buffers (populated by safetensors on model load)
        self.register_buffer(
            "qweight",
            torch.empty(
                (in_features, out_features // self.pack_factor),
                dtype=torch.int32,
                device=device,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.empty(
                (in_features // self.group_size, out_features // self.pack_factor),
                dtype=torch.int32,
                device=device,
            ),
        )
        self.register_buffer(
            "scales",
            torch.empty(
                (in_features // self.group_size, out_features),
                dtype=dtype,
                device=device,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.empty((out_features,), dtype=dtype, device=device),
            )
        else:
            self.bias = None

        self._is_hf_initialized = True

    def dequantize(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize 4-bit packed AWQ weights to a floating-point weight matrix."""
        device = self.qweight.device
        target_dtype = dtype or self.scales.dtype

        # 8 4-bit packed integers per 32-bit int
        shifts = torch.tensor(
            [0, 4, 8, 12, 16, 20, 24, 28],
            device=device,
            dtype=torch.int32,
        )

        w_int = ((self.qweight.unsqueeze(-1) >> shifts) & 0x0F).reshape(
            self.in_features, self.out_features
        )
        z_int = ((self.qzeros.unsqueeze(-1) >> shifts) & 0x0F).reshape(
            self.in_features // self.group_size, self.out_features
        )

        zeros = z_int.repeat_interleave(self.group_size, dim=0)[:self.in_features]
        scales = self.scales.repeat_interleave(self.group_size, dim=0)[:self.in_features]

        weight = (w_int - zeros).to(target_dtype) * scales.to(target_dtype)
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize(dtype=x.dtype)
        # F.linear expects weight shape (out_features, in_features)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight.t(), bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={self.bits}, group_size={self.group_size}"
        )


def _try_create_native_awq_quantizer_cls():
    """Create NativeAwqQuantizer subclassing HfQuantizer if transformers is present."""
    try:
        from transformers.quantizers.base import HfQuantizer
        from transformers.quantizers.quantizers_utils import should_convert_module
    except ImportError:
        return None

    class NativeAwqQuantizer(HfQuantizer):
        """Native AWQ quantizer that replaces Linear layers with NativeAWQLinear."""

        requires_calibration = False

        def __init__(self, quantization_config, **kwargs):
            super().__init__(quantization_config, **kwargs)

        def validate_environment(self, *args, **kwargs):
            # No external compiled packages (gptqmodel/autoawq) are required.
            pass

        def update_dtype(self, dtype):
            if dtype is None:
                return torch.float16
            return dtype

        def _process_model_before_weight_loading(self, model, **kwargs):
            group_size = getattr(self.quantization_config, "group_size", 128)
            bits = getattr(self.quantization_config, "bits", 4)
            try:
                base_skips = self.get_modules_to_not_convert(
                    model,
                    getattr(self.quantization_config, "modules_to_not_convert", None),
                    getattr(model, "_keep_in_fp32_modules", None),
                    add_default_skips=True,
                )
            except Exception:
                base_skips = list(
                    getattr(self.quantization_config, "modules_to_not_convert", None) or []
                ) + list(getattr(model, "_keep_in_fp32_modules", None) or [])

            # Multimodal and hybrid attention layers that remain unquantized in AWQ checkpoints
            extra_skips = [
                "visual",
                "vision_tower",
                "merger",
                "multi_modal_projector",
                "in_proj_a",
                "in_proj_b",
            ]
            self.modules_to_not_convert = list(set(base_skips + extra_skips))

            try:
                from transformers.pytorch_utils import Conv1D
                linear_classes = (nn.Linear, Conv1D)
            except ImportError:
                linear_classes = (nn.Linear,)

            replaced_count = 0
            for module_name, module in model.named_modules():
                if not isinstance(module, linear_classes):
                    continue
                if self.modules_to_not_convert and not should_convert_module(
                    module_name, self.modules_to_not_convert
                ):
                    continue
                if module_name.endswith("lm_head"):
                    continue

                parent = model
                sub_names = module_name.split(".")
                for sub in sub_names[:-1]:
                    parent = getattr(parent, sub)

                in_f = getattr(module, "in_features", None) or getattr(module, "nx", None)
                out_f = getattr(module, "out_features", None) or getattr(module, "nf", None)
                if in_f is None or out_f is None:
                    continue

                has_bias = module.bias is not None
                new_module = NativeAWQLinear(
                    in_features=in_f,
                    out_features=out_f,
                    bias=has_bias,
                    group_size=group_size,
                    bits=bits,
                )
                setattr(parent, sub_names[-1], new_module)
                replaced_count += 1

            logger.debug(
                "awq: replaced %d linear layers with NativeAWQLinear (group_size=%d, bits=%d)",
                replaced_count,
                group_size,
                bits,
            )

        def _process_model_after_weight_loading(self, model, **kwargs):
            return model

        def is_serializable(self):
            return True

        @property
        def is_trainable(self):
            return False

    return NativeAwqQuantizer


def register_native_awq_quantizer() -> bool:
    """Register NativeAwqQuantizer in transformers AUTO_QUANTIZER_MAPPING."""
    try:
        from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING

        quantizer_cls = _try_create_native_awq_quantizer_cls()
        if quantizer_cls is not None:
            AUTO_QUANTIZER_MAPPING["awq"] = quantizer_cls
            logger.debug("awq: registered NativeAwqQuantizer in AUTO_QUANTIZER_MAPPING")
            return True
    except Exception as e:
        logger.debug("awq: could not register NativeAwqQuantizer: %s", e)
    return False
