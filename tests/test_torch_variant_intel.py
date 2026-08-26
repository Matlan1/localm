# SPDX-License-Identifier: AGPL-3.0-or-later
"""Intel HF torch routing: a vendor-neutral runtime pick (vulkan/own) on an Intel box
routes the optional HuggingFace PyTorch to the native XPU wheels rather than
falling through to "none". The backend pick fully drives the variant for explicit
vendor picks, and a neutral pick NEVER pulls ROCm."""

from localm.hwdetect import Detection, recommended_torch_variant

_INTEL = Detection(vendors=["intel"], gpu_names="intel(r) arc(tm) a770")
_NVIDIA = Detection(vendors=["nvidia"], gpu_names="nvidia geforce rtx 4070")
_AMD = Detection(vendors=["amd"], gpu_names="amd radeon rx 6800")
_NONE = Detection(vendors=[])


def test_explicit_backend_picks_drive_the_variant():
    assert recommended_torch_variant("cuda", _AMD) == "cuda"      # honour the pick
    assert recommended_torch_variant("amd-rocm", _NVIDIA) == "rocm"
    assert recommended_torch_variant("hip", _NONE) == "rocm"
    assert recommended_torch_variant("sycl", _NONE) == "xpu"      # Intel GGUF -> Intel HF
    assert recommended_torch_variant("cpu", _NVIDIA) == "none"


def test_neutral_pick_routes_intel_to_xpu_not_none():
    # vulkan/own on an Intel box -> xpu.
    assert recommended_torch_variant("vulkan", _INTEL) == "xpu"
    assert recommended_torch_variant("own", _INTEL) == "xpu"


def test_neutral_pick_other_vendors_unchanged():
    assert recommended_torch_variant("vulkan", _NVIDIA) == "cuda"
    assert recommended_torch_variant("vulkan", _AMD) == "none"    # never rocm on a neutral pick
    assert recommended_torch_variant("vulkan", _NONE) == "none"
