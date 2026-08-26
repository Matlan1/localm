# SPDX-License-Identifier: AGPL-3.0-or-later
"""The PyTorch variant (for the HuggingFace/transformers backend) follows the
user's chosen llama.cpp BACKEND, not just the detected GPU vendor.

Choosing torch from the DETECTED vendor alone gives an AMD box whose user picked
``[2] vulkan`` the AMD ROCm torch stack, and on Windows the ROCm extra hard-pins
the gfx103X wheel (pyproject ``[tool.uv.sources]``).

`recommended_torch_variant(backend, det)` is the single shared policy both
setup.bat and setup.sh call (via ``python -m localm.hwdetect torch <backend>``).
Returns one of "cuda" | "rocm" | "xpu" | "none".
"""

from __future__ import annotations

import pytest

from localm import hwdetect

AMD = hwdetect.Detection(vendors=["amd"], gpu_names="amd radeon rx 6900 xt")
NVIDIA = hwdetect.Detection(vendors=["nvidia"], gpu_names="nvidia geforce rtx 4090")
INTEL = hwdetect.Detection(vendors=["intel"], gpu_names="intel arc a770")
NOGPU = hwdetect.Detection(vendors=[], gpu_names="")
MIXED = hwdetect.Detection(vendors=["nvidia", "amd"], gpu_names="rtx 4090 / radeon")


def tv(backend, det):
    return hwdetect.recommended_torch_variant(backend, det)


# --------------------------- vendor-specific backend picks ---------------- #

def test_cuda_backend_always_cuda():
    assert tv("cuda", NVIDIA) == "cuda"
    assert tv("cuda", NOGPU) == "cuda"          # the user explicitly asked for CUDA


def test_rocm_backend_always_rocm():
    assert tv("amd-rocm", AMD) == "rocm"
    assert tv("hip", AMD) == "rocm"             # Linux AMD backend name
    assert tv("amd-rocm", NOGPU) == "rocm"      # explicit ROCm pick is honored


# --------------------------- vendor-neutral backend picks ---------------- #

def test_vulkan_backend_never_installs_rocm_even_on_amd():
    # AMD hardware, user picked vulkan.
    assert tv("vulkan", AMD) == "none"
    assert tv("vulkan", AMD) != "rocm"


def test_cpu_backend_never_installs_a_vendor_torch():
    assert tv("cpu", AMD) == "none"
    assert tv("cpu", NVIDIA) == "none"


def test_vulkan_on_nvidia_keeps_cuda_torch():
    # NVIDIA's CUDA torch is a clean pip wheel (no toolkit), so a vendor-neutral
    # pick still routes HF torch to cuda.
    assert tv("vulkan", NVIDIA) == "cuda"
    assert tv("vulkan", MIXED) == "cuda"        # any NVIDIA present -> cuda


def test_vulkan_no_gpu_is_none_but_intel_is_xpu():
    assert tv("vulkan", NOGPU) == "none"
    # Intel Arc/Xe gets the native PyTorch XPU wheels (they self-provision the oneAPI
    # runtime), so a vendor-neutral pick on Intel routes HF torch to xpu, not none.
    assert tv("vulkan", INTEL) == "xpu"


def test_own_backend_follows_vendor_but_never_forces_rocm():
    assert tv("own", NVIDIA) == "cuda"
    assert tv("own", INTEL) == "xpu"            # Intel -> the native XPU wheels
    assert tv("own", AMD) == "none"             # power user: do not force ROCm
    assert tv("own", NOGPU) == "none"


# --------------------------- the invariant (negative test of the oracle) -- #

@pytest.mark.parametrize("det", [AMD, NVIDIA, INTEL, NOGPU, MIXED])
@pytest.mark.parametrize("vendor_neutral", ["vulkan", "cpu"])
def test_vendor_neutral_pick_never_yields_rocm(vendor_neutral, det):
    """A vendor-neutral or no-GPU runtime pick must never drag in the ROCm
    stack, whatever hardware is detected."""
    assert tv(vendor_neutral, det) != "rocm"


def test_old_naive_vendor_gate_would_have_been_wrong():
    """A vendor-only gate answers rocm for an AMD box regardless of the backend
    pick; the policy answers none for a vulkan pick on the same box."""
    naive_vendor_gate = "rocm" if "amd" in AMD.vendors else "none"
    assert naive_vendor_gate == "rocm"               # the naive result
    assert tv("vulkan", AMD) == "none"               # what tv() returns


def test_variant_is_always_a_known_token():
    for backend in ("cuda", "amd-rocm", "hip", "vulkan", "cpu", "own", "metal", ""):
        for det in (AMD, NVIDIA, INTEL, NOGPU, MIXED):
            assert hwdetect.recommended_torch_variant(backend, det) in (
                "cuda", "rocm", "xpu", "none")


# --------------------------- the CLI the shells call --------------------- #

def test_cli_torch_mode_is_deterministic_for_hw_independent_backends(capsys):
    # cpu/amd-rocm do not depend on the test machine's hardware, so these are
    # safe to assert exactly regardless of where the suite runs.
    assert hwdetect.main(["torch", "cpu"]) == 0
    assert capsys.readouterr().out.strip() == "none"
    assert hwdetect.main(["torch", "amd-rocm"]) == 0
    assert capsys.readouterr().out.strip() == "rocm"
    assert hwdetect.main(["torch", "cuda"]) == 0
    assert capsys.readouterr().out.strip() == "cuda"


def test_cli_torch_mode_emits_one_known_token(capsys):
    assert hwdetect.main(["torch", "vulkan"]) == 0
    out = capsys.readouterr().out.strip()
    assert out in ("cuda", "xpu", "none")        # depends on this box's GPU vendor


def test_cli_default_mode_unchanged(capsys):
    # The no-arg invocation the installers already use must still print
    # "<vendor> <install-backend>" (two tokens).
    assert hwdetect.main() == 0
    parts = capsys.readouterr().out.strip().split()
    assert len(parts) == 2
    assert parts[1] in ("vulkan", "cpu", "metal", "amd-rocm")


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
