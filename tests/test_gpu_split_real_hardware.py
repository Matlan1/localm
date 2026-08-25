# SPDX-License-Identifier: AGPL-3.0-or-later
"""REAL end-to-end tests of the split-GPU load/inference path across two genuinely distinct, real GPU devices (real VRAM, real allocator, real driver) - Tier 2 of the layered testing strategy in issues/issues.txt's GPU-SPLIT-TESTING entry and dev-notes/split-gpu-testing-research-2026-07-13.md."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.real_multi_gpu_hardware]

_TINY_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
_TINY_FILE = "SmolLM2-135M-Instruct-Q4_K_M.gguf"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "tier2_gpu_split"))


def _split_indices() -> list:
    raw = os.environ.get("LOCALM_TEST_MULTI_GPU_SPLIT_INDICES", "0,1")
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _smi_tool() -> str:
    """Which vendor tool to use for an INDEPENDENT VRAM reading - independent of list_gpus()/vram_capacity(), which are themselves under test here, so the raw delta measurement below does not just check the code against itself."""
    if shutil.which("nvidia-smi"):
        return "nvidia-smi"
    if shutil.which("rocm-smi"):
        return "rocm-smi"
    pytest.skip("neither nvidia-smi nor rocm-smi found on PATH - cannot "
                "independently measure per-device VRAM on this box")


def _device_memory_used_bytes(tool: str, index: int) -> int:
    if tool == "nvidia-smi":
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-i", str(index)],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip()
        return int(out) * 1024 * 1024
    # rocm-smi's --json key text has drifted across ROCm releases; match
    # loosely (case-insensitive "used" + "memory") rather than a single exact
    # key name, and fail with the raw output if nothing matches so a real
    # drift is an actionable error, not a wrong silent number.
    out = subprocess.run(
        ["rocm-smi", "--showmeminfo", "vram", "--json", "-d", str(index)],
        capture_output=True, text=True, timeout=15, check=True,
    ).stdout
    data = json.loads(out)
    card = next(iter(data.values()))
    used_key = next(
        (k for k in card if "used" in k.lower() and "memory" in k.lower()), None)
    if used_key is None:
        raise RuntimeError(
            f"could not find a 'used memory' key in rocm-smi --json output "
            f"for device {index}: {data!r}")
    return int(card[used_key])


@pytest.fixture(scope="module")
def two_real_gpus():
    from localm.discover import list_gpus
    gpus = list_gpus()
    if len(gpus) < 2:
        pytest.skip(
            f"list_gpus() detected fewer than 2 devices ({gpus!r}) - this "
            f"file needs a real 2-GPU box (the Tier 2 rental); see "
            f"scripts/tier2_gpu_split/README.md")
    return gpus


@pytest.fixture(scope="module")
def tiny_model_path():
    try:
        from localm.inference.backends.llamacpp import _loader
        _loader.load_lib()
    except Exception as e:
        pytest.skip(f"native llama runtime not provisioned (run 'localm setup-llama'): {e}")
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo_id=_TINY_REPO, filename=_TINY_FILE)
    except Exception as e:
        pytest.skip(f"could not fetch {_TINY_REPO}/{_TINY_FILE}: {e}")


def test_split_load_honors_configured_ratio(two_real_gpus, tiny_model_path, monkeypatch):
    """PASS/FAIL oracle #1 - 'real allocator behaviour': a deliberately lopsided (9:1) split across two REAL devices must land ~90% of the VRAM delta on device 0, measured INDEPENDENTLY via nvidia-smi/rocm-smi (not through list_gpus()/vram_capacity(), which other tests in this file exercise directly)."""
    indices = _split_indices()
    assert len(indices) >= 2, "LOCALM_TEST_MULTI_GPU_SPLIT_INDICES must name >= 2 devices"
    tool = _smi_tool()
    ratios = [9.0] + [1.0] * (len(indices) - 1)
    monkeypatch.setattr(
        "localm.config.load_config",
        lambda: {"gpu_split_indices": indices, "gpu_split_ratios": ratios},
    )

    before = [_device_memory_used_bytes(tool, i) for i in indices]

    from localm.inference.backends.llamacpp.llama import LlamaCpp
    llm = LlamaCpp(tiny_model_path, n_ctx=512, n_gpu_layers=99)
    try:
        after = [_device_memory_used_bytes(tool, i) for i in indices]
        deltas = [max(0, a - b) for a, b in zip(after, before)]
        total = sum(deltas)
        assert total > 0, (
            f"no measurable VRAM delta on any configured device after load "
            f"(before={before}, after={after}) - the split may not have "
            f"landed on these devices at all")
        dev0_share = deltas[0] / total
        assert dev0_share >= 0.70, (
            f"configured gpu_split_ratios={ratios} asked for ~90% of the VRAM "
            f"delta on device {indices[0]}, actual share {dev0_share:.0%} "
            f"(deltas={deltas}) - the configured ratio was not honored")

        result = llm.create_chat_completion(
            [{"role": "user", "content": "In one short sentence, what color is a clear daytime sky?"}],
            max_tokens=40, temperature=0.0, seed=1, stream=False,
        )
        text = result["choices"][0]["message"]["content"].strip()
        assert len(text) >= 10 and any(c.isalpha() for c in text), f"incoherent split-load output: {text!r}"
    finally:
        llm.close()


def test_combined_vram_budgeting_admits_model_too_big_for_one_device(two_real_gpus, monkeypatch):
    """PASS/FAIL oracle #2 - 'combined-VRAM budgeting under real pressure': proves the #770 fix (localm/inference/backends/llamacpp/_sizing.py's combined-capacity budgeting) holds on REAL hardware."""
    indices = _split_indices()
    assert len(indices) >= 2
    monkeypatch.setattr(
        "localm.config.load_config",
        lambda: {"gpu_split_indices": indices, "gpu_split_ratios": None},
    )

    from localm.discover import list_gpus, vram_capacity
    combined = vram_capacity(combined_only=True)
    if combined.get("devices", 0) < 2 or "free" not in combined:
        pytest.skip(
            f"vram_capacity(combined_only=True) did not return a measurable "
            f"2+-device combined free figure ({combined!r}) - cannot pick a "
            f"model to exercise the combined-budgeting path")

    gpus = {g["index"]: g for g in list_gpus()}
    device_frees = [gpus[i]["free"] for i in indices
                    if i in gpus and gpus[i].get("free") is not None]
    if len(device_frees) < 2:
        pytest.skip("could not read real free VRAM for both configured devices")
    single_free = min(device_frees)
    combined_free = combined["free"]

    from model_selection import select_model_for_combined_test
    candidate = select_model_for_combined_test(single_free, combined_free)
    if candidate is None:
        pytest.skip(
            f"no candidate GGUF in model_selection.CANDIDATE_TABLE spans "
            f"(single_free={single_free}, combined_free={combined_free}) - "
            f"see scripts/tier2_gpu_split/model_selection.py")

    # Download EVERY part before loading the first: a split-GGUF quant (most
    # candidates big enough to matter here ship as 2+ sibling files - see
    # model_selection.py's module docstring) needs all its parts present
    # alongside each other for localm's own split-GGUF loader
    # (localm/model_manager.py's missing_split_parts) to find them.
    from huggingface_hub import hf_hub_download
    part_paths = [hf_hub_download(repo_id=candidate.repo_id, filename=part)
                  for part in candidate.parts]
    model_path = part_paths[0]

    from localm.discover import applied_split_device_count
    from localm.inference.backends.gguf import GgufBackend
    backend = GgufBackend(model_path, n_ctx=512, n_gpu_layers=99)
    try:
        backend.load()  # runs the REAL _check_vram() preflight before any native call
        assert applied_split_device_count() >= 2, (
            "the loader did not apply a 2+-device split for a model sized "
            "specifically to need it - the combined-budgeting path was not "
            "actually exercised")
        text = "".join(backend.chat_stream(
            [{"role": "user", "content": "In one short sentence, name a primary color."}],
            max_tokens=40, temperature=0.0, seed=1,
        ))
        assert len(text) >= 5 and any(c.isalpha() for c in text), f"incoherent output: {text!r}"
    finally:
        backend.unload()


def test_genuine_oom_refused_cleanly_at_split_boundary(two_real_gpus, tiny_model_path, monkeypatch):
    """PASS/FAIL oracle #3 - 'a genuine OOM at the split boundary': a request that genuinely cannot fit even the COMBINED split capacity must be refused safely, fast, and cleanly - never a hang or a driver crash."""
    indices = _split_indices()
    assert len(indices) >= 2
    monkeypatch.setattr(
        "localm.config.load_config",
        lambda: {"gpu_split_indices": indices, "gpu_split_ratios": None},
    )

    from localm.discover import vram_capacity
    combined = vram_capacity(combined_only=True)
    if combined.get("devices", 0) < 2 or "total" not in combined:
        pytest.skip(
            f"vram_capacity(combined_only=True) did not return a measurable "
            f"combined total ({combined!r})")

    # uint32-safe (max ~4.29e9) and, at any plausible per-token KV cost for
    # any model small enough to smoke-test with, guaranteed to dwarf a
    # real-world combined VRAM total on its own - this test only needs
    # "unambiguously exceeds", not a precisely-tuned boundary value.
    huge_ctx = 2_000_000_000

    from localm.inference.backends.gguf import GgufBackend
    start = time.monotonic()
    doomed = GgufBackend(tiny_model_path, n_ctx=huge_ctx, n_gpu_layers=99)
    with pytest.raises(RuntimeError, match="cannot fit across this split"):
        doomed.load()  # _check_vram() fires here, before any worker is spawned
    elapsed = time.monotonic() - start
    assert elapsed < 30, (
        f"the combined-capacity refusal took {elapsed:.1f}s (expected < 30s "
        f"for a pure arithmetic preflight) - a slow refusal suggests this "
        f"fell through to an actual native allocation attempt instead of "
        f"being caught by the Python-side preflight")

    # The GPU/driver must still be healthy after the refusal.
    healthy = GgufBackend(tiny_model_path, n_ctx=512, n_gpu_layers=99)
    healthy.load()
    healthy.unload()


def test_amd_rocm_hip_split_path_executes(two_real_gpus, tiny_model_path, monkeypatch):
    """PASS/FAIL oracle #4 - 'the amd-rocm/HIP split path that has never once been executed' (issues/issues.txt GPU-SPLIT-TESTING)."""
    if os.environ.get("LOCALM_TEST_TIER2_BACKEND") != "amd-rocm":
        pytest.skip(
            "only meaningful on the amd-rocm/HIP backend run - set "
            "LOCALM_TEST_TIER2_BACKEND=amd-rocm (run_gate.py sets this "
            "automatically for the AMD/Hot Aisle run)")
    indices = _split_indices()
    assert len(indices) >= 2
    tool = _smi_tool()
    if tool != "rocm-smi":
        pytest.skip(
            f"LOCALM_TEST_TIER2_BACKEND=amd-rocm but the detected VRAM tool "
            f"is {tool!r}, not rocm-smi - inconsistent environment")
    monkeypatch.setattr(
        "localm.config.load_config",
        lambda: {"gpu_split_indices": indices, "gpu_split_ratios": [1.0] * len(indices)},
    )
    before = [_device_memory_used_bytes(tool, i) for i in indices]

    from localm.inference.backends.llamacpp.llama import LlamaCpp
    llm = LlamaCpp(tiny_model_path, n_ctx=512, n_gpu_layers=99)
    try:
        after = [_device_memory_used_bytes(tool, i) for i in indices]
        deltas = [max(0, a - b) for a, b in zip(after, before)]
        assert all(d > 0 for d in deltas), (
            f"expected a positive VRAM delta on every configured AMD device "
            f"after a split load, got deltas={deltas} (before={before}, "
            f"after={after}) - the HIP split path did not actually spread "
            f"the load across both devices")
        result = llm.create_chat_completion(
            [{"role": "user", "content": "In one short sentence, what is 2 plus 2?"}],
            max_tokens=40, temperature=0.0, seed=1, stream=False,
        )
        text = result["choices"][0]["message"]["content"].strip()
        assert len(text) >= 5 and any(c.isalpha() for c in text), f"incoherent HIP split output: {text!r}"
    finally:
        llm.close()


def test_adversarial_uneven_ratio_and_short_device_refusal(two_real_gpus, tiny_model_path, monkeypatch):
    """PASS/FAIL oracle #5 - the research doc's explicitly-requested adversarial configs (dev-notes/split-gpu-testing-research-2026-07-13.md Tier 2 step 7): an even more extreme ratio than test #1, and a PINNED split sized so one device's computed share cannot fit that device's own real free VRAM. gpu_spli..."""
    indices = _split_indices()
    assert len(indices) >= 2
    tool = _smi_tool()

    # Part A: a more extreme ratio than test #1's 9:1.
    monkeypatch.setattr(
        "localm.config.load_config",
        lambda: {"gpu_split_indices": indices, "gpu_split_ratios": [99.0, 1.0]},
    )
    before = [_device_memory_used_bytes(tool, i) for i in indices]
    from localm.inference.backends.llamacpp.llama import LlamaCpp
    llm = LlamaCpp(tiny_model_path, n_ctx=512, n_gpu_layers=99)
    try:
        after = [_device_memory_used_bytes(tool, i) for i in indices]
        deltas = [max(0, a - b) for a, b in zip(after, before)]
        total = sum(deltas)
        assert total > 0, "no measurable VRAM delta for the 99:1 ratio load"
        dev0_share = deltas[0] / total
        assert dev0_share >= 0.85, (
            f"a 99:1 ratio should skew harder than test #1's 9:1 floor "
            f"(0.70) - got share {dev0_share:.0%} (deltas={deltas})")
    finally:
        llm.close()

    # Part B: gpu_split_shortfall() with a PINNED ratio and a required total
    # sized so ONE device's computed equal share exceeds ITS OWN real free
    # VRAM by a clear margin.
    from localm.discover import GPU_PROBE_OK, gpu_split_shortfall, list_gpus
    gpus = {g["index"]: g for g in list_gpus()}
    frees = [gpus[i]["free"] for i in indices
             if i in gpus and gpus[i].get("free") is not None]
    if len(frees) < 2:
        pytest.skip("could not read real free VRAM for both configured devices")
    monkeypatch.setattr(
        "localm.config.load_config",
        lambda: {"gpu_split_indices": indices, "gpu_split_ratios": [1.0] * len(indices)},
    )
    smallest_free = min(frees)
    required_total = int((smallest_free + 4 * 1024 ** 3) * len(indices))
    shortfall, status = gpu_split_shortfall(required_total, return_status=True)
    if status != GPU_PROBE_OK:
        pytest.skip(f"GPU probe was not fresh (status={status!r}) - cannot "
                    f"assert on a shortfall computed from stale data")
    assert shortfall, (
        f"gpu_split_shortfall({required_total}) with a pinned equal split "
        f"reported no shortfall, even though the computed per-device share "
        f"({required_total // len(indices)}) exceeds the smallest configured "
        f"device's real free VRAM ({smallest_free}) by design - the "
        f"refuse-when-short path did not fire against real hardware")
