# SPDX-License-Identifier: AGPL-3.0-or-later
r"""REAL end-to-end tests of the split-GPU load/inference path across two
genuinely distinct, real GPU devices (real VRAM, real allocator, real driver) -
Tier 2 of the layered testing strategy in issues/issues.txt's GPU-SPLIT-TESTING
entry and dev-notes/split-gpu-testing-research-2026-07-13.md. Tier 1
(tests/test_gpu_split_native_vulkan.py) proved the native split path exists and
works using a real GPU plus a free SOFTWARE second Vulkan device (Mesa
lavapipe) - by design it cannot touch real VRAM pressure/allocator/OOM behavior
(lavapipe is backed by system RAM, not a real second memory domain) or the
amd-rocm/HIP backend at all (no software HIP implementation exists). This file
is what closes that gap; it only means anything on a REAL 2-GPU box, owned or
rented. If you already have two GPUs, run it directly - nothing here needs a
rental (see PREREQUISITES below; the gate is two visible devices, an smi tool
and a provisioned runtime). See scripts/tier2_gpu_split/README.md for the exact
spec, and for the rental path when no such box is to hand.

PREREQUISITES (this file does not perform them - do them yourself on a box you
own, or scripts/tier2_gpu_split/provision_remote.sh does them for a rental):
  1. A real 2+ GPU Linux box (NVIDIA via the vulkan backend, or AMD via the
     amd-rocm backend), provisioned with localm + the matching native runtime
     (localm setup-llama --backend <vulkan|amd-rocm>) and, purely so this
     harness's own assertions can read real per-device free VRAM, a torch
     build that makes list_gpus()/vram_capacity() see both devices (discover.py
     has an nvidia-smi subprocess fallback but no rocm-smi equivalent, so the
     amd-rocm run specifically needs a real torch-ROCm-Linux build - see the
     provisioning script for why).
  2. Set LOCALM_TEST_REAL_MULTI_GPU=1 to opt in (tests/conftest.py's
     real_multi_gpu_hardware resource gate) - this is never set outside a Tier
     2 run, so this file is always skipped in the normal suite and in CI.
  3. Optionally LOCALM_TEST_MULTI_GPU_SPLIT_INDICES (default "0,1") if
     list_gpus() orders the two devices differently, and
     LOCALM_TEST_TIER2_BACKEND=amd-rocm|vulkan (run_gate.py sets this) so the
     amd-rocm-only test can skip cleanly on the other backend's run.

Unlike Tier 1 (gated on a Vulkan-loader env var that is only read once per
process), there is no comparable single-shot registration hazard here - real
CUDA/ROCm devices are visible to list_gpus()/torch at any point, so these tests
do not need to run in process isolation from each other.
"""

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
    """Which vendor tool to use for an INDEPENDENT VRAM reading - independent
    of list_gpus()/vram_capacity(), which are themselves under test here, so
    the raw delta measurement below does not just check the code against
    itself."""
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
    # rocm-smi's --json key text drifts across ROCm releases: match loosely
    # (case-insensitive "used" plus "memory") and fail with the raw output if
    # nothing matches.
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
    """PASS/FAIL oracle #1 - "real allocator behaviour": a deliberately
    lopsided (9:1) split across two REAL devices must land ~90% of the VRAM
    delta on device 0, measured INDEPENDENTLY via nvidia-smi/rocm-smi (not
    through list_gpus()/vram_capacity(), which other tests in this file
    exercise directly). Ports tests/test_gpu_split_native_vulkan.py's Tier 1
    oracle from a Vulkan-specific native log regex to a backend-agnostic VRAM
    delta, since a per-layer log format is not guaranteed on the cuda/amd-rocm
    builds the way it was confirmed for vulkan.

    PASS: dev0's share of the total VRAM delta >= 0.70, and a real chat
          completion returns coherent text.
    FAIL: an even/reversed split, no measurable delta, a hang, a crash, or
          incoherent output.
    """
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
    """PASS/FAIL oracle #2 - "combined-VRAM budgeting under real pressure":
    proves the #770 fix (localm/inference/backends/llamacpp/_sizing.py's
    combined-capacity budgeting) holds on REAL hardware. The 2026-07-21 checkup
    (dev-notes/checkup/REPORT-2026-07-21-gpu-split.md) proved this only at the
    code level ("mocked repro driving the REAL methods with only the VRAM
    reading... faked... NOT live-verified on real 2-GPU hardware (none here -
    this is exactly the Tier 2 gap)") - this is that live verification, using a
    real model whose size is picked (scripts/tier2_gpu_split/model_selection.py)
    from LIVE per-device free VRAM, not a hardcoded guess.

    Deliberately drives this through GgufBackend, NOT a raw LlamaCpp
    construction: _check_vram() (the preflight the #770 fix lives in) is called
    by GgufBackend.load() (localm/inference/backends/gguf.py:157) BEFORE it
    spawns the isolated worker - LlamaCpp.__init__ itself never calls it at all
    (an earlier draft of this test constructed LlamaCpp directly, which meant
    the "construction does not raise" assertion was trivially true regardless
    of whether the combined-budget arithmetic was even correct - caught by
    review before this ever ran for real).

    PASS: GgufBackend.load() does not raise (proves _check_vram did not
          hard-refuse), applied_split_device_count() >= 2 (proves a real
          2+-device split was actually applied, not degraded to single-device),
          and a real chat completion returns coherent text.
    FAIL: a RuntimeError refusal, applied_split_device_count() < 2, or
          incoherent/empty output.
    """
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

    # Download EVERY part before loading the first: localm's split-GGUF loader
    # (model_manager.missing_split_parts) needs all sibling parts present
    # alongside each other.
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
    """PASS/FAIL oracle #3 - "a genuine OOM at the split boundary": a request
    that genuinely cannot fit even the COMBINED split capacity must be refused
    safely, fast, and cleanly - never a hang or a driver crash. Reuses the tiny
    model but drives n_ctx absurdly high so the KV cache ALONE exceeds combined
    capacity, exercising _check_vram()'s real combined-capacity refusal
    (localm/inference/backends/llamacpp/_sizing.py:591-642) against REAL
    per-device numbers.

    Deliberately drives this through GgufBackend.load(), NOT a raw LlamaCpp
    construction: _check_vram() is only ever called by GgufBackend.load()
    (gguf.py:157) before it spawns the isolated worker - LlamaCpp.__init__
    itself never calls it (caught by review before this test ever ran for
    real: as originally written it constructed LlamaCpp directly, so the
    expected RuntimeError could never fire and the test would instead have
    attempted a REAL native allocation for a 2-billion-token KV cache on the
    rented box - precisely the "materially higher-risk exercise" the
    paragraph below says this test exists to avoid).

    Scoped deliberately to the Python-side preflight (a pure arithmetic check
    BEFORE any native allocation attempt, and BEFORE GgufBackend.load() ever
    spawns a worker process), not a forced native-allocator crash: the
    worker-subprocess isolation in
    localm/inference/backends/llamacpp/_worker.py exists precisely so a real
    native OOM/crash cannot take down the parent, but deliberately trying to
    trigger that on a metered rented box is a materially higher-risk exercise
    than proving the documented preflight refusal - see
    scripts/tier2_gpu_split/README.md for why that is out of scope here.

    PASS: RuntimeError containing "cannot fit across this split" (the exact
          combined-capacity wording _check_vram raises), within 30s (it is
          pure arithmetic, no native call attempted, no worker spawned); a
          trivial follow-up single-device-sized load still succeeds
          (GPU/driver not wedged).
    FAIL: no exception, a hang past 30s, a crash, or the follow-up load fails.
    """
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

    # uint32-safe (max ~4.29e9) and large enough to dwarf any real combined VRAM
    # total at any plausible per-token KV cost.
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
    """PASS/FAIL oracle #4 - "the amd-rocm/HIP split path that has never once
    been executed" (issues/issues.txt GPU-SPLIT-TESTING). The lowest possible
    bar for a path with zero prior runs on any hardware, ever: does a split
    load across 2 real AMD devices via the amd-rocm backend even complete and
    produce coherent output. test_split_load_honors_configured_ratio already
    proves ratio PRECISION generically (any backend, including this one, when
    the gate runs on the AMD box) - this test exists so the HIP path has its
    own always-named, always-collected proof point, and skips outright (never
    silently "passes" for the wrong reason) when not run against amd-rocm.

    PASS: both configured AMD devices show a positive VRAM delta after load;
          coherent chat completion.
    FAIL: all layers land on one device, a load error, or incoherent output.
    """
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
    """PASS/FAIL oracle #5 - the research doc's explicitly-requested
    adversarial configs (dev-notes/split-gpu-testing-research-2026-07-13.md
    Tier 2 step 7): an even more extreme ratio than test #1, and a PINNED
    split sized so one device's computed share cannot fit that device's own
    real free VRAM. gpu_split_shortfall()'s docstring is explicit that this
    per-device hazard is a PINNED-ratio phenomenon specifically (with ratios
    unset, the loader's own live auto-adaptation makes this refusal
    structurally impossible) - so this test pins ratios deliberately, unlike
    tests #2/#3 which leave them auto.

    PASS: a 99:1 ratio skews the VRAM delta even harder than test #1's 9:1
          floor; gpu_split_shortfall() (a fresh, non-stale probe) reports a
          non-empty shortfall for a pinned-equal-split total that exceeds the
          smallest configured device's real free VRAM.
    FAIL: no skew from the extreme ratio, or an empty shortfall despite a
          per-device requirement that provably exceeds real free VRAM.
    """
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
