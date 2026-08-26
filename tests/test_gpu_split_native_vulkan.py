# SPDX-License-Identifier: AGPL-3.0-or-later
r"""REAL end-to-end test of the split-GPU load/inference path across two
genuinely distinct native ggml-vulkan devices - the gap identified in
dev-notes/split-gpu-testing-research-2026-07-13.md: tests/test_gpu_split_wiring.py
already proves the Python-side wiring (apply_gpu_split() writes the right
mp.tensor_split/mp.split_mode/mp.main_gpu) against a fully MOCKED ctypes api, but
never touches the real native loader. This file does the opposite: only
localm.config.load_config is mocked (to inject gpu_split_indices/gpu_split_ratios,
exactly like the wiring tests), everything else - LlamaCpp construction,
apply_gpu_split, the native llama.dll call, the actual model load and a real
chat completion - runs for REAL, against two real Vulkan devices.

PREREQUISITES (this test does not perform them - see the doc's Tier 1 section):
  1. A second real Vulkan device is registered ADDITIVELY alongside whatever
     GPU(s) are already installed (Mesa lavapipe is the researched/recommended
     choice - free, and the same technique llama.cpp's own upstream CI already
     uses). Register it via VK_ADD_DRIVER_FILES (NOT VK_ICD_FILENAMES/
     VK_DRIVER_FILES - those REPLACE the driver search and would hide the real
     GPU). Run unelevated (the Vulkan loader ignores the *_DRIVER_FILES env vars
     for elevated processes).
  2. localm's `vulkan` native runtime build is provisioned (`localm setup-llama
     vulkan`), so ggml-vulkan is the active backend.
  3. Set LOCALM_TEST_LAVAPIPE_ICD to the second device's ICD manifest path
     BEFORE launching pytest (not inside a test - see the ordering caveat below).
     This both gates the test (tests/conftest.py's real_vulkan_split resource
     gate) and, unless VK_ADD_DRIVER_FILES is already exported separately, is
     used AS the VK_ADD_DRIVER_FILES value.

ORDERING CAVEAT - run this file in ISOLATION, not mixed into a full suite run:
  the native Vulkan loader only reads VK_ADD_DRIVER_FILES once, when the ggml
  backend is first registered inside this process (triggered by the FIRST
  load_lib()/LlamaCpp() call). If an earlier test in the same pytest session
  already loaded the native runtime (e.g. any real_gguf-marked test that RAN
  earlier - its lazy resource gate, or the test itself, calls load_lib() at
  that test's setup), the env var
  would already be too late for THIS process. The fixture below guards this: it
  refuses (skips, with a clear reason) rather than silently running against
  whatever device set happened to already be registered - a false pass here
  would be worse than a skip (AGENTS.md rule 5). Invoke it as its own run:

      $env:LOCALM_TEST_LAVAPIPE_ICD = "Z:\path\to\lvp_icd.x86_64.json"
      pytest -m real_vulkan_split tests/test_gpu_split_native_vulkan.py -v -s

Also optionally set LOCALM_TEST_VULKAN_SPLIT_INDICES (default "0,1") if the
real enumeration order puts the two devices at different indices - confirm the
real order first with `vulkaninfo --summary` or the native startup log this
test itself captures and prints (run once with -s to see it before trusting the
default).

NOT covered here (see the doc for why): real VRAM pressure/OOM behavior (a
software Vulkan device is backed by system RAM, not a real second memory
domain), and the amd-rocm/HIP backend (no software HIP implementation exists -
that path needs the Tier 2 real-hardware cloud rental instead).
"""

from __future__ import annotations

import os
import re

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.real_vulkan_split]

# Same pinned tiny model as tests/test_gguf_smoke_integration.py - deliberately
# reused rather than a second constant, so both real-load tests exercise an
# identical, already-verified-working GGUF and any divergence in behavior is
# attributable to the split configuration, not the model.
_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
_FILE = "SmolLM2-135M-Instruct-Q4_K_M.gguf"


def _split_indices() -> list:
    raw = os.environ.get("LOCALM_TEST_VULKAN_SPLIT_INDICES", "0,1")
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


@pytest.fixture(scope="module")
def vulkan_split_model_path():
    try:
        from localm.inference.backends.llamacpp import _loader
    except Exception as e:
        pytest.skip(f"native llamacpp binding unavailable: {e}")

    if _loader._loaded_lib is not None:
        pytest.skip(
            "the native runtime was already loaded earlier in this pytest "
            "session, so VK_ADD_DRIVER_FILES (read once, at first backend "
            "registration) cannot be honored for this test. Run this file in "
            "ISOLATION: pytest -m real_vulkan_split "
            "tests/test_gpu_split_native_vulkan.py -v -s"
        )

    icd_path = os.environ.get("LOCALM_TEST_LAVAPIPE_ICD")
    if icd_path and not os.environ.get("VK_ADD_DRIVER_FILES"):
        # Additive registration: appends to the loader's normal driver search
        # (the real GPU's already-installed driver stays discovered too) -
        # see the module docstring for why VK_ICD_FILENAMES/VK_DRIVER_FILES
        # must NOT be used here instead.
        os.environ["VK_ADD_DRIVER_FILES"] = icd_path

    if not os.environ.get("GGML_VK_VISIBLE_DEVICES"):
        # CONFIRMED BY A LIVE RUN (2026-07-13, real AMD GPU + real lavapipe):
        # ggml-vulkan's default auto-selection EXCLUDES CPU-type devices (it
        # printed "Found 1 Vulkan devices" even though vkEnumeratePhysicalDevices
        # correctly reported 2 via VK_ADD_DRIVER_FILES alone) - lavapipe reports
        # deviceType=PHYSICAL_DEVICE_TYPE_CPU, so it is silently filtered out
        # unless explicitly named here. Without this, the whole point of this
        # test (proving the native split path across 2 devices) silently
        # degrades to a 1-device run that would still pass every assertion
        # below for the wrong reason. GGML_VK_VISIBLE_DEVICES is REQUIRED, not
        # optional, whenever the second device is CPU-type (a real GPU pair
        # would not need this).
        os.environ["GGML_VK_VISIBLE_DEVICES"] = ",".join(str(i) for i in _split_indices())

    try:
        _loader.load_lib()
    except Exception as e:
        pytest.skip(f"native llama runtime not provisioned (run 'localm setup-llama vulkan'): {e}")

    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id=_REPO, filename=_FILE)
    except Exception as e:
        pytest.skip(f"could not fetch {_REPO}/{_FILE}: {e}")
    return path


def test_split_load_uses_both_native_devices(vulkan_split_model_path, monkeypatch, capfd):
    """The actual DONE oracle for GPU-SPLIT-VKINDEX (issues/issues.txt): load a
    real GGUF with a deliberately LOPSIDED configured split ratio (9:1) across
    two real, independent native Vulkan devices, and confirm from the native
    backend's OWN per-layer device-assignment log that the ratio was actually
    HONORED - not just that both devices happen to receive some layers (which
    can also happen via llama.cpp's own unrelated auto-split fallback; see the
    CONFIRMED BUG note below) and not just that list_gpus() (which is
    Vulkan-blind, see discover.py) was told about them.

    CONFIRMED BUG (live run, 2026-07-13, real AMD RX 6900 XT + real Mesa
    lavapipe): this assertion currently FAILS. discover.apply_gpu_split()
    validates the configured gpu_split_indices against list_gpus() - which
    only enumerates via torch.cuda (CUDA/HIP) or nvidia-smi and is
    structurally blind to Vulkan - so on a Vulkan-only box list_gpus() never
    reports the second (Vulkan-enumerated) device index, and
    apply_gpu_split() SILENTLY DROPS it ("gpu_split_indices contains 1
    device(s) not currently detected; dropping them from the split", logged
    at WARNING, discover.py:640) rather than applying the configured split.
    With fewer than 2 valid devices left, it leaves mp.tensor_split/
    mp.split_mode at the native LlamaCpp default, which for
    SPLIT_MODE_LAYER means llama.cpp's OWN built-in auto-split (proportional
    to each device's free memory) silently takes over instead - independent
    of, and in this live run's case DIRECTLY CONTRADICTING, the user's
    explicit 9:1 request (the observed result was an even 15/15 layer split,
    identical to a separate control run using an explicit 1:1 ratio - proving
    the configured ratio had zero effect either way). This is a live instance
    of exactly the class of bug the project's own working agreement calls out
    as a cardinal sin: silently overriding a user's explicit choice instead of
    informing them and offering the alternative. FIX: teach list_gpus() a
    Vulkan enumeration path, or have apply_gpu_split() validate against the
    native ggml device count/order (llama_max_devices()) directly instead of
    the Vulkan-blind list_gpus() when the active backend is vulkan."""
    indices = _split_indices()
    assert len(indices) >= 2, (
        "LOCALM_TEST_VULKAN_SPLIT_INDICES must name at least 2 device indices"
    )
    # Deliberately LOPSIDED, not equal: an equal ratio cannot distinguish
    # "the configured split was honored" from "llama.cpp's own unrelated
    # auto-split fallback happened to produce a similar-looking result" -
    # exactly how this bug went unnoticed. 9:1 is unambiguous either way.
    ratios = [9.0] + [1.0] * (len(indices) - 1)

    monkeypatch.setattr(
        "localm.config.load_config",
        lambda: {"gpu_split_indices": indices, "gpu_split_ratios": ratios},
    )

    from localm.inference.backends.llamacpp.llama import LlamaCpp

    # Deliberately do NOT discard capfd's buffer before construction: the
    # ggml-vulkan device-enumeration print may happen at BACKEND REGISTRATION
    # (inside the vulkan_split_model_path fixture's load_lib() call, which ran
    # during this test's setup phase) rather than at model-load time, and a
    # premature readouterr() here would silently swallow exactly the signal
    # this test exists to check. The regex assertion below is tolerant of the
    # extra fixture-setup output this captures alongside it.
    llm = LlamaCpp(
        vulkan_split_model_path,
        n_ctx=512,
        n_gpu_layers=99,
        verbose=True,   # native stderr goes straight to the real fd 2 -
                         # uncaptured by localm's own _quiet_stderr/_capture_stderr
                         # suppression - so pytest's capfd can see it below.
    )
    try:
        out = capfd.readouterr()
        native_log = out.out + out.err
        print("\n--- captured native load output ---\n" + native_log)

        # CONFIRMED real format (2026-07-13 live run) - each KV-cache layer's
        # assigned backend device, one line per layer: "llama_kv_cache: layer
        # <n>: dev = Vulkan<idx>". This is the ground-truth per-layer
        # placement list_gpus() cannot provide (it never enumerates Vulkan at
        # all - the exact gap GPU-SPLIT-VKINDEX is about), and a stronger
        # oracle than the startup "ggml_vulkan: <idx> = <name>" enumeration
        # line (which only proves a device EXISTS, not that any layer landed
        # on it).
        layer_devices = re.findall(r"dev\s*=\s*Vulkan(\d+)", native_log)
        assert layer_devices, (
            f"found no 'dev = Vulkan<n>' per-layer device-assignment lines "
            f"in the native load log at all - the model may have loaded "
            f"single-device or the log format has changed. Captured log:\n{native_log}"
        )
        seen = sorted(set(layer_devices))
        assert len(seen) >= 2, (
            f"expected layers split across >= 2 native Vulkan devices, all "
            f"{len(layer_devices)} layers landed on device(s) {seen} only. "
            f"Captured log:\n{native_log}\n"
            f"Check that VK_ADD_DRIVER_FILES is registering the second device "
            f"additively (see the module docstring) - this exact silent "
            f"degrade-to-one-device is the upstream failure class in "
            f"ggml-org/llama.cpp#15974."
        )

        # The real oracle: with ratios=[9.0, 1.0] (device 0 should get ~90% of
        # layers), device 0's SHARE must be clearly the larger one - not an
        # even split, which is what apply_gpu_split()'s Vulkan-blindness bug
        # (GPU-SPLIT-VKINDEX, see this test's docstring) currently produces
        # instead by silently falling back to llama.cpp's own unrelated
        # auto-split. 70% is a deliberately generous floor (not the literal
        # 90%) so this isn't brittle to llama.cpp's own internal rounding -
        # it only needs to distinguish "honored" from "ignored", not verify
        # exact arithmetic (that's test_gpu_split_wiring.py's job, on the
        # mocked Python layer).
        dev0_count = layer_devices.count(str(indices[0]))
        dev0_share = dev0_count / len(layer_devices)
        assert dev0_share >= 0.70, (
            f"configured gpu_split_ratios={ratios} asked for ~90% of layers "
            f"on device {indices[0]}, but it actually got {dev0_count}/"
            f"{len(layer_devices)} ({dev0_share:.0%}) - the configured ratio "
            f"was not honored. This is GPU-SPLIT-VKINDEX: "
            f"apply_gpu_split() silently dropped the configured split because "
            f"list_gpus() (CUDA/HIP/nvidia-smi only) does not recognize the "
            f"native Vulkan device index, and llama.cpp's own unrelated "
            f"auto-split (proportional to free memory) silently took over "
            f"instead. See this test's docstring for the confirmed root cause "
            f"and fix. Captured log:\n{native_log}"
        )

        # And the actual, behavioral half of the oracle: a real forward pass
        # across the split must still produce coherent text, not garbage/empty
        # output (the failure mode in ggml-org/llama.cpp#22817's crash, or a
        # silent NaN-producing misconfiguration).
        result = llm.create_chat_completion(
            [{"role": "user", "content": "In one short sentence, what color is a clear daytime sky?"}],
            max_tokens=40, temperature=0.0, seed=1, stream=False,
        )
        text = result["choices"][0]["message"]["content"].strip()
        assert len(text) >= 10, f"suspiciously short split-load output: {text!r}"
        assert any(c.isalpha() for c in text), f"no words in split-load output: {text!r}"
    finally:
        llm.close()


def test_auto_split_ratios_from_native_free_vram(
        vulkan_split_model_path, monkeypatch, capfd, caplog):
    """End-to-end oracle for the AUTO split distribution on the vulkan build:
    with gpu_split_indices configured and NO ratios pinned, the parent-side
    discover.resolve_auto_split_ratios() must read per-device free VRAM from
    the NATIVE registry (the crash-isolated probe daemon - the only source in
    ggml-vulkan's own index space, GPU-SPLIT-VKINDEX), and a load pinned with
    those ratios (exactly what GgufBackend._load_native forwards through
    GgufWorker/LlamaCpp) must place layers in that proportion - proving the
    whole "query free vram from each card, compare and distribute" feature
    against two genuinely distinct native devices, not mocks.

    On the researched setup (a real ~16 GB GPU + lavapipe backed by more
    system RAM) the free readings are genuinely asymmetric, so the resulting
    layer split is visibly NOT the historical equal split - but the assertion
    band is anchored to the ratios THIS run computed, not to any hardcoded
    hardware expectation."""
    indices = _split_indices()
    assert len(indices) >= 2, (
        "LOCALM_TEST_VULKAN_SPLIT_INDICES must name at least 2 device indices"
    )
    cfg = {"gpu_split_indices": indices, "gpu_split_ratios": None}
    monkeypatch.setattr("localm.config.load_config", lambda: cfg)

    from localm import discover

    with caplog.at_level("INFO"):
        ratios = discover.resolve_auto_split_ratios(cfg)
    assert ratios is not None and len(ratios) == len(indices), (
        f"resolve_auto_split_ratios() declined on the vulkan build "
        f"(got {ratios!r}) - the native probe daemon should have answered "
        f"per-device free/total for devices {indices}. Check "
        f"discover.native_gpu_devices() output and the daemon log."
    )
    assert all(r > 0 for r in ratios)
    assert any("auto" in r.message and "split" in r.message
               for r in caplog.records), (
        "the auto distribution decision must be logged at INFO")
    print(f"\n--- auto ratios computed from native free VRAM: "
          f"{dict(zip(indices, [f'{r:.3f}' for r in ratios]))} ---")

    from localm.inference.backends.llamacpp.llama import LlamaCpp

    llm = LlamaCpp(
        vulkan_split_model_path,
        n_ctx=512,
        n_gpu_layers=99,
        verbose=True,                 # native per-layer placement to real fd 2
        gpu_split_ratios=ratios,       # the parent-pinned auto distribution
    )
    try:
        out = capfd.readouterr()
        native_log = out.out + out.err
        print("\n--- captured native load output ---\n" + native_log)

        layer_devices = re.findall(r"dev\s*=\s*Vulkan(\d+)", native_log)
        assert layer_devices, (
            f"found no 'dev = Vulkan<n>' per-layer device-assignment lines in "
            f"the native load log. Captured log:\n{native_log}"
        )
        seen = sorted(set(layer_devices))
        assert len(seen) >= 2, (
            f"expected the auto split to place layers on >= 2 native devices, "
            f"all {len(layer_devices)} layers landed on {seen} only. "
            f"Captured log:\n{native_log}"
        )
        # Each device's actual layer share must track the auto ratio this run
        # computed. 0.15 is a generous band for llama.cpp's whole-layer
        # rounding on a ~30-layer model (the 9:1 test above uses the same
        # tolerance philosophy) - it distinguishes "the pinned auto ratios
        # were honored" from "the equal split or llama.cpp's own fallback
        # happened instead" without being brittle to rounding.
        total = len(layer_devices)
        for idx, ratio in zip(indices, ratios):
            share = layer_devices.count(str(idx)) / total
            assert abs(share - ratio) <= 0.15, (
                f"device {idx}: actual layer share {share:.2f} deviates from "
                f"the auto ratio {ratio:.2f} by more than 0.15 - the pinned "
                f"auto distribution was not honored. Placement: "
                f"{ {i: layer_devices.count(str(i)) for i in indices} } of "
                f"{total} layers. Captured log:\n{native_log}"
            )

        # Behavioral half: a real forward pass across the auto split still
        # produces coherent text.
        result = llm.create_chat_completion(
            [{"role": "user", "content": "In one short sentence, what color is grass?"}],
            max_tokens=40, temperature=0.0, seed=1, stream=False,
        )
        text = result["choices"][0]["message"]["content"].strip()
        assert len(text) >= 10, f"suspiciously short auto-split output: {text!r}"
        assert any(c.isalpha() for c in text), f"no words in auto-split output: {text!r}"
    finally:
        llm.close()
