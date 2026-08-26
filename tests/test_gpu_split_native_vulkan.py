# SPDX-License-Identifier: AGPL-3.0-or-later
r"""REAL end-to-end test of the split-GPU load/inference path across two
genuinely distinct native ggml-vulkan devices. tests/test_gpu_split_wiring.py
proves the Python-side wiring (apply_gpu_split() writes the right
mp.tensor_split/mp.split_mode/mp.main_gpu) against a fully MOCKED ctypes api;
here only localm.config.load_config is mocked (to inject
gpu_split_indices/gpu_split_ratios), and everything else - LlamaCpp
construction, apply_gpu_split, the native llama.dll call, the actual model load
and a real chat completion - runs for REAL against two real Vulkan devices.

PREREQUISITES (this test does not perform them):
  1. A second real Vulkan device is registered ADDITIVELY alongside whatever
     GPU(s) are already installed (Mesa lavapipe is the recommended choice).
     Register it via VK_ADD_DRIVER_FILES (NOT VK_ICD_FILENAMES/
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
  skips, with a clear reason, rather than running against whatever device set
  happened to already be registered. Invoke it as its own run:

      $env:LOCALM_TEST_LAVAPIPE_ICD = "Z:\path\to\lvp_icd.x86_64.json"
      pytest -m real_vulkan_split tests/test_gpu_split_native_vulkan.py -v -s

Also optionally set LOCALM_TEST_VULKAN_SPLIT_INDICES (default "0,1") if the
real enumeration order puts the two devices at different indices - confirm the
real order first with `vulkaninfo --summary` or the native startup log this
test itself captures and prints (run once with -s to see it before trusting the
default).

NOT covered here: real VRAM pressure/OOM behavior (a software Vulkan device is
backed by system RAM, not a real second memory domain), and the amd-rocm/HIP
backend (no software HIP implementation exists).
"""

from __future__ import annotations

import os
import re

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.real_vulkan_split]

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
        # Additive registration: appends to the loader's normal driver search,
        # so the real GPU's already-installed driver stays discovered too.
        os.environ["VK_ADD_DRIVER_FILES"] = icd_path

    if not os.environ.get("GGML_VK_VISIBLE_DEVICES"):
        # ggml-vulkan's default auto-selection EXCLUDES CPU-type devices, and
        # lavapipe reports deviceType=PHYSICAL_DEVICE_TYPE_CPU, so it has to be
        # named here or the run silently degrades to one device.
        # GGML_VK_VISIBLE_DEVICES is REQUIRED whenever the second device is
        # CPU-type; a real GPU pair would not need it.
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
    """Load a real GGUF with a LOPSIDED configured split ratio (9:1) across two
    real, independent native Vulkan devices, and confirm from the native
    backend's OWN per-layer device-assignment log that the ratio was HONORED -
    not just that both devices receive some layers (which llama.cpp's own
    auto-split fallback also produces) and not just that list_gpus() (which is
    Vulkan-blind, see discover.py) was told about them."""
    indices = _split_indices()
    assert len(indices) >= 2, (
        "LOCALM_TEST_VULKAN_SPLIT_INDICES must name at least 2 device indices"
    )
    # LOPSIDED, not equal: an equal ratio cannot distinguish a honoured split
    # from llama.cpp's own auto-split fallback. 9:1 is unambiguous either way.
    ratios = [9.0] + [1.0] * (len(indices) - 1)

    monkeypatch.setattr(
        "localm.config.load_config",
        lambda: {"gpu_split_indices": indices, "gpu_split_ratios": ratios},
    )

    from localm.inference.backends.llamacpp.llama import LlamaCpp

    # Do NOT discard capfd's buffer before construction: the ggml-vulkan
    # device-enumeration print may happen at BACKEND REGISTRATION (inside the
    # vulkan_split_model_path fixture's load_lib() call) rather than at
    # model-load time. The regex assertion below tolerates the extra
    # fixture-setup output this captures alongside it.
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

        # Each KV-cache layer's assigned backend device, one line per layer:
        # "llama_kv_cache: layer <n>: dev = Vulkan<idx>". This is the
        # ground-truth per-layer placement; the startup
        # "ggml_vulkan: <idx> = <name>" line only proves a device EXISTS.
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

        # With ratios=[9.0, 1.0], device 0's SHARE must be clearly the larger
        # one, not an even split. 70% is a generous floor rather than the literal
        # 90%, so the check is not brittle to llama.cpp's whole-layer rounding:
        # it distinguishes "honored" from "ignored" rather than verifying exact
        # arithmetic.
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

        # A real forward pass across the split still produces coherent text,
        # not garbage or empty output.
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
    ggml-vulkan's own index space), and a load pinned with those ratios (what
    GgufBackend._load_native forwards through GgufWorker/LlamaCpp) must place
    layers in that proportion.

    The assertion band is anchored to the ratios THIS run computed, not to any
    hardcoded hardware expectation."""
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
        # Each device's actual layer share tracks the auto ratio this run
        # computed. 0.15 is a generous band for llama.cpp's whole-layer rounding
        # on a ~30-layer model.
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
