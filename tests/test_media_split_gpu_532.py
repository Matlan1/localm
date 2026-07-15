# SPDX-License-Identifier: AGPL-3.0-or-later
"""REG-532: media generation must tell ComfyUI which ONE device to use.

WHY THIS EXISTS (verified against the real ComfyUI source, git 867404b):
ComfyUI MASKS devices - `main.py:78-81` consumes `--cuda-device` by setting
`CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` - and then loads a model onto ONE
device (`model_management.py:194`, `get_torch_device()` -> `torch.cuda.current_device()`).
It has NO tensor_split equivalent, and no MultiGPU/DisTorch node is installed. So
media CANNOT span a GPU split the way the GGUF embedder does via `apply_gpu_split(mp)`.

Today localm passes ComfyUI no device at all. On a split box the swap gate
(`vram.decide_media_swap`, which reads COMBINED free across the split) and the actual
loader therefore disagree about which hardware they mean: the gate sees 2x4 GB = 8 GB
free and says a 4 GB media job fits, so the chat model is kept; the media model then
lands on ONE card with 4 GB and OOMs (or spills to shared RAM, or trips the ROCm TDR).

The fix is NOT to make the gate read single-GPU (rejected: it would need a
_RAW_ACCESSOR_GUARDS self-exemption, and capacity questions must use the combined
number). It is to make the media consumer pick ONE card deliberately, name it to
ComfyUI, and preflight THAT card for the WHOLE model.

Spec + full rationale: dev-notes/media-split-gpu/SPEC.md
"""

from __future__ import annotations

import pytest

import localm.config as cfg
from localm.media import managed_comfy as mc


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Throwaway LOCALM_HOME wired through both the lazy home_dir() and the
    import-frozen config module attrs, so load_config() and managed_comfy path
    resolution agree on the same tmp dir (see "Test home isolation (import-time)")."""
    h = tmp_path / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    return h


def _fake_gpus(monkeypatch, *specs):
    """Patch list_gpus() to report the given (index, free_bytes) devices.

    Patched at its definition site in localm.discover so every caller
    (resolve_gpu_split's validation, the device chooser) sees the same fake box.
    """
    import localm.discover as disc
    gpus = [{"index": i, "name": f"fake{i}", "free": free, "total": free * 2}
            for i, free in specs]
    monkeypatch.setattr(disc, "list_gpus", lambda **kw: list(gpus))
    return gpus


GB = 1024 ** 3


def _cuda_device_arg(cmd: str):
    """The value passed to --cuda-device in a launch command, or None."""
    parts = cmd.split()
    for i, p in enumerate(parts):
        if p == "--cuda-device":
            return parts[i + 1] if i + 1 < len(parts) else ""
        if p.startswith("--cuda-device="):
            return p.split("=", 1)[1]
    return None


# --------------------------------------------------------------------------- #
#  CHECK 1: managed_comfy_launch_cmd() names ONE deliberately-chosen device.   #
# --------------------------------------------------------------------------- #

def test_split_box_picks_the_card_with_most_free_vram(home, monkeypatch):
    """THE REG-532 CORE. Split [0,1]; card 1 has MORE free VRAM -> ComfyUI must be
    told to use card 1.

    This is also the #661 trap negative-test: `resolve_main_gpu_index(None)` returns
    **0** (discover.py:528-540), so an unset main_gpu_index does NOT mean "no split",
    it means "device 0". Choosing the device with an IDENTITY default here would
    silently pick card 0 and ignore the split - which is the exact bug PR #661 shipped
    and reverted (it dropped a peer holding VRAM on GPU 1 because `1 == 0` was false).
    Which card a whole-model workload lands on is a CAPACITY-informed choice.
    """
    _fake_gpus(monkeypatch, (0, 2 * GB), (1, 7 * GB))
    cfg.save_config({"gpu_split_indices": [0, 1], "gpu_split_ratios": [0.5, 0.5]})

    got = _cuda_device_arg(mc.managed_comfy_launch_cmd())

    assert got == "1", (
        f"expected --cuda-device 1 (card 1 has 7 GB free vs card 0's 2 GB), got {got!r}. "
        "Picking 0 here is the resolve_main_gpu_index(None) -> 0 trap: an identity "
        "default answering a capacity question.")


def test_split_box_never_emits_a_device_set(home, monkeypatch):
    """`--cuda-device 0,1` would be a FACADE.

    It is expressible (cli_args.py:52 takes a comma-separated list), but main.py:78-81
    consumes it as CUDA_VISIBLE_DEVICES masking only, and the model still lands on ONE
    card - chosen by ComfyUI's own `current_device()` rule (the first visible), not by
    us. Passing the set would LOOK like split support while changing placement not at
    all, leave us unable to preflight the right card, and make an upstream ComfyUI
    "first visible" change break us silently. Pick one card and name it.
    """
    _fake_gpus(monkeypatch, (0, 2 * GB), (1, 7 * GB))
    cfg.save_config({"gpu_split_indices": [0, 1], "gpu_split_ratios": [0.5, 0.5]})

    got = _cuda_device_arg(mc.managed_comfy_launch_cmd())

    assert got is not None and "," not in got, (
        f"--cuda-device must name exactly ONE device, got {got!r}. A comma-separated "
        "set only masks visibility; ComfyUI cannot shard a model across the split.")


def test_single_gpu_box_with_main_gpu_index_names_that_card(home, monkeypatch):
    """No split configured, main_gpu_index = 1 -> honour the user's explicit choice."""
    _fake_gpus(monkeypatch, (0, 4 * GB), (1, 4 * GB))
    cfg.save_config({"main_gpu_index": 1})

    assert _cuda_device_arg(mc.managed_comfy_launch_cmd()) == "1"


def test_unconfigured_box_emits_no_device_flag(home, monkeypatch):
    """NEGATIVE-TEST: nothing configured -> do NOT invent a device.

    A plain single-GPU box must keep working byte-identically to today. Emitting
    `--cuda-device 0` here would be inventing a choice the user never made, and would
    mask any second card they later add without telling them.
    """
    _fake_gpus(monkeypatch, (0, 8 * GB))
    cfg.save_config({})

    assert _cuda_device_arg(mc.managed_comfy_launch_cmd()) is None


def test_launch_cmd_still_targets_the_managed_venv_and_port(home, monkeypatch):
    """Guard the parts of the launch command the device flag must not disturb."""
    _fake_gpus(monkeypatch, (0, 2 * GB), (1, 7 * GB))
    cfg.save_config({"gpu_split_indices": [0, 1]})

    cmd = mc.managed_comfy_launch_cmd()

    assert "main.py" in cmd
    assert "--listen 127.0.0.1" in cmd
    assert f"--port {mc.MANAGED_COMFY_PORT}" in cmd
