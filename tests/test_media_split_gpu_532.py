# SPDX-License-Identifier: AGPL-3.0-or-later
"""Media generation names a PREFERRED GPU without MASKING the others away.

ComfyUI core ships per-component GPU PLACEMENT - `SelectModelDevice` /
`SelectCLIPDevice` / `SelectVAEDevice` (`comfy_extras/nodes_multigpu.py`, registered at
`nodes.py:2440`), which call `deepclone_multigpu` to rehome a component onto another
card with its own independent weights. So a second GPU CAN carry real work.

Masking destroys that capability. ComfyUI has two flags that both read as "use
this card", and they are NOT interchangeable:

  --cuda-device N    -> CUDA_VISIBLE_DEVICES = "N"        (main.py:78-81)  DELETES the
                        other cards from torch's view, so every gpu:1 placement node
                        silently becomes a no-op.
  --default-device N -> CUDA_VISIBLE_DEVICES = "N,0,1,.." (main.py:69-76)  REORDERS so
                        N leads and the rest stay usable.

The swap gate (`vram.decide_media_swap`) reads COMBINED free VRAM across the split; the
loader must agree about which card leads, or a 4 GB job "fits" in 2x4 GB combined, the
chat model is kept, and the media model OOMs on one 4 GB card. Naming the preferred card
makes them agree WITHOUT amputating the box.
"""

from __future__ import annotations

import pytest

import localm.config as cfg
from localm.media import managed_comfy as mc


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Throwaway LOCALM_HOME wired through both the lazy home_dir() and the
    import-frozen config module attrs, so load_config() and managed_comfy path
    resolution agree on the same tmp dir."""
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

    Also pins non-Vulkan so resolve_gpu_split's membership validation actually runs,
    regardless of what native backend is provisioned in the ambient environment: on
    Vulkan, list_gpus() cannot see Vulkan-only devices, so resolve_gpu_split passes
    the configured indices through UNCHECKED.
    """
    import localm.discover as disc
    gpus = [{"index": i, "name": f"fake{i}", "free": free, "total": free * 2}
            for i, free in specs]
    monkeypatch.setattr(disc, "list_gpus", lambda **kw: list(gpus))
    monkeypatch.setattr(disc, "_native_backend_has_vulkan", lambda: False)
    return gpus


GB = 1024 ** 3


def _flag(cmd: str, name: str):
    """The value passed to --<name> in a launch command, or None."""
    parts = cmd.split()
    for i, p in enumerate(parts):
        if p == f"--{name}":
            return parts[i + 1] if i + 1 < len(parts) else ""
        if p.startswith(f"--{name}="):
            return p.split("=", 1)[1]
    return None


# --------------------------------------------------------------------------- #
#  Never mask.                                                                 #
# --------------------------------------------------------------------------- #

def test_never_emits_cuda_device_which_would_mask_the_other_cards(home, monkeypatch):
    """--cuda-device must NEVER be emitted.

    It sets CUDA_VISIBLE_DEVICES to a single id (main.py:78-81), which deletes every
    other card from torch's view and turns ComfyUI core's SelectModelDevice /
    SelectCLIPDevice / SelectVAEDevice nodes into silent no-ops - a gpu:1 that does not
    exist does nothing.
    """
    _fake_gpus(monkeypatch, (0, 2 * GB), (1, 7 * GB))
    cfg.save_config({"gpu_split_indices": [0, 1], "gpu_split_ratios": [0.5, 0.5]})

    cmd = mc.managed_comfy_launch_cmd()

    assert _flag(cmd, "cuda-device") is None, (
        "--cuda-device masks the other GPUs out of existence and disables ComfyUI's "
        "own per-component placement nodes. Use --default-device, which reorders.")


def test_split_box_prefers_the_card_with_most_free_vram(home, monkeypatch):
    """Split [0,1]; card 1 has MORE free VRAM -> ComfyUI defaults to card 1.

    resolve_main_gpu_index(None) returns 0 (discover.py:528-540), so an unset
    main_gpu_index does NOT mean "no split", it means "device 0". Choosing via that
    identity default would silently pick card 0 and ignore the split. Which card
    leads is a CAPACITY question.
    """
    _fake_gpus(monkeypatch, (0, 2 * GB), (1, 7 * GB))
    cfg.save_config({"gpu_split_indices": [0, 1], "gpu_split_ratios": [0.5, 0.5]})

    got = _flag(mc.managed_comfy_launch_cmd(), "default-device")

    assert got == "1", (
        f"expected --default-device 1 (card 1 has 7 GB free vs card 0's 2 GB), got "
        f"{got!r}. Picking 0 is the resolve_main_gpu_index(None) -> 0 trap: an identity "
        "default answering a capacity question.")


def test_single_gpu_box_with_main_gpu_index_names_that_card(home, monkeypatch):
    """No split configured, main_gpu_index = 1 -> honour the user's explicit choice."""
    _fake_gpus(monkeypatch, (0, 4 * GB), (1, 4 * GB))
    cfg.save_config({"main_gpu_index": 1})

    assert _flag(mc.managed_comfy_launch_cmd(), "default-device") == "1"


def test_unconfigured_box_emits_no_device_flag(home, monkeypatch):
    """NEGATIVE-TEST: nothing configured -> do NOT invent a device.

    Naming a device here would invent a choice the user never made, and would mask
    any second card they later add.
    """
    _fake_gpus(monkeypatch, (0, 8 * GB))
    cfg.save_config({})

    cmd = mc.managed_comfy_launch_cmd()
    assert _flag(cmd, "default-device") is None
    assert _flag(cmd, "cuda-device") is None


def test_launch_cmd_still_targets_the_managed_venv_and_port(home, monkeypatch):
    """Guard the parts of the launch command the device flag must not disturb."""
    _fake_gpus(monkeypatch, (0, 2 * GB), (1, 7 * GB))
    cfg.save_config({"gpu_split_indices": [0, 1]})

    cmd = mc.managed_comfy_launch_cmd()

    assert "main.py" in cmd
    assert "--listen 127.0.0.1" in cmd
    assert f"--port {mc.MANAGED_COMFY_PORT}" in cmd


# --------------------------------------------------------------------------- #
#  A user's OWN ComfyUI: the child env must ORDER, not MASK.                   #
# --------------------------------------------------------------------------- #

def test_own_comfy_child_env_keeps_every_card_visible(home, monkeypatch):
    """The env route must ORDER the devices, not pin one.

    localm cannot pass argv to a user's own launcher, so the child env is the only
    lever. Setting CUDA_VISIBLE_DEVICES="1" would mask exactly as --cuda-device does.
    We must emit the full order ("1,0"), which is what --default-device itself writes.
    """
    from localm.media.comfy_client import comfy_child_env
    _fake_gpus(monkeypatch, (0, 2 * GB), (1, 7 * GB))
    cfg.save_config({"gpu_split_indices": [0, 1], "gpu_split_ratios": [0.5, 0.5],
                     "comfy_target": "external"})

    env = comfy_child_env() or {}

    assert env.get("CUDA_VISIBLE_DEVICES") == "1,0", (
        f"expected the full ordered list '1,0' (preferred card leads, other card still "
        f"visible), got {env.get('CUDA_VISIBLE_DEVICES')!r}. A single id is a MASK.")
    assert env.get("HIP_VISIBLE_DEVICES") == "1,0", (
        "HIP must match CUDA: on the ZLUDA/ROCm path CUDA is emulated over HIP, and "
        "ComfyUI's own main.py sets both.")


def test_own_comfy_child_env_unset_when_nothing_configured(home, monkeypatch):
    """NEGATIVE-TEST: a plain box spawns exactly as today, no vars injected."""
    from localm.media.comfy_client import comfy_child_env
    _fake_gpus(monkeypatch, (0, 8 * GB))
    cfg.save_config({"comfy_target": "external"})

    env = comfy_child_env() or {}

    assert env.get("CUDA_VISIBLE_DEVICES") is None
    assert env.get("HIP_VISIBLE_DEVICES") is None


# --------------------------------------------------------------------------- #
#  INDEX SPACE: never leak a Vulkan-space index into a torch decision.         #
# --------------------------------------------------------------------------- #

def test_split_with_no_torch_visible_device_names_nothing(home, monkeypatch):
    """A split whose indices torch cannot see must name NO device.

    resolve_gpu_split() passes indices through UNVALIDATED on the `vulkan` llama.cpp
    build, because list_gpus() is structurally blind to Vulkan-only devices. Those
    indices live in ggml-vulkan's index space. Media is torch (ComfyUI), so handing one
    of them to ComfyUI as a CUDA/HIP id would point it at the WRONG CARD. Returning None
    (ComfyUI keeps its own default) is the only honest answer.
    """
    import localm.discover as disc
    # A box where torch sees ONLY device 0, but a split names 3 and 4 (as a Vulkan-only
    # split legitimately can). Vulkan pass-through is ON, so resolve_gpu_split does not
    # drop them.
    gpus = [{"index": 0, "name": "fake0", "free": 8 * GB, "total": 16 * GB}]
    monkeypatch.setattr(disc, "list_gpus", lambda **kw: list(gpus))
    monkeypatch.setattr(disc, "_native_backend_has_vulkan", lambda: True)
    cfg.save_config({"gpu_split_indices": [3, 4], "gpu_split_ratios": [0.5, 0.5]})

    assert disc.resolve_preferred_device() is None, (
        "a Vulkan-space index must never be emitted as a torch/CUDA device id")
    assert _flag(mc.managed_comfy_launch_cmd(), "default-device") is None


# --------------------------------------------------------------------------- #
#  THE INDEX-SPACE GATE: gpu:N is a POSITION in the order we impose.           #
# --------------------------------------------------------------------------- #

def test_gpu_option_is_a_position_in_the_order_we_impose(home, monkeypatch):
    """gpu:N maps through visible_device_order(), NOT through our own device ids.

    ComfyUI's get_gpu_device_options (model_management.py:246-257) emits
    `gpu:{i} for i in range(len(get_all_torch_devices()))`, and get_all_torch_devices
    enumerates torch AFTER the reorder we impose. So on a box where card 1 is preferred,
    the visible order is [1, 0] and therefore:
        our device 1 -> gpu:0   (it LEADS, so it is torch index 0)
        our device 0 -> gpu:1
    Getting this backwards puts a component on the wrong card and STILL RENDERS:
    a silent wrong answer.
    """
    import localm.discover as disc
    _fake_gpus(monkeypatch, (0, 2 * GB), (1, 7 * GB))
    cfg.save_config({"gpu_split_indices": [0, 1], "gpu_split_ratios": [0.5, 0.5]})

    assert disc.visible_device_order() == [1, 0], "card 1 has more free VRAM, so it leads"
    assert disc.comfy_gpu_option(1) == "gpu:0", (
        "our card 1 leads the visible order, so ComfyUI sees it at position 0")
    assert disc.comfy_gpu_option(0) == "gpu:1", (
        "our card 0 is second in the visible order, so ComfyUI sees it at position 1")


def test_gpu_option_none_for_unknown_or_unconfigured(home, monkeypatch):
    """NEGATIVE-TEST: never guess a position."""
    import localm.discover as disc
    _fake_gpus(monkeypatch, (0, 2 * GB), (1, 7 * GB))
    cfg.save_config({"gpu_split_indices": [0, 1]})
    assert disc.comfy_gpu_option(7) is None, "a device not in the order has no position"

    cfg.save_config({})
    assert disc.comfy_gpu_option(0) is None, (
        "nothing configured means no order is imposed, so no gpu:N can be named")
