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
import localm.vram as vram
from localm.media import comfy_client
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

    Also pins non-Vulkan so resolve_gpu_split's membership validation actually
    runs, regardless of what native backend is provisioned in the ambient
    environment (GPU-SPLIT-VKINDEX: on Vulkan, list_gpus() cannot see Vulkan-only
    devices, so resolve_gpu_split deliberately passes the configured indices
    through UNCHECKED - which would make these tests depend on how the box that
    runs them happens to be provisioned). Same pin, same reason, as #671.
    """
    import localm.discover as disc
    gpus = [{"index": i, "name": f"fake{i}", "free": free, "total": free * 2}
            for i, free in specs]
    monkeypatch.setattr(disc, "list_gpus", lambda **kw: list(gpus))
    monkeypatch.setattr(disc, "_native_backend_has_vulkan", lambda: False)
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


# --------------------------------------------------------------------------- #
#  CHECK 2: a USER'S OWN ComfyUI gets the device via the child ENV.            #
#  localm cannot rewrite their launcher (.bat, possibly ZLUDA-wrapped), so the #
#  env is the only lever. ComfyUI's own main.py:78-81 does nothing with        #
#  --cuda-device except set exactly these two variables.                       #
# --------------------------------------------------------------------------- #

def test_own_comfy_child_env_pins_the_chosen_card(home, monkeypatch):
    """Split [0,1], card 1 emptier -> the spawned ComfyUI is masked to card 1.

    Nothing is installed under the throwaway LOCALM_HOME, so managed_comfy_active()
    is False and this is the user's-own path.
    """
    _fake_gpus(monkeypatch, (0, 2 * GB), (1, 7 * GB))
    conf = {"gpu_split_indices": [0, 1], "gpu_split_ratios": [0.5, 0.5]}
    cfg.save_config(conf)

    env = comfy_client.comfy_child_env(conf)

    assert env is not None, "expected a child env pinning the device, got None (inherit)"
    assert env.get("CUDA_VISIBLE_DEVICES") == "1"
    assert env.get("HIP_VISIBLE_DEVICES") == "1", (
        "HIP_VISIBLE_DEVICES must be set too: this is the ZLUDA/ROCm path where CUDA "
        "is emulated over HIP, and ComfyUI's own main.py sets both.")


def test_managed_comfy_child_env_does_not_pin_the_device(home, monkeypatch):
    """The managed instance carries its device on the ARGV (--cuda-device), so the
    env must NOT also set it: one source of truth, not two that could disagree."""
    _fake_gpus(monkeypatch, (0, 2 * GB), (1, 7 * GB))
    conf = {"gpu_split_indices": [0, 1], "gpu_split_ratios": [0.5, 0.5]}
    cfg.save_config(conf)
    monkeypatch.setattr(mc, "managed_comfy_active", lambda c=None: True)

    env = comfy_client.comfy_child_env(conf)

    if env is not None:
        assert "CUDA_VISIBLE_DEVICES" not in env
        assert "HIP_VISIBLE_DEVICES" not in env


def test_unconfigured_box_child_env_is_untouched(home, monkeypatch):
    """NEGATIVE-TEST: nothing configured -> spawn exactly as today.

    Masking a plain box to an invented card would also hide a second GPU the user
    later adds, which is the opposite of what this feature is for.
    """
    _fake_gpus(monkeypatch, (0, 8 * GB))
    cfg.save_config({})

    env = comfy_client.comfy_child_env({})

    if env is not None:
        assert "CUDA_VISIBLE_DEVICES" not in env
        assert "HIP_VISIBLE_DEVICES" not in env


# --------------------------------------------------------------------------- #
#  CHECK 3: the per-device preflight. THE LITERAL REG-532 SCENARIO.            #
# --------------------------------------------------------------------------- #

def test_reg532_combined_says_fits_but_no_single_card_does(home, monkeypatch):
    """THE BUG, as a test. Two cards, 4 GB free EACH (8 GB combined), a 4 GB job.

    decide_media_swap reads the COMBINED 8 GB, sees 8 >= 4 + headroom, and says "both
    fit, keep the chat model loaded". The media model then lands WHOLE on ONE 4 GB
    card and OOMs. The combined gate is not wrong - it cannot see placement. The
    preflight must catch it.

    Note a per-device RATIO check would PASS this wrongly: each card would be asked
    for its 50% share (2 GB) and 4 GB free covers that. The whole-model predicate is
    what makes this fail correctly.
    """
    _fake_gpus(monkeypatch, (0, 4 * GB), (1, 4 * GB))
    conf = {"gpu_split_indices": [0, 1], "gpu_split_ratios": [0.5, 0.5]}
    cfg.save_config(conf)
    s = {"vram_estimate_bytes": 4 * GB, "swap_policy": "auto"}

    # The gate itself is UNCHANGED and still says "fits" on the combined number.
    assert vram.decide_media_swap(s, read_free=lambda: 8 * GB) is False

    shortfall = vram.media_single_device_shortfall(s, config=conf)

    assert shortfall is not None, (
        "preflight must refuse: 8 GB combined free, but the 4 GB model lands WHOLE "
        "on one 4 GB card. This is the REG-532 OOM.")
    assert shortfall["index"] in (0, 1)
    assert shortfall["free"] == 4 * GB
    assert shortfall["needed"] == 4 * GB + vram._DEFAULT_HEADROOM, (
        "the per-device check must use the SAME headroom as the aggregate, so it is "
        "not held to a thinner margin than the ceiling it composes with")


def test_preflight_allows_a_job_that_genuinely_fits_the_chosen_card(home, monkeypatch):
    """NEGATIVE-TEST: do NOT block a load that would have worked.

    Card 1 has 20 GB free and takes the whole 4 GB model comfortably.
    """
    _fake_gpus(monkeypatch, (0, 4 * GB), (1, 20 * GB))
    conf = {"gpu_split_indices": [0, 1], "gpu_split_ratios": [0.5, 0.5]}
    cfg.save_config(conf)
    s = {"vram_estimate_bytes": 4 * GB, "swap_policy": "auto"}

    assert vram.media_single_device_shortfall(s, config=conf) is None


def test_preflight_is_a_noop_without_a_split(home, monkeypatch):
    """No split -> the combined reading already IS the single card's. Nothing to add."""
    _fake_gpus(monkeypatch, (0, 4 * GB))
    conf = {}
    cfg.save_config(conf)
    s = {"vram_estimate_bytes": 40 * GB, "swap_policy": "auto"}

    assert vram.media_single_device_shortfall(s, config=conf) is None


def test_preflight_respects_an_explicit_swap_policy(home, monkeypatch):
    """'never' is the user's explicit choice (e.g. a big workstation card).

    Never silently override an explicit user selection: detect, inform, offer.
    """
    _fake_gpus(monkeypatch, (0, 4 * GB), (1, 4 * GB))
    conf = {"gpu_split_indices": [0, 1], "gpu_split_ratios": [0.5, 0.5]}
    cfg.save_config(conf)
    s = {"vram_estimate_bytes": 4 * GB, "swap_policy": "never"}

    assert vram.media_single_device_shortfall(s, config=conf) is None


# --------------------------------------------------------------------------- #
#  CHECK 8: the user-visible shortfall notice.                                 #
# --------------------------------------------------------------------------- #

def test_split_box_gets_a_notice_that_media_uses_one_card(home, monkeypatch):
    """The user configured a split and is NOT getting it for media: say so.

    Asserts the OBSERVABLE returned string, NOT that a log record exists - #637
    shipped a dead module-scope logger.debug() green because caplog forces level 0
    under pytest while production emitted nothing.
    """
    _fake_gpus(monkeypatch, (0, 2 * GB), (1, 7 * GB))
    conf = {"gpu_split_indices": [0, 1], "gpu_split_ratios": [0.5, 0.5]}
    cfg.save_config(conf)

    note = vram.media_split_notice(conf)

    assert note is not None
    assert "GPU 1" in note, "should name the card it actually chose"
    assert "ONE card" in note


def test_unsplit_box_gets_no_notice(home, monkeypatch):
    """NEGATIVE-TEST: never nag a single-GPU user about a shortfall that cannot exist."""
    _fake_gpus(monkeypatch, (0, 8 * GB))
    cfg.save_config({})

    assert vram.media_split_notice({}) is None
