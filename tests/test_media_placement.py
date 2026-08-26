# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-component GPU PLACEMENT for media generation (the multi-GPU ceiling).

The single-GPU floor names ONE preferred card without masking. This suite covers
the ceiling: on a box ComfyUI sees as 2+ GPUs, push the CLIP text-encoder and the
VAE onto the second card so it carries real work, freeing the compute card for the
model. Mechanism: inject ComfyUI core's `SelectCLIPDevice` / `SelectVAEDevice`
nodes, located BY CLASS, never by hardcoded id.

These tests exercise the REAL shipped workflow JSONs and the REAL helper code
paths, not mocks of them.

localm's MANAGED ComfyUI is pinned at v0.31.1, which DOES ship the placement nodes
(they first appear in tag v0.23.0). A decline on the managed install is therefore
upstream's own rule that the device combo only offers gpu:N once ComfyUI sees MORE
THAN ONE card (model_management.get_gpu_device_options), which on a single-GPU box
yields ['default', 'cpu'] and nothing to place onto.

The capability probe still has real work to do: a ComfyUI of the USER'S OWN may
predate the nodes. These tests cover the probe/plan/inject/notice plumbing;
whether weights actually land on card 1 needs 2 real GPUs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import localm.config as cfg


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Throwaway LOCALM_HOME wired through both home_dir() and the import-frozen
    config attrs."""
    h = tmp_path / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    return h


# --------------------------------------------------------------------------- #
#  Real shipped workflows: the actual JSON, not an invented fixture.           #
# --------------------------------------------------------------------------- #

_REPO = Path(__file__).resolve().parent.parent

GB = 1024 ** 3


def _fake_gpus(monkeypatch, *specs):
    """Patch list_gpus() to report the given (index, free_bytes) devices, and pin
    non-Vulkan so resolve_gpu_split's membership validation runs regardless of how
    the ambient box is provisioned.

    Honors return_status the same way the real list_gpus() does: a bare list by
    default, (list, GPU_PROBE_OK) when a caller opts in.
    media_single_device_shortfall() calls list_gpus(return_status=True), so a
    stand-in that always returned a bare list would make that call's
    `gpus, status = list_gpus(...)` unpack the device dicts instead of a
    status."""
    import localm.discover as disc
    gpus = [{"index": i, "name": f"fake{i}", "free": free, "total": free * 2}
            for i, free in specs]
    monkeypatch.setattr(
        disc, "list_gpus",
        lambda **kw: (list(gpus), disc.GPU_PROBE_OK) if kw.get("return_status")
        else list(gpus))
    monkeypatch.setattr(disc, "_native_backend_has_vulkan", lambda: False)
    return gpus


def _load_workflow(relpath: str) -> dict:
    return json.loads((_REPO / relpath).read_text(encoding="utf-8"))


SHIPPED = {
    "image": "localm/image_gen/flux_workflow.example.json",
    "music": "localm/music_gen/ace_workflow.json",
    "video": "localm/video_gen/wan_workflow.json",
}


def _object_info(*, model=True, clip=True, vae=True, gpu_count=2):
    """A fake ComfyUI /object_info map for the Select*Device nodes, in the REAL shape.

    The Select*Device nodes are v3 (comfy_api schema) nodes, so their device Combo
    serializes as ``["COMBO", {"options": [...]}]`` - the io_type string first,
    choices under ``meta["options"]`` - NOT the classic v1 ``[choices_list, meta]``.
    This fixture uses the v3 shape, which is what the real server emits.

    `gpu_count` controls how many gpu:N options the device combo offers, mirroring
    get_gpu_device_options: "default","cpu" always, plus gpu:0..gpu:{n-1} only when
    the child enumerates >1 device.
    """
    opts = ["default", "cpu"]
    if gpu_count > 1:
        opts += [f"gpu:{i}" for i in range(gpu_count)]

    def spec(kind):
        return {"input": {"required": {
            kind: [kind.upper(), {}],
            "device": ["COMBO", {"options": opts, "default": "default"}],
        }}, "output": [kind.upper()]}

    info = {}
    if model:
        info["SelectModelDevice"] = spec("model")
    if clip:
        info["SelectCLIPDevice"] = spec("clip")
    if vae:
        # the VAE node omits cpu, but that does not matter to the probe
        info["SelectVAEDevice"] = spec("vae")
    # a couple of unrelated nodes so "present" is a real membership test
    info["KSampler"] = {"input": {"required": {"seed": ["INT"]}}}
    return info


# --------------------------------------------------------------------------- #
#  P1: capability probe                                                        #
# --------------------------------------------------------------------------- #

def test_combo_options_parses_both_v1_and_v3_shapes():
    """_combo_options must read choices from BOTH combo serializations ComfyUI emits, or
    the probe silently finds zero GPUs on a v3 node and placement never happens:
    - v1 (classic loaders): ``[choices_list, meta]`` - choices are element 0.
    - v3 (Select*Device / comfy_api): ``["COMBO", {"options": choices_list}]``.
    A non-combo input (``["INT", {...}]`` / ``["MODEL", {...}]``) is NOT a combo -> None."""
    from localm.media.comfy_client import _combo_options
    v1 = {"input": {"required": {"ckpt_name": [["a.safetensors", "b.safetensors"], {}]}}}
    assert _combo_options(v1, "ckpt_name") == ["a.safetensors", "b.safetensors"]
    v3 = {"input": {"required": {
        "device": ["COMBO", {"options": ["default", "cpu", "gpu:0", "gpu:1"]}]}}}
    assert _combo_options(v3, "device") == ["default", "cpu", "gpu:0", "gpu:1"]
    # non-combo inputs, both eras, must return None (not mistaken for a combo)
    assert _combo_options({"input": {"required": {"x": ["INT", {"default": 0}]}}}, "x") is None
    assert _combo_options({"input": {"required": {"m": ["MODEL", {}]}}}, "m") is None
    assert _combo_options({"input": {"required": {}}}, "device") is None


def test_probe_available_when_nodes_present_and_two_gpus(monkeypatch):
    from localm.media import comfy_client as cc
    monkeypatch.setattr(cc, "comfy_object_info",
                        lambda url, timeout=10.0: _object_info(gpu_count=2))
    cap = cc.probe_placement_capability("http://127.0.0.1:8188")
    assert cap.available is True
    assert cap.gpu_options == ["gpu:0", "gpu:1"], (
        "the probe must read the LIVE device combo, which is authoritative about how "
        "many cards ComfyUI actually sees - not localm's own list_gpus (blind to Vulkan)")


def test_probe_unavailable_when_node_absent(monkeypatch):
    """NEGATIVE: a ComfyUI with NO Select*Device nodes. The probe must say
    unavailable and NAME why, so placement never injects a node ComfyUI would
    reject. This is the user's-own-ComfyUI case (anything older than tag v0.23.0),
    not the managed install."""
    from localm.media import comfy_client as cc
    monkeypatch.setattr(cc, "comfy_object_info",
                        lambda url, timeout=10.0: _object_info(model=False, clip=False,
                                                               vae=False, gpu_count=2))
    cap = cc.probe_placement_capability("http://127.0.0.1:8188")
    assert cap.available is False
    assert cap.reason and "placement" in cap.reason.lower()


def test_probe_unavailable_when_only_one_gpu(monkeypatch):
    """NEGATIVE: nodes present but the child sees ONE GPU (this very box). The device
    combo then offers no gpu:N, so there is nothing to place onto -> unavailable."""
    from localm.media import comfy_client as cc
    monkeypatch.setattr(cc, "comfy_object_info",
                        lambda url, timeout=10.0: _object_info(gpu_count=1))
    cap = cc.probe_placement_capability("http://127.0.0.1:8188")
    assert cap.available is False
    assert [o for o in cap.gpu_options if o.startswith("gpu:")] == []


def test_probe_unavailable_when_server_unreachable(monkeypatch):
    """NEGATIVE: /object_info could not be fetched -> unavailable, best-effort, named."""
    from localm.media import comfy_client as cc
    monkeypatch.setattr(cc, "comfy_object_info", lambda url, timeout=10.0: None)
    cap = cc.probe_placement_capability("http://127.0.0.1:8188")
    assert cap.available is False
    assert cap.reason


# --------------------------------------------------------------------------- #
#  The pin and the prose must not drift apart: the help text may not describe  #
#  localm's own managed ComfyUI as unable to place.                            #
# --------------------------------------------------------------------------- #

def _ver(s: str) -> tuple:
    """'v0.31.1' -> (0, 31, 1). Numeric compare, so v0.9.2 < v0.23.0, which a plain
    string compare gets backwards."""
    return tuple(int(p) for p in s.lstrip("v").split("."))


def _placement_help() -> str:
    from localm.settings_schema import CORE_FIELDS
    field = next(f for f in CORE_FIELDS if f.key == "comfy_gpu_placement")
    return field.help


def test_pinned_comfyui_is_new_enough_for_per_component_placement():
    """The shipped pin must be at or after the first upstream tag carrying
    comfy_extras/nodes_multigpu.py, which is what the help text below claims in
    prose."""
    from localm.media.managed_comfy_fresh import (COMFYUI_PINNED_VERSION,
                                                  COMFYUI_PLACEMENT_MIN_VERSION)
    assert _ver(COMFYUI_PINNED_VERSION) >= _ver(COMFYUI_PLACEMENT_MIN_VERSION), (
        f"pin {COMFYUI_PINNED_VERSION} predates {COMFYUI_PLACEMENT_MIN_VERSION}, the "
        "first ComfyUI release with the Select*Device nodes - so the managed install "
        "cannot place, and comfy_gpu_placement's help text must say so honestly")


def test_placement_help_does_not_call_localms_own_comfyui_too_old():
    """Given the pin above DOES offer placement, the user-facing text must not still
    tell users localm's own managed ComfyUI is the thing holding them back."""
    import re
    from localm.media.managed_comfy_fresh import (COMFYUI_PINNED_VERSION,
                                                  COMFYUI_PLACEMENT_MIN_VERSION)
    assert _ver(COMFYUI_PINNED_VERSION) >= _ver(COMFYUI_PLACEMENT_MIN_VERSION)

    help_text = _placement_help()
    # managed ... older/too old/does not/predates in one breath is the shape
    # being refused.
    bad = re.search(
        r"managed[^.]{0,80}?\b(is older|older than|too old|predates|cannot|does not "
        r"(?:offer|support))", help_text, re.IGNORECASE)
    assert not bad, (
        "comfy_gpu_placement's help text still describes localm's OWN managed ComfyUI "
        f"as unable to place, but the pin is {COMFYUI_PINNED_VERSION}: {bad.group(0)!r}")

    # No retired pin version may be quoted as the current one.
    for stale in re.findall(r"\bv\d+\.\d+\.\d+\b", help_text):
        assert stale == COMFYUI_PINNED_VERSION, (
            f"help text names ComfyUI {stale} but the pin is {COMFYUI_PINNED_VERSION}")


def test_placement_help_still_states_the_two_gpu_precondition():
    """The other direction: two or more GPUs is a REAL precondition. Upstream's
    get_gpu_device_options only appends gpu:N when it sees MORE THAN ONE card, so on
    a single-GPU box the combo is ['default', 'cpu'] and there is nothing to place
    onto."""
    help_text = _placement_help().lower()
    assert "two or more" in help_text or "2+" in help_text, (
        "the help text must keep stating that placement needs 2+ GPUs")
    assert "single-gpu" in help_text or "single gpu" in help_text, (
        "the help text must keep telling single-GPU users nothing changes for them")


# --------------------------------------------------------------------------- #
#  P2: planner (pure)                                                          #
# --------------------------------------------------------------------------- #

def test_plan_pushes_clip_and_vae_to_second_card(home):
    from localm.discover import plan_media_placement
    cfg.save_config({"gpu_split_indices": [0, 1]})
    plan = plan_media_placement(cfg.load_config(), gpu_options=["gpu:0", "gpu:1"])
    assert plan == {"model": None, "clip": "gpu:1", "vae": "gpu:1"}, (
        "v1 policy: keep the big model on the preferred card (gpu:0, no injection) and "
        "offload the smaller CLIP + VAE to the second card - no free-VRAM read needed")


def test_plan_none_when_fewer_than_two_gpu_options(home):
    """NEGATIVE: single card visible -> no plan (single-GPU floor unchanged). Gating on
    the probe's gpu_options, NOT split_device_count, so the Vulkan hole is never
    inherited."""
    from localm.discover import plan_media_placement
    cfg.save_config({"gpu_split_indices": [0, 1]})
    for opts in ([], ["gpu:0"], None):
        assert plan_media_placement(cfg.load_config(), gpu_options=opts) is None


# --------------------------------------------------------------------------- #
#  P3: injection BY CLASS into the real shipped workflows                      #
# --------------------------------------------------------------------------- #

def _link(node, input_name):
    return node["inputs"].get(input_name)


@pytest.mark.parametrize("medium", ["image", "music", "video"])
def test_inject_places_clip_and_vae_by_class(medium):
    from localm.media.comfy_client import inject_device_placement, find_component_producer
    wf = _load_workflow(SHIPPED[medium])
    clip_prod = find_component_producer(wf, "clip")
    vae_prod = find_component_producer(wf, "vae")
    assert clip_prod is not None, f"{medium}: a CLIP producer must be findable by class"
    assert vae_prod is not None, f"{medium}: a VAE producer must be findable by class"

    # who consumed the CLIP/VAE outputs BEFORE injection (excluding nothing yet)
    def consumers(prod):
        pid, slot = prod
        out = []
        for nid, node in wf.items():
            for name, val in node.get("inputs", {}).items():
                if isinstance(val, list) and len(val) == 2 and val == [pid, slot]:
                    out.append((nid, name))
        return out

    clip_consumers = consumers(clip_prod)
    vae_consumers = consumers(vae_prod)
    assert clip_consumers, f"{medium}: the template must have a CLIP consumer to rewire"

    inject_device_placement(wf, {"model": None, "clip": "gpu:1", "vae": "gpu:1"})

    # exactly one SelectCLIPDevice + one SelectVAEDevice added, device gpu:1
    sel_clip = [(i, n) for i, n in wf.items() if n.get("class_type") == "SelectCLIPDevice"]
    sel_vae = [(i, n) for i, n in wf.items() if n.get("class_type") == "SelectVAEDevice"]
    assert len(sel_clip) == 1 and len(sel_vae) == 1
    ci, cnode = sel_clip[0]
    vi, vnode = sel_vae[0]
    assert cnode["inputs"]["device"] == "gpu:1"
    assert vnode["inputs"]["device"] == "gpu:1"

    # the select node reads FROM the loader...
    assert _link(cnode, "clip") == list(clip_prod), (
        f"{medium}: SelectCLIPDevice must read the CLIP loader's original output")
    assert _link(vnode, "vae") == list(vae_prod)

    # ...and every ORIGINAL consumer now reads from the select node, not the loader
    for nid, name in clip_consumers:
        assert wf[nid]["inputs"][name] == [ci, 0], (
            f"{medium}: consumer {nid}.{name} must be rewired to SelectCLIPDevice")
    for nid, name in vae_consumers:
        assert wf[nid]["inputs"][name] == [vi, 0]


def test_inject_locates_by_class_not_id():
    """Transforms that hardcode ids ("6","31",...) KeyError on a user's arbitrary
    graph. Injection must locate BY CLASS: renumber every node to a fresh id space
    and confirm it still finds and rewires the loaders."""
    from localm.media.comfy_client import inject_device_placement
    wf = _load_workflow(SHIPPED["image"])
    # remap all ids to a disjoint space (1000+), rewriting links too
    old_ids = list(wf.keys())
    remap = {old: str(1000 + i) for i, old in enumerate(old_ids)}
    new_wf = {}
    for old, node in wf.items():
        node = copy.deepcopy(node)
        for name, val in node.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2 and str(val[0]) in remap:
                node["inputs"][name] = [remap[str(val[0])], val[1]]
        new_wf[remap[old]] = node

    inject_device_placement(new_wf, {"model": None, "clip": "gpu:1", "vae": "gpu:1"})
    assert any(n.get("class_type") == "SelectCLIPDevice" for n in new_wf.values())
    assert any(n.get("class_type") == "SelectVAEDevice" for n in new_wf.values())


def test_inject_skips_missing_component_without_crashing():
    """NEGATIVE: a graph with no VAE loader -> VAE placement is skipped with a note, not
    a KeyError. (Build a minimal graph with only a CLIP loader.)"""
    from localm.media.comfy_client import inject_device_placement
    wf = {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "x.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0], "text": "hi"}},
    }
    notes = inject_device_placement(wf, {"model": None, "clip": "gpu:1", "vae": "gpu:1"})
    assert any(n.get("class_type") == "SelectCLIPDevice" for n in wf.values())
    assert not any(n.get("class_type") == "SelectVAEDevice" for n in wf.values())
    assert notes, "a skipped component must be surfaced in the returned notes (rule 5)"


def test_inject_skips_gguf_clip_loader_with_specific_reason():
    """A CUSTOM workflow whose CLIP comes from a GGUF loader must NOT be
    cross-device-moved: GGUF loaders do not register the cached_patcher_init factory
    that deepclone_multigpu needs, so wrapping one trips a ComfyUI RuntimeError.
    localm skips it with a SPECIFIC reason, never a generic 'no loader'. Only
    CORE-loaded components are relocated."""
    from localm.media.comfy_client import inject_device_placement
    wf = {
        "1": {"class_type": "DualCLIPLoaderGGUF",
              "inputs": {"clip_name1": "a.gguf", "clip_name2": "b.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0], "text": "hi"}},
    }
    notes = inject_device_placement(wf, {"model": None, "clip": "gpu:1"})
    # No SelectCLIPDevice was injected (the GGUF CLIP was NOT moved).
    assert not any(n.get("class_type") == "SelectCLIPDevice" for n in wf.values()), (
        "a GGUF CLIP loader must not be wrapped for cross-device placement yet")
    # The CLIPTextEncode still reads straight from the loader (untouched).
    assert wf["2"]["inputs"]["clip"] == ["1", 0]
    # The skip is surfaced with the SPECIFIC GGUF reason, not the generic no-loader one.
    joined = " ".join(notes).lower()
    assert "gguf" in joined and "clip" in joined, (
        "the skip note must name the GGUF loader as the reason (rule 5), got: " + joined)


# --------------------------------------------------------------------------- #
#  P4: the notice that becomes a lie the moment placement works                #
# --------------------------------------------------------------------------- #

def test_notice_reflects_active_placement(home):
    """An ACTIVE plan drives the notice, even with no chat split configured: placement is
    capability-driven (2+ GPUs visible), not chat-split-driven."""
    from localm.vram import media_split_notice
    cfg.save_config({})
    plan = {"model": None, "clip": "gpu:1", "vae": "gpu:1"}
    notice = media_split_notice(cfg.load_config(), placement=plan)
    assert notice
    assert "runs on ONE card" not in notice, (
        "with placement active the 'runs on ONE card' clause is a lie")
    low = notice.lower()
    assert "vae" in low or "encoder" in low or "gpu 1" in low or "component" in low, (
        "the placement notice must tell the user components were placed on the 2nd card")


def test_notice_states_honest_reason_when_capability_unavailable(home, monkeypatch):
    """Split configured that RESOLVES to 2 real cards, but the ComfyUI cannot place (old
    pin / node absent): the notice must give the HONEST reason from the capability, not
    the flat single-card message. Needs a faked 2-GPU box so split_device_count >= 2 -
    on a 1-GPU box the split does not resolve and the notice correctly stays silent."""
    from localm.vram import media_split_notice
    from localm.media.comfy_client import PlacementCapability
    _fake_gpus(monkeypatch, (0, 6 * GB), (1, 6 * GB))
    cfg.save_config({"gpu_split_indices": [0, 1]})
    cap = PlacementCapability(available=False, gpu_options=[],
                              reason="this ComfyUI does not offer per-component placement")
    notice = media_split_notice(cfg.load_config(), placement=None, capability=cap)
    assert notice and "per-component placement" in notice


def test_notice_none_without_split_and_no_active_placement(home):
    """NEGATIVE: no split configured AND no active placement -> no notice, ever (do not
    nag a single-GPU user). A plan whose targets are all None is not active placement."""
    from localm.vram import media_split_notice
    cfg.save_config({})
    assert media_split_notice(cfg.load_config()) is None
    assert media_split_notice(cfg.load_config(),
                              placement={"model": None, "clip": None, "vae": None}) is None


# --------------------------------------------------------------------------- #
#  P5: media_single_device_shortfall - the whole-model-on-one-card check.      #
#  The v1 placement policy keeps the model on the preferred card.             #
# --------------------------------------------------------------------------- #

def test_shortfall_refuses_reg532_scenario(home, monkeypatch):
    """Two cards, 4 GB free EACH (8 GB combined), a 4 GB job. decide_media_swap says
    "fits" on the 8 GB combined number and keeps the chat model loaded; the
    per-device check must STILL refuse, because no SINGLE card holds the whole 4 GB
    model + headroom. Asserted against the WHOLE-model predicate: a
    proportional/ratio share would wrongly pass this."""
    from localm.vram import media_single_device_shortfall, decide_media_swap
    _fake_gpus(monkeypatch, (0, 4 * GB), (1, 4 * GB))
    conf = {"gpu_split_indices": [0, 1]}
    settings = {"swap_policy": "auto", "vram_estimate_bytes": 4 * GB}
    # The combined gate is fooled: 8 GB "fits" a 4 GB job, so it would NOT swap.
    assert decide_media_swap(settings, read_free=lambda: 8 * GB) is False
    # The per-device check is not fooled: neither 4 GB card holds 4 GB + 1 GB headroom.
    shortfall = media_single_device_shortfall(settings, config=conf)
    assert shortfall is not None, "no single 4 GB card holds a 4 GB model + headroom"
    assert shortfall["free"] == 4 * GB
    assert shortfall["needed"] > 4 * GB
    assert shortfall["index"] in (0, 1)


def test_shortfall_none_when_chosen_card_fits_whole_model(home, monkeypatch):
    """NEGATIVE: it must NOT refuse a load that would have worked. Card 0 has 4 GB (would
    fail), card 1 has 12 GB (fits). resolve_preferred_device picks the most-free card (1),
    which holds the whole 4 GB model + headroom -> None. This also proves the check reads
    the card actually chosen, not just card 0."""
    from localm.vram import media_single_device_shortfall
    _fake_gpus(monkeypatch, (0, 4 * GB), (1, 12 * GB))
    conf = {"gpu_split_indices": [0, 1]}
    settings = {"swap_policy": "auto", "vram_estimate_bytes": 4 * GB}
    assert media_single_device_shortfall(settings, config=conf) is None


def test_shortfall_none_when_policy_not_auto(home, monkeypatch):
    """NEGATIVE: an explicit always/never swap policy is the user's choice and is not
    second-guessed, even in the OOM scenario."""
    from localm.vram import media_single_device_shortfall
    _fake_gpus(monkeypatch, (0, 4 * GB), (1, 4 * GB))
    conf = {"gpu_split_indices": [0, 1]}
    for policy in ("always", "never"):
        settings = {"swap_policy": policy, "vram_estimate_bytes": 4 * GB}
        assert media_single_device_shortfall(settings, config=conf) is None


def test_shortfall_none_when_probe_did_not_complete_fresh(home, monkeypatch):
    """The OOM scenario, but the probe this call made is TIMEOUT/BUSY, not
    GPU_PROBE_OK - a served last-known-good list is not a current measurement.
    Must return None (cannot check this call), never compute a shortfall, and
    never a stale one, from it."""
    from localm.vram import media_single_device_shortfall
    import localm.discover as disc
    gpus = [{"index": 0, "name": "fake0", "free": 4 * GB, "total": 8 * GB},
            {"index": 1, "name": "fake1", "free": 4 * GB, "total": 8 * GB}]
    monkeypatch.setattr(
        disc, "list_gpus",
        lambda **kw: (list(gpus), disc.GPU_PROBE_TIMEOUT)
        if kw.get("return_status") else list(gpus))
    monkeypatch.setattr(disc, "_native_backend_has_vulkan", lambda: False)
    conf = {"gpu_split_indices": [0, 1]}
    settings = {"swap_policy": "auto", "vram_estimate_bytes": 4 * GB}
    assert media_single_device_shortfall(settings, config=conf) is None


def test_shortfall_none_on_single_gpu_box(home, monkeypatch):
    """NEGATIVE: no split configured (single-GPU box) -> the combined reading already IS
    the single card's, so there is nothing extra to check -> None."""
    from localm.vram import media_single_device_shortfall
    _fake_gpus(monkeypatch, (0, 4 * GB))
    settings = {"swap_policy": "auto", "vram_estimate_bytes": 4 * GB}
    assert media_single_device_shortfall(settings, config={}) is None


# --------------------------------------------------------------------------- #
#  P6: resolve_media_placement - the shared plugin entry point that decides    #
#  the plan + notice. OFF by default; all three outcomes are covered.          #
# --------------------------------------------------------------------------- #

def test_resolve_placement_disabled_by_default_plain_box(home):
    """With the flag absent AND no split there is no plan and no notice: the
    feature never activates itself."""
    from localm.media.comfy_client import resolve_media_placement
    cfg.save_config({})
    plan, notice = resolve_media_placement(cfg.load_config(), "http://x")
    assert plan is None and notice is None


def test_resolve_placement_disabled_keeps_legacy_notice_on_split(home, monkeypatch):
    """Flag OFF but a split configured -> no plan, but the legacy single-card notice still
    fires (behaviour unchanged from before placement existed)."""
    from localm.media.comfy_client import resolve_media_placement
    _fake_gpus(monkeypatch, (0, 6 * GB), (1, 6 * GB))
    cfg.save_config({"gpu_split_indices": [0, 1]})
    plan, notice = resolve_media_placement(cfg.load_config(), "http://x")
    assert plan is None
    assert notice and "one card" in notice.lower()


def test_resolve_placement_active_when_enabled_and_capable(home, monkeypatch):
    """Flag ON + the running ComfyUI reports the nodes and 2 GPUs -> a real plan (CLIP+VAE
    to gpu:1) and the placement-active notice. This is the wiring that makes the second
    card carry work. The probe is stubbed, so no real ComfyUI is contacted."""
    from localm.media import comfy_client as cc
    from localm.media.comfy_client import resolve_media_placement, PlacementCapability
    _fake_gpus(monkeypatch, (0, 6 * GB), (1, 6 * GB))
    monkeypatch.setattr(cc, "probe_placement_capability",
                        lambda api_url, **kw: PlacementCapability(True, ["gpu:0", "gpu:1"]))
    cfg.save_config({"comfy_gpu_placement": True, "gpu_split_indices": [0, 1]})
    plan, notice = resolve_media_placement(cfg.load_config(), "http://x")
    assert plan == {"model": None, "clip": "gpu:1", "vae": "gpu:1"}
    assert notice and "runs on ONE card" not in notice


def test_resolve_placement_states_reason_when_enabled_but_incapable(home, monkeypatch):
    """Flag ON but the ComfyUI cannot place (old pin without the nodes, or one GPU):
    no plan, and the notice must give the HONEST reason. Enabled-but-not-delivered
    is SAID, never silent."""
    from localm.media import comfy_client as cc
    from localm.media.comfy_client import resolve_media_placement, PlacementCapability
    _fake_gpus(monkeypatch, (0, 6 * GB), (1, 6 * GB))
    monkeypatch.setattr(
        cc, "probe_placement_capability",
        lambda api_url, **kw: PlacementCapability(
            False, [], "this ComfyUI does not offer per-component GPU placement"))
    cfg.save_config({"comfy_gpu_placement": True, "gpu_split_indices": [0, 1]})
    plan, notice = resolve_media_placement(cfg.load_config(), "http://x")
    assert plan is None
    assert notice and "per-component" in notice


def test_resolve_placement_declines_without_configured_split(home, monkeypatch):
    """Gate 2: even with the toggle ON and a fully capable ComfyUI (nodes present, 2
    gpu:N options), placement needs a configured 2+ card split. No split ->
    single-card floor, no plan."""
    from localm.media import comfy_client as cc
    from localm.media.comfy_client import resolve_media_placement, PlacementCapability
    _fake_gpus(monkeypatch, (0, 6 * GB), (1, 6 * GB))
    monkeypatch.setattr(cc, "probe_placement_capability",
                        lambda api_url, **kw: PlacementCapability(True, ["gpu:0", "gpu:1"]))
    # Toggle ON, capable probe, but NO gpu_split_indices configured.
    cfg.save_config({"comfy_gpu_placement": True})
    plan, notice = resolve_media_placement(cfg.load_config(), "http://x")
    assert plan is None, "gate 2 must decline placement without a configured 2+ split"


def test_resolve_placement_disabled_ignores_capable_comfyui(home, monkeypatch):
    """Gate 1: the experimental toggle OFF wins even when everything else is ready
    (a 2+ split configured AND a capable ComfyUI)."""
    from localm.media import comfy_client as cc
    from localm.media.comfy_client import resolve_media_placement, PlacementCapability
    _fake_gpus(monkeypatch, (0, 6 * GB), (1, 6 * GB))
    monkeypatch.setattr(cc, "probe_placement_capability",
                        lambda api_url, **kw: PlacementCapability(True, ["gpu:0", "gpu:1"]))
    # Everything ready EXCEPT the toggle (absent -> default False).
    cfg.save_config({"gpu_split_indices": [0, 1]})
    plan, notice = resolve_media_placement(cfg.load_config(), "http://x")
    assert plan is None, "gate 1 (toggle off) must keep the single-card floor"
    assert notice and "one card" in notice.lower()


# --------------------------------------------------------------------------- #
#  P7: LIVE WIRING - the plan reaches injection inside generate_* and mutates  #
#  the SUBMITTED workflow.                                                     #
# --------------------------------------------------------------------------- #

def test_generate_music_injects_placement_into_submitted_workflow(tmp_path, monkeypatch):
    """Drive the REAL generate_music to the submit boundary and capture the workflow
    it would POST. A placement plan passed in must have injected SelectCLIPDevice /
    SelectVAEDevice by the time of submit."""
    from localm.music_gen import comfy as mcomfy
    from localm.media import comfy_client
    captured = {}

    def fake_submit(api_url, workflow, **kw):
        captured["wf"] = copy.deepcopy(workflow)
        return comfy_client.SUBMIT_ERROR, RuntimeError("stop after capture")

    with patch.object(mcomfy, "ensure_comfy", return_value=(True, "up")), \
         patch.object(mcomfy, "preflight_models", return_value=(True, "")), \
         patch.object(mcomfy, "_localm_unload", MagicMock()), \
         patch.object(mcomfy, "comfy_submit_prompt", fake_submit):
        mcomfy.generate_music(
            "synthwave", tmp_path / "out.flac", swap=False,
            placement={"model": None, "clip": "gpu:1", "vae": "gpu:1"})

    wf = captured.get("wf")
    assert wf is not None, "generate_music never reached submit"
    sel_clip = [n for n in wf.values() if n.get("class_type") == "SelectCLIPDevice"]
    sel_vae = [n for n in wf.values() if n.get("class_type") == "SelectVAEDevice"]
    assert sel_clip, "the placement plan did NOT reach inject inside generate_music (dead wiring)"
    assert sel_vae
    assert sel_clip[0]["inputs"]["device"] == "gpu:1"


def test_generate_music_no_placement_leaves_workflow_untouched(tmp_path, monkeypatch):
    """NEGATIVE: placement=None (the default / disabled path) must inject NOTHING - the
    submitted workflow is the plain shipped graph, no Select*Device nodes."""
    from localm.music_gen import comfy as mcomfy
    from localm.media import comfy_client
    captured = {}

    def fake_submit(api_url, workflow, **kw):
        captured["wf"] = copy.deepcopy(workflow)
        return comfy_client.SUBMIT_ERROR, RuntimeError("stop after capture")

    with patch.object(mcomfy, "ensure_comfy", return_value=(True, "up")), \
         patch.object(mcomfy, "preflight_models", return_value=(True, "")), \
         patch.object(mcomfy, "_localm_unload", MagicMock()), \
         patch.object(mcomfy, "comfy_submit_prompt", fake_submit):
        mcomfy.generate_music("synthwave", tmp_path / "out.flac", swap=False, placement=None)

    wf = captured.get("wf")
    assert wf is not None
    assert not any(str(n.get("class_type", "")).startswith("Select")
                   and n.get("class_type") in ("SelectModelDevice", "SelectCLIPDevice",
                                                "SelectVAEDevice")
                   for n in wf.values()), "no placement plan must mean no injection"
