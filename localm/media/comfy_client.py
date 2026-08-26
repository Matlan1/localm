# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Shared ComfyUI client - generic plumbing for image / video / music generation.

One ComfyUI server, one set of helpers. This module holds everything that is
NOT specific to a single medium's workflow: role-based node resolution, model
preflight against ``/object_info``, the VRAM handoff, server reachability and
launch, and the queue / poll / download transport shared by every generator.

The image, video, and music modules import from here and keep only their own
workflow shaping (which nodes to inject, which output keys to read).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from localm.http_ssl import verified_urlopen
from localm.pathsafe import confined_under, is_unc_or_device_path


# ---------------------------------------------------------------------------
#  Role-based node resolution (resolve nodes by class_type + graph edges)
# ---------------------------------------------------------------------------
#
#  A ComfyUI workflow is a dict of {node_id: {"class_type": str, "inputs": {...}}}.
#  An input value is either a literal or a LINK [source_node_id, output_index].
#  Hardcoding node ids (workflow["8"]["inputs"]["seed"]) breaks the moment a user
#  exports their own graph from ComfyUI, because the ids are arbitrary. Resolving
#  by ROLE - find the sampler by class_type, then follow its positive / negative /
#  latent edges to their source nodes - works on any graph that wires those roles,
#  so a local override no longer has to preserve the template's exact ids (I3).

# Sampler node classes that carry the seed / steps / cfg knobs and the
# positive / negative / latent_image edges we trace the other roles from.
_SAMPLER_CLASSES = (
    "KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced",
)


def _is_link(value) -> bool:
    """True when an input value is a ComfyUI link: ``[source_node_id, output_index]``."""
    return (isinstance(value, list) and len(value) == 2
            and isinstance(value[0], (str, int)) and not isinstance(value[0], bool)
            and isinstance(value[1], int) and not isinstance(value[1], bool))


def _link_source_id(node: dict, input_name: str) -> Optional[str]:
    """The source node id an input is wired to, or None when it is a literal."""
    if not isinstance(node, dict):
        return None
    value = node.get("inputs", {}).get(input_name)
    return str(value[0]) if _is_link(value) else None


def find_node_by_class(workflow: dict, *class_types: str):
    """First ``(id, node)`` whose class_type is one of *class_types*, else
    ``(None, None)``. Dict insertion order is preserved, so the first matching node
    in the graph wins."""
    for nid, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") in class_types:
            return str(nid), node
    return None, None


def find_nodes_by_class(workflow: dict, *class_types: str) -> list:
    """All ``(id, node)`` pairs whose class_type is one of *class_types*."""
    return [(str(nid), node) for nid, node in workflow.items()
            if isinstance(node, dict) and node.get("class_type") in class_types]


def resolve_sampler_roles(workflow: dict) -> dict:
    """Resolve ``{sampler, positive, negative, latent}`` to ``(node_id, node)`` by
    following the sampler's input edges, so node ids can be anything.

    A role is ``(None, None)`` when the sampler or that edge is absent. ``positive``
    / ``negative`` point at whatever conditioning node feeds them (a CLIPTextEncode,
    a TextEncodeAceStepAudio, a ConditioningZeroOut, ...); the caller injects the
    field that node actually has rather than assuming a text box."""
    roles = {"sampler": (None, None), "positive": (None, None),
             "negative": (None, None), "latent": (None, None)}
    sid, sampler = find_node_by_class(workflow, *_SAMPLER_CLASSES)
    if sampler is None:
        return roles
    roles["sampler"] = (sid, sampler)
    for role, input_name in (("positive", "positive"),
                             ("negative", "negative"),
                             ("latent", "latent_image")):
        src = _link_source_id(sampler, input_name)
        if src is not None and src in workflow:
            roles[role] = (src, workflow[src])
    return roles


def set_seed_on(node: dict, seed: int) -> None:
    """Set whichever seed field a sampler node actually has (KSampler uses ``seed``;
    a RandomNoise / KSamplerAdvanced uses ``noise_seed``). Never invents a field the
    node does not declare, which ComfyUI would reject."""
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return
    if "seed" in inputs:
        inputs["seed"] = seed
    elif "noise_seed" in inputs:
        inputs["noise_seed"] = seed


def set_seed_on_all(workflow: dict, seed: int) -> int:
    """Set the seed on EVERY sampler/noise node in the graph and return how many were
    set. A multi-sampler graph (a two-stage Wan high/low-noise split, an SDXL refiner)
    must get the seed on each stage, or a later stage stays on the template's fixed
    seed - making "random" output partially deterministic. Only touches nodes that
    actually declare a seed field (set_seed_on is a no-op otherwise)."""
    n = 0
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") in _SAMPLER_CLASSES or \
                node.get("class_type") == "RandomNoise":
            inputs = node.get("inputs")
            if isinstance(inputs, dict) and ("seed" in inputs or "noise_seed" in inputs):
                set_seed_on(node, seed)
                n += 1
    return n


def next_node_id(workflow: dict) -> str:
    """A node id not already used by *workflow* (max numeric id + 1, else a stable
    name). Injected helper nodes (e.g. a LoadImage for img2img/img2video) use this
    instead of a hardcoded id, which could clobber a node in a user's own graph."""
    numeric = [int(k) for k in workflow if str(k).isdigit()]
    if numeric:
        return str(max(numeric) + 1)
    base, i = "localm_node_", 0
    while f"{base}{i}" in workflow:
        i += 1
    return f"{base}{i}"


# ---------------------------------------------------------------------------
#  Per-component GPU placement: inject core Select*Device nodes by class
# ---------------------------------------------------------------------------
#
#  On a box ComfyUI sees as 2+ GPUs, ComfyUI core's SelectModelDevice /
#  SelectCLIPDevice / SelectVAEDevice nodes rehome a component onto another card
#  (deepclone_multigpu -> the loader's cached_patcher_init factory -> independent
#  weights on the target). We inject them so the second card carries real work.
#
#  Located BY CLASS, never by id: the existing template transforms hardcode ids
#  ("6","31",...) and KeyError on a user's arbitrary exported graph. A loader may
#  emit several components (CheckpointLoaderSimple: model=slot0, clip=slot1,
#  vae=slot2); a dedicated loader emits its one component at slot 0.

# Component -> ordered (loader class_type, output slot) candidates. First match in
# the graph wins, so a checkpoint's own clip/vae outputs are preferred over a
# separate loader only when both somehow exist (the shipped graphs never do).
_COMPONENT_LOADERS = {
    "model": (("CheckpointLoaderSimple", 0), ("UNETLoader", 0),
              ("UnetLoaderGGUF", 0), ("UnetLoaderGGUFAdvanced", 0)),
    "clip": (("CheckpointLoaderSimple", 1), ("CLIPLoader", 0), ("DualCLIPLoader", 0),
             ("DualCLIPLoaderGGUF", 0), ("CLIPLoaderGGUF", 0)),
    "vae": (("CheckpointLoaderSimple", 2), ("VAELoader", 0)),
}

# Component -> the core placement node and the input name that carries the component.
_SELECT_NODE = {"model": "SelectModelDevice", "clip": "SelectCLIPDevice",
                "vae": "SelectVAEDevice"}

# Loaders that do NOT register ComfyUI's ``cached_patcher_init`` factory (ComfyUI-GGUF
# calls ``load_diffusion_model_state_dict`` / ``load_text_encoder_state_dicts`` directly,
# bypassing the path-based wrappers that set it - custom_nodes/ComfyUI-GGUF, pin 6ea2651).
# Cross-device placement of such a component needs that factory (``deepclone_multigpu``
# re-invokes it), so moving one would trip a ComfyUI RuntimeError that degrades back to a
# single card. Registering it is a deliberate follow-up (see SPEC-placement.md, out-of-
# scope #2). Until it lands we SKIP a GGUF-loaded component with a SPECIFIC reason rather
# than emit a node ComfyUI cannot honor - matching the deferral's own justification (only
# CORE-loaded components are relocated) and surfacing WHY at localm's level (rule 5). The
# shipped image workflow's UNet is GGUF, but the size-free plan keeps the model on its
# card (never moved), so this only ever fires for a user's CUSTOM GGUF CLIP/VAE graph.
_GGUF_LOADERS = frozenset({
    "UnetLoaderGGUF", "UnetLoaderGGUFAdvanced", "DualCLIPLoaderGGUF", "CLIPLoaderGGUF",
})


def find_component_producer(workflow: dict, component: str):
    """The ``(node_id, output_slot)`` producing *component*'s output, located BY CLASS
    (so a user's arbitrary graph works), or ``None`` when no such loader is present."""
    for class_type, slot in _COMPONENT_LOADERS.get(component, ()):  # noqa: E501
        nid, _node = find_node_by_class(workflow, class_type)
        if nid is not None:
            return (nid, slot)
    return None


def _reroute_output(workflow: dict, producer_id, slot: int, new_producer_id: str) -> None:
    """Repoint every input wired to ``[producer_id, slot]`` onto ``[new_producer_id, 0]``.

    Called BEFORE the new node is inserted, so the new node's own input (which will read
    from the loader) is never itself rewired. Id comparison is str-tolerant because a
    link source id may be int or str across hand-written vs exported graphs."""
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        for name, val in node.get("inputs", {}).items():
            if _is_link(val) and str(val[0]) == str(producer_id) and val[1] == slot:
                node["inputs"][name] = [new_producer_id, 0]


def inject_device_placement(workflow: dict, plan: Optional[dict]) -> list:
    """Inject the core ``Select*Device`` nodes to place each component named in *plan* on
    its target ``gpu:N``, rewiring the loader's consumers through the new node.

    *plan* is ``{"model": "gpu:N"|None, "clip": ..., "vae": ...}``
    (:func:`localm.discover.plan_media_placement`); a ``None``/empty target keeps that
    component on its loader default (no injection). Returns human-readable notes: a
    component that could NOT be placed (no matching loader in this graph) is surfaced in
    the notes, never silently dropped (AGENTS rule 5). The caller pushes these to the
    user so a requested-but-skipped placement is visible."""
    notes = []
    for component, target in (plan or {}).items():
        if not target:
            continue
        select_class = _SELECT_NODE.get(component)
        if select_class is None:
            continue
        producer = find_component_producer(workflow, component)
        if producer is None:
            notes.append(f"could not place {component} on {target}: no {component} loader "
                         "in this workflow (left on the default card)")
            continue
        pid, slot = producer
        producer_class = workflow.get(pid, {}).get("class_type", "")
        if producer_class in _GGUF_LOADERS:
            # A GGUF loader has no cross-device factory yet (see _GGUF_LOADERS): moving it
            # would trip a ComfyUI RuntimeError and degrade back to one card anyway. Skip
            # it here with the specific reason instead of emitting a node that cannot work.
            notes.append(f"could not place {component} on {target}: this workflow's "
                         f"{component} uses a GGUF loader ({producer_class}), which cannot "
                         "yet be moved across cards (a pending follow-up); left on the "
                         "default card")
            continue
        new_id = next_node_id(workflow)  # reserved; not yet in workflow, so reroute skips it
        _reroute_output(workflow, pid, slot, new_id)
        workflow[new_id] = {
            "class_type": select_class,
            "inputs": {component: [pid, slot], "device": target},
        }
        notes.append(f"placing {component} on {target}")
    return notes


def resolve_media_placement(config: Optional[dict], api_url: str):
    """Decide per-component GPU placement for one media job: ``(plan, notice)``.

    The single DRY entry point the image/music/video plugins share (their VRAM/placement
    preamble is otherwise byte-identical). ``plan`` is passed to ``generate_*`` to inject
    the ``Select*Device`` nodes; ``notice`` is the user-facing line the plugin pushes.

    Three outcomes, all honest (rule 5 - a requested capability that is not delivered is
    stated, never silently dropped):

    - **Placement disabled** (the default until the 2-GPU mechanism is proven on real
      hardware - the experimental ``comfy_gpu_placement`` toggle is off, or no 2+ card
      split is configured): returns ``(None, <legacy notice>)`` - behaviour is exactly as
      before, the single-card notice on a configured split and nothing on a plain box.
    - **Enabled and the running ComfyUI can place** (probe finds the Select*Device nodes
      and 2+ GPUs): returns ``(plan, <placement notice>)`` - the second card carries the
      text encoder + VAE.
    - **Enabled but the running ComfyUI cannot** (an old ComfyUI of the USER'S OWN,
      predating the nodes - or only one GPU visible): returns ``(None, <honest
      reason notice>)`` so the user learns WHY placement did not happen.

    Best-effort and never raises: the probe swallows its own errors and any doubt yields
    no plan (single-card floor). ``api_url`` must point at a ComfyUI already confirmed
    alive (call after ``ensure_available`` / ``ensure_comfy``) so ``/object_info`` reflects
    the running device set."""
    from localm.vram import media_split_notice
    cfg = config or {}
    # Gate 1 - the experimental toggle (default off, maintainer decision 2026-07-16).
    # Off -> byte-identical to today: single-card floor, legacy notice.
    if not cfg.get("comfy_gpu_placement"):
        return None, media_split_notice(cfg)
    # Gate 2 - only when a 2+ card split is actually applied (maintainer decision). A
    # cheap pre-check before the /object_info probe that ties placement to a deliberate
    # multi-GPU setup. Uses applied_split_device_count (the loader-truth count that
    # mirrors apply_gpu_split's own gate, added in #704), NOT split_device_count: the
    # latter filters against a list_gpus() that is structurally blind to Vulkan-only
    # devices, so on the vulkan build a live 2-way split collapses to <2 and would
    # wrongly decline placement (GPU-SPLIT-VKINDEX). applied_split_device_count is
    # Vulkan-sound. The probe in Gate 3 is the authoritative "how many cards does
    # ComfyUI see" check either way.
    from localm.discover import applied_split_device_count, plan_media_placement
    if applied_split_device_count(cfg) < 2:
        return None, media_split_notice(cfg)
    # Gate 3 - the running ComfyUI must actually offer the Select*Device nodes AND
    # enumerate 2+ GPUs (probe /object_info). An older ComfyUI (including localm's own
    # managed pin) or a single visible card declines with a stated reason, never a
    # graph ComfyUI would reject.
    cap = probe_placement_capability(api_url)
    plan = plan_media_placement(cfg, gpu_options=cap.gpu_options) if cap.available else None
    return plan, media_split_notice(cfg, placement=plan, capability=cap)


# ---------------------------------------------------------------------------
#  Pre-submit model validation (/object_info) - fail BEFORE the LLM unload
# ---------------------------------------------------------------------------
#
#  Before unloading the chat model (an expensive VRAM handoff) we ask ComfyUI which
#  model files each loader can actually see (GET /object_info) and confirm the
#  workflow's filenames exist. A missing file is named EXACTLY, and where the user
#  has the same model in a different precision/quant we auto-substitute the single
#  unambiguous variant. This turns a late "rejected (HTTP 400) after the unload"
#  into an early, specific error pointing at the Workflow panel (I3 / MEDIA-1).
#
#  Best-effort by design: when /object_info cannot be read it does NOT block - the
#  submit-time validation still applies. We never escalate a best-effort early
#  check into a hard failure that breaks an otherwise-working setup (AGENTS rule 5).

# Extensions that mark a combo's options as model files (so a value not in the list
# is a missing-model error we can name/fix), as opposed to an enum like sampler_name.
_MODEL_FILE_EXTS = (".safetensors", ".ckpt", ".gguf", ".pt", ".pth", ".bin",
                    ".sft", ".onnx", ".pte")


def comfy_object_info(api_url: str, timeout: float = 10.0) -> Optional[dict]:
    """ComfyUI's full ``/object_info`` map ``{class_type: spec}``, or None when it
    cannot be fetched or parsed (so the caller treats preflight as best-effort)."""
    try:
        with _comfy_urlopen(f"{api_url}/object_info", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _combo_options(spec: dict, input_name: str) -> Optional[list]:
    """The list of literal choices for a combo input of an /object_info node spec,
    or None when the input is not a combo.

    ComfyUI serves TWO combo shapes and both are live, so both must be parsed:

    - **v1 nodes** (the classic ``INPUT_TYPES`` dict, e.g. CheckpointLoaderSimple):
      ``[choices_list, meta?]`` - the choices ARE the first element. A non-combo v1
      input's first element is a type-name string ("INT", "MODEL", ...), not a list.
    - **v3 nodes** (the ``comfy_api`` schema, e.g. the Select*Device multigpu nodes):
      ``["COMBO", {"options": choices_list, ...}]`` - the first element is the io_type
      string "COMBO" and the choices live under ``meta["options"]``. Verified against
      the real serialization: server.py serves ``INPUT_TYPES()`` -> ``get_v1_info`` ->
      ``add_to_dict_v1`` emits ``(input.get_io_type(), input.as_dict())`` and
      ``Combo.get_io_type()`` is "COMBO" with the choices inside ``as_dict()``
      (comfy_api/latest/_io.py, ComfyUI git 867404b). Missing this shape is why a
      naive reader silently finds zero options on a v3 node and declines placement.

    A non-combo input (v1 or v3) returns None either way: its meta dict has no
    ``options`` list."""
    if not isinstance(spec, dict):
        return None
    io = spec.get("input")
    if not isinstance(io, dict):
        return None
    for section in ("required", "optional"):
        sec = io.get(section)
        if isinstance(sec, dict) and input_name in sec:
            entry = sec[input_name]
            if not (isinstance(entry, list) and entry):
                continue
            # v1: the first element is the choices list.
            if isinstance(entry[0], list):
                return [o for o in entry[0] if isinstance(o, str)]
            # v3: first element is the io_type string, choices under meta["options"].
            if (isinstance(entry[0], str) and len(entry) > 1
                    and isinstance(entry[1], dict)):
                opts = entry[1].get("options")
                if isinstance(opts, list):
                    return [o for o in opts if isinstance(o, str)]
    return None


@dataclass(frozen=True)
class PlacementCapability:
    """Whether the RUNNING ComfyUI can do per-component GPU placement, and onto which
    positions. ``gpu_options`` are the live ``gpu:N`` strings the ``SelectModelDevice``
    device combo offers (ComfyUI's OWN index space); ``reason`` is set (and human-
    readable) whenever ``available`` is False, so a decline is never a silent mystery."""
    available: bool
    gpu_options: list
    reason: str = ""


# The three core placement nodes. All three must be present for a complete plan; the
# device combo is read from the model node (all three carry the same options set).
_PLACEMENT_NODES = ("SelectModelDevice", "SelectCLIPDevice", "SelectVAEDevice")


def probe_placement_capability(api_url: str, *,
                               timeout: float = 10.0) -> PlacementCapability:
    """Ask the LIVE ComfyUI (``/object_info``) whether it offers per-component placement
    and how many GPUs it enumerates. This is what makes placement SAFE by construction:

    - It is the guard for a ComfyUI that PREDATES the multigpu nodes (first added
      upstream 2026-05-25, tag v0.23.0). There the probe returns unavailable and
      placement declines cleanly, rather than injecting a node ComfyUI would reject.
      That is now only ever a ComfyUI of the user's OWN: localm's managed pin is
      v0.31.1 and DOES offer the nodes (verified live against ``/object_info``; see
      ``COMFYUI_PLACEMENT_MIN_VERSION`` next to the pin itself).
    - It reads ComfyUI's OWN device enumeration (the ``gpu:N`` combo), which is
      authoritative about how many cards ComfyUI actually sees. (The caller's Gate 2
      uses ``applied_split_device_count`` as a cheap Vulkan-sound pre-filter; this probe
      is the authoritative device count either way.)

    Unavailable (with a named reason) when: ``/object_info`` cannot be read; any of the
    three ``Select*Device`` nodes is missing (old ComfyUI); or the device combo offers
    fewer than two ``gpu:N`` options (ComfyUI sees one card - nothing to place onto).
    Best-effort: on any doubt it returns unavailable, and the caller falls back to the
    single-card floor."""
    info = comfy_object_info(api_url, timeout=timeout)
    if not isinstance(info, dict):
        return PlacementCapability(False, [], "could not read ComfyUI /object_info")
    missing = [n for n in _PLACEMENT_NODES if n not in info]
    if missing:
        return PlacementCapability(
            False, [],
            "this ComfyUI does not offer per-component GPU placement (needs a ComfyUI "
            "with the multigpu nodes, upstream 2026-05-25 or newer)")
    options = _combo_options(info["SelectModelDevice"], "device") or []
    gpu_options = [o for o in options if isinstance(o, str) and o.startswith("gpu:")]
    if len(gpu_options) < 2:
        return PlacementCapability(
            False, gpu_options,
            "ComfyUI sees fewer than two GPUs, so there is no second card to place onto")
    return PlacementCapability(True, gpu_options)


def _looks_like_model_files(options: list, current: Optional[str] = None) -> bool:
    """True when a combo's options look like model files (a mismatch is then a
    missing-model error we can name), not an enum like sampler_name / scheduler.

    With 2+ live options this is decided from *options* ALONE, exactly as
    before *current* existed as a parameter: a real enum (sampler_name,
    scheduler, ...) always has several live choices, and none of them can
    ever look like a filename, so blending an unrelated *current* value into
    that vote could only ever wrongly flip a genuine enum - it never helps.

    With 0 or 1 live options there is not enough of a sample to judge alone:
    that is exactly what "ComfyUI has NOTHING of this file type installed"
    (an empty list) or "this loader's only live choice is a non-file
    sentinel" (e.g. a VAE loader offering just its built-in pixel-space
    passthrough when no external VAE is installed) looks like. There *current*
    - the node's value already in the workflow JSON, a concrete filename
    regardless of whether ComfyUI has it installed - is the fallback signal,
    so the slot still surfaces as "install this" instead of silently
    vanishing from the picker (and from preflight_models(), which walks the
    same slots). A single-option enum whose lone value is itself a stray
    extension-like string is the residual false-positive this cannot rule
    out, but that requires a hand-crafted/corrupted workflow - implausible
    from ComfyUI's own UI - and this validation is best-effort by design."""
    opts = options or []
    if len(opts) > 1:
        hits = sum(1 for o in opts if o.lower().endswith(_MODEL_FILE_EXTS))
        return hits >= max(1, len(opts) // 2)
    if opts and opts[0].lower().endswith(_MODEL_FILE_EXTS):
        return True
    return bool(current) and current.lower().endswith(_MODEL_FILE_EXTS)


def model_type_for_node(class_type: str) -> str:
    """The localm ``MODEL_TYPES`` value a ComfyUI loader node's model-file input
    holds, or ``"unknown"`` when the node name carries no usable signal.

    ONE derivation, shared by every caller: the ComfyUI-folder scanner
    (``model_manager/scan.py``, which reconciles its folder walk against
    ``/object_info``) and the media plugins' model-role wiring both key off this.
    A second copy of the heuristic elsewhere would drift, and the two would then
    disagree about the same file.

    Node-NAME based on purpose, not the input name: ComfyUI's ecosystem renames
    inputs freely across custom nodes (``unet_name`` / ``model_name`` /
    ``ckpt_name``) while the class name keeps the loader's role legible
    (``UNETLoader``, ``UnetLoaderGGUFAdvanced``, ``DualCLIPLoader``,
    ``VAELoader``, ``LoraLoader``). Order matters: a checkpoint loader is checked
    LAST of the positive cases so a name carrying both signals resolves to the
    more specific one.

    ``checkpoint`` maps to ``diffusion-unet`` deliberately, matching
    ``scan.SUBFOLDER_MAPPING["checkpoints"]``: an all-in-one checkpoint bundles
    UNet + text encoder + VAE, and the registry already files it under the UNet
    type from its folder. Without this case the scanner's ``/object_info`` pass
    stayed silent on checkpoints (it only ever overwrites with a KNOWN type), and
    the music plugin - whose shipped ACE workflow's only model slot is a
    ``CheckpointLoaderSimple`` - could never match its own declared
    ``music-unet`` role.

    A node this cannot classify returns ``"unknown"`` rather than a guess: that
    is a real answer ("we could not tell"), and callers must not collapse it into
    "there is nothing here"."""
    name = (class_type or "").lower()
    if "unet" in name or "diffusion" in name:
        return "diffusion-unet"
    if "clip" in name or "textencode" in name:
        return "text-encoder"
    if "vae" in name:
        return "vae"
    if "lora" in name:
        return "lora"
    if "checkpoint" in name or "ckpt" in name:
        return "diffusion-unet"
    return "unknown"


def _normalize_model_base(name: str) -> str:
    """A precision/quant-insensitive key for a model filename, so e.g.
    ``wan_5B_fp16.safetensors`` and ``wan_5B_fp8_scaled.safetensors`` share a base
    and one can stand in for the other. Drops the extension and ONLY unambiguous
    precision / quant markers (fp16/fp8/bf16/e4m3fn/scaled/qN...), keeping everything
    else.

    Crucially it does NOT drop bare digits or lone single letters: those are version
    / variant discriminators (``wan2.1`` vs ``wan2.2``, ``model_s`` vs ``model_m``,
    ``vae_1`` vs ``vae_2``) and collapsing them would merge genuinely different models
    into one base and trigger a WRONG auto-substitution - the opposite of "a single
    unambiguous precision variant"."""
    import re as _re
    n = name.lower().rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    for ext in _MODEL_FILE_EXTS:
        if n.endswith(ext):
            n = n[: -len(ext)]
            break
    drop = _re.compile(
        r"^(fp8|fp16|fp32|f8|f16|f32|bf16|e\dm\d[a-z0-9]*|scaled|default|q\d+)$")
    kept = [t for t in _re.split(r"[^a-z0-9]+", n) if t and not drop.match(t)]
    return "".join(kept)


def _pick_variant(missing_name: str, options: list) -> Optional[str]:
    """The single unambiguous precision/quant variant of *missing_name* among
    *options*, or None when there are zero or more than one candidates (we never
    guess between several)."""
    base = _normalize_model_base(missing_name)
    if not base:
        return None
    cands = [o for o in options
             if o != missing_name and _normalize_model_base(o) == base]
    return cands[0] if len(cands) == 1 else None


@dataclass(frozen=True)
class MissingModelSlot:
    """One workflow input whose model file ComfyUI's /object_info reports as not
    installed, and no unambiguous precision/quant variant was found to sub in."""
    class_type: str
    input_name: str
    filename: str
    available_options: list


def _format_missing(missing: list) -> str:
    lines = ["ComfyUI is missing model files this workflow needs:"]
    for cls, field, name, options in missing:
        shown = ", ".join(options[:8]) + (", ..." if len(options) > 8 else "")
        avail = f" Available {field}: {shown}." if options else ""
        lines.append(
            f"  - '{name}' (the {field} for the {cls} node) is not installed.{avail}")
    # NEW-COMFY-PREFLIGHT-MESSAGE-WRONG-AFTER-GUI-SWAP: this used to assert "The
    # chat model was NOT unloaded", which is only true when THIS function's own
    # caller (generate_image()/generate_music()/generate_video()) is the one
    # that would have done the unload. The GUI plugins (image/music/video
    # plug.py) unload the chat model THEMSELVES, earlier, before ever calling
    # in here - so that claim was false exactly there, and the contradiction
    # showed up verbatim in a real transcript ("Chat model unloaded - freed 9.4
    # GB" immediately followed by "The chat model was NOT unloaded"). Whether an
    # unload already happened is not something this function can know from
    # *missing* alone, so it no longer claims either way - the actual VRAM
    # state is reported correctly by the caller's own reload-on-every-exit-path
    # step regardless of what this message says.
    lines.append(
        "Install the file(s) into ComfyUI's models folder, or pick a workflow whose "
        "models you have on the Workflow panel (Settings -> Media), then run again.")
    return "\n".join(lines)


def workflow_model_slots(workflow: dict, api_url: str) -> Optional[list]:
    """Every model-file combo slot in *workflow*, resolved against ComfyUI's live
    ``/object_info``: ``[{"node_id", "class_type", "input_name", "current",
    "options"}, ...]``. This is the node/input walk ``preflight_models()`` uses to
    validate a workflow's model choices, exposed here for a model-picker UI too -
    one shared walk, not two independently-maintained ones.

    None when ``/object_info`` cannot be fetched (ComfyUI unreachable) - distinct
    from ``[]`` (reachable, but this workflow genuinely has no model-file inputs),
    so a caller can tell "cannot determine" from "determined: none"."""
    info = comfy_object_info(api_url)
    if not info:
        return None
    slots = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        spec = info.get(node.get("class_type"))
        inputs = node.get("inputs")
        if not isinstance(spec, dict) or not isinstance(inputs, dict):
            continue
        for input_name, value in inputs.items():
            if not isinstance(value, str):
                continue
            options = _combo_options(spec, input_name)
            if options is None or not _looks_like_model_files(options, current=value):
                continue
            slots.append({
                "node_id": str(node_id),
                "class_type": node.get("class_type"),
                "input_name": input_name,
                "current": value,
                "options": options,
            })
    return slots


def apply_model_overrides(workflow: dict, overrides: dict) -> int:
    """Apply per-node model-slot overrides to *workflow* in place.

    *overrides* is ``{node_id: {input_name: value}}`` (the shape a client builds
    from ``workflow_model_slots()``'s ``node_id``/``input_name``). Only writes a
    field that ALREADY exists as a plain string input on that node - never creates
    a new key, never touches a link/number input - so a malformed or stale
    override can at worst no-op, never corrupt the graph. Returns how many fields
    were actually changed."""
    changed = 0
    if not isinstance(overrides, dict):
        return 0
    for node_id, fields in overrides.items():
        node = workflow.get(str(node_id))
        if not isinstance(node, dict) or not isinstance(fields, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for input_name, value in fields.items():
            if (input_name in inputs and isinstance(inputs[input_name], str)
                    and isinstance(value, str) and inputs[input_name] != value):
                inputs[input_name] = value
                changed += 1
    return changed


def preflight_models(workflow: dict, api_url: str, *, on_progress=None) -> tuple[bool, str]:
    """Validate every loader's model file against ComfyUI ``/object_info`` BEFORE the
    caller unloads the chat model.

    Mutates *workflow* in place to substitute the single unambiguous precision/quant
    variant for a missing file. Returns ``(ok, message)``: ``ok=False`` with a
    specific, Workflow-panel-pointing error when a required model is missing and no
    one variant fits; ``ok=True`` (empty message) otherwise. Best-effort: returns
    ``(True, "")`` when /object_info is unavailable (defer to submit-time validation)."""
    slots = workflow_model_slots(workflow, api_url)
    if slots is None:
        return True, ""        # cannot validate -> defer to submit-time validation
    missing: list = []
    subs: list = []
    for slot in slots:
        value, options = slot["current"], slot["options"]
        if value in options:
            continue            # the file is present - good
        variant = _pick_variant(value, options)
        if variant is not None:
            workflow[slot["node_id"]]["inputs"][slot["input_name"]] = variant
            subs.append((slot["class_type"], slot["input_name"], value, variant))
        else:
            missing.append((slot["class_type"], slot["input_name"], value, options))
    if on_progress:
        for cls, field, old, new in subs:
            try:
                on_progress(
                    f"Model '{old}' is not installed; substituting '{new}' "
                    f"(same model, different precision) for the {cls} node.")
            except Exception:
                pass
    if missing:
        return False, _format_missing(missing)
    return True, ""


def describe_missing_models(workflow: dict, api_url: str) -> list:
    """Read-only variant of the ``preflight_models`` check: reports missing model
    slots as ``MissingModelSlot`` entries WITHOUT applying substitutions or
    otherwise mutating *workflow*. Used by a pre-check (e.g. before a user
    clicks Generate) that must not have side effects on the caller's workflow
    dict. Built on the same ``workflow_model_slots()`` walk ``preflight_models``
    uses - one shared walk, not a third independently-maintained one. Returns
    ``[]`` when /object_info is unavailable (same best-effort behavior as
    ``preflight_models``) or nothing is missing."""
    slots = workflow_model_slots(workflow, api_url)
    if slots is None:
        return []
    return [MissingModelSlot(slot["class_type"], slot["input_name"],
                             slot["current"], slot["options"])
            for slot in slots if not slot_is_satisfied(slot)]


def slot_is_satisfied(slot: dict) -> bool:
    """True when ComfyUI can actually serve the model file a *slot* asks for:
    either the exact filename is among the live options, or a single unambiguous
    precision/quant variant of it is (which ``preflight_models`` substitutes in).

    Split out of ``describe_missing_models`` so the model-picker/role surfaces
    report "installed" by the SAME rule preflight uses to decide "missing" -
    two rules would eventually disagree, and a picker calling a slot fine while
    generation refuses it is the worst version of that.

    A non-string ``current`` is unsatisfied rather than an exception. It cannot
    arise from ``workflow_model_slots`` (which only emits string-valued inputs),
    so this changes no reachable behaviour; it only keeps a hand-built slot dict
    from raising deep inside the filename normaliser."""
    value = slot.get("current")
    options = slot.get("options") or []
    if not isinstance(value, str):
        return False
    if value in options:
        return True             # the file is present - good
    # an unambiguous substitute exists - not "missing"
    return _pick_variant(value, options) is not None


# ---------------------------------------------------------------------------
#  VRAM management
# ---------------------------------------------------------------------------

def _localm_unload(localm_url: Optional[str] = None,
                    instance_token: Optional[str] = None) -> Optional[dict]:
    """
    Ask a localm server to release its model from GPU memory.

    Reads LOCALM_URL from the environment if *localm_url* is not given, and
    authenticates via ``auth.resolve_bearer_headers`` (the owner key - env,
    else the persisted ``auth.key`` - or, in open/keyless mode, *instance_token*).
    The ``/v1/models/unload`` endpoint requires the models-write scope, so an
    UNAUTHENTICATED POST is rejected with 401 and the chat model stays resident
    in VRAM - the media model then loads on top of it, exceeds total VRAM and
    hangs the GPU driver (the AMD TDR the user hit). For the same reason the
    built-in TLS cert of a loopback ``https`` self-call must be trusted, exactly
    as the media-job model reload does (``localm.tls.requests_verify``); plain
    ``urllib`` would reject the self-signed cert and silently skip the unload.

    *instance_token*: this is a FALLBACK path, reached only when the primary,
    already-authenticated ``vram.unload_chat_for_media`` call (which threads
    the same instance token) reported failure. A genuinely open (keyless)
    server still needs a credential here for the same reason it needs one on
    the primary path - see ``selfclient.self_request``'s docstring.

    Silent no-op when the URL is unset. Returns the server's JSON result
    (``status``, plus ``vram_freed`` / ``vram_before_bytes`` /
    ``vram_after_bytes`` when VRAM is measurable at all) on success, or None on
    any failure - never blocks generation if localm is not in the picture.

    Do not read the VRAM numbers without checking ``vram_reading_uncertain``:
    when set, the GPU probe behind them timed out or was busy, so they may be a
    stale cached reading and ``vram_freed`` may be null (unverifiable) rather
    than a real True/False - see http_server._add_vram_fields. A box with no VRAM
    telemetry at all simply omits the three fields, as before.
    """
    url = (localm_url or os.environ.get("LOCALM_URL", "")).rstrip("/")
    if not url:
        return None
    try:
        import requests as _rq

        from localm import tls as _tls
        from localm.auth import resolve_bearer_headers
        headers = resolve_bearer_headers(instance_token)
        # Unload waits for any in-flight generation to finish AND for VRAM to be
        # actually reclaimed before it returns (it must not free the context
        # mid-decode), so give it time.
        resp = _rq.post(f"{url}/models/unload", headers=headers, timeout=180,
                        verify=_tls.requests_verify(url))
        if not resp.ok:
            return None
        try:
            return resp.json()
        except Exception:
            return {"status": "unloaded"}
    except Exception:
        return None


class ComfyRedirectRefused(Exception):
    """A ComfyUI HTTP call tried to redirect and localm refused it. See
    _RefuseRedirect (CHK-COMFY-REDIRECT)."""


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    """LM-DA-045: sanitize_comfy_url (CHK-COMFY-APIURL, below) screens only the
    CONFIGURED comfy_api_url, at resolution time. A redirect target is chosen by
    the remote ComfyUI AFTER that check has already run, and urllib's default
    opener follows up to 10 such redirects with no validation at all - so a
    hostile or compromised ComfyUI (SECURITY.md: it "may be another machine,
    over plain http") could answer any request with a 3xx straight past the
    guard, e.g. to a cloud-metadata address. ComfyUI's HTTP API has no
    legitimate reason to redirect, so every hop is refused outright here rather
    than re-validated per hop (CHK-COMFY-REDIRECT)."""

    def redirect_request(self, req, fp, code, msg, hdrs, newurl):
        raise ComfyRedirectRefused(
            f"ComfyUI tried to redirect (HTTP {code}, to {newurl!r}); refusing "
            "- the redirect target is not policy-checked")


def _comfy_urlopen(req_or_url, *, timeout=None):
    """Every ComfyUI HTTP call in this module goes through this, never a bare
    urllib.request.urlopen - see _RefuseRedirect / CHK-COMFY-REDIRECT."""
    return verified_urlopen(req_or_url, timeout=timeout, handlers=(_RefuseRedirect,))


_COMFY_LOOPBACK_DEFAULT = "http://127.0.0.1:8188"


def _host_is_link_local(host: str) -> bool:
    """True if *host* is (or resolves to) a link-local / cloud-metadata address
    (169.254.0.0/16, fe80::/10). A ComfyUI never lives there; loopback / LAN /
    public do, and are allowed."""
    import ipaddress
    import socket
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_link_local
    except ValueError:
        pass
    try:
        for info in socket.getaddrinfo(host, None):
            try:
                if ipaddress.ip_address(info[4][0]).is_link_local:
                    return True
            except ValueError:
                continue
    except (socket.gaierror, OSError):
        return False
    return False


def sanitize_comfy_url(url: str) -> str:
    """Return *url* unless its host is link-local / cloud-metadata (or the guard
    itself cannot validate it), in which case warn and fall back to the loopback
    default. Defense-in-depth so an ADMIN-set comfy_api_url cannot turn the comfy
    control calls (free / interrupt / stop) into an SSRF probe of cloud metadata
    (CHK-COMFY-APIURL). Loopback + LAN + public are allowed - a real ComfyUI runs
    on any of those."""
    try:
        if _host_is_link_local(urllib.parse.urlparse(url).hostname or ""):
            from localm.debuglog import logger
            logger.warning(
                "comfy_api_url %r targets a link-local/metadata address; ignoring "
                "it and using the loopback default (CHK-COMFY-APIURL)", url)
            return _COMFY_LOOPBACK_DEFAULT
    except Exception as e:
        # FAIL CLOSED: if the guard itself cannot parse or verify the URL (e.g.
        # urlparse raising "Invalid IPv6 URL"), refusing is the only honest
        # outcome - returning the URL unchecked would silently approve exactly
        # what this guard exists to refuse (AGENTS.md rule 5). A URL the guard
        # cannot parse is not a working ComfyUI endpoint anyway.
        from localm.debuglog import logger
        logger.warning(
            "comfy_api_url %r could not be validated (%s); ignoring it and "
            "using the loopback default (CHK-COMFY-APIURL)", url, e)
        return _COMFY_LOOPBACK_DEFAULT
    return url


def default_api_url() -> str:
    """ComfyUI base URL: FLUX_API_URL env override, then a localm-MANAGED
    instance when one is installed and selected (coexistence, decision 6), then
    the ``comfy_api_url`` config key, else the ComfyUI default port. A link-local
    / cloud-metadata target is refused and falls back to loopback
    (CHK-COMFY-APIURL).

    The managed-ComfyUI hook is the ONLY managed touch in this module and is
    confined to TARGET RESOLUTION (never the launch/spawn path): it returns None
    - byte-identical to before - until a managed instance actually exists on
    disk, so nothing changes for a user who has not opted in."""
    env = os.environ.get("FLUX_API_URL")
    if env:
        return sanitize_comfy_url(env.rstrip("/"))
    try:
        from localm.media.managed_comfy import managed_comfy_api_url_if_active
        managed = managed_comfy_api_url_if_active()
        if managed:
            return managed
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("managed-ComfyUI URL lookup failed; falling back to the configured "
                     "/ default ComfyUI URL: %s", e)
    try:
        from localm.config import load_config
        cfg_url = load_config().get("comfy_api_url")
        if cfg_url:
            return sanitize_comfy_url(str(cfg_url).rstrip("/"))
    except Exception:
        pass
    return _COMFY_LOOPBACK_DEFAULT


def free_comfy_vram(api_url: Optional[str] = None) -> bool:
    """
    Ask ComfyUI to unload its models and free VRAM (POST /free).

    Returns True when ComfyUI accepted the request. Used after generation
    so the chat model can be reloaded immediately instead of spilling into
    system RAM next to a resident FLUX. Both an HTTP error (older ComfyUI
    builds without /free) and a network error return False, but each is now
    logged at debug level so --debug-discoverable can tell the two apart;
    callers then leave the LLM reload lazy.
    """
    url = (api_url or default_api_url()).rstrip("/")
    try:
        body = json.dumps({"unload_models": True, "free_memory": True}).encode()
        req = urllib.request.Request(
            f"{url}/free", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with _comfy_urlopen(req, timeout=30):
            return True
    except urllib.error.HTTPError as e:
        # Distinguish the documented older-build case (no /free endpoint) from a
        # real network failure below, so a missing endpoint is not mistaken for
        # ComfyUI being unreachable.
        from localm.debuglog import logger
        logger.debug("free_comfy_vram: ComfyUI /free returned HTTP %s (%s) at %s; "
                     "likely an older build without /free", e.code, e.reason, url)
        return False
    except (urllib.error.URLError, Exception) as e:
        # A real network/connection failure reaching ComfyUI, not a missing endpoint.
        from localm.debuglog import logger
        logger.debug("free_comfy_vram: could not reach ComfyUI /free at %s: %s", url, e)
        return False


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _comfy_alive(api_url: str, timeout: float = 3.0) -> bool:
    """Quick reachability probe so callers can fail fast with a clear error."""
    try:
        with _comfy_urlopen(f"{api_url}/system_stats", timeout=timeout):
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
#  Readiness cache - de-duplicate back-to-back reachability probes
#
#  Before this, ensure_comfy() pinged /system_stats on EVERY call, and every
#  media submission called it twice back-to-back (once at the route-handler
#  layer via ensure_available, once again at the top of the generator
#  function) - plain redundant network round-trips, on top of the GUI's own
#  5-second poll (settings.js, removed separately). So a confirmation is
#  remembered per api_url and reused instead of re-pinging.
#
#  It is remembered for a few SECONDS, not for the process lifetime. The
#  original version cached it forever, on the premise that "ComfyUI does not
#  appear or disappear on its own between requests" - which is false on this
#  stack: an OOM on a large render, a ZLUDA/ROCm torch crash, the user closing
#  its window, or VRAM eviction all kill it mid-session. A permanent entry then
#  reported a DEAD ComfyUI as running, so every later generation submitted to a
#  dead server and failed with an opaque ConnectionError, with both recovery
#  paths (auto-launch from comfy_workdir, and the actionable "not reachable"
#  message) silently disabled until the user hit Stop or reopened a media page -
#  and on the CLI / the coder's generate_image path, where nothing re-primes the
#  cache, for the whole process (REG-444).
#
#  The TTL keeps the entire win: the double-ping inside one submission is
#  milliseconds apart, far inside the window. A ping is a ~1ms loopback call
#  against a generation costing seconds to minutes, so a SHORT window is nearly
#  free and a longer one would only widen the stale gap for no gain.
#  mark_comfy_dead() (from stop_comfy(), and internally when ensure_comfy() can
#  no longer reach it) still clears the entry outright.
# ---------------------------------------------------------------------------

# How long a confirmed-reachable result stays trusted. Sized for the back-to-back
# double-ping this cache exists to collapse, NOT for "ComfyUI stays up".
_CONFIRM_TTL_SECONDS = 5.0

# api_url -> time.monotonic() when it was last confirmed reachable. monotonic, so
# a wall-clock change (NTP, DST) cannot make an entry look fresh or ancient.
_confirmed_alive: dict[str, float] = {}


def mark_comfy_alive(api_url: str) -> None:
    _confirmed_alive[api_url.rstrip("/")] = time.monotonic()


def mark_comfy_dead(api_url: str) -> None:
    _confirmed_alive.pop(api_url.rstrip("/"), None)


def is_comfy_confirmed(api_url: Optional[str] = None) -> bool:
    """True when *api_url* was confirmed reachable within the last
    ``_CONFIRM_TTL_SECONDS`` and nothing has invalidated it since.

    Deliberately time-bounded: a confirmation is evidence that ComfyUI was up a
    moment ago, never a promise that it still is (see the section comment above).
    """
    seen = _confirmed_alive.get((api_url or default_api_url()).rstrip("/"))
    return seen is not None and (time.monotonic() - seen) < _CONFIRM_TTL_SECONDS


def warm_comfy_status_async(api_url: Optional[str] = None) -> None:
    """Fire-and-forget readiness check for the "on app start" trigger: primes
    the cache without blocking plugin registration and without attempting to
    launch ComfyUI (an app boot should not decide FOR the user that this
    session needs ComfyUI running - only the on-demand triggers do that)."""
    import threading

    url = (api_url or default_api_url()).rstrip("/")

    def _check() -> None:
        if _comfy_alive(url):
            mark_comfy_alive(url)

    threading.Thread(target=_check, name="comfy-status-warm", daemon=True).start()


def history_execution_error(entry: dict) -> Optional[str]:
    """Return a human-readable ComfyUI execution error from a ``/history`` entry,
    or None when the job did not error.

    When a node crashes mid-render (a missing model, an ACE-Step/ComfyUI version
    mismatch, an OOM) ComfyUI still records the prompt in ``/history`` but with
    ``status.status_str == "error"`` and an ``("execution_error", {...})`` message
    carrying the node type and exception text. The poll loops otherwise only look
    for an output artifact and, finding none, blame a generic "no output" - so the
    real cause was hidden and the user was told to read the ComfyUI console (issue
    I2). Surfacing it here turns that into the actual reason."""
    status = entry.get("status") or {}
    for m in (status.get("messages") or []):
        if isinstance(m, (list, tuple)) and len(m) >= 2 and m[0] == "execution_error":
            info = m[1] if isinstance(m[1], dict) else {}
            node = info.get("node_type") or info.get("node_id")
            exc = (info.get("exception_message")
                   or info.get("exception_type") or "").strip()
            detail = " ".join(p for p in (exc, f"(node {node})" if node else "") if p)
            if detail:
                return detail
    if status.get("status_str") == "error":
        return "ComfyUI reported an execution error (no detail in /history)."
    return None


# ---------------------------------------------------------------------------
#  Reactive ComfyUI __func__ regression detection + offer (MEDIA-1)
# ---------------------------------------------------------------------------
#
#  A ComfyUI CORE regression in comfy_api/internal/__init__.py's
#  make_locked_method_func does `getattr(type_obj, func).__func__`, assuming a
#  node's FUNCTION is a bound method. A node whose FUNCTION resolves to a plain
#  function (the core audio VAEDecodeAudio, used by native ACE-Step) has no
#  `.__func__` -> AttributeError. Refs: Comfy-Org/ComfyUI #12116,
#  patientx/ComfyUI-Zluda #424. localm only submits a workflow + polls, so this is
#  purely upstream. We do NOT assume ComfyUI is broken: only when a REAL generation
#  hits this exact error do we offer a localm-side, in-memory shim (see comfy_shim/).

def is_known_comfy_func_regression(detail) -> bool:
    """True when a ComfyUI execution-error detail is the known upstream
    make_locked_method_func __func__ regression (Comfy-Org/ComfyUI #12116,
    patientx/ComfyUI-Zluda #424). ONLY this specific error should trigger the
    reactive offer; every other execution error is unrelated and behaves as before."""
    if not detail:
        return False
    text = str(detail)
    return ("has no attribute '__func__'" in text
            or "make_locked_method_func" in text)


def comfy_exec_error_message(payload, api_url: Optional[str] = None) -> str:
    """Build the message for a ComfyUI POLL_EXEC_ERROR. For an unrelated error this
    is the plain wording every generator used before. For the known __func__
    regression it is a richer, actionable message: what it is, that the fix is
    localm-side and in-memory (writes NOTHING into the user's ComfyUI install and
    self-expires once ComfyUI is fixed), and how to apply it."""
    detail = "" if payload is None else str(payload)
    if not is_known_comfy_func_regression(detail):
        return f"ComfyUI execution failed: {detail}"
    localm_started = spawned_pid(api_url) is not None
    apply_hint = (
        "localm can restart the ComfyUI it launched and retry with the fix."
        if localm_started else
        "Close your ComfyUI first (localm never touches a ComfyUI it did not "
        "start), then let localm launch a fixed one (needs comfy_workdir set).")
    return (
        "ComfyUI execution failed: " + detail + "\n"
        "This is a known ComfyUI core regression (Comfy-Org/ComfyUI #12116, "
        "patientx/ComfyUI-Zluda #424): a node whose function is a plain function "
        "(the native ACE-Step audio decode) trips an internal __func__ access.\n"
        "localm can apply an in-memory, localm-side compatibility shim - it patches "
        "ONLY a ComfyUI that localm itself starts (via a PYTHONPATH env var), writes "
        "NOTHING into your ComfyUI install, and self-expires once ComfyUI ships its "
        "own fix. " + apply_hint + "\n"
        "To stop being asked, turn it on: localm config comfy_func_shim on")


# ---------------------------------------------------------------------------
#  Managed-ComfyUI re-offer: the __func__ regression re-offers managed ComfyUI ONCE
# ---------------------------------------------------------------------------
#
#  The T1 shim (above) fixes THIS run in memory. As a durable answer, decision 1
#  + 8 of the managed-ComfyUI design let the __func__ crash ALSO offer localm's
#  own managed, patched ComfyUI (`localm comfy setup`) - the fix-for-good - but at
#  most ONCE (a persisted flag), never when a managed instance already exists (it
#  is moot: localm routes to the patched managed one, decision 6), and never for an
#  unrelated error. This is a pure decision + message; the CLI offer point
#  (cli/media.py) presents it alongside the shim offer.

_MANAGED_SETUP_OFFERED_KEY = "comfy_managed_setup_offered"


def should_offer_managed_comfy_setup(detail, cfg: Optional[dict] = None) -> bool:
    """True when the ONE-TIME durable-fix offer (set up localm's own managed,
    patched ComfyUI) should be surfaced for a media error.

    Gated on all three (design decisions 1 + 8): the error is the known upstream
    ``__func__`` regression, NO managed ComfyUI is installed yet (else the offer is
    moot - localm already routes to the patched managed instance, decision 6), and
    localm has not already made this offer (the persisted
    ``comfy_managed_setup_offered`` flag). Any one false -> no offer, so it never
    nags. A managed-install check that itself errors fails SAFE toward not offering
    (never surface something the user may already have)."""
    if not is_known_comfy_func_regression(detail):
        return False
    try:
        from localm.media.managed_comfy import is_managed_comfy_installed
        if is_managed_comfy_installed():
            return False
    except Exception:
        return False
    if cfg is None:
        try:
            from localm.config import load_config
            cfg = load_config()
        except Exception as e:
            from localm.debuglog import logger
            logger.debug("could not load config to check the managed-ComfyUI setup-offer "
                         "flag; the one-time offer may show again: %s", e)
            cfg = {}
    try:
        return not bool(cfg.get(_MANAGED_SETUP_OFFERED_KEY))
    except AttributeError:
        return False


def managed_comfy_setup_offer_message() -> str:
    """The one-time durable-fix offer text: set up localm's OWN managed, patched
    ComfyUI so the recurring upstream regression cannot recur. Points at the
    explicit opt-in command (`localm comfy setup`); the user's own ComfyUI is left
    untouched. Framed as the fix-for-good next to the shim's fix-this-run."""
    return (
        "This is a recurring upstream ComfyUI bug. For a durable fix - not just a "
        "patch for this run - localm can set up its OWN managed, patched ComfyUI so "
        "it cannot recur; your own ComfyUI is left untouched. To set it up:\n"
        "  localm comfy setup")


def mark_managed_comfy_setup_offered() -> None:
    """Persist ``comfy_managed_setup_offered=True`` so the durable-fix offer is made
    at most ONCE (it never nags). Written directly (like the shim's
    ``comfy_func_shim``), not through the settings form. Best-effort: a failure to
    persist must not break the media error path (worst case the offer reappears next
    time - annoying, not harmful), so it is logged at debug, never muted blind."""
    try:
        from localm.config import update_config
        update_config(lambda cfg: cfg.__setitem__(_MANAGED_SETUP_OFFERED_KEY, True))
    except Exception:
        from localm.debuglog import logger
        logger.debug("could not persist %s (the managed-ComfyUI offer may show "
                     "again next time)", _MANAGED_SETUP_OFFERED_KEY, exc_info=True)


def _derive_workdir_from_cmd(launch_cmd: str) -> Optional[str]:
    """The folder of the launcher script, so a .bat / .sh that references paths
    relative to its own location (the ComfyUI + ZLUDA convention, e.g. a copied
    ``launch-comfyui.bat`` next to ``python_embeded`` / ``venv``) runs from the
    right place even when ``comfy_workdir`` was not set. Best-effort: returns the
    parent of the first token that is an existing file, else None."""
    import shlex
    try:
        tokens = shlex.split(launch_cmd, posix=(os.name != "nt"))
    except ValueError:
        tokens = launch_cmd.split()
    for tok in tokens:
        tok = tok.strip().strip('"').strip("'")
        if not tok:
            continue
        try:
            p = Path(tok)
        except Exception:
            break
        if p.is_file():
            return str(p.parent)
        break  # only the first token is the program/script being launched
    return None


# Common ComfyUI launcher scripts, in priority order. A user's own
# launch-comfyui.* (the convention from the ComfyUI + ZLUDA community, and what
# this repo's own issue reporter hand-made) is preferred over the stock launcher,
# so localm uses the setup the user already has instead of imposing its own.
_LAUNCHER_NAMES_WIN = (
    "launch-comfyui.bat", "comfyui.bat", "run_nvidia_gpu.bat",
    "run_amd_gpu.bat", "run_cpu.bat", "run.bat",
)
_LAUNCHER_NAMES_POSIX = (
    "launch-comfyui.sh", "comfyui.sh", "run_nvidia_gpu.sh",
    "run_amd_gpu.sh", "run_cpu.sh", "run.sh",
)


def _venv_python(folder: Path) -> Optional[Path]:
    """A ComfyUI install's own venv interpreter, if present (venv/ or .venv/)."""
    if os.name == "nt":
        cands = [folder / "venv" / "Scripts" / "python.exe",
                 folder / ".venv" / "Scripts" / "python.exe"]
    else:
        cands = [folder / "venv" / "bin" / "python",
                 folder / ".venv" / "bin" / "python"]
    for c in cands:
        if c.is_file():
            return c
    return None


def discover_launch_cmd(folder: Path) -> Optional[str]:
    """Build a launch command for an existing ComfyUI install in *folder*.

    Prefers a launcher script the user already has (their own launch-comfyui.*,
    else the stock comfyui.* / run.*), falling back to running ``main.py`` with
    the install's own venv. Returns an absolute, quoted command string (run with
    *folder* as the working dir) or None when nothing recognizable is found.
    Absolute paths keep it cwd-independent and cross-platform (no PATH lookup,
    no bare-name resolution differences between cmd.exe and a POSIX shell)."""
    try:
        if not folder.is_dir():
            return None
    except OSError:
        return None
    names = _LAUNCHER_NAMES_WIN if os.name == "nt" else _LAUNCHER_NAMES_POSIX
    for name in names:
        p = folder / name
        if p.is_file():
            return f'"{p}"'
    py = _venv_python(folder)
    main = folder / "main.py"
    if py and main.is_file():
        return f'"{py}" "{main}"'
    return None


def _amd_rocm_launch_env() -> Optional[dict]:
    """Child env for the ComfyUI launch with ROCm's bin on PATH, or None to inherit.

    ZLUDA's ``cublas64_11.dll`` / ``cusparse64_11.dll`` shims load rocBLAS / rocSPARSE
    from ``%HIP_PATH%\\bin`` at ``import torch`` time. A localm process spawned from a
    context that has only HIP_PATH set (not ROCm\\bin on PATH) would otherwise make a
    ZLUDA ComfyUI die importing torch (WinError 126 on cublas64_11.dll). No-op off
    Windows, without HIP_PATH, when the bin is missing, or when it is already on PATH.
    """
    import sys as _sys
    if _sys.platform != "win32":
        return None
    hip = os.environ.get("HIP_PATH")
    if not hip:
        return None
    rocm_bin = os.path.join(hip, "bin")
    if not os.path.isdir(rocm_bin):
        return None
    cur = os.environ.get("PATH", "")
    if any(rocm_bin.lower() == p.strip().lower() for p in cur.split(os.pathsep) if p):
        return None
    env = dict(os.environ)
    env["PATH"] = rocm_bin + os.pathsep + cur
    return env


# ---------------------------------------------------------------------------
#  Reactive __func__ shim: apply ONLY to a ComfyUI localm spawns (MEDIA-1)
# ---------------------------------------------------------------------------
#
#  The fix is a localm-owned sitecustomize.py in comfy_shim/. localm never writes it
#  into the user's ComfyUI install: it only adds the shim DIRECTORY to the PYTHONPATH
#  of a ComfyUI process localm ITSELF spawns, so the interpreter auto-imports it and
#  patches the regression in memory. Default = off (no shim on PYTHONPATH). It turns
#  on only per the reactive offer: `enable_func_shim_once()` for this process, or the
#  persistent `comfy_func_shim` config for every future localm-spawned ComfyUI. If
#  localm did not spawn ComfyUI, the shim is simply absent.

# Process-local one-shot: "apply once" for this run without persisting a preference.
_func_shim_once = False


def comfy_shim_dir() -> Path:
    """The localm-owned directory holding the ComfyUI __func__ compatibility
    sitecustomize.py. Always inside the localm package, never in a ComfyUI folder."""
    return Path(__file__).resolve().parent / "comfy_shim"


def enable_func_shim_once() -> None:
    """Arrange for the NEXT ComfyUI that localm spawns to get the shim on its child
    PYTHONPATH, for this process only (does not persist a preference)."""
    global _func_shim_once
    _func_shim_once = True


def func_shim_enabled(cfg: Optional[dict] = None) -> bool:
    """Whether a ComfyUI localm spawns should get the shim on its child PYTHONPATH:
    the process one-shot, or the persistent ``comfy_func_shim`` config. Default off."""
    if _func_shim_once:
        return True
    if cfg is None:
        try:
            from localm.config import load_config
            cfg = load_config()
        except Exception as e:
            from localm.debuglog import logger
            logger.debug("could not load config to check the comfy_func_shim setting; "
                         "the shim stays disabled for this spawn: %s", e)
            cfg = {}
    try:
        return bool(cfg.get("comfy_func_shim"))
    except AttributeError:
        return False


def comfy_child_env(cfg: Optional[dict] = None) -> Optional[dict]:
    """The environment for a ComfyUI process localm SPAWNS. Starts from the AMD/ROCm
    launch env (or the inherited env), and layers on two things when they apply:

    - ONLY when the shim is enabled, PREPENDS the localm-owned shim dir to PYTHONPATH
      (preserving any pre-existing PYTHONPATH).
    - ONLY for a USER'S OWN ComfyUI, ORDERS ``CUDA_VISIBLE_DEVICES``/
      ``HIP_VISIBLE_DEVICES`` so the preferred card leads and EVERY OTHER CARD STAYS
      VISIBLE (REG-532).

    ORDER, NEVER MASK. This shipped wrong once (f094d3d0) as a bare
    ``CUDA_VISIBLE_DEVICES=<one id>``, which deletes the other cards from torch's view
    and silently disables ComfyUI core's per-component placement nodes
    (``SelectModelDevice``/``SelectCLIPDevice``/``SelectVAEDevice``,
    ``comfy_extras/nodes_multigpu.py``, registered ``nodes.py:2440``): a ``gpu:1`` that
    no longer exists is a no-op. So we emit the full ordered list ("1,0,2"), exactly
    what ComfyUI's own ``--default-device`` writes at ``main.py:69-76``, rather than
    the single id ``--cuda-device`` writes at ``main.py:78-81``.

    Why the ENV here and the ARGV for the managed instance: for a managed ComfyUI
    localm builds the command itself, so it passes ``--default-device`` (see
    ``managed_comfy_launch_cmd``). A user's own ComfyUI is started by THEIR launcher -
    often a .bat, possibly ZLUDA-wrapped - which localm must not rewrite, so the child
    env is the only lever. Both routes end in the same place: those two env vars.
    Setting both mirrors ComfyUI itself, which matters on the ZLUDA/ROCm path where
    CUDA is emulated over HIP. Deliberately NOT set for the managed instance, so its
    device has exactly ONE source of truth (the argv) rather than two that could
    disagree.

    Nothing is set when the user configured no split and no main GPU, or when no
    torch-visible device can be named honestly: ``visible_device_order()`` returns None
    and a plain box spawns exactly as it does today, rather than being pinned to an
    invented card (which would also hide a second GPU the user later adds). Never
    writes anything to disk."""
    base = _amd_rocm_launch_env()

    order = None
    try:
        from localm.media.managed_comfy import managed_comfy_active
        if not managed_comfy_active(cfg):
            from localm.discover import visible_device_order
            order = visible_device_order(cfg)
    except Exception as e:
        # Do NOT silently spawn with no device: that is the REG-532 defect (the swap
        # gate reads combined free VRAM while the model lands on one card). We cannot
        # fail the launch over it - that would break a working single-GPU setup - so
        # surface it and continue exactly as before (rule 5: a note, not silence, and
        # not an escalation).
        from localm.debuglog import logger
        logger.warning("could not resolve a GPU device order for the ComfyUI child env "
                       "(%s); launching without one. On a multi-GPU box the media VRAM "
                       "check may not match the card ComfyUI uses.", e)

    shim_on = func_shim_enabled(cfg)
    if not shim_on and not order:
        return base          # unchanged: exactly what the launch used before
    env = base if base is not None else dict(os.environ)
    if shim_on:
        shim = str(comfy_shim_dir())
        prev = env.get("PYTHONPATH", "")
        # Avoid a duplicate entry if we are re-spawning a ComfyUI that already had it.
        if prev.split(os.pathsep)[:1] == [shim]:
            env["PYTHONPATH"] = prev
        else:
            env["PYTHONPATH"] = shim + (os.pathsep + prev if prev else "")
    if order:
        # The FULL order, not order[0]: every card stays visible, the preferred one
        # merely leads. Emitting just the first id here is the masking bug this
        # replaced.
        visible = ",".join(str(i) for i in order)
        env["CUDA_VISIBLE_DEVICES"] = visible
        env["HIP_VISIBLE_DEVICES"] = visible
    return env


# NEW-STOPCOMFY: the ComfyUI processes localm itself launched, keyed by api_url,
# so we can terminate/restart the one WE spawned (and never a ComfyUI the user
# started themselves - we only have handles to our own). Guarded by a lock because
# a launch and a stop can race across request threads.
import threading as _threading

_spawned_procs: dict = {}
_spawned_lock = _threading.Lock()


def _remember_spawned(api_url: str, proc) -> None:
    with _spawned_lock:
        _spawned_procs[api_url] = proc


def _take_spawned(api_url: str):
    """Pop and return the proc localm launched for *api_url*, or None."""
    with _spawned_lock:
        return _spawned_procs.pop(api_url, None)


def spawned_pid(api_url: Optional[str] = None) -> Optional[int]:
    """The PID of the ComfyUI localm launched for *api_url* if it is still ours
    and running, else None (localm did not launch it, or it has exited)."""
    api_url = (api_url or default_api_url()).rstrip("/")
    with _spawned_lock:
        proc = _spawned_procs.get(api_url)
    if proc is None:
        return None
    try:
        return proc.pid if proc.poll() is None else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  NEW-COMFY-LAUNCH-NO-SERIALIZATION-LOCK: serialize ensure_comfy()'s whole
#  check-then-launch sequence per api_url.
#
#  Without this, two callers for the SAME api_url that both find ComfyUI down
#  (confirmed live: a generation submission racing the separate "Launch
#  ComfyUI" button, both reaching ensure_comfy() within milliseconds of each
#  other, neither with anything positive cached yet) each independently
#  decide to spawn a process and launch a SECOND ComfyUI with no
#  --port/--database-url override to keep them apart - they collide on the
#  same default port and database lock, and BOTH launches fail, even though
#  nothing else was ever running. The 5-second readiness cache
#  (_CONFIRM_TTL_SECONDS above) only de-duplicates a CONFIRMED-ALIVE result;
#  it has nothing to offer while both callers are independently still mid-
#  launch. This lock closes that: only one caller per api_url ever reaches
#  the spawn step at a time: a second caller blocks until the first's whole
#  attempt resolves (success or timeout), then re-checks aliveness under the
#  lock before ever considering a launch of its own.
#
#  KNOWN NARROW TRADE-OFF, not fixed here: a caller reached via
#  run_in_threadpool_bounded (imagine_comfy_launch, the generate submission
#  path) budgets ~comfy_launch_wait_seconds()+30s for its OWN attempt - not
#  for "queue behind someone else's attempt, THEN make my own". If caller A's
#  launch is slow and eventually fails (using its full wait_seconds), a
#  concurrent caller B can spend nearly all of ITS budget just waiting on
#  this lock, then get ThreadCallTimeout'd almost immediately after finally
#  acquiring it - a misleading "timed out" HTTP response even though (per
#  _threadpool_timeout.py's abandon_on_cancel=True contract) B's own launch
#  attempt keeps running to completion on the abandoned worker thread and may
#  still succeed moments later. Narrow (needs a genuinely slow-then-failing
#  first attempt, not just a fast failure - a fast failure releases this lock
#  quickly) and no worse than the pre-lock behaviour it replaces (both
#  callers colliding and BOTH failing); flagged rather than fixed since
#  closing it needs the outer timeout budget to account for lock wait time,
#  a change to every ensure_comfy() call site, not this lock alone.
# ---------------------------------------------------------------------------

_launch_locks: dict = {}
_launch_locks_guard = _threading.Lock()


def _launch_lock_for(api_url: str) -> "_threading.Lock":
    """The per-api_url lock serializing ensure_comfy()'s launch decision.
    Created on first use; never removed (one lock per distinct api_url this
    process ever launches for - unbounded only in the sense that the set of
    distinct api_urls a single process targets is itself small and stable)."""
    with _launch_locks_guard:
        lock = _launch_locks.get(api_url)
        if lock is None:
            lock = _threading.Lock()
            _launch_locks[api_url] = lock
        return lock


# ---------------------------------------------------------------------------
#  NEW-COMFY-SILENT-PARTIAL-APPLY: surface ComfyUI's own console warnings
# ---------------------------------------------------------------------------
#
#  A node whose weights only PARTLY match the model (a LoRA with incompatible
#  key naming, a checkpoint with missing UNet/CLIP/VAE keys, ...) is not an
#  error to ComfyUI: it logs a `logging.warning(...)` line and the run
#  completes normally, so history_execution_error() never sees it and the
#  caller is told the generation succeeded. This is only observable at all
#  when localm itself launched the ComfyUI process (spawned_pid() is not
#  None) - a remote or already-running instance has no process here to read
#  from, and that is a real, structural gap, not a bug in this mechanism.
#
#  _COMFY_SILENT_PARTIAL_APPLY_PATTERNS was built by grepping a real installed
#  ComfyUI checkout's comfy/*.py for logging.warning() calls with this exact
#  shape (component weights partly/fully unmatched, execution continues). Not
#  exhaustive - a fork, a newer ComfyUI release, or a custom node can log
#  differently. Extend this table as new cases are found; do not assume it is
#  complete.
_COMFY_SILENT_PARTIAL_APPLY_PATTERNS = (
    ("lora key not loaded",
     "a LoRA patch key did not match the model and was skipped"),
    ("WARNING SHAPE MISMATCH",
     "a LoRA patch's tensor shape did not match; that layer's weight was not merged"),
    ("Calculate Weight Failed",
     "applying one of the model's weight patches failed"),
    ("patch type not recognized",
     "an unrecognized weight-patch type was skipped"),
    ("clip missing:",
     "some CLIP/text-encoder weights were not found in the checkpoint"),
    ("Missing VAE keys",
     "some VAE weights were not found in the checkpoint"),
    ("No VAE weights detected, VAE not initalized",
     "no VAE weights were found in the checkpoint at all"),
    ("unet missing:",
     "some UNet weights were not found in the checkpoint"),
    ("unet unexpected:",
     "the checkpoint has UNet weights ComfyUI's model definition does not expect"),
    ("missing controlnet keys:",
     "some ControlNet weights were not found in the checkpoint"),
    ("missing clip vision:",
     "some CLIP-vision encoder weights were not found"),
    ("missing audio encoder:",
     "some audio-encoder weights were not found"),
    ("unexpected audio encoder:",
     "the checkpoint has audio-encoder weights ComfyUI's model definition does not expect"),
)


def comfy_launch_log_path(api_url: str) -> Path:
    """Where ensure_comfy redirects a self-launched ComfyUI's stdout+stderr
    (see the launch block below), scoped to *api_url*.

    MUST be per-instance, not one shared path: image/video/music each resolve
    their OWN per-plugin comfy.api_url/workdir/launch_cmd (localm/plugins/
    media_config.py), so two independently self-launched ComfyUI instances can
    be alive at once. ensure_comfy() truncates this file on every fresh
    launch (open(..., "w")) - a single shared path would let one instance's
    launch truncate/interleave-corrupt another still-running instance's
    console log out from under it, and let comfy_console_warnings_since()
    silently misattribute one instance's real warning onto the other's
    generation record. A plain deterministic path (not tied to any one
    process handle), so a caller can locate it without having launched
    ComfyUI itself in this call - same api_url always maps to the same path."""
    from localm.config import home_dir
    import hashlib
    api_url = api_url.rstrip("/")
    suffix = hashlib.sha256(api_url.encode("utf-8")).hexdigest()[:12]
    return home_dir() / f"comfy-launch-{suffix}.log"


def _launch_log_tail(log_path: Optional[Path], limit: int = 600) -> str:
    """The last *limit* chars of a self-launched ComfyUI's captured console
    output at *log_path*, so a launch-failure message can show WHY instead of
    just naming the file for the user to go open by hand. "" (nothing to add)
    when there is no log path, the file is empty, or it cannot be read -
    never raises: this only enriches an already-failing launch's message."""
    if log_path is None:
        return ""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if not text:
        return ""
    return text if len(text) <= limit else "..." + text[-limit:]


@dataclass(frozen=True)
class ComfyConsoleTail:
    """Opaque token from comfy_console_tail_start, passed to
    comfy_console_warnings_since. Carries the spawned PID alongside the byte
    offset so the later read can verify it is still looking at the SAME
    process's log, not a same-api_url process that died and was relaunched
    in between (which truncates the log via ensure_comfy's "w" open and
    re-registers a NEW pid under the same _spawned_procs[api_url] key -
    checking liveness alone cannot tell those apart)."""
    offset: int
    pid: int


def comfy_console_tail_start(api_url: str) -> Optional[ComfyConsoleTail]:
    """Marks 'now' in the self-launched ComfyUI's console log, so a caller can
    later read only what it printed during one generation (see
    comfy_console_warnings_since). None when localm did not launch this
    ComfyUI itself, or the log does not exist - there is nothing to tail, and
    no offset would be meaningful."""
    pid = spawned_pid(api_url)
    if pid is None:
        return None
    try:
        offset = comfy_launch_log_path(api_url).stat().st_size
    except OSError:
        return None
    return ComfyConsoleTail(offset=offset, pid=pid)


def comfy_console_warnings_since(api_url: str,
                                 tail: Optional[ComfyConsoleTail]) -> tuple:
    """Human-readable warnings ComfyUI printed to its own console between
    *tail* (from comfy_console_tail_start, called before the prompt was
    submitted) and now, matched against _COMFY_SILENT_PARTIAL_APPLY_PATTERNS.
    Each returned string names the condition and, when it recurred, how many
    times (e.g. "a LoRA patch key did not match the model and was skipped
    (x152)").

    Returns (checked, warnings). ``checked`` is True ONLY when localm actually
    performed a real read of ComfyUI's console covering *tail*'s window - the
    caller should derive any "did we actually check" signal (e.g. a sidecar's
    comfy_console_checked field) from THIS, not from whether tail_start
    returned non-None earlier, because the process can die or be replaced for
    the same api_url in between the two calls. ``warnings`` is [] both when
    checked is True and nothing matched (a genuine clean read) and whenever
    checked is False (nothing to report). See NEW-COMFY-SILENT-PARTIAL-APPLY
    in issues.txt: for a remote or pre-existing ComfyUI, checked is always
    False, because there is no local process to read from at all."""
    if tail is None:
        return False, []
    if spawned_pid(api_url) != tail.pid:
        # Not merely "not alive" - not the SAME process any more. Whether it
        # died outright or was replaced by a relaunch (which truncates the
        # log at tail.offset and hands the api_url key to a new pid), the
        # bytes at tail.offset in whatever now exists are not attributable to
        # the generation being asked about.
        return False, []
    try:
        with open(comfy_launch_log_path(api_url), "rb") as f:
            f.seek(tail.offset)
            new_bytes = f.read()
    except OSError:
        return False, []
    lines = new_bytes.decode("utf-8", errors="replace").splitlines()
    matches = []
    for substring, label in _COMFY_SILENT_PARTIAL_APPLY_PATTERNS:
        # Per-line, not a whole-buffer substring count: a pattern must occur
        # WITHIN one console line to count, so it cannot straddle two
        # unrelated lines a byte-offset read happened to butt together.
        # Residual, accepted risk (LOW - see NEW-COMFY-SILENT-PARTIAL-APPLY):
        # this does not prove the matched line came from logging.warning()
        # rather than, say, ComfyUI echoing a user-supplied filename that
        # happens to contain one of these substrings verbatim. The label text
        # emitted is always the fixed catalogue string, never the matched
        # content, so a false match is a spoofed diagnostic, not an
        # injection.
        count = sum(line.count(substring) for line in lines)
        if count:
            matches.append(f"{label} (x{count})" if count > 1 else label)
    return True, matches


def _kill_process_tree(proc) -> None:
    """Terminate *proc* AND its children. On Windows the launcher we spawn is a
    `cmd /S /c "<bat>"` whose real ComfyUI (python) is a CHILD, so terminating the
    cmd alone would orphan it - use taskkill /T. On POSIX the child was started in
    its own session (start_new_session), so signal the whole process group."""
    import os as _os
    import signal as _signal
    import subprocess as _sp
    import sys as _sys
    pid = getattr(proc, "pid", None)
    if not pid:
        return
    if _sys.platform == "win32":
        try:
            _sp.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True, timeout=15)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        return
    # POSIX: signal the process group, then wait, then hard-kill the group.
    try:
        pgid = _os.getpgid(pid)
    except Exception:
        pgid = None
    try:
        if pgid is not None:
            _os.killpg(pgid, _signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
        return
    except Exception:
        pass
    try:
        if pgid is not None:
            _os.killpg(pgid, _signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        pass


def stop_comfy(api_url: Optional[str] = None) -> tuple[bool, str]:
    """Stop ComfyUI (NEW-STOPCOMFY). Always aborts the in-flight render + clears
    the queue and frees VRAM first (graceful). Then, IF localm launched this
    ComfyUI, terminates the process tree we spawned; if localm did NOT launch it,
    the process is left alone (we only kill our own) and the caller is told so."""
    api_url = (api_url or default_api_url()).rstrip("/")
    mark_comfy_dead(api_url)   # about to stop it either way - the next check must be real
    interrupt_comfy(api_url)                       # abort render + clear queue
    try:
        free_comfy_vram(api_url)
    except Exception:
        pass
    proc = _take_spawned(api_url)
    if proc is None:
        if _comfy_alive(api_url, timeout=2.0):
            mark_comfy_alive(api_url)   # localm did not launch it and did not stop it
            return True, ("Aborted the in-flight render and cleared the queue. "
                          "localm did not launch this ComfyUI, so its process was "
                          "left running - stop it where you started it.")
        return True, "ComfyUI is not running."
    try:
        if proc.poll() is None:
            _kill_process_tree(proc)
    except Exception as e:
        return False, f"Could not stop the ComfyUI localm launched: {e}"
    return True, "Stopped the ComfyUI that localm launched."


def restart_comfy(api_url: Optional[str] = None, on_progress=None,
                  wait_seconds: Optional[int] = None,
                  launch_cmd: Optional[str] = None,
                  workdir: Optional[str] = None) -> tuple[bool, str]:
    """Stop the ComfyUI localm launched (if any), then launch a fresh one
    (NEW-STOPCOMFY). Only meaningful when localm has a launch command configured."""
    stop_comfy(api_url)
    return ensure_comfy(api_url=api_url, on_progress=on_progress,
                        wait_seconds=wait_seconds, launch_cmd=launch_cmd,
                        workdir=workdir)


def comfy_launch_wait_seconds(cfg: Optional[dict] = None) -> int:
    """The wait budget ``ensure_comfy`` gives an unpinned launch: the
    configured ``comfy_launch_timeout`` (a ZLUDA/ROCm cold start compiles GPU
    kernels and can take minutes), falling back to 300s, floored at 30s.

    Extracted out of ``ensure_comfy`` so a CALLER that needs to know this
    budget ahead of time - a route wrapping ``ensure_available``/
    ``restart_comfy`` in ``run_in_threadpool_bounded`` needs a timeout at
    least this large, or it would abort a launch that is still legitimately
    progressing - reads the exact same number ``ensure_comfy`` will actually
    wait, rather than a second, independently-maintained guess that could
    silently drift smaller than a user's own configured timeout. Pass an
    already-loaded *cfg* to avoid a second ``load_config()`` disk read when
    the caller already has one."""
    if cfg is None:
        from localm.config import load_config
        cfg = load_config()
    try:
        wait_seconds = int(cfg.get("comfy_launch_timeout") or 300)
    except (TypeError, ValueError):
        wait_seconds = 300
    return max(30, wait_seconds)


def ensure_comfy(api_url: Optional[str] = None, on_progress=None,
                 wait_seconds: Optional[int] = None,
                 launch_cmd: Optional[str] = None,
                 workdir: Optional[str] = None) -> tuple[bool, str]:
    """
    Make sure ComfyUI is reachable, launching it when configured.

    Used by every generator (image, music, video) from any caller - GUI,
    CLI, or the coder's generate_image tool. When ComfyUI is down and a launch
    command is configured, the command is started (optionally in *workdir*) and
    polled until the API answers.

    *launch_cmd* / *workdir* let a caller pass per-plugin config; when not given
    they fall back to the global ``comfy_launch_cmd`` / ``comfy_workdir`` config
    keys (kept for callers not yet migrated to per-plugin config).

    Returns (ok, message); the message explains what to configure when
    nothing could be launched.

    The whole decide-then-launch sequence is serialized per api_url via
    _launch_lock_for() (NEW-COMFY-LAUNCH-NO-SERIALIZATION-LOCK) - see that
    lock's own module-level comment for why. The cheap aliveness checks below
    run BEFORE acquiring it, so a caller that finds ComfyUI already up never
    waits on anything.
    """
    def _say(text: str) -> None:
        if on_progress:
            try:
                on_progress(text)
            except Exception:
                pass

    api_url = (api_url or default_api_url()).rstrip("/")
    if is_comfy_confirmed(api_url):
        return True, "ComfyUI is running."
    if _comfy_alive(api_url):
        mark_comfy_alive(api_url)
        return True, "ComfyUI is running."
    mark_comfy_dead(api_url)

    with _launch_lock_for(api_url):
        # Re-check under the lock: another caller may have finished an entire
        # launch (or be mid-launch and about to succeed) while we were
        # waiting for it above - the classic double-checked pattern. Without
        # this a caller that waited through someone else's whole launch would
        # still go on to attempt a redundant one of its own.
        if is_comfy_confirmed(api_url) or _comfy_alive(api_url):
            mark_comfy_alive(api_url)
            return True, "ComfyUI is running."
        return _launch_and_wait(
            api_url, launch_cmd, workdir, wait_seconds, _say)


def _launch_and_wait(api_url: str, launch_cmd: Optional[str],
                     workdir: Optional[str], wait_seconds: Optional[int],
                     _say) -> tuple[bool, str]:
    """The actual decide-config / spawn / poll-until-up body of ensure_comfy(),
    split out so the lock in ensure_comfy() wraps it without needing to
    re-indent it. Always called with _launch_lock_for(api_url) already held -
    not meant to be called directly."""
    import shlex
    import subprocess
    import sys as _sys
    import time as _t
    from localm.config import load_config

    cfg = load_config()

    # A localm-managed instance (decision 6) knows its own launch command - its
    # own venv + main.py, never the user's comfy_workdir/comfy_launch_cmd/
    # discovery (a raw managed checkout has no bundled launcher script for
    # discovery to find). Only applies when the CALLER did not already pass an
    # explicit workdir/launch_cmd of its own - same "caller override wins"
    # precedent default_api_url() already follows for the URL.
    managed_launch_cmd = None
    if workdir is None and not launch_cmd:
        try:
            from localm.media.managed_comfy import (
                managed_comfy_active, managed_comfy_launch_cmd, managed_comfy_workdir)
            if managed_comfy_active(cfg):
                # Atomic: only adopt the managed workdir if BOTH calls succeed.
                # Setting `workdir` from the first call and then having the
                # second raise would leave `workdir` pointed at the managed
                # folder with no matching launch_cmd, so the code below would
                # fall through to unrelated global-config/discovery logic
                # against that folder instead of a clean "not managed" outcome.
                managed_workdir = managed_comfy_workdir()
                managed_launch_cmd = managed_comfy_launch_cmd()
                workdir = managed_workdir
        except Exception:
            managed_launch_cmd = None

    # Resolve the ComfyUI folder (working dir): explicit arg / managed, then
    # config. It anchors both launcher discovery and the cwd a relative
    # launcher name runs from, so a bare "launch-comfyui.bat" works once the
    # folder is known.
    if workdir is None:
        workdir = cfg.get("comfy_workdir")

    # Resolve the launch command: explicit arg / managed, then config, then -
    # when the ComfyUI folder is known - auto-discover a launcher inside it
    # (the user's own launch-comfyui.bat, else the stock comfyui.bat / run.bat).
    # This is the "work with the install the user already has" path: pointing
    # localm at the ComfyUI folder is enough; naming a script is optional.
    if not launch_cmd:
        launch_cmd = managed_launch_cmd or cfg.get("comfy_launch_cmd")
    discovered = False
    if not launch_cmd and workdir:
        found = discover_launch_cmd(Path(workdir))
        if found:
            launch_cmd, discovered = found, True
    if not launch_cmd:
        return False, (
            f"ComfyUI is not reachable at {api_url}.\n"
            "Point localm at your ComfyUI install so it can start it for you:\n"
            '  localm config comfy_workdir "<path-to-your-ComfyUI-folder>"\n'
            "localm then runs the launcher it finds there (your own "
            "launch-comfyui.bat, or the stock comfyui.bat / run.bat).\n"
            "To name a specific launcher instead of auto-detecting:\n"
            '  localm config comfy_launch_cmd "<path>\\launch-comfyui.bat"\n'
            "Or start ComfyUI yourself first (default http://127.0.0.1:8188), or "
            "set the FLUX_API_URL environment variable if it runs elsewhere."
        )

    # Honour the configurable timeout when the caller did not pin one - see
    # comfy_launch_wait_seconds's own docstring for why this is a shared
    # helper rather than inline logic. The 30s floor applies unconditionally,
    # even to an explicitly-passed wait_seconds (unchanged from before this
    # was extracted - a caller-supplied value below 30 was never honoured).
    if wait_seconds is None:
        wait_seconds = comfy_launch_wait_seconds(cfg)
    wait_seconds = max(30, wait_seconds)

    # MEDIA-2: optionally start ComfyUI headless. Off by default (keep the
    # current behavior); when comfy_disable_auto_launch is set, append ComfyUI's
    # --disable-auto-launch so it does not pop open its own web page (localm has
    # its own GUI). Shared by image/music/video. The stock run_*.bat / comfyui.*
    # and a bare "python main.py" forward extra args to main.py; a launcher that
    # drops args simply ignores the flag (no error), so this stays non-breaking.
    if cfg.get("comfy_disable_auto_launch") and \
            "--disable-auto-launch" not in launch_cmd:
        launch_cmd = launch_cmd + " --disable-auto-launch"

    _say(f"ComfyUI not running - launching: {launch_cmd}")
    if discovered:
        _say(f"Found a ComfyUI launcher in {workdir}")
    # The command is the user's own config value (their launcher script).
    # On Windows pass `cmd /S /c "<line>"` as a single string: /S strips the
    # outer quotes and runs the line verbatim, so quoted executable paths
    # survive (a `["cmd", "/c", line]` list gets re-quoted by subprocess and
    # mangles them). POSIX uses shlex.
    if not workdir:
        # No configured ComfyUI folder: fall back to the launcher file's own
        # folder so a .bat/.sh that uses paths relative to itself (ComfyUI +
        # ZLUDA) still works when the user gave an absolute launcher path.
        workdir = _derive_workdir_from_cmd(launch_cmd)
        if workdir:
            _say(f"Running the launcher from {workdir}")
    workdir = workdir or None
    argv = shlex.split(launch_cmd, posix=(_sys.platform != "win32"))
    if _sys.platform == "win32":
        argv = [a.strip('"\'') for a in argv]
        if argv and (argv[0].lower().endswith(".bat") or argv[0].lower().endswith(".cmd")):
            argv = ["cmd", "/d", "/c"] + argv
    # Redirect the launcher's own stdout+stderr to a log file instead of
    # discarding them, so a ComfyUI that fails to start leaves its reason on
    # disk for the user (and for --debug-discoverable). Best-effort: fall back
    # to DEVNULL if the log cannot be opened, so launching still proceeds.
    # Opening in "w" mode TRUNCATES the file on every fresh spawn, which is
    # what lets comfy_console_warnings_since() trust that anything past a
    # given offset belongs to the currently-running process, not a stale run.
    # Scoped to api_url (comfy_launch_log_path) so a DIFFERENT self-launched
    # ComfyUI instance (image/video/music can each point at their own) never
    # shares - and truncates - this one's log.
    launch_log_path = comfy_launch_log_path(api_url)
    try:
        launch_out = open(launch_log_path, "w", encoding="utf-8", errors="replace")
    except OSError:
        launch_out = subprocess.DEVNULL
        launch_log_path = None
    # NEW-STOPCOMFY: start the launcher in its OWN process group/session so the
    # whole ComfyUI tree (the launcher + the python it spawns) can later be
    # terminated together (see stop_comfy / _kill_process_tree). Harmless to the
    # normal run; only changes signal grouping.
    _popen_kw: dict = {}
    if _sys.platform == "win32":
        _popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        _popen_kw["start_new_session"] = True
    try:
        proc = subprocess.Popen(argv, cwd=workdir,
                         env=comfy_child_env(cfg),
                         stdout=launch_out,
                         stderr=subprocess.STDOUT,
                         **_popen_kw)
        _remember_spawned(api_url, proc)   # retain the handle so Stop can reach it
        _t.sleep(0.5)
        if proc.poll() is not None and proc.returncode != 0:
            _take_spawned(api_url)         # it died; drop the dead handle
            detail = _launch_log_tail(launch_log_path)
            return False, (
                f"ComfyUI launcher exited immediately with code {proc.returncode}."
                + (f" Its captured output:\n{detail}" if detail else ""))
    except Exception as e:
        return False, f"Could not launch ComfyUI ({launch_cmd}): {e}"
    finally:
        # Close the PARENT's copy of the log handle on every path: the child
        # inherited its own dup'd descriptor, so leaving this open would leak a
        # file handle per launch (and on Windows hold a write lock the next
        # launch's re-open would contend with). DEVNULL is a sentinel, not a file.
        if hasattr(launch_out, "close"):
            launch_out.close()

    deadline = _t.monotonic() + wait_seconds
    last_said = 0.0
    while _t.monotonic() < deadline:
        if _comfy_alive(api_url):
            mark_comfy_alive(api_url)
            return True, "ComfyUI is up."
        elapsed = wait_seconds - (deadline - _t.monotonic())
        if elapsed - last_said >= 15:
            _say(f"Waiting for ComfyUI… ({int(elapsed)}s)")
            last_said = elapsed
        _t.sleep(2)
    # Point the user at the captured launcher log (when we managed to open one):
    # a ComfyUI that died on startup wrote its own error there, not to the window.
    tail = _launch_log_tail(launch_log_path)
    if launch_log_path and tail:
        log_hint = f" The launcher's own output ({launch_log_path}):\n{tail}"
    elif launch_log_path:
        log_hint = (f" The launcher's own output was captured to {launch_log_path} - "
                    "check it for the reason it failed to start.")
    else:
        log_hint = ""
    return False, (
        f"ComfyUI did not come up within {wait_seconds // 60} minutes - "
        "check the launcher window for errors. If it is just a slow first "
        "(ZLUDA / ROCm) start, raise the limit: "
        "localm config comfy_launch_timeout 600"
        f"{log_hint}"
    )


def comfy_http_error_detail(e: "urllib.error.HTTPError") -> str:
    """
    Human-readable detail from a ComfyUI /prompt error response.

    A 400 from /prompt means the workflow failed validation - not a
    connectivity problem. The response body is JSON naming the failing
    node and why (a model file missing from ComfyUI's models directory
    is the usual cause). Shared by image, music, and video generation.
    """
    try:
        body = json.loads(e.read().decode("utf-8", "replace"))
    except Exception:
        return f"HTTP {e.code}: {e.reason}"
    lines = []
    err = body.get("error") or {}
    if err.get("message"):
        msg = err["message"]
        if err.get("details"):
            msg += f" - {err['details']}"
        lines.append(msg)
    for node_id, info in (body.get("node_errors") or {}).items():
        cls = info.get("class_type") or f"node {node_id}"
        for ne in info.get("errors", []):
            msg = ne.get("message", "")
            if ne.get("details"):
                msg += f" ({ne['details']})"
            lines.append(f"{cls}: {msg}")
    return "\n".join(lines) or f"HTTP {e.code}: {e.reason}"


def _image_dimensions(path: Path) -> tuple[int, int]:
    """Return (width, height) from a PNG or JPEG without any external libs."""
    try:
        with open(path, "rb") as f:
            data = f.read(32)
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
        if data[:2] == b"\xff\xd8":
            # JPEG - scan for SOF0/SOF2 markers
            full = path.read_bytes()
            i = 2
            while i < len(full) - 8:
                if full[i] != 0xFF:
                    break
                marker = full[i + 1]
                length = int.from_bytes(full[i + 2:i + 4], "big")
                if marker in (0xC0, 0xC2):
                    h = int.from_bytes(full[i + 5:i + 7], "big")
                    w = int.from_bytes(full[i + 7:i + 9], "big")
                    return w, h
                i += 2 + length
    except Exception:
        pass
    return 1024, 1024


# Magic-byte signatures of the raster formats ComfyUI's LoadImage node reads.
# ((offset, bytes), ...) per format; WEBP and the RIFF family need a second
# window, hence the tuple-of-windows shape rather than a flat prefix list.
_IMAGE_SIGNATURES: tuple[tuple[tuple[int, bytes], ...], ...] = (
    ((0, b"\x89PNG\r\n\x1a\n"),),                  # PNG
    ((0, b"\xff\xd8\xff"),),                       # JPEG
    ((0, b"GIF87a"),),                             # GIF
    ((0, b"GIF89a"),),
    # BMP's signature really is only two bytes. That is weak in the abstract, but
    # not here: the threat is a caller naming SOMEONE ELSE'S file (auth.key, a
    # session store) to have it transmitted, and they control WHICH file, never
    # its first two bytes. A file that happens to start "BM" is not a file an
    # attacker can arrange to contain a secret.
    ((0, b"BM"),),                                 # BMP
    ((0, b"RIFF"), (8, b"WEBP")),                  # WebP
    ((0, b"II*\x00"),),                            # TIFF, little-endian
    ((0, b"MM\x00*"),),                            # TIFF, big-endian
)


def looks_like_image(head: bytes) -> bool:
    """True when *head* (the first bytes of a file) starts with the signature of
    a raster image format ComfyUI can load."""
    return any(all(head[off:off + len(sig)] == sig for off, sig in windows)
               for windows in _IMAGE_SIGNATURES)


def _upload_image(image_path: Path, api_url: str) -> str:
    """
    Upload a local image to ComfyUI via POST /upload/image.

    Returns the filename ComfyUI assigned (used in the LoadImage node).
    Raises on failure.

    Sniffs the magic bytes FIRST, before reading the body or opening the socket.
    This upload TRANSMITS the file, and sanitize_comfy_url deliberately permits
    api_url to be a LAN or public host over plaintext http, so "whatever bytes
    the caller named" must never leave the machine. media/paths.py confines
    WHERE an HTTP caller's input_image may live; this is the choke point that
    also covers the CLI and the coder tool, for which that path policy does not
    apply. Both gates are needed: neither subsumes the other.
    """
    with open(image_path, "rb") as f:
        head = f.read(16)
    if not looks_like_image(head):
        # Deliberately does NOT say "is not an image": this allowlist is narrower
        # than "image". localm itself accepts .heic/.heif elsewhere
        # (gui/web.py _SHARE_IMAGE_EXTS, i.e. an ordinary iPhone photo), and
        # those are refused here because the backend's loader cannot be assumed
        # to read them. Telling a user their photo "is not an image" would be
        # false and would send them debugging the wrong thing; name the
        # supported set and let them convert.
        raise ValueError(
            f"{image_path.name} is not in a format this upload supports "
            f"(PNG, JPEG, GIF, BMP, WebP or TIFF - detected from the file's "
            f"own header, not its extension); refusing to upload it to ComfyUI")
    boundary = "LocalcoderUploadBoundary"
    img_bytes = image_path.read_bytes()
    content_type = "image/jpeg" if image_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'
        f"Content-Type: {content_type}\r\n"
        f"\r\n"
    ).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{api_url}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with _comfy_urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode())
    name = result.get("name")
    if not name:
        raise RuntimeError(f"ComfyUI upload returned no filename: {result}")
    return name


# ---------------------------------------------------------------------------
#  Output containment
# ---------------------------------------------------------------------------
#
#  Default: localm LEAVES ComfyUI's own copies alone. A user may run ComfyUI for
#  its own gallery and legitimately want the generated files and the /history
#  entry, so deleting them is NOT the default. Containment is opt-in (the
#  comfy_delete_outputs config / a per-plugin delete_outputs), and privacy mode
#  forces it on (no trace anywhere). When enabled, after the artifact has been
#  fetched into localm's own location we clear ComfyUI's /history entry AND delete
#  its on-disk copy of the output plus any img2img source we uploaded into input/.
#  Deleting files needs ComfyUI's output/ dir; when it cannot be resolved we
#  return a loud warning rather than silently leaving a copy the user asked to
#  remove. Shared by image, music, and video generation.

def _comfy_output_root(comfy_output_dir: Optional[str] = None) -> Optional[Path]:
    """ComfyUI's output/ directory, or None when it cannot be resolved.

    Order: explicit arg, COMFY_OUTPUT_DIR env, the ``comfy_output_dir`` config
    key, then a derived ``<comfy_workdir>/output`` when that exists.

    ``comfy_output_dir`` (the arg, env var, and config key alike) is settable
    by a caller holding only the config:write scope - privileged, but NOT
    ADMIN (inference/routes/config.py's set_media_config ADMIN-gates
    launch_cmd/api_url/workdir but deliberately not this key). This is the
    READ-TIME choke point every caller of this function goes through, so the
    UNC/device guard belongs HERE rather than at each call site or only at
    the config-write boundary: confined_under() (used downstream by
    contain_comfy_artifacts) validates the RELATIVE path handed to it, never
    the base it is confined under, so a UNC-shaped base reaches its
    .resolve() call - the SMB dial - before any containment check can refuse
    it. A write-side check alone would also leave an ALREADY-PERSISTED config
    value from before this fix unguarded, which is why read time is
    authoritative here, not a backup for a config-write-side check."""
    cand = comfy_output_dir or os.environ.get("COMFY_OUTPUT_DIR")
    if not cand:
        try:
            from localm.config import load_config
            cfg = load_config()
            cand = cfg.get("comfy_output_dir")
            if not cand:
                wd = cfg.get("comfy_workdir")
                if wd:
                    derived = Path(wd) / "output"
                    if derived.is_dir():
                        return derived
        except Exception:
            return None
    if not cand:
        return None
    if is_unc_or_device_path(cand):
        # Fail safe, like the "cannot be resolved" case this function already
        # documents - but SURFACE it (AGENTS.md rule 5), since the caller's
        # own "set the ComfyUI output dir" warning would otherwise read as
        # nothing being configured, when something dangerous was.
        from localm.debuglog import logger
        logger.warning(
            "comfy_output_dir is a UNC or device path (%r) - refusing to use "
            "it as the ComfyUI output root; containment/cleanup will report "
            "it as unresolvable rather than dial it", cand)
        return None
    return Path(cand)


def clear_comfy_history(api_url: str, prompt_id: str) -> bool:
    """Remove this job from ComfyUI's /history (the Queue/History panel and
    gallery views read it). POST /history {"delete": [prompt_id]}. Best-effort;
    returns True when ComfyUI accepted the request."""
    if not prompt_id:
        return False
    try:
        body = json.dumps({"delete": [prompt_id]}).encode()
        req = urllib.request.Request(
            f"{api_url}/history", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with _comfy_urlopen(req, timeout=10):
            return True
    except Exception:
        return False


def contain_comfy_artifacts(
    api_url: str,
    prompt_id: str,
    info: dict,
    *,
    comfy_output_dir: Optional[str] = None,
    uploaded_input: Optional[str] = None,
    delete_outputs: bool = False,
) -> str:
    """Optionally remove ComfyUI's own copies of a generation.

    By DEFAULT (delete_outputs=False) this is a no-op: ComfyUI keeps its /history
    entry and its on-disk output, because a user may run ComfyUI for its own
    gallery and want them. When the user opts in (or privacy mode forces no-trace),
    it clears the history entry and deletes ComfyUI's duplicate output plus any
    img2img source we uploaded into input/. Returns a WARNING string when
    containment was requested but a copy could NOT be removed (so the user is told
    a copy remains), or "" otherwise."""
    if not delete_outputs:
        return ""   # keep ComfyUI's history + on-disk copy by default

    warnings: list = []
    if not clear_comfy_history(api_url, prompt_id):
        warnings.append(
            "ComfyUI's /history entry for this generation could not be cleared "
            "and remains visible in its Queue/History panel")
    root = _comfy_output_root(comfy_output_dir)
    # Remove the uploaded img2img source from ComfyUI's input/ dir (sibling of
    # output/). Surface a failure (do not silence): it is still a stray copy of
    # the user's input that they asked to contain.
    #
    # `uploaded_input` is ComfyUI's OWN reply (result["name"] from /upload/image),
    # so it is remote data driving an unlink(). Confine it: pathlib would let an
    # absolute component REPLACE the base outright and a "../.." walk out of it,
    # which on a LAN or public api_url (sanitize_comfy_url permits both, over
    # plaintext http) hands a hostile or compromised ComfyUI arbitrary file
    # deletion with localm's privileges.
    if uploaded_input and root is not None:
        try:
            inp = confined_under(root.parent / "input", uploaded_input)
        except ValueError:
            # AGENTS rule 5: NOT silently skipped. Containment was requested and
            # did not happen, so the user is told - a success we did not achieve
            # is never reported as one.
            warnings.append(
                f"ComfyUI returned an out-of-bounds input filename "
                f"({uploaded_input!r}); its copy in ComfyUI's input folder was "
                f"not removed")
        else:
            try:
                if inp.exists():
                    inp.unlink()
            except OSError as e:
                warnings.append(
                    f"a copy of your input image remains in ComfyUI's input folder ({e})")

    # Delete ComfyUI's on-disk copy of the output. type "temp" is auto-purged
    # by ComfyUI, so only a real "output" artifact needs removing.
    if (info.get("type") or "output") == "output":
        if root is None:
            warnings.append(
                "a copy of this file remains in ComfyUI's output folder; localm "
                "cleared the ComfyUI history entry but cannot delete the file "
                "until you set the ComfyUI output dir (Settings -> Media, or: "
                "localm config comfy_output_dir <path>)")
        else:
            # `subfolder` and `filename` are parsed straight out of ComfyUI's
            # /history JSON - remote data, same as uploaded_input above. Nesting
            # is legitimate here (that is what `subfolder` IS), so this needs
            # confined_under's nested form, not confined_name.
            rel = "/".join(
                p for p in (str(info.get("subfolder", "") or ""),
                            str(info.get("filename", "") or "")) if p)
            try:
                copy = confined_under(root, rel)
            except ValueError:
                warnings.append(
                    f"ComfyUI returned an out-of-bounds output filename "
                    f"({rel!r}); its copy in ComfyUI's output folder was not "
                    f"removed")
            else:
                try:
                    if copy.exists():
                        copy.unlink()
                except OSError as e:
                    warnings.append(f"could not delete ComfyUI's copy of the output ({e})")

    return ("WARNING: " + "; ".join(warnings) + ".") if warnings else ""


def interrupt_comfy(api_url: str) -> bool:
    """Abort ComfyUI's currently running prompt and clear its queue (POST
    ``/interrupt`` + POST ``/queue {"clear": true}``).

    Best-effort, used to honour a user's Stop so a long media gen actually stops
    instead of running to completion. Shared by image, music, and video. Returns
    True when the interrupt was accepted."""
    ok = False
    try:
        req = urllib.request.Request(f"{api_url}/interrupt", data=b"", method="POST")
        with _comfy_urlopen(req, timeout=10):
            ok = True
    except Exception:
        pass
    try:
        body = json.dumps({"clear": True}).encode()
        req = urllib.request.Request(
            f"{api_url}/queue", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with _comfy_urlopen(req, timeout=10):
            pass
    except Exception:
        pass
    return ok


def _with_warning(message: str, warning: str) -> str:
    """Append a containment warning to a success message when present."""
    return f"{message}\n{warning}" if warning else message


# ---------------------------------------------------------------------------
#  Shared transport: queue -> poll -> download
# ---------------------------------------------------------------------------
#
#  The three generators (image / video / music) share the exact same ComfyUI
#  transport: POST /prompt to queue, poll /history until the job finishes (with
#  the same cancel handling, the same execution-error check, the same
#  last-poll-error tracking on timeout), then GET /view to fetch the artifact.
#  Only the medium-specific bits differ (which output keys to read, the wording
#  of each error, how progress is shown), so those stay in each generator and the
#  shared loop below takes them as parameters. Behavior, retry cadence, timeouts,
#  and output selection are unchanged from the per-module originals.

# Sentinel "kind" values returned by comfy_submit_prompt so the caller can build
# its own medium-specific error text without this helper hardcoding any.
SUBMIT_OK = "ok"
SUBMIT_NO_ID = "no_prompt_id"
SUBMIT_HTTP_ERROR = "http_error"
SUBMIT_URL_ERROR = "url_error"
SUBMIT_ERROR = "error"


def comfy_submit_prompt(api_url: str, workflow: dict, *, timeout: float = 10.0):
    """Queue *workflow* in ComfyUI (POST /prompt) and return ``(kind, value)``.

    ``kind`` is one of the ``SUBMIT_*`` sentinels:
      - ``SUBMIT_OK``        -> value is the prompt_id (str)
      - ``SUBMIT_NO_ID``     -> value is None (accepted but no prompt_id returned)
      - ``SUBMIT_HTTP_ERROR``-> value is the ``urllib.error.HTTPError``
      - ``SUBMIT_URL_ERROR`` -> value is the ``urllib.error.URLError``
      - ``SUBMIT_ERROR``     -> value is the ``Exception``

    The caller maps each kind to its own (unchanged) message text. This mirrors
    the per-module try/except exactly: an HTTPError, a URLError, then a catch-all
    Exception, in that order."""
    try:
        req = urllib.request.Request(
            f"{api_url}/prompt",
            data=json.dumps({"prompt": workflow}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with _comfy_urlopen(req, timeout=timeout) as response:
            prompt_id = json.loads(response.read().decode("utf-8")).get("prompt_id")
        if not prompt_id:
            return SUBMIT_NO_ID, None
        return SUBMIT_OK, prompt_id
    except urllib.error.HTTPError as e:
        return SUBMIT_HTTP_ERROR, e
    except urllib.error.URLError as e:
        return SUBMIT_URL_ERROR, e
    except Exception as e:
        return SUBMIT_ERROR, e


# Poll outcomes (the "status" element of comfy_poll_until_done's return).
POLL_CANCELLED = "cancelled"
POLL_EXEC_ERROR = "exec_error"
POLL_FINISHED = "finished"
POLL_TIMEOUT = "timeout"


def comfy_poll_until_done(
    api_url: str,
    prompt_id: str,
    *,
    max_poll_seconds: float,
    cancel_check=None,
    on_tick=None,
    history_timeout: float = 5.0,
    sleep_seconds: float = 2.0,
):
    """Poll ComfyUI ``/history/<prompt_id>`` until the job finishes, is cancelled,
    errors, or times out. Returns ``(status, value)``:

      - ``POLL_CANCELLED``  -> value None. ``cancel_check()`` returned truthy; this
        helper has already issued ``interrupt_comfy`` + ``clear_comfy_history``.
      - ``POLL_EXEC_ERROR`` -> value is the ``history_execution_error`` detail (str).
      - ``POLL_FINISHED``   -> value is the finished ``history[prompt_id]`` entry
        (dict); the caller reads its ``"outputs"`` for the medium's artifact.
      - ``POLL_TIMEOUT``    -> value is the last poll Exception (or None if none).

    ``on_tick(elapsed_seconds)`` is called once per iteration BEFORE the /history
    request (``elapsed_seconds`` is ``time.time() - start_time``), so the caller can
    drive its own progress UI. Retry cadence, the in-loop exception swallow + retry,
    and the timeout-with-last-error semantics match the per-module originals."""
    import time
    start_time = time.time()
    last_poll_err: Optional[Exception] = None
    while time.time() - start_time < max_poll_seconds:
        if cancel_check and cancel_check():
            interrupt_comfy(api_url)
            clear_comfy_history(api_url, prompt_id)
            return POLL_CANCELLED, None
        if on_tick is not None:
            on_tick(time.time() - start_time)
        try:
            hist_req = urllib.request.Request(f"{api_url}/history/{prompt_id}")
            with _comfy_urlopen(hist_req, timeout=history_timeout) as response:
                history = json.loads(response.read().decode("utf-8"))
            if prompt_id in history:
                entry = history[prompt_id]
                err = history_execution_error(entry)
                if err:
                    return POLL_EXEC_ERROR, err
                return POLL_FINISHED, entry
        except Exception as e:
            # Keep retrying within the loop (ComfyUI may just be busy), but
            # remember the last failure so a crashed/unreachable ComfyUI is
            # distinguishable from a merely slow one if we time out below.
            last_poll_err = e
        time.sleep(sleep_seconds)
    return POLL_TIMEOUT, last_poll_err


def select_output_info(entry: dict, output_keys) -> Optional[dict]:
    """The first artifact info dict in a finished ``/history`` entry's outputs,
    scanning each node's output for any of *output_keys* (a non-empty list under
    that key yields its first element). Returns None when no node carries one.

    ``output_keys`` is the medium's save-node output keys, e.g. ``("images",)``
    for image, ``("videos", "gifs", "images")`` for video, ``("audio",)`` for
    music. The first key (in the given order) that a node has a non-empty list
    for wins, matching the per-module poll loops."""
    for node_output in entry.get("outputs", {}).values():
        for key in output_keys:
            if node_output.get(key):
                return node_output[key][0]
    return None


def comfy_fetch_output(api_url: str, info: dict, output_path: Path, *,
                       timeout: float) -> None:
    """Download a finished artifact from ComfyUI (GET /view?filename&subfolder&type)
    and write the bytes to *output_path* (creating parent dirs). Builds the query
    from *info*'s ``filename`` / ``subfolder`` / ``type`` (``type`` defaults to
    "output"). Raises on any failure - the caller wraps it in its own error text."""
    params = urllib.parse.urlencode({
        "filename": info.get("filename"),
        "subfolder": info.get("subfolder", ""),
        "type": info.get("type", "output"),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _comfy_urlopen(f"{api_url}/view?{params}", timeout=timeout) as response:
        output_path.write_bytes(response.read())
