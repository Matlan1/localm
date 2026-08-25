# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model roles for the media plugins: the consumer side of two registry APIs."""

from __future__ import annotations

from typing import Optional

from localm.media.comfy_client import model_type_for_node, slot_is_satisfied

# The component model types a media plugin can declare a role for. 'llm',
# 'mmproj', 'embedding' and 'unknown' are in MODEL_TYPES too but are not
# ComfyUI workflow components, so a registry lookup for them here would only
# ever offer a chat model as a VAE.
COMPONENT_TYPES = ("diffusion-unet", "text-encoder", "vae", "lora")


def plugin_model_roles(app, plugin: str) -> list:
    """The model roles *plugin* registered, in declaration order."""
    manager = getattr(getattr(app, "state", None), "plugin_manager", None)
    if manager is None:
        # "no manager attached" and "this plugin declared none" both yield an
        # empty list, and only one of them is normal. Say which happened rather
        # than letting a missing manager read as a plugin that declares nothing
        # (the same reason http_server carries a sentinel for this state).
        from localm.debuglog import logger
        logger.debug("no plugin manager on app.state; model roles for %r "
                     "cannot be read", plugin)
        return []
    try:
        roles = manager.get_all_model_roles()
    except Exception:
        # A broken plugin's descriptor must not take the model picker down with
        # it; the picker still works from the slots alone. Surfaced, not hidden.
        from localm.debuglog import logger
        logger.warning("could not read model roles for plugin %r", plugin,
                       exc_info=True)
        return []
    return [r for r in roles if r.get("plugin_name") == plugin]


def registry_models_of_type(model_type: str, registry: Optional[dict] = None) -> list:
    """Registered models of *model_type*, as ``[{'name', 'filename'}, ...]``."""
    from localm.model_manager import _entry_path
    from localm.model_manager.registry import models_of_type
    out = []
    for name, entry in sorted(models_of_type(model_type, registry).items()):
        epath = _entry_path(entry)
        if epath is None:
            # A malformed entry is skipped rather than crashing the picker, the
            # same way every other registry consumer routes through _entry_path.
            continue
        out.append({
            "name": name,
            "filename": epath.replace("\\", "/").rsplit("/", 1)[-1],
        })
    return out


def registry_models_by_type(registry: Optional[dict] = None) -> dict:
    """``{model_type: [{'name', 'filename'}, ...]}`` for every component type."""
    from localm.config import load_registry
    reg = load_registry() if registry is None else registry
    return {t: registry_models_of_type(t, reg) for t in COMPONENT_TYPES}


def _pair_roles_to_slots(slots: list, roles: list) -> dict:
    """``{slot index: role}`` pairing each slot with a declared role of the SAME model_type, positionally within that type."""
    by_type: dict = {}
    for role in roles:
        by_type.setdefault(role.get("model_type"), []).append(role)
    paired: dict = {}
    used: dict = {}
    for index, slot in enumerate(slots):
        if not isinstance(slot, dict):
            continue
        mtype = model_type_for_node(slot.get("class_type"))
        candidates = by_type.get(mtype) or []
        i = used.get(mtype, 0)
        if i < len(candidates):
            paired[index] = candidates[i]
            used[mtype] = i + 1
    return paired


def annotate_slots(slots: Optional[list], roles: list) -> Optional[list]:
    """*slots* with ``model_type``, ``role_id``/``role_label`` and ``installed`` added to each entry. ``None`` in, ``None`` out - 'ComfyUI could not be asked' survives the annotation instead of turning into an empty list."""
    if slots is None:
        return None
    paired = _pair_roles_to_slots(slots, roles)
    out = []
    for index, slot in enumerate(slots):
        if not isinstance(slot, dict):
            # A backend broke workflow_model_slots' documented shape. Pass the
            # entry through rather than 500-ing the whole picker, but SAY so -
            # an un-annotated row that nobody logged is the hidden-problem shape
            # rule 5 forbids.
            from localm.debuglog import logger
            logger.warning(
                "model-slot entry is %s, not the documented dict shape - "
                "left unannotated", type(slot).__name__)
            out.append(slot)
            continue
        role = paired.get(index)
        annotated = dict(slot)
        annotated["model_type"] = model_type_for_node(slot.get("class_type"))
        annotated["role_id"] = role.get("role_id") if role else None
        annotated["role_label"] = role.get("label") if role else None
        annotated["installed"] = slot_is_satisfied(slot)
        out.append(annotated)
    return out


def describe_roles(roles: list, slots: Optional[list],
                   registry: Optional[dict] = None) -> list:
    """Per-role status for the model picker."""
    from localm.config import load_registry
    reg = load_registry() if registry is None else registry
    paired = _pair_roles_to_slots(slots, roles) if slots is not None else {}
    slot_by_role = {}
    for index, slot in enumerate(slots or []):
        role = paired.get(index)
        if role is not None and isinstance(slot, dict):
            slot_by_role[role.get("role_id")] = slot

    out = []
    for role in roles:
        role_id = role.get("role_id")
        mtype = role.get("model_type")
        slot = slot_by_role.get(role_id)
        known = registry_models_of_type(mtype, reg) if mtype else []
        installed = None if slot is None else slot_is_satisfied(slot)
        if installed is not False:
            # Only where ComfyUI has FAILED to serve the slot. MEASURED against a
            # live server before this gate existed: every same-type model you own
            # was listed under every satisfied slot too, so the video page
            # advertised a flux UNet and a music checkpoint as things to go
            # install. True, useless, and it buried the one case that is neither.
            registry_only = []
        else:
            options = {str(o).replace("\\", "/").rsplit("/", 1)[-1].lower()
                       for o in (slot.get("options") or [])}
            registry_only = [m for m in known
                             if m["filename"].lower() not in options]
        out.append({
            "role_id": role_id,
            "label": role.get("label"),
            "model_type": mtype,
            "required": bool(role.get("required", True)),
            "description": role.get("description", ""),
            "slot": None if slot is None else {
                "node_id": slot.get("node_id"),
                "input_name": slot.get("input_name"),
            },
            "current": None if slot is None else slot.get("current"),
            "in_workflow": None if slots is None else slot is not None,
            "installed": installed,
            "registry_models": known,
            "registry_only": registry_only,
        })
    return out


def resolve_model_roles(slots: Optional[list], roles: list,
                        registry: Optional[dict] = None) -> dict:
    """The whole model-picker payload for one media plugin, in one pass."""
    reg = registry
    if reg is None:
        from localm.config import load_registry
        reg = load_registry()
    return {
        "reachable": slots is not None,
        "slots": annotate_slots(slots, roles) or [],
        "roles": describe_roles(roles, slots, reg),
        "registry_models": registry_models_by_type(reg),
    }
