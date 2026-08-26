# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model roles for the media plugins: the consumer side of two registry APIs.

It joins two registry APIs:

* ``Host.register_model_role(ModelRoleDescriptor)`` (``plugins/engine.py``), by
  which a plugin declares WHICH model types it needs and what to call them. The
  image, music and video plugins all declare theirs.
* the registry's ``model_type`` slice - 'diffusion-unet' / 'text-encoder' /
  'vae' / 'lora' - which the Import-from-ComfyUI scan fills in
  (``model_manager/scan.py``).

Each model-file slot of the active workflow is mapped to the localm
``model_type`` it holds, paired with the plugin's declared roles, and matched
against the registry's own models of that type. That answers three questions for
the model picker:

* what IS this dropdown - "Diffusion model (UNet)" rather than ``unet_name``;
* is a declared REQUIRED role missing from this workflow entirely;
* does this box already have a model of the right type registered that ComfyUI
  is not offering (i.e. it lives outside ComfyUI's model folders).

"ComfyUI could not be asked" is never collapsed into "there is nothing here":
``slots=None`` (unreachable) yields ``None`` for every ComfyUI-derived answer,
while the registry answers - which need no ComfyUI - are still returned.
"""

from __future__ import annotations

from typing import Optional

from localm.media.comfy_client import model_type_for_node, slot_is_satisfied

# The component model types a media plugin can declare a role for. 'llm',
# 'mmproj', 'embedding' and 'unknown' are in MODEL_TYPES too but are not
# ComfyUI workflow components, so a registry lookup for them here would only
# ever offer a chat model as a VAE.
COMPONENT_TYPES = ("diffusion-unet", "text-encoder", "vae", "lora")


def plugin_model_roles(app, plugin: str) -> list:
    """The model roles *plugin* registered, in declaration order.

    Reads the live plugin manager off ``app.state`` (the same handle
    ``GET /api/models/roles`` uses) and filters by ``plugin_name``, which
    ``PluginHost.register_model_role`` stamps from the plugin's own spec, so a
    plugin can only ever see its own. Returns ``[]`` when no manager is
    attached (a bare test app) or the plugin registered none - both are
    "nothing declared", which is a real answer here rather than an error."""
    manager = getattr(getattr(app, "state", None), "plugin_manager", None)
    if manager is None:
        # "no manager attached" and "this plugin declared none" both yield an
        # empty list, so the log line says which happened.
        from localm.debuglog import logger
        logger.debug("no plugin manager on app.state; model roles for %r "
                     "cannot be read", plugin)
        return []
    try:
        roles = manager.get_all_model_roles()
    except Exception:
        # A broken plugin's descriptor does not take the model picker down; it
        # still works from the slots alone.
        from localm.debuglog import logger
        logger.warning("could not read model roles for plugin %r", plugin,
                       exc_info=True)
        return []
    return [r for r in roles if r.get("plugin_name") == plugin]


def registry_models_of_type(model_type: str, registry: Optional[dict] = None) -> list:
    """Registered models of *model_type*, as ``[{"name", "filename"}, ...]``.

    ``filename`` is the BASENAME of the registered path, because that is the
    name ComfyUI reports in an ``/object_info`` combo, which is what makes the two
    lists comparable. The full path is NOT returned."""
    from localm.model_manager import _entry_path
    from localm.model_manager.registry import models_of_type
    out = []
    for name, entry in sorted(models_of_type(model_type, registry).items()):
        epath = _entry_path(entry)
        if epath is None:
            # A malformed entry is skipped rather than crashing the picker.
            continue
        out.append({
            "name": name,
            "filename": epath.replace("\\", "/").rsplit("/", 1)[-1],
        })
    return out


def registry_models_by_type(registry: Optional[dict] = None) -> dict:
    """``{model_type: [{"name", "filename"}, ...]}`` for every component type.

    One registry read covers all four types."""
    from localm.config import load_registry
    reg = load_registry() if registry is None else registry
    return {t: registry_models_of_type(t, reg) for t in COMPONENT_TYPES}


def _pair_roles_to_slots(slots: list, roles: list) -> dict:
    """``{slot index: role}`` pairing each slot with a declared role of the SAME
    model_type, positionally within that type.

    Keyed by INDEX rather than ``id(slot)``, so a list holding the same dict
    twice cannot alias.

    The match is POSITIONAL within a model_type: ComfyUI's graph carries no role
    names, so a plugin declaring two text-encoder roles ('CLIP-L', 'T5/CLIP-G')
    against a ``DualCLIPLoader``'s two combo inputs is matched by order. The
    pairing is only ever used to ADD a friendly label; every caller keeps the raw
    ``input_name`` alongside it.

    Surplus slots of a type stay unpaired (role stays None) and surplus roles get
    no slot; neither is an error."""
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
    """*slots* with ``model_type``, ``role_id``/``role_label`` and ``installed``
    added to each entry. ``None`` in, ``None`` out - "ComfyUI could not be asked"
    survives the annotation instead of turning into an empty list.

    The originals are NOT mutated; ``workflow_model_slots`` is shared with
    ``preflight_models``.

    ``installed`` uses ``comfy_client.slot_is_satisfied``, the same rule
    preflight uses to decide a model is missing."""
    if slots is None:
        return None
    paired = _pair_roles_to_slots(slots, roles)
    out = []
    for index, slot in enumerate(slots):
        if not isinstance(slot, dict):
            # A backend broke workflow_model_slots' documented shape: the entry
            # is passed through unannotated and a warning is logged.
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
    """Per-role status for the model picker.

    Each entry carries the declaration (``role_id``, ``label``, ``model_type``,
    ``required``, ``description``), where it landed in the active workflow, and
    which registered models could fill it:

    * ``slot``   - ``{"node_id", "input_name"}`` of the paired slot, else None.
    * ``in_workflow`` - True / False / **None**. None means ComfyUI could not be
      reached, so the active workflow's slots are unknown; False means it was
      read and this role has no slot in it.
    * ``installed`` - whether ComfyUI actually has the file the slot names, or
      None when there is no slot to ask about.
    * ``registry_models`` - this box's registered models of the role's type.
      Independent of ComfyUI, so it is populated even when ComfyUI is down.
    * ``registry_only`` - registered models of this type that ComfyUI is NOT
      offering for the slot, and ONLY when the slot is unsatisfied. On a slot
      ComfyUI serves fine this is always empty."""
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
            # registry_only is populated only where ComfyUI has FAILED to serve
            # the slot.
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
    """The whole model-picker payload for one media plugin, in one pass.

    ``{"reachable", "slots", "roles", "registry_models"}``. Each media backend
    exposes a thin binding onto this (``_comfy_model_roles``), so the registry
    slice is read through the seam the backend already owns.

    ``reachable`` is False exactly when *slots* is None (ComfyUI could not be
    asked). ``registry_models`` is returned in BOTH cases: it comes from this
    box's own registry and needs no ComfyUI at all."""
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
