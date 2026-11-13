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


def _component_entries(model_type: str, registry: Optional[dict]) -> list:
    """``[(name, entry, filename), ...]`` for the registered component files of
    *model_type*, sorted by name. GGUF text-model LoRA adapters, registered
    folders and entries without a path are left out."""
    from localm.model_manager import _entry_path, is_gguf_adapter_entry
    from localm.model_manager.registry import _entry_is_folder, models_of_type
    out = []
    for name, entry in sorted(models_of_type(model_type, registry).items()):
        if is_gguf_adapter_entry(entry) or _entry_is_folder(entry):
            continue
        epath = _entry_path(entry)
        if epath is None:
            # A malformed entry is skipped rather than crashing the picker.
            continue
        out.append((name, entry, epath.replace("\\", "/").rsplit("/", 1)[-1]))
    return out


def registry_models_of_type(model_type: str, registry: Optional[dict] = None) -> list:
    """Registered models of *model_type*, as ``[{"name", "filename"}, ...]``.

    ``filename`` is the BASENAME of the registered path, because that is the
    name ComfyUI reports in an ``/object_info`` combo, which is what makes the two
    lists comparable. The full path is NOT returned. A GGUF text-model LoRA
    adapter is a ``lora`` entry too but is not a ComfyUI component, and neither
    is a registered folder (such as a part of a diffusers pipeline): both are
    left out."""
    return [{"name": name, "filename": filename}
            for name, _entry, filename in _component_entries(model_type, registry)]


def _not_family(families: frozenset):
    return lambda info: info.kind != "music" and info.family not in families


# The companion slots of a diffusion family that hold a text encoder.
_TEXT_SLOTS = ("clip_l", "clip_g", "t5xxl", "llm")


def _image_vae_fits(info) -> bool:
    """Whether *info* can fill the image VAE role of a workflow of unknown
    family: a VAE some image family decodes with (``families.accepts``, so the
    3D VAE classified ``wan2.1`` that Qwen-Image and Anima use is kept), or
    any other non-music file outside the video families."""
    from localm.media.sdcpp import families
    if info.kind == "music":
        return False
    if info.kind != "vae":
        return info.family not in families.VIDEO_FAMILIES
    return any(families.accepts("vae", fam, info) for fam in families.IMAGE_FAMILIES)


def _family_fits(family: str, variant: Optional[str]) -> dict:
    """``{role_id: test}`` for the image component roles of a workflow whose
    diffusion model is of the image family *family* (*variant* is passed to
    ``families.accepts``): a text-encoder role takes a file that fits any
    text-encoder slot of that family, the VAE role a VAE that family decodes
    with."""
    from localm.media.sdcpp import families
    req = families.requirements(families.FileInfo("diffusion", family))
    text = [r.slot for r in req if r.slot in _TEXT_SLOTS]

    def text_fits(info) -> bool:
        return any(families.accepts(s, family, info, variant) for s in text)

    def vae_fits(info) -> bool:
        return families.accepts("vae", family, info, variant)

    return {"image-clip1": text_fits, "image-clip2": text_fits, "image-vae": vae_fits}


def _role_fits(family: Optional[str] = None, variant: Optional[str] = None) -> dict:
    """``{role_id: test}`` for the roles the built-in media plugins declare;
    each test takes a classified file (``families.FileInfo``) and says whether
    it can fill that role. With *family*, the diffusion family of the active
    workflow's model (and *variant*, its size), when that is an image family,
    the image text-encoder and VAE roles take only files that family can
    use."""
    from localm.media.sdcpp import families
    image, video = families.IMAGE_FAMILIES, families.VIDEO_FAMILIES
    fits = {
        "image-unet": _not_family(video),
        "image-vae": _image_vae_fits,
        "image-clip1": lambda info: info.kind in ("clip_l", "clip_h", "llm"),
        "image-clip2": lambda info: info.kind in ("t5xxl", "clip_g"),
        "video-unet": _not_family(image),
        "video-vae": _not_family(image),
        "video-clip": lambda info: info.kind in ("t5xxl", "umt5", "llm", "clip_l"),
        "music-unet": lambda info: info.kind == "music",
        "music-clip": lambda info: info.kind in ("music", "umt5", "t5xxl"),
        "music-vae": lambda info: info.family not in image | video,
    }
    if family in image:
        fits.update(_family_fits(family, variant))
    return fits


def workflow_family(slots: Optional[list],
                    registry: Optional[dict] = None) -> tuple[Optional[str], Optional[str]]:
    """``(family, variant)`` of the diffusion model the active workflow loads:
    the registered file whose basename is the ``current`` value of the
    workflow's diffusion-model slots, classified from its header. ``(None,
    None)`` when *slots* is None, no such slot names a registered file, the
    file is not a diffusion model of a known family, or two slots name models
    of different families. Never raises."""
    if not slots:
        return None, None
    from localm.media.sdcpp import families
    entries = None
    found: set = set()
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        if model_type_for_node(str(slot.get("class_type") or "")) != "diffusion-unet":
            continue
        current = slot.get("current")
        if not isinstance(current, str) or not current:
            continue
        if entries is None:
            entries = {filename.lower(): entry for _name, entry, filename
                       in _component_entries("diffusion-unet", registry)}
        entry = entries.get(current.replace("\\", "/").rsplit("/", 1)[-1].lower())
        info = _entry_info(entry) if entry is not None else None
        if info is None or info.kind != "diffusion" or not families.known_family(info.family):
            continue
        found.add((info.family, info.detail.get("variant")))
    if len(found) == 1:
        return next(iter(found))
    return None, None


def _entry_info(entry):
    """The classified file of registry *entry* (``families.FileInfo``), or None
    when it cannot be read: a network or device path, or a header that names
    no known kind. Never raises."""
    from localm.media.sdcpp import families
    music = families.music_entry(entry)
    if music is not None:
        return music
    from localm.model_manager.registry import _entry_path
    from localm.pathsafe import is_unc_or_device_path
    raw = _entry_path(entry)
    if not raw or is_unc_or_device_path(raw):
        return None
    info = families.inspect_file(raw)
    return None if info.kind == "unknown" else info


def registry_models_for_role(role: dict, registry: Optional[dict] = None,
                             family: Optional[str] = None,
                             variant: Optional[str] = None) -> list:
    """Registered models that can fill *role*, as ``[{"name", "filename"}, ...]``:
    the models of the role's ``model_type`` (see :func:`registry_models_of_type`),
    less the files whose header shows they belong elsewhere (a CLIP-L file for a
    T5 role, an image checkpoint for the music model role, a music file for an
    image role). With *family* (and *variant*), the diffusion family of the
    active workflow's model (:func:`workflow_family`), an image text-encoder
    or VAE role keeps only the files that image family can use. A file whose header cannot
    be read or classified is kept. A role a built-in media plugin did not
    declare is not filtered."""
    mtype = role.get("model_type")
    if not mtype:
        return []
    fits = _role_fits(family, variant).get(role.get("role_id"))
    out = []
    for name, entry, filename in _component_entries(mtype, registry):
        if fits is not None:
            info = _entry_info(entry)
            if info is not None and not fits(info):
                continue
        out.append({"name": name, "filename": filename})
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
    * ``registry_models`` - this box's registered models that can fill the
      role (:func:`registry_models_for_role`), narrowed to the family of the
      workflow's diffusion model when :func:`workflow_family` can tell it.
      Populated even when ComfyUI is down.
    * ``registry_only`` - registered models of this type that ComfyUI is NOT
      offering for the slot, and ONLY when the slot is unsatisfied. On a slot
      ComfyUI serves fine this is always empty."""
    from localm.config import load_registry
    reg = load_registry() if registry is None else registry
    paired = _pair_roles_to_slots(slots, roles) if slots is not None else {}
    family, variant = workflow_family(slots, reg)
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
        known = registry_models_for_role(role, reg, family, variant)
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
