"""Stable identity of a registered model.

Every registry entry carries an ``id`` that never changes for the entry's life.
Its name is what people see and type. A record that stores a reference to a
model keeps the name field older builds read, and beside it a ``model_refs`` map
``{name as written in this record: id}``.

Only stored references are read through :func:`stored_to_current`. What a person
or an API client sends matches current names only, so a former name is refused
there.

Stdlib only: :mod:`localm.config` imports this module.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Optional

REFS_KEY = "model_refs"
FORMER_KEY = "former_names"

# config.json keys holding one model name.
CONFIG_NAME_KEYS = ("embedding_model", "rag_rerank_model", "coder_reviewer_model",
                    "spec_draft_model", "default_model", "model")
CONFIG_LIST_KEYS = ("pinned_models",)
CONFIG_MAP_KEYS = ("model_avatar_overrides",)
# Keys of ``plugins.<media>.native`` holding a model name or a file path. Must
# cover native_models.media_keys for every media. See
# test_native_setting_keys_cover_every_media_slot.
NATIVE_NAME_KEYS = ("model", "vae", "clip_l", "clip_g", "t5xxl", "llm", "llm_vision",
                    "clip_vision", "dit", "text_encoder", "lm")


@dataclass(frozen=True)
class ModelIndex:
    """One registry state as lookups: ``by_id`` id to name, ``by_name`` name to
    id, ``former`` former name to current name, ``first_name`` current name of a
    renamed entry to the first name it had."""
    by_id: dict = field(default_factory=dict)
    by_name: dict = field(default_factory=dict)
    former: dict = field(default_factory=dict)
    first_name: dict = field(default_factory=dict)


EMPTY_INDEX = ModelIndex()

_rename_listeners: list = []


def add_rename_listener(listener: Callable[[str, str], None]) -> None:
    """Have *listener(old_name, new_name)* called when this process renames a
    model. Adding the same function again has no effect."""
    if listener not in _rename_listeners:
        _rename_listeners.append(listener)


def notify_renamed(old_name: str, new_name: str) -> None:
    """Call every rename listener. A listener that raises is logged and the rest
    still run, so this never raises."""
    for listener in list(_rename_listeners):
        try:
            listener(old_name, new_name)
        except Exception as e:
            logging.getLogger("localm").warning(
                "A model rename listener failed: %s", e)


def new_model_id() -> str:
    """A fresh model ID."""
    return uuid.uuid4().hex


def build_index(reg: Any) -> ModelIndex:
    """The :class:`ModelIndex` of registry dict *reg*. A non-dict entry or one
    without an ID still counts as a name for ``by_name``, with the id ``""``.
    Ignores anything that is not a dict."""
    if not isinstance(reg, dict):
        return EMPTY_INDEX
    by_id: dict = {}
    by_name: dict = {}
    former: dict = {}
    first_name: dict = {}
    for name, entry in reg.items():
        rid = entry.get("id") if isinstance(entry, dict) else None
        rid = rid if isinstance(rid, str) else ""
        by_name[name] = rid
        if rid and rid not in by_id:
            by_id[rid] = name
        olds = entry.get(FORMER_KEY) if isinstance(entry, dict) else None
        if isinstance(olds, list):
            for old in olds:
                if isinstance(old, str):
                    former[old] = name
                    first_name.setdefault(name, old)
    return ModelIndex(by_id, by_name, former, first_name)


def needs_ids(reg: Any) -> bool:
    """True when a dict entry of *reg* has no usable ID or shares one with an
    earlier entry."""
    if not isinstance(reg, dict):
        return False
    seen: set = set()
    for entry in reg.values():
        if not isinstance(entry, dict):
            continue
        rid = entry.get("id")
        if not (isinstance(rid, str) and rid) or rid in seen:
            return True
        seen.add(rid)
    return False


def ensure_ids(reg: Any, previous: Optional[dict] = None) -> bool:
    """Give every dict entry of *reg* an ID of its own, in place. *previous* maps
    a name to the entry it had before this write: an entry replaced under the same
    name takes its ID and ``former_names`` back. When two entries hold one ID (an
    older build's alias copies the whole entry), the first in file order keeps it
    and the other gets a new ID and loses ``former_names``. Returns whether
    anything changed."""
    if not isinstance(reg, dict):
        return False
    changed = False
    seen = {e["id"] for e in reg.values()
            if isinstance(e, dict) and isinstance(e.get("id"), str) and e["id"]}
    kept: set = set()
    for name, entry in reg.items():
        if not isinstance(entry, dict):
            continue
        rid = entry.get("id")
        if isinstance(rid, str) and rid:
            if rid not in kept:
                kept.add(rid)
                continue
            entry.pop(FORMER_KEY, None)
        else:
            old = (previous or {}).get(name)
            old_id = old.get("id") if isinstance(old, dict) else None
            if isinstance(old_id, str) and old_id and old_id not in seen:
                entry["id"] = old_id
                seen.add(old_id)
                kept.add(old_id)
                if isinstance(old.get(FORMER_KEY), list):
                    entry.setdefault(FORMER_KEY, old[FORMER_KEY])
                changed = True
                continue
        fresh = new_model_id()
        entry["id"] = fresh
        seen.add(fresh)
        kept.add(fresh)
        changed = True
    return changed


def stored_to_current(stored: Any, refs: Any, index: ModelIndex) -> Any:
    """The current name of a stored reference: the entry ``refs[stored]`` names,
    else *stored* when it is a current name, else the entry that once had that
    name, else *stored* unchanged. A value that is not a string is returned as is."""
    if not isinstance(stored, str) or not stored:
        return stored
    if isinstance(refs, dict):
        name = index.by_id.get(refs.get(stored))
        if name is not None:
            return name
    if stored in index.by_name:
        return stored
    return index.former.get(stored, stored)


def refs_for(names: Iterable, index: ModelIndex) -> dict:
    """``{name: id}`` for each of *names* that is a registered entry with an ID."""
    out: dict = {}
    for name in names:
        rid = index.by_name.get(name) if isinstance(name, str) else None
        if rid:
            out[name] = rid
    return out


def attach_refs(record: dict, names: Iterable, index: ModelIndex) -> None:
    """Set ``record["model_refs"]`` from *names*, or remove it when none is
    registered."""
    refs = refs_for(names, index)
    if refs:
        record[REFS_KEY] = refs
    else:
        record.pop(REFS_KEY, None)


def follow_bases(reg: Any, index: ModelIndex) -> None:
    """Point every entry's ``base`` at its base model's current name and rebuild
    its ``model_refs``, in place."""
    if not isinstance(reg, dict):
        return
    for entry in reg.values():
        if not isinstance(entry, dict):
            continue
        base = entry.get("base")
        if isinstance(base, str) and base:
            base = stored_to_current(base, entry.get(REFS_KEY), index)
            entry["base"] = base
            attach_refs(entry, (base,), index)
        else:
            entry.pop(REFS_KEY, None)


def rename_entry(reg: dict, old_name: str, new_name: str) -> None:
    """Move entry *old_name* of *reg* to the key *new_name*, recording the old
    name in its ``former_names``. Caller checked that *old_name* exists and
    *new_name* is free."""
    entry = reg.pop(old_name)
    if isinstance(entry, dict):
        olds = entry.get(FORMER_KEY)
        olds = [n for n in olds if isinstance(n, str) and n not in (old_name, new_name)] \
            if isinstance(olds, list) else []
        entry[FORMER_KEY] = olds + [old_name]
    reg[new_name] = entry


def _native_blocks(cfg: dict) -> list:
    plugins = cfg.get("plugins")
    if not isinstance(plugins, dict):
        return []
    blocks = [b.get("native") for b in plugins.values() if isinstance(b, dict)]
    return [b for b in blocks if isinstance(b, dict)]


def config_names(cfg: dict) -> list:
    """Every string in *cfg* that may name a model, from the keys listed above."""
    names: list = [cfg.get(k) for k in CONFIG_NAME_KEYS]
    for key in CONFIG_LIST_KEYS:
        value = cfg.get(key)
        if isinstance(value, list):
            names.extend(value)
    for key in CONFIG_MAP_KEYS:
        value = cfg.get(key)
        if isinstance(value, dict):
            names.extend(value)
    for block in _native_blocks(cfg):
        names.extend(block.get(k) for k in NATIVE_NAME_KEYS)
    return [n for n in names if isinstance(n, str) and n]


def config_to_current(cfg: dict, refs: Any, index: ModelIndex) -> None:
    """Replace every stored model name in *cfg* with its current name, in place."""
    if not index.by_name:
        return
    for key in CONFIG_NAME_KEYS:
        if key in cfg:
            cfg[key] = stored_to_current(cfg[key], refs, index)
    for key in CONFIG_LIST_KEYS:
        value = cfg.get(key)
        if isinstance(value, list):
            cfg[key] = [stored_to_current(n, refs, index) for n in value]
    for key in CONFIG_MAP_KEYS:
        value = cfg.get(key)
        if isinstance(value, dict):
            cfg[key] = {stored_to_current(n, refs, index): v for n, v in value.items()}
    for block in _native_blocks(cfg):
        for key in NATIVE_NAME_KEYS:
            if key in block:
                block[key] = stored_to_current(block[key], refs, index)
