# SPDX-License-Identifier: AGPL-3.0-or-later
"""One capability-signal system for every registered model.

Answers "can this model do X" for the capabilities routing consumes:
vision, audio input, tool use, reasoning, and context length. Every answer is a TRI-STATE -
``True`` / ``False`` / ``None`` for the booleans, an ``int`` / ``None`` for
context length - where ``None`` means NOT INSPECTED, never "no".

Callers must not render a negative for ``None`` and must not route on it: an
entry on an unmounted drive, a header that truncated, and a model nobody has
looked at yet all produce ``None``, and none of them is evidence that the model
lacks the capability.

The four signals come from three different places, and the difference is
load-bearing rather than incidental:

vision           LIVE per call, delegated to registry.model_vision_capability.
                 It depends on a SIBLING FILE (the mmproj projector) that can
                 appear, move or vanish independently of the model, so a stored
                 answer goes stale without the model itself changing.
audio            LIVE per call, delegated to registry.model_audio_capability,
                 for the same reason as vision.
tool_use         PERSISTED on the registry entry at registration, read from the
                 model's own chat template.
context_length   PERSISTED likewise, from the model's own header.
reasoning        LIVE per call from the model's own chat template (a GGUF header
                 or a HuggingFace directory's tokenizer_config.json), kept per
                 file in memory; never from the model's name.

tool_use and context_length are cached precisely because they are baked into the
model file's own immutable bytes: the same file at the same path answers the
same way forever, so the staleness argument that forbids caching vision does not
apply to them. They follow the architecture/expert_count precedent in
registry.py exactly, key presence and all: a key ABSENT means never checked, a
key PRESENT means confirmed, and every read tests ``is not None`` rather than
truthiness so a confirmed ``False``/``0`` stays distinct from an absence.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from . import registry as _registry
from .gguf import _gguf_capability_probe, chat_template_tool_signal

logger = logging.getLogger(__name__)

# Substrings that mark a chat template as built for a model that reasons before
# it answers: the thinking switch the template reads (``enable_thinking``,
# ``reasoning_effort``), the reasoning block it writes or strips (``<think>``,
# ``</think>``), the reasoning field it renders (``reasoning_content``), and the
# harmony analysis channel. Each is a template construct, never prose.
_REASONING_TEMPLATE_MARKERS = (
    "enable_thinking", "reasoning_effort", "<think>", "</think>",
    "reasoning_content", "<|channel|>analysis",
)

VISION = "vision"
AUDIO = "audio_input"
TOOL_USE = "tool_use"
REASONING = "reasoning"

# The boolean capabilities, in the order a report lists them. Context length is
# deliberately NOT here: it is an integer whose routing question is "at least how
# many tokens", not a yes/no, and folding it in would force every caller to
# special-case one member of its own enumeration.
BOOLEAN_CAPABILITIES = (VISION, AUDIO, TOOL_USE, REASONING)

CONTEXT_LENGTH = "context_length"

# Registry-entry keys holding a capability read once at registration.
_ENTRY_TOOL_USE_KEY = TOOL_USE
_ENTRY_CONTEXT_LENGTH_KEY = CONTEXT_LENGTH


def _entry_for(name: str, reg: Optional[dict]) -> Optional[dict]:
    """The registry entry dict for *name*, or None when absent or malformed.

    A registry value that is not a dict (a bare path string from a hand-edited
    registry.json) has no capability keys to read and would raise on ``.get``,
    so it resolves to "nothing to inspect" rather than crashing the caller."""
    reg = _registry._mm.load_registry() if reg is None else reg
    if not isinstance(reg, dict):
        return None
    entry = reg.get(name)
    return entry if isinstance(entry, dict) else None


def _hf_dir_chat_template(model_dir: Path) -> Optional[str]:
    """The chat template declared by a HuggingFace model directory, from
    ``tokenizer_config.json``. None when the file is absent, unreadable, or
    declares none.

    Newer exports may state a LIST of named templates (``[{"name": ...,
    "template": ...}]``) instead of one string; the entry named "default" wins,
    else the first, matching how transformers itself picks one."""
    try:
        cfg = model_dir / "tokenizer_config.json"
        if not cfg.is_file():
            return None
        data = json.loads(cfg.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError, RecursionError) as e:
        logger.debug("chat template read failed for %s (%s): %s",
                     model_dir, type(e).__name__, e)
        return None
    tmpl = data.get("chat_template") if isinstance(data, dict) else None
    if isinstance(tmpl, str):
        return tmpl
    if isinstance(tmpl, list):
        named = [t for t in tmpl if isinstance(t, dict) and "template" in t]
        for t in named:
            if t.get("name") == "default" and isinstance(t["template"], str):
                return t["template"]
        for t in named:
            if isinstance(t["template"], str):
                return t["template"]
    return None


def _hf_dir_context_length(model_dir: Path) -> Optional[int]:
    """The trained context window a HuggingFace model directory declares, from
    ``config.json``'s ``max_position_embeddings``. None when absent, unreadable,
    or not a positive integer."""
    try:
        cfg = model_dir / "config.json"
        if not cfg.is_file():
            return None
        data = json.loads(cfg.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError, RecursionError) as e:
        logger.debug("context length read failed for %s (%s): %s",
                     model_dir, type(e).__name__, e)
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("max_position_embeddings")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    return None


def model_tool_use_capability(name: str, *,
                              reg: Optional[dict] = None) -> Optional[bool]:
    """Whether ONE registered model is known to emit STRUCTURED tool calls - a
    TRI-STATE.

    ``True``  its chat template renders tool calls.
    ``False`` its template was read and renders none.
    ``None``  nobody has read it: the entry is missing or malformed, the file is
              a format with no template to read, or it was registered before this
              capability existed and the opportunistic backfill has not reached
              it yet.

    Answers from the value stored on the registry entry at registration
    (``gguf_capability_metadata``), falling back to a live read of a
    HuggingFace-format directory's ``tokenizer_config.json``, which has no
    registration-time capture. Does NOT probe a GGUF file live: that read costs
    real time per model and routing evaluates every candidate, which is the whole
    reason the value is captured once at registration instead.

    A ``False`` is a FITNESS signal, never a refusal - localm's coder drives
    tools by prompting for ``<tool_call>`` XML and parsing the reply, which works
    against any instruction-following model."""
    entry = _entry_for(name, reg)
    if entry is None:
        return None
    stored = entry.get(_ENTRY_TOOL_USE_KEY)
    if isinstance(stored, bool):
        return stored
    epath = _registry._entry_path(entry)
    if epath is None:
        return None
    try:
        p = Path(epath)
        if p.is_dir():
            return chat_template_tool_signal(_hf_dir_chat_template(p))
    except (OSError, ValueError) as e:
        logger.debug("tool-use capability probe failed for %r (%s): %s",
                     name, type(e).__name__, e)
    return None


def model_context_length(name: str, *,
                         reg: Optional[dict] = None) -> Optional[int]:
    """The trained context window of ONE registered model, or ``None`` when
    unknown.

    Stored on the entry at registration for a GGUF (from its
    ``<architecture>.context_length`` header key), read live from ``config.json``
    for a HuggingFace-format directory. ``None`` is "not inspected", never
    "small": a caller comparing a prompt size against this must treat None as
    "cannot say", not as a model too small to use.

    This is the model's TRAINED window, which is an upper bound on what a load
    can configure - not the ``n_ctx`` a currently-loaded engine happens to run
    with (Engine.context_capacity)."""
    entry = _entry_for(name, reg)
    if entry is None:
        return None
    stored = entry.get(_ENTRY_CONTEXT_LENGTH_KEY)
    if isinstance(stored, int) and not isinstance(stored, bool) and stored > 0:
        return stored
    epath = _registry._entry_path(entry)
    if epath is None:
        return None
    try:
        p = Path(epath)
        if p.is_dir():
            return _hf_dir_context_length(p)
    except (OSError, ValueError) as e:
        logger.debug("context length probe failed for %r (%s): %s",
                     name, type(e).__name__, e)
    return None


def chat_template_reasoning_signal(template: Optional[str]) -> Optional[bool]:
    """Whether a chat template is built for a reasoning model - a TRI-STATE.

    ``True``  the template carries a reasoning construct (see
              ``_REASONING_TEMPLATE_MARKERS``).
    ``False`` the template was read and carries none.
    ``None``  no template was read, so there is nothing to judge."""
    if template is None:
        return None
    low = template.lower()
    return any(m in low for m in _REASONING_TEMPLATE_MARKERS)


_reasoning_by_file: dict[tuple[str, int, int], Optional[bool]] = {}


def _gguf_reasoning_signal(path: Path) -> Optional[bool]:
    """The reasoning signal of the chat template inside a GGUF file. The header
    read is bounded (``gguf._GGUF_CAPABILITY_PROBE_BYTES``) and the answer is
    kept per file, size and modification time, so a model file is read once.
    A header that could not be walked in full answers ``None`` and is not kept."""
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key in _reasoning_by_file:
        return _reasoning_by_file[key]
    meta = _gguf_capability_probe(path)
    template = meta.get("chat_template")
    if template is None:
        return None
    signal = chat_template_reasoning_signal(template)
    if len(_reasoning_by_file) > 512:
        _reasoning_by_file.clear()
    _reasoning_by_file[key] = signal
    return signal


def model_reasoning_capability(name: str, *,
                               reg: Optional[dict] = None) -> Optional[bool]:
    """Whether ONE registered model reasons before it answers - a TRI-STATE.

    ``True``  its chat template carries a reasoning construct (a thinking
              switch, a ``<think>`` block, a reasoning field or the harmony
              analysis channel).
    ``False`` its template was read and carries none.
    ``None``  nobody could read a template: the entry is missing or malformed,
              the file is neither a GGUF nor a HuggingFace directory with a
              ``tokenizer_config.json``, or its header declares no template.

    The answer comes from the model's own template, the same source as
    ``model_tool_use_capability``, never from the model's name. The name-marker
    heuristic in ``inference.model_family`` stays what it is, a prompt-tuning
    hint for the coder and the chat ``<think>`` nudge, and is not a capability."""
    entry = _entry_for(name, reg)
    if entry is None:
        return None
    epath = _registry._entry_path(entry)
    if epath is None:
        return None
    try:
        p = Path(epath)
        if p.is_dir():
            return chat_template_reasoning_signal(_hf_dir_chat_template(p))
        if p.suffix.lower() == ".gguf" and p.is_file():
            return _gguf_reasoning_signal(p)
    except (OSError, ValueError) as e:
        logger.debug("reasoning capability probe failed for %r (%s): %s",
                     name, type(e).__name__, e)
    return None


def model_capability(name: str, capability: str, *, reg: Optional[dict] = None,
                     dir_cache: Optional[dict] = None) -> Optional[bool]:
    """One BOOLEAN capability of one registered model, as a tri-state.

    *capability* is one of ``BOOLEAN_CAPABILITIES``. An unrecognised name raises
    ValueError rather than answering None, which would be indistinguishable from
    a real "not inspected" and would let a typo silently mean "no model
    qualifies" forever.

    Does disk I/O, so callers on an event loop must run it in an executor."""
    if capability == VISION:
        return _registry.model_vision_capability(name, reg=reg,
                                                 dir_cache=dir_cache)
    if capability == AUDIO:
        return _registry.model_audio_capability(name, reg=reg,
                                                dir_cache=dir_cache)
    if capability == TOOL_USE:
        return model_tool_use_capability(name, reg=reg)
    if capability == REASONING:
        return model_reasoning_capability(name, reg=reg)
    raise ValueError(f"unknown capability {capability!r}; "
                     f"expected one of {BOOLEAN_CAPABILITIES}")


def model_capabilities(name: str, *, reg: Optional[dict] = None,
                       dir_cache: Optional[dict] = None) -> dict:
    """Every capability of one registered model, as
    ``{"vision": ..., "audio_input": ..., "tool_use": ..., "reasoning": ...,
    "context_length": ...}``.

    Each value keeps its own tri-state; a key is always present, so a caller
    reads ``None`` explicitly rather than inferring it from a missing key."""
    reg = _registry._mm.load_registry() if reg is None else reg
    out = {c: model_capability(name, c, reg=reg, dir_cache=dir_cache)
           for c in BOOLEAN_CAPABILITIES}
    out[CONTEXT_LENGTH] = model_context_length(name, reg=reg)
    return out


def models_with_capability(capability: str, *, reg: Optional[dict] = None,
                           dir_cache: Optional[dict] = None) -> list[str]:
    """Registered model names CONFIRMED to have *capability*, sorted.

    POSITIVE MEMBERSHIP ONLY, exactly like ``vision_capable_models``: a name is
    listed only when its capability probe returned ``True``. Absence means "not
    confirmed to have it" and must never be read as "confirmed to lack it" -
    ``model_capability`` is what tells those apart.

    That is the right set for ROUTING, which needs somewhere it can send a
    request and must not send one to a model nobody has inspected. It is the
    wrong set for a badge, which has to show unknown as unknown.

    The registry is loaded ONCE and threaded into every per-name call, and one
    directory listing is shared across models in the same folder."""
    reg = _registry._mm.load_registry() if reg is None else reg
    if not isinstance(reg, dict):
        return []
    dir_cache = {} if dir_cache is None else dir_cache
    return sorted(n for n in reg
                  if model_capability(n, capability, reg=reg,
                                      dir_cache=dir_cache) is True)


def models_with_context_at_least(tokens: int, *,
                                 reg: Optional[dict] = None) -> list[str]:
    """Registered model names whose CONFIRMED trained context window is at least
    *tokens*, sorted by that window descending so the roomiest comes first.

    Positive membership only, on the same reasoning as
    ``models_with_capability``: a model whose window is unknown is not listed,
    because routing a request to it would be a guess that it fits."""
    reg = _registry._mm.load_registry() if reg is None else reg
    if not isinstance(reg, dict):
        return []
    sized = []
    for n in reg:
        ctx = model_context_length(n, reg=reg)
        if ctx is not None and ctx >= tokens:
            sized.append((ctx, n))
    return [n for _, n in sorted(sized, key=lambda t: (-t[0], t[1]))]
