# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ``kv_cache_type`` setting: the element type llama.cpp stores the KV cache
in, the ggml type id a context is created with, and the per-token size of a KV
cache in that type.

``f16`` is llama.cpp's own default. ``q8_0`` stores each 32-element block in 34
bytes (1.0625 bytes per element against f16's 2), so the same VRAM holds about
1.88 times the context. A quantised KV cache needs flash attention; llama.cpp
turns it on for a context created with flash attention left on auto, and refuses
the context when flash attention is forced off or the model cannot use it.
"""

from __future__ import annotations

from typing import Any, Optional

KV_CACHE_F16 = "f16"
KV_CACHE_Q8_0 = "q8_0"

KV_CACHE_TYPES = (KV_CACHE_F16, KV_CACHE_Q8_0)
DEFAULT_KV_CACHE_TYPE = KV_CACHE_F16

# ggml_type enum value of each setting value (ggml.h: GGML_TYPE_F16 = 1,
# GGML_TYPE_Q8_0 = 8). See test_ggml_type_ids_match_the_runtime.
_GGML_TYPE = {KV_CACHE_F16: 1, KV_CACHE_Q8_0: 8}

# Bytes per KV element as (numerator, denominator): f16 is 2, q8_0 is 34 bytes
# per 32-element block.
_BYTES_PER_ELEMENT = {KV_CACHE_F16: (2, 1), KV_CACHE_Q8_0: (34, 32)}


def coerce_kv_cache_type(value: Any) -> Optional[str]:
    """*value* as one of :data:`KV_CACHE_TYPES`, or None when it is not one.

    Only an exact string match counts: case and surrounding whitespace are not
    folded, and a non-string is never a type."""
    if isinstance(value, str) and value in KV_CACHE_TYPES:
        return value
    return None


def resolve_kv_cache_type(cfg: dict) -> str:
    """The ``kv_cache_type`` a load should use under config *cfg*.

    An absent key gives the default. A stored value that is not one of
    :data:`KV_CACHE_TYPES` also gives the default and logs a warning naming the
    value."""
    raw = cfg.get("kv_cache_type", DEFAULT_KV_CACHE_TYPE)
    kind = coerce_kv_cache_type(raw)
    if kind is None:
        from localm.debuglog import logger
        logger.warning("kv_cache_type is set to %r, which is not one of %s; "
                       "using %s", raw, ", ".join(KV_CACHE_TYPES), DEFAULT_KV_CACHE_TYPE)
        return DEFAULT_KV_CACHE_TYPE
    return kind


def ggml_type_id(kv_cache_type: str) -> int:
    """The ggml_type enum value for *kv_cache_type*. Raises ValueError for a
    value that is not one of :data:`KV_CACHE_TYPES`."""
    try:
        return _GGML_TYPE[kv_cache_type]
    except KeyError:
        raise ValueError("kv_cache_type must be one of %s, got %r"
                         % (", ".join(KV_CACHE_TYPES), kv_cache_type)) from None


def kv_bytes_for_type(f16_bytes: int, kv_cache_type: Optional[str]) -> int:
    """*f16_bytes* (a KV size computed at 2 bytes per element) rescaled to
    *kv_cache_type*, rounded up. None or ``f16`` returns *f16_bytes* unchanged,
    and so does 0. Raises ValueError for any other value that is not one of
    :data:`KV_CACHE_TYPES`."""
    f16_bytes = int(f16_bytes)
    if kv_cache_type is None:
        return f16_bytes
    ggml_type_id(kv_cache_type)
    if f16_bytes <= 0:
        return f16_bytes
    num, den = _BYTES_PER_ELEMENT[kv_cache_type]
    # f16_bytes / 2 elements, times num/den bytes each, rounded up.
    return -((-f16_bytes * num) // (2 * den))


def apply_kv_cache_type(cp: Any, kv_cache_type: Optional[str]) -> None:
    """Set *cp*'s ``type_k`` and ``type_v`` for *kv_cache_type*. None or ``f16``
    leaves *cp* untouched (llama.cpp's own default is f16). Raises ValueError
    for a value that is not one of :data:`KV_CACHE_TYPES`."""
    if kv_cache_type is None:
        return
    type_id = ggml_type_id(kv_cache_type)
    if kv_cache_type == KV_CACHE_F16:
        return
    cp.type_k = type_id
    cp.type_v = type_id


def refusal_status(status: str, kv_cache_type: Optional[str]) -> str:
    """*status* for a refused context, with a ``:kv_cache_type=<type>`` detail
    suffix when the KV cache was asked for in a quantised type. f16 or None
    returns *status* unchanged."""
    if kv_cache_type is None or kv_cache_type == KV_CACHE_F16:
        return status
    return f"{status}:kv_cache_type={kv_cache_type}"


def refused_type(status: str) -> Optional[str]:
    """The KV cache type a :func:`refusal_status` *status* names, or None when
    it names none."""
    _, _sep, detail = status.partition(":kv_cache_type=")
    return detail or None


def context_refused_note(kv_cache_type: Optional[str]) -> str:
    """The sentence a refused context creation appends when the KV cache was
    asked for in a quantised type, "" for f16 or None."""
    if kv_cache_type is None or kv_cache_type == KV_CACHE_F16:
        return ""
    return (f" The KV cache type is {kv_cache_type} (the kv_cache_type setting), which "
            f"needs flash attention; if this model or runtime cannot use flash "
            f"attention, set kv_cache_type to {KV_CACHE_F16} and load the model again.")
