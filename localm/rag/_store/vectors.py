# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vector helpers: the optional numpy binding and its pure-Python
fallback, stored-vector validation, and cosine similarity."""

from __future__ import annotations

import math
from typing import Optional

from localm.debuglog import logger as _log
from localm.rag import store as _st


# numpy is optional: absent -> None, and every caller degrades to pure Python.
# Bound ONCE at module import rather than per call.
try:
    import numpy as _numpy
except ImportError:      # optional dependency - every caller degrades to pure Python
    _numpy = None


#: True when numpy imported but is a namespace stub / attribute-less object
#: rather than a real install (``__file__`` is None for a PEP 420 namespace
#: package).
_NUMPY_IS_STUB = _numpy is not None and getattr(_numpy, "__file__", None) is None


def _warn_numpy_degrade(exc: Exception, operation: str) -> None:
    """Announce the pure-Python fallback ONCE per process, saying which case it is.

    Three states, three branches:

    1. numpy ABSENT      - the default install has no numpy. Debug level, no
                           warning.
    2. numpy is a STUB   - the install is BROKEN. Something put a bare 'numpy'
                           directory on sys.path. Warns, naming the artefact so
                           it can be deleted.
    3. numpy present but
       otherwise UNUSABLE - warns as unexpected.
    """
    if _st._NUMPY_DEGRADE_LOGGED:
        return
    _st._NUMPY_DEGRADE_LOGGED.add(True)
    # The log record carries the exception's text, never the exception object.
    # See test_the_notice_does_not_keep_the_callers_frames_alive.
    reason = str(exc)
    if _st._numpy is None:
        # Absent: logged at debug, never as a warning. Branches on the MODULE
        # STATE, never on the exception's text.
        _log.debug("numpy is not installed; using the pure-Python %s (%s: %s).",
                   operation, type(exc).__name__, reason)
        return
    if _st._NUMPY_IS_STUB:
        _log.warning(
            "numpy imported as an EMPTY NAMESPACE PACKAGE from %s - it is not a real "
            "install, and this will break anything else here that imports numpy. Most "
            "likely a bare 'numpy' directory left on sys.path by a failed or "
            "partially-removed install; find and remove it. Falling back to "
            "pure-Python %s (%s: %s).",
            getattr(_st._numpy, "__path__", None) or "an unknown path",
            operation, type(exc).__name__, reason)
    else:
        _log.warning(
            "numpy is present but unusable (%s: %s); falling back to pure-Python %s. "
            "Results are identical, it is slower on large collections.",
            type(exc).__name__, reason, operation)


#: Set once when numpy has been found present-but-unusable; the pure-Python
#: cosine fallback then announces itself exactly once per process.
_NUMPY_DEGRADE_LOGGED: set = set()


def _first_dim(vectors: list) -> Optional[int]:
    """Dimensionality of the first non-empty vector, or None."""
    for v in vectors:
        if v:
            return len(v)
    return None


def _well_formed_vectors(vectors) -> bool:
    """Cheap (O(n)) structural check that *vectors* is what ``_save`` writes: a
    list whose entries are each a null placeholder (a missing embedding) or a
    list/tuple."""
    return isinstance(vectors, list) and all(
        (not v) or isinstance(v, (list, tuple)) for v in vectors)


def _vectors_finite(vectors) -> bool:
    """True when every component of every present vector is a FINITE number.

    Structure is already validated by ``_well_formed_vectors``; this checks the
    values."""
    try:
        np = _st._numpy
        if np is None:
            raise ImportError("numpy is not installed")
        for v in vectors:
            if not v:
                continue
            try:
                arr = np.asarray(v, dtype="float64")
            except (ValueError, TypeError):
                return False                       # non-numeric component
            if not np.isfinite(arr).all():
                return False
        return True
    # AttributeError as well as ImportError: an attribute-less numpy raises
    # AttributeError rather than failing to import.
    except (ImportError, AttributeError) as e:
        _warn_numpy_degrade(e, "vector validation")
        for v in vectors:
            if not v:
                continue
            for x in v:
                try:
                    if not math.isfinite(x):
                        return False
                except TypeError:
                    return False
        return True


def _maxnorm(scores: list[float]) -> list[float]:
    """*scores* divided by their maximum; unchanged when the maximum is not
    positive."""
    top = max(scores, default=0.0)
    return [s / top for s in scores] if top > 0 else list(scores)


def _cosine(a: list, b: list) -> float:
    if len(a) != len(b):
        # Callers (_vector_scores) guarantee equal lengths; a mismatch raises
        # rather than being scored as a real (zero) similarity.
        raise ValueError(
            f"cosine similarity needs equal-length vectors "
            f"(got {len(a)} and {len(b)})")
    try:
        np = _st._numpy
        if np is None:
            raise ImportError("numpy is not installed")
        va, vb = np.asarray(a, dtype="float32"), np.asarray(b, dtype="float32")
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        sim = float(va @ vb) / denom if denom else 0.0
    except (ImportError, AttributeError) as e:
        # ImportError is the ordinary "numpy not installed" case; AttributeError
        # is an importable but attribute-less numpy, where np.asarray is missing.
        # Not a bare except: a real numerical error from a usable numpy still
        # propagates. The degrade is announced once per process.
        _warn_numpy_degrade(e, "cosine similarity")
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        sim = dot / (na * nb) if na and nb else 0.0
    # A NaN/inf component makes the similarity non-finite; non-finite is
    # returned as a miss (0.0) and never leaves this function.
    return sim if math.isfinite(sim) else 0.0
