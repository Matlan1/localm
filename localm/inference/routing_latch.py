# SPDX-License-Identifier: AGPL-3.0-or-later
"""Remember which models failed to load, so capability routing skips them.

``RoutingLatch`` records a failed model load under the model's name and a
fingerprint of everything that decides whether that load can work: the model's
files, the load-related config keys and the provisioned llama.cpp runtime.
``skipped`` answers which recorded models routing must leave out of its
candidates right now.

A record stops applying when any of these happens:

- the fingerprint no longer matches (the model file, a load setting or the
  runtime changed);
- the model loads successfully (``record_success``);
- its backoff elapses. The delay starts at ``BACKOFF_BASE_S`` and doubles with
  each consecutive failure under an unchanged fingerprint, up to
  ``BACKOFF_MAX_S``. The record is kept while it is due, so a failed retry
  lengthens the next delay. Failures recorded while the record is still inside
  its backoff count once together.

Only routing consults the latch. A load the user asked for by name never does.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

from localm.inference.capability_routing import SkippedCandidate

BACKOFF_BASE_S = 600.0
BACKOFF_MAX_S = 6 * 3600.0
REASON_MAX_CHARS = 160

LOAD_CONFIG_KEYS = (
    "n_ctx", "n_ctx_max", "n_ctx_grow", "ctx_auto",
    "n_gpu_layers", "n_gpu_layers_auto", "n_cpu_moe", "use_mmap", "mtp_enabled",
    "mtp_draft_tokens", "spec_source", "spec_draft_tokens", "spec_draft_model",
    "vram_overhead_mb", "gpu_split_ratios", "gpu_split_indices",
    "main_gpu_index", "gguf_load_timeout_s", "hf_load_timeout_s",
    "hf_trust_remote_code", "binary_dir", "llama_runtime_pin",
)


def short_reason(detail: object) -> str:
    """*detail* as one line of at most ``REASON_MAX_CHARS`` characters:
    whitespace collapsed, cut with ``...`` when longer."""
    text = " ".join(str(detail or "").split())
    if len(text) <= REASON_MAX_CHARS:
        return text
    return text[:REASON_MAX_CHARS - 3].rstrip() + "..."


def _file_identity(path: object) -> list:
    """``[path, size, mtime_ns]`` for *path*, with None for size and mtime when
    it cannot be read."""
    if not path:
        return [None, None, None]
    try:
        st = os.stat(path)
    except OSError:
        return [str(path), None, None]
    return [str(path), st.st_size, st.st_mtime_ns]


def load_fingerprint(name: str) -> str:
    """A short digest of what decides whether loading model *name* can work:
    its model and projector files (path, size, modification time), the
    load-related config keys (``LOAD_CONFIG_KEYS``), the newest entry of
    ``llama_runtime_history`` and the identity of the provisioned runtime
    directory (``installed_runtime_identity``), which also changes when
    ``setup-llama --from`` or ``--url`` replaces the runtime.

    Never raises. A part that cannot be read contributes the marker
    ``"unreadable"``, so an unreadable state still yields a stable digest."""
    from localm.debuglog import logger

    parts: dict = {}
    try:
        from localm.config import load_config
        cfg = load_config()
        parts["config"] = {k: cfg.get(k) for k in LOAD_CONFIG_KEYS}
        history = cfg.get("llama_runtime_history")
        parts["runtime"] = (history[-1] if isinstance(history, list) and history
                            else None)
    except Exception as exc:
        logger.debug("routing latch: load config unreadable for %s: %s", name, exc)
        parts["config"] = parts["runtime"] = "unreadable"
    try:
        from localm.setup_llama import installed_runtime_identity
        parts["runtime_dir"] = installed_runtime_identity()
    except Exception as exc:
        logger.debug("routing latch: runtime directory unreadable for %s: %s",
                     name, exc)
        parts["runtime_dir"] = "unreadable"
    try:
        from localm.model_manager import get_model_info, get_model_mmproj
        info = get_model_info(name)
        parts["model"] = _file_identity(info[0] if info else None)
        parts["mmproj"] = _file_identity(get_model_mmproj(name))
    except Exception as exc:
        logger.debug("routing latch: model files unreadable for %s: %s", name, exc)
        parts["model"] = parts["mmproj"] = "unreadable"
    blob = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class LoadFailure:
    """One recorded load failure: the model, the fingerprint it failed under,
    when it failed (``failed_at``, epoch seconds), the short ``reason``, how
    many consecutive times it has failed under that fingerprint and when routing
    may try it again (``retry_at``, epoch seconds)."""

    model: str
    fingerprint: str
    failed_at: float
    reason: str
    attempts: int
    retry_at: float


def backoff_seconds(attempts: int) -> float:
    """The delay before routing retries a model that has failed *attempts*
    consecutive times: ``BACKOFF_BASE_S`` doubled per extra failure, capped at
    ``BACKOFF_MAX_S``."""
    n = max(1, int(attempts))
    return min(BACKOFF_MAX_S, BACKOFF_BASE_S * (2 ** (n - 1)))


class RoutingLatch:
    """Thread-safe record of models whose last load failed.

    *clock* returns epoch seconds and *fingerprint* maps a model name to its
    load fingerprint; both can be replaced for tests."""

    def __init__(self, *, clock: Callable[[], float] = time.time,
                 fingerprint: Callable[[str], str] = load_fingerprint) -> None:
        self._clock = clock
        self._fingerprint = fingerprint
        self._lock = threading.Lock()
        self._records: Dict[str, LoadFailure] = {}

    def fingerprint(self, name: str) -> str:
        """The current load fingerprint of *name*."""
        return self._fingerprint(name)

    def record_failure(self, name: str, reason: object, *,
                       fingerprint: Optional[str] = None) -> LoadFailure:
        """Record that loading *name* failed with *reason*.

        *fingerprint* is the one taken when the load started; omitted, it is
        taken now. A failure under the fingerprint already recorded adds to
        ``attempts`` once that record's backoff has elapsed; while it is still
        inside its backoff the failure refreshes ``failed_at``, ``reason`` and
        ``retry_at`` and leaves ``attempts`` unchanged. Under a different
        fingerprint ``attempts`` starts again at 1."""
        fp = fingerprint if fingerprint is not None else self._fingerprint(name)
        now = self._clock()
        with self._lock:
            prior = self._records.get(name)
            if prior is None or prior.fingerprint != fp:
                attempts = 1
            elif now < prior.retry_at:
                attempts = prior.attempts
            else:
                attempts = prior.attempts + 1
            record = LoadFailure(
                model=name, fingerprint=fp, failed_at=now,
                reason=short_reason(reason), attempts=attempts,
                retry_at=now + backoff_seconds(attempts))
            self._records[name] = record
        return record

    def record_success(self, name: str) -> None:
        """Forget any failure recorded for *name*."""
        with self._lock:
            self._records.pop(name, None)

    def clear(self) -> None:
        """Forget every recorded failure."""
        with self._lock:
            self._records.clear()

    def failure(self, name: str) -> Optional[LoadFailure]:
        """The record for *name*, or None. Does not check the fingerprint or the
        backoff."""
        with self._lock:
            return self._records.get(name)

    def skipped(self, names: Optional[Iterable[str]] = None
                ) -> Dict[str, SkippedCandidate]:
        """The models routing must leave out right now, keyed by name: recorded
        failures still inside their backoff whose recorded fingerprint matches
        the current one. A record whose fingerprint changed is dropped. With
        *names*, only the records of those models are examined.

        Costs nothing when nothing has failed: no config or file is read."""
        wanted = None if names is None else set(names)
        with self._lock:
            if not self._records:
                return {}
            pending = [r for r in self._records.values()
                       if wanted is None or r.model in wanted]
        now = self._clock()
        out: Dict[str, SkippedCandidate] = {}
        stale = []
        for rec in pending:
            if self._fingerprint(rec.model) != rec.fingerprint:
                stale.append(rec)
                continue
            if now < rec.retry_at:
                out[rec.model] = SkippedCandidate(
                    model=rec.model, failed_at=rec.failed_at,
                    retry_at=rec.retry_at, reason=rec.reason)
        if stale:
            with self._lock:
                for rec in stale:
                    if self._records.get(rec.model) is rec:
                        del self._records[rec.model]
        return out
