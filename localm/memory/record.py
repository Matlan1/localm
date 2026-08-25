# SPDX-License-Identifier: AGPL-3.0-or-later
"""
One remembered thing: the ``MemoryRecord`` dataclass.

A record is a small, distilled memory (a durable fact/preference, or an episodic
summary), NOT a raw transcript. Fields carry everything retrieval + forgetting +
provenance need:

  id          stable 16-hex id (so vectors and edits key on it, not on position)
  kind        "semantic" (facts/preferences) | "episodic" (past interactions)
  text        the memory content (bounded; see store.MAX_TEXT_LEN)
  importance  0..1 salience, set at write time (Generative-Agents "importance")
  created     first stored (unix seconds)
  updated     last edited/consolidated
  last_used   last retrieved (drives recency decay + reinforcement)
  uses        retrieval count (reinforcement)
  source      "user" (typed /remember - trusted) | "synth" (LLM-distilled -
              untrusted, importance-capped) | "import" (migrated legacy memory)
  meta        free-form extras (e.g. episodic: outcome/files); forward-compatible
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

VALID_KINDS = ("semantic", "episodic")
VALID_SOURCES = ("user", "synth", "import")


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _clamp01(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if v != v:                       # NaN
        return default
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


@dataclass
class MemoryRecord:
    text: str
    id: str = field(default_factory=_new_id)
    kind: str = "semantic"
    importance: float = 0.5
    created: float = 0.0
    updated: float = 0.0
    last_used: float = 0.0
    uses: int = 0
    source: str = "synth"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        now = time.time()
        self.text = (self.text or "").strip()
        if self.kind not in VALID_KINDS:
            self.kind = "semantic"
        if self.source not in VALID_SOURCES:
            self.source = "synth"
        self.importance = _clamp01(self.importance)
        # Timestamps default to "now" when unset (0), so a hand-built record is
        # never treated as epoch-old by the recency decay.
        if not self.created:
            self.created = now
        if not self.updated:
            self.updated = self.created
        if not self.last_used:
            self.last_used = self.created
        try:
            self.uses = max(0, int(self.uses))
        except (TypeError, ValueError):
            self.uses = 0
        if not isinstance(self.meta, dict):
            self.meta = {}

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "text": self.text,
            "importance": self.importance, "created": self.created,
            "updated": self.updated, "last_used": self.last_used,
            "uses": self.uses, "source": self.source, "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryRecord":
        # Keep only known fields so a forward-compat record with extra keys still
        # loads (mirrors coder Episode.from_dict).
        known = set(cls.__dataclass_fields__)     # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})
