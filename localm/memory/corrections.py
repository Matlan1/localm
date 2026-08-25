# SPDX-License-Identifier: AGPL-3.0-or-later
"""A PENDING CORRECTION: a proposed supersession of a TRUSTED (user/import) memory that the user has not yet reviewed."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

VALID_ACTIONS = ("update", "delete")


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
class PendingCorrection:
    """A proposed update/delete of a trusted memory, awaiting user review."""

    target_id: str
    action: str
    proposed_text: str = ""
    target_text: str = ""
    confidence: float = 0.5
    source: str = "synth"
    created: float = 0.0
    id: str = field(default_factory=_new_id)

    def __post_init__(self) -> None:
        if self.action not in VALID_ACTIONS:
            self.action = "update"
        self.proposed_text = (self.proposed_text or "").strip()
        self.target_text = (self.target_text or "").strip()
        self.confidence = _clamp01(self.confidence)
        if not self.created:
            self.created = time.time()

    def dedup_key(self) -> tuple:
        """Identity for de-duplication: two proposals with the same target, action, and proposed text are the same suggestion (do not stack duplicates when the same contradiction is distilled run after run)."""
        return (self.target_id, self.action, self.proposed_text.casefold())

    def to_dict(self) -> dict:
        return {
            "id": self.id, "target_id": self.target_id, "action": self.action,
            "proposed_text": self.proposed_text, "target_text": self.target_text,
            "confidence": self.confidence, "source": self.source,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PendingCorrection":
        known = set(cls.__dataclass_fields__)     # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})
