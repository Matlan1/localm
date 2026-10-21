# SPDX-License-Identifier: AGPL-3.0-or-later
"""Turns the worker's WARN/ERROR log lines into the lines a job log shows.

stable-diffusion.cpp logs the failures it recovers from at WARN/ERROR level
before it recovers: a VAE decode that does not fit in VRAM logs a memory
shortfall, a failed segment and a failed compute, then logs that it retries
with tiling, and the image is saved. ``RecoveredWarnings`` holds such a
sequence back and, when the retry line arrives, replaces it with one plain
line. A generation refused for its VRAM budget logs a memory shortfall, a
failed segment and the failed sampling; ``budget_retry`` drops that sequence
when the backend retries without the budget. Lines that are not followed by a
recovery are shown unchanged by ``flush``, so a real failure is never hidden.
Every raw line goes to the debug log whether or not it is collapsed.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

_PRECURSORS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"model manager memory on ",
    r"cannot make enough memory available",
    r"vae segment \d+/\d+ \([^)]*\) failed during",
    r"\bvae decode compute failed",
    r"\breducing vae decode tiles from",
    r"\bsegment \d+/\d+ \([^)]*\) failed during weight preparation",
    r"\Adiffusion model compute failed\b",
    r"\Adiffusion model sampling failed\b",
    r"\A(?:hires )?sampling for image \d+/\d+ failed after\b",
    r"\Asampling(?:\([^)]*\))? failed after\b",
))

_TILE_FAILED = re.compile(r"vae decode compute failed while processing a tile", re.IGNORECASE)

_VAE_RETRY = re.compile(
    r"vae decode ran out of memory; retrying with (spatial\+temporal|spatial|temporal) tiling",
    re.IGNORECASE)

_NO_PROJECTION = re.compile(r"no latent to rgb projection known", re.IGNORECASE)

NO_PREVIEW = "No live preview is available for this model."

_RETRY_WORDS = {
    "spatial": "tiles",
    "temporal": "time slices",
    "spatial+temporal": "tiles and time slices",
}


def vae_retry_message(mode: str, smaller: bool) -> str:
    """The job-log line for a VAE decode retry in *mode* (``spatial``,
    ``temporal`` or ``spatial+temporal``); *smaller* when the decode was already
    tiled and the tiles are being reduced."""
    words = _RETRY_WORDS.get(mode.lower(), "tiles")
    if smaller:
        return f"Still not enough VRAM; decoding in smaller {words}."
    return f"Not enough VRAM to decode in one piece; decoding in {words}."


def vae_escalation_message(mode: str) -> str:
    """The job-log line for a second decode retry that changes *mode*."""
    return f"Still not enough VRAM; decoding in {_RETRY_WORDS.get(mode.lower(), 'tiles')}."


class RecoveredWarnings:
    """Relays worker log lines to *say*, collapsing recovered sequences. Once
    *cancelled* returns True the worker's failure lines are the cancel itself:
    they go to the debug log only."""

    def __init__(self, say: Callable[[str], None],
                 cancelled: Optional[Callable[[], bool]] = None) -> None:
        self._say = say
        self._cancelled = cancelled
        self._pending: list[str] = []
        self._no_preview_said = False
        self._retry_mode: Optional[str] = None

    def _is_cancelled(self) -> bool:
        try:
            return bool(self._cancelled is not None and self._cancelled())
        except Exception:
            return False

    def _raw(self, text: str) -> None:
        self._say(f"stable-diffusion.cpp: {text}")

    def line(self, text: str) -> None:
        """Handle one WARN/ERROR line from the worker."""
        from localm.debuglog import logger
        logger.debug("sd.cpp: %s", text)
        if self._is_cancelled():
            self._pending.clear()
            return
        retry = _VAE_RETRY.search(text)
        if retry:
            smaller = any(_TILE_FAILED.search(p) for p in self._pending)
            self._pending.clear()
            mode = retry.group(1).lower()
            if self._retry_mode is not None and mode != self._retry_mode and not smaller:
                self._say(vae_escalation_message(mode))
            else:
                self._say(vae_retry_message(mode, smaller))
            self._retry_mode = mode
            return
        if _NO_PROJECTION.search(text):
            if not self._no_preview_said:
                self._no_preview_said = True
                self.flush()
                self._say(NO_PREVIEW)
            return
        if any(p.search(text) for p in _PRECURSORS):
            self._pending.append(text)
            return
        self.flush()
        self._raw(text)

    def sampling(self) -> None:
        """Sampling is running again: held lines were not part of a decode
        retry, so show them."""
        self._retry_mode = None
        self.flush()

    def budget_retry(self) -> None:
        """The generation is retried without its VRAM budget: drop the held
        lines of the refused attempt. Call ``flush`` when an attempt starts, so
        only that attempt's lines are held here."""
        self._pending.clear()
        self._retry_mode = None

    def flush(self) -> None:
        """Show every held line unchanged. Call when the job ends."""
        pending, self._pending = self._pending, []
        if self._is_cancelled():
            return
        for text in pending:
            self._raw(text)
