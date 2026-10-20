# SPDX-License-Identifier: AGPL-3.0-or-later
"""Byte counters for a running Hugging Face download.

huggingface_hub reports two byte streams through the ``tqdm_class`` it is
given: bytes received from the network (``update_transfer``) and bytes written
to the destination file (``update``). A Xet download writes its file in bursts
once a contiguous run of chunks is complete, so the size of the file on disk
can stay flat for seconds while data arrives; the network stream is the
steady one. :data:`METER` counts both, and :func:`meter_tqdm_class` is the
``tqdm_class`` that feeds it. The bars draw nothing.

This module imports nothing from the package.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

# Seconds of history a transfer rate is averaged over.
RATE_WINDOW_S = 6.0

# Seconds without a byte after which a download is reported as stalled.
STALL_AFTER_S = 5.0


class RateWindow:
    """Bytes per second over the last :data:`RATE_WINDOW_S` seconds of a
    cumulative byte count. Thread-safe."""

    def __init__(self, window: float = RATE_WINDOW_S) -> None:
        self._window = window
        self._lock = threading.Lock()
        self._samples: deque[tuple[float, int]] = deque()

    def add(self, total: int, now: Optional[float] = None) -> None:
        """Record that the cumulative count is *total* at time *now*."""
        t = time.monotonic() if now is None else now
        with self._lock:
            self._samples.append((t, total))
            self._trim(t)

    def _trim(self, now: float) -> None:
        cutoff = now - self._window
        while len(self._samples) > 2 and self._samples[1][0] <= cutoff:
            self._samples.popleft()

    def rate(self, now: Optional[float] = None) -> Optional[float]:
        """Bytes per second, 0.0 when nothing arrived in the window, or None
        before two samples exist."""
        t = time.monotonic() if now is None else now
        with self._lock:
            self._trim(t)
            if len(self._samples) < 2:
                return None
            t0, b0 = self._samples[0]
            t1, b1 = self._samples[-1]
            span = max(t, t1) - t0
            if span <= 0:
                return None
            return max(0.0, (b1 - b0) / span)


class TransferMeter:
    """Counts the bytes of the download running in this process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rate = RateWindow()
        self._transferred = 0
        self._written = 0
        self._last_activity: Optional[float] = None

    def reset(self) -> None:
        """Forget every count; call before a download starts."""
        with self._lock:
            self._rate = RateWindow()
            self._transferred = 0
            self._written = 0
            self._last_activity = None

    def add_transferred(self, n: int) -> None:
        """Count *n* bytes received from the network (negative to take back
        bytes that were discarded)."""
        if not n:
            return
        now = time.monotonic()
        with self._lock:
            self._transferred = max(0, self._transferred + int(n))
            if n > 0:
                self._last_activity = now
            total = self._transferred
            rate = self._rate
        rate.add(total, now)

    def add_written(self, n: int) -> None:
        """Count *n* bytes written to the destination file."""
        if not n:
            return
        with self._lock:
            self._written = max(0, self._written + int(n))

    @property
    def transferred(self) -> int:
        """Bytes received from the network since :meth:`reset`."""
        with self._lock:
            return self._transferred

    @property
    def written(self) -> int:
        """Bytes written to the file since :meth:`reset`."""
        with self._lock:
            return self._written

    def rate(self) -> Optional[float]:
        """Network bytes per second, or None when no byte has been counted."""
        with self._lock:
            if self._last_activity is None:
                return None
            rate = self._rate
        return rate.rate()

    def seconds_idle(self) -> Optional[float]:
        """Seconds since the last network byte, or None when none arrived."""
        with self._lock:
            if self._last_activity is None:
                return None
            return time.monotonic() - self._last_activity


METER = TransferMeter()


class MeterBar:
    """A progress bar for huggingface_hub that draws nothing and counts into
    :data:`METER`.

    A bar created by ``snapshot_download`` for its network total is told apart
    by its description (``Downloading bytes``): its ``update`` carries network
    bytes. For every other byte bar ``update`` carries bytes written and
    ``update_transfer`` carries network bytes. A bar whose unit is not bytes
    (a count of files) is ignored.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.n = kwargs.get("initial") or 0
        self.total = kwargs.get("total")
        desc = str(kwargs.get("desc") or "")
        if kwargs.get("unit", "it") != "B":
            self._role = "ignore"
        elif desc.startswith("Downloading bytes"):
            self._role = "transfer"
        else:
            self._role = "written"

    def __enter__(self) -> MeterBar:
        return self

    def __exit__(self, *exc) -> None:
        return None

    def update(self, n=1) -> None:
        n = int(n or 0)
        self.n += n
        if self._role == "transfer":
            METER.add_transferred(n)
        elif self._role == "written":
            METER.add_written(n)

    def update_transfer(self, n=1) -> None:
        if self._role != "ignore":
            METER.add_transferred(int(n or 0))

    def set_postfix_str(self, *args, **kwargs) -> None:
        return None

    def set_transfer_postfix_str(self, *args, **kwargs) -> None:
        return None

    def set_description(self, *args, **kwargs) -> None:
        return None

    def refresh(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def meter_tqdm_class() -> type:
    """The ``tqdm_class`` to hand to ``hf_hub_download`` / ``snapshot_download``
    so the download's byte counts reach :data:`METER`."""
    return MeterBar


def format_rate(bytes_per_sec: Optional[float]) -> str:
    """``'12.3 MB/s'``; ``'--'`` when unknown."""
    if bytes_per_sec is None:
        return "--"
    value = float(bytes_per_sec)
    for unit in ("B/s", "KB/s", "MB/s", "GB/s"):
        if value < 1000 or unit == "GB/s":
            return f"{value:.0f} {unit}" if unit == "B/s" else f"{value:.1f} {unit}"
        value /= 1000.0
    return "--"


def format_duration(seconds: Optional[float]) -> str:
    """``'1h 02m'`` / ``'3m 05s'`` / ``'42s'``; ``'--'`` when unknown."""
    if seconds is None or seconds < 0 or seconds != seconds or seconds == float("inf"):
        return "--"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def format_size(n: float) -> str:
    """``'491.4 MB'`` (decimal units, as the hub and the GUI show them)."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000.0
    return f"{n} B"


def eta_seconds(downloaded: int, total: int, rate: Optional[float]) -> Optional[int]:
    """Seconds left at *rate*, or None when the total or the rate is unknown or
    the rate is zero."""
    if not total or not rate or rate <= 0 or downloaded > total:
        return None
    return int(round((total - downloaded) / rate))


def describe(downloaded: int, total: int, rate: Optional[float],
             label: str = "") -> str:
    """One line a person can read: ``'name: 34% (167.1 MB / 491.4 MB) at 22.0
    MB/s, 15s left'``."""
    head = f"{label}: " if label else ""
    if total:
        pct = min(100.0, downloaded * 100.0 / total)
        text = f"{head}{pct:.0f}% ({format_size(downloaded)} / {format_size(total)})"
    else:
        text = f"{head}{format_size(downloaded)}"
    if rate is not None:
        text += f" at {format_rate(rate)}"
        eta = eta_seconds(downloaded, total, rate)
        if eta is not None:
            text += f", {format_duration(eta)} left"
    return text
