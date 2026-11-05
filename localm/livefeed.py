# SPDX-License-Identifier: AGPL-3.0-or-later
"""One sampler that pushes hardware stats to every connected GUI tab.

The sampler is a single asyncio task per process, alive only while at least one
tab is connected, so the GPU is probed once per interval however many tabs are
open and not at all when none is. Each tick it takes a reading, keeps it when
the values a tab would DISPLAY changed (:func:`display_key`), and wakes every
connection to send it.

:func:`kick` is the other way a reading is taken: any thread calls it when
something changed what the meter shows (a model loaded or unloaded, a job
started or ended). The sampler then takes a forced reading at once, bypassing
the VRAM refresh throttle, and again at each of :data:`FOLLOW_UPS_S` because a
driver releases memory after the unload call returns. With no tab connected a
kick does nothing: the next tab to connect starts with a forced reading.

A connection reads the feed's state (:attr:`LiveFeed.seq`, :attr:`LiveFeed.latest`)
after each wake instead of queueing frames, so a slow client sees the latest
value and never a backlog.
"""

from __future__ import annotations

import asyncio
import math
from typing import Awaitable, Callable, Optional

INTERVAL_S = 2.5
FOLLOW_UPS_S = (0.75, 2.0, 4.0)

_GIB = 1024 ** 3


def _num(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _round_half_up(value) -> Optional[int]:
    n = _num(value)
    return None if n is None else math.floor(n + 0.5)


def _gib(value) -> Optional[str]:
    n = _num(value)
    return None if n is None else f"{n / _GIB:.1f}"


def _band(used, total) -> Optional[str]:
    u, t = _num(used), _num(total)
    if u is None or not t:
        return None
    frac = u / t
    return "full" if frac >= 0.9 else "busy" if frac >= 0.7 else "ok"


def display_key(stats: dict) -> tuple:
    """The part of *stats* the status bar shows: CPU and GPU percent as whole
    numbers, RAM and VRAM as GB to one decimal, and the VRAM fullness band, for
    the board and for each card of a multi-card board. Two readings with an
    equal key render identically, so only a change of key is worth a frame."""
    cpu = stats.get("cpu") or {}
    ram = stats.get("ram") or {}
    gpu = stats.get("gpu") or {}
    vram = stats.get("vram") or {}
    devices = tuple(
        (d.get("index"), _gib(d.get("used")), _gib(d.get("total")),
         _band(d.get("used"), d.get("total")))
        for d in (vram.get("devices") or ()) if isinstance(d, dict))
    return (
        _round_half_up(cpu.get("percent")),
        _gib(ram.get("used")), _gib(ram.get("total")),
        _round_half_up(ram.get("percent")),
        _round_half_up(gpu.get("percent")),
        _gib(vram.get("used")), _gib(vram.get("total")),
        _band(vram.get("used"), vram.get("total")),
        devices,
    )


class Subscription:
    """One connection's handle on the feed. Wakes when the feed has something
    new; the connection then reads the feed's state."""

    def __init__(self, feed: "LiveFeed") -> None:
        self._feed = feed
        self._event = asyncio.Event()
        self.seen_seq = 0

    async def wait(self, timeout: float) -> bool:
        """Block until the feed wakes this connection or *timeout* seconds pass.
        True when woken."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout)
            woke = True
        except TimeoutError:
            woke = False
        self._event.clear()
        return woke

    def wake(self) -> None:
        self._event.set()

    def close(self) -> None:
        self._feed._unsubscribe(self)


class LiveFeed:
    """The sampler and its subscribers. *sample* is ``async (fresh: bool) ->
    dict``: the current stats, taken now when *fresh* (a forced reading) and
    otherwise subject to the usual refresh throttle."""

    def __init__(self, sample: Callable[[bool], Awaitable[dict]], *,
                 interval_s: float = INTERVAL_S,
                 follow_ups_s: tuple = FOLLOW_UPS_S) -> None:
        self._sample = sample
        self._interval_s = interval_s
        self._follow_up_delays = tuple(follow_ups_s)
        self._subs: set[Subscription] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._kicked: Optional[asyncio.Event] = None
        self._follow_ups: list[float] = []
        self._key: Optional[tuple] = None
        self._latest: Optional[dict] = None
        self._seq = 0
        self._failing = False
        self.samples = 0
        self.forced_samples = 0

    @property
    def seq(self) -> int:
        """Counts the readings whose displayed values differ from the one
        before; 0 until the first reading lands."""
        return self._seq

    @property
    def latest(self) -> Optional[dict]:
        return self._latest

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)

    def subscribe(self) -> Subscription:
        """Register a connection and start the sampler if it is not running.
        Must be called on the event loop that will serve the connection."""
        sub = Subscription(self)
        self._subs.add(sub)
        if self._task is None or self._task.done():
            self._loop = asyncio.get_running_loop()
            self._kicked = asyncio.Event()
            self._follow_ups = []
            self._latest = None
            self._key = None
            self._task = self._loop.create_task(self._run(), name="localm-live-feed")
        return sub

    def _unsubscribe(self, sub: Subscription) -> None:
        self._subs.discard(sub)
        if not self._subs and self._task is not None:
            self._task.cancel()
            self._task = None
            self._loop = None

    def kick(self, reason: str) -> bool:
        """Ask for a forced reading now and the follow-ups after it. Safe from
        any thread. False when no tab is connected, so nobody is waiting for
        one."""
        loop = self._loop
        if loop is None:
            return False
        try:
            loop.call_soon_threadsafe(self._kick_on_loop, reason)
        except RuntimeError:
            return False
        return True

    def _kick_on_loop(self, reason: str) -> None:
        if not self._subs or self._kicked is None:
            return
        now = asyncio.get_running_loop().time()
        self._follow_ups = [now + d for d in self._follow_up_delays]
        self._kicked.set()

    async def _run(self) -> None:
        forced = True
        while True:
            try:
                stats = await self._sample(forced)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._note_failure(exc)
                stats = None
            else:
                self._failing = False
            self.samples += 1
            if forced:
                self.forced_samples += 1
            if stats is not None:
                self._publish(stats)
            for sub in list(self._subs):
                sub.wake()
            forced = await self._pause()

    def _note_failure(self, exc: Exception) -> None:
        from localm.debuglog import logger
        if self._failing:
            logger.debug("live stats: sampling still failing: %s", exc)
            return
        self._failing = True
        logger.warning("live stats: sampling failed (%s); the status bar keeps "
                       "its last reading until sampling recovers", exc)

    def _publish(self, stats: dict) -> None:
        key = display_key(stats)
        if self._seq and key == self._key:
            return
        self._key = key
        self._latest = stats
        self._seq += 1

    async def _pause(self) -> bool:
        """Wait for the next reading; True when it is to be forced (a kick, or
        a follow-up coming due), False for a routine tick."""
        loop = asyncio.get_running_loop()
        now = loop.time()
        self._follow_ups = [t for t in self._follow_ups if t > now]
        due_follow_up = bool(self._follow_ups) and self._follow_ups[0] - now < self._interval_s
        delay = self._follow_ups[0] - now if due_follow_up else self._interval_s
        if not self._kicked.is_set():
            try:
                await asyncio.wait_for(self._kicked.wait(), delay)
            except TimeoutError:
                pass
        if self._kicked.is_set():
            self._kicked.clear()
            return True
        if due_follow_up and self._follow_ups:
            self._follow_ups.pop(0)
            return True
        return False


_feed: Optional[LiveFeed] = None


def install(feed: Optional[LiveFeed]) -> None:
    """Make *feed* the one :func:`kick` reaches."""
    global _feed
    _feed = feed


def kick(reason: str) -> bool:
    """Tell the installed feed that something changed what the status bar
    shows. Safe from any thread; False when there is no feed or no tab."""
    feed = _feed
    return feed.kick(reason) if feed is not None else False
