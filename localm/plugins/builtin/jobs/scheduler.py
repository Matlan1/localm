# SPDX-License-Identifier: AGPL-3.0-or-later
"""The job scheduler and a self-contained 5-field cron matcher.

The matcher and the due/tick logic are PURE and clock-injectable, so they run
without real time or a running event loop:

  - ``cron_match(expr, when)``    -> bool   (no I/O, no global state)
  - ``JobScheduler.due(job, now)``-> bool   (interval vs cron, given a clock)
  - ``JobScheduler.tick(now=...)``         (runs due+enabled jobs once)

``start()`` / ``stop()`` drive ``tick`` from an asyncio loop that sleeps between
wakeups (never a busy-loop). A disabled or not-due job is never run.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Optional

from localm.plugins.builtin.jobs.store import Job, JobStore

# How often the loop wakes to look for due jobs.
DEFAULT_POLL_SECONDS = 30

# How far back a restarted server will catch up a missed cron slot. A slot
# older than this is not back-fired. One day.
_CATCHUP_WINDOW_SECONDS = 24 * 60 * 60


# --------------------------------------------------------------------------- #
#  Cron matcher (minute hour day-of-month month day-of-week)                  #
# --------------------------------------------------------------------------- #
# Field ranges (inclusive). day-of-week: 0-6 with 0 = Sunday; 7 is also Sunday.
_FIELD_BOUNDS = (
    (0, 59),    # minute
    (0, 23),    # hour
    (1, 31),    # day of month
    (1, 12),    # month
    (0, 6),     # day of week (0 = Sunday)
)


def _parse_field(field: str, lo: int, hi: int) -> set:
    """Expand one cron field into the set of matching integers.

    Supports ``*``, single values, comma lists (``a,b``), ranges (``a-b``), and
    steps (``*/n`` or ``a-b/n`` or ``a/n``). Raises ValueError on a malformed
    field or an out-of-range value."""
    values: set = set()
    field = field.strip()
    if field == "":
        raise ValueError("empty cron field")
    for part in field.split(","):
        part = part.strip()
        if not part:
            raise ValueError("empty cron list element")
        step = 1
        had_step = "/" in part
        if had_step:
            base, _, step_s = part.partition("/")
            if not step_s.isdigit() or int(step_s) < 1:
                raise ValueError(f"invalid step in cron field: {part!r}")
            step = int(step_s)
            part = base.strip()
        if part == "*" or part == "":
            start, end = lo, hi
        elif "-" in part:
            a, _, b = part.partition("-")
            if not (a.strip().lstrip("-").isdigit() and b.strip().lstrip("-").isdigit()):
                raise ValueError(f"invalid range in cron field: {part!r}")
            start, end = int(a), int(b)
        else:
            if not part.lstrip("-").isdigit():
                raise ValueError(f"invalid value in cron field: {part!r}")
            # A bare value with a step means "from this value to the field max,
            # every n" (Vixie cron: 0/15 == 0,15,30,45). Without a step it is a
            # single value.
            start = int(part)
            end = hi if had_step else start
        if start > end:
            raise ValueError(f"range start > end in cron field: {part!r}")
        for v in range(start, end + 1, step):
            if v < lo or v > hi:
                raise ValueError(f"value {v} out of range [{lo},{hi}] in cron field")
            values.add(v)
    return values


def parse_cron(expr: str) -> list:
    """Parse a 5-field cron expression into a list of 5 value-sets. Raises
    ValueError when the expression does not have exactly 5 fields or a field is
    malformed."""
    if not isinstance(expr, str):
        raise ValueError("cron expression must be a string")
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(
            f"cron expression must have 5 fields (minute hour dom month dow), "
            f"got {len(fields)}: {expr!r}")
    sets = []
    for idx, (raw, (lo, hi)) in enumerate(zip(fields, _FIELD_BOUNDS)):
        if idx == 4:
            # day-of-week: accept 7 as an alias for Sunday (Vixie cron), then
            # normalise to 0 so cron_match (which computes dow 0..6) still works.
            dow = _parse_field(raw, 0, 7)
            sets.append({0 if v == 7 else v for v in dow})
        else:
            sets.append(_parse_field(raw, lo, hi))
    return sets


def validate_cron(expr: str) -> None:
    """Raise ValueError if *expr* is not a valid 5-field cron expression."""
    parse_cron(expr)


def cron_match(expr: str, when: Optional[float] = None) -> bool:
    """True if the cron *expr* fires at the wall-clock time *when* (epoch
    seconds; defaults to now). Pure: no global state, no side effects.

    Day-of-month and day-of-week are matched cron-style: if BOTH are restricted
    (neither is ``*``), the job fires when EITHER matches; otherwise both must
    match. (This mirrors Vixie cron.)"""
    if when is None:
        when = time.time()
    minute_s, hour_s, dom_s, month_s, dow_s = parse_cron(expr)
    lt = time.localtime(when)
    # Python weekday(): Monday=0..Sunday=6; cron dow: Sunday=0..Saturday=6.
    cron_dow = (lt.tm_wday + 1) % 7
    if lt.tm_min not in minute_s:
        return False
    if lt.tm_hour not in hour_s:
        return False
    if lt.tm_mon not in month_s:
        return False
    fields = expr.split()
    dom_restricted = fields[2].strip() != "*"
    dow_restricted = fields[4].strip() != "*"
    dom_ok = lt.tm_mday in dom_s
    dow_ok = cron_dow in dow_s
    if dom_restricted and dow_restricted:
        return dom_ok or dow_ok
    return dom_ok and dow_ok


# --------------------------------------------------------------------------- #
#  Scheduler                                                                   #
# --------------------------------------------------------------------------- #

class JobScheduler:
    """Runs enabled + due jobs on a periodic asyncio loop.

    The decision logic is pure and clock-injectable (``due`` / ``tick`` take a
    ``now``). ``start`` spawns the loop on the running event loop; ``stop``
    cancels it. A disabled or not-due job is never run; a job that errors is
    recorded and never crashes the tick.
    """

    def __init__(self, store: Optional[JobStore] = None, *,
                 run_job: Optional[Callable] = None,
                 engine: Optional[Callable] = None,
                 poll_seconds: int = DEFAULT_POLL_SECONDS) -> None:
        self.store = store or JobStore()
        # run_job is injectable; defaults to the real runner.
        self._run_job = run_job
        # engine resolver: a zero-arg callable returning the inference engine
        # (or None to let the runner load one via the model manager).
        self._engine = engine
        self.poll_seconds = max(1, int(poll_seconds))
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event() if False else None   # set lazily in start()
        self._running = False
        # Track interval cron firings within the same minute so a 30s poll does
        # not run a cron job twice in its matching minute: job id -> last fired
        # cron-minute epoch (floor to 60s).
        self._cron_fired: dict = {}
        # Log-once/on-change state for a FAILING tick: the signature of the last
        # logged failure, so an identical one is not re-logged every poll.
        self._last_tick_error: Optional[str] = None

    # ---- pure decision logic ----------------------------------------------
    def due(self, job: Job, now: float) -> bool:
        """True if *job* should run at *now* (epoch seconds). Does NOT consider
        ``enabled``; the caller filters that."""
        if job.schedule_kind == "interval":
            try:
                interval = int(job.schedule)
            except (TypeError, ValueError):
                return False
            if job.last_run is None:
                return True
            return (now - job.last_run) >= interval
        if job.schedule_kind == "cron":
            minute = int(now // 60) * 60
            already = self._cron_fired.get(job.id) == minute
            try:
                matches_now = cron_match(str(job.schedule), now)
            except ValueError:
                return False
            if matches_now:
                # Avoid double-firing within the same matching minute on a
                # sub-minute poll: only fire once per cron-minute per job.
                return not already
            # Catch-up: a slot that came due while the server was DOWN fires
            # ONCE on the next tick, for any missed matching minute strictly
            # after last_run. Only for jobs that have run before, so a
            # freshly-created cron job never back-fires history, and bounded by
            # _CATCHUP_WINDOW_SECONDS. record_result stamps last_run, so the
            # caught-up slot is not re-fired next tick.
            if job.last_run is None or already:
                return False
            return self._missed_cron_slot(job, now)
        return False

    def _missed_cron_slot(self, job: Job, now: float) -> bool:
        """True if a cron-matching minute lies in ``(last_run, now)`` within the
        bounded catch-up window. Scans minute-by-minute from just after now back
        to the window floor, short-circuiting on the first match."""
        try:
            expr = str(job.schedule)
            floor = max(float(job.last_run), now - _CATCHUP_WINDOW_SECONDS)
            m = int(now // 60) * 60 - 60          # the minute before "now"
            while m > floor:
                if cron_match(expr, m):
                    return True
                m -= 60
        except (ValueError, TypeError):
            return False
        return False

    def _resolve_run(self) -> Callable:
        if self._run_job is not None:
            return self._run_job
        from localm.plugins.builtin.jobs.runner import run_job as _run
        return _run

    def tick(self, now: Optional[float] = None) -> list:
        """Run every enabled + due job once. Pure-ish: side effects are limited
        to running each due job and recording its result. Returns the list of
        job ids that ran this tick. Never raises out (a per-job error is caught
        and recorded).

        Overlap guard: if a previous run (this scheduler or a GUI "run now") is
        still in flight, the tick SKIPS its due jobs rather than stacking a second
        model load. Skipped jobs are due again next tick - an interval job runs as
        soon as the slow run finishes, and a cron job is not marked fired, so its
        minute is not consumed."""
        if now is None:
            now = time.time()
        current = self.store.list()
        # Prune _cron_fired for jobs that no longer exist.
        if self._cron_fired:
            live = {j.id for j in current}
            for jid in [k for k in self._cron_fired if k not in live]:
                del self._cron_fired[jid]
        due_jobs = [j for j in current if j.enabled and self.due(j, now)]
        if not due_jobs:
            return []

        from localm.plugins.builtin.jobs.runguard import run_slot

        ran: list = []
        run_job = self._resolve_run()
        engine = self._engine() if callable(self._engine) else self._engine
        with run_slot() as got_slot:
            if not got_slot:
                from localm.debuglog import logger
                logger.info(
                    "jobs: skipping %d due job(s) this tick - a previous job run is "
                    "still in progress (avoids stacking model loads / VRAM OOM)",
                    len(due_jobs))
                return ran
            for job in due_jobs:
                if job.schedule_kind == "cron":
                    self._cron_fired[job.id] = int(now // 60) * 60
                try:
                    result = run_job(job, engine=engine)
                except Exception as e:    # the runner shouldn't raise, but guard
                    result = {"status": "error", "output": "", "error": str(e),
                              "started": now, "finished": time.time()}
                try:
                    self.store.record_result(job.id, result)
                except Exception as e:
                    # Recording must never crash the scheduler loop; the
                    # persistence failure is logged at debug instead.
                    from localm.debuglog import logger
                    logger.debug(
                        "jobs scheduler: could not record result for %s: %s",
                        job.id, e)
                ran.append(job.id)
        return ran

    # ---- async loop --------------------------------------------------------
    def _note_tick_error(self, exc: BaseException) -> None:
        """Log a failing tick ONCE per distinct error, not every poll. A
        persistent failure (an unreadable jobs.json makes ``store.list()`` raise
        every tick and halts all scheduled jobs) is logged at WARNING the first
        time and on every change of signature."""
        signature = f"{type(exc).__name__}: {exc}"
        if signature == self._last_tick_error:
            return
        self._last_tick_error = signature
        from localm.debuglog import logger
        logger.warning(
            "jobs scheduler: tick failed - scheduled jobs will not fire until this "
            "clears: %s", signature)

    def _note_tick_ok(self) -> None:
        """After a tick succeeds, clear the failure state (noting recovery once) so
        a later recurrence is logged again rather than suppressed as a duplicate."""
        if self._last_tick_error is None:
            return
        self._last_tick_error = None
        from localm.debuglog import logger
        logger.info(
            "jobs scheduler: tick recovered; scheduled jobs are firing again")

    async def _loop(self) -> None:
        assert self._stop is not None
        while not self._stop.is_set():
            try:
                # Run the (blocking) tick off the event loop so a slow job run
                # never stalls the loop or the server.
                await asyncio.get_running_loop().run_in_executor(None, self.tick)
            except Exception as e:
                # A tick must never kill the loop; the failure is logged
                # once/on-change instead.
                self._note_tick_error(e)
            else:
                self._note_tick_ok()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_seconds)
            except asyncio.TimeoutError:
                pass        # normal wakeup; loop again

    def start(self) -> bool:
        """Start the scheduler loop on the running event loop. Returns False
        (no-op) when there is no running loop (e.g. a sync test or headless
        CLI) or it is already running, so callers can guard safely."""
        if self._running:
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False        # no event loop: caller is not under a server
        self._stop = asyncio.Event()
        self._task = loop.create_task(self._loop())
        self._running = True
        return True

    def stop(self) -> None:
        """Signal the loop to stop and cancel its task. Safe to call when not
        running."""
        if self._stop is not None:
            self._stop.set()
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running
