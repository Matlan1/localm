# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hang detection with recovery: the server must never be able to hang
undetected and unrecovered.

The pre-existing hang watchdog (http_server._start_hang_watchdog) watches ONLY
the event-loop heartbeat and only writes a stack-trace file when it fires, so a
wedge that leaves the loop healthy is neither detected nor surfaced.

This module is the alarm half. One off-loop daemon thread runs three detectors
and drives a staged pipeline:

  detect -> SURFACE (unmissable: native status window turns red + CRITICAL
  log) -> if still hung, RECOVER (auto-restart via the same path as the tray
  Restart button, hardened for a wedged process) -> if the condition clears
  before the restart stage, un-surface and say so.

The detectors, and the honesty boundary between them:

* LOOP FREEZE (detector A): the event-loop heartbeat gap. Zero false
  positives by construction - this codebase's own design rule is that
  nothing may block the loop at all, so a 10s+ gap is always a defect.
  Surfaces at hang_watchdog_threshold() (10s default), auto-restarts after
  LOCALM_HANG_RESTART_SECS (60s default) of CONTINUOUS freeze.
* TRANSPORT DEATH (detector C): a raw self-HTTP probe of /health on the
  actual bind address. Any response bytes count as alive (on a TLS bind the
  plaintext probe gets portmux's 308 redirect - still bytes, and the relay
  runs on the same loop, so a frozen loop fails this too). Catches the
  accept/portmux/socket layer dying in ways the in-process heartbeat cannot
  see. Surfaces after 4 consecutive failures, restarts after 8.
* REQUEST STARVATION (detector S): handlers wedged on something (a lock, an
  executor) while the loop itself stays healthy. Observes when requests have
  been in flight longer than LOCALM_HANG_STARVATION_SECS (600s default) while
  nothing at all makes response progress. LOG-ONLY FORENSICS: no threshold
  separates this state from legitimate long silent work with certainty (model
  downloads, non-streaming generations, the 300s ComfyUI launch), and anything
  user-facing must have zero false positives, so S never touches the window,
  never warns, never restarts. It records which requests (method + path + age),
  proof of total starvation, an immediate stack snapshot of every thread, and a
  follow-up snapshot one window later so identical blocked frames prove
  "genuinely wedged" over "merely idle".

Detection and surfacing are NOT privacy-gated: nothing here writes chat
content or paths anywhere - the surface text is generic, the CRITICAL line
goes through the normal logger, and a restart is an action, not a
disclosure. (The stack-dump FILE the pre-existing watchdog writes keeps its
own privacy gate, unchanged, in http_server.) LOCALM_HANG_RECOVERY=off
disables this module entirely; =surface keeps detection + surfacing but
never auto-restarts; =restart (the default) enables the full pipeline.

A restart storm is bounded without touching disk: each auto-restart appends
a timestamp to LOCALM_HANG_RESTART_HISTORY in this process's environment
immediately before the re-exec, and os.execv carries the environment into
the replacement process. More than 3 hang-restarts inside 30 minutes
downgrades further recovery to surface-only, so a hang that survives
restarts cannot restart-loop forever.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from typing import Callable, Optional, Tuple

from localm.debuglog import logger

_RECOVERY_ENV = "LOCALM_HANG_RECOVERY"
_RESTART_SECS_ENV = "LOCALM_HANG_RESTART_SECS"
_STARVATION_SECS_ENV = "LOCALM_HANG_STARVATION_SECS"
_PROBE_ENV = "LOCALM_HANG_PROBE"
_HISTORY_ENV = "LOCALM_HANG_RESTART_HISTORY"

_STORM_WINDOW_S = 1800.0
_STORM_LIMIT = 3


def recovery_mode() -> str:
    """"restart" (default: detect, surface, auto-restart), "surface" (detect
    and surface, never restart), or "off" (this module does not run)."""
    v = os.environ.get(_RECOVERY_ENV, "").strip().lower()
    return v if v in ("off", "surface", "restart") else "restart"


def restart_after_seconds() -> float:
    """How long a CONTINUOUS loop freeze must last before auto-restart. Floored
    well above the surface threshold, so the staged pipeline always surfaces
    first and a brief transient stall never restarts anyone."""
    try:
        return max(15.0, float(os.environ.get(_RESTART_SECS_ENV, "60")))
    except ValueError:
        return 60.0


def starvation_seconds() -> float:
    """Detector S's window. 0 disables it. The default sits WELL above every
    bounded long operation this codebase has on a request path (the ~330s
    comfy-launch worst case is the longest): S covers the unbounded wedge, not
    slow-but-finite work."""
    try:
        return max(0.0, float(os.environ.get(_STARVATION_SECS_ENV, "600")))
    except ValueError:
        return 600.0


class RequestProgress:
    """In-flight request bookkeeping for detector S, fed by the pure-ASGI
    middleware below. All numbers, no request content.

    The tiny lock is held only for dict updates and snapshot copies
    (microseconds); the loop-side cost is two locked dict operations per
    request, which cannot meaningfully stall the loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_id = 0
        # token -> (start_monotonic, "METHOD /path"). Method+path so a
        # starvation record can name WHICH requests are stuck (the same
        # granularity the debug request log already records); never bodies,
        # never query strings, never chat content.
        self._inflight: dict[int, tuple[float, str]] = {}
        self._last_progress = time.monotonic()

    def start(self, label: str = "?") -> int:
        with self._lock:
            self._next_id += 1
            token = self._next_id
            self._inflight[token] = (time.monotonic(), label)
            return token

    def progress(self) -> None:
        with self._lock:
            self._last_progress = time.monotonic()

    def finish(self, token: int) -> None:
        with self._lock:
            self._inflight.pop(token, None)
            self._last_progress = time.monotonic()

    def snapshot(self) -> Tuple[int, float, float]:
        """(in-flight count, oldest in-flight age seconds, seconds since the
        last response progress). Ages are 0.0 when nothing is in flight."""
        now = time.monotonic()
        with self._lock:
            count = len(self._inflight)
            oldest = (min(t for t, _ in self._inflight.values())
                      if self._inflight else now)
            return count, now - oldest, now - self._last_progress

    def observe(self) -> Tuple[Tuple[Tuple[float, str], ...], float]:
        """(per-request (age, "METHOD /path") pairs, seconds since the last
        response progress) - the starvation watch's view. Per-request, not
        just the oldest, so a record can name exactly which requests are
        stuck rather than claiming something vague about everything in
        flight."""
        now = time.monotonic()
        with self._lock:
            entries = tuple((now - t, label)
                            for t, label in self._inflight.values())
            return entries, now - self._last_progress


class RequestProgressMiddleware:
    """Pure ASGI: counts an http request in flight from arrival to completion
    and records every response body chunk as progress, so detector S can tell
    "slow but moving" from "nothing is answering at all".

    /health and /whoami are excluded: they are neither counted in flight nor
    recorded as progress. Everything else (pages, stats, activity, chat) stays
    tracked."""

    _EXCLUDED_PATHS = frozenset({"/health", "/whoami"})

    def __init__(self, app, tracker: Optional["RequestProgress"] = None) -> None:
        self.app = app
        # The process singleton by default; injectable so tests can use an
        # isolated tracker instead of sharing cross-test state.
        self.tracker: RequestProgress = tracker if tracker is not None else _TRACKER

    async def __call__(self, scope, receive, send):
        if (scope.get("type") != "http"
                or scope.get("path") in self._EXCLUDED_PATHS):
            await self.app(scope, receive, send)
            return
        token = self.tracker.start(
            f"{scope.get('method', '?')} {scope.get('path', '?')}")

        async def send_tracked(message):
            if message.get("type") == "http.response.body":
                self.tracker.progress()
            await send(message)

        try:
            await self.app(scope, receive, send_tracked)
        finally:
            self.tracker.finish(token)


# One process-wide tracker: the middleware writes it, the alarm thread reads
# it, and tests may reset it via fresh HangAlarm instances with their own.
_TRACKER = RequestProgress()


def tracker() -> RequestProgress:
    return _TRACKER


def _probe_once(host: str, port: int, timeout: float = 5.0) -> bool:
    """One end-to-end aliveness probe: TCP connect, minimal plaintext GET,
    any response byte within *timeout* counts as alive. The GET stays plaintext
    for a TLS bind too, where portmux's first-byte peek answers it with a 308
    redirect."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\n"
                      b"Connection: close\r\n\r\n")
            return bool(s.recv(1))
    except OSError:
        return False


def _probe_host(bind_host: Optional[str]) -> str:
    """The address the self-probe should dial for a given bind host: the
    wildcard binds are reachable on loopback; a specific address is only
    guaranteed reachable on itself.

    Delegates to ``bindhost.self_connect_host``. The result feeds
    ``socket.create_connection``, which takes a bare address, so it is NOT
    passed through ``url_host`` here - bracketing is for URL authorities only."""
    from localm.bindhost import self_connect_host
    return self_connect_host(bind_host)


def _restart_history() -> list[float]:
    out = []
    for part in os.environ.get(_HISTORY_ENV, "").split(","):
        part = part.strip()
        if part:
            try:
                out.append(float(part))
            except ValueError:
                continue
    return out


def _storm_active(now_epoch: float) -> bool:
    recent = [t for t in _restart_history() if now_epoch - t < _STORM_WINDOW_S]
    return len(recent) >= _STORM_LIMIT


def _record_restart(now_epoch: float) -> None:
    """Append to the in-environment restart history BEFORE the re-exec so the
    replacement process inherits it (os.execv passes the current environment
    through). Environment, not a file, so privacy mode's "nothing written
    automatically" promise is untouched."""
    hist = [t for t in _restart_history()
            if now_epoch - t < _STORM_WINDOW_S] + [now_epoch]
    os.environ[_HISTORY_ENV] = ",".join(f"{t:.0f}" for t in hist)


class HangAlarm:
    """The staged detect -> surface -> recover pipeline. Everything external
    is injected so tests can drive it with tiny thresholds and spy callbacks:

    * heartbeat_gap() -> seconds since the loop last ticked, or None before
      the first tick (no reading is never treated as a reading).
    * inflight() -> RequestProgress.snapshot() triple, or None to disable S.
    * probe_target() -> (host, port) to self-probe, or None while unknown
      (the port is only known once the server has advertised).
    * surface(text) / recovered() -> the unmissable human-facing state; in
      the GUI these drive the native status window (red text + un-hide).
    * restart(reason) -> the recovery action. Expected not to return (it
      re-execs the process); the alarm latches after calling it once.
    * dump(reason) -> optional extra forensics hook (the privacy-gated
      stack-trace file), called once per distinct incident.
    """

    def __init__(self, *,
                 heartbeat_gap: Callable[[], Optional[float]],
                 inflight: Optional[Callable[[], Tuple[int, float, float]]],
                 probe_target: Optional[Callable[[], Optional[Tuple[str, int]]]],
                 surface: Callable[[str], None],
                 recovered: Callable[[], None],
                 restart: Callable[[str], None],
                 dump: Optional[Callable[[str], None]] = None,
                 surface_after: float = 10.0,
                 restart_after: float = 60.0,
                 starvation_after: float = 300.0,
                 probe_interval: float = 30.0,
                 probe_timeout: float = 5.0,
                 probe_surface_fails: int = 4,
                 probe_restart_fails: int = 8,
                 allow_restart: bool = True,
                 poll: float = 1.0) -> None:
        self._hb_gap = heartbeat_gap
        self._inflight = inflight
        self._probe_target = probe_target
        self._surface = surface
        self._recovered = recovered
        self._restart = restart
        self._dump = dump
        self.surface_after = surface_after
        self.restart_after = max(restart_after, surface_after)
        self.starvation_after = starvation_after
        self.probe_interval = probe_interval
        self.probe_timeout = probe_timeout
        self.probe_surface_fails = probe_surface_fails
        self.probe_restart_fails = probe_restart_fails
        self.allow_restart = allow_restart
        self.poll = poll
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Detector state
        self._loop_suspect = False     # one prior sample already over threshold
        self._probe_due = 0.0
        self._probe_fails = 0
        self._active: dict[str, str] = {}   # detector -> surfaced text
        self._restart_latched = False
        self._dumped: set[str] = set()
        # Log-only starvation incident state (never enters _active).
        self._starve_since: Optional[float] = None
        self._starve_second_dump_due: Optional[float] = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "HangAlarm":
        t = threading.Thread(target=self._run, name="localm-hang-alarm",
                             daemon=True)
        self._thread = t
        t.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    # -- surfaced-state management ----------------------------------------

    def _set_active(self, detector: str, text: Optional[str]) -> None:
        """Transition one detector's surfaced state and re-render the union.
        Never raises: a broken surface hook must not kill the alarm."""
        was = dict(self._active)
        if text is None:
            self._active.pop(detector, None)
        else:
            self._active[detector] = text
        if self._active == was:
            return
        try:
            if self._active:
                joined = " / ".join(self._active[k] for k in sorted(self._active))
                logger.critical("HANG ALARM: %s", joined)
                self._surface(joined)
                if self._dump is not None and detector not in self._dumped and text:
                    self._dumped.add(detector)
                    try:
                        self._dump(joined)
                    except Exception:
                        logger.debug("hang-alarm dump hook failed", exc_info=True)
            else:
                logger.warning("hang alarm cleared: server is responding again")
                self._recovered()
                self._dumped.clear()
        except Exception:
            logger.debug("hang-alarm surface hook failed", exc_info=True)

    def _maybe_restart(self, reason: str) -> None:
        if self._restart_latched:
            return
        if not self.allow_restart:
            return
        now_epoch = time.time()
        if _storm_active(now_epoch):
            # Keep the surface up but refuse to loop: a hang that survives
            # restarts will not be fixed by a fourth one.
            self._set_active(
                "storm",
                "auto-restart suppressed (%d recent hang restarts) - restart "
                "manually once the cause is fixed" % len(_restart_history()))
            return
        self._restart_latched = True
        _record_restart(now_epoch)
        logger.critical(
            "HANG ALARM: auto-restarting the server (%s). Set %s=surface to "
            "disable auto-restart.", reason, _RECOVERY_ENV)
        try:
            self._restart(reason)
        except Exception:
            # The restart action re-execs and should never return, so any
            # exception means recovery itself failed - the loudest state we
            # have left is the surface.
            logger.critical("hang-alarm restart action FAILED", exc_info=True)
            self._set_active("restart-failed",
                             "automatic restart failed - restart manually")

    # -- detectors ---------------------------------------------------------

    def _check_loop(self) -> None:
        gap = self._hb_gap()
        if gap is None or gap < self.surface_after:
            self._loop_suspect = False
            self._set_active("loop", None)
            return
        if not self._loop_suspect:
            # Require two consecutive over-threshold samples before acting:
            # after a system sleep/resume both this thread and the loop wake
            # together, and a single stale-looking gap sampled before the
            # loop's first post-resume tick is not a freeze. A real freeze
            # holds the gap across every later sample.
            self._loop_suspect = True
            return
        self._set_active(
            "loop", "server event loop frozen for %.0fs" % gap)
        if gap >= self.restart_after:
            self._maybe_restart("event loop frozen %.0fs" % gap)

    def _check_probe(self, now: float) -> None:
        if self._probe_target is None or now < self._probe_due:
            return
        self._probe_due = now + self.probe_interval
        target = None
        try:
            target = self._probe_target()
        except Exception:
            logger.debug("hang-alarm probe target lookup failed", exc_info=True)
        if target is None:
            return   # not listening yet (pre-advertise): nothing to measure
        host, port = target
        if _probe_once(host, port, self.probe_timeout):
            self._probe_fails = 0
            self._set_active("probe", None)
            return
        self._probe_fails += 1
        if self._probe_fails >= self.probe_surface_fails:
            self._set_active(
                "probe",
                "server is not answering its own port (%d probes failed)"
                % self._probe_fails)
        if self._probe_fails >= self.probe_restart_fails:
            self._maybe_restart(
                "%d consecutive self-probe failures" % self._probe_fails)

    def _check_starvation(self) -> None:
        """LOG-ONLY FORENSICS - never a user-facing warning, never a restart
        trigger: anything shown to a user must have zero false positives, and
        no threshold can separate "requests wedged by a defect" from
        "legitimately slow work" with certainty (a model download, an hour-long
        non-streaming generation, and the 300s ComfyUI launch are all real and
        all silent). The zero-false-positive detectors (loop freeze, transport
        death) own the user-facing alarms and restarts. This one writes to the
        log: WHICH requests (method + path + age), the proof they were starved
        (nothing at all completing), a stack snapshot from the moment it began
        showing WHERE every thread was blocked, a second snapshot one window
        later (identical blocked frames = genuinely wedged, not idle), and what
        to do about it."""
        if self._inflight is None or self.starvation_after <= 0:
            return
        entries, progress_age = self._inflight()
        stuck = sorted((e for e in entries if e[0] > self.starvation_after),
                       reverse=True)
        if self._starve_since is not None:
            if not stuck:
                logger.warning(
                    "request-starvation watch cleared: the stuck requests "
                    "finished after %.1f minutes",
                    (time.monotonic() - self._starve_since) / 60.0)
                self._starve_since = None
                self._starve_second_dump_due = None
            elif (self._starve_second_dump_due is not None
                    and time.monotonic() >= self._starve_second_dump_due):
                # The follow-up snapshot: compared with the first, identical
                # blocked frames turn "might be stuck" into "provably has
                # not moved for a full window".
                self._starve_second_dump_due = None
                self._try_dump("request starvation follow-up - compare with "
                               "the first snapshot: identical frames for the "
                               "listed requests mean they are genuinely "
                               "wedged, not idle")
            return
        if stuck and progress_age > self.starvation_after:
            self._starve_since = time.monotonic()
            self._starve_second_dump_due = (
                self._starve_since + self.starvation_after)
            listed = ", ".join(
                f"{label} (in flight {age / 60.0:.1f} min)"
                for age, label in stuck[:8])
            if len(stuck) > 8:
                listed += f", +{len(stuck) - 8} more"
            logger.critical(
                "%d request(s) stuck with nothing completing for over %.1f "
                "minutes: %s. Writing a stack snapshot now and again in %.1f "
                "minutes - identical blocked frames across the two mean these "
                "are genuinely wedged. If the app is unusable, restart the "
                "server and include the snapshots in a bug report. (log-only "
                "diagnostic, ADR-0012)",
                len(stuck), progress_age / 60.0, listed,
                self.starvation_after / 60.0)
            self._try_dump("request starvation: " + listed)

    def _try_dump(self, reason: str) -> None:
        if self._dump is None:
            return
        try:
            self._dump(reason)
        except Exception:
            logger.debug("hang-alarm dump hook failed", exc_info=True)

    # -- main loop ---------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.wait(self.poll):
            now = time.monotonic()
            try:
                self._check_loop()
                self._check_probe(now)
                self._check_starvation()
            except Exception:
                # The alarm must never crash out from under the server it
                # watches - a dead watchdog is indistinguishable from a
                # healthy quiet one (same lesson as the cold-start guard in
                # _start_hang_watchdog).
                logger.debug("hang-alarm cycle failed (continuing)",
                             exc_info=True)
