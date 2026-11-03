# SPDX-License-Identifier: AGPL-3.0-or-later
"""Detect a server-wide stall and keep a small in-memory record of where it was.

Two kinds of stall stop the server from answering:

* GIL: one thread holds the interpreter lock inside a long C call (a large
  ``json.dumps``, a native library called without releasing it). No Python
  thread runs, so a Python watchdog thread cannot see it until it is over.
* LOOP: the event loop thread runs one long piece of Python code, or waits,
  while other threads keep running.

A keeper thread re-arms ``faulthandler.dump_traceback_later`` every *poll*
seconds. The dump runs on faulthandler's own native thread, which needs no
GIL, so it fires only when the keeper could not re-arm it for *threshold*
seconds: a GIL stall. The dump goes to an in-memory sink (an anonymous pipe on
Windows, an unnamed temporary file elsewhere) and is parsed when the keeper
runs again. A LOOP stall is seen by the keeper itself, from the heartbeat
gap, and the loop thread's frames are read with ``sys._current_frames()``.

Each record holds the stall's length, its kind and code locations only
(module-relative file paths, line numbers and function names), never a
variable, an argument or an absolute path. Nothing is written to disk unless
the caller passes a *trace_sink*, which receives the raw dump text.
"""

from __future__ import annotations

import collections
import faulthandler
import itertools
import os
import re
import sys
import tempfile
import threading
import time
from typing import Callable, Optional

from localm.debuglog import logger

#: Bytes the Windows pipe buffers without a reader; a full dump fits.
_SINK_BYTES = 4 << 20

#: Records kept.
_KEEP = 8

#: Frames kept per record, innermost first.
_FRAMES_KEPT = 12

_THREAD_RE = re.compile(r"^(?:Current thread|Thread) (0x[0-9a-fA-F]+)")
_FRAME_RE = re.compile(r'^\s+File "(.*)", line (\d+) in (.*)$')

#: (file name, function) of stdlib frames a thread sits in while it waits with
#: the GIL released.
_WAIT_SITES = frozenset({
    ("threading.py", "wait"), ("threading.py", "_wait_for_tstate_lock"),
    ("queue.py", "get"), ("selectors.py", "select"),
    ("windows_events.py", "_poll"), ("windows_events.py", "select"),
    ("thread.py", "_worker"),
})

_ids = itertools.count(1)


def short_path(path: str) -> str:
    """*path* without the part that names this machine: from the last
    ``localm/``, ``plugins/``, ``site-packages/`` or standard-library
    directory on, else the file name alone."""
    p = path.replace("\\", "/")
    best = None
    for marker, keep in (("/site-packages/", False), ("/Lib/", False),
                         ("/localm/", True), ("/plugins/", True)):
        i = p.rfind(marker)
        if i >= 0 and (best is None or i > best[0]):
            best = (i, marker, keep)
    m = re.search(r"/lib/python3[^/]*/", p)
    if m and (best is None or m.start() > best[0]):
        return p[m.end():]
    if best is None:
        return p.rsplit("/", 1)[-1]
    i, marker, keep = best
    return p[i + 1:] if keep else p[i + len(marker):]


def parse_dump(text: str) -> dict:
    """{thread ident: [(file, line, function), ...] innermost first} for a
    faulthandler dump."""
    threads: dict = {}
    current = None
    for line in text.splitlines():
        m = _THREAD_RE.match(line)
        if m:
            current = threads.setdefault(int(m.group(1), 16), [])
            continue
        m = _FRAME_RE.match(line)
        if m and current is not None:
            current.append((m.group(1), int(m.group(2)), m.group(3)))
    return threads


def _waiting(frames: list) -> bool:
    if not frames:
        return True
    path, _line, func = frames[0]
    return (path.replace("\\", "/").rsplit("/", 1)[-1], func) in _WAIT_SITES


def _frame_text(frame) -> str:
    path, line, func = frame
    return f"{short_path(path)}:{line} in {func}"


def describe(frames: list) -> str:
    """The innermost frame, plus "called from <innermost localm frame>" when
    the innermost frame is not localm's own code."""
    if not frames:
        return "unknown code"
    inner = frames[0]
    text = _frame_text(inner)
    if not short_path(inner[0]).startswith(("localm/", "plugins/")):
        for f in frames[1:]:
            if short_path(f[0]).startswith(("localm/", "plugins/")):
                text += f", called from {_frame_text(f)}"
                break
    return text


class _ThreadCpu:
    """CPU seconds per Python thread, where the platform can tell; ``read``
    returns {} where it cannot."""

    def __init__(self) -> None:
        self._win = None
        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes
                k32 = ctypes.WinDLL("kernel32", use_last_error=True)
                k32.OpenThread.restype = wintypes.HANDLE
                k32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                k32.GetThreadTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
                k32.CloseHandle.argtypes = [wintypes.HANDLE]
                self._win = (ctypes, wintypes, k32)
            except (OSError, AttributeError) as e:
                logger.debug("stall watch: per-thread CPU unavailable: %s", e)

    def _one_win(self, native_id: int) -> Optional[float]:
        ctypes, wintypes, k32 = self._win
        h = k32.OpenThread(0x0800, False, native_id)   # THREAD_QUERY_LIMITED_INFORMATION
        if not h:
            return None
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            if not k32.GetThreadTimes(h, *[ctypes.byref(t) for t in times]):
                return None
            total = 0
            for t in times[2:]:
                total += (t.dwHighDateTime << 32) | t.dwLowDateTime
            return total / 1e7
        finally:
            k32.CloseHandle(h)

    @staticmethod
    def _one_linux(native_id: int) -> Optional[float]:
        try:
            with open(f"/proc/self/task/{native_id}/stat", "rb") as fh:
                fields = fh.read().rsplit(b")", 1)[1].split()
            return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
        except (OSError, ValueError, IndexError):
            return None

    def read(self) -> dict:
        """{thread ident: CPU seconds} for every live ``threading`` thread."""
        if self._win is None and not sys.platform.startswith("linux"):
            return {}
        out = {}
        for t in threading.enumerate():
            nid = getattr(t, "native_id", None)
            if t.ident is None or nid is None:
                continue
            cpu = self._one_win(nid) if self._win is not None else self._one_linux(nid)
            if cpu is not None:
                out[t.ident] = cpu
        return out


class _Sink:
    """A file descriptor faulthandler writes into, read back in memory."""

    def __init__(self) -> None:
        self._pipe = None
        self._tmp = None
        if sys.platform == "win32":
            import _winapi
            import msvcrt
            r, w = _winapi.CreatePipe(None, _SINK_BYTES)
            self.fd = msvcrt.open_osfhandle(w, os.O_WRONLY)
            self._pipe = (r, msvcrt.open_osfhandle(r, os.O_RDONLY), _winapi)
        else:
            self._tmp = tempfile.TemporaryFile()
            self.fd = self._tmp.fileno()

    def drain(self) -> str:
        """Everything written since the last drain."""
        if self._pipe is not None:
            handle, rfd, winapi = self._pipe
            chunks = []
            while True:
                avail = winapi.PeekNamedPipe(handle, 0)[0]
                if not avail:
                    break
                chunks.append(os.read(rfd, avail))
            data = b"".join(chunks)
        else:
            size = os.fstat(self.fd).st_size
            data = os.pread(self.fd, size, 0) if size else b""
            if size:
                os.ftruncate(self.fd, 0)
                os.lseek(self.fd, 0, os.SEEK_SET)
        return data.decode("utf-8", "backslashreplace")

    def close(self) -> None:
        if self._pipe is not None:
            os.close(self._pipe[1])
            os.close(self.fd)
            self._pipe = None
        elif self._tmp is not None:
            self._tmp.close()
            self._tmp = None


class StallWatch:
    """The keeper thread and the records it keeps. ``start()`` then
    ``stop()``; ``records()`` any time.

    *heartbeat_gap* returns seconds since the event loop last ticked, or None
    before its first tick. *loop_thread* is the event loop thread's ident.
    *trace_sink*, when given, receives the raw text of each native dump."""

    def __init__(self, *, threshold: float,
                 heartbeat_gap: Callable[[], Optional[float]],
                 loop_thread: Optional[int],
                 trace_sink: Optional[Callable[[str], None]] = None,
                 poll: float = 1.0) -> None:
        self.threshold = threshold
        self.poll = poll
        self._hb_gap = heartbeat_gap
        self._loop_thread = loop_thread
        self._trace_sink = trace_sink
        self._records: collections.deque = collections.deque(maxlen=_KEEP)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sink: Optional[_Sink] = None
        self._cpu = _ThreadCpu()
        self._open_loop: Optional[dict] = None
        self._loop_suspect = False

    def start(self) -> StallWatch:
        """Open the sink and start the keeper. Raises OSError when the sink
        cannot be created."""
        self._sink = _Sink()
        self._thread = threading.Thread(target=self._run, name="localm-stall-watch",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the keeper, cancel the armed dump and close the sink."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        try:
            faulthandler.cancel_dump_traceback_later()
        finally:
            if self._sink is not None:
                self._sink.close()
                self._sink = None

    def records(self) -> list:
        """The kept records, oldest first, each a dict: id, at (epoch
        seconds), seconds, kind ("gil" or "loop"), where, frames, ongoing."""
        with self._lock:
            return [dict(r, frames=list(r["frames"])) for r in self._records]

    def _add(self, record: dict) -> None:
        with self._lock:
            self._records.append(record)

    def _arm(self) -> None:
        faulthandler.dump_traceback_later(self.threshold, repeat=False,
                                          file=self._sink.fd, exit=False)

    def _run(self) -> None:
        """Each poll: drain the sink, then take a CPU reading and re-arm. A
        dump is attributed to the arm that fired it: the current one when
        *threshold* has passed since it, else the one before (a stall that
        began before the re-arm)."""
        try:
            cur = (time.monotonic(), self._cpu.read())
            self._arm()
            prev = cur
            while not self._stop.wait(self.poll):
                now = time.monotonic()
                text = self._sink.drain()
                if text:
                    time.sleep(0.05)
                    text += self._sink.drain()
                    fired = cur if now - cur[0] >= self.threshold else prev
                    self._gil_stall(text, now - fired[0], fired[1])
                else:
                    self._check_loop()
                cpu = self._cpu.read()
                prev, cur = cur, (time.monotonic(), cpu)
                self._arm()
        except Exception:
            logger.warning("stall watch stopped after an error", exc_info=True)
        finally:
            try:
                faulthandler.cancel_dump_traceback_later()
            except Exception:
                logger.debug("stall watch: could not cancel the armed dump",
                             exc_info=True)

    def _gil_stall(self, text: str, away: float, cpu_before: dict) -> None:
        """Record the GIL stall a native dump (*text*) caught. *away* is how
        long the keeper could not run. The holder is the thread that used at
        least half of *away* in CPU time, and the stall's length is that CPU
        time. Failing that, the candidates are the threads not sitting in a
        stdlib wait (those that ended since the dump, when per-thread CPU
        readings exist and any did), newest first; the holder is the first,
        ``where`` says how many others there were, and the length is *away*.
        Nothing is recorded when there is no candidate."""
        if self._trace_sink is not None:
            try:
                self._trace_sink(text)
            except Exception:
                logger.debug("stall watch: trace sink failed", exc_info=True)
        threads = parse_dump(text)
        threads.pop(threading.get_ident(), None)
        cpu_after = self._cpu.read()
        holder = None
        seconds = away
        busy = [t for t, frames in threads.items() if not _waiting(frames)]
        others = 0
        if cpu_after:
            spent = {t: cpu_after[t] - cpu_before.get(t, 0.0)
                     for t in threads if t in cpu_after}
            busiest = max(spent, key=spent.get, default=None)
            if busiest is not None and spent[busiest] >= 0.5 * away:
                holder, seconds = busiest, spent[busiest]
        if holder is None:
            ended = [t for t in busy if cpu_after and t not in cpu_after]
            candidates = ended or busy
            if not candidates:
                return
            holder, others = candidates[0], len(candidates) - 1
        frames = threads.get(holder) or []
        where = describe(frames)
        if others:
            where += f" (or one of {others} other running thread{'s' if others > 1 else ''})"
        self._add({"id": next(_ids), "at": time.time() - seconds,
                   "seconds": round(seconds, 1), "kind": "gil", "where": where,
                   "frames": [_frame_text(f) for f in frames[:_FRAMES_KEPT]],
                   "ongoing": False})
        logger.warning("server stall: Python was busy for %.1f s in %s", seconds, where)

    def _check_loop(self) -> None:
        """Open a LOOP record on the second consecutive poll that finds the
        heartbeat gap at or over the threshold, keep its length current while
        it lasts, and close it on the first poll under the threshold."""
        gap = self._hb_gap()
        rec = self._open_loop
        if gap is not None and gap >= self.threshold:
            if rec is None and not self._loop_suspect:
                self._loop_suspect = True
                return
            if rec is None:
                frame = sys._current_frames().get(self._loop_thread)
                frames = []
                while frame is not None and len(frames) < _FRAMES_KEPT:
                    frames.append((frame.f_code.co_filename, frame.f_lineno,
                                   frame.f_code.co_name))
                    frame = frame.f_back
                rec = {"id": next(_ids), "at": time.time() - gap,
                       "seconds": round(gap, 1), "kind": "loop",
                       "where": describe(frames),
                       "frames": [_frame_text(f) for f in frames], "ongoing": True}
                self._open_loop = rec
                self._add(rec)
            else:
                with self._lock:
                    rec["seconds"] = round(gap, 1)
            return
        self._loop_suspect = False
        if rec is not None:
            with self._lock:
                rec["seconds"] = round(time.time() - rec["at"], 1)
                rec["ongoing"] = False
            self._open_loop = None
            logger.warning("server stall: the event loop was blocked for %.1f s in %s",
                           rec["seconds"], rec["where"])


_WATCH: Optional[StallWatch] = None


def set_active(watch: Optional[StallWatch]) -> None:
    """Make *watch* the one ``recent_stalls`` reads (None when stopped)."""
    global _WATCH
    _WATCH = watch


def recent_stalls() -> list:
    """The active watch's records, or [] when none runs."""
    w = _WATCH
    return w.records() if w is not None else []
