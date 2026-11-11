# SPDX-License-Identifier: AGPL-3.0-or-later
"""Detect a server-wide stall and keep a small in-memory record of where it was.

Two kinds of stall are recorded:

* GIL: one thread holds the interpreter lock inside a long C call (a large
  ``json.dumps``, a native library called without releasing it) and no other
  Python thread runs.
* LOOP: the event loop thread runs one long piece of code while other threads
  keep running.

A keeper thread re-arms ``faulthandler.dump_traceback_later`` every *poll*
seconds. The dump runs on faulthandler's own native thread without the GIL and
fires only when the keeper could not re-arm it for *threshold* seconds: a GIL
stall. The dump goes to an in-memory sink (an anonymous pipe on Windows, an
unnamed temporary file elsewhere) and is parsed when the keeper runs again. A
LOOP stall is read by the keeper from the heartbeat gap, with the loop
thread's frames from ``sys._current_frames()``.

When the event loop thread is known, a record's ``where`` and ``frames`` are
that thread's (what the loop was blocked in), and the record keeps every
thread's stack with its name, ident and daemon flag, the event loop thread
first (``records(with_threads=True)``).

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
from localm.pathscrub import short_code_path as short_path

#: Bytes the Windows pipe buffers without a reader; a full dump fits.
_SINK_BYTES = 4 << 20

#: Records kept.
_KEEP = 8

#: Frames kept per record, innermost first.
_FRAMES_KEPT = 12

#: Frames kept per thread in a record's ``threads`` list and in a rendered
#: stack, innermost first.
_THREAD_FRAMES = 64

_THREAD_RE = re.compile(r"^(?:Current thread|Thread) (0x[0-9a-fA-F]+)")

#: (file name, function) of stdlib frames a thread sits in while it waits with
#: the GIL released.
_WAIT_SITES = frozenset({
    ("threading.py", "wait"), ("threading.py", "_wait_for_tstate_lock"),
    ("queue.py", "get"), ("selectors.py", "select"),
    ("windows_events.py", "_poll"), ("windows_events.py", "select"),
    ("thread.py", "_worker"), ("connection.py", "_exhaustive_wait"),
})

_ids = itertools.count(1)
_BOOT = f"{os.getpid()}-{int(time.time())}"


def _new_id() -> str:
    """A record id unique across processes and restarts."""
    return f"{_BOOT}-{next(_ids)}"


def _parse_frame(line: str) -> Optional[tuple]:
    """(file, line, function) for a dump line ``  File "x", line N in f``, else
    None."""
    body = line.lstrip()
    if not body.startswith('File "'):
        return None
    path, sep, rest = body[6:].rpartition('", line ')
    number, sep2, func = rest.partition(" in ")
    if not sep or not sep2 or not number.isdecimal():
        return None
    return path, int(number), func


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
        frame = _parse_frame(line)
        if frame is not None and current is not None:
            current.append(frame)
    return threads


def live_frames() -> dict:
    """{thread ident: [(file, line, function), ...] innermost first} for every
    thread of this process, read from ``sys._current_frames()``, at most
    ``_THREAD_FRAMES`` frames each. The calling thread is included."""
    out: dict = {}
    for ident, frame in sys._current_frames().items():
        frames = []
        while frame is not None and len(frames) < _THREAD_FRAMES:
            frames.append((frame.f_code.co_filename, frame.f_lineno,
                           frame.f_code.co_name))
            frame = frame.f_back
        out[ident] = frames
    return out


def thread_names() -> dict:
    """{ident: (name, daemon)} for every live ``threading.Thread``."""
    return {t.ident: (t.name, t.daemon) for t in threading.enumerate()
            if t.ident is not None}


def thread_entries(frames_by_ident: dict, loop_thread: Optional[int],
                   names: Optional[dict] = None) -> list:
    """One dict per thread in *frames_by_ident*: ident, name and daemon (from
    the live ``threading.Thread``, else from *names* as ``thread_names``
    returned it earlier, else None), loop (True for *loop_thread*) and frames
    (as given). The event loop thread comes first; the others keep their
    order."""
    known = dict(names or {})
    known.update(thread_names())
    entries = []
    for ident, frames in frames_by_ident.items():
        name, daemon = known.get(ident, (None, None))
        entries.append({"ident": ident, "name": name, "daemon": daemon,
                        "loop": loop_thread is not None and ident == loop_thread,
                        "frames": list(frames)})
    entries.sort(key=lambda e: not e["loop"])
    return entries


def _thread_header(entry: dict) -> str:
    name = entry["name"] if entry["name"] is not None else "?"
    daemon = "?" if entry["daemon"] is None else str(entry["daemon"])
    tag = " [event loop thread]" if entry["loop"] else ""
    return f'Thread 0x{entry["ident"]:08x} "{name}" daemon={daemon}{tag}'


def format_stacks(entries: list) -> str:
    """Every thread in *entries* (see ``thread_entries``) as text, in order:
    a ``Thread 0x<ident> "<name>" daemon=<flag>`` header (the event loop thread
    tagged ``[event loop thread]``), then its frames innermost first, at most
    ``_THREAD_FRAMES``. A (file, line, function) frame is written as
    ``  File "<file>", line N in <function>`` (``parse_dump`` reads it back); a
    string frame is written as it is."""
    out = []
    for entry in entries:
        out.append(f"{_thread_header(entry)} (most recent call first):")
        for frame in entry["frames"][:_THREAD_FRAMES]:
            if isinstance(frame, str):
                out.append(f"  {frame}")
            else:
                path, line, func = frame
                out.append(f'  File "{path}", line {line} in {func}')
        out.append("")
    return "\n".join(out)


def loop_summary(entries: list) -> str:
    """One line naming what the event loop thread in *entries* was running
    (``describe`` of its frames), or saying it was not found or had no frame."""
    for entry in entries:
        if entry["loop"]:
            frames = [f for f in entry["frames"] if not isinstance(f, str)]
            if frames:
                return f"the event loop thread was in {describe(frames)}"
            if entry["frames"]:
                return f"the event loop thread was in {entry['frames'][0]}"
            return "the event loop thread had no Python frame"
    return "the event loop thread was not found"


def _short_entries(entries: list) -> list:
    """*entries* with every frame as its ``_frame_text`` string."""
    return [dict(e, frames=[f if isinstance(f, str) else _frame_text(f)
                            for f in e["frames"][:_THREAD_FRAMES]])
            for e in entries]


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
    returns {} where it cannot. Every call keeps the GIL: Windows calls go
    through ``ctypes.PyDLL``, Linux reads ``time.pthread_getcpuclockid``
    clocks."""

    def __init__(self) -> None:
        self._win: Optional[tuple] = None
        self._linux = sys.platform.startswith("linux") and hasattr(time, "pthread_getcpuclockid")
        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes
                k32 = ctypes.PyDLL("kernel32")
                k32.OpenThread.restype = wintypes.HANDLE
                k32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                k32.GetThreadTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
                k32.CloseHandle.argtypes = [wintypes.HANDLE]
                self._win = (ctypes, wintypes, k32)
            except (OSError, AttributeError) as e:
                logger.debug("stall watch: per-thread CPU unavailable: %s", e)

    @staticmethod
    def _one_win(win: tuple, native_id: int) -> Optional[float]:
        ctypes, wintypes, k32 = win
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
    def _one_linux(ident: int) -> Optional[float]:
        try:
            return time.clock_gettime(time.pthread_getcpuclockid(ident))
        except (OSError, OverflowError):
            return None

    def read(self) -> dict:
        """{thread ident: CPU seconds} for every live ``threading`` thread."""
        if self._win is None and not self._linux:
            return {}
        out = {}
        for t in threading.enumerate():
            if t.ident is None:
                continue
            if self._win is not None:
                nid = getattr(t, "native_id", None)
                cpu = None if nid is None else self._one_win(self._win, nid)
            else:
                cpu = self._one_linux(t.ident)
            if cpu is not None:
                out[t.ident] = cpu
        return out


class _Sink:
    """A file descriptor faulthandler writes into, read back in memory."""

    def __init__(self) -> None:
        self._pipe: Optional[tuple] = None
        self._tmp = None
        self._read_off = 0
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
        """Everything written since the last drain. The POSIX file is never
        truncated or seeked: faulthandler writes at its own offset."""
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
            data = b""
            if size > self._read_off:
                data = os.pread(self.fd, size - self._read_off, self._read_off)
                self._read_off += len(data)
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
        self._names: dict = {}
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

    def records(self, *, with_threads: bool = False) -> list:
        """The kept records, oldest first, each a dict: id, at (epoch
        seconds), seconds, kind ("gil" or "loop"), where, frames, ongoing.
        ``frames`` is the event loop thread's stack when the loop thread is
        known, else the stalling thread's. With *with_threads* each record also
        carries ``threads``: every thread's stack at the stall (see
        ``thread_entries``; frames as ``file:line in function`` strings), the
        event loop thread first."""
        with self._lock:
            out = []
            for r in self._records:
                rec = {k: v for k, v in r.items() if k != "threads"}
                rec["frames"] = list(r["frames"])
                if with_threads:
                    rec["threads"] = [dict(t, frames=list(t["frames"]))
                                      for t in r.get("threads", [])]
                out.append(rec)
            return out

    def _add(self, record: dict) -> None:
        with self._lock:
            self._records.append(record)

    def _arm(self, sink: _Sink) -> None:
        faulthandler.dump_traceback_later(self.threshold, repeat=False,
                                          file=sink.fd, exit=False)

    def _run(self) -> None:
        """Each poll: drain the sink, then take a CPU reading and re-arm. A
        dump is attributed to the arm that fired it: the current one when
        *threshold* has passed since it, else the one before (a stall that
        began before the re-arm)."""
        sink = self._sink
        if sink is None:
            return
        try:
            cur = (time.monotonic(), self._cpu.read())
            self._names = thread_names()
            self._arm(sink)
            prev = cur
            while not self._stop.wait(self.poll):
                now = time.monotonic()
                text = sink.drain()
                if text:
                    time.sleep(0.05)
                    text += sink.drain()
                    fired = cur if now - cur[0] >= self.threshold else prev
                    self._gil_stall(text, now - fired[0], fired[1])
                else:
                    self._check_loop()
                cpu = self._cpu.read()
                self._names = thread_names()
                prev, cur = cur, (time.monotonic(), cpu)
                self._arm(sink)
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
        Nothing is recorded when there is no candidate.

        When the event loop thread is known and in the dump, the record's
        ``where`` and ``frames`` are the loop thread's, and ``where`` adds the
        holder when the CPU readings prove one or when it ended since the dump.
        The trace sink receives the dump's first line (faulthandler's
        ``Timeout`` line), a line naming what the loop thread was in, then
        every thread's stack, the event loop thread first (``format_stacks``);
        the raw dump when it parses to no thread."""
        threads = parse_dump(text)
        threads.pop(threading.get_ident(), None)
        entries = thread_entries(threads, self._loop_thread, self._names)
        if self._trace_sink is not None:
            head = text.lstrip().split("\n", 1)[0]
            head = "" if _THREAD_RE.match(head) else f"{head}\n"
            try:
                self._trace_sink(f"{head}{loop_summary(entries)}\n{format_stacks(entries)}"
                                 if entries else text)
            except Exception:
                logger.debug("stall watch: trace sink failed", exc_info=True)
        cpu_after = self._cpu.read()
        holder = None
        proven = False
        seconds = away
        busy = [t for t, frames in threads.items() if not _waiting(frames)]
        others = 0
        if cpu_after:
            spent = {t: cpu_after[t] - cpu_before.get(t, 0.0)
                     for t in threads if t in cpu_after}
            busiest = max(spent, key=lambda t: spent[t], default=None)
            if busiest is not None and spent[busiest] >= 0.5 * away:
                holder, seconds, proven = busiest, spent[busiest], True
        ended: list = []
        if holder is None:
            ended = [t for t in busy if cpu_after and t not in cpu_after]
            candidates = ended or busy
            if not candidates:
                logger.info("stall watch: no Python thread ran for %.1f s and every "
                            "thread was waiting (%s); not recorded as a stall",
                            away, loop_summary(entries))
                return
            holder, others = candidates[0], len(candidates) - 1
        loop_frames = (threads.get(self._loop_thread)
                       if self._loop_thread is not None else None)
        if loop_frames is not None:
            frames = loop_frames
            where = f"{describe(frames)} on the event loop thread"
            name = next((e["name"] for e in entries if e["ident"] == holder), None)
            label = f"thread {name or hex(holder)}"
            if holder != self._loop_thread and proven:
                where += (f"; {label} used {seconds:.1f} s of CPU "
                          f"in {describe(threads.get(holder) or [])}")
            elif holder != self._loop_thread and holder in ended:
                where += (f"; {label} ended after running "
                          f"in {describe(threads.get(holder) or [])}")
        else:
            frames = threads.get(holder) or []
            where = describe(frames)
            if others:
                where += f" (or one of {others} other running thread{'s' if others > 1 else ''})"
        self._add({"id": _new_id(), "at": time.time() - seconds,
                   "seconds": round(seconds, 1), "kind": "gil", "where": where,
                   "frames": [_frame_text(f) for f in frames[:_FRAMES_KEPT]],
                   "threads": _short_entries(entries),
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
                stacks = live_frames()
                stacks.pop(threading.get_ident(), None)
                frames = ([] if self._loop_thread is None
                          else stacks.get(self._loop_thread, []))[:_FRAMES_KEPT]
                rec = {"id": _new_id(), "at": time.time() - gap,
                       "seconds": round(gap, 1), "kind": "loop",
                       "where": describe(frames),
                       "frames": [_frame_text(f) for f in frames],
                       "threads": _short_entries(thread_entries(stacks, self._loop_thread)),
                       "ongoing": True}
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


def recent_stalls(*, with_threads: bool = False) -> list:
    """The active watch's records (see ``StallWatch.records``), or [] when
    none runs."""
    w = _WATCH
    return w.records(with_threads=with_threads) if w is not None else []


def recent_stall_stacks() -> str:
    """The newest recorded stall that carries thread stacks, as text: one line
    with its kind, length and ``where``, then every thread's stack, the event
    loop thread first (``format_stacks``, frames as ``file:line in function``).
    Empty when no such stall is recorded."""
    for rec in reversed(recent_stalls(with_threads=True)):
        if rec.get("threads"):
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rec["at"]))
            return (f"===== LOCALM STALL WATCH: {rec['kind']} stall of "
                    f"{rec['seconds']:.1f} s at {when} =====\n"
                    f"{rec['where']}\n{format_stacks(rec['threads'])}")
    return ""
