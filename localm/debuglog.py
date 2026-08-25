# SPDX-License-Identifier: AGPL-3.0-or-later
"""Debug mode for localm."""

from __future__ import annotations

import collections
import contextlib
import itertools
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

_ENV_VAR = "LOCALM_DEBUG"

logger = logging.getLogger("localm")


def debug_enabled() -> bool:
    return bool(os.environ.get(_ENV_VAR))


def debug_content_enabled() -> bool:
    """Whether the debug log may include raw CHAT CONTENT (a user prompt or a model reply)."""
    if not debug_enabled():
        return False
    try:
        from localm.audit import SessionMode, effective_mode
        # Suppressed if ANY surface is privacy: the backend that produces the
        # content is surface-agnostic and serves coder sessions through the same
        # generation path as chat and server.
        for surface in ("server", "chat", "coder"):
            if effective_mode(surface) == SessionMode.PRIVACY:
                return False
        return True
    except Exception:
        return False   # fail-safe: no chat content on disk


def honor_env_debug() -> None:
    """Open the debug log file when debug was requested via the LOCALM_DEBUG env var (e.g. ``LOCALM_DEBUG=1 localm run ...``), not only via the ``--debug`` flag."""
    if debug_enabled() and log_file_path() is None:
        enable_debug()


_deferred_records: "list[tuple[int, str, tuple]]" = []


def defer_log(level: int, msg: str, *args) -> None:
    """Queue a diagnostic raised BEFORE any log handler exists, for replay once one does."""
    if len(_deferred_records) < 100:
        _deferred_records.append((level, msg, args))


def _flush_deferred() -> None:
    """Emit and clear anything defer_log() queued."""
    global _deferred_records
    pending, _deferred_records = _deferred_records, []
    for level, msg, args in pending:
        logger.log(level, msg, *args)


def uvicorn_log_level() -> str:
    """The uvicorn log level for a server launch: verbose ``info`` in debug mode so the console window shows requests / connections / errors live (SRV-5), otherwise the quiet ``warning`` default."""
    return "info" if debug_enabled() else "warning"


# --------------------------------------------------------------------------- #
#  Always-on in-memory recent-activity buffer                                  #
#                                                                              #
#  A bounded ring buffer of recent INFO+ log records, captured regardless of    #
#  debug mode, so the bug reporter can show the last breadcrumbs.              #
#                                                                              #
#  INFO and above ONLY. Raw pre-scrub model output is logged at DEBUG, so it    #
#  never lands here. Nothing here is written to disk on its own.               #
#                                                                              #
#  PROCESS-LOCAL. install_ring_buffer() runs once, at CLI startup, in the       #
#  server/parent process. A spawned worker child is a fresh interpreter that    #
#  never runs it, so an INFO call inside a child never reaches this buffer; a   #
#  child-side breadcrumb needs an explicit relay back to the parent.           #
# --------------------------------------------------------------------------- #

_RING_CAPACITY = 400


class _RingBufferHandler(logging.Handler):
    """A logging handler that keeps the last N rendered records in memory."""

    def __init__(self, capacity: int = _RING_CAPACITY) -> None:
        super().__init__(level=logging.INFO)
        self._buf: "collections.deque[str]" = collections.deque(maxlen=capacity)
        self.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            render = self.format          # bound logging.Handler renderer
            self._buf.append(render(record))
        except Exception:
            # A logging handler must never raise into the code that logged.
            self.handleError(record)

    def snapshot(self) -> list:
        return list(self._buf)


def flush_log_handlers() -> None:
    """Flush all file handlers on the localm logger."""
    for h in list(logger.handlers):
        try:
            h.flush()
        except Exception:
            pass


def dump_ring_buffer() -> None:
    """Save the in-memory ring buffer to disk to survive os.execv."""
    if _ring_handler:
        try:
            import json
            from localm.config import HOME_DIR
            path = HOME_DIR / "run" / "ring_buffer.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(list(_ring_handler._buf)), encoding="utf-8")
        except Exception:
            pass

def load_ring_buffer() -> None:
    """Load the saved ring buffer from disk after os.execv."""
    if _ring_handler:
        try:
            from localm.config import HOME_DIR
            import json
            path = HOME_DIR / "run" / "ring_buffer.json"
            if path.exists():
                items = json.loads(path.read_text(encoding="utf-8"))
                _ring_handler._buf.extend(items)
                path.unlink(missing_ok=True)
        except Exception:
            pass

_ring_handler: Optional[_RingBufferHandler] = None


def install_ring_buffer(capacity: int = _RING_CAPACITY) -> bool:
    """Attach the always-on in-memory recent-activity buffer to the localm logger."""
    global _ring_handler
    if _ring_handler is not None:
        return False
    handler = _RingBufferHandler(capacity)
    logger.addHandler(handler)
    # The localm logger otherwise inherits the root's WARNING threshold, which
    # would drop the INFO breadcrumbs. Adds no console output: a non-debug run
    # has no stream handler on this logger. A later enable_debug() drops the
    # level to DEBUG; the handler's own INFO level still excludes DEBUG records.
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    _ring_handler = handler
    load_ring_buffer()
    return True


def recent_activity() -> list:
    """The buffered recent log lines (oldest first), or [] when not installed."""
    return _ring_handler.snapshot() if _ring_handler is not None else []


def _stable_console_stream():
    """A private duplicate of the current stderr, taken once, so the console mirror is immune to the OS-level fd-2 redirection the llama.cpp backend uses to silence native model output (``_quiet_stderr`` / ``_capture_stderr`` in inference/backends/llamacpp/llama.py dup2 a file over fd 2 around every model..."""
    stream = sys.stderr
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        return None
    try:
        dup_fd = os.dup(fd)
    except OSError:
        return None
    try:
        return os.fdopen(
            dup_fd, "w", buffering=1,
            encoding=getattr(stream, "encoding", None) or "utf-8",
            errors="backslashreplace", closefd=True)
    except OSError:
        # Close our own just-created descriptor when fdopen fails, so the dup is
        # not leaked. The caller falls back to live stderr.
        with contextlib.suppress(OSError):
            os.close(dup_fd)
        return None


def _add_console_handler() -> None:
    """Mirror debug logs to the server console (stderr), so a --debug run shows activity live in the window instead of only in the log file (SRV-5)."""
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            return
    stream = _stable_console_stream()
    if stream is None:
        # No duplicable stderr fd (a detached process): fall back to the live
        # stream. The file handler still carries every record.
        stream = sys.stderr
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
    logger.addHandler(handler)


@contextlib.contextmanager
def suppress_console_mirror():
    """Temporarily detach the debug-mode console-mirroring handler (see ``_add_console_handler`` above) so a log record emitted during this block reaches only the FILE handler (when debug mode is on), never the shared terminal - nothing is silently lost, only the LIVE view is paused, same 'never drop a rec..."""
    mirror = None
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            mirror = h
            break
    if mirror is None:
        yield
        return
    logger.removeHandler(mirror)
    try:
        yield
    finally:
        logger.addHandler(mirror)


def log_file_path() -> Optional[Path]:
    """The active debug log file, or None when debug mode is off."""
    value = os.environ.get(_ENV_VAR, "")
    if value and value not in ("1", "true", "yes"):
        return Path(value)
    return None


def logs_dir() -> Path:
    from localm.config import HOME_DIR
    return HOME_DIR / "logs"


# --- Hang watchdog (event-loop stall capture) ------------------------------ #
# A plain thread off the event loop that dumps every thread's stack to a file
# when the loop stops ticking.
#
# On by default; the trace file is opened lazily, only if a stall happens.
# LOCALM_HANG_WATCHDOG=0/false/off turns it off; =1/true/on also enables the
# verbose extras (asyncio debug and slow-callback logging).
_HANG_ENV = "LOCALM_HANG_WATCHDOG"
_HANG_SECS_ENV = "LOCALM_HANG_WATCHDOG_SECS"
_HANG_OFF = frozenset({"0", "false", "off", "no"})
_HANG_ON = frozenset({"1", "true", "on", "yes"})


def hang_watchdog_active() -> bool:
    """Whether the event-loop stall watchdog runs."""
    return os.environ.get(_HANG_ENV, "").strip().lower() not in _HANG_OFF


def hang_watchdog_verbose() -> bool:
    """Whether the extra, noisier diagnostics (asyncio debug mode + slow-callback logging) are on."""
    return os.environ.get(_HANG_ENV, "").strip().lower() in _HANG_ON


def hang_watchdog_threshold() -> float:
    """Seconds the event loop may go without a heartbeat before it is declared stalled and stacks are dumped."""
    try:
        return max(2.0, float(os.environ.get(_HANG_SECS_ENV, "10")))
    except ValueError:
        return 10.0


def hang_trace_path() -> Path:
    """One hang-trace file per run under the logs dir (appended to on each stall)."""
    d = logs_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"hang_{time.strftime('%Y-%m-%d_%H%M%S')}_{os.getpid()}.log"


# --- Native-fault trace for an isolated CHILD process ---------------------- #
# A child that dies from a native signal never regains control in Python, so only
# a signal-safe writer armed before the fault can record why, which is what
# faulthandler is. The parent picks the path, hands it to the child, and relays
# whatever the child left there into the shared debug log.
_crash_trace_counter = itertools.count()


def child_crash_trace_path(tag: str) -> Path:
    """A fresh per-child native-fault trace file under the logs dir. *tag* names the kind of child (e.g. 'gguf-worker') so a leftover file says what died."""
    d = logs_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"crash_{tag}_{os.getpid()}_{next(_crash_trace_counter)}.txt"


def enable_debug() -> Path:
    """Turn on debug mode for this process and its children."""
    existing = log_file_path()
    if existing is not None:
        return existing

    logs_dir().mkdir(parents=True, exist_ok=True)
    path = logs_dir() / f"localm_{time.strftime('%Y-%m-%d_%H%M%S')}_{os.getpid()}.log"
    os.environ[_ENV_VAR] = str(path)

    # buffering=1 is line-buffered, so each record is flushed immediately and no
    # lines are lost on a kill or os.execv. delay=True stops FileHandler opening
    # the file itself; the stream is then set to a manually opened line-buffered
    # handle, so baseFilename is preserved and the fd is opened once.
    handler = logging.FileHandler(path, mode="a", encoding="utf-8", delay=True)
    handler.stream = open(path, "a", buffering=1, encoding="utf-8",
                         errors="backslashreplace")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    _add_console_handler()   # SRV-5: also show debug activity in the console
    install_ring_buffer()    # keep the in-memory breadcrumb buffer for reports

    _install_thread_hook()
    logger.debug("debug mode enabled (pid %d)", os.getpid())
    _flush_deferred()   # replay import-time diagnostics now the log file exists
    return path


def _install_thread_hook() -> None:
    """Mirror uncaught thread exceptions into the debug log."""
    import threading

    previous = threading.excepthook

    def _hook(args):
        try:
            logger.error(
                "uncaught exception in thread %s",
                getattr(args.thread, "name", "?"),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
        except Exception:
            pass
        previous(args)

    # Idempotent: don't stack hooks on repeated enable calls
    if getattr(threading.excepthook, "_localm_hook", False):
        return
    _hook._localm_hook = True
    threading.excepthook = _hook


def attach_child_logging() -> None:
    """In a child process that inherited LOCALM_DEBUG: attach the file handler so this process's logger writes to the shared log file too."""
    path = log_file_path()
    if path is None or any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", "") == str(path)
        for h in logger.handlers
    ):
        return
    # buffering=1 = line-buffered: same guarantee as enable_debug() above.
    # delay=True + manual stream avoids opening the file twice.
    handler = logging.FileHandler(path, mode="a", encoding="utf-8", delay=True)
    handler.stream = open(path, "a", buffering=1, encoding="utf-8",
                         errors="backslashreplace")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    _add_console_handler()   # SRV-5: a managed/child server is verbose too
    install_ring_buffer()    # buffer breadcrumbs for a bug report in this child too
    _install_thread_hook()


def native_stderr_target() -> Optional[int]:
    """File descriptor that native (llama.cpp) stderr should be redirected to during suppression windows: the debug log in debug mode, else None (caller falls back to devnull)."""
    path = log_file_path()
    if path is None:
        return None
    try:
        return os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    except OSError:
        return None


def record_native_line(text: str) -> None:
    """Append a (possibly already-grouped) native log line straight into the recent-activity ring buffer, so the GUI status window's live log tail shows it too - see appface.py, which polls recent_activity() on a timer."""
    if _ring_handler is None:
        return
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    _ring_handler._buf.append(f"{stamp} INFO    localm.native: {text}")


class _LineGrouper:
    """Collapses a stream of lines into repeat-count groups, tolerating a small REPEATING SET of up to _MAX_PENDING distinct lines - not just a line repeated immediately after itself."""

    # Bounded: enough to hold a short repeating cycle open across one-off lines
    # interleaved with it, while keeping reordering latency to at most
    # _MAX_PENDING lines.
    _MAX_PENDING = 8

    # Grouping key: the line with every run of digits replaced, so lines that
    # differ only in an embedded number share one pending slot.
    _DIGITS_RE = re.compile(r"\d+")
    _PLACEHOLDER = "<N>"

    # Cap on how many distinct raw variants of one template are retained, purely
    # so a stream that never repeats cannot turn this into unbounded memory.
    # Past the cap the variants are no longer individually recoverable and the
    # count is reported as "N+".
    _MAX_VARIANTS = 256

    # TEMPLATE COLLAPSE IS A FALLBACK FOR WHAT EXACT MATCHING CANNOT SERVE, not
    # a replacement for it. Two conditions, BOTH required:
    #
    #   1. more distinct variants than _MAX_PENDING. At or below that, every
    #      variant fits in its own slot and exact matching already groups each
    #      one with its own count - which is strictly more informative than a
    #      placeholder. Collapsing there would DESTROY working output.
    #   2. genuine repetition - the average variant seen at least
    #      _COLLAPSE_MIN_REPEATS times. Without this, "load_tensors: layer 0
    #      assigned to device ROCm0" through "layer 27 ..." would compress 28
    #      informative lines into one, losing every layer number to save
    #      nothing. That is a LIST of distinct messages, not a flood.
    #
    # The measured cases separate cleanly on both, so neither is a fine
    # judgement: the captured flood is 105 variants at 85 repeats each; a load
    # report is 28 variants at 1; the two-id case in the tests is 2 variants,
    # which condition 1 alone already protects.
    _COLLAPSE_MIN_REPEATS = 2

    def __init__(self, emit) -> None:
        self._emit = emit
        # key -> [count, {variant: count}, overflowed]. The inner dict is
        # insertion-ordered, so variants can be replayed in arrival order.
        self._pending: "collections.OrderedDict[str, list]" = collections.OrderedDict()

    @classmethod
    def _key(cls, line: str) -> str:
        return cls._DIGITS_RE.sub(cls._PLACEHOLDER, line)

    def feed(self, line: str) -> None:
        key = self._key(line)
        entry = self._pending.get(key)
        if entry is not None:
            self._pending.move_to_end(key)
            entry[0] += 1
            if line in entry[1]:
                entry[1][line] += 1
            elif len(entry[1]) < self._MAX_VARIANTS:
                entry[1][line] = 1
            else:
                entry[2] = True
            return
        if len(self._pending) >= self._MAX_PENDING:
            _oldest_key, oldest = self._pending.popitem(last=False)
            self._emit_one(*oldest)
        self._pending[key] = [1, {line: 1}, False]

    def _emit_one(self, count: int, variants: dict, overflowed: bool) -> None:
        # ONE variant -> byte-identical to the pre-template behaviour, so every
        # line that groups correctly today keeps its exact present output.
        if len(variants) == 1 and not overflowed:
            line, n = next(iter(variants.items()))
            self._emit(line if n <= 1 else f"{line}({n})")
            return
        # Anything exact matching could have handled, or that is a list rather
        # than a flood, is emitted per-variant with its own count - byte-identical
        # to the pre-template behaviour. See _COLLAPSE_MIN_REPEATS for both
        # conditions and why each is load-bearing.
        if not overflowed and (len(variants) <= self._MAX_PENDING
                               or count < self._COLLAPSE_MIN_REPEATS * len(variants)):
            for line, n in variants.items():
                self._emit(line if n <= 1 else f"{line}({n})")
            return
        # A genuine flood. The varying part becomes a placeholder rather than one
        # arbitrary value, because picking one would misreport it as THE value.
        # The distinct count is KEPT - 105 distinct graph ids is a different
        # situation from 2, and rule 5 asks for a counted line, never a silenced
        # one.
        n_distinct = f"{len(variants)}+" if overflowed else str(len(variants))
        template = self._key(next(iter(variants)))
        self._emit(f"{template} (x{count}, {n_distinct} distinct)")

    def flush(self) -> None:
        for entry in self._pending.values():
            self._emit_one(*entry)
        self._pending.clear()


@contextlib.contextmanager
def dedup_native_stderr():
    """Redirect native (llama.cpp/ggml) stderr through a background reader that collapses consecutive IDENTICAL lines into 'line(N)' before re-emitting - fixes console/GUI spam from a tight native logging loop (e.g. ggml-cuda's 'CUDA Graph id N reused', printed once per token during generation) without sil..."""
    # _stable_console_stream() MUST run before fd 2 is redirected below: it
    # duplicates sys.stderr.fileno() (= fd 2) to get a handle that survives
    # the redirect. Calling it AFTER the dup2 would duplicate the PIPE's
    # write end instead of the real stderr - every "print to the terminal"
    # would then loop straight back into the same pipe the reader thread is
    # draining, which reads it again and re-emits it forever: a silent,
    # CPU-spinning infinite loop with no forward progress (this exact bug
    # was caught live - the whole generation call hung indefinitely).
    console = _stable_console_stream()

    saved_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)

    debug_fd = native_stderr_target()

    # Latch: warn ONCE if a native line cannot be written to the persisted debug
    # log, then stay silent. A persistently-failing fd must not spam the log, and
    # the warning is itself drained back through this same reader - the latch caps
    # that at one line.
    _debug_write_failed = False

    def _write_debug(data: bytes) -> None:
        nonlocal _debug_write_failed
        if debug_fd is None:
            return
        try:
            os.write(debug_fd, data)
        except OSError as e:
            if not _debug_write_failed:
                _debug_write_failed = True
                # The line still reaches the console and the ring buffer (via
                # _emit / grouper below), so this is degraded, not a silent drop;
                # the docstring's "nothing silently lost" holds because we say so.
                logger.warning("debuglog: native stderr line could not be written "
                               "to the persisted debug log (%s); further such "
                               "failures this session are suppressed", e)

    def _emit(text: str) -> None:
        if console is not None:
            with contextlib.suppress(OSError, ValueError):
                console.write(text + "\n")
                console.flush()
        record_native_line(text)

    grouper = _LineGrouper(_emit)

    def _reader() -> None:
        buf = b""
        try:
            while True:
                try:
                    chunk = os.read(read_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    _write_debug(raw + b"\n")
                    grouper.feed(raw.decode("utf-8", errors="replace"))
        finally:
            with contextlib.suppress(OSError):
                os.close(read_fd)
        if buf:
            _write_debug(buf)
            grouper.feed(buf.decode("utf-8", errors="replace"))
        grouper.flush()

    thread = threading.Thread(target=_reader, name="native-stderr-dedup", daemon=True)
    thread.start()
    try:
        yield
    finally:
        os.dup2(saved_fd, 2)
        thread.join(timeout=2.0)
        os.close(saved_fd)
        if debug_fd is not None:
            with contextlib.suppress(OSError):
                os.close(debug_fd)
        if console is not None:
            with contextlib.suppress(OSError):
                console.close()
