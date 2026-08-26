# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Debug mode for localm.

Enabled with ``--debug`` on ``localm gui`` / ``serve`` / ``run`` or by
setting ``LOCALM_DEBUG=1``. When active:

- A timestamped log file is created under ``<data dir>/logs/``.
- Python logging (logger ``localm``) writes DEBUG records to it.
- The native llama.cpp stderr stream - normally suppressed to keep chat
  output clean - is redirected INTO the log file instead of discarded.
  Native aborts (e.g. batch-size violations) print their reason there,
  which is exactly the information needed to analyse a hard crash.
- The raw, pre-scrub model output (including internal markers such as
  thinking-channel tags) is written to the log after each generation.
  Chat output itself stays scrubbed - tags never reach the user.

The env var is the single source of truth so child processes (jobs,
managed servers) inherit debug mode automatically. LOCALM_DEBUG holds the
log file path so every process in the tree appends to the same file.
"""

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


def native_fault_hint() -> str:
    """Trailing parenthetical for a native-fault message that already logged
    the full trace via ``logger.error``: names the debug log file when one
    exists, or how to get one next time when it does not."""
    if debug_enabled():
        return "full trace in the debug log"
    return "rerun with --debug to save the full trace to a log file"


def debug_content_enabled() -> bool:
    """Whether the debug log may include raw CHAT CONTENT (a user prompt or a model
    reply). True only when the debug log is on AND no relevant session mode is
    privacy - so privacy mode NEVER persists chat content to the debug log, even
    when ``--debug`` or ``keep_diagnostics`` enabled the log file for OPERATIONAL
    diagnostics. Fail-safe: no content when the mode cannot be resolved.

    This is the gate for any raw prompt/response logging (e.g. the GGUF backend's
    "raw model output" line). Operational lines (requests, timings, errors) stay
    on ``debug_enabled()``; only chat CONTENT is held to this stricter bar."""
    if not debug_enabled():
        return False
    try:
        from localm.audit import SessionMode, any_coder_session_is_privacy, effective_mode
        # Suppress content if ANY of these surfaces is privacy: the backend that
        # produces the content (llamacpp/llama.py) is surface-agnostic and serves
        # coder sessions through the same generation path as chat/server, so a
        # coder-only privacy override (coder_mode / .localcoder/config.toml) must
        # suppress it too, even when chat/server are not privacy.
        for surface in ("server", "chat", "coder"):
            if effective_mode(surface) == SessionMode.PRIVACY:
                return False
        # effective_mode("coder") above is called with no cwd, so it can never
        # see a project's own .localcoder/config.toml privacy pin. A coder
        # session that resolved privacy that way publishes it here instead.
        if any_coder_session_is_privacy():
            return False
        return True
    except Exception:
        return False   # fail-safe: no chat content on disk


def honor_env_debug() -> None:
    """Open the debug log file when debug was requested via the LOCALM_DEBUG env
    var (e.g. ``LOCALM_DEBUG=1 localm run ...``), not only via the ``--debug``
    flag. A truthy-but-non-path value ("1"/"true"/"yes") is the user's request;
    a real path means we inherited an already-open log from a parent process, so
    leave enable_debug() to no-op."""
    if debug_enabled() and log_file_path() is None:
        enable_debug()


_deferred_records: "list[tuple[int, str, tuple]]" = []


def defer_log(level: int, msg: str, *args) -> None:
    """Queue a diagnostic raised BEFORE any log handler exists, for replay once
    one does. Use this instead of ``logger.<level>()`` at MODULE-IMPORT scope.

    Import-time code in ``localm/cli/**`` runs before Click invokes ``main()``
    (which calls install_ring_buffer) and before any ``enable_debug()``, so at
    that moment the localm logger has no handler and still inherits the root's
    WARNING threshold. A plain ``logger.debug()`` there is therefore dropped AT
    THE CALL and no later handler can recover it. Queued records are replayed by
    enable_debug(), so they reach the debug log whichever way debug is turned on
    (LOCALM_DEBUG via honor_env_debug, or a per-command --debug flag later in the
    run).

    Bounded: a runaway producer can never grow this without limit."""
    if len(_deferred_records) < 100:
        _deferred_records.append((level, msg, args))


def _flush_deferred() -> None:
    """Emit and clear anything defer_log() queued. Idempotent: the queue is
    drained, so a second enable_debug() call replays nothing twice."""
    global _deferred_records
    pending, _deferred_records = _deferred_records, []
    for level, msg, args in pending:
        logger.log(level, msg, *args)


def uvicorn_log_level() -> str:
    """The uvicorn log level for a server launch: verbose ``info`` in debug mode
    so the console window shows requests / connections / errors live, otherwise
    the quiet ``warning`` default."""
    return "info" if debug_enabled() else "warning"


# --------------------------------------------------------------------------- #
#  Always-on in-memory recent-activity buffer                                  #
#                                                                              #
#  A bounded, in-memory ring buffer that captures recent INFO+ log records     #
#  ALWAYS, so the bug reporter can show the last breadcrumbs (model loads,     #
#  backend pick, swaps, warnings, errors) regardless of debug mode - the       #
#  on-disk debug log only exists under --debug.                                #
#                                                                              #
#  Privacy: INFO and above ONLY. The raw, pre-scrub model output (chat content)#
#  is logged at DEBUG (inference/backends/llamacpp/llama.py), so it never lands #
#  in this buffer even when --debug is on. Nothing here is written to disk on   #
#  its own; it only leaves the machine if the user files AND sends a report,    #
#  which they review and edit first.                                           #
#                                                                              #
#  PROCESS-LOCAL, NOT WHOLE-APP. install_ring_buffer() (below) is called once,  #
#  at CLI startup, in the SERVER/parent process only. An isolated worker child  #
#  (multiprocessing "spawn" - llamacpp/_runner.py, _hf_runner.py, etc.) is a    #
#  fresh interpreter that never runs that call, so this buffer does not exist   #
#  there: an INFO-level logger.info(...) inside a child is a genuine NO-OP      #
#  unless the child has ALSO run attach_child_logging() (which requires --debug #
#  already on - see that function below) - it never reaches THIS buffer either  #
#  way, since a spawned child cannot share this module's in-memory state with   #
#  its parent. INFO alone does not guarantee "reaches a bug report"; it only    #
#  does so for code that executes in the process that called install_ring_     #
#  buffer(). A child-side breadcrumb needs an explicit relay over its own IPC   #
#  (an extra envelope back to the parent, which then logs it itself) to land    #
#  here: llamacpp/_runner.py's ModelRunner.chat_stream logs its own             #
#  parent-side markers rather than relying on the child's INFO calls.           #
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
    """Flush all file handlers on the localm logger.

    Call immediately before os.execv() so no buffered log lines are lost
    when the process image is replaced. The file handlers use buffering=1
    (line-buffered), so this is a guard against any remaining buffer; it
    never raises."""
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
    """Attach the always-on in-memory recent-activity buffer to the localm logger.

    Idempotent. Captures INFO and above (never DEBUG, so no raw model output /
    chat content) into a bounded ring the bug reporter can dump. Returns True if
    it installed on this call.
    """
    global _ring_handler
    if _ring_handler is not None:
        return False
    handler = _RingBufferHandler(capacity)
    logger.addHandler(handler)
    # The localm logger otherwise inherits the root's WARNING threshold, which
    # would drop the INFO breadcrumbs we want. Lower it to INFO. This adds NO
    # console output: a non-debug run has no stream handler on this logger, and
    # INFO is below the root's WARNING lastResort, so nothing new is printed.
    # A later enable_debug() drops the level further to DEBUG; the handler's own
    # INFO level still keeps DEBUG (chat content) out of the buffer.
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    _ring_handler = handler
    load_ring_buffer()
    return True


def recent_activity() -> list:
    """The buffered recent log lines (oldest first), or [] when not installed."""
    return _ring_handler.snapshot() if _ring_handler is not None else []


def _stable_console_stream():
    """A private duplicate of the current stderr, taken once, so the console
    mirror is immune to the OS-level fd-2 redirection the llama.cpp backend uses
    to silence native model output (``_quiet_stderr`` / ``_capture_stderr`` in
    inference/backends/llamacpp/llama.py dup2 a file over fd 2 around every model
    load and every generation).

    The mirror writes through stderr. Without this isolation it writes through
    fd 2 *while that fd is being juggled*, which on Windows raises
    "OSError: [WinError 6] The handle is invalid" on nearly every log line during
    those windows - flooding the console and burying real errors.
    Duplicating the fd once keeps the mirror pointed at the original console no
    matter what later happens to fd 2.

    This only changes WHERE the mirror writes; it never drops a record. Every
    record is ALSO written by the file handler unconditionally, so no log line,
    and in particular no error, is silenced by this path. Returns None when
    stderr has no duplicable fd (e.g. a fully detached process); the caller then
    falls back to the live stream rather than losing the mirror.
    """
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
        # Cleanup of OUR OWN just-created descriptor on a construction failure -
        # this suppresses nothing of the application's; it only avoids leaking
        # the dup when fdopen itself fails. The caller falls back to live stderr.
        with contextlib.suppress(OSError):
            os.close(dup_fd)
        return None


def _add_console_handler() -> None:
    """Mirror debug logs to the server console (stderr), so a --debug run shows
    activity live in the window instead of only in the log file.
    Idempotent: a real (non-file) StreamHandler is added at most once. A
    FileHandler is a StreamHandler subclass, so it is explicitly excluded.

    The stream is a private duplicate of stderr (see ``_stable_console_stream``)
    so the mirror is NOT disrupted by the fd-2 redirection that silences native
    llama.cpp output. The mirror never swallows a record either way: the file
    handler always carries every line, so nothing is hidden if a console write
    ever does fail (logging then reports that failure loudly)."""
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            return
    stream = _stable_console_stream()
    if stream is None:
        # No duplicable stderr fd (e.g. a detached process): fall back to the
        # live stream. No worse than before, and the file handler still carries
        # every record - this fallback never silences anything.
        stream = sys.stderr
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
    logger.addHandler(handler)


@contextlib.contextmanager
def suppress_console_mirror():
    """Temporarily detach the debug-mode console-mirroring handler (see
    ``_add_console_handler`` above) so a log record emitted during this block
    reaches only the FILE handler (when debug mode is on), never the shared
    terminal - nothing is lost, only the LIVE view is paused, same
    "never drop a record" contract as ``dedup_native_stderr``.

    The console mirror's stream is ``_stable_console_stream()``, a private
    duplicate of stderr, so it does not write through fd 2 at all: an fd-2
    redirect (``_quiet_stderr``/``_capture_stderr``/``dedup_native_stderr`` in
    ``inference/backends/llamacpp/llama.py``) does not reach it, and this
    context manager is what pauses it.

    The case this covers: an isolated child process (the GGUF chat backend's
    model-load worker) calling ``logger.info(...)`` in debug mode writes
    straight to the shared terminal from a DIFFERENT process than the one
    rendering a parent-owned Rich ``Progress``/``Live`` display - the parent
    has zero visibility into that write, so its cursor-position bookkeeping
    desyncs and a stale frame is stranded on screen (see
    ``llamacpp/llama.py``'s ``LlamaCpp.__init__``, which pairs this with its
    merged native-call redirect scope).

    A no-op when no console mirror is currently attached (debug mode off, or
    never enabled) - nothing to suppress, nothing to restore."""
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
# The server runs on a single asyncio event loop; if any callback blocks it (a
# synchronous driver/subprocess/IO call), the WHOLE server freezes while the box
# looks idle. The watchdog (wired in http_server.lifespan) runs a plain thread
# OFF the loop that dumps every thread's stack to a file when the loop stops
# ticking, so an intermittent freeze is captured instead of lost.
#
# It is ON BY DEFAULT: a non-technical tester never has to set anything, and its
# steady-state cost is ~one wakeup/second on a background thread (the trace file
# is opened LAZILY, only if a real stall happens, so a healthy run leaves nothing
# behind). LOCALM_HANG_WATCHDOG=0/false/off turns it off; =1/true/on ALSO enables
# the verbose extras (asyncio debug + slow-callback logging).
_HANG_ENV = "LOCALM_HANG_WATCHDOG"
_HANG_SECS_ENV = "LOCALM_HANG_WATCHDOG_SECS"
_HANG_OFF = frozenset({"0", "false", "off", "no"})
_HANG_ON = frozenset({"1", "true", "on", "yes"})


def hang_watchdog_active() -> bool:
    """Whether the event-loop stall watchdog runs. ON unless explicitly disabled
    (LOCALM_HANG_WATCHDOG=0/false/off), so a tester's freeze is captured with no
    setup on their part."""
    return os.environ.get(_HANG_ENV, "").strip().lower() not in _HANG_OFF


def hang_watchdog_verbose() -> bool:
    """Whether the extra, noisier diagnostics (asyncio debug mode + slow-callback
    logging) are on. Explicit opt-in only (LOCALM_HANG_WATCHDOG=1/true/on)."""
    return os.environ.get(_HANG_ENV, "").strip().lower() in _HANG_ON


def hang_watchdog_threshold() -> float:
    """Seconds the event loop may go without a heartbeat before it is declared
    stalled and stacks are dumped. Floored at 2s so a normal slow callback does
    not trip it; default 10s."""
    try:
        return max(2.0, float(os.environ.get(_HANG_SECS_ENV, "10")))
    except ValueError:
        return 10.0


def hang_trace_path() -> Path:
    """One hang-trace file per run under the logs dir (appended to on each
    stall). Kept next to the debug log so a bug report picks it up."""
    d = logs_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"hang_{time.strftime('%Y-%m-%d_%H%M%S')}_{os.getpid()}.log"


# --- Native-fault trace for an isolated CHILD process ---------------------- #
# A child that dies from a native SIGNAL (SIGILL/SIGSEGV/SIGABRT/SIGBUS/SIGFPE)
# never regains control in Python, so no `except` clause and no logging call in
# that child can record why - see llamacpp/_runner.py's _runner_entry. Only a
# signal-safe writer armed BEFORE the fault can, which is what faulthandler is.
# The parent arms nothing itself: it picks the path, hands it to the child, and
# relays whatever the child left there into the shared debug log.
_crash_trace_counter = itertools.count()


def child_crash_trace_path(tag: str) -> Path:
    """A fresh per-child native-fault trace file under the logs dir.
    *tag* names the kind of child (e.g. "gguf-worker") so a leftover file says
    what died.

    CALLED BY THE PARENT, ONCE, and the resulting path is passed to the child
    explicitly rather than recomputed there: `logs_dir()` reads
    `config.HOME_DIR`, which is frozen at IMPORT time, so a spawned child that
    inherited a different LOCALM_HOME would resolve a DIFFERENT directory and
    the two sides would disagree about where the trace went."""
    d = logs_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"crash_{tag}_{os.getpid()}_{next(_crash_trace_counter)}.txt"


def enable_debug() -> Path:
    """
    Turn on debug mode for this process and its children.

    Idempotent: a second call returns the existing log file.
    """
    existing = log_file_path()
    if existing is not None:
        return existing

    logs_dir().mkdir(parents=True, exist_ok=True)
    path = logs_dir() / f"localm_{time.strftime('%Y-%m-%d_%H%M%S')}_{os.getpid()}.log"
    os.environ[_ENV_VAR] = str(path)

    # buffering=1 = line-buffered: each log record is flushed to disk immediately
    # so no lines are lost if the process is killed or os.execv'd.
    # delay=True prevents FileHandler from opening the file internally; we then
    # set stream to a manually opened line-buffered handle so baseFilename is
    # preserved (used by attach_child_logging to deduplicate) while the fd is
    # opened exactly once with the correct buffer mode.
    handler = logging.FileHandler(path, mode="a", encoding="utf-8", delay=True)
    handler.stream = open(path, "a", buffering=1, encoding="utf-8",
                         errors="backslashreplace")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    _add_console_handler()   # also show debug activity in the console
    install_ring_buffer()    # keep the in-memory breadcrumb buffer for reports

    _install_thread_hook()
    logger.debug("debug mode enabled (pid %d)", os.getpid())
    _flush_deferred()   # replay import-time diagnostics now the log file exists
    return path


def _install_thread_hook() -> None:
    """
    Mirror uncaught thread exceptions into the debug log.

    Worker threads (generation, jobs, agent sessions) otherwise print their
    tracebacks only to the console, where they scroll away - the log file
    must carry them too.
    """
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
    """
    In a child process that inherited LOCALM_DEBUG: attach the file handler
    so this process's logger writes to the shared log file too.
    """
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
    _add_console_handler()   # a managed/child server is verbose too
    install_ring_buffer()    # buffer breadcrumbs for a bug report in this child too
    _install_thread_hook()


def native_stderr_target() -> Optional[int]:
    """
    File descriptor that native (llama.cpp) stderr should be redirected to
    during suppression windows: the debug log in debug mode, else None
    (caller falls back to devnull).

    The caller owns the descriptor and must close it after dup2.
    """
    path = log_file_path()
    if path is None:
        return None
    try:
        return os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    except OSError:
        return None


def record_native_line(text: str) -> None:
    """Append a (possibly already-grouped) native log line straight into the
    recent-activity ring buffer, so the GUI status window's live log tail
    shows it too - see appface.py, which polls recent_activity() on a timer.

    Bypasses the logging.Handler chain (does not call logger.info()): a
    debug-mode run already has its own console StreamHandler mirroring
    *structured* localm log calls to the terminal (_add_console_handler), and
    routing native text through the same logger would print it a second time
    there. dedup_native_stderr() below writes the terminal copy itself
    directly, so this function's only job is the ring buffer / GUI side.
    """
    if _ring_handler is None:
        return
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    _ring_handler._buf.append(f"{stamp} INFO    localm.native: {text}")


class _LineGrouper:
    """Collapses a stream of lines into repeat-count groups, tolerating a small
    REPEATING SET of up to _MAX_PENDING distinct lines - not just a line
    repeated immediately after itself.

    Native ggml/llama.cpp stderr often alternates between a handful of
    DISTINCT lines rather than repeating one line twice in a row (e.g.
    ggml-cuda's "CUDA graph warmup complete" / "...warmup reset", toggled
    every call), and the same stream also carries "CUDA Graph id N reused",
    where N varies.

    Grouping is therefore keyed on a DIGIT-NORMALISED template, not the raw
    line. A template seen with ONE variant renders as the bare line or
    "line(N)"; a template collapsed across variants renders as
    "<template> (xCOUNT, N distinct)".

    Up to _MAX_PENDING distinct lines are held open at once as an LRU set,
    each with its own running count: a line that matches one already pending
    increments it in place and marks it most-recently-used; a genuinely new
    distinct line, once the set is full, evicts and emits the LEAST-recently-
    used slot to make room. LRU (not first-seen order) matters here because a
    one-off line that never repeats (the "id N" line above) must cycle
    straight through the set without evicting a slot that IS actively
    repeating - otherwise every unrelated one-off line between repeats would
    reset the count on the very thing this class exists to collapse.
    _MAX_PENDING=1 reduces to a single-line lookback exactly. Nothing is
    dropped: every distinct line is eventually emitted exactly once, either
    bare (a run of 1) or with its repeat count.

    Generic - no CUDA-specific string appears in this class - so another
    backend's own repeating cycle collapses too."""

    # Small and bounded: enough to hold a short repeating cycle open across
    # one-off lines interleaved with it, while keeping a genuinely
    # non-repeating stream's reordering latency (at most _MAX_PENDING lines)
    # small.
    _MAX_PENDING = 8

    # Grouping key: the line with every run of digits replaced, so lines that
    # differ ONLY in an embedded number share one pending slot.
    _DIGITS_RE = re.compile(r"\d+")
    _PLACEHOLDER = "<N>"

    # Cap on how many distinct raw variants of one template are retained, purely
    # so a stream that never repeats cannot turn this into unbounded memory.
    # Past the cap the variants are no longer individually recoverable and the
    # count is reported as "N+".
    _MAX_VARIANTS = 256

    # Template collapse is a FALLBACK for what exact matching cannot serve, not
    # a replacement for it. Two conditions, BOTH required:
    #
    #   1. more distinct variants than _MAX_PENDING. At or below that, every
    #      variant fits in its own slot and exact matching already groups each
    #      one with its own count, which is more informative than a
    #      placeholder.
    #   2. genuine repetition - the average variant seen at least
    #      _COLLAPSE_MIN_REPEATS times. Without this, "load_tensors: layer 0
    #      assigned to device ROCm0" through "layer 27 ..." would compress 28
    #      informative lines into one, losing every layer number. That is a
    #      LIST of distinct messages, not a flood.
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
        # ONE variant -> the bare line, or the line with its repeat count.
        if len(variants) == 1 and not overflowed:
            line, n = next(iter(variants.items()))
            self._emit(line if n <= 1 else f"{line}({n})")
            return
        # Anything exact matching could have handled, or that is a list rather
        # than a flood, is emitted per-variant with its own count. See
        # _COLLAPSE_MIN_REPEATS for both conditions.
        if not overflowed and (len(variants) <= self._MAX_PENDING
                               or count < self._COLLAPSE_MIN_REPEATS * len(variants)):
            for line, n in variants.items():
                self._emit(line if n <= 1 else f"{line}({n})")
            return
        # A genuine flood. The varying part becomes a placeholder rather than one
        # arbitrary value, and the distinct count is KEPT, so the line is
        # counted rather than silenced.
        n_distinct = f"{len(variants)}+" if overflowed else str(len(variants))
        template = self._key(next(iter(variants)))
        self._emit(f"{template} (x{count}, {n_distinct} distinct)")

    def flush(self) -> None:
        for entry in self._pending.values():
            self._emit_one(*entry)
        self._pending.clear()


@contextlib.contextmanager
def dedup_native_stderr():
    """
    Redirect native (llama.cpp/ggml) stderr through a background reader that
    collapses consecutive IDENTICAL lines into "line(N)" before re-emitting -
    fixes console/GUI spam from a tight native logging loop (e.g. ggml-cuda's
    "CUDA Graph id N reused", printed once per token during generation)
    without silently discarding the information the way _quiet_stderr's
    devnull suppression does.

    The grouped line is:
      - written to a stable duplicate of the REAL stderr, so a terminal
        still shows it live, grouped instead of spammed;
      - appended to the always-on recent-activity ring buffer via
        record_native_line(), so the GUI status window's log tail (which
        already polls that same buffer) shows the identical grouped view.

    Nothing is SILENTLY lost from the persisted record: in debug mode, every
    RAW (ungrouped) line is ALSO appended to the debug log file
    (native_stderr_target()), exactly as _quiet_stderr already does for the
    windows it covers - only the two LIVE views are grouped, never the file.
    If a persisted-log write itself fails (disk full, a closed fd), the line
    still reaches the console and the ring buffer and a single warning is
    emitted (see the _write_debug latch below) - degraded, never a silent drop.

    Draining a pipe requires an active reader (the writer blocks once it
    fills), so this spins up a background thread - entering/exiting this
    context is NOT cheap. Callers must wrap a whole generation loop in ONE
    call, never re-enter it per native call / per token, or both the
    dedup grouping (state resets on every entry) and the per-entry thread
    overhead break.
    """
    # _stable_console_stream() MUST run before fd 2 is redirected below: it
    # duplicates sys.stderr.fileno() (= fd 2) to get a handle that survives
    # the redirect. Calling it AFTER the dup2 would duplicate the PIPE's
    # write end instead of the real stderr - every "print to the terminal"
    # would then loop straight back into the same pipe the reader thread is
    # draining, which reads it again and re-emits it forever: a silent,
    # CPU-spinning infinite loop with no forward progress.
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
                # _emit / grouper below), so this is degraded, not a silent drop.
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
