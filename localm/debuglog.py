# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Debug mode for localm.

Enabled with ``--debug`` on ``localm gui`` / ``serve`` / ``run`` or by
setting ``LOCALM_DEBUG=1``. When active:

- A timestamped log file is created under ``~/.localm/logs/``.
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
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

_ENV_VAR = "LOCALM_DEBUG"

logger = logging.getLogger("localm")


def debug_enabled() -> bool:
    return bool(os.environ.get(_ENV_VAR))


def honor_env_debug() -> None:
    """Open the debug log file when debug was requested via the LOCALM_DEBUG env
    var (e.g. ``LOCALM_DEBUG=1 localm run ...``), not only via the ``--debug``
    flag. Previously the env var flipped debug SEMANTICS on (debug_enabled() ->
    True, verbose uvicorn) but nothing ever called enable_debug(), so no log file
    was written - a silent half-on state (REC-DEBUGENV). A truthy-but-non-path
    value ("1"/"true"/"yes") is the user's request; a real path means we inherited
    an already-open log from a parent process, so leave enable_debug() to no-op."""
    if debug_enabled() and log_file_path() is None:
        enable_debug()


def uvicorn_log_level() -> str:
    """The uvicorn log level for a server launch: verbose ``info`` in debug mode
    so the console window shows requests / connections / errors live (SRV-5),
    otherwise the quiet ``warning`` default."""
    return "info" if debug_enabled() else "warning"


# --------------------------------------------------------------------------- #
#  Always-on in-memory recent-activity buffer                                  #
#                                                                              #
#  A bug report is only useful if it carries what the app was DOING before it  #
#  broke. The on-disk debug log only exists under --debug, which a tester will #
#  not have enabled, so a normal report had no activity trail at all. This     #
#  bounded, in-memory ring buffer captures recent INFO+ log records ALWAYS, so #
#  the bug reporter can show the last breadcrumbs (model loads, backend pick,  #
#  swaps, warnings, errors) regardless of debug mode.                          #
#                                                                              #
#  Privacy: INFO and above ONLY. The raw, pre-scrub model output (chat content)#
#  is logged at DEBUG (inference/backends/llamacpp/llama.py), so it never lands #
#  in this buffer even when --debug is on. Nothing here is written to disk on   #
#  its own; it only leaves the machine if the user files AND sends a report,    #
#  which they review and edit first.                                           #
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
    when the process image is replaced (Task 1: save-bug / log durability).
    The file handlers use buffering=1 (line-buffered) so this is a belt-and-
    suspenders guard against any remaining buffer; it never raises."""
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
    those windows (LOG-1) - flooding the console and burying real errors.
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
    activity live in the window instead of only in the log file (SRV-5).
    Idempotent: a real (non-file) StreamHandler is added at most once. A
    FileHandler is a StreamHandler subclass, so it is explicitly excluded.

    The stream is a private duplicate of stderr (see ``_stable_console_stream``)
    so the mirror is NOT disrupted by the fd-2 redirection that silences native
    llama.cpp output - the cause of the LOG-1 "[WinError 6] The handle is
    invalid" log flood. The mirror never swallows a record either way: the file
    handler always carries every line, so nothing is hidden if a console write
    ever does fail (logging then reports that failure loudly, as before)."""
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


def log_file_path() -> Optional[Path]:
    """The active debug log file, or None when debug mode is off."""
    value = os.environ.get(_ENV_VAR, "")
    if value and value not in ("1", "true", "yes"):
        return Path(value)
    return None


def logs_dir() -> Path:
    from localm.config import HOME_DIR
    return HOME_DIR / "logs"


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
    # so no lines are lost if the process is killed or os.execv'd (Task 1: save-bug).
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
    _add_console_handler()   # SRV-5: also show debug activity in the console
    install_ring_buffer()    # keep the in-memory breadcrumb buffer for reports

    _install_thread_hook()
    logger.debug("debug mode enabled (pid %d)", os.getpid())
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
    _add_console_handler()   # SRV-5: a managed/child server is verbose too
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
