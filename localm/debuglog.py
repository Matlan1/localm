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

import logging
import os
import time
from pathlib import Path
from typing import Optional

_ENV_VAR = "LOCALM_DEBUG"

logger = logging.getLogger("localm")


def debug_enabled() -> bool:
    return bool(os.environ.get(_ENV_VAR))


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

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

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
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
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
