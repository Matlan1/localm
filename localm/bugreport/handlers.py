# SPDX-License-Identifier: AGPL-3.0-or-later
"""Process-wide handlers that turn an uncaught exception in the main thread, a
background thread or an asyncio task into a saved report.
"""

from __future__ import annotations

import sys
import threading

import localm.bugreport as _br

# --------------------------------------------------------------------------- #
#  Process-wide net: catch a bug ANYWHERE, not just in a CLI command           #
#                                                                              #
#  Bugs are not confined to setup: a crash in a background thread (model       #
#  preload, the jobs runner, the coder loop) and an uncaught main-thread       #
#  exception both route through the same "sorry X for Y" + report path.        #
#                                                                              #
#  Limit: this catches PYTHON exceptions. A native crash inside llama.dll      #
#  (segfault), an OS OOM-kill, or os._exit cannot be caught in-process -       #
#  that needs subprocess isolation.                                            #
# --------------------------------------------------------------------------- #


def _handle_main_exception(exc_type, exc, tb) -> None:
    """sys.excepthook: last-resort net for an uncaught main-thread exception.
    Ctrl+C stays quiet; everything else becomes a friendly report. Fully guarded:
    if reporting itself fails, defer to the interpreter's default hook."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    try:
        interactive = bool(getattr(sys.stdin, "isatty", lambda: False)())
        _br.report_failure(
            summary=getattr(exc, "summary", None) or "localm hit an unexpected error",
            reason=getattr(exc, "reason", None) or str(exc),
            error=exc, context=getattr(exc, "context", None) or {},
            interactive=interactive)
    except Exception:
        sys.__excepthook__(exc_type, exc, tb)


def _handle_thread_exception(args) -> None:
    """threading.excepthook: log a crash in a background thread and save a
    report (never prompt - a worker thread has no user attention), so the
    failure is visible without taking the whole app down."""
    exc = getattr(args, "exc_value", None)
    if exc is None or isinstance(exc, SystemExit):
        return
    try:
        tname = getattr(getattr(args, "thread", None), "name", "?")
        _br.report_failure(
            summary=f"a background task crashed (thread '{tname}')",
            reason=str(exc), error=exc, context={"thread": tname},
            interactive=False)
    except Exception:
        try:
            threading.__excepthook__(args)
        except Exception:
            # Last resort: the bug reporter itself failed AND the stdlib default
            # hook failed. There is nothing left to fall back to, and an
            # excepthook must not raise (it would mask the original crash).
            pass


def install_global_handlers(force: bool = False) -> bool:
    """Install the process-wide graceful-failure net (main thread + background
    threads). Idempotent. A no-op under pytest unless *force* (so the test runner
    keeps its own exception/thread handling). Returns True if it installed."""
    if _br._handlers_installed:
        return False
    if not force and "pytest" in sys.modules:
        return False
    sys.excepthook = _handle_main_exception
    threading.excepthook = _handle_thread_exception
    _br._handlers_installed = True
    return True


def install_asyncio_handler(loop) -> bool:
    """Route an uncaught asyncio task/loop exception through the bug reporter
    (saved, never prompted - a server loop has no user at the keyboard). Returns
    True if installed. Non-exception loop messages fall through to the default."""
    import asyncio
    try:
        def _h(loop, ctx):
            exc = ctx.get("exception")
            if isinstance(exc, (KeyboardInterrupt, SystemExit,
                                asyncio.CancelledError)):
                return   # normal control flow, not a crash
            if exc is None:
                loop.default_exception_handler(ctx)   # message-only: keep default
                return
            try:
                _br.report_failure(
                    summary="an async task crashed",
                    reason=ctx.get("message") or str(exc),
                    error=exc, context={"asyncio": ctx.get("message", "")},
                    interactive=False)
            except Exception:
                loop.default_exception_handler(ctx)
        loop.set_exception_handler(_h)
        return True
    except Exception:
        return False
