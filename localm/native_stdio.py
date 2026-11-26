# SPDX-License-Identifier: AGPL-3.0-or-later
"""Keeping what native code prints on stdout and stderr off the console: the
banner a ggml build prints while its library loads and registers its backends
(device and ``load_backend:`` lines). In debug mode it goes to the debug log;
a load error can quote its tail."""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Optional

from localm import debuglog


class CapturedNativeStdio:
    """Holder yielded by :func:`capture_native_stdio`; ``tail()`` reads whatever
    landed in its file from offset *start* on while the block was open."""

    def __init__(self, path: str, start: int = 0) -> None:
        self.path = path
        self._start = start

    def tail(self, max_chars: int = 1500) -> str:
        """The last *max_chars* of the captured text, "" when nothing was
        captured or the file cannot be read."""
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self._start)
                data = fh.read()
        except OSError:
            return ""
        text = data.decode("utf-8", errors="replace").strip()
        return text[-max_chars:] if len(text) > max_chars else text


def _flush_python_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()


@contextlib.contextmanager
def capture_native_stdio(keep_path: Optional[str] = None):
    """Redirect fd 1 (stdout) and fd 2 (stderr) into one file for the block.
    Yields a :class:`CapturedNativeStdio`.

    In debug mode the descriptors point straight at the debug log (appended), so
    the text is there even when the process dies inside the block. Otherwise
    they point at *keep_path* (truncated first) or a new temp file, which is
    removed when the block ends; a process that dies inside the block leaves
    it, so a parent that named *keep_path* can read it. Read ``tail()`` inside
    the block.

    This only moves the OS-level descriptors. A logging record emitted in the
    block reaches the debug-mode console mirror, which writes to its own
    duplicate of stderr; pair it with ``debuglog.suppress_console_mirror`` to
    keep those off the console too."""
    fd = None
    start = 0
    remove = True
    target = ""
    path = debuglog.log_file_path()
    if path is not None:
        debuglog.drain_debug_log()
        try:
            fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT)
            start = os.fstat(fd).st_size
            remove = False
            target = str(path)
        except OSError as e:
            debuglog.logger.warning(
                "native output: could not open the debug log %s (%s); it goes to a "
                "temp file instead", path, e)
            fd = None
    if fd is None:
        if keep_path:
            target = keep_path
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        else:
            import tempfile
            fd, target = tempfile.mkstemp(prefix="localm_loadlib_", suffix=".log")
    _flush_python_stdio()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    try:
        yield CapturedNativeStdio(target, start)
    finally:
        _flush_python_stdio()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        if remove:
            with contextlib.suppress(OSError):
                os.unlink(target)
