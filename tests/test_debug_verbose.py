# SPDX-License-Identifier: AGPL-3.0-or-later
"""Debug mode must make the SERVER CONSOLE verbose: mirror debug logs to stderr
and raise uvicorn's log level, not just write to a file."""

import logging
import os
import sys

import pytest

from localm import debuglog


def _console_handlers():
    return [h for h in debuglog.logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)]


def test_uvicorn_log_level_follows_debug(monkeypatch):
    monkeypatch.delenv("LOCALM_DEBUG", raising=False)
    assert debuglog.uvicorn_log_level() == "warning"
    monkeypatch.setenv("LOCALM_DEBUG", "1")
    assert debuglog.uvicorn_log_level() == "info"


def test_console_handler_added_once():
    saved = list(debuglog.logger.handlers)
    try:
        # baseline: drop any existing console handler
        debuglog.logger.handlers = [
            h for h in debuglog.logger.handlers if h not in _console_handlers()]
        debuglog._add_console_handler()
        debuglog._add_console_handler()
        assert len(_console_handlers()) == 1   # idempotent, not stacked
    finally:
        debuglog.logger.handlers = saved


def test_console_stream_is_a_private_stderr_dup():
    """The console mirror must write through a PRIVATE duplicate of stderr,
    not fd 2 itself. A distinct fileno() proves the isolation."""
    stream = debuglog._stable_console_stream()
    if stream is None:
        pytest.skip("stderr has no duplicable fd in this environment")
    try:
        assert stream.fileno() != sys.stderr.fileno()
    finally:
        stream.close()


def test_console_stream_survives_fd_redirect(tmp_path, monkeypatch):
    """The mirror must keep writing WHILE its underlying fd is redirected
    elsewhere (what _quiet_stderr/_capture_stderr do to fd 2 around a model load
    or generation). Run against a stand-in stderr, never the real fd 2 or
    pytest's own capture."""
    standin = open(tmp_path / "stderr.txt", "w", encoding="utf-8")
    monkeypatch.setattr(sys, "stderr", standin)
    stream = debuglog._stable_console_stream()
    assert stream is not None
    assert stream.fileno() != standin.fileno()        # a genuine private dup
    saved = os.dup(standin.fileno())
    other = open(tmp_path / "other.txt", "w", encoding="utf-8")
    try:
        os.dup2(other.fileno(), standin.fileno())     # redirect the fd, as the backend does
        stream.write("mirror-still-alive\n")          # must NOT raise (private dup)
        stream.flush()
    finally:
        os.dup2(saved, standin.fileno())
        os.close(saved)
        other.close()
        stream.close()
        standin.close()
