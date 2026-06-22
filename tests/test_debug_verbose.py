# SPDX-License-Identifier: AGPL-3.0-or-later
"""SRV-5: debug mode must make the SERVER CONSOLE verbose - mirror debug logs to
stderr and raise uvicorn's log level - not just write to a file."""

import logging

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
