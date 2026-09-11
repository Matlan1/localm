# SPDX-License-Identifier: AGPL-3.0-or-later
"""install_ring_buffer() must not silence the console entirely.

Before this fix, install_ring_buffer() added the FIRST handler ever attached
to the "localm" logger. CPython's Logger.callHandlers counts a handler before
the level test and only fires logging.lastResort (the stderr WARNING+
fallback) when it finds zero handlers - so that one stream-less ring handler
silenced every WARNING+ record on the console for the life of the process,
on every ordinary (non-debug) run.

The fix makes install_ring_buffer() also attach a real WARNING+ console
mirror via _add_console_handler(), while INFO stays console-silent (it still
reaches the ring buffer) and a later enable_debug()/attach_child_logging()
can still lower that same handler to show everything.
"""

import logging
import subprocess
import sys

import pytest

from localm import debuglog


def _reset_logger_state():
    """Strip every handler and level override so a test starts from the
    same state a fresh interpreter would: no handlers, level NOTSET."""
    debuglog.logger.handlers = []
    debuglog.logger.setLevel(logging.NOTSET)
    debuglog._ring_handler = None


@pytest.fixture
def clean_logger():
    saved_handlers = list(debuglog.logger.handlers)
    saved_level = debuglog.logger.level
    saved_ring = debuglog._ring_handler
    _reset_logger_state()
    try:
        yield
    finally:
        debuglog.logger.handlers = saved_handlers
        debuglog.logger.setLevel(saved_level)
        debuglog._ring_handler = saved_ring


def test_warning_reaches_console_after_install(clean_logger, tmp_path, monkeypatch):
    standin = open(tmp_path / "stderr.txt", "w", encoding="utf-8")
    monkeypatch.setattr(sys, "stderr", standin)
    try:
        debuglog.install_ring_buffer()
        logging.getLogger("localm.netpolicy").warning("PROBE-WARN-7Q4M")
        logging.getLogger("localm.netpolicy").info("PROBE-INFO-7Q4M")
        for h in debuglog.logger.handlers:
            h.flush()
        standin.flush()
    finally:
        standin.close()
    written = (tmp_path / "stderr.txt").read_text(encoding="utf-8")
    assert "PROBE-WARN-7Q4M" in written
    assert "PROBE-INFO-7Q4M" not in written
    joined = "\n".join(debuglog.recent_activity())
    assert "PROBE-WARN-7Q4M" in joined
    assert "PROBE-INFO-7Q4M" in joined


def test_debug_mode_still_shows_info_through_the_same_handler(clean_logger, tmp_path, monkeypatch):
    """enable_debug()'s own _add_console_handler() call must lower the
    WARNING mirror install_ring_buffer() created, not stack a second one."""
    standin = open(tmp_path / "stderr.txt", "w", encoding="utf-8")
    monkeypatch.setattr(sys, "stderr", standin)
    try:
        debuglog.install_ring_buffer()
        debuglog._add_console_handler()  # what enable_debug()/attach_child_logging() do
        logging.getLogger("localm.netpolicy").info("PROBE-INFO-DEBUG-9K2L")
        for h in debuglog.logger.handlers:
            h.flush()
        standin.flush()
    finally:
        standin.close()
    written = (tmp_path / "stderr.txt").read_text(encoding="utf-8")
    assert "PROBE-INFO-DEBUG-9K2L" in written
    non_file_stream_handlers = [
        h for h in debuglog.logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    assert len(non_file_stream_handlers) == 1


def test_two_arm_subprocess_control_proves_the_regression_and_the_fix():
    """Control arm (no install_ring_buffer): a bare warning must reach
    stderr on its own - this proves the instrument can see the warning at
    all, so the ring arm's assertion means something. Ring arm: the same
    warning must still reach stderr after install_ring_buffer()."""
    probe = "PROBE-SUBPROC-WARN-4M8X"
    control_code = (
        "import logging, sys\n"
        "logging.getLogger('localm.netpolicy').warning(%r)\n" % probe
    )
    ring_code = (
        "import logging, sys\n"
        "from localm import debuglog\n"
        "debuglog.install_ring_buffer()\n"
        "logging.getLogger('localm.netpolicy').warning(%r)\n" % probe
    )
    control = subprocess.run(
        [sys.executable, "-c", control_code],
        capture_output=True, text=True, timeout=30,
    )
    assert probe in control.stderr, (
        f"control arm did not see its own warning on stderr; "
        f"the instrument cannot see this warning at all: {control.stderr!r}"
    )
    ring = subprocess.run(
        [sys.executable, "-c", ring_code],
        capture_output=True, text=True, timeout=30,
    )
    assert probe in ring.stderr, (
        f"install_ring_buffer() silenced a WARNING that reaches stderr "
        f"without it: {ring.stderr!r}"
    )
