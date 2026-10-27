# SPDX-License-Identifier: AGPL-3.0-or-later
"""A background model-preload failure (localm gui's automatic warm-up load at
startup) must be LOGGED, not only printed to console.

console.print never reaches the debug log file, so a user whose only symptom is a
failed preload (they never explicitly try to chat) would file a bug report showing
nothing wrong. _report_preload_failure logs the exception with its full traceback
in addition to the console notice.
"""

from __future__ import annotations

import logging

from localm.plugins.gui.cli import _report_preload_failure


class _FakeConsole:
    def __init__(self):
        self.printed = []

    def print(self, msg):
        self.printed.append(msg)


def test_preload_failure_is_logged_with_traceback(caplog):
    console = _FakeConsole()
    try:
        raise RuntimeError("Native llama runtime failed to load: [WinError 2]")
    except RuntimeError as e:
        with caplog.at_level(logging.ERROR, logger="localm"):
            _report_preload_failure(console, e)

    # Notifies the console.
    assert any("Background model load failed" in m for m in console.printed)

    # Reaches the logger, which the bug-report digest reads from disk.
    records = [r for r in caplog.records if r.name == "localm"]
    assert any("background model preload failed" in r.message for r in records)
    assert any(r.exc_info for r in records)   # the traceback is attached


def test_an_expected_refusal_is_one_warning_line_without_a_traceback(caplog):
    from localm.inference.backends.base import ModelRefusedError
    console = _FakeConsole()
    try:
        raise ModelRefusedError("This model is quantized with EXL2 and needs an NVIDIA GPU.")
    except ModelRefusedError as e:
        with caplog.at_level(logging.DEBUG, logger="localm"):
            _report_preload_failure(console, e)

    assert any("EXL2" in m for m in console.printed)
    records = [r for r in caplog.records if r.name == "localm"]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert "EXL2" in records[0].getMessage()
    assert not records[0].exc_info
