# SPDX-License-Identifier: AGPL-3.0-or-later
"""Import-time diagnostics must survive until a log handler exists.

The `except ImportError` branch of `_wire_plugin_cli_entries` runs at
module-import time, before Click invokes main() to install the ring buffer and
before any enable_debug(). At that moment the localm logger has no handler and
inherits the root's WARNING level, so a `logger.debug(...)` record is dropped AT
THE CALL.

These tests do NOT use caplog: caplog attaches a root handler and forces level
0, which masks the production condition under test (no handler, level inherited
from root).
"""
import logging

import pytest


@pytest.fixture
def pristine_logging(monkeypatch):
    """Reproduce the REAL import-time logging state, and restore it afterwards.

    The window being reproduced is "no handler on the localm logger, level
    NOTSET so the root's WARNING is what is in force". Two things make a naive
    reset silently wrong:

    1. logging.Logger.isEnabledFor() MEMOISES its answer per level in
       Logger._cache, and only setLevel()/_clear_cache() invalidate it. Assigning
       `.level` directly (e.g. monkeypatch.setattr(logger, "level", NOTSET))
       leaves a stale entry, so after any earlier test in the same worker called
       enable_debug() (test_debug_scrub.py does), isEnabledFor(DEBUG) keeps
       answering True and the window is NOT reproduced.
    2. enable_debug() is idempotent via the LOCALM_DEBUG env var: if an earlier
       test left it set, enable_debug() returns the OLD path without opening a
       new log, and assertions read the wrong file.

    Does NOT use caplog, which attaches a root handler and forces level 0.
    """
    from localm import debuglog
    root = logging.getLogger()
    saved_level = debuglog.logger.level
    saved_handlers = list(debuglog.logger.handlers)
    saved_root_level = root.level

    monkeypatch.delenv("LOCALM_DEBUG", raising=False)   # else enable_debug() no-ops
    debuglog.logger.handlers = []
    debuglog.logger.setLevel(logging.NOTSET)            # setLevel() clears the cache
    root.setLevel(logging.WARNING)                      # the real default at import
    logging.Logger.manager._clear_cache()
    try:
        yield
    finally:
        # enable_debug() adds a file handler and lowers the level; leaving that in
        # place would pollute every later test in this worker.
        debuglog.logger.handlers = saved_handlers
        debuglog.logger.setLevel(saved_level)
        root.setLevel(saved_root_level)
        logging.Logger.manager._clear_cache()


class TestDeferLog:
    def test_defer_log_survives_the_no_handler_import_window(
            self, tmp_path, monkeypatch, pristine_logging):
        """A record queued while the logger has NO handler and an inherited
        WARNING level still reaches the debug log once enable_debug() runs."""
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        from localm import debuglog

        # Precondition: a DEBUG record is dropped in this state.
        assert not debuglog.logger.isEnabledFor(logging.DEBUG), (
            "precondition: a DEBUG record must be dropped in this state - if this "
            "fails the test is no longer reproducing the import-time window"
        )
        monkeypatch.setattr(debuglog, "_deferred_records", [])

        debuglog.defer_log(logging.DEBUG, "queued %s here", "diagnostic")

        # Nothing emitted yet, but nothing lost either.
        assert len(debuglog._deferred_records) == 1

        path = debuglog.enable_debug()
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "queued diagnostic here" in text
        # Drained, so a second enable_debug() cannot double-replay it.
        assert debuglog._deferred_records == []

    def test_flush_is_idempotent(self, tmp_path, monkeypatch, pristine_logging):
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        from localm import debuglog
        monkeypatch.setattr(debuglog, "_deferred_records", [])
        debuglog.defer_log(logging.DEBUG, "once")
        debuglog._flush_deferred()
        debuglog._flush_deferred()   # no records left; must not raise
        assert debuglog._deferred_records == []

    def test_queue_is_bounded(self, monkeypatch):
        from localm import debuglog
        monkeypatch.setattr(debuglog, "_deferred_records", [])
        for i in range(250):
            debuglog.defer_log(logging.DEBUG, "msg %d", i)
        assert len(debuglog._deferred_records) <= 100


class TestPluginCliWiringDiagnostic:
    def test_broken_plugin_cli_import_is_recorded_not_silent(self, monkeypatch):
        """A first-party plugin CLI that fails to import must leave a trace.

        Drives the REAL _wire_plugin_cli_entries() with a forced ImportError
        and asserts the diagnostic lands in the deferred queue, the mechanism
        that works at import scope.
        """
        import importlib

        from localm import debuglog
        from localm.cli import maintenance

        monkeypatch.setattr(debuglog, "_deferred_records", [])

        def _boom(mod_name, *a, **kw):
            raise ImportError(f"No module named {mod_name!r}")

        monkeypatch.setattr(importlib, "import_module", _boom)
        maintenance._wire_plugin_cli_entries()

        assert debuglog._deferred_records, (
            "a broken first-party plugin CLI vanished with no trace - the "
            "import-failure diagnostic was lost"
        )
        rendered = [msg % args for _lvl, msg, args in debuglog._deferred_records]
        assert any("not wired (import failed)" in r for r in rendered)


@pytest.mark.parametrize("entry", ["logger.debug(", "logger.info(", "logger.warning("])
def test_no_direct_logger_call_at_import_scope_in_wiring(entry):
    """Guard the class of bug, not just the instance: the wiring function runs at
    import scope, so a direct logger.* call inside it can never emit."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "localm" / "cli" / "maintenance.py"
    body = src.read_text(encoding="utf-8")
    start = body.index("def _wire_plugin_cli_entries")
    end = body.index("_wire_plugin_cli_entries()", start)
    assert entry not in body[start:end], (
        f"{entry} inside _wire_plugin_cli_entries runs before any log handler "
        "exists and is dropped at the call - use defer_log() instead"
    )
