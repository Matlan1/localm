# SPDX-License-Identifier: AGPL-3.0-or-later
"""Import-time diagnostics must survive until a log handler exists.

Regression cover for the dud fix that shipped in #637: the
`except ImportError` branch of `_wire_plugin_cli_entries` was given a
`logger.debug(...)` that could NEVER emit, because the function runs at
module-import time - before Click invokes main() to install the ring buffer and
before any enable_debug(). At that moment the localm logger has no handler and
inherits the root's WARNING level, so a DEBUG record is dropped AT THE CALL.

These tests deliberately do NOT use caplog: caplog attaches a root handler and
forces level 0, which MASKS the exact production condition (no handler, level
inherited from root) that made the original fix dead code. A caplog-based test
would have false-passed on the broken code.
"""
import logging

import pytest


class TestDeferLog:
    def test_defer_log_survives_the_no_handler_import_window(self, tmp_path, monkeypatch):
        """A record queued while the logger has NO handler and an inherited
        WARNING level still reaches the debug log once enable_debug() runs."""
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        from localm import debuglog

        # Reproduce the real import-time state: no handlers, level NOTSET so the
        # root's WARNING is what is in force. This is what kills a bare
        # logger.debug() - assert that precondition rather than assuming it.
        monkeypatch.setattr(debuglog.logger, "handlers", [])
        monkeypatch.setattr(debuglog.logger, "level", logging.NOTSET)
        assert not debuglog.logger.isEnabledFor(logging.DEBUG), (
            "precondition: a DEBUG record must be dropped in this state - if this "
            "fails the test is no longer reproducing the import-time window"
        )
        monkeypatch.setattr(debuglog, "_deferred_records", [])

        debuglog.defer_log(logging.DEBUG, "queued %s here", "diagnostic")

        # Nothing emitted yet, but nothing lost either.
        assert len(debuglog._deferred_records) == 1

        monkeypatch.delenv("LOCALM_DEBUG_FILE", raising=False)
        path = debuglog.enable_debug()
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "queued diagnostic here" in text
        # Drained, so a second enable_debug() cannot double-replay it.
        assert debuglog._deferred_records == []

    def test_flush_is_idempotent(self, tmp_path, monkeypatch):
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

        Drives the REAL _wire_plugin_cli_entries() with a forced ImportError and
        asserts the diagnostic lands in the deferred queue - i.e. it is recorded
        through a mechanism that actually works at import scope, rather than a
        logger.debug() that is dropped on the floor.
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
