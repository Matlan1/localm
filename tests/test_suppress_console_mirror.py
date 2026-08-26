# SPDX-License-Identifier: AGPL-3.0-or-later
"""suppress_console_mirror(): temporarily detach the debug-mode console-
mirroring StreamHandler (see debuglog._add_console_handler) so a log record
emitted inside the block reaches only the FILE handler, never the shared
terminal.

The mirror's own stream (_stable_console_stream) is immune to the OS-level
fd-2 redirect the llamacpp backend uses to silence native stderr, so without
this a debug-mode log call from inside the GGUF chat backend's isolated child
(_apply_cpu_moe's _dbg.info) writes straight to the shared terminal, racing the
parent's Rich load-spinner. tests/test_moe_placement_report.py covers
LlamaCpp.__init__'s merged native-call scope engaging this.
"""

import logging

from localm import debuglog


_saved_level = None


def setup_function():
    # Start every test from a known handler set: no console mirror attached,
    # matching debug-mode-off (the common case for a plain pytest run).
    for h in list(debuglog.logger.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            debuglog.logger.removeHandler(h)
    # logger.level defaults to NOTSET, which INHERITS the root logger's WARNING
    # threshold, so a bare .info() call would be filtered before reaching a
    # handler. DEBUG matches the state these tests model: attach_child_logging
    # and enable_debug both set exactly this level.
    global _saved_level
    _saved_level = debuglog.logger.level
    debuglog.logger.setLevel(logging.DEBUG)


def teardown_function():
    for h in list(debuglog.logger.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            debuglog.logger.removeHandler(h)
    debuglog.logger.setLevel(_saved_level)


def _mirror_handler(stream):
    h = logging.StreamHandler(stream)
    h.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
    return h


def test_noop_when_no_mirror_is_attached():
    """Debug mode off (or never enabled): nothing to suppress, nothing to
    restore - must not raise, and must not touch an absent handler set."""
    before = list(debuglog.logger.handlers)
    with debuglog.suppress_console_mirror():
        pass
    assert debuglog.logger.handlers == before


def test_mirror_receives_nothing_while_suppressed():
    import io

    stream = io.StringIO()
    mirror = _mirror_handler(stream)
    debuglog.logger.addHandler(mirror)
    try:
        with debuglog.suppress_console_mirror():
            debuglog.logger.info("this must not reach the mirror")
        assert stream.getvalue() == "", (
            f"a record emitted inside the block reached the mirror anyway: "
            f"{stream.getvalue()!r}")
    finally:
        debuglog.logger.removeHandler(mirror)


def test_mirror_is_restored_and_receives_records_again_after_the_block():
    import io

    stream = io.StringIO()
    mirror = _mirror_handler(stream)
    debuglog.logger.addHandler(mirror)
    try:
        with debuglog.suppress_console_mirror():
            debuglog.logger.info("suppressed")
        debuglog.logger.info("this one must reach the mirror")
        assert "suppressed" not in stream.getvalue()
        assert "this one must reach the mirror" in stream.getvalue()
    finally:
        debuglog.logger.removeHandler(mirror)


def test_file_handler_is_never_touched(tmp_path):
    """The file handler (the persisted debug-log record) must keep receiving
    every record regardless - this only ever narrows the LIVE view, matching
    dedup_native_stderr's own "never silently drop a record" contract. A
    REAL FileHandler, not a StreamHandler standing in for one: this is
    exactly the isinstance(..., FileHandler) distinction
    suppress_console_mirror (and _add_console_handler before it) relies on
    to tell the two apart."""
    log_path = tmp_path / "debug.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    debuglog.logger.addHandler(file_handler)
    try:
        with debuglog.suppress_console_mirror():
            debuglog.logger.info("must still reach the file handler")
        file_handler.flush()
        assert "must still reach the file handler" in log_path.read_text(encoding="utf-8")
    finally:
        debuglog.logger.removeHandler(file_handler)
        file_handler.close()


def test_handler_object_identity_is_preserved_across_the_block():
    """Restoring must re-attach the SAME handler instance, not a new one with
    equivalent config - a caller (or another test) holding a reference to the
    original handler must still see it live in logger.handlers afterward."""
    import io

    mirror = _mirror_handler(io.StringIO())
    debuglog.logger.addHandler(mirror)
    try:
        with debuglog.suppress_console_mirror():
            assert mirror not in debuglog.logger.handlers
        assert mirror in debuglog.logger.handlers
    finally:
        debuglog.logger.removeHandler(mirror)


def test_exception_inside_the_block_still_restores_the_mirror():
    import io

    mirror = _mirror_handler(io.StringIO())
    debuglog.logger.addHandler(mirror)
    try:
        try:
            with debuglog.suppress_console_mirror():
                raise ValueError("boom")
        except ValueError:
            pass
        assert mirror in debuglog.logger.handlers, (
            "an exception inside the block left the mirror permanently "
            "detached - every subsequent debug-mode log line would silently "
            "stop reaching the console")
    finally:
        debuglog.logger.removeHandler(mirror)
