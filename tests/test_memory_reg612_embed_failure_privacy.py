# SPDX-License-Identifier: AGPL-3.0-or-later
"""The embed-failure log must not leak memory CONTENT in privacy mode.

`_embed_one`'s except branch logs the record being embedded:

    _dbg.debug("memory embed_one failed for %r: %s", text[:80], e)

`text` is a memory record - a synthesized fact about the user, or an episodic
summary of their chat sessions. It is chat-derived CONTENT, and this writes a
snippet of it to the persisted debug log file.

Every other content-logging site in the codebase (llama.py, jobs/webtool.py)
gates raw content on `debug_content_enabled()`, which returns False in privacy
mode even when `debug_enabled()` is True. A plain `logger.debug` is gated only
on debug_enabled(), so privacy mode does NOT suppress it.

The leak: the user runs in privacy mode with the debug log on for operational
diagnostics (--debug / LOCALM_DEBUG / keep_diagnostics). A real embedder raises
inside embed_fn([text]) - a worker restart, an OOM, a dim mismatch - and their
memory content lands in a file on disk.

Re-silencing the branch is NOT the answer: the content is gated, the failure is
always logged.

The branch only runs when a REAL embedder raises from embed_fn; mock embedders
return vectors and never throw.
"""

from __future__ import annotations

import logging

import pytest

from localm.memory import MemoryStore

SECRET_TEXT = "User's bank PIN hint is his mother's maiden name Vandermeulen"


def _boom_embedder(texts):
    """A REAL embedder failing the way a real one does - a worker restart, an
    OOM, a dim mismatch. Mock embedders return vectors and never reach the
    except branch, which is exactly why the suite missed this."""
    raise RuntimeError("embedding worker died")


@pytest.fixture
def _debug_log(monkeypatch, caplog):
    """Debug log ON (the operational-diagnostics case), so the only thing
    deciding whether CONTENT is written is the content gate.

    The logger is `localm` (debuglog.py:37), NOT `localm.debug`. Capturing the
    wrong name made every leak assertion pass VACUOUSLY against an empty blob -
    a test that cannot fail. `_assert_capturing` below guards that regression."""
    monkeypatch.setattr("localm.debuglog.debug_enabled", lambda: True)
    caplog.set_level(logging.DEBUG, logger="localm")
    return caplog


def _embed_one(tmp_path, text=SECRET_TEXT):
    s = MemoryStore("owner", "chat", root=tmp_path)
    return s._embed_one(text, _boom_embedder)


def _captured_blob(caplog) -> str:
    """The captured debug output, PROVEN to be live capture.

    A "the secret is not in the log" assertion is worthless against an empty
    blob: it passes whether or not the leak exists. Every read of the log goes
    through here, which fails loudly if nothing was captured at all.
    """
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert blob.strip(), (
        "NOTHING was captured from the 'localm' logger, so a not-in-log "
        "assertion would pass vacuously. The embed failure must always be "
        "logged (rule 5), so an empty blob means the harness is broken or the "
        "failure was silently swallowed.")
    return blob


class TestPrivacyModeDoesNotLeakContent:
    @pytest.fixture(autouse=True)
    def _privacy(self, monkeypatch):
        monkeypatch.setattr("localm.debuglog.debug_content_enabled", lambda: False)

    def test_memory_content_is_not_written_to_the_debug_log(self, tmp_path, _debug_log):
        _embed_one(tmp_path)
        blob = _captured_blob(_debug_log)
        assert "Vandermeulen" not in blob, f"leaked memory content: {blob}"
        assert SECRET_TEXT not in blob, f"leaked memory content: {blob}"

    def test_no_fragment_of_the_content_survives(self, tmp_path, _debug_log):
        # Asserts on PREFIXES as well as the full string, so a leaked snippet is
        # caught too.
        _embed_one(tmp_path)
        blob = _captured_blob(_debug_log)
        for fragment in ("bank PIN", "maiden name", SECRET_TEXT[:40]):
            assert fragment not in blob, f"leaked {fragment!r}: {blob}"

    def test_the_failure_is_STILL_surfaced(self, tmp_path, _debug_log):
        """The CONTENT is gated, never the failure. Re-silencing the branch
        would pass the leak tests above while hiding a real embedder fault."""
        _embed_one(tmp_path)
        blob = _captured_blob(_debug_log)
        assert "embedding worker died" in blob, \
            f"the underlying error must still be reported: {blob}"

    def test_the_write_still_degrades_rather_than_raising(self, tmp_path, _debug_log):
        # A memory write must not crash on an embedder hiccup.
        assert _embed_one(tmp_path) is None


class TestNonPrivacyModeStillLogsContent:
    """NEGATIVE CASE: the gate must actually be a GATE, not a blanket removal. If
    the fix just deleted the content, these fail - and the debug log would lose a
    genuinely useful diagnostic for the non-privacy user it is meant for."""

    @pytest.fixture(autouse=True)
    def _not_privacy(self, monkeypatch):
        monkeypatch.setattr("localm.debuglog.debug_content_enabled", lambda: True)

    def test_content_is_logged_when_the_content_gate_allows_it(self, tmp_path,
                                                              _debug_log):
        _embed_one(tmp_path)
        blob = _captured_blob(_debug_log)
        assert "Vandermeulen" in blob, \
            f"content should be available to a non-privacy debug session: {blob}"

    def test_the_failure_is_reported_here_too(self, tmp_path, _debug_log):
        _embed_one(tmp_path)
        blob = _captured_blob(_debug_log)
        assert "embedding worker died" in blob


class TestGateIsTheRealOne:
    """Guards the wiring itself: the fix must consult debug_content_enabled(),
    the shared privacy gate, not re-implement its own weaker check."""

    def test_the_content_gate_is_actually_consulted(self, tmp_path, monkeypatch,
                                                    _debug_log):
        calls = []

        def _tracked():
            calls.append(1)
            return False

        monkeypatch.setattr("localm.debuglog.debug_content_enabled", _tracked)
        _embed_one(tmp_path)
        assert calls, "debug_content_enabled() was never consulted"
