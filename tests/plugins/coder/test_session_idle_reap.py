# SPDX-License-Identifier: AGPL-3.0-or-later
"""SessionManager.reap_idle() closes and removes a session that has gone idle (no
_push() activity) past a threshold, so a GUI coder session the user abandons
without an explicit DELETE - a closed tab, a killed browser - still writes the
"session ended" audit record a normally-ended session gets.

It is called opportunistically from list(), so no background thread or shutdown
hook is involved.
"""

from __future__ import annotations

from pathlib import Path

from localm.plugins.coder.sessions import CoderSession, SessionManager, _IDLE_REAP_SECONDS


class _StubBackend:
    model_id = "stub-model"
    native_tools = False

    def set_tools(self, defs):
        pass


def _session(tmp_path: Path) -> CoderSession:
    return CoderSession(tmp_path, _StubBackend(), mode="privacy", auto_verify=False)


def test_reap_idle_leaves_a_recently_active_session_alone(tmp_path):
    mgr = SessionManager()
    s = mgr.create(_session(tmp_path))
    now = s.last_activity_at + 10   # well under the threshold
    assert mgr.reap_idle(max_idle_s=_IDLE_REAP_SECONDS, now=now) == []
    assert mgr.get(s.id) is not None
    assert s.closed is False


def test_reap_idle_closes_and_removes_a_session_past_the_threshold(tmp_path):
    """An abandoned session is closed and removed from
    SessionManager._sessions."""
    mgr = SessionManager()
    s = mgr.create(_session(tmp_path))
    now = s.last_activity_at + _IDLE_REAP_SECONDS + 1
    assert mgr.reap_idle(max_idle_s=_IDLE_REAP_SECONDS, now=now) == [s.id]
    assert mgr.get(s.id) is None
    assert s.closed is True


def test_reap_idle_never_touches_a_busy_session(tmp_path):
    """A session running a long task (a slow model, a big verify) is not
    reaped, even when it has pushed no new event for longer than the
    threshold."""
    mgr = SessionManager()
    s = mgr.create(_session(tmp_path))
    s.busy = True
    now = s.last_activity_at + _IDLE_REAP_SECONDS + 1
    assert mgr.reap_idle(max_idle_s=_IDLE_REAP_SECONDS, now=now) == []
    assert mgr.get(s.id) is not None
    assert s.closed is False


def test_a_push_resets_the_idle_clock(tmp_path):
    s = _session(tmp_path)
    stale = s.last_activity_at
    s._push({"type": "info", "text": "still here"})
    assert s.last_activity_at > stale


def test_list_opportunistically_reaps_without_a_background_thread(tmp_path):
    """An ordinary list() call performs the sweep; there is no timer to start
    or stop."""
    mgr = SessionManager()
    s = mgr.create(_session(tmp_path))
    s.last_activity_at -= (_IDLE_REAP_SECONDS + 1)   # force idle, no real wait
    assert mgr.list() == []
    assert mgr.get(s.id) is None
