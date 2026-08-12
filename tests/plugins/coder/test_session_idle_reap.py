# SPDX-License-Identifier: AGPL-3.0-or-later
"""NEW-CODER-NO-TOOLCALL-SILENT residual (E): a GUI coder session the user
abandons without an explicit DELETE - a closed tab, a killed browser - never
triggered close(), so it never wrote the "session ended" audit record a
normally-ended session gets. No idle reaper existed at all.

SessionManager.reap_idle() closes and removes a session that has gone idle
(no _push() activity) past a threshold, called opportunistically from list()
so no new background thread or shutdown hook is needed.
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
    """THE DEFECT. Before reap_idle existed, nothing ever called close() on an
    abandoned session - it just sat in SessionManager._sessions forever."""
    mgr = SessionManager()
    s = mgr.create(_session(tmp_path))
    now = s.last_activity_at + _IDLE_REAP_SECONDS + 1
    assert mgr.reap_idle(max_idle_s=_IDLE_REAP_SECONDS, now=now) == [s.id]
    assert mgr.get(s.id) is None
    assert s.closed is True


def test_reap_idle_never_touches_a_busy_session(tmp_path):
    """A long-running task (a slow model, a big verify) is still very much
    alive even though it has pushed no NEW event in a while at the exact
    moment reap_idle runs - reaping it mid-task would corrupt the run."""
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
    """No timer to start or stop: an ordinary list() call (the GUI already
    polls this for the sidebar) is what performs the sweep."""
    mgr = SessionManager()
    s = mgr.create(_session(tmp_path))
    s.last_activity_at -= (_IDLE_REAP_SECONDS + 1)   # force idle, no real wait
    assert mgr.list() == []
    assert mgr.get(s.id) is None
