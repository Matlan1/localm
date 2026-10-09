# SPDX-License-Identifier: AGPL-3.0-or-later
"""A coder session's JSONL audit log is deletable once the session is closed.

On Windows an open handle blocks deleting the file, so these assert on the
file itself: after the session ends, the log can be unlinked.
"""

from pathlib import Path

import pytest

from localm.plugins.coder.sessions import CoderSession, SessionManager


class _StubBackend:
    model_id = "stub-model"
    native_tools = False

    def set_tools(self, defs):
        pass


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr("localm.audit._SESSIONS_DIR", home / "sessions")
    return home / "sessions"


@pytest.fixture
def open_sessions(monkeypatch):
    """Closes any session a test leaves open, so a failing assertion cannot
    leave a handle behind."""
    opened = []
    real_init = CoderSession.__init__

    def _recording_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        opened.append(self)

    monkeypatch.setattr(CoderSession, "__init__", _recording_init)
    yield
    for session in opened:
        if not session.closed:
            session.close()
        try:
            session.agent._audit.close()
        except Exception:
            pass


def _session(tmp_path: Path) -> CoderSession:
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    return CoderSession(proj, _StubBackend(), mode="log", auto_verify=False)


def _assert_deletable(path: Path) -> None:
    assert path.exists()
    path.unlink()
    assert not path.exists()


def test_log_is_deletable_after_session_close(tmp_path, sessions_dir, open_sessions):
    session = _session(tmp_path)
    log = session.audit_log_path()
    assert log is not None and log.parent == sessions_dir
    session.close()
    _assert_deletable(log)


def test_log_is_deletable_when_storing_the_episode_raises(
        tmp_path, sessions_dir, open_sessions, monkeypatch):
    session = _session(tmp_path)
    log = session.audit_log_path()

    def _boom(*a, **kw):
        raise RuntimeError("episode storage failed")

    monkeypatch.setattr(type(session.agent), "_maybe_store_episode", _boom)
    session.close()
    _assert_deletable(log)


def test_manager_remove_releases_the_log(tmp_path, sessions_dir, open_sessions):
    manager = SessionManager()
    session = manager.create(_session(tmp_path))
    log = session.audit_log_path()
    manager.remove(session.id)
    _assert_deletable(log)


def test_manager_close_all_releases_every_log(tmp_path, sessions_dir, open_sessions):
    manager = SessionManager()
    logs = []
    for _ in range(2):
        s = manager.create(_session(tmp_path))
        logs.append(s.audit_log_path())
    assert logs[0] != logs[1]
    assert manager.close_all() == 2
    for log in logs:
        _assert_deletable(log)


def test_manager_reap_idle_releases_the_log(tmp_path, sessions_dir, open_sessions):
    manager = SessionManager()
    session = manager.create(_session(tmp_path))
    log = session.audit_log_path()
    reaped = manager.reap_idle(max_idle_s=0, now=session.last_activity_at + 10)
    assert reaped == [session.id]
    _assert_deletable(log)
