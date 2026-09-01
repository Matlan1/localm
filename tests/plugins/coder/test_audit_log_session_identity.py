# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two coder sessions created close together, in the same process, must not
share an audit log file.

AuditLog named its file by timestamp+pid alone (second resolution, pid shared
by every session in one server process). Two CoderSessions constructed within
the same second landed on the identical filename and their JSONL records
interleaved into one file, so GET /api/coder/sessions/{id}/log returned the
same merged content for both sessions regardless of which one, or which
project, actually produced a given line.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from localm.plugins.coder.sessions import CoderSession


class _StubBackend:
    model_id = "stub-model"
    native_tools = False

    def set_tools(self, defs):
        pass


def _session(cwd: Path, **kw) -> CoderSession:
    return CoderSession(cwd, _StubBackend(), mode="log", auto_verify=False, **kw)


def test_sessions_in_different_projects_get_distinct_audit_logs(tmp_path):
    proj_a = tmp_path / "a"
    proj_b = tmp_path / "b"
    proj_a.mkdir()
    proj_b.mkdir()
    with patch("localm.audit._SESSIONS_DIR", tmp_path / "sessions"), \
         patch("time.strftime", return_value="2026-01-01_000000"):
        s1 = _session(proj_a)
        s2 = _session(proj_b)
    try:
        assert s1.agent._checkpoint_id != s2.agent._checkpoint_id
        assert s1.audit_log_path() != s2.audit_log_path()
        assert s1.audit_log_path().exists()
        assert s2.audit_log_path().exists()
    finally:
        s1.agent.close()
        s2.agent.close()


def test_sessions_in_the_same_project_get_distinct_audit_logs(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    with patch("localm.audit._SESSIONS_DIR", tmp_path / "sessions"), \
         patch("time.strftime", return_value="2026-01-01_000000"):
        s1 = _session(proj)
        s2 = _session(proj)
    try:
        assert s1.audit_log_path() != s2.audit_log_path()
    finally:
        s1.agent.close()
        s2.agent.close()


def test_each_sessions_log_holds_only_its_own_records(tmp_path):
    """The reported symptom: two sessions' logs must not merge, and a
    consumer reading either file back can attribute every line."""
    proj_a = tmp_path / "a"
    proj_b = tmp_path / "b"
    proj_a.mkdir()
    proj_b.mkdir()
    with patch("localm.audit._SESSIONS_DIR", tmp_path / "sessions"), \
         patch("time.strftime", return_value="2026-01-01_000000"):
        s1 = _session(proj_a)
        s2 = _session(proj_b)
        s1.agent._audit.notice("test", "from session A")
        s2.agent._audit.notice("test", "from session B")
        s1.agent.close()
        s2.agent.close()

    entries_a = [json.loads(line) for line in
                 s1.audit_log_path().read_text(encoding="utf-8").splitlines()]
    entries_b = [json.loads(line) for line in
                 s2.audit_log_path().read_text(encoding="utf-8").splitlines()]

    assert all(e["session"] == s1.agent._checkpoint_id for e in entries_a)
    assert all(e["session"] == s2.agent._checkpoint_id for e in entries_b)
    assert not any(e["data"].get("message") == "from session B" for e in entries_a)
    assert not any(e["data"].get("message") == "from session A" for e in entries_b)
