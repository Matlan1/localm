# SPDX-License-Identifier: AGPL-3.0-or-later
"""The resume checkpoint lives under HOME/checkpoints/<digest>.json, not in the
project tree (<cwd>/.localcoder/checkpoint.json). The legacy in-project path is
still READ (back-compat) and cleared. Project-local config.toml, LOCALCODER.md,
and full-mode transcripts are unaffected (not exercised here)."""

import json
from unittest.mock import patch

from localm.plugins.coder.audit import SessionMode


class _Stub:
    model_id = "m"
    native_tools = False
    supports_grammar = False
    last_usage = {"total_tokens": 0}

    def chat(self, messages, **kw):
        return "Done."

    def chat_stream(self, messages, **kw):
        yield "Done."


def _agent(cwd, **kw):
    from localm.plugins.coder.agent import Agent
    with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        PM.build.return_value.file_count.return_value = 0
        PM.build.return_value.truncated = False
        return Agent(_Stub(), cwd=cwd, auto_approve=True, self_verify=False, **kw)


def test_checkpoint_written_under_home_not_project(tmp_path, monkeypatch):
    home = tmp_path / "home"
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    proj = tmp_path / "proj"; proj.mkdir()

    a = _agent(proj, mode=SessionMode.LOG)
    a._messages = [{"role": "user", "content": "hi"}]
    a.save_checkpoint()

    cp = a._checkpoint_path
    assert home in cp.parents                 # under HOME/checkpoints/...
    assert cp.is_file()
    assert not (proj / ".localcoder").exists()  # NOTHING left in the project tree


def test_legacy_in_project_checkpoint_is_read_then_cleared(tmp_path, monkeypatch):
    home = tmp_path / "home"
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    proj = tmp_path / "proj"; proj.mkdir()

    legacy = proj / ".localcoder" / "checkpoint.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({
        "version": 1, "turns": 2, "total_tokens": 5,
        "messages": [{"role": "user", "content": "x"}],
    }), encoding="utf-8")

    a = _agent(proj, mode=SessionMode.LOG)
    data = a.load_checkpoint()                 # falls back to the legacy path
    assert data is not None and data["turns"] == 2

    a.clear_checkpoint()                        # clears BOTH locations
    assert not legacy.exists()


def test_checkpoint_info_probe_without_an_agent(tmp_path, monkeypatch):
    home = tmp_path / "home"
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    from localm.plugins.coder.agent import checkpoint_info, _checkpoint_path_for
    proj = tmp_path / "proj"; proj.mkdir()

    assert checkpoint_info(proj) is None        # nothing saved yet

    cp = _checkpoint_path_for(proj, "abc123")
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({
        "version": 1, "turns": 1, "total_tokens": 7,
        "interrupted_at": "2026-06-22T10:00:00",
        "messages": [{"role": "user", "content": "a"},
                     {"role": "assistant", "content": "b"}],
    }), encoding="utf-8")

    info = checkpoint_info(proj)
    assert info["turns"] == 1 and info["messages"] == 2 and info["total_tokens"] == 7
    assert info["interrupted_at"] == "2026-06-22T10:00:00"
    assert info["id"] == "abc123"


def test_distinct_projects_get_distinct_checkpoint_dirs(tmp_path, monkeypatch):
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    from localm.plugins.coder.agent.checkpoint import _project_dir_for
    p1 = tmp_path / "one"; p1.mkdir()
    p2 = tmp_path / "two"; p2.mkdir()
    assert _project_dir_for(p1) != _project_dir_for(p2)
    assert _project_dir_for(p1) == _project_dir_for(p1)   # stable per project


def test_distinct_sessions_in_one_project_get_distinct_checkpoint_files(
        tmp_path, monkeypatch):
    """Two sessions in the SAME project must land in two different files. One
    shared file per project means each session silently overwrites the last."""
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    from localm.plugins.coder.agent import _checkpoint_path_for
    proj = tmp_path / "proj"; proj.mkdir()
    assert _checkpoint_path_for(proj, "session-a") != _checkpoint_path_for(proj, "session-b")
    assert _checkpoint_path_for(proj, "session-a") == _checkpoint_path_for(proj, "session-a")


def test_save_checkpoint_failure_is_logged_not_silent(tmp_path, monkeypatch, caplog):
    """NEW-CODER-CHECKPOINT-NONATOMIC: a save failure must not be swallowed
    outright - the user has no other way to learn their resume checkpoint did
    not persist."""
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    proj = tmp_path / "proj"; proj.mkdir()
    a = _agent(proj, mode=SessionMode.LOG)
    a._messages = [{"role": "user", "content": "hi"}]

    with patch("localm.plugins.coder.agent.persistence.atomic_write",
               side_effect=OSError("disk full")):
        with caplog.at_level("WARNING"):
            a.save_checkpoint()          # must not raise

    assert any("save_checkpoint" in r.message for r in caplog.records)
    assert not a._checkpoint_path.exists()   # no torn/partial file either


def test_save_checkpoint_uses_the_atomic_write_helper(tmp_path, monkeypatch):
    """The write goes through storekit.atomic_write (tmp + os.replace), not a
    bare write_text, so a crash mid-write cannot leave a torn checkpoint."""
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    proj = tmp_path / "proj"; proj.mkdir()
    a = _agent(proj, mode=SessionMode.LOG)
    a._messages = [{"role": "user", "content": "hi"}]

    with patch("localm.plugins.coder.agent.persistence.atomic_write") as mock_write:
        a.save_checkpoint()
    assert mock_write.call_count == 1
    written_path, written_data = mock_write.call_args[0]
    assert written_path == a._checkpoint_path
    assert json.loads(written_data)["messages"] == a._messages


def test_checkpoint_info_reports_unreadable_for_a_corrupt_checkpoint(
        tmp_path, monkeypatch):
    """A user whose only checkpoint is corrupt must be told it exists but
    could not be read - not the same "nothing to resume" answer as never
    having saved one at all."""
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    from localm.plugins.coder.agent import checkpoint_info, _checkpoint_path_for
    proj = tmp_path / "proj"; proj.mkdir()

    assert checkpoint_info(proj) is None        # nothing saved yet - unchanged

    cp = _checkpoint_path_for(proj, "abc123")
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text("{not valid json", encoding="utf-8")

    assert checkpoint_info(proj) == {"unreadable": True}
