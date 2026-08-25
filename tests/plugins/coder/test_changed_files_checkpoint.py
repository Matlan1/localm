# SPDX-License-Identifier: AGPL-3.0-or-later
"""The changed-files tracker (GUI 'files changed' / session_diff) must survive a checkpoint save/resume cycle."""

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
        kw.setdefault("mode", SessionMode.LOG)
        kw.setdefault("auto_approve", True)
        return Agent(_Stub(), cwd=cwd, self_verify=False, **kw)


def _call(agent, name, **args):
    return agent._execute_tool(
        __import__("localm.plugins.coder.parser", fromlist=["ToolCall"])
        .ToolCall(name=name, args=args, raw="", start=0, end=0),
        interactive=False)


def test_changed_files_survive_a_real_checkpoint_resume_cycle(tmp_path, monkeypatch):
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    proj = tmp_path / "proj"
    proj.mkdir()

    first = _agent(proj)
    result = _call(first, "write_file", path="app.py", content="x = 1\n")
    assert result.ok
    assert first.changed_files() == [
        {"path": "app.py", "writes": 1, "created": True,
         "exists": True, "last_tool": "write_file"}
    ]
    first._messages = [{"role": "user", "content": "write app.py"}]
    first.save_checkpoint()

    # A genuinely fresh Agent for the same project - the load-bearing half:
    # if save wrote nothing usable, this finds an empty tracker regardless of
    # what the first agent's in-memory state still (correctly) shows.
    second = _agent(proj)
    assert second.changed_files() == []          # fresh agent, nothing yet
    data = second.load_checkpoint()
    assert data is not None
    assert "changed_files" in data                # pins the SAVE half
    second.resume_checkpoint(data)

    assert second.changed_files() == [
        {"path": "app.py", "writes": 1, "created": True,
         "exists": True, "last_tool": "write_file"}
    ]
    # session_diff needs the ORIGINAL snapshot too (None here - app.py was
    # newly created), not just the writes count.
    diff = second.session_diff("app.py")
    assert "+x = 1" in diff


def test_older_checkpoint_without_changed_files_key_resumes_with_empty_tracker(
        tmp_path, monkeypatch):
    """A checkpoint saved before this fix existed has no 'changed_files' key at all - resuming it must not crash, and leaves the tracker empty (the same lossy behaviour it already had, not a new failure mode)."""
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    proj = tmp_path / "proj"
    proj.mkdir()

    a = _agent(proj)
    a.resume_checkpoint({"version": 1, "turns": 2, "total_tokens": 5,
                         "messages": [{"role": "user", "content": "x"}]})
    assert a.changed_files() == []
    assert a._turns == 2


def test_garbage_changed_files_in_a_checkpoint_are_dropped_not_trusted(
        tmp_path, monkeypatch):
    """The checkpoint is plain user-writable JSON: a hand-edited or corrupted changed_files value must normalise, not crash or land unvalidated."""
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    proj = tmp_path / "proj"
    proj.mkdir()

    a = _agent(proj)
    a.resume_checkpoint({
        "version": 1, "messages": [],
        "changed_files": {
            "kept.py": {"original": None, "writes": 2, "last_tool": "write_file"},
            "bad_b64.py": {"original": "not valid base64!!!", "writes": 1,
                           "last_tool": "write_file"},
            "bad_writes.py": {"original": None, "writes": "lots",
                              "last_tool": "write_file"},
            123: {"original": None, "writes": 1, "last_tool": "write_file"},
            "not_a_dict.py": "surprise",
        },
    })
    files = {f["path"]: f for f in a.changed_files()}
    assert set(files) == {"kept.py", "bad_writes.py"}
    assert files["kept.py"]["writes"] == 2
    assert files["bad_writes.py"]["writes"] == 1   # non-int writes normalised to 1

    a.resume_checkpoint({"version": 1, "messages": [], "changed_files": "not a dict"})
    assert a.changed_files() == []

    a.resume_checkpoint({"version": 1, "messages": [], "changed_files": 17})
    assert a.changed_files() == []
