# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for localm.plugins.coder.audit session-mode logic and agent integration.

Covers:
  - SessionMode enum / parse_mode()
  - NullAuditLog (no-op, no files created)
  - AuditLog (writes JSONL)
  - make_audit_log() factory
  - Agent.close() - privacy: no files; log: closes JSONL; full: writes markdown
  - _write_session_markdown content
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from localm.plugins.coder.audit import (
    AuditLog,
    NullAuditLog,
    SessionMode,
    make_audit_log,
    parse_mode,
)


# ---------------------------------------------------------------------------
#  SessionMode / parse_mode
# ---------------------------------------------------------------------------

class TestSessionMode:
    @pytest.mark.parametrize("mode,expected", [
        (SessionMode.PRIVACY, "privacy"),
        (SessionMode.LOG, "log"),
        (SessionMode.FULL, "full"),
    ])
    def test_value(self, mode, expected):
        assert mode.value == expected

    @pytest.mark.parametrize("raw,expected", [
        ("privacy", SessionMode.PRIVACY),
        ("log", SessionMode.LOG),
        ("full", SessionMode.FULL),
    ])
    def test_parse(self, raw, expected):
        assert parse_mode(raw) == expected

    def test_parse_uppercase(self):
        assert parse_mode("LOG") == SessionMode.LOG
        assert parse_mode("PRIVACY") == SessionMode.PRIVACY
        assert parse_mode("FULL") == SessionMode.FULL

    def test_parse_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown session mode"):
            parse_mode("cloud")

    def test_parse_whitespace_stripped(self):
        assert parse_mode("  full  ") == SessionMode.FULL


# ---------------------------------------------------------------------------
#  NullAuditLog
# ---------------------------------------------------------------------------

class TestNullAuditLog:
    def test_path_is_none(self):
        log = NullAuditLog()
        assert log.path is None

    def test_methods_are_no_ops(self, tmp_path):
        log = NullAuditLog()
        # None of these should raise or write anything
        log.set_turn(1)
        log.user("hello")
        log.llm("response", tokens=100, reasoning="scratchpad")
        log.tool_call("read_file", {"path": "x.py"})
        log.tool_result("read_file", ok=True, summary="10 lines")
        log.close()
        # No files created anywhere
        assert not list(tmp_path.rglob("*.jsonl"))

    def test_close_is_idempotent(self):
        log = NullAuditLog()
        log.close()
        log.close()   # should not raise


# ---------------------------------------------------------------------------
#  AuditLog (real writer)
# ---------------------------------------------------------------------------

class TestAuditLog:
    def test_creates_jsonl_file(self, tmp_path):
        with patch("localm.plugins.coder.audit._SESSIONS_DIR", tmp_path):
            log = AuditLog(label="test")
        assert log.path.exists()
        assert log.path.suffix == ".jsonl"

    def test_label_in_filename(self, tmp_path):
        with patch("localm.plugins.coder.audit._SESSIONS_DIR", tmp_path):
            log = AuditLog(label="myagent")
        assert "myagent" in log.path.name

    def test_writes_start_event(self, tmp_path):
        with patch("localm.plugins.coder.audit._SESSIONS_DIR", tmp_path):
            log = AuditLog(label="t")
        events = [json.loads(l) for l in log.path.read_text().splitlines()]
        assert events[0]["type"] == "system"
        assert events[0]["data"]["msg"] == "session started"

    def test_writes_user_event(self, tmp_path):
        with patch("localm.plugins.coder.audit._SESSIONS_DIR", tmp_path):
            log = AuditLog(label="t")
        log.user("hello world")
        log.close()
        events = [json.loads(l) for l in log.path.read_text().splitlines()]
        user_evts = [e for e in events if e["type"] == "user"]
        assert len(user_evts) == 1
        assert user_evts[0]["data"]["content"] == "hello world"

    def test_writes_llm_event(self, tmp_path):
        with patch("localm.plugins.coder.audit._SESSIONS_DIR", tmp_path):
            log = AuditLog(label="t")
        log.set_turn(2)
        log.llm("model response", tokens=500)
        log.close()
        events = [json.loads(l) for l in log.path.read_text().splitlines()]
        llm_evts = [e for e in events if e["type"] == "llm"]
        assert llm_evts[0]["turn"] == 2
        assert llm_evts[0]["data"]["tokens"] == 500

    def test_llm_event_records_reasoning_separately(self, tmp_path):
        """A thinking model's reasoning is stored in its OWN field, never
        appended to the visible-answer 'content' field."""
        with patch("localm.plugins.coder.audit._SESSIONS_DIR", tmp_path):
            log = AuditLog(label="t")
        log.llm("The answer.", tokens=10, reasoning="because reasons")
        log.close()
        events = [json.loads(l) for l in log.path.read_text().splitlines()]
        llm_evts = [e for e in events if e["type"] == "llm"]
        assert llm_evts[0]["data"]["content"] == "The answer."
        assert llm_evts[0]["data"]["reasoning"] == "because reasons"

    def test_llm_event_without_reasoning_arg_is_empty_string(self, tmp_path):
        """Back-compat: a caller that never passes reasoning= still gets a
        well-formed (empty) field, not a missing key."""
        with patch("localm.plugins.coder.audit._SESSIONS_DIR", tmp_path):
            log = AuditLog(label="t")
        log.llm("plain answer")
        log.close()
        events = [json.loads(l) for l in log.path.read_text().splitlines()]
        llm_evts = [e for e in events if e["type"] == "llm"]
        assert llm_evts[0]["data"]["reasoning"] == ""

    def test_close_writes_end_event(self, tmp_path):
        with patch("localm.plugins.coder.audit._SESSIONS_DIR", tmp_path):
            log = AuditLog(label="t")
        log.close()
        events = [json.loads(l) for l in log.path.read_text().splitlines()]
        assert events[-1]["type"] == "system"
        assert events[-1]["data"]["msg"] == "session ended"


# ---------------------------------------------------------------------------
#  make_audit_log factory
# ---------------------------------------------------------------------------

class TestMakeAuditLog:
    def test_privacy_returns_null(self):
        result = make_audit_log(SessionMode.PRIVACY)
        assert isinstance(result, NullAuditLog)

    @pytest.mark.parametrize("mode", [SessionMode.LOG, SessionMode.FULL])
    def test_non_privacy_returns_real(self, mode, tmp_path):
        with patch("localm.plugins.coder.audit._SESSIONS_DIR", tmp_path):
            result = make_audit_log(mode, label="x")
        assert isinstance(result, AuditLog)
        result.close()


# ---------------------------------------------------------------------------
#  Agent.close() - per mode
# ---------------------------------------------------------------------------

def _make_agent(tmp_path, mode):
    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = "test-model"
    backend.last_usage = {}
    with patch("localm.plugins.coder.agent.make_audit_log") as mock_factory, \
         patch("localm.plugins.coder.agent.load_memory", return_value=""), \
         patch("localm.plugins.coder.agent.ProjectMap") as mock_pm:
        mock_pm.build.return_value.file_count.return_value = 0
        mock_factory.return_value = NullAuditLog()
        agent = Agent(backend=backend, cwd=tmp_path, mode=mode)
    return agent


class TestAgentClose:
    @pytest.mark.parametrize("mode", [SessionMode.PRIVACY, SessionMode.LOG])
    def test_non_full_close_returns_none(self, mode, tmp_path):
        agent = _make_agent(tmp_path, mode)
        result = agent.close()
        assert result is None

    def test_full_close_writes_markdown(self, tmp_path):
        agent = _make_agent(tmp_path, SessionMode.FULL)
        # Add a simple conversation
        agent._messages = [
            {"role": "user",      "content": "Write a hello world"},
            {"role": "assistant", "content": "Here it is: print('hello world')"},
        ]
        agent._turns = 1
        agent._model_name = "gemma4-4b"

        result = agent.close()
        assert result is not None
        assert result.exists()
        assert result.suffix == ".md"
        assert result.parent == tmp_path / ".localcoder" / "sessions"

    def test_full_markdown_contains_user_message(self, tmp_path):
        agent = _make_agent(tmp_path, SessionMode.FULL)
        agent._messages = [
            {"role": "user", "content": "My specific question here"},
        ]
        agent._turns = 1

        result = agent.close()
        content = result.read_text()
        assert "My specific question here" in content

    def test_full_markdown_skips_tool_results(self, tmp_path):
        agent = _make_agent(tmp_path, SessionMode.FULL)
        agent._messages = [
            {"role": "user",      "content": "do a thing"},
            {"role": "assistant", "content": "<tool_call>{\"name\": \"read_file\", \"args\": {\"path\": \"x.py\"}}</tool_call>"},
            {"role": "user",      "content": "<tool_result name='read_file'>lots of code here</tool_result>"},
            {"role": "assistant", "content": "Done reading."},
        ]
        agent._turns = 2

        result = agent.close()
        content = result.read_text()
        # Tool result blobs should NOT appear
        assert "<tool_result" not in content
        assert "lots of code here" not in content
        # But tool call summary should appear
        assert "read_file" in content

    def test_full_markdown_contains_model_name(self, tmp_path):
        agent = _make_agent(tmp_path, SessionMode.FULL)
        agent._model_name = "deepseek-r1"
        agent._messages = []
        agent._turns = 0

        result = agent.close()
        content = result.read_text()
        assert "deepseek-r1" in content

    def test_full_markdown_tool_call_summary(self, tmp_path):
        agent = _make_agent(tmp_path, SessionMode.FULL)
        agent._messages = [
            {"role": "assistant", "content": (
                'Calling read_file.\n'
                '<tool_call>{"name": "write_file", "args": {"path": "out.py", "content": "x"}}</tool_call>'
            )},
        ]
        agent._turns = 1

        result = agent.close()
        content = result.read_text()
        assert "write_file" in content
        assert "out.py" in content

    def test_full_markdown_summarises_a_fenced_json_tool_call(self, tmp_path):
        """```json fences are one of the 5 shapes parse_tool_calls recognises
        (name-gated); unsummarised, the raw fence markers and JSON leak
        verbatim into the transcript."""
        agent = _make_agent(tmp_path, SessionMode.FULL)
        agent._messages = [
            {"role": "assistant", "content": (
                'Reading it now.\n'
                '```json\n{"name": "read_file", "args": {"path": "a.py"}}\n```'
            )},
        ]
        agent._turns = 1

        result = agent.close()
        content = result.read_text()
        assert "Reading it now." in content
        assert "read_file" in content
        assert "a.py" in content
        assert "```" not in content
        assert '"name"' not in content

    def test_full_markdown_summarises_a_bare_json_tool_call(self, tmp_path):
        """A bare top-level {"name":...,"args":...} object with no wrapper at
        all is the other shape that can leak raw JSON into the transcript."""
        agent = _make_agent(tmp_path, SessionMode.FULL)
        agent._messages = [
            {"role": "assistant", "content": (
                '{"name": "write_file", "args": {"path": "out.py", "content": "x"}}'
            )},
        ]
        agent._turns = 1

        result = agent.close()
        content = result.read_text()
        assert "write_file" in content
        assert "out.py" in content
        assert '"name"' not in content
        assert '"args"' not in content

    def test_full_markdown_shows_placeholder_for_a_malformed_tool_call(self, tmp_path):
        """A <tool_call> block whose JSON body never parsed (loop.py's repair
        path persists the raw attempt to history before the repair succeeds)
        must still render the same generic placeholder it always has, not
        leak raw XML."""
        agent = _make_agent(tmp_path, SessionMode.FULL)
        agent._messages = [
            {"role": "assistant", "content": (
                'Let me check.\n<tool_call>\n'
                '{"name": "read_file", "args": {"path": "a.py"'
                '\n</tool_call>'
            )},
        ]
        agent._turns = 1

        result = agent.close()
        content = result.read_text()
        assert "Let me check." in content
        assert "(tool call)" in content
        assert "<tool_call>" not in content

    def test_full_markdown_header_renders_for_a_purely_malformed_tool_call(self, tmp_path):
        """No prose survives at all, only the malformed block - the header
        line must still appear (via `calls or malformed`), matching what the
        old strip_xml_tool_calls-only code path did for the same shape."""
        agent = _make_agent(tmp_path, SessionMode.FULL)
        agent._messages = [
            {"role": "assistant", "content": (
                '<tool_call>\n{"name": "read_file", "args": {"path": "a.py"\n</tool_call>'
            )},
        ]
        agent._turns = 1

        result = agent.close()
        content = result.read_text()
        assert f"**{agent.name}**:" in content
        assert "(tool call)" in content

    def test_privacy_no_files_created(self, tmp_path):
        agent = _make_agent(tmp_path, SessionMode.PRIVACY)
        agent._messages = [{"role": "user", "content": "secret message"}]
        agent.close()
        # No .localcoder/sessions directory should exist
        sessions_dir = tmp_path / ".localcoder" / "sessions"
        assert not sessions_dir.exists()


# ---------------------------------------------------------------------------
#  _write_session_markdown collision (REG: coder session markdown transcript
#  collision - the sibling of the audit-log collision fixed by
#  TestSessionIdentity in test_audit_modes.py)
# ---------------------------------------------------------------------------

class TestSessionMarkdownIdentity:
    """Two coder sessions in the SAME project, closed within the same second,
    must not land on one transcript file. The filename used to be built from
    a bare one-second-resolution timestamp with no session identity at all,
    so the second session's close() silently overwrote the first session's
    entire transcript (write_text, not append)."""

    def test_checkpoint_id_is_part_of_the_filename(self, tmp_path):
        agent = _make_agent(tmp_path, SessionMode.FULL)
        agent._messages = [{"role": "user", "content": "hi"}]
        agent._turns = 1

        result = agent.close()
        assert agent._checkpoint_id in result.name

    def test_two_sessions_same_second_get_distinct_transcript_files(self, tmp_path):
        a1 = _make_agent(tmp_path, SessionMode.FULL)
        a1._messages = [{"role": "user", "content": "message from session A"}]
        a1._turns = 1
        a2 = _make_agent(tmp_path, SessionMode.FULL)
        a2._messages = [{"role": "user", "content": "message from session B"}]
        a2._turns = 1

        with patch("time.strftime", return_value="2026-01-01_000000"):
            path1 = a1.close()
            path2 = a2.close()

        assert path1 != path2
        assert path1.exists() and path2.exists()
        text1 = path1.read_text(encoding="utf-8")
        text2 = path2.read_text(encoding="utf-8")
        assert "message from session A" in text1
        assert "message from session B" in text2
        assert "message from session B" not in text1
        assert "message from session A" not in text2
