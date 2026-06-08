"""
Tests for localm.plugins.coder.privacy

Covers:
  - suppress_readline_history(): no crash when readline unavailable; when
    available, sets history length to 0 and clears history
  - subprocess_privacy_env(): returns dict with expected vars zeroed
  - warn_external_provider(): calls console.print with expected text
  - tool_run_shell with _privacy=True: subprocess receives zeroed HISTFILE
  - spawn_agent mode inheritance (via agent._execute_tool injection)
"""

import sys
import pytest
from unittest.mock import MagicMock, patch, call

from localm.plugins.coder.privacy import (
    subprocess_privacy_env,
    suppress_readline_history,
    warn_external_provider,
)


# ---------------------------------------------------------------------------
#  suppress_readline_history
# ---------------------------------------------------------------------------

class TestSuppressReadlineHistory:
    def test_no_crash_without_readline(self):
        """Should silently succeed even when readline is not importable."""
        with patch.dict("sys.modules", {"readline": None}):
            suppress_readline_history()   # must not raise

    def test_handles_attributeerror_gracefully(self):
        """readline present but without set_history_length should not crash."""
        mock_rl = MagicMock(spec=[])   # empty spec — no attributes
        with patch.dict("sys.modules", {"readline": mock_rl}):
            suppress_readline_history()   # must not raise

    def test_real_readline_set_to_zero(self):
        """Integration: if readline is actually available, history length is 0."""
        try:
            import readline as _rl
        except ImportError:
            pytest.skip("readline not available in this environment")

        suppress_readline_history()
        assert _rl.get_history_length() == 0

    def test_real_readline_history_cleared(self):
        """Integration: any accumulated history is wiped."""
        try:
            import readline as _rl
        except ImportError:
            pytest.skip("readline not available in this environment")

        _rl.add_history("secret prompt")
        suppress_readline_history()
        assert _rl.get_current_history_length() == 0


# ---------------------------------------------------------------------------
#  subprocess_privacy_env
# ---------------------------------------------------------------------------

class TestSubprocessPrivacyEnv:
    def test_returns_dict(self):
        result = subprocess_privacy_env()
        assert isinstance(result, dict)

    def test_histfile_is_devnull(self):
        result = subprocess_privacy_env()
        import os
        expected = "NUL" if sys.platform == "win32" else os.devnull
        assert result["HISTFILE"] == expected

    def test_histsize_is_zero(self):
        assert subprocess_privacy_env()["HISTSIZE"] == "0"

    def test_histfilesize_is_zero(self):
        assert subprocess_privacy_env()["HISTFILESIZE"] == "0"

    def test_histignore_blocks_all(self):
        assert subprocess_privacy_env()["HISTIGNORE"] == "*"

    def test_lesshistfile_is_devnull(self):
        result = subprocess_privacy_env()
        import os
        expected = "NUL" if sys.platform == "win32" else os.devnull
        assert result["LESSHISTFILE"] == expected

    def test_includes_existing_env_vars(self):
        """Should be a copy of os.environ with overrides, not an empty dict."""
        import os
        result = subprocess_privacy_env()
        # PATH (or PATHEXT on Windows) should be present
        assert "PATH" in result or "PATHEXT" in result

    def test_does_not_mutate_os_environ(self):
        """Must return a copy, not modify os.environ in place."""
        import os
        original_histfile = os.environ.get("HISTFILE", "__not_set__")
        subprocess_privacy_env()
        assert os.environ.get("HISTFILE", "__not_set__") == original_histfile


# ---------------------------------------------------------------------------
#  warn_external_provider
# ---------------------------------------------------------------------------

class TestWarnExternalProvider:
    # console is imported at module level in privacy.py — patch it there
    _PATCH = "localm.plugins.coder.privacy.console"

    def test_prints_warning_for_openai(self):
        with patch(self._PATCH) as mock_console:
            warn_external_provider("openai")
        mock_console.print.assert_called_once()
        text = mock_console.print.call_args[0][0]
        assert "OpenAI" in text

    def test_prints_warning_for_anthropic(self):
        with patch(self._PATCH) as mock_console:
            warn_external_provider("anthropic")
        text = mock_console.print.call_args[0][0]
        assert "Anthropic" in text

    def test_prints_warning_for_unknown_provider(self):
        with patch(self._PATCH) as mock_console:
            warn_external_provider("mycloud")
        mock_console.print.assert_called_once()
        text = mock_console.print.call_args[0][0]
        assert "mycloud" in text or "privacy" in text.lower()


# ---------------------------------------------------------------------------
#  tool_run_shell — _privacy flag propagation
# ---------------------------------------------------------------------------

class TestRunShellPrivacy:
    def test_privacy_flag_passes_custom_env(self, tmp_path):
        """When _privacy=True, subprocess should receive the zeroed HISTFILE env."""
        from localm.plugins.coder.tools import tool_run_shell

        captured_env = {}

        def fake_run(cmd, **kwargs):
            captured_env.update(kwargs.get("env") or {})
            m = MagicMock()
            m.returncode = 0
            m.stdout = "ok"
            m.stderr = ""
            return m

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_run_shell(tmp_path, "echo hi", _privacy=True)

        import os
        expected_null = "NUL" if sys.platform == "win32" else os.devnull
        assert captured_env.get("HISTFILE") == expected_null
        assert captured_env.get("HISTSIZE") == "0"

    def test_no_privacy_flag_passes_none_env(self, tmp_path):
        """Without _privacy, env should be None (inherit parent)."""
        from localm.plugins.coder.tools import tool_run_shell

        captured_kwargs = {}

        def fake_run(cmd, **kwargs):
            captured_kwargs.update(kwargs)
            m = MagicMock()
            m.returncode = 0
            m.stdout = "ok"
            m.stderr = ""
            return m

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_run_shell(tmp_path, "echo hi", _privacy=False)

        assert captured_kwargs.get("env") is None


# ---------------------------------------------------------------------------
#  spawn_agent mode inheritance
# ---------------------------------------------------------------------------

class TestSpawnAgentModeInheritance:
    # Agent is imported locally inside tool_spawn_agent — patch at definition
    _PATCH = "localm.plugins.coder.agent.Agent"

    def test_child_inherits_privacy_mode(self, tmp_path):
        """spawn_agent must pass parent.mode to the child Agent."""
        from localm.plugins.coder.audit import SessionMode
        from localm.plugins.coder.tools import tool_spawn_agent

        parent = MagicMock()
        parent.mode = SessionMode.PRIVACY
        parent.backend.model_id = "test"
        parent.cwd = tmp_path

        with patch(self._PATCH) as MockAgent:
            MockAgent.return_value.run_task.return_value = "done"
            MockAgent.return_value.turns = 1
            tool_spawn_agent(tmp_path, "do something", _parent_agent=parent)

        assert MockAgent.call_count == 1
        _, kwargs = MockAgent.call_args
        assert kwargs.get("mode") == SessionMode.PRIVACY

    def test_child_inherits_log_mode(self, tmp_path):
        from localm.plugins.coder.audit import SessionMode
        from localm.plugins.coder.tools import tool_spawn_agent

        parent = MagicMock()
        parent.mode = SessionMode.LOG
        parent.backend.model_id = "test"
        parent.cwd = tmp_path

        with patch(self._PATCH) as MockAgent:
            MockAgent.return_value.run_task.return_value = "done"
            MockAgent.return_value.turns = 1
            tool_spawn_agent(tmp_path, "task", _parent_agent=parent)

        _, kwargs = MockAgent.call_args
        assert kwargs.get("mode") == SessionMode.LOG
