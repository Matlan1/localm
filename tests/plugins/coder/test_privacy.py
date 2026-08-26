# SPDX-License-Identifier: AGPL-3.0-or-later
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

import os
import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

from localm.plugins.coder.audit import SessionMode
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
        mock_rl = MagicMock(spec=[])   # empty spec - no attributes
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
    @pytest.fixture
    def result(self):
        return subprocess_privacy_env()

    def test_returns_dict(self, result):
        assert isinstance(result, dict)

    @pytest.mark.parametrize("key,expected", [
        ("HISTFILE", "NUL" if sys.platform == "win32" else os.devnull),
        ("HISTSIZE", "0"),
        ("HISTFILESIZE", "0"),
        ("HISTIGNORE", "*"),
        ("LESSHISTFILE", "NUL" if sys.platform == "win32" else os.devnull),
    ])
    def test_privacy_env_values(self, result, key, expected):
        assert result[key] == expected

    def test_includes_existing_env_vars(self):
        """Should be a copy of os.environ with overrides, not an empty dict."""
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
    # console is imported at module level in privacy.py - patch it there
    _PATCH = "localm.plugins.coder.privacy.console"

    @pytest.mark.parametrize("provider,expected_substr", [
        ("openai", "OpenAI"),
        ("anthropic", "Anthropic"),
        ("mycloud", None),  # unknown provider falls back to a generic notice
    ])
    def test_prints_warning(self, provider, expected_substr):
        with patch(self._PATCH) as mock_console:
            warn_external_provider(provider)
        mock_console.print.assert_called_once()
        text = mock_console.print.call_args[0][0]
        if expected_substr is not None:
            assert expected_substr in text
        else:
            assert "mycloud" in text or "privacy" in text.lower()


# ---------------------------------------------------------------------------
#  clear_shell_history_traces
# ---------------------------------------------------------------------------

from localm.plugins.coder.privacy import (
    _scrub_history_file,
    clear_shell_history_traces,
    _psreadline_history_paths,
    _unix_history_paths,
)


class TestScrubHistoryFile:
    def _pattern(self, name="localcoder"):
        import re
        return re.compile(
            r"(^|[|&;`]\s*|sudo\s+)" + re.escape(name) + r"\b",
            re.IGNORECASE,
        )

    def test_removes_matching_line(self, tmp_path):
        f = tmp_path / "history.txt"
        f.write_text("git status\nlocalcoder --model x\ngit log\n")
        changed = _scrub_history_file(f, self._pattern())
        assert changed
        lines = f.read_text().splitlines()
        assert "localcoder --model x" not in lines
        assert "git status" in lines
        assert "git log" in lines

    def test_no_change_when_no_match(self, tmp_path):
        f = tmp_path / "history.txt"
        original = "git status\ngit log\n"
        f.write_text(original)
        changed = _scrub_history_file(f, self._pattern())
        assert not changed
        assert f.read_text() == original

    def test_handles_zsh_extended_format(self, tmp_path):
        f = tmp_path / "history.txt"
        f.write_text(
            "git status\n"
            ": 1718000000:0;localcoder --model gemma4-4b\n"
            "git log\n"
        )
        changed = _scrub_history_file(f, self._pattern())
        assert changed
        lines = f.read_text().splitlines()
        assert not any("localcoder" in l for l in lines)
        assert "git status" in lines

    def test_case_insensitive(self, tmp_path):
        f = tmp_path / "history.txt"
        f.write_text("LOCALCODER --mode log\n")
        changed = _scrub_history_file(f, self._pattern())
        assert changed
        assert "LOCALCODER" not in f.read_text()

    def test_sudo_prefix_removed(self, tmp_path):
        f = tmp_path / "history.txt"
        f.write_text("sudo localcoder --model phi4\n")
        changed = _scrub_history_file(f, self._pattern())
        assert changed

    def test_does_not_remove_unrelated_lines_containing_substring(self, tmp_path):
        f = tmp_path / "history.txt"
        # "notlocalcoder" should NOT be removed (word boundary)
        f.write_text("notlocalcoder\npip install localcoderlib\n")
        changed = _scrub_history_file(f, self._pattern())
        assert not changed

    def test_returns_false_on_missing_file(self, tmp_path):
        f = tmp_path / "nonexistent.txt"
        changed = _scrub_history_file(f, self._pattern())
        assert not changed

    def test_preserves_crlf_line_endings(self, tmp_path):
        f = tmp_path / "history.txt"
        f.write_bytes(b"git status\r\nlocalcoder run\r\ngit log\r\n")
        _scrub_history_file(f, self._pattern())
        content = f.read_bytes()
        # All remaining lines should use CRLF
        assert b"\r\n" in content
        assert b"localcoder" not in content

    def test_empty_file_after_all_removed(self, tmp_path):
        f = tmp_path / "history.txt"
        f.write_text("localcoder --model x\nlocalcoder --mode log\n")
        _scrub_history_file(f, self._pattern())
        # File should exist but be effectively empty (or just a newline)
        assert f.exists()


class TestScrubHistoryFileFailureWarnings:
    """A failure to scrub a history file must always be surfaced, on both the
    read and the write path: a silent failure means secret-bearing command lines
    stay on disk with no indication anything went wrong."""

    _PATCH = "localm.plugins.coder.privacy.console"

    def _pattern(self):
        import re
        return re.compile(r"localcoder\b", re.IGNORECASE)

    def test_read_failure_prints_warning(self, tmp_path, monkeypatch):
        f = tmp_path / "history.txt"
        f.write_text("localcoder --model x\n")

        def _raise_read(self, *a, **k):
            raise PermissionError("denied")

        monkeypatch.setattr(Path, "read_bytes", _raise_read)

        with patch(self._PATCH) as mock_console:
            changed = _scrub_history_file(f, self._pattern())

        assert changed is False
        mock_console.print.assert_called_once()
        text = mock_console.print.call_args[0][0]
        assert "warning" in text.lower()
        assert str(f) in text

    def test_write_failure_prints_warning(self, tmp_path, monkeypatch):
        f = tmp_path / "history.txt"
        f.write_text("localcoder --model x\n")

        def _raise_write(self, *a, **k):
            raise PermissionError("denied")

        monkeypatch.setattr(Path, "write_bytes", _raise_write)

        with patch(self._PATCH) as mock_console:
            changed = _scrub_history_file(f, self._pattern())

        assert changed is False
        mock_console.print.assert_called_once()
        text = mock_console.print.call_args[0][0]
        assert "warning" in text.lower()
        assert str(f) in text


class TestClearShellHistoryTraces:
    def test_cleans_bash_history_file(self, tmp_path):
        hist = tmp_path / ".bash_history"
        hist.write_text("git status\nlocalcoder --mode privacy\ngit log\n")
        with patch(
            "localm.plugins.coder.privacy._unix_history_paths",
            return_value=[hist],
        ), patch(
            "localm.plugins.coder.privacy._psreadline_history_paths",
            return_value=[],
        ):
            modified = clear_shell_history_traces("localcoder")
        assert modified == 1
        assert "localcoder" not in hist.read_text()

    def test_cleans_psreadline_file(self, tmp_path):
        hist = tmp_path / "ConsoleHost_history.txt"
        hist.write_text("git status\nlocalcoder --model gemma4-4b\n")
        with patch(
            "localm.plugins.coder.privacy._psreadline_history_paths",
            return_value=[hist],
        ), patch(
            "localm.plugins.coder.privacy._unix_history_paths",
            return_value=[],
        ):
            modified = clear_shell_history_traces("localcoder")
        assert modified == 1
        assert "localcoder" not in hist.read_text()

    def test_returns_zero_when_nothing_to_clean(self, tmp_path):
        hist = tmp_path / ".bash_history"
        hist.write_text("git status\ngit log\n")
        with patch(
            "localm.plugins.coder.privacy._unix_history_paths",
            return_value=[hist],
        ), patch(
            "localm.plugins.coder.privacy._psreadline_history_paths",
            return_value=[],
        ):
            modified = clear_shell_history_traces("localcoder")
        assert modified == 0

    def test_skips_nonexistent_files(self, tmp_path):
        missing = tmp_path / "no_such_history"
        with patch(
            "localm.plugins.coder.privacy._unix_history_paths",
            return_value=[missing],
        ), patch(
            "localm.plugins.coder.privacy._psreadline_history_paths",
            return_value=[],
        ):
            modified = clear_shell_history_traces("localcoder")
        assert modified == 0

    def test_cleans_multiple_files(self, tmp_path):
        bash = tmp_path / ".bash_history"
        zsh  = tmp_path / ".zsh_history"
        bash.write_text("localcoder run\n")
        zsh.write_text("localcoder --model phi4\n")
        with patch(
            "localm.plugins.coder.privacy._unix_history_paths",
            return_value=[bash, zsh],
        ), patch(
            "localm.plugins.coder.privacy._psreadline_history_paths",
            return_value=[],
        ):
            modified = clear_shell_history_traces("localcoder")
        assert modified == 2


class TestHistoryPaths:
    def test_psreadline_only_on_windows(self):
        with patch("localm.plugins.coder.privacy.sys") as mock_sys:
            mock_sys.platform = "linux"
            paths = _psreadline_history_paths()
        assert paths == []

    def test_unix_paths_empty_on_windows(self):
        with patch("localm.plugins.coder.privacy.sys") as mock_sys:
            mock_sys.platform = "win32"
            paths = _unix_history_paths()
        assert paths == []

    def test_unix_uses_histfile_env(self, tmp_path):
        fake_hist = tmp_path / "custom_hist"
        fake_hist.write_text("")
        with patch.dict(os.environ, {"HISTFILE": str(fake_hist)}), \
             patch("localm.plugins.coder.privacy.sys") as mock_sys:
            mock_sys.platform = "linux"
            paths = _unix_history_paths()
        assert fake_hist in paths

    def test_unix_does_not_include_devnull_histfile(self):
        with patch.dict(os.environ, {"HISTFILE": "/dev/null"}), \
             patch("localm.plugins.coder.privacy.sys") as mock_sys:
            mock_sys.platform = "linux"
            paths = _unix_history_paths()
        # /dev/null should be excluded (we set it ourselves in subprocess env)
        assert not any(str(p) in ("/dev/null", "NUL") for p in paths)


# ---------------------------------------------------------------------------
#  tool_run_shell - _privacy flag propagation
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
    # Agent is imported locally inside tool_spawn_agent - patch at definition
    _PATCH = "localm.plugins.coder.agent.Agent"

    @pytest.mark.parametrize("mode", [SessionMode.PRIVACY, SessionMode.LOG])
    def test_child_inherits_mode(self, tmp_path, mode):
        """spawn_agent must pass parent.mode to the child Agent."""
        from localm.plugins.coder.tools import tool_spawn_agent

        parent = MagicMock()
        parent.mode = mode
        parent.backend.model_id = "test"
        parent.cwd = tmp_path

        with patch(self._PATCH) as MockAgent:
            MockAgent.return_value.run_task.return_value = "done"
            MockAgent.return_value.turns = 1
            tool_spawn_agent(tmp_path, "do something", _parent_agent=parent)

        assert MockAgent.call_count == 1
        _, kwargs = MockAgent.call_args
        assert kwargs.get("mode") == mode
