# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for localm.plugins.coder.privacy.clear_shell_history_traces().

Both coder invocation forms must be scrubbed: the documented "localm coder
<task>" form and the legacy "localcoder ..." form. These tests construct a fake
history file containing BOTH, run the scrub against it, and assert that BOTH are
removed while unrelated lines are kept.
"""



from localm.plugins.coder.privacy import clear_shell_history_traces


def _run_scrub(hist_path, binary_name="localcoder"):
    """Point both history-path resolvers at our single fake file and scrub."""
    from unittest.mock import patch

    with patch(
        "localm.plugins.coder.privacy._unix_history_paths",
        return_value=[hist_path],
    ), patch(
        "localm.plugins.coder.privacy._psreadline_history_paths",
        return_value=[],
    ):
        return clear_shell_history_traces(binary_name)


class TestScrubMatchesBothInvocations:
    def test_removes_both_localm_coder_and_localcoder_lines(self, tmp_path):
        hist = tmp_path / ".bash_history"
        hist.write_text(
            "git status\n"
            "localm coder fix the bug --mode privacy\n"
            "localcoder --model gemma4-4b\n"
            "git commit -m 'wip'\n"
        )

        modified = _run_scrub(hist)

        assert modified == 1
        remaining = hist.read_text().splitlines()
        # Both coder invocations gone.
        assert "localm coder fix the bug --mode privacy" not in remaining
        assert not any("localm coder" in line for line in remaining)
        assert not any("localcoder" in line for line in remaining)
        # Unrelated lines preserved.
        assert "git status" in remaining
        assert "git commit -m 'wip'" in remaining

    def test_keeps_plain_localm_chat_lines(self, tmp_path):
        """A bare 'localm <subcommand>' that is NOT the coder must survive: only
        coder invocations are in scope for this scrub.
        """
        hist = tmp_path / ".bash_history"
        hist.write_text(
            "localm gui --no-model\n"
            "localm serve\n"
            "localm coder do the thing\n"
        )

        modified = _run_scrub(hist)

        assert modified == 1
        remaining = hist.read_text().splitlines()
        assert "localm gui --no-model" in remaining
        assert "localm serve" in remaining
        assert not any("localm coder" in line for line in remaining)

    def test_localm_coder_after_pipe_or_sudo(self, tmp_path):
        hist = tmp_path / ".bash_history"
        hist.write_text(
            "true && localm coder run --model x\n"
            "sudo localm coder --model phi4\n"
            "echo done\n"
        )

        modified = _run_scrub(hist)

        assert modified == 1
        remaining = hist.read_text().splitlines()
        assert not any("localm coder" in line for line in remaining)
        assert "echo done" in remaining

    def test_localm_coder_in_zsh_extended_format(self, tmp_path):
        hist = tmp_path / ".zsh_history"
        hist.write_text(
            "git status\n"
            ": 1718000000:0;localm coder --model gemma4-4b\n"
            ": 1718000001:0;localcoder run\n"
            "git log\n"
        )

        modified = _run_scrub(hist)

        assert modified == 1
        remaining = hist.read_text().splitlines()
        assert not any("localm coder" in line for line in remaining)
        assert not any("localcoder" in line for line in remaining)
        assert "git status" in remaining
        assert "git log" in remaining

    def test_no_change_when_only_unrelated_lines(self, tmp_path):
        hist = tmp_path / ".bash_history"
        original = "git status\nls -la\nlocalm gui\n"
        hist.write_text(original)

        modified = _run_scrub(hist)

        assert modified == 0
        assert hist.read_text() == original

    def test_word_boundary_not_substring(self, tmp_path):
        """'localm coderx' / 'mylocalcoder' style substrings must be kept."""
        hist = tmp_path / ".bash_history"
        original = (
            "pip install localcoderlib\n"
            "notlocalcoder\n"
            "localm coderfoo run\n"
        )
        hist.write_text(original)

        modified = _run_scrub(hist)

        assert modified == 0
        assert hist.read_text() == original
