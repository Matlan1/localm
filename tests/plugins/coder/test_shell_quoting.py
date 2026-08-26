# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quoted paths through the coder's shell tool.

Quoting a path is the normal way to pass one containing spaces. Two hazards on
the same chain, both covered here:

  A. the argv route keeping the quote CHARACTERS in the token, so the process is
     handed a filename with literal quotes in it; and
  B. the shell route handing cmd.exe an argv list, which subprocess renders with
     list2cmdline - MSVCRT escaping that cmd.exe misreads.

Each mechanism is also driven unfixed, in the same run, so a pass here cannot be
for an unrelated reason.

Every target is a disposable file the test creates under its own tmp_path.
Nothing here reads, stats, or names a file that belongs to the machine.
"""

import shlex
import subprocess
import sys

import pytest

from localm.plugins.coder.tools import _shell_argv
from localm.plugins.coder.tools.base import run_subprocess
from localm.plugins.coder.tools.shell import _split_command

MARK = "PAYLOAD-TOKEN"


def _went_through_the_shell(launched) -> bool:
    """Did *launched* - whatever reached subprocess - go through the platform shell?

    The launch form differs by platform: POSIX gets an argv list (execv receives
    it verbatim), Windows a raw command-line STRING, because an argv list is
    re-quoted by list2cmdline in syntax cmd.exe misreads. See
    tools/base.py:platform_shell.
    """
    if isinstance(launched, str):
        return launched.startswith("cmd /C ")
    return launched[0] in ("cmd", "/bin/sh")


def _payload(tmp_path):
    """A disposable file whose directory AND name both contain spaces."""
    d = tmp_path / "a dir with spaces"
    d.mkdir()
    f = d / "payload file.txt"
    f.write_text(MARK + "\n", encoding="utf-8")
    return d, f


def _reader(d):
    """A disposable script that prints the file named by its first argument."""
    script = d / "reader script.py"
    script.write_text(
        "import sys\n"
        "print(open(sys.argv[1], encoding='utf-8').read().strip())\n",
        encoding="utf-8")
    return script


# ---------------------------------------------------------------------------
#  Splitting
# ---------------------------------------------------------------------------

class TestSplitCommandRemovesQuotes:
    """What reaches the process must be the path, not the quotes around it.

    Lexical, so these run on every platform; the Windows backslash rules are
    asserted separately below.
    """

    def test_quoted_argument_with_spaces_is_one_token_without_quotes(self):
        assert _split_command('reader "a dir with spaces/f.txt"') == [
            "reader", "a dir with spaces/f.txt"]

    def test_embedded_quote_is_grouped_and_the_quotes_removed(self):
        """The case that rules out stripping quotes off the tokens.

        posix=False honours a quote only where one OPENS a token, so it splits
        this into ['--message="a', 'b"']. No post-pass repairs a boundary; and
        even given the right boundary, tok.strip('\"') would leave
        '--message="a b'.
        """
        assert _split_command('git commit --message="a b"') == [
            "git", "commit", "--message=a b"]
        assert _split_command('git commit -m"a b"') == ["git", "commit", "-ma b"]

    def test_hash_inside_an_argument_is_not_a_comment(self):
        """shlex's default commenter would silently truncate the message, so
        clearing it is load-bearing rather than tidy."""
        assert _split_command('git commit -m "fix #42 now"') == [
            "git", "commit", "-m", "fix #42 now"]

    def test_empty_quoted_argument_survives(self):
        assert _split_command('prog ""') == ["prog", ""]

    def test_unquoted_command_is_unchanged(self):
        assert _split_command("python -m pytest -q") == [
            "python", "-m", "pytest", "-q"]

    def test_malformed_quoting_raises_so_the_caller_can_fall_back(self):
        with pytest.raises(ValueError):
            _split_command('prog "unterminated')


@pytest.mark.skipif(sys.platform != "win32", reason="Windows backslash rules")
class TestWindowsBackslashesSurviveSplitting:
    """Clearing shlex's escape character is what keeps these intact. Plain
    posix=True - the other candidate fix - reads a backslash as an escape and
    would flatten every one of them."""

    def test_unquoted_windows_path_keeps_its_separators(self):
        assert _split_command(r"dir sub\dir\file.txt") == [
            "dir", r"sub\dir\file.txt"]

    def test_quoted_windows_path_with_spaces_keeps_its_separators(self):
        assert _split_command(r'type "a dir\with spaces\f.txt"') == [
            "type", r"a dir\with spaces\f.txt"]

    def test_unc_path_keeps_both_leading_separators(self):
        assert _split_command(r'copy "\\host\share\a b" .') == [
            "copy", r"\\host\share\a b", "."]

    def test_trailing_separator_before_the_closing_quote(self):
        assert _split_command(r'dir "a dir\"') == ["dir", "a dir" + "\\"]


# ---------------------------------------------------------------------------
#  Real execution
# ---------------------------------------------------------------------------

class TestQuotedPathsReallyExecute:
    """Real commands reading a real file whose path contains spaces."""

    def test_argv_route_reads_a_quoted_path_with_spaces(self, tmp_path):
        """Interpreter, script and target are all quoted paths, two of them
        containing spaces - so argv[0] and a later argument are both covered."""
        d, payload = _payload(tmp_path)
        command = '"%s" "%s" "%s"' % (sys.executable, _reader(d), payload)

        assert not _went_through_the_shell(_shell_argv(command)), "expected argv mode"
        result = run_subprocess(_shell_argv(command), tmp_path, timeout=60)
        assert result.ok, result.stderr
        assert MARK in (result.stdout or "")

    def test_shell_route_reads_a_quoted_path_with_spaces(self, tmp_path):
        """The shell route is where the cmd.exe re-quoting defect lived. A cmd
        builtin has no executable on disk so it cannot use argv mode; on POSIX
        an operator forces the same route."""
        _, payload = _payload(tmp_path)
        command = ('type "%s"' % payload if sys.platform == "win32"
                   else 'cat "%s" && true' % payload)

        assert _went_through_the_shell(_shell_argv(command)), "expected shell mode"
        result = run_subprocess(_shell_argv(command), tmp_path, timeout=60)
        assert result.ok, result.stderr
        assert MARK in (result.stdout or "")

    def test_unquoted_path_without_spaces_still_works(self, tmp_path):
        """Regression guard: the unquoted form must not have been traded away
        to make the quoted one work."""
        (tmp_path / "plain.txt").write_text(MARK + "\n", encoding="utf-8")
        command = (r"type .\plain.txt" if sys.platform == "win32"
                   else "cat ./plain.txt")
        result = run_subprocess(_shell_argv(command), tmp_path, timeout=60)
        assert result.ok, result.stderr
        assert MARK in (result.stdout or "")

    def test_shell_operators_still_work_with_a_quoted_path(self, tmp_path):
        """Handing cmd the command line verbatim must not cost us the operators
        that are the whole reason for the shell route."""
        _, payload = _payload(tmp_path)
        reader = "type" if sys.platform == "win32" else "cat"
        command = '%s "%s" && echo OPERATORS-OK' % (reader, payload)
        result = run_subprocess(_shell_argv(command), tmp_path, timeout=60)
        assert result.ok, result.stderr
        assert MARK in (result.stdout or ""), result.stdout
        assert "OPERATORS-OK" in (result.stdout or ""), result.stdout


# ---------------------------------------------------------------------------
#  The old split modes
# ---------------------------------------------------------------------------

class TestTheQuotingDefectsFire:
    """Both defect mechanisms, run against the same real files, in the same run
    that shows the fix working.

    Without this, the tests above could pass for a reason unrelated to the fix,
    and a regression to either mechanism would look like a change nobody has to
    explain.
    """

    def test_the_old_split_mode_got_the_token_boundary_wrong(self):
        """Cross-platform, no execution needed: posix=False splits an
        embedded-quote argument in two, where the fix keeps it whole."""
        assert shlex.split('git commit --message="a b"', posix=False) == [
            "git", "commit", '--message="a', 'b"']
        assert _split_command('git commit --message="a b"') == [
            "git", "commit", "--message=a b"]

    @pytest.mark.skipif(sys.platform != "win32",
                        reason="both execution defects were Windows-only")
    def test_the_old_split_left_quotes_in_the_argument(self, tmp_path):
        """Defect A: posix=False does not remove quotes, so the process got a
        filename with literal quote characters in it and could not open it."""
        d, payload = _payload(tmp_path)
        args = '"%s" "%s"' % (_reader(d), payload)

        old = [sys.executable] + shlex.split(args, posix=False)
        assert any('"' in tok for tok in old[1:]), old
        old_result = run_subprocess(old, tmp_path, timeout=60)
        assert not old_result.ok, (
            "the old split was expected to FAIL: %r" % (old_result.stdout,))

        new_result = run_subprocess(
            [sys.executable] + _split_command(args), tmp_path, timeout=60)
        assert new_result.ok, new_result.stderr
        assert MARK in (new_result.stdout or "")

    @pytest.mark.skipif(sys.platform != "win32",
                        reason="both execution defects were Windows-only")
    def test_the_old_shell_wrapping_mangled_a_quoted_path(self, tmp_path):
        """Defect B: ['cmd', '/C', command] is rendered by list2cmdline, which
        escapes each embedded quote MSVCRT-style - syntax cmd.exe misreads."""
        _, payload = _payload(tmp_path)
        command = 'type "%s"' % payload

        assert '\\"' in subprocess.list2cmdline(["cmd", "/C", command])
        old_result = run_subprocess(["cmd", "/C", command], tmp_path, timeout=60)
        assert not old_result.ok, (
            "the old wrapping was expected to FAIL: %r" % (old_result.stdout,))
        assert MARK not in (old_result.stdout or "")

        new_result = run_subprocess(_shell_argv(command), tmp_path, timeout=60)
        assert new_result.ok, new_result.stderr
        assert MARK in (new_result.stdout or "")
