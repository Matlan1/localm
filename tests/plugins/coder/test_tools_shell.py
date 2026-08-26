# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for shell and test-runner tools in localm.plugins.coder.tools:
  tool_run_shell, tool_run_tests, _detect_test_runner
"""

import json
import shutil
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from localm.plugins.coder.tools import (
    tool_run_shell,
    tool_run_tests,
    _detect_test_runner,
    _needs_shell,
)
from localm.plugins.coder.tools.shell import _js_test_command


def _launcher(launched) -> str:
    """The program *launched* actually starts.

    The platform shell's launch form differs by platform: POSIX gets an argv list
    (execv receives it verbatim), Windows a raw command-line STRING, because an
    argv list is re-quoted by list2cmdline in syntax cmd.exe misreads. See
    tools/base.py:platform_shell.
    """
    return launched.split()[0] if isinstance(launched, str) else launched[0]


# ---------------------------------------------------------------------------
#  tool_run_shell
# ---------------------------------------------------------------------------

class TestRunShell:
    def _make_proc(self, stdout="", stderr="", returncode=0):
        p = MagicMock()
        p.stdout = stdout
        p.stderr = stderr
        p.returncode = returncode
        return p

    def _patch_run(self, **kwargs):
        return patch(
            "localm.plugins.coder.tools.subprocess.run",
            return_value=self._make_proc(**kwargs),
        )

    def test_success_returns_ok(self, tmp_path):
        with self._patch_run(stdout="hello\n", returncode=0):
            r = tool_run_shell(tmp_path, "echo hello")
        assert r.ok
        assert "hello" in r.output

    def test_nonzero_exit_returns_not_ok(self, tmp_path):
        with self._patch_run(stdout="", stderr="error", returncode=1):
            r = tool_run_shell(tmp_path, "false")
        assert not r.ok
        assert "exit_code" in r.output

    def test_output_contains_exit_code_xml(self, tmp_path):
        with self._patch_run(stdout="out\n", returncode=0):
            r = tool_run_shell(tmp_path, "echo out")
        assert "<exit_code>0</exit_code>" in r.output

    def test_stderr_included_in_output(self, tmp_path):
        with self._patch_run(stdout="", stderr="warning!\n", returncode=0):
            r = tool_run_shell(tmp_path, "cmd")
        assert "STDERR" in r.output
        assert "warning!" in r.output

    def test_timeout_returns_error(self, tmp_path):
        with patch(
            "localm.plugins.coder.tools.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["cmd"], 30),
        ):
            r = tool_run_shell(tmp_path, "sleep 100", timeout=30)
        assert not r.ok
        assert "timed out" in r.output.lower()

    def test_no_output_shows_placeholder(self, tmp_path):
        with self._patch_run(stdout="", stderr="", returncode=0):
            r = tool_run_shell(tmp_path, "true")
        assert "(no output)" in r.output

    def test_summary_contains_command_prefix(self, tmp_path):
        with self._patch_run(stdout="", returncode=0):
            r = tool_run_shell(tmp_path, "echo test")
        assert "echo test" in r.summary

    def test_large_output_truncated(self, tmp_path):
        big = "x" * 50_000
        with self._patch_run(stdout=big, returncode=0):
            r = tool_run_shell(tmp_path, "bigcmd")
        assert r.ok
        assert r.truncated

    def test_privacy_mode_passes_env(self, tmp_path):
        captured_env = {}

        def fake_run(cmd, **kwargs):
            captured_env.update(kwargs.get("env") or {})
            return self._make_proc(stdout="ok")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_run_shell(tmp_path, "echo hi", _privacy=True)

        assert captured_env.get("HISTFILE") in ("NUL", "/dev/null")
        assert captured_env.get("HISTSIZE") == "0"

    def test_simple_command_uses_arg_list(self, tmp_path):
        """Commands whose executable exists on PATH run as an argument list."""
        captured_cmd = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return self._make_proc(stdout="")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_run_shell(tmp_path, "git status")   # git resolves via PATH

        # Must NOT go through cmd/sh - first token names the executable.
        # resolve_runner() now absolutises argv[0] for every resolvable
        # command (not only npm/yarn/npx), so this is the resolved path
        # rather than the bare name.
        from pathlib import Path
        assert Path(captured_cmd[0]).stem.lower() == "git"
        assert "status" in captured_cmd

    def test_shell_builtin_routed_through_shell(self, tmp_path):
        """echo/dir/type have no executable on disk - must use the shell,
        otherwise argument-list mode fails with 'file not found'."""
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return self._make_proc(stdout="hi\n")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run), \
             patch("shutil.which", return_value=None):
            r = tool_run_shell(tmp_path, "echo hi")

        assert r.ok
        assert _launcher(captured["cmd"]) in ("cmd", "/bin/sh"), captured["cmd"]

    def test_pipe_uses_shell(self, tmp_path):
        """Commands with pipe operators are routed through the system shell."""
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return self._make_proc(stdout="")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_run_shell(tmp_path, "echo hi | cat")

        launched = captured["cmd"]
        assert _launcher(launched) in ("cmd", "/bin/sh"), launched
        # The command text reaches the shell unchanged, not re-quoted.
        tail = launched if isinstance(launched, str) else launched[-1]
        assert tail.endswith("echo hi | cat"), tail


# ---------------------------------------------------------------------------
#  _needs_shell
# ---------------------------------------------------------------------------

class TestNeedsShell:
    def test_simple_command_no_shell(self):
        assert not _needs_shell("python -m pytest")
        assert not _needs_shell("echo hello")
        assert not _needs_shell("git status")

    def test_pipe_needs_shell(self):
        assert _needs_shell("echo hi | cat")

    def test_redirect_needs_shell(self):
        assert _needs_shell("echo hi > out.txt")
        assert _needs_shell("cat < in.txt")

    def test_and_operator_needs_shell(self):
        assert _needs_shell("make && echo done")

    def test_subshell_needs_shell(self):
        assert _needs_shell("echo $(pwd)")

    def test_quoted_pipe_no_shell(self):
        # A pipe inside quotes is not a shell operator
        assert not _needs_shell("python -c 'print(\"|\")'")

    def test_glob_star_needs_shell(self):
        assert _needs_shell("ls *.py")

    def test_semicolon_needs_shell(self):
        assert _needs_shell("cd src; ls")


# ---------------------------------------------------------------------------
#  _detect_test_runner
# ---------------------------------------------------------------------------

class TestDetectTestRunner:
    def test_detects_pytest_for_python_project(self, tmp_path):
        # Empty dir - defaults to pytest
        cmd = _detect_test_runner(tmp_path)
        assert "pytest" in " ".join(cmd)

    def test_python_runner_uses_current_interpreter(self, tmp_path):
        # The pytest runner invokes sys.executable, not a bare python off PATH.
        import sys
        cmd = _detect_test_runner(tmp_path)
        assert cmd[0] == sys.executable
        assert cmd[1:3] == ["-m", "pytest"]

    def test_detects_cargo_for_rust(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "x"\n')
        cmd = _detect_test_runner(tmp_path)
        assert "cargo" in cmd[0]

    def test_detects_go_test(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/x\n")
        cmd = _detect_test_runner(tmp_path)
        assert "go" in cmd[0]

    def test_detects_npm_when_no_yarn_lock(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "x"}')
        cmd = _detect_test_runner(tmp_path)
        assert "npm" in cmd[0]

    def test_detects_yarn_when_yarn_lock_present(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "x"}')
        (tmp_path / "yarn.lock").write_text("")
        cmd = _detect_test_runner(tmp_path)
        assert "yarn" in cmd[0]


class TestPassWithNoTestsActuallyReachesTheRunner:
    """Appended bare, ``npm test --passWithNoTests`` never reaches the runner.

    npm parses an unknown ``--flag`` as a CLI config and forwards only nopt's
    POSITIONAL remainder to the package script, so the flag is decorative and a
    JS project with no tests is billed a verification failure anyway. It goes
    through npm's own documented ``--`` separator instead, and only to a runner
    that actually has the flag, because with the separator it really does arrive
    and most runners reject an unknown option outright.
    """

    @staticmethod
    def _pkg(tmp_path, test_script):
        (tmp_path / "package.json").write_text(
            json.dumps({"name": "p", "private": True,
                        "scripts": {"test": test_script}}), encoding="utf-8")

    @pytest.mark.parametrize("script", [
        "jest",
        "jest --ci",
        "vitest run",
        "cross-env NODE_ENV=test jest",
        "node_modules/.bin/jest.cmd",
    ])
    def test_a_runner_that_has_the_flag_gets_it_past_the_separator(
            self, tmp_path, script):
        self._pkg(tmp_path, script)
        assert _detect_test_runner(tmp_path)[1:] == [
            "test", "--", "--passWithNoTests"]

    @pytest.mark.parametrize("script", [
        # This repository's own test script. node --test rejects an unknown
        # option, so no separator is inserted for it.
        "node --test --test-force-exit tests-js/*.test.mjs",
        "mocha",
        "node tools/jest-codemod.js",   # merely MENTIONS jest, is not jest
        "jest && eslint .",             # appended args land on eslint, not jest
    ])
    def test_a_runner_without_the_flag_gets_a_plain_test_command(
            self, tmp_path, script):
        self._pkg(tmp_path, script)
        assert _detect_test_runner(tmp_path)[1:] == ["test"]

    @pytest.mark.parametrize("body", ['{"name": "x"}', "{not json"])
    def test_no_readable_test_script_means_no_flag(self, tmp_path, body):
        (tmp_path / "package.json").write_text(body, encoding="utf-8")
        assert _detect_test_runner(tmp_path)[1:] == ["test"]

    def test_yarn_gets_the_flag_bare_because_a_separator_would_break_it(
            self, tmp_path):
        """npm and yarn are OPPOSITES here. yarn classic already forwards a
        bare flag to the script, and warns that a future yarn "will forward any
        explicit -- as-is to the scripts", which would hand the runner a literal
        `--` and demote the flag to a positional argument. Giving yarn npm's
        separator would break a case that works today."""
        self._pkg(tmp_path, "jest")
        (tmp_path / "yarn.lock").write_text("", encoding="utf-8")
        cmd = _detect_test_runner(tmp_path)
        assert "yarn" in cmd[0]
        assert cmd[1:] == ["test", "--passWithNoTests"]

    def test_the_explicit_yarn_runner_also_omits_the_separator(self, tmp_path):
        self._pkg(tmp_path, "jest")
        assert _js_test_command(tmp_path, "yarn")[1:] == [
            "test", "--passWithNoTests"]

    def test_the_explicit_npm_runner_builds_the_same_command_as_auto(
            self, tmp_path):
        """run_tests(runner="npm") and the verify oracle's auto-detection share
        one builder, so this pins that they agree."""
        self._pkg(tmp_path, "jest")
        assert _js_test_command(tmp_path, "npm") == _detect_test_runner(tmp_path)

    def test_the_flag_really_arrives_at_the_runner(self, tmp_path):
        """The end-to-end that the argv assertions above cannot make: run the
        detected command through the REAL npm and read back what the package
        script actually received.

        The second half is the fires-control: the bare, no-separator argv is
        run against the SAME project with the SAME npm in the SAME test, and the
        flag does NOT arrive. Without it a green first half would only prove
        that npm exists, not that the separator is what delivers the flag, which
        a pure argv-shape assertion cannot see.
        """
        if shutil.which("npm") is None:
            pytest.skip("npm is not installed on this box")
        # Named jest.js so detection recognises a supported runner; the file
        # itself only reports its argv, which is the thing under test.
        (tmp_path / "jest.js").write_text(
            'console.log("ARGV=" + JSON.stringify(process.argv.slice(2)));\n',
            encoding="utf-8")
        self._pkg(tmp_path, "node jest.js")

        cmd = _detect_test_runner(tmp_path)
        assert cmd[1:] == ["test", "--", "--passWithNoTests"]
        fixed = subprocess.run(cmd, cwd=tmp_path, capture_output=True,
                               text=True, timeout=180)
        assert "--passWithNoTests" in fixed.stdout, (
            "the separator form did not deliver the flag to the script: "
            f"stdout={fixed.stdout!r} stderr={fixed.stderr!r}")

        bare = subprocess.run([cmd[0], "test", "--passWithNoTests"],
                              cwd=tmp_path, capture_output=True, text=True,
                              timeout=180)
        assert "ARGV=[]" in bare.stdout, (
            "npm forwarded a BARE --passWithNoTests after all. If that is real, "
            "the separator is unnecessary and this fix needs re-measuring - do "
            "not just relax the assertion. "
            f"stdout={bare.stdout!r} stderr={bare.stderr!r}")


# ---------------------------------------------------------------------------
#  tool_run_tests
# ---------------------------------------------------------------------------

class TestRunTests:
    def _make_proc(self, stdout="", stderr="", returncode=0):
        p = MagicMock()
        p.stdout = stdout
        p.stderr = stderr
        p.returncode = returncode
        return p

    def _patch_run(self, **kwargs):
        return patch(
            "localm.plugins.coder.tools.subprocess.run",
            return_value=self._make_proc(**kwargs),
        )

    def test_auto_runner_runs_tests(self, tmp_path):
        with self._patch_run(stdout="1 passed\n", returncode=0):
            r = tool_run_tests(tmp_path, runner="auto")
        assert r.ok
        assert "passed" in r.summary

    def test_explicit_pytest_runner(self, tmp_path):
        captured = []

        def fake_run(cmd, **kwargs):
            captured.extend(cmd)
            return self._make_proc(stdout="2 passed")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_run_tests(tmp_path, runner="pytest")
        assert "pytest" in " ".join(captured)

    def test_failure_returns_not_ok(self, tmp_path):
        with self._patch_run(stdout="1 failed\n", returncode=1):
            r = tool_run_tests(tmp_path)
        assert not r.ok
        assert "failed" in r.summary

    def test_timeout_returns_error(self, tmp_path):
        with patch(
            "localm.plugins.coder.tools.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["pytest"], 120),
        ):
            r = tool_run_tests(tmp_path)
        assert not r.ok
        assert "timed out" in r.output.lower()

    def test_runner_not_found_returns_error(self, tmp_path):
        with patch(
            "localm.plugins.coder.tools.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            r = tool_run_tests(tmp_path, runner="pytest")
        assert not r.ok
        assert "not found" in r.output.lower()

    def test_unknown_runner_returns_error(self, tmp_path):
        r = tool_run_tests(tmp_path, runner="not_a_runner")
        assert not r.ok
        assert "unknown runner" in r.output.lower()

    def test_path_arg_appended_to_command(self, tmp_path):
        captured = []

        def fake_run(cmd, **kwargs):
            captured.extend(cmd)
            return self._make_proc(stdout="ok")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_run_tests(tmp_path, runner="pytest", path="tests/")
        assert "tests/" in captured

    def test_extra_args_appended(self, tmp_path):
        captured = []

        def fake_run(cmd, **kwargs):
            captured.extend(cmd)
            return self._make_proc(stdout="ok")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_run_tests(tmp_path, runner="pytest", extra_args="-v -k test_foo")
        assert "-v" in captured
        assert "-k" in captured
        assert "test_foo" in captured

    def test_output_contains_runner_and_status(self, tmp_path):
        with self._patch_run(stdout="all good\n", returncode=0):
            r = tool_run_tests(tmp_path, runner="pytest")
        assert "<runner>" in r.output
        assert "<status>" in r.output


class TestCallerArgsReachTheRunner:
    """``run_tests``' own ``path`` and ``extra_args`` have the same npm problem
    ``--passWithNoTests`` has one code path over: appended bare, npm swallows
    anything flag-shaped and quietly runs a plain suite instead.

    ``npm test --watch`` gives the package script ``ARGV=[]`` plus an "Unknown
    cli config" warning, while ``npm test -- --watch`` delivers it, so
    ``run_tests(runner="npm", extra_args="--watch")`` would report success for a
    run nobody asked for.
    """

    @staticmethod
    def _pkg(tmp_path, test_script="node argv.js"):
        (tmp_path / "package.json").write_text(
            json.dumps({"name": "p", "private": True,
                        "scripts": {"test": test_script}}), encoding="utf-8")

    @staticmethod
    def _cmd_for(tmp_path, **kwargs) -> list:
        """The argv ``run_tests`` would launch, with the subprocess stubbed."""
        captured = []

        def fake_run(cmd, **_kw):
            captured.append(list(cmd))
            p = MagicMock()
            p.stdout, p.stderr, p.returncode = "ok", "", 0
            return p

        with patch("localm.plugins.coder.tools.subprocess.run",
                   side_effect=fake_run):
            tool_run_tests(tmp_path, **kwargs)
        assert len(captured) == 1, captured
        return captured[0]

    def test_npm_gets_the_separator_before_a_flag(self, tmp_path):
        self._pkg(tmp_path)
        assert self._cmd_for(tmp_path, runner="npm", extra_args="--watch")[1:] \
            == ["test", "--", "--watch"]

    def test_npm_gets_the_separator_before_a_path_too(self, tmp_path):
        """A positional needs no separator of its own (``npm test somepath``
        already arrives), but one in front of it is measurably inert - ``npm
        test -- somepath`` delivers the same ``["somepath"]`` - and a call
        passing BOTH has to put them on the same side of it."""
        self._pkg(tmp_path)
        assert self._cmd_for(tmp_path, runner="npm", path="tests")[1:] \
            == ["test", "--", "tests"]

    def test_npm_path_and_flags_land_together_past_one_separator(self, tmp_path):
        self._pkg(tmp_path)
        assert self._cmd_for(tmp_path, runner="npm", path="tests",
                             extra_args="--watch -t slow")[1:] \
            == ["test", "--", "tests", "--watch", "-t", "slow"]

    def test_the_auto_detected_npm_command_gets_it_as_well(self, tmp_path):
        """``auto`` is the branch the model actually reaches, and it does not go
        through the explicit npm branch, so it needs its own pin."""
        self._pkg(tmp_path)
        assert self._cmd_for(tmp_path, extra_args="--watch")[1:] \
            == ["test", "--", "--watch"]

    def test_an_existing_separator_is_never_doubled(self, tmp_path):
        """The ``--passWithNoTests`` command already carries one and everything
        appended lands after it. A second reaches the runner as a literal
        argument - ``npm test -- -- --watch`` delivers ``["--", "--watch"]`` -
        demoting the flag behind it."""
        self._pkg(tmp_path, "jest")
        assert self._cmd_for(tmp_path, runner="npm", extra_args="--watch")[1:] \
            == ["test", "--", "--passWithNoTests", "--watch"]

    def test_a_caller_written_separator_is_honoured_not_doubled(self, tmp_path):
        self._pkg(tmp_path)
        assert self._cmd_for(tmp_path, runner="npm",
                             extra_args="-- --watch")[1:] == ["test", "--", "--watch"]

    def test_no_caller_args_means_no_separator(self, tmp_path):
        self._pkg(tmp_path)
        assert self._cmd_for(tmp_path, runner="npm")[1:] == ["test"]

    def test_yarn_never_gets_a_separator(self, tmp_path):
        """npm and yarn are opposites. yarn classic forwards a bare flag today
        and warns that a future yarn "will forward any explicit -- as-is to the
        scripts", which would hand the runner a literal ``--`` and demote the
        flag to a positional. Giving yarn npm's separator would break the case
        npm needs it for."""
        self._pkg(tmp_path)
        (tmp_path / "yarn.lock").write_text("", encoding="utf-8")
        assert self._cmd_for(tmp_path, runner="yarn", path="tests",
                             extra_args="--watch")[1:] \
            == ["test", "tests", "--watch"]
        # ...and through auto-detection, which is where yarn.lock is read.
        assert self._cmd_for(tmp_path, extra_args="--watch")[1:] \
            == ["test", "--watch"]

    @pytest.mark.parametrize("runner", ["pytest", "cargo", "go"])
    def test_the_other_runners_are_untouched(self, runner, tmp_path):
        """They parse their own argv, so their flags already arrive; a
        separator here would be an invented argument."""
        cmd = self._cmd_for(tmp_path, runner=runner, path="tests",
                            extra_args="--watch")
        assert "--" not in cmd
        assert cmd[-2:] == ["tests", "--watch"]

    def test_the_args_really_arrive_when_run_tests_launches_npm(self, tmp_path):
        """The end-to-end the argv assertions above cannot make: launch the REAL
        npm through ``run_tests`` and read back what the package script actually
        received.

        The second half is the fires-control: the bare-appended argv is run
        against the SAME project with the SAME npm in the SAME test, and the
        flag does NOT arrive while the positional does. Without it a green first
        half would only prove that npm exists, not that the separator is what
        delivers the flag, which an argv-shape assertion cannot see.
        """
        if shutil.which("npm") is None:
            pytest.skip("npm is not installed on this box")
        (tmp_path / "argv.js").write_text(
            'console.log("ARGV=" + JSON.stringify(process.argv.slice(2)));\n',
            encoding="utf-8")
        # node argv.js is not a --passWithNoTests runner, so it carries no
        # separator of its own and the only one in play is the one run_tests adds.
        self._pkg(tmp_path, "node argv.js")
        (tmp_path / "sub").mkdir()

        r = tool_run_tests(tmp_path, runner="npm", path="sub",
                           extra_args="--watch")
        assert r.ok, r.output
        assert 'ARGV=["sub","--watch"]' in r.output, (
            "run_tests did not deliver its caller args to the package script: "
            f"{r.output!r}")

        npm = _js_test_command(tmp_path, "npm")[0]
        bare = subprocess.run([npm, "test", "sub", "--watch"], cwd=tmp_path,
                              capture_output=True, text=True, timeout=180)
        assert 'ARGV=["sub"]' in bare.stdout, (
            "npm forwarded a BARE --watch after all. If that is real, the "
            "separator is unnecessary and this fix needs re-measuring - do not "
            "just relax the assertion. "
            f"stdout={bare.stdout!r} stderr={bare.stderr!r}")
