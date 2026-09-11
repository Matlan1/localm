# SPDX-License-Identifier: AGPL-3.0-or-later
"""A crashing server must never take the console window with it.

A console window closes the instant its owning process exits, so whatever the
crash printed last - the error itself - goes with it. That matters most for a
native fault, which is also the case least likely to have managed to write its
trace file, leaving the console as the only record.

The trap this pins: ``if errorlevel 1`` is a >= test against a SIGNED value, so
it matches an ordinary failure and silently misses a native fault, which exits
NEGATIVE (an access violation exits -1073741819). The window therefore stayed
open for tidy failures and closed for violent ones.

The batch tests drive the REAL shipped localm.bat with its localm invocation
swapped for an exit code, so they cannot pass against a file that no longer
holds the window.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# An access violation and a stack buffer overrun, as cmd reports them.
ACCESS_VIOLATION = "-1073741819"
STACK_OVERRUN = "-1073740791"


# --------------------------------------------------------------------------- #
#  The launcher's own console hold                                             #
# --------------------------------------------------------------------------- #

def _load_launcher_pyw(module_name):
    """Import launcher.pyw under *module_name*.

    ``.pyw`` is in ``importlib.machinery.SOURCE_SUFFIXES`` only on Windows, so
    ``spec_from_file_location`` returns None elsewhere and ``spec.loader``
    raises AttributeError. Naming the loader keeps the import working on every
    platform. See test_launcher_pyw_loads_without_the_pyw_suffix.
    """
    import importlib.util
    from importlib.machinery import SourceFileLoader

    path = str(ROOT / "launcher.pyw")
    spec = importlib.util.spec_from_file_location(
        module_name, path, loader=SourceFileLoader(module_name, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _hold(cmd, env=None):
    sys.path.insert(0, str(ROOT))
    mod = _load_launcher_pyw("localm_launcher_pyw")
    return mod._console_hold(cmd, env)


class TestConsoleHold:
    def test_a_native_fault_is_caught_as_well_as_an_ordinary_failure(self):
        """Both signs, or the violent crashes are exactly the ones lost."""
        tail = _hold(["localm", "gui"], {})
        assert tail == " || pause", tail
        assert "errorlevel" not in tail, (
            "a negative exit code is never >= 1, so this is what catches a "
            f"native fault: {tail}")

    def test_a_clean_exit_still_closes_outside_debug(self):
        tail = _hold(["localm", "gui"], {})
        assert not tail.strip().endswith("& pause")

    def test_debug_holds_the_window_whatever_happened(self):
        """The log is the reason the console is open."""
        assert _hold(["localm", "gui", "--debug"], {}) == " & pause"

    def test_debug_is_also_read_from_the_environment(self):
        """The coder mode has no --debug flag and uses LOCALM_DEBUG instead."""
        assert _hold(["localm", "coder"], {"LOCALM_DEBUG": "1"}) == " & pause"

    def test_no_environment_falls_back_to_the_process_environment(self, monkeypatch):
        monkeypatch.setenv("LOCALM_DEBUG", "1")
        assert _hold(["localm", "coder"], None) == " & pause"

    def test_an_unset_debug_variable_is_not_debug(self):
        tail = _hold(["localm", "gui"], {"LOCALM_DEBUG": ""})
        assert tail != " & pause"


# --------------------------------------------------------------------------- #
#  The Windows command line _spawn_detached hands to cmd.exe                   #
# --------------------------------------------------------------------------- #

CMDLINE_MARKER = "LAUNCHER-CMDLINE-PROBE-RAN"
SECOND_STUB_MARKER = "LAUNCHER-CMDLINE-SECOND-STUB-RAN"


def _win_cmdline(cmd, env=None):
    mod = _load_launcher_pyw("localm_launcher_pyw")
    return mod._windows_command_line(cmd, env)


def _write_marker_stub(directory, marker=CMDLINE_MARKER, exit_code=0):
    """A .bat that prints a fixed marker and exits *exit_code*.

    It never echoes its own arguments: batch re-expands ``%*`` into the
    ``echo`` line, so an argument carrying a metacharacter would be split by
    the batch parser and the stub would appear to fail for a reason that has
    nothing to do with the command line under test.
    """
    directory.mkdir(parents=True, exist_ok=True)
    bat = directory / "probe.bat"
    bat.write_text(
        "@echo off\r\necho {}\r\nexit /b {}\r\n".format(marker, exit_code),
        encoding="utf-8")
    return bat


def _run_cmdline(line):
    """Run a cmd.exe /c command line for real, with the two traps this file's
    own docstring warns about: pause blocks on stdin without DEVNULL, and a
    hung child needs a bounded timeout."""
    return subprocess.run(line, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=15)


class TestWindowsCommandLineBuilder:
    """Structural checks on the string _windows_command_line returns. None of
    these execute cmd.exe, so they run on every platform."""

    def test_the_hold_is_not_escaped(self):
        line = _win_cmdline(["localm", "gui"], {})
        assert line.endswith(' || pause'), line
        assert "^|" not in line, line
        assert "^&" not in line, line

    def test_the_debug_hold_is_not_escaped(self):
        line = _win_cmdline(["localm", "gui", "--debug"], {})
        assert line.endswith(' & pause'), line
        assert "^&" not in line, line

    def test_every_argument_is_quoted(self):
        line = _win_cmdline(["prog", "a", "b c"], {})
        assert '"prog"' in line
        assert '"a"' in line
        assert '"b c"' in line

    def test_the_whole_line_starts_with_cmd_and_one_outer_quote(self):
        line = _win_cmdline(["prog"], {})
        assert line.startswith('cmd.exe /c "')

    def test_an_embedded_quote_is_refused_not_corrupted(self):
        """A literal quote toggles cmd.exe's own quote-tracking regardless of
        the backslash escaping applied for the child's argv parser, so an
        embedded quote next to a metacharacter (e.g. `x"&calc&"y`) cannot be
        represented safely through both layers at once. Refusing to build
        the command line is the safe response."""
        with pytest.raises(ValueError):
            _win_cmdline(["prog", "--cwd", 'x"&calc&"y'], {})

    def test_a_plain_trailing_backslash_survives(self):
        line = _win_cmdline(["prog", "D:\\"], {})
        assert '"D:\\\\"' in line


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe only")
class TestWindowsCommandLineExecution:
    """Drives the real string through a real cmd.exe, in tmp_path, against
    hostile directory names. A source-level assertion cannot tell a working
    fix from one that merely looks right: the mandatory hold test passes for
    a shape that does not run at all, so this has to execute."""

    def test_a_space_in_the_path_runs(self, tmp_path):
        stub = _write_marker_stub(tmp_path / "My Repo")
        line = _win_cmdline([str(stub)], {})
        out = _run_cmdline(line)
        assert CMDLINE_MARKER in out.stdout, (line, out.stdout, out.stderr)

    def test_an_ampersand_in_the_path_runs(self, tmp_path):
        stub = _write_marker_stub(tmp_path / "R&D")
        line = _win_cmdline([str(stub)], {})
        out = _run_cmdline(line)
        assert CMDLINE_MARKER in out.stdout, (line, out.stdout, out.stderr)

    def test_a_space_and_an_ampersand_together_run(self, tmp_path):
        """The case that separates a half-fix from a fix: blind caret-escaping
        passes the ampersand-only case above and fails this one, because it
        also escapes the ampersand that sits inside this path's own quotes."""
        stub = _write_marker_stub(tmp_path / "My R&D Dir")
        line = _win_cmdline([str(stub)], {})
        out = _run_cmdline(line)
        assert CMDLINE_MARKER in out.stdout, (line, out.stdout, out.stderr)

    def test_no_injection_from_an_argument_value(self, tmp_path):
        main_stub = _write_marker_stub(tmp_path / "main")
        second_stub = _write_marker_stub(tmp_path / "second", marker=SECOND_STUB_MARKER)
        payload = "X&{}".format(second_stub)
        assert " " not in payload
        cmd = [str(main_stub), "--cwd", payload]
        line = _win_cmdline(cmd, {})
        out = _run_cmdline(line)
        assert CMDLINE_MARKER in out.stdout, (line, out.stdout, out.stderr)
        assert SECOND_STUB_MARKER not in out.stdout, (line, out.stdout, out.stderr)

    def test_the_hold_still_pauses_on_a_failing_child_with_a_space_in_the_path(self, tmp_path):
        stub = _write_marker_stub(tmp_path / "My Repo", exit_code=1)
        line = _win_cmdline([str(stub)], {})
        out = _run_cmdline(line)
        assert "press any key" in out.stdout.lower(), (line, out.stdout, out.stderr)


# --------------------------------------------------------------------------- #
#  The shipped localm.bat                                                      #
# --------------------------------------------------------------------------- #

def _run_bat(tmp_path, exit_code, args):
    """Run the REAL localm.bat with its localm call replaced by an exit code.

    stdin is /dev/null so `pause` returns at once; its prompt in stdout is what
    says the window was held."""
    src = (ROOT / "localm.bat").read_text(encoding="utf-8")
    stub = f"cmd /c exit /b {exit_code}"
    patched = (src.replace('"%PY%" -m localm %*', stub)
                  .replace('"%PY%" -m localm run %MODEL%', stub))
    assert stub in patched, "the localm invocation moved; this test is blind"
    bat = tmp_path / "probe.bat"
    bat.write_text(patched, encoding="utf-8")
    out = subprocess.run(["cmd", "/c", str(bat), *args], capture_output=True,
                         text=True, stdin=subprocess.DEVNULL, cwd=str(tmp_path))
    return "press any key" in (out.stdout or "").lower()


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe only")
class TestLocalmBatHold:
    def test_a_native_fault_holds_the_window(self, tmp_path):
        """The regression: this code is negative, so `if errorlevel 1` never
        matched it and the window closed on the crash worth reading."""
        assert _run_bat(tmp_path, ACCESS_VIOLATION, ["run", "x"])

    def test_a_stack_overrun_holds_the_window(self, tmp_path):
        assert _run_bat(tmp_path, STACK_OVERRUN, ["run", "x"])

    def test_an_ordinary_failure_holds_the_window(self, tmp_path):
        assert _run_bat(tmp_path, "1", ["run", "x"])

    def test_a_clean_exit_closes(self, tmp_path):
        assert not _run_bat(tmp_path, "0", ["run", "x"])

    def test_debug_holds_the_window_even_on_a_clean_exit(self, tmp_path):
        assert _run_bat(tmp_path, "0", ["gui", "--debug"])
