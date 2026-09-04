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

def _hold(cmd, env=None):
    sys.path.insert(0, str(ROOT))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "localm_launcher_pyw", ROOT / "launcher.pyw")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["localm_launcher_pyw"] = mod
    spec.loader.exec_module(mod)
    return mod._console_hold(cmd, env)


class TestConsoleHold:
    def test_a_native_fault_is_caught_as_well_as_an_ordinary_failure(self):
        """Both signs, or the violent crashes are exactly the ones lost."""
        tail = _hold(["localm", "gui"], {})
        assert "if errorlevel 1 pause" in tail, tail
        assert "if not errorlevel 0 pause" in tail, (
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
