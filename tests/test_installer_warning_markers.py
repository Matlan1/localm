# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.bat's `EnableDelayedExpansion` (active for nearly the whole script) drops
a lone, unpaired `!` on any line it scans - including a hand-typed literal `!`
inside an echo statement's own source text, with no install-path bang required
to trigger it at all. Every `[!]` warning/error marker in the file was affected:
`echo  [!] uv ... is not installed.` printed as ` [] uv ... is not installed.`,
on every install, regardless of the clone's directory name.

A single `^` before the bang (`[^!]`) does NOT fix this - confirmed directly, it
produces the identical broken output, and on a line that also carries a real
`!VAR!` reference it is worse: the escaped-but-still-scanned bang pairs with the
variable's own opening `!`, eating everything in between (including unrelated
text like "Provisioning failed - run later:") and leaving the variable
unsubstituted. `setlocal DisableDelayedExpansion` / `endlocal` (the fix already
used for the two plain %CD% banners) does not work at these sites either: several
of them need a REAL `!VAR!` reference on the SAME line to still expand, which
disabling delayed expansion for the line would break.

The fix that survives every shape actually present in this file - flat, inside
1-3 levels of nested parens, a single-line `if errorlevel 1 echo`, a `||` fallback,
immediately after an unrelated setlocal/endlocal wrap - is a DOUBLE caret,
`[^^!]`. Confirmed empirically against a real cmd.exe for every one of those
shapes before this fix was written; see TestBangSurvivesEveryStructuralShape.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BAT = ROOT / "setup.bat"


@pytest.fixture(scope="module")
def bat():
    return BAT.read_text(encoding="utf-8", errors="replace")


def test_no_unescaped_bang_marker_remains(bat):
    """Regression guard: `[!]` (no carets) must never reappear. A future site
    added the naive way reproduces the exact bug this file exists to catch."""
    assert not re.search(r"\[!\]", bat), (
        "found an unescaped [!] warning marker - it will silently print as "
        "[] under this file's EnableDelayedExpansion; escape it as [^^!]")


def test_every_known_site_uses_the_double_caret_escape(bat):
    sites = [
        'echo  [^^!] uv (the Python package manager localm builds on) is not installed.',
        'echo  [^^!] uv still is not callable after the install attempt.',
        'echo  [^^!] Could not create the environment.',
        'echo  [^^!] Install failed - see the error above.',
        'uv pip install -p .venv -e ".[gpu,audio]" || echo  [^^!] ROCm torch install failed.',
        'uv pip install -p .venv %TORCHSPEC% || echo  [^^!] torch install failed.',
        'uv pip install -p .venv -e ".[hf,audio]" || echo  [^^!] transformers install failed.',
        'echo  [^^!] Provisioning failed - run later: .venv\\Scripts\\localm setup-llama --from "!LLAMABUILD!"',
        'echo  [^^!] Provisioning failed - run later: .venv\\Scripts\\localm setup-llama --backend %BACKEND%',
        "if errorlevel 1 echo  [^^!] Could not build LocaLM.exe",
        "if errorlevel 1 echo  [^^!] Could not record the install manifest",
        'echo  [^^!] No venv Python found - only the marked .venv will be removed.',
        'echo  [^^!] No path given - using the portable .\\home instead.',
    ]
    for site in sites:
        assert site in bat, site
    assert bat.count("[^^!]") == len(sites), (
        "site count changed - update this list (and the executing coverage "
        "below) rather than just this assertion")


def _line_containing(bat_text, needle):
    for line in bat_text.splitlines():
        if needle in line:
            return line
    raise AssertionError("no line contains: {!r}".format(needle))


def _own_backend_block(bat_text):
    """The `if /i "%BACKEND%"=="own" ( ... ) else ( ... )` block that provisions
    llama.cpp, sliced from the real setup.bat text by its own boundaries -
    3 levels of nested parens deep at the site under test, the highest-risk
    shape in the file."""
    start = bat_text.index('if /i "%BACKEND%"=="own" (')
    end = bat_text.index("rem ---- choose where data lives", start)
    return bat_text[start:end]


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe only")
class TestBangSurvivesEveryStructuralShape:
    """Drives the REAL lines/blocks (sliced out of setup.bat, never hand-
    retyped) through a real cmd.exe under active EnableDelayedExpansion,
    asserting the actual printed text contains a literal `[!]` - not `[]`
    (the pre-fix bug) and not a stray caret (an over-escaped attempt)."""

    def _run(self, tmp_path, preamble, body, stdin_input=None):
        probe = tmp_path / "probe.bat"
        probe.write_text(
            "@echo off\r\nsetlocal EnableDelayedExpansion\r\n"
            + preamble + body +
            "\r\nexit /b 0\r\n",
            encoding="utf-8")
        if stdin_input is None:
            return subprocess.run(["cmd", "/c", str(probe)], capture_output=True, text=True,
                                  stdin=subprocess.DEVNULL, timeout=15, cwd=str(tmp_path))
        return subprocess.run(["cmd", "/c", str(probe)], capture_output=True, text=True,
                              input=stdin_input, timeout=15, cwd=str(tmp_path))

    def test_flat_toplevel_line_no_other_bang(self, bat, tmp_path):
        line = _line_containing(
            bat, "uv (the Python package manager localm builds on) is not installed.")
        out = self._run(tmp_path, "", line)
        assert "[!] uv (the Python package manager localm builds on) is not installed." \
            in out.stdout, (out.stdout, out.stderr)

    def test_single_line_if_errorlevel_no_parens(self, bat, tmp_path):
        line = _line_containing(bat, "Could not build LocaLM.exe")
        out = self._run(tmp_path, "cmd /c exit /b 1\r\n", line)
        assert "[!] Could not build LocaLM.exe" in out.stdout, (out.stdout, out.stderr)

    def test_one_level_else_block(self, bat, tmp_path):
        start = bat.index('if exist "%PYBIN%" (')
        end = bat.index("\n)\n", bat.index("No venv Python found", start)) + len("\n)")
        block = bat[start:end]
        assert "No venv Python found" in block, "the else-block boundaries moved; update this test"
        preamble = 'set "PYBIN=nonexistent-marker-for-test.exe"\r\n'
        out = self._run(tmp_path, preamble, block)
        assert "[!] No venv Python found - only the marked .venv will be removed." \
            in out.stdout, (out.stdout, out.stderr)

    def test_three_level_nested_paren_with_a_real_bang_variable_on_the_same_line(
            self, bat, tmp_path):
        """The single highest-risk site in the file: 3 levels of nested `if (
        ... )` blocks, and the echo line itself ALSO carries a real
        `!LLAMABUILD!` delayed-expansion variable reference that must keep
        substituting correctly on the exact same line as the literal `[!]`.

        LLAMABUILD is fed to the block's own real `set /p` prompt via stdin
        (matching how a real user's typed path reaches it) rather than
        pre-set with a plain `set "LLAMABUILD=...!..."` - that assignment
        shape drops the bang from its OWN value under active delayed
        expansion (a different, already-known defect - see
        `dev-notes/installer-b1-b2-fix-2026-09-11.md`), which would corrupt
        the test's input before the site under test ever ran."""
        block = _own_backend_block(bat)
        real_call = '.venv\\Scripts\\localm setup-llama --from "!LLAMABUILD!"'
        assert block.count(real_call) == 2, (
            "expected the real call once and the echo's suggested retry once; "
            "the block shape moved - update this test")
        stubbed = block.replace(real_call, "cmd /c exit /b 1", 1)
        stubbed += "\r\ngoto :after_flush_stub\r\n:flush\r\nexit /b 0\r\n:after_flush_stub\r\n"
        out = self._run(tmp_path, 'set "BACKEND=own"\r\n', stubbed,
                        stdin_input="C:\\some\\bang!path\r\n")
        expected = (
            '[!] Provisioning failed - run later: .venv\\Scripts\\localm '
            'setup-llama --from "C:\\some\\bang!path"'
        )
        assert expected in out.stdout, (out.stdout, out.stderr)

    def test_double_or_fallback_after_a_failed_pip_install(self, bat, tmp_path):
        """The `cmd || echo [^^!] ...` shape - distinct from `if errorlevel 1`,
        since the marker fires from a short-circuited command, not a separate
        conditional line."""
        line = _line_containing(bat, "ROCm torch install failed")
        stubbed = line.replace(
            'uv pip install -p .venv -e ".[gpu,audio]"', "cmd /c exit /b 1", 1)
        out = self._run(tmp_path, "", stubbed)
        assert "[!] ROCm torch install failed." in out.stdout, (out.stdout, out.stderr)
