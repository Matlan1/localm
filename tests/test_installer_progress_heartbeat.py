# SPDX-License-Identifier: AGPL-3.0-or-later
"""The setup heartbeat must never run over a command that draws live progress.

MECHANISM: uv draws a live byte-progress readout whenever its output goes to a
terminal, and redraws it IN PLACE (cursor up N lines, rewrite). The heartbeat is
a detached writer printing into that same console. Its line shifts the cursor, so
uv's next frame lands a line low: the previous frame is stranded for good and the
heartbeat's line is overwritten by the redraw that follows it. During
`Installing PyTorch (AMD ROCm, gfx103X) + transformers ...` that leaves one
orphaned progress frame per heartbeat tick and no heartbeat line at all.

So the heartbeat is correct in exactly ONE situation - the command's output is
CAPTURED (redirected to a file, or into a shell variable) and the console would
otherwise be silent for a long time. That is the venv-creation retry, and only it.

These tests run against the REAL shipped setup.bat / setup.sh, not a fixture, so
they catch the heartbeat being copied back to a live-output site by anyone, in
any future edit.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BAT = ROOT / "setup.bat"
SH = ROOT / "setup.sh"


def _lines(p):
    return p.read_text(encoding="utf-8", errors="replace").splitlines()


def _heartbeat_regions(lines, start_tok, stop_tok, is_call):
    """[(start_lineno, [covered lines])] for each heartbeat_start .. next stop."""
    regions = []
    open_at = None
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if not is_call(s):
            continue
        if start_tok in s:
            open_at = (i, [])
        elif stop_tok in s and open_at is not None:
            regions.append(open_at)
            open_at = None
        elif open_at is not None:
            open_at[1].append(s)
    if open_at is not None:            # unterminated: still report what it covers
        regions.append(open_at)
    return regions


def _bat_is_call(s):
    low = s.lower()
    return low.startswith("call :heartbeat") or not low.startswith("rem")


def _sh_is_call(s):
    return not s.startswith("#")


def _captures_output(cmd: str) -> bool:
    """True when this command's stdout does NOT reach the console live."""
    return ">" in cmd or "$(" in cmd or "`" in cmd


CASES = [
    pytest.param(BAT, "call :heartbeat_start", "call :heartbeat_stop", _bat_is_call,
                 id="setup.bat"),
    pytest.param(SH, "heartbeat_start ", "heartbeat_stop", _sh_is_call, id="setup.sh"),
]


@pytest.mark.parametrize("path,start_tok,stop_tok,is_call", CASES)
def test_no_heartbeat_over_a_command_that_writes_live_to_the_console(
        path, start_tok, stop_tok, is_call):
    """Every uv invocation under a heartbeat must have its output captured."""
    regions = _heartbeat_regions(_lines(path), start_tok, stop_tok, is_call)
    assert regions, f"no heartbeat regions found in {path.name} - parser is broken"

    offenders = []
    for start_line, covered in regions:
        for cmd in covered:
            if cmd.startswith("uv ") and not _captures_output(cmd):
                offenders.append(f"{path.name}:{start_line} guards live command: {cmd}")

    assert not offenders, (
        "A heartbeat is running over a command that writes live progress to the "
        "console; its ticks will strand uv's progress frames on screen:\n  "
        + "\n  ".join(offenders))


@pytest.mark.parametrize("path,start_tok,stop_tok,is_call", CASES)
def test_the_only_heartbeat_left_is_the_captured_venv_creation(
        path, start_tok, stop_tok, is_call):
    """Pins the ONE legitimate site, so a re-copy elsewhere shows up as a count."""
    regions = _heartbeat_regions(_lines(path), start_tok, stop_tok, is_call)
    assert len(regions) == 1, (
        f"{path.name} has {len(regions)} heartbeat region(s), expected exactly 1 "
        "(the captured `uv venv` retry). A new one is only correct if that "
        "command's output is captured - see this module's docstring.")
    covered = " ".join(regions[0][1])
    assert "uv venv" in covered, (
        f"{path.name}: the surviving heartbeat no longer guards `uv venv`")


def test_the_torch_and_base_installs_print_their_own_progress():
    """The two sites from the bug report must have no heartbeat around them."""
    for path, installs in (
        (BAT, ['uv pip install -p .venv -e ".[%EXTRAS%]"',
               'uv pip install -p .venv -e ".[gpu,audio]"',
               "uv pip install -p .venv %TORCHSPEC%"]),
        (SH, ['uv pip install -p .venv -e ".[${EXTRAS}]"',
              "uv pip install -p .venv $TORCHSPEC"]),
    ):
        text = path.read_text(encoding="utf-8", errors="replace")
        for cmd in installs:
            assert cmd in text, f"{path.name}: install line moved or changed: {cmd}"
        start = "call :heartbeat_start" if path is BAT else "heartbeat_start "
        for chunk in text.split(start)[1:]:
            head = chunk.split("heartbeat_stop")[0]
            for cmd in installs:
                assert cmd not in head, (
                    f"{path.name}: `{cmd}` is back under a heartbeat - it writes "
                    "live progress to the console and the ticks will corrupt it.")
