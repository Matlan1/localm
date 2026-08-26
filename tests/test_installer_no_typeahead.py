# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup must never let TYPE-AHEAD answer a question the user has not seen.

Setup runs long steps between questions (a uv download, a Python download, a venv
build, a backend provision). Anything typed while one of those runs sits in the
terminal's input queue and is delivered to the NEXT prompt - so one stray Enter
silently accepts a default, and a double Enter accepts two questions in a row.
A `set /p` in place of a `choice` only makes the confirming Enter belong to its
own prompt; it does not empty the queue, and a `choice` prompt consumes a
buffered keypress outright. The queue must be emptied right before each question.

Static assertions: the real failure needs a terminal with keystrokes queued
during a multi-second download, which no unit test can stage. What CAN be pinned
mechanically is that every prompt is preceded by the drop, so a prompt added
later cannot quietly reintroduce it.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _bat_lines() -> list[str]:
    return (ROOT / "setup.bat").read_text(encoding="utf-8").split("\n")


def _sh() -> str:
    return (ROOT / "setup.sh").read_text(encoding="utf-8")


def _is_prompt(line: str) -> bool:
    s = line.strip().lower()
    if s.startswith("rem"):
        return False
    return s.startswith("set /p ") or s.startswith("choice /c ")


class TestBatchFlushesBeforeEveryPrompt:
    def test_every_prompt_is_preceded_by_a_flush(self):
        lines = _bat_lines()
        unguarded = []
        for i, ln in enumerate(lines):
            if not _is_prompt(ln):
                continue
            # the nearest preceding non-blank, non-comment line must be the flush
            j = i - 1
            while j >= 0 and (not lines[j].strip()
                              or lines[j].strip().lower().startswith("rem")):
                j -= 1
            if j < 0 or lines[j].strip().lower() != "call :flush":
                unguarded.append(f"  line {i + 1}: {ln.strip()[:70]}")
        assert not unguarded, (
            "these setup.bat prompts can be answered by type-ahead the user never "
            "saw; put `call :flush` immediately before each:\n" + "\n".join(unguarded))

    def test_the_flush_subroutine_exists(self):
        assert "\n:flush\n" in "\n".join(_bat_lines())

    def test_the_flush_actually_empties_the_console_queue(self):
        """It must call the real API, not just echo something."""
        body = "\n".join(_bat_lines())
        start = body.index("\n:flush\n")
        sub = body[start:start + 500]
        assert "FlushInputBuffer" in sub, \
            "the flush must actually empty the console input queue"

    def test_the_flush_can_never_block_an_install(self):
        """Best-effort: with no console (piped/CI) the flush must no-op, not error
        out and abort setup."""
        body = "\n".join(_bat_lines())
        start = body.index("\n:flush\n")
        sub = body[start:start + 500]
        assert "catch" in sub, "a flush failure must not break setup"
        assert ">nul 2>nul" in sub, "a flush must not print noise into the install"

    def test_there_is_at_least_one_prompt_to_guard(self):
        """Guards the guard: if the greps ever stop matching, the test above would
        pass vacuously over zero prompts."""
        assert sum(1 for ln in _bat_lines() if _is_prompt(ln)) >= 10


class TestShellDrainsBeforeEveryPrompt:
    def test_every_prompt_goes_through_ask(self):
        """setup.sh funnels all prompts through ask(), so the drain lives in one
        place. A bare `read -r -p` outside ask() would bypass it."""
        stray = [ln.strip() for ln in _sh().split("\n")
                 if re.search(r"^\s*read\s+-r\s+-p", ln)]
        # exactly one: the read inside ask() itself
        assert len(stray) == 1, (
            "setup.sh must ask only via ask() (which drains type-ahead first); "
            f"found {len(stray)} raw read -r -p sites: {stray}")

    def test_ask_drains_before_reading(self):
        src = _sh()
        start = src.index("ask() {")
        body = src[start:src.index("\n}", start)]
        drain = body.index("read -r -t")
        prompt = body.index('read -r -p "$prompt"')
        assert drain < prompt, "ask() must drain type-ahead BEFORE it prompts"

    def test_the_drain_is_terminal_only(self):
        """Piped input is deliberate (a script feeding answers); draining it would
        eat the caller's real answers."""
        src = _sh()
        start = src.index("ask() {")
        body = src[start:src.index("\n}", start)]
        assert "[ -t 0 ]" in body, \
            "the drain must only run when stdin is a terminal"

    def test_yes_mode_never_reaches_the_drain(self):
        """--yes returns the default before any read, so a non-interactive install
        is untouched."""
        src = _sh()
        start = src.index("ask() {")
        body = src[start:src.index("\n}", start)]
        yes = body.index('if [ "$YES" = 1 ]')
        drain = body.index("[ -t 0 ]")
        assert yes < drain, "--yes must short-circuit before the drain"
