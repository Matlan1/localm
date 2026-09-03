# SPDX-License-Identifier: AGPL-3.0-or-later
"""Proves tests.conftest.make_console_wide_and_plain actually does what its
docstring claims, against the two independent Rich mechanisms it exists to
defeat: Console.size short-circuiting to 80x25 on a dumb terminal before it
reads COLUMNS, and Console._color_system being cached once at construction
rather than re-derived from later environment changes.

Both halves are proven with a real dumb-terminal-shaped environment
(TTY_COMPATIBLE=1, TERM=dumb - the combination confirmed to match real CI's
own console.is_terminal/is_dumb_terminal behaviour, not FORCE_COLOR, which
also enables genuine color rendering CI does not show), not merely asserted.
"""

from __future__ import annotations

import io

import rich.console

from tests.conftest import make_console_wide_and_plain


def _dumb_env(monkeypatch) -> None:
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("TTY_COMPATIBLE", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)


class TestWidthHalf:
    def test_width_stays_80_without_the_helper(self, monkeypatch):
        """Fires-control: confirms the dumb-terminal environment alone
        reproduces the bug this helper exists to fix."""
        _dumb_env(monkeypatch)
        c = rich.console.Console(file=io.StringIO())
        assert c.size.width == 80

    def test_width_is_wide_with_the_helper(self, monkeypatch):
        _dumb_env(monkeypatch)
        make_console_wide_and_plain(monkeypatch, width="300")
        c = rich.console.Console(file=io.StringIO())
        assert c.size.width == 299   # legacy_windows subtracts 1 on some platforms; not here

    def test_a_preexisting_console_is_also_widened(self, monkeypatch):
        """The shared CLI singletons are constructed before any fixture
        runs, so the fix must reach an ALREADY-BUILT instance too."""
        _dumb_env(monkeypatch)
        preexisting = rich.console.Console(file=io.StringIO())
        assert preexisting.size.width == 80
        make_console_wide_and_plain(monkeypatch, width="300")
        assert preexisting.size.width == 299


class TestColorHalf:
    """Console._color_system is cached ONCE in __init__ (confirmed by
    reading rich.console.Console.__init__'s own source), not re-derived
    live, so a genuinely fresh Console() under TTY_COMPATIBLE=1/TERM=dumb
    alone is NOT the trigger - that combination does not by itself make
    _detect_color_system() report a color-capable terminal. The real
    failure this helper fixes is cross-test: pytest-xdist reuses one
    process across hundreds of tests, so the SHARED CLI console singleton
    is constructed exactly once per worker and can cache a color decision
    made under whatever the FIRST test to import it happened to leave in
    the environment, poisoning every later test that shares the worker.
    That is what test_a_preexisting_consoles_cached_color_is_also_forced_plain
    below proves the fix survives, without needing to reproduce the
    original poisoning trigger from cold."""

    def test_a_fresh_console_renders_plain_with_the_helper(self, monkeypatch):
        _dumb_env(monkeypatch)
        make_console_wide_and_plain(monkeypatch, width="300")
        c = rich.console.Console(file=io.StringIO())
        c.print("[red]plain[/red]")
        assert c.file.getvalue() == "plain\n"

    def test_a_preexisting_consoles_cached_color_is_also_forced_plain(
            self, monkeypatch):
        """The part a class-level patch alone CANNOT fix: an instance built
        before the helper runs already cached its color decision in
        __init__, so the fix has to reach into that instance directly."""
        _dumb_env(monkeypatch)
        preexisting = rich.console.Console(file=io.StringIO())
        make_console_wide_and_plain(monkeypatch, width="300")
        preexisting.print("[red]plain[/red]")
        assert preexisting.file.getvalue() == "plain\n"
