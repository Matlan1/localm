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
        # legacy_windows subtracts 1 from the reported width on Windows
        # without a modern terminal - platform-dependent, not a fixed number.
        assert c.size.width == 300 - c.legacy_windows

    def test_a_preexisting_singleton_is_also_widened(self, monkeypatch):
        """The shared CLI singletons are constructed before any fixture
        runs, so the fix must reach an ALREADY-BUILT instance too.

        Targets the REAL localm.cli._core.console, not an arbitrary fresh
        Console(). A fresh, unrelated instance is a claim the fix was never
        designed to make: os.get_terminal_size() on the real file
        descriptors succeeds on some platforms (observed: a real GitHub
        Actions runner, returning a genuine 80x24) in a way it never does
        under WSL's pipes, and Console.size's os.get_terminal_size() branch
        runs before COLUMNS is consulted - orthogonal to is_dumb_terminal,
        and irrelevant to the singletons the product code actually uses."""
        _dumb_env(monkeypatch)
        from localm.cli import _core
        assert _core.console.size.width == 80
        make_console_wide_and_plain(monkeypatch, width="300")
        assert _core.console.size.width == 300 - _core.console.legacy_windows


    def test_a_poisoned_shared_singletons_width_is_also_reset(self, monkeypatch):
        """Console.width has a real setter (console.width = N). If some
        UNRELATED test elsewhere in the suite leaves the shared CLI
        singleton's _width non-None, Console.size's FIRST check short-
        circuits before is_dumb_terminal/COLUMNS ever run - this is the
        exact failure mode that made keys.py's Table tests still fail after
        the is_dumb_terminal fix alone: the shared console renders correctly
        in isolation but not when some other test in the same xdist worker
        set an explicit width on it first. The helper can only defend the
        specific singleton objects it holds a reference to - not an
        arbitrary Console() built elsewhere, which is why this targets the
        real localm.cli._core.console rather than a fresh instance."""
        _dumb_env(monkeypatch)
        from localm.cli import _core
        # Poison through monkeypatch too (not a raw assignment): monkeypatch
        # snapshots whatever a value WAS at the moment of its OWN first
        # setattr call and restores exactly that at teardown. A raw
        # assignment here would poison the value monkeypatch snapshots
        # instead of the true original, leaking width=80 into every test
        # that runs after this one in the same process - which is exactly
        # what happened the first time this test was written.
        monkeypatch.setattr(_core.console, "_width", 80)
        make_console_wide_and_plain(monkeypatch, width="300")
        assert _core.console.size.width == 300 - _core.console.legacy_windows


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
