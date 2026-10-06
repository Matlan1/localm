# SPDX-License-Identifier: AGPL-3.0-or-later
"""A Tcl library read failure must retry and then fail, never read as no display."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests import _tk_root
from tests._tk_root import ATTEMPTS, build_tk_root

tkinter = pytest.importorskip("tkinter")

_TESTS = Path(__file__).resolve().parent

INIT_READ = (
    "Can't find a usable init.tcl in the following directories: \n"
    "    {D:\\py\\tcl\\tcl8.6}\n\n"
    'D:/py/tcl/tcl8.6/init.tcl: couldn\'t read file "D:/py/tcl/tcl8.6/init.tcl": '
    "No error\n\nThis probably means that Tcl wasn't installed properly.\n")
TK_READ = (
    "Can't find a usable tk.tcl in the following directories: \n"
    "    D:/py/tcl/tk8.6\n\n"
    'D:/py/tcl/tk8.6/tk.tcl: couldn\'t read file "D:/py/tcl/tk8.6/scale.tcl": '
    "no such file or directory\n\nThis probably means that tk wasn't installed "
    "properly.\n")
FIND_LIBRARY = 'invalid command name "tcl_findLibrary"'
LIBRARY_READ_ERRORS = [INIT_READ, TK_READ, FIND_LIBRARY]

NO_DISPLAY_ERRORS = [
    "no display name and no $DISPLAY environment variable",
    'couldn\'t connect to display ":0"',
]


class _Factory:
    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "root"


def _build(factory):
    delays = []
    outcome = None
    try:
        outcome = ("built", build_tk_root(factory, sleep=delays.append))
    except pytest.fail.Exception as e:
        outcome = ("failed", str(e))
    except pytest.skip.Exception as e:
        outcome = ("skipped", str(e))
    return outcome, delays


@pytest.mark.parametrize("message", LIBRARY_READ_ERRORS)
def test_a_library_read_failure_is_retried_and_the_root_is_returned(message):
    factory = _Factory(tkinter.TclError(message))
    outcome, delays = _build(factory)
    assert factory.calls == 2
    assert outcome == ("built", "root")
    assert len(delays) == 1


def test_a_failure_that_clears_after_three_attempts_still_builds():
    factory = _Factory(*[tkinter.TclError(INIT_READ)] * 3)
    outcome, _ = _build(factory)
    assert factory.calls == 4
    assert outcome == ("built", "root")


@pytest.mark.parametrize("message", LIBRARY_READ_ERRORS)
def test_a_library_read_failure_that_never_clears_fails_loudly(message):
    factory = _Factory(*[tkinter.TclError(message)] * 50)
    outcome, delays = _build(factory)
    assert factory.calls == ATTEMPTS
    assert len(delays) == ATTEMPTS - 1
    assert outcome[0] == "failed"
    assert message.splitlines()[0] in outcome[1]


@pytest.mark.parametrize("message", NO_DISPLAY_ERRORS)
def test_a_missing_display_skips_without_retrying(message):
    factory = _Factory(*[tkinter.TclError(message)] * 50)
    outcome, delays = _build(factory)
    assert factory.calls == 1
    assert delays == []
    assert outcome[0] == "skipped"
    assert "no display" in outcome[1]


def test_a_library_read_failure_is_not_taken_for_a_missing_display():
    for message in LIBRARY_READ_ERRORS:
        assert _tk_root.is_no_display(tkinter.TclError(message)) is False
    for message in NO_DISPLAY_ERRORS:
        assert _tk_root.is_no_display(tkinter.TclError(message)) is True


def test_an_error_that_is_not_a_tcl_error_is_not_retried():
    factory = _Factory(RuntimeError("boom"))
    raised = None
    try:
        build_tk_root(factory, sleep=lambda _: None)
    except RuntimeError as e:
        raised = e
    assert factory.calls == 1
    assert raised is not None


def _skips_on_tcl_error(source: str) -> list[int]:
    """Lines of ``except ...TclError`` handlers that call ``pytest.skip``."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ExceptHandler) or node.type is None:
            continue
        names = {n.attr if isinstance(n, ast.Attribute) else getattr(n, "id", "")
                 for n in ast.walk(node.type)}
        if "TclError" not in names:
            continue
        for call in (n for b in node.body for n in ast.walk(b)
                     if isinstance(n, ast.Call)):
            func = call.func
            if (isinstance(func, ast.Attribute) and func.attr == "skip"
                    and getattr(func.value, "id", "") == "pytest"):
                hits.append(node.lineno)
    return hits


_OLD_PATTERN = (
    "try:\n"
    "    root = tk.Tk()\n"
    "except tk.TclError as e:\n"
    "    pytest.skip(f'no display: {e}')\n")
_NEW_PATTERN = "root = build_tk_root(tk.Tk)\n"


def test_the_sentinel_flags_the_old_pattern_and_passes_the_new_one():
    assert _skips_on_tcl_error(_OLD_PATTERN) == [3]
    assert _skips_on_tcl_error(_NEW_PATTERN) == []


def test_no_test_skips_on_a_caught_tcl_error():
    offenders = []
    for path in sorted(_TESTS.glob("*.py")):
        if path.name in (Path(__file__).name, "_tk_root.py"):
            continue
        source = path.read_text(encoding="utf-8")
        if "TclError" not in source:
            continue
        for line in _skips_on_tcl_error(source):
            offenders.append(f"{path.name}:{line}")
    assert offenders == [], (
        "a Tcl library read failure would read as a missing display here; "
        "build the window with tests._tk_root.build_tk_root: "
        f"{offenders}")
