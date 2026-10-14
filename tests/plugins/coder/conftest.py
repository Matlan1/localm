# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fixtures shared by the coder plugin's tests."""

import sys

import pytest

_FINDERS = ("_psreadline_history_paths", "_unix_history_paths")


def _history_finder_namespaces():
    """The module dicts that hold the history path finders for a
    ``clear_shell_history_traces`` the coder code can call: the current
    ``localm.plugins.coder.privacy`` plus the globals of every such function
    reachable from an already-imported coder module, so a function imported
    before a module purge and re-import is covered too."""
    from localm.plugins.coder import privacy
    spaces = {id(privacy.__dict__): privacy.__dict__}
    for name, mod in list(sys.modules.items()):
        if not name.startswith(("localm.plugins.coder", "_localm_plugin_coder")):
            continue
        ns = getattr(getattr(mod, "clear_shell_history_traces", None), "__globals__", None)
        if ns is not None and all(f in ns for f in _FINDERS):
            spaces[id(ns)] = ns
    return list(spaces.values())


@pytest.fixture(autouse=True)
def _no_real_shell_history(monkeypatch):
    """The coder CLI scrubs shell history files on a privacy-mode exit
    (``clear_shell_history_traces``). Its path finders answer no paths in every
    test here, in every module namespace the scrub can run from, so a test that
    runs the CLI never reads or rewrites the real PowerShell, bash or zsh
    history. A test that exercises the scrub patches the finders itself or
    calls the functions it imported directly.
    See test_no_test_reaches_the_real_shell_history."""
    for ns in _history_finder_namespaces():
        for finder in _FINDERS:
            monkeypatch.setitem(ns, finder, lambda: [])
