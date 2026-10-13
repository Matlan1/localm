# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fixtures shared by the coder plugin's tests."""

import pytest


@pytest.fixture(autouse=True)
def _no_real_shell_history(monkeypatch):
    """The coder CLI scrubs shell history files on a privacy-mode exit
    (``clear_shell_history_traces``). Its path finders answer no paths in every
    test here, so a test that runs the CLI never reads or rewrites the real
    PowerShell, bash or zsh history. A test that exercises the scrub patches
    the finders itself or calls the functions it imported directly.
    See test_no_test_reaches_the_real_shell_history."""
    from localm.plugins.coder import privacy
    monkeypatch.setattr(privacy, "_psreadline_history_paths", lambda: [])
    monkeypatch.setattr(privacy, "_unix_history_paths", lambda: [])
