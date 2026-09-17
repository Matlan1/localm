#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The one place a script appends to a file the GitHub Actions runner names in
an environment variable: ``GITHUB_STEP_SUMMARY`` (the job's markdown summary)
and ``GITHUB_OUTPUT`` (a step's ``name=value`` outputs).

Both files are created by the Actions runtime for the current job and named
only through the job's own environment, which nothing but the runner that
launches the job controls. Every script under scripts/ that publishes to
either goes through :func:`append`; tests/test_ci_runner_files.py fails on any
other ``os.environ`` read of these two names under scripts/, so the runner-owned
file is opened at exactly one site.

Stdlib only. Import from a sibling script by putting scripts/ on sys.path:

    SCRIPTS = Path(__file__).resolve().parent
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import ci_runner_files
"""

from __future__ import annotations

import os

STEP_SUMMARY = "GITHUB_STEP_SUMMARY"
OUTPUT = "GITHUB_OUTPUT"

RUNNER_FILE_VARS = (STEP_SUMMARY, OUTPUT)


def append(var: str, text: str) -> bool:
    """Append ``text`` to the runner file named by environment variable
    ``var`` (one of :data:`RUNNER_FILE_VARS`). Returns True when the variable
    is set and the write happened, False when it is unset or empty (a local
    run outside Actions). Raises ValueError for any other ``var`` and OSError
    when the runner-named file cannot be written."""
    if var not in RUNNER_FILE_VARS:
        raise ValueError(f"not a runner file variable: {var!r}")
    path = os.environ.get(var)
    if not path:
        return False
    # codeql[py/path-injection] the path is the Actions runner's own per-job file, set only by the job's launcher
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
    return True
