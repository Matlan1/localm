# SPDX-License-Identifier: AGPL-3.0-or-later
"""``stdin_is_interactive`` (in-app and in the standalone reporter) is False
for stdin read from the null device, which on Windows reports ``isatty()``
True although nobody can answer a prompt there."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

_PROBE = (
    "import importlib.util, json, sys\n"
    "from localm import bugreport\n"
    "spec = importlib.util.spec_from_file_location('ri', sys.argv[1])\n"
    "ri = importlib.util.module_from_spec(spec); spec.loader.exec_module(ri)\n"
    "print(json.dumps({'isatty': sys.stdin.isatty(),\n"
    "                  'in_app': bugreport.stdin_is_interactive(),\n"
    "                  'standalone': ri.stdin_is_interactive()}))\n"
)


def _probe(stdin) -> dict:
    env = dict(os.environ, PYTHONPATH=str(_REPO))
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, str(_REPO / "scripts" / "report_issue.py")],
        stdin=stdin, capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_null_device_stdin_is_not_interactive():
    got = _probe(subprocess.DEVNULL)
    assert got["in_app"] is False
    assert got["standalone"] is False


@pytest.mark.skipif(os.name != "nt", reason="the null device reports isatty only on Windows")
def test_the_windows_null_device_does_report_isatty():
    assert _probe(subprocess.DEVNULL)["isatty"] is True


def test_piped_stdin_is_not_interactive():
    got = _probe(subprocess.PIPE)
    assert got == {"isatty": False, "in_app": False, "standalone": False}
