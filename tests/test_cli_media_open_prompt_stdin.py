# SPDX-License-Identifier: AGPL-3.0-or-later
"""The media commands' "Open it now?" offer is made only where a person can
answer it: never with stdin on the null device, which on Windows reports
``isatty()`` True."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

_PROBE = (
    "import json, sys\n"
    "from pathlib import Path\n"
    "from localm.cli import media\n"
    "asked = []\n"
    "media.console.input = lambda prompt='': asked.append(prompt) or 'n'\n"
    "media._offer_open(Path('x.png'))\n"
    "print(json.dumps({'isatty': sys.stdin.isatty(), 'interactive': media._is_interactive(),\n"
    "                  'asked': asked}))\n"
)


def _probe(stdin) -> dict:
    env = dict(os.environ, PYTHONPATH=str(_REPO))
    proc = subprocess.run([sys.executable, "-c", _PROBE], stdin=stdin,
                          capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_no_open_offer_when_stdin_is_the_null_device():
    got = _probe(subprocess.DEVNULL)
    assert got["interactive"] is False
    assert got["asked"] == []


def test_no_open_offer_when_stdin_is_piped():
    got = _probe(subprocess.PIPE)
    assert got == {"isatty": False, "interactive": False, "asked": []}
