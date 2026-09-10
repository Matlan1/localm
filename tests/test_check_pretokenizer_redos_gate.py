# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/check_pretokenizer_redos.py --gate: the exit code follows the verdicts.

The sweep itself needs MSVC and the network and is not run here; this pins
the one offline decision the CI pre-flight relies on, that a flagged pattern
turns into a non-zero exit under --gate and into exit 0 without it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_pretokenizer_redos.py"


@pytest.fixture(scope="module")
def redos():
    spec = importlib.util.spec_from_file_location("check_pretokenizer_redos", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gate_fails_on_a_flagged_probe_and_only_then(redos):
    flagged = [("pattern", "bait", "QUADRATIC-ish (x4 -> x16)")]
    assert redos._exit_code([], gate=True) == 0
    assert redos._exit_code(flagged, gate=True) == 1
    assert redos._exit_code(flagged, gate=False) == 0, "the default stays a report"
    assert redos._exit_code([], gate=False) == 0


def test_main_wires_the_gate_flag(redos):
    """The flag exists and main() returns _exit_code's verdict rather than a
    literal 0: read from the source, since main() itself needs MSVC to run."""
    src = _SCRIPT.read_text(encoding="utf-8")
    assert 'ap.add_argument("--gate"' in src
    assert "return _exit_code(concerning, args.gate)" in src
    assert "\n    return 0\n" not in src.split("def main()", 1)[1].split("def _exit_code", 1)[0]
