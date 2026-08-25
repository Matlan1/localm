# SPDX-License-Identifier: AGPL-3.0-or-later
"""Out-of-range / non-integer ports must be rejected CLEANLY at the CLI, not
crash deep in the socket layer (pick_port -> connect_ex: "port must be
0-65535"). Regression for 4 captured test-station bug reports.
"""

import pytest
from click.testing import CliRunner

from localm.cli import serve as serve_cmd
from localm.plugins.gui.cli import main as gui_cmd


@pytest.mark.parametrize("bad", ["70000", "0", "-1", "abc"])
def test_serve_rejects_bad_port(bad):
    r = CliRunner().invoke(serve_cmd, ["somemodel", "-p", bad])
    assert r.exit_code != 0
    assert "Invalid value" in r.output      # click's clean parse error
    assert "Traceback" not in r.output      # never a raw crash


@pytest.mark.parametrize("bad", ["70000", "0", "abc"])
def test_gui_rejects_bad_port(bad):
    # The GUI path reaches pick_port even with no model; IntRange catches the
    # bad value at parse time.
    r = CliRunner().invoke(gui_cmd, ["-p", bad])
    assert r.exit_code != 0
    assert "Invalid value" in r.output
    assert "Traceback" not in r.output


def test_valid_port_passes_validation():
    # A valid port is accepted by the IntRange (the command then proceeds and
    # fails later for an unrelated reason - not a port parse error).
    r = CliRunner().invoke(serve_cmd, ["no-such-model", "-p", "8642"])
    assert "Invalid value" not in r.output
