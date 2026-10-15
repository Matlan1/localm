# SPDX-License-Identifier: AGPL-3.0-or-later
"""enable_debug() with a logs folder that cannot be created: debug mode still
turns on, on the console only, and nothing raises."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_PROBE = r"""
import json, logging
from localm import debuglog
first = debuglog.enable_debug()
second = debuglog.enable_debug()
debuglog.logger.debug("probe-debug-line")
print("RESULT " + json.dumps({
    "first": None if first is None else str(first),
    "second": None if second is None else str(second),
    "enabled": debuglog.debug_enabled(),
    "level": debuglog.logger.level,
}))
"""


def _run_probe(home: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("LOCALM_DEBUG", None)
    env["LOCALM_HOME"] = str(home)
    env["PYTHONPATH"] = str(REPO)
    return subprocess.run([sys.executable, "-c", _PROBE], env=env, cwd=str(home),
                          capture_output=True, text=True, timeout=120)


def test_a_logs_folder_that_cannot_be_created_falls_back_to_the_console(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "logs").write_text("a file where the logs folder should be", encoding="utf-8")

    proc = _run_probe(home)

    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(x for x in proc.stdout.splitlines() if x.startswith("RESULT "))
    result = json.loads(line[len("RESULT "):])
    assert result["first"] is None and result["second"] is None, result
    assert result["enabled"] is True
    assert result["level"] == 10
    out = proc.stdout + proc.stderr
    assert "could not be created" in out, out[-2000:]
    assert out.count("could not be created") == 1, "warned more than once"
    assert "probe-debug-line" in out, "debug output did not reach the console"


def test_a_writable_logs_folder_still_gets_the_file(tmp_path):
    home = tmp_path / "home"
    home.mkdir()

    proc = _run_probe(home)

    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(x for x in proc.stdout.splitlines() if x.startswith("RESULT "))
    result = json.loads(line[len("RESULT "):])
    assert result["first"] and result["first"] == result["second"], result
    log = Path(result["first"])
    assert log.parent == home / "logs"
    assert "probe-debug-line" in log.read_text(encoding="utf-8")
