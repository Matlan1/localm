# SPDX-License-Identifier: AGPL-3.0-or-later
"""A process that inherits LOCALM_DEBUG naming a log file (a server re-executed by
Restart server, or any child) writes its own debug records to that same file."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_FIRST = r"""
from localm import debuglog
print("LOG " + str(debuglog.enable_debug()))
debuglog.logger.debug("first-process-line")
"""

_ENABLE = r"""
from localm import debuglog
returned = debuglog.enable_debug()
again = debuglog.enable_debug()
debuglog.logger.debug("restarted-process-line")
print("LOG " + str(returned) + " " + str(again))
"""

_HONOR = r"""
from localm import debuglog
debuglog.honor_env_debug()
debuglog.logger.debug("env-process-line")
"""

_CHILD = r"""
from localm import debuglog
debuglog.attach_child_logging()
debuglog.attach_child_logging()
debuglog.logger.debug("child-process-line")
"""


def _run(code: str, home: Path, debug_env: str | None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("LOCALM_DEBUG", None)
    if debug_env is not None:
        env["LOCALM_DEBUG"] = debug_env
    env["LOCALM_HOME"] = str(home)
    env["PYTHONPATH"] = str(REPO)
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(home),
                          capture_output=True, text=True, timeout=120,
                          creationflags=flags)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc


def _first_process(home: Path) -> Path:
    proc = _run(_FIRST, home, None)
    line = next(x for x in proc.stdout.splitlines() if x.startswith("LOG "))
    return Path(line[4:])


def test_a_restarted_process_appends_to_the_inherited_log_file(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    log = _first_process(home)

    proc = _run(_ENABLE, home, str(log))

    text = log.read_text(encoding="utf-8")
    assert "first-process-line" in text
    assert text.count("restarted-process-line") == 1, text
    assert f"LOG {log} {log}" in proc.stdout
    assert list((home / "logs").iterdir()) == [log], "a second log file was created"


def test_honor_env_debug_opens_an_inherited_log_path(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    log = _first_process(home)

    _run(_HONOR, home, str(log))

    text = log.read_text(encoding="utf-8")
    assert text.count("env-process-line") == 1, text


def test_a_child_process_still_shares_the_log_once_per_record(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    log = _first_process(home)

    _run(_CHILD, home, str(log))

    text = log.read_text(encoding="utf-8")
    assert text.count("child-process-line") == 1, text
