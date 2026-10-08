# SPDX-License-Identifier: AGPL-3.0-or-later
"""The launcher runs under pythonw.exe, where sys.stdout and sys.stderr are None.

Importing the model manager (which the launcher's models-folder scan does) must
not touch a stream that is not there. Each case runs in a fresh interpreter,
because the streams are configured at import time.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_DRIVER = """
import sys, traceback
out = sys.argv[1]
sys.stdout = None
sys.stderr = None
try:
    {body}
    result = "ok"
except Exception:
    result = traceback.format_exc()
with open(out, "w", encoding="utf-8") as fh:
    fh.write(result)
"""


def _run_without_streams(body: str, tmp_path: Path) -> str:
    home = tmp_path / "home"
    home.mkdir()
    out = tmp_path / "result.txt"
    env = {
        "PYTHONPATH": str(REPO_ROOT),
        "LOCALM_HOME": str(home),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    import os
    env = {**os.environ, **env}
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER.format(body=body), str(out)],
        cwd=str(tmp_path), env=env, capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    return out.read_text(encoding="utf-8")


@pytest.mark.skipif(sys.platform != "win32",
                    reason="the UTF-8 stream reconfigure only runs on Windows")
def test_model_manager_imports_without_console_streams(tmp_path):
    result = _run_without_streams("import localm.model_manager", tmp_path)
    assert result == "ok", result


@pytest.mark.skipif(sys.platform != "win32",
                    reason="the UTF-8 stream reconfigure only runs on Windows")
def test_models_folder_scan_runs_without_console_streams(tmp_path):
    body = ("from localm.model_manager import sync_models_dir; "
            "sync_models_dir(backfill_mmproj=False)")
    result = _run_without_streams(body, tmp_path)
    assert result == "ok", result


@pytest.mark.skipif(sys.platform != "win32",
                    reason="the UTF-8 stream reconfigure only runs on Windows")
def test_cli_core_imports_without_console_streams(tmp_path):
    result = _run_without_streams("import localm.cli._core", tmp_path)
    assert result == "ok", result
