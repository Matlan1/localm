# SPDX-License-Identifier: AGPL-3.0-or-later
"""A test that patches sys.platform and reaches an `import psutil` that fails and
is swallowed must not leave psutil unimportable for the tests after it."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("psutil")

ROOT = Path(__file__).resolve().parents[1]
_OTHER = "linux" if sys.platform == "win32" else "win32"

_SCRIPT = f"""
import sys
{{first}}
real = sys.platform
sys.platform = {_OTHER!r}
try:
    import psutil
except Exception:
    pass
sys.platform = real
import psutil
print("PSUTIL_OK", psutil.virtual_memory().total > 0)
"""


def _run(first: str) -> subprocess.CompletedProcess:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run([sys.executable, "-c", _SCRIPT.format(first=first)], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=120, creationflags=flags)


def test_psutil_survives_a_failed_import_under_a_patched_platform():
    r = _run("import tests")
    assert r.returncode == 0, r.stderr
    assert "PSUTIL_OK True" in r.stdout, r.stdout + r.stderr
