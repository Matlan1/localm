# SPDX-License-Identifier: AGPL-3.0-or-later
"""A throwaway install folder for driving the shipped launch scripts.

``make_windows_clone`` copies a shipped script into a new folder next to a
``.venv`` whose ``Scripts\\python.exe`` is a copy of the running venv's
interpreter launcher (a real program, with its ``pyvenv.cfg``), and a fake
``localm`` package that records how it was started. The fake package is put
first on ``PYTHONPATH``; the copied venv has no site-packages, so the real
localm is not importable there.

The fake package reads its registry from ``LOCALM_PROBE_REG`` (a JSON object of
name to entry) and, when started as ``python -m localm``, writes
``{"cwd", "argv"}`` as JSON to ``LOCALM_PROBE_OUT`` and exits with
``LOCALM_PROBE_RC`` (default 0).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CREATE_NO_WINDOW = 0x08000000

_FAKE_INIT = ""
_FAKE_MAIN = (
    "import json, os, sys\n"
    "out = os.environ.get('LOCALM_PROBE_OUT')\n"
    "if out:\n"
    "    with open(out, 'w', encoding='utf-8') as f:\n"
    "        json.dump({'cwd': os.getcwd(), 'argv': sys.argv[1:]}, f)\n"
    "sys.exit(int(os.environ.get('LOCALM_PROBE_RC', '0')))\n"
)
_FAKE_CONFIG = (
    "import json, os\n"
    "def load_registry():\n"
    "    return json.loads(os.environ.get('LOCALM_PROBE_REG', '{}'))\n"
)
_FAKE_MODEL_MANAGER = (
    "def is_llm(entry):\n"
    "    return isinstance(entry, dict) and entry.get('model_type', 'llm') == 'llm'\n"
)


def venv_launcher() -> Path:
    """The running venv's ``Scripts\\python.exe``; raises when not in a venv."""
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("the tests are not running from a venv")
    return Path(sys.prefix) / "Scripts" / "python.exe"


def make_windows_clone(tmp_path: Path, script: str = "localm.bat", *,
                       venv: bool = True, text: str | None = None) -> Path:
    """A folder holding *script* (the shipped file, or *text* when given) and,
    with *venv*, a runnable ``.venv``. Returns the folder."""
    clone = tmp_path / "install dir"
    clone.mkdir()
    body = text if text is not None else (ROOT / script).read_text(encoding="utf-8")
    (clone / script).write_bytes(body.replace("\r\n", "\n").replace("\n", "\r\n").encode())
    if venv:
        scripts = clone / ".venv" / "Scripts"
        scripts.mkdir(parents=True)
        src = venv_launcher()
        shutil.copy2(src, scripts / "python.exe")
        shutil.copy2(src.parent.parent / "pyvenv.cfg", clone / ".venv" / "pyvenv.cfg")
    fake = tmp_path / "fake"
    (fake / "localm" / "model_manager").mkdir(parents=True)
    (fake / "localm" / "__init__.py").write_text(_FAKE_INIT, encoding="utf-8")
    (fake / "localm" / "__main__.py").write_text(_FAKE_MAIN, encoding="utf-8")
    (fake / "localm" / "config.py").write_text(_FAKE_CONFIG, encoding="utf-8")
    (fake / "localm" / "model_manager" / "__init__.py").write_text(
        _FAKE_MODEL_MANAGER, encoding="utf-8")
    return clone


def run_bat(clone: Path, args: list, *, cwd: Path, registry: dict | None = None,
            extra_env: dict | None = None, script: str = "localm.bat"):
    """Run ``cmd /c <clone>\\<script> *args`` from *cwd* with stdin closed (so a
    ``pause`` returns at once). Returns ``(completed, probe)`` where *probe* is
    what the fake ``python -m localm`` recorded, or None when it did not run."""
    out = cwd / "probe.json"
    if out.exists():
        out.unlink()
    env = {k: v for k, v in os.environ.items()
           if k not in ("LOCALM_MODEL", "PYTHONHOME", "VIRTUAL_ENV")}
    env["PYTHONPATH"] = str(clone.parent / "fake")
    env["PYTHONNOUSERSITE"] = "1"
    env["LOCALM_PROBE_OUT"] = str(out)
    env["LOCALM_PROBE_REG"] = json.dumps(registry or {})
    env.update(extra_env or {})
    done = subprocess.run(["cmd", "/c", str(clone / script), *args],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL,
                          cwd=str(cwd), env=env, timeout=60,
                          creationflags=CREATE_NO_WINDOW)
    probe = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    return done, probe
