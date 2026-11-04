# SPDX-License-Identifier: AGPL-3.0-or-later
"""The installers read uv's and pip's output as UTF-8, whatever the locale code
page: a double-byte code page or a non-ASCII install path never raises while the
output is read."""

from __future__ import annotations

import subprocess
import sys

from localm import setup_llama
from localm.plugins import deps

_BYTES = ("Using environment at: C:\\Users\\\u7528\u6237\\.venv\n").encode("utf-8") + b"bad \xff\xfe\n"
_CODE = f"import sys; sys.stdout.buffer.write({_BYTES!r}); sys.stdout.flush(); sys.exit(1)"
_EXPECTED = "C:\\Users\\\u7528\u6237\\.venv"
_NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def test_plugin_installer_reads_output_as_utf8(monkeypatch):
    real_popen = subprocess.Popen
    monkeypatch.setattr(deps.uv_tool, "find_uv", lambda root: None)
    monkeypatch.setattr(
        deps.subprocess, "Popen",
        lambda cmd, **kw: real_popen([sys.executable, "-c", _CODE], **kw, **_NO_WINDOW))
    seen = []
    ok, out = deps._run_pip(["somepkg"], on_progress=seen.append)
    assert ok is False
    assert _EXPECTED in out and "\ufffd" in out
    assert any(_EXPECTED in line for line in seen)


def test_runtime_wheel_installer_reads_output_as_utf8(monkeypatch, capsys):
    real_run = subprocess.run
    results = []

    def run(cmd, **kw):
        result = real_run([sys.executable, "-c", _CODE], **kw, **_NO_WINDOW)
        results.append(result)
        return result

    monkeypatch.setitem(sys.modules, "localm_llama_runtime", None)
    monkeypatch.setattr(setup_llama.runtime_dir.uv_tool, "own_uv", lambda root: None)
    monkeypatch.setattr(setup_llama.runtime_dir.subprocess, "run", run)
    assert setup_llama._install_runtime_wheel(setup_llama.runtime_dir.Path("pkg")) is False
    assert results and _EXPECTED in results[0].stdout and "\ufffd" in results[0].stdout
