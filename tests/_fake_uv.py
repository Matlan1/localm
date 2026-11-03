# SPDX-License-Identifier: AGPL-3.0-or-later
"""A stand-in uv.exe for tests that run the Windows setup scripts.

``build(folder)`` compiles a small console program to ``folder/uv.exe`` with
Windows PowerShell's Add-Type. It appends ``uv <args>`` to the file named by
STUB_LOG (with `` TEMP=<TEMP> TMP=<TMP>`` appended) when that is set, answers
``--version`` with ``uv <version>`` (the version setup.bat pins, or
STUB_UV_VERSION when that is set), creates ``.venv\\Scripts`` for ``venv``
(and copies itself there as python.exe when STUB_VENV_PYTHON=1), answers
``-m localm.setup_spec extras`` (as that python) with ``voice``, and exits 0.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_SOURCE = r"""
using System;
using System.IO;
public class Stub {
    public static int Main(string[] a) {
        string log = Environment.GetEnvironmentVariable("STUB_LOG");
        if (log != null) File.AppendAllText(log, "uv " + string.Join(" ", a)
            + " TEMP=" + Environment.GetEnvironmentVariable("TEMP")
            + " TMP=" + Environment.GetEnvironmentVariable("TMP") + "\r\n");
        if (a.Length > 0 && a[0] == "--version") {
            string v = Environment.GetEnvironmentVariable("STUB_UV_VERSION");
            Console.WriteLine("uv " + (string.IsNullOrEmpty(v) ? "@@PINNED@@" : v) + " (stub)");
        }
        if (a.Length > 0 && a[0] == "venv") {
            Directory.CreateDirectory(Path.Combine(".venv", "Scripts"));
            if (Environment.GetEnvironmentVariable("STUB_VENV_PYTHON") == "1")
                File.Copy(System.Reflection.Assembly.GetExecutingAssembly().Location,
                          Path.Combine(".venv", "Scripts", "python.exe"), true);
        }
        if (a.Length > 2 && a[0] == "-m" && a[1] == "localm.setup_spec" && a[2] == "extras")
            Console.WriteLine("voice");
        return 0;
    }
}
"""


def pinned_version() -> str:
    """The uv version setup.bat pins (its UV_INSTALLER_VERSION)."""
    text = (ROOT / "setup.bat").read_text(encoding="utf-8")
    m = re.search(r'^set "UV_INSTALLER_VERSION=([^"]+)"', text, re.M)
    assert m, "setup.bat no longer pins UV_INSTALLER_VERSION"
    return m.group(1)


def _powershell() -> str:
    return os.path.join(os.environ["SystemRoot"], "System32", "WindowsPowerShell", "v1.0",
                        "powershell.exe")


def build(folder: Path) -> Path:
    """Compiles the stand-in to ``folder/uv.exe`` and returns its path."""
    folder.mkdir(parents=True, exist_ok=True)
    src = folder / "stub.cs"
    src.write_text(_SOURCE.replace("@@PINNED@@", pinned_version()), encoding="utf-8")
    out = folder / "uv.exe"
    cmd = ("Add-Type -TypeDefinition (Get-Content -Raw -LiteralPath $env:STUB_SRC) "
           "-OutputAssembly $env:STUB_OUT -OutputType ConsoleApplication")
    r = subprocess.run([_powershell(), "-NoProfile", "-Command", cmd],
                       env={**os.environ, "STUB_SRC": str(src), "STUB_OUT": str(out)},
                       capture_output=True, text=True, timeout=120, creationflags=_NO_WINDOW)
    assert out.is_file(), r.stdout + r.stderr
    src.unlink()
    return out


def place(built: Path, folder: Path) -> Path:
    """Copies a built stand-in to ``folder/uv.exe`` and returns that path."""
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / "uv.exe"
    shutil.copyfile(built, dest)
    return dest
