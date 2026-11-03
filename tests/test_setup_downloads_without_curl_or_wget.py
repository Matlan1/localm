# SPDX-License-Identifier: AGPL-3.0-or-later
"""Setup gets uv on a system with neither curl nor wget.

A fresh Debian installed without the desktop and "standard system utilities"
tasks has neither curl (priority optional) nor wget (priority standard), but it
always has apt (priority required) and so /usr/lib/apt/apt-helper. setup.sh and
setup-gui.sh download the uv installer with curl, else wget, else apt-helper,
and give Astral's installer (which itself only knows curl and wget) a wget that
downloads with apt-helper. installer/gui.py does the same for the installer it
runs.

The script functions run here with their system folders pointing at an empty
stub folder and APT_HELPER at a logging stand-in.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests._posix_sysdirs import bash_path, sys_prog_function

ROOT = Path(__file__).resolve().parents[1]


def _bash():
    found = shutil.which("bash")
    return None if not found or "system32" in found.lower() else found


needs_bash = pytest.mark.skipif(_bash() is None, reason="no bash on PATH")

_FAKE_HELPER = ('#!/bin/sh\n'
                'echo "helper $*" >> "$HELPER_LOG"\n'
                '[ "$1" = download-file ] || exit 9\n'
                'printf "payload from %s" "$2" > "$3"\n')


def _function(text: str, name: str) -> str:
    start = text.index(name + "() {")
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def _setup(tmp_path: Path, script: str, *, with_curl: bool = False):
    text = (ROOT / script).read_text(encoding="utf-8")
    empty = tmp_path / "sys"
    empty.mkdir()
    if with_curl:
        (empty / "curl").write_text('#!/bin/sh\necho "curl $*" >> "$HELPER_LOG"\n',
                                    encoding="utf-8", newline="\n")
        (empty / "curl").chmod(0o755)
    helper = tmp_path / "apt-helper"
    helper.write_text(_FAKE_HELPER, encoding="utf-8", newline="\n")
    helper.chmod(0o755)
    head = (f'LOCALM_SYSDIRS="{bash_path(empty)}"\n'
            f'APT_HELPER="{bash_path(helper)}"\n')
    assert 'APT_HELPER="/usr/lib/apt/apt-helper"' in text
    return head + sys_prog_function(text) + _function(text, "download") \
        + _function(text, "installer_path")


def _run(tmp_path: Path, body: str):
    log = tmp_path / "helper.log"
    env = dict(os.environ, HELPER_LOG=bash_path(log))
    r = subprocess.run([_bash(), "-c", body], cwd=str(tmp_path), env=env,
                       capture_output=True, text=True, timeout=60)
    return r, (log.read_text().splitlines() if log.exists() else [])


@needs_bash
@pytest.mark.parametrize("script", ["setup.sh", "setup-gui.sh"])
def test_download_falls_back_to_apt_helper(tmp_path, script):
    body = _setup(tmp_path, script) + 'download https://x.invalid/i.sh got && cat got\n'
    r, log = _run(tmp_path, body)
    assert r.returncode == 0, r.stdout + r.stderr
    assert log == ["helper download-file https://x.invalid/i.sh got"], log
    assert "payload from https://x.invalid/i.sh" in r.stdout


@needs_bash
@pytest.mark.parametrize("script", ["setup.sh", "setup-gui.sh"])
def test_the_installer_gets_a_wget_that_downloads_with_apt_helper(tmp_path, script):
    body = (_setup(tmp_path, script)
            + 'p="$(installer_path "$PWD/work")" || exit 3\n'
            + 'PATH="$p:$PATH" wget --header "Authorization: Bearer t" '
              'https://x.invalid/uv.tar.gz -O out.tgz && cat out.tgz\n')
    r, log = _run(tmp_path, body)
    assert r.returncode == 0, r.stdout + r.stderr
    assert log == ["helper download-file https://x.invalid/uv.tar.gz out.tgz"], log
    assert "payload from https://x.invalid/uv.tar.gz" in r.stdout


@needs_bash
@pytest.mark.parametrize("script", ["setup.sh", "setup-gui.sh"])
def test_with_curl_there_is_no_shim_and_no_apt_helper(tmp_path, script):
    body = (_setup(tmp_path, script, with_curl=True)
            + 'p="$(installer_path "$PWD/work")"; [ "$p" = "$PATH" ] || exit 4\n'
            + '[ ! -e work/bin/wget ] || exit 5\n'
            + 'download https://x.invalid/i.sh got\n')
    r, log = _run(tmp_path, body)
    assert r.returncode == 0, r.stdout + r.stderr
    assert log == ["curl -fsSL -o got https://x.invalid/i.sh"], log


def _gui(monkeypatch):
    import importlib.util
    import sys
    path = ROOT / "installer" / "gui.py"
    spec = importlib.util.spec_from_file_location("localm_installer_gui_dl", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["localm_installer_gui_dl"] = mod
    spec.loader.exec_module(mod)
    return mod


@needs_bash
def test_the_window_gives_the_installer_the_same_wget(tmp_path, monkeypatch):
    gui = _gui(monkeypatch)
    helper = tmp_path / "apt-helper"
    helper.write_text(_FAKE_HELPER, encoding="utf-8", newline="\n")
    helper.chmod(0o755)
    monkeypatch.setattr(gui, "IS_WINDOWS", False)
    monkeypatch.setattr(gui, "system_program", lambda name: None)
    monkeypatch.setattr(gui, "APT_HELPER", str(helper))
    monkeypatch.setattr(gui.os, "access", lambda p, mode: True)
    shim = gui.installer_download_shim(tmp_path / "scratch")
    assert shim == str(tmp_path / "scratch" / "bin")
    text = (Path(shim) / "wget").read_text(encoding="utf-8")
    text = text.replace(str(helper), bash_path(helper))
    (Path(shim) / "wget").write_text(text, encoding="utf-8", newline="\n")
    r, log = _run(tmp_path, f'sh "{bash_path(Path(shim) / "wget")}" '
                            'https://x.invalid/uv.tar.gz -O out.tgz && cat out.tgz')
    assert r.returncode == 0, r.stdout + r.stderr
    assert log == ["helper download-file https://x.invalid/uv.tar.gz out.tgz"], log


def test_the_window_writes_no_shim_when_curl_or_wget_is_there(tmp_path, monkeypatch):
    gui = _gui(monkeypatch)
    monkeypatch.setattr(gui, "IS_WINDOWS", False)
    monkeypatch.setattr(gui, "system_program",
                        lambda name: "/usr/bin/wget" if name == "wget" else None)
    assert gui.installer_download_shim(tmp_path / "scratch") is None
    assert not (tmp_path / "scratch").exists()
