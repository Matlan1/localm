# SPDX-License-Identifier: AGPL-3.0-or-later
"""appface.open_logs opens the server logfile in the OS viewer, best-effort."""

from __future__ import annotations

import sys

import pytest

from localm import appface


@pytest.fixture()
def openers(monkeypatch):
    calls = []
    monkeypatch.setattr("os.startfile",
                        lambda p, *a, **k: calls.append(("startfile", p)),
                        raising=False)
    monkeypatch.setattr(appface.webbrowser, "open",
                        lambda url, *a, **k: calls.append(("webbrowser", url)) or True)
    return calls


@pytest.mark.parametrize("empty", [None, ""])
def test_an_empty_logfile_opens_nothing(openers, empty):
    appface.open_logs(empty)
    assert openers == []


def test_windows_opens_the_file_with_the_shell(openers, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    appface.open_logs("server.log")
    assert openers == [("startfile", "server.log")]


def test_windows_accepts_a_path_object(openers, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    log = tmp_path / "server.log"
    appface.open_logs(log)
    assert openers == [("startfile", str(log))]


@pytest.mark.parametrize("plat", ["linux", "darwin"])
def test_elsewhere_opens_a_file_url(openers, monkeypatch, plat):
    monkeypatch.setattr(sys, "platform", plat)
    appface.open_logs("/var/log/server.log")
    assert openers == [("webbrowser", "file:///var/log/server.log")]


@pytest.mark.parametrize("plat", ["win32", "linux"])
def test_an_opener_that_raises_does_not_propagate(monkeypatch, plat):
    monkeypatch.setattr(sys, "platform", plat)
    attempts = []

    def boom(*a, **k):
        attempts.append(a)
        raise OSError("no viewer")

    monkeypatch.setattr("os.startfile", boom, raising=False)
    monkeypatch.setattr(appface.webbrowser, "open", boom)
    assert appface.open_logs("server.log") is None
    assert len(attempts) == 1
