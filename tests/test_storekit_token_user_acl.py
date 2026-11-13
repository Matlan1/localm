# SPDX-License-Identifier: AGPL-3.0-or-later
"""A private file is locked to the user of the process token, whatever the
USERNAME and USER variables say."""

import re
import subprocess
import sys

import pytest

from localm import storekit


def test_current_user_sid_is_none_off_windows(monkeypatch):
    monkeypatch.setattr(storekit.os, "name", "posix")
    assert storekit.current_user_sid() is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ACLs")
def test_current_user_sid_is_a_sid_string():
    sid = storekit.current_user_sid()
    assert sid and re.fullmatch(r"S-1-\d+(-\d+)+", sid), sid


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ACLs")
def test_lock_down_follows_the_token_not_the_username_variable(tmp_path, monkeypatch):
    from tests._win_acl import token_user_name
    monkeypatch.setenv("USERNAME", "Guest")
    monkeypatch.setenv("USER", "Guest")
    f = tmp_path / "secret.json"
    f.write_text("{}", encoding="utf-8")
    assert storekit.restrict_file_perms(f) is True
    out = subprocess.run(["icacls", str(f)], capture_output=True, text=True,
                         check=True, creationflags=subprocess.CREATE_NO_WINDOW).stdout
    me = token_user_name()
    assert f"{me}:(F)".lower() in out.lower(), out
    assert "guest" not in out.lower(), out
    assert "(I)" not in out, out


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ACLs")
def test_lock_down_refuses_when_the_sid_cannot_be_read(tmp_path, monkeypatch):
    monkeypatch.setattr(storekit, "current_user_sid", lambda: None)
    f = tmp_path / "secret.json"
    f.write_text("{}", encoding="utf-8")
    before = subprocess.run(["icacls", str(f)], capture_output=True, text=True,
                            creationflags=subprocess.CREATE_NO_WINDOW).stdout
    assert storekit.restrict_file_perms(f) is False
    after = subprocess.run(["icacls", str(f)], capture_output=True, text=True,
                           creationflags=subprocess.CREATE_NO_WINDOW).stdout
    assert after == before

