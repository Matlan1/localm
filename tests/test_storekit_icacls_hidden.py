"""restrict_file_perms runs icacls without a console window of its own (Windows)."""

import subprocess
import sys

import pytest

from localm import storekit

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="icacls is Windows only")


def test_restrict_file_perms_runs_icacls_without_a_console(tmp_path, monkeypatch):
    target = tmp_path / "secret.txt"
    target.write_text("x", encoding="utf-8")
    calls = []
    real_run = subprocess.run

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    assert storekit.restrict_file_perms(target) is True
    icacls_calls = [kw for a, kw in calls if a and "icacls" in str(a[0][0]).lower()]
    assert len(icacls_calls) == 1
    assert icacls_calls[0].get("creationflags", 0) & subprocess.CREATE_NO_WINDOW
    acl = real_run(["icacls", str(target)], capture_output=True, text=True,
                   creationflags=subprocess.CREATE_NO_WINDOW).stdout
    assert "(I)" not in acl
