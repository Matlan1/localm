# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup-llama's load-test must report a FAILED provision when the native runtime
LOADS but registers ZERO compute backends.

``load_lib()`` does not raise when no ggml compute backend registers - it logs a
warning and returns the handle - so a build that loads yet cannot compute exits 0
under a returncode-only check, and setup would report SUCCESS.

``_native_loads_ok()`` load-tests with ``compute_backends_available()`` and treats
a load with no backend (subprocess exit 88) as a failure with a clear reason,
distinct from a clean computing load (0) and a genuine load crash (non-zero plus
a native traceback).
"""

from __future__ import annotations

import pytest

from localm import setup_llama


class _FakeProc:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode: int, stderr: str = "", stdout: str = ""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


def test_no_backend_load_is_a_failure(monkeypatch):
    # The runtime loaded but registered no compute backend (exit 88).
    monkeypatch.setattr(setup_llama.subprocess, "run",
                        lambda *a, **k: _FakeProc(88))
    ok, detail = setup_llama._native_loads_ok()
    assert ok is False
    assert "no compute backends" in detail


def test_clean_computing_load_passes(monkeypatch):
    # Loaded AND a backend registered (exit 0).
    monkeypatch.setattr(setup_llama.subprocess, "run",
                        lambda *a, **k: _FakeProc(0))
    ok, detail = setup_llama._native_loads_ok()
    assert ok is True
    assert detail == ""


def test_load_crash_still_reported(monkeypatch):
    # A genuine load failure (mismatched build) stays a failure, surfacing the
    # last line of the native error - not collapsed into the no-backend message.
    err = ("Traceback (most recent call last):\n"
           "RuntimeError: Failed to load llama.dll: not a valid Win32 application")
    monkeypatch.setattr(setup_llama.subprocess, "run",
                        lambda *a, **k: _FakeProc(1, stderr=err))
    ok, detail = setup_llama._native_loads_ok()
    assert ok is False
    assert "Failed to load llama.dll" in detail
    assert "no compute backends" not in detail


def test_multiline_load_error_reports_cause_not_hint(monkeypatch):
    # load_lib() raises a MULTI-LINE RuntimeError whose FIRST line is the real
    # dlopen cause and whose trailing lines are re-provision hints.
    # _native_loads_ok() surfaces the cause, not splitlines()[-1].
    err = (
        "Traceback (most recent call last):\n"
        '  File "run.py", line 1, in <module>\n'
        "    _loader.load_lib()\n"
        "RuntimeError: Failed to load libllama.so from /venv/lib/libllama.so: "
        "libgomp.so.1: cannot open shared object file: No such file or directory\n"
        "This usually means the provisioned build does not match this machine.\n"
        "  localm setup-llama --backend vulkan --force   (any GPU, no vendor toolkit)\n"
        "  localm setup-llama --backend cpu --force       (no GPU)\n"
        "  localm setup-llama --backend amd-rocm --force  (AMD RX 6000)"
    )
    monkeypatch.setattr(setup_llama.subprocess, "run",
                        lambda *a, **k: _FakeProc(1, stderr=err))
    ok, detail = setup_llama._native_loads_ok()
    assert ok is False
    assert "libgomp.so.1: cannot open shared object file" in detail  # real cause
    assert "--backend amd-rocm --force" not in detail                # not the hint


def test_informative_error_line_falls_back_to_last_line():
    # Non-traceback output (no recognisable exception header): return the last
    # non-empty line, and a safe default for empty output.
    assert setup_llama._informative_error_line(
        "noise line\nactual tail") == "actual tail"
    assert setup_llama._informative_error_line("") == "library failed to load"


def _quick(monkeypatch):
    from localm.setup_llama import load_probe
    monkeypatch.setattr(load_probe, "_PAUSE_S", 0)
    monkeypatch.setattr(setup_llama.subprocess, "run", lambda *a, **k: _FakeProc(0))
    return load_probe


def test_a_test_file_that_cannot_be_deleted_does_not_fail_the_build(monkeypatch):
    probe = _quick(monkeypatch)

    def held(path, ignore_errors=False, **_k):
        if not ignore_errors:
            raise PermissionError(13, "in use")
    monkeypatch.setattr(probe.shutil, "rmtree", held)
    assert setup_llama._native_loads_ok() == (True, "")


def test_without_a_temporary_folder_the_data_folder_is_used(monkeypatch, tmp_path):
    from localm import config
    probe = _quick(monkeypatch)
    real = probe.tempfile.mkdtemp
    used = []

    def mkdtemp(prefix=None, dir=None):
        if dir is None:
            raise FileNotFoundError("no usable temporary directory")
        used.append(dir)
        return real(prefix=prefix, dir=dir)
    monkeypatch.setattr(probe.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(config, "home_dir", lambda: tmp_path)
    assert setup_llama._native_loads_ok() == (True, "")
    assert used == [tmp_path] and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fails, loads", [(2, True), (3, False)])
def test_a_test_that_cannot_be_run_is_tried_again_and_is_no_failed_build(monkeypatch, fails, loads):
    probe = _quick(monkeypatch)
    left = [fails]

    def run(*_a, **_k):
        if left[0]:
            left[0] -= 1
            raise OSError("no interpreter")
        return _FakeProc(0)
    monkeypatch.setattr(setup_llama.subprocess, "run", run)
    for test in (setup_llama._native_loads_ok, setup_llama._native_gpu_loads_ok):
        left[0] = fails
        if loads:
            assert test() == (True, "")
        else:
            with pytest.raises(probe.LoadTestNotRun, match="no interpreter"):
                test()
