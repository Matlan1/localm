# SPDX-License-Identifier: AGPL-3.0-or-later
"""CHK-SETUP-NOBACKEND regression: setup-llama's load-test must report a FAILED
provision when the native runtime LOADS but registers ZERO compute backends.

Before this fix, ``_native_loads_ok()`` ran ``load_lib()`` in a subprocess and
checked only ``returncode == 0``. ``load_lib()`` does not raise when no ggml
compute backend registers - it logs a warning and returns the handle - so a build
that loads yet cannot compute ("no backends are loaded") exited 0 and setup
reported SUCCESS. ``_provision_with_fallback`` then never offered the working
Vulkan/CPU fallback, and the user only discovered the broken runtime at the first
model load, with the real cause (no registered backend) already lost. That is the
AGENTS.md rule-5 anti-pattern ("it did not crash, so it is fine") at setup level.

The fix load-tests with ``compute_backends_available()`` and treats a load with no
backend (subprocess exit 88) as a failure with a clear reason, distinct from a
clean computing load (0) and a genuine load crash (non-zero + native traceback).
"""

from __future__ import annotations

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


def test_subprocess_spawn_error_is_reported(monkeypatch):
    def _boom(*a, **k):
        raise OSError("cannot spawn interpreter")
    monkeypatch.setattr(setup_llama.subprocess, "run", _boom)
    ok, detail = setup_llama._native_loads_ok()
    assert ok is False
    assert "cannot spawn interpreter" in detail
