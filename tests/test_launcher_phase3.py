# SPDX-License-Identifier: AGPL-3.0-or-later
"""BUG-15 (M2 Phase 3): the launcher's Launch action must be cross-platform.

The launcher spawned child processes with
``creationflags=subprocess.CREATE_NEW_CONSOLE``. That flag (and the
``CREATE_NEW_CONSOLE`` attribute itself) is Windows-only. On Linux/macOS the
attribute does not exist, so referencing it raises ``AttributeError`` - which
the launcher's bare ``except Exception`` swallowed into a silent
"Launch failed". The setup.sh ``.desktop`` entry was therefore a dead button on
POSIX.

These tests pin the contract for the extracted spawn helper:
- on POSIX (``sys.platform != "win32"``) the child is spawned WITHOUT
  ``creationflags`` (and never touches ``subprocess.CREATE_NEW_CONSOLE``), so no
  ``AttributeError`` can occur; and
- on Windows (``sys.platform == "win32"``) ``creationflags`` is still passed
  with ``CREATE_NEW_CONSOLE`` so the visible-console behavior is unchanged.

The launcher lives at the repo root as ``launcher.pyw`` (a ``.pyw`` so a
double-click runs it without a console). That extension is not importable by
plain ``import``, so we load it from its file path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parent.parent
LAUNCHER_PATH = REPO_DIR / "launcher.pyw"


def _load_launcher():
    """Import launcher.pyw as a module without instantiating any Tk window.

    The module only ``import``s tkinter at load time (it does not build a root
    window until ``Launcher()`` is constructed), so this is safe headless.
    """
    spec = importlib.util.spec_from_file_location("localm_launcher", str(LAUNCHER_PATH))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def launcher_mod():
    return _load_launcher()


class _PopenSpy:
    """Records the args/kwargs of the most recent call; mimics a live Popen."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self  # stand-in for a Popen object; the launcher ignores it

    @property
    def last_kwargs(self):
        assert self.calls, "Popen was never called"
        return self.calls[-1][1]


def test_helper_exists(launcher_mod):
    """The spawn must be factored into a module-level, testable helper.

    (The launch logic was buried in a Tk button handler; BUG-15's fix extracts
    it so the cross-platform behavior can be asserted without a display.)
    """
    assert hasattr(launcher_mod, "_spawn_detached"), (
        "expected a module-level _spawn_detached() helper to test the "
        "cross-platform spawn"
    )


def test_posix_spawn_omits_creationflags(launcher_mod, monkeypatch):
    """On POSIX the child is spawned WITHOUT creationflags and WITHOUT touching
    subprocess.CREATE_NEW_CONSOLE (the attribute that does not exist there)."""
    spy = _PopenSpy()
    monkeypatch.setattr(launcher_mod.subprocess, "Popen", spy)
    # Simulate a POSIX interpreter where the Windows-only flag is absent, so any
    # stray reference to it would raise AttributeError exactly as on Linux/macOS.
    monkeypatch.setattr(launcher_mod.sys, "platform", "linux")
    monkeypatch.delattr(launcher_mod.subprocess, "CREATE_NEW_CONSOLE", raising=False)

    # Must not raise (the old code raised AttributeError here).
    launcher_mod._spawn_detached(["echo", "hi"], cwd="/tmp")

    kwargs = spy.last_kwargs
    assert "creationflags" not in kwargs, (
        "POSIX spawn must not pass the Windows-only creationflags kwarg; got "
        f"{kwargs!r}"
    )


def test_windows_spawn_passes_create_new_console(launcher_mod, monkeypatch):
    """On Windows the spawn still passes creationflags=CREATE_NEW_CONSOLE so the
    child opens in its own console window (behavior must not regress)."""
    spy = _PopenSpy()
    monkeypatch.setattr(launcher_mod.subprocess, "Popen", spy)
    monkeypatch.setattr(launcher_mod.sys, "platform", "win32")
    # Pin a known sentinel so the assertion does not depend on the host's value.
    sentinel = 0x10  # real CREATE_NEW_CONSOLE value, but any int works here
    monkeypatch.setattr(
        launcher_mod.subprocess, "CREATE_NEW_CONSOLE", sentinel, raising=False
    )

    launcher_mod._spawn_detached(["python", "-m", "localm", "gui"], cwd="C:\\repo")

    kwargs = spy.last_kwargs
    assert kwargs.get("creationflags") == sentinel, (
        "Windows spawn must keep creationflags=CREATE_NEW_CONSOLE; got "
        f"{kwargs!r}"
    )


def test_spawn_forwards_cwd_and_env(launcher_mod, monkeypatch):
    """The helper forwards cwd and (optional) env to Popen on both platforms."""
    spy = _PopenSpy()
    monkeypatch.setattr(launcher_mod.subprocess, "Popen", spy)
    monkeypatch.setattr(launcher_mod.sys, "platform", "linux")
    monkeypatch.delattr(launcher_mod.subprocess, "CREATE_NEW_CONSOLE", raising=False)

    env = {"LOCALM_API_KEY": "secret"}
    launcher_mod._spawn_detached(["x"], cwd="/work", env=env)

    args, kwargs = spy.calls[-1]
    assert args[0] == ["x"]
    assert kwargs.get("cwd") == "/work"
    assert kwargs.get("env") == env


def test_launch_handler_uses_helper_on_posix(launcher_mod, monkeypatch):
    """End-to-end-ish: the buried Launch handler routes through the helper, so a
    POSIX user clicking Launch spawns a real child instead of getting the
    swallowed 'Launch failed' from an AttributeError.

    We avoid building a Tk root by calling the launch path on a lightweight
    stand-in object that provides only what _launch / _import_from_url touch.
    """
    spy = _PopenSpy()
    monkeypatch.setattr(launcher_mod.subprocess, "Popen", spy)
    monkeypatch.setattr(launcher_mod.sys, "platform", "linux")
    monkeypatch.delattr(launcher_mod.subprocess, "CREATE_NEW_CONSOLE", raising=False)

    Launcher = launcher_mod.Launcher

    class _Var:
        def __init__(self, value):
            self._value = value

        def get(self):
            return self._value

    class _FakeLauncher:
        def __init__(self):
            self.messages = []
            self.keep_open = _Var(True)
            self.port = _Var("")
            self.privacy_global = _Var("privacy")

        # --- the handful of collaborators the spawn path calls ---
        def status_msg(self, text, error=False):
            self.messages.append((text, error))

        def after(self, *_a, **_k):
            pass

        def destroy(self):
            pass

    fake = _FakeLauncher()
    # askstring is a module-level import inside launcher; patch it to return a spec.
    monkeypatch.setattr(
        launcher_mod.simpledialog, "askstring", lambda *a, **k: "org/model"
    )

    # Call the real (unbound) handler against the fake instance.
    Launcher._import_from_url(fake)

    assert spy.calls, "the Launch path must spawn a child on POSIX"
    assert "creationflags" not in spy.last_kwargs
    # No error status was set (no swallowed AttributeError).
    assert not any(err for _t, err in fake.messages), fake.messages
