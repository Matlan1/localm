# SPDX-License-Identifier: AGPL-3.0-or-later
"""S5 slice: the localm-managed ComfyUI DISCOVERY HINT in `localm doctor`.

The managed-ComfyUI feature is opt-in and off by default, surfaced for discovery
as a NON-INSTALLING hint in `localm doctor`. This proves the hint is:
  - shown only when NO managed ComfyUI is installed;
  - replaced by an installed-status line (with the managed root path) when one
    IS installed;
  - purely informational: it never adds a failure or changes doctor's verdict.

Drives the real click doctor command through the ``cli_runner`` fixture (which
pins a throwaway LOCALM_HOME) and stubs the llama-lib / smi / torch probes so the
only variable under test is the managed-ComfyUI state - it never touches real GPU
state.
"""

import importlib
import importlib.machinery
import sys
import types

import localm.cli as cli
from localm.media import managed_comfy as mc

# localm.cli re-exports `doctor` as the click Command itself, shadowing the
# submodule name - go through importlib for the module.
doctor_mod = importlib.import_module("localm.cli.doctor")

_SETUP_HINT = "localm comfy setup"
_CROSS = "✗"  # the red-cross glyph doctor uses for a FAILED check


def _fake_torch_no_gpu():
    """A stand-in torch with no CUDA device, so the GPU probe is deterministic
    and the heavy real torch is never imported."""
    mod = types.ModuleType("torch")
    # transformers (probed by doctor's package check) calls
    # importlib.util.find_spec("torch"), which raises if torch.__spec__ is None.
    # A bare ModuleType has no spec, so the stub is given a real one.
    mod.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)

    class _Cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

    mod.cuda = _Cuda()
    return mod


def _stub_probes(monkeypatch):
    """Neutralize doctor's hardware probes so the only thing that varies between
    runs is the managed-ComfyUI install state."""
    import subprocess

    def _raise(*a, **k):
        raise FileNotFoundError("not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    monkeypatch.setattr(cli, "find_binary_dir", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_no_gpu())
    # The fake torch stub lacks the internals transformers needs, so a real
    # transformers in this venv would fail its lazy AutoTokenizer/AutoProcessor/
    # AutoModelForCausalLM resolution against it.
    monkeypatch.setattr(doctor_mod, "_check_hf_backend_usable", lambda *a, **k: None)
    # rich soft-wraps at the console width, which defaults to 80 cols under the
    # non-tty CliRunner capture. Render wide so substring assertions see the
    # unbroken line.
    monkeypatch.setenv("COLUMNS", "400")


def _install_managed():
    """Create the S1 on-disk layout that makes is_managed_comfy_installed() true,
    under whatever LOCALM_HOME the cli_runner fixture pinned. Uses the module's
    own path accessors so it is platform-agnostic (venv interpreter path differs
    on Windows vs POSIX). Includes the completion marker: main.py + venv alone
    means "still installing"."""
    from localm.media.managed_comfy_provision import MARKER_FILENAME
    paths = mc.managed_comfy_paths()
    paths.main_py.parent.mkdir(parents=True, exist_ok=True)
    paths.main_py.write_text("# stand-in ComfyUI main.py\n", encoding="utf-8")
    paths.venv_python.parent.mkdir(parents=True, exist_ok=True)
    paths.venv_python.write_text("", encoding="utf-8")
    (paths.root / MARKER_FILENAME).write_text("{}", encoding="utf-8")
    assert mc.is_managed_comfy_installed()
    return paths


def _fail_lines(output):
    """Doctor's failure lines (those carrying the red-cross glyph)."""
    return [ln for ln in output.splitlines() if _CROSS in ln]


def test_doctor_hint_when_not_installed(cli_runner, monkeypatch):
    """No managed ComfyUI -> the discovery hint appears; no installed line."""
    _stub_probes(monkeypatch)
    assert mc.is_managed_comfy_installed() is False   # throwaway home: nothing set up

    result = cli_runner.invoke(cli.doctor, [])
    assert result.exit_code == 0, result.output
    out = result.output
    assert _SETUP_HINT in out
    assert "manage its own ComfyUI" in out
    # Nothing is installed, so the installed-status line must NOT show.
    assert "installed at" not in out


def test_doctor_status_when_installed_replaces_hint(cli_runner, monkeypatch):
    """A managed ComfyUI on disk -> the installed-status line (with the managed
    root path) replaces the setup hint."""
    _stub_probes(monkeypatch)
    paths = _install_managed()

    result = cli_runner.invoke(cli.doctor, [])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "installed" in out.lower()
    assert str(paths.root) in out                     # names the managed root
    # Do not nudge to set up what is already set up.
    assert _SETUP_HINT not in out


def test_doctor_hint_is_informational_never_a_failure(cli_runner, monkeypatch):
    """The managed-ComfyUI line must never read as a doctor failure, and its
    install state must not change doctor's set of failures either way."""
    _stub_probes(monkeypatch)

    # Not installed: capture doctor's failures and the ComfyUI line(s).
    res_hint = cli_runner.invoke(cli.doctor, [])
    fails_hint = _fail_lines(res_hint.output)
    comfy_lines = [ln for ln in res_hint.output.splitlines() if "ComfyUI" in ln]
    assert comfy_lines, "expected a managed-ComfyUI line in doctor output"
    assert all(_CROSS not in ln for ln in comfy_lines)   # never a failure glyph

    # Now install a managed instance and re-run.
    _install_managed()
    res_status = cli_runner.invoke(cli.doctor, [])
    fails_status = _fail_lines(res_status.output)

    # Managed-ComfyUI state changes NOTHING about doctor's failures.
    assert fails_hint == fails_status
    assert res_hint.exit_code == 0
    assert res_status.exit_code == 0
