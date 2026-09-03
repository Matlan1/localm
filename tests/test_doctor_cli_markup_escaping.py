# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm doctor` prints real probe output - a subprocess's stderr, a native
ABI mismatch detail, a GPU/driver device name, a plugin's declared dependency
string, a filesystem path - straight into ``rich.console.Console.print()``
f-strings. Rich parses any ``[...]`` in a printed string as markup, not just
inside a command's own literal ``[style]`` tags, so any of that free text can
silently corrupt the report it appears in. Rich renders these as:

    Console().print('report[draft].txt')       -> prints "report.txt"
    Console().print('notes[bold red].md')       -> prints "notes.md"

The bracketed span is either dropped outright or consumed as a (bogus) style
directive, in both cases silently - never a crash, just a wrong diagnostic
shown to a user debugging a broken GPU/runtime/plugin install.

Every case here monkeypatches the REAL probe function or module doctor.py
calls (``diagnostics.check_llama_lib``, ``subprocess.run``, a fake ``torch``
module, ``localm.setup_llama.installed_build``/``pinned_tag``,
``doctor_mod._provisioned_backend_name``/``_probe_gpu_devices``,
``importlib.metadata.version``, ``PluginManager.all_missing_deps``,
``managed_comfy.is_managed_comfy_installed``/``managed_comfy_paths``) to
return a realistic bracketed value, then drives the real `doctor` command
through `cli_runner` end to end - never mocking ``console.print`` itself.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.metadata
import sys
import types

import localm.cli as cli
from localm import diagnostics

# localm.cli re-exports `doctor` as the click Command itself, shadowing the
# submodule name - go through importlib for the module (same convention as
# test_doctor_gpu_verdict.py / test_doctor_cli_phase3.py).
doctor_mod = importlib.import_module("localm.cli.doctor")

# One name Rich DROPS outright, one it consumes as a (bogus) style tag - the
# two distinct failure shapes test_rag_cli_markup_escaping.py's docstring
# describes, reused here so both shapes are exercised across this file.
BRACKET_DROP = "firmware[legacy].bin"
BRACKET_STYLE = "driver[bold red]crash"


def _fake_torch_no_gpu():
    """A stand-in torch with no CUDA device - keeps `doctor` deterministic and
    avoids importing the heavy real torch, matching the sibling doctor tests'
    own _fake_torch_no_gpu (test_doctor_managed_comfy_hint.py)."""
    mod = types.ModuleType("torch")
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


def _stub_unrelated_probes(monkeypatch):
    """Neutralize every doctor probe NOT under test in a given test, so the
    real `doctor` CLI command can be driven end to end without touching real
    hardware/subprocess state. Mirrors test_doctor_cli_phase3.py's /
    test_doctor_managed_comfy_hint.py's own established stubbing pattern."""
    import subprocess

    def _raise(*a, **k):
        raise FileNotFoundError("not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    monkeypatch.setattr(cli, "find_binary_dir", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_no_gpu())
    monkeypatch.setattr(doctor_mod, "_check_hf_backend_usable", lambda *a, **k: None)
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)
    # Pin the width of the console the CLI prints through. Console.size returns
    # _width/_height before it consults TERM or COLUMNS, and returns 80x25
    # outright on a dumb terminal, where COLUMNS alone does not survive.
    monkeypatch.setenv("COLUMNS", "400")
    from localm.cli import _core
    monkeypatch.setattr(_core.console, "_width", 400)
    monkeypatch.setattr(_core.console, "_height", 25)


# --------------------------------------------------------------------------- #
#  _render() - shared by check_llama_lib / check_native_abi / worker_spawn /  #
#  venv_creation / check_hf_backend, so this is the single highest-value site #
# --------------------------------------------------------------------------- #

def _fake_llama_lib_result(status, text, note="", hints=()):
    return diagnostics.CheckResult(
        key="llama_lib", label="llama.dll / llama.so", status=status,
        summary="fake probe result for markup-escaping test",
        findings=(diagnostics.Finding(status, text, note=note, hints=hints),))


class TestRenderFindingsMarkupEscaping:
    def test_finding_text_bracket_drop_survives_verbatim(self, cli_runner, monkeypatch):
        """diagnostics.check_llama_lib's own real findings interpolate a raw
        binary-dir listing (see diagnostics.py's `contents: {files}` site) -
        realistic bracketed content, reproduced here via the same function."""
        _stub_unrelated_probes(monkeypatch)
        bad_text = f"binary dir found (C:\\bin) but no llama .dll/.so - contents: ['{BRACKET_DROP}']"
        monkeypatch.setattr(
            diagnostics, "check_llama_lib",
            lambda find_binary_dir: _fake_llama_lib_result(diagnostics.WARN, bad_text))

        out = cli_runner.invoke(cli.doctor, []).output
        assert bad_text in out, (
            f"a probe Finding's text must survive verbatim, not be silently "
            f"mangled by Rich markup parsing: {out!r}")

    def test_finding_note_and_hints_bracket_style_survive_verbatim(
            self, cli_runner, monkeypatch):
        """note/hints are rendered on separate lines from text (_render's own
        idiom) - both must be escaped independently."""
        _stub_unrelated_probes(monkeypatch)
        note = f"struct layout {BRACKET_STYLE}"
        hint = f"see {BRACKET_STYLE} for detail"
        monkeypatch.setattr(
            diagnostics, "check_llama_lib",
            lambda find_binary_dir: _fake_llama_lib_result(
                diagnostics.WARN, "native ABI not verified", note=note, hints=(hint,)))

        out = cli_runner.invoke(cli.doctor, []).output
        assert note in out, f"a Finding's note must survive verbatim: {out!r}"
        assert hint in out, f"a Finding's hint must survive verbatim: {out!r}"


# --------------------------------------------------------------------------- #
#  _check_gpu_driver() - nvidia-smi/rocm-smi's own stdout/stderr             #
# --------------------------------------------------------------------------- #

class TestGpuDriverMarkupEscaping:
    def test_smi_success_device_name_bracket_style_survives_verbatim(
            self, monkeypatch, capsys):
        """A successful nvidia-smi's own reported device name/memory line."""
        import subprocess
        device_line = f"NVIDIA GeForce {BRACKET_STYLE}, 24564 MiB"

        def _run(cmd, *a, **k):
            if cmd and cmd[0] == "nvidia-smi":
                return subprocess.CompletedProcess(cmd, 0, device_line, "")
            raise FileNotFoundError(cmd[0] if cmd else "?")
        monkeypatch.setattr(subprocess, "run", _run)

        assert doctor_mod._check_gpu_driver() is True
        out = capsys.readouterr().out
        assert device_line in out, (
            f"nvidia-smi's own reported device name must survive verbatim: {out!r}")

    def test_smi_failure_text_bracket_drop_survives_verbatim(self, monkeypatch, capsys):
        """A failing nvidia-smi's own stdout/stderr reason text."""
        import subprocess
        error_text = f"Failed to initialize NVML: {BRACKET_DROP}"

        def _run(cmd, *a, **k):
            if cmd and cmd[0] == "nvidia-smi":
                return subprocess.CompletedProcess(cmd, 9, error_text, "")
            raise FileNotFoundError(cmd[0] if cmd else "?")
        monkeypatch.setattr(subprocess, "run", _run)

        assert doctor_mod._check_gpu_driver() is False
        out = capsys.readouterr().out
        # Rich's non-tty console defaults to 80 cols and this line is longer
        # than that, so it wraps mid-string on a plain capsys capture with no
        # COLUMNS override - collapse the wrap before asserting, the same
        # idiom test_doctor_cli_phase3.py uses for the same reason. The
        # escaping under test is unaffected either way: what matters here is
        # that the bracketed text is not DROPPED or turned into markup, not
        # which column it happens to wrap at.
        assert error_text in " ".join(out.split()), (
            f"a failing smi tool's own error text must survive verbatim, not "
            f"be mangled by Rich: {out!r}")


# --------------------------------------------------------------------------- #
#  _check_vram_torch() - the CUDA/ROCm driver's own device name (props.name), #
#  and a dynamically-named exception class (type(e).__name__)                 #
# --------------------------------------------------------------------------- #

class TestVramTorchMarkupEscaping:
    def test_torch_device_name_bracket_style_survives_verbatim(
            self, cli_runner, monkeypatch):
        from localm import discover, gpu_usage

        device_name = f"AMD Radeon {BRACKET_STYLE}"
        mod = types.ModuleType("torch")
        mod.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)

        class _Props:
            name = device_name

        class _Cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 1

            @staticmethod
            def get_device_properties(i):
                return _Props()

            @staticmethod
            def mem_get_info(i):
                return (8 * 1024**3, 16 * 1024**3)

        mod.cuda = _Cuda()

        _stub_unrelated_probes(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", mod)
        monkeypatch.setattr(
            discover, "list_gpus",
            lambda *, deadline=None, return_status=False: (
                ([], discover.GPU_PROBE_OK) if return_status else []))
        monkeypatch.setattr(gpu_usage, "raw_reading_is_process_scoped", lambda: False)
        monkeypatch.setattr(doctor_mod, "_check_native_abi", lambda: None)

        out = cli_runner.invoke(cli.doctor, []).output
        assert device_name in out, (
            f"the torch-reported GPU device name must survive verbatim: {out!r}")

    def test_torch_probe_exception_class_name_bracket_drop_survives_verbatim(
            self, cli_runner, monkeypatch):
        """A dynamically-constructed exception class can carry an arbitrary
        __name__ - forced here the same way TestLockMessageEscaping in
        test_rag_cli_markup_escaping.py forces a real exception rather than
        mocking console.print."""
        bad_cls = type(f"Torch{BRACKET_DROP}Error", (RuntimeError,), {})
        mod = types.ModuleType("torch")
        mod.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)

        class _Cuda:
            @staticmethod
            def is_available():
                raise bad_cls("torch cuda blew up")
        mod.cuda = _Cuda()

        _stub_unrelated_probes(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", mod)

        out = cli_runner.invoke(cli.doctor, []).output
        assert bad_cls.__name__ in out, (
            f"a dynamically-named exception class must survive verbatim in "
            f"the probe-failed line: {out!r}")


# --------------------------------------------------------------------------- #
#  _check_runtime_build() - the unvalidated `build` marker token, and `pin`   #
#  (escaped as defense-in-depth even though pinned_tag() validates on read)   #
# --------------------------------------------------------------------------- #

class TestRuntimeBuildMarkupEscaping:
    def test_build_bracket_drop_survives_verbatim(self, cli_runner, monkeypatch, tmp_path):
        _stub_unrelated_probes(monkeypatch)
        bindir = tmp_path / "bin"
        bindir.mkdir()
        monkeypatch.setattr(cli, "find_binary_dir", lambda: bindir)

        build_tag = f"b1307{BRACKET_DROP}"
        from localm import setup_llama
        monkeypatch.setattr(setup_llama, "installed_build", lambda: build_tag)
        monkeypatch.setattr(setup_llama, "pinned_tag", lambda: None)

        out = cli_runner.invoke(cli.doctor, []).output
        assert build_tag in out, (
            f"the installed build tag must survive verbatim: {out!r}")

    def test_pin_bracket_style_survives_verbatim(self, cli_runner, monkeypatch, tmp_path):
        """pinned_tag() validates its stored value via is_safe_tag() on every
        real read, so this exercises the DEFENSE-IN-DEPTH escape rather than a
        reachable-today bug - the mock bypasses that validation the same way
        mocking any other already-guarded internal would."""
        _stub_unrelated_probes(monkeypatch)
        bindir = tmp_path / "bin"
        bindir.mkdir()
        monkeypatch.setattr(cli, "find_binary_dir", lambda: bindir)

        pin_tag = f"b1200{BRACKET_STYLE}"
        from localm import setup_llama
        monkeypatch.setattr(setup_llama, "installed_build", lambda: None)
        monkeypatch.setattr(setup_llama, "pinned_tag", lambda: pin_tag)

        out = cli_runner.invoke(cli.doctor, []).output
        assert pin_tag in out, f"the pinned tag must survive verbatim: {out!r}"


# --------------------------------------------------------------------------- #
#  _check_gpu_verdict() - the provisioned backend name (`tag`, `backend`),    #
#  the native probe's device names, and the JSON-round-tripped probe error   #
# --------------------------------------------------------------------------- #

class TestGpuVerdictMarkupEscaping:
    def test_probe_device_name_bracket_drop_survives_verbatim(
            self, cli_runner, monkeypatch, tmp_path):
        bindir = tmp_path / "bin"
        bindir.mkdir()
        _stub_unrelated_probes(monkeypatch)
        monkeypatch.setattr(cli, "find_binary_dir", lambda: bindir)
        monkeypatch.setattr(doctor_mod, "_check_llama_lib", lambda fbd: True)
        monkeypatch.setattr(doctor_mod, "_check_native_abi", lambda: None)

        device_name = f"gpu0{BRACKET_DROP}"
        monkeypatch.setattr(doctor_mod, "_provisioned_backend_name", lambda fbd: "vulkan")
        monkeypatch.setattr(
            doctor_mod, "_probe_gpu_devices",
            lambda: {"loaded": True, "devices": [[device_name, 1]], "error": ""})
        monkeypatch.setattr(doctor_mod, "_loader_gpu_type", lambda: 1)

        out = cli_runner.invoke(cli.doctor, []).output
        assert device_name in out, (
            f"a native GPU compute-device name must survive verbatim: {out!r}")

    def test_backend_tag_and_probe_error_bracket_style_survive_verbatim(
            self, cli_runner, monkeypatch, tmp_path):
        bindir = tmp_path / "bin"
        bindir.mkdir()
        _stub_unrelated_probes(monkeypatch)
        monkeypatch.setattr(cli, "find_binary_dir", lambda: bindir)
        monkeypatch.setattr(doctor_mod, "_check_llama_lib", lambda fbd: True)
        monkeypatch.setattr(doctor_mod, "_check_native_abi", lambda: None)

        backend_name = f"weird{BRACKET_STYLE}"
        probe_error_text = f"native load failed: {BRACKET_STYLE}"
        monkeypatch.setattr(doctor_mod, "_provisioned_backend_name", lambda fbd: backend_name)
        monkeypatch.setattr(
            doctor_mod, "_probe_gpu_devices",
            lambda: {"loaded": False, "devices": [], "error": probe_error_text})

        out = cli_runner.invoke(cli.doctor, []).output
        assert backend_name in out, (
            f"the provisioned-backend tag must survive verbatim: {out!r}")
        assert probe_error_text in out, (
            f"the runtime probe's own captured error must survive verbatim: {out!r}")


# --------------------------------------------------------------------------- #
#  _check_packages() - an installed package's own distribution version string #
# --------------------------------------------------------------------------- #

class TestPackagesVersionMarkupEscaping:
    def test_package_version_bracket_style_survives_verbatim(self, cli_runner, monkeypatch):
        _stub_unrelated_probes(monkeypatch)
        real_version = importlib.metadata.version
        fake_version = f"13.0{BRACKET_STYLE}"

        def _version(dist_name):
            if dist_name == "rich":
                return fake_version
            return real_version(dist_name)
        monkeypatch.setattr(importlib.metadata, "version", _version)

        out = cli_runner.invoke(cli.doctor, []).output
        assert fake_version in out, (
            f"an installed package's own reported version must survive "
            f"verbatim: {out!r}")


# --------------------------------------------------------------------------- #
#  _check_plugin_deps() - a plugin manifest's own declared name/requirement   #
# --------------------------------------------------------------------------- #

class TestPluginDepsMarkupEscaping:
    def test_plugin_name_and_requirement_survive_verbatim(self, cli_runner, monkeypatch):
        _stub_unrelated_probes(monkeypatch)
        from localm.plugins.engine import PluginManager

        plugin_name = f"myplugin{BRACKET_DROP}"
        requirement = f"some-pkg{BRACKET_STYLE}>=1.0"
        monkeypatch.setattr(
            PluginManager, "all_missing_deps",
            lambda self, *, enabled_only=True: {plugin_name: [requirement]})

        out = cli_runner.invoke(cli.doctor, []).output
        assert plugin_name in out, (
            f"a plugin's own declared name must survive verbatim: {out!r}")
        assert requirement in out, (
            f"a plugin's own declared pip requirement must survive verbatim: {out!r}")


# --------------------------------------------------------------------------- #
#  _check_managed_comfy() - the managed-ComfyUI root, a real filesystem path  #
# --------------------------------------------------------------------------- #

class TestManagedComfyMarkupEscaping:
    def test_installed_root_path_bracket_style_survives_verbatim(
            self, cli_runner, monkeypatch, tmp_path):
        _stub_unrelated_probes(monkeypatch)
        from localm.media import managed_comfy as mc_mod

        fake_root = tmp_path / f"comfy{BRACKET_STYLE}"
        fake_paths = mc_mod.ManagedComfyPaths(
            root=fake_root, models_dir=tmp_path, main_py=tmp_path / "main.py",
            venv_python=tmp_path / "venv_python", extra_model_paths=tmp_path / "x.yaml")
        monkeypatch.setattr(mc_mod, "is_managed_comfy_installed", lambda: True)
        monkeypatch.setattr(mc_mod, "managed_comfy_paths", lambda: fake_paths)

        out = cli_runner.invoke(cli.doctor, []).output
        assert str(fake_root) in out, (
            f"the managed-ComfyUI root path must survive verbatim: {out!r}")
