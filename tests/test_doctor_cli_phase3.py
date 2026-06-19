# SPDX-License-Identifier: AGPL-3.0-or-later
"""M2 Phase 3 tests for `localm doctor` and the REPL `_handle_command`.

Covers:
  BUG-1  doctor must check llama lib *integrity* (size>0), not mere existence.
  BUG-2  doctor must not print "CPU mode only" when torch reports a GPU.
  FAC-4  doctor must show a real version for 'rich' (importlib.metadata).
  GAP-CLI-3  /temp and /tokens must clamp absurd/negative/zero values.
  SEC-9  /save must be confined to the cwd (reject traversal/absolute escapes).

The doctor tests drive the real click command through the ``cli_runner``
fixture and monkeypatch the smi/torch/rich probes. The REPL tests unit-call
``_handle_command`` directly.
"""

import sys
import types

import pytest

import localm.cli as cli


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _fake_torch(gpu_names):
    """Build a stand-in ``torch`` module exposing the bits doctor touches."""
    mod = types.ModuleType("torch")

    class _Props:
        def __init__(self, name):
            self.name = name

    class _Cuda:
        @staticmethod
        def is_available():
            return bool(gpu_names)

        @staticmethod
        def device_count():
            return len(gpu_names)

        @staticmethod
        def get_device_properties(i):
            return _Props(gpu_names[i])

        @staticmethod
        def mem_get_info(i):
            # 8 GB free / 16 GB total
            return (8 * 1024**3, 16 * 1024**3)

    mod.cuda = _Cuda()
    return mod


def _no_smi(monkeypatch):
    """Make every smi subprocess probe fail (no nvidia-smi / rocm-smi)."""
    import subprocess

    def _raise(*a, **k):
        raise FileNotFoundError("not found")

    monkeypatch.setattr(subprocess, "run", _raise)


def _install_torch(monkeypatch, gpu_names):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(gpu_names))


# --------------------------------------------------------------------------- #
#  BUG-1 - llama lib integrity (size>0)                                        #
# --------------------------------------------------------------------------- #

def test_doctor_flags_zero_byte_llama_lib(cli_runner, tmp_path, monkeypatch):
    """A 0-byte llama.dll must NOT be reported as a healthy 'found' lib."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "llama.dll").write_bytes(b"")  # zeroed / truncated

    monkeypatch.setattr(cli, "find_binary_dir", lambda: bindir)
    _no_smi(monkeypatch)
    _install_torch(monkeypatch, [])  # stub torch; this test is about the lib

    result = cli_runner.invoke(cli.doctor, [])
    out = result.output
    # Must not declare the empty lib healthy.
    assert "llama.dll found" not in out
    # Must call out the empty / corrupt file explicitly.
    assert "empty" in out.lower() or "0 byte" in out.lower() or "corrupt" in out.lower()


def test_doctor_warns_on_suspiciously_tiny_llama_lib(cli_runner, tmp_path, monkeypatch):
    """A 1-byte llama.dll is present and non-empty but implausibly small."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "llama.dll").write_bytes(b"\x00")  # 1 byte

    monkeypatch.setattr(cli, "find_binary_dir", lambda: bindir)
    _no_smi(monkeypatch)
    _install_torch(monkeypatch, [])  # stub torch; this test is about the lib

    result = cli_runner.invoke(cli.doctor, [])
    out = result.output.lower()
    assert "suspicious" in out or "too small" in out or "tiny" in out


def test_doctor_accepts_plausible_llama_lib(cli_runner, tmp_path, monkeypatch):
    """A normally sized llama.dll is reported as found and healthy."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "llama.dll").write_bytes(b"\x00" * (2 * 1024 * 1024))  # 2 MB

    monkeypatch.setattr(cli, "find_binary_dir", lambda: bindir)
    _no_smi(monkeypatch)
    _install_torch(monkeypatch, [])  # stub torch; this test is about the lib

    result = cli_runner.invoke(cli.doctor, [])
    assert "llama.dll found" in result.output


# --------------------------------------------------------------------------- #
#  BUG-2 - no self-contradiction (CPU-only + torch GPU in same run)            #
# --------------------------------------------------------------------------- #

def test_doctor_no_cpu_only_warning_when_torch_sees_gpu(cli_runner, monkeypatch):
    """nvidia-smi/rocm-smi absent BUT torch sees a GPU -> no 'CPU mode only'."""
    monkeypatch.setattr(cli, "find_binary_dir", lambda: None)
    _no_smi(monkeypatch)
    _install_torch(monkeypatch, ["RTX 4090"])

    result = cli_runner.invoke(cli.doctor, [])
    out = result.output
    # The torch GPU line must still appear...
    assert "RTX 4090" in out
    # ...but the contradictory CPU-only warning must NOT.
    assert "CPU mode only" not in out


def test_doctor_cpu_only_warning_when_no_gpu_at_all(cli_runner, monkeypatch):
    """No smi AND torch reports no GPU -> the CPU-only warning is legitimate."""
    monkeypatch.setattr(cli, "find_binary_dir", lambda: None)
    _no_smi(monkeypatch)
    _install_torch(monkeypatch, [])  # torch present, no CUDA device

    result = cli_runner.invoke(cli.doctor, [])
    assert "CPU mode only" in result.output


def test_doctor_cpu_only_warning_when_torch_missing(cli_runner, monkeypatch):
    """No smi AND torch not installed -> CPU-only warning still legitimate."""
    monkeypatch.setattr(cli, "find_binary_dir", lambda: None)
    _no_smi(monkeypatch)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    # Force `import torch` to fail even if torch is installed in the venv.
    monkeypatch.setitem(sys.modules, "torch", None)

    result = cli_runner.invoke(cli.doctor, [])
    assert "CPU mode only" in result.output


# --------------------------------------------------------------------------- #
#  FAC-4 - rich version via importlib.metadata                                 #
# --------------------------------------------------------------------------- #

def test_doctor_shows_rich_version(cli_runner, monkeypatch):
    """'rich' has no __version__ attr; doctor must still print a real version."""
    import importlib.metadata as ilm

    rich_ver = ilm.version("rich")

    monkeypatch.setattr(cli, "find_binary_dir", lambda: None)
    _no_smi(monkeypatch)
    # Stub torch so the doctor's GPU probe does not import the heavy real torch
    # (keeps this test about the package-version report, not the GPU section).
    _install_torch(monkeypatch, [])

    result = cli_runner.invoke(cli.doctor, [])
    out = result.output
    # The rich line must carry its actual installed version, not a blank.
    assert rich_ver in out
    # Sanity: rich genuinely lacks __version__, which is why the old code blanked.
    import rich
    assert getattr(rich, "__version__", None) is None


# --------------------------------------------------------------------------- #
#  GAP-CLI-3 - /temp and /tokens clamping                                      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    ("/temp -1", 0.0),
    ("/temp 0", 0.0),
    ("/temp 5", 2.0),
    ("/temp 99", 2.0),
    ("/temp 0.7", 0.7),
])
def test_temp_is_clamped(raw, expected):
    gen = {}
    cli._handle_command(raw, [], gen)
    assert gen["temperature"] == pytest.approx(expected)


@pytest.mark.parametrize("raw,expected_min", [
    ("/tokens -5", 1),
    ("/tokens 0", 1),
])
def test_tokens_floor(raw, expected_min):
    gen = {}
    cli._handle_command(raw, [], gen)
    assert gen["max_tokens"] >= 1
    assert gen["max_tokens"] == expected_min


def test_tokens_absurdly_large_is_capped():
    gen = {}
    cli._handle_command("/tokens 100000000000", [], gen)
    # Must not store the absurd value verbatim.
    assert gen["max_tokens"] < 100000000000
    assert gen["max_tokens"] >= 1


def test_tokens_normal_value_passes_through():
    gen = {}
    cli._handle_command("/tokens 2048", [], gen)
    assert gen["max_tokens"] == 2048


# --------------------------------------------------------------------------- #
#  SEC-9 - /save confined to cwd                                              #
# --------------------------------------------------------------------------- #

def test_save_rejects_parent_traversal(tmp_path, monkeypatch):
    """/save ../escape.json must NOT write outside the cwd."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    msgs = [{"role": "user", "content": "hi"}]
    cli._handle_command("/save ../escape.json", msgs, {})

    # The escape target must not exist outside cwd.
    assert not (tmp_path / "escape.json").exists()
    assert not (workdir / ".." / "escape.json").resolve().exists()


def test_save_rejects_absolute_path(tmp_path, monkeypatch):
    """/save <absolute path outside cwd> must be rejected."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(workdir)

    target = outside / "leak.json"
    cli._handle_command(f"/save {target}", [{"role": "user", "content": "hi"}], {})

    assert not target.exists()


def test_save_allows_in_cwd(tmp_path, monkeypatch):
    """A plain filename inside the cwd is still allowed."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    cli._handle_command("/save chat.json", [{"role": "user", "content": "hi"}], {})
    assert (workdir / "chat.json").exists()


def test_save_allows_subdir_in_cwd(tmp_path, monkeypatch):
    """A relative path into a subdir of cwd is allowed."""
    workdir = tmp_path / "work"
    (workdir / "sub").mkdir(parents=True)
    monkeypatch.chdir(workdir)

    cli._handle_command("/save sub/chat.json", [{"role": "user", "content": "hi"}], {})
    assert (workdir / "sub" / "chat.json").exists()
