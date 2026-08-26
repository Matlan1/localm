# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for `localm doctor` and the REPL `_handle_command`.

Covers:
  - doctor must check the llama lib's *integrity* (size>0), not mere existence.
  - doctor must not print "CPU mode only" when torch reports a GPU.
  - doctor must show a real version for 'rich' (importlib.metadata).
  - /temp and /tokens must clamp absurd/negative/zero values.
  - /save must be confined to the cwd (reject traversal/absolute escapes).

The doctor tests drive the real click command through the ``cli_runner``
fixture and monkeypatch the smi/torch/rich probes. The REPL tests unit-call
``_handle_command`` directly.
"""

import importlib
import sys
import types

import pytest

import localm.cli as cli

# localm.cli re-exports `doctor` as the click Command itself, shadowing the
# submodule name - go through importlib for the module.
doctor_mod = importlib.import_module("localm.cli.doctor")


@pytest.fixture(autouse=True)
def _neutralise_native_lib_loaded():
    """_loader.native_lib_loaded() is True for the rest of ANY xdist worker in
    which a real_gguf-gated test has RUN (conftest.py's lazy resource gate - or
    the test itself - calls load_lib() at that test's setup, and _loaded_lib is
    never reset). Once True, doctor.py's own _check_vram_torch() skips the torch
    attempt ENTIRELY (see its docstring - the same known-doomed DLL-identity
    conflict), so test_doctor_no_cpu_only_warning_when_torch_sees_gpu's fake
    "RTX 4090" torch never gets read at all.

    Stays an opt-in, module-scoped fixture rather than a global one: another
    test module unit-tests native_lib_loaded() itself, and a global override
    would silently defeat that test's own mock instead of guarding against the
    real cross-worker pollution. Patches the FUNCTION, not the underlying
    _loaded_lib variable (there is no separate cache variable here) - restored
    after every test."""
    from localm.inference.backends.llamacpp import _loader
    saved = _loader.native_lib_loaded
    _loader.native_lib_loaded = lambda: False
    yield
    _loader.native_lib_loaded = saved


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
    # The stub is only a `.cuda` stand-in, so a real installed transformers
    # would fail its lazy Auto* resolution against it. Skip that check here.
    monkeypatch.setattr(doctor_mod, "_check_hf_backend_usable", lambda *a, **k: None)


# --------------------------------------------------------------------------- #
#  llama lib integrity (size>0)                                                #
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
#  No self-contradiction (CPU-only + torch GPU in same run)                    #
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
#  The torch VRAM probe OMITS an untrusted free reading rather than printing   #
#  it with a caveat. Covers both dimensions: free_scope and probe freshness    #
#  (list_gpus() without return_status=True can serve a stale list).            #
# --------------------------------------------------------------------------- #

def _corrected_gpus(monkeypatch, gpus, status):
    """Control discover.list_gpus(deadline=..., return_status=True), the
    device-global correction call _check_vram_torch makes."""
    from localm import discover
    monkeypatch.setattr(
        discover, "list_gpus",
        lambda *, deadline=None, return_status=False: (
            (list(gpus), status) if return_status else list(gpus)))


class TestDoctorVramReadingHonesty:
    def test_process_scoped_correction_is_omitted_not_caveated(
        self, cli_runner, monkeypatch
    ):
        from localm import discover
        monkeypatch.setattr(cli, "find_binary_dir", lambda: None)
        _no_smi(monkeypatch)
        _install_torch(monkeypatch, ["RTX 4090"])
        _corrected_gpus(monkeypatch, [
            {"index": 0, "total": 16 * 1024**3, "free": 15 * 1024**3,
             "free_scope": discover.FREE_SCOPE_PROCESS}
        ], discover.GPU_PROBE_OK)

        out = cli_runner.invoke(cli.doctor, []).output
        assert "GB free" not in out
        assert "16.0 GB total" in out
        assert "free VRAM reading unavailable on this platform" in " ".join(out.split())

    def test_device_scoped_fresh_correction_is_shown(self, cli_runner, monkeypatch):
        from localm import discover
        monkeypatch.setattr(cli, "find_binary_dir", lambda: None)
        _no_smi(monkeypatch)
        _install_torch(monkeypatch, ["RTX 4090"])
        _corrected_gpus(monkeypatch, [
            {"index": 0, "total": 16 * 1024**3, "free": 15 * 1024**3,
             "free_scope": discover.FREE_SCOPE_DEVICE}
        ], discover.GPU_PROBE_OK)

        out = cli_runner.invoke(cli.doctor, []).output
        assert "15.0 GB free" in out
        assert "free VRAM reading unavailable" not in " ".join(out.split())

    def test_stale_correction_is_omitted_even_with_device_scope(
        self, cli_runner, monkeypatch
    ):
        """A served last-known-good list (TIMEOUT/BUSY/INCONCLUSIVE) is not a
        current measurement even when it carries a FREE_SCOPE_DEVICE tag from
        the earlier successful probe that produced it. Calling list_gpus()
        without return_status=True substitutes a stale-but-tagged-device
        correction and prints it as fact."""
        from localm import discover
        monkeypatch.setattr(cli, "find_binary_dir", lambda: None)
        _no_smi(monkeypatch)
        _install_torch(monkeypatch, ["RTX 4090"])
        _corrected_gpus(monkeypatch, [
            {"index": 0, "total": 16 * 1024**3, "free": 15 * 1024**3,
             "free_scope": discover.FREE_SCOPE_DEVICE}
        ], discover.GPU_PROBE_TIMEOUT)

        out = cli_runner.invoke(cli.doctor, []).output
        assert "GB free" not in out
        assert "16.0 GB total" in out
        assert "free VRAM reading unavailable on this platform" in " ".join(out.split())

    def test_uncorrected_raw_reading_on_blind_platform_is_omitted(
        self, cli_runner, monkeypatch
    ):
        """No corrected entry exists (list_gpus returns nothing for this index -
        the cold-probe/registry-fallback case) so the raw torch.cuda.mem_get_info
        value stands; on a platform gpu_usage says is known process-scoped-blind,
        that raw figure must be omitted too, exactly like the corrected case."""
        from localm import discover
        monkeypatch.setattr(cli, "find_binary_dir", lambda: None)
        _no_smi(monkeypatch)
        _install_torch(monkeypatch, ["RTX 4090"])   # mem_get_info -> 8/16 GB
        _corrected_gpus(monkeypatch, [], discover.GPU_PROBE_OK)
        monkeypatch.setattr(
            "localm.gpu_usage.raw_reading_is_process_scoped", lambda: True)

        out = cli_runner.invoke(cli.doctor, []).output
        assert "GB free" not in out
        assert "16.0 GB total" in out
        assert "free VRAM reading unavailable on this platform" in " ".join(out.split())

    def test_uncorrected_raw_reading_on_device_global_platform_is_shown(
        self, cli_runner, monkeypatch
    ):
        """The complement: no corrected entry, but this platform is NOT known
        blind (Linux/NVIDIA) - the raw torch reading stands unqualified."""
        from localm import discover
        monkeypatch.setattr(cli, "find_binary_dir", lambda: None)
        _no_smi(monkeypatch)
        _install_torch(monkeypatch, ["RTX 4090"])   # mem_get_info -> 8/16 GB
        _corrected_gpus(monkeypatch, [], discover.GPU_PROBE_OK)
        monkeypatch.setattr(
            "localm.gpu_usage.raw_reading_is_process_scoped", lambda: False)

        out = cli_runner.invoke(cli.doctor, []).output
        assert "8.0 GB free" in out
        assert "free VRAM reading unavailable" not in " ".join(out.split())


# --------------------------------------------------------------------------- #
#  rich version via importlib.metadata                                         #
# --------------------------------------------------------------------------- #

def test_doctor_shows_rich_version(cli_runner, monkeypatch):
    """'rich' has no __version__ attr; doctor must still print a real version."""
    import importlib.metadata as ilm

    rich_ver = ilm.version("rich")

    monkeypatch.setattr(cli, "find_binary_dir", lambda: None)
    _no_smi(monkeypatch)
    # Stub torch so the doctor's GPU probe does not import the real torch.
    _install_torch(monkeypatch, [])

    result = cli_runner.invoke(cli.doctor, [])
    out = result.output
    # The rich line must carry its actual installed version, not a blank.
    assert rich_ver in out
    # rich genuinely lacks __version__.
    import rich
    assert getattr(rich, "__version__", None) is None


# --------------------------------------------------------------------------- #
#  /temp and /tokens clamping                                                  #
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
#  /save confined to cwd                                                      #
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
