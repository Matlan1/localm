# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-platform native-binary detection / loader logic.

These run on any host by monkeypatching sys.platform, so the Linux and macOS
code paths are exercised from the Windows CI too.
"""

import os
import sys

import pytest


@pytest.mark.parametrize("plat,expected", [
    ("win32", "llama.dll"),
    ("linux", "libllama.so"),
    ("darwin", "libllama.dylib"),
])
def test_loader_lib_filename(monkeypatch, plat, expected):
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(sys, "platform", plat)
    assert _loader.lib_filename() == expected


@pytest.mark.parametrize("plat,expected", [
    ("win32", "ggml*.dll"),
    ("linux", "libggml*.so*"),
    ("darwin", "libggml*.dylib"),
])
def test_loader_ggml_glob(monkeypatch, plat, expected):
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(sys, "platform", plat)
    assert _loader._ggml_glob() == expected


@pytest.mark.parametrize("plat,expected", [
    ("win32", ("llama.dll",)),
    ("linux", ("libllama.so",)),
    ("darwin", ("libllama.dylib",)),
])
def test_config_loadable_lib_names(monkeypatch, plat, expected):
    from localm import config
    monkeypatch.setattr(sys, "platform", plat)
    assert config._loadable_lib_names() == expected


@pytest.mark.parametrize("plat,expected", [
    ("win32", "llama.dll"),
    ("linux", "libllama.so"),
    ("darwin", "libllama.dylib"),
])
def test_setup_llama_lib_name(monkeypatch, plat, expected):
    from localm import setup_llama
    monkeypatch.setattr(sys, "platform", plat)
    assert setup_llama._lib_name() == expected


def test_setup_llama_is_wanted_linux(monkeypatch, tmp_path):
    from localm import setup_llama
    monkeypatch.setattr(sys, "platform", "linux")
    assert setup_llama._is_wanted(tmp_path / "libllama.so")
    assert setup_llama._is_wanted(tmp_path / "libggml-base.so")
    assert setup_llama._is_wanted(tmp_path / "libfoo.so.1")          # versioned
    assert not setup_llama._is_wanted(tmp_path / "llama.dll")
    assert not setup_llama._is_wanted(tmp_path / "readme.txt")


def test_setup_llama_is_wanted_windows(monkeypatch, tmp_path):
    from localm import setup_llama
    monkeypatch.setattr(sys, "platform", "win32")
    assert setup_llama._is_wanted(tmp_path / "llama.dll")
    assert setup_llama._is_wanted(tmp_path / "ggml-hip.dll")
    assert not setup_llama._is_wanted(tmp_path / "libllama.so")


@pytest.mark.parametrize("name", [
    "llama-server.exe", "llama-cli.exe", "llama-bench.exe",
    "ggml-rpc-server.exe", "llama.exe",
])
def test_setup_llama_never_ships_an_executable(monkeypatch, tmp_path, name):
    """localm loads the runtime in-process via ctypes and never shells out to a
    bundled binary, so the upstream archives' command-line tools (including an
    RPC server daemon) are never copied into a Windows install.

    darwin and Linux match libraries only, so those installs carry no bundled
    executable at all; the assertion below pins that asymmetry."""
    monkeypatch.setattr(sys, "platform", "win32")
    assert not setup_llama_mod()._is_wanted(tmp_path / name)


def setup_llama_mod():
    from localm import setup_llama
    return setup_llama


@pytest.mark.parametrize("plat,exe", [
    ("linux", "llama-cli"), ("darwin", "llama-cli"), ("win32", "llama-cli.exe"),
])
def test_no_platform_ships_executables(monkeypatch, tmp_path, plat, exe):
    """The invariant itself, stated once for all three platforms: whatever the
    naming convention, an EXECUTABLE is never copied into the runtime. Linux and
    darwin already satisfied this by construction; this pins it so a future
    'just add .exe back' cannot pass unnoticed."""
    monkeypatch.setattr(sys, "platform", plat)
    assert not setup_llama_mod()._is_wanted(tmp_path / exe)


def test_find_binary_dir_detects_linux_so(monkeypatch, tmp_path):
    from localm import config
    monkeypatch.setattr(sys, "platform", "linux")
    (tmp_path / "libllama.so").write_bytes(b"\x00")
    monkeypatch.setattr(config, "load_config", lambda: {"binary_dir": str(tmp_path)})
    assert config.find_binary_dir() == tmp_path


def test_find_binary_dir_none_when_lib_absent(monkeypatch, tmp_path):
    from localm import config
    monkeypatch.setattr(sys, "platform", "linux")
    # a Windows llama.dll must NOT satisfy Linux detection
    (tmp_path / "llama.dll").write_bytes(b"\x00")
    monkeypatch.setattr(config, "load_config", lambda: {"binary_dir": str(tmp_path)})
    assert config.find_binary_dir() is None


def test_add_to_search_path_covers_pypi_fetched_cuda_runtime_libs(monkeypatch, tmp_path):
    """_add_to_search_path adds the runtime binary DIRECTORY to LD_LIBRARY_PATH
    unconditionally, for every backend - it has no per-file or per-backend logic
    at all, so setup-llama's PyPI-fetched libcudart.so/libcublas.so/libnccl.so,
    which land in that exact directory, are covered by the SAME mechanism ROCm
    and every other backend relies on. This does not prove the files are
    individually findable via dlopen (that needs a real Linux+CUDA box); it pins
    the one thing verifiable here, that the directory they were fetched into is
    unconditionally added to the search path."""
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    for name in ("libllama.so", "libggml-cuda.so", "libcudart.so.12",
                "libcublas.so.12", "libnccl.so.2"):
        (runtime_dir / name).write_bytes(b"\x7fELF")

    _loader._add_to_search_path(runtime_dir)

    assert str(runtime_dir) in os.environ["LD_LIBRARY_PATH"].split(os.pathsep)
