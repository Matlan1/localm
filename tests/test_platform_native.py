# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-platform native-binary detection / loader logic.

These run on any host by monkeypatching sys.platform, so the Linux and macOS
code paths are exercised from the Windows CI too.
"""

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
    bundled binary, so the upstream archives' ~49 command-line tools were dead
    weight in every Windows install (3.2 MB, including an RPC server daemon).

    The deciding evidence is the platform asymmetry asserted below: darwin and
    Linux have ALWAYS matched libraries only, so those installs never carried a
    single bundled executable and demonstrably work. Windows was the outlier."""
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
