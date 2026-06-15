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
    assert setup_llama._is_wanted(tmp_path / "llama-server.exe")
    assert not setup_llama._is_wanted(tmp_path / "libllama.so")


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
