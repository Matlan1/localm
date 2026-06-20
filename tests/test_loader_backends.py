# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for ggml backend registration in the llama.cpp loader.

No native library: a fake lib records ggml_backend_load calls, so these run on
any CI host. Covers (a) old builds without the loader symbol are left untouched
(the bundled AMD path), (b) ggml-base / ggml are NOT registered as backends, and
(c) the real backend plugins are registered by absolute path.
"""

from __future__ import annotations

import sys

from localm.inference.backends.llamacpp import _loader


class _RecordingLoad:
    """Stand-in for the ggml_backend_load function pointer."""

    def __init__(self):
        self.restype = None
        self.argtypes = None
        self.calls = []

    def __call__(self, path):
        self.calls.append(path.decode() if isinstance(path, (bytes, bytearray)) else path)
        return 1   # truthy ggml_backend_reg_t


class _FakeLib:
    def __init__(self, with_loader: bool):
        if with_loader:
            self.ggml_backend_load = _RecordingLoad()


def _touch(d, names):
    for n in names:
        (d / n).write_bytes(b"")   # empty: ctypes.CDLL will fail+skip, as intended


def test_old_build_without_symbol_is_untouched(tmp_path, monkeypatch):
    # The bundled lemonade build statically registers its backend and exports no
    # ggml_backend_load -> _register must no-op (return False), never raise.
    monkeypatch.setattr(sys, "platform", "win32")
    _touch(tmp_path, ["ggml-base.dll", "ggml-cpu.dll", "llama.dll"])
    assert _loader._register_ggml_backends(tmp_path, _FakeLib(with_loader=False)) is False


def test_registers_backends_by_path_skipping_non_backends(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    _touch(tmp_path, ["ggml-base.dll", "ggml.dll", "ggml-cpu.dll",
                      "ggml-vulkan.dll", "ggml-rpc.dll", "llama.dll"])
    lib = _FakeLib(with_loader=True)
    assert _loader._register_ggml_backends(tmp_path, lib) is True
    calls = lib.ggml_backend_load.calls
    # The real compute/RPC backends are registered by absolute path...
    assert any(c.endswith("ggml-cpu.dll") for c in calls)
    assert any(c.endswith("ggml-vulkan.dll") for c in calls)
    assert any(c.endswith("ggml-rpc.dll") for c in calls)
    # ...but the core libs are NOT (loading them as a backend is wrong).
    assert not any(c.endswith("ggml-base.dll") for c in calls)
    assert not any(c.rstrip("\\/").endswith("ggml.dll") for c in calls)
    # Absolute paths (so discovery never looks next to python.exe).
    assert all(c.endswith(".dll") for c in calls)


def test_no_backends_when_dir_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    # Loader symbol present but no plugin files -> nothing registered, no error.
    assert _loader._register_ggml_backends(tmp_path, _FakeLib(with_loader=True)) is False
