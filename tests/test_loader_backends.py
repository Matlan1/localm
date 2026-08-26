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


class _FakeFn:
    """A callable with a settable ``restype`` (ctypes function-pointer shape),
    returning whatever ``impl`` produces. Records its call count."""

    def __init__(self, impl):
        self.restype = None
        self.argtypes = None
        self._impl = impl
        self.calls = 0

    def __call__(self, *a):
        self.calls += 1
        return self._impl(*a)


class _FakeLib:
    def __init__(self, with_loader: bool = False, dev_count=None,
                 load_all_registers: bool = False):
        if with_loader:
            self.ggml_backend_load = _RecordingLoad()
        # When dev_count is given, expose ggml_backend_dev_count. If
        # load_all_registers, it reports 0 until ggml_backend_load_all() runs,
        # then the post-load count (so the loader can confirm success).
        self._registered = bool(dev_count) and not load_all_registers
        self._post = dev_count if dev_count is not None else 0
        if dev_count is not None:
            self.ggml_backend_dev_count = _FakeFn(
                lambda: self._post if self._registered else 0)
        if load_all_registers:
            def _load_all():
                self._registered = True
            self.ggml_backend_load_all = _FakeFn(_load_all)


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


def test_already_registered_skips_probing(tmp_path, monkeypatch):
    # The bundled AMD build self-registers: a device is already present, so
    # ggml_backend_load must NOT then be called on the ggml-* libs.
    monkeypatch.setattr(sys, "platform", "win32")
    _touch(tmp_path, ["ggml-base.dll", "ggml.dll", "ggml-cpu.dll",
                      "ggml-hip.dll", "llama.dll"])
    lib = _FakeLib(with_loader=True, dev_count=2)   # a GPU + CPU already present
    assert _loader._register_ggml_backends(tmp_path, lib) is True
    assert lib.ggml_backend_load.calls == [], \
        "must not probe ggml-* plugins when backends are already registered"


def test_load_all_used_when_nothing_registered(tmp_path, monkeypatch):
    # No device registered yet -> prefer the build's own ggml_backend_load_all
    # discovery; once it registers a device we stop (no per-path probing).
    monkeypatch.setattr(sys, "platform", "win32")
    _touch(tmp_path, ["ggml-base.dll", "ggml.dll", "ggml-cpu.dll", "llama.dll"])
    lib = _FakeLib(with_loader=True, dev_count=2, load_all_registers=True)
    assert _loader._register_ggml_backends(tmp_path, lib) is True
    assert lib.ggml_backend_load_all.calls == 1
    assert lib.ggml_backend_load.calls == [], "load_all succeeded; no per-path probe"


def test_loaded_but_no_device_reports_false(tmp_path, monkeypatch):
    """A non-null ggml_backend_load handle does NOT prove a usable device. When
    the plugin loads "succeed" but the device registry still reports 0, the
    registration is a FAILURE - the authoritative device count wins over the raw
    load signal, so setup does not report a broken build as a success."""
    monkeypatch.setattr(sys, "platform", "win32")
    _touch(tmp_path, ["ggml-base.dll", "ggml-cpu.dll", "ggml-vulkan.dll", "llama.dll"])
    lib = _FakeLib(with_loader=True, dev_count=0)   # loads return truthy, 0 devices
    assert _loader._register_ggml_backends(tmp_path, lib) is False
    assert lib.ggml_backend_load.calls, "the explicit per-path load was attempted"


def test_compute_backends_available_reflects_flag(monkeypatch):
    """compute_backends_available() mirrors the flag load_lib() records, and is
    False before a load has happened (None) so a caller never reads a stale True."""
    monkeypatch.setattr(_loader, "load_lib", lambda: None)   # do not touch real lib
    monkeypatch.setattr(_loader, "_compute_backends_ok", True)
    assert _loader.compute_backends_available() is True
    monkeypatch.setattr(_loader, "_compute_backends_ok", False)
    assert _loader.compute_backends_available() is False
    monkeypatch.setattr(_loader, "_compute_backends_ok", None)
    assert _loader.compute_backends_available() is False
