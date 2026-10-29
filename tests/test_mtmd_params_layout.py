# SPDX-License-Identifier: AGPL-3.0-or-later
"""``n_threads`` and ``image_min_tokens`` of ``mtmd_context_params`` are written
where the runtime in use keeps them: the layout is found by where the default
struct holds the address of ``mtmd_default_marker()``, and an unrecognised
layout is left alone (``n_threads`` at byte 4 as before, no minimum tokens)."""

from __future__ import annotations

import ctypes
import struct

import pytest

from localm.inference.backends.llamacpp import mtmd as lmtmd

MARKER = ctypes.create_string_buffer(b"<__media__>")
MARKER_FN = ctypes.CFUNCTYPE(ctypes.c_void_p)(lambda: ctypes.addressof(MARKER))

# marker offset, n_threads offset, image_min_tokens offset
OLD = (16, 4, 32)
NEW = (32, 20, 48)


class _FakeRuntime:
    """A runtime whose default params follow one layout and that records the
    params it is asked to open a context with."""

    def __init__(self, layout, *, min_default=-1, init_results=(0x7000,)):
        self.layout = layout
        self.min_default = min_default
        self.opened = []
        self.freed = []
        self._results = list(init_results)
        self.mtmd_default_marker = MARKER_FN

    def mtmd_context_params_default(self):
        p = lmtmd._MtmdParams()
        marker_at, threads_at, min_at = self.layout
        raw = (ctypes.c_uint8 * 64).from_address(ctypes.addressof(p))
        for i in range(64):
            raw[i] = 0xAB
        struct.pack_into("<Q", raw, marker_at, ctypes.addressof(MARKER))
        struct.pack_into("<i", raw, threads_at, 4)
        struct.pack_into("<i", raw, min_at, self.min_default)
        struct.pack_into("<i", raw, min_at + 4, -1)
        raw[0] = 1
        return p

    def mtmd_init_from_file(self, path, model, params):
        raw = ctypes.string_at(ctypes.byref(params), 64)
        marker_at, threads_at, min_at = self.layout
        self.opened.append({
            "use_gpu": raw[0],
            "n_threads": struct.unpack_from("<i", raw, threads_at)[0],
            "min": struct.unpack_from("<i", raw, min_at)[0],
            "max": struct.unpack_from("<i", raw, min_at + 4)[0],
            "byte4": struct.unpack_from("<i", raw, 4)[0],
            "raw": raw,
        })
        return self._results.pop(0) if self._results else 0x7000

    def mtmd_free(self, ctx):
        self.freed.append(ctx)


def _ctx(runtime, *, on_gpu=True):
    c = lmtmd.MtmdContext.__new__(lmtmd.MtmdContext)
    c._m = runtime
    c._mmproj_path = "x.gguf"
    c._model_ptr = 1
    c._gpu_index = 0
    c.on_gpu = on_gpu
    c._ctx = 0x7000
    c._embd = None
    c._embd_bytes = 0
    return c


@pytest.mark.parametrize("layout", [OLD, NEW])
def test_a_known_layout_is_found_and_its_offsets_are_used(layout):
    rt = _FakeRuntime(layout)
    found = lmtmd._params_layout(rt, rt.mtmd_context_params_default())
    assert found == lmtmd._ParamLayout(layout[1], layout[2])
    _ctx(rt)._open(use_gpu=True)
    assert rt.opened[-1]["n_threads"] == lmtmd._encode_threads()
    assert rt.opened[-1]["min"] == -1


@pytest.mark.parametrize("layout", [OLD, NEW])
def test_the_minimum_image_tokens_land_in_their_field_and_nowhere_else(layout):
    rt = _FakeRuntime(layout)
    _ctx(rt)._open(use_gpu=True)
    base = rt.opened[-1]["raw"]
    _ctx(rt)._open(use_gpu=True, min_image_tokens=1024)
    changed = [i for i in range(64) if rt.opened[-1]["raw"][i] != base[i]]
    assert rt.opened[-1]["min"] == 1024
    assert rt.opened[-1]["max"] == -1
    assert changed and set(changed) <= set(range(layout[2], layout[2] + 4))


def test_the_new_layout_no_longer_writes_n_threads_into_padding():
    rt = _FakeRuntime(NEW)
    _ctx(rt)._open(use_gpu=True)
    assert rt.opened[-1]["byte4"] == struct.unpack("<i", bytes([0xAB] * 4))[0]


class _UnknownRuntime(_FakeRuntime):
    def mtmd_context_params_default(self):
        p = lmtmd._MtmdParams()
        raw = (ctypes.c_uint8 * 64).from_address(ctypes.addressof(p))
        for i in range(64):
            raw[i] = 0
        return p


def test_an_unrecognised_layout_keeps_n_threads_at_byte_4_and_sets_no_minimum():
    rt = _UnknownRuntime(OLD)
    rt.mtmd_default_marker = MARKER_FN
    c = _ctx(rt)
    c._open(use_gpu=True, min_image_tokens=1024)
    assert rt.opened[-1]["byte4"] == lmtmd._encode_threads()
    assert c.apply_min_image_tokens(1024) is False
    assert rt.freed == []


def test_a_runtime_without_a_default_marker_function_is_unrecognised():
    class _Bare:
        def mtmd_context_params_default(self):
            return lmtmd._MtmdParams()
    assert lmtmd._params_layout(_Bare(), _Bare().mtmd_context_params_default()) is None


@pytest.mark.parametrize("layout", [OLD, NEW])
def test_apply_min_image_tokens_rebuilds_the_context_on_the_same_device(layout):
    rt = _FakeRuntime(layout, init_results=(0x8000,))
    c = _ctx(rt, on_gpu=False)
    assert c.apply_min_image_tokens(1024) is True
    assert rt.freed == [0x7000]
    assert c._ctx == 0x8000
    assert rt.opened[-1]["min"] == 1024
    assert rt.opened[-1]["use_gpu"] == 0


def test_apply_min_image_tokens_leaves_a_runtime_with_another_default_alone():
    rt = _FakeRuntime(NEW, min_default=256)
    c = _ctx(rt)
    assert c.apply_min_image_tokens(1024) is False
    assert rt.freed == [] and rt.opened == []


def test_a_failed_rebuild_restores_a_context_without_the_setting():
    rt = _FakeRuntime(NEW, init_results=(0, 0x9000))
    c = _ctx(rt)
    assert c.apply_min_image_tokens(1024) is False
    assert c._ctx == 0x9000
    assert [o["min"] for o in rt.opened] == [1024, -1]


def test_a_rebuild_that_cannot_be_restored_raises():
    rt = _FakeRuntime(NEW, init_results=(0, 0))
    with pytest.raises(lmtmd.MtmdUnavailable):
        _ctx(rt).apply_min_image_tokens(1024)


@pytest.mark.parametrize("layout", [OLD, NEW])
def test_the_cpu_retry_keeps_the_applied_minimum_image_tokens(layout):
    rt = _FakeRuntime(layout, init_results=(0x8000, 0x8100))
    c = _ctx(rt, on_gpu=True)
    assert c.apply_min_image_tokens(1024) is True
    assert c.retry_on_cpu() is True
    assert rt.opened[-1]["use_gpu"] == 0
    assert rt.opened[-1]["min"] == 1024
    assert c._ctx == 0x8100


@pytest.mark.parametrize("layout", [OLD, NEW])
def test_the_cpu_retry_of_a_context_that_applied_nothing_asks_for_no_minimum(layout):
    rt = _FakeRuntime(layout, init_results=(0x8100,))
    c = _ctx(rt, on_gpu=True)
    assert c.retry_on_cpu() is True
    assert rt.opened[-1]["use_gpu"] == 0
    assert rt.opened[-1]["min"] == -1


def test_a_failed_minimum_rebuild_is_not_remembered_for_the_cpu_retry():
    rt = _FakeRuntime(NEW, init_results=(0, 0x9000, 0x9100))
    c = _ctx(rt, on_gpu=True)
    assert c.apply_min_image_tokens(1024) is False
    assert c.retry_on_cpu() is True
    assert [o["min"] for o in rt.opened] == [1024, -1, -1]


def test_the_shipped_runtime_layout_is_recognised():
    """Against the real mtmd of the provisioned runtime: its default struct must
    match one of the known layouts, with the documented defaults in the fields."""
    try:
        m = lmtmd._load_lib()
    except lmtmd.MtmdUnavailable:
        pytest.skip("no multimodal runtime provisioned")
    params = m.mtmd_context_params_default()
    layout = lmtmd._params_layout(m, params)
    assert layout is not None, "the shipped mtmd has a params layout localm does not know"
    raw = ctypes.string_at(ctypes.byref(params), 64)
    assert struct.unpack_from("<i", raw, layout.n_threads)[0] == 4
    assert struct.unpack_from("<i", raw, layout.image_min_tokens)[0] == -1
