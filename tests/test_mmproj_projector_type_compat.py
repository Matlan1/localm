# SPDX-License-Identifier: AGPL-3.0-or-later
"""A vision projector that records no projector type (llava-1.5-era exports) is
loaded from a copy that records the type its tensors show, because the bundled
runtime refuses it otherwise ("unknown projector type")."""

import logging
import os
import struct

import pytest

import localm.config as config
from localm.inference.backends.llamacpp import mtmd as mtmd_mod
from localm.model_manager import gguf as gguf_mod
from localm.model_manager.gguf import (
    _gguf_header_layout,
    gguf_mmproj_inferred_projector_type,
    gguf_n_embd,
    write_gguf_with_string_kv,
)
from tests.test_mmproj_discovery import (
    _T_STRING,
    _T_UINT32,
    _real_llava15_mmproj_gguf,
    _real_llava16_mmproj_gguf,
    _real_text_model_gguf,
    _write_gguf,
)


def _with_tensor_data(path, alignment=32):
    """Append alignment padding and distinct tensor bytes to a header written by
    ``_write_gguf`` (f16 tensors, offsets laid out back to back)."""
    with open(path, "rb") as f:
        layout = _gguf_header_layout(f)
    raw = path.read_bytes()
    pad = -len(raw) % alignment
    data_len = 0
    # _write_gguf lays tensors out back to back at 2 bytes per element.
    with open(path, "rb") as f:
        f.seek(24)
        for _ in range(layout.kv_count):
            gguf_mod._gguf_read_string_stream(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            gguf_mod._gguf_skip_value_stream(f, vtype)
        for _ in layout.tensor_names:
            gguf_mod._gguf_read_string_stream(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            f.read(12)
            n = 1
            for d in dims:
                n *= d
            data_len += 2 * n
    data = bytes((i * 7 + 3) % 251 for i in range(data_len))
    path.write_bytes(raw + b"\x00" * pad + data)
    return data


def _small_llava15(path):
    """The llava-1.5 projector layout (no projector-type key, GGUF v2) with small
    tensors and real tensor data."""
    _write_gguf(path, [
        ("general.architecture", _T_STRING, "clip"),
        ("clip.vision.embedding_length", _T_UINT32, 16),
        ("clip.vision.projection_dim", _T_UINT32, 768)],
        [("mm.0.weight", (16, 64)), ("mm.0.bias", (64,)),
         ("mm.2.weight", (64, 64)), ("mm.2.bias", (64,))], version=2)
    return _with_tensor_data(path)


def _data_section(path):
    with open(path, "rb") as f:
        layout = _gguf_header_layout(f)
    raw = path.read_bytes()
    start = layout.info_end + (-layout.info_end % layout.alignment)
    return start, raw[start:]


def _header_string(path, key):
    with open(path, "rb") as f:
        kv_count = _gguf_header_layout(f).kv_count
        f.seek(24)
        for _ in range(kv_count):
            k = gguf_mod._gguf_read_string_stream(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            if k == key and vtype == _T_STRING:
                return gguf_mod._gguf_read_string_stream(f)
            gguf_mod._gguf_skip_value_stream(f, vtype)
    return None


class TestInferredProjectorType:

    def test_llava15_projector_reads_as_mlp(self, tmp_path):
        assert gguf_mmproj_inferred_projector_type(
            _real_llava15_mmproj_gguf(tmp_path / "mmproj-model-f16.gguf")) == "mlp"

    def test_projector_that_records_its_type_needs_nothing(self, tmp_path):
        assert gguf_mmproj_inferred_projector_type(
            _real_llava16_mmproj_gguf(tmp_path / "mmproj-model-f16.gguf")) is None

    def test_vision_projector_type_key_counts_as_recorded(self, tmp_path):
        proj = _write_gguf(tmp_path / "mmproj.gguf", [
            ("general.architecture", _T_STRING, "clip"),
            ("clip.vision.projector_type", _T_STRING, "qwen25vl")],
            [("mm.2.weight", (64, 64))])
        assert gguf_mmproj_inferred_projector_type(proj) is None

    @pytest.mark.parametrize("tensor, expected", [
        ("mm.model.mb_block.1.block.2.1.bias", "ldp"),
        ("mm.model.peg.0.bias", "ldpv2"),
        ("mm.3.weight", "mlp")])
    def test_mobilevlm_and_yi_layouts(self, tmp_path, tensor, expected):
        proj = _write_gguf(tmp_path / "mmproj.gguf", [
            ("general.architecture", _T_STRING, "clip")], [(tensor, (64,))])
        assert gguf_mmproj_inferred_projector_type(proj) == expected

    def test_unrecognised_layout_is_not_guessed(self, tmp_path):
        proj = _write_gguf(tmp_path / "mmproj.gguf", [
            ("general.architecture", _T_STRING, "clip")],
            [("v.patch_embd.weight", (14, 14, 3, 1024))])
        assert gguf_mmproj_inferred_projector_type(proj) is None

    def test_text_model_and_unreadable_files_are_none(self, tmp_path):
        assert gguf_mmproj_inferred_projector_type(
            _real_text_model_gguf(tmp_path / "m.gguf", "llama", 4096)) is None
        bad = tmp_path / "bad.gguf"
        bad.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\xff" * 40)
        assert gguf_mmproj_inferred_projector_type(bad) is None
        assert gguf_mmproj_inferred_projector_type(tmp_path / "missing.gguf") is None


class TestWriteWithStringKv:

    def test_copy_records_the_key_and_keeps_every_tensor_byte(self, tmp_path):
        src = tmp_path / "src.gguf"
        data = _small_llava15(src)
        before = src.read_bytes()
        dst = tmp_path / "dst.gguf"

        write_gguf_with_string_kv(src, dst, "clip.projector_type", "mlp")

        start, copied = _data_section(dst)
        assert copied == data, "tensor data changed in the copy"
        assert start % 32 == 0
        assert src.read_bytes() == before, "the source file was modified"
        assert _header_string(dst, "clip.projector_type") == "mlp"
        with open(src, "rb") as f:
            src_layout = _gguf_header_layout(f)
        with open(dst, "rb") as f:
            dst_layout = _gguf_header_layout(f)
        assert dst_layout.tensor_names == src_layout.tensor_names
        assert dst_layout.keys == src_layout.keys | {"clip.projector_type"}
        assert dst_layout.version == src_layout.version
        assert gguf_n_embd(dst) == gguf_n_embd(src) == 64
        assert gguf_mmproj_inferred_projector_type(dst) is None

    def test_existing_key_is_refused(self, tmp_path):
        src = _real_llava16_mmproj_gguf(tmp_path / "src.gguf")
        with pytest.raises(ValueError, match="already has"):
            write_gguf_with_string_kv(src, tmp_path / "dst.gguf", "clip.projector_type", "mlp")


@pytest.fixture
def compat_cache(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setattr(config, "cache_dir", lambda: root)
    return root / "mmproj-compat"


class TestCompatibleMmprojPath:

    def test_typeless_projector_is_served_from_a_typed_copy(self, tmp_path, compat_cache):
        src = tmp_path / "mmproj-model-f16.gguf"
        data = _small_llava15(src)
        before = src.read_bytes()

        out = mtmd_mod.compatible_mmproj_path(str(src))

        assert _data_section(type(src)(out))[1] == data
        assert src.read_bytes() == before, "the source file was modified"
        assert os.path.dirname(out) == str(compat_cache)
        assert _header_string(type(src)(out), "clip.projector_type") == "mlp"

    def test_copy_is_reused_until_the_source_changes(self, tmp_path, compat_cache, monkeypatch):
        src = tmp_path / "mmproj-model-f16.gguf"
        _small_llava15(src)
        first = mtmd_mod.compatible_mmproj_path(str(src))

        writes = []
        real_write = gguf_mod.write_gguf_with_string_kv
        monkeypatch.setattr(gguf_mod, "write_gguf_with_string_kv",
                            lambda *a: writes.append(a) or real_write(*a))
        assert mtmd_mod.compatible_mmproj_path(str(src)) == first
        assert writes == [], "an up-to-date copy was rewritten"

        st = src.stat()
        os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))
        assert mtmd_mod.compatible_mmproj_path(str(src)) == first
        assert len(writes) == 1, "a changed source did not refresh its copy"

    def test_projector_that_records_its_type_is_used_as_is(self, tmp_path, compat_cache):
        src = _real_llava16_mmproj_gguf(tmp_path / "mmproj-model-f16.gguf")
        assert mtmd_mod.compatible_mmproj_path(str(src)) == str(src)
        assert not compat_cache.exists()

    def test_failed_write_falls_back_to_the_original_and_says_so(
            self, tmp_path, compat_cache, monkeypatch, caplog):
        src = tmp_path / "mmproj-model-f16.gguf"
        _small_llava15(src)

        def _fail(src_path, dst_path, key, value):
            dst_path.write_bytes(b"partial")
            raise OSError("disk full")
        monkeypatch.setattr(gguf_mod, "write_gguf_with_string_kv", _fail)

        with caplog.at_level(logging.WARNING, logger="localm"):
            out = mtmd_mod.compatible_mmproj_path(str(src))

        assert out == str(src)
        assert list(compat_cache.iterdir()) == [], "a partial copy was left behind"
        assert any("disk full" in r.getMessage() for r in caplog.records)

    def test_too_little_space_falls_back_to_the_original_and_says_so(
            self, tmp_path, compat_cache, monkeypatch, caplog):
        src = tmp_path / "mmproj-model-f16.gguf"
        _small_llava15(src)
        import shutil
        import types
        monkeypatch.setattr(shutil, "disk_usage", lambda p: types.SimpleNamespace(free=0))
        with caplog.at_level(logging.WARNING, logger="localm"):
            out = mtmd_mod.compatible_mmproj_path(str(src))
        assert out == str(src)
        assert any("free" in r.getMessage() for r in caplog.records)


def test_load_mmproj_hands_mtmd_the_compatible_path(tmp_path, compat_cache, monkeypatch):
    from localm.inference.backends.llamacpp import llama as llama_mod
    src = tmp_path / "mmproj-model-f16.gguf"
    _small_llava15(src)
    seen = {}

    class _Recorder:
        def __init__(self, mmproj_path, model_ptr, gpu_index=0):
            seen["path"] = mmproj_path
            self.supports_vision = True

    monkeypatch.setattr(mtmd_mod, "MtmdContext", _Recorder)
    inst = llama_mod.LlamaCpp.__new__(llama_mod.LlamaCpp)
    inst._model_ptr = 0xBEEF
    inst._load_mmproj(str(src), verbose=True)

    assert seen["path"] != str(src)
    assert _header_string(type(src)(seen["path"]), "clip.projector_type") == "mlp"
