# SPDX-License-Identifier: AGPL-3.0-or-later
"""general.alignment must be a nonzero power of two, as the GGUF loader requires."""
import struct

import pytest

from localm.model_manager import gguf


def _lstr(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _header_with_alignment(alignment: int) -> bytes:
    kvs = (_lstr("general.alignment") + struct.pack("<II", 4, alignment)
           + _lstr("general.architecture") + struct.pack("<I", 8) + _lstr("clip"))
    return b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + kvs + b"\x00" * 100


@pytest.mark.parametrize("alignment", [0, 3, 33, 48, 1000, 2 ** 20 - 1])
def test_a_rewrite_refuses_an_alignment_that_is_not_a_power_of_two(tmp_path, alignment):
    src, dst = tmp_path / "src.gguf", tmp_path / "dst.gguf"
    src.write_bytes(_header_with_alignment(alignment))
    with pytest.raises(ValueError):
        gguf.write_gguf_with_string_kv(src, dst, "k.new", "v")
    assert not dst.exists()


@pytest.mark.parametrize("alignment", [1, 2, 32, 64, 4096, 2 ** 20])
def test_a_rewrite_accepts_a_power_of_two_alignment(tmp_path, alignment):
    src, dst = tmp_path / "src.gguf", tmp_path / "dst.gguf"
    src.write_bytes(_header_with_alignment(alignment))
    gguf.write_gguf_with_string_kv(src, dst, "k.new", "v")
    assert b"k.new" in dst.read_bytes()
