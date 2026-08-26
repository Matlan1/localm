# SPDX-License-Identifier: AGPL-3.0-or-later
"""_has_gguf_magic enforces a minimum plausible file size, and
_gguf_recently_written enforces a settle period on top of it.

A file that has the 4-byte GGUF magic but almost no body (a placeholder, a
half-written copy that got just the header) must be skipped by auto-registration
instead of passing the magic check and crashing a later model load with an
opaque ggml error. A file that clears BOTH the magic and size checks can still
be an in-progress copy - _gguf_recently_written is the second gate that catches
that case, by refusing to trust a file whose mtime is too fresh.

A file can defeat BOTH of those at once: the magic and the size floor live at
the START of the file, so they survive a copy truncated at the TAIL, and a
backdated mtime reads as settled rather than mid-copy. _has_gguf_magic's third
check - the file must reach at least as far as its own header declares its
last tensor starts - is what TestDeclaredSizeCheck below pins."""

import os
import struct
import time

from localm.model_manager.gguf import (
    _GGUF_MIN_BYTES,
    _GGUF_SETTLE_SECONDS,
    _gguf_declared_min_size,
    _gguf_recently_written,
    _has_gguf_magic,
)


def _s(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf_header(tensors, *, version=3, alignment=32) -> bytes:
    """A real GGUF magic + version + counts + (empty KV) + tensor-info block
    for *tensors* = [(name, size_bytes), ...], alignment-padded - offsets are
    assigned contiguously from the size_bytes, exactly as a real writer lays
    tensors out. Returns only the HEADER bytes; the caller decides how much
    (if any) real tensor DATA to append, to build either a complete file or
    one truncated partway through."""
    out = [b"GGUF", struct.pack("<I", version), struct.pack("<QQ", len(tensors), 0)]
    offset = 0
    for name, size in tensors:
        out.append(_s(name))
        out.append(struct.pack("<I", 1))       # n_dims
        out.append(struct.pack("<Q", 1))        # dims[0] (unused by the check)
        out.append(struct.pack("<I", 0))        # ggml_type F32 (unused by the check)
        out.append(struct.pack("<Q", offset))
        offset += size
    body = b"".join(out)
    remainder = len(body) % alignment
    if remainder:
        body += b"\0" * (alignment - remainder)
    return body


def test_bad_magic_rejected_even_when_large(tmp_path):
    f = tmp_path / "foreign.gguf"
    f.write_bytes(b"not a model".ljust(_GGUF_MIN_BYTES * 2, b"\x00"))
    assert _has_gguf_magic(f) is False


def test_magic_just_below_floor_rejected(tmp_path):
    # Covers the header-only-stub case too: any size < floor takes this same
    # `<` branch.
    f = tmp_path / "below.gguf"
    f.write_bytes(b"GGUF".ljust(_GGUF_MIN_BYTES - 1, b"\x00"))
    assert _has_gguf_magic(f) is False


def test_magic_at_floor_accepted(tmp_path):
    f = tmp_path / "at-floor.gguf"
    f.write_bytes(b"GGUF".ljust(_GGUF_MIN_BYTES, b"\x00"))
    assert _has_gguf_magic(f) is True


def test_magic_above_floor_accepted(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * (_GGUF_MIN_BYTES * 4))
    assert _has_gguf_magic(f) is True


def test_missing_file_rejected(tmp_path):
    assert _has_gguf_magic(tmp_path / "absent.gguf") is False


def test_freshly_written_file_reads_as_recently_written(tmp_path):
    # A file written this instant is indistinguishable, by mtime alone, from
    # the tail end of an in-progress copy - it must be deferred.
    f = tmp_path / "copying.gguf"
    f.write_bytes(b"GGUF".ljust(_GGUF_MIN_BYTES, b"\x00"))
    assert _gguf_recently_written(f) is True


def test_backdated_file_reads_as_settled(tmp_path):
    f = tmp_path / "settled.gguf"
    f.write_bytes(b"GGUF".ljust(_GGUF_MIN_BYTES, b"\x00"))
    old = time.time() - (_GGUF_SETTLE_SECONDS + 5)
    os.utime(f, (old, old))
    assert _gguf_recently_written(f) is False


def test_missing_file_is_not_recently_written(tmp_path):
    # Fails safe: an unreadable file only ever defers registration via this
    # check, never blocks it - it fails a different, earlier check instead.
    assert _gguf_recently_written(tmp_path / "absent.gguf") is False


class TestDeclaredSizeCheck:
    """A copy cut short keeps its magic (start of file) and can have an old
    mtime (backdated here, as a stand-in for a genuinely settled but incomplete
    copy) - both the magic check and the settle check pass. _has_gguf_magic's
    third check - does the file reach as far as its own header says its last
    tensor starts - is what catches this shape."""

    def test_truncated_file_with_backdated_mtime_is_rejected(self, tmp_path):
        f = tmp_path / "truncated.gguf"
        # weight.0 declares 4096 bytes, so weight.1 (never reached) starts at
        # offset 4096 - the file below has only a few hundred bytes of body
        # after the header, nowhere near even weight.0's own declared size.
        header = _gguf_header([("weight.0", 4096), ("weight.1", 1)])
        f.write_bytes(header.ljust(_GGUF_MIN_BYTES + 4, b"\0"))
        old = time.time() - (_GGUF_SETTLE_SECONDS + 5)
        os.utime(f, (old, old))

        # The magic+floor guard and the settle guard both read this as fine.
        assert f.stat().st_size >= _GGUF_MIN_BYTES
        assert _gguf_recently_written(f) is False

        declared_min = _gguf_declared_min_size(f)
        assert declared_min is not None and declared_min > f.stat().st_size
        assert _has_gguf_magic(f) is False

    def test_complete_small_file_is_still_accepted(self, tmp_path):
        # A genuinely complete file - actual size covers every declared tensor
        # offset - must not be rejected just because it happens to be small.
        f = tmp_path / "complete.gguf"
        header = _gguf_header([("weight.0", 2000)])
        data = b"\xAB" * 2000
        f.write_bytes(header + data)
        assert len(header) + len(data) >= _GGUF_MIN_BYTES
        assert _has_gguf_magic(f) is True

    def test_unparseable_header_is_not_treated_as_truncated(self, tmp_path):
        # The other fixtures in this file (magic + zero padding) have no real
        # tensor-info section at all - version reads as 0, which must read as
        # "no signal", not "truncated", so those files still register.
        f = tmp_path / "no-real-header.gguf"
        f.write_bytes(b"GGUF".ljust(_GGUF_MIN_BYTES, b"\x00"))
        assert _gguf_declared_min_size(f) is None
        assert _has_gguf_magic(f) is True

    def test_hostile_kv_array_count_does_not_crash(self, tmp_path):
        # A KV value of type ARRAY carries its own element count, unbounded
        # by anything else in the format. An enormous one makes
        # _gguf_skip_value_stream ask to seek past what Python's file API can
        # address at all (f.seek raises ValueError, not OSError, for an
        # offset outside a Py_ssize_t) - this must read as an ordinary parse
        # failure (None), not escape and crash the caller.
        f = tmp_path / "hostile.gguf"
        header = b"".join([
            b"GGUF", struct.pack("<I", 3), struct.pack("<QQ", 0, 1),
            _s("evil"), struct.pack("<I", 9),        # ARRAY
            struct.pack("<I", 4),                     # element type: uint32
            struct.pack("<Q", 2 ** 62),                # declared count
        ])
        f.write_bytes(header.ljust(_GGUF_MIN_BYTES, b"\0"))
        assert _gguf_declared_min_size(f) is None
        assert _has_gguf_magic(f) is True
