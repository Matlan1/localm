# SPDX-License-Identifier: AGPL-3.0-or-later
"""_has_gguf_magic enforces a minimum plausible file size, and
_gguf_recently_written enforces a settle period on top of it.

A file that has the 4-byte GGUF magic but almost no body (a placeholder, a
half-written copy that got just the header) must be skipped by auto-registration
instead of passing the magic check and crashing a later model load with an
opaque ggml error. A file that clears BOTH the magic and size checks can still
be an in-progress copy; _gguf_recently_written is the second gate for that case,
refusing to trust a file whose mtime is too fresh."""

import os
import time

from localm.model_manager.gguf import (
    _GGUF_MIN_BYTES,
    _GGUF_SETTLE_SECONDS,
    _gguf_recently_written,
    _has_gguf_magic,
)


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
