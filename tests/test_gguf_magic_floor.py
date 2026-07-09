# SPDX-License-Identifier: AGPL-3.0-or-later
"""_has_gguf_magic enforces a minimum plausible file size.

A file that has the 4-byte GGUF magic but almost no body (a placeholder, a
half-written copy that got just the header) must be skipped by auto-registration
instead of passing the magic check and crashing a later model load with an
opaque ggml error."""

from localm.model_manager.gguf import _GGUF_MIN_BYTES, _has_gguf_magic


def test_bad_magic_rejected_even_when_large(tmp_path):
    f = tmp_path / "foreign.gguf"
    f.write_bytes(b"not a model".ljust(_GGUF_MIN_BYTES * 2, b"\x00"))
    assert _has_gguf_magic(f) is False


def test_magic_just_below_floor_rejected(tmp_path):
    # Covers the header-only-stub case too: any size < floor takes this same
    # `<` branch, and the floor-1 boundary is the strictly more precise check.
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
