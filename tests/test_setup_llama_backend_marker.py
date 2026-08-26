# SPDX-License-Identifier: AGPL-3.0-or-later
"""The .localm-backend marker: a backend name plus an OPTIONAL build tag.

The marker is parsed by TOKEN: `_provisioned_backend` returns the first,
`_provisioned_build` the second when present. A one-token marker and a two-token
marker answer the provision guard identically, so both the old and the new shape
are compatible without a version check.

A whole-file `.strip()` would return a string equal to no backend name for a
two-token marker, the guard's `have == want` would never hold, and EVERY
invocation would re-provision.
"""

from __future__ import annotations

from localm.setup_llama import (
    _BACKEND_MARKER,
    _provisioned_backend,
    _provisioned_build,
    _record_provisioned_backend,
)


def _write_raw(target, text):
    """A marker written by hand, byte for byte, NOT through the recorder, so an
    old on-disk format is reproduced rather than re-derived from current code."""
    target.mkdir(parents=True, exist_ok=True)
    (target / _BACKEND_MARKER).write_text(text, encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
#  The old marker format.                                                     #
# --------------------------------------------------------------------------- #

def test_old_single_token_marker_still_names_its_backend(tmp_path):
    t = _write_raw(tmp_path / "bin", "amd-rocm\n")
    assert _provisioned_backend(t) == "amd-rocm"


def test_old_single_token_marker_reports_no_build(tmp_path):
    """None, not "" and not a guess: a one-token marker records no build."""
    t = _write_raw(tmp_path / "bin", "amd-rocm\n")
    assert _provisioned_build(t) is None


def test_old_format_still_satisfies_the_provision_guard(tmp_path):
    """A one-token marker still satisfies the guard's `have == want`, so such an
    install short-circuits rather than re-downloading on every run."""
    t = _write_raw(tmp_path / "bin", "vulkan\n")
    assert _provisioned_backend(t) == "vulkan"      # have
    assert _provisioned_backend(t) == "vulkan".lower()   # == want


# --------------------------------------------------------------------------- #
#  The new format resolves the SAME backend.                                  #
# --------------------------------------------------------------------------- #

def test_two_token_marker_resolves_the_identical_backend(tmp_path):
    """Same answer as the one-token marker. Returning "amd-rocm b1307" would
    make the guard re-provision every time."""
    old = _write_raw(tmp_path / "old", "amd-rocm\n")
    new = _write_raw(tmp_path / "new", "amd-rocm b1307\n")
    assert _provisioned_backend(new) == _provisioned_backend(old) == "amd-rocm"


def test_two_token_marker_exposes_the_build(tmp_path):
    t = _write_raw(tmp_path / "bin", "amd-rocm b1307\n")
    assert _provisioned_build(t) == "b1307"


def test_an_unexpected_third_token_does_not_break_the_backend(tmp_path):
    """A marker written by a LATER release that adds a field still resolves its
    backend here."""
    t = _write_raw(tmp_path / "bin", "amd-rocm b1307 gfx1030\n")
    assert _provisioned_backend(t) == "amd-rocm"
    assert _provisioned_build(t) == "b1307"


# --------------------------------------------------------------------------- #
#  Round trip through the real recorder.                                       #
# --------------------------------------------------------------------------- #

def test_recording_with_a_build_round_trips(tmp_path):
    t = tmp_path / "bin"
    t.mkdir()
    _record_provisioned_backend(t, "amd-rocm", build="b1307")
    assert _provisioned_backend(t) == "amd-rocm"
    assert _provisioned_build(t) == "b1307"


def test_recording_without_a_build_writes_the_old_shape(tmp_path):
    """A backend whose tag would cost a network call records one token only."""
    t = tmp_path / "bin"
    t.mkdir()
    _record_provisioned_backend(t, "vulkan")
    assert (t / _BACKEND_MARKER).read_text(encoding="utf-8").strip() == "vulkan"
    assert _provisioned_backend(t) == "vulkan"
    assert _provisioned_build(t) is None


def test_the_positional_two_arg_call_still_works(tmp_path):
    """Call sites pass (target, backend) positionally; the build parameter is
    keyword-optional."""
    t = tmp_path / "bin"
    t.mkdir()
    _record_provisioned_backend(t, "cpu")
    assert _provisioned_backend(t) == "cpu"


# --------------------------------------------------------------------------- #
#  Degenerate markers stay "unknown", never a bogus backend name.              #
# --------------------------------------------------------------------------- #

def test_missing_marker_is_unknown(tmp_path):
    t = tmp_path / "bin"
    t.mkdir()
    assert _provisioned_backend(t) is None
    assert _provisioned_build(t) is None


def test_empty_marker_is_unknown_not_empty_string(tmp_path):
    """An empty string would compare unequal to every backend AND read as
    'recorded', which is the worst of both."""
    t = _write_raw(tmp_path / "bin", "\n")
    assert _provisioned_backend(t) is None
    assert _provisioned_build(t) is None


def test_whitespace_only_marker_is_unknown(tmp_path):
    t = _write_raw(tmp_path / "bin", "   \n\t\n")
    assert _provisioned_backend(t) is None
    assert _provisioned_build(t) is None


def test_ragged_whitespace_between_tokens_is_tolerated(tmp_path):
    """Split on arbitrary whitespace, not a single space, so a marker carrying a
    tab or a double space still resolves."""
    t = _write_raw(tmp_path / "bin", "  amd-rocm \t b1307  \n")
    assert _provisioned_backend(t) == "amd-rocm"
    assert _provisioned_build(t) == "b1307"
