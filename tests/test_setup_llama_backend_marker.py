# SPDX-License-Identifier: AGPL-3.0-or-later
"""The .localm-backend marker: an OPTIONAL build tag, added without a migration.

`setup-llama` could not say what a re-provision was replacing, because the
marker recorded a backend name and never a version - so a real b1288 -> b1307
upgrade announced itself as a bare "Re-downloading", which reads as a no-op.

The obvious fix breaks the install. `_provisioned_backend` did a bare `.strip()`
of the whole file, so writing "amd-rocm b1307" would return a string that equals
no backend name, the provision guard's `have == want` would never hold, and
EVERY invocation would re-provision - the destructive-then-restorative path the
runtime-in-use refusal exists to stop. A silent infinite re-provision is a far
worse bug than the message it fixes.

So the marker is parsed by TOKEN: `_provisioned_backend` returns the first,
`_provisioned_build` the second when present. Both formats answer the guard
identically, which is what makes this backward AND forward compatible by
construction rather than by a version check - the right property for a file
written by releases that cannot be revised afterwards.

The old-format tests below are the ones that matter: every install in the field
today has a one-token marker.
"""

from __future__ import annotations

from localm.setup_llama import (
    _BACKEND_MARKER,
    _provisioned_backend,
    _provisioned_build,
    _record_provisioned_backend,
)


def _write_raw(target, text):
    """A marker written by hand, byte for byte - NOT through the recorder, so an
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
    """None, not "" and not a guess. A marker predating the tag genuinely does
    not know what is installed, and the caller is required to say so."""
    t = _write_raw(tmp_path / "bin", "amd-rocm\n")
    assert _provisioned_build(t) is None


def test_old_format_still_satisfies_the_provision_guard(tmp_path):
    """The actual regression risk, asserted as the guard states it: an install
    whose marker predates this change must still short-circuit rather than
    re-download on every run."""
    t = _write_raw(tmp_path / "bin", "vulkan\n")
    assert _provisioned_backend(t) == "vulkan"      # have
    assert _provisioned_backend(t) == "vulkan".lower()   # == want


# --------------------------------------------------------------------------- #
#  The new format resolves the SAME backend.                                  #
# --------------------------------------------------------------------------- #

def test_two_token_marker_resolves_the_identical_backend(tmp_path):
    """Same answer as the one-token marker, which is what makes the tag safe to
    add: had this returned "amd-rocm b1307", the guard would re-provision every
    single time."""
    old = _write_raw(tmp_path / "old", "amd-rocm\n")
    new = _write_raw(tmp_path / "new", "amd-rocm b1307\n")
    assert _provisioned_backend(new) == _provisioned_backend(old) == "amd-rocm"


def test_two_token_marker_exposes_the_build(tmp_path):
    t = _write_raw(tmp_path / "bin", "amd-rocm b1307\n")
    assert _provisioned_build(t) == "b1307"


def test_an_unexpected_third_token_does_not_break_the_backend(tmp_path):
    """Forward compatibility: a marker written by some LATER release that adds a
    field must still resolve its backend here, not wedge an older localm into
    re-provisioning forever."""
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
    """A backend whose tag costs a network call records one token, exactly as
    before - so this change adds no new failure mode for cuda/vulkan/cpu."""
    t = tmp_path / "bin"
    t.mkdir()
    _record_provisioned_backend(t, "vulkan")
    assert (t / _BACKEND_MARKER).read_text(encoding="utf-8").strip() == "vulkan"
    assert _provisioned_backend(t) == "vulkan"
    assert _provisioned_build(t) is None


def test_the_positional_two_arg_call_still_works(tmp_path):
    """Existing call sites pass (target, backend) positionally. The build
    parameter is keyword-optional so none of them had to change."""
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
    """Split on arbitrary whitespace, not a single space: a marker that picked
    up a tab or a double space must not silently become an unknown backend."""
    t = _write_raw(tmp_path / "bin", "  amd-rocm \t b1307  \n")
    assert _provisioned_backend(t) == "amd-rocm"
    assert _provisioned_build(t) == "b1307"
