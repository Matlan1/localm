# SPDX-License-Identifier: AGPL-3.0-or-later
"""cpu_backend_select: prune every CPU-tier backend .so but the one safe for
this machine, so ggml's directory-scan loaders (and localm's own RTLD_GLOBAL
preload) can never simultaneously map two tiers whose identically-named global
symbols would otherwise collide. See the module's own docstring for the full
mechanism.

Real .so files cannot be fabricated here, so `_probe_score` (the isolated
subprocess that calls a real candidate's own `ggml_backend_score()`) is
monkeypatched to canned per-candidate verdicts; everything downstream of that
- file layout, marker contents, locking, fast-path behaviour - is exercised for
real against real files under `tmp_path`.
"""

from __future__ import annotations

import json
import os
import shutil

import pytest

from localm import cpu_backend_select as cbs


def _touch(path, name: str):
    (path / name).write_bytes(b"")
    return path / name


@pytest.fixture
def lib_dir(tmp_path):
    d = tmp_path / "lib"
    d.mkdir()
    return d


# --------------------------- _cpu_fingerprint ------------------------------ #

def test_fingerprint_never_raises_and_is_a_string():
    fp = cbs._cpu_fingerprint()
    assert isinstance(fp, str)
    assert fp  # never empty - falls back to "unknown" rather than ""


# ------------------------------ candidates --------------------------------- #

def test_candidates_matches_only_cpu_tier_files(lib_dir):
    _touch(lib_dir, "libggml-cpu-haswell.so")
    _touch(lib_dir, "libggml-cpu-alderlake.so")
    _touch(lib_dir, "libggml-base.so.0")          # not a CPU tier
    _touch(lib_dir, "libggml-hip.so")              # not a CPU tier
    _touch(lib_dir, "_unused-libggml-cpu-zen4.so")  # already pruned
    found = {p.name for p in cbs._candidates(lib_dir)}
    assert found == {"libggml-cpu-haswell.so", "libggml-cpu-alderlake.so"}


# -------------------------------- marker ------------------------------------ #

def test_marker_roundtrip(lib_dir):
    _touch(lib_dir, "libggml-cpu-haswell.so")
    cbs._write_marker(lib_dir, "libggml-cpu-haswell.so")
    data = cbs._read_marker(lib_dir)
    assert data["tier"] == "libggml-cpu-haswell.so"
    assert data["fingerprint"] == cbs._cpu_fingerprint()
    assert data["schema"] == cbs._MARKER_SCHEMA


def test_no_marker_is_not_current(lib_dir):
    assert cbs._marker_is_current(lib_dir) is False


def test_marker_current_when_winner_present_and_no_siblings(lib_dir):
    _touch(lib_dir, "libggml-cpu-haswell.so")
    cbs._write_marker(lib_dir, "libggml-cpu-haswell.so")
    assert cbs._marker_is_current(lib_dir) is True


def test_marker_not_current_when_winner_file_missing(lib_dir):
    cbs._write_marker(lib_dir, "libggml-cpu-haswell.so")  # file never created
    assert cbs._marker_is_current(lib_dir) is False


def test_marker_not_current_when_fingerprint_stale(lib_dir, monkeypatch):
    _touch(lib_dir, "libggml-cpu-haswell.so")
    cbs._write_marker(lib_dir, "libggml-cpu-haswell.so")
    # Simulate the runtime directory having been copied to different hardware.
    monkeypatch.setattr(cbs, "_cpu_fingerprint", lambda: "a-different-machine")
    assert cbs._marker_is_current(lib_dir) is False


def test_marker_not_current_when_unpruned_sibling_remains(lib_dir):
    """A marker naming a winner is not enough on its own - if a sibling tier
    is STILL present un-pruned (e.g. an interrupted prior run), the collision
    hazard the marker claims to have closed is still on disk."""
    _touch(lib_dir, "libggml-cpu-haswell.so")
    _touch(lib_dir, "libggml-cpu-alderlake.so")  # not renamed - still a hazard
    cbs._write_marker(lib_dir, "libggml-cpu-haswell.so")
    assert cbs._marker_is_current(lib_dir) is False


# ------------------------- ensure_cpu_tier_selected ------------------------- #

def test_no_candidates_is_a_noop(lib_dir):
    assert cbs.ensure_cpu_tier_selected(lib_dir) is None
    assert list(lib_dir.iterdir()) == []


def test_single_candidate_selected_without_probing(lib_dir, monkeypatch):
    _touch(lib_dir, "libggml-cpu-haswell.so")
    called = []
    monkeypatch.setattr(cbs, "_probe_score",
                        lambda c, d: called.append(c) or 999)
    result = cbs.ensure_cpu_tier_selected(lib_dir)
    assert result == "libggml-cpu-haswell.so"
    assert called == []  # nothing to choose between - no subprocess spent
    assert (lib_dir / "libggml-cpu-haswell.so").exists()
    assert cbs._read_marker(lib_dir)["tier"] == "libggml-cpu-haswell.so"


def test_highest_scoring_candidate_wins_and_losers_are_renamed(lib_dir, monkeypatch):
    _touch(lib_dir, "libggml-cpu-haswell.so")
    _touch(lib_dir, "libggml-cpu-alderlake.so")
    _touch(lib_dir, "libggml-cpu-zen4.so")
    scores = {
        "libggml-cpu-haswell.so": 100,
        "libggml-cpu-alderlake.so": 0,     # rejected - AVX-512-only, unsupported
        "libggml-cpu-zen4.so": None,       # probe failed to load at all
    }
    monkeypatch.setattr(cbs, "_probe_score",
                        lambda c, d: scores[c.name])

    result = cbs.ensure_cpu_tier_selected(lib_dir)

    assert result == "libggml-cpu-haswell.so"
    names = {p.name for p in lib_dir.iterdir() if p.suffix == ".so"}
    assert "libggml-cpu-haswell.so" in names        # winner: untouched name
    assert "_unused-libggml-cpu-alderlake.so" in names
    assert "_unused-libggml-cpu-zen4.so" in names
    # After selection, ggml's own directory-scan pattern finds exactly ONE
    # CPU-tier candidate.
    assert [p.name for p in cbs._candidates(lib_dir)] == ["libggml-cpu-haswell.so"]


def test_all_candidates_unusable_leaves_directory_untouched(lib_dir, monkeypatch):
    _touch(lib_dir, "libggml-cpu-haswell.so")
    _touch(lib_dir, "libggml-cpu-alderlake.so")
    monkeypatch.setattr(cbs, "_probe_score", lambda c, d: 0)

    result = cbs.ensure_cpu_tier_selected(lib_dir)

    assert result is None
    names = {p.name for p in lib_dir.iterdir()}
    assert names == {"libggml-cpu-haswell.so", "libggml-cpu-alderlake.so"}
    assert cbs._read_marker(lib_dir) is None


def test_current_marker_short_circuits_without_probing(lib_dir, monkeypatch):
    _touch(lib_dir, "libggml-cpu-haswell.so")
    cbs._write_marker(lib_dir, "libggml-cpu-haswell.so")
    monkeypatch.setattr(cbs, "_probe_score",
                        lambda c, d: (_ for _ in ()).throw(
                            AssertionError("must not probe on the fast path")))
    result = cbs.ensure_cpu_tier_selected(lib_dir)
    assert result == "libggml-cpu-haswell.so"


def test_stale_fingerprint_triggers_reselection(lib_dir, monkeypatch):
    _touch(lib_dir, "libggml-cpu-haswell.so")
    _touch(lib_dir, "libggml-cpu-zen4.so")
    cbs._write_marker(lib_dir, "libggml-cpu-haswell.so")
    monkeypatch.setattr(cbs, "_cpu_fingerprint", lambda: "moved-to-new-hardware")
    monkeypatch.setattr(cbs, "_probe_score",
                        lambda c, d: {"libggml-cpu-haswell.so": 0,
                                       "libggml-cpu-zen4.so": 50}[c.name])
    result = cbs.ensure_cpu_tier_selected(lib_dir)
    assert result == "libggml-cpu-zen4.so"
    assert (lib_dir / "_unused-libggml-cpu-haswell.so").exists()


# ---------------------------------- lock ------------------------------------ #

def test_lock_is_reclaimed_from_a_dead_pid(lib_dir, monkeypatch):
    lock = lib_dir / cbs._LOCK_NAME
    lock.mkdir()
    # A PID essentially guaranteed not to be alive.
    (lock / cbs._LOCK_OWNER_FILE).write_text(json.dumps({"pid": 999999}),
                                              encoding="utf-8")
    with cbs._lock(lib_dir) as acquired:
        assert acquired is True
    assert not lock.exists()  # released on exit


def test_lock_short_circuits_when_marker_already_current_under_a_live_holder(
        lib_dir, monkeypatch):
    _touch(lib_dir, "libggml-cpu-haswell.so")
    cbs._write_marker(lib_dir, "libggml-cpu-haswell.so")
    lock = lib_dir / cbs._LOCK_NAME
    lock.mkdir()
    (lock / cbs._LOCK_OWNER_FILE).write_text(
        json.dumps({"pid": os.getpid()}), encoding="utf-8")  # genuinely alive
    monkeypatch.setattr(cbs, "_LOCK_POLL_SECONDS", 0.01)
    with cbs._lock(lib_dir) as acquired:
        assert acquired is False  # never took it - the live holder already finished
    shutil.rmtree(lock)  # clean up the lock this test created by hand
