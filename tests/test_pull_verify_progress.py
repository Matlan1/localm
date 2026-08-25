# SPDX-License-Identifier: AGPL-3.0-or-later
"""ADR-0009 P9 + P10: the pull path must say which stage it is in."""

import hashlib
import json
from unittest.mock import MagicMock

import pytest

from localm import model_manager as mm
from localm.model_manager import _shared


def _events(capsys):
    """Every progress payload emitted so far, in order."""
    out = capsys.readouterr().out
    return [json.loads(line.split(mm.PROGRESS_SENTINEL, 1)[1])
            for line in out.splitlines() if mm.PROGRESS_SENTINEL in line]


@pytest.fixture()
def gui(monkeypatch):
    monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")


@pytest.fixture()
def many_blocks(monkeypatch):
    """Make a small file span many hash blocks, so throttling is observable."""
    monkeypatch.setattr(mm, "_HASH_BLOCK_BYTES", 64, raising=False)


@pytest.fixture()
def big_file(tmp_path):
    p = tmp_path / "model.gguf"
    p.write_bytes(b"x" * 4096)          # 64 blocks once many_blocks applies
    return p


# ------------------------------------------------- the stage is now nameable

class TestThePhaseFieldDistinguishesTheStages:
    def test_verification_reports_the_verify_phase(self, gui, big_file, capsys):
        mm._verify_digest(big_file)
        evs = _events(capsys)
        assert evs, "hashing a file in GUI mode emitted nothing at all"
        assert {e["phase"] for e in evs} == {"verify"}, (
            f"verification did not identify its own stage: {evs[:3]}")

    def test_the_download_stage_still_says_download(self, gui, capsys):
        """The other half of P9: the two stages must be TELLABLE APART."""
        with mm._snapshot_progress(lambda: 50, 100) as outcome:
            outcome.ok()
        phases = {e["phase"] for e in _events(capsys)}
        assert phases == {"download"}, f"the download stage was mislabelled: {phases}"

    def test_the_digest_is_unchanged(self, gui, big_file, capsys):
        """Reporting must not alter the answer."""
        expected = hashlib.sha256(big_file.read_bytes()).hexdigest()
        assert mm._verify_digest(big_file) == expected
        _events(capsys)                          # drain, keeps output off stdout


# --------------------------------------------------- honest numbers, per ADR-0008

class TestTheNumbersAreHonest:
    def test_it_reaches_the_end_of_the_file(self, gui, many_blocks, big_file,
                                            capsys):
        mm._verify_digest(big_file)
        evs = _events(capsys)
        assert evs[-1]["downloaded"] == 4096, (
            f"the last verify event stopped short of the file: {evs[-1]}")
        assert evs[-1]["pct"] == 100.0

    def test_an_unsizeable_file_reports_null_never_zero(self, gui, many_blocks,
                                                        big_file, monkeypatch,
                                                        capsys):
        """`_sha256_file` passes total=0 when it cannot stat the file."""
        real_stat = type(big_file).stat

        def _boom(self, *a, **kw):
            if self == big_file:
                raise OSError("no size for you")
            return real_stat(self, *a, **kw)

        monkeypatch.setattr(type(big_file), "stat", _boom)
        mm._verify_digest(big_file)
        evs = _events(capsys)
        assert evs, "an unsizeable file reported nothing at all"
        assert all(e["pct"] is None for e in evs), (
            f"fabricated a percentage with no denominator: {evs[:3]}")
        assert evs[-1]["downloaded"] == 4096, "dropped an honest byte count too"

    def test_progress_is_monotonic(self, gui, many_blocks, big_file, capsys):
        mm._verify_digest(big_file)
        counts = [e["downloaded"] for e in _events(capsys)]
        # Non-emptiness first: `[] == sorted([])` is true, so without this the
        # assertion below is satisfied by emitting nothing at all. Measured -
        # this test stayed green under the fires-control that removed verify
        # reporting entirely, which is the whole "could not have failed" class.
        assert counts, "emitted no verify progress to be monotonic about"
        assert counts == sorted(counts), f"verify progress went backwards: {counts}"


# ------------------------------------------- the emit rate is ours, not the hasher's

class TestTheEmitRateIsThrottled:
    def test_it_does_not_emit_once_per_block(self, gui, many_blocks, big_file,
                                             capsys):
        """`_sha256_file` calls back after EVERY block, at a rate set by disk and CPU throughput."""
        mm._verify_digest(big_file)              # 4096 / 64 = 64 blocks
        evs = _events(capsys)
        # `0 < 64` too, so emitting nothing would "pass" a bare upper bound.
        assert evs, "emitted nothing at all, which is not what throttling means"
        assert len(evs) < 64, (
            f"emitted {len(evs)} events for 64 blocks: the throttle is not working")

    def test_the_final_event_survives_the_throttle(self, gui, many_blocks,
                                                   big_file, monkeypatch, capsys):
        """The throttle must never swallow the LAST event: that is the one that says the wait is over."""
        # Patched on the module that DEFINES _verify_digest. It moved from
        # `pull` to `_shared` so `registry` could reach it without an import
        # cycle, and `_pull.time` stopped being the clock it reads - the test
        # went red on exactly that, which is what a patch surface pinned to the
        # wrong module looks like when it is caught rather than silently inert.
        monkeypatch.setattr(_shared.time, "monotonic", lambda: 10_000.0)
        mm._verify_digest(big_file)
        evs = _events(capsys)
        assert evs, "with time frozen, the final event was throttled away"
        assert evs[-1]["downloaded"] == 4096 and evs[-1]["pct"] == 100.0


# ------------------------------------------------ the CLI is a surface too

class TestTheCliSurface:
    def test_no_sentinel_is_printed_outside_gui_mode(self, monkeypatch, big_file,
                                                     capsys):
        """The sentinel is a control-character framing meant for a parent process."""
        monkeypatch.delenv("LOCALM_PROGRESS_JSON", raising=False)
        monkeypatch.setattr(mm, "_hash_with_progress",
                            MagicMock(return_value="deadbeef"))
        mm._verify_digest(big_file)
        assert _events(capsys) == [], "leaked GUI progress framing into the CLI"

    def test_the_cli_bar_is_told_what_it_is_waiting_for(self, monkeypatch,
                                                        big_file):
        """Asserted from OUTSIDE the call (catalogue item 13): a side_effect that raises would be an input to the code under test, not an assertion."""
        spy = MagicMock(return_value="deadbeef")
        monkeypatch.delenv("LOCALM_PROGRESS_JSON", raising=False)
        monkeypatch.setattr(mm, "_hash_with_progress", spy)
        assert mm._verify_digest(big_file, purpose="to verify the download") == "deadbeef"
        spy.assert_called_once()
        assert spy.call_args.kwargs["purpose"] == "to verify the download"

    def test_a_none_from_the_bar_falls_back_rather_than_propagating(
            self, monkeypatch, big_file):
        """_hash_with_progress returns None for a directory."""
        monkeypatch.delenv("LOCALM_PROGRESS_JSON", raising=False)
        monkeypatch.setattr(mm, "_hash_with_progress", MagicMock(return_value=None))
        expected = hashlib.sha256(big_file.read_bytes()).hexdigest()
        assert mm._verify_digest(big_file) == expected
