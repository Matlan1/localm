# SPDX-License-Identifier: AGPL-3.0-or-later
"""The pull path must say which stage it is in.

THE DEFECT IS NOT MERE ABSENCE, and the two halves ship together. The
download's own terminal event announces 100% and THEN the caller hashes the
file, which for a multi-GB model is minutes of total silence - so the channel
says "the download is finished" and keeps working. The only field that could
tell the stages apart, `phase`, was dead: `_emit_progress` defaults it to
"download" and every call site in `pull.py` took the default.

WHAT THE FIXTURES HAVE TO BE ABLE TO EXPRESS, since a test cannot fail on a case
its fixture cannot build:

* One 4 MiB block yields exactly ONE callback, which can never demonstrate
  throttling and can never distinguish "throttled" from "emitted every block".
  `_HASH_BLOCK_BYTES` is therefore shrunk through `_mm` (this module's own
  convention) so a small fixture file spans many blocks.
* A file whose size always stats can never produce the no-denominator case, so
  one test makes `stat` raise.
* A GUI-mode-only fixture can never catch the regression that matters most to a
  CLI user: sentinel control characters printed into an ordinary terminal. The
  env var is therefore left UNSET in its own test rather than assumed.
"""

import hashlib
import json
from unittest.mock import MagicMock

import pytest

from localm import model_manager as mm
from localm.model_manager import _shared


def _events(capsys):
    """Every progress payload emitted so far, in order. Call ONCE per test:
    capsys.readouterr() DRAINS the buffer, so a second call returns [] and any
    assertion against it passes trivially."""
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


# ----------------------------------------------------- the stage is nameable

class TestThePhaseFieldDistinguishesTheStages:
    def test_verification_reports_the_verify_phase(self, gui, big_file, capsys):
        mm._verify_digest(big_file)
        evs = _events(capsys)
        assert evs, "hashing a file in GUI mode emitted nothing at all"
        assert {e["phase"] for e in evs} == {"verify"}, (
            f"verification did not identify its own stage: {evs[:3]}")

    def test_the_download_stage_still_says_download(self, gui, capsys):
        """The two stages must be TELLABLE APART. A test that only pinned
        'verify' would pass just as well if every stage were relabelled, which
        is the bug in a different costume."""
        with mm._snapshot_progress(lambda: 50, 100) as outcome:
            outcome.ok()
        phases = {e["phase"] for e in _events(capsys)}
        assert phases == {"download"}, f"the download stage was mislabelled: {phases}"

    def test_the_digest_is_unchanged(self, gui, big_file, capsys):
        """Reporting must not alter the answer. Guards the whole unit against
        being a security regression dressed as a UX one."""
        expected = hashlib.sha256(big_file.read_bytes()).hexdigest()
        assert mm._verify_digest(big_file) == expected
        _events(capsys)                          # drain, keeps output off stdout


# ------------------------------------------------------------- honest numbers

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
        """`_sha256_file` passes total=0 when it cannot stat the file. That is
        'we could not size this', and a percentage derived from it would be
        fabricated. The fixture has to MAKE stat fail: a file that always sizes
        can never reach this branch."""
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
        # assertion below is satisfied by emitting nothing at all.
        assert counts, "emitted no verify progress to be monotonic about"
        assert counts == sorted(counts), f"verify progress went backwards: {counts}"


# ------------------------------------------- the emit rate is ours, not the hasher's

class TestTheEmitRateIsThrottled:
    def test_it_does_not_emit_once_per_block(self, gui, many_blocks, big_file,
                                             capsys):
        """`_sha256_file` calls back after EVERY block, at a rate set by disk
        and CPU throughput. Forwarding that 1:1 puts hundreds of events a second
        on the GUI's stdout pipe for a large model."""
        mm._verify_digest(big_file)              # 4096 / 64 = 64 blocks
        evs = _events(capsys)
        # `0 < 64` too, so emitting nothing would "pass" a bare upper bound.
        assert evs, "emitted nothing at all, which is not what throttling means"
        assert len(evs) < 64, (
            f"emitted {len(evs)} events for 64 blocks: the throttle is not working")

    def test_the_final_event_survives_the_throttle(self, gui, many_blocks,
                                                   big_file, monkeypatch, capsys):
        """The throttle must never swallow the LAST event: that is the one that
        says the wait is over. Frozen time makes every event throttle-eligible,
        so only the explicit final-block exemption can let one through."""
        # Patched on the module that DEFINES _verify_digest: it lives in `_shared`,
        # not in `pull`, and `_pull.time` is not the clock it reads.
        monkeypatch.setattr(_shared.time, "monotonic", lambda: 10_000.0)
        mm._verify_digest(big_file)
        evs = _events(capsys)
        assert evs, "with time frozen, the final event was throttled away"
        assert evs[-1]["downloaded"] == 4096 and evs[-1]["pct"] == 100.0


# ------------------------------------------------ the CLI is a surface too

class TestTheCliSurface:
    def test_no_sentinel_is_printed_outside_gui_mode(self, monkeypatch, big_file,
                                                     capsys):
        """The sentinel is a control-character framing meant for a parent
        process. Printing it into an ordinary terminal is the regression this
        whole channel has to avoid, and only an env-UNSET fixture can catch it."""
        monkeypatch.delenv("LOCALM_PROGRESS_JSON", raising=False)
        monkeypatch.setattr(mm, "_hash_with_progress",
                            MagicMock(return_value="deadbeef"))
        mm._verify_digest(big_file)
        assert _events(capsys) == [], "leaked GUI progress framing into the CLI"

    def test_the_cli_bar_is_told_what_it_is_waiting_for(self, monkeypatch,
                                                        big_file):
        """Asserted from OUTSIDE the call: a side_effect that raises would be an
        input to the code under test, not an assertion.

        Patched on the PACKAGE, because _verify_digest reads
        _mm._hash_with_progress at call time rather than binding it at import."""
        spy = MagicMock(return_value="deadbeef")
        monkeypatch.delenv("LOCALM_PROGRESS_JSON", raising=False)
        monkeypatch.setattr(mm, "_hash_with_progress", spy)
        assert mm._verify_digest(big_file, purpose="to verify the download") == "deadbeef"
        spy.assert_called_once()
        assert spy.call_args.kwargs["purpose"] == "to verify the download"

    def test_a_none_from_the_bar_falls_back_rather_than_propagating(
            self, monkeypatch, big_file):
        """_hash_with_progress returns None for a directory. No caller here
        passes one, but returning None from a function typed `-> str` would turn
        a wrong answer into a confusing AttributeError at the .lower() call
        site, so the fallback is asserted rather than assumed."""
        monkeypatch.delenv("LOCALM_PROGRESS_JSON", raising=False)
        monkeypatch.setattr(mm, "_hash_with_progress", MagicMock(return_value=None))
        expected = hashlib.sha256(big_file.read_bytes()).hexdigest()
        assert mm._verify_digest(big_file) == expected
