# SPDX-License-Identifier: AGPL-3.0-or-later
"""Before the first byte lands, "0%" is a claim, not a measurement.

Seeding the progress row from a real measurement makes the zero HONEST, but on a
fresh pull the measurement IS 0, so `_emit_progress` computes
`pct = round(0 * 100 / total, 1)` = 0.0 and the GUI renders a confident
"0% . 0 B / 1.04 GB" through DNS, TLS and the first HTTP round trip. A rendered
0% cannot be told from a stalled download.

With the total known and nothing on disk, TWO frames carry `pct: 0.0` in both
context managers - the opening seed AND the first poll tick - so fixing the seed
alone does not close it: both poll loops start at `last = -1`, so the first
`dl == 0` reading differs from `last` and emits.

WHAT THE FIXTURES MUST BE ABLE TO EXPRESS:

* A fixture with bytes already on disk can never produce the fresh case, and a
  fixture with none can never prove the resume percentage survives. Both are
  here, because the fix has to remove one while keeping the other - an
  unconditional `pct: None` seed would pass half of this file and throw away a
  true resume percentage.
* A fixture that always succeeds can never reach the terminal-exactness rule: a
  FAILED pull that landed nothing genuinely IS at zero, and reporting "unknown"
  there is the mirror error. One case below never calls `.ok()`.
* Only exercising `_snapshot_progress` can never catch `_download_progress`.
  Both are driven.
"""

import json

import pytest

from localm import model_manager as mm


def _events(capsys):
    """Progress payloads emitted so far. Call ONCE: readouterr() drains."""
    out = capsys.readouterr().out
    return [json.loads(line.split(mm.PROGRESS_SENTINEL, 1)[1])
            for line in out.splitlines() if mm.PROGRESS_SENTINEL in line]


@pytest.fixture()
def gui(monkeypatch):
    monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")


def _parts(tmp_path, n=1):
    return [tmp_path / f"m{i}.gguf" for i in range(n)]


class TestAFreshPullNeverClaimsZeroPercent:
    def test_snapshot_emits_no_percentage_before_the_first_byte(self, gui, capsys):
        with mm._snapshot_progress(lambda: 0, 1_000_000) as outcome:
            outcome.ok()
        inflight = [e for e in _events(capsys) if e["downloaded"] == 0]
        assert inflight, "the fixture produced no pre-first-byte event to judge"
        assert all(e["pct"] is None for e in inflight), (
            f"claimed a percentage before any byte landed: {inflight}")

    def test_download_emits_no_percentage_before_the_first_byte(
            self, gui, tmp_path, capsys):
        """The sibling context manager. Same defect, different function - a test
        that only drove one of them would leave the other free to regress."""
        with mm._download_progress(_parts(tmp_path), 1_000_000,
                                   base_dir=tmp_path) as outcome:
            outcome.ok()
        inflight = [e for e in _events(capsys) if e["downloaded"] == 0]
        assert inflight, "the fixture produced no pre-first-byte event to judge"
        assert all(e["pct"] is None for e in inflight), (
            f"claimed a percentage before any byte landed: {inflight}")

    def test_the_byte_count_is_still_reported(self, gui, capsys):
        """Withholding the percentage must not withhold everything. The GUI's
        indeterminate branch renders "downloading... <bytes>", so the count is
        what the user actually has during that window."""
        with mm._snapshot_progress(lambda: 0, 1_000_000) as outcome:
            outcome.ok()
        first = _events(capsys)[0]
        assert first["downloaded"] == 0 and first["total"] == 1_000_000, (
            f"dropped the honest fields along with the percentage: {first}")


class TestARealMeasurementIsStillReported:
    def test_a_resume_reports_its_percentage_from_the_very_first_event(
            self, gui, capsys):
        """429304 of 4683073 is a TRUE 9.2% before any new byte moves, and an
        unconditional `pct: None` seed would throw it away - trading this defect
        for its mirror image."""
        with mm._snapshot_progress(lambda: 429_304, 4_683_073) as outcome:
            outcome.ok()
        first = _events(capsys)[0]
        assert first["pct"] == 9.2, (
            f"discarded a real resume measurement: {first}")

    def test_a_percentage_appears_as_soon_as_a_byte_lands(self, gui, capsys):
        """The suppression is scoped to zero, not to the seed. One byte is
        enough to make a percentage meaningful again."""
        seen = [0, 1, 500_000]
        with mm._snapshot_progress(lambda: seen.pop(0) if seen else 500_000,
                                   1_000_000) as outcome:
            outcome.ok()
        evs = _events(capsys)
        numeric = [e for e in evs if e["pct"] is not None]
        assert numeric, f"never recovered a percentage after bytes landed: {evs}"
        assert all(e["downloaded"] > 0 for e in numeric), (
            f"a percentage was attached to a zero reading: {numeric}")


class TestTheTerminalEventKeepsExactSemantics:
    def test_a_failed_pull_that_landed_nothing_reports_zero_not_unknown(
            self, gui, capsys):
        """The mirror error: when the run is OVER, zero bytes is a known fact and
        "unknown" would be the opposite lie, so the suppression is a parameter
        rather than a blanket rule in _emit_progress. `.ok()` is never called, so
        this is the failure path."""
        with mm._snapshot_progress(lambda: 0, 1_000_000):
            pass
        last = _events(capsys)[-1]
        assert last["downloaded"] == 0
        assert last["pct"] == 0.0, (
            f"a finished run reported its known zero as unknown: {last}")

    def test_a_successful_pull_still_reports_one_hundred(self, gui, capsys):
        with mm._snapshot_progress(lambda: 1_000_000, 1_000_000) as outcome:
            outcome.ok()
        assert _events(capsys)[-1]["pct"] == 100.0


class TestAnUnsizedDownloadIsUnaffected:
    def test_no_total_still_streams_an_indeterminate_count(self, gui, capsys):
        """total == 0 already produced `pct: null`; this pins that the new flag
        did not change it, and that such a download still reports its bytes."""
        with mm._snapshot_progress(lambda: 4096, 0) as outcome:
            outcome.ok()
        evs = _events(capsys)
        assert evs and all(e["pct"] is None for e in evs)
        assert any(e["downloaded"] == 4096 for e in evs), (
            f"an unsized download stopped reporting its byte count: {evs}")
