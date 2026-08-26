# SPDX-License-Identifier: AGPL-3.0-or-later
"""The CLI REPL perf readout (`localm run` / chat): tok/s is the DECODE rate, never
the load-folded rate. Regression lock for the "0.6 tok/s cold on a 64 tok/s GPU"
bug, where the first call after a model load charged the load seconds against the
generation throughput and looked like a silent CPU fallback.

_perf_line is a pure function of (n_tokens, t0, first_at, end), so these assert the
exact reported string without touching real timing.
"""

from __future__ import annotations

from localm.cli.chat import _perf_line


def test_rate_is_over_gen_window_not_total():
    # 5.0s load, 0.5s decode, 50 tokens. Decode rate = 50 / 0.5 = 100 tok/s.
    line = _perf_line(50, t0=0.0, first_at=5.0, end=5.5)
    assert "100.0 tok/s" in line
    assert "load 5.0s" in line
    assert "gen 0.5s" in line


def test_warm_call_uses_single_time_form():
    # A warm call's tiny prefill (< 0.1s) is noise, so keep the familiar single
    # time, not a load/gen split. 40 tokens / 1.0s decode = 40 tok/s.
    line = _perf_line(40, t0=0.0, first_at=0.02, end=1.02)
    assert "40.0 tok/s" in line
    assert "load" not in line
    assert "(1.0s)" in line


def test_single_token_omits_rate_but_shows_load():
    # One token has no decode interval, so tok/s is omitted; the load time is
    # still shown, so a cold start is visible.
    line = _perf_line(1, t0=0.0, first_at=5.0, end=5.5)
    assert "tok/s" not in line
    assert "1 tokens" in line
    assert "load 5.0s" in line


def test_none_when_total_too_quick():
    # Total <= 0.5s: nothing worth showing.
    assert _perf_line(10, t0=0.0, first_at=0.1, end=0.4) is None


def test_none_when_nothing_generated():
    assert _perf_line(0, t0=0.0, first_at=None, end=5.0) is None


def test_burst_arrival_omits_rate_but_keeps_load():
    # A 7.4s delayed first token followed by the rest arriving in a
    # near-instantaneous burst implies ~54,700 tok/s for 19 tokens, which is
    # impossible for single-stream decode. The plausibility floor omits the rate
    # while still reporting the real load/TTFT.
    line = _perf_line(19, t0=0.0, first_at=7.4, end=7.4 + 19 / 54786.62)
    assert "tok/s" not in line, line
    assert "19 tokens" in line, line
    assert "load 7.4s" in line, line
