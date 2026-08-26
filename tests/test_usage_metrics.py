# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for TTFT / throughput metrics in the usage field of HTTP responses.

tok/s must be measured over the DECODE window only, never over total wall time,
so a cold start's model-load time is never charged against the generation rate.
"""

import json
import os
import time

from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from localm.inference.http_server import (
    _decode_elapsed,
    _tokens_per_sec,
    _ttft_ms,
    create_app,
)


class TestMetricHelpers:
    def test_ttft_none_when_no_token(self):
        assert _ttft_ms(10.0, None) is None

    def test_ttft_milliseconds(self):
        assert _ttft_ms(10.0, 10.25) == 250.0

    def test_decode_elapsed_none_when_no_first_token(self):
        assert _decode_elapsed(None, 10.0) is None

    def test_decode_elapsed_is_end_minus_first_token(self):
        # The window starts at the FIRST token, so the pre-first-token load/prefill
        # span (gen_start .. first_token) is excluded by construction.
        assert _decode_elapsed(3.0, 10.0) == 7.0

    def test_tokens_per_sec(self):
        assert _tokens_per_sec(50, 2.0) == 25.0

    def test_tokens_per_sec_none_when_zero_tokens(self):
        assert _tokens_per_sec(0, 2.0) is None

    def test_tokens_per_sec_none_when_single_token(self):
        # One token has no decode interval to time; dividing by the near-zero
        # trailing window would report a meaningless huge rate, so report None.
        assert _tokens_per_sec(1, 2.0) is None

    def test_tokens_per_sec_none_when_zero_elapsed(self):
        assert _tokens_per_sec(50, 0.0) is None

    def test_tokens_per_sec_none_when_no_window(self):
        assert _tokens_per_sec(50, None) is None

    def test_tokens_per_sec_none_when_implausibly_fast(self):
        # Below the plausibility floor (see _MIN_SEC_PER_TOKEN): this implies
        # 50,000 tok/s, physically impossible for single-stream decode.
        assert _tokens_per_sec(19, 19 / 54786.62) is None
        assert _tokens_per_sec(29, 29 / 137701.81) is None

    def test_tokens_per_sec_accepts_realistic_rate_near_the_floor(self):
        # A real, plausible rate (19 tokens over a 0.172s decode window, ~110
        # tok/s) is NOT rejected by the plausibility floor.
        assert _tokens_per_sec(19, 0.172) is not None
        assert abs(_tokens_per_sec(19, 0.172) - 110.47) < 0.1


def _make_engine():
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.count_tokens.return_value = 5

    def _stream(messages, **kw):
        # A small, realistic gap between pieces: with no delay at all the decode
        # window is sub-millisecond Python overhead, which the plausibility floor
        # (see _MIN_SEC_PER_TOKEN) rejects for a claimed 5 tokens. This fixture
        # covers the ORDINARY case, not the burst-arrival edge case.
        yield "hello"
        time.sleep(0.01)
        yield " world"

    engine.chat_stream.side_effect = _stream
    type(engine).loaded = property(lambda self: True)
    return engine


def _slow_load_engine(load_delay=0.4, reported_tokens=20, pieces=6, decode_gap=0.005):
    """An engine whose first token is preceded by a big synthetic load delay, then
    emits its remaining pieces quickly. This is the cold-start shape: a long wait
    before the first token (model load + prefill), then fast decode. Used to prove
    the reported rate reflects the FAST decode, not the slow load.

    decode_gap=0.005: the 5 gaps between 6 pieces give a 25ms decode window for
    a claimed 20 tokens (12.5ms/token average), above the 1ms/token plausibility
    floor (see _MIN_SEC_PER_TOKEN) and fast relative to the 400ms load."""
    engine = MagicMock()
    engine.display_name = "slow-model"
    engine.count_tokens.return_value = reported_tokens

    def _stream(messages, **kw):
        time.sleep(load_delay)          # model load + prompt prefill, before token 1
        for i in range(pieces):
            if i:
                time.sleep(decode_gap)   # small inter-token gap (decode)
            yield f"t{i} "

    engine.chat_stream.side_effect = _stream
    type(engine).loaded = property(lambda self: True)
    return engine


def _burst_after_delay_engine(delay=0.3, pieces=20):
    """An engine with a delay before the first token (a contended first token,
    or a cold load), then the REST arriving in a near-instantaneous burst (no
    inter-token gap at all) - what a GPU scheduler produces when a delayed first
    request finally gets an uncontended run. completion_tokens / decode_elapsed
    would report tens of thousands of tok/s if not for the plausibility floor."""
    engine = MagicMock()
    engine.display_name = "burst-model"
    engine.count_tokens.return_value = pieces

    def _stream(messages, **kw):
        time.sleep(delay)
        for i in range(pieces):
            yield f"t{i} "   # no sleep at all between pieces - a true burst

    engine.chat_stream.side_effect = _stream
    type(engine).loaded = property(lambda self: True)
    return engine


def _single_token_engine():
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.count_tokens.return_value = 1          # a one-token reply ("ON")
    engine.chat_stream.side_effect = lambda messages, **kw: iter(["ON"])
    type(engine).loaded = property(lambda self: True)
    return engine


CHAT_PAYLOAD = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "hi"}],
}


def _final_usage(resp):
    """The usage object from the last usage-bearing SSE chunk (the 'done' chunk)."""
    usage = None
    for line in resp.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        data = json.loads(line[len("data: "):])
        if data.get("usage"):
            usage = data["usage"]
    return usage


class TestUsageMetricsInResponses:
    def setup_method(self):
        os.environ.pop("LOCALM_API_KEY", None)

    def test_non_streaming_chat_has_tokens_per_sec(self):
        with TestClient(create_app(_make_engine())) as client:
            r = client.post("/v1/chat/completions", json=CHAT_PAYLOAD)
        assert r.status_code == 200
        usage = r.json()["usage"]
        assert usage["tokens_per_sec"] is not None
        assert usage["tokens_per_sec"] > 0
        # TTFT IS observable even without streaming: the handler drives chat_stream
        # internally, so the first-token boundary is measured and reported.
        assert usage["ttft_ms"] is not None
        assert usage["ttft_ms"] >= 0

    def test_chat_reports_same_context_capacity_streaming_and_not(self):
        eng = _make_engine()
        # MUST be set explicitly: a bare MagicMock coerces to 1 through pydantic's
        # __int__ path.
        eng.context_capacity.return_value = 65536
        with TestClient(create_app(eng)) as client:
            ns = client.post("/v1/chat/completions", json=CHAT_PAYLOAD).json()["usage"]
            st = _final_usage(client.post(
                "/v1/chat/completions", json={**CHAT_PAYLOAD, "stream": True}))
        assert ns["context_capacity"] == 65536
        assert ns["context_capacity"] == st["context_capacity"]

    def test_streaming_chat_final_chunk_has_ttft(self):
        with TestClient(create_app(_make_engine())) as client:
            r = client.post(
                "/v1/chat/completions",
                json={**CHAT_PAYLOAD, "stream": True},
            )
        assert r.status_code == 200
        usage = _final_usage(r)
        assert usage is not None
        assert usage["ttft_ms"] is not None
        assert usage["ttft_ms"] >= 0
        assert usage["tokens_per_sec"] is not None

    def test_streaming_completions_final_chunk_has_metrics(self):
        with TestClient(create_app(_make_engine())) as client:
            r = client.post(
                "/v1/completions",
                json={"model": "test-model", "prompt": "hi", "stream": True},
            )
        assert r.status_code == 200
        usage = _final_usage(r)
        assert usage is not None
        assert usage["ttft_ms"] is not None
        assert usage["tokens_per_sec"] is not None

    # --- load must NOT be folded into the rate ------------------------------- #

    def _assert_rate_excludes_load(self, usage):
        # The load delay is captured as TTFT...
        assert usage["ttft_ms"] >= 300, usage
        # ...and, crucially, the decode-only rate exceeds tokens/TTFT. A load-folded
        # rate is tokens/(TTFT + decode), which is STRICTLY BELOW tokens/TTFT, so
        # this inequality holds only when the load was excluded from the rate.
        ttft_s = usage["ttft_ms"] / 1000.0
        assert usage["tokens_per_sec"] > usage["completion_tokens"] / ttft_s, usage

    def test_streaming_chat_rate_excludes_load(self):
        with TestClient(create_app(_slow_load_engine())) as client:
            r = client.post("/v1/chat/completions",
                            json={**CHAT_PAYLOAD, "stream": True})
        self._assert_rate_excludes_load(_final_usage(r))

    def test_streaming_completion_rate_excludes_load(self):
        with TestClient(create_app(_slow_load_engine())) as client:
            r = client.post("/v1/completions",
                            json={"model": "test-model", "prompt": "hi", "stream": True})
        self._assert_rate_excludes_load(_final_usage(r))

    def test_non_streaming_chat_rate_excludes_load(self):
        with TestClient(create_app(_slow_load_engine())) as client:
            r = client.post("/v1/chat/completions", json=CHAT_PAYLOAD)
        assert r.status_code == 200
        self._assert_rate_excludes_load(r.json()["usage"])

    # --- single-token replies omit the (unmeasurable) rate -------------------- #

    def test_streaming_single_token_omits_rate_keeps_ttft(self):
        with TestClient(create_app(_single_token_engine())) as client:
            r = client.post("/v1/chat/completions",
                            json={**CHAT_PAYLOAD, "stream": True})
        usage = _final_usage(r)
        assert usage["tokens_per_sec"] is None      # no decode interval to measure
        assert usage["ttft_ms"] is not None         # first-token time still reported

    def test_non_streaming_single_token_omits_rate(self):
        with TestClient(create_app(_single_token_engine())) as client:
            r = client.post("/v1/chat/completions", json=CHAT_PAYLOAD)
        usage = r.json()["usage"]
        assert usage["tokens_per_sec"] is None

    # --- a burst-arrival decode window omits the rate, never a false one ------- #
    #
    # first_token_at is a single sample, so a contended first token followed by an
    # uncontended burst collapses the decode window toward zero. A rate that
    # cannot physically be true (see _MIN_SEC_PER_TOKEN) is not reported at all.

    def test_streaming_chat_burst_arrival_omits_rate_keeps_ttft(self):
        with TestClient(create_app(_burst_after_delay_engine())) as client:
            r = client.post("/v1/chat/completions",
                            json={**CHAT_PAYLOAD, "stream": True})
        usage = _final_usage(r)
        assert usage["ttft_ms"] >= 250, usage    # the real delay is still reported
        assert usage["tokens_per_sec"] is None, usage

    def test_streaming_completion_burst_arrival_omits_rate(self):
        with TestClient(create_app(_burst_after_delay_engine())) as client:
            r = client.post("/v1/completions",
                            json={"model": "test-model", "prompt": "hi", "stream": True})
        usage = _final_usage(r)
        assert usage["ttft_ms"] >= 250, usage
        assert usage["tokens_per_sec"] is None, usage

    def test_non_streaming_chat_burst_arrival_omits_rate(self):
        with TestClient(create_app(_burst_after_delay_engine())) as client:
            r = client.post("/v1/chat/completions", json=CHAT_PAYLOAD)
        usage = r.json()["usage"]
        assert usage["ttft_ms"] >= 250, usage
        assert usage["tokens_per_sec"] is None, usage
