# SPDX-License-Identifier: AGPL-3.0-or-later
"""The chat-completion ``usage`` object is built by one helper, so the streamed
and non-streamed replies carry the same keys and values for the same engine."""

import ast
import json
import os
import time
from pathlib import Path

from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from localm.inference import http_server
from localm.inference.http_server import _build_usage, create_app

CHAT_PAYLOAD = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "hi"}],
}

MTP = {"state": "on", "drafted": 12, "accepted": 9, "paused_steps": 1}
SPEC = {"source": "ngram", "state": "on", "drafted": 7, "accepted": 5}

TIMING_KEYS = {"ttft_ms", "tokens_per_sec"}


def _engine():
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.count_tokens.return_value = 5
    engine.count_messages_tokens.return_value = 11
    engine.context_capacity.return_value = 65536
    engine.mtp_usage.return_value = dict(MTP)
    engine.speculation_usage.return_value = dict(SPEC)

    def _stream(messages, **kw):
        yield "hello"
        time.sleep(0.01)
        yield " world"

    engine.chat_stream.side_effect = _stream
    type(engine).loaded = property(lambda self: True)
    return engine


def _streamed_usage(resp):
    usage = None
    for line in resp.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        data = json.loads(line[len("data: "):])
        if data.get("usage"):
            usage = data["usage"]
    return usage


class TestStreamedNonStreamedParity:
    def setup_method(self):
        os.environ.pop("LOCALM_API_KEY", None)

    def test_same_engine_output_gives_the_same_usage(self):
        with TestClient(create_app(_engine())) as client:
            plain = client.post("/v1/chat/completions", json=CHAT_PAYLOAD)
            streamed = client.post("/v1/chat/completions",
                                   json={**CHAT_PAYLOAD, "stream": True})
        ns = plain.json()["usage"]
        st = _streamed_usage(streamed)
        assert st is not None
        assert set(ns) == set(st)
        stable = set(ns) - TIMING_KEYS
        assert {k: ns[k] for k in stable} == {k: st[k] for k in stable}
        assert ns["prompt_tokens"] == 11
        assert ns["completion_tokens"] == 5
        assert ns["total_tokens"] == 16
        assert ns["context_capacity"] == 65536
        assert ns["mtp"]["drafted"] == 12
        assert ns["speculation"]["source"] == "ngram"
        for usage in (ns, st):
            assert usage["ttft_ms"] is not None
            assert usage["tokens_per_sec"] is not None


class TestBuildUsage:
    def test_counts_are_summed(self):
        usage = _build_usage(7, 3)
        assert (usage.prompt_tokens, usage.completion_tokens,
                usage.total_tokens) == (7, 3, 10)

    def test_without_timing_or_engine_only_counts_and_capacity(self):
        usage = _build_usage(9, context_capacity=4096)
        assert usage.total_tokens == 9
        assert usage.context_capacity == 4096
        assert usage.ttft_ms is None
        assert usage.tokens_per_sec is None
        assert usage.mtp is None
        assert usage.speculation is None

    def test_decode_rate_excludes_the_time_before_the_first_token(self):
        usage = _build_usage(1, 50, gen_start=0.0, first_token_at=5.0,
                             gen_end=7.0)
        assert usage.ttft_ms == 5000.0
        assert usage.tokens_per_sec == 25.0

    def test_no_first_token_gives_no_timing(self):
        usage = _build_usage(1, 0, gen_start=0.0, first_token_at=None,
                             gen_end=3.0)
        assert usage.ttft_ms is None
        assert usage.tokens_per_sec is None

    def test_snapshot_outcome_wins_over_the_live_engine(self):
        engine = _engine()
        snap = {"mtp": None, "speculation": None}
        usage = _build_usage(1, 1, engine=engine, outcome=snap)
        assert usage.mtp is None
        assert usage.speculation is None

    def test_missing_snapshot_key_reads_the_engine(self):
        usage = _build_usage(1, 1, engine=_engine(), outcome={})
        assert usage.mtp.accepted == 9
        assert usage.speculation.accepted == 5


class TestSingleBuildSite:
    def test_usage_info_is_constructed_only_inside_build_usage(self):
        tree = ast.parse(Path(http_server.__file__).read_text(encoding="utf-8"))
        sites = []
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "UsageInfo"):
                    sites.append(fn.name)
        assert set(sites) == {"_build_usage"}, sites
