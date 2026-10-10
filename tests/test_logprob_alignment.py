# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mapping visible reply text back to the tokens that spelled it.

``ScrubMap`` must describe exactly what ``scrub_text`` does, ``ThinkSplitter``
and ``ToolCallStream`` must report the stream offsets of the text they emit,
``_ReplyRouter`` must compose them, and ``LogprobAligner`` must return each
token of a visible range once, in order, and refuse text the tokens do not
spell. Property cases run over random token splits of random replies built
from the markers each stage reacts to."""

from __future__ import annotations

import random

import pytest

from localm.inference.backends.llamacpp.llama import (
    _filtered_stream, _scrub_stream, _utf8_pieces)
from localm.inference.http_server import _ReplyRouter
from localm.inference.logprobs import (
    LogprobAligner, LogprobAlignmentError, SegMap, chat_logprobs_content,
    completion_logprobs)
from localm.inference.tool_calling import CLOSE_TAG, OPEN_TAG, ToolCallStream
from localm.textnorm import ScrubMap, ThinkSplitter, scrub_stream, scrub_text

ATOMS = ["<think>", "</think>", OPEN_TAG, CLOSE_TAG,
         '{"name": "f", "arguments": {}}', "STOP", "ab", " ", "\n", "x", "<", "{",
         "}", "[", "<|channel|>analysis<|message|>", "<|channel|>final<|message|>",
         "<|end|>", "<|start|>assistant", '<|"|>', "<unused3>", "[TOOL_CALLS]",
         "é", "€", "<reasoning>", "</reasoning>", "<|im_end|>"]


def random_text(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(ATOMS) for _ in range(n))


def split(rng: random.Random, data: bytes) -> list[bytes]:
    cuts = sorted(rng.sample(range(1, len(data)), min(len(data) - 1, rng.randint(0, 12))))
    edges = [0, *cuts, len(data)]
    return [data[a:b] for a, b in zip(edges, edges[1:], strict=False) if b > a]


# ------------------------------------------------------------------ ScrubMap


def rebuild(text: str, spans) -> str:
    return "".join(text[s0:s1] if rep is None else rep for s0, s1, _o0, _o1, rep in spans)


def test_scrub_map_covers_text_and_output_end_to_end():
    rng = random.Random(7)
    for _ in range(2000):
        text = random_text(rng, rng.randint(0, 10))
        spans = ScrubMap().spans(text)
        assert rebuild(text, spans) == scrub_text(text)
        assert [s[0] for s in spans] == [0, *[s[1] for s in spans[:-1]]][:len(spans)]
        assert not spans or spans[-1][1] == len(text)
        for s0, s1, o0, o1, rep in spans:
            assert (o1 - o0) == (s1 - s0 if rep is None else len(rep))


def test_a_growing_text_maps_its_settled_part_once_and_the_rest_like_one_pass():
    rng = random.Random(11)
    for _ in range(300):
        text = random_text(rng, rng.randint(5, 40))
        m = ScrubMap()
        for end in sorted(rng.sample(range(len(text) + 1), min(6, len(text) + 1))) + [len(text)]:
            spans = m.spans(text[:end])
            assert rebuild(text[:end], spans) == scrub_text(text[:end])
        assert m.spans(text) == ScrubMap().spans(text)


def test_a_marker_completed_by_later_text_is_mapped_as_a_marker():
    m = ScrubMap()
    head = "word " * 20 + "<|chan"
    assert rebuild(head, m.spans(head)) == head
    full = head + "nel|>final<|message|>Answer"
    spans = m.spans(full)
    assert rebuild(full, spans) == scrub_text(full) == "word " * 20 + "\n</think>\nAnswer"


# ------------------------------------------------------------------ stage offsets


def test_think_splitter_spans_are_the_slices_its_content_came_from():
    rng = random.Random(3)
    for _ in range(2000):
        text = random_text(rng, rng.randint(0, 12))
        ts = ThinkSplitter(exit_marker=rng.choice([None, "x"]))
        pieces = [text[i:i + 3] for i in range(0, len(text), 3)]
        content, spans = [], []
        for p in pieces:
            c, _r = ts.feed(p)
            content.append(c)
            spans += ts.spans
            assert "".join(text[a:b] for a, b in ts.spans) == c
        c, _r = ts.flush()
        content.append(c)
        spans += ts.spans
        assert "".join(text[a:b] for a, b in spans) == "".join(content)


def test_tool_call_stream_spans_are_the_slices_its_text_events_are():
    rng = random.Random(5)
    for _ in range(2000):
        text = random_text(rng, rng.randint(0, 12))
        stream = ToolCallStream({"f"})
        step = rng.choice([1, 2, 5, 100])
        for i in range(0, len(text) + 1, step):
            events = stream.feed(text[i:i + step]) if i < len(text) else stream.finish()
            texts = [v for k, v in events if k == "text"]
            assert [text[a:b] for a, b in stream.spans] == texts


def test_router_text_spans_spell_the_visible_text():
    rng = random.Random(1)
    for _ in range(3000):
        text = random_text(rng, rng.randint(0, 12))
        router = _ReplyRouter(rng.choice([None, ["STOP"], ["b <", "x"]]),
                              rng.choice([None, {"f"}]),
                              {"max_tool_calls": rng.choice([None, 1])})
        step = rng.choice([1, 2, 3, 7, 1000])
        visible = []
        for i in range(0, len(text), step):
            visible += [v for k, v in router.feed(text[i:i + step]) if k == "text"]
        visible += [v for k, v in router.flush() if k == "text"]
        assert "".join(text[a:b] for a, b in router.text_spans) == "".join(visible)
        assert [s for spans in router.event_spans for s in spans] == router.text_spans


def test_seg_map_merges_neighbours_and_maps_across_pieces():
    m = SegMap()
    m.add(10, 3)
    m.add(13, 2)
    m.add(40, 4)
    m.add(0, 0)
    assert m.total == 9
    assert m.map(0, 9) == [(10, 15), (40, 44)]
    assert m.map(4, 6) == [(14, 15), (40, 41)]
    assert m.map(6, 6) == []


# ------------------------------------------------------------------ the aligner


def received_text(token_bytes: list[bytes]) -> str:
    """What the server receives for *token_bytes*: the GGUF worker's decode chain
    then the engine's scrub, piece by piece."""
    return "".join(scrub_stream(_scrub_stream(_filtered_stream(_utf8_pieces(iter(token_bytes))))))


def records(token_bytes: list[bytes]) -> list:
    return [(b, -float(i), ((b, -float(i)),)) for i, b in enumerate(token_bytes)]


def owners(tokens: list[bytes], spans) -> list[int]:
    """The tokens whose bytes overlap a character inside *spans*, for a reply
    whose received text is its bytes decoded unchanged."""
    text = b"".join(tokens).decode("utf-8")
    char_at = [0]
    for ch in text:
        char_at.append(char_at[-1] + len(ch.encode("utf-8")))
    token_at = [0]
    for t in tokens:
        token_at.append(token_at[-1] + len(t))
    wanted = set()
    for a, b in spans:
        for j in range(a, b):
            lo, hi = char_at[j], char_at[j + 1]
            wanted.update(i for i in range(len(tokens))
                          if token_at[i] < hi and lo < token_at[i + 1])
    return sorted(wanted)


def test_the_tokens_reported_are_exactly_those_that_spelled_visible_text():
    rng = random.Random(9)
    checked = 0
    for _ in range(3000):
        raw = random_text(rng, rng.randint(1, 10)).encode("utf-8")
        tokens = split(rng, raw)
        got = received_text(tokens)
        router = _ReplyRouter(rng.choice([None, ["STOP"]]), rng.choice([None, {"f"}]), {})
        router.feed(got)
        router.flush()
        aligner = LogprobAligner()
        aligner.add(records(tokens))
        aligner.finish()
        aligner.received(got)
        entries = aligner.take(router.text_spans)
        indices = [-int(e["logprob"]) for e in entries]
        assert indices == sorted(set(indices))
        if got == raw.decode("utf-8"):
            assert indices == owners(tokens, router.text_spans)
            checked += 1
    assert checked > 300


def test_streaming_alignment_matches_whole_reply_alignment():
    rng = random.Random(21)
    for _ in range(800):
        raw = random_text(rng, rng.randint(1, 10)).encode("utf-8")
        tokens = split(rng, raw)
        recs = records(tokens)
        stops = rng.choice([None, ["STOP"]])
        tools = rng.choice([None, {"f"}])
        whole_text = received_text(tokens)
        router = _ReplyRouter(stops, tools, {})
        router.feed(whole_text)
        router.flush()
        aligner = LogprobAligner()
        aligner.add(recs)
        aligner.finish()
        aligner.received(whole_text)
        whole = aligner.take(router.text_spans)

        router = _ReplyRouter(stops, tools, {})
        aligner = LogprobAligner()
        sink: list = []

        def scored(recs=recs, sink=sink):
            for r in recs:
                sink.append(r)
                yield r[0]

        streamed: list = []
        seen = 0
        for piece in scrub_stream(_scrub_stream(_filtered_stream(_utf8_pieces(scored())))):
            aligner.add(sink)
            sink.clear()
            aligner.received(piece)
            router.feed(piece)
            streamed += aligner.take(router.text_spans[seen:])
            seen = len(router.text_spans)
        aligner.add(sink)
        aligner.finish()
        router.flush()
        streamed += aligner.take(router.text_spans[seen:])
        assert [e["bytes"] for e in streamed] == [e["bytes"] for e in whole]


def test_received_text_the_tokens_do_not_spell_is_an_error():
    aligner = LogprobAligner()
    aligner.add(records([b"Hello"]))
    aligner.received("Hullo")
    with pytest.raises(LogprobAlignmentError, match="does not match"):
        aligner.take([(0, 5)])
    aligner = LogprobAligner()
    aligner.add(records([b"Hi"]))
    aligner.received("Hi there")
    with pytest.raises(LogprobAlignmentError, match="past the text"):
        aligner.take([(0, 8)])


def test_a_token_is_returned_once_even_when_two_ranges_touch_it():
    aligner = LogprobAligner()
    aligner.add(records([b"abc", b"def"]))
    aligner.received("abcdef")
    assert [e["bytes"] for e in aligner.take([(0, 1)])] == [b"abc"]
    assert [e["bytes"] for e in aligner.take([(1, 4)])] == [b"def"]
    assert aligner.take([(4, 6)]) == []


def test_a_marker_rewritten_into_visible_text_reports_the_marker_tokens():
    tokens = [b"say ", b'<|"', b'|>', b"hi"]
    got = received_text(tokens)
    assert got == 'say "hi'
    aligner = LogprobAligner()
    aligner.add(records(tokens))
    aligner.finish()
    aligner.received(got)
    entries = aligner.take([(0, len(got))])
    assert [e["bytes"] for e in entries] == tokens
    assert [e["offset"] for e in entries] == [0, 4, 4, 5]


def test_response_shapes():
    entries = [{"bytes": b"Hi", "logprob": -0.5, "top": ((b"Hi", -0.5), (b"Yo", -1.0)),
                "offset": 0},
               {"bytes": b"\xe2\x82", "logprob": -0.1, "top": (), "offset": 2}]
    assert chat_logprobs_content(entries, 1) == [
        {"token": "Hi", "logprob": -0.5, "bytes": [72, 105],
         "top_logprobs": [{"token": "Hi", "logprob": -0.5, "bytes": [72, 105]}]},
        {"token": "�", "logprob": -0.1, "bytes": [226, 130], "top_logprobs": []}]
    assert completion_logprobs(entries, 2, base_offset=3) == {
        "tokens": ["Hi", "�"], "token_logprobs": [-0.5, -0.1],
        "top_logprobs": [{"Hi": -0.5, "Yo": -1.0}, {"�": -0.1}],
        "text_offset": [3, 5]}
