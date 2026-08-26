# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recall precision.

- Absolute relevance gate: recall injects NOTHING when no stored memory relates
  to the query, rather than always surfacing the top-k. A memory is eligible
  only on a lexical content-word hit OR an absolute cosine match, mirroring the
  coder episode gate.
- Injected facts render in full (up to the block budget), not hard-sliced at
  150 chars mid-word; an over-cap line truncates at a word boundary with an
  ellipsis.
- The recall query windows the recent user turns, so an anaphoric follow-up
  ("yes, do that") still carries the prior turn's topic.
"""

from __future__ import annotations

import pytest

import localm.memory as M
from localm.memory import MemoryRecord, MemoryStore
from localm.memory.store import REL_COS_MIN, TINY_CORPUS

NOW = 1_800_000_000.0
DAY = 86400.0


def _measure_embed(texts):
    """3-axis one-hot topic stub (measurement / editor / other), so distinct topics
    are ORTHOGONAL - a paraphrased measurement query cosine-matches a metric fact
    (no shared token) but an editor query and generic fillers do NOT (each on its
    own axis). A coarser 2-class stub would make every non-measurement text
    cosine-identical."""
    meas = ("metric", "unit", "units", "measure", "measurement", "measurements",
            "system")
    editor = ("vim", "editor", "keybinding", "keybindings", "tutorial")
    out = []
    for t in texts:
        lo = t.lower()
        if any(w in lo for w in meas):
            out.append([1.0, 0.0, 0.0])
        elif any(w in lo for w in editor):
            out.append([0.0, 1.0, 0.0])
        else:
            out.append([0.0, 0.0, 1.0])
    return out


# ----------------------------------------------------- absolute relevance gate #

def test_offtopic_query_injects_nothing(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    for t in ("User prefers metric units", "User uses Vim as an editor",
              "User lives in Ghent"):
        s.add(MemoryRecord(text=t, source="user"))
    # No shared content word, no embedder -> recall stays SILENT.
    assert s.recall("what is the capital of France", k=6, now=NOW) == []


def test_ontopic_query_recalls_only_the_matching_fact(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User prefers metric units", source="user"))
    s.add(MemoryRecord(text="User uses Vim as an editor", source="user"))
    hits = s.recall("which editor do I use", k=6, now=NOW)
    assert [h.text for h in hits] == ["User uses Vim as an editor"]


def test_user_fact_is_gated_not_always_injected(tmp_path):
    # A high-importance USER fact unrelated to the query is not injected.
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User was born in 1990", source="user", importance=0.9))
    assert s.recall("recommend a pasta recipe", k=6, now=NOW) == []


def test_relevant_user_fact_is_recalled(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="User was born in 1990", source="user", importance=0.9))
    hits = s.recall("what year was I born", k=6, now=NOW)
    assert hits and hits[0].text == "User was born in 1990"


def test_semantic_gate_recalls_paraphrase_above_cosine_floor(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    for i in range(TINY_CORPUS + 1):
        s.add(MemoryRecord(text=f"random filler note {i}", source="synth"),
              embed_fn=_measure_embed)
    s.add(MemoryRecord(text="User prefers metric for all measurements",
                       source="user"), embed_fn=_measure_embed)
    # No shared content word with the metric fact ("measurement"!="measurements"),
    # so only the semantic (cosine) gate can surface it.
    hits = s.recall("which measurement do I like", k=3, embed_fn=_measure_embed,
                    now=NOW)
    assert hits and "metric" in hits[0].text


def test_semantic_gate_silent_below_cosine_floor(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    for i in range(TINY_CORPUS + 1):
        s.add(MemoryRecord(text=f"random filler note {i}", source="synth"),
              embed_fn=_measure_embed)
    s.add(MemoryRecord(text="User prefers metric units", source="user"),
          embed_fn=_measure_embed)
    # A query orthogonal to every record's topic and sharing no content word.
    assert s.recall("vim keybindings tutorial", k=3, embed_fn=_measure_embed,
                    now=NOW) == []


def test_cos_floor_constant_in_range():
    assert 0.4 <= REL_COS_MIN <= 0.6


# ------------------------------------------------------------ injection render #

def test_long_fact_renders_in_full_not_sliced_at_150():
    long_text = (
        "User is a marine biologist specialising in deep-sea cephalopods and "
        "prefers detailed technical answers with inline citations, and enjoys "
        "sailing on weekends near the coast of Brittany in north-western France")
    assert len(long_text) > 150
    block = M.render_memories([MemoryRecord(text=long_text)])
    assert long_text in block                       # rendered whole


def test_over_cap_line_truncates_at_word_boundary_with_ellipsis():
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    block = M.render_memories([MemoryRecord(text=text)], line_chars=25)
    # Find the rendered fact line and its content after "- ".
    line = next(ln for ln in block.splitlines() if ln.startswith("- "))
    content = line[2:]
    assert content.endswith("...")
    head = content[:-3].rstrip()
    assert text.startswith(head)                    # a real prefix of the fact
    # Word boundary: the char in the original right after the prefix is a space,
    # so nothing was cut mid-word.
    assert text[len(head)] == " "


# ---------------------------------------------------------- windowed recall query #

def test_recall_query_windows_recent_user_turns():
    from localm.plugins.builtin.memory.plug import _recall_query
    messages = [
        {"role": "user", "content": "Tell me about the Vim editor"},
        {"role": "assistant", "content": "Vim is a modal text editor."},
        {"role": "user", "content": "yes, do that"},
    ]
    q = _recall_query(messages)
    assert "yes, do that" in q
    assert "Vim" in q                                # prior turn's topic is included


def test_recall_query_capped_and_recent_first():
    from localm.plugins.builtin.memory.plug import _recall_query
    messages = [{"role": "user", "content": "x" * 1000},
                {"role": "user", "content": "the latest question"}]
    q = _recall_query(messages, max_chars=100)
    assert len(q) <= 100
    assert "the latest question" in q               # newest turn is never dropped


def test_recall_query_empty_without_user_turn():
    from localm.plugins.builtin.memory.plug import _recall_query
    assert _recall_query([{"role": "assistant", "content": "hi"}]) == ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
