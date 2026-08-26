# SPDX-License-Identifier: AGPL-3.0-or-later
"""Once HttpEngine.chat_stream re-wraps a thinking model's reasoning as inline
<think>...</think> (localm run's default attach mode), cli/chat.py neither
resends nor logs that raw scratchpad as the visible answer: it runs
textnorm.strip_think first, as every INTERNAL consumer of model output does.
transcript.exchange is the one exception - it splits the raw text itself and
keeps the reasoning in a collapsed block."""

from unittest.mock import MagicMock

from localm.cli.chat import _interactive


def _engine(pieces_by_turn):
    """A mock engine whose chat_stream yields the next pieces list on each call
    (one entry per expected turn). `seen_messages` records a SNAPSHOT (shallow
    copy) of the `messages` argument at each call, since `messages` is mutated
    in place afterward - a raw call_args_list reference would show later state."""
    e = MagicMock()
    e.display_name = "test-model"
    e.count_tokens.return_value = 5
    e.context_capacity.return_value = None
    calls = iter(pieces_by_turn)
    e.seen_messages = []

    def fake_chat_stream(messages, **k):
        e.seen_messages.append(list(messages))
        return iter(next(calls))

    e.chat_stream.side_effect = fake_chat_stream
    return e


def _drive(engine, inputs, audit=None, transcript=None, monkeypatch=None):
    from localm.cli import chat as chat_mod
    it = iter(inputs)

    def fake_input(_prompt):
        v = next(it)
        if isinstance(v, BaseException):
            raise v
        return v

    monkeypatch.setattr(chat_mod.console, "input", fake_input)
    _interactive(engine, None, {}, audit=audit, transcript=transcript)


def test_interactive_strips_think_before_storing_history_and_audit(monkeypatch):
    eng = _engine([
        ["<think>", "secret reasoning", "</think>", "The answer."],
        ["ok"],
    ])
    audit = MagicMock()
    transcript = MagicMock()

    _drive(eng, ["hi", "again", KeyboardInterrupt()],
          audit=audit, transcript=transcript, monkeypatch=monkeypatch)

    # The SECOND chat_stream call's `messages` snapshot is exactly what the
    # first turn stored as conversation history - assert it is the STRIPPED
    # visible answer, not the raw <think> scratchpad.
    second_call_messages = eng.seen_messages[1]
    assistant_turns = [m for m in second_call_messages if m.get("role") == "assistant"]
    assert len(assistant_turns) == 1
    stored = assistant_turns[0]["content"]
    assert stored == "The answer."
    assert "<think>" not in stored and "secret reasoning" not in stored

    # audit.llm got the stripped text too (first call = the reasoning turn).
    logged = audit.llm.call_args_list[0].args[0]
    assert logged == "The answer."
    assert "secret reasoning" not in logged

    # transcript still gets the RAW text (it splits reasoning out itself).
    _, raw_assistant = transcript.exchange.call_args_list[0].args
    assert "<think>" in raw_assistant and "secret reasoning" in raw_assistant


def test_interactive_no_reasoning_is_unaffected(monkeypatch):
    """NEGATIVE: an ordinary (non-thinking) reply is stored/logged unchanged."""
    eng = _engine([["Just a plain answer."]])
    audit = MagicMock()
    transcript = MagicMock()

    _drive(eng, ["hi", KeyboardInterrupt()],
          audit=audit, transcript=transcript, monkeypatch=monkeypatch)

    audit.llm.assert_called_once_with("Just a plain answer.")
    _, raw_assistant = transcript.exchange.call_args.args
    assert raw_assistant == "Just a plain answer."
