# SPDX-License-Identifier: AGPL-3.0-or-later
"""A model that emits NO tool call at all must be escalated until it makes one,
not silently accepted as finished.

Every gate in _handle_no_tool_calls is reached through
looks_like_tool_attempt(), so the MALFORMED case (the model nearly succeeding)
and the ZERO-attempt case (the total failure) need separate handling.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.inference.gbnf import TOOL_CALLS_AFTER_THINK, check_grammar_structure
from localm.plugins.coder.agent.constants import _MAX_NOCALL_ESCALATIONS
from localm.plugins.coder.agent.loop import implies_action, response_similarity
from tests.conftest import final_answer as _final_answer


def _make_agent(tmp_path: Path, **kwargs):
    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = "test-model"
    backend.native_tools = False
    backend.supports_grammar = True
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        return Agent(backend=backend, cwd=tmp_path, **kwargs)


def _nocall_prompts(agent) -> list:
    return [m for m in agent._messages
            if m["role"] == "user" and "[no tool call]" in str(m.get("content", ""))]


def _notice_kinds(agent) -> list:
    return [c.args[0] for c in agent._audit.notice.call_args_list]


# --------------------------------------------------------------------------- #
#  implies_action - the precondition                                           #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text", [
    "Create three files: a.txt, b.txt and c.txt",
    "create 257 files in this folder",              # scaled
    "fix the crash",
    "read config.py and tell me the port",           # a READ still needs a tool
    "show me what is in the settings module",
    "run the tests",
    "what does main.py do?",                         # no verb, but names a file
])
def test_action_requests_are_recognised(text):
    assert implies_action(text) is True


@pytest.mark.parametrize("text", [
    "what is a Python closure?",
    "explain the difference between a thread and a process",
    "why is recursion sometimes slower than iteration?",
    "",
])
def test_pure_questions_are_not_action_requests(text):
    assert implies_action(text) is False


# --------------------------------------------------------------------------- #
#  The ladder                                                                  #
# --------------------------------------------------------------------------- #

def test_zero_tool_call_on_an_action_request_is_escalated_not_accepted(tmp_path):
    """A refusal like "I can't create files." must not be accepted as the finished
    answer on turn 1: it is re-prompted, noticed and recorded in the audit."""
    agent = _make_agent(tmp_path, max_turns=8)
    responses = iter([
        "As an AI I don't have the capability to create files on your machine.",
        "Here is a shell script you could run yourself instead: touch a.txt",
        "I am unable to touch the filesystem, sorry about that.",
    ])
    with patch.object(agent, "_call_llm", side_effect=lambda *a, **k: next(responses)), \
         patch("localm.plugins.coder.agent.parse_tool_calls", return_value=[]):
        result = agent.run_task("create a.txt with the text hello")

    # It was asked again, with the format, instead of being taken at its word.
    assert len(_nocall_prompts(agent)) == _MAX_NOCALL_ESCALATIONS
    assert "<tool_call>" in str(_nocall_prompts(agent)[0]["content"])
    # And the failure to enforce is stated, not hidden.
    assert "tool use not achieved" in result
    assert _notice_kinds(agent).count("no_tool_call") >= 1


def test_escalation_is_bounded(tmp_path):
    """It must not re-ask forever: the ladder has a cap and then reports."""
    agent = _make_agent(tmp_path, max_turns=12)
    n = [0]

    def _reply(*a, **k):
        n[0] += 1
        return f"Refusal number {n[0]}: I cannot do that, and here is a different excuse."

    with patch.object(agent, "_call_llm", side_effect=_reply), \
         patch("localm.plugins.coder.agent.parse_tool_calls", return_value=[]):
        agent.run_task("create a.txt")
    assert len(_nocall_prompts(agent)) == _MAX_NOCALL_ESCALATIONS


def test_the_report_never_tells_the_user_to_change_model(tmp_path):
    """Binding requirement: the job is to make the USER'S chosen model call the
    tool. The last-resort message reports failed enforcement; it must never
    suggest picking a different model, which is the user's call to make."""
    agent = _make_agent(tmp_path, max_turns=8)
    with patch.object(agent, "_call_llm", side_effect=lambda *a, **k: f"no {id(object())}"), \
         patch("localm.plugins.coder.agent.parse_tool_calls", return_value=[]):
        result = agent.run_task("create a.txt")
    lowered = result.lower()
    for banned in ("another model", "different model", "switch model", "choose a model",
                   "try a different", "better model", "capable model"):
        assert banned not in lowered, f"the report suggests changing model: {result!r}"


def test_a_question_is_still_answered_without_escalation(tmp_path):
    """Control. A request that needs no tool must be unaffected - otherwise the
    ladder would nag on every conversational turn."""
    agent = _make_agent(tmp_path, max_turns=6)
    with patch.object(agent, "_call_llm", return_value="A closure captures its enclosing scope."), \
         patch("localm.plugins.coder.agent.parse_tool_calls", return_value=[]):
        result = agent.run_task("what is a Python closure?")
    assert _final_answer(result) == "A closure captures its enclosing scope."
    assert _nocall_prompts(agent) == []
    assert "tool use not achieved" not in result


def test_no_escalation_once_the_model_has_called_a_tool(tmp_path):
    """Control. A model that used a tool and then summarised in prose is
    finishing normally, not refusing."""
    agent = _make_agent(tmp_path, max_turns=6)
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    responses = iter([
        '<tool_call>\n{"name": "read_file", "args": {"path": "a.py"}}\n</tool_call>',
        "The file sets x to 1.",
    ])
    with patch.object(agent, "_call_llm", side_effect=lambda *a, **k: next(responses)):
        result = agent.run_task("read a.py and tell me what it does")
    assert _final_answer(result) == "The file sets x to 1."
    assert _nocall_prompts(agent) == []
    assert "tool use not achieved" not in result


# --------------------------------------------------------------------------- #
#  Rung 2: the grammar is actually bound from the first token                  #
# --------------------------------------------------------------------------- #

def test_second_rung_binds_the_forcing_grammar_from_the_first_token(tmp_path):
    """The lazy grammar cannot rescue this case - it engages only once the model
    starts a <tool_call>, i.e. it is gated on the very thing that is failing. The
    forcing rung must bind from token one: no trigger, not lazy."""
    agent = _make_agent(tmp_path)
    agent._force_tool_grammar = True
    kw = agent._llm_kwargs()
    assert kw["grammar"] == TOOL_CALLS_AFTER_THINK
    assert "grammar_triggers" not in kw
    assert not kw.get("grammar_lazy")

    agent._force_tool_grammar = False
    lazy = agent._llm_kwargs()
    assert lazy["grammar_lazy"] is True and lazy["grammar_triggers"]


def test_forcing_is_one_shot(tmp_path):
    """A constrained sampler must not leak into later turns: the whole session
    would then be unable to emit a plain answer."""
    agent = _make_agent(tmp_path, max_turns=8)
    seen = []

    def _reply(*a, **k):
        seen.append(bool(agent._llm_kwargs().get("grammar_lazy") is not True))
        return f"still refusing, variation {len(seen)}"

    with patch.object(agent, "_call_llm", side_effect=_reply), \
         patch("localm.plugins.coder.agent.parse_tool_calls", return_value=[]):
        agent.run_task("create a.txt")
    # Exactly one turn ran with forcing armed, and it was not the last.
    assert seen.count(True) == 1, seen
    assert agent._force_tool_grammar is False


def test_forcing_respects_an_explicit_config_optout(tmp_path):
    """coder_tool_grammar=False is an explicit user choice to leave sampling
    unconstrained. The rescue path must not quietly re-impose a grammar; it says
    forcing is unavailable instead."""
    agent = _make_agent(tmp_path)
    with patch("localm.config.load_config", return_value={"coder_tool_grammar": False}):
        assert agent.can_force_tool_calls() is False
        assert "grammar" not in agent._llm_kwargs()


def test_forcing_unavailable_is_reported_with_its_reason(tmp_path):
    agent = _make_agent(tmp_path, max_turns=8)
    agent.backend.supports_grammar = False
    with patch.object(agent, "_call_llm", side_effect=lambda *a, **k: f"no {len(agent._messages)}"), \
         patch("localm.plugins.coder.agent.parse_tool_calls", return_value=[]):
        result = agent.run_task("create a.txt")
    assert "tool use not achieved" in result
    assert "Constrained sampling" in result


# --------------------------------------------------------------------------- #
#  The forcing grammar itself                                                  #
# --------------------------------------------------------------------------- #

def test_forcing_grammar_allows_thinking_but_still_requires_a_call():
    """Bare TOOL_CALLS_ONLY lets a thinking model mask its <think> opener into
    the leading whitespace and return a WHITESPACE-ONLY reply. The forced variant
    must therefore permit a reasoning block - including the template-pre-opened
    dialect, where generation starts INSIDE the block and the first token is
    prose, not a tag - while still requiring a real tool call."""
    assert "<think>" in TOOL_CALLS_AFTER_THINK
    assert "</think>" in TOOL_CALLS_AFTER_THINK
    # tool-block+ is required, so no prelude alone can satisfy the root rule.
    assert "tool-block+" in TOOL_CALLS_AFTER_THINK


def test_forcing_grammar_requires_the_opening_think_tag():
    """The prelude's opening tag must NOT be optional, and this is the single
    property that decides whether this grammar forces anything at all.

    An optional opener turns the prelude into an ARBITRARY PROSE prelude - any
    leading text parses as "reasoning that has not closed yet" - so the grammar
    constrains nothing and the model can ramble to max_tokens without emitting a
    tool call.

    That regression is invisible to a shape test that only greps for "<think>",
    so this asserts the tag is mandatory rather than merely present."""
    assert '("<think>" think-body "</think>")?' in TOOL_CALLS_AFTER_THINK
    assert '"<think>"?' not in TOOL_CALLS_AFTER_THINK, (
        "an optional opening tag lets arbitrary prose satisfy the prelude, "
        "which defeats forcing entirely")


def test_forcing_grammar_prelude_is_bounded_below_the_native_ceiling():
    """An unbounded prelude keeps alive a parse in which the model may emit text
    forever and never reach the call, silently defeating the forcing. The bound
    must also stay under llama.cpp's own repetition ceiling of 1999 on the
    bundled runtime (2000 is rejected as "exceeds sane defaults"). localm's
    MAX_GRAMMAR_REPEAT_COUNT is aligned at 1900 with the same margin, so this
    also checks the prelude bound stays inside the structural pre-check's own
    ceiling, not just the native one."""
    import re
    bounds = [int(m) for m in re.findall(r"\{0,(\d+)\}", TOOL_CALLS_AFTER_THINK)]
    assert bounds, "the reasoning prelude must carry an explicit {0,N} bound"
    assert max(bounds) <= 1900, bounds
    check_grammar_structure(TOOL_CALLS_AFTER_THINK)      # raises if out of bounds


# --------------------------------------------------------------------------- #
#  The repeat breaker: similarity, and only on turns with no tool call         #
# --------------------------------------------------------------------------- #

def test_near_duplicate_refusals_trip_the_breaker(tmp_path):
    """Near-duplicate refusals (roughly 0.72 to 0.91 similar) must trip the
    breaker; an exact-fingerprint, consecutive-only breaker resets to zero on any
    difference and never fires.

    The turns carry a REPEATED, IDENTICAL tool call, because that is the only
    shape in which this can happen: a turn with no call at all ends the task in
    _handle_no_tool_calls, so a call-less refusal loop never reaches a fifth turn
    to be counted. The shape is a model re-issuing the same call and re-narrating
    around it, reworded just enough each time."""
    base = ('<tool_call>\n{"name": "read_file", "args": {"path": "a.py"}}\n</tool_call>\n'
            "Message {n}: I have re-read the file and will now wait for further "
            "instructions before doing anything else.")
    assert response_similarity(base.replace("{n}", "1"),
                               base.replace("{n}", "2")) > 0.85

    agent = _make_agent(tmp_path, max_turns=30)
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    n = [0]

    def _reply(*a, **k):
        n[0] += 1
        return base.replace("{n}", str(n[0]))

    with patch.object(agent, "_call_llm", side_effect=_reply), \
         patch.object(agent, "_execute_tools", side_effect=lambda *a, **k: ["<tool_result>ok"]):
        result = agent.run_task("read a.py")
    assert "circuit breaker" in result
    assert n[0] < 30, "the breaker never fired; the loop ran to max_turns"


def test_the_old_exact_breaker_would_not_have_fired_on_these(tmp_path):
    """Each successive refusal differs from the last, so a breaker built on exact
    string equality with a counter that resets to zero on any difference could
    never count past one."""
    base = ("Message {n}: I have re-read the file and will now wait for further "
            "instructions before doing anything else.")
    a, b = base.replace("{n}", "1"), base.replace("{n}", "2")
    assert a.strip() != b.strip()                      # exact match: no hit
    assert response_similarity(a, b) >= 0.85           # similarity: hit


def test_repetitive_tool_work_does_not_trip_the_breaker(tmp_path):
    """Legitimate tool use is repetitive BY NATURE - reading five files differs
    only in the path and scores ~0.97 similar. Scoring every response would abort
    a working session for doing exactly what it was asked, so only turns that
    produced NO call may count."""
    agent = _make_agent(tmp_path, max_turns=12)
    for i in range(6):
        (tmp_path / f"f{i}.py").write_text(f"x = {i}\n", encoding="utf-8")
    calls = [f'<tool_call>\n{{"name": "read_file", "args": {{"path": "f{i}.py"}}}}\n</tool_call>'
             for i in range(6)]
    assert response_similarity(calls[0], calls[1]) > 0.85      # they ARE near-identical
    responses = iter(calls + ["All six read."])
    with patch.object(agent, "_call_llm", side_effect=lambda *a, **k: next(responses)):
        result = agent.run_task("read every f*.py file")
    assert "circuit breaker" not in result
    assert _final_answer(result) == "All six read."


# --------------------------------------------------------------------------- #
#  The server can refuse a grammar outright - degrade, don't crash             #
# --------------------------------------------------------------------------- #

def test_grammar_unsupported_400_degrades_instead_of_crashing(tmp_path):
    """HTTPBackend advertises supports_grammar=True for ANY localm server (it
    cannot know which backend is actually loaded server-side - see http.py). A
    grammar request against an incapable backend is a hard 400
    (GrammarUnsupportedError), so the FIRST grammar-bearing turn against such a
    server must not crash the whole task with an unhandled CoderServerError. It
    must disable grammar for the rest of the session and retry the same turn
    unconstrained."""
    from localm.inference.backends.base import GRAMMAR_UNSUPPORTED_MESSAGE
    from localm.plugins.coder.backends.http import CoderServerError

    agent = _make_agent(tmp_path)
    calls = []

    def _chat(messages, **kw):
        calls.append(kw)
        if len(calls) == 1:
            raise CoderServerError(
                "HTTP 400 error from http://x/v1/chat/completions: "
                + GRAMMAR_UNSUPPORTED_MESSAGE)
        return "ok"

    agent.backend.chat = _chat
    result = agent._call_llm([{"role": "user", "content": "hi"}], interactive=False)

    assert result == "ok"
    assert len(calls) == 2
    assert "grammar" in calls[0]            # the failed attempt DID carry one
    assert "grammar" not in calls[1]        # the retry omits it
    assert agent._grammar_confirmed_unsupported is True
    assert _notice_kinds(agent).count("grammar_unsupported") == 1


def test_grammar_unsupported_disables_forcing_too(tmp_path):
    """Once the server has authoritatively refused, the forced (rung-2)
    grammar must not be offered either - can_force_tool_calls() must report
    False so the escalation ladder reports failed enforcement instead of
    trying (and failing the same way) again."""
    agent = _make_agent(tmp_path)
    agent._grammar_confirmed_unsupported = True
    assert agent.can_force_tool_calls() is False
    assert "grammar" not in agent._llm_kwargs()


def test_unrelated_server_error_still_propagates(tmp_path):
    """Only the grammar-unsupported refusal is swallowed. Any other server
    error must still surface - a broad catch here would silently retry a
    genuinely broken request forever."""
    from localm.plugins.coder.backends.http import CoderServerError

    agent = _make_agent(tmp_path)

    def _chat(messages, **kw):
        raise CoderServerError(
            "HTTP 500 error from http://x/v1/chat/completions: boom")

    agent.backend.chat = _chat
    with pytest.raises(CoderServerError, match="boom"):
        agent._call_llm([{"role": "user", "content": "hi"}], interactive=False)
    assert agent._grammar_confirmed_unsupported is False


def test_invalid_grammar_is_not_treated_as_unsupported(tmp_path):
    """A malformed grammar (OUR OWN bug - the bundled TOOL_CALLS_* grammar
    failing to parse) is a different failure than a capability gap and must
    not be silently swallowed the same way."""
    from localm.plugins.coder.backends.http import CoderServerError

    agent = _make_agent(tmp_path)

    def _chat(messages, **kw):
        raise CoderServerError(
            "HTTP 400 error from http://x/v1/chat/completions: "
            "Invalid grammar: unexpected token")

    agent.backend.chat = _chat
    with pytest.raises(CoderServerError, match="Invalid grammar"):
        agent._call_llm([{"role": "user", "content": "hi"}], interactive=False)
    assert agent._grammar_confirmed_unsupported is False


# --------------------------------------------------------------------------- #
#  The prompt no longer hands the model an excuse                              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("model_name", ["qwen2.5-coder-7b", "some-thinking-model", "tiny-1b"])
def test_identity_line_states_capability_not_deployment(tmp_path, model_name):
    """The system prompt says the model runs on the user's own machine and
    never contains the phrase "fully offline"."""
    from localm.plugins.coder.prompts import build_system_prompt
    prompt = build_system_prompt(tmp_path, model_name=model_name)
    assert "fully offline" not in prompt
    assert "your own machine" in prompt or "user's own machine" in prompt


# --------------------------------------------------------------------------- #
#  The server can refuse the LAZY form specifically - degrade NARROWLY          #
# --------------------------------------------------------------------------- #

def test_lazy_grammar_unsupported_400_degrades_instead_of_crashing(tmp_path):
    """A lazy-grammar refusal must be recognised, not propagated.

    A lazy refusal is a SECOND, distinct refusal message.
    _disable_grammar_on_unsupported matches on the exact message string, so a
    message it does not know returns False and the CoderServerError propagates,
    crashing the whole task. Every ordinary tool-call turn sends
    grammar_lazy=True, so against an HF-backed server that is the FIRST turn.
    """
    from localm.inference.backends.base import GRAMMAR_LAZY_UNSUPPORTED_MESSAGE
    from localm.plugins.coder.backends.http import CoderServerError

    agent = _make_agent(tmp_path)
    calls = []

    def _chat(messages, **kw):
        calls.append(kw)
        if len(calls) == 1:
            raise CoderServerError(
                "HTTP 400 error from http://x/v1/chat/completions: "
                + GRAMMAR_LAZY_UNSUPPORTED_MESSAGE)
        return "ok"

    agent.backend.chat = _chat
    result = agent._call_llm([{"role": "user", "content": "hi"}], interactive=False)

    assert result == "ok"
    assert len(calls) == 2
    assert "grammar_lazy" in calls[0]       # the failed attempt asked for lazy
    assert "grammar" not in calls[1]        # the retry omits the grammar entirely
    assert _notice_kinds(agent).count("lazy_grammar_unsupported") == 1


def test_lazy_refusal_leaves_the_forced_grammar_available(tmp_path):
    """The narrow half: a lazy-only refusal must not discard the forced grammar.

    A server that cannot enforce a grammar from a TRIGGER may still enforce one
    from the first token (an HF backend with the [grammar] extra is exactly
    that). The forced rung is what rescues a turn that produced no tool call at
    all. Contrast test_grammar_unsupported_disables_forcing_too, where the server
    refused grammar OUTRIGHT and forcing correctly goes away.
    """
    agent = _make_agent(tmp_path)
    agent._lazy_grammar_confirmed_unsupported = True

    assert agent.can_force_tool_calls() is True
    # An ordinary (non-forced) turn no longer asks for the lazy grammar...
    assert "grammar_lazy" not in agent._llm_kwargs()
    assert "grammar" not in agent._llm_kwargs()
    # ...but the forced rung still binds a grammar from the first token.
    agent._force_tool_grammar = True
    forced_kw = agent._llm_kwargs()
    assert "grammar" in forced_kw
    assert "grammar_lazy" not in forced_kw
