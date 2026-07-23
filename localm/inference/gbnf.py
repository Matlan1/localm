# SPDX-License-Identifier: AGPL-3.0-or-later
"""
GBNF grammar strings for constrained sampling.

These grammars are passed to ``llama_sampler_init_grammar`` so the sampler
masks any token that would violate the grammar at the current parse position.
The result is structurally valid output without post-hoc repair.

Usage example::

    from localm.inference.gbnf import JSON_OBJECT
    tokens = engine.chat_stream(messages, grammar=JSON_OBJECT)

All grammars use ``root`` as the entry rule, which is what
``llama_sampler_init_grammar`` expects.
"""

from __future__ import annotations

import re

#  JSON grammars

# Any valid JSON value at the root
JSON_VALUE = r"""
root   ::= ws value ws
value  ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (string ws ":" ws value ws ("," ws string ws ":" ws value ws)*)? "}"
array  ::= "[" ws (value ws ("," ws value ws)*)? "]"
string ::= "\"" ([^\"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws     ::= ([ \t\n\r])*
""".strip()

# Root must be a JSON object (dict)
JSON_OBJECT = r"""
root   ::= ws object ws
object ::= "{" ws (member ws ("," ws member ws)*)? "}"
member ::= string ws ":" ws value
value  ::= object | array | string | number | "true" | "false" | "null"
array  ::= "[" ws (value ws ("," ws value ws)*)? "]"
string ::= "\"" ([^\"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws     ::= ([ \t\n\r])*
""".strip()

# Root must be a JSON array
JSON_ARRAY = r"""
root   ::= ws array ws
array  ::= "[" ws (value ws ("," ws value ws)*)? "]"
value  ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (member ws ("," ws member ws)*)? "}"
member ::= string ws ":" ws value
string ::= "\"" ([^\"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws     ::= ([ \t\n\r])*
""".strip()


#  Tool-call grammar (coder XML format)

# Constrains the whole response to one or more <tool_call>...</tool_call> blocks
# separated by optional whitespace.
#
# STRICT mode (grammar=TOOL_CALLS_ONLY alone) forces tool-only output from the
# first token (no thinking/prose). Live-tested 2026-07-02: a thinking model
# masked off its <think> opener stalls into the leading `ws` (whitespace-only
# reply), so strict suits only a "must call a tool NOW" routing step.
#
# LAZY mode (grammar_lazy=True + TOOL_CALL_TRIGGER) is the general-turn form:
# free text and <think> flow unconstrained; the grammar engages only when the
# model starts a <tool_call>, from where the call must be valid JSON.
TOOL_CALLS_ONLY = r"""
root       ::= opt-ws tool-block+ opt-ws
tool-block ::= "<tool_call>" opt-ws json-obj opt-ws "</tool_call>" opt-ws
json-obj   ::= "{" ws "\"name\"" ws ":" ws string ws "," ws "\"args\"" ws ":" ws object ws "}"
object     ::= "{" ws (member ws ("," ws member ws)*)? "}"
member     ::= string ws ":" ws value
value      ::= object | array | string | number | "true" | "false" | "null"
array      ::= "[" ws (value ws ("," ws value ws)*)? "]"
string     ::= "\"" ([^\"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number     ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws         ::= ([ \t\n\r])*
opt-ws     ::= [ \t\n\r]? [ \t\n\r]? [ \t\n\r]?
""".strip()

# Lazy-grammar trigger for TOOL_CALLS_ONLY: full-match-with-capture-group form
# per llama.cpp's trigger_patterns contract (the grammar is fed from capture
# group 1, so enforcement starts exactly at the tag). Verified live 2026-07-02
# on the bundled runtime: prose before the tag flows free; a started
# <tool_call> is forced to a valid call.
TOOL_CALL_TRIGGER = r"[\s\S]*?(<tool_call>[\s\S]*)"


#  Structural pre-validation (LM-FZ-001)

# llama.cpp's GBNF parser is a hand-written recursive-descent parser with no
# bound on nesting depth, alternation count, or repeat-count size. A grammar
# string built of thousands of unmatched "(" drives its recursion far enough
# to overflow the native call stack - observed live as both a caught
# "exception: stack overflow" and a hard STATUS_ACCESS_VIOLATION process
# crash for the identical input, depending on heap layout at the moment of
# overflow. These bounds are deliberately generous for any real grammar (the
# largest production grammar in this file, TOOL_CALLS_ONLY, nests 3 levels
# deep) while being far below anything that risks the native stack.
MAX_GRAMMAR_BYTES = 65536
MAX_GRAMMAR_NESTING_DEPTH = 128
MAX_GRAMMAR_REPEAT_COUNT = 10000

_REPEAT_COUNT_RE = re.compile(r"\{(\d+)(?:,(\d+))?\}")


def check_grammar_structure(grammar: str) -> None:
    """Reject a grammar whose size or structural complexity could drive the
    native GBNF parser into stack overflow, BEFORE any of it reaches that
    parser. Pure Python, no native call - safe to run unconditionally on
    every grammar-bearing request, including the path where up-front
    validation would otherwise be deferred (RunnerBusy) straight into a
    generation-time native call.

    Raises :class:`InvalidGrammarError` (the same typed error the native
    validation path raises for a malformed grammar) so callers can treat both
    as one clean 400, never a native fault."""
    from localm.inference.backends.base import InvalidGrammarError

    if len(grammar) > MAX_GRAMMAR_BYTES:
        raise InvalidGrammarError(
            f"grammar is {len(grammar)} bytes, over the {MAX_GRAMMAR_BYTES}-byte limit")

    depth = 0
    max_depth = 0
    for ch in grammar:
        if ch in "([":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch in ")]":
            depth = max(0, depth - 1)
    if max_depth > MAX_GRAMMAR_NESTING_DEPTH:
        raise InvalidGrammarError(
            f"grammar nests {max_depth} levels deep, over the "
            f"{MAX_GRAMMAR_NESTING_DEPTH}-level limit (a common shape behind a "
            "native parser stack overflow)")

    for m in _REPEAT_COUNT_RE.finditer(grammar):
        for group in m.groups():
            if group is not None and int(group) > MAX_GRAMMAR_REPEAT_COUNT:
                raise InvalidGrammarError(
                    f"grammar repeat count {{{m.group(0)}}} exceeds the "
                    f"{MAX_GRAMMAR_REPEAT_COUNT} limit")


