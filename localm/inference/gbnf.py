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

# ---------------------------------------------------------------------------
#  JSON grammars
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
#  Tool-call grammar (coder XML format)
# ---------------------------------------------------------------------------

# Constrains the entire response to one or more <tool_call>…</tool_call> blocks
# separated by optional whitespace. Useful when you want the model to *only*
# emit tool calls and nothing else (e.g. a routing step).
#
# Normal conversational turns - which mix free text with tool calls - cannot
# be constrained this way without restricting the model too aggressively.
# Use the text parser for those turns.
TOOL_CALLS_ONLY = r"""
root       ::= ws tool-block+ ws
tool-block ::= "<tool_call>" ws json-obj ws "</tool_call>" ws
json-obj   ::= "{" ws "\"name\"" ws ":" ws string ws "," ws "\"args\"" ws ":" ws object ws "}"
object     ::= "{" ws (member ws ("," ws member ws)*)? "}"
member     ::= string ws ":" ws value
value      ::= object | array | string | number | "true" | "false" | "null"
array      ::= "[" ws (value ws ("," ws value ws)*)? "]"
string     ::= "\"" ([^\"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number     ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws         ::= ([ \t\n\r])*
""".strip()


