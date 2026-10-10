# SPDX-License-Identifier: AGPL-3.0-or-later
"""OpenAI ``response_format`` as a GBNF grammar.

``{"type": "text"}`` asks for nothing. ``{"type": "json_object"}`` constrains
the reply to a JSON object. ``{"type": "json_schema", "json_schema": {"name",
"schema", "strict"}}`` constrains it to the schema: with ``strict`` true a
schema keyword the grammar cannot enforce is refused, otherwise that keyword
is left out and the rest of the schema is still enforced.

Pure Python: no engine, no web framework."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from localm.inference.gbnf import check_grammar_structure
from localm.inference.json_schema_grammar import (
    SchemaGrammarError, loosen_schema, schema_to_grammar,
)

_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_RULE_DEF = re.compile(r"^\s*([A-Za-z0-9-]+)\s*::=", re.MULTILINE)
_TOKENS = re.compile(
    r'"(?:\\.|[^"\\])*"'          # string literal
    r"|\[(?:\\.|[^\]\\])*\]"      # character class
    r"|#[^\n]*"                   # comment
    r"|\{[0-9,\s]*\}"             # repeat count
    r"|[A-Za-z0-9-]+"             # rule name
    r"|.",
    re.DOTALL)

# A <think> block of at most 1900 characters that cannot contain "</t" before
# its closing tag; the same bound as gbnf.TOOL_CALLS_AFTER_THINK.
_THINK_RULES = (
    'think-body ::= think-char{0,1900}\n'
    'think-char ::= [^<] | "<" [^/] | "</" [^t]\n'
    'think-ws ::= [ \\t\\n\\r]? [ \\t\\n\\r]?\n')


class ResponseFormatError(ValueError):
    """The ``response_format`` of a request is malformed or cannot be enforced."""


@dataclass(frozen=True)
class ResponseFormat:
    kind: str                                  # "json_object" or "json_schema"
    name: Optional[str] = None
    schema: Optional[dict[str, Any]] = None
    strict: bool = False


def parse_response_format(value: Any) -> Optional[ResponseFormat]:
    """The format *value* asks for, or ``None`` for plain text (absent, null or
    ``{"type": "text"}``). Raises :class:`ResponseFormatError` for anything
    malformed."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ResponseFormatError("response_format must be an object")
    kind = value.get("type")
    if kind == "text":
        return None
    if kind == "json_object":
        return ResponseFormat("json_object")
    if kind != "json_schema":
        raise ResponseFormatError(
            f"response_format.type {kind!r} is not supported: expected "
            f"'text', 'json_object' or 'json_schema'")
    spec = value.get("json_schema")
    if not isinstance(spec, dict):
        raise ResponseFormatError("response_format.json_schema must be an object")
    name = spec.get("name")
    if not isinstance(name, str) or not _NAME.match(name):
        raise ResponseFormatError(
            "response_format.json_schema.name must be 1 to 64 letters, digits, "
            "underscores or dashes")
    strict = spec.get("strict")
    if strict is not None and not isinstance(strict, bool):
        raise ResponseFormatError("response_format.json_schema.strict must be a boolean")
    schema = spec.get("schema")
    if schema is None:
        return ResponseFormat("json_object", name=name, strict=bool(strict))
    if not isinstance(schema, dict):
        raise ResponseFormatError("response_format.json_schema.schema must be an object")
    return ResponseFormat("json_schema", name=name, schema=schema, strict=bool(strict))


def format_grammar(fmt: ResponseFormat) -> tuple[str, list[str]]:
    """``(grammar, dropped)``: the grammar (entry rule ``root``) that makes a reply
    satisfy *fmt*, and the schema keywords left out of it (only when *fmt* is not
    strict; a keyword that gives the schema its shape is never left out).
    Raises :class:`ResponseFormatError` when the schema cannot be enforced."""
    if fmt.kind == "json_object" or fmt.schema is None:
        return schema_to_grammar({"type": "object"}), []
    if fmt.strict:
        try:
            return schema_to_grammar(fmt.schema), []
        except SchemaGrammarError as exc:
            raise ResponseFormatError(
                f"response_format.json_schema.schema: {exc}") from None
    loosened, dropped, error = loosen_schema(fmt.schema)
    if loosened is None:
        raise ResponseFormatError(
            f"response_format.json_schema.schema: {error or 'cannot be compiled'}")
    return schema_to_grammar(loosened), dropped


def prefix_rules(grammar: str, prefix: str) -> str:
    """*grammar* with every rule it defines renamed to ``prefix + name``, at its
    definition and at every reference. String literals, character classes,
    comments and repeat counts are left as they are, also when a rule name is
    all digits."""
    defined = set(_RULE_DEF.findall(grammar))
    return "".join(
        prefix + tok if tok in defined else tok for tok in _TOKENS.findall(grammar))


def combine(*alternatives: str) -> str:
    """One grammar whose ``root`` matches any of *alternatives* (each a grammar
    with entry rule ``root``)."""
    if len(alternatives) == 1:
        return alternatives[0]
    parts = [prefix_rules(g, f"alt{i}-") for i, g in enumerate(alternatives)]
    root = " | ".join(f"alt{i}-root" for i in range(len(parts)))
    return f"root ::= {root}\n" + "\n".join(p.rstrip("\n") for p in parts) + "\n"


def after_think(grammar: str) -> str:
    """*grammar* behind an optional bounded ``<think>...</think>`` block, for a
    reasoning model asked to keep its reasoning on."""
    inner = prefix_rules(grammar, "fmt-")
    return ('root ::= ("<think>" think-body "</think>" think-ws)? fmt-root\n'
            + _THINK_RULES + inner.rstrip("\n") + "\n")


def checked(grammar: str) -> str:
    """*grammar*, after the structural check every grammar passes before it
    reaches the native parser. Raises ``InvalidGrammarError`` when it fails."""
    check_grammar_structure(grammar)
    return grammar
