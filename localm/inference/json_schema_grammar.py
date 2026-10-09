# SPDX-License-Identifier: AGPL-3.0-or-later
"""A JSON schema compiled to a GBNF grammar, so constrained sampling can only
produce JSON that satisfies the schema.

``schema_to_grammar(schema)`` returns grammar text whose entry rule is
``root``. Properties are generated in the order the schema declares them;
required ones always, optional ones when the model chooses them. An object
with declared properties has no others unless ``additionalProperties`` is
true or a schema.

Supported: ``type`` (also a list), ``enum``, ``const``, ``properties``,
``required``, ``additionalProperties``, ``items``, ``prefixItems`` (a fixed
tuple), ``minItems``, ``maxItems``, ``minLength``, ``maxLength``, ``minimum``
and ``maximum`` (and the exclusive forms) on integers, ``format`` ``date``,
``time``, ``date-time`` and ``uuid``, ``anyOf``, ``oneOf``, ``allOf`` of
objects, and local ``$ref`` (recursive schemas included). Annotations
(``title``, ``description``, ``default``, ``examples``) and ``uniqueItems``
are accepted and not enforced. Any other keyword is refused with a
``SchemaGrammarError`` naming it, rather than dropped."""

from __future__ import annotations

import json
import math
import re
from typing import Any

MAX_SCHEMA_BYTES = 65536
MAX_RULES = 2000
MAX_REPEAT = 1000
MAX_NESTING = 32
MAX_INTEGER_BOUND = 10 ** 15


class SchemaGrammarError(ValueError):
    """The schema is malformed, too large, or uses something the grammar
    cannot express."""


_ANNOTATIONS = frozenset({
    "title", "description", "default", "examples", "example", "$schema", "$id",
    "$comment", "deprecated", "readOnly", "writeOnly", "$defs", "definitions",
    "uniqueItems", "contentMediaType", "contentEncoding",
})

_KEYWORDS = frozenset({
    "type", "enum", "const", "properties", "required", "additionalProperties",
    "items", "prefixItems", "minItems", "maxItems", "minLength", "maxLength",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "format",
    "anyOf", "oneOf", "allOf", "$ref",
})

_TYPES = frozenset({"object", "array", "string", "number", "integer", "boolean", "null"})

_DIGIT = "[0-9]"

_PRIMITIVES: dict[str, str] = {
    "ws": '| " " | "\\n" [ \\t]{0,20}',
    "char": '[^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
    "string": '"\\"" char* "\\"" ws',
    "boolean": '("true" | "false") ws',
    "null": '"null" ws',
    "integer": '("-"? ([0-9] | [1-9] [0-9]{0,15})) ws',
    "number": ('("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? '
               '([eE] [-+]? [0-9]{1,3})? ws'),
    "value": "object-any | array-any | string | number | boolean | null",
    "object-any": '"{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws',
    "array-any": '"[" ws (value ("," ws value)*)? "]" ws',
}

_PRIMITIVE_DEPS: dict[str, tuple[str, ...]] = {
    "string": ("char", "ws"),
    "boolean": ("ws",), "null": ("ws",), "integer": ("ws",), "number": ("ws",),
    "value": ("object-any", "array-any", "string", "number", "boolean", "null"),
    "object-any": ("string", "value", "ws"),
    "array-any": ("value", "ws"),
}

_FORMATS: dict[str, str] = {
    "date": ('"\\"" [0-9]{4} "-" ("0" [1-9] | "1" [0-2]) "-" '
             '("0" [1-9] | [12] [0-9] | "3" [01]) "\\"" ws'),
    "time": ('"\\"" ([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] '
             '("." [0-9]{1,9})? ("Z" | ("+" | "-") ([01] [0-9] | "2" [0-3]) ":" '
             '[0-5] [0-9]) "\\"" ws'),
    "date-time": ('"\\"" [0-9]{4} "-" ("0" [1-9] | "1" [0-2]) "-" '
                  '("0" [1-9] | [12] [0-9] | "3" [01]) "T" ([01] [0-9] | "2" [0-3]) '
                  '":" [0-5] [0-9] ":" [0-5] [0-9] ("." [0-9]{1,9})? ("Z" | '
                  '("+" | "-") ([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9]) "\\"" ws'),
    "uuid": ('"\\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" '
             '[0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\\"" ws'),
}


def literal(text: str) -> str:
    """*text* as a GBNF string literal."""
    out = ['"']
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\x{ord(ch):02X}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "rule"


def _digit_class(lo: int, hi: int) -> str:
    return f"[{lo}]" if lo == hi else f"[{lo}-{hi}]"


def _repeat(lo: int, hi: int | None) -> str:
    """The GBNF repetition suffix for lo..hi occurrences (hi None: no upper bound)."""
    if hi is None:
        return "*" if lo == 0 else "+" if lo == 1 else f"{{{lo},}}"
    if lo == hi:
        return f"{{{lo}}}"
    if lo == 0 and hi == 1:
        return "?"
    return f"{{{lo},{hi}}}"


class _Compiler:
    def __init__(self, root: Any) -> None:
        self.root = root
        self.rules: dict[str, str] = {}
        self.refs: dict[str, str] = {}
        self.used: set[str] = set()
        self.depth = 0

    # ------------------------------------------------------------ rules

    def name_for(self, hint: str) -> str:
        base = _slug(hint)
        if base in self.rules or base in _PRIMITIVES:
            n = 2
            while f"{base}-{n}" in self.rules:
                n += 1
            base = f"{base}-{n}"
        if len(self.rules) >= MAX_RULES:
            raise SchemaGrammarError(
                f"the schema needs more than {MAX_RULES} grammar rules; simplify it")
        self.rules[base] = ""
        return base

    def add(self, hint: str, body: str) -> str:
        name = self.name_for(hint)
        self.rules[name] = body
        return name

    def prim(self, name: str) -> str:
        self.used.add(name)
        return name

    # ------------------------------------------------------------ schema walk

    def visit(self, schema: Any, hint: str) -> str:
        self.depth += 1
        try:
            if self.depth > MAX_NESTING:
                raise SchemaGrammarError(
                    f"the schema nests more than {MAX_NESTING} levels deep")
            return self._visit(schema, hint)
        finally:
            self.depth -= 1

    def _visit(self, schema: Any, hint: str) -> str:
        if schema is True or (isinstance(schema, dict) and not schema):
            return self.prim("value")
        if schema is False:
            raise SchemaGrammarError("a schema of false allows no value")
        if not isinstance(schema, dict):
            raise SchemaGrammarError("a schema must be an object or a boolean")
        for key in schema:
            if key in _KEYWORDS or key in _ANNOTATIONS:
                continue
            raise SchemaGrammarError(f"JSON schema keyword '{key}' is not supported")
        if "$ref" in schema:
            return self._ref(schema["$ref"], hint)
        if "const" in schema:
            return self._literals([schema["const"]], hint)
        if "enum" in schema:
            values = schema["enum"]
            if not isinstance(values, list) or not values:
                raise SchemaGrammarError("enum must be a non-empty list")
            return self._literals(values, hint)
        for key in ("anyOf", "oneOf"):
            if key in schema:
                return self._any_of(schema, key, hint)
        if "allOf" in schema:
            return self._visit(self._merge_all_of(schema), hint)
        declared = schema.get("type")
        if isinstance(declared, list):
            if not declared:
                raise SchemaGrammarError("type must not be an empty list")
            alts = [self.visit({**schema, "type": t}, f"{hint}-{t}") for t in declared]
            return self.add(hint, " | ".join(alts))
        kind = declared if declared is not None else self._infer_type(schema)
        if kind is None:
            return self.prim("value")
        if kind not in _TYPES:
            raise SchemaGrammarError(f"unknown JSON schema type {kind!r}")
        return getattr(self, "_" + kind)(schema, hint)

    @staticmethod
    def _infer_type(schema: dict) -> str | None:
        if any(k in schema for k in ("properties", "required", "additionalProperties")):
            return "object"
        if any(k in schema for k in ("items", "prefixItems", "minItems", "maxItems")):
            return "array"
        if any(k in schema for k in ("minLength", "maxLength", "format")):
            return "string"
        if any(k in schema for k in ("minimum", "maximum", "exclusiveMinimum",
                                     "exclusiveMaximum")):
            return "number"
        return None

    # ------------------------------------------------------------ composition

    def _literals(self, values: list, hint: str) -> str:
        texts = [json.dumps(v, ensure_ascii=False, separators=(",", ":")) for v in values]
        body = "(" + " | ".join(literal(t) for t in texts) + ") ws"
        self.prim("ws")
        return self.add(hint, body)

    def _any_of(self, schema: dict, key: str, hint: str) -> str:
        subs = schema[key]
        if not isinstance(subs, list) or not subs:
            raise SchemaGrammarError(f"{key} must be a non-empty list")
        extra = [k for k in schema if k not in (key,) and k not in _ANNOTATIONS]
        if extra:
            raise SchemaGrammarError(
                f"'{extra[0]}' next to {key} is not supported; put it inside each alternative")
        alts = [self.visit(sub, f"{hint}-{i}") for i, sub in enumerate(subs)]
        return self.add(hint, " | ".join(alts))

    def _merge_all_of(self, schema: dict) -> dict:
        subs = schema["allOf"]
        if not isinstance(subs, list) or not subs:
            raise SchemaGrammarError("allOf must be a non-empty list")
        merged: dict[str, Any] = {k: v for k, v in schema.items() if k != "allOf"}
        for sub in subs:
            sub = self._deref(sub)
            if not isinstance(sub, dict):
                raise SchemaGrammarError("allOf takes object schemas")
            for key, value in sub.items():
                if key in _ANNOTATIONS:
                    continue
                if key == "properties":
                    props = dict(merged.get("properties") or {})
                    for name, p in value.items():
                        if name in props and props[name] != p:
                            raise SchemaGrammarError(
                                f"allOf defines property '{name}' twice with different schemas")
                        props[name] = p
                    merged["properties"] = props
                elif key == "required":
                    merged["required"] = list(dict.fromkeys(
                        list(merged.get("required") or []) + list(value)))
                elif key == "type" and merged.get("type") in (None, value):
                    merged["type"] = value
                elif key not in merged or merged[key] == value:
                    merged[key] = value
                else:
                    raise SchemaGrammarError(f"allOf combines '{key}' in a way that is not supported")
        return merged

    def _deref(self, schema: Any) -> Any:
        seen = 0
        while isinstance(schema, dict) and "$ref" in schema:
            seen += 1
            if seen > MAX_NESTING:
                raise SchemaGrammarError("$ref chain is too long")
            schema = self._resolve(schema["$ref"])
        return schema

    def _resolve(self, ref: Any) -> Any:
        if not isinstance(ref, str) or not ref.startswith("#"):
            raise SchemaGrammarError(
                f"only local $ref values ('#/...') are supported, not {ref!r}")
        node: Any = self.root
        for part in [p for p in ref[1:].split("/") if p != ""]:
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            else:
                raise SchemaGrammarError(f"$ref {ref!r} does not resolve")
        return node

    def _ref(self, ref: Any, hint: str) -> str:
        if isinstance(ref, str) and ref in self.refs:
            return self.refs[ref]
        target = self._resolve(ref)
        name = self.name_for(str(ref).rsplit("/", 1)[-1] or "ref")
        self.refs[ref] = name
        self.rules[name] = self.visit(target, name + "-body")
        return name

    # ------------------------------------------------------------ primitives

    def _boolean(self, schema: dict, hint: str) -> str:
        return self.prim("boolean")

    def _null(self, schema: dict, hint: str) -> str:
        return self.prim("null")

    def _number(self, schema: dict, hint: str) -> str:
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if key in schema:
                raise SchemaGrammarError(
                    f"'{key}' is supported on integers only, not on number")
        return self.prim("number")

    def _integer(self, schema: dict, hint: str) -> str:
        lo, hi = self._integer_bounds(schema)
        if lo is None and hi is None:
            return self.prim("integer")
        if lo is not None and hi is not None and lo > hi:
            raise SchemaGrammarError("the integer bounds leave no value")
        self.prim("ws")
        return self.add(hint, f"({self._int_range(lo, hi, hint)}) ws")

    @staticmethod
    def _integer_bounds(schema: dict) -> tuple[int | None, int | None]:
        def num(key: str) -> float | None:
            value = schema.get(key)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value):
                raise SchemaGrammarError(f"'{key}' must be a number")
            return value

        lo = num("minimum")
        hi = num("maximum")
        ex_lo = num("exclusiveMinimum")
        ex_hi = num("exclusiveMaximum")
        low = None if lo is None else math.ceil(lo)
        high = None if hi is None else math.floor(hi)
        if ex_lo is not None:
            cand = math.floor(ex_lo) + 1
            low = cand if low is None else max(low, cand)
        if ex_hi is not None:
            cand = math.ceil(ex_hi) - 1
            high = cand if high is None else min(high, cand)
        for bound in (low, high):
            if bound is not None and abs(bound) > MAX_INTEGER_BOUND:
                raise SchemaGrammarError(
                    f"integer bounds beyond +-{MAX_INTEGER_BOUND} are not supported")
        return low, high

    def _int_range(self, lo: int | None, hi: int | None, hint: str) -> str:
        alts: list[str] = []
        if lo is None or lo < 0:
            mag_lo = 1 if (hi is None or hi >= 0) else -hi
            mag_hi = None if lo is None else -lo
            alts.append('"-" ' + self._magnitude(mag_lo, mag_hi, hint + "-neg"))
        if hi is None or hi >= 0:
            alts.append(self._magnitude(max(lo, 0) if lo is not None else 0, hi,
                                        hint + "-pos"))
        return " | ".join(alts)

    def _magnitude(self, a: int, b: int | None, hint: str) -> str:
        """Decimal integers a..b (a >= 0; b None for no bound), no leading zero."""
        la = len(str(a))
        alts: list[str] = []
        length = la
        while True:
            lo_l = a if length == la else 10 ** (length - 1)
            hi_l = (10 ** length - 1) if (b is None or length < len(str(b))) else b
            alts.append(self._same_length(str(lo_l), str(hi_l), hint))
            if b is None:
                alts.append(f"[1-9] [0-9]{{{la},}}")
                break
            if length == len(str(b)):
                break
            length += 1
        return self.add(hint, " | ".join(alts))

    def _same_length(self, lo: str, hi: str, hint: str) -> str:
        """Atom for the digit strings of the same length as lo and hi, between them."""
        if lo == hi:
            return literal(lo)
        if len(lo) == 1:
            return _digit_class(int(lo), int(hi))
        n = len(lo) - 1
        if lo[0] == hi[0]:
            return literal(lo[0]) + " " + self._same_length(lo[1:], hi[1:], hint)
        alts = []
        if lo[1:] == "0" * n:
            alts.append(_digit_class(int(lo[0]), int(lo[0])) + " " + self._any_digits(n))
        else:
            alts.append(literal(lo[0]) + " " + self._same_length(lo[1:], "9" * n, hint))
        mid_lo, mid_hi = int(lo[0]) + 1, int(hi[0]) - 1
        if mid_lo <= mid_hi:
            alts.append(_digit_class(mid_lo, mid_hi) + " " + self._any_digits(n))
        if hi[1:] == "9" * n:
            alts.append(_digit_class(int(hi[0]), int(hi[0])) + " " + self._any_digits(n))
        else:
            alts.append(literal(hi[0]) + " " + self._same_length("0" * n, hi[1:], hint))
        return "(" + " | ".join(
            self.add(hint + "-alt", alt) if " " in alt else alt for alt in alts) + ")"

    @staticmethod
    def _any_digits(n: int) -> str:
        return _DIGIT if n == 1 else f"{_DIGIT}{{{n}}}"

    def _string(self, schema: dict, hint: str) -> str:
        fmt = schema.get("format")
        if fmt in _FORMATS:
            self.prim("ws")
            return self.add(hint, _FORMATS[fmt])
        lo = schema.get("minLength", 0)
        hi = schema.get("maxLength")
        for key, value in (("minLength", lo), ("maxLength", hi)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                      or value < 0 or value > MAX_REPEAT):
                raise SchemaGrammarError(
                    f"'{key}' must be an integer from 0 to {MAX_REPEAT}")
        if hi is not None and lo > hi:
            raise SchemaGrammarError("minLength is above maxLength")
        if lo == 0 and hi is None:
            return self.prim("string")
        self.prim("char")
        self.prim("ws")
        count = f"{{{lo},}}" if hi is None else f"{{{lo},{hi}}}"
        return self.add(hint, f'"\\"" char{count} "\\"" ws')

    # ------------------------------------------------------------ array

    def _array(self, schema: dict, hint: str) -> str:
        self.prim("ws")
        items = schema.get("items")
        prefix = schema.get("prefixItems")
        lo = schema.get("minItems", 0)
        hi = schema.get("maxItems")
        for key, value in (("minItems", lo), ("maxItems", hi)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                      or value < 0 or value > MAX_REPEAT):
                raise SchemaGrammarError(
                    f"'{key}' must be an integer from 0 to {MAX_REPEAT}")
        if hi is not None and lo > hi:
            raise SchemaGrammarError("minItems is above maxItems")
        if prefix is not None:
            if not isinstance(prefix, list) or not prefix:
                raise SchemaGrammarError("prefixItems must be a non-empty list")
            if lo or hi is not None or items not in (None, False):
                raise SchemaGrammarError(
                    "prefixItems with minItems, maxItems or items is not supported")
            seq = ' "," ws '.join(
                self.visit(p, f"{hint}-{i}") for i, p in enumerate(prefix))
            return self.add(hint, f'"[" ws {seq} "]" ws')
        if items is False or hi == 0:
            return self.add(hint, '"[" ws "]" ws')
        item = self.visit(True if items is None else items, f"{hint}-item")
        more = f'("," ws {item})' + _repeat(max(lo - 1, 0), None if hi is None else hi - 1)
        if hi is not None and hi - 1 == 0:
            more = ""
        inner = f"{item} {more}".rstrip()
        if lo == 0:
            inner = f"({inner})?"
        return self.add(hint, f'"[" ws {inner} "]" ws')

    # ------------------------------------------------------------ object

    def _object(self, schema: dict, hint: str) -> str:
        self.prim("ws")
        props = schema.get("properties")
        if props is None:
            props = {}
        if not isinstance(props, dict):
            raise SchemaGrammarError("properties must be an object")
        required = schema.get("required") or []
        if not isinstance(required, list) or not all(isinstance(r, str) for r in required):
            raise SchemaGrammarError("required must be a list of strings")
        extra = schema.get("additionalProperties")
        props = dict(props)
        for name in required:
            props.setdefault(name, True)
        members: list[tuple[str, bool, bool]] = []
        for name, sub in props.items():
            value = self.visit(sub, f"{hint}-{name}")
            kv = self.add(f"{hint}-{name}-kv", f'{literal(json.dumps(name, ensure_ascii=False))} ws ":" ws {value}')
            members.append((kv, name in required, False))
        if extra is True or isinstance(extra, dict) or (extra is None and not props):
            value = self.visit(True if extra in (None, True) else extra, f"{hint}-extra")
            key = self.prim("string")
            kv = self.add(f"{hint}-extra-kv", f'{key} ":" ws {value}')
            members.append((kv, False, True))
        elif extra is not None and extra is not False:
            raise SchemaGrammarError("additionalProperties must be a boolean or a schema")
        if not members:
            return self.add(hint, '"{" ws "}" ws')
        memo: dict[tuple[int, bool], str] = {}

        def rest(i: int, started: bool) -> str:
            """Rule name for members[i:] given whether a member was already written;
            '' when nothing can follow."""
            if i >= len(members):
                return ""
            key = (i, started)
            if key in memo:
                return memo[key]
            kv, is_required, repeat = members[i]
            after_started = rest(i + 1, True)
            comma_kv = f'"," ws {kv}'
            if is_required:
                body = (comma_kv if started else kv) + (f" {after_started}" if after_started else "")
            elif started:
                one = f"({comma_kv})" + ("*" if repeat else "?")
                body = one + (f" {after_started}" if after_started else "")
            else:
                first = kv + (f" ({comma_kv})*" if repeat else "")
                with_member = first + (f" {after_started}" if after_started else "")
                skipped = rest(i + 1, False)
                body = (f"({with_member} | {skipped})" if skipped
                        else f"({with_member})?")
            memo[key] = self.add(f"{hint}-m{i}{'c' if started else ''}", body)
            return memo[key]

        chain = rest(0, False)
        return self.add(hint, f'"{{" ws {chain} "}}" ws')

    # ------------------------------------------------------------ output

    def render(self, root_expr: str) -> str:
        self.rules["root"] = root_expr
        needed: set[str] = set()
        stack = list(self.used)
        while stack:
            name = stack.pop()
            if name in needed:
                continue
            needed.add(name)
            stack.extend(_PRIMITIVE_DEPS.get(name, ()))
        lines = [f"root ::= {self.rules['root']}"]
        lines += [f"{n} ::= {b}" for n, b in self.rules.items() if n != "root"]
        lines += [f"{n} ::= {_PRIMITIVES[n]}" for n in _PRIMITIVES if n in needed]
        return "\n".join(lines) + "\n"


def schema_to_grammar(schema: Any) -> str:
    """The GBNF grammar (entry rule ``root``) for the JSON *schema*.

    Raises :class:`SchemaGrammarError` for a schema that is malformed, too
    large, or uses a keyword the grammar cannot enforce."""
    try:
        size = len(json.dumps(schema, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise SchemaGrammarError(f"the schema is not valid JSON: {exc}") from None
    if size > MAX_SCHEMA_BYTES:
        raise SchemaGrammarError(
            f"the schema is {size} bytes, over the {MAX_SCHEMA_BYTES}-byte limit")
    comp = _Compiler(schema)
    comp.rules["root"] = ""
    comp.refs["#"] = "root"
    root = comp.visit(schema, "root-body")
    if root == "root":
        raise SchemaGrammarError("the schema is only a reference to itself")
    return comp.render(root)
