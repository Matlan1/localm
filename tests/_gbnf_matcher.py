# SPDX-License-Identifier: AGPL-3.0-or-later
"""A small GBNF matcher for tests: does a grammar accept a given text?

Understands the subset the JSON-schema compiler and localm's own grammars use:
``name ::= alternatives`` one rule per line, string literals with escapes,
character classes (ranges, negation, escapes), groups, references, ``|``, and
the repetitions ``*`` ``+`` ``?`` ``{m}`` ``{m,}`` ``{m,n}``. It tracks the set
of end positions of every match, so alternations and loops need no
backtracking limits."""

from __future__ import annotations

import re

_NAME = re.compile(r"[A-Za-z0-9_-]+")
_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"', "]": "]",
            "[": "[", "-": "-", "^": "^"}


class GrammarSyntaxError(ValueError):
    pass


class _Parser:
    def __init__(self, text: str) -> None:
        self.s = text
        self.i = 0

    def ws(self) -> None:
        while self.i < len(self.s) and self.s[self.i] in " \t":
            self.i += 1

    def peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def alts(self):
        seqs = [self.seq()]
        while True:
            self.ws()
            if self.peek() == "|":
                self.i += 1
                seqs.append(self.seq())
            else:
                return seqs[0] if len(seqs) == 1 else ("alt", seqs)

    def seq(self):
        items = []
        while True:
            self.ws()
            ch = self.peek()
            if ch in ("", "|", ")"):
                return ("seq", items)
            items.append(self.item())

    def item(self):
        node = self.atom()
        while True:
            ch = self.peek()
            if ch == "*":
                self.i += 1
                node = ("rep", node, 0, None)
            elif ch == "+":
                self.i += 1
                node = ("rep", node, 1, None)
            elif ch == "?":
                self.i += 1
                node = ("rep", node, 0, 1)
            elif ch == "{":
                m = re.compile(r"\{(\d+)(,(\d*))?\}").match(self.s, self.i)
                if not m:
                    raise GrammarSyntaxError(f"bad repetition at {self.i}")
                lo = int(m.group(1))
                hi = lo if m.group(2) is None else (int(m.group(3)) if m.group(3) else None)
                self.i = m.end()
                node = ("rep", node, lo, hi)
            else:
                return node

    def escape(self) -> str:
        self.i += 1
        ch = self.s[self.i]
        if ch == "x":
            val = chr(int(self.s[self.i + 1:self.i + 3], 16))
            self.i += 3
            return val
        if ch == "u":
            val = chr(int(self.s[self.i + 1:self.i + 5], 16))
            self.i += 5
            return val
        if ch in _ESCAPES:
            self.i += 1
            return _ESCAPES[ch]
        raise GrammarSyntaxError(f"unknown escape \\{ch}")

    def atom(self):
        ch = self.peek()
        if ch == '"':
            self.i += 1
            out = []
            while self.peek() != '"':
                if self.peek() == "":
                    raise GrammarSyntaxError("unterminated literal")
                out.append(self.escape() if self.peek() == "\\" else self.take())
            self.i += 1
            return ("lit", "".join(out))
        if ch == "[":
            self.i += 1
            negate = self.peek() == "^"
            if negate:
                self.i += 1
            ranges = []
            while self.peek() != "]":
                if self.peek() == "":
                    raise GrammarSyntaxError("unterminated class")
                lo = self.escape() if self.peek() == "\\" else self.take()
                hi = lo
                if self.peek() == "-" and self.s[self.i + 1:self.i + 2] != "]":
                    self.i += 1
                    hi = self.escape() if self.peek() == "\\" else self.take()
                ranges.append((ord(lo), ord(hi)))
            self.i += 1
            return ("cls", negate, tuple(ranges))
        if ch == "(":
            self.i += 1
            node = self.alts()
            self.ws()
            if self.peek() != ")":
                raise GrammarSyntaxError("missing )")
            self.i += 1
            return node
        m = _NAME.match(self.s, self.i)
        if not m:
            raise GrammarSyntaxError(f"unexpected {ch!r} at {self.i}")
        self.i = m.end()
        return ("ref", m.group(0))

    def take(self) -> str:
        ch = self.s[self.i]
        self.i += 1
        return ch


class Grammar:
    def __init__(self, text: str) -> None:
        self.rules: dict[str, tuple] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, sep, body = line.partition("::=")
            if not sep:
                raise GrammarSyntaxError(f"not a rule: {line!r}")
            parser = _Parser(body.strip())
            node = parser.alts()
            parser.ws()
            if parser.i != len(parser.s):
                raise GrammarSyntaxError(f"trailing text in rule {name.strip()!r}")
            self.rules[name.strip()] = node
        for name, node in self.rules.items():
            self._check_refs(node, name)

    def _check_refs(self, node, owner: str) -> None:
        kind = node[0]
        if kind == "ref" and node[1] not in self.rules:
            raise GrammarSyntaxError(f"rule {owner!r} uses undefined rule {node[1]!r}")
        if kind in ("alt", "seq"):
            for child in node[1]:
                self._check_refs(child, owner)
        elif kind == "rep":
            self._check_refs(node[1], owner)

    def accepts(self, text: str, start: str = "root") -> bool:
        memo: dict = {}

        def run(node, i: int) -> frozenset:
            key = (id(node), i)
            if key in memo:
                return memo[key]
            kind = node[0]
            if kind == "lit":
                out = frozenset({i + len(node[1])}) if text.startswith(node[1], i) else frozenset()
            elif kind == "cls":
                hit = False
                if i < len(text):
                    c = ord(text[i])
                    hit = any(lo <= c <= hi for lo, hi in node[2]) != node[1]
                out = frozenset({i + 1}) if hit else frozenset()
            elif kind == "ref":
                out = run(self.rules[node[1]], i)
            elif kind == "alt":
                out = frozenset().union(*(run(n, i) for n in node[1]))
            elif kind == "seq":
                pos = {i}
                for child in node[1]:
                    pos = set().union(*(run(child, p) for p in pos)) if pos else set()
                    if not pos:
                        break
                out = frozenset(pos)
            else:
                _kind, child, lo, hi = node
                cur = {i}
                for _ in range(lo):
                    cur = set().union(*(run(child, p) for p in cur)) if cur else set()
                res = set(cur)
                if hi is None:
                    seen, frontier = set(cur), set(cur)
                    while frontier:
                        nxt = set().union(*(run(child, p) for p in frontier)) - seen
                        seen |= nxt
                        res |= nxt
                        frontier = nxt
                else:
                    for _ in range(hi - lo):
                        cur = set().union(*(run(child, p) for p in cur)) if cur else set()
                        if not cur:
                            break
                        res |= cur
                out = frozenset(res)
            memo[key] = out
            return out

        return len(text) in run(("ref", start), 0)
