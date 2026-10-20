# SPDX-License-Identifier: AGPL-3.0-or-later
"""Structural file outlines: the classes, functions and methods a source file
declares, each with the exact line span it occupies.

Python is parsed with ``ast``, so spans are exact. Every other supported
language (and a Python file that does not parse) goes through a regex scan
that finds declaration lines by shape and estimates each end line from
indentation; those spans are approximate and the result says so.

The result feeds the ``outline_file`` coder tool (tools/outline.py), which
lets the model read a large file's shape first and then ``read_file`` just the
span it needs.
"""

from __future__ import annotations

import ast
import re
import warnings
from dataclasses import dataclass

from .indexer import _lang_for_ext


@dataclass(frozen=True)
class OutlineEntry:
    kind:  str      # "class", "function", "method", "struct", ...
    text:  str      # the declaration as written, e.g. "def area(self) -> float"
    start: int      # 1-based first line (a decorated Python def starts at its first decorator)
    end:   int      # 1-based last line, inclusive
    depth: int      # 0 = top level, 1 = member of a depth-0 entry, ...


@dataclass(frozen=True)
class Outline:
    entries: list[OutlineEntry]
    method:  str        # "ast" or "regex"
    lang:    str
    lines:   int        # total lines in the file


_MAX_DECL_CHARS = 110   # a declaration longer than this is shown cut short


def _count_lines(text: str) -> int:
    return text.count("\n") + (1 if text and not text.endswith("\n") else 0)


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _MAX_DECL_CHARS else text[:_MAX_DECL_CHARS - 3] + "..."


# ---------------------------------------------------------------------------
#  Python: exact, via ast
# ---------------------------------------------------------------------------

def _py_signature(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(b) for b in node.bases]
        bases += [f"{k.arg}={ast.unparse(k.value)}" for k in node.keywords if k.arg]
        return f"class {node.name}" + (f"({', '.join(bases)})" if bases else "")
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){ret}"


def _py_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(d.lineno for d in decorators)])


def _py_collect(body: list[ast.stmt], depth: int, in_class: bool,
                out: list[OutlineEntry]) -> None:
    """Declarations in *body*, descending into class bodies and into the
    compound statements (``if``, ``try``, ``with``) that wrap declarations at
    module or class level. A function body is not descended: its nested
    helpers are implementation detail, not structure."""
    for node in body:
        if isinstance(node, ast.ClassDef):
            out.append(OutlineEntry("class", _clip(_py_signature(node)),
                                    _py_start(node), node.end_lineno or node.lineno, depth))
            _py_collect(node.body, depth + 1, True, out)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "method" if in_class else "function"
            out.append(OutlineEntry(kind, _clip(_py_signature(node)),
                                    _py_start(node), node.end_lineno or node.lineno, depth))
        elif isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            _py_collect(node.body, depth, in_class, out)
            _py_collect(node.orelse, depth, in_class, out)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            _py_collect(node.body, depth, in_class, out)
        elif isinstance(node, ast.Try) or type(node).__name__ == "TryStar":
            _py_collect(node.body, depth, in_class, out)
            for handler in node.handlers:
                _py_collect(handler.body, depth, in_class, out)
            _py_collect(node.orelse, depth, in_class, out)
            _py_collect(node.finalbody, depth, in_class, out)


def _outline_python_ast(text: str) -> list[OutlineEntry] | None:
    """Exact outline, or None when *text* does not parse as Python."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    out: list[OutlineEntry] = []
    try:
        _py_collect(tree.body, 0, False, out)
    except RecursionError:
        return None
    return out


# ---------------------------------------------------------------------------
#  Everything else: regex declaration scan, estimated end lines
# ---------------------------------------------------------------------------

_NOT_A_DECLARATION: frozenset[str] = frozenset({
    "if", "else", "elif", "for", "foreach", "while", "switch", "case", "catch",
    "return", "do", "try", "finally", "with", "synchronized", "sizeof", "typeof",
    "new", "throw", "using", "lock", "fixed", "unless", "until", "when",
})

_IDENT = r"[A-Za-z_$][\w$]*"

# Keywords that open a statement; a line starting with one is never a declaration.
_STATEMENT_START = (r"\s*(?:return|throw|new|else|if|for|foreach|while|switch|case|catch|"
                    r"do|try|yield|await|assert|break|continue|goto|delete|using)\b")

# One (kind, pattern) list per language. Each pattern's group 1 is the leading
# indentation and group 2 the declared name; matching is per physical line.
_PATTERNS: dict[str, list[tuple[str, re.Pattern[str]]]] = {}


def _p(lang_names: tuple[str, ...], *specs: tuple[str, str]) -> None:
    compiled = [(kind, re.compile(rx)) for kind, rx in specs]
    for name in lang_names:
        _PATTERNS[name] = compiled


_JS_MODS = r"(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?"
_p(("javascript", "typescript"),
   ("class",    rf"^(\s*){_JS_MODS}class\s+({_IDENT})"),
   ("interface", rf"^(\s*){_JS_MODS}interface\s+({_IDENT})"),
   ("type",     rf"^(\s*){_JS_MODS}type\s+({_IDENT})\s*(?:<[^=]*>)?\s*="),
   ("enum",     rf"^(\s*){_JS_MODS}(?:const\s+)?enum\s+({_IDENT})"),
   ("function", rf"^(\s*){_JS_MODS}(?:async\s+)?function\s*\*?\s*({_IDENT})"),
   ("function", rf"^(\s*)(?:export\s+)?(?:const|let|var)\s+({_IDENT})\s*(?::[^=]+)?=\s*"
                rf"(?:async\s*)?(?:function\b|\([^)]*\)\s*(?::[^=]+)?=>|{_IDENT}\s*=>)"),
   ("method",   rf"^(\s+)(?:(?:public|private|protected|static|readonly|override|async|get|set)\s+)*"
                rf"({_IDENT})\s*(?:<[^>]*>)?\([^)]*\)\s*(?::\s*[^{{;=]+)?\{{\s*$"),
   )

_p(("go",),
   ("type",     rf"^(\s*)type\s+({_IDENT})\s+(?:struct|interface)\b"),
   ("function", rf"^(\s*)func\s+(?:\([^)]*\)\s*)?({_IDENT})"),
   )

_p(("rust",),
   ("struct",   rf"^(\s*)(?:pub(?:\([^)]*\))?\s+)?struct\s+({_IDENT})"),
   ("enum",     rf"^(\s*)(?:pub(?:\([^)]*\))?\s+)?enum\s+({_IDENT})"),
   ("trait",    rf"^(\s*)(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?trait\s+({_IDENT})"),
   ("impl",     r"^(\s*)(?:unsafe\s+)?impl(?:<[^>]*>)?\s+([^{]+?)\s*\{?\s*$"),
   ("mod",      rf"^(\s*)(?:pub(?:\([^)]*\))?\s+)?mod\s+({_IDENT})"),
   ("function", rf"^(\s*)(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+({_IDENT})"),
   )

_JAVA_MODS = (r"(?:(?:public|private|protected|internal|static|final|abstract|sealed|"
              r"partial|open|data|override|virtual|synchronized|native|suspend)\s+)*")
_p(("java", "csharp", "kotlin", "swift"),
   ("class",    rf"^(\s*){_JAVA_MODS}(?:class|interface|enum|record|struct|object|protocol|extension)\s+({_IDENT})"),
   ("function", rf"^(\s*){_JAVA_MODS}(?:fun|func)\s+(?:<[^>]*>\s*)?({_IDENT})"),
   ("method",   rf"^(\s*)(?!{_STATEMENT_START}){_JAVA_MODS}[\w<>\[\],.?\s]+?\s+({_IDENT})\s*\([^;]*$"),
   )

_p(("ruby",),
   ("class",    rf"^(\s*)(?:class|module)\s+({_IDENT}(?:::{_IDENT})*)"),
   ("function", rf"^(\s*)def\s+(?:self\.)?({_IDENT}[?!=]?)"),
   )

_p(("php",),
   ("class",    rf"^(\s*)(?:(?:abstract|final)\s+)?(?:class|interface|trait|enum)\s+({_IDENT})"),
   ("function", rf"^(\s*)(?:(?:public|private|protected|static|final|abstract)\s+)*function\s+&?({_IDENT})"),
   )

_p(("lua",),
   ("function", rf"^(\s*)(?:local\s+)?function\s+({_IDENT}(?:[.:]{_IDENT})*)"),
   )

_p(("shell",),
   ("function", rf"^(\s*)function\s+({_IDENT})"),
   ("function", rf"^(\s*)({_IDENT})\s*\(\)\s*\{{?"),
   )

_C_FUNC = (rf"^()(?!{_STATEMENT_START})(?:[A-Za-z_][\w\s\*&:<>,~]*?[\s\*&])?"
           rf"({_IDENT}(?:::~?{_IDENT})*)\s*\([^;]*$")
_p(("c", "cpp"),
   ("class",    rf"^(\s*)(?:typedef\s+)?(?:class|struct|union|enum(?:\s+class)?|namespace)\s+({_IDENT})\b[^;]*$"),
   ("function", _C_FUNC),
   )

_p(("generic",),
   ("function", rf"^(\s*)(?:def|fn|func|function|sub|proc)\s+({_IDENT})"),
   ("class",    rf"^(\s*)class\s+({_IDENT})"),
   )

_MAX_REGEX_LINE = 400   # a longer line (minified code) is not scanned


def _regex_declarations(lines: list[str], lang: str) -> list[tuple[int, int, str, str]]:
    """(1-based line, indent width, kind, declaration text) per declaration line."""
    patterns = _PATTERNS.get(lang) or _PATTERNS["generic"]
    found: list[tuple[int, int, str, str]] = []
    for lineno, line in enumerate(lines, start=1):
        if len(line) > _MAX_REGEX_LINE or not line.strip():
            continue
        stripped = line.lstrip()
        if stripped.startswith(("//", "#", "*", "/*", "--")):
            continue
        for kind, rx in patterns:
            m = rx.match(line)
            if m is None:
                continue
            name = m.group(2)
            if name in _NOT_A_DECLARATION:
                continue
            indent = len(m.group(1).expandtabs(4))
            decl = stripped.rstrip().rstrip("{").rstrip(":").rstrip()
            found.append((lineno, indent, kind, _clip(decl)))
            break
    return found


def _outline_regex(text: str, lang: str) -> list[OutlineEntry]:
    lines = text.splitlines()
    decls = _regex_declarations(lines, lang)
    out: list[OutlineEntry] = []
    stack: list[int] = []        # indentation of each enclosing declaration
    for i, (lineno, indent, kind, decl) in enumerate(decls):
        while stack and stack[-1] >= indent:
            stack.pop()
        depth = len(stack)
        stack.append(indent)
        end = len(lines)
        for next_line, next_indent, _k, _d in decls[i + 1:]:
            if next_indent <= indent:
                end = next_line - 1
                break
        while end > lineno and not lines[end - 1].strip():
            end -= 1
        if depth > 0 and kind == "function":
            kind = "method"
        out.append(OutlineEntry(kind, decl, lineno, max(end, lineno), depth))
    return out


# ---------------------------------------------------------------------------
#  Public entry point
# ---------------------------------------------------------------------------

def outline_source(text: str, suffix: str) -> Outline:
    """Outline *text*, a source file with extension *suffix* (e.g. ``".py"``)."""
    lang = _lang_for_ext(suffix.lower())
    lines = _count_lines(text)
    if lang == "python":
        entries = _outline_python_ast(text)
        if entries is not None:
            return Outline(entries, "ast", lang, lines)
    return Outline(_outline_regex(text, lang), "regex", lang, lines)


def format_outline(outline: Outline, max_chars: int) -> tuple[str, int, str]:
    """Render *outline* as text of at most *max_chars* characters.

    Returns ``(text, shown_entries, note)``. When the full outline is too long
    it is rendered one nesting level shallower until it fits; *note* says so
    (empty when nothing was cut). Past depth 0 the entries are cut from the end."""
    entries = outline.entries
    if not entries:
        return "", 0, ""
    deepest = max(e.depth for e in entries)
    width = len(str(outline.lines))

    def render(limit_depth: int, cap: int | None) -> tuple[str, int]:
        rows = [e for e in entries if e.depth <= limit_depth]
        if cap is not None:
            rows = rows[:cap]
        body = "\n".join(
            f"{'  ' * e.depth}L{e.start:>{width}}-{e.end:<{width}}  {e.text}" for e in rows)
        return body, len(rows)

    for limit in range(deepest, -1, -1):
        body, shown = render(limit, None)
        if len(body) <= max_chars:
            hidden = len(entries) - shown
            note = ""
            if hidden:
                note = (f"{hidden} deeper member(s) omitted to fit; outline a narrower "
                        "span with read_file, or grep for the name")
            return body, shown, note
    shown = len([e for e in entries if e.depth == 0])
    lo, hi = 1, shown
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(render(0, mid)[0]) <= max_chars:
            lo = mid
        else:
            hi = mid - 1
    body, shown_n = render(0, lo)
    note = (f"only the first {shown_n} of {len(entries)} entries fit; use grep or "
            "read_file with offset/limit to reach the rest")
    return body, shown_n, note
