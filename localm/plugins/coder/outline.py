# SPDX-License-Identifier: AGPL-3.0-or-later
"""Structural file outlines: the classes, functions and methods a source file
declares, each with the exact line span it occupies.

Python is parsed with ``ast``, so spans are exact. Every other supported
language (and a Python file that does not parse) goes through a regex scan
that finds declaration lines by shape and estimates each end line from
indentation; those spans are approximate and the result says so.

Line numbers are 1-based and follow ``str.splitlines`` - the same line
breaking ``read_file`` uses for its offset/limit - so a span can be passed to
``read_file`` as is.

The result feeds the ``outline_file`` coder tool (tools/outline.py), which
lets the model read a large file's shape first and then ``read_file`` just the
span it needs.
"""

from __future__ import annotations

import ast
import re
import warnings
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

from .indexer import _SYMBOL_LANGS, _lang_for_ext


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
    method:  str        # "ast", "regex", or "unsupported" (no outline for this file type)
    lang:    str
    lines:   int        # total lines, counted the way read_file reports them


@dataclass(frozen=True)
class FormattedOutline:
    body:      str
    shown:     int           # entries rendered
    total:     int           # entries that matched (end line >= from_line)
    next_from: int | None    # from_line that continues after a cut, None when nothing was cut


_MAX_DECL_CHARS = 110   # a declaration longer than this is shown cut short

# Extensions the indexer has no symbol language for but that share a grammar
# with one it does.
_EXT_LANG: dict[str, str] = {
    ".h": "c", ".hpp": "cpp", ".hh": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".mjs": "javascript", ".cjs": "javascript", ".pyi": "python",
}


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


def _py_start(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return min([node.lineno, *(d.lineno for d in node.decorator_list)])


_Match: Any = getattr(ast, "Match", None)
_TryStar: Any = getattr(ast, "TryStar", None)


def _py_collect_try(node: Any, depth: int, in_class: bool, out: list[OutlineEntry]) -> None:
    """A ``try`` / ``try*`` statement's body, handlers, else and finally blocks."""
    _py_collect(node.body, depth, in_class, out)
    for handler in node.handlers:
        _py_collect(handler.body, depth, in_class, out)
    _py_collect(node.orelse, depth, in_class, out)
    _py_collect(node.finalbody, depth, in_class, out)


def _py_collect_match(node: Any, depth: int, in_class: bool, out: list[OutlineEntry]) -> None:
    for case in node.cases:
        _py_collect(case.body, depth, in_class, out)


def _py_collect(body: list[ast.stmt], depth: int, in_class: bool,
                out: list[OutlineEntry]) -> None:
    """Declarations in *body*, descending into class bodies and into the
    compound statements (``if``, ``try``, ``with``, ``match``) that wrap
    declarations at module or class level. A function body is not descended:
    its nested helpers are implementation detail, not structure."""
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
        elif isinstance(node, ast.Try) or (_TryStar is not None and isinstance(node, _TryStar)):
            _py_collect_try(node, depth, in_class, out)
        elif _Match is not None and isinstance(node, _Match):
            _py_collect_match(node, depth, in_class, out)


_AST_LINE_BREAK = re.compile(r"\r\n|\r|\n")


def _line_translator(text: str):
    """A function mapping an ``ast`` line number to the line number
    ``str.splitlines`` (and so ``read_file``) gives the same position, or None
    when the two agree for this text.

    ``ast`` breaks lines only at \\n, \\r and \\r\\n; ``splitlines`` also breaks
    at form feed, vertical tab, the file/group/record separators, NEL and the
    Unicode line and paragraph separators."""
    ast_starts = [0] + [m.end() for m in _AST_LINE_BREAK.finditer(text)]
    split_starts = [0]
    for piece in text.splitlines(keepends=True):
        split_starts.append(split_starts[-1] + len(piece))
    split_starts.pop()
    if ast_starts[:len(split_starts)] == split_starts and \
            len(ast_starts) - len(split_starts) <= 1:
        return None

    def translate(lineno: int) -> int:
        offset = ast_starts[min(lineno, len(ast_starts)) - 1]
        return bisect_right(split_starts, offset)

    return translate


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
    translate = _line_translator(text)
    if translate is None:
        return out
    return [OutlineEntry(e.kind, e.text, translate(e.start), translate(e.end), e.depth)
            for e in out]


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
                rf"({_IDENT})\s*(?:<[^>()]*>)?\([^()]*\)\s*(?::\s*[^{{;=()]+)?\{{\s*$"),
   )

_p(("python",),
   ("class",    rf"^(\s*)class\s+({_IDENT})"),
   ("function", rf"^(\s*)(?:async\s+)?def\s+({_IDENT})"),
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

_JVM_MODS = (r"(?:(?:public|private|protected|internal|static|final|abstract|sealed|"
             r"partial|open|data|override|virtual|synchronized|native|suspend|async|"
             r"extern|readonly|unsafe)\s+)*")
_CLASS_LIKE = rf"^(\s*){_JVM_MODS}(?:class|interface|enum|record|struct|object|protocol|extension)\s+({_IDENT})"
# A return type, optionally generic (bounded) and/or an array or nullable.
_JVM_TYPE = r"[\w.]+(?:<[^()]{0,80}>)?(?:\[\])*\??"
_p(("java", "csharp"),
   ("class",    _CLASS_LIKE),
   ("method",   rf"^(\s*)(?!{_STATEMENT_START}){_JVM_MODS}{_JVM_TYPE}\s+({_IDENT})\s*\([^;]*$"),
   )
_p(("kotlin", "swift"),
   ("class",    _CLASS_LIKE),
   ("function", rf"^(\s*){_JVM_MODS}(?:fun|func)\s+(?:<[^>]*>\s*)?({_IDENT})"),
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

_MAX_REGEX_LINE = 400   # a longer line (minified code) is not scanned


def _lang_of(suffix: str) -> str:
    suffix = suffix.lower()
    return _EXT_LANG.get(suffix) or _lang_for_ext(suffix)


def supported_suffixes() -> list[str]:
    """Extensions outline_source() can outline, sorted."""
    exts = {ext for ext, lang in _SYMBOL_LANGS.items()
            if lang == "python" or lang in _PATTERNS}
    exts |= {ext for ext, lang in _EXT_LANG.items() if lang == "python" or lang in _PATTERNS}
    return sorted(exts)


def _regex_declarations(lines: list[str], lang: str) -> list[tuple[int, int, str, str]]:
    """(1-based line, indent width, kind, declaration text) per declaration line."""
    patterns = _PATTERNS.get(lang, [])
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
            if name in _NOT_A_DECLARATION and (kind == "method" or lang in ("c", "cpp")):
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
#  Public entry points
# ---------------------------------------------------------------------------

def outline_source(text: str, suffix: str) -> Outline:
    """Outline *text*, a source file with extension *suffix* (e.g. ``".py"``).

    The result's method is "unsupported" (and entries empty) for a file type
    with no outline support, so a caller can tell that apart from a file that
    declares nothing."""
    text = text.removeprefix(chr(0xFEFF))
    lang = _lang_of(suffix)
    lines = text.count("\n") + 1
    if lang == "python":
        entries = _outline_python_ast(text)
        if entries is not None:
            return Outline(entries, "ast", lang, lines)
        return Outline(_outline_regex(text, "python"), "regex", lang, lines)
    if lang not in _PATTERNS:
        return Outline([], "unsupported", lang, lines)
    return Outline(_outline_regex(text, lang), "regex", lang, lines)


def format_outline(outline: Outline, max_chars: int, from_line: int = 1) -> FormattedOutline:
    """Render the entries of *outline* that end at or after *from_line*, as text
    of at most *max_chars* characters, in file order.

    An enclosing class whose span covers *from_line* is included, so a page that
    starts inside a class still shows which class it is. When the entries do not
    all fit, the rest are cut from the end and ``next_from`` is the *from_line*
    that continues from the first one left out."""
    rows = [e for e in outline.entries if e.end >= from_line]
    used = 0
    lines: list[str] = []
    next_from: int | None = None
    for e in rows:
        line = f"{'  ' * e.depth}L{e.start}-{e.end}  {e.text}"
        cost = len(line) + (1 if lines else 0)
        if lines and used + cost > max_chars:
            next_from = e.start
            break
        lines.append(line)
        used += cost
    return FormattedOutline("\n".join(lines), len(lines), len(rows), next_from)
