# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GUI never opens a native browser dialog.

Every GUI script and page (the shell under localm/plugins/gui/static and every
built-in plugin's static folder, vendored libraries excluded) is read with its
comments and string contents blanked, and must contain no call of alert(),
confirm() or prompt() (bare, or through window, globalThis, self, top or
parent, dotted or bracketed), no onbeforeunload handler, and no beforeunload
listener that sets returnValue or calls preventDefault. Questions go through
the in-page dialogs in app/helpers.js instead."""

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PLUGINS = _ROOT / "localm" / "plugins"

_NAMES = ("alert", "confirm", "prompt")
_KEPT_STRINGS = set(_NAMES) | {"beforeunload"}
_REGEX_BEFORE = set("(,=:[!&|?{};+-*%<>~^")
_REGEX_KEYWORDS = {"return", "typeof", "case", "do", "else", "in", "of", "new", "delete", "void",
                   "throw", "yield", "await"}

_DIALOG_CALL = re.compile(
    r"(?:\b(?:window|globalThis|self|top|parent)\s*(?:\.\s*(?P<dot>alert|confirm|prompt)"
    r"|\[\s*\"(?P<bracket>alert|confirm|prompt)\"\s*\])"
    r"|(?<![\w.$])(?P<bare>alert|confirm|prompt))\s*\(")
_ON_BEFOREUNLOAD = re.compile(r"\bonbeforeunload\s*=")
_BEFOREUNLOAD_LISTENER = re.compile(r"addEventListener\s*\(\s*\"beforeunload\"")


def _gui_files() -> list[Path]:
    """Every GUI script and page shipped as static content, vendor excluded."""
    roots = [_PLUGINS / "gui" / "static"]
    roots += sorted(p for p in (_PLUGINS / "builtin").glob("*/static") if p.is_dir())
    out = []
    for root in roots:
        for p in sorted(root.rglob("*")):
            if p.suffix in (".js", ".mjs", ".html") and "vendor" not in p.relative_to(root).parts:
                out.append(p)
    return out


def blank_js(src: str) -> str:
    """*src* with comments, regex literal bodies and string contents replaced
    by spaces (newlines kept, so offsets and line numbers stay). A string whose
    whole content is a dialog name or "beforeunload" is kept, in double quotes.
    Code inside template-literal substitutions is kept."""
    out = []
    i, n = 0, len(src)
    stack = []  # "`" while inside a template literal, "{" for a substitution's braces
    prev = ""   # last significant code character or keyword

    def keep_string(body: str, quote_len: int) -> str:
        if body in _KEPT_STRINGS:
            return '"' + body + '"'
        return " " * quote_len + re.sub(r"[^\n]", " ", body) + " " * quote_len

    while i < n:
        c = src[i]
        if stack and stack[-1] == "`":
            if c == "\\":
                out.append("  " if src[i + 1:i + 2] != "\n" else " \n")
                i += 2
                continue
            if c == "`":
                stack.pop()
                out.append(" ")
                prev = ")"
                i += 1
                continue
            if c == "$" and src[i + 1:i + 2] == "{":
                stack.append("{")
                out.append("  ")
                prev = "{"
                i += 2
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        if c == "/" and src[i + 1:i + 2] == "/":
            j = src.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
            continue
        if c == "/" and src[i + 1:i + 2] == "*":
            j = src.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append(re.sub(r"[^\n]", " ", src[i:j]))
            i = j
            continue
        if c in "'\"":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            body = src[i + 1:j]
            if j < n and src[j] == "\n":
                out.append(" " + re.sub(r"[^\n]", " ", body))
                i = j
            else:
                out.append(keep_string(body, 1))
                i = j + 1
            prev = ")"
            continue
        if c == "`":
            stack.append("`")
            out.append(" ")
            i += 1
            continue
        if c == "/" and (prev == "" or prev in _REGEX_BEFORE or prev in _REGEX_KEYWORDS):
            j = i + 1
            in_class = False
            while j < n and src[j] != "\n":
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == "[":
                    in_class = True
                elif src[j] == "]":
                    in_class = False
                elif src[j] == "/" and not in_class:
                    break
                j += 1
            out.append(" " * (j + 1 - i))
            i = j + 1
            prev = ")"
            continue
        if c == "{" and stack:
            stack.append("{")
        elif c == "}" and stack and stack[-1] == "{":
            stack.pop()
            if stack and stack[-1] == "`":
                out.append(" ")
                i += 1
                continue
        if c.isalnum() or c in "_$":
            m = re.match(r"[\w$]+", src[i:])
            word = m.group(0)
            out.append(word)
            prev = word if word in _REGEX_KEYWORDS else "a"
            i += len(word)
            continue
        out.append(c)
        if not c.isspace():
            prev = c
        i += 1
    return "".join(out)


def blank_html(src: str) -> str:
    """*src* with every inline script and every on* attribute value blanked as
    JavaScript, and everything else (markup, text, HTML comments) blanked to
    spaces. Newlines are kept, so line numbers stay."""
    src = re.sub(r"<!--.*?-->", lambda m: re.sub(r"[^\n]", " ", m.group(0)), src, flags=re.S)
    out = list(re.sub(r"[^\n]", " ", src))
    spans = [(m.start(1), m.group(1))
             for m in re.finditer(r"<script\b[^>]*>(?=(.*?)</script>)", src, flags=re.S | re.I)]
    spans += [(m.start(1) + 1, m.group(1)[1:-1])
              for m in re.finditer(r"\son\w+\s*=\s*(\"[^\"]*\"|'[^']*')", src)]
    for start, code in spans:
        blanked = blank_js(code)
        out[start:start + len(code)] = list(blanked.ljust(len(code))[:len(code)])
    return "".join(out)


def _call_end(code: str, start: int) -> int:
    """Index just past the parenthesised call opened at or after *start*."""
    depth = 0
    for k in range(code.index("(", start), len(code)):
        if code[k] == "(":
            depth += 1
        elif code[k] == ")":
            depth -= 1
            if depth == 0:
                return k + 1
    return len(code)


def native_dialog_uses(code: str) -> list[tuple[int, str]]:
    """(line, what) for every native dialog use in blanked JavaScript *code*."""
    hits = []

    def line_of(pos: int) -> int:
        return code.count("\n", 0, pos) + 1

    for m in _DIALOG_CALL.finditer(code):
        name = m.group("dot") or m.group("bracket") or m.group("bare")
        before = code[max(0, m.start() - 9):m.start()]
        if m.group("bare") and re.search(r"\bfunction\s*$", before):
            continue
        hits.append((line_of(m.start()), f"{name}() call"))
    for m in _ON_BEFOREUNLOAD.finditer(code):
        hits.append((line_of(m.start()), "onbeforeunload handler"))
    for m in _BEFOREUNLOAD_LISTENER.finditer(code):
        body = code[m.start():_call_end(code, m.start())]
        if re.search(r"\breturnValue\b|\bpreventDefault\b", body):
            hits.append((line_of(m.start()), "beforeunload listener asking the browser to confirm"))
    return hits


def _uses_in(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    code = blank_html(text) if path.suffix == ".html" else blank_js(text)
    return native_dialog_uses(code)


def test_the_scan_covers_the_shell_and_every_plugin_static_folder():
    files = {p.relative_to(_ROOT).as_posix() for p in _gui_files()}
    for expected in ("localm/plugins/gui/static/app/helpers.js",
                     "localm/plugins/gui/static/pages/settings.js",
                     "localm/plugins/gui/static/index.html",
                     "localm/plugins/builtin/jobs/static/jobs.js",
                     "localm/plugins/builtin/tts/static/tts.js"):
        assert expected in files
    assert not any("/vendor/" in f for f in files)


def test_no_native_browser_dialog_in_gui_javascript():
    found = [f"{p.relative_to(_ROOT).as_posix()}:{line}: {what}"
             for p in _gui_files() for line, what in _uses_in(p)]
    assert found == [], "native browser dialogs in GUI code:\n" + "\n".join(found)


@pytest.mark.parametrize("snippet", [
    'if (confirm("Delete?")) go();',
    "window.confirm(msg);",
    "const ok = window . alert ( 1 );",
    'globalThis.prompt("name");',
    'self.alert("x");',
    'window["confirm"]("x");',
    "const v = await Promise.resolve(prompt(`Name ${who}`));",
    "x = `${confirm('y')}`;",
    "window.onbeforeunload = () => 'unsaved';",
    'window.addEventListener("beforeunload", (e) => { e.returnValue = ""; });',
    "addEventListener('beforeunload', function (e) { e.preventDefault(); });",
])
def test_the_scan_finds_a_native_dialog(snippet):
    assert native_dialog_uses(blank_js(snippet)), snippet


@pytest.mark.parametrize("snippet", [
    "// window.confirm() is not used here",
    "/* prompt(\"x\") */ go();",
    'const s = "press confirm(ok) to continue";',
    "deferredInstall.prompt();",
    'confirmDanger(t("x"), msg, label, onYes);',
    "await confirmDangerAsync(title, body, label);",
    "const re = /confirm\\(/; go();",
    'window.addEventListener("pagehide", storeDraft);',
    'window.addEventListener("beforeunload", storeDraft);',
    "function alert(x) { return x; }",
])
def test_the_scan_ignores_what_is_not_a_native_dialog(snippet):
    assert native_dialog_uses(blank_js(snippet)) == [], snippet


def test_the_scan_reads_inline_scripts_and_handlers_in_html():
    page = ('<p>Do not confirm(this)</p><!-- alert(1) -->'
            '<script nonce="n">if (window.confirm("x")) go();</script>'
            '<button onclick="prompt(1)">x</button>')
    assert sorted(what for _, what in native_dialog_uses(blank_html(page))) == [
        "confirm() call", "prompt() call"]
