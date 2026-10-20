# SPDX-License-Identifier: AGPL-3.0-or-later
"""outline_file: the structural outline extractor (outline.py), the tool
(tools/outline.py), and how the tool is wired into the registry, the
restricted-session allowlist and the --scope gate."""

from __future__ import annotations

import re
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from localm.plugins.coder.agent import Agent, _SCOPED_TOOLS
from localm.plugins.coder.outline import format_outline, outline_source
from localm.plugins.coder.roles import ROLE_PRESETS
from localm.plugins.coder.tools import SAFE_RESTRICTED_TOOLS, TOOL_REGISTRY
from localm.plugins.coder.tools.base import _MAX_OUTPUT
from localm.plugins.coder.tools.files import tool_read_file
from localm.plugins.coder.tools.outline import tool_outline_file

_PY = '''\
"""module doc"""
import os


def top(a, b=1) -> int:
    def inner():
        return 1
    return a + b


async def fetch(url: str):
    pass


@decorator
@other(arg)
class Shape(Base, metaclass=Meta):
    """doc"""

    def area(self) -> float:
        return 0.0

    @property
    def name(self):
        return "x"

    class Meta:
        def helper(self):
            pass


if True:
    def conditional():
        pass

try:
    def in_try():
        pass
except ImportError:
    def in_except():
        pass

match os.name:
    case "nt":
        def in_match():
            pass
'''


def _by_name(entries):
    return {e.text.split("(")[0].split()[-1]: e for e in entries}


def _tuples(out):
    return [(e.kind, e.text, e.start, e.end, e.depth) for e in out.entries]


def test_python_outline_has_exact_spans_and_nesting():
    out = outline_source(_PY, ".py")
    assert out.method == "ast" and out.lang == "python"
    by = _by_name(out.entries)
    assert (by["top"].kind, by["top"].start, by["top"].end, by["top"].depth) == ("function", 5, 8, 0)
    assert (by["fetch"].start, by["fetch"].end) == (11, 12)
    assert by["fetch"].text.startswith("async def fetch")
    shape = by["Shape"]
    assert shape.kind == "class" and shape.depth == 0
    assert shape.start == 15, "a decorated class starts at its first decorator"
    assert shape.end == 29
    assert shape.text == "class Shape(Base, metaclass=Meta)"
    assert (by["area"].kind, by["area"].start, by["area"].end, by["area"].depth) == ("method", 20, 21, 1)
    assert by["name"].start == 23, "a decorated method starts at its decorator"
    assert by["helper"].depth == 2 and by["Meta"].depth == 1
    assert out.lines == _PY.count("\n") + 1


def test_python_outline_does_not_list_nested_function_helpers():
    names = [e.text for e in outline_source(_PY, ".py").entries]
    assert not any("inner" in n for n in names)


def test_python_outline_descends_into_if_try_and_match_blocks():
    by = _by_name(outline_source(_PY, ".py").entries)
    assert {"conditional", "in_try", "in_except", "in_match"} <= set(by)
    assert by["conditional"].depth == 0


def test_python_syntax_error_falls_back_to_regex_and_says_so():
    broken = "def ok():\n    pass\n\nasync def also():\n    pass\n\nclass Broken(:\n    def m(self):\n        pass\n"
    out = outline_source(broken, ".py")
    assert out.method == "regex"
    texts = [e.text for e in out.entries]
    assert texts[0] == "def ok()" and "async def also()" in texts
    assert any(t.startswith("class Broken") for t in texts)


def test_python_null_byte_does_not_raise():
    out = outline_source("def a():\n    pass\n\x00\n", ".py")
    assert out.method in ("ast", "regex")


def test_empty_file_has_no_entries():
    out = outline_source("", ".py")
    assert out.entries == [] and out.method == "ast"


_BOM_SRC = (
    "import os\n\n\nclass A:\n    @property\n    def p(self):\n        return '''\ndef fake():\n    pass\n'''\n\n"
    "    def long(self,\n             a):\n        return a\n\n\ndef top():\n    pass\n"
)


def test_a_utf8_bom_does_not_push_python_off_the_exact_parser():
    plain = outline_source(_BOM_SRC, ".py")
    with_bom = outline_source("\ufeff" + _BOM_SRC, ".py")
    assert plain.method == with_bom.method == "ast"
    assert _tuples(with_bom) == _tuples(plain)
    assert not any("fake" in e.text for e in with_bom.entries)


def test_a_bom_hides_no_declaration_on_line_one_for_regex_languages():
    out = outline_source("\ufefffunction f() {}\n", ".js")
    assert [e.text for e in out.entries] == ["function f() {}"]


def test_spans_follow_the_line_breaks_read_file_uses(tmp_path):
    src = "def a():\n    pass\n\x0c\ndef b():\n    pass\n"
    out = outline_source(src, ".py")
    b = [e for e in out.entries if e.text == "def b()"][0]
    (tmp_path / "f.py").write_text(src, encoding="utf-8", newline="")
    got = tool_read_file(tmp_path, "f.py", offset=b.start, limit=b.end - b.start + 1)
    assert got.output.split("<content>\n", 1)[1].startswith("def b():\n    pass")
    uni = outline_source('X = "a\u2028b"\ndef b():\n    pass\n', ".py")
    assert [(e.text, e.start, e.end) for e in uni.entries] == [("def b()", 3, 4)]


def test_crlf_and_lone_cr_line_numbers_match_the_unix_ones():
    unix = outline_source("def a():\n    pass\n\ndef b():\n    pass\n", ".py")
    crlf = outline_source("def a():\r\n    pass\r\n\r\ndef b():\r\n    pass\r\n", ".py")
    cr = outline_source("def a():\r    pass\r\rdef b():\r    pass\r", ".py")
    assert _tuples(unix) == _tuples(crlf) == _tuples(cr)


_JS = '''\
export class Widget extends Base {
  constructor(a) {
    super(a);
  }
  render() {
    if (this.x) {
      return 1;
    }
  }
}

export async function load(url) {
  return fetch(url);
}

const add = (a, b) => a + b;
export const mul = async (a, b) => {
  return a * b;
};
interface Opts { a: number }
'''


def test_javascript_regex_outline_finds_classes_functions_and_methods():
    out = outline_source(_JS, ".js")
    assert out.method == "regex"
    assert [(e.kind, e.text, e.start, e.depth) for e in out.entries] == [
        ("class", "export class Widget extends Base", 1, 0),
        ("method", "constructor(a)", 2, 1),
        ("method", "render()", 5, 1),
        ("function", "export async function load(url)", 12, 0),
        ("function", "const add = (a, b) => a + b;", 16, 0),
        ("function", "export const mul = async (a, b) =>", 17, 0),
        ("interface", "interface Opts { a: number }", 20, 0),
    ]
    widget = out.entries[0]
    assert (widget.start, widget.end) == (1, 10)
    assert next(e for e in out.entries if e.text.startswith("const add")).end == 16


def test_mjs_cjs_and_header_extensions_are_outlined():
    assert outline_source("function f() {}\n", ".mjs").method == "regex"
    assert outline_source("function f() {}\n", ".cjs").entries
    assert outline_source("int add(int a, int b);\nstruct S {\n};\n", ".h").entries[0].kind == "class"
    assert outline_source("def f(): ...\n", ".pyi").method == "ast"


def test_go_and_rust_and_java_outlines():
    go = outline_source("package x\n\ntype S struct {\n}\n\nfunc (s *S) Run() {\n}\n\nfunc main() {\n}\n", ".go")
    assert [(e.kind, e.text, e.start, e.end) for e in go.entries] == [
        ("type", "type S struct", 3, 4),
        ("function", "func (s *S) Run()", 6, 7),
        ("function", "func main()", 9, 10),
    ]
    rs = outline_source("pub struct P;\n\nimpl P {\n    pub fn new() -> P {\n        P\n    }\n}\n\nfn main() {}\n", ".rs")
    kinds = {(e.kind, e.depth) for e in rs.entries}
    assert ("struct", 0) in kinds and ("impl", 0) in kinds and ("method", 1) in kinds
    java = outline_source(
        "public class A {\n    public int f(int x) {\n        return g(x);\n    }\n}\n", ".java")
    assert [e.kind for e in java.entries] == ["class", "method"]
    assert java.entries[1].text.startswith("public int f(")


def test_a_method_named_like_a_keyword_is_still_declared():
    rs = outline_source("impl P {\n    pub fn new() -> P {\n        P\n    }\n    fn try(&self) {\n    }\n}\n", ".rs")
    assert [e.text for e in rs.entries] == ["impl P", "pub fn new() -> P", "fn try(&self)"]
    rb = outline_source("class A\n  def when(x)\n  end\n  def self.new\n  end\nend\n", ".rb")
    assert [e.text for e in rb.entries] == ["class A", "def when(x)", "def self.new"]


def test_c_outline_does_not_mistake_statements_for_functions():
    src = "int add(int a, int b)\n{\n    return a + b;\n}\n\nreturn foo(1)\n"
    names = [e.text for e in outline_source(src, ".c").entries]
    assert names == ["int add(int a, int b)"]


def test_call_statements_are_not_declared_as_methods():
    kotlin = ("class A {\n  fun f() {\n    setContentView(R.layout.main)\n    initViews()\n"
              "    println(\"x\")\n  }\n}\n")
    assert [e.text for e in outline_source(kotlin, ".kt").entries] == ["class A", "fun f()"]
    swift = ("class V {\n  func load() {\n    setupUI()\n    guard isReady(x) else { return }\n  }\n}\n")
    assert [e.text for e in outline_source(swift, ".swift").entries] == ["class V", "func load()"]
    java = ("class A {\n  void f() {\n    assertEquals(\n    doSomething(a,\n    super(\n"
            "    String s = make(\n    Foo x = new Foo(\n  }\n}\n")
    assert [e.text for e in outline_source(java, ".java").entries] == ["class A", "void f()"]
    js = ("describe('x', function() {\n  beforeEach(function () {\n    foo(function () {\n"
          "      setTimeout(function() {\n  it('works', function () {\n")
    assert outline_source(js, ".js").entries == []


def test_prose_in_a_markdown_file_is_not_outlined():
    out = outline_source("function names are bad\nclass of 1999\n", ".md")
    assert out.method == "unsupported" and out.entries == []


def test_long_line_is_not_scanned():
    src = "function a() {}\n" + "function b() {" + "x" * 1000 + "}\n"
    texts = [e.text for e in outline_source(src, ".js").entries]
    assert texts == ["function a() {}"]


def test_hostile_whitespace_lines_do_not_stall_any_language():
    line = " " * 200 + "a " * 100 + "\n"
    for suffix in (".java", ".cs", ".kt", ".swift", ".js", ".ts", ".c", ".cpp", ".rs", ".go",
                   ".rb", ".php", ".lua", ".sh", ".py"):
        src = line * 150
        t0 = time.perf_counter()
        outline_source(src, suffix)
        outline_source(src.replace("a a", "a ( a"), suffix)
        assert time.perf_counter() - t0 < 2.0, f"{suffix} regex scan is superlinear"


def test_regex_end_line_is_before_the_next_sibling_and_skips_trailing_blanks():
    src = "function a() {\n  x;\n}\n\n\nfunction b() {\n  y;\n}\n"
    a, b = outline_source(src, ".js").entries
    assert (a.start, a.end) == (1, 3)
    assert (b.start, b.end) == (6, 8)


# ---------------------------------------------------------------------------
#  format_outline: stays inside its character budget and pages
# ---------------------------------------------------------------------------

def _big_python(n_classes: int, n_methods: int) -> str:
    parts = []
    for c in range(n_classes):
        parts.append(f"class C{c}:\n")
        for m in range(n_methods):
            parts.append(f"    def method_{m}(self):\n        pass\n")
        parts.append("\n")
    return "".join(parts)


def test_format_outline_cuts_from_the_end_and_names_the_continuation():
    out = outline_source(_big_python(20, 10), ".py")
    full = format_outline(out, 10 ** 6)
    assert full.shown == full.total == 220 and full.next_from is None
    cut = format_outline(out, 600)
    assert len(cut.body) <= 600 and 0 < cut.shown < 220
    assert cut.next_from is not None
    assert "method_" in cut.body, "members must not be dropped wholesale"


def test_format_outline_pages_reach_every_declaration():
    out = outline_source(_big_python(40, 12), ".py")
    seen: set[int] = set()
    from_line, pages = 1, 0
    while from_line is not None:
        page = format_outline(out, 900, from_line)
        pages += 1
        assert pages < 200, "paging must terminate"
        seen |= {int(n) for n in re.findall(r"L\s*(\d+)-", page.body)}
        from_line = page.next_from
    assert {e.start for e in out.entries} <= seen
    assert pages > 3


def test_format_outline_keeps_the_enclosing_class_on_a_continuation_page():
    out = outline_source(_big_python(1, 40), ".py")
    first = format_outline(out, 700)
    assert first.next_from is not None
    second = format_outline(out, 700, first.next_from)
    assert second.body.splitlines()[0].endswith("class C0")


def test_format_outline_nothing_from_a_line_past_the_end():
    out = outline_source("def a():\n    pass\n", ".py")
    page = format_outline(out, 1000, 99)
    assert page.shown == 0 and page.body == "" and page.next_from is None


# ---------------------------------------------------------------------------
#  The tool
# ---------------------------------------------------------------------------

def test_tool_reports_spans_and_the_ast_label(tmp_path):
    (tmp_path / "m.py").write_text(_PY, encoding="utf-8")
    r = tool_outline_file(tmp_path, "m.py")
    assert r.ok and r.truncated is False
    assert '<path>m.py</path>' in r.output
    assert 'method="ast"' in r.output
    assert "class Shape(Base, metaclass=Meta)" in r.output
    assert "L15-29" in r.output.replace(" ", "")
    assert "read_file(path, offset=" in r.output
    assert "(ast)" in r.summary
    assert f'lines="{_PY.count(chr(10)) + 1}"' in r.output, "the line total matches read_file's"


def test_tool_labels_regex_results_as_approximate(tmp_path):
    (tmp_path / "a.js").write_text(_JS, encoding="utf-8")
    r = tool_outline_file(tmp_path, "a.js")
    assert r.ok
    assert "approximate" in r.output and 'method="ast"' not in r.output


def test_tool_outlines_a_python_file_with_a_bom_exactly(tmp_path):
    (tmp_path / "b.py").write_bytes(b"\xef\xbb\xbf" + _BOM_SRC.encode("utf-8"))
    r = tool_outline_file(tmp_path, "b.py")
    assert r.ok and 'method="ast"' in r.output and "fake" not in r.output


def test_tool_says_when_nothing_was_found(tmp_path):
    (tmp_path / "n.py").write_text("X = 1\n", encoding="utf-8")
    r = tool_outline_file(tmp_path, "n.py")
    assert r.ok and "no class or function declarations found" in r.output


def test_tool_says_when_the_file_type_is_not_supported(tmp_path):
    (tmp_path / "notes.md").write_text("# t\nfunction names are bad\n", encoding="utf-8")
    r = tool_outline_file(tmp_path, "notes.md")
    assert not r.ok and "no support for .md" in r.output and ".py" in r.output
    (tmp_path / "Makefile").write_text("all:\n", encoding="utf-8")
    assert "extensionless" in tool_outline_file(tmp_path, "Makefile").output


def test_tool_refuses_paths_outside_cwd(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (tmp_path / "secret.py").write_text("def leak():\n    pass\n", encoding="utf-8")
    r = tool_outline_file(root, "../secret.py")
    assert not r.ok and "outside the working directory" in r.output
    assert "leak" not in r.output


def test_tool_missing_directory_and_oversize(tmp_path):
    assert not tool_outline_file(tmp_path, "nope.py").ok
    (tmp_path / "d").mkdir()
    r = tool_outline_file(tmp_path, "d")
    assert not r.ok and "directory" in r.output
    big = tmp_path / "big.py"
    big.write_bytes(b"#" * 2_000_001)
    r = tool_outline_file(tmp_path, "big.py")
    assert not r.ok and "outline limit" in r.output


def test_tool_rejects_a_non_numeric_from_line(tmp_path):
    (tmp_path / "m.py").write_text("def a():\n    pass\n", encoding="utf-8")
    r = tool_outline_file(tmp_path, "m.py", from_line="abc")
    assert not r.ok and "from_line" in r.output


def test_tool_output_stays_under_the_cap_and_pages_with_from_line(tmp_path):
    (tmp_path / "big.py").write_text(_big_python(400, 5), encoding="utf-8")
    r = tool_outline_file(tmp_path, "big.py")
    assert r.ok and r.truncated is True
    assert len(r.output) <= _MAX_OUTPUT
    assert "continue with outline_file(path, from_line=" in r.output
    start = int(r.output.rsplit("from_line=", 1)[1].split(")")[0])
    second = tool_outline_file(tmp_path, "big.py", from_line=start)
    assert second.ok and len(second.output) <= _MAX_OUTPUT
    assert f"L{start}" in second.output.replace(" ", "")


def test_tool_output_cap_holds_for_a_very_long_path_and_the_regex_label(tmp_path):
    deep = tmp_path
    for _ in range(5):
        deep = deep / ("d" * 24)
    deep.mkdir(parents=True)
    (deep / "big.go").write_text("".join(f"func f{i}() {{\n}}\n" for i in range(500)), encoding="utf-8")
    rel = deep.relative_to(tmp_path).as_posix() + "/big.go"
    assert len(rel) > 125
    r = tool_outline_file(tmp_path, rel)
    assert r.ok and r.truncated is True
    assert len(r.output) <= _MAX_OUTPUT


def test_tool_from_line_past_the_end_says_so(tmp_path):
    (tmp_path / "m.py").write_text("def a():\n    pass\n", encoding="utf-8")
    r = tool_outline_file(tmp_path, "m.py", from_line=50)
    assert r.ok and "no declarations end at or after line 50" in r.output


def test_tool_handles_a_non_utf8_file(tmp_path):
    (tmp_path / "l.py").write_bytes(b"def caf\xe9():\n    pass\n")
    r = tool_outline_file(tmp_path, "l.py")
    assert r.ok and "def caf" in r.output


# ---------------------------------------------------------------------------
#  Registry, restricted allowlist, scope, roles
# ---------------------------------------------------------------------------

def test_outline_file_is_registered_with_a_required_path():
    tool = TOOL_REGISTRY["outline_file"]
    assert tool.params["path"]["required"] is True
    assert tool.params["from_line"]["required"] is False
    assert not tool.destructive and not tool.ask_by_default


def test_outline_file_is_in_the_restricted_allowlist_and_read_only_roles():
    assert "outline_file" in SAFE_RESTRICTED_TOOLS
    assert "outline_file" in ROLE_PRESETS["reviewer"].allowed_tools


def _agent(tmp_path: Path, **kwargs) -> Agent:
    backend = MagicMock()
    backend.model_id = "test-model"
    with patch("localm.plugins.coder.agent.ProjectMap") as pm, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        pm.build.return_value.file_count.return_value = 0
        return Agent(backend=backend, cwd=tmp_path, **kwargs)


def _call(name: str, **args):
    c = MagicMock()
    c.name = name
    c.args = args
    return c


def test_a_restricted_session_may_run_outline_file(tmp_path):
    a = _agent(tmp_path, restricted=True)
    assert "outline_file" not in a.disabled_tools
    assert "run_shell" in a.disabled_tools


def test_outline_file_is_confined_by_scope(tmp_path):
    assert "outline_file" in _SCOPED_TOOLS
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "ok.py").write_text("def f():\n    pass\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("def g():\n    pass\n", encoding="utf-8")
    a = _agent(tmp_path, scope="src/**")
    inside = a._execute_tool(_call("outline_file", path="src/ok.py"), interactive=False)
    assert inside.ok and "def f()" in inside.output
    outside = a._execute_tool(_call("outline_file", path="other.py"), interactive=False)
    assert not outside.ok and "outside the active scope" in outside.output
    assert "def g" not in outside.output
