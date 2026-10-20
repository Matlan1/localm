# SPDX-License-Identifier: AGPL-3.0-or-later
"""outline_file: the structural outline extractor (outline.py), the tool
(tools/outline.py), and how the tool is wired into the registry, the
restricted-session allowlist and the --scope gate."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from localm.plugins.coder.agent import Agent, _SCOPED_TOOLS
from localm.plugins.coder.outline import format_outline, outline_source
from localm.plugins.coder.tools import SAFE_RESTRICTED_TOOLS, TOOL_REGISTRY
from localm.plugins.coder.tools.outline import tool_outline_file
from localm.plugins.coder.roles import ROLE_PRESETS

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
'''


def _by_name(entries):
    return {e.text.split("(")[0].split()[-1]: e for e in entries}


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
    assert shape.end == 31
    assert shape.text == "class Shape(Base, metaclass=Meta)"
    assert (by["area"].kind, by["area"].start, by["area"].end, by["area"].depth) == ("method", 19, 20, 1)
    assert by["name"].start == 22, "a decorated method starts at its decorator"
    assert by["helper"].depth == 2 and by["Meta"].depth == 1
    assert out.lines == _PY.count("\n")


def test_python_outline_does_not_list_nested_function_helpers():
    names = [e.text for e in outline_source(_PY, ".py").entries]
    assert not any("inner" in n for n in names)


def test_python_outline_descends_into_if_and_try_blocks():
    by = _by_name(outline_source(_PY, ".py").entries)
    assert {"conditional", "in_try", "in_except"} <= set(by)
    assert by["conditional"].depth == 0


def test_python_syntax_error_falls_back_to_regex_and_says_so():
    broken = "def ok():\n    pass\n\nclass Broken(:\n    def m(self):\n        pass\n"
    out = outline_source(broken, ".py")
    assert out.method == "regex"
    texts = [e.text for e in out.entries]
    assert "def ok()" in texts[0] and any("class Broken" in t for t in texts)


def test_python_null_byte_does_not_raise():
    out = outline_source("def a():\n    pass\n\x00\n", ".py")
    assert out.method in ("ast", "regex")


def test_empty_file_has_no_entries():
    out = outline_source("", ".py")
    assert out.entries == [] and out.lines == 0


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
    got = {e.text.split("(")[0].split()[-1]: e for e in out.entries}
    assert {"Widget", "load", "add", "mul", "render", "constructor"} <= set(got)
    assert got["Widget"].kind == "class" and got["Widget"].depth == 0
    assert got["render"].kind == "method" and got["render"].depth == 1
    assert got["Widget"].start == 1 and got["Widget"].end == 10
    assert "if" not in got, "control flow must not read as a method"


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


def test_c_outline_does_not_mistake_statements_for_functions():
    src = "int add(int a, int b)\n{\n    return a + b;\n}\n\nreturn foo(1)\n"
    names = [e.text for e in outline_source(src, ".c").entries]
    assert names == ["int add(int a, int b)"]


def test_long_line_is_not_scanned():
    src = "function a() {}\n" + "function b() {" + "x" * 1000 + "}\n"
    texts = [e.text for e in outline_source(src, ".js").entries]
    assert texts == ["function a() {}"]


def test_unknown_extension_uses_the_generic_pattern():
    out = outline_source("def alpha\nfn beta\nclass Gamma\n", ".zzz")
    assert out.method == "regex" and len(out.entries) == 3


def test_regex_end_line_is_before_the_next_sibling_and_skips_trailing_blanks():
    src = "function a() {\n  x;\n}\n\n\nfunction b() {\n  y;\n}\n"
    a, b = outline_source(src, ".js").entries
    assert (a.start, a.end) == (1, 3)
    assert (b.start, b.end) == (6, 8)


# ---------------------------------------------------------------------------
#  format_outline: stays inside its character budget
# ---------------------------------------------------------------------------

def _big_python(n_classes: int, n_methods: int) -> str:
    parts = []
    for c in range(n_classes):
        parts.append(f"class C{c}:\n")
        for m in range(n_methods):
            parts.append(f"    def method_{m}(self):\n        pass\n")
        parts.append("\n")
    return "".join(parts)


def test_format_outline_drops_depth_before_dropping_entries():
    out = outline_source(_big_python(20, 10), ".py")
    full, shown_full, note_full = format_outline(out, 10 ** 6)
    assert shown_full == 220 and note_full == ""
    body, shown, note = format_outline(out, len(full) - 1)
    assert len(body) <= len(full) - 1
    assert shown == 20, "only the classes remain once methods no longer fit"
    assert "deeper member(s) omitted" in note and "200" in note


def test_format_outline_cuts_top_level_entries_as_a_last_resort():
    out = outline_source(_big_python(300, 0), ".py")
    body, shown, note = format_outline(out, 500)
    assert len(body) <= 500 and 0 < shown < 300
    assert f"only the first {shown} of 300" in note


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
    assert "L15-31" in r.output.replace(" ", "")
    assert "read_file(path, offset=" in r.output
    assert "(ast)" in r.summary


def test_tool_labels_regex_results_as_approximate(tmp_path):
    (tmp_path / "a.js").write_text(_JS, encoding="utf-8")
    r = tool_outline_file(tmp_path, "a.js")
    assert r.ok
    assert "approximate" in r.output and 'method="ast"' not in r.output


def test_tool_says_when_nothing_was_found(tmp_path):
    (tmp_path / "n.py").write_text("X = 1\n", encoding="utf-8")
    r = tool_outline_file(tmp_path, "n.py")
    assert r.ok and "no class or function declarations found" in r.output


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


def test_tool_output_stays_under_the_tool_output_cap(tmp_path):
    from localm.plugins.coder.tools.base import _MAX_OUTPUT
    (tmp_path / "big.py").write_text(_big_python(400, 5), encoding="utf-8")
    r = tool_outline_file(tmp_path, "big.py")
    assert r.ok and r.truncated is True
    assert len(r.output) <= _MAX_OUTPUT
    assert "[outline cut:" in r.output


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
