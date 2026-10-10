# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every JSON parse in ``localm/`` is covered against hostile nesting.

``json.loads`` raises ``RecursionError`` (a ``RuntimeError``, not a ``ValueError``)
for deeply nested input. A parse is covered when an enclosing ``try`` or
``contextlib.suppress`` handles ``Exception``, ``BaseException`` or
``RecursionError``, or when the parse is in ``_CALLER_HANDLES``/``_NOT_HOSTILE``
below with the reason it is safe. A new parse that is none of these fails here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "localm"

_BROAD = {"Exception", "BaseException", "RecursionError"}
_JSON_BASES = {"json", "_json", "tomllib", "tomli"}

# Parses with no enclosing handler of their own, and why that is safe. A key is
# "<path relative to localm/>::<function>".
_CALLER_HANDLES = {
    "inference/backends/llamacpp/_loader.py::_parse":
        "only called by _probe_roundtrip, which catches Exception",
    "plugins/builtin/chat/plug.py::_conv_meta":
        "only called inside a try that catches Exception",
    "inference/backends/_hf_fp8.py::_safetensors_header":
        "only called by expanded_bf16_bytes, which catches RecursionError",
}
_NOT_HOSTILE = {
    "plugins/gui/routes/models/acquisition.py::_build_check_workflow":
        "parses the workflow templates shipped with localm",
}


def _handler_names(handler: ast.ExceptHandler) -> set[str]:
    if handler.type is None:
        return {"BaseException"}
    nodes = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return {getattr(n, "attr", None) or getattr(n, "id", None) for n in nodes}


def _suppress_names(node: ast.With) -> set[str]:
    names: set[str] = set()
    for item in node.items:
        call = item.context_expr
        if isinstance(call, ast.Call) and (
                getattr(call.func, "attr", None) == "suppress"
                or getattr(call.func, "id", None) == "suppress"):
            for arg in call.args:
                names.add(getattr(arg, "attr", None) or getattr(arg, "id", None))
    return names


def _is_parse(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Attribute):
        if func.attr in ("load", "loads"):
            return isinstance(func.value, ast.Name) and func.value.id in _JSON_BASES
        return func.attr == "json" and not call.args and not call.keywords
    return False


def unguarded_parses(source: str) -> list[tuple[int, str, str]]:
    """``(line, function, why)`` for each parse in *source* that no enclosing
    handler protects against ``RecursionError``. ``why`` is ``"narrow"`` when a
    handler exists but misses it, ``"none"`` when there is no handler."""
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree)
               for child in ast.iter_child_nodes(node)}
    found = []
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and _is_parse(call)):
            continue
        covered = narrow = False
        function = "<module>"
        cur = call
        while cur in parents:
            parent = parents[cur]
            if isinstance(parent, ast.Try) and cur in parent.body:
                caught = set().union(*(_handler_names(h) for h in parent.handlers)) \
                    if parent.handlers else set()
                if caught & _BROAD:
                    covered = True
                    break
                if caught:
                    narrow = True
            if isinstance(parent, ast.With) and cur in parent.body:
                caught = _suppress_names(parent)
                if caught & _BROAD:
                    covered = True
                    break
                if caught:
                    narrow = True
            if function == "<module>" and isinstance(
                    parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function = parent.name
            cur = parent
        if not covered:
            found.append((call.lineno, function, "narrow" if narrow else "none"))
    return found


def _scan_package() -> list[str]:
    hits = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        try:
            found = unguarded_parses(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for line, function, why in found:
            key = f"{rel}::{function}"
            if why == "none" and (key in _CALLER_HANDLES or key in _NOT_HOSTILE):
                continue
            hits.append(f"{rel}:{line} ({function}, {why})")
    return hits


def test_no_json_parse_in_the_package_is_unguarded_against_deep_nesting():
    assert _scan_package() == []


def test_every_exemption_still_names_an_unguarded_parse():
    stale = []
    for key in {**_CALLER_HANDLES, **_NOT_HOSTILE}:
        rel, function = key.split("::")
        found = unguarded_parses((PACKAGE / rel).read_text(encoding="utf-8"))
        if not any(fn == function and why == "none" for _, fn, why in found):
            stale.append(key)
    assert stale == []


# ---------------------------------------------------------------- the scanner

@pytest.mark.parametrize("source, expected", [
    ("import json\nx = json.loads(s)\n", [(2, "<module>", "none")]),
    ("import json\ndef f(s):\n    return json.loads(s)\n", [(3, "f", "none")]),
    ("import json\ntry:\n    json.loads(s)\nexcept ValueError:\n    pass\n",
     [(3, "<module>", "narrow")]),
    ("import json\ntry:\n    json.loads(s)\nexcept json.JSONDecodeError:\n    pass\n",
     [(3, "<module>", "narrow")]),
    ("import json, contextlib\nwith contextlib.suppress(ValueError):\n    json.loads(s)\n",
     [(3, "<module>", "narrow")]),
    ("import json\nr.json()\n", [(2, "<module>", "none")]),
])
def test_the_scanner_flags_what_it_should(source, expected):
    assert unguarded_parses(source) == expected


@pytest.mark.parametrize("source", [
    "import json\ntry:\n    json.loads(s)\nexcept (ValueError, RecursionError):\n    pass\n",
    "import json\ntry:\n    json.loads(s)\nexcept Exception:\n    pass\n",
    "import json\ntry:\n    json.loads(s)\nexcept:\n    pass\n",
    "import json, contextlib\nwith contextlib.suppress(OSError, RecursionError):\n    json.loads(s)\n",
    "import json\ntry:\n    try:\n        json.loads(s)\n    except ValueError:\n        pass\nexcept Exception:\n    pass\n",
])
def test_the_scanner_accepts_covered_parses(source):
    assert unguarded_parses(source) == []
