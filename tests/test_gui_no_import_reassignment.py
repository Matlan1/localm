# SPDX-License-Identifier: AGPL-3.0-or-later
"""A GUI ES module must never REASSIGN a name it imports.

ES module import bindings are read-only. ``import { X } from "./m.js"; X = ...``
throws ``TypeError: Assignment to constant variable`` in the REAL browser and
aborts the function mid-way. The jsdom harness STRIPS import/export into ONE
shared scope, where the same reassignment silently works, so the JS suite cannot
see it.

This static check reads the GUI modules as text and fails if any reassigns a
name it imports; mutate a shared object in place (``VIEWS.length = 0;
VIEWS.push(...)``) or add a setter in the owning module instead.
"""

import re
from pathlib import Path

_GUI = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "gui" / "static"


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)   # block comments
    src = re.sub(r"(?m)//.*$", "", src)               # line comments
    return src


def _imported_names(src: str) -> set:
    """Local names introduced by ``import { a, b as c } from "..."`` statements."""
    names = set()
    for m in re.finditer(r"import\s*\{([^}]*)\}\s*from", src):
        for part in m.group(1).split(","):
            p = part.strip()
            if p:
                names.add(p.split(" as ")[-1].strip())   # `b as c` -> local name c
    return names


def _reassigns(src: str, name: str) -> bool:
    """True if *name* is assigned as a bare binding (``name = ...``), excluding a
    local (re)declaration, a member/index write (``name.x =`` / ``name[i] =``), and
    a comparison (``name ==`` / ``name =>``)."""
    esc = re.escape(name)
    for line in src.splitlines():
        if re.search(r"\b(?:const|let|var)\s+" + esc + r"\b", line):
            continue                                   # local declaration, allowed
        if re.search(r"(?<![.\w])" + esc + r"\s*=\s*(?![=>])", line):
            return True
    return False


def _gui_modules():
    return sorted((_GUI / "app").glob("*.js")) + sorted((_GUI / "pages").glob("*.js"))


def test_no_gui_module_reassigns_an_imported_binding():
    offenders = []
    for js in _gui_modules():
        raw = js.read_text(encoding="utf-8")
        src = _strip_comments(raw)
        for name in _imported_names(raw):
            if _reassigns(src, name):
                offenders.append(f"{js.name}: reassigns imported '{name}'")
    assert not offenders, (
        "A GUI ES module reassigns a read-only import (throws 'Assignment to "
        "constant variable' in the browser; mutate in place or use a setter):\n  "
        + "\n  ".join(offenders))


def test_detector_catches_a_planted_reassignment():
    """The detector flags an obvious import-reassignment."""
    bad = 'import { VIEWS } from "./tabs.js";\nfunction f(){ VIEWS = [1,2]; }\n'
    assert _reassigns(_strip_comments(bad), "VIEWS") is True
    # ...but NOT an in-place mutation or a local shadow or a comparison.
    ok = ('import { VIEWS, el } from "./tabs.js";\n'
          'VIEWS.length = 0; VIEWS.push(1);\n'
          'const el = document.createElement("div");\n'
          'if (VIEWS === el) {}\n')
    ok = _strip_comments(ok)
    assert _reassigns(ok, "VIEWS") is False
    assert _reassigns(ok, "el") is False
