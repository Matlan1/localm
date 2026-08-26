# SPDX-License-Identifier: AGPL-3.0-or-later
"""Static ESM-integrity guards for the GUI - the class of bug the JS suite CANNOT
catch.

The GUI ships as native ES modules, but the jsdom JS harness (tests-js/harness.mjs)
STRIPS import/export and runs every module in one shared classic-script realm. That
merged realm hides three whole classes of "the page will not load in a real
browser" bug because the real module GRAPH is never linked or evaluated:

  1. an ``import { X } from "./m.js"`` where m.js does not actually export ``X`` -
     a hard module-graph load failure in the browser (blank page), but ``X`` just
     resolves to ``undefined`` in the merged realm and the suite passes;
  2. a top-level ``JSON.parse(localStorage.getItem(k) || "[]")`` - throws
     SyntaxError on a CORRUPT (not merely absent) value, and because it runs during
     module evaluation of a widely-imported module it aborts the whole graph and
     blanks the app. Must go through helpers.readStoredJSON, which branches
     missing-vs-corrupt and surfaces corruption instead of crashing boot.

(The third, reassigning a read-only import binding, is guarded by
test_gui_no_import_reassignment.py.)

These are static checks, because the harness structurally cannot make them.
"""

import re
from pathlib import Path

_GUI = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "gui" / "static"


def _gui_modules():
    return sorted((_GUI / "app").glob("*.js")) + sorted((_GUI / "pages").glob("*.js"))


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"(?m)//.*$", "", src)
    return src


def _top_statements(src: str):
    for line in src.splitlines():
        s = line.strip()
        if s.startswith(("import ", "import{", "export ", "export{")):
            yield s


def _parse_module(path: Path):
    src = _strip_comments(path.read_text(encoding="utf-8"))
    exports, star_sources, has_default = set(), [], False
    imports = []  # (source_spec, exported_name)
    for s in _top_statements(src):
        m = re.match(r"export\s+(?:async\s+)?function\s+([\w$]+)", s)
        if m:
            exports.add(m.group(1)); continue
        m = re.match(r"export\s+class\s+([\w$]+)", s)
        if m:
            exports.add(m.group(1)); continue
        m = re.match(r"export\s+(?:const|let|var)\s+([\w$]+)", s)
        if m:
            exports.add(m.group(1))
            for extra in re.finditer(r",\s*([\w$]+)\s*=", s):  # multi-declarator
                exports.add(extra.group(1))
            continue
        if re.match(r"export\s+default\b", s):
            has_default = True; continue
        m = re.match(r"export\s*\*\s*from\s*['\"]([^'\"]+)['\"]", s)
        if m:
            star_sources.append(m.group(1)); continue
        m = re.match(r"export\s*\{([^}]*)\}\s*from\s*['\"]([^'\"]+)['\"]", s)  # re-export
        if m:
            for part in m.group(1).split(","):
                p = part.strip()
                if p:
                    exports.add(p.split(" as ")[-1].strip())
            continue
        m = re.match(r"export\s*\{([^}]*)\}", s)  # local export list
        if m:
            for part in m.group(1).split(","):
                p = part.strip()
                if p:
                    exports.add(p.split(" as ")[-1].strip())
            continue
        m = re.match(r"import\s*\{([^}]*)\}\s*from\s*['\"]([^'\"]+)['\"]", s)
        if m:
            for part in m.group(1).split(","):
                p = part.strip()
                if p:
                    imports.append((m.group(2), p.split(" as ")[0].strip()))
            continue
        m = re.match(r"import\s+([\w$]+)\s*,\s*\{([^}]*)\}\s*from\s*['\"]([^'\"]+)['\"]", s)
        if m:
            imports.append((m.group(3), "default"))
            for part in m.group(2).split(","):
                p = part.strip()
                if p:
                    imports.append((m.group(3), p.split(" as ")[0].strip()))
            continue
        m = re.match(r"import\s+([\w$]+)\s+from\s*['\"]([^'\"]+)['\"]", s)
        if m:
            imports.append((m.group(2), "default"))
            continue
    return {"exports": exports, "star": star_sources, "imports": imports,
            "has_default": has_default}


def _exports_name(mods, mod_path, name, seen=None):
    seen = seen or set()
    if mod_path in seen:
        return False
    seen.add(mod_path)
    info = mods.get(mod_path)
    if info is None:
        return None  # external / not-scanned (vendor) - skip
    if name == "default":
        return info["has_default"]
    if name in info["exports"]:
        return True
    for star in info["star"]:
        if _exports_name(mods, (mod_path.parent / star).resolve(), name, seen):
            return True
    return False


def test_every_named_import_resolves_to_a_real_export():
    """A missing/misnamed export is a browser module-graph load failure (blank
    page) that the merged-realm JS suite cannot see."""
    mods = {p.resolve(): _parse_module(p) for p in _gui_modules()}
    broken = []
    for path, info in mods.items():
        for spec, name in info["imports"]:
            if not spec.startswith("."):
                continue  # bare specifier (vendor/package) - out of scope
            src = (path.parent / spec).resolve()
            ok = _exports_name(mods, src, name)
            if ok is False:
                broken.append(f"{path.name}: imports '{name}' from '{spec}' which does not export it")
    assert not broken, (
        "GUI module import(s) reference a name the target module does not export "
        "(the page would fail to load in a real browser):\n  " + "\n  ".join(broken))


def test_no_unguarded_json_parse_of_localstorage():
    """A raw JSON.parse(localStorage...) throws on a CORRUPT value; at module top
    level that blanks the whole app. Reads must go through helpers.readStoredJSON."""
    offenders = []
    for js in _gui_modules():
        src = _strip_comments(js.read_text(encoding="utf-8"))
        # JSON.parse( ... localStorage ... ) on one logical read (allow a newline)
        for m in re.finditer(r"JSON\.parse\s*\([^)]*localStorage", src, flags=re.S):
            line = src[:m.start()].count("\n") + 1
            offenders.append(f"{js.name}:{line}")
    assert not offenders, (
        "Raw JSON.parse(localStorage...) found - a corrupt entry would crash boot. "
        "Use helpers.readStoredJSON(key, fallback) which branches missing-vs-corrupt "
        "and surfaces corruption:\n  " + "\n  ".join(offenders))


def test_readstoredjson_helper_exists():
    """The guarded reader the above rule points people to must exist."""
    helpers = (_GUI / "app" / "helpers.js").read_text(encoding="utf-8")
    assert "export function readStoredJSON(" in helpers
