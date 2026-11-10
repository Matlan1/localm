"""Ratcheting sentinel for modules that re-derive the session mode themselves.

An automatic writer asks ``localm.audit.automatic_write_allowed``. Every module
outside ``localm/audit.py`` that calls ``effective_mode`` /
``effective_unattended_mode`` or compares a value with ``"privacy"`` /
``.PRIVACY`` is counted against ``.github/session-mode-baseline.json``. The
counts may only go down: a new or grown count fails, and a lowered count fails
until the baseline file is lowered to match.

Regenerate after migrating a writer:
    python tests/test_session_mode_sentinel.py --write
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "localm"
BASELINE = REPO / ".github" / "session-mode-baseline.json"
EXEMPT = {"localm/audit.py"}
MODE_FUNCS = {"effective_mode", "effective_unattended_mode"}


def _is_privacy_operand(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and node.value == "privacy":
        return True
    return isinstance(node, ast.Attribute) and node.attr == "PRIVACY"


def _call_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def count_sites(tree: ast.AST) -> int:
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in MODE_FUNCS:
            n += 1
        elif isinstance(node, ast.Compare):
            if any(_is_privacy_operand(o) for o in [node.left, *node.comparators]):
                n += 1
    return n


def scan() -> dict[str, int]:
    found: dict[str, int] = {}
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel in EXEMPT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        n = count_sites(tree)
        if n:
            found[rel] = n
    return found


def _load_baseline() -> dict[str, int]:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_count_sites_sees_each_shape():
    tree = ast.parse(
        "from localm.audit import effective_mode, SessionMode\n"
        "a = effective_mode('chat')\n"
        "b = audit.effective_unattended_mode('coder')\n"
        "c = mode == 'privacy'\n"
        "d = SessionMode.PRIVACY != m\n"
        "e = other == 'log'\n"
    )
    assert count_sites(tree) == 4


def test_no_module_re_derives_the_mode_beyond_the_baseline():
    base = _load_baseline()
    found = scan()
    grew = {f: (base.get(f, 0), n) for f, n in found.items() if n > base.get(f, 0)}
    assert not grew, (
        "These modules re-derive the session mode instead of calling "
        "audit.automatic_write_allowed (baseline, now): " + repr(grew))


def test_baseline_only_goes_down():
    base = _load_baseline()
    found = scan()
    stale = {f: (b, found.get(f, 0)) for f, b in base.items() if found.get(f, 0) < b}
    assert not stale, (
        "Sites were migrated; lower the baseline with "
        "`python tests/test_session_mode_sentinel.py --write` (baseline, now): "
        + repr(stale))


if __name__ == "__main__" and "--write" in sys.argv:
    BASELINE.write_text(json.dumps(scan(), indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"wrote {BASELINE}")
