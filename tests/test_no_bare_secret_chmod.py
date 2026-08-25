# SPDX-License-Identifier: AGPL-3.0-or-later
"""LM-DA-044/LM-DA-027: no THIRD generation of a bare ``chmod(..., 0o600)`` locking down a secret file, bypassing ``config.restrict_file_perms``."""

from __future__ import annotations

import ast
import pathlib

from localm import config as _config

# (relative-to-repo-root path, enclosing qualname) for every reviewed
# non-secret chmod(..., 0o600) site. Paths use forward slashes regardless of
# host OS so the assertion is platform-stable.
_REVIEWED_SITES = {
    ("localm/bugreport.py", "save_report"),
}


def _enclosing_qualnames(tree: ast.AST):
    """Map every ``ast.Call`` in *tree* to the dotted name of the def/class it sits inside (``'<module>'`` for a call at module level)."""
    found: list[tuple[ast.Call, str]] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                visit(child, f"{prefix}.{child.name}"
                      if prefix != "<module>" else child.name)
                continue
            if isinstance(child, ast.Call):
                found.append((child, prefix))
            visit(child, prefix)

    visit(tree, "<module>")
    return found


def _bare_chmod_0600_sites(root: pathlib.Path):
    """Every ``<expr>.chmod(...)`` call under *root* whose arguments include a literal ``0o600`` - positional or keyword (``mode=0o600``)."""
    hits = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = path.relative_to(root.parent).as_posix()
        for node, qualname in _enclosing_qualnames(tree):
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "chmod":
                continue
            args = list(node.args) + [kw.value for kw in node.keywords]
            if any(isinstance(a, ast.Constant) and a.value == 0o600 for a in args):
                hits.append((rel, qualname))
    return hits


# --------------------------------------------------------------------------- #
#  Fires-control for the instrument, run BEFORE trusting its clean result      #
# --------------------------------------------------------------------------- #

def test_the_scan_detects_a_planted_os_chmod_call():
    planted = ast.parse("import os\nos.chmod(path, 0o600)\n")
    hits = [n for n in ast.walk(planted)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "chmod"
            and any(isinstance(a, ast.Constant) and a.value == 0o600
                    for a in n.args)]
    assert hits, "the scan cannot see a planted os.chmod(path, 0o600); it proves nothing"


def test_the_scan_detects_a_planted_path_chmod_call():
    """``Path.chmod(0o600)`` is a real second spelling (bugreport.py uses it), not covered by matching only ``os.chmod``."""
    planted = ast.parse("path.chmod(0o600)\n")
    hits = [n for n in ast.walk(planted)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "chmod"
            and any(isinstance(a, ast.Constant) and a.value == 0o600
                    for a in n.args)]
    assert hits, "the scan cannot see a planted path.chmod(0o600); it proves nothing"


def test_the_scan_detects_a_planted_keyword_mode_call():
    planted = ast.parse("os.chmod(path, mode=0o600)\n")
    hits = [n for n in ast.walk(planted)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "chmod"
            and any(isinstance(kw.value, ast.Constant) and kw.value.value == 0o600
                    for kw in n.keywords)]
    assert hits, "the scan cannot see a planted os.chmod(path, mode=0o600); it proves nothing"


def test_a_different_mode_or_a_variable_is_not_flagged():
    """0o700 (directory lockdown - tls.py/instances.py/gpu_registry.py all still chmod their run/tls dirs to 0700 directly, out of this finding's scope) and a *variable* mode (restrict_file_perms's own implementation) must not be flagged, or this test would fail on legitimate code the moment it is written."""
    planted = ast.parse(
        "os.chmod(path, 0o700)\n"
        "os.chmod(path, mode)\n"
    )
    hits = [n for n in ast.walk(planted)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "chmod"
            and any(isinstance(a, ast.Constant) and a.value == 0o600
                    for a in list(n.args) + [kw.value for kw in n.keywords])]
    assert hits == []


def test_the_scan_reports_the_enclosing_function_and_survives_a_line_shift(
        tmp_path):
    """Fires-control for the REAL scanner, and the regression guard for the detachment described in the module docstring."""
    root = tmp_path / "localm"
    root.mkdir()
    body = (
        "import os\n"
        "\n"
        "def outer():\n"
        "    path.chmod(0o600)\n"
        "\n"
        "class C:\n"
        "    def meth(self):\n"
        "        os.chmod(path, mode=0o600)\n"
        "\n"
        "path.chmod(0o600)\n"
        "path.chmod(0o700)\n"          # a different mode is not this finding
        "os.chmod(path, mode)\n"       # a variable mode is not a literal
    )
    expected = {("localm/x.py", "outer"),
                ("localm/x.py", "C.meth"),
                ("localm/x.py", "<module>")}

    (root / "x.py").write_text(body, encoding="utf-8")
    assert set(_bare_chmod_0600_sites(root)) == expected

    (root / "x.py").write_text("# pad\n" * 40 + body, encoding="utf-8")
    assert set(_bare_chmod_0600_sites(root)) == expected, (
        "an unrelated edit above the call changed the reported site, so every "
        "reviewed entry detaches the next time its file grows")


# --------------------------------------------------------------------------- #
#  The real assertion                                                          #
# --------------------------------------------------------------------------- #

def test_every_bare_chmod_0600_is_a_reviewed_site():
    """A NEW (unreviewed) hit is a candidate fourth-generation instance of the LM-DA-044/LM-DA-027 class - route it through config.restrict_file_perms, or review it and extend _REVIEWED_SITES with the same rigor as the module docstring above."""
    root = pathlib.Path(_config.__file__).resolve().parent
    hits = set(_bare_chmod_0600_sites(root))

    unreviewed = hits - _REVIEWED_SITES
    missing = _REVIEWED_SITES - hits
    assert not unreviewed, (
        "found a bare chmod(..., 0o600) outside config.restrict_file_perms - "
        "a documented Windows no-op that leaves the file inheriting the data "
        f"dir's ACL instead of being user-restricted (see this file's module "
        f"docstring): {sorted(unreviewed)}")
    assert not missing, (
        "a previously-reviewed site no longer exists (moved, renamed, fixed, "
        f"or removed) - update _REVIEWED_SITES to match: {sorted(missing)}")
