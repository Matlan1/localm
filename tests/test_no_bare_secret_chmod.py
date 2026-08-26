# SPDX-License-Identifier: AGPL-3.0-or-later
"""No bare ``chmod(..., 0o600)`` locking down a secret file, bypassing
``config.restrict_file_perms``.

``config.restrict_file_perms`` exists because a bare POSIX
``os.chmod(path, 0o600)`` is a no-op on Windows: it leaves the file inheriting
the data directory's ACL (commonly ``BUILTIN\\Users`` read) instead of being
restricted to the current user. Any NEW bare ``chmod(..., 0o600)`` under
``localm/`` fails here instead of shipping as a silent extra instance.

NOT a lint rule against the exact call site config.restrict_file_perms itself
makes: it passes 0o600 as its *mode* default, a `Name` node, never a literal
`Constant` at the call site, so the AST scan already cannot see it.

If you land a new site here, EITHER route it through
``config.restrict_file_perms`` (the common case) OR add it to the allowlist
below with a review comment proving the file carries no secret - never widen
the scan to stop noticing it.

  localm/bugreport.py::save_report   its ``path.chmod(0o600)`` on the saved
                             bug-report markdown. Reviewed: the report carries
                             NO secrets (no API key, env, config secrets, or
                             chat content), so the 0600 here is multi-user-box
                             world-readability hygiene, not credential
                             protection, and a Windows no-op degrades it to
                             "as readable as anything else in the data dir",
                             not to "a leaked secret".

THE ALLOWLIST IS KEYED ON THE ENCLOSING FUNCTION, NOT ON A LINE NUMBER. A
line-keyed pin detaches whenever an unrelated edit lands above the site, and
reports the reviewed site as unreviewed. A function RENAME still detaches, which
is correct - that is a change worth re-reading the review comment for.
"""

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
    """Map every ``ast.Call`` in *tree* to the dotted name of the def/class it
    sits inside (``"<module>"`` for a call at module level).

    ``ast.walk`` alone cannot answer this: it flattens the tree, so a node
    knows its own line but not its owner."""
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
    """Every ``<expr>.chmod(...)`` call under *root* whose arguments include a
    literal ``0o600`` - positional or keyword (``mode=0o600``). AST, not grep,
    so a comment or docstring mentioning ``0o600`` is never mistaken for a
    call, and a call passing a *variable* (e.g. ``restrict_file_perms``'s own
    ``os.chmod(path, mode)``) is never mistaken for the literal.

    Reported as (path, enclosing qualname)."""
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
#  The scanner fires on a synthetic bare chmod                                 #
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
    """``Path.chmod(0o600)`` is a real second spelling (bugreport.py uses it),
    not covered by matching only ``os.chmod``."""
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
    """0o700 (directory lockdown - tls.py/instances.py/gpu_registry.py all
    still chmod their run/tls dirs to 0700 directly, out of this finding's
    scope) and a *variable* mode (restrict_file_perms's own implementation)
    must not be flagged, or this test would fail on legitimate code the
    moment it is written."""
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
    """Fires-control for the REAL scanner, and the regression guard for the
    line-keyed detachment described in the module docstring.

    The three planted tests above re-apply the predicate by hand, so they are
    blind to a change in ``_bare_chmod_0600_sites`` itself - they would stay
    green if it stopped scanning entirely. This one drives the actual function.

    The second half is the point: padding the file shifts every line and must
    not change the reported sites.
    """
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
    """A NEW (unreviewed) hit is a candidate bypass of
    config.restrict_file_perms - route it through that helper, or review it and
    extend _REVIEWED_SITES with the same rigor as the module docstring above.
    Never widen this test to stop noticing it."""
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
