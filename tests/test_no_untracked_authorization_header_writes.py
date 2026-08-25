# SPDX-License-Identifier: AGPL-3.0-or-later
"""No SIXTH site independently re-derives the owner-key-else-instance-token precedence for calling localm's own gated HTTP routes."""

from __future__ import annotations

import ast
import pathlib

from localm import auth as _auth

# (relative-to-repo-root path, line number) for every reviewed site. Paths use
# forward slashes regardless of host OS so the assertion is platform-stable.
_REVIEWED_SITES = {
    ("localm/auth.py", 251),
    ("localm/cli/rag.py", 443),
    ("localm/inference/http_engine.py", 108),
    ("localm/inference/http_engine.py", 203),
    ("localm/plugins/coder/backends/http.py", 353),
    ("localm/plugins/gui/cli.py", 125),
}


def _is_authorization_key(node) -> bool:
    return isinstance(node, ast.Constant) and node.value == "Authorization"


def _authorization_header_writes(root: pathlib.Path):
    """Every place under *root* that BUILDS an ``Authorization`` header: ``headers['Authorization'] = ...`` (subscript assignment), ``{'Authorization': ...}`` (dict literal, however it is used - a direct return, a ternary, a kwarg), or ``dict(Authorization=...)`` (keyword-call form)."""
    hits = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = path.relative_to(root.parent).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Subscript)
                            and _is_authorization_key(target.slice)):
                        hits.append((rel, node.lineno))
            elif isinstance(node, ast.Dict):
                for key in node.keys:
                    if key is not None and _is_authorization_key(key):
                        hits.append((rel, node.lineno))
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "dict"):
                for kw in node.keywords:
                    if kw.arg == "Authorization":
                        hits.append((rel, node.lineno))
    return hits


# --------------------------------------------------------------------------- #
#  Fires-control for the instrument, run BEFORE trusting its clean result      #
# --------------------------------------------------------------------------- #

def test_the_scan_detects_a_planted_subscript_write():
    planted = ast.parse('headers["Authorization"] = f"Bearer {key}"\n')
    hits = [n for n in ast.walk(planted)
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Subscript) and _is_authorization_key(t.slice)]
    assert hits, "the scan cannot see a subscript-assignment write; it proves nothing"


def test_the_scan_detects_a_planted_dict_literal_write():
    planted = ast.parse('h = {"Authorization": f"Bearer {tok}"} if tok else {}\n')
    hits = [n for n in ast.walk(planted) if isinstance(n, ast.Dict)
            for k in n.keys if k is not None and _is_authorization_key(k)]
    assert hits, "the scan cannot see a dict-literal write; it proves nothing"


def test_the_scan_detects_a_planted_dict_keyword_call_write():
    """dict(Authorization=...) is a real third spelling, not covered by either the subscript or the literal form above."""
    planted = ast.parse('h = dict(Authorization=f"Bearer {tok}")\n')
    hits = [n for n in ast.walk(planted)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "dict"
            for kw in n.keywords if kw.arg == "Authorization"]
    assert hits, "the scan cannot see a dict(Authorization=...) write; it proves nothing"


def test_a_mere_read_or_membership_check_is_not_flagged():
    """`headers.get('Authorization')` and `'Authorization' in headers` are reads, not writes, and must not be counted - or the scan would flag every caller that merely inspects a response header."""
    planted = ast.parse(
        'a = headers.get("Authorization")\n'
        'b = "Authorization" in headers\n'
    )
    writes = [n for n in ast.walk(planted)
              if isinstance(n, ast.Assign)
              for t in n.targets
              if isinstance(t, ast.Subscript) and _is_authorization_key(t.slice)]
    assert writes == []


# --------------------------------------------------------------------------- #
#  The real assertion                                                          #
# --------------------------------------------------------------------------- #

def test_every_authorization_header_write_is_a_reviewed_site():
    """A NEW (unreviewed) hit is a candidate sixth site of the credential- precedence class - add it to auth.resolve_bearer_headers's callers, or review it and extend _REVIEWED_SITES with the same rigor as the module docstring above."""
    root = pathlib.Path(_auth.__file__).resolve().parent
    hits = set(_authorization_header_writes(root))

    unreviewed = hits - _REVIEWED_SITES
    missing = _REVIEWED_SITES - hits
    assert not unreviewed, (
        "found an Authorization header built outside every reviewed site - "
        f"review it (see this file's module docstring): {sorted(unreviewed)}")
    assert not missing, (
        "a previously-reviewed site no longer exists (moved, renamed, or "
        f"removed) - update _REVIEWED_SITES to match: {sorted(missing)}")
