# SPDX-License-Identifier: AGPL-3.0-or-later
"""LM-DA-044/LM-DA-027: no THIRD generation of a bare ``chmod(..., 0o600)``
locking down a secret file, bypassing ``config.restrict_file_perms``.

``config.restrict_file_perms`` (config.py:749) exists precisely because a bare
POSIX ``os.chmod(path, 0o600)`` is a documented no-op on Windows
(config.py:768-769, tls.py's own former comment) - it leaves the file
inheriting the data directory's ACL (commonly ``BUILTIN\\Users`` read) instead
of being restricted to the current user. Three writers drifted onto the bare
form before this test existed: ``tls._write_private`` (the TLS CA and leaf
PRIVATE KEYS - LM-DA-044), and the token-bearing registry writers
``instances.register_instance``/``instances.set_mode`` and
``gpu_registry.write_entry`` (LM-DA-027, folded into LM-DA-044). Each was a
separate incident because ``config.py``'s own docstring documented the helper
and that did not stop a fourth site from reaching for the familiar bare
pattern instead - "one implementation so a fourth caller cannot get a weaker
fourth variant" held only by convention. This test makes that convention
mechanical: any NEW bare ``chmod(..., 0o600)`` under ``localm/`` fails here
instead of shipping as a silent fourth (now fifth) instance.

Deliberately NOT "no module may call os.chmod with 0o600" as a lint rule
against the exact call site config.restrict_file_perms itself makes -
excluded below because it passes 0o600 as its *mode* default, a `Name` node,
never a literal `Constant` at the call site (the AST scan already cannot see
it; nothing to allowlist).

If you land a new site here, EITHER route it through
``config.restrict_file_perms`` (the common case) OR add it to the allowlist
below with a review comment proving the file carries no secret - never widen
the scan to stop noticing it.

  localm/bugreport.py:887   ``save_report``'s ``path.chmod(0o600)`` on the
                             saved bug-report markdown. Reviewed: the module
                             docstring and the site's own why-comment both
                             state the report deliberately carries NO secrets
                             (no API key, env, config secrets, or chat
                             content) - the 0600 here is multi-user-box
                             world-readability hygiene, not credential
                             protection, so a Windows no-op degrades it to
                             "as readable as anything else in the data dir",
                             not to "a leaked secret".
"""

from __future__ import annotations

import ast
import pathlib

from localm import config as _config

# (relative-to-repo-root path, line number) for every reviewed non-secret
# chmod(..., 0o600) site. Paths use forward slashes regardless of host OS so
# the assertion is platform-stable.
_REVIEWED_SITES = {
    ("localm/bugreport.py", 887),
}


def _bare_chmod_0600_sites(root: pathlib.Path):
    """Every ``<expr>.chmod(...)`` call under *root* whose arguments include a
    literal ``0o600`` - positional or keyword (``mode=0o600``). AST, not grep,
    so a comment or docstring mentioning ``0o600`` is never mistaken for a
    call, and a call passing a *variable* (e.g. ``restrict_file_perms``'s own
    ``os.chmod(path, mode)``) is never mistaken for the literal."""
    hits = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = path.relative_to(root.parent).as_posix()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "chmod"):
                continue
            args = list(node.args) + [kw.value for kw in node.keywords]
            if any(isinstance(a, ast.Constant) and a.value == 0o600 for a in args):
                hits.append((rel, node.lineno))
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


# --------------------------------------------------------------------------- #
#  The real assertion                                                          #
# --------------------------------------------------------------------------- #

def test_every_bare_chmod_0600_is_a_reviewed_site():
    """A NEW (unreviewed) hit is a candidate fourth-generation instance of the
    LM-DA-044/LM-DA-027 class - route it through config.restrict_file_perms,
    or review it and extend _REVIEWED_SITES with the same rigor as the module
    docstring above. Never widen this test to stop noticing it."""
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
