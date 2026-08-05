# SPDX-License-Identifier: AGPL-3.0-or-later
"""No SIXTH site independently re-derives the owner-key-else-instance-token
precedence for calling localm's own gated HTTP routes.

Four sites did exactly that, one at a time: `selfclient.read_activity`
(#953/#1094), `selfclient.self_request` (#1114), `cli/models.py`'s
`unload_cmd`/`stop_cmd` (#1121), and `media/comfy_client.py`'s
`_localm_unload` (this change). Each was a separate incident because the bug
lives in how each site ASSEMBLES its credential, not because the precedence
rule itself was ever wrong - so a test on the shared helper
(`auth.resolve_bearer_headers`, and `tests/test_selfclient.py`'s pinned
precedence) verifies the RULE beautifully and is structurally incapable of
noticing a caller that never invokes it. This test sits one layer further
out: it finds every place in the tree that BUILDS an ``Authorization`` header
at all, and asserts the set is exactly the reviewed one below - so a genuinely
new (unreviewed) site fails loudly here instead of shipping as a silent sixth
instance of the same class.

Deliberately NOT "every site must call get_api_key()" - two of the reviewed
sites below never call it (nor resolve_bearer_headers) and are still correct,
because the credential precedence only applies to routes _origin_guard
actually gates (an unsafe method, or a metadata GET, not listed in
_CROSS_ORIGIN_OK). The property is "resolves through the shared precedence
where the route needs it", not "calls one named function" - asserting the
latter would flag these two as violations:

  localm/auth.py:228,230        the canonical implementation itself
                                 (resolve_bearer_headers).

  localm/cli/rag.py:392         POSTs /v1/embeddings, which IS listed in
                                 http_server.py's _CROSS_ORIGIN_OK - the
                                 open-mode gate never applies, so get_api_key()
                                 alone (no instance_token fallback) is correct,
                                 the same classification as coder/plug.py and
                                 jobs/runner.py's chat/completions calls (see
                                 dev-notes/FIX-2026-08-05-cli-unload-stop-open-
                                 mode-auth.md).

  localm/inference/http_engine.py:108,203
                                 HttpEngine/remote_model_status route the
                                 `localm run`/`chat` ATTACH flow, a different
                                 credential class entirely: the token sent is
                                 always the discovered instance's own registry
                                 attach token (never this process's owner
                                 key), used for /v1/chat/completions (exempt,
                                 same as above) and a best-effort /v1/config
                                 capacity probe that already degrades
                                 gracefully (try/except -> None) on a keyed
                                 target instead of hard-failing the attach -
                                 not the "primary function silently or hard-
                                 breaks in the default configuration" shape
                                 this precedence class exists to fix. Recorded
                                 here as a known, low-severity, DIFFERENT-
                                 command candidate rather than silently
                                 rediscovered later; not fixed in this change
                                 (out of scope - see the PR/commit notes).

  localm/plugins/coder/backends/http.py:353
                                 (the dict literal's opening line - the
                                 "Authorization" key text itself is on 354,
                                 one line down; ast.Dict.lineno points at the
                                 literal's start, not each key) the standalone
                                 coder CLI's own HTTP backend,
                                 targeting an arbitrary (possibly remote,
                                 possibly a different install's) server via
                                 --api-key/$LOCALM_API_KEY/a placeholder.
                                 Legitimately env/flag-scoped, not a self-call
                                 to THIS install.

  localm/plugins/gui/cli.py:109 `_mount_remote_gui` asks a SIBLING localm
                                 instance (found via discovery, "api" mode) to
                                 mount its own GUI, using THAT instance's own
                                 registry attach token - the route
                                 (/v1/surfaces/) is itself listed in
                                 _CROSS_ORIGIN_OK and does its own strict auth
                                 server-side; this process's own
                                 get_api_key() would not even be the right
                                 credential for a different instance's home.

If you land a new site here, EITHER route it through
``auth.resolve_bearer_headers`` (the common case) OR add it to the allowlist
below with a review comment matching the rigor above - never widen the scan
to stop noticing it.
"""

from __future__ import annotations

import ast
import pathlib

from localm import auth as _auth

# (relative-to-repo-root path, line number) for every reviewed site. Paths use
# forward slashes regardless of host OS so the assertion is platform-stable.
_REVIEWED_SITES = {
    ("localm/auth.py", 228),
    ("localm/auth.py", 230),
    ("localm/cli/rag.py", 392),
    ("localm/inference/http_engine.py", 108),
    ("localm/inference/http_engine.py", 203),
    ("localm/plugins/coder/backends/http.py", 353),
    ("localm/plugins/gui/cli.py", 109),
}


def _is_authorization_key(node) -> bool:
    return isinstance(node, ast.Constant) and node.value == "Authorization"


def _authorization_header_writes(root: pathlib.Path):
    """Every place under *root* that BUILDS an ``Authorization`` header:
    ``headers["Authorization"] = ...`` (subscript assignment), ``{"Authorization": ...}``
    (dict literal, however it is used - a direct return, a ternary, a kwarg),
    or ``dict(Authorization=...)`` (keyword-call form). AST, not grep, so a
    comment mentioning the header or a mere ``"Authorization" in headers`` /
    ``headers.get(...)`` read is never mistaken for a write."""
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
    """dict(Authorization=...) is a real third spelling, not covered by
    either the subscript or the literal form above."""
    planted = ast.parse('h = dict(Authorization=f"Bearer {tok}")\n')
    hits = [n for n in ast.walk(planted)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "dict"
            for kw in n.keywords if kw.arg == "Authorization"]
    assert hits, "the scan cannot see a dict(Authorization=...) write; it proves nothing"


def test_a_mere_read_or_membership_check_is_not_flagged():
    """`headers.get("Authorization")` and `"Authorization" in headers` are
    reads, not writes, and must not be counted - or the scan would flag every
    caller that merely inspects a response header."""
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
    """A NEW (unreviewed) hit is a candidate sixth site of the credential-
    precedence class - add it to auth.resolve_bearer_headers's callers, or
    review it and extend _REVIEWED_SITES with the same rigor as the module
    docstring above. Never widen this test to stop noticing it."""
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
