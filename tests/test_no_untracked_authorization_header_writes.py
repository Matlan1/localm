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

  localm/auth.py, resolve_bearer_headers()
                                 the canonical implementation itself
                                 (resolve_bearer_headers, wrapping the raw-
                                 token precedence in resolve_bearer_token).

  localm/cli/rag.py, _cli_rag_embed_fn._embed()
                                 POSTs /v1/embeddings, which IS listed in
                                 http_server.py's _CROSS_ORIGIN_OK - the
                                 open-mode gate never applies, so get_api_key()
                                 alone (no instance_token fallback) is correct,
                                 the same classification as coder/plug.py and
                                 jobs/runner.py's chat/completions calls (see
                                 dev-notes/FIX-2026-08-05-cli-unload-stop-open-
                                 mode-auth.md).

  localm/inference/http_engine.py, HttpEngine._headers() and
  remote_model_status()
                                 route the `localm run`/`chat` ATTACH flow.
                                 These two are a generic HTTP client: they
                                 send whatever *token* the caller passes in,
                                 so they never need to call resolve_bearer_*
                                 themselves. RESOLVING the right token is the
                                 CALLER's job - cli/chat.py's `run()` and
                                 coder/cli/_main.py's `_build_backend()` now
                                 both call auth.resolve_bearer_token(target
                                 ["token"]) before constructing HttpEngine /
                                 HTTPBackend, so a keyed self-attach sends the
                                 owner key (checkup 2026-08-11 item 12 fixed
                                 this - it was flagged here as a known,
                                 unfixed, low-severity gap by the original
                                 2026-08-05 credential-resolution-helper PR;
                                 that flag turned out to be the "primary
                                 function hard-breaks in the default
                                 configuration" shape after all, contrary to
                                 its own "not a hard-block" reasoning at the
                                 time - see dev-notes/FIX-2026-08-05-
                                 credential-resolution-helper.md for the
                                 original, since-corrected rationale). The raw
                                 instance token remains the correct value to
                                 pass in for the open-mode fallback.

  localm/plugins/coder/backends/http.py, HTTPBackend._headers()
                                 the standalone coder CLI's own HTTP backend,
                                 targeting an arbitrary (possibly remote,
                                 possibly a different install's) server via
                                 --api-key/$LOCALM_API_KEY/a placeholder.
                                 Legitimately env/flag-scoped, not a self-call
                                 to THIS install.

  localm/plugins/gui/cli.py, _mount_remote_gui()
                                 asks a SIBLING localm instance (found via
                                 discovery, "api" mode) to mount its own GUI,
                                 using THAT instance's own registry attach
                                 token - the route (/v1/surfaces/) is itself
                                 listed in _CROSS_ORIGIN_OK and does its own
                                 strict auth server-side; this process's own
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

# (relative-to-repo-root path, enclosing scope, unparsed value expression) for
# every reviewed site. Paths use forward slashes regardless of host OS so the
# assertion is platform-stable. The scope is "<module>" for module-level code
# or a dotted "Outer.inner"/"Class.method" path; see _enclosing_scopes.
_REVIEWED_SITES = {
    ("localm/auth.py", "resolve_bearer_headers", "f'Bearer {token}'"),
    ("localm/cli/rag.py", "_cli_rag_embed_fn._embed", "f'Bearer {key}'"),
    ("localm/inference/http_engine.py", "HttpEngine._headers", "f'Bearer {self._token}'"),
    ("localm/inference/http_engine.py", "remote_model_status", "f'Bearer {token}'"),
    ("localm/plugins/coder/backends/http.py", "HTTPBackend._headers", "f'Bearer {self._api_key}'"),
    ("localm/plugins/gui/cli.py", "_mount_remote_gui", "f'Bearer {token}'"),
}


def _is_authorization_key(node) -> bool:
    return isinstance(node, ast.Constant) and node.value == "Authorization"


def _enclosing_scopes(tree: ast.AST) -> dict:
    """Maps ``id(node)`` to the dotted def/class path each node sits inside:
    ``"<module>"`` at module level, ``"outer.inner"`` for a nested def,
    ``"Class.method"`` for a method."""
    scopes: dict = {}

    def visit(node: ast.AST, stack: list) -> None:
        scopes[id(node)] = ".".join(stack) if stack else "<module>"
        child_stack = (stack + [node.name]
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                       else stack)
        for child in ast.iter_child_nodes(node):
            visit(child, child_stack)

    visit(tree, [])
    return scopes


def _authorization_header_hits(tree: ast.AST):
    """Every ``(enclosing scope, unparsed value expression)`` pair for a
    place in *tree* that BUILDS an ``Authorization`` header:
    ``headers["Authorization"] = ...`` (subscript assignment),
    ``{"Authorization": ...}`` (dict literal, however it is used - a direct
    return, a ternary, a kwarg), or ``dict(Authorization=...)`` (keyword-call
    form). AST, not grep, so a comment mentioning the header or a mere
    ``"Authorization" in headers`` / ``headers.get(...)`` read is never
    mistaken for a write.

    The value is ``ast.unparse``d rather than taken by line number or raw
    source text, so the key tracks the header's CODE rather than its
    position or formatting: see test_key_is_stable_when_code_moves_but_does_
    not_change and test_key_changes_when_the_reviewed_expression_itself_
    changes."""
    scopes = _enclosing_scopes(tree)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Subscript)
                        and _is_authorization_key(target.slice)):
                    hits.append((scopes[id(node)], ast.unparse(node.value)))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if key is not None and _is_authorization_key(key):
                    hits.append((scopes[id(node)], ast.unparse(value)))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "dict"):
            for kw in node.keywords:
                if kw.arg == "Authorization":
                    hits.append((scopes[id(node)], ast.unparse(kw.value)))
    return hits


def _authorization_header_writes(root: pathlib.Path):
    """Every ``(relative path, enclosing scope, unparsed value expression)``
    site under *root* that builds an ``Authorization`` header - see
    ``_authorization_header_hits`` for what counts as a write."""
    hits = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = path.relative_to(root.parent).as_posix()
        hits.extend((rel, scope, value) for scope, value in _authorization_header_hits(tree))
    return hits


# --------------------------------------------------------------------------- #
#  Fires-control for the instrument, run BEFORE trusting its clean result      #
# --------------------------------------------------------------------------- #

def test_the_scan_detects_a_planted_subscript_write():
    tree = ast.parse('headers["Authorization"] = f"Bearer {key}"\n')
    hits = _authorization_header_hits(tree)
    assert hits, "the scan cannot see a subscript-assignment write; it proves nothing"


def test_the_scan_detects_a_planted_dict_literal_write():
    tree = ast.parse('h = {"Authorization": f"Bearer {tok}"} if tok else {}\n')
    hits = _authorization_header_hits(tree)
    assert hits, "the scan cannot see a dict-literal write; it proves nothing"


def test_the_scan_detects_a_planted_dict_keyword_call_write():
    """dict(Authorization=...) is a real third spelling, not covered by
    either the subscript or the literal form above."""
    tree = ast.parse('h = dict(Authorization=f"Bearer {tok}")\n')
    hits = _authorization_header_hits(tree)
    assert hits, "the scan cannot see a dict(Authorization=...) write; it proves nothing"


def test_a_mere_read_or_membership_check_is_not_flagged():
    """`headers.get("Authorization")` and `"Authorization" in headers` are
    reads, not writes, and must not be counted - or the scan would flag every
    caller that merely inspects a response header."""
    tree = ast.parse(
        'a = headers.get("Authorization")\n'
        'b = "Authorization" in headers\n'
    )
    assert _authorization_header_hits(tree) == []


def test_key_is_stable_when_code_moves_but_does_not_change():
    """Unrelated lines inserted above a reviewed site - a new comment, a
    blank line, an explanatory note - must not change its key, or a routine
    edit elsewhere in the file turns into a spurious failure here."""
    before = ast.parse(
        'def resolve_bearer_headers():\n'
        '    token = get_token()\n'
        '    if token:\n'
        '        return {"Authorization": f"Bearer {token}"}\n'
        '    return {}\n'
    )
    after = ast.parse(
        '# a new comment\n'
        '\n'
        '\n'
        'def resolve_bearer_headers():\n'
        '    # an explanatory note that was not here before\n'
        '    token = get_token()\n'
        '\n'
        '    if token:\n'
        '        return {"Authorization": f"Bearer {token}"}\n'
        '    return {}\n'
    )
    assert _authorization_header_hits(before) == _authorization_header_hits(after)


def test_key_changes_when_the_reviewed_expression_itself_changes():
    """A site whose CODE changes - not merely its position - must re-key, so
    it is flagged for re-review rather than silently inheriting the old
    approval."""
    original = ast.parse(
        'def f():\n'
        '    return {"Authorization": f"Bearer {token}"}\n'
    )
    edited = ast.parse(
        'def f():\n'
        '    return {"Authorization": f"Basic {token}"}\n'
    )
    assert _authorization_header_hits(original) != _authorization_header_hits(edited)


def test_identical_value_text_in_different_scopes_does_not_collide():
    """Two sites that happen to compute the header identically must still be
    told apart by their enclosing scope - otherwise a genuinely new site
    could hide behind an already-reviewed key that merely looks the same."""
    tree = ast.parse(
        'def site_one():\n'
        '    headers["Authorization"] = f"Bearer {token}"\n'
        'def site_two():\n'
        '    headers["Authorization"] = f"Bearer {token}"\n'
    )
    hits = _authorization_header_hits(tree)
    assert len(hits) == 2
    assert len(set(hits)) == 2


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
