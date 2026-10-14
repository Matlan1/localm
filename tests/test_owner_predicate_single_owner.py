# SPDX-License-Identifier: AGPL-3.0-or-later
"""The "is this caller the owner" question has one implementation.

``caller_scopes`` returns None for BOTH open mode and "a key is configured but
the request presents no valid credential". Every authorisation site must ask
through the named predicates in ``http_server`` (``caller_is_operator``,
``caller_is_admin_key``, ``caller_grants``) rather than re-deriving the answer
from that None, so a route that is ever mounted without a ``require_scope``
dependency cannot read a credential-less caller as the owner.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import auth
from localm import scopes as S
from localm.inference import http_server as hs

KEY = "owner-key-one-0123456789abcdef"
PKG = Path(hs.__file__).resolve().parents[1]

PROBES = {
    "state": lambda r: hs.caller_auth_state(r),
    "operator": lambda r: hs.caller_is_operator(r),
    "admin_key": lambda r: hs.caller_is_admin_key(r),
    "config_write": lambda r: hs.caller_grants(r, S.CONFIG_WRITE),
    "coder_full": lambda r: hs.caller_grants(r, S.CODER_FULL),
}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    return tmp_path


@pytest.fixture
def client(home):
    app = FastAPI()

    @app.get("/probe")
    def probe(request: hs.Request):
        return {name: fn(request) for name, fn in PROBES.items()}

    return TestClient(app)


def _answers(client, key=None):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return client.get("/probe", headers=headers).json()


def test_open_mode_is_the_operator_but_not_an_admin_key(client):
    got = _answers(client)
    assert got == {"state": hs.CALLER_OPEN, "operator": True, "admin_key": False,
                   "config_write": True, "coder_full": True}


def test_no_credential_in_protected_mode_is_anonymous_and_nothing(client):
    auth.set_api_key(KEY)
    got = _answers(client)
    assert got == {"state": hs.CALLER_ANONYMOUS, "operator": False,
                   "admin_key": False, "config_write": False, "coder_full": False}


def test_wrong_key_in_protected_mode_is_anonymous(client):
    auth.set_api_key(KEY)
    got = _answers(client, "not-the-owner-key-0123456789abcdef")
    assert got["state"] == hs.CALLER_ANONYMOUS
    assert got["operator"] is False and got["config_write"] is False


def test_owner_key_is_operator_admin_key_and_grants_everything(client):
    auth.set_api_key(KEY)
    got = _answers(client, KEY)
    assert got == {"state": hs.CALLER_KEYED, "operator": True, "admin_key": True,
                   "config_write": True, "coder_full": True}


@pytest.mark.parametrize("scope_list,expect", [
    ([S.CHAT], dict(operator=False, admin_key=False, config_write=False, coder_full=False)),
    ([S.CONFIG_WRITE], dict(operator=False, admin_key=False, config_write=True, coder_full=False)),
    ([S.CODER_FULL], dict(operator=False, admin_key=False, config_write=False, coder_full=True)),
])
def test_scoped_key_is_keyed_and_only_what_it_holds(client, scope_list, expect):
    auth.set_api_key(KEY)
    minted = auth.create_key("device", scope_list, allow_privileged=True)
    got = _answers(client, minted["key"])
    assert got.pop("state") == hs.CALLER_KEYED
    assert got == expect


def test_admin_scoped_minted_key_is_operator_and_admin_key(client):
    auth.set_api_key(KEY)
    minted = auth.create_key("device", [S.ADMIN], allow_privileged=True)
    got = _answers(client, minted["key"])
    assert got["operator"] is True and got["admin_key"] is True


# ---- AST sentinel: no inline re-derivation of the owner/scope answer ----

# Helpers that ARE the owner predicate, in http_server.py.
_OWNERS_OF_THE_PATTERN = {("inference/http_server.py", "caller_grants"),
                          ("inference/http_server.py", "caller_is_admin_key")}

# Checks on a scope set that is not a request caller's: mount_gui verifies a raw
# presented token, _cwd_trusted reads a stored keystore record.
_NOT_A_REQUEST_CALLER = {("inference/routes/system.py", "mount_gui"),
                         ("plugins/builtin/jobs/runner.py", "_cwd_trusted")}


def _none_checked_names(boolop: ast.BoolOp) -> set:
    names = set()
    for node in ast.walk(boolop):
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                and len(node.ops) == 1 and isinstance(node.ops[0], (ast.Is, ast.IsNot))
                and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value is None):
            names.add(node.left.id)
    return names


def _scope_tested_names(boolop: ast.BoolOp) -> set:
    names = set()
    for node in ast.walk(boolop):
        if (isinstance(node, ast.Compare) and len(node.ops) == 1
                and isinstance(node.ops[0], (ast.In, ast.NotIn))
                and isinstance(node.comparators[0], ast.Name)):
            left = node.left
            leaf = left.attr if isinstance(left, ast.Attribute) else getattr(left, "id", "")
            if leaf in {"ADMIN", "CODER_FULL"}:
                names.add(node.comparators[0].id)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "grants" and node.args
                and isinstance(node.args[0], ast.Name)):
            names.add(node.args[0].id)
    return names


def inline_owner_derivations(tree: ast.AST) -> list:
    """(lineno, enclosing function) of each ``X is [not] None`` combined, in one
    boolean expression, with an ADMIN/CODER_FULL membership test or a
    ``grants(X, ...)`` call on the same name X."""
    parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BoolOp):
            continue
        if _none_checked_names(node) & _scope_tested_names(node):
            fn, cur = "<module>", node
            while cur in parents:
                cur = parents[cur]
                if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fn = cur.name
                    break
            hits.append((node.lineno, fn))
    return hits


def test_sentinel_flags_the_inline_shapes():
    for src in (
        "owner = held is None or scopes.ADMIN in held",
        "if held is not None and scopes.ADMIN not in held:\n    pass",
        "ok = held is not None and scopes.ADMIN in held",
        "ok = held is None or scopes.grants(held, scopes.CONFIG_WRITE)",
        "ok = held is not None and (S.ADMIN in held or S.CODER_FULL in held)",
    ):
        assert inline_owner_derivations(ast.parse(src)), src
    for src in (
        "owner = caller_is_operator(request)",
        "x = held is None",
        "y = scopes.ADMIN in held",
        "z = a is None or b in c",
    ):
        assert not inline_owner_derivations(ast.parse(src)), src


def test_no_module_re_derives_the_owner_answer_inline():
    offenders = []
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for lineno, fn in inline_owner_derivations(tree):
            if (rel, fn) not in _OWNERS_OF_THE_PATTERN | _NOT_A_REQUEST_CALLER:
                offenders.append(f"{rel}:{lineno} in {fn}")
    assert not offenders, (
        "inline owner/scope derivation from a None scope set; use "
        "http_server.caller_is_operator / caller_is_admin_key / caller_grants:\n  "
        + "\n  ".join(offenders))
