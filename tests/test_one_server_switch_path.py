# SPDX-License-Identifier: AGPL-3.0-or-later
"""One code path decides what a command does when a localm server is already
running for its project with another model: server_switch.serve_requested_model.

These scans fail when a command attaches to a running server without being
classified below, when one of the model-serving commands stops calling the
shared path, when another module asks a server to load a model, or when the
old refusal texts come back."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

LOCALM = Path(__file__).resolve().parents[1] / "localm"

# Commands that attach to the running server with a model of their own: each
# must call serve_requested_model.
SERVES_A_MODEL = {
    "cli/chat.py": "localm run MODEL",
    "plugins/gui/cli.py": "localm gui MODEL, localm serve MODEL, the launcher",
    "plugins/coder/cli/_main.py": "localm coder --model MODEL",
}

# Callers of attach_target/find_attachable that name no model to serve.
NO_MODEL = {
    "instances.py": "defines attach_target and find_attachable",
    "cli/_core.py": "server address and headers for management commands",
    "cli/models.py": "rename, status, unload and stop act on the server as it is",
    "cli/plugins.py": "plugin management on the running server",
    "cli/rag.py": "embedding requests use the configured embedding model",
    "plugins/builtin/jobs/cli.py": "runs a job on the server; the job names its model per request",
    "plugins/builtin/jobs/runner.py": "the server's own job runner calling itself",
}

# Modules whose string literals may name the model load route.
LOAD_ROUTE_ALLOWED = {
    "inference/server_switch.py": "the one switch request",
    "inference/routes/models.py": "defines POST /v1/models/load",
    "plugins/gui/routes/models/inventory.py": "defines POST /api/models/load",
    "vram.py": "the server reloading its own model",
    "plugins/coder/backends/http.py": "an attached coder session's /model command",
}

OLD_REFUSALS = ("not the requested", "(the running server serves")


def _modules():
    for path in sorted(LOCALM.rglob("*.py")):
        rel = path.relative_to(LOCALM).as_posix()
        yield rel, path.read_text(encoding="utf-8")


def _called_names(tree: ast.AST) -> set:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute):
                names.add(fn.attr)
            elif isinstance(fn, ast.Name):
                names.add(fn.id)
    return names


def _docstring_nodes(tree: ast.AST) -> set:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                ids.add(id(body[0].value))
    return ids


def _string_literals(tree: ast.AST):
    skip = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in skip):
            yield node.value


def test_every_module_that_attaches_to_a_running_server_is_classified():
    unclassified = []
    for rel, src in _modules():
        called = _called_names(ast.parse(src))
        if {"attach_target", "find_attachable"} & called or (
                rel == "instances.py"):
            if rel not in SERVES_A_MODEL and rel not in NO_MODEL:
                unclassified.append(rel)
    assert unclassified == [], (
        "these modules attach to a running localm server: a command that names "
        "a model calls server_switch.serve_requested_model and is listed in "
        f"SERVES_A_MODEL, otherwise list it in NO_MODEL: {unclassified}")


@pytest.mark.parametrize("rel", sorted(SERVES_A_MODEL))
def test_each_model_serving_command_calls_the_shared_path(rel):
    called = _called_names(ast.parse((LOCALM / rel).read_text(encoding="utf-8")))
    assert "serve_requested_model" in called, rel


def test_only_the_shared_path_probes_and_switches():
    probes = []
    for rel, src in _modules():
        if rel in ("inference/http_engine.py", "inference/server_switch.py"):
            continue
        called = _called_names(ast.parse(src))
        if {"remote_model_status", "served_model_is_requested",
                "remote_active_model", "switch_server_model"} & called:
            probes.append(rel)
    assert probes == [], (
        "probe or switch a running server through "
        f"server_switch.serve_requested_model: {probes}")


def test_no_other_module_asks_a_server_to_load_a_model():
    found = []
    for rel, src in _modules():
        if any("/models/load" in s for s in _string_literals(ast.parse(src))):
            found.append(rel)
    assert sorted(found) == sorted(LOAD_ROUTE_ALLOWED), (
        "only server_switch.switch_server_model asks a running server to load "
        f"a model: {sorted(set(found) - set(LOAD_ROUTE_ALLOWED))}")


def test_the_old_refusal_texts_are_gone():
    hits = [(rel, text) for rel, src in _modules() for text in OLD_REFUSALS
            if text in src]
    assert hits == []


CREDENTIAL_RESOLVERS = {"resolve_bearer_token", "resolve_bearer_headers",
                        "get_api_key"}


def test_the_switch_sends_the_credential_its_caller_gives():
    src = (LOCALM / "inference/server_switch.py").read_text(encoding="utf-8")
    assert CREDENTIAL_RESOLVERS & _called_names(ast.parse(src)) == set()


def test_the_coder_resolves_its_attach_credential_in_one_function():
    tree = ast.parse((LOCALM / "plugins/coder/cli/_main.py").read_text(
        encoding="utf-8"))
    parent = {child: node for node in ast.walk(tree)
              for child in ast.iter_child_nodes(node)}
    where = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (
                getattr(node.func, "id", None) == "resolve_bearer_token"
                or getattr(node.func, "attr", None) == "resolve_bearer_token"):
            fn = parent.get(node)
            while fn is not None and not isinstance(
                    fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = parent.get(fn)
            where.append(getattr(fn, "name", None))
    assert where == ["_attach_credential"]


SWITCH_CALLS = {"serve_requested_model", "_serve_requested_model", "_serve_on_attached"}
CREDENTIAL_SOURCES = {"resolve_bearer_token", "_attach_credential"}


def _call_name(node):
    fn = getattr(node, "func", None)
    return getattr(fn, "attr", None) or getattr(fn, "id", None)


def _credential_arg(call):
    if len(call.args) > 1:
        return call.args[1]
    for kw in call.keywords:
        if kw.arg == "token":
            return kw.value
    return None


def _unresolved_switch_credentials(tree):
    """(function, line) of every call into the switch path whose credential is
    not a resolve_bearer_token(...) or _attach_credential(...) result: passed
    directly, or a name whose last assignment before the call, in the same
    function, is one; a parameter of a switch-path wrapper passing its own
    argument on is accepted."""
    bad = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        assigned = []
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                source = (isinstance(node.value, ast.Call)
                          and _call_name(node.value) in CREDENTIAL_SOURCES)
                assigned += [(node.lineno, t.id, source)
                             for t in node.targets if isinstance(t, ast.Name)]
        params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call) and _call_name(node) in SWITCH_CALLS):
                continue
            arg = _credential_arg(node)
            if isinstance(arg, ast.Call):
                ok = _call_name(arg) in CREDENTIAL_SOURCES
            elif isinstance(arg, ast.Name):
                before = [a for a in assigned if a[1] == arg.id and a[0] < node.lineno]
                if before:
                    ok = max(before)[2]
                else:
                    ok = arg.id in params and fn.name in SWITCH_CALLS
            else:
                ok = False
            if not ok:
                bad.append((fn.name, node.lineno))
    return bad


@pytest.mark.parametrize("rel", sorted(SERVES_A_MODEL))
def test_every_switch_credential_is_resolved_by_its_command(rel):
    tree = ast.parse((LOCALM / rel).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and _call_name(n) in SWITCH_CALLS]
    assert calls, rel
    assert _unresolved_switch_credentials(tree) == [], rel


def test_the_credential_scan_sees_a_raw_instance_token():
    tree = ast.parse(
        'def attach(tgt):\n'
        '    key = _attach_credential(api_key, explicit, tgt.get("token"))\n'
        '    _serve_on_attached(tgt["base_url"], key, model)\n'
        '    _serve_on_attached(tgt["base_url"], tgt.get("token"), model)\n'
        '    key = tgt.get("token")\n'
        '    _serve_on_attached(tgt["base_url"], key, model)\n')
    assert _unresolved_switch_credentials(tree) == [("attach", 4), ("attach", 6)]


def test_the_scan_sees_a_planted_attach_and_load_request():
    tree = ast.parse(
        'def f():\n'
        '    """Mentions /models/load in a docstring."""\n'
        '    t = instances.attach_target(home, root)\n'
        '    requests.post(base + "/models/load", json={})\n')
    assert "attach_target" in _called_names(tree)
    assert list(_string_literals(tree)) == ["/models/load"]
