# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/check_hygiene.py check 10: a test never reads a top-level JSON key
that the matching route's handler cannot produce.

A handler whose response is built from dict literals has a CLOSED shape; each
key a test reads from that route's JSON must be one the handler can return.
Any other construction leaves the shape OPEN and unjudged.

These tests pin what makes the check worth having. It FIRES on a key the route
dropped, however the test spells the read (a bound name, an inline chain, a
positive membership test, an f-string path, an awaited client, a rebinding
through `.json()`, a binding inside a `try*` block). It does NOT fire when the
shape is open (including every handler mutation the walk does not model), when
the key is produced by a tracked mutation or a same-module helper, on a write
into the body, on a negative membership test, on `.get()`, inside a
`pytest.raises` block, on the error keys, on a path the test file serves itself
(by template or router prefix), on a path behind a non-literal base, on a path
variable that differs between branches, on an ambiguous path, on a name shadowed
by a comprehension or lambda, or in a function that mentions a helper the
route's keys came from. The last section binds the check to the real tree: the
route whose dropped field motivated it is judged, and dropping that field again
is caught.
"""

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_check_hygiene():
    spec = importlib.util.spec_from_file_location(
        "check_hygiene", REPO_ROOT / "scripts" / "check_hygiene.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ROUTE = '''
from fastapi import FastAPI

app = FastAPI()


@app.get("/api/thing/status")
async def thing_status():
    return {"installed": True, "state": "ready"}
'''

_TEST = '''
def test_reads(client):
    r = client.get("/api/thing/status")
    body = r.json()
    assert body["installed"] is True
    assert body["{key}"] == "x"
'''


def _tree(tmp_path, files: dict[str, str]) -> list[Path]:
    """A throwaway checkout from {relative path: source}; returns the tracked
    file list the check takes."""
    out = []
    for rel, src in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
        out.append(p)
    return out


def _check(tmp_path, monkeypatch, files):
    ch = _load_check_hygiene()
    tracked = _tree(tmp_path, files)
    monkeypatch.setattr(ch, "REPO", tmp_path)
    return ch._response_key_violations(tracked)


def _run(tmp_path, monkeypatch, route=_ROUTE, test=_TEST, key="state"):
    return _check(tmp_path, monkeypatch, {
        "localm/routes.py": route,
        "tests/test_thing.py": test.replace("{key}", key),
    })


# --------------------------------------------------------------------------- #
#  NEGATIVE: it must actually fire                                             #
# --------------------------------------------------------------------------- #

def test_dropped_key_fires_and_names_test_route_and_handler(tmp_path, monkeypatch):
    problems = _run(tmp_path, monkeypatch, key="enabled")
    assert len(problems) == 1, problems
    msg = problems[0]
    assert msg.startswith("tests/test_thing.py:6 (test_reads) reads key 'enabled' "
                          "from the JSON of GET /api/thing/status"), msg
    assert "localm/routes.py:8 thing_status can return only: installed, state." in msg


def test_present_key_is_silent(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch, key="state") == []


def test_inline_chain_fires(tmp_path, monkeypatch):
    test = '''
def test_inline(client):
    assert client.get("/api/thing/status").json()["gone"] == 1
'''
    problems = _run(tmp_path, monkeypatch, test=test)
    assert len(problems) == 1 and "'gone'" in problems[0], problems


def test_positive_membership_fires(tmp_path, monkeypatch):
    test = '''
def test_member(client):
    body = client.get("/api/thing/status").json()
    assert "gone" in body
'''
    problems = _run(tmp_path, monkeypatch, test=test)
    assert len(problems) == 1 and "'gone'" in problems[0], problems


def test_fstring_path_matches_a_parameterised_route(tmp_path, monkeypatch):
    route = '''
@app.get("/api/things/{name}")
async def one_thing(name: str):
    return {"name": name}
'''
    test = '''
def test_param(client):
    name = "abc"
    body = client.get(f"/api/things/{name}").json()
    assert body["size"] == 3
'''
    problems = _run(tmp_path, monkeypatch, route=_ROUTE + route, test=test)
    assert len(problems) == 1, problems
    assert "GET /api/things/abc" in problems[0] and "one_thing" in problems[0], problems[0]


def test_awaited_async_client_fires(tmp_path, monkeypatch):
    test = '''
async def test_async(ac):
    r = await ac.get("/api/thing/status")
    data = r.json()
    assert data["gone"]
'''
    problems = _run(tmp_path, monkeypatch, test=test)
    assert len(problems) == 1 and "'gone'" in problems[0], problems


def test_query_string_and_literal_host_are_stripped(tmp_path, monkeypatch):
    test = '''
def test_url(client):
    body = client.get("/api/thing/status?verbose=1").json()
    assert body["gone"]
    data = client.get("http://testserver/api/thing/status").json()
    assert data["also_gone"]
'''
    problems = _run(tmp_path, monkeypatch, test=test)
    assert len(problems) == 2, problems
    assert all("GET /api/thing/status" in p for p in problems)


def test_a_path_behind_a_non_literal_base_is_not_resolved(tmp_path, monkeypatch):
    test = '''
def test_url(client, base):
    body = client.get(f"{base}/api/thing/status").json()
    assert body["gone"]
    other = client.get(base + "/api/thing/status").json()
    assert other["gone"]
'''
    assert _run(tmp_path, monkeypatch, test=test) == []


def test_a_literal_bound_in_the_function_is_substituted_into_an_fstring(tmp_path, monkeypatch):
    test = '''
def test_url(client):
    base = "/api/thing"
    body = client.get(f"{base}/status").json()
    assert body["gone"]
'''
    problems = _run(tmp_path, monkeypatch, test=test)
    assert len(problems) == 1 and "GET /api/thing/status" in problems[0], problems


def test_key_absent_from_both_branch_routes_fires_once(tmp_path, monkeypatch):
    route = '''
@app.get("/api/other")
async def other():
    return {"other": 1}
'''
    test = '''
def test_branches(client, flag):
    if flag:
        r = client.get("/api/thing/status")
    else:
        r = client.get("/api/other")
    body = r.json()
    assert body["gone"]
'''
    problems = _run(tmp_path, monkeypatch, route=_ROUTE + route, test=test)
    assert len(problems) == 1, problems
    assert "'gone'" in problems[0]


def test_multiple_returns_form_a_union_and_a_key_in_none_fires(tmp_path, monkeypatch):
    route = '''
from fastapi import FastAPI

app = FastAPI()


@app.get("/api/thing/status")
async def thing_status(flag: bool):
    if flag:
        return {"a": 1}
    return {"b": 2}
'''
    test = '''
def test_union(client):
    body = client.get("/api/thing/status").json()
    assert body["a"] or body["b"]
    assert body["c"]
'''
    problems = _run(tmp_path, monkeypatch, route=route, test=test)
    assert len(problems) == 1 and "'c'" in problems[0], problems


def test_jsonresponse_content_is_a_shape(tmp_path, monkeypatch):
    route = '''
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()


@app.post("/api/thing/act")
async def act():
    return JSONResponse(status_code=202, content={"accepted": True})
'''
    test = '''
def test_act(client):
    body = client.post("/api/thing/act").json()
    assert body["accepted"]
    assert body["job"]
'''
    problems = _run(tmp_path, monkeypatch, route=route, test=test)
    assert len(problems) == 1 and "'job'" in problems[0], problems


def test_hygiene_ok_marker_silences_the_line(tmp_path, monkeypatch):
    test = '''
def test_marked(client):
    body = client.get("/api/thing/status").json()
    assert body["gone"]  # hygiene-ok
    assert body["also_gone"]
'''
    problems = _run(tmp_path, monkeypatch, test=test)
    assert len(problems) == 1 and "'also_gone'" in problems[0], problems


# --------------------------------------------------------------------------- #
#  POSITIVE: the shapes it must leave alone                                    #
# --------------------------------------------------------------------------- #

def test_tracked_mutations_count_as_produced(tmp_path, monkeypatch):
    route = '''
from fastapi import FastAPI

app = FastAPI()


def _extra():
    return {"from_helper": 1}


@app.get("/api/thing/status")
async def thing_status():
    body = {"installed": True}
    body["assigned"] = 1
    body.update({"literal": 2})
    body.update(kw=3)
    body.setdefault("default", 4)
    body.update(_extra())
    return body
'''
    test = '''
def test_all_produced(client):
    body = client.get("/api/thing/status").json()
    assert body["installed"] and body["assigned"] and body["literal"]
    assert body["kw"] and body["default"] and body["from_helper"]
    assert body["gone"]
'''
    problems = _run(tmp_path, monkeypatch, route=route, test=test)
    assert len(problems) == 1 and "'gone'" in problems[0], problems
    assert "assigned, default, from_helper, installed, kw, literal" in problems[0]


def test_open_shapes_are_not_judged(tmp_path, monkeypatch):
    route = '''
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from localm.other import build, Model

app = FastAPI()


@app.get("/api/open/call")
async def open_call():
    return build()


@app.get("/api/open/update")
async def open_update():
    body = {"a": 1}
    body.update(build())
    return body


@app.get("/api/open/spread")
async def open_spread(extra):
    return {"a": 1, **extra}


@app.get("/api/open/model", response_model=Model)
async def open_model():
    return {"a": 1}


@app.get("/api/open/list")
async def open_list():
    return [{"a": 1}]

'''
    test = '''
def test_open(client):
    assert client.get("/api/open/call").json()["gone"]
    assert client.get("/api/open/update").json()["gone"]
    assert client.get("/api/open/spread").json()["gone"]
    assert client.get("/api/open/model").json()["gone"]
    assert client.get("/api/open/list").json()["gone"]
'''
    problems = _run(tmp_path, monkeypatch, route=route, test=test)
    assert problems == [], problems


def test_negative_membership_get_and_raises_are_not_reads(tmp_path, monkeypatch):
    test = '''
import pytest


def test_soft(client):
    body = client.get("/api/thing/status").json()
    assert "gone" not in body
    assert body.get("gone") is None
    with pytest.raises(KeyError):
        body["gone"]
'''
    assert _run(tmp_path, monkeypatch, test=test) == []


def test_error_keys_are_always_allowed(tmp_path, monkeypatch):
    test = '''
def test_errors(client):
    body = client.get("/api/thing/status").json()
    assert body["detail"] and body["errors"]
'''
    assert _run(tmp_path, monkeypatch, test=test) == []


def test_nested_reads_only_judge_the_first_level(tmp_path, monkeypatch):
    test = '''
def test_nested(client):
    body = client.get("/api/thing/status").json()
    assert body["state"]["gone"] == 1
'''
    assert _run(tmp_path, monkeypatch, test=test) == []


def test_a_path_the_test_file_serves_itself_is_skipped(tmp_path, monkeypatch):
    test = '''
from fastapi import FastAPI

fake = FastAPI()


@fake.get("/api/thing/status")
def fake_status():
    return {"gone": 1}


def test_fake(client):
    body = client.get("/api/thing/status").json()
    assert body["gone"]
'''
    assert _run(tmp_path, monkeypatch, test=test) == []


def test_an_ambiguous_path_is_skipped(tmp_path, monkeypatch):
    route = '''
@app.get("/api/things/{name}")
async def one_thing(name: str):
    return {"name": name}


@app.get("/api/things/{path:path}")
async def any_thing(path: str):
    return {"path": path}
'''
    test = '''
def test_ambiguous(client):
    body = client.get("/api/things/abc").json()
    assert body["gone"]
'''
    assert _run(tmp_path, monkeypatch, route=_ROUTE + route, test=test) == []


def test_a_rebound_name_follows_its_latest_route(tmp_path, monkeypatch):
    route = '''
@app.get("/api/other")
async def other():
    return {"other": 1}
'''
    test = '''
def test_rebind(client):
    r = client.get("/api/thing/status")
    assert r.json()["state"]
    r = client.get("/api/other")
    assert r.json()["other"]
    r = "not a response"
    assert r.json()["gone"]
'''
    assert _run(tmp_path, monkeypatch, route=_ROUTE + route, test=test) == []


def test_unknown_route_and_non_path_receivers_are_skipped(tmp_path, monkeypatch):
    test = '''
import os


def test_unknown(client, cache):
    assert client.get("/api/nowhere").json()["gone"]
    assert os.environ.get("HOME")
    assert cache.get("key").json()["gone"]
'''
    assert _run(tmp_path, monkeypatch, test=test) == []


def test_missing_trees_are_a_no_op(tmp_path, monkeypatch):
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    assert ch._response_key_violations([]) == []


def test_a_write_into_the_body_is_not_a_read(tmp_path, monkeypatch):
    test = '''
def test_write(client):
    body = client.get("/api/thing/status").json()
    body["extra"] = 1
    del body["state"]
    client.post("/api/other", json=body)
'''
    assert _run(tmp_path, monkeypatch, test=test) == []


def test_unmodelled_handler_mutations_open_the_shape(tmp_path, monkeypatch):
    route = '''
from fastapi import FastAPI

app = FastAPI()


def _fill(d):
    d["filled"] = 1


@app.get("/api/open/call-arg")
async def call_arg():
    body = {"a": 1}
    _fill(body)
    return body


@app.get("/api/open/nested-setdefault")
async def nested_setdefault():
    out = {"a": 1}
    out.setdefault("warnings", []).append("x")
    return out


@app.get("/api/open/augassign")
async def augassign():
    body = {"a": 1}
    body |= {"b": 2}
    return body


@app.get("/api/open/alias")
async def alias():
    body = {"a": 1}
    out = body
    out["b"] = 2
    return body


@app.get("/api/open/closure")
async def closure():
    body = {"a": 1}

    def add(k):
        body[k] = 1

    add("b")
    return body


@app.get("/api/open/read-in-value")
async def read_in_value():
    body = {"a": 1}
    body["b"] = body["a"] + 1
    return body
'''
    test = '''
def test_open(client):
    assert client.get("/api/open/call-arg").json()["filled"]
    assert client.get("/api/open/nested-setdefault").json()["warnings"]
    assert client.get("/api/open/augassign").json()["b"]
    assert client.get("/api/open/alias").json()["b"]
    assert client.get("/api/open/closure").json()["b"]
    assert client.get("/api/open/read-in-value").json()["b"]
'''
    assert _run(tmp_path, monkeypatch, route=route, test=test) == []


def test_a_path_variable_differing_between_branches_is_dropped(tmp_path, monkeypatch):
    route = '''
@app.get("/api/other")
async def other():
    return {"other": 1}
'''
    test = '''
def test_branches(client, flag):
    if flag:
        url = "/api/thing/status"
    else:
        url = "/api/other"
    body = client.get(url).json()
    assert body["state"] or body["other"]
'''
    assert _run(tmp_path, monkeypatch, route=_ROUTE + route, test=test) == []


def test_own_routes_match_by_template_and_router_prefix(tmp_path, monkeypatch):
    route = '''
@app.get("/api/things/{name}")
async def one_thing(name: str):
    return {"name": name}


@app.get("/api/x/status")
async def x_status():
    return {"x": 1}
'''
    test = '''
from fastapi import APIRouter, FastAPI

fake = FastAPI()
fake_router = APIRouter(prefix="/api/x")


@fake.get("/api/things/{name}")
def fake_thing(name):
    return {"gone": 1}


@fake_router.get("/status")
def fake_status():
    return {"gone": 1}


def test_fake(client):
    assert client.get("/api/things/abc").json()["gone"]
    assert client.get("/api/x/status").json()["gone"]
'''
    assert _run(tmp_path, monkeypatch, route=_ROUTE + route, test=test) == []


def test_a_literal_and_a_parameterised_template_both_matching_is_ambiguous(tmp_path, monkeypatch):
    route = '''
@app.get("/api/things/{name}")
async def one_thing(name: str):
    return {"name": name}


@app.get("/api/things/abc")
async def the_abc_thing():
    return {"abc": 1}
'''
    test = '''
def test_ambiguous(client):
    assert client.get("/api/things/abc").json()["name"]
'''
    assert _run(tmp_path, monkeypatch, route=_ROUTE + route, test=test) == []


def test_a_function_mentioning_the_routes_helper_is_not_judged(tmp_path, monkeypatch):
    route = '''
from fastapi import FastAPI

app = FastAPI()


def _build():
    return {"built": 1}


@app.get("/api/thing/status")
async def thing_status():
    return _build()
'''
    test = '''
def test_patched(client, monkeypatch):
    monkeypatch.setattr("localm.routes._build", lambda: {"patched": 1})
    assert client.get("/api/thing/status").json()["patched"]


def test_unpatched(client):
    assert client.get("/api/thing/status").json()["gone"]
'''
    problems = _run(tmp_path, monkeypatch, route=route, test=test)
    assert len(problems) == 1 and "test_unpatched" in problems[0], problems


def test_a_name_shadowed_by_a_comprehension_or_lambda_is_not_judged(tmp_path, monkeypatch):
    test = '''
def test_shadow(client, others):
    body = client.get("/api/thing/status").json()
    assert all("gone" in body for body in others)
    assert any((lambda body: body["gone"])(o) for o in others)
    assert body["state"]
'''
    assert _run(tmp_path, monkeypatch, test=test) == []


def test_a_rebinding_through_json_and_a_try_star_block_are_followed(tmp_path, monkeypatch):
    test = '''
def test_rebind(client):
    r = client.get("/api/thing/status")
    r = r.json()
    assert r["gone"]


def test_try_star(client):
    try:
        body = client.get("/api/thing/status").json()
    except* ValueError:
        raise
    assert body["also_gone"]
'''
    problems = _run(tmp_path, monkeypatch, test=test)
    assert len(problems) == 2, problems
    assert "'gone'" in problems[0] and "'also_gone'" in problems[1]


# --------------------------------------------------------------------------- #
#  The real tree                                                               #
# --------------------------------------------------------------------------- #

_MOTIVATING_ROUTE = REPO_ROOT / "localm" / "plugins" / "gui" / "routes" / "comfy.py"
_MOTIVATING_TEST = REPO_ROOT / "tests" / "test_managed_comfy_s5_gui.py"


def test_real_tree_is_clean():
    ch = _load_check_hygiene()
    tracked = ch._tracked_files()
    assert tracked, "git ls-files returned nothing"
    assert ch._response_key_violations(tracked) == []


def test_the_motivating_route_is_judged():
    ch = _load_check_hygiene()
    routes = ch._collect_json_routes([_MOTIVATING_ROUTE])
    handlers = routes[("GET", "/api/comfy/managed-status")]
    keys = handlers[0][3]
    assert keys is not None, "the managed-status shape must stay readable"
    assert {"installed", "state", "target", "managed_active", "update_available"} <= keys


def test_dropping_the_motivating_field_again_is_caught(tmp_path, monkeypatch):
    route_src = _MOTIVATING_ROUTE.read_text(encoding="utf-8")
    assert route_src.count('"state": state,') == 1
    files = {
        "localm/plugins/gui/routes/comfy.py": route_src.replace('"state": state,', ""),
        "tests/test_managed_comfy_s5_gui.py": _MOTIVATING_TEST.read_text(encoding="utf-8"),
    }
    problems = _check(tmp_path, monkeypatch, files)
    assert problems, "the dropped field went unnoticed"
    assert all("reads key 'state' from the JSON of GET /api/comfy/managed-status" in p
               for p in problems), problems
