# SPDX-License-Identifier: AGPL-3.0-or-later
"""GET /api/activity: discover what a server is doing WITHOUT holding a job id.

Every other way to reach a job needs an id the caller already has, and that id
is handed out exactly once - in the body of the POST that started the job. A
second browser tab, a second device, or the same tab after a reload therefore
cannot learn that a model pull is running, even though the server has the whole
record. This route answers "what is happening" without being told what to look
for.

Two properties get equal weight here:

* A caller that never saw the id MUST find the operation.
* A caller must NOT see another principal's operation on a keyed server, and
  the same 404-style silence the events route uses applies - absence from the
  list, never a redacted entry proving one exists.

The default configuration is open mode, where there are no owners at all, so
the filter admits everyone. That is correct for a single-owner local server and
is pinned below: a per-principal list would show a user nothing on their own
machine.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import scopes as S
from localm.plugins.gui.jobs import Job
from localm.plugins.gui.web import attach_gui


@pytest.fixture
def app(tmp_path, monkeypatch):
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import attach_engine
    a = FastAPI()
    attach_engine(a)
    attach_gui(a, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda n: None, active_model=lambda: "model-a")
    return a


@pytest.fixture
def headless_app(tmp_path, monkeypatch):
    """A bare `localm serve`: attach_engine only, no GUI. The route must exist
    here too."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import attach_engine
    a = FastAPI()
    attach_engine(a)
    return a


def _hdr(key):
    return {"Authorization": f"Bearer {key}"}


def _inject(app, *, owner=None, kind="pull", label=None, status="running"):
    """Register a job directly, with a pre-queued end event so streaming it
    returns promptly. Mirrors test_key_scope_jobs.py's helper."""
    j = Job(id=uuid.uuid4().hex[:12], kind=kind, argv=["secret", "argv"],
            owner=owner, label=label)
    j.status = status
    j.push({"type": "end", "status": "done"})
    app.state.jobs._jobs[j.id] = j
    return j


# ------------------------------------------- finding an operation by listing

def test_a_client_that_never_saw_the_id_finds_the_operation(app):
    """The id is never given to this caller; it still finds it."""
    job = _inject(app, label="Model pull owner/repo")
    with TestClient(app) as c:
        r = c.get("/api/activity")
        assert r.status_code == 200, r.text
        ops = r.json()["operations"]
    assert [o["id"] for o in ops] == [job.id]
    assert ops[0]["kind"] == "pull"
    assert ops[0]["label"] == "Model pull owner/repo"
    assert ops[0]["status"] == "running"
    assert ops[0]["cancellable"] is True


def test_a_discovered_id_can_then_be_streamed(app):
    """Discovery is only useful if it hands back something attachable: the id
    from the listing must work on the events route."""
    _inject(app)
    with TestClient(app) as c:
        found = c.get("/api/activity").json()["operations"][0]["id"]
        r = c.get(f"/api/jobs/{found}/events")
        assert r.status_code == 200, r.text
        assert "data:" in r.text


def test_the_route_exists_on_a_headless_server(headless_app):
    """The registry is at kernel level, so a bare `localm serve` answers this
    too; the CLI and MCP surfaces read it."""
    _inject(headless_app, label="Indexing docs")
    with TestClient(headless_app) as c:
        r = c.get("/api/activity")
        assert r.status_code == 200, r.text
        assert [o["label"] for o in r.json()["operations"]] == ["Indexing docs"]


# ------------------------------------------------------------- keyed server

class TestKeyedServer:
    def test_another_principal_does_not_see_your_operation(self, app):
        from localm import auth
        a = auth.create_key("A", [S.MODELS_READ])["key"]
        b = auth.create_key("B", [S.MODELS_READ])["key"]
        _inject(app, owner=auth._hash_key(a))
        with TestClient(app) as c:
            ops = c.get("/api/activity", headers=_hdr(b)).json()["operations"]
        assert ops == [], "B must not learn that A has an operation running"

    def test_the_creator_sees_its_own_operation(self, app):
        from localm import auth
        a = auth.create_key("A", [S.MODELS_READ])["key"]
        job = _inject(app, owner=auth._hash_key(a))
        with TestClient(app) as c:
            ops = c.get("/api/activity", headers=_hdr(a)).json()["operations"]
        assert [o["id"] for o in ops] == [job.id]

    def test_admin_sees_every_operation(self, app, monkeypatch):
        from localm import auth
        a = auth.create_key("A", [S.MODELS_READ])["key"]
        job = _inject(app, owner=auth._hash_key(a))
        monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
        with TestClient(app) as c:
            ops = c.get("/api/activity",
                        headers=_hdr("ownersecret")).json()["operations"]
        assert [o["id"] for o in ops] == [job.id]

    def test_a_hidden_operation_leaves_no_trace_in_the_response(self, app):
        """Absent, not redacted. A placeholder row would confirm that another
        principal has something running, which is what the events route's
        indistinguishable 404 exists to prevent."""
        from localm import auth
        a = auth.create_key("A", [S.MODELS_READ])["key"]
        b = auth.create_key("B", [S.MODELS_READ])["key"]
        job = _inject(app, owner=auth._hash_key(a), label="Model pull private/repo")
        with TestClient(app) as c:
            body = c.get("/api/activity", headers=_hdr(b)).text
        assert job.id not in body
        assert "private/repo" not in body
        assert "operations" in body


# --------------------------------------------------------- open mode is the
#                                                            DEFAULT

def test_open_mode_shows_unowned_operations_to_any_caller(app):
    """With no owner key and no keystore - how localm runs out of the box -
    principal_id() returns None, so jobs are unowned and everyone sees them.

    This is CORRECT for a single-owner local server: a filter that demanded a
    matching principal would show the user an empty list on their own machine
    while the pull they just started was running.
    """
    from localm import auth

    # Precondition: this fixture is in open mode.
    assert not auth.any_key_configured(), (
        "precondition: this fixture must genuinely be open mode")

    job = _inject(app, owner=None)
    with TestClient(app) as c:
        ops = c.get("/api/activity").json()["operations"]
    assert [o["id"] for o in ops] == [job.id]


# ------------------------------------------------------------ payload shape

def test_pct_is_absent_until_something_reports_progress(app):
    """A pull that has not read a byte count is at an UNKNOWN percentage, not
    at 0 percent. A client must be able to tell those apart."""
    _inject(app)
    with TestClient(app) as c:
        op = c.get("/api/activity").json()["operations"][0]
    assert "pct" not in op
    assert "phase" not in op


def test_pct_appears_once_progress_is_reported(app):
    job = _inject(app)
    job.push({"type": "progress", "pct": 42.5, "phase": "download"})
    with TestClient(app) as c:
        op = c.get("/api/activity").json()["operations"][0]
    assert op["pct"] == 42.5
    assert op["phase"] == "download"


def test_the_response_never_carries_argv(app):
    """argv holds the resolved model spec and any host path the caller passed.
    Injected unowned, because an owned job is correctly INVISIBLE to an
    unauthenticated open-mode caller."""
    _inject(app, owner=None)
    with TestClient(app) as c:
        body = c.get("/api/activity").text
        op = c.get("/api/activity").json()["operations"][0]
    assert "argv" not in op
    assert "secret" not in body


def test_the_response_never_carries_the_owner_hash(app):
    """owner is a keystore hash and must not reach a listing. Checked on the
    KEYED path, since that is the only configuration where a job has an owner
    AND the caller is allowed to see it."""
    from localm import auth
    key = auth.create_key("A", [S.MODELS_READ])["key"]
    digest = auth._hash_key(key)
    job = _inject(app, owner=digest)
    with TestClient(app) as c:
        r = c.get("/api/activity", headers=_hdr(key))
    body = r.text
    ops = r.json()["operations"]
    assert [o["id"] for o in ops] == [job.id], "precondition: the owner can see it"
    assert "owner" not in ops[0]
    assert digest not in body


def test_finished_operations_are_listed_with_their_finish_time(app):
    job = _inject(app, status="done")
    job.mark_finished()
    with TestClient(app) as c:
        op = c.get("/api/activity").json()["operations"][0]
    assert op["status"] == "done"
    assert op["finished_at"] is not None
    assert op["cancellable"] is False


def test_newest_first(app):
    old = _inject(app, label="old")
    old.created_at = time.time() - 600
    _inject(app, label="new")
    with TestClient(app) as c:
        ops = c.get("/api/activity").json()["operations"]
    assert [o["label"] for o in ops] == ["new", "old"]


def test_an_idle_server_reports_an_empty_list_not_an_error(app):
    """An empty list is a real answer - "I looked, nothing is running" - and
    must be distinguishable by the client from "I have not asked yet" and from
    "I asked and could not reach the server". The server's part of that is
    answering 200 with an explicit empty list rather than erroring."""
    with TestClient(app) as c:
        r = c.get("/api/activity")
    assert r.status_code == 200
    assert r.json()["operations"] == []


def test_the_reply_carries_the_server_clock(app):
    """A client must not compute an operation's age against its OWN clock.

    created_at exists so a user can tell a six-second operation from a six-hour
    one, and a client subtracting a server epoch from its local Date.now() is
    wrong by however much the two clocks disagree. The reference clock ships
    alongside it.
    """
    job = _inject(app)
    job.created_at = time.time() - 120
    with TestClient(app) as c:
        body = c.get("/api/activity").json()
    assert "now" in body, "the client needs a reference clock, not its own"
    age = body["now"] - body["operations"][0]["created_at"]
    assert 110 < age < 130, f"age computed from the server clock was {age}"
