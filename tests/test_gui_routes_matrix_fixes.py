# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI server routes: a log export makes a new folder every time and none when
there is nothing to export, an upload names the files it did not save, and an
API path no route answers gets a JSON answer that says why."""

import time

import pytest
from fastapi.testclient import TestClient

from tests.test_upload import scoped_app  # noqa: F401


@pytest.fixture
def client(scoped_app):  # noqa: F811
    with TestClient(scoped_app) as c:
        yield c


def _logs(scoped_app):  # noqa: F811
    from localm.debuglog import logs_dir
    d = logs_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------- log export

def test_export_with_no_logs_creates_no_folder(client, scoped_app, tmp_path):  # noqa: F811
    from localm.config import home_dir
    from localm.debuglog import logs_dir
    for src in (logs_dir(), home_dir()):
        if src.is_dir():
            for p in src.glob("*.log"):
                p.unlink()
    dest = tmp_path / "out"
    dest.mkdir()
    r = client.post("/api/logs/export", json={"dest": str(dest)})
    assert r.status_code == 200, r.text
    assert r.json()["copied"] == 0 and r.json()["found"] == 0
    assert list(dest.iterdir()) == []


def test_two_exports_in_one_second_get_two_folders(client, scoped_app,  # noqa: F811
                                                   tmp_path, monkeypatch):
    (_logs(scoped_app) / "a.log").write_text("first", encoding="utf-8")
    monkeypatch.setattr(time, "strftime", lambda fmt, *a: "20260101-000000")
    dest = tmp_path / "out"
    dest.mkdir()
    first = client.post("/api/logs/export", json={"dest": str(dest)}).json()
    (_logs(scoped_app) / "a.log").write_text("second", encoding="utf-8")
    second = client.post("/api/logs/export", json={"dest": str(dest)}).json()
    assert first["dest"] != second["dest"]
    folders = sorted(p.name for p in dest.iterdir())
    assert folders == ["localm-logs-20260101-000000", "localm-logs-20260101-000000-2"]
    assert (dest / folders[0] / "a.log").read_text(encoding="utf-8") == "first"
    assert (dest / folders[1] / "a.log").read_text(encoding="utf-8") == "second"


# ---------------------------------------------------------------- uploads

def test_upload_names_an_empty_file_it_skipped(client):
    r = client.post("/api/upload", files=[("file", ("good.txt", b"data")),
                                          ("file", ("empty.txt", b""))])
    assert r.status_code == 200, r.text
    body = r.json()
    assert [u["name"] for u in body["uploaded"]] == ["good.txt"]
    assert body["skipped"] == [{"name": "empty.txt", "reason": "the file is empty"}]
    assert "empty.txt: the file is empty" in body["warning"]


def test_upload_of_only_an_empty_file_says_which(client):
    r = client.post("/api/upload", files={"file": ("empty.txt", b"")})
    assert r.status_code == 400
    assert "empty.txt: the file is empty" in r.json()["detail"]


def test_upload_with_nothing_skipped_has_no_warning(client):
    body = client.post("/api/upload", files={"file": ("a.txt", b"x")}).json()
    assert "skipped" not in body and "warning" not in body


# ------------------------------------------------- unanswered API paths

def test_route_of_a_plugin_not_installed_names_the_plugin(client):
    r = client.post("/api/web/search", json={"query": "x"})
    assert r.status_code == 404
    assert r.json()["detail"].startswith("The web plugin is not installed or not enabled")


def test_route_of_a_disabled_plugin_says_enable_it(client, scoped_app):  # noqa: F811
    mgr = scoped_app.state.plugin_manager
    mgr.set_installed_state("web", True)
    mgr.set_enabled_state("web", False)
    r = client.post("/api/web/search", json={"query": "x"})
    assert r.status_code == 404
    assert "The web plugin is not installed or not enabled" in r.json()["detail"]
    assert "localm plugin enable web" in r.json()["detail"]


def test_unknown_path_under_an_active_plugin_is_no_such_endpoint(client):
    r = client.get("/api/chat/no-such-route")
    assert r.status_code == 404
    assert r.json()["detail"] == "No such API endpoint: GET /api/chat/no-such-route"


def test_unknown_api_path_is_a_json_404(client):
    r = client.post("/api/no-such-thing/x")
    assert r.status_code == 404
    assert r.json()["detail"] == "No such API endpoint: POST /api/no-such-thing/x"


def test_existing_api_route_with_another_method_is_405(client):
    r = client.delete("/api/upload")
    assert r.status_code == 405


def test_static_files_are_still_served(client):
    r = client.get("/sw.js")
    assert r.status_code == 200
    r = client.get("/index.html")
    assert r.status_code == 200
