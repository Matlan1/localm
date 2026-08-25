# SPDX-License-Identifier: AGPL-3.0-or-later
"""Download a coder-created file (pull output onto a phone)."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.builtin.coder import plug


class _FakeSession:
    def __init__(self, cwd, tracked):
        self.cwd = cwd
        self._tracked = tracked

    def changed_files(self):
        return [{"path": p, "writes": 1, "created": True,
                 "exists": (Path(self.cwd) / p).is_file(), "last_tool": "write_file"}
                for p in self._tracked]


@pytest.fixture
def coder_client(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    # write_bytes (not write_text) so Windows does not translate \n -> \r\n.
    (tmp_path / "src" / "app.py").write_bytes(b"print('hi')\n")
    sess = _FakeSession(tmp_path, ["src/app.py", "gone.txt"])  # gone.txt not on disk
    monkeypatch.setattr(plug, "_get_session", lambda request, sid: sess)
    app = FastAPI()
    app.include_router(plug._router)
    return TestClient(app), tmp_path


def test_download_tracked_file(coder_client):
    client, _ = coder_client
    r = client.get("/api/coder/sessions/s1/files/download",
                   params={"path": "src/app.py"})
    assert r.status_code == 200
    assert r.content == b"print('hi')\n"
    assert "app.py" in r.headers.get("content-disposition", "")


def test_untracked_file_is_not_downloadable(coder_client):
    client, root = coder_client
    (root / "secret.txt").write_text("nope", encoding="utf-8")   # exists but untracked
    r = client.get("/api/coder/sessions/s1/files/download",
                   params={"path": "secret.txt"})
    assert r.status_code == 404


def test_path_traversal_rejected(coder_client):
    client, _ = coder_client
    r = client.get("/api/coder/sessions/s1/files/download",
                   params={"path": "../../../etc/passwd"})
    assert r.status_code in (400, 404)        # untracked + escaping -> refused


def test_tracked_but_deleted_file_404(coder_client):
    client, _ = coder_client
    r = client.get("/api/coder/sessions/s1/files/download",
                   params={"path": "gone.txt"})
    assert r.status_code == 404


def test_missing_path_is_400(coder_client):
    client, _ = coder_client
    r = client.get("/api/coder/sessions/s1/files/download")
    assert r.status_code == 400
