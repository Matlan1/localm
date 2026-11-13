# SPDX-License-Identifier: AGPL-3.0-or-later
"""An error answer carries the error field its protocol's clients read, beside
the "detail" localm's own pages read: OpenAI's error object on /v1 paths,
Ollama's "error" string on the Ollama paths, and "detail" alone elsewhere."""

from __future__ import annotations

import os
from unittest.mock import patch

from fastapi.testclient import TestClient

from localm.inference.app_assembly.errors import error_body
from localm.inference.http_server import create_app

SECRET = "protocol-error-bodies-owner-key"


def test_an_openai_path_refusal_carries_the_openai_error_object():
    with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
        client = TestClient(create_app(None))
        r = client.get("/v1/models")
    assert r.status_code == 401
    detail = r.json()["detail"]
    assert r.json() == {"detail": detail,
                        "error": {"message": detail, "type": "authentication_error",
                                  "param": None, "code": None}}


def test_an_openai_path_validation_error_carries_the_openai_error_object():
    with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
        client = TestClient(create_app(None))
        r = client.post("/v1/chat/completions", json={"messages": "not a list"},
                        headers={"Authorization": f"Bearer {SECRET}"})
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["message"] == body["detail"]
    assert "errors" in body


def test_a_gui_path_refusal_keeps_detail_alone():
    with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
        client = TestClient(create_app(None))
        r = client.get("/api/session/whoami-not-a-route")
    assert set(r.json()) == {"detail"}


def test_an_unhandled_os_error_names_its_cause_without_the_path():
    with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
        app = create_app(None)

        @app.get("/v1/probe-os-error")
        def _boom():
            raise PermissionError(13, "Permission denied", "D:/secret/folder/file.bin")

        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/v1/probe-os-error", headers={"Authorization": f"Bearer {SECRET}"})
    assert r.status_code == 500
    assert r.json()["detail"] == "Internal server error: Permission denied"
    assert r.json()["error"]["type"] == "server_error"
    assert "secret" not in r.text


def test_the_body_shape_per_path():
    assert error_body("/api/pull", 403, "no") == {"detail": "no", "error": "no"}
    assert error_body("/api/chat", 400, "no") == {"detail": "no", "error": "no"}
    assert error_body("/api/blobs/sha256:ab", 404, "no")["error"] == "no"
    assert error_body("/v1/messages", 401, "no")["error"]["type"] == "authentication_error"
    assert error_body("/v1/messages", 401, "no")["type"] == "error"
    assert error_body("/v1/embeddings", 503, "no")["error"]["type"] == "server_error"
    assert error_body("/api/plugins", 403, "no") == {"detail": "no"}
    assert error_body("/health", 503, "no") == {"detail": "no"}


def test_the_open_mode_gate_answers_an_ollama_client_in_its_shape():
    from localm import auth
    assert not auth.any_key_configured()
    client = TestClient(create_app(None))
    r = client.post("/api/pull", json={"model": "x"})
    assert r.status_code == 403
    assert r.json()["error"] == r.json()["detail"]
    assert "API key" in r.json()["error"]


def test_an_os_error_without_a_file_stays_generic():
    with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
        app = create_app(None)

        @app.get("/v1/probe-connect-error")
        def _boom():
            raise ConnectionRefusedError(10061, "Connect call failed ('10.0.0.9', 8188)")

        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/v1/probe-connect-error", headers={"Authorization": f"Bearer {SECRET}"})
    assert r.status_code == 500
    assert r.json()["detail"] == "Internal server error"
    assert "10.0.0.9" not in r.text


def test_every_ollama_management_path_is_a_route():
    from localm.inference import ollama_protocol as O
    paths = {getattr(r, "path", None) for r in create_app(None).routes}
    assert O.MANAGEMENT_PATHS <= paths
