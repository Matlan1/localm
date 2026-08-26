# SPDX-License-Identifier: AGPL-3.0-or-later
"""A single global exception handler: an unexpected error in any route returns a
clean JSON 500 carrying no traceback and no exception detail, and the server
stays responsive."""

from fastapi.testclient import TestClient

from localm.inference.http_server import create_app


def test_exception_handler_is_registered():
    app = create_app(None)
    assert Exception in app.exception_handlers


def test_unhandled_error_returns_json_500_without_leak():
    app = create_app(None)
    secret = "leak-marker-do-not-show-7731"

    async def _boom():
        raise RuntimeError(secret)

    app.add_api_route("/_boom_test", _boom, methods=["GET"])
    client = TestClient(app, raise_server_exceptions=False)

    r = client.get("/_boom_test")
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal server error"}
    assert secret not in r.text  # the exception detail must not reach the client


def test_server_stays_up_after_an_error():
    """A failing request must not wedge the app - a following request still works."""
    app = create_app(None)

    async def _boom():
        raise RuntimeError("boom")

    app.add_api_route("/_boom_test", _boom, methods=["GET"])
    client = TestClient(app, raise_server_exceptions=False)

    assert client.get("/_boom_test").status_code == 500
    assert client.get("/_boom_test").status_code == 500
