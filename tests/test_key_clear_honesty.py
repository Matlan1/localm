"""A key clear that FAILED must never report success.

An undeletable ``auth.key`` (an AV or indexer lock, a permissions or profile
change) must not produce a green tick on the CLI or ``{"cleared": true}`` from
the API while the key on disk still grants access.

These drive the real functions and assert from OUTSIDE the call. The failure is
injected at the lowest honest point - ``Path.unlink`` raising ``OSError``, which
is exactly what a locked file does - so every layer above it runs for real.
"""

import pytest

from localm import auth
from localm.cli import main


@pytest.fixture
def runner(cli_runner, monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    return cli_runner


def _unlink_denied_for(monkeypatch, *names):
    """Make ``Path.unlink`` raise for the named files only, as a real lock does.

    Everything else still deletes for real, so a test that expects a PARTIAL
    failure genuinely exercises the partial path.
    """
    from pathlib import Path
    real = Path.unlink

    def fake(self, *a, **kw):
        if self.name in names:
            raise PermissionError(13, "The process cannot access the file")
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", fake)


class TestClearApiKeyReportsWhatSurvived:
    def test_clean_clear_returns_no_failures(self, runner):
        runner.invoke(main, ["key", "generate"])
        assert auth.get_api_key() is not None
        assert auth.clear_api_key() == []
        assert auth.get_api_key() is None

    def test_undeletable_key_file_is_reported(self, runner, monkeypatch):
        runner.invoke(main, ["key", "generate"])
        key_name = auth.key_file().name
        _unlink_denied_for(monkeypatch, key_name)

        failures = auth.clear_api_key()

        assert failures, "an undeletable key file must be reported, not swallowed"
        assert any(key_name in f["path"] for f in failures)
        # The credential SURVIVED, so the caller must be able to tell.
        assert auth.key_file().exists()

    def test_report_names_the_reason_not_just_the_file(self, runner, monkeypatch):
        runner.invoke(main, ["key", "generate"])
        _unlink_denied_for(monkeypatch, auth.key_file().name)
        failures = auth.clear_api_key()
        assert any("PermissionError" in f["error"] for f in failures)

    def test_the_failure_is_also_written_to_the_local_log(
            self, runner, monkeypatch, caplog):
        """The HTTP route logs nothing of its own; clear_api_key records each
        failure here with the path and the OS error, which is the route's only
        local signal."""
        import logging
        runner.invoke(main, ["key", "generate"])
        key_name = auth.key_file().name
        _unlink_denied_for(monkeypatch, key_name)

        with caplog.at_level(logging.WARNING, logger="localm"):
            auth.clear_api_key()

        warned = [r.getMessage() for r in caplog.records
                  if r.levelno >= logging.WARNING]
        assert any(key_name in m for m in warned), (
            "auth.clear_api_key must name the file it could not remove")
        assert any("PermissionError" in m or "cannot access" in m
                   for m in warned), "and must say why it failed"

    def test_label_is_path_free_so_a_network_caller_can_use_it(
            self, runner, monkeypatch):
        """``what`` must never carry the path: it is the ONLY field the HTTP
        route puts on the wire."""
        runner.invoke(main, ["key", "generate"])
        _unlink_denied_for(monkeypatch, auth.key_file().name)
        home = str(auth.key_file().parent)
        for f in auth.clear_api_key():
            assert home not in f["what"]
            assert "Error" not in f["what"]


class TestKeyClearCliDoesNotClaimSuccess:
    def test_success_still_says_cleared(self, runner):
        runner.invoke(main, ["key", "generate"])
        r = runner.invoke(main, ["key", "clear", "--yes"])
        assert r.exit_code == 0
        assert "cleared" in r.output.lower()
        assert "not fully cleared" not in r.output.lower()

    def test_failed_clear_does_not_print_a_success_tick(self, runner, monkeypatch):
        runner.invoke(main, ["key", "generate"])
        _unlink_denied_for(monkeypatch, auth.key_file().name)

        r = runner.invoke(main, ["key", "clear", "--yes"])

        out = r.output.lower()
        assert "not fully cleared" in out, (
            "a failed clear reported as success is the rule-5 violation itself")
        assert "may still grant access" in out
        assert "api key cleared - open mode" not in out


class TestClearRouteDoesNotClaimSuccess:
    """The GUI renders open-mode state from ``cleared``, so a false true is
    shown to the user as a keyless install."""

    def _client(self):
        from fastapi.testclient import TestClient
        from localm.inference.http_server import create_app
        return TestClient(create_app(None))

    def test_clean_clear_reports_cleared(self, runner):
        runner.invoke(main, ["key", "generate"])
        c = self._client()
        r = c.post("/api/auth/key/clear",
                   headers={"Authorization": f"Bearer {auth.get_api_key()}"})
        assert r.status_code == 200
        assert r.json() == {"cleared": True, "warnings": []}

    def test_failed_clear_reports_not_cleared_with_reasons(self, runner, monkeypatch):
        runner.invoke(main, ["key", "generate"])
        key = auth.get_api_key()
        _unlink_denied_for(monkeypatch, auth.key_file().name)

        c = self._client()
        r = c.post("/api/auth/key/clear", headers={"Authorization": f"Bearer {key}"})

        assert r.status_code == 200, "the session revocation half DID happen"
        body = r.json()
        assert body["cleared"] is False, (
            "cleared:true while the key file survives is a lie the GUI shows as "
            "open mode")
        assert body["warnings"], "the caller must learn WHAT survived"
        assert auth.key_file().exists()

    def test_response_discloses_no_path_and_no_exception_text(
            self, runner, monkeypatch):
        """The absolute path carries the account name and raw OS exception text
        is stack-trace exposure, so neither may ride out on an HTTP response.
        The local CLI still shows both."""
        runner.invoke(main, ["key", "generate"])
        key = auth.get_api_key()
        _unlink_denied_for(monkeypatch, auth.key_file().name)
        home = str(auth.key_file().parent)

        c = self._client()
        r = c.post("/api/auth/key/clear", headers={"Authorization": f"Bearer {key}"})

        blob = r.text
        assert home not in blob, "the data dir path must not go on the wire"
        assert "PermissionError" not in blob, "no raw exception text on the wire"
        assert "the API key file" in blob, "but it must still say WHAT survived"
