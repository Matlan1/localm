# SPDX-License-Identifier: AGPL-3.0-or-later
"""GET /api/changelog serves the release CHANGELOG.md for the Settings "Show
changelog" button: read-only, scoped like its Updates sibling. Returns
{available, version, markdown} or {available: false} when the file is absent
from this build (surfaced honestly, never a fake empty success). The changelog
path is resolved via updater.repo_root() so it is correct in dev AND in an
installed release; tests monkeypatch it to a throwaway dir for hermetic runs."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from localm import updater
from localm.inference.http_server import create_app


def _engine():
    e = MagicMock()
    e.display_name = "m"
    e.loaded = True
    return e


def _get(app, path):
    with TestClient(app) as c:
        return c.get(path, headers={"Authorization": f"Bearer {app.state.shell_token}"})


def _open_mode(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)


def test_changelog_endpoint_serves_markdown(monkeypatch, tmp_path):
    """The route returns the full changelog markdown, version history and all."""
    _open_mode(monkeypatch)
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n### Added\n- a new thing\n\n"
        "## [0.1.0] - 2026-07-04\n\nFirst tagged release.\n",
        encoding="utf-8")
    monkeypatch.setattr(updater, "repo_root", lambda: tmp_path)
    data = _get(create_app(_engine()), "/api/changelog").json()
    assert data["available"] is True
    # Full history, newest first: both the newest section and an older release
    # section must be present (this is the whole point - it is the record).
    assert "## [Unreleased]" in data["markdown"]
    assert "## [0.1.0]" in data["markdown"]
    assert "First tagged release." in data["markdown"]


def test_changelog_endpoint_missing_file_is_honest(monkeypatch, tmp_path):
    """No CHANGELOG in this build -> available:false, never a faked empty success."""
    _open_mode(monkeypatch)
    monkeypatch.setattr(updater, "repo_root", lambda: tmp_path)  # empty dir, no file
    data = _get(create_app(_engine()), "/api/changelog").json()
    assert data["available"] is False


def test_changelog_endpoint_serves_the_real_shipped_file(monkeypatch):
    """End-to-end with the REAL repo CHANGELOG.md (no monkeypatch): it is found and
    contains a version section - proving repo_root() resolution actually works."""
    _open_mode(monkeypatch)
    data = _get(create_app(_engine()), "/api/changelog").json()
    assert data["available"] is True
    assert "# Changelog" in data["markdown"]
    assert "## [0.1.0]" in data["markdown"]
