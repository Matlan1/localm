# SPDX-License-Identifier: AGPL-3.0-or-later
"""GET /api/changelog serves the release CHANGELOG.md for the Settings 'Show changelog' button: read-only, scoped like its Updates sibling."""

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


def test_changelog_endpoint_serves_released_history_without_the_unreleased_section(
        monkeypatch, tmp_path):
    """RELEASED history, newest first - and NOT the in-progress section."""
    _open_mode(monkeypatch)
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n### Added\n- an unshipped thing\n\n"
        "### Fixed\n- an unshipped security fix\n\n"
        "## [0.1.0] - 2026-07-04\n\nFirst tagged release.\n",
        encoding="utf-8")
    monkeypatch.setattr(updater, "repo_root", lambda: tmp_path)
    data = _get(create_app(_engine()), "/api/changelog").json()
    assert data["available"] is True
    md = data["markdown"]

    # Gone: the heading AND its content. The bullets are the payload - a strip that
    # removed only the heading would leave the disclosure intact.
    assert "[Unreleased]" not in md
    assert "an unshipped thing" not in md
    assert "an unshipped security fix" not in md

    # Kept: the released record, in full.
    assert "## [0.1.0]" in md
    assert "First tagged release." in md
    assert "# Changelog" in md, "the file header is not part of the unreleased section"


def test_changelog_endpoint_serves_a_file_with_no_unreleased_section_unchanged(
        monkeypatch, tmp_path):
    """The other arm."""
    _open_mode(monkeypatch)
    original = ("# Changelog\n\n## [0.2.0] - 2026-08-01\n\nSecond release.\n\n"
                "## [0.1.0] - 2026-07-04\n\nFirst tagged release.\n")
    (tmp_path / "CHANGELOG.md").write_text(original, encoding="utf-8")
    monkeypatch.setattr(updater, "repo_root", lambda: tmp_path)
    data = _get(create_app(_engine()), "/api/changelog").json()
    assert data["markdown"] == original


def test_changelog_endpoint_keeps_published_prereleases(monkeypatch, tmp_path):
    """A prerelease is PUBLISHED - it is on GitHub, so it shipped, and it stays."""
    _open_mode(monkeypatch)
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n- not shipped\n\n"
        "## [0.1.5rc2] - 2026-08-08\n\nA published prerelease.\n",
        encoding="utf-8")
    monkeypatch.setattr(updater, "repo_root", lambda: tmp_path)
    md = _get(create_app(_engine()), "/api/changelog").json()["markdown"]
    assert "## [0.1.5rc2]" in md
    assert "A published prerelease." in md
    assert "not shipped" not in md


def test_changelog_endpoint_strips_an_unreleased_only_file_rather_than_serving_it(
        monkeypatch, tmp_path):
    """`[Unreleased]` last/only (a project with no releases yet) runs to end of file."""
    _open_mode(monkeypatch)
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\nSome header prose.\n\n## [Unreleased]\n\n- nothing shipped yet\n",
        encoding="utf-8")
    monkeypatch.setattr(updater, "repo_root", lambda: tmp_path)
    md = _get(create_app(_engine()), "/api/changelog").json()["markdown"]
    assert "nothing shipped yet" not in md
    assert "[Unreleased]" not in md
    assert "Some header prose." in md, "only the section goes, not the whole document"


def test_strip_unreleased_cannot_eat_a_following_release(tmp_path):
    """The failure mode a regex span would have: matching past the section's end and silently removing a real release."""
    from localm.inference.routes.admin import _strip_unreleased
    src = ("# Changelog\n\n## [Unreleased]\n\n- pending\n\n"
           "## [0.3.0] - 2026-08-09\n\n- three\n\n"
           "## [0.2.0] - 2026-08-01\n\n- two\n\n"
           "## [0.1.0] - 2026-07-04\n\n- one\n")
    out = _strip_unreleased(src)
    assert "- pending" not in out
    for kept in ("## [0.3.0]", "- three", "## [0.2.0]", "- two", "## [0.1.0]", "- one"):
        assert kept in out, f"{kept} was eaten by the strip"


def test_changelog_endpoint_missing_file_is_honest(monkeypatch, tmp_path):
    """No CHANGELOG in this build -> available:false, never a faked empty success."""
    _open_mode(monkeypatch)
    monkeypatch.setattr(updater, "repo_root", lambda: tmp_path)  # empty dir, no file
    data = _get(create_app(_engine()), "/api/changelog").json()
    assert data["available"] is False


def test_changelog_endpoint_serves_the_real_shipped_file(monkeypatch):
    """End-to-end with the REAL repo CHANGELOG.md (no monkeypatch): it is found and contains a version section - proving repo_root() resolution actually works."""
    _open_mode(monkeypatch)
    data = _get(create_app(_engine()), "/api/changelog").json()
    assert data["available"] is True
    md = data["markdown"]
    assert "# Changelog" in md
    assert "## [0.1.0]" in md
    # The HEADING, not the bare substring. Measured on the real file: "[Unreleased]"
    # also occurs in the header sentence explaining the convention and in a bullet
    # INSIDE a released section that refers back to a correction made in the
    # unreleased one. Asserting the substring is absent would fail on both of those
    # and would push a future fix toward deleting text out of a shipped release.
    assert "## [Unreleased]" not in md, (
        "the real shipped changelog must reach users without its in-progress section")
    assert "\n[Unreleased]:" not in md, "the dangling link definition goes too"


def test_strip_leaves_unreleased_MENTIONS_inside_released_sections_alone(tmp_path):
    """The over-reach guard, and the real file genuinely has this shape: a shipped release's notes can refer back to the unreleased section."""
    from localm.inference.routes.admin import _strip_unreleased
    src = ("# Changelog\n\n"
           "The `[Unreleased]` section is maintained until it is cut.\n\n"
           "## [Unreleased]\n\n- pending work\n\n"
           "## [0.2.0] - 2026-08-09\n\n"
           "- corrected the `[Unreleased]` entry that claimed otherwise\n\n"
           "[Unreleased]: https://example.invalid/compare/v0.1.0...HEAD\n"
           "[0.2.0]: https://example.invalid/releases/v0.2.0\n")
    out = _strip_unreleased(src)
    assert "- pending work" not in out
    assert "## [Unreleased]" not in out
    assert "[Unreleased]: https://" not in out, "the dangling link definition goes"
    # ...but everything that merely MENTIONS it, in prose or in shipped notes, stays.
    assert "The `[Unreleased]` section is maintained until it is cut." in out
    assert "- corrected the `[Unreleased]` entry that claimed otherwise" in out
    assert "[0.2.0]: https://example.invalid/releases/v0.2.0" in out
