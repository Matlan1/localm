# SPDX-License-Identifier: AGPL-3.0-or-later
"""The live-read version signal (localm/_version) the self-updater compares against
the latest release tag. Editable installs do not refresh dist-info on a code swap,
so the running version MUST come from the VERSION file at runtime, not metadata."""

from localm import _version


def test_read_version_from_file(tmp_path, monkeypatch):
    vf = tmp_path / "VERSION"
    vf.write_text("0.9.3\n", encoding="utf-8")
    monkeypatch.setattr(_version, "version_file", lambda: vf)
    assert _version.read_version() == "0.9.3"


def test_read_version_falls_back_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(_version, "version_file", lambda: tmp_path / "nope")
    v = _version.read_version()
    assert isinstance(v, str) and v   # installed metadata or 'unknown', never a raise


def test_repo_version_file_is_present_and_read():
    # The real repo-root VERSION file must exist and be what read_version reports.
    assert _version.version_file().is_file(), "repo must ship a VERSION file"
    assert _version.read_version() == _version.version_file().read_text(
        encoding="utf-8").strip()


def test_normalize_strips_leading_v():
    assert _version.normalize("v0.2.0") == "0.2.0"
    assert _version.normalize("0.2.0") == "0.2.0"
    assert _version.normalize("V1.0") == "1.0"
    assert _version.normalize("vibe") == "vibe"   # not a version tag, left as-is


def test_is_newer_basic():
    assert _version.is_newer("0.2.0", "0.1.0") is True
    assert _version.is_newer("v0.2.0", "0.1.9") is True
    assert _version.is_newer("1.0.0", "0.9.9") is True
    assert _version.is_newer("0.1.0", "0.1.0") is False
    assert _version.is_newer("0.1.0", "0.2.0") is False


def test_is_newer_edge_cases():
    assert _version.is_newer("0.2.0", "unknown") is True   # no signal -> offer
    assert _version.is_newer("", "0.1.0") is False          # empty candidate
    assert _version.is_newer("0.2", "0.1.5") is True         # uneven lengths
    assert _version.is_newer("0.1", "0.1.0") is False        # 0.1 == 0.1.0
    assert _version.is_newer("0.1.0.1", "0.1.0") is True
