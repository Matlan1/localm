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


def test_is_newer_prerelease_truth_table():
    """The full truth table this fix is built from - drive is_newer() directly
    rather than asserting on the parser's internals, so a future refactor of
    _parse/_prerelease_suffix is free as long as this table still holds."""
    # A stable release always outranks a prerelease of the SAME numeric version -
    # the filed bug: upgrading FROM an rc TO the matching final release.
    assert _version.is_newer("0.1.4", "0.1.4-rc1") is True
    # ...and never the other direction (never "downgrade" final -> rc automatically).
    assert _version.is_newer("0.1.4-rc1", "0.1.4") is False
    # Between two prereleases of the same numeric version, the higher ordinal wins -
    # a second bug the truth table surfaced beyond the filed one (rc1<->rc2 tied
    # at False in BOTH directions before this fix, i.e. neither was ever offered
    # as an upgrade over the other).
    assert _version.is_newer("0.1.4-rc2", "0.1.4-rc1") is True
    assert _version.is_newer("0.1.4-rc1", "0.1.4-rc2") is False
    # Identical prerelease, or identical final: a true tie, never newer.
    assert _version.is_newer("0.1.4-rc1", "0.1.4-rc1") is False
    assert _version.is_newer("0.1.4", "0.1.4") is False
    # A prerelease of a LATER numeric version still wins on the numeric compare
    # alone (the tie-break only applies when the release tuples are equal).
    assert _version.is_newer("0.1.5-rc1", "0.1.4") is True
    assert _version.is_newer("0.1.4", "0.1.5-rc1") is False


def test_is_newer_prerelease_never_more_permissive_than_before():
    """Anti-rollback load-bearing property (updater._refuse_downgrade calls
    is_newer directly): the prerelease tie-break must only ADD resolution to
    cases that previously tied at False, never flip an already-correct numeric
    verdict. A malformed/adversarial version must still never be treated as
    newer than a well-formed one it is not actually ahead of."""
    assert _version.is_newer("0.1.3", "0.1.4") is False          # plain older candidate
    assert _version.is_newer("0.1.3-rc1", "0.1.4") is False      # older AND a prerelease
    assert _version.is_newer("not-a-version", "0.1.3") is False  # malformed candidate
    assert _version.is_newer("0.1.4", "not-a-version") is True   # malformed current (unchanged: fails open toward offering, not toward accepting a downgrade)
