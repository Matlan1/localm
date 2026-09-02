# SPDX-License-Identifier: AGPL-3.0-or-later
"""The live-read version signal (localm/_version) the self-updater compares against
the latest release tag. Editable installs do not refresh dist-info on a code swap,
so the running version MUST come from the VERSION file at runtime, not metadata."""

import sys
import types

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


# --------------------- read_version() fallback order (3-way) --------------------
#
# VERSION file (live) > installed metadata > the release build's baked
# localm/_build_version.py constant > "unknown". These control all three sources
# independently so the ORDER, not just each source's own presence, is pinned.

def _block_file_and_metadata(monkeypatch, tmp_path):
    """Shared setup: the VERSION file is unreadable and importlib.metadata.version
    raises, so read_version() can only resolve via the baked constant or
    "unknown"."""
    monkeypatch.setattr(_version, "version_file", lambda: tmp_path / "nope")

    def _raise(name):
        raise ModuleNotFoundError(f"No package metadata was found for {name}")
    monkeypatch.setattr("importlib.metadata.version", _raise)


def test_read_version_falls_back_to_baked_constant(monkeypatch, tmp_path):
    """Third fallback: a release build's baked localm/_build_version.py constant,
    reached only when neither the VERSION file nor installed metadata resolves."""
    _block_file_and_metadata(monkeypatch, tmp_path)
    fake = types.ModuleType("localm._build_version")
    fake.VERSION = "7.7.7-baked"
    monkeypatch.setitem(sys.modules, "localm._build_version", fake)
    assert _version.read_version() == "7.7.7-baked"


def test_read_version_prefers_metadata_over_baked_constant(monkeypatch, tmp_path):
    """Installed metadata still wins over the baked constant when both are
    available - the baked constant is the LAST fallback, not an alternate."""
    monkeypatch.setattr(_version, "version_file", lambda: tmp_path / "nope")
    monkeypatch.setattr("importlib.metadata.version", lambda name: "5.5.5-metadata")
    fake = types.ModuleType("localm._build_version")
    fake.VERSION = "5.5.5-baked-should-not-be-read"
    monkeypatch.setitem(sys.modules, "localm._build_version", fake)
    assert _version.read_version() == "5.5.5-metadata"


def test_read_version_unknown_when_all_three_sources_are_absent(monkeypatch, tmp_path):
    """The pre-existing terminal case is unchanged: with no VERSION file, no
    metadata, and no localm._build_version module at all (an ordinary source
    checkout, never built into a signed release), read_version() still returns
    "unknown" rather than raising - the baked-constant import must be guarded."""
    _block_file_and_metadata(monkeypatch, tmp_path)
    monkeypatch.delitem(sys.modules, "localm._build_version", raising=False)
    assert _version.read_version() == "unknown"


def test_read_version_baked_constant_empty_string_is_treated_as_absent(monkeypatch, tmp_path):
    """A present-but-empty baked constant (a malformed or half-written generated
    file) must not be reported as a real version."""
    _block_file_and_metadata(monkeypatch, tmp_path)
    fake = types.ModuleType("localm._build_version")
    fake.VERSION = ""
    monkeypatch.setitem(sys.modules, "localm._build_version", fake)
    assert _version.read_version() == "unknown"


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
    """The full truth table, driving is_newer() directly rather than asserting on
    the parser's internals, so a future refactor of _parse/_prerelease_suffix is
    free as long as this table still holds."""
    # A stable release always outranks a prerelease of the SAME numeric version:
    # upgrading FROM an rc TO the matching final release.
    assert _version.is_newer("0.1.4", "0.1.4-rc1") is True
    # ...and never the other direction (never "downgrade" final -> rc automatically).
    assert _version.is_newer("0.1.4-rc1", "0.1.4") is False
    # Between two prereleases of the same numeric version, the higher ordinal
    # wins.
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
    cases the numeric compare alone leaves tied at False, never flip an
    already-correct numeric verdict. A malformed or adversarial version must
    never be treated as newer than a well-formed one it is not ahead of."""
    assert _version.is_newer("0.1.3", "0.1.4") is False          # plain older candidate
    assert _version.is_newer("0.1.3-rc1", "0.1.4") is False      # older AND a prerelease
    assert _version.is_newer("not-a-version", "0.1.3") is False  # malformed candidate
    assert _version.is_newer("0.1.4", "not-a-version") is True   # malformed current (unchanged: fails open toward offering, not toward accepting a downgrade)


# ------------------------------ comparable() -------------------------------
#
# is_newer() returns False both for a genuine tie/older release AND for a tag
# it could not parse as a version at all - "stable"/"nightly"/"release-5" all
# degrade to the same (0,) tuple _parse() gives a real "0.0.0"-shaped version,
# so the caller cannot tell "not newer" from "could not tell" from the bool
# alone. comparable() is the second signal a caller reads ALONGSIDE is_newer().

def test_comparable_true_for_two_real_versions():
    assert _version.comparable("0.2.0", "0.1.0") is True
    assert _version.comparable("v0.2.0", "0.1.9") is True


def test_comparable_true_for_a_genuine_tie_or_older():
    # A real, ordered comparison - "not newer" here really does mean not newer,
    # never "could not tell".
    assert _version.comparable("0.1.0", "0.1.0") is True
    assert _version.comparable("0.1.0", "0.2.0") is True


def test_comparable_false_for_non_numeric_leading_candidate():
    # Each of these ties at False against a real version, and none of them is
    # actually "not newer" - they are unrecognized.
    for tag in ("stable", "nightly", "release-5"):
        assert _version.is_newer(tag, "0.1.4") is False
        assert _version.comparable(tag, "0.1.4") is False, tag


def test_comparable_false_for_non_numeric_leading_current():
    assert _version.comparable("0.1.5", "nightly") is False


def test_comparable_true_when_current_has_no_signal():
    # Matches is_newer's own special case: a fresh install with no local
    # version signal still sees any real candidate as comparable (an update).
    assert _version.comparable("0.2.0", "unknown") is True
    assert _version.comparable("0.2.0", "") is True


def test_comparable_false_for_empty_candidate():
    assert _version.comparable("", "0.1.0") is False


# ---------- pinned to the ACTUAL shipped tag shape (unhyphenated) ----------
#
# Every truth-table test above uses the hyphenated "0.1.4-rc1" form. This
# project's real tags and VERSION file use the UNHYPHENATED form instead. The
# code handles both by construction (_prerelease_suffix's `.lstrip("-_")`);
# these pin the shape that actually ships.

def test_is_newer_pinned_to_shipped_tag_shape():
    assert _version.is_newer("v0.1.5rc2", "v0.1.5rc1") is True
    assert _version.is_newer("v0.1.5rc1", "v0.1.5rc2") is False
    assert _version.is_newer("v0.1.5", "v0.1.5rc2") is True     # final outranks any rc
    assert _version.is_newer("v0.1.5rc1", "v0.1.5") is False
    assert _version.is_newer("v0.1.5rc1", "v0.1.5rc1") is False  # exact tie
    assert _version.normalize("v0.1.5rc2") == "0.1.5rc2"
    assert _version.is_newer("v0.1.5rc3", "v0.1.5rc2") is True
    assert _version.is_newer("v0.1.5rc2", "v0.1.5rc3") is False


def test_shipping_version_is_offered_as_an_update_to_every_earlier_tag():
    """The version in VERSION must compare NEWER than every published tag before
    it, or `localm update` silently never offers this release to the users
    already on one of them. `_refuse_downgrade` is gated on the same call, so a
    break here is quiet in both directions.

    Reads VERSION rather than hard-coding a pair: a hard-coded one stops
    describing the release the moment someone bumps VERSION without touching
    this file, which is exactly when the check is load-bearing."""
    shipping = _version.read_version()
    for earlier in ("0.1.0", "0.1.1", "0.1.2", "0.1.3", "0.1.4",
                    "0.1.5rc1", "0.1.5rc2"):
        if _version.normalize(shipping) == earlier:
            continue    # re-cutting an existing tag: nothing to order against
        assert _version.is_newer(shipping, earlier) is True, (
            f"VERSION {shipping!r} is not newer than published {earlier!r}, so "
            "the updater would not offer it")
        assert _version.is_newer(earlier, shipping) is False, (
            f"published {earlier!r} compares newer than VERSION {shipping!r}")
        assert _version.comparable(shipping, earlier) is True


def test_prerelease_suffix_shipped_tag_shape():
    assert _version._prerelease_suffix("0.1.5rc1") == ("rc", 1)
    assert _version._prerelease_suffix("0.1.5rc2") == ("rc", 2)
    assert _version._prerelease_suffix("0.1.5rc3") == ("rc", 3)
    assert _version._prerelease_suffix("0.1.5") is None


# --------------------------------------------------------------------------- #
#  Every hardcoded version literal must agree with the VERSION file            #
#                                                                              #
#  The release version lives in SEVEN places: the VERSION file, pyproject,     #
#  uv.lock, and four hardcoded literals in the product itself. Bumping a        #
#  release meant remembering all of them, and NOTHING checked that they agreed  #
#  - so a missed site shipped silently, reporting a stale version in the API's  #
#  OpenAPI document, to MCP clients, and from `localm --version` whenever the   #
#  VERSION file was unreadable.                                                 #
#                                                                              #
#  Caught for real while cutting 0.1.5: the VERSION file and pyproject were     #
#  bumped and `localm.__version__` still said 0.1.5rc3. These assert on the     #
#  SOURCE literals rather than only on imported values, because a literal in    #
#  an `except` fallback or a dict is not reachable at import time - and those   #
#  are exactly the ones that rot unnoticed.                                     #
# --------------------------------------------------------------------------- #

import re                       # noqa: E402
from pathlib import Path        # noqa: E402

import pytest                   # noqa: E402

_REPO = Path(__file__).resolve().parent.parent


def _declared_version() -> str:
    return (_REPO / "VERSION").read_text(encoding="utf-8").strip()


# (path, regex capturing the version literal in group 1, what it feeds)
_VERSION_SITES = [
    ("localm/__init__.py", r'__version__ = "([^"]+)"',
     "localm.__version__"),
    ("localm/cli/_core.py", r'return "(\d[^"]*)"\s*$',
     "the `localm --version` fallback when the VERSION file is unreadable"),
    ("localm/inference/http_server.py", r'\n    app = FastAPI\((?:.|\n)*?version="([^"]+)"',
     "the FastAPI app version (published in the OpenAPI document)"),
    ("localm/plugins/coder/mcp.py", r'"clientInfo": \{"name": "localcoder", "version": "([^"]+)"\}',
     "the clientInfo localm sends to an MCP server"),
    ("localm/plugins/mcpserver/server.py", r'SERVER_VERSION = "([^"]+)"',
     "the version localm's own MCP server reports to clients"),
]


@pytest.mark.parametrize("rel,pattern,what", _VERSION_SITES,
                         ids=[s[0] for s in _VERSION_SITES])
def test_hardcoded_version_matches_version_file(rel, pattern, what):
    src = (_REPO / rel).read_text(encoding="utf-8")
    m = re.search(pattern, src, re.M)
    assert m, f"{rel}: no version literal matched - the site moved, update _VERSION_SITES"
    assert m.group(1) == _declared_version(), (
        f"{rel} declares {m.group(1)!r} but VERSION says {_declared_version()!r}. "
        f"This literal feeds {what}; bump it with the release.")


def test_pyproject_and_lock_match_version_file():
    want = _declared_version()
    pyproject = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version = "([^"]+)"', pyproject, re.M)
    assert m and m.group(1) == want, f"pyproject.toml version != VERSION ({want})"
    # uv.lock pins the project's OWN version; a stale one makes the lockfile
    # drift check fail in CI rather than here, which is a slower way to learn it.
    lock = (_REPO / "uv.lock").read_text(encoding="utf-8")
    m = re.search(r'\nname = "localm"\nversion = "([^"]+)"', lock)
    assert m and m.group(1) == want, f"uv.lock localm version != VERSION ({want})"
