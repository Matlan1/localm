# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline tests for scripts/check_comfyui_pin.py.

No real network calls: the GitHub releases API is either driven through the
injectable ``opener`` seam on ``_fetch_releases`` (unit tests of the fetch
wrapper itself) or, for the two end-to-end tests, by monkeypatching only the
true leaf dependency ``_fetch_releases_http`` - the one function that actually
opens a socket - so ``main()`` and the real ``_fetch_releases`` try/except
logic still run for real.

Proves: (a) a stale pin is reported with the correct behind-count and the
actual latest tag, (b) a current pin is reported as current, (c) version
comparison is numeric, not lexical (a plain string sort would get a
v0.9.2-vs-v0.31.1 comparison backwards), (d) draft/prerelease releases never
count as "latest" or toward "behind", (e) an unreachable API is reported as
"could not check" and never as "up to date", and (f) the script always exits 0
regardless of outcome.
"""

from __future__ import annotations

import importlib.util
import urllib.error
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_comfyui_pin.py"
_spec = importlib.util.spec_from_file_location("check_comfyui_pin", _PATH)
pincheck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pincheck)


def _release(tag, *, prerelease=False, draft=False):
    return {"tag_name": tag, "prerelease": prerelease, "draft": draft}


# --------------------------------------------------------------------------- #
#  Bound to the real shipped file                                             #
# --------------------------------------------------------------------------- #

def test_reads_the_real_pinned_version_and_it_parses():
    pinned = pincheck._pinned_version()
    assert isinstance(pinned, str) and pinned
    assert pincheck._parse_version(pinned) is not None, (
        f"COMFYUI_PINNED_VERSION={pinned!r} in managed_comfy_fresh.py no longer "
        "parses as a plain vX.Y[.Z] tag"
    )


# --------------------------------------------------------------------------- #
#  _parse_version                                                             #
# --------------------------------------------------------------------------- #

def test_parse_version_numeric_tuple():
    assert pincheck._parse_version("v0.31.1") == (0, 31, 1)
    assert pincheck._parse_version("0.9.2") == (0, 9, 2)
    assert pincheck._parse_version("v1") == (1,)


def test_parse_version_rejects_suffixed_and_non_version_tags():
    assert pincheck._parse_version("v0.31.1-rc1") is None
    assert pincheck._parse_version("nightly") is None
    assert pincheck._parse_version(None) is None
    assert pincheck._parse_version("") is None


def test_version_ordering_is_numeric_not_lexical():
    """'v0.9.2' > 'v0.31.1' as strings ('9' > '3'), which would wrongly report
    a 22-releases-stale pin as current. This is the exact bug the pinned
    constant sat undetected behind for five weeks."""
    assert "v0.9.2" > "v0.31.1"  # lexical comparison
    assert pincheck._parse_version("v0.9.2") < pincheck._parse_version("v0.31.1")

    result = pincheck._compare("v0.9.2", [_release("v0.31.1")])
    assert result["status"] == "stale"
    assert result["behind"] == 1
    assert result["latest"] == "v0.31.1"


# --------------------------------------------------------------------------- #
#  _compare (pure - no I/O)                                                   #
# --------------------------------------------------------------------------- #

def test_compare_current_when_no_release_is_newer():
    result = pincheck._compare("v0.31.1", [_release("v0.31.1"), _release("v0.30.0")])
    assert result == {"status": "current", "latest": "v0.31.1"}


def test_compare_reports_stale_with_exact_behind_count():
    releases = [
        _release("v0.31.1"),
        _release("v0.31.0"),
        _release("v0.30.2"),
        _release("v0.8.5"),   # older than the pin - must not be counted
        _release("v0.9.2"),   # equal to the pin - must not be counted
    ]
    result = pincheck._compare("v0.9.2", releases)
    assert result["status"] == "stale"
    assert result["latest"] == "v0.31.1"
    assert result["behind"] == 3
    assert result["capped"] is False


def test_minor_version_is_an_integer_not_a_decimal_fraction():
    """0.10.0 reads, misleadingly, like it could be 'point-one-zero' < 'point-
    nine' if compared as decimal fractions - it is not: minor version 10 comes
    after minor version 9, same as 1.10 > 1.9 in ordinary semver. A comparator
    that parsed each dotted component as one float (0.9 vs 0.10) would get
    this backwards; tuple-of-ints must not."""
    assert pincheck._parse_version("v0.10.0") > pincheck._parse_version("v0.9.2")
    result = pincheck._compare("v0.9.2", [_release("v0.10.0")])
    assert result == {"status": "stale", "latest": "v0.10.0", "behind": 1, "capped": False}


def test_compare_excludes_draft_and_prerelease_from_latest_and_count():
    releases = [
        _release("v0.31.0", prerelease=True),   # newest tag, but a prerelease
        _release("v0.30.5", draft=True),        # newer than pin, but a draft
        _release("v0.30.1"),                    # the only eligible newer release
        _release("v0.29.0"),                    # older than pin
    ]
    result = pincheck._compare("v0.30.0", releases)
    assert result["status"] == "stale"
    assert result["latest"] == "v0.30.1"
    assert result["behind"] == 1


def test_compare_no_eligible_releases_is_no_data_not_current():
    """An API response with nothing usable in it must never be read as 'the
    pin is current' - that would be a false positive, the exact failure mode
    constraint 2 in the brief forbids."""
    assert pincheck._compare("v0.31.1", [])["status"] == "no_data"
    assert pincheck._compare(
        "v0.31.1", [_release("nightly"), {"tag_name": None}]
    )["status"] == "no_data"


def test_compare_unparseable_pin():
    result = pincheck._compare("not-a-version", [_release("v0.31.1")])
    assert result == {"status": "unparseable_pin"}


def test_compare_stale_marks_capped_when_page_is_full():
    releases = [_release(f"v0.{50 + i}.0") for i in range(pincheck._PER_PAGE)]
    result = pincheck._compare("v0.1.0", releases)
    assert result["status"] == "stale"
    assert result["capped"] is True


# --------------------------------------------------------------------------- #
#  _fetch_releases (the injectable seam - no real network, no urllib patch)   #
# --------------------------------------------------------------------------- #

def test_fetch_releases_returns_data_from_a_working_opener():
    fixture = [_release("v0.31.1")]
    result = pincheck._fetch_releases("owner/repo", opener=lambda repo: fixture)
    assert result is fixture


def test_fetch_releases_returns_none_never_raises_on_network_error():
    def _raiser(repo):
        raise urllib.error.URLError("simulated network failure")

    result = pincheck._fetch_releases("owner/repo", opener=_raiser)
    assert result is None


def test_fetch_releases_returns_none_on_malformed_response_shape():
    result = pincheck._fetch_releases("owner/repo", opener=lambda repo: {"not": "a list"})
    assert result is None


# --------------------------------------------------------------------------- #
#  main() end-to-end - only the true leaf (_fetch_releases_http) is patched.  #
# --------------------------------------------------------------------------- #

def test_main_reports_could_not_check_on_unreachable_api_never_current(monkeypatch, capsys):
    def _raiser(repo):
        raise urllib.error.URLError("simulated: rate limited")

    monkeypatch.setattr(pincheck, "_fetch_releases_http", _raiser)
    rc = pincheck.main(["--pinned", "v0.9.2"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "could not check" in out.lower()
    assert "up to date" not in out.lower()
    assert "behind" not in out.lower()


def test_main_reports_stale_with_remedy_and_exits_zero(monkeypatch, capsys):
    fixture = [_release("v0.31.1"), _release("v0.30.0")]
    monkeypatch.setattr(pincheck, "_fetch_releases_http", lambda repo: fixture)
    rc = pincheck.main(["--pinned", "v0.9.2"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "2 release(s) behind" in out
    assert "v0.31.1" in out
    assert "COMFYUI_PINNED_VERSION" in out  # names the remedy site
    assert "could not check" not in out.lower()


def test_main_reports_current_and_exits_zero(monkeypatch, capsys):
    fixture = [_release("v0.31.1")]
    monkeypatch.setattr(pincheck, "_fetch_releases_http", lambda repo: fixture)
    rc = pincheck.main(["--pinned", "v0.31.1"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "up to date" in out.lower()
    assert "behind" not in out.lower()
