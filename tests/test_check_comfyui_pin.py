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
"could not check" and never as "up to date", and (f) the default-mode script
always exits 0 regardless of outcome.

A second block covers ``--gate`` mode (added alongside pin-currency.yml):
0/1/2 exit codes tracking CURRENT/STALE/UNKNOWN, the age-vs-tolerance boundary,
and that a missing release date - real and reproduced here, not synthetic; the
live bundled pin genuinely has no GitHub Release object for its exact tag -
reports UNKNOWN rather than guessing.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import urllib.error
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_comfyui_pin.py"
_spec = importlib.util.spec_from_file_location("check_comfyui_pin", _PATH)
pincheck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pincheck)


def _release(tag, *, prerelease=False, draft=False, published_at=None):
    return {"tag_name": tag, "prerelease": prerelease, "draft": draft,
            "published_at": published_at}


def _day(n: int) -> dt.datetime:
    return dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=n)


@pytest.fixture(autouse=True)
def _no_ambient_actions_env(monkeypatch):
    """Every test runs with the GitHub Actions annotation env vars cleared, so a
    real GITHUB_ACTIONS=true set by the CI runner that is running THIS pytest
    process cannot leak into a --gate test that does not expect annotations."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)


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


# --------------------------------------------------------------------------- #
#  assess() - pure, age-aware comparison for --gate                          #
# --------------------------------------------------------------------------- #

def test_assess_current_pin_needs_no_date_at_all():
    releases = [_release("v0.31.1", published_at="2026-01-01T00:00:00Z")]
    result = pincheck.assess("v0.31.1", releases, pin_date=None, max_age_days=21)
    assert result["status"] == pincheck.CURRENT
    assert result["days_behind"] is None


@pytest.mark.parametrize("days, status", [
    (0, "current"),
    (5, "behind"),
    (21, "behind"),
    (22, "stale"),
])
def test_assess_splits_behind_vs_stale_at_the_tolerance_boundary(days, status):
    """AT the tolerance is still within it - same boundary check_llama_pin.py
    pins for its own assess(). The end-to-end exit-code mapping for BEHIND/STALE
    is covered separately by the test_gate_* functions below."""
    if status == "current":
        releases = [_release("v0.31.1", published_at="2026-01-01T00:00:00Z")]
    else:
        releases = [_release("v0.32.0", published_at=_day(days).strftime("%Y-%m-%dT%H:%M:%SZ")),
                    _release("v0.31.1", published_at="2026-01-01T00:00:00Z")]
    result = pincheck.assess("v0.31.1", releases, _day(0), 21)
    assert result["status"] == status


def test_assess_reports_unknown_when_a_date_is_missing_even_though_something_is_newer():
    """FIRES: paired with the case above where both dates are present. Something
    IS newer (v0.32.0 > v0.31.1), so _compare()'s own status is "stale", but
    with no pin_date the age cannot be computed - assess() must not silently
    treat that as either BEHIND or STALE."""
    releases = [_release("v0.32.0", published_at="2026-02-01T00:00:00Z"),
                _release("v0.31.1", published_at=None)]
    result = pincheck.assess("v0.31.1", releases, pin_date=None, max_age_days=21)
    assert result["status"] == pincheck.UNKNOWN
    assert result["days_behind"] is None
    assert result["newest_date"] == pincheck._parse_date("2026-02-01T00:00:00Z")

    # the mirror case: pin_date known, but the newest release's own date missing
    releases = [_release("v0.32.0", published_at=None),
                _release("v0.31.1", published_at="2026-01-01T00:00:00Z")]
    result = pincheck.assess("v0.31.1", releases, pin_date=_day(0), max_age_days=21)
    assert result["status"] == pincheck.UNKNOWN
    assert result["days_behind"] is None


@pytest.mark.parametrize("pinned, releases", [
    ("not-a-version", [_release("v0.31.1")]),
    ("v0.31.1", []),
])
def test_assess_passes_unparseable_and_no_data_through_unchanged(pinned, releases):
    """Neither status has an age to compute, so assess() must not invent one -
    it returns exactly what _compare() would, plus the (unused) age fields."""
    result = pincheck.assess(pinned, releases, pin_date=_day(0), max_age_days=21)
    assert result["status"] in ("unparseable_pin", "no_data")
    assert result["days_behind"] is None
    assert result == dict(pincheck._compare(pinned, releases),
                          days_behind=None, pin_date=_day(0), newest_date=None)


def test_eligible_releases_carries_the_date_through_the_existing_filters():
    """The date-carrying refactor must not change WHICH releases are eligible -
    draft/prerelease/unparseable are still excluded, exactly as _compare()'s own
    tests already prove for the undated case."""
    releases = [
        _release("v0.31.0", prerelease=True, published_at="2026-01-05T00:00:00Z"),
        _release("v0.30.5", draft=True, published_at="2026-01-04T00:00:00Z"),
        _release("v0.30.1", published_at="2026-01-03T00:00:00Z"),
        _release("nightly", published_at="2026-01-02T00:00:00Z"),
    ]
    eligible = pincheck._eligible_releases(releases)
    assert [tag for _, tag, _ in eligible] == ["v0.30.1"]
    assert eligible[0][2] == _day(2)


# --------------------------------------------------------------------------- #
#  release_date() / _pin_published_at() - resolving the pin's own date        #
# --------------------------------------------------------------------------- #

def test_pin_published_at_finds_it_in_the_raw_unfiltered_page():
    """Searches the RAW page, not the eligible-only one: a pin that currently
    reads as prerelease/draft (unlikely, but not this function's problem) must
    still have its own date found."""
    releases = [_release("v0.31.1", prerelease=True, published_at="2026-03-01T00:00:00Z")]
    assert pincheck._pin_published_at(releases, "v0.31.1") == _day(59)
    assert pincheck._pin_published_at(releases, "v0.9.9") is None


def test_release_date_uses_the_injectable_opener_and_never_raises():
    calls = []

    def opener(repo, tag):
        calls.append((repo, tag))
        return {"published_at": "2026-04-01T00:00:00Z"}
    assert pincheck.release_date("v0.31.1", "owner/repo", opener=opener) == _day(90)
    assert calls == [("owner/repo", "v0.31.1")]

    def raiser(repo, tag):
        raise urllib.error.HTTPError("url", 404, "Not Found", {}, None)
    assert pincheck.release_date("v0.31.1", opener=raiser) is None

    def malformed(repo, tag):
        return ["not", "a", "dict"]
    assert pincheck.release_date("v0.31.1", opener=malformed) is None


def test_release_date_by_tag_http_hits_the_real_endpoint_shape(monkeypatch):
    """Only the true leaf (urlopen) is patched, so the URL-building and
    header-setting in _fetch_release_by_tag_http runs for real."""
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["accept"] = req.get_header("Accept")
        return _FakeHTTP({"published_at": "2026-05-01T00:00:00Z"})
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = pincheck._fetch_release_by_tag_http("owner/repo", "v1.2.3")
    assert result == {"published_at": "2026-05-01T00:00:00Z"}
    assert captured["url"] == "https://api.github.com/repos/owner/repo/releases/tags/v1.2.3"
    assert captured["accept"] == "application/vnd.github+json"


# --------------------------------------------------------------------------- #
#  --gate: exit codes, annotations, and the real bundled pin                  #
# --------------------------------------------------------------------------- #

def test_gate_default_max_age_matches_llama_pins_own_default():
    """Both scripts are the same MAINTENANCE SIGNAL shape; a silent divergence
    here would mean one pin tolerates staleness the other does not, for no
    stated reason."""
    llama_path = _PATH.parent / "check_llama_pin.py"
    spec = importlib.util.spec_from_file_location("check_llama_pin_ref", llama_path)
    llama = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llama)
    assert pincheck.DEFAULT_MAX_AGE_DAYS == llama.DEFAULT_MAX_AGE_DAYS == 21


def test_gate_current_exits_zero_and_never_prints_behind(monkeypatch, capsys):
    fixture = [_release("v0.31.1", published_at="2026-01-01T00:00:00Z")]
    monkeypatch.setattr(pincheck, "_fetch_releases_http", lambda repo: fixture)
    rc = pincheck.main(["--pinned", "v0.31.1", "--gate"])
    out = capsys.readouterr().out
    assert rc == pincheck.EXIT_CURRENT == 0
    assert "OK: the pin is current" in out
    assert "BEHIND" not in out


def test_gate_stale_exits_one_and_names_the_remedy(monkeypatch, capsys):
    fixture = [_release("v0.31.1", published_at="2026-02-01T00:00:00Z"),
               _release("v0.9.2", published_at="2026-01-01T00:00:00Z")]
    monkeypatch.setattr(pincheck, "_fetch_releases_http", lambda repo: fixture)
    rc = pincheck.main(["--pinned", "v0.9.2", "--gate", "--max-age-days", "10"])
    out = capsys.readouterr().out
    assert rc == pincheck.EXIT_STALE == 1
    assert "STALE" in out
    assert "31 day(s)" in out
    assert "COMFYUI_PINNED_VERSION" in out


def test_gate_behind_within_tolerance_exits_current(monkeypatch, capsys):
    fixture = [_release("v0.31.1", published_at="2026-01-06T00:00:00Z"),
               _release("v0.9.2", published_at="2026-01-01T00:00:00Z")]
    monkeypatch.setattr(pincheck, "_fetch_releases_http", lambda repo: fixture)
    rc = pincheck.main(["--pinned", "v0.9.2", "--gate", "--max-age-days", "21"])
    out = capsys.readouterr().out
    assert rc == pincheck.EXIT_CURRENT == 0
    assert "within the 21-day tolerance" in out


def test_gate_reports_unknown_never_current_on_unreachable_api(monkeypatch, capsys):
    def _raiser(repo):
        raise urllib.error.URLError("simulated: rate limited")
    monkeypatch.setattr(pincheck, "_fetch_releases_http", _raiser)
    rc = pincheck.main(["--pinned", "v0.9.2", "--gate"])
    out = capsys.readouterr().out
    assert rc == pincheck.EXIT_UNKNOWN == 2
    assert "COULD NOT CHECK" in out
    assert "NOT 'the pin is up to date'" in out
    assert "OK:" not in out and "STALE" not in out


def test_gate_falls_back_to_a_per_tag_lookup_when_the_pin_is_off_the_page(monkeypatch):
    """The realistic, LIVE-REPRODUCED case: the fetched page does not carry the
    pinned tag's own entry at all (upstream never published a GitHub Release
    for that exact tag - measured true for this repo's own v0.31.1 pin), so
    the per-tag fallback is what supplies pin_date."""
    fixture = [_release("v0.32.0", published_at="2026-02-01T00:00:00Z")]
    calls = []

    def by_tag(repo, tag):
        calls.append((repo, tag))
        return {"published_at": "2026-01-01T00:00:00Z"}

    monkeypatch.setattr(pincheck, "_fetch_releases_http", lambda repo: fixture)
    monkeypatch.setattr(pincheck, "_fetch_release_by_tag_http", by_tag)
    rc = pincheck.main(["--pinned", "v0.31.1", "--gate", "--max-age-days", "10"])

    assert calls == [(pincheck._REPO, "v0.31.1")]
    assert rc == pincheck.EXIT_STALE == 1


def test_gate_reports_unknown_when_even_the_per_tag_lookup_fails(monkeypatch, capsys):
    """The pin is behind by count, but NEITHER its own date NOR a fallback
    lookup can be resolved - UNKNOWN, never a guessed STALE or CURRENT."""
    fixture = [_release("v0.32.0", published_at="2026-02-01T00:00:00Z")]

    def failing_by_tag(repo, tag):
        raise urllib.error.HTTPError("url", 404, "Not Found", {}, None)

    monkeypatch.setattr(pincheck, "_fetch_releases_http", lambda repo: fixture)
    monkeypatch.setattr(pincheck, "_fetch_release_by_tag_http", failing_by_tag)
    rc = pincheck.main(["--pinned", "v0.31.1", "--gate"])
    out = capsys.readouterr().out

    assert rc == pincheck.EXIT_UNKNOWN == 2
    assert "COULD NOT CHECK" in out
    assert "date is missing" in out


def test_gate_annotates_and_summarises_only_under_github_actions(monkeypatch, capsys, tmp_path):
    fixture = [_release("v0.31.1", published_at="2026-02-01T00:00:00Z"),
               _release("v0.9.2", published_at="2026-01-01T00:00:00Z")]
    monkeypatch.setattr(pincheck, "_fetch_releases_http", lambda repo: fixture)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    rc = pincheck.main(["--pinned", "v0.9.2", "--gate", "--max-age-days", "5"])
    assert rc == pincheck.EXIT_STALE
    assert "::error::" not in capsys.readouterr().out

    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    rc = pincheck.main(["--pinned", "v0.9.2", "--gate", "--max-age-days", "5"])
    out = capsys.readouterr().out
    assert rc == pincheck.EXIT_STALE
    assert "::error::" in out and "STALE" in out
    text = summary.read_text(encoding="utf-8")
    assert "STALE" in text and "31 day(s)" in text and "v0.31.1" in text


def test_default_mode_is_byte_for_byte_unaffected_by_gate_machinery(monkeypatch, capsys):
    """The existing default-mode contract (always exit 0, no --gate vocabulary)
    must survive the --gate addition untouched."""
    fixture = [_release("v0.31.1"), _release("v0.30.0")]
    monkeypatch.setattr(pincheck, "_fetch_releases_http", lambda repo: fixture)
    rc = pincheck.main(["--pinned", "v0.9.2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "EXIT_STALE" not in out
    assert "tolerance" not in out
    assert "::" not in out


class _FakeHTTP:
    """A urlopen() context manager standing in for a real HTTP response, for
    tests that patch urllib.request.urlopen directly rather than using the
    opener= seam."""

    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
