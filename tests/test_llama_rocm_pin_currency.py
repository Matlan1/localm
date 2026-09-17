# SPDX-License-Identifier: AGPL-3.0-or-later
"""The pin CONSTANT's own safety, and the currency check that stops it rotting.

Companion to test_llama_pin_constant_and_currency.py, which covers the
separate ggml-org/llama.cpp pin (_PINNED_TAG). This file covers _ROCM_TAG's own
tag safety and its currency check, scripts/check_llama_rocm_pin.py - a
different tag series sourced from lemonade-sdk/llamacpp-rocm.

A second block covers ``--gate`` mode (added alongside pin-currency.yml): 0/1/2
exit codes tracking CURRENT/STALE/UNKNOWN and the age-vs-tolerance boundary,
mirroring test_llama_pin_constant_and_currency.py's own gate coverage since
assess() here is the identical shape.

Nothing here asserts the pin's VALUE.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import pytest

from localm import setup_llama as sl

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_llama_rocm_pin.py"


@pytest.fixture(scope="module")
def currency():
    """scripts/check_llama_rocm_pin.py, loaded by path. Not a package module: it
    is stdlib-only so the CI job can run it with nothing installed."""
    spec = importlib.util.spec_from_file_location("check_llama_rocm_pin", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _no_ambient_actions_env(monkeypatch):
    """Every test runs with the GitHub Actions annotation env vars cleared, so a
    real GITHUB_ACTIONS=true set by the CI runner that is running THIS pytest
    process cannot leak into a --gate test that does not expect annotations."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)


def _day(n: int) -> dt.datetime:
    return dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=n)


def _releases(*pairs):
    """[(tag, day-offset or None), ...] newest-first -> upstream_releases() shape."""
    return [{"tag": t, "published_at": (_day(d) if d is not None else None)}
            for t, d in pairs]


# --------------------------------------------------------------------------- #
#  The pin constant                                                            #
# --------------------------------------------------------------------------- #

def test_the_pin_is_a_tag_that_can_safely_reach_a_url():
    """_ROCM_TAG is interpolated into a lemonade-sdk release URL the same way
    _PINNED_TAG is interpolated into a ggml-org one (_release_assets)."""
    assert sl.is_safe_tag(sl._ROCM_TAG)


# --------------------------------------------------------------------------- #
#  The currency check                                                          #
# --------------------------------------------------------------------------- #

def test_currency_reads_the_same_pin_the_code_uses(currency):
    """It reads the constant BY TEXT so the CI job needs no install. The cost of
    that is a second source of truth, so the two are compared here - otherwise a
    rename would leave the check silently reporting an old value."""
    assert currency.pinned_tag() == sl._ROCM_TAG


def test_currency_compares_build_numbers_numerically_not_lexically(currency):
    assert currency._build_number("b1288") < currency._build_number("b10375")
    assert sorted(["b1288", "b10375", "b1307"],
                  key=currency._build_number) == ["b1288", "b1307", "b10375"]


def test_currency_refuses_to_report_currency_it_did_not_verify(currency, monkeypatch, capsys):
    """A blocked lookup and an up-to-date pin must never print the same thing.
    This check exists to make a stale pin visible; one that says "OK" when it
    could not reach the API would hide exactly what it was built to surface."""
    monkeypatch.setattr(currency.urllib.request, "urlopen", _boom)
    assert currency.main([]) == 0, "a maintenance signal never fails the build"
    out = capsys.readouterr().out
    assert "COULD NOT CHECK" in out
    assert "NOT 'the pin is up to date'" in out
    assert "OK:" not in out


def test_currency_reports_a_gap_and_says_how_to_close_it(currency, monkeypatch, capsys):
    monkeypatch.setattr(currency, "upstream_tags",
                        lambda: (["b99999", "b99998", currency.pinned_tag()], ""))
    assert currency.main([]) == 0
    out = capsys.readouterr().out
    assert "BEHIND by 2 release(s)" in out
    assert "_ROCM_TAG" in out, (
        "advancing this pin has no automated confirm step, so the remedy must "
        "name the constant a maintainer edits by hand")


def test_currency_does_not_call_a_lexically_larger_older_tag_newer(
        currency, monkeypatch, capsys):
    """THE COMPARISON ITSELF, on a value that DISCRIMINATES.

    The test above cannot catch a lexical comparison: b99999 and b99998 are both
    lexically AND numerically greater than the pin, so a lexical and a numeric
    implementation agree on that fixture.

    A tag needs FEWER DIGITS to discriminate: the real pin has 4 digits
    (b1307), so a 3-digit decoy sorts AFTER it as a string ('9' > '1') while
    being far older as a number. A lexical comparison reports the pin as
    behind; the correct one reports it current."""
    older = "b999"
    assert older > currency.pinned_tag(), (
        "this fixture only discriminates while the pin has more digits than the "
        "decoy - if that ever stops holding, pick a smaller decoy")
    monkeypatch.setattr(currency, "upstream_tags", lambda: ([older], ""))

    assert currency.main([]) == 0
    out = capsys.readouterr().out
    assert "OK: the pin is current" in out, out
    assert "BEHIND" not in out


def test_currency_skips_releases_whose_assets_are_not_uploaded_yet(currency, monkeypatch):
    """Upstream publishes a release before its archives finish uploading.
    Counting one of those as "behind" overstates the gap and would point the
    advance step at a tag that cannot be downloaded yet."""
    payload = [
        {"tag_name": "b99999", "draft": False, "prerelease": False, "assets": []},
        {"tag_name": "b99998", "draft": False, "prerelease": False,
         "assets": [{"name": "x"}]},
        {"tag_name": "b99997", "draft": True, "prerelease": False,
         "assets": [{"name": "x"}]},
    ]
    monkeypatch.setattr(currency.urllib.request, "urlopen",
                        lambda *a, **k: _FakeHTTP(payload))
    tags, err = currency.upstream_tags()
    assert err == ""
    assert tags == ["b99998"], "asset-less and draft releases are not candidates"


def test_currency_excludes_a_draft_or_prerelease_release_from_candidates(
        currency, monkeypatch):
    """CONTROL for the test below: unlike ggml-org/llama.cpp (whose prerelease
    flag is set on every release with no signal value, per check_llama_pin.py),
    lemonade-sdk/llamacpp-rocm's draft and prerelease flags are meaningful, so a
    release flagged either one must be excluded here - the opposite filter from
    check_llama_pin.py's own, for its different upstream."""
    payload = [
        {"tag_name": "b99999", "draft": False, "prerelease": True,
         "assets": [{"name": "x"}] * 14},
        {"tag_name": "b99998", "draft": True, "prerelease": False,
         "assets": [{"name": "x"}] * 14},
    ]
    monkeypatch.setattr(currency.urllib.request, "urlopen",
                        lambda *a, **k: _FakeHTTP(payload))
    tags, err = currency.upstream_tags()
    assert tags == [], "a draft or prerelease release must never become a candidate"
    assert err != ""


def test_currency_counts_a_real_lemonade_sdk_shaped_release_as_a_candidate(
        currency, monkeypatch):
    """FIRES: paired with the control above, on the payload shape verified live
    against the lemonade-sdk/llamacpp-rocm releases API - draft=False,
    prerelease=False, 14 uploaded assets, a plain 'bNNNN' tag - which must
    survive the filter the control just proved can exclude something."""
    payload = [
        {"tag_name": "b1321", "draft": False, "prerelease": False,
         "assets": [{"name": "llama-b1321-ubuntu-rocm-gfx103X-x64.zip"}] * 14},
    ]
    monkeypatch.setattr(currency.urllib.request, "urlopen",
                        lambda *a, **k: _FakeHTTP(payload))
    tags, err = currency.upstream_tags()
    assert err == ""
    assert tags == ["b1321"]


# --------------------------------------------------------------------------- #
#  assess() - identical shape to check_llama_pin.py's own                     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("days, status, gate_exit", [
    (0, "current", 0),
    (5, "behind", 0),
    (21, "behind", 0),
    (22, "stale", 1),
    (None, "unknown", 2),
])
def test_gate_exit_codes_follow_the_age_against_the_tolerance(
        currency, monkeypatch, days, status, gate_exit):
    """The three outcomes stay distinct all the way to the exit code, and the
    boundary is exactly the tolerance: AT the tolerance is still within it.
    Without --gate every outcome is exit 0."""
    pin = currency.pinned_tag()
    n = currency._build_number(pin)
    if status == "current":
        releases = _releases((pin, 0))
    else:
        releases = _releases((f"b{n + 1}", days))
    monkeypatch.setattr(currency, "upstream_releases", lambda: (releases, ""))
    monkeypatch.setattr(currency, "release_date", lambda tag: _day(0))
    result = currency.assess(
        pin, releases, _day(0) if status != "current" else None, 21)
    assert result["status"] == status
    assert currency.main(["--gate", "--max-age-days", "21"]) == gate_exit
    assert currency.main(["--max-age-days", "21"]) == 0


def test_age_comes_from_the_two_release_dates_not_the_clock(currency):
    """A quiet upstream does not age the pin: only the distance between the
    pinned release and the newest one counts, and it is a pure function of the
    two dates."""
    pin = "b100"
    releases = _releases(("b200", 100), ("b150", 50))
    assert currency.assess(pin, releases, _day(80), 21)["days_behind"] == 20
    assert currency.assess(pin, releases, _day(80), 21)["status"] == currency.BEHIND
    assert currency.assess(pin, releases, _day(60), 21)["days_behind"] == 40
    assert currency.assess(pin, releases, _day(60), 21)["status"] == currency.STALE
    assert currency.assess(pin, releases, _day(60), 21)["builds_behind"] == 100
    assert currency.assess(pin, releases, None, 21)["status"] == currency.UNKNOWN


# --------------------------------------------------------------------------- #
#  --gate: exit codes, annotations, the pin's own date, and the real bundle    #
# --------------------------------------------------------------------------- #

def test_gate_default_max_age_matches_llama_pins_own_default(currency):
    """Both scripts are the same MAINTENANCE SIGNAL shape; a silent divergence
    here would mean one pin tolerates staleness the other does not, for no
    stated reason."""
    llama_path = _SCRIPT.parent / "check_llama_pin.py"
    spec = importlib.util.spec_from_file_location("check_llama_pin_ref", llama_path)
    llama = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llama)
    assert currency.DEFAULT_MAX_AGE_DAYS == llama.DEFAULT_MAX_AGE_DAYS == 21


def test_gate_current_exits_zero_and_never_prints_behind(currency, monkeypatch, capsys):
    pin = currency.pinned_tag()
    monkeypatch.setattr(currency, "upstream_releases", lambda: (_releases((pin, 0)), ""))
    rc = currency.main(["--gate"])
    out = capsys.readouterr().out
    assert rc == currency.EXIT_CURRENT == 0
    assert "OK: the pin is current" in out
    assert "BEHIND" not in out


def test_gate_reports_unknown_never_current_on_unreachable_api(currency, monkeypatch, capsys):
    monkeypatch.setattr(currency, "upstream_releases", lambda: ([], "simulated: rate limited"))
    rc = currency.main(["--gate"])
    out = capsys.readouterr().out
    assert rc == currency.EXIT_UNKNOWN == 2
    assert "COULD NOT CHECK" in out
    assert "NOT 'the pin is up to date'" in out
    assert "OK:" not in out and "STALE" not in out


def test_gate_unparseable_pin_reports_unknown(currency, capsys):
    rc = currency.main(["--pinned", "not-a-tag", "--gate"])
    out = capsys.readouterr().out
    assert rc == currency.EXIT_UNKNOWN == 2
    assert "COULD NOT CHECK" in out


def test_gate_falls_back_to_release_date_when_pin_is_off_the_page(currency, monkeypatch, capsys):
    """The realistic case check_llama_pin.py's own gate already covers: an old
    pin has fallen off the fetched page, so its date needs a second, per-tag
    lookup."""
    pin = currency.pinned_tag()
    n = currency._build_number(pin)
    monkeypatch.setattr(currency, "upstream_releases", lambda: (
        _releases((f"b{n + 5}", 40),), ""))
    seen = []

    def by_tag(tag):
        seen.append(tag)
        return _day(0)
    monkeypatch.setattr(currency, "release_date", by_tag)
    rc = currency.main(["--gate"])
    out = capsys.readouterr().out
    assert rc == currency.EXIT_STALE
    assert seen == [pin]
    assert "age: 40 day(s)" in out


def test_gate_reports_unknown_when_even_the_per_tag_lookup_fails(currency, monkeypatch, capsys):
    pin = currency.pinned_tag()
    n = currency._build_number(pin)
    monkeypatch.setattr(currency, "upstream_releases", lambda: (
        _releases((f"b{n + 5}", 40),), ""))
    monkeypatch.setattr(currency, "release_date", lambda tag: None)
    rc = currency.main(["--gate"])
    out = capsys.readouterr().out
    assert rc == currency.EXIT_UNKNOWN == 2
    assert "age: UNKNOWN" in out
    assert "COULD NOT CHECK" in out
    assert "STALE" not in out
    assert currency.main([]) == 0, "and without --gate it is still only a report"


def test_release_date_uses_the_injectable_opener_and_never_raises(currency):
    calls = []

    def opener(repo, tag):
        calls.append((repo, tag))
        return {"published_at": "2026-04-01T00:00:00Z"}
    assert currency.release_date("b1307", "owner/repo", opener=opener) == dt.datetime(
        2026, 4, 1, tzinfo=dt.timezone.utc)
    assert calls == [("owner/repo", "b1307")]

    def raiser(repo, tag):
        raise OSError("no network")
    assert currency.release_date("b1307", opener=raiser) is None

    def malformed(repo, tag):
        return ["not", "a", "dict"]
    assert currency.release_date("b1307", opener=malformed) is None


def test_gate_annotates_and_summarises_only_where_actions_will_show_it(
        currency, monkeypatch, capsys, tmp_path):
    """Under GITHUB_ACTIONS a stale gate emits an ::error annotation and writes
    the step summary; outside Actions it prints neither."""
    pin = currency.pinned_tag()
    n = currency._build_number(pin)
    monkeypatch.setattr(currency, "upstream_releases", lambda: (
        _releases((f"b{n + 7}", 30),), ""))
    monkeypatch.setattr(currency, "release_date", lambda tag: _day(0))

    assert currency.main(["--gate"]) == currency.EXIT_STALE
    assert "::error::" not in capsys.readouterr().out

    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert currency.main(["--gate"]) == currency.EXIT_STALE
    out = capsys.readouterr().out
    assert "::error::" in out and "STALE" in out
    text = summary.read_text(encoding="utf-8")
    assert "STALE" in text and "30 day(s)" in text and f"b{n + 7}" in text

    # The pre-existing default-mode path is untouched by --gate: it never
    # annotates at all, under GITHUB_ACTIONS or not (see
    # test_default_mode_is_unaffected_by_the_gate_machinery for the full check).
    assert currency.main([]) == 0
    out = capsys.readouterr().out
    assert "::" not in out


def test_default_mode_is_unaffected_by_the_gate_machinery(currency, monkeypatch, capsys):
    """The existing default-mode contract (always exit 0, no --gate vocabulary)
    must survive the --gate addition untouched."""
    pin = currency.pinned_tag()
    n = currency._build_number(pin)
    monkeypatch.setattr(currency, "upstream_tags", lambda: ([f"b{n + 1}", pin], ""))
    rc = currency.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "EXIT_STALE" not in out
    assert "tolerance" not in out
    assert "::" not in out


def _boom(*a, **k):
    raise OSError("no network")


class _FakeHTTP:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
