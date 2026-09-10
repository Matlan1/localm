# SPDX-License-Identifier: AGPL-3.0-or-later
"""The pin CONSTANT's own invariants, and the currency check that stops it rotting.

This file covers three properties of the pin:

  * it states what it rests on, PER BACKEND, so a confirmation cannot read as
    green while silently skipping backends it could not test;
  * it is compared against upstream by scripts/check_llama_pin.py, whose --gate
    mode turns "too far behind" into a non-zero exit and keeps "could not check"
    distinct from both "current" and "stale";
  * the workflow that runs that gate has the shape that makes the exit code
    reach someone: push to master and a schedule, never pull_request, read-only
    permissions.

Nothing here asserts the pin's VALUE.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import re
from pathlib import Path

import pytest
import yaml

from localm import setup_llama as sl

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "check_llama_pin.py"
_WORKFLOW = _ROOT / ".github" / "workflows" / "llama-pin-currency.yml"
_CI = _ROOT / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def currency():
    """scripts/check_llama_pin.py, loaded by path. Not a package module: it is
    stdlib-only so the CI job can run it with nothing installed."""
    spec = importlib.util.spec_from_file_location("check_llama_pin", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Every test in this file runs with urlopen raising, so no test can reach
    GitHub by accident; a test that needs a payload installs its own."""
    monkeypatch.setattr("urllib.request.urlopen", _boom)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)


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
    """The pin is interpolated into a release URL path segment exactly like a
    user's --tag, so it must pass the same predicate. It is written by hand in
    this file, which is the one entry point _validated_tag never sees."""
    assert sl.is_safe_tag(sl._PINNED_TAG)
    assert sl._PINNED_TAG.lower() not in (sl._TRACK_LATEST, sl._TRACK_DEFAULT), (
        "the pin must not collide with the words that mean 'track upstream' or "
        "'use the pin' - either would make --tag ambiguous")


def test_every_backend_states_what_its_pin_rests_on():
    """A NEW BACKEND CANNOT BE ADDED WITHOUT SAYING WHAT ITS PIN RESTS ON.

    The entry is required, not optional: an absent entry would read as "covered
    by the confirmation like everything else"."""
    backends = {b for plat in sl._ASSET_MATCH.values() for b in plat} | {"amd-rocm"}
    missing = sorted(backends - set(sl._PIN_CONFIRMATION))
    assert not missing, (
        f"no _PIN_CONFIRMATION entry for {missing}; say what the pin rests on "
        "for each, including 'NOT measured' when that is the truth")


def test_no_backend_gets_to_be_vague_about_confirmation():
    """Each entry must either CLAIM a measurement or DISCLAIM one, with no middle
    ground. Wording like "should be fine" is exactly the shape that turns into a
    false "confirmed" when someone summarises this table later."""
    for backend, note in sl._PIN_CONFIRMATION.items():
        claims = "load + generate, measured" in note
        disclaims = "NOT measured" in note
        assert claims != disclaims, (
            f"_PIN_CONFIRMATION[{backend!r}] must say either 'load + generate, "
            f"measured' or 'NOT measured', not both and not neither: {note!r}")


def test_the_untested_backends_are_the_ones_needing_absent_hardware():
    """Pins the actual asymmetry, not a count, so this fails if a measured backend
    is downgraded to save a test run, or an unmeasured one is upgraded without
    being measured.

    cpu and vulkan are measurable here (any machine; this project's own AMD box).
    cuda, sycl, hip and metal need hardware nobody here has. amd-rocm is NOT in
    the measured set even though this box could run it: it ships from a different
    tag series (_ROCM_TAG), so this pin's confirmation never touched it."""
    measured = {b for b, note in sl._PIN_CONFIRMATION.items()
                if "load + generate, measured" in note}
    assert measured == {"cpu", "vulkan"}, measured


# --------------------------------------------------------------------------- #
#  The currency check: reading and comparing                                   #
# --------------------------------------------------------------------------- #

def test_currency_reads_the_same_pin_the_code_uses(currency):
    """It reads the constant BY TEXT so the CI job needs no install. The cost of
    that is a second source of truth, so the two are compared here - otherwise a
    rename would leave the check silently reporting an old value."""
    assert currency.pinned_tag() == sl._PINNED_TAG


def test_currency_compares_build_numbers_numerically_not_lexically(currency):
    """b9870 vs b10375 is the trap: '9' sorts after '1', so a string comparison
    calls the OLDER pin current. It starts being wrong at the exact moment the
    digit count changes, which happens once and then looks fine forever after."""
    assert currency._build_number("b9870") < currency._build_number("b10375")
    assert sorted(["b9870", "b10375", "b10361"],
                  key=currency._build_number) == ["b9870", "b10361", "b10375"]


def test_currency_refuses_to_report_currency_it_did_not_verify(currency, capsys):
    """A blocked lookup and an up-to-date pin must never print the same thing,
    and under --gate a blocked lookup is exit 2: neither the 0 that would read
    as current nor the 1 that would read as stale."""
    assert currency.main([]) == 0, "a maintenance signal never fails the build"
    out = capsys.readouterr().out
    assert "COULD NOT CHECK" in out
    assert "NOT 'the pin is up to date'" in out
    assert "OK:" not in out

    assert currency.main(["--gate"]) == currency.EXIT_UNKNOWN
    out = capsys.readouterr().out
    assert "COULD NOT CHECK" in out
    assert "STALE" not in out


def test_currency_reports_a_gap_and_says_how_to_close_it(currency, monkeypatch, capsys):
    pin = currency.pinned_tag()
    n = currency._build_number(pin)
    monkeypatch.setattr(currency, "upstream_releases", lambda: (
        _releases((f"b{n + 2}", 3), (f"b{n + 1}", 2), (pin, 0)), ""))
    assert currency.main([]) == 0
    out = capsys.readouterr().out
    assert "BEHIND: 2 builds, 2 newer release(s)" in out
    assert "age: 3 day(s)" in out
    assert "confirm_llama_runtime.py" in out, (
        "a bump without the confirm is the untested-build problem this exists "
        "to remove, so the remedy must name the confirm step")
    assert "bump_llama_pin.py" in out, "and the script that does the mechanical half"


def test_currency_does_not_call_a_lexically_larger_older_tag_newer(
        currency, monkeypatch, capsys):
    """THE COMPARISON ITSELF, on a value that DISCRIMINATES.

    A tag needs FEWER DIGITS to discriminate: 'b9999' sorts AFTER 'b10375' as a
    string and is far older as a number. A lexical comparison reports the pin as
    behind; the correct one reports it current."""
    older = "b9999"
    assert older > currency.pinned_tag(), (
        "this fixture only discriminates while the pin has more digits than the "
        "decoy - if that ever stops holding, pick a smaller decoy")
    monkeypatch.setattr(currency, "upstream_releases", lambda: (_releases((older, 0)), ""))

    assert currency.main(["--gate"]) == currency.EXIT_CURRENT
    out = capsys.readouterr().out
    assert "OK: the pin is current" in out, out
    assert "BEHIND" not in out


def test_currency_skips_releases_whose_assets_are_not_uploaded_yet(currency, monkeypatch):
    """Upstream publishes a release before its ~25 archives finish uploading.
    Counting one of those as "behind" overstates the gap and would point the
    advance step at a tag that cannot be downloaded yet. Every entry is flagged
    prerelease, matching what ggml-org/llama.cpp actually sends, so this also
    pins that the asset/draft checks are what exclude b99999 and b99997 - not
    the prerelease flag."""
    payload = [
        {"tag_name": "b99999", "draft": False, "prerelease": True, "assets": []},
        {"tag_name": "b99998", "draft": False, "prerelease": True,
         "assets": [{"name": "x"}], "published_at": "2026-01-02T00:00:00Z"},
        {"tag_name": "b99997", "draft": True, "prerelease": True,
         "assets": [{"name": "x"}]},
    ]
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeHTTP(payload))
    tags, err = currency.upstream_tags()
    assert err == ""
    assert tags == ["b99998"], "asset-less and draft releases are not candidates"
    releases, _ = currency.upstream_releases()
    assert releases[0]["published_at"] == _day(1), "the release date rides along"


def test_currency_treats_a_prerelease_flagged_release_as_a_real_candidate(
        currency, monkeypatch):
    """Every release ggml-org/llama.cpp publishes is flagged prerelease, with no
    signal value: a real, asset-bearing, non-draft release must still surface."""
    payload = [
        {"tag_name": "b99999", "draft": False, "prerelease": True,
         "assets": [{"name": "x"}] * 16},
    ]
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeHTTP(payload))
    tags, err = currency.upstream_tags()
    assert err == ""
    assert tags == ["b99999"], (
        "a prerelease-flagged, asset-bearing, non-draft release must survive "
        "the filter")


def test_currency_reports_the_release_count_as_a_floor_when_the_page_ends_above_the_pin(
        currency, monkeypatch, capsys):
    """When every release on the page is newer than the pin, the page ended
    before reaching it and the count is a lower bound. That case used to be
    reported only when the raw page was full, so one draft or asset-less entry
    on the page hid the floor and "99 releases" stood in for a gap five times
    that size."""
    pin = currency.pinned_tag()
    n = currency._build_number(pin)
    monkeypatch.setattr(currency, "upstream_releases", lambda: (
        _releases((f"b{n + 3}", 3), (f"b{n + 2}", 2), (f"b{n + 1}", 1)), ""))
    monkeypatch.setattr(currency, "release_date", lambda tag: _day(0))
    assert currency.main([]) == 0
    out = capsys.readouterr().out
    assert "3+ newer release(s)" in out
    assert "floor" in out

    monkeypatch.setattr(currency, "upstream_releases", lambda: (
        _releases((f"b{n + 1}", 1), (pin, 0)), ""))
    assert currency.main([]) == 0
    out = capsys.readouterr().out
    assert "1 newer release(s)" in out
    assert "1+" not in out and "floor" not in out


def test_the_pin_date_is_looked_up_by_tag_when_the_page_does_not_reach_it(
        currency, monkeypatch, capsys):
    """A stale pin is exactly the one that has fallen off the first page, so its
    date has to come from a second lookup. When that lookup fails too, the age
    is UNKNOWN and --gate exits 2, never 1 and never 0."""
    pin = currency.pinned_tag()
    n = currency._build_number(pin)
    monkeypatch.setattr(currency, "upstream_releases", lambda: (
        _releases((f"b{n + 5}", 40),), ""))
    seen = []

    def by_tag(tag):
        seen.append(tag)
        return _day(0)
    monkeypatch.setattr(currency, "release_date", by_tag)
    assert currency.main(["--gate"]) == currency.EXIT_STALE
    assert seen == [pin]
    assert "age: 40 day(s)" in capsys.readouterr().out

    monkeypatch.setattr(currency, "release_date", lambda tag: None)
    assert currency.main(["--gate"]) == currency.EXIT_UNKNOWN
    out = capsys.readouterr().out
    assert "age: UNKNOWN" in out
    assert "COULD NOT CHECK" in out
    assert "STALE" not in out
    assert currency.main([]) == 0, "and without --gate it is still only a report"


# --------------------------------------------------------------------------- #
#  The gate                                                                    #
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


def test_gate_annotates_and_summarises_only_where_actions_will_show_it(
        currency, monkeypatch, capsys, tmp_path):
    """Under GITHUB_ACTIONS a stale gate emits an ::error annotation and writes
    the step summary; outside Actions it prints neither. The summary carries
    the numbers a person needs without opening the log."""
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

    assert currency.main([]) == 0, "without --gate a stale pin is a warning, not an error"
    out = capsys.readouterr().out
    assert "::warning::" in out and "::error::" not in out


def test_github_token_rides_as_a_bearer_header_when_present(currency, monkeypatch):
    captured = []

    def capture(req, timeout=None):
        captured.append(req)
        return _FakeHTTP({"published_at": "2026-01-01T00:00:00Z"})
    monkeypatch.setattr("urllib.request.urlopen", capture)

    assert currency.release_date("b1") == _day(0)
    assert captured[-1].get_header("Authorization") is None

    monkeypatch.setenv("GITHUB_TOKEN", "tok-123")
    assert currency.release_date("b1") == _day(0)
    assert captured[-1].get_header("Authorization") == "Bearer tok-123"


# --------------------------------------------------------------------------- #
#  The workflow that carries the gate                                          #
# --------------------------------------------------------------------------- #

def _load_workflow(path: Path) -> dict:
    wf = yaml.safe_load(path.read_text(encoding="utf-8"))
    # PyYAML reads the bare key `on` as the boolean True.
    wf["on"] = wf.pop(True, wf.get("on"))
    return wf


def test_the_currency_workflow_runs_the_gate_where_a_red_reaches_someone():
    """Bound to the real file: the gate runs on every push to master and on its
    own schedule, never on pull_request (a stale pin is never a PR's failure),
    with read-only permissions, and it invokes the script in --gate mode."""
    wf = _load_workflow(_WORKFLOW)
    on = wf["on"]
    assert on["push"]["branches"] == ["master"]
    assert "schedule" in on and on["schedule"][0]["cron"]
    assert "workflow_dispatch" in on
    assert "pull_request" not in on and "pull_request_target" not in on

    ci_cron = _load_workflow(_CI)["on"]["schedule"][0]["cron"]
    assert on["schedule"][0]["cron"] != ci_cron, (
        "a second sample of GitHub's scheduler needs a different slot")

    assert wf["permissions"] == {"contents": "read"}
    for name, job in wf["jobs"].items():
        assert "permissions" not in job, f"job {name} must not widen permissions"
        for step in job["steps"]:
            uses = step.get("uses")
            if uses:
                assert re.search(r"@[0-9a-f]{40}(\s|$)", uses), f"{name}: unpinned {uses}"
            with_ = step.get("with") or {}
            if uses and uses.startswith("actions/checkout@"):
                assert with_.get("persist-credentials") is False

    gate = wf["jobs"]["currency"]
    runs = "\n".join(s.get("run", "") for s in gate["steps"])
    assert "check_llama_pin.py --gate" in runs
    assert 'if [ "$rc" = "2" ]' in runs, "could-not-check must not turn the job red"
    assert 'exit "$rc"' in runs, "and stale must"

    preflight = wf["jobs"]["candidate-preflight"]
    assert "workflow_dispatch" in preflight["if"] and "candidate_tag" in preflight["if"]
    runs = "\n".join(s.get("run", "") for s in preflight["steps"])
    for script in ("check_llama_abi.py", "confirm_llama_runtime.py",
                   "check_pretokenizer_redos.py", "bump_llama_pin.py"):
        assert script in runs, f"the pre-flight must run {script}"
    assert "--write" not in runs, "the pre-flight writes nothing"


def test_the_abi_check_pin_report_survives_a_failure_earlier_in_the_job():
    """The report step in ci.yml sits after three steps that track a moving
    upstream target; without `if: always()` a header drift skips it, which is
    the run where the report matters most."""
    ci = _load_workflow(_CI)
    steps = ci["jobs"]["abi-check"]["steps"]
    report = [s for s in steps if "check_llama_pin.py" in s.get("run", "")]
    assert len(report) == 1
    assert report[0].get("if") == "always()"
    assert report[0].get("continue-on-error") is True


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
