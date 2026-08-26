# SPDX-License-Identifier: AGPL-3.0-or-later
"""The pin CONSTANT's own invariants, and the currency check that stops it rotting.

This file covers two properties of the pin itself:

  * it states what it rests on, PER BACKEND, so a confirmation cannot read as
    green while silently skipping backends it could not test;
  * it is visibly compared against upstream, so a pin nobody advances is
    detectable.

Nothing here asserts the pin's VALUE.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from localm import setup_llama as sl

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_llama_pin.py"


@pytest.fixture(scope="module")
def currency():
    """scripts/check_llama_pin.py, loaded by path. Not a package module: it is
    stdlib-only so the CI job can run it with nothing installed."""
    spec = importlib.util.spec_from_file_location("check_llama_pin", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
#  The currency check                                                          #
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
    """A blocked lookup and an up-to-date pin must never print the same thing.
    This check exists to make a stale pin visible; one that says "OK" when it
    could not reach the API would hide exactly what it was built to surface."""
    currency.urllib.request.urlopen = _boom
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
    assert "confirm_llama_runtime.py" in out, (
        "a bump without the confirm is the untested-build problem this exists "
        "to remove, so the remedy must name the confirm step")


def test_currency_does_not_call_a_lexically_larger_older_tag_newer(
        currency, monkeypatch, capsys):
    """THE COMPARISON ITSELF, on a value that DISCRIMINATES.

    The test above cannot catch a lexical comparison: b99999 and b99998 are both
    lexically AND numerically greater than the pin, so a lexical and a numeric
    implementation agree on that fixture.

    A tag needs FEWER DIGITS to discriminate: 'b9999' sorts AFTER 'b10375' as a
    string and is far older as a number. A lexical comparison reports the pin as
    behind; the correct one reports it current."""
    older = "b9999"
    assert older > currency.pinned_tag(), (
        "this fixture only discriminates while the pin has more digits than the "
        "decoy - if that ever stops holding, pick a smaller decoy")
    monkeypatch.setattr(currency, "upstream_tags", lambda: ([older], ""))

    assert currency.main([]) == 0
    out = capsys.readouterr().out
    assert "OK: the pin is current" in out, out
    assert "BEHIND" not in out


def test_currency_skips_releases_whose_assets_are_not_uploaded_yet(currency, monkeypatch):
    """Upstream publishes a release before its ~25 archives finish uploading.
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
