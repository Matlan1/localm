# SPDX-License-Identifier: AGPL-3.0-or-later
"""The pin CONSTANT's own safety, and the currency check that stops it rotting.

Companion to test_llama_pin_constant_and_currency.py, which covers the
separate ggml-org/llama.cpp pin (_PINNED_TAG). This file covers _ROCM_TAG's own
tag safety and its currency check, scripts/check_llama_rocm_pin.py - a
different tag series sourced from lemonade-sdk/llamacpp-rocm.

Nothing here asserts the pin's VALUE.
"""

from __future__ import annotations

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
