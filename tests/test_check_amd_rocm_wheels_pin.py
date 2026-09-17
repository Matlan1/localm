# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline tests for scripts/check_amd_rocm_wheels_pin.py.

No real network calls: the AMD wheel index is driven through the injectable
``opener`` seam on ``_fetch_index`` (unit tests of the fetch wrapper itself)
or, for the end-to-end tests, by monkeypatching only the true leaf dependency
``_fetch_index_http`` - the one function that actually opens a socket - so
``main()`` and the real ``_fetch_index`` try/except logic still run for real.

Proves: (a) a wheel filename is parsed per PEP 427 with the version correctly
unquoted, (b) version comparison is by parsed tuple, not string, across a
digit-count change, (c) the win_amd64/cp312 filter actually discriminates
(a wheel for a different platform or python tag is excluded), (d) a
py3-none-only package reports no python-ABI ceiling rather than a false one,
(e) an unreachable index is reported as "could not check" and never as
"current", one package's failure never hides another's result, and (f) the
script always exits 0 regardless of outcome (report-only, no --gate).
"""

from __future__ import annotations

import importlib.util
import urllib.error
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_amd_rocm_wheels_pin.py"
_spec = importlib.util.spec_from_file_location("check_amd_rocm_wheels_pin", _PATH)
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)


def _href(fname: str) -> str:
    return f"../{fname}"


# --------------------------------------------------------------------------- #
#  Bound to the real shipped files                                            #
# --------------------------------------------------------------------------- #

def test_reads_the_real_pinned_torch_stack_versions_and_they_parse():
    for pkg in checker._TORCH_STACK:
        pinned = checker._pinned_torch_stack_version(pkg)
        assert isinstance(pinned, str) and pinned, pkg
        assert checker._base_version_tuple(pinned) is not None, (
            f"{pkg}=={pinned!r} in pyproject.toml no longer parses as a plain version")


def test_reads_the_real_pinned_rocm_sdk_versions_from_uv_lock_and_they_parse():
    for pkg in checker._ROCM_SDK:
        pinned = checker._pinned_rocm_sdk_version(pkg)
        assert isinstance(pinned, str) and pinned, pkg
        assert checker._base_version_tuple(pinned) is not None, (
            f"{pkg} version {pinned!r} in uv.lock no longer parses as a plain version")


def test_rocm_sdk_pin_is_not_in_pyproject_toml_only_in_uv_lock():
    """Pins the actual reason this script reads two different files: these two
    packages carry no ==version constraint in pyproject.toml at all (only an
    index reference), so _pinned_torch_stack_version's regex must not find one -
    if it ever does, this script's split reasoning about where the pin lives is
    wrong and needs re-checking, not silently ignoring."""
    for pkg in checker._ROCM_SDK:
        assert checker._pinned_torch_stack_version(pkg) is None, pkg


def test_reads_the_real_requires_python_floor():
    floor = checker._requires_python_floor()
    assert floor is not None
    assert checker._floor_minor(floor) == 12, (
        f"requires-python={floor!r} - if the floor really moved, this test's "
        "expectation needs updating, not the check")


# --------------------------------------------------------------------------- #
#  _parse_wheel                                                               #
# --------------------------------------------------------------------------- #

def test_parse_wheel_five_token_shape_and_unquotes_the_version():
    """The exact shape observed live on the real AMD index for torch/
    torchvision: version-build local suffix percent-encoded as %2B."""
    w = checker._parse_wheel(_href("torch-2.11.0%2Brocm7.13.0-cp312-cp312-win_amd64.whl"))
    assert w == {"version": "2.11.0+rocm7.13.0", "pytag": "cp312",
                 "abitag": "cp312", "platform": "win_amd64"}


def test_parse_wheel_py3_none_shape():
    """The exact shape observed live for rocm-sdk-core/rocm-sdk-libraries-
    gfx103x-all: no local-version suffix, no cp-tag."""
    w = checker._parse_wheel(_href("rocm_sdk_core-7.13.0-py3-none-win_amd64.whl"))
    assert w == {"version": "7.13.0", "pytag": "py3", "abitag": "none",
                 "platform": "win_amd64"}


def test_parse_wheel_six_token_build_tag_shape_still_parses():
    w = checker._parse_wheel(_href("pkg-1.0-1-py3-none-any.whl"))
    assert w == {"version": "1.0", "pytag": "py3", "abitag": "none", "platform": "any"}


def test_parse_wheel_ignores_non_wheel_and_malformed_hrefs():
    assert checker._parse_wheel(_href("rocm-7.13.0.tar.gz")) is None
    assert checker._parse_wheel(_href("not-a-wheel-name")) is None
    assert checker._parse_wheel("") is None


# --------------------------------------------------------------------------- #
#  _base_version_tuple                                                       #
# --------------------------------------------------------------------------- #

def test_base_version_tuple_strips_the_local_suffix():
    assert checker._base_version_tuple("2.11.0+rocm7.13.0") == (2, 11, 0)
    assert checker._base_version_tuple("7.13.0") == (7, 13, 0)


def test_base_version_tuple_rejects_unparseable_strings():
    assert checker._base_version_tuple("not-a-version") is None
    assert checker._base_version_tuple("") is None


def test_base_version_tuple_orders_numerically_not_lexically():
    """'2.9.1' > '2.11.0' as strings ('9' > '1'), which would wrongly report a
    real newer release as older. This is the same trap check_comfyui_pin.py's
    own version comparator is tested against."""
    assert "2.9.1" > "2.11.0"  # lexical comparison
    assert checker._base_version_tuple("2.9.1") < checker._base_version_tuple("2.11.0")


# --------------------------------------------------------------------------- #
#  newest_win_amd64_version / newest_win_amd64_pytag                          #
# --------------------------------------------------------------------------- #

def _wheel(version, pytag, platform="win_amd64", abitag=None):
    return {"version": version, "pytag": pytag, "abitag": abitag or pytag, "platform": platform}


def test_newest_win_amd64_version_filters_platform_and_pytag():
    wheels = [
        _wheel("2.9.1+rocm7.13.0", "cp312"),
        _wheel("2.11.0+rocm7.13.0", "cp312"),
        _wheel("2.11.0+rocm7.13.0", "cp313"),          # excluded: wrong pytag
        _wheel("3.0.0+rocm7.13.0", "cp312", platform="linux_x86_64"),  # excluded: wrong platform
    ]
    result = checker.newest_win_amd64_version(wheels, pytag="cp312")
    assert result == ("2.11.0+rocm7.13.0", (2, 11, 0))


def test_newest_win_amd64_version_with_no_pytag_filter_for_py3_none_packages():
    wheels = [_wheel("7.12.0", "py3"), _wheel("7.13.0", "py3")]
    assert checker.newest_win_amd64_version(wheels) == ("7.13.0", (7, 13, 0))


def test_newest_win_amd64_version_none_when_nothing_matches():
    wheels = [_wheel("1.0", "cp312", platform="linux_x86_64")]
    assert checker.newest_win_amd64_version(wheels, pytag="cp312") is None
    assert checker.newest_win_amd64_version([], pytag="cp312") is None


def test_newest_win_amd64_pytag_picks_the_highest_cp_tag_numerically():
    wheels = [_wheel("1.0", "cp310"), _wheel("1.0", "cp39"), _wheel("1.0", "cp313")]
    assert checker.newest_win_amd64_pytag(wheels) == "cp313", (
        "cp39 < cp313 numerically even though '39' > '3' as a bare string suffix")


def test_newest_win_amd64_pytag_none_for_a_py3_none_only_package():
    """A py3-none-only package imposes no ceiling - this must read as
    "nothing to report", never as a false cp0/cp-none floor."""
    wheels = [_wheel("7.13.0", "py3")]
    assert checker.newest_win_amd64_pytag(wheels) is None


def test_newest_win_amd64_pytag_ignores_non_win_amd64_wheels():
    wheels = [_wheel("1.0", "cp314", platform="linux_x86_64"), _wheel("1.0", "cp312")]
    assert checker.newest_win_amd64_pytag(wheels) == "cp312"


# --------------------------------------------------------------------------- #
#  _floor_minor                                                              #
# --------------------------------------------------------------------------- #

def test_floor_minor_parses_the_lower_bound():
    assert checker._floor_minor(">=3.12,<3.13") == 12
    assert checker._floor_minor(">=3.9") == 9


def test_floor_minor_none_when_unparseable():
    assert checker._floor_minor("not a constraint") is None


# --------------------------------------------------------------------------- #
#  _fetch_index (the injectable seam - no real network, no urllib patch)      #
# --------------------------------------------------------------------------- #

def test_fetch_index_returns_hrefs_from_a_working_opener():
    html = '<a href="../x-1.0-py3-none-any.whl">x</a>'
    result = checker._fetch_index("torch", opener=lambda pkg: html)
    assert result == ["../x-1.0-py3-none-any.whl"]


def test_fetch_index_returns_none_never_raises_on_network_error():
    def _raiser(pkg):
        raise urllib.error.URLError("simulated network failure")
    assert checker._fetch_index("torch", opener=_raiser) is None


def test_fetch_index_returns_none_on_malformed_response_shape():
    assert checker._fetch_index("torch", opener=lambda pkg: {"not": "a string"}) is None


# --------------------------------------------------------------------------- #
#  _report_package                                                            #
# --------------------------------------------------------------------------- #

def test_report_package_current(capsys):
    wheels = [checker._parse_wheel(_href("rocm_sdk_core-7.13.0-py3-none-win_amd64.whl"))]
    checker._report_package("rocm-sdk-core", "7.13.0", wheels)
    out = capsys.readouterr().out
    assert "current" in out and "STALE" not in out


def test_report_package_stale(capsys):
    wheels = [checker._parse_wheel(
        _href("torch-2.11.0%2Brocm7.13.0-cp312-cp312-win_amd64.whl"))]
    checker._report_package("torch", "2.9.1+rocm7.13.0", wheels)
    out = capsys.readouterr().out
    assert "STALE" in out and "2.11.0" in out


def test_report_package_unreachable_never_current(capsys):
    checker._report_package("torch", "2.9.1+rocm7.13.0", None)
    out = capsys.readouterr().out
    assert "could not check" in out
    assert "current" not in out and "STALE" not in out


def test_report_package_unreadable_pin(capsys):
    checker._report_package("torch", None, [])
    out = capsys.readouterr().out
    assert "could not read the pinned version" in out


def test_report_package_no_matching_wheel_on_the_page(capsys):
    wheels = [checker._parse_wheel(_href("torch-2.11.0-cp313-cp313-win_amd64.whl"))]
    checker._report_package("torch", "2.9.1+rocm7.13.0", wheels)
    out = capsys.readouterr().out
    assert "no matching" in out
    assert "STALE" not in out and "current" not in out


# --------------------------------------------------------------------------- #
#  _report_python_abi                                                        #
# --------------------------------------------------------------------------- #

def test_report_python_abi_bump_looks_safe_when_a_newer_cp_tag_is_published(capsys, monkeypatch):
    monkeypatch.setattr(checker, "_requires_python_floor", lambda: ">=3.12,<3.13")
    wheels = {
        "torch": [_wheel("2.11.0", "cp312"), _wheel("2.11.0", "cp313")],
        "torchvision": [_wheel("0.26.0", "cp312"), _wheel("0.26.0", "cp313")],
        "rocm-sdk-core": [_wheel("7.13.0", "py3")],
        "rocm-sdk-libraries-gfx103x-all": [_wheel("7.13.0", "py3")],
    }
    checker._report_python_abi(wheels)
    out = capsys.readouterr().out
    assert "cp313" in out
    assert "looks ROCm-safe TODAY" in out


def test_report_python_abi_not_safe_when_no_newer_cp_tag_exists(capsys, monkeypatch):
    monkeypatch.setattr(checker, "_requires_python_floor", lambda: ">=3.12,<3.13")
    wheels = {
        "torch": [_wheel("2.11.0", "cp312")],
        "torchvision": [_wheel("0.26.0", "cp312")],
        "rocm-sdk-core": [_wheel("7.13.0", "py3")],
        "rocm-sdk-libraries-gfx103x-all": [_wheel("7.13.0", "py3")],
    }
    checker._report_python_abi(wheels)
    out = capsys.readouterr().out
    assert "would NOT be ROCm-safe today" in out


def test_report_python_abi_unreachable_index_reports_could_not_determine(capsys):
    checker._report_python_abi({"torch": None, "torchvision": None,
                                "rocm-sdk-core": None, "rocm-sdk-libraries-gfx103x-all": None})
    out = capsys.readouterr().out
    assert "could not determine" in out


def test_report_python_abi_never_reports_a_ceiling_from_py3_none_alone(capsys):
    """If torch/torchvision's own index happened to publish nothing usable but
    the two py3-none packages did, there is still no cp-tag ceiling to report -
    py3-none packages must never be read as imposing one."""
    wheels = {"torch": [], "torchvision": [],
              "rocm-sdk-core": [_wheel("7.13.0", "py3")],
              "rocm-sdk-libraries-gfx103x-all": [_wheel("7.13.0", "py3")]}
    checker._report_python_abi(wheels)
    out = capsys.readouterr().out
    assert "could not determine" in out or "no win_amd64 wheel" in out


def test_report_python_abi_never_claims_safe_when_one_torch_stack_package_lags(
        capsys, monkeypatch):
    """THE BUG THIS PINS: torch publishing cp313 must NOT make the script claim
    cp313 is safe when torchvision only publishes up to cp312 - the ceiling is
    the MINIMUM across the whole torch stack, never a tag pooled across both
    packages' wheel lists combined. A realistic case (the two packages'
    release cadences are independent, so one publishing a new ABI ahead of the
    other is ordinary, not a fixture artifact)."""
    monkeypatch.setattr(checker, "_requires_python_floor", lambda: ">=3.12,<3.13")
    wheels = {
        "torch": [_wheel("2.11.0", "cp312"), _wheel("2.11.0", "cp313")],
        "torchvision": [_wheel("0.26.0", "cp312")],  # no cp313 yet
        "rocm-sdk-core": [_wheel("7.13.0", "py3")],
        "rocm-sdk-libraries-gfx103x-all": [_wheel("7.13.0", "py3")],
    }
    checker._report_python_abi(wheels)
    out = capsys.readouterr().out
    assert "looks ROCm-safe" not in out, (
        "torch's own cp313 must never be reported as THE safe ceiling when "
        "torchvision has no cp313 wheel")
    assert "would NOT be ROCm-safe today" in out
    assert "torch cp313" in out and "torchvision cp312" in out, (
        "each package's own ceiling must be shown, not a pooled/combined one")


def test_report_python_abi_incomplete_when_one_package_is_unreachable(capsys):
    """One package's fetch failing must not silently fall back to reporting a
    ceiling based only on the package that succeeded - that is exactly as
    unsafe as the pooled-max bug: the unreachable package's real ceiling is
    unknown and could be lower."""
    wheels = {
        "torch": [_wheel("2.11.0", "cp313")],
        "torchvision": None,  # fetch failed
        "rocm-sdk-core": [_wheel("7.13.0", "py3")],
        "rocm-sdk-libraries-gfx103x-all": [_wheel("7.13.0", "py3")],
    }
    checker._report_python_abi(wheels)
    out = capsys.readouterr().out
    assert "could not determine a full ceiling" in out
    assert "torchvision" in out, "the package that could not be checked must be named"
    assert "torch cp313" in out, "what WAS reached is still shown"
    assert "looks ROCm-safe" not in out, (
        "torch's own cp313 must not be reported as a verified-safe ceiling while "
        "torchvision's real ceiling is unknown")


# --------------------------------------------------------------------------- #
#  main() end-to-end - only the true leaf (_fetch_index_http) is patched.    #
# --------------------------------------------------------------------------- #

def test_main_reports_could_not_check_on_unreachable_index_never_current(monkeypatch, capsys):
    def _raiser(pkg):
        raise urllib.error.URLError("simulated: rate limited")
    monkeypatch.setattr(checker, "_fetch_index_http", _raiser)
    rc = checker.main([])
    out = capsys.readouterr().out

    assert rc == 0
    assert "could not check" in out
    assert "current" not in out.split("Python ABI")[0]


def test_main_always_exits_zero_report_only(monkeypatch, capsys):
    """No --gate exists at all: the script is a maintenance signal, never a
    build gate, regardless of what it finds."""
    real_torch_html = (
        '<a href="../torch-99.0.0-cp312-cp312-win_amd64.whl">x</a>'  # deliberately far ahead
    )

    def opener(pkg):
        if pkg == "torch":
            return real_torch_html
        return '<a href="../x-1.0-py3-none-win_amd64.whl">x</a>'
    monkeypatch.setattr(checker, "_fetch_index_http", opener)
    rc = checker.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "STALE" in out  # confirms this run actually exercised the stale path


def test_main_one_package_failing_does_not_hide_the_others(monkeypatch, capsys):
    def opener(pkg):
        if pkg == "torch":
            raise urllib.error.URLError("simulated failure for torch only")
        return '<a href="../x-1.0-py3-none-win_amd64.whl">x</a>'
    monkeypatch.setattr(checker, "_fetch_index_http", opener)
    rc = checker.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "torch: pinned" in out and "could not check" in out
    # torchvision (also a _TORCH_STACK member) must still be reported, since
    # only torch's own fetch failed.
    assert "torchvision:" in out
