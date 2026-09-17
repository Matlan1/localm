#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Report staleness of localm's AMD ROCm wheel pins, and Python-ABI headroom.

localm's ``[gpu]`` extra pins torch, torchvision, rocm-sdk-core and
rocm-sdk-libraries-gfx103x-all from AMD's own wheel index
(``https://repo.amd.com/rocm/whl/gfx103X-all/``), an ordinary PEP 503 simple
index. torch and torchvision carry an explicit ``==version`` in
``pyproject.toml``'s ``dependencies``; rocm-sdk-core and
rocm-sdk-libraries-gfx103x-all carry none there (only an index reference in
``[tool.uv.sources]``), so their actual pinned version is whatever ``uv.lock``
resolved and recorded.

This is a MAINTENANCE SIGNAL, the same shape as scripts/check_llama_pin.py,
scripts/check_comfyui_pin.py and scripts/check_llama_rocm_pin.py for this
tree's other pinned dependencies, but REPORT-ONLY with no ``--gate`` mode:
unlike those three, there is no single upstream "release" to compare
against - four independently-versioned packages plus a separate Python-ABI
signal do not reduce to one pass/fail verdict without inventing an aggregation
rule this script has no basis for. It always exits 0 and changes nothing.

TWO THINGS REPORTED, since they share the same fetch:

  (a) PACKAGE STALENESS - for each of the four packages, whether the pinned
      version is still the newest version the index publishes a wheel for
      (win_amd64, and for torch/torchvision specifically cp312 - the ABI
      localm ships on; rocm-sdk-core/rocm-sdk-libraries-gfx103x-all publish
      py3-none wheels with no cp-tag at all).
  (b) PYTHON ABI AVAILABILITY - the newest cp-tag (cp312, cp313, ...) the
      index publishes a win_amd64 wheel for at all, compared against
      ``requires-python``'s floor in pyproject.toml, so a Python bump's
      ROCm-safety can be read off directly instead of re-derived by hand.
      This NEVER touches requires-python, target-version, or any Python
      version pin - report only; bumping those is the maintainer's call,
      exactly like every other pin covered by this file's sibling scripts.

Fails soft per package: an unreachable index, a malformed response, or a
version that does not parse are each reported as "could not check" for that
one package - never a false "current" and never a crash that hides the other
three packages' results.

Usage:
    python scripts/check_amd_rocm_wheels_pin.py

Stdlib only (urllib + re), so it runs anywhere without extra installs - it
does not need localm, uv, or the AMD wheels themselves installed.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote

_INDEX_BASE = "https://repo.amd.com/rocm/whl/gfx103X-all"
_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT_PATH = _ROOT / "pyproject.toml"
_UV_LOCK_PATH = _ROOT / "uv.lock"

# torch/torchvision carry an explicit ==version in pyproject.toml; the two
# rocm-sdk-* packages do not (see _pinned_rocm_sdk_version). `rocm` itself
# (the meta-package pyproject.toml also declares) ships no wheel at all - an
# sdist only - so it has no ABI/version signal to report here.
_TORCH_STACK = ("torch", "torchvision")
_ROCM_SDK = ("rocm-sdk-core", "rocm-sdk-libraries-gfx103x-all")
_PACKAGES = _TORCH_STACK + _ROCM_SDK

_HREF_RE = re.compile(r'href="([^"]+)"')
_REQUIRES_PYTHON_RE = re.compile(r'^requires-python\s*=\s*"([^"]+)"', re.M)


# --------------------------------------------------------------------------- #
#  Reading the pins                                                           #
# --------------------------------------------------------------------------- #

def _pyproject_text() -> str:
    return _PYPROJECT_PATH.read_text(encoding="utf-8")


def _pinned_torch_stack_version(pkg: str) -> "str | None":
    """'2.11.0+rocm7.13.0' out of pyproject.toml's dependencies list, by text
    (not by parsing TOML): matches this tree's other check_*.py scripts, which
    all read their pin the same way so the CI job needs nothing installed.
    None when the line is not there in the expected shape."""
    m = re.search(rf'"{re.escape(pkg)}==([^";]+);', _pyproject_text())
    return m.group(1) if m else None


def _pinned_rocm_sdk_version(pkg: str, path: Path = _UV_LOCK_PATH) -> "str | None":
    """rocm-sdk-core / rocm-sdk-libraries-gfx103x-all carry NO version
    constraint in pyproject.toml - only an index reference - so uv resolves
    whatever the AMD index currently publishes, and uv.lock records the
    result. That lockfile entry, not pyproject.toml, is this dependency's
    actual pin."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(
        rf'^\[\[package\]\]\nname = "{re.escape(pkg)}"\nversion = "([^"]+)"',
        text, re.M)
    return m.group(1) if m else None


def _requires_python_floor() -> "str | None":
    m = _REQUIRES_PYTHON_RE.search(_pyproject_text())
    return m.group(1) if m else None


def _floor_minor(requires_python: str) -> "int | None":
    """'>=3.12,<3.13' -> 12. Reads the first '>=3.Y' clause; this repo has
    pinned exactly one such constraint since requires-python was written."""
    m = re.search(r">=\s*3\.(\d+)", requires_python)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
#  Fetching + parsing the AMD wheel index                                     #
# --------------------------------------------------------------------------- #

def _fetch_index_http(pkg: str) -> str:
    """Real index page fetch. Raises on any failure; see _fetch_index, which
    never lets that propagate."""
    url = f"{_INDEX_BASE}/{pkg}/"
    req = urllib.request.Request(
        url, headers={"User-Agent": "localm-check-amd-rocm-wheels-pin"})
    with urllib.request.urlopen(req, timeout=20) as r:  # noqa: S310 - fixed https:// URL
        return r.read().decode("utf-8")


def _fetch_index(pkg: str, *, opener=None) -> "list[str] | None":
    """href targets listed on *pkg*'s AMD index page, or None if the page
    could not be fetched for ANY reason. None must be read as "unknown",
    never as "no wheels" and never as "current".

    *opener* is injectable so a test can drive the unreachable/malformed
    paths with a plain function, the same seam scripts/check_comfyui_pin.py
    uses for _fetch_releases."""
    if opener is None:
        opener = _fetch_index_http
    try:
        html = opener(pkg)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        print(f"  (could not reach the AMD wheel index for {pkg}: {e})", file=sys.stderr)
        return None
    if not isinstance(html, str):
        print(f"  (unexpected response shape for {pkg}: {type(html).__name__})", file=sys.stderr)
        return None
    return _HREF_RE.findall(html)


def _parse_wheel(href: str) -> "dict | None":
    """One href -> {version, pytag, abitag, platform}, or None when it is not
    a plain 5- or 6-token PEP 427 wheel name (a build-tag wheel has 6 tokens;
    every wheel observed on this index has 5, but a stray non-.whl href, or a
    future 6-token wheel, must not crash the sweep - it is simply excluded).

    *version* is unquoted (the index URL-encodes '+' as '%2B') and otherwise
    left exactly as published, local-version suffix included."""
    fname = href.rsplit("/", 1)[-1]
    if not fname.endswith(".whl"):
        return None
    parts = fname[:-4].split("-")
    if len(parts) == 5:
        _dist, ver, pytag, abitag, plat = parts
    elif len(parts) == 6:
        _dist, ver, _build, pytag, abitag, plat = parts
    else:
        return None
    return {"version": unquote(ver), "pytag": pytag, "abitag": abitag, "platform": plat}


def _base_version_tuple(version: str) -> "tuple[int, ...] | None":
    """'2.11.0+rocm7.13.0' -> (2, 11, 0). The local-version suffix is the AMD
    ROCm release tag shared by every wheel in one publish - it does not order
    numerically against a differently-suffixed version and is not what makes
    one of these packages newer than another; only the base version does."""
    base = version.split("+", 1)[0]
    if not re.match(r"^\d+(\.\d+)*$", base):
        return None
    return tuple(int(p) for p in base.split("."))


def newest_win_amd64_version(
        wheels: "list[dict]", *, pytag: "str | None" = None
) -> "tuple[str, tuple[int, ...]] | None":
    """(raw version string, parsed base tuple) of the highest-version
    win_amd64 wheel, optionally filtered to an exact *pytag* (torch/
    torchvision: the ABI localm ships on; pass None for a py3-none package,
    where pytag does not distinguish anything). None when nothing matches."""
    candidates = []
    for w in wheels:
        if w["platform"] != "win_amd64":
            continue
        if pytag is not None and w["pytag"] != pytag:
            continue
        t = _base_version_tuple(w["version"])
        if t is None:
            continue
        candidates.append((t, w["version"]))
    if not candidates:
        return None
    t, v = max(candidates, key=lambda pair: pair[0])
    return v, t


_CP3_TAG_RE = re.compile(r"^cp3(\d+)$")


def _cp3_tag_minor(tag: str) -> "int | None":
    """'cp312' -> 12, 'cp39' -> 9: this codebase's entire Python-version domain
    is 3.x (matching _floor_minor's own >=3.Y assumption), so a tag's minor is
    everything after the "cp3" prefix, never a bare slice off "cp" - "cp312"[2:]
    is "312", not "12". None for anything not shaped like a Python 3.x cp-tag
    (py3-none, cp27, ...)."""
    m = _CP3_TAG_RE.match(tag)
    return int(m.group(1)) if m else None


def newest_win_amd64_pytag(wheels: "list[dict]") -> "str | None":
    """The highest cpNNN python tag published for a win_amd64 wheel, or None
    when no win_amd64 wheel in *wheels* carries a Python-3.x cp-tag at all (a
    py3-none-only package imposes no ceiling and is excluded by this filter,
    not reported as a false floor)."""
    by_minor = {}
    for w in wheels:
        if w["platform"] != "win_amd64":
            continue
        minor = _cp3_tag_minor(w["pytag"])
        if minor is not None:
            by_minor[minor] = w["pytag"]
    if not by_minor:
        return None
    return by_minor[max(by_minor)]


# --------------------------------------------------------------------------- #
#  Reporting                                                                  #
# --------------------------------------------------------------------------- #

def _report_package(pkg: str, pinned: "str | None", wheels: "list[dict] | None") -> None:
    if pinned is None:
        print(f"{pkg}: could not read the pinned version from the source - has it moved?")
        return
    if wheels is None:
        print(f"{pkg}: pinned {pinned}; could not check (AMD wheel index unreachable)")
        return

    pytag = "cp312" if pkg in _TORCH_STACK else None
    newest = newest_win_amd64_version(wheels, pytag=pytag)
    where = f"win_amd64, {pytag}" if pytag else "win_amd64"
    if newest is None:
        print(f"{pkg}: pinned {pinned}; the index published no matching {where} wheel "
              "to compare against")
        return
    newest_raw, newest_tuple = newest

    pinned_tuple = _base_version_tuple(pinned)
    if pinned_tuple is None:
        print(f"{pkg}: pinned version {pinned!r} does not parse as a plain version; "
              "skipping the comparison")
        return

    if newest_tuple > pinned_tuple:
        print(f"{pkg}: STALE - pinned {pinned}, newest published ({where}) is {newest_raw}")
    else:
        print(f"{pkg}: current - pinned {pinned} is the newest published ({where})")


def _report_python_abi(wheels_by_pkg: "dict[str, list[dict] | None]") -> None:
    torch_stack_wheels = [
        w for pkg in _TORCH_STACK for w in (wheels_by_pkg.get(pkg) or [])
    ]
    if not any(wheels_by_pkg.get(pkg) is not None for pkg in _TORCH_STACK):
        print("Python ABI: could not determine the newest published win_amd64 cp-tag "
              "(the torch/torchvision index was unreachable)")
        return

    newest_pytag = newest_win_amd64_pytag(torch_stack_wheels)
    if newest_pytag is None:
        print("Python ABI: torch/torchvision published no win_amd64 wheel with a "
              "cpNNN tag at all - could not determine a ceiling")
        return

    floor = _requires_python_floor()
    print(f"Python ABI: the AMD index's newest win_amd64 wheel for torch/torchvision "
          f"is built for {newest_pytag}. rocm-sdk-core and "
          "rocm-sdk-libraries-gfx103x-all ship py3-none wheels and impose no ceiling.")
    print(f"pyproject.toml's requires-python floor is {floor!r}.")

    floor_minor = _floor_minor(floor) if floor else None
    newest_minor = _cp3_tag_minor(newest_pytag)
    if floor_minor is None:
        print("  could not read a >=3.Y floor from requires-python to compare against")
    elif newest_minor > floor_minor:
        print(f"  a Python bump beyond 3.{floor_minor} looks ROCm-safe TODAY, up to "
              f"{newest_pytag} (torch/torchvision already publish a win_amd64 wheel "
              "there). Report only - this script changes nothing.")
    else:
        print(f"  no win_amd64 wheel exists beyond 3.{floor_minor} yet; a Python bump "
              "would NOT be ROCm-safe today.")


# --------------------------------------------------------------------------- #
#  Entry point                                                                #
# --------------------------------------------------------------------------- #

def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)

    wheels_by_pkg: "dict[str, list[dict] | None]" = {}
    for pkg in _PACKAGES:
        hrefs = _fetch_index(pkg)
        if hrefs is None:
            wheels_by_pkg[pkg] = None
        else:
            wheels_by_pkg[pkg] = [w for w in (_parse_wheel(h) for h in hrefs) if w is not None]

    print("AMD ROCm wheel pin currency (report only; changes nothing)")
    print()
    for pkg in _TORCH_STACK:
        _report_package(pkg, _pinned_torch_stack_version(pkg), wheels_by_pkg[pkg])
    for pkg in _ROCM_SDK:
        _report_package(pkg, _pinned_rocm_sdk_version(pkg), wheels_by_pkg[pkg])
    print()
    _report_python_abi(wheels_by_pkg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
