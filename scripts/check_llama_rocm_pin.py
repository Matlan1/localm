#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Report how far localm's pinned lemonade-sdk llama.cpp ROCm build has fallen behind upstream.

``localm/setup_llama.py`` installs ``_ROCM_TAG``, one lemonade-sdk/llamacpp-rocm
release confirmed to work on gfx103X hardware, rather than whatever that repo
published most recently. This is a SEPARATE tag series from ``_PINNED_TAG``
(ggml-org/llama.cpp, checked by scripts/check_llama_pin.py) - see the comments
on ``_ROCM_TAG`` and ``_tag_for`` in setup_llama.py.

This is a MAINTENANCE SIGNAL, not a build gate, the same shape as
scripts/check_llama_pin.py and scripts/check_comfyui_pin.py for this tree's
other pinned dependencies. It always exits 0.

Unlike ggml-org/llama.cpp, lemonade-sdk/llamacpp-rocm's releases carry a
meaningful draft/prerelease flag, so a candidate is filtered the way
scripts/check_comfyui_pin.py filters ComfyUI's: excluded on draft OR
prerelease. The tag SHAPE is "bNNNN" like _PINNED_TAG rather than ComfyUI's
"vX.Y.Z", so tags are compared as an integer build number, the same approach
scripts/check_llama_pin.py uses.

There is no confirm_llama_runtime.py-style automated confirmation for this
pin: that script's backend list excludes amd-rocm, because the amd-rocm build
never resolves from an upstream ggml-org tag at all. Advancing _ROCM_TAG means
a maintainer running the newer lemonade-sdk build through localm's real loader
on AMD ROCm hardware, confirming it loads and generates, and then updating
_ROCM_TAG, DEFAULT_URL, DEFAULT_URL_SHA256 and the affected
_PINNED_FALLBACK_SHA256 entries in setup_llama.py together - a person's
decision, not something this script does.

Fails soft on the API: unreachable, rate-limited, or a malformed response all
print a clearly-labelled "could not check" and exit 0 - never a false "up to
date".

Usage:
    python scripts/check_llama_rocm_pin.py
    python scripts/check_llama_rocm_pin.py --pinned b1300   # sanity-check a hypothetical

Stdlib only (urllib + re + json), so the CI job running it does not need
localm installed - same as check_llama_pin.py and check_comfyui_pin.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

_REPO = "lemonade-sdk/llamacpp-rocm"
_SETUP_PATH = Path(__file__).resolve().parent.parent / "localm" / "setup_llama.py"
_PIN_RE = re.compile(r'^_ROCM_TAG\s*=\s*"([^"]+)"', re.M)
# lemonade-sdk build tags are "b" plus a monotonically increasing build number,
# the same shape as ggml-org's own but a different, unrelated numbering.
_TAG_RE = re.compile(r"^b(\d+)$")
_PER_PAGE = 100


def pinned_tag(path: Path = _SETUP_PATH) -> str:
    """_ROCM_TAG out of setup_llama.py, BY TEXT rather than by importing it.

    Importing would drag in click, rich and the rest of localm, which this script
    is independent of so the CI job can run it with nothing installed. Reading by
    text also keeps the script usable when the tree it is reading does not import
    cleanly."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"could not read {path}: {e}")
    m = _PIN_RE.search(text)
    if not m:
        raise SystemExit(
            f"no _ROCM_TAG assignment found in {path}. If the constant was "
            "renamed, this script needs updating - it is not evidence the pin "
            "is fine.")
    return m.group(1)


def _build_number(tag: str):
    m = _TAG_RE.match((tag or "").strip())
    return int(m.group(1)) if m else None


def upstream_tags() -> "tuple[list, str]":
    """(tags newest-first, error). Never raises: a failed lookup is reported as
    an error string, never as an empty list that would read as "nothing newer"."""
    url = f"https://api.github.com/repos/{_REPO}/releases?per_page={_PER_PAGE}"
    req = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github+json",
                      "User-Agent": "localm-check-llama-rocm-pin"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            releases = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return [], f"HTTP {e.code} from the GitHub releases API"
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"
    if not isinstance(releases, list):
        return [], "the releases API returned something that is not a list"
    tags = []
    for rel in releases:
        if not isinstance(rel, dict) or rel.get("draft") or rel.get("prerelease"):
            continue
        tag = rel.get("tag_name")
        # Skip releases whose assets have not finished uploading.
        if isinstance(tag, str) and _build_number(tag) is not None and rel.get("assets"):
            tags.append(tag)
    if not tags:
        return [], "no usable release with uploaded assets in the API response"
    tags.sort(key=lambda t: _build_number(t), reverse=True)
    return tags, ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pinned", default=None,
                    help="check this tag instead of the one in setup_llama.py")
    args = ap.parse_args(argv)

    pin = args.pinned or pinned_tag()
    pin_n = _build_number(pin)
    print(f"localm pins the lemonade-sdk ROCm build at {pin}")
    if pin_n is None:
        print(f"COULD NOT CHECK: {pin!r} is not upstream's bNNNNN tag shape, so "
              "it cannot be compared. This is not a statement that the pin is "
              "current.")
        return 0

    tags, err = upstream_tags()
    if err:
        print(f"COULD NOT CHECK upstream releases: {err}")
        print("This is NOT 'the pin is up to date' - nothing was compared.")
        return 0

    newest = tags[0]
    behind = [t for t in tags if (_build_number(t) or 0) > pin_n]
    print(f"upstream newest with assets: {newest}")
    if not behind:
        print(f"OK: the pin is current (nothing newer than {pin} in the last "
              f"{len(tags)} releases).")
        return 0
    print(f"BEHIND by {len(behind)} release(s): {', '.join(behind[:10])}"
          f"{' ...' if len(behind) > 10 else ''}")
    if len(behind) >= _PER_PAGE:
        print(f"(that is the whole {_PER_PAGE}-release page, so the real gap may "
              "be larger)")
    print("\nTo advance the pin: run the newer lemonade-sdk build through "
          "localm's real loader on AMD ROCm hardware and confirm it loads AND "
          "generates. There is no confirm_llama_runtime.py-style automated "
          "check for this pin - that script's backend list excludes amd-rocm, "
          "since this build never resolves from an upstream ggml-org tag. Once "
          "confirmed, update in setup_llama.py together:")
    print(f"    _ROCM_TAG = {newest!r}")
    print("    DEFAULT_URL, DEFAULT_URL_SHA256, and the affected "
          "_PINNED_FALLBACK_SHA256 entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
