#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Report how far localm's pinned llama.cpp build has fallen behind upstream."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

_REPO = "ggml-org/llama.cpp"
_SETUP_PATH = Path(__file__).resolve().parent.parent / "localm" / "setup_llama.py"
_PIN_RE = re.compile(r'^_PINNED_TAG\s*=\s*"([^"]+)"', re.M)
# Upstream's build tags are "b" plus a monotonically increasing build number.
# Parsed to an int rather than compared as strings, for the same reason
# check_comfyui_pin.py parses versions: "b9870" > "b10375" lexically, since '9'
# sorts after '1', so a string comparison reports a NEWER pin as behind - and it
# starts doing so silently at the exact moment the digit count changes.
_TAG_RE = re.compile(r"^b(\d+)$")
_PER_PAGE = 100


def pinned_tag(path: Path = _SETUP_PATH) -> str:
    """_PINNED_TAG out of setup_llama.py, BY TEXT rather than by importing it."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"could not read {path}: {e}")
    m = _PIN_RE.search(text)
    if not m:
        raise SystemExit(
            f"no _PINNED_TAG assignment found in {path}. If the constant was "
            "renamed, this script needs updating - it is not evidence the pin "
            "is fine.")
    return m.group(1)


def _build_number(tag: str):
    m = _TAG_RE.match((tag or "").strip())
    return int(m.group(1)) if m else None


def upstream_tags() -> "tuple[list, str]":
    """(tags newest-first, error)."""
    url = f"https://api.github.com/repos/{_REPO}/releases?per_page={_PER_PAGE}"
    req = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github+json",
                      "User-Agent": "localm-check-llama-pin"})
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
        # Same asset filter setup_llama._recent_tags applies: a release is
        # published before its ~25 archives finish uploading, and a tag with no
        # assets is not something anyone could install, so counting it as
        # "behind" would overstate the gap.
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
    print(f"localm pins llama.cpp {pin}")
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
    print("\nTo advance the pin - and it is one action, not two, because a bump "
          "without the confirm is exactly the untested-build problem the pin "
          "exists to remove:")
    print(f"    python scripts/confirm_llama_runtime.py --tag {newest} --backend cpu")
    print("    # plus --backend vulkan on a machine with a GPU, then bump "
          "_PINNED_TAG,")
    print("    # its _PINNED_FALLBACK_SHA256 entries, and _PIN_CONFIRMATION.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
