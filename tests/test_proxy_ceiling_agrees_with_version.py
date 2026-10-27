# SPDX-License-Identifier: AGPL-3.0-or-later
"""The update proxy's legacy ceiling and localm's version order agree.

A release the proxy offers to a client from 0.3.1 or earlier (one at or below
``LEGACY_CLIENT_CEILING``) must also order at or below the ceiling for
:mod:`localm._version`, which the release tooling uses to decide whether a build
is offered to those clients.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from localm import _version

WORKER = Path(__file__).resolve().parents[1] / "tools" / "bugreport-proxy" / "worker.js"


def _worker_source() -> str:
    return WORKER.read_text(encoding="utf-8")


def _js_regex(name: str) -> re.Pattern[str]:
    m = re.search(rf"const {name} = /(.+)/(\w*);", _worker_source())
    assert m, f"{name} not found in worker.js"
    return re.compile(m.group(1), re.IGNORECASE if "i" in m.group(2) else 0)


def _ceiling() -> str:
    m = re.search(r'const LEGACY_CLIENT_CEILING = "(v[^"]+)";', _worker_source())
    assert m, "LEGACY_CLIENT_CEILING not found in worker.js"
    return m.group(1)


def _tag_key(tag: str):
    """worker.js tagKey."""
    m = re.match(r"^v(\d+)\.(\d+)\.(\d+)(.*)$", tag)
    if not m:
        return None
    suffix = re.sub(r"^[-.]", "", m.group(4))
    if suffix and not _js_regex("PRERELEASE_SUFFIX_RE").match(suffix):
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), 0 if suffix else 1, suffix.lower())


def _proxy_offers_legacy(tag: str, ceiling: str) -> bool:
    """worker.js tagAtMost(tag, ceiling)."""
    ka, kb = _tag_key(tag), _tag_key(ceiling)
    if ka is None or kb is None:
        return False
    if ka[:4] != kb[:4]:
        return ka[:4] < kb[:4]
    return ka[4] <= kb[4]


def _candidate_tags(ceiling: str) -> list[str]:
    names = ["rc", "a", "b", "alpha", "beta", "RC", "dev", "post", "hotfix", ""]
    numbers = ["1", "12", ".1", "-1", "_1", ""]
    seps = ["", "-", ".", "+", "_"]
    major, minor, patch = (int(x) for x in ceiling[1:].split(".")[:3])
    bases = [ceiling, f"v{major}.{minor}.{max(patch - 1, 0)}", f"v{major}.{minor}.{patch + 1}"]
    return sorted({f"{b}{s}{n}{d}" for b in bases for s in seps for n in names for d in numbers})


def test_the_worker_constants_are_found():
    assert _ceiling().startswith("v")
    assert _js_regex("PRERELEASE_SUFFIX_RE").match("rc1")


@pytest.mark.parametrize("tag", _candidate_tags(_ceiling()))
def test_every_tag_the_proxy_offers_old_clients_orders_at_or_below_the_ceiling(tag):
    ceiling = _ceiling()
    if _proxy_offers_legacy(tag, ceiling):
        assert not _version.is_newer(tag, ceiling), (
            f"the proxy offers {tag} to clients from 0.3.1 and earlier, "
            f"but localm orders it above {ceiling}")


def test_the_dotted_prerelease_is_not_offered_to_old_clients():
    ceiling = _ceiling()
    for tag in (f"{ceiling}-rc.1", f"{ceiling}rc.1", f"{ceiling}b.2", f"{ceiling}alpha.3"):
        assert not _proxy_offers_legacy(tag, ceiling), tag
