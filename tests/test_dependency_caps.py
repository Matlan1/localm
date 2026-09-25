# SPDX-License-Identifier: AGPL-3.0-or-later
"""Upper bounds in pyproject.toml that keep a fresh install resolvable.

setup.bat, setup.sh and the graphical installer run `uv pip install -e .`,
which resolves from pyproject.toml and never reads uv.lock, so these bounds are
what a new install actually gets.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def _base_requirement(name: str) -> Requirement:
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    reqs = [Requirement(d) for d in deps["project"]["dependencies"]]
    found = [r for r in reqs if r.name.lower() == name]
    assert len(found) == 1, f"expected one {name} requirement, got {found}"
    return found[0]


def test_huggingface_hub_stays_below_what_tokenizers_accepts():
    """tokenizers 0.22.x, which faster-whisper and transformers pull in,
    requires huggingface-hub<2.0. Allowing 2.x makes the resolver take it and
    fall back to tokenizers 0.13.3, which has no Python 3.12 wheel and needs a
    Rust compiler to build, so the install fails on most machines."""
    spec = _base_requirement("huggingface-hub").specifier
    assert not spec.contains("2.0.0"), spec
    assert spec.contains("1.30.0"), spec


def test_the_lock_agrees_with_the_cap():
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    hub = [p for p in lock["package"] if p["name"] == "huggingface-hub"]
    assert hub, "huggingface-hub is missing from uv.lock"
    for p in hub:
        assert _base_requirement("huggingface-hub").specifier.contains(p["version"]), p
